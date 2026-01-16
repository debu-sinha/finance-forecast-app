"""
FastAPI backend for Databricks Finance Forecasting Platform
"""
import os
import gc
import logging
import numpy as np
import tempfile
import glob as glob_module
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.schemas import (
    TrainRequest, TrainResponse, DeployRequest, DeployResponse,
    HealthResponse, ForecastMetrics, ModelResult, CovariateImpact,
    BatchTrainRequest, BatchTrainResponse, BatchResultItem,
    AggregateRequest, AggregateResponse, TestModelRequest, TestModelResponse,
    ModelTestResult, BatchDeployRequest, BatchDeployResponse, BatchSegmentInfo,
    DataAnalysisRequest, DataAnalysisResponse
)
from backend.models.prophet import train_prophet_model, prepare_prophet_data
from backend.models.arima import train_arima_model, train_sarimax_model
from backend.models.ets import train_exponential_smoothing_model
from backend.models.xgboost import train_xgboost_model
from backend.models.utils import register_model_to_unity_catalog, validate_mlflow_run_artifacts
from backend.deploy_service import (
    deploy_model_to_serving, get_endpoint_status, delete_endpoint,
    list_endpoints, get_databricks_client, test_model_inference
)
from backend.ai_service import analyze_dataset, generate_forecast_insights, generate_executive_summary
from backend.data_analyzer import analyze_time_series, get_analysis_summary_for_ui

# Simple Mode - Autopilot forecasting for finance users
try:
    from backend.simple_mode import simple_mode_router
    SIMPLE_MODE_AVAILABLE = True
except ImportError:
    SIMPLE_MODE_AVAILABLE = False
    simple_mode_router = None

# AI Thinker - Opus 4.5 powered intelligent analysis
try:
    from backend.ai_thinker import thinker_router
    THINKER_AVAILABLE = True
except ImportError:
    THINKER_AVAILABLE = False
    thinker_router = None

# Configure logging with both console and file handlers
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'training.log')

# Create formatter
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Root logger setup
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Console handler (stdout)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

# File handler (writes to backend/logs/training.log)
file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

# Add handlers to root logger
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)
logger.info(f"📝 Logging to file: {LOG_FILE}")


def truncate_log_file():
    """Truncate the log file to start fresh for a new training run."""
    global file_handler
    try:
        # Close the current file handler
        file_handler.close()
        root_logger.removeHandler(file_handler)

        # Truncate the file
        with open(LOG_FILE, 'w') as f:
            f.truncate(0)

        # Re-create and add the file handler
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

        logger.info(f"📝 Log file truncated and ready for new training run: {LOG_FILE}")
    except Exception as e:
        logger.warning(f"Could not truncate log file: {e}")


def cleanup_temp_files_and_memory():
    """Clean up temp files and free memory to prevent 'too many open files' errors."""
    try:
        # Clean up temp files that may have been created during MLflow logging
        temp_patterns = [
            '/tmp/*.csv',
            '/tmp/*.pkl',
            '/tmp/*.json',
            '/tmp/tmp*',
        ]
        for pattern in temp_patterns:
            for f in glob_module.glob(pattern):
                try:
                    os.remove(f)
                except:
                    pass

        # Force garbage collection to release file handles
        gc.collect()

        logger.debug("Cleaned up temp files and ran garbage collection")
    except Exception as e:
        logger.warning(f"Cleanup warning (non-fatal): {e}")


def log_dataframe_summary(df, name: str, show_sample: bool = True):
    """Log a comprehensive summary of a DataFrame for debugging."""
    import pandas as pd

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"📋 {name.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"   Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    logger.info(f"   Columns: {list(df.columns)}")

    # Log data types
    logger.info(f"   Data types:")
    for col in df.columns:
        logger.info(f"      - {col}: {df[col].dtype}")

    # Log null counts
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.info(f"   Null counts:")
        for col, count in null_counts.items():
            if count > 0:
                logger.info(f"      - {col}: {count} nulls ({count/len(df)*100:.1f}%)")
    else:
        logger.info(f"   Null counts: No nulls in data")

    # Log date range if there's a datetime column
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or col in ['ds', 'date', 'Date', 'DATE']:
            try:
                col_dates = pd.to_datetime(df[col])
                logger.info(f"   Date range ({col}): {col_dates.min()} to {col_dates.max()}")
            except:
                pass

    # Log numeric column stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        logger.info(f"   Numeric column stats:")
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            logger.info(f"      - {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}, std={df[col].std():.4f}")

    # Log sample rows
    if show_sample and len(df) > 0:
        logger.info(f"   First 3 rows:")
        for idx, row in df.head(3).iterrows():
            row_str = ", ".join([f"{k}={v}" for k, v in row.items()])
            logger.info(f"      [{idx}] {row_str[:200]}...")
        logger.info(f"   Last 3 rows:")
        for idx, row in df.tail(3).iterrows():
            row_str = ", ".join([f"{k}={v}" for k, v in row.items()])
            logger.info(f"      [{idx}] {row_str[:200]}...")

    logger.info(f"{'='*60}")

# Resolve "inherit" in DATABRICKS_HOST
if os.environ.get("DATABRICKS_HOST") == "inherit":
    try:
        from databricks.sdk import WorkspaceClient
        os.environ["DATABRICKS_HOST"] = WorkspaceClient().config.host
        logger.info(f"Resolved DATABRICKS_HOST: {os.environ['DATABRICKS_HOST']}")
    except Exception as e:
        logger.warning(f"Failed to resolve DATABRICKS_HOST: {e}")

app = FastAPI(title="Databricks Finance Forecasting API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = "dist" if os.path.exists("dist") else "../dist"
if os.path.exists(static_dir):
    app.mount("/assets", StaticFiles(directory=f"{static_dir}/assets"), name="assets")

# Register Simple Mode routes
if SIMPLE_MODE_AVAILABLE and simple_mode_router:
    app.include_router(simple_mode_router)
    logger.info("✅ Simple Mode routes registered at /api/simple/*")

# Register AI Thinker routes
if THINKER_AVAILABLE and thinker_router:
    app.include_router(thinker_router)
    logger.info("✅ AI Thinker routes registered at /api/thinker/*")

@app.get("/")
async def root():
    return FileResponse(f"{static_dir}/index.html") if os.path.exists(f"{static_dir}/index.html") else {"message": "API Running"}

@app.post("/api/analyze")
async def analyze_data(request: dict):
    try:
        return JSONResponse(content=analyze_dataset(request.get('sample_data', []), request.get('columns', [])))
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return JSONResponse(content={"summary": "Analysis failed.", "suggestedCovariates": [], "suggestedGroupColumns": []})

@app.post("/api/insights")
async def generate_insights_endpoint(request: dict):
    try:
        return JSONResponse(content=generate_forecast_insights(**request))
    except Exception as e:
        logger.error(f"Insights error: {e}")
        return JSONResponse(content={"explanation": "Model trained.", "pythonCode": "# Prophet training code"})

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    status_data = {"status": "healthy", "message": "Backend operational", "databricks_connected": False, "mlflow_enabled": False}
    try:
        get_databricks_client().serving_endpoints.list()
        status_data["databricks_connected"] = True
    except Exception as e:
        logger.debug(f"Databricks connection check failed (expected in dev): {type(e).__name__}")

    try:
        import mlflow
        mlflow.set_tracking_uri("databricks")
        status_data["mlflow_enabled"] = True
    except Exception as e:
        logger.debug(f"MLflow setup failed (expected in dev): {type(e).__name__}")
    return HealthResponse(**status_data)


@app.post("/api/analyze-data", response_model=DataAnalysisResponse)
async def analyze_data(request: DataAnalysisRequest):
    """
    Analyze time series data and provide intelligent recommendations for:
    - Which models to use based on data characteristics
    - Hyperparameter ranges to explore
    - Data quality assessment and warnings

    Call this endpoint before training to get data-driven model/hyperparameter recommendations.
    """
    try:
        import pandas as pd

        logger.info(f"📊 Analyzing data: {len(request.data)} rows, time_col={request.time_col}, target_col={request.target_col}")

        # Convert to DataFrame
        df = pd.DataFrame(request.data)

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data provided")

        if request.time_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Time column '{request.time_col}' not found in data")

        if request.target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_col}' not found in data")

        # Run analysis
        result = analyze_time_series(
            df=df,
            time_col=request.time_col,
            target_col=request.target_col,
            frequency=request.frequency
        )

        # Convert to UI-friendly format
        summary = get_analysis_summary_for_ui(result)

        return DataAnalysisResponse(**summary)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data analysis failed: {str(e)}")


@app.post("/api/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    try:
        import pandas as pd

        # TRUNCATE LOG FILE at the start of each training run (unless batch training)
        if not request.batch_id:
            truncate_log_file()

        logger.info(f"")
        logger.info(f"{'#'*70}")
        logger.info(f"#  NEW TRAINING RUN STARTED")
        logger.info(f"{'#'*70}")
        logger.info(f"Training request: target={request.target_col}, horizon={request.horizon}, data_rows={len(request.data)}")

        # Log segment/filter info for batch training
        if request.filters:
            logger.info(f"📋 Segment filters: {request.filters}")
        if request.batch_segment_id:
            logger.info(f"📋 Batch segment: {request.batch_segment_id}")

        # Log hyperparameter filter info (from data analysis)
        if request.hyperparameter_filters:
            logger.info(f"📊 Using data-driven hyperparameter filters for {len(request.hyperparameter_filters)} models")
            for model_name, filters in request.hyperparameter_filters.items():
                logger.info(f"   - {model_name}: {list(filters.keys())}")

        # ========================================
        # LOG RAW INPUT DATA
        # ========================================
        raw_input_df = pd.DataFrame(request.data)
        log_dataframe_summary(raw_input_df, "RAW INPUT DATA (from frontend)")

        # ========================================
        # LOG RAW PROMOTIONS/EVENTS DATA (if covariates provided)
        # ========================================
        if request.covariates and len(request.covariates) > 0:
            logger.info(f"")
            logger.info(f"{'='*60}")
            logger.info(f"📋 RAW PROMOTIONS/COVARIATES")
            logger.info(f"{'='*60}")
            logger.info(f"   Covariates requested: {request.covariates}")
            # Check which covariates are present in the data
            present_covariates = [c for c in request.covariates if c in raw_input_df.columns]
            missing_covariates = [c for c in request.covariates if c not in raw_input_df.columns]
            logger.info(f"   Present in data: {present_covariates}")
            if missing_covariates:
                logger.info(f"   ⚠️ Missing from data: {missing_covariates}")
            # Log covariate statistics
            for cov in present_covariates:
                non_null_count = raw_input_df[cov].notna().sum()
                non_zero_count = (raw_input_df[cov] != 0).sum() if raw_input_df[cov].dtype in ['int64', 'float64'] else 0
                logger.info(f"   - {cov}: non-null={non_null_count}, non-zero={non_zero_count}, unique={raw_input_df[cov].nunique()}")
            logger.info(f"{'='*60}")
        else:
            logger.info(f"📋 No covariates/promotions provided in request")

        # ========================================
        # LOG DATE FILTER PARAMETERS
        # ========================================
        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"📅 DATE FILTER PARAMETERS")
        logger.info(f"{'='*60}")
        logger.info(f"   from_date: {request.from_date or '(not specified - use all data)'}")
        logger.info(f"   to_date: {request.to_date or '(not specified - will use max date in data)'}")
        # Parse raw dates to show the range
        if request.time_col in raw_input_df.columns:
            raw_dates = pd.to_datetime(raw_input_df[request.time_col], errors='coerce')
            logger.info(f"   Raw data date range: {raw_dates.min()} to {raw_dates.max()}")
            logger.info(f"   Total periods in raw data: {len(raw_input_df)}")
        logger.info(f"{'='*60}")

        if not request.data or request.horizon <= 0:
            raise ValueError("Invalid data or horizon")
        
        # Set random seeds for reproducibility
        import random
        import numpy as np
        seed = request.random_seed if request.random_seed is not None else 42
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"🌱 Set random seed: {seed} for reproducibility")

        # CRITICAL: Filter out target column from covariates to prevent data leakage
        # The target column should NEVER be used as a covariate/feature
        safe_covariates = [c for c in (request.covariates or []) if c != request.target_col]
        if len(safe_covariates) != len(request.covariates or []):
            logger.warning(f"🚨 Removed target column '{request.target_col}' from covariates to prevent data leakage")

        df = prepare_prophet_data(request.data, request.time_col, request.target_col, safe_covariates)

        # ========================================
        # LOG COMBINED/PROCESSED DATASET (after prepare_prophet_data)
        # ========================================
        log_dataframe_summary(df, "COMBINED/PROCESSED DATASET (after prepare_prophet_data)")

        # Store pre-filter data for logging (shallow copy for logging only)
        pre_filter_df = df

        # Apply date range filtering using a single mask (more efficient than multiple .copy())
        original_len = len(df)
        import pandas as pd

        # Default to_date to the end of the time series if not provided
        max_date = df['ds'].max()
        effective_to_date = request.to_date if request.to_date else str(max_date.date())

        # Build combined date mask (avoids multiple DataFrame copies)
        date_mask = pd.Series(True, index=df.index)

        if request.from_date:
            from_date = pd.to_datetime(request.from_date).normalize()
            date_mask &= (df['ds'] >= from_date)

        to_date = pd.to_datetime(effective_to_date).normalize()
        date_mask &= (df['ds'] <= to_date)

        # Apply mask once with single copy
        df = df[date_mask].copy()

        # Log filtering results
        if request.from_date:
            logger.info(f"📅 Filtered data: {original_len} -> {len(df)} rows (from_date >= {request.from_date})")
        if request.to_date:
            logger.info(f"📅 Filtered data: to_date <= {request.to_date}")
        else:
            logger.info(f"📅 Applied default to_date filter: to_date <= {effective_to_date} (max date in dataset)")
        
        if len(df) == 0:
            raise ValueError(f"No data remaining after date filtering (from_date: {request.from_date}, to_date: {effective_to_date})")

        if request.from_date or not request.to_date:
            logger.info(f"📅 Final filtered dataset: {len(df)} rows (from {df['ds'].min()} to {df['ds'].max()})")

        # ========================================
        # LOG DATE-FILTERED DATASET
        # ========================================
        log_dataframe_summary(df, f"DATE-FILTERED DATASET (from_date={request.from_date or 'N/A'}, to_date={effective_to_date})")

        # ========================================
        # TRAIN / EVAL / HOLDOUT SPLIT
        # ========================================
        # Strategy:
        #   - Holdout: final validation set (never seen during hyperparameter tuning)
        #   - Eval: hyperparameter tuning validation
        #   - Train: model training
        # Split ratios: ~70% train, ~15% eval, ~15% holdout (adjusted based on data size)

        total_rows = len(df)

        # Calculate split sizes - minimum 1 period for eval and holdout
        eval_size = request.test_size or min(request.horizon, max(1, total_rows // 7))
        holdout_size = min(request.horizon, max(1, total_rows // 7))

        # Ensure we have enough data for training
        min_train_size = max(request.horizon * 2, 30)  # At least 2x horizon or 30 points
        if total_rows - eval_size - holdout_size < min_train_size:
            # Not enough data for 3-way split, fall back to 2-way split
            logger.warning(f"⚠️ Not enough data for 3-way split (need {min_train_size + eval_size + holdout_size}, have {total_rows}). Using 2-way train/eval split.")
            holdout_size = 0
            eval_size = request.test_size or min(request.horizon, len(df) // 5)

        # Create the splits
        if holdout_size > 0:
            holdout_df = df.iloc[-holdout_size:].copy()
            eval_df = df.iloc[-(holdout_size + eval_size):-holdout_size].copy()
            train_df = df.iloc[:-(holdout_size + eval_size)].copy()
        else:
            holdout_df = pd.DataFrame()  # Empty holdout
            eval_df = df.iloc[-eval_size:].copy()
            train_df = df.iloc[:-eval_size].copy()

        # For backward compatibility, test_df refers to eval_df (used by model training functions)
        test_df = eval_df
        test_size = eval_size

        # ========================================
        # LOG TRAIN/EVAL/HOLDOUT SPLIT
        # ========================================
        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"📊 TRAIN/EVAL/HOLDOUT SPLIT")
        logger.info(f"{'='*60}")
        logger.info(f"   Total rows: {total_rows}")
        logger.info(f"   Train rows: {len(train_df)} ({len(train_df)/total_rows*100:.1f}%)")
        logger.info(f"   Eval rows: {len(eval_df)} ({len(eval_df)/total_rows*100:.1f}%) - for hyperparameter tuning")
        if holdout_size > 0:
            logger.info(f"   Holdout rows: {len(holdout_df)} ({len(holdout_df)/total_rows*100:.1f}%) - for final model selection")
        else:
            logger.info(f"   Holdout rows: 0 (insufficient data for 3-way split)")
        logger.info(f"{'='*60}")

        # Log TRAIN set details
        log_dataframe_summary(train_df, "TRAINING SET")

        # Log EVAL set details
        log_dataframe_summary(eval_df, "EVAL SET (hyperparameter tuning)")

        # Log HOLDOUT set details if exists
        if holdout_size > 0:
            log_dataframe_summary(holdout_df, "HOLDOUT SET (final model selection)")

        # Log the split points
        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"📅 SPLIT POINT VERIFICATION")
        logger.info(f"{'='*60}")
        logger.info(f"   Train ends at: {train_df['ds'].max()}")
        logger.info(f"   Eval starts at: {eval_df['ds'].min()}")
        if holdout_size > 0:
            logger.info(f"   Eval ends at: {eval_df['ds'].max()}")
            logger.info(f"   Holdout starts at: {holdout_df['ds'].min()}")
            logger.info(f"   Holdout ends at: {holdout_df['ds'].max()}")
        if len(train_df) > 0 and len(eval_df) > 0:
            train_end = train_df['ds'].max()
            eval_start = eval_df['ds'].min()
            gap_days = (eval_start - train_end).days
            logger.info(f"   Gap between train end and eval start: {gap_days} days")
            if gap_days > 7:
                logger.warning(f"   ⚠️ Large gap detected! This may indicate data issues.")
        logger.info(f"{'='*60}")

        import mlflow
        from datetime import datetime
        mlflow.set_tracking_uri("databricks")
        experiment_base = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/finance-forecasting")

        # For batch training, use a consistent experiment name based on batch_id
        # This ensures all segments from the same batch are grouped together
        if request.batch_id:
            experiment_name = f"{experiment_base}-batch-{request.batch_id}"
            logger.info(f"Batch training mode: using experiment {experiment_name}")
        else:
            experiment_name = f"{experiment_base}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.set_experiment(experiment_name)

        model_results = []
        best_mape = float('inf')
        best_model_name = None
        best_run_id = None
        artifact_uri_ref = None
        parent_run_id = None
        experiment_id = None

        # Create descriptive run name
        if request.batch_id and request.batch_segment_id:
            run_name = f"Segment: {request.batch_segment_id}"
        elif request.filters:
            filter_str = " | ".join(f"{k}={v}" for k, v in request.filters.items())
            run_name = f"Forecast: {filter_str}"
        else:
            run_name = "Finance_Forecast_Training"

        with mlflow.start_run(run_name=run_name) as parent_run:
            parent_run_id = parent_run.info.run_id
            experiment_id = parent_run.info.experiment_id
            
            # Log original uploaded data (including promotions/covariates)
            try:
                import json
                original_data_df = pd.DataFrame(request.data)
                original_data_df.to_csv("/tmp/original_uploaded_data.csv", index=False)
                mlflow.log_artifact("/tmp/original_uploaded_data.csv", "input_data")  # Changed to input_data folder
                logger.info(f"Logged original uploaded data to input_data/: {len(original_data_df)} rows with columns: {list(original_data_df.columns)}")

                # Log pre-filter data (after type conversion but before date filtering)
                pre_filter_df.to_csv("/tmp/pre_filter_data.csv", index=False)
                mlflow.log_artifact("/tmp/pre_filter_data.csv", "datasets/raw")
                logger.info(f"Logged pre-filter data to datasets/raw/: {len(pre_filter_df)} rows")

                # Log post-filter data (after date range filtering)
                df.to_csv("/tmp/post_filter_data.csv", index=False)
                mlflow.log_artifact("/tmp/post_filter_data.csv", "datasets/processed")
                logger.info(f"Logged post-filter data to datasets/processed/: {len(df)} rows")
                
                # Add train/eval/holdout split metadata
                split_metadata = {
                    "train_size": len(train_df),
                    "eval_size": len(eval_df),
                    "holdout_size": len(holdout_df) if holdout_size > 0 else 0,
                    "total_size": len(df),
                    "train_eval_split_date": str(train_df['ds'].max()),
                    "eval_holdout_split_date": str(eval_df['ds'].max()) if holdout_size > 0 else None,
                    "train_percentage": round(len(train_df) / len(df) * 100, 2),
                    "eval_percentage": round(len(eval_df) / len(df) * 100, 2),
                    "holdout_percentage": round(len(holdout_df) / len(df) * 100, 2) if holdout_size > 0 else 0,
                    "train_date_range": {
                        "start": str(train_df['ds'].min()),
                        "end": str(train_df['ds'].max())
                    },
                    "eval_date_range": {
                        "start": str(eval_df['ds'].min()),
                        "end": str(eval_df['ds'].max())
                    },
                    "holdout_date_range": {
                        "start": str(holdout_df['ds'].min()),
                        "end": str(holdout_df['ds'].max())
                    } if holdout_size > 0 else None
                }
                with open("/tmp/train_eval_holdout_split.json", "w") as f:
                    json.dump(split_metadata, f, indent=2)
                mlflow.log_artifact("/tmp/train_eval_holdout_split.json", "metadata")
                logger.info(f"Logged train/eval/holdout split metadata: train={len(train_df)}, eval={len(eval_df)}, holdout={len(holdout_df) if holdout_size > 0 else 0}")
            except Exception as e:
                logger.warning(f"Could not log original uploaded data or metadata: {e}")
            
            # Log all training parameters and data filters for reproducibility
            mlflow.log_param("time_column", request.time_col)
            mlflow.log_param("target_column", request.target_col)
            mlflow.log_param("covariates", str(request.covariates))
            mlflow.log_param("horizon", request.horizon)
            mlflow.log_param("frequency", request.frequency)
            mlflow.log_param("seasonality_mode", request.seasonality_mode)
            mlflow.log_param("regressor_method", request.regressor_method)
            mlflow.log_param("country", request.country)
            mlflow.log_param("models_trained", str(request.models))
            mlflow.log_param("eval_size", eval_size)
            mlflow.log_param("holdout_size", holdout_size)
            mlflow.log_param("total_data_points", len(df))
            mlflow.log_param("training_data_points", len(train_df))
            mlflow.log_param("eval_data_points", len(eval_df))
            mlflow.log_param("holdout_data_points", len(holdout_df) if holdout_size > 0 else 0)
            mlflow.log_param("random_seed", seed)
            logger.info(f"🌱 Logged random seed: {seed} to MLflow")

            # Log batch training context for easier tracking and filtering
            if request.batch_id:
                mlflow.set_tag("batch_id", request.batch_id)
                mlflow.set_tag("batch_mode", "true")
                mlflow.log_param("batch_segment_id", request.batch_segment_id or "unknown")
                mlflow.log_param("batch_segment_index", request.batch_segment_index or 0)
                mlflow.log_param("batch_total_segments", request.batch_total_segments or 0)
                logger.info(f"Batch context: segment {request.batch_segment_index}/{request.batch_total_segments} - {request.batch_segment_id}")
            else:
                mlflow.set_tag("batch_mode", "false")

            # Log data filters/context (if any filters were applied in the UI)
            # Note: Filters are applied client-side before sending data, so we log what we received
            mlflow.log_param("data_start_date", str(df['ds'].min()))
            mlflow.log_param("data_end_date", str(df['ds'].max()))
            
            # Log date range filters if provided
            if request.from_date:
                mlflow.log_param("filter_from_date", request.from_date)
                logger.info(f"📅 Logged date filter: from_date = {request.from_date}")
            mlflow.log_param("filter_to_date", effective_to_date)
            if request.to_date:
                logger.info(f"📅 Logged date filter: to_date = {request.to_date}")
            else:
                logger.info(f"📅 Logged date filter: to_date = {effective_to_date} (default: max date in dataset)")
            if request.from_date or request.to_date:
                mlflow.log_param("original_data_points", original_len)
                mlflow.log_param("filtered_data_points", len(df))
                logger.info(f"📅 Date filtering: {original_len} -> {len(df)} data points")
            
            # Log filter criteria if provided
            if request.filters:
                for filter_key, filter_value in request.filters.items():
                    logger.info(f"Logging filter to MLflow: {filter_key} = {filter_value}")
                    mlflow.log_param(f"filter_{filter_key}", str(filter_value))
                logger.info(f"Logged {len(request.filters)} filter criteria")
            else:
                logger.info("ℹ️ No data filters provided in request")
            
            logger.info(f"Logged training parameters to parent run {parent_run.info.run_id}")

            # Progress tracking
            total_models = len(request.models)
            completed_models = 0

            for model_idx, model_type in enumerate(request.models, 1):
                logger.info(f"")
                logger.info(f"{'='*60}")
                logger.info(f"📊 Training model {model_idx}/{total_models}: {model_type.upper()}")
                logger.info(f"{'='*60}")
                try:
                    result = None
                    if model_type == 'prophet':
                        run_id, _, metrics, val, fcst, uri, impacts = train_prophet_model(
                            request.data, request.time_col, request.target_col, safe_covariates,
                            request.horizon, request.frequency, request.seasonality_mode, test_size,
                            request.regressor_method, request.country, seed, request.future_features,
                            request.hyperparameter_filters,
                            train_df_override=train_df,  # Pass pre-split data
                            test_df_override=test_df,    # Pass pre-split data
                            forecast_start_date=to_date  # Forecast starts from user's to_date
                        )
                        result = ModelResult(
                            model_type='prophet', model_name='Prophet (MLflow)', run_id=run_id,
                            metrics=ForecastMetrics(
                                rmse=str(metrics['rmse']), mape=str(metrics['mape']), r2=str(metrics['r2']),
                                cv_mape=str(metrics['cv_mape']) if metrics.get('cv_mape') else None,
                                cv_mape_std=str(metrics['cv_mape_std']) if metrics.get('cv_mape_std') else None
                            ),
                            validation=val, forecast=fcst, covariate_impacts=[CovariateImpact(**i) for i in impacts],
                            is_best=False
                        )
                    elif model_type == 'arima':
                        run_id, _, metrics, val, fcst, uri, params = train_arima_model(
                            train_df, test_df, request.horizon, request.frequency, None, seed,
                            original_data=request.data, covariates=safe_covariates,
                            hyperparameter_filters=request.hyperparameter_filters,
                            forecast_start_date=to_date  # Forecast starts from user's to_date
                        )
                        val = val.rename(columns={'ds': request.time_col})
                        fcst = fcst.rename(columns={'ds': request.time_col})
                        result = ModelResult(
                            model_type='arima', model_name=f"ARIMA{params}" if params else "ARIMA", run_id=run_id,
                            metrics=ForecastMetrics(
                                rmse=str(metrics['rmse']), mape=str(metrics['mape']), r2=str(metrics['r2']),
                                cv_mape=str(metrics['cv_mape']) if metrics.get('cv_mape') else None,
                                cv_mape_std=str(metrics['cv_mape_std']) if metrics.get('cv_mape_std') else None
                            ),
                            validation=val.to_dict('records'), forecast=fcst.to_dict('records'), covariate_impacts=[], is_best=False
                        )
                    elif model_type == 'exponential_smoothing':
                        seasonal_periods = 52 if request.frequency == 'weekly' else 7 if request.frequency == 'daily' else 12
                        # ETS needs at least 2 seasonal cycles for reliable estimation (statsmodels requirement)
                        min_required = int(seasonal_periods * 2)
                        if len(train_df) < min_required:
                            logger.warning(f" Skipping ETS: Need at least {min_required} data points for seasonal={seasonal_periods}, but only have {len(train_df)}")
                            continue
                        run_id, _, metrics, val, fcst, uri, params = train_exponential_smoothing_model(
                            train_df, test_df, request.horizon, request.frequency, seasonal_periods, seed,
                            original_data=request.data, covariates=safe_covariates,
                            hyperparameter_filters=request.hyperparameter_filters,
                            forecast_start_date=to_date  # Forecast starts from user's to_date
                        )
                        val = val.rename(columns={'ds': request.time_col})
                        fcst = fcst.rename(columns={'ds': request.time_col})
                        result = ModelResult(
                            model_type='exponential_smoothing', model_name=f"ExpSmoothing({params.get('trend')}, {params.get('seasonal')})", run_id=run_id,
                            metrics=ForecastMetrics(
                                rmse=str(metrics['rmse']), mape=str(metrics['mape']), r2=str(metrics['r2']),
                                cv_mape=str(metrics['cv_mape']) if metrics.get('cv_mape') else None,
                                cv_mape_std=str(metrics['cv_mape_std']) if metrics.get('cv_mape_std') else None
                            ),
                            validation=val.to_dict('records'), forecast=fcst.to_dict('records'), covariate_impacts=[], is_best=False
                        )
                    elif model_type == 'sarimax':
                        # SARIMAX - Seasonal ARIMA with eXogenous variables (supports covariates)
                        run_id, _, metrics, val, fcst, uri, params = train_sarimax_model(
                            train_df, test_df, request.horizon, request.frequency,
                            covariates=safe_covariates, random_seed=seed, original_data=request.data,
                            country=request.country, hyperparameter_filters=request.hyperparameter_filters,
                            forecast_start_date=to_date  # Forecast starts from user's to_date
                        )
                        val = val.rename(columns={'ds': request.time_col})
                        fcst = fcst.rename(columns={'ds': request.time_col})
                        order_str = f"{params.get('order', '?')}" if params else "?"
                        seasonal_str = f"{params.get('seasonal_order', '?')}" if params else "?"
                        result = ModelResult(
                            model_type='sarimax', model_name=f"SARIMAX{order_str}x{seasonal_str}", run_id=run_id,
                            metrics=ForecastMetrics(
                                rmse=str(metrics['rmse']), mape=str(metrics['mape']), r2=str(metrics['r2']),
                                cv_mape=str(metrics['cv_mape']) if metrics.get('cv_mape') else None,
                                cv_mape_std=str(metrics['cv_mape_std']) if metrics.get('cv_mape_std') else None
                            ),
                            validation=val.to_dict('records'), forecast=fcst.to_dict('records'), covariate_impacts=[], is_best=False
                        )
                    elif model_type == 'xgboost':
                        # XGBoost - Gradient boosting with calendar features and full covariate support
                        run_id, _, metrics, val, fcst, uri, params = train_xgboost_model(
                            train_df, test_df, request.horizon, request.frequency,
                            covariates=safe_covariates, random_seed=seed, original_data=request.data,
                            country=request.country, hyperparameter_filters=request.hyperparameter_filters,
                            forecast_start_date=to_date  # Forecast starts from user's to_date
                        )
                        val = val.rename(columns={'ds': request.time_col})
                        fcst = fcst.rename(columns={'ds': request.time_col})
                        depth = params.get('max_depth', '?') if params else '?'
                        n_est = params.get('n_estimators', '?') if params else '?'
                        result = ModelResult(
                            model_type='xgboost', model_name=f"XGBoost(depth={depth}, n={n_est})", run_id=run_id,
                            metrics=ForecastMetrics(
                                rmse=str(metrics['rmse']), mape=str(metrics['mape']), r2=str(metrics['r2']),
                                cv_mape=str(metrics['cv_mape']) if metrics.get('cv_mape') else None,
                                cv_mape_std=str(metrics['cv_mape_std']) if metrics.get('cv_mape_std') else None
                            ),
                            validation=val.to_dict('records'), forecast=fcst.to_dict('records'), covariate_impacts=[], is_best=False
                        )

                    if result:
                        completed_models += 1
                        logger.info(f"✅ {model_type.upper()} completed - MAPE: {metrics['mape']:.2f}%")
                        logger.info(f"📈 Progress: {completed_models}/{total_models} models completed")
                        if metrics['mape'] < best_mape:
                            best_mape = metrics['mape']
                            best_model_name = result.model_name
                            best_run_id = run_id
                            artifact_uri_ref = uri
                            logger.info(f"🏆 New best model: {result.model_name} (MAPE: {best_mape:.2f}%)")
                        model_results.append(result)

                        # Validate MLflow artifacts for this model run
                        try:
                            validation_result = validate_mlflow_run_artifacts(run_id)
                            if not validation_result.get("validation_passed"):
                                logger.warning(f"   ⚠️ Artifact validation issues for {result.model_name}: {validation_result.get('issues', [])}")
                        except Exception as val_error:
                            logger.warning(f"   Could not validate artifacts: {val_error}")

                        # Cleanup after each model to prevent 'too many open files' errors
                        cleanup_temp_files_and_memory()

                except Exception as e:
                    logger.error(f"{model_type} failed: {e}", exc_info=True)
                    # Add failed model to results with error info so frontend can show why it failed
                    error_msg = str(e)[:300]  # Truncate long errors
                    failed_result = ModelResult(
                        model_type=model_type,
                        model_name=f"{model_type.upper()} (Failed)",
                        run_id="",
                        metrics=ForecastMetrics(rmse="N/A", mape="N/A", r2="N/A"),
                        validation=[],
                        forecast=[],
                        covariate_impacts=[],
                        is_best=False,
                        error=error_msg
                    )
                    model_results.append(failed_result)
                    logger.warning(f"Model {model_type} failed: {error_msg}")

        if not model_results: raise Exception("All models failed")

        # ========================================
        # HOLDOUT EVALUATION - Final Model Selection
        # ========================================
        # After all models are trained, evaluate them on holdout set
        # This provides unbiased estimate of model performance
        if holdout_size > 0 and len(model_results) > 0:
            logger.info(f"")
            logger.info(f"{'='*60}")
            logger.info(f"🔒 HOLDOUT EVALUATION - Final Model Selection")
            logger.info(f"{'='*60}")
            logger.info(f"   Evaluating {len(model_results)} models on holdout set ({holdout_size} rows)")
            logger.info(f"   Holdout period: {holdout_df['ds'].min()} to {holdout_df['ds'].max()}")

            holdout_results = []
            for res in model_results:
                if res.run_id and res.metrics.mape != "N/A":
                    try:
                        # Load the model and predict on holdout
                        import mlflow.pyfunc
                        model_uri = f"runs:/{res.run_id}/model"
                        loaded_model = mlflow.pyfunc.load_model(model_uri)

                        # Prepare holdout input
                        holdout_input = pd.DataFrame({
                            'periods': [holdout_size],
                            'start_date': [str(eval_df['ds'].max().date())]  # Start from end of eval
                        })

                        # Get predictions
                        holdout_predictions = loaded_model.predict(holdout_input)

                        # Calculate holdout MAPE
                        if isinstance(holdout_predictions, pd.DataFrame) and 'yhat' in holdout_predictions.columns:
                            # Align predictions with actuals
                            pred_df = holdout_predictions[['ds', 'yhat']].copy()
                            pred_df['ds'] = pd.to_datetime(pred_df['ds'])
                            actual_df = holdout_df[['ds', 'y']].copy()

                            merged = actual_df.merge(pred_df, on='ds', how='inner')
                            if len(merged) > 0:
                                holdout_mape = float(np.mean(np.abs((merged['y'] - merged['yhat']) / (merged['y'] + 1e-10))) * 100)
                                holdout_results.append({
                                    'model_name': res.model_name,
                                    'run_id': res.run_id,
                                    'eval_mape': float(res.metrics.mape),
                                    'holdout_mape': holdout_mape,
                                    'result': res
                                })
                                logger.info(f"   {res.model_name}: Eval MAPE={res.metrics.mape}%, Holdout MAPE={holdout_mape:.2f}%")
                            else:
                                logger.warning(f"   {res.model_name}: No overlapping dates for holdout evaluation")
                        else:
                            logger.warning(f"   {res.model_name}: Predictions format not supported for holdout eval")
                    except Exception as e:
                        logger.warning(f"   {res.model_name}: Holdout evaluation failed - {str(e)[:100]}")

            # Select best model based on holdout performance
            if holdout_results:
                holdout_results.sort(key=lambda x: x['holdout_mape'])
                best_holdout = holdout_results[0]
                logger.info(f"")
                logger.info(f"   🏆 Best model on HOLDOUT: {best_holdout['model_name']}")
                logger.info(f"      Holdout MAPE: {best_holdout['holdout_mape']:.2f}%")
                logger.info(f"      Eval MAPE: {best_holdout['eval_mape']:.2f}%")

                # Update best model selection
                best_model_name = best_holdout['model_name']
                best_mape = best_holdout['holdout_mape']
                best_run_id = best_holdout['run_id']

                # Log holdout metrics to MLflow
                with mlflow.start_run(run_id=parent_run_id):
                    mlflow.log_metric("best_holdout_mape", best_holdout['holdout_mape'])
                    mlflow.log_param("best_model_selected_by", "holdout")
                    for hr in holdout_results:
                        mlflow.log_metric(f"holdout_mape_{hr['model_name'].replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '_')}", hr['holdout_mape'])

                logger.info(f"{'='*60}")
            else:
                logger.warning(f"   ⚠️ No models could be evaluated on holdout. Using eval MAPE for selection.")
                logger.info(f"{'='*60}")
        else:
            if holdout_size == 0:
                logger.info(f"")
                logger.info(f"ℹ️ No holdout set available - using eval MAPE for model selection")

        # Log final training summary
        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"🎉 TRAINING COMPLETE - {len(model_results)}/{total_models} models succeeded")
        logger.info(f"{'='*60}")
        for res in model_results:
            status_icon = "🏆" if res.model_name == best_model_name else "  "
            mape_str = res.metrics.mape if res.metrics.mape != "N/A" else "Failed"
            logger.info(f"{status_icon} {res.model_name}: MAPE={mape_str}")
        logger.info(f"")
        logger.info(f"🏆 Best model: {best_model_name} (MAPE: {best_mape:.2f}%)")
        if holdout_size > 0:
            logger.info(f"   (Selected based on holdout performance)")
        else:
            logger.info(f"   (Selected based on eval performance)")
        logger.info(f"{'='*60}")

        # Build MLflow URLs for experiment and runs
        databricks_host = os.environ.get("DATABRICKS_HOST", "")
        if databricks_host:
            # Remove trailing slash if present
            databricks_host = databricks_host.rstrip("/")
            experiment_url = f"{databricks_host}/ml/experiments/{experiment_id}" if experiment_id else None
            logger.info(f"🔗 Experiment URL: {experiment_url}")
        else:
            experiment_url = None
            logger.warning("DATABRICKS_HOST not set, cannot generate MLflow URLs")

        for res in model_results:
            if res.model_name == best_model_name: res.is_best = True
            # Add experiment and run URLs
            if databricks_host and res.run_id:
                res.experiment_url = experiment_url
                res.run_url = f"{databricks_host}/ml/experiments/{experiment_id}/runs/{res.run_id}" if experiment_id else None
                logger.info(f"🔗 Run URL for {res.model_name}: {res.run_url}")

        # Register ALL models to Unity Catalog and run pre-deployment tests
        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"📦 REGISTERING AND TESTING MODELS")
        logger.info(f"{'='*60}")
        logger.info(f"Attempting to register {len(model_results)} models to Unity Catalog...")

        full_model_name = f"{request.catalog_name}.{request.schema_name}.{request.model_name}"

        for res in model_results:
            if not res.run_id:
                logger.error(f"Cannot register {res.model_name}: missing run_id")
                continue
            try:
                logger.info(f"📦 Registering {res.model_name} (Run: {res.run_id})...")
                # Exclude 'source' tag as it's restricted by Unity Catalog tag policies
                tags_to_set = {
                    "run_id": res.run_id,
                    "model_type": res.model_type,
                    "is_best": str(res.is_best),
                    "parent_run_id": parent_run_id or "",
                    "experiment_id": experiment_id or ""
                }
                version = register_model_to_unity_catalog(
                    f"runs:/{res.run_id}/model",
                    full_model_name,
                    tags_to_set
                )
                logger.info(f"   ✅ Registered {res.model_name} as version {version} in Unity Catalog")
                res.registered_version = str(version)

                # Run pre-deployment test on the registered model
                logger.info(f"   🧪 Testing {res.model_name} v{version}...")
                try:
                    test_result = test_model_inference(
                        model_name=full_model_name,
                        model_version=str(version),
                        test_periods=3,  # Quick test with 3 periods
                        frequency=request.frequency
                    )

                    res.test_result = ModelTestResult(
                        test_passed=test_result["test_passed"],
                        message=test_result["message"],
                        load_time_seconds=test_result.get("load_time_seconds"),
                        inference_time_seconds=test_result.get("inference_time_seconds"),
                        error_details=test_result.get("error_details")
                    )

                    if test_result["test_passed"]:
                        logger.info(f"   ✅ TEST PASSED (Load: {test_result.get('load_time_seconds', 0):.2f}s, Inference: {test_result.get('inference_time_seconds', 0):.3f}s)")
                    else:
                        logger.warning(f"   ❌ TEST FAILED: {test_result['message']}")
                        # Mark model as not deployable in tags
                        try:
                            import mlflow
                            mlflow.set_registry_uri("databricks-uc")
                            client = mlflow.MlflowClient()
                            client.set_model_version_tag(full_model_name, str(version), "test_passed", "false")
                            client.set_model_version_tag(full_model_name, str(version), "test_error", test_result.get("error_details", "Unknown error")[:250])
                        except Exception as tag_error:
                            logger.warning(f"   Could not set test result tags: {tag_error}")

                except Exception as test_error:
                    logger.error(f"   ❌ Test error for {res.model_name}: {test_error}")
                    res.test_result = ModelTestResult(
                        test_passed=False,
                        message=f"Test execution failed: {str(test_error)}",
                        error_details=str(test_error)
                    )

            except Exception as e:
                logger.error(f"Auto-register failed for {res.model_name}: {e}", exc_info=True)

        # Log summary of test results
        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"🧪 MODEL TEST SUMMARY")
        logger.info(f"{'='*60}")
        tested_models = [r for r in model_results if r.test_result]
        passed_models = [r for r in tested_models if r.test_result.test_passed]
        failed_models = [r for r in tested_models if not r.test_result.test_passed]

        logger.info(f"   Total tested: {len(tested_models)}")
        logger.info(f"   ✅ Passed: {len(passed_models)}")
        logger.info(f"   ❌ Failed: {len(failed_models)}")

        if passed_models:
            logger.info(f"   Deployable models:")
            for r in passed_models:
                logger.info(f"      - {r.model_name} v{r.registered_version}")
        if failed_models:
            logger.info(f"   ⚠️ Non-deployable models (failed tests):")
            for r in failed_models:
                logger.info(f"      - {r.model_name} v{r.registered_version}: {r.test_result.message[:100]}")
        logger.info(f"{'='*60}")

        # Prepare history data for chart visualization
        # Convert df back to original column names for the frontend
        history_df = df.copy()
        history_df = history_df.rename(columns={'ds': request.time_col, 'y': request.target_col})

        # Debug: Log date range and sample values to verify alignment
        if request.time_col in history_df.columns and len(history_df) > 0:
            logger.info(f"📅 History date range: {history_df[request.time_col].min()} to {history_df[request.time_col].max()}")
            # Log first and last few rows to verify data alignment
            if len(history_df) >= 3:
                logger.info(f"📅 First 3 dates: {list(history_df[request.time_col].head(3))}")
                logger.info(f"📅 Last 3 dates: {list(history_df[request.time_col].tail(3))}")
                logger.info(f"📊 First 3 target values: {list(history_df[request.target_col].head(3))}")
                logger.info(f"📊 Last 3 target values: {list(history_df[request.target_col].tail(3))}")

        # Convert datetime to string for JSON serialization
        if request.time_col in history_df.columns:
            history_df[request.time_col] = history_df[request.time_col].astype(str)
        history_data = history_df.to_dict('records')
        logger.info(f"📊 Returning {len(history_data)} history records for chart visualization")

        return TrainResponse(models=[m.dict() for m in model_results], best_model=best_model_name, artifact_uri=artifact_uri_ref or "N/A", history=history_data)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/executive-summary")
async def get_executive_summary_endpoint(request: dict):
    try:
        return {"summary": generate_executive_summary(
            request['bestModelName'], request['bestModelMetrics'], request['allModels'],
            request['targetCol'], request['timeCol'], request.get('covariates', []),
            request['forecastHorizon'], request['frequency'],
            request.get('actualsComparison')  # Optional actuals comparison data
        )}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/deploy", response_model=DeployResponse)
async def deploy_model(request: DeployRequest):
    try:
        model_version = request.model_version
        if request.run_id and not model_version:
            model_version = register_model_to_unity_catalog(
                f"runs:/{request.run_id}/model", request.model_name,
                {"source": "finance_forecasting_app", "run_id": request.run_id}
            )
        
        if not model_version: raise ValueError("Version or Run ID required")

        result = deploy_model_to_serving(
            request.model_name, model_version, request.endpoint_name,
            request.workload_size, request.scale_to_zero
        )
        result['deployed_version'] = model_version
        return DeployResponse(**result)
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test-model", response_model=TestModelResponse)
async def test_model_endpoint(request: TestModelRequest):
    """
    Test a registered model by loading it as pyfunc and running inference.
    This validates the model can be loaded and produces valid predictions
    before deploying to a serving endpoint.

    Use this endpoint to verify model functionality before deployment.
    """
    try:
        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"🧪 MODEL PRE-DEPLOYMENT TEST")
        logger.info(f"{'='*60}")
        logger.info(f"   Model: {request.model_name} v{request.model_version}")
        logger.info(f"   Test periods: {request.test_periods}")
        logger.info(f"   Start date: {request.start_date or 'auto (tomorrow)'}")
        logger.info(f"   Frequency: {request.frequency}")

        result = test_model_inference(
            model_name=request.model_name,
            model_version=request.model_version,
            test_periods=request.test_periods,
            start_date=request.start_date,
            frequency=request.frequency
        )

        if result["test_passed"]:
            logger.info(f"   ✅ TEST PASSED")
        else:
            logger.warning(f"   ❌ TEST FAILED: {result['message']}")

        logger.info(f"{'='*60}")

        return TestModelResponse(**result)
    except Exception as e:
        logger.error(f"Model test endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/endpoints/{endpoint_name}/status")
async def get_endpoint_status_api(endpoint_name: str):
    return JSONResponse(content=get_endpoint_status(endpoint_name))

@app.delete("/api/endpoints/{endpoint_name}")
async def delete_endpoint_api(endpoint_name: str):
    return JSONResponse(content=delete_endpoint(endpoint_name))

@app.get("/api/endpoints")
async def list_endpoints_api():
    return JSONResponse(content=list_endpoints())

@app.post("/api/register")
async def register_model_endpoint(model_uri: str, model_name: str = os.getenv("UC_MODEL_NAME", "main.default.finance_forecast_model")):
    try:
        version = register_model_to_unity_catalog(model_uri, model_name, {"source": "finance_forecasting_app"})
        return {"model_name": model_name, "version": version, "status": "registered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train-batch", response_model=BatchTrainResponse)
async def train_batch(request: BatchTrainRequest):
    """
    Train multiple forecasting models in parallel.

    This endpoint allows you to submit multiple training requests at once,
    processing them in parallel for faster batch forecasting across segments.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
    import traceback

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"🚀 BATCH TRAINING STARTED")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Total segments: {len(request.requests)}")
    logger.info(f"🔧 Requested workers: {request.max_workers}")

    # Limit max_workers based on environment (Databricks Apps has 4 vCPU limit)
    max_workers = min(request.max_workers, int(os.environ.get('MLFLOW_MAX_WORKERS', '2')))
    logger.info(f"⚙️ Using {max_workers} parallel workers")

    results = []

    def train_single(req: TrainRequest, index: int) -> BatchResultItem:
        """Train a single segment and return result."""
        segment_id = f"segment_{index}"
        if req.filters:
            segment_id = "_".join(f"{k}={v}" for k, v in req.filters.items())

        try:
            # Import here to avoid circular imports in thread
            import asyncio

            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Call the train_model function synchronously
                result = loop.run_until_complete(train_model(req))
                return BatchResultItem(
                    filters=req.filters,
                    segment_id=segment_id,
                    status="success",
                    result=result,
                    error=None
                )
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Batch training failed for {segment_id}: {e}")
            logger.error(traceback.format_exc())
            return BatchResultItem(
                filters=req.filters,
                segment_id=segment_id,
                status="error",
                result=None,
                error=str(e)
            )

    # Process requests in parallel with timeout protection
    # Each segment gets up to 10 minutes to complete
    SEGMENT_TIMEOUT_SECONDS = int(os.environ.get('BATCH_SEGMENT_TIMEOUT', '600'))  # 10 minutes default

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(train_single, req, i): i
            for i, req in enumerate(request.requests)
        }

        completed_segments = 0
        total_segments = len(request.requests)

        for future in as_completed(future_to_index, timeout=SEGMENT_TIMEOUT_SECONDS * total_segments):
            index = future_to_index[future]
            try:
                # Per-segment timeout
                result = future.result(timeout=SEGMENT_TIMEOUT_SECONDS)
                results.append(result)
                completed_segments += 1
                status_icon = "✅" if result.status == "success" else "❌"
                logger.info(f"{status_icon} Completed segment {result.segment_id} [{completed_segments}/{total_segments}]")
            except FuturesTimeoutError:
                logger.error(f"⏱️ Timeout for segment {index} after {SEGMENT_TIMEOUT_SECONDS}s")
                results.append(BatchResultItem(
                    filters=request.requests[index].filters,
                    segment_id=f"segment_{index}",
                    status="error",
                    result=None,
                    error=f"Training timeout after {SEGMENT_TIMEOUT_SECONDS} seconds"
                ))
            except Exception as e:
                logger.error(f"Unexpected error for segment {index}: {e}")
                results.append(BatchResultItem(
                    filters=request.requests[index].filters,
                    segment_id=f"segment_{index}",
                    status="error",
                    result=None,
                    error=str(e)
                ))

    # Sort results by original order
    results.sort(key=lambda x: int(x.segment_id.split("_")[-1]) if x.segment_id.startswith("segment_") else 0)

    successful = sum(1 for r in results if r.status == "success")
    failed = len(results) - successful

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"🎉 BATCH TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Successful: {successful}/{len(request.requests)} segments")
    logger.info(f"❌ Failed: {failed}/{len(request.requests)} segments")
    logger.info(f"{'='*60}")

    return BatchTrainResponse(
        total_requests=len(request.requests),
        successful=successful,
        failed=failed,
        results=results
    )


@app.post("/api/deploy-batch", response_model=BatchDeployResponse)
async def deploy_batch_models(request: BatchDeployRequest):
    """
    Deploy multiple segment models as a single router endpoint.

    This creates a PyFunc model that routes incoming requests to the appropriate
    segment-specific model based on the filter values in the request.
    The router model is registered to Unity Catalog and deployed to a serving endpoint.
    """
    import mlflow
    import json
    import tempfile
    import cloudpickle

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"🚀 BATCH DEPLOYMENT STARTED")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Segments to deploy: {len(request.segments)}")
    logger.info(f"🎯 Endpoint name: {request.endpoint_name}")

    try:
        # Build the segment routing table
        # Maps segment filter string -> model version
        segment_routing = {}
        for seg in request.segments:
            # Create a canonical key from filters (sorted for consistency)
            filter_key = "|".join(f"{k}={v}" for k, v in sorted(seg.filters.items()))
            segment_routing[filter_key] = {
                "segment_id": seg.segment_id,
                "model_version": seg.model_version,
                "filters": seg.filters
            }

        logger.info(f"📋 Routing table built with {len(segment_routing)} segments")

        # Full model name in Unity Catalog
        full_model_name = f"{request.catalog_name}.{request.schema_name}.{request.model_name}"
        router_model_name = f"{request.catalog_name}.{request.schema_name}.{request.model_name}_router"

        # Create the router PyFunc model
        class SegmentRouterModel(mlflow.pyfunc.PythonModel):
            """
            A router model that dispatches forecast requests to segment-specific models.

            Input format (DataFrame or dict):
            {
                "periods": 12,
                "start_date": "2024-01-01",
                "filters": {"region": "US", "product": "Widget"}  # Optional - identifies segment
            }

            The model will:
            1. Look up the appropriate segment model based on filters
            2. Load that model from Unity Catalog
            3. Return the forecast from that model
            """

            def __init__(self, routing_table, base_model_name):
                self.routing_table = routing_table
                self.base_model_name = base_model_name
                self._model_cache = {}

            def _get_filter_key(self, filters):
                """Convert filters dict to canonical key."""
                if not filters:
                    return None
                return "|".join(f"{k}={v}" for k, v in sorted(filters.items()))

            def _load_segment_model(self, model_version):
                """Load a segment model from Unity Catalog (with caching)."""
                cache_key = f"{self.base_model_name}@{model_version}"
                if cache_key not in self._model_cache:
                    model_uri = f"models:/{self.base_model_name}/{model_version}"
                    self._model_cache[cache_key] = mlflow.pyfunc.load_model(model_uri)
                return self._model_cache[cache_key]

            def predict(self, context, model_input):
                """
                Route prediction to appropriate segment model.

                model_input can be:
                - DataFrame with columns: periods, start_date, filters (JSON string)
                - Dict with keys: periods, start_date, filters
                """
                import pandas as pd
                import json

                # Handle DataFrame input
                if isinstance(model_input, pd.DataFrame):
                    if len(model_input) == 0:
                        return pd.DataFrame()

                    results = []
                    for _, row in model_input.iterrows():
                        periods = int(row.get('periods', 12))
                        start_date = str(row.get('start_date', ''))

                        # Parse filters
                        filters_raw = row.get('filters', {})
                        if isinstance(filters_raw, str):
                            try:
                                filters = json.loads(filters_raw)
                            except:
                                filters = {}
                        else:
                            filters = filters_raw if filters_raw else {}

                        # Find matching segment
                        filter_key = self._get_filter_key(filters)
                        segment_info = self.routing_table.get(filter_key)

                        if not segment_info:
                            # Try to find a default or return error
                            if len(self.routing_table) == 1:
                                # Only one segment, use it as default
                                segment_info = list(self.routing_table.values())[0]
                            else:
                                results.append({
                                    "error": f"No matching segment for filters: {filters}",
                                    "available_segments": list(self.routing_table.keys())
                                })
                                continue

                        # Load and call segment model
                        try:
                            segment_model = self._load_segment_model(segment_info['model_version'])

                            # Create input for segment model
                            segment_input = pd.DataFrame([{
                                'periods': periods,
                                'start_date': start_date
                            }])

                            forecast = segment_model.predict(segment_input)

                            # Add segment info to result
                            if isinstance(forecast, pd.DataFrame):
                                forecast_dict = forecast.to_dict('records')
                            else:
                                forecast_dict = forecast

                            results.append({
                                "segment_id": segment_info['segment_id'],
                                "filters": segment_info['filters'],
                                "forecast": forecast_dict
                            })
                        except Exception as e:
                            results.append({
                                "segment_id": segment_info.get('segment_id', 'unknown'),
                                "error": str(e)
                            })

                    return pd.DataFrame(results)

                # Handle dict input (single request)
                elif isinstance(model_input, dict):
                    periods = model_input.get('periods', 12)
                    start_date = model_input.get('start_date', '')
                    filters = model_input.get('filters', {})

                    filter_key = self._get_filter_key(filters)
                    segment_info = self.routing_table.get(filter_key)

                    if not segment_info and len(self.routing_table) == 1:
                        segment_info = list(self.routing_table.values())[0]

                    if not segment_info:
                        return {
                            "error": f"No matching segment for filters: {filters}",
                            "available_segments": list(self.routing_table.keys())
                        }

                    segment_model = self._load_segment_model(segment_info['model_version'])
                    segment_input = pd.DataFrame([{'periods': periods, 'start_date': start_date}])
                    forecast = segment_model.predict(segment_input)

                    return {
                        "segment_id": segment_info['segment_id'],
                        "filters": segment_info['filters'],
                        "forecast": forecast.to_dict('records') if hasattr(forecast, 'to_dict') else forecast
                    }

                else:
                    return {"error": f"Unsupported input type: {type(model_input)}"}

        # Create and log the router model
        mlflow.set_experiment(f"/Shared/finance-forecasting-router")

        with mlflow.start_run(run_name=f"router_{request.endpoint_name}") as run:
            # Log router metadata
            mlflow.log_param("num_segments", len(segment_routing))
            mlflow.log_param("endpoint_name", request.endpoint_name)
            mlflow.log_param("base_model_name", full_model_name)

            # Log routing table as artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(segment_routing, f, indent=2)
                routing_file = f.name
            mlflow.log_artifact(routing_file, "routing")
            os.unlink(routing_file)

            # Create the router model instance
            router_model = SegmentRouterModel(
                routing_table=segment_routing,
                base_model_name=full_model_name
            )

            # Define model signature
            from mlflow.models.signature import ModelSignature
            from mlflow.types.schema import Schema, ColSpec

            input_schema = Schema([
                ColSpec("long", "periods"),
                ColSpec("string", "start_date"),
                ColSpec("string", "filters")  # JSON string of filter dict
            ])

            output_schema = Schema([
                ColSpec("string", "segment_id"),
                ColSpec("string", "filters"),
                ColSpec("string", "forecast"),
                ColSpec("string", "error")
            ])

            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            # Log the model
            mlflow.pyfunc.log_model(
                artifact_path="router_model",
                python_model=router_model,
                signature=signature,
                pip_requirements=[
                    "mlflow>=2.0",
                    "pandas>=1.0",
                    "cloudpickle>=2.0"
                ]
            )

            router_run_id = run.info.run_id
            logger.info(f"✅ Router model logged with run_id: {router_run_id}")

        # Register the router model to Unity Catalog
        router_version = register_model_to_unity_catalog(
            f"runs:/{router_run_id}/router_model",
            router_model_name,
            {
                "source": "finance_forecasting_app",
                "type": "router",
                "num_segments": str(len(segment_routing)),
                "base_model": full_model_name
            }
        )

        logger.info(f"✅ Router model registered: {router_model_name} v{router_version}")

        # Deploy the router model to serving endpoint
        result = deploy_model_to_serving(
            router_model_name,
            router_version,
            request.endpoint_name,
            request.workload_size,
            request.scale_to_zero
        )

        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"🎉 BATCH DEPLOYMENT COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"📍 Endpoint: {request.endpoint_name}")
        logger.info(f"📊 Segments: {len(segment_routing)}")
        logger.info(f"🔗 Router model: {router_model_name} v{router_version}")
        logger.info(f"{'='*60}")

        return BatchDeployResponse(
            status="success",
            message=f"Router endpoint created with {len(segment_routing)} segments",
            endpoint_name=request.endpoint_name,
            endpoint_url=result.get('endpoint_url'),
            deployed_segments=len(segment_routing),
            router_model_version=router_version
        )

    except Exception as e:
        logger.error(f"Batch deployment failed: {e}", exc_info=True)
        return BatchDeployResponse(
            status="error",
            message=str(e),
            endpoint_name=request.endpoint_name
        )


@app.post("/api/aggregate", response_model=AggregateResponse)
async def aggregate_data(request: AggregateRequest):
    """
    Aggregate time series data from a higher frequency to a lower frequency.

    For example, convert daily data to weekly or monthly data before training.
    This is useful when you have daily data but want to forecast at a weekly level.
    """
    import pandas as pd

    logger.info(f"Aggregation request: {request.source_frequency} -> {request.target_frequency}")

    # Validate frequency conversion
    freq_order = {'daily': 0, 'weekly': 1, 'monthly': 2}
    if freq_order.get(request.source_frequency, -1) >= freq_order.get(request.target_frequency, -1):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot aggregate from {request.source_frequency} to {request.target_frequency}. Target must be lower frequency than source."
        )

    # Convert to DataFrame
    df = pd.DataFrame(request.data)
    original_rows = len(df)

    # Parse dates
    df[request.time_col] = pd.to_datetime(df[request.time_col])
    df = df.sort_values(request.time_col)

    # Determine pandas frequency string
    freq_map = {
        'weekly': 'W',
        'monthly': 'M'
    }
    target_freq = freq_map.get(request.target_frequency)
    if not target_freq:
        raise HTTPException(status_code=400, detail=f"Unsupported target frequency: {request.target_frequency}")

    # Build aggregation dictionary
    agg_dict = {}
    agg_methods_used = {}

    # Target column aggregation
    target_agg = request.aggregation_method
    agg_dict[request.target_col] = target_agg
    agg_methods_used[request.target_col] = target_agg

    # Covariate aggregation
    for cov in request.covariates:
        if cov not in df.columns:
            logger.warning(f"Covariate '{cov}' not found in data, skipping")
            continue

        if cov in request.covariate_aggregation:
            # Use specified aggregation method
            agg_method = request.covariate_aggregation[cov]
        else:
            # Auto-detect: binary columns use 'max', continuous use 'mean'
            unique_vals = df[cov].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
                agg_method = 'max'  # Binary flag - take max (1 if any 1 exists)
            else:
                agg_method = 'mean'  # Continuous - take mean

        agg_dict[cov] = agg_method
        agg_methods_used[cov] = agg_method

    # Set time column as index for resampling
    df = df.set_index(request.time_col)

    # Perform aggregation
    try:
        aggregated = df.resample(target_freq).agg(agg_dict)
        aggregated = aggregated.dropna()  # Remove periods with no data
        aggregated = aggregated.reset_index()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Aggregation failed: {str(e)}")

    # Convert back to records
    aggregated[request.time_col] = aggregated[request.time_col].dt.strftime('%Y-%m-%d')
    result_data = aggregated.to_dict('records')

    logger.info(f"Aggregation complete: {original_rows} -> {len(result_data)} rows")

    return AggregateResponse(
        data=result_data,
        original_rows=original_rows,
        aggregated_rows=len(result_data),
        source_frequency=request.source_frequency,
        target_frequency=request.target_frequency,
        aggregation_methods=agg_methods_used
    )


# SPA catch-all route - MUST be at the end after all API routes
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if full_path.startswith("api"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    return FileResponse(f"{static_dir}/index.html") if os.path.exists(f"{static_dir}/index.html") else JSONResponse(status_code=404, content={"error": "Frontend not found"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)

