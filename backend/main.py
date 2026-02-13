"""
FastAPI backend for Databricks Finance Forecasting Platform
"""
import os
import gc
import logging
import uuid
import numpy as np
import tempfile
import glob as glob_module
import hashlib
import json
from datetime import datetime
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
    DataAnalysisRequest, DataAnalysisResponse,
    ForecastAdvisorRequest, ForecastAdvisorResponse,
    HolidayAnalysisRequest, HolidayAnalysisResponse,
    # Multi-user architecture schemas
    SessionCreateRequest, SessionResponse, TrainAsyncRequest, TrainAsyncResponse,
    JobStatusResponse, JobResultsResponse, UserHistoryResponse, UserHistoryItem,
    ReproduceJobRequest, ReproduceJobResponse
)
from backend.utils.logging_utils import log_io, log_route_io_middleware
from backend.utils.pipeline_trace import PipelineTrace, store_trace, get_latest_trace, list_trace_ids
from backend.models.prophet import train_prophet_model, prepare_prophet_data
from backend.models.arima import train_arima_model, train_sarimax_model
from backend.models.ets import train_exponential_smoothing_model
from backend.models.xgboost import train_xgboost_model
from backend.models.utils import register_model_to_unity_catalog, validate_mlflow_run_artifacts

# Advanced forecasting models
try:
    from backend.models.statsforecast_models import train_statsforecast_model
    STATSFORECAST_AVAILABLE = True
except ImportError:
    STATSFORECAST_AVAILABLE = False
    train_statsforecast_model = None

try:
    from backend.models.chronos_model import train_chronos_model
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    train_chronos_model = None

try:
    from backend.models.ensemble import train_ensemble_model, create_ensemble_forecast
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    train_ensemble_model = None
    create_ensemble_forecast = None

# AutoML Best Practices (AutoGluon, Nixtla, Greykite patterns)
try:
    from backend.models.automl_utils import (
        check_data_quality,
        validate_forecast,
        detect_overfitting,
        DataQualityReport,
        ForecastValidationResult,
        OverfittingReport,
        log_automl_summary,
    )
    from backend.models.presets import (
        get_preset,
        get_preset_models,
        recommend_preset,
        log_preset_info,
        PresetConfig,
    )
    from backend.preprocessing import validate_data_quality
    AUTOML_UTILS_AVAILABLE = True
except ImportError as e:
    AUTOML_UTILS_AVAILABLE = False
    # Note: logger not yet defined at import time, will log after setup
    _automl_import_error = str(e)

try:
    from backend.models.conformal import ConformalPredictor, add_conformal_intervals
    CONFORMAL_AVAILABLE = True
except ImportError:
    CONFORMAL_AVAILABLE = False
    ConformalPredictor = None
    add_conformal_intervals = None
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

# Job Delegation API - Offload heavy training to dedicated cluster
try:
    from backend.services.job_api import router as job_api_router
    from backend.services.job_delegation import is_delegation_enabled
    JOB_DELEGATION_AVAILABLE = True
except ImportError:
    JOB_DELEGATION_AVAILABLE = False
    job_api_router = None
    is_delegation_enabled = lambda: False

# Multi-User Architecture Services (Lakebase PostgreSQL)
try:
    from backend.services.lakebase_client import (
        LakebaseClient,
        get_lakebase_client,
        close_lakebase_client,
        compute_data_hash,
    )
    from backend.services.session_manager import SessionManager, get_session_manager
    from backend.services.job_service import JobService, get_job_service
    from backend.services.history_service import HistoryService, get_history_service
    MULTIUSER_AVAILABLE = True
except ImportError as e:
    MULTIUSER_AVAILABLE = False
    # Note: logger not yet defined, will log after logger setup

# Configure logging with both console and file handlers
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'training.log')
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Create formatter
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Root logger setup
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Console handler (stdout)
console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
console_handler.setFormatter(log_formatter)

# File handler (writes to backend/logs/training.log)
file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

# Add handlers to root logger
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Debug file handler (only created when LOG_LEVEL=DEBUG to avoid disk usage)
if LOG_LEVEL == "DEBUG":
    from logging.handlers import RotatingFileHandler
    debug_log_file = os.path.join(LOG_DIR, 'debug.log')
    debug_handler = RotatingFileHandler(
        debug_log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8'
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(log_formatter)
    root_logger.addHandler(debug_handler)

logger = logging.getLogger(__name__)
logger.info(f"📝 Logging to file: {LOG_FILE}")

# Log deferred import warnings
if not AUTOML_UTILS_AVAILABLE:
    logger.warning(f"AutoML utilities not available: {_automl_import_error if '_automl_import_error' in dir() else 'unknown error'}")


@log_io
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


@log_io
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


@log_io
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

# Add request/response I/O logging middleware (active at DEBUG level only)
log_route_io_middleware(app)

# Mount static files
static_dir = "dist" if os.path.exists("dist") else "../dist"
if os.path.exists(static_dir):
    app.mount("/assets", StaticFiles(directory=f"{static_dir}/assets"), name="assets")

# Register Simple Mode routes
if SIMPLE_MODE_AVAILABLE and simple_mode_router:
    app.include_router(simple_mode_router)
    logger.info("✅ Simple Mode routes registered at /api/simple/*")

# Register Job Delegation API routes
if JOB_DELEGATION_AVAILABLE and job_api_router:
    app.include_router(job_api_router)
    delegation_status = "enabled" if is_delegation_enabled() else "disabled (set ENABLE_CLUSTER_DELEGATION=true)"
    logger.info(f"✅ Job Delegation API routes registered at /api/v2/jobs/* - Delegation: {delegation_status}")

# Log Multi-User Architecture availability
if MULTIUSER_AVAILABLE:
    logger.info("✅ Multi-User Architecture services available (Lakebase PostgreSQL)")
else:
    logger.info("⚠️ Multi-User Architecture services not available (optional - requires Lakebase)")

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
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
        mlflow.set_tracking_uri(tracking_uri)
        status_data["mlflow_enabled"] = True
    except Exception as e:
        logger.debug(f"MLflow setup failed (expected in dev): {type(e).__name__}")
    return HealthResponse(**status_data)


# =============================================================================
# DEBUG / PIPELINE TRACE ENDPOINTS
# =============================================================================

@app.get("/api/debug/pipeline-trace")
async def get_pipeline_trace_endpoint(trace_id: str = None):
    """Retrieve the most recent pipeline trace, or a specific one by trace_id."""
    trace_data = get_latest_trace(trace_id)
    if not trace_data:
        return {"error": "No pipeline traces available. Run a training first.", "available_traces": []}
    return trace_data


@app.get("/api/debug/pipeline-traces")
async def list_pipeline_traces():
    """List all stored pipeline trace IDs."""
    return {"trace_ids": list_trace_ids()}


@app.get("/api/debug/training-log")
async def get_training_log(lines: int = 500):
    """Return the last N lines of the training log file."""
    try:
        with open(LOG_FILE, 'r') as f:
            all_lines = f.readlines()
        return {"log": "".join(all_lines[-lines:]), "total_lines": len(all_lines)}
    except FileNotFoundError:
        return {"log": "", "total_lines": 0}


# =============================================================================
# MULTI-USER ARCHITECTURE ENDPOINTS (Lakebase PostgreSQL)
# =============================================================================

@app.post("/api/session/create", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """
    Create a new user session.

    Sessions provide isolated state management for concurrent users.
    Each session tracks execution history and enables reproducibility.
    """
    if not MULTIUSER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Multi-user architecture not available. Ensure Lakebase PostgreSQL is configured."
        )

    try:
        session_manager = get_session_manager()
        session_id = await session_manager.create_session(
            user_id=request.user_id,
            user_email=request.user_email,
            session_config=request.session_config or {}
        )

        session = await session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=500, detail="Failed to retrieve created session")

        logger.info(f"Created session {session_id} for user {request.user_id}")
        return SessionResponse(**session)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")


@app.get("/api/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session details by ID."""
    if not MULTIUSER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Multi-user architecture not available")

    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return SessionResponse(**session)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Session retrieval failed: {str(e)}")


@app.post("/api/train-async", response_model=TrainAsyncResponse)
async def train_async(request: TrainAsyncRequest):
    """
    Submit an async training job to the dedicated cluster.

    This endpoint returns immediately with a job_id for status polling.
    The actual training runs on a dedicated beefy cluster (64 vCPU, 256GB RAM)
    via Databricks Jobs, allowing heavy workloads without blocking the App.

    Flow:
    1. Validates session and stores upload data
    2. Creates execution history record in Lakebase
    3. Submits job to Databricks Jobs API
    4. Returns job_id for polling via /api/job/{job_id}/status
    """
    if not MULTIUSER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Multi-user architecture not available")

    try:
        import pandas as pd

        # Validate session
        session_manager = get_session_manager()
        session = await session_manager.validate_session(request.session_id)
        if not session:
            raise HTTPException(status_code=401, detail=f"Invalid or expired session: {request.session_id}")

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Compute data hash for reproducibility
        data_json = json.dumps(request.data, sort_keys=True, default=str)
        data_hash = hashlib.md5(data_json.encode()).hexdigest()

        # Store uploaded data and create upload record
        lakebase_client = get_lakebase_client()
        upload_id = str(uuid.uuid4())

        await lakebase_client.create_upload(
            upload_id=upload_id,
            user_id=request.user_id,
            session_id=request.session_id,
            file_name=f"training_data_{job_id[:8]}.json",
            columns=[col for col in request.data[0].keys()] if request.data else [],
            row_count=len(request.data),
            data_hash=data_hash
        )

        # Prepare request parameters for history
        request_params = {
            "time_col": request.time_col,
            "target_col": request.target_col,
            "covariates": request.covariates,
            "horizon": request.horizon,
            "frequency": request.frequency,
            "seasonality_mode": request.seasonality_mode,
            "models": request.models,
            "random_seed": request.random_seed,
            "confidence_level": request.confidence_level,
            "hyperparameter_filters": request.hyperparameter_filters,
        }

        # Create execution history record
        history_service = get_history_service()
        await history_service.create_execution(
            job_id=job_id,
            session_id=request.session_id,
            user_id=request.user_id,
            request_params=request_params,
            data_upload_id=upload_id,
            data_row_count=len(request.data),
            data_hash=data_hash
        )

        # Submit to Databricks Jobs
        job_service = get_job_service()
        run_id = await job_service.submit_training_job(
            job_id=job_id,
            session_id=request.session_id,
            user_id=request.user_id,
            data_upload_id=upload_id,
            train_request=request_params,
            priority=request.priority
        )

        # Update execution record with Databricks run ID
        await history_service.update_status(
            job_id=job_id,
            status="RUNNING",
            databricks_run_id=run_id
        )

        # Update session activity
        await session_manager.update_activity(request.session_id)

        logger.info(f"Submitted async training job {job_id} (Databricks run: {run_id}) for user {request.user_id}")

        return TrainAsyncResponse(
            job_id=job_id,
            databricks_run_id=run_id,
            status="SUBMITTED",
            message="Training job submitted to dedicated cluster",
            poll_url=f"/api/job/{job_id}/status"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Async training submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training submission failed: {str(e)}")


@app.get("/api/job/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Poll for job completion status.

    Returns current job state and progress information.
    When status is COMPLETED, use /api/job/{job_id}/results to get forecast data.
    """
    if not MULTIUSER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Multi-user architecture not available")

    try:
        history_service = get_history_service()
        job_service = get_job_service()

        # Get execution record from Lakebase
        execution = await history_service.get_execution(job_id)
        if not execution:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        # If still running, sync status with Databricks
        if execution.get("status") == "RUNNING" and execution.get("databricks_run_id"):
            await job_service.sync_job_status(job_id)
            # Re-fetch updated status
            execution = await history_service.get_execution(job_id)

        return JobStatusResponse(
            job_id=job_id,
            status=execution.get("status", "UNKNOWN"),
            databricks_run_id=execution.get("databricks_run_id"),
            submitted_at=execution.get("submitted_at", ""),
            started_at=execution.get("started_at"),
            completed_at=execution.get("completed_at"),
            duration_seconds=execution.get("duration_seconds"),
            progress_message=execution.get("progress_message"),
            error_message=execution.get("error_message"),
            best_model=execution.get("best_model"),
            best_mape=execution.get("best_mape"),
            mlflow_run_id=execution.get("mlflow_run_id")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


@app.get("/api/job/{job_id}/results", response_model=JobResultsResponse)
async def get_job_results(job_id: str):
    """
    Get forecast results for a completed job.

    Returns full model results, forecasts, and validation data.
    Only available when job status is COMPLETED.
    """
    if not MULTIUSER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Multi-user architecture not available")

    try:
        history_service = get_history_service()
        lakebase_client = get_lakebase_client()

        # Get execution record
        execution = await history_service.get_execution(job_id)
        if not execution:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if execution.get("status") != "COMPLETED":
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is not completed (status: {execution.get('status')})"
            )

        # Get forecast results from Lakebase
        results = await lakebase_client.get_forecast_results(job_id)
        if not results:
            raise HTTPException(status_code=404, detail=f"No results found for job {job_id}")

        # Find best model result
        best_model = execution.get("best_model", "")
        best_result = next((r for r in results if r.get("model_name") == best_model), results[0] if results else None)

        return JobResultsResponse(
            job_id=job_id,
            status="COMPLETED",
            models=[{
                "model_name": r.get("model_name"),
                "mape": r.get("mape"),
                "rmse": r.get("rmse"),
                "mae": r.get("mae"),
                "r2": r.get("r2"),
                "mlflow_run_id": r.get("mlflow_run_id")
            } for r in results],
            best_model=best_model,
            forecast=json.loads(best_result.get("forecast_json", "[]")) if best_result else [],
            validation=json.loads(best_result.get("validation_json", "[]")) if best_result else [],
            mlflow_run_id=execution.get("mlflow_run_id", ""),
            mlflow_experiment_url=execution.get("mlflow_experiment_url")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job results retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Results retrieval failed: {str(e)}")


@app.get("/api/history/{user_id}", response_model=UserHistoryResponse)
async def get_user_history(user_id: str, limit: int = 50):
    """
    Get execution history for a user.

    Returns a list of past training jobs with their status and results summary.
    Useful for reproducibility and tracking past forecasts.
    """
    if not MULTIUSER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Multi-user architecture not available")

    try:
        session_manager = get_session_manager()
        history = await session_manager.get_user_history(user_id, limit=limit)

        return UserHistoryResponse(
            user_id=user_id,
            total_executions=len(history),
            executions=[
                UserHistoryItem(
                    job_id=item.get("job_id", ""),
                    submitted_at=item.get("submitted_at", ""),
                    status=item.get("status", "UNKNOWN"),
                    best_model=item.get("best_model"),
                    best_mape=item.get("best_mape"),
                    duration_seconds=item.get("duration_seconds"),
                    horizon=item.get("horizon", 0),
                    frequency=item.get("frequency", ""),
                    models=item.get("models", [])
                )
                for item in history
            ]
        )

    except Exception as e:
        logger.error(f"User history retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")


@app.post("/api/job/{job_id}/reproduce", response_model=ReproduceJobResponse)
async def reproduce_job(job_id: str, request: ReproduceJobRequest):
    """
    Reproduce a previous execution with exact same parameters.

    Creates a new job with identical configuration to the original.
    Useful for validating reproducibility or re-running with updated data.
    """
    if not MULTIUSER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Multi-user architecture not available")

    try:
        history_service = get_history_service()
        job_service = get_job_service()
        session_manager = get_session_manager()

        # Validate session
        session = await session_manager.validate_session(request.session_id)
        if not session:
            raise HTTPException(status_code=401, detail=f"Invalid or expired session: {request.session_id}")

        # Get reproduction parameters
        repro_params = await history_service.get_reproduction_params(job_id)
        if not repro_params:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or cannot be reproduced")

        original_params = repro_params.get("original_request", {})
        original_data_hash = repro_params.get("data_hash", "")
        data_upload_id = repro_params.get("data_upload_id", "")

        # Generate new job ID
        new_job_id = str(uuid.uuid4())

        # Create execution history record for new job
        await history_service.create_execution(
            job_id=new_job_id,
            session_id=request.session_id,
            user_id=request.user_id,
            request_params=original_params,
            data_upload_id=data_upload_id,
            data_row_count=repro_params.get("data_row_count", 0),
            data_hash=original_data_hash,
            reproduced_from=job_id
        )

        # Submit to Databricks Jobs
        run_id = await job_service.submit_training_job(
            job_id=new_job_id,
            session_id=request.session_id,
            user_id=request.user_id,
            data_upload_id=data_upload_id,
            train_request=original_params,
            priority="NORMAL"
        )

        # Update execution record with Databricks run ID
        await history_service.update_status(
            job_id=new_job_id,
            status="RUNNING",
            databricks_run_id=run_id
        )

        logger.info(f"Reproduced job {job_id} as {new_job_id} for user {request.user_id}")

        return ReproduceJobResponse(
            new_job_id=new_job_id,
            reproduced_from=job_id,
            original_params=original_params,
            data_hash_match=True,  # Will be verified during training
            status="SUBMITTED",
            poll_url=f"/api/job/{new_job_id}/status"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job reproduction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reproduction failed: {str(e)}")


# =============================================================================
# END MULTI-USER ARCHITECTURE ENDPOINTS
# =============================================================================


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


@app.post("/api/detect-dimensions")
async def detect_dimensions(request: dict):
    """
    Detect dimension columns in multi-dimensional data.

    This endpoint analyzes the data and identifies:
    - Dimension columns (categorical columns suitable for grouping/filtering)
    - Numeric measure columns (suitable for aggregation)
    - Whether aggregation is recommended based on data structure

    Call this before training to understand data structure and configure aggregation.
    """
    try:
        import pandas as pd
        from backend.preprocessing import detect_dimension_columns

        data = request.get('data', [])
        time_col = request.get('time_col', 'ds')
        target_col = request.get('target_col', 'y')

        if not data:
            raise HTTPException(status_code=400, detail="No data provided")

        df = pd.DataFrame(data)

        logger.info(f"🔍 Detecting dimensions: {len(df)} rows, time_col={time_col}, target_col={target_col}")

        if time_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Time column '{time_col}' not found")
        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found")

        # Detect dimensions
        result = detect_dimension_columns(df, time_col, target_col)

        # Convert to response format
        dimensions_response = {}
        for col, info in result['dimensions'].items():
            dimensions_response[col] = {
                'n_unique': info['n_unique'],
                'type': info['type'],
                'sample_values': [str(v) for v in info['sample_values']],
                'null_count': info['null_count']
            }

        # Check if aggregation is recommended
        unique_dates = df[time_col].nunique()
        row_count = len(df)
        aggregation_recommended = row_count > unique_dates

        recommendation = ""
        if aggregation_recommended:
            ratio = row_count / unique_dates
            recommendation = (
                f"Data has {row_count:,} rows but only {unique_dates} unique dates "
                f"(~{ratio:.0f} rows per date). Aggregation is recommended. "
                f"Detected dimensions: {list(result['dimensions'].keys())}"
            )

        return {
            'dimensions': dimensions_response,
            'numeric_measures': result['numeric_measures'],
            'date_col': time_col,
            'target_col': target_col,
            'row_count': row_count,
            'unique_dates': unique_dates,
            'aggregation_recommended': aggregation_recommended,
            'recommendation': recommendation
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dimension detection failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Dimension detection failed: {str(e)}")


@app.post("/api/forecast-advisor", response_model=ForecastAdvisorResponse)
async def forecast_advisor(request: ForecastAdvisorRequest):
    """
    Smart Forecast Advisor — analyzes all data slices using research-backed
    forecastability metrics (spectral entropy, STL-based trend/seasonal strength),
    recommends model selection, training windows, and aggregation levels.

    Call this before training to get data-driven recommendations across all slices.
    """
    try:
        import pandas as pd
        from backend.forecast_advisor import ForecastAdvisor

        logger.info(
            f"Forecast Advisor: {len(request.data)} rows, target={request.target_col}, "
            f"dimensions={request.dimension_cols}"
        )

        df = pd.DataFrame(request.data)

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data provided")

        if request.time_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Time column '{request.time_col}' not found in data"
            )
        if request.target_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{request.target_col}' not found in data"
            )
        for col in request.dimension_cols:
            if col not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dimension column '{col}' not found in data"
                )

        advisor = ForecastAdvisor()
        result = advisor.analyze_all_slices(
            df=df,
            time_col=request.time_col,
            target_col=request.target_col,
            dimension_cols=request.dimension_cols,
            frequency=request.frequency,
            horizon=request.horizon,
        )

        # Convert dataclass result to response model
        slice_analyses = []
        for sa in result.slice_analyses:
            slice_analyses.append({
                "slice_name": sa.slice_name,
                "filters": sa.filters,
                "forecastability_score": sa.forecastability_score,
                "grade": sa.grade.value,
                "spectral_entropy": sa.spectral_entropy,
                "trend_strength": sa.trend_strength,
                "seasonal_strength": sa.seasonal_strength,
                "total_growth_pct": sa.total_growth_pct,
                "recent_growth_pct": sa.recent_growth_pct,
                "data_quality": {
                    "n_observations": sa.data_quality.n_observations,
                    "has_sufficient_history": sa.data_quality.has_sufficient_history,
                    "missing_pct": sa.data_quality.missing_pct,
                    "gap_count": sa.data_quality.gap_count,
                    "anomalous_week_count": sa.data_quality.anomalous_week_count,
                    "anomalous_weeks": sa.data_quality.anomalous_weeks,
                    "volume_level": sa.data_quality.volume_level,
                    "weekly_mean": sa.data_quality.weekly_mean,
                    "warnings": sa.data_quality.warnings,
                },
                "recommended_models": sa.recommended_models,
                "excluded_models": sa.excluded_models,
                "model_exclusion_reasons": sa.model_exclusion_reasons,
                "recommended_training_window": sa.recommended_training_window,
                "training_window_reason": sa.training_window_reason,
                "expected_mape_range": list(sa.expected_mape_range),
            })

        agg_recs = []
        for ar in result.aggregation_recommendations:
            agg_recs.append({
                "from_slices": ar.from_slices,
                "to_slice": ar.to_slice,
                "reason": ar.reason,
                "combined_forecastability_score": ar.combined_forecastability_score,
                "improvement_pct": ar.improvement_pct,
            })

        return ForecastAdvisorResponse(
            slice_analyses=slice_analyses,
            aggregation_recommendations=agg_recs,
            summary=result.summary,
            overall_data_quality=result.overall_data_quality,
            total_slices=result.total_slices,
            forecastable_slices=result.forecastable_slices,
            problematic_slices=result.problematic_slices,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast advisor failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Forecast advisor failed: {str(e)}")


@app.post("/api/holiday-analysis", response_model=HolidayAnalysisResponse)
async def holiday_analysis(request: HolidayAnalysisRequest):
    """
    Holiday and Event Impact Analysis — detects and quantifies holiday/event
    impacts using STL remainder analysis, then matches anomalous weeks to known
    holidays with impact quantification.

    Call this before training to understand holiday effects in the data.
    """
    try:
        import pandas as pd
        from backend.holiday_analyzer import HolidayAnalyzer

        logger.info(
            f"Holiday Analysis: {len(request.data)} rows, target={request.target_col}"
        )

        df = pd.DataFrame(request.data)

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data provided")

        if request.time_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Time column '{request.time_col}' not found in data"
            )
        if request.target_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{request.target_col}' not found in data"
            )

        analyzer = HolidayAnalyzer()
        result = analyzer.analyze(
            df=df,
            time_col=request.time_col,
            target_col=request.target_col,
            frequency=request.frequency,
            country=request.country,
        )

        # Convert dataclass result to response model
        holiday_impacts = []
        for hi in result.holiday_impacts:
            holiday_impacts.append({
                "holiday_name": hi.holiday_name,
                "avg_lift_pct": hi.avg_lift_pct,
                "consistency": hi.consistency,
                "direction": hi.direction,
                "confidence": hi.confidence,
                "yearly_impacts": {str(k): v for k, v in hi.yearly_impacts.items()},
                "recommendation": hi.recommendation,
            })

        anomalous_events = []
        for ae in result.anomalous_events:
            anomalous_events.append({
                "week_date": ae.week_date,
                "deviation_pct": ae.deviation_pct,
                "direction": ae.direction,
                "matched_holiday": ae.matched_holiday,
                "is_recurring": ae.is_recurring,
                "note": ae.note,
            })

        return HolidayAnalysisResponse(
            holiday_impacts=holiday_impacts,
            anomalous_events=anomalous_events,
            summary=result.summary,
            training_recommendations=result.training_recommendations,
            detected_partial_weeks=result.detected_partial_weeks,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Holiday analysis failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Holiday analysis failed: {str(e)}")


@app.post("/api/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    try:
        import pandas as pd

        # TRUNCATE LOG FILE at the start of each training run (unless batch training)
        if not request.batch_id:
            truncate_log_file()

        # Initialize pipeline trace for debugging
        trace = PipelineTrace(target_col=request.target_col, time_col=request.time_col)

        logger.info(f"")
        logger.info(f"{'#'*70}")
        logger.info(f"#  NEW TRAINING RUN STARTED (trace_id={trace.trace_id})")
        logger.info(f"{'#'*70}")
        logger.info(f"Training request: target={request.target_col}, horizon={request.horizon}, data_rows={len(request.data)}")

        # TRACE Step 0: Request parameters
        trace.add_step("REQUEST_PARAMS", details={
            "target_col": request.target_col,
            "time_col": request.time_col,
            "horizon": request.horizon,
            "frequency": request.frequency,
            "seasonality_mode": request.seasonality_mode,
            "covariates": request.covariates or [],
            "models": request.models,
            "from_date": request.from_date,
            "to_date": request.to_date,
            "filters": request.filters or {},
            "random_seed": request.random_seed,
            "regressor_method": request.regressor_method,
            "data_rows_received": len(request.data),
        })

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

        # TRACE Step 1: Raw input
        trace.add_step("RAW_INPUT", raw_input_df, request.target_col, request.time_col,
                        request.covariates, {"source": "frontend_payload"})

        # ========================================
        # DATA AGGREGATION (for multi-dimensional data)
        # ========================================
        # If aggregation config is provided, aggregate the data before forecasting
        # This handles datasets with multiple dimensions (e.g., region, segment, product)
        aggregation_report = None
        if request.aggregation and request.aggregation.enabled:
            try:
                from backend.preprocessing import (
                    prepare_data_with_aggregation,
                    detect_dimension_columns,
                    aggregate_time_series
                )

                logger.info(f"")
                logger.info(f"{'='*70}")
                logger.info(f"🔄 AGGREGATION ENABLED")
                logger.info(f"{'='*70}")
                logger.info(f"   Group by: {request.aggregation.group_by_cols or 'None (total)'}")
                logger.info(f"   Aggregation method: {request.aggregation.agg_method}")
                logger.info(f"   Dimension filters: {request.aggregation.filter_dimensions or 'None'}")
                logger.info(f"   Auto-detect incomplete: {request.aggregation.auto_detect_incomplete}")

                # First, detect available dimensions
                dim_info = detect_dimension_columns(
                    raw_input_df,
                    request.time_col,
                    request.target_col
                )
                logger.info(f"   Detected dimensions: {list(dim_info['dimensions'].keys())}")

                # Perform aggregation with data quality checks
                # Pass covariates so they're preserved during aggregation
                raw_input_df, aggregation_report = prepare_data_with_aggregation(
                    df=raw_input_df,
                    date_col=request.time_col,
                    target_col=request.target_col,
                    group_by_cols=request.aggregation.group_by_cols,
                    agg_method=request.aggregation.agg_method,
                    filter_dimensions=request.aggregation.filter_dimensions,
                    auto_detect_incomplete=request.aggregation.auto_detect_incomplete,
                    frequency=request.frequency,
                    covariate_cols=request.covariates  # Preserve covariates during aggregation
                )

                # Update request.data with aggregated data for downstream processing
                request.data = raw_input_df.to_dict(orient='records')

                log_dataframe_summary(raw_input_df, "AGGREGATED DATA (after dimension aggregation)")

                # TRACE Step 2: After aggregation
                trace.add_step("AFTER_AGGREGATION", raw_input_df, request.target_col, request.time_col,
                                request.covariates, {
                                    "aggregation_method": request.aggregation.agg_method,
                                    "group_by_cols": request.aggregation.group_by_cols,
                                    "filter_dimensions": request.aggregation.filter_dimensions,
                                    "aggregation_report_keys": list(aggregation_report.keys()) if aggregation_report else [],
                                })

                # Log incomplete data detection results
                if aggregation_report.get('incomplete_detection', {}).get('n_dropped', 0) > 0:
                    inc_report = aggregation_report['incomplete_detection']
                    logger.warning(f"⚠️ INCOMPLETE DATA REMOVED:")
                    logger.warning(f"   Dropped {inc_report['n_dropped']} incomplete period(s)")
                    logger.warning(f"   Dropped dates: {inc_report['dropped_dates']}")
                    logger.info(f"   This prevents models from detecting false 'crashes' in the data")

            except Exception as e:
                logger.error(f"❌ Aggregation failed: {e}")
                logger.warning("   Continuing with raw data (no aggregation)")
                import traceback
                logger.error(traceback.format_exc())

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

        # Confidence level for prediction intervals (default 0.95 = 95%)
        # Lower values (e.g., 0.80) give narrower uncertainty ranges
        confidence_level = getattr(request, 'confidence_level', None) or 0.95
        logger.info(f"📊 Confidence level for prediction intervals: {confidence_level*100:.0f}%")

        # Inject confidence_level into hyperparameter_filters so all models can access it
        hp_filters = dict(request.hyperparameter_filters or {})
        hp_filters['_global'] = hp_filters.get('_global', {})
        hp_filters['_global']['confidence_level'] = confidence_level
        request_hp_filters = hp_filters

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

        # TRACE Step 3: After prepare_prophet_data (column renaming, type conversion)
        trace.add_step("AFTER_PREPARE_DATA", df, "y", "ds", safe_covariates, {
            "original_time_col": request.time_col,
            "original_target_col": request.target_col,
            "renamed_to": {"time": "ds", "target": "y"},
            "covariates_present": [c for c in safe_covariates if c in df.columns],
            "covariates_missing": [c for c in safe_covariates if c not in df.columns],
        })

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

        # TRACE Step 4: After date filtering
        trace.add_step("AFTER_DATE_FILTER", df, "y", "ds", safe_covariates, {
            "from_date": request.from_date or "(all)",
            "to_date": request.to_date,
            "effective_to_date": effective_to_date,
            "rows_before_filter": original_len,
            "rows_after_filter": len(df),
            "rows_dropped": original_len - len(df),
        })

        # ========================================
        # DATA QUALITY VALIDATION (AutoML Best Practices)
        # ========================================
        # Implements Greykite/AutoTS-style data quality checks:
        # - Minimum data requirements
        # - Missing value detection/handling
        # - Outlier detection (IQR method)
        # - Duplicate date detection
        # - Time gap detection
        # - Variance checks
        if AUTOML_UTILS_AVAILABLE:
            try:
                df, data_quality_report = validate_data_quality(
                    df=df,
                    date_col='ds',
                    target_col='y',
                    frequency=request.frequency,
                    auto_fix=True  # Automatically fix minor issues
                )

                # Store for later use in ensemble creation
                historical_mean = data_quality_report.stats.get('mean', df['y'].mean())
                historical_std = data_quality_report.stats.get('std', df['y'].std())

                # Log any critical issues
                if not data_quality_report.is_valid:
                    logger.warning(f"⚠️ Data quality issues detected: {data_quality_report.issues}")
                    # Continue anyway but warn user
                if data_quality_report.recommendations:
                    for rec in data_quality_report.recommendations:
                        logger.info(f"   💡 Recommendation: {rec}")
            except Exception as e:
                logger.warning(f"Data quality validation failed: {e}. Continuing with original data.")
                historical_mean = df['y'].mean()
                historical_std = df['y'].std()
                data_quality_report = None
        else:
            # Fallback if automl_utils not available
            historical_mean = df['y'].mean()
            historical_std = df['y'].std()
            data_quality_report = None

        # TRACE Step 5: After data quality validation
        trace.add_step("AFTER_DATA_QUALITY", df, "y", "ds", safe_covariates, {
            "quality_valid": data_quality_report.is_valid if data_quality_report else "N/A",
            "quality_issues": data_quality_report.issues[:5] if data_quality_report and data_quality_report.issues else [],
            "transformations_applied": data_quality_report.transformations_applied[:5] if data_quality_report and hasattr(data_quality_report, 'transformations_applied') and data_quality_report.transformations_applied else [],
            "historical_mean": float(historical_mean) if historical_mean else None,
            "historical_std": float(historical_std) if historical_std else None,
        })

        # ========================================
        # SMART AUTO-OPTIMIZATION (Forecast Advisor)
        # ========================================
        # When auto_optimize=True (default), analyze data characteristics
        # and automatically configure training window, model selection,
        # log transform, and horizon guidance. Generic and dataset-agnostic.
        auto_optimize_info = None
        auto_optimize_enabled = getattr(request, 'auto_optimize', True)

        if auto_optimize_enabled and len(df) >= 52:
            try:
                from backend.forecast_advisor import ForecastAdvisor
                advisor = ForecastAdvisor()

                # Run advisor on the prepared data
                advisor_values = df['y'].dropna().values.astype(float)
                advisor_dates = pd.to_datetime(df['ds'])
                advisor_config = advisor.auto_configure_training(
                    values=advisor_values,
                    dates=advisor_dates,
                    frequency=request.frequency,
                    requested_horizon=request.horizon,
                )

                logger.info(f"")
                logger.info(f"{'='*60}")
                logger.info(f"AUTO-OPTIMIZE: Forecast Advisor Results")
                logger.info(f"{'='*60}")
                logger.info(f"   Forecastability: {advisor_config.grade} (score {advisor_config.forecastability_score}/100)")
                logger.info(f"   Growth: {advisor_config.growth_pct:.1f}%")
                logger.info(f"   Training window: {advisor_config.training_window_weeks or 'all'} weeks")
                logger.info(f"   Log transform: {advisor_config.log_transform}")
                logger.info(f"   Recommended models: {advisor_config.recommended_models}")
                logger.info(f"   Recommended horizon: {advisor_config.recommended_horizon} (max reliable: {advisor_config.max_reliable_horizon})")
                logger.info(f"   Expected MAPE: {advisor_config.expected_mape_range[0]:.1f}%-{advisor_config.expected_mape_range[1]:.1f}%")
                if advisor_config.data_warnings:
                    for w in advisor_config.data_warnings[:3]:
                        logger.info(f"   Warning: {w}")
                logger.info(f"{'='*60}")

                # Apply training window: trim old data if advisor recommends
                if advisor_config.from_date and not request.from_date:
                    advisor_from = pd.Timestamp(advisor_config.from_date)
                    before_count = len(df)
                    df = df[df['ds'] >= advisor_from].reset_index(drop=True)
                    logger.info(
                        f"   Auto-optimize: trimmed training data from {before_count} to {len(df)} rows "
                        f"(from_date={advisor_config.from_date}, window={advisor_config.training_window_weeks}w)"
                    )

                # Apply model selection: intersect advisor recommendations with user's list
                # Only modify if user used defaults (didn't explicitly pick models)
                user_models = request.models
                default_models = ["prophet"]
                if user_models == default_models or set(user_models) == set(default_models):
                    # User used defaults — apply advisor's full recommendation
                    request.models = advisor_config.recommended_models
                    logger.info(f"   Auto-optimize: models set to {request.models} (advisor recommended)")
                else:
                    # User explicitly chose models — warn about excluded ones but respect choice
                    excluded_in_user = [m for m in user_models if m in advisor_config.excluded_models]
                    if excluded_in_user:
                        logger.warning(
                            f"   Auto-optimize: user selected {excluded_in_user} which advisor excludes. "
                            f"Reasons: {', '.join(advisor_config.model_exclusion_reasons.get(m, '?') for m in excluded_in_user)}"
                        )

                # Apply log transform recommendation (only if user left it on 'auto')
                user_log_transform = getattr(request, 'log_transform', 'auto') or 'auto'
                if user_log_transform == 'auto':
                    # Advisor provides more nuanced recommendation than the simple auto-detect
                    request.log_transform = advisor_config.log_transform
                    logger.info(f"   Auto-optimize: log_transform set to '{advisor_config.log_transform}' ({advisor_config.log_transform_reason})")

                # Build auto_optimize_info for response
                from backend.schemas import AutoOptimizeInfo
                auto_optimize_info = AutoOptimizeInfo(
                    enabled=True,
                    forecastability_score=advisor_config.forecastability_score,
                    grade=advisor_config.grade,
                    training_window_weeks=advisor_config.training_window_weeks,
                    from_date_applied=advisor_config.from_date if not request.from_date else None,
                    log_transform=advisor_config.log_transform,
                    models_selected=advisor_config.recommended_models,
                    models_excluded=advisor_config.excluded_models,
                    recommended_horizon=advisor_config.recommended_horizon,
                    max_reliable_horizon=advisor_config.max_reliable_horizon,
                    expected_mape_range=list(advisor_config.expected_mape_range),
                    growth_pct=advisor_config.growth_pct,
                    summary=advisor_config.summary,
                )

                # TRACE step
                trace.add_step("AUTO_OPTIMIZE", df, "y", "ds", safe_covariates, {
                    "forecastability_score": advisor_config.forecastability_score,
                    "grade": advisor_config.grade,
                    "growth_pct": advisor_config.growth_pct,
                    "training_window": advisor_config.training_window_weeks,
                    "from_date": advisor_config.from_date,
                    "log_transform": advisor_config.log_transform,
                    "models": advisor_config.recommended_models,
                    "recommended_horizon": advisor_config.recommended_horizon,
                    "max_reliable_horizon": advisor_config.max_reliable_horizon,
                })

            except Exception as e:
                logger.warning(f"Auto-optimize failed (non-fatal, using defaults): {e}")
                auto_optimize_info = AutoOptimizeInfo(enabled=False)

        # ========================================
        # LOG TRANSFORM FOR HIGH-GROWTH SERIES
        # ========================================
        # Log1p transform linearizes exponential growth, dramatically improving
        # forecast accuracy on high-growth segments.
        # All models train in log space; forecasts are inverse-transformed (expm1)
        # back to original scale before returning to the user.
        log_transform_applied = False
        log_transform_mode = getattr(request, 'log_transform', 'auto') or 'auto'

        if log_transform_mode != 'never' and len(df) >= 52:
            should_transform = False
            growth_pct = 0.0

            if log_transform_mode == 'always':
                should_transform = True
                logger.info("Log transform: mode='always', applying log1p to target")
            elif log_transform_mode == 'auto':
                # Detect high growth: compare first 26 weeks vs last 26 weeks
                sorted_y = df.sort_values('ds')['y']
                first_half = sorted_y.head(len(sorted_y) // 2).mean()
                second_half = sorted_y.tail(len(sorted_y) // 2).mean()
                if first_half > 0:
                    growth_pct = (second_half - first_half) / first_half * 100
                    should_transform = growth_pct > 100  # >100% growth triggers transform
                    if should_transform:
                        logger.info(f"Log transform: auto-detected {growth_pct:.0f}% growth (>100% threshold), applying log1p")
                    else:
                        logger.info(f"Log transform: auto-detected {growth_pct:.0f}% growth (<100% threshold), skipping")

            if should_transform:
                # Validate: log1p requires non-negative values
                if (df['y'] < 0).any():
                    logger.warning("Log transform: skipped — negative values in target column")
                else:
                    df['y'] = np.log1p(df['y'])
                    log_transform_applied = True
                    logger.info(f"Log transform applied: y range [{df['y'].min():.2f}, {df['y'].max():.2f}] (log scale)")

                    # Update historical stats to reflect log space
                    historical_mean = df['y'].mean()
                    historical_std = df['y'].std()

                    # TRACE: Log transform step
                    trace.add_step("LOG_TRANSFORM", df, "y", "ds", safe_covariates, {
                        "transform": "log1p",
                        "mode": log_transform_mode,
                        "growth_pct": round(growth_pct, 1),
                        "y_min_log": float(df['y'].min()),
                        "y_max_log": float(df['y'].max()),
                    })

        # ========================================
        # CRITICAL: VALIDATE DATA IS SORTED BY DATE
        # ========================================
        # Time series splits REQUIRE chronologically sorted data.
        # Without this, train/eval/holdout sets would have temporal leakage.
        if not df['ds'].is_monotonic_increasing:
            logger.warning("⚠️ Data not sorted chronologically - sorting by date to prevent temporal leakage")
            df = df.sort_values('ds').reset_index(drop=True)
            logger.info(f"✅ Data sorted: {df['ds'].min()} to {df['ds'].max()}")

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

        # TRACE Step 6: Train split
        trace.add_step("TRAIN_SPLIT", train_df, "y", "ds", safe_covariates, {
            "split_type": "train",
            "total_rows": total_rows,
            "train_rows": len(train_df),
            "train_pct": round(len(train_df) / total_rows * 100, 1),
            "date_start": str(train_df['ds'].min()) if len(train_df) > 0 else None,
            "date_end": str(train_df['ds'].max()) if len(train_df) > 0 else None,
        })

        # Log EVAL set details
        log_dataframe_summary(eval_df, "EVAL SET (hyperparameter tuning)")

        # TRACE Step 7: Eval split
        trace.add_step("EVAL_SPLIT", eval_df, "y", "ds", safe_covariates, {
            "split_type": "eval",
            "eval_rows": len(eval_df),
            "eval_pct": round(len(eval_df) / total_rows * 100, 1),
            "date_start": str(eval_df['ds'].min()) if len(eval_df) > 0 else None,
            "date_end": str(eval_df['ds'].max()) if len(eval_df) > 0 else None,
        })

        # Log HOLDOUT set details if exists
        if holdout_size > 0:
            log_dataframe_summary(holdout_df, "HOLDOUT SET (final model selection)")

            # TRACE Step 8: Holdout split
            trace.add_step("HOLDOUT_SPLIT", holdout_df, "y", "ds", safe_covariates, {
                "split_type": "holdout",
                "holdout_rows": len(holdout_df),
                "holdout_pct": round(len(holdout_df) / total_rows * 100, 1),
                "date_start": str(holdout_df['ds'].min()),
                "date_end": str(holdout_df['ds'].max()),
            })

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
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
        mlflow.set_tracking_uri(tracking_uri)
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
            mlflow.log_param("confidence_level", confidence_level)
            logger.info(f"🌱 Logged random seed: {seed} and confidence_level: {confidence_level} to MLflow")

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
                            request_hp_filters,
                            train_df_override=train_df,  # Pass pre-split data
                            test_df_override=test_df,    # Pass pre-split data
                            forecast_start_date=to_date,  # Forecast starts from user's to_date
                            pipeline_trace=trace          # Pass pipeline trace for debugging
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
                            hyperparameter_filters=request_hp_filters,
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
                            hyperparameter_filters=request_hp_filters,
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
                            country=request.country, hyperparameter_filters=request_hp_filters,
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
                            country=request.country, hyperparameter_filters=request_hp_filters,
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
                    elif model_type == 'statsforecast':
                        # StatsForecast - Fast statistical models (AutoARIMA, AutoETS, AutoTheta)
                        if not STATSFORECAST_AVAILABLE:
                            logger.warning("StatsForecast not available. Install with: pip install statsforecast>=1.7.0")
                            continue
                        run_id, _, metrics, val, fcst, uri, params = train_statsforecast_model(
                            train_df, test_df, request.horizon, request.frequency,
                            random_seed=seed, original_data=request.data,
                            hyperparameter_filters=request_hp_filters,
                            forecast_start_date=to_date,
                            model_type='auto'  # Uses AutoARIMA by default
                        )
                        val = val.rename(columns={'ds': request.time_col})
                        fcst = fcst.rename(columns={'ds': request.time_col})
                        model_name_str = params.get('model_name', 'AutoARIMA') if params else 'StatsForecast'
                        result = ModelResult(
                            model_type='statsforecast', model_name=f"StatsForecast ({model_name_str})", run_id=run_id,
                            metrics=ForecastMetrics(
                                rmse=str(metrics['rmse']), mape=str(metrics['mape']), r2=str(metrics['r2']),
                                cv_mape=str(metrics['cv_mape']) if metrics.get('cv_mape') else None,
                                cv_mape_std=str(metrics['cv_mape_std']) if metrics.get('cv_mape_std') else None
                            ),
                            validation=val.to_dict('records'), forecast=fcst.to_dict('records'), covariate_impacts=[], is_best=False
                        )
                    elif model_type == 'chronos':
                        # Chronos - Zero-shot foundation model
                        if not CHRONOS_AVAILABLE:
                            logger.warning("Chronos not available. Install with: pip install chronos-forecasting torch")
                            continue
                        # Get model size from hyperparameter filters or use default
                        chronos_filters = request_hp_filters.get('Chronos', {})
                        model_size = chronos_filters.get('model_size', 'small')
                        run_id, _, metrics, val, fcst, uri, params = train_chronos_model(
                            train_df, test_df, request.horizon, request.frequency,
                            random_seed=seed, original_data=request.data,
                            forecast_start_date=to_date,
                            model_size=model_size
                        )
                        val = val.rename(columns={'ds': request.time_col})
                        fcst = fcst.rename(columns={'ds': request.time_col})
                        result = ModelResult(
                            model_type='chronos', model_name=f"Chronos-Bolt ({model_size})", run_id=run_id,
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

                        # TRACE Step 9: Per-model result
                        fcst_values = [r.get('yhat', 0) for r in (result.forecast if isinstance(result.forecast, list) else [])]
                        trace.add_step(f"MODEL_RESULT_{model_type.upper()}", train_df, "y", "ds", safe_covariates, {
                            "model_name": result.model_name,
                            "mape": metrics.get('mape'),
                            "rmse": metrics.get('rmse'),
                            "r2": metrics.get('r2'),
                            "cv_mape": metrics.get('cv_mape'),
                            "validation_rows": len(val) if isinstance(val, (list, pd.DataFrame)) else 0,
                            "forecast_rows": len(fcst_values),
                            "forecast_min": min(fcst_values) if fcst_values else None,
                            "forecast_max": max(fcst_values) if fcst_values else None,
                            "forecast_mean": sum(fcst_values) / len(fcst_values) if fcst_values else None,
                            "historical_mean": float(train_df['y'].mean()),
                            "forecast_vs_history_ratio": (sum(fcst_values) / len(fcst_values)) / max(float(train_df['y'].mean()), 1e-10) if fcst_values else None,
                        })

                        # Penalize models with flat forecasts to prevent selection as best
                        try:
                            fcst_records = result.forecast
                            if isinstance(fcst_records, list) and len(fcst_records) > 1:
                                from backend.models.utils import detect_flat_forecast
                                yhat_vals = np.array([r.get('yhat', 0) for r in fcst_records])
                                flat_info = detect_flat_forecast(yhat_vals, train_df['y'].values)
                                if flat_info['is_flat']:
                                    logger.warning(f"🚨 {result.model_name} produces flat forecast — penalizing for selection")
                                    metrics['mape'] = metrics['mape'] + 1000.0
                        except Exception as flat_err:
                            logger.debug(f"Flat forecast check skipped for {model_type}: {flat_err}")

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
        # ENSEMBLE MODEL CREATION (Optional)
        # ========================================
        # Create ensemble forecast by combining successful models
        if ENSEMBLE_AVAILABLE and len([r for r in model_results if r.run_id and r.metrics.mape != "N/A"]) >= 2:
            try:
                logger.info(f"")
                logger.info(f"{'='*60}")
                logger.info(f"🔀 CREATING ENSEMBLE MODEL")
                logger.info(f"{'='*60}")

                # Prepare model results for ensemble
                ensemble_inputs = []
                for res in model_results:
                    if res.run_id and res.metrics.mape != "N/A":
                        try:
                            mape_val = float(res.metrics.mape)
                            # Convert validation and forecast back to DataFrames
                            val_df_for_ensemble = pd.DataFrame(res.validation)
                            fcst_df_for_ensemble = pd.DataFrame(res.forecast)

                            # Rename columns back to standard names
                            if request.time_col != 'ds':
                                val_df_for_ensemble = val_df_for_ensemble.rename(columns={request.time_col: 'ds'})
                                fcst_df_for_ensemble = fcst_df_for_ensemble.rename(columns={request.time_col: 'ds'})

                            ensemble_inputs.append({
                                'model_name': res.model_name,
                                'metrics': {'mape': mape_val, 'rmse': float(res.metrics.rmse), 'r2': float(res.metrics.r2)},
                                'val_df': val_df_for_ensemble,
                                'fcst_df': fcst_df_for_ensemble
                            })
                        except (ValueError, TypeError) as e:
                            logger.warning(f"   Skipping {res.model_name} for ensemble: {e}")
                            continue

                if len(ensemble_inputs) >= 2:
                    # Create ensemble with AutoML best practices filtering
                    # - Validates model predictions (NaN, inf, extreme values, all-zeros)
                    # - Filters overfitting models (train vs eval MAPE ratio)
                    # - Uses MAPE-weighted ensemble from AutoGluon patterns
                    ensemble_run_id, _, ensemble_metrics, ensemble_val, ensemble_fcst, ensemble_uri, ensemble_info = train_ensemble_model(
                        model_results=ensemble_inputs,
                        train_df=train_df,
                        test_df=test_df,
                        horizon=request.horizon,
                        frequency=request.frequency,
                        random_seed=seed,
                        weighting_method='inverse_mape',
                        forecast_start_date=to_date,
                        # AutoML best practices parameters
                        historical_mean=historical_mean,
                        historical_std=historical_std,
                        max_mape_ratio=3.0,  # Only include models with MAPE <= 3x best
                        filter_overfitting=True,  # Filter models with train/eval ratio > threshold
                        max_overfitting_severity='high'  # Threshold: low=1.5, medium=2.0, high=3.0
                    )

                    # Rename columns for consistency
                    ensemble_val = ensemble_val.rename(columns={'ds': request.time_col})
                    ensemble_fcst = ensemble_fcst.rename(columns={'ds': request.time_col})

                    # Create ensemble result
                    weights_str = ", ".join([f"{k[:10]}:{v:.2f}" for k, v in ensemble_info.get('weights', {}).items()])
                    ensemble_result = ModelResult(
                        model_type='ensemble',
                        model_name=f"Ensemble ({len(ensemble_inputs)} models)",
                        run_id=ensemble_run_id,
                        metrics=ForecastMetrics(
                            rmse=str(ensemble_metrics['rmse']),
                            mape=str(ensemble_metrics['mape']),
                            r2=str(ensemble_metrics['r2']),
                            cv_mape=None,
                            cv_mape_std=None
                        ),
                        validation=ensemble_val.to_dict('records'),
                        forecast=ensemble_fcst.to_dict('records'),
                        covariate_impacts=[],
                        is_best=False
                    )

                    # Check if ensemble is best
                    if ensemble_metrics['mape'] < best_mape:
                        best_mape = ensemble_metrics['mape']
                        best_model_name = ensemble_result.model_name
                        best_run_id = ensemble_run_id
                        artifact_uri_ref = ensemble_uri
                        logger.info(f"🏆 Ensemble is new best model: MAPE={best_mape:.2f}%")

                    model_results.append(ensemble_result)
                    logger.info(f"✅ Ensemble created with weights: {weights_str}")

                    # TRACE Step 10: Ensemble
                    ens_fcst_vals = [r.get('yhat', 0) for r in ensemble_fcst.to_dict('records')]
                    trace.add_step("ENSEMBLE", train_df, "y", "ds", safe_covariates, {
                        "n_models": len(ensemble_inputs),
                        "weights": ensemble_info.get('weights', {}),
                        "ensemble_mape": ensemble_metrics.get('mape'),
                        "ensemble_rmse": ensemble_metrics.get('rmse'),
                        "forecast_min": min(ens_fcst_vals) if ens_fcst_vals else None,
                        "forecast_max": max(ens_fcst_vals) if ens_fcst_vals else None,
                        "forecast_mean": sum(ens_fcst_vals) / len(ens_fcst_vals) if ens_fcst_vals else None,
                    })

            except Exception as e:
                logger.warning(f"Ensemble creation failed: {e}")
                # Continue without ensemble - not a critical failure

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
                        logger.info(f"   {res.model_name}: Loading model from {model_uri}")

                        try:
                            loaded_model = mlflow.pyfunc.load_model(model_uri)
                            logger.info(f"   {res.model_name}: Model loaded successfully")
                        except Exception as load_error:
                            logger.error(f"   {res.model_name}: Failed to load model from {model_uri}: {load_error}")
                            # Try loading from backup
                            try:
                                backup_uri = f"runs:/{res.run_id}/model_backup"
                                logger.info(f"   {res.model_name}: Attempting to load from backup: {backup_uri}")
                                import pickle
                                client = mlflow.tracking.MlflowClient()
                                local_path = client.download_artifacts(res.run_id, "model_backup")
                                backup_files = [f for f in os.listdir(local_path) if f.endswith('.pkl')]
                                if backup_files:
                                    with open(os.path.join(local_path, backup_files[0]), 'rb') as f:
                                        backup_data = pickle.load(f)
                                    logger.info(f"   {res.model_name}: Loaded model from backup")
                                    # Use the wrapper from backup if available
                                    if 'wrapper' in backup_data:
                                        loaded_model = backup_data['wrapper']
                                    else:
                                        raise Exception("Backup doesn't contain model wrapper")
                                else:
                                    raise Exception("No backup pickle files found")
                            except Exception as backup_error:
                                logger.error(f"   {res.model_name}: Backup load also failed: {backup_error}")
                                raise load_error  # Re-raise original error

                        # Prepare holdout input based on model signature
                        # XGBoost uses Mode 1 (periods/start_date), Prophet/ARIMA use Mode 2 (ds)
                        model_signature = loaded_model.metadata.signature
                        expected_cols = []
                        if model_signature and model_signature.inputs:
                            expected_cols = [col_spec.get('name', '') for col_spec in model_signature.inputs.to_dict()]

                        # Detect if model expects periods/start_date (XGBoost Mode 1)
                        uses_periods_format = 'periods' in expected_cols and 'start_date' in expected_cols

                        if uses_periods_format:
                            # XGBoost Mode 1: periods + start_date
                            holdout_dates = pd.to_datetime(holdout_df['ds'])
                            start_date = holdout_dates.min()
                            num_periods = len(holdout_dates)
                            holdout_input = pd.DataFrame({
                                'periods': [num_periods],
                                'start_date': [start_date.strftime('%Y-%m-%d')]
                            })
                            logger.info(f"   {res.model_name}: Using periods/start_date format (periods={num_periods}, start={start_date.strftime('%Y-%m-%d')})")
                        else:
                            # Prophet/ARIMA Mode 2: ds column with dates
                            holdout_input = holdout_df[['ds']].copy()
                            holdout_input['ds'] = pd.to_datetime(holdout_input['ds']).dt.strftime('%Y-%m-%d')

                        # Add covariates from holdout_df if model expects them (Mode 2 only)
                        try:
                            if model_signature and model_signature.inputs and not uses_periods_format:

                                for col_name in expected_cols:
                                    if col_name == 'ds':
                                        continue  # Already added

                                    if col_name in holdout_df.columns:
                                        # Direct column available
                                        holdout_input[col_name] = holdout_df[col_name].values
                                    elif col_name in df.columns:
                                        # Try to get from full dataset (for covariates)
                                        # Match by date
                                        holdout_dates = pd.to_datetime(holdout_input['ds'])
                                        df_lookup = df.set_index('ds')[col_name]
                                        holdout_input[col_name] = holdout_dates.map(df_lookup).fillna(0).values
                                    else:
                                        # Feature not available - set to 0 (common for binary indicators)
                                        logger.warning(f"   ⚠️ Feature '{col_name}' not in holdout data - using default 0")
                                        holdout_input[col_name] = 0

                        except Exception as sig_error:
                            logger.warning(f"   Could not extract signature: {sig_error}")

                        # Get predictions
                        holdout_predictions = loaded_model.predict(holdout_input)

                        # DEBUG: Log prediction details
                        logger.info(f"   {res.model_name} holdout debug:")
                        logger.info(f"      Input shape: {holdout_input.shape}, columns: {list(holdout_input.columns)}")
                        if isinstance(holdout_predictions, pd.DataFrame):
                            logger.info(f"      Output shape: {holdout_predictions.shape}, columns: {list(holdout_predictions.columns)}")
                            # Check for all-zero predictions (common sign of model loading issues)
                            if 'yhat' in holdout_predictions.columns:
                                pred_values = holdout_predictions['yhat'].values
                                non_zero_count = (pred_values != 0).sum()
                                if non_zero_count == 0:
                                    logger.error(f"   ❌ {res.model_name}: ALL PREDICTIONS ARE ZERO! This indicates a model loading or inference issue.")
                                    logger.error(f"      Possible causes: Model not properly serialized, date format mismatch, or feature mismatch")
                                    logger.error(f"      Input sample: {holdout_input.head(2).to_dict()}")
                                else:
                                    logger.info(f"      Prediction stats: min={pred_values.min():,.0f}, max={pred_values.max():,.0f}, mean={pred_values.mean():,.0f}")
                        else:
                            logger.info(f"      Output type: {type(holdout_predictions)}, len: {len(holdout_predictions) if hasattr(holdout_predictions, '__len__') else 'N/A'}")

                        # Calculate holdout MAPE with robust outlier handling
                        if isinstance(holdout_predictions, pd.DataFrame) and 'yhat' in holdout_predictions.columns:
                            # Align predictions with actuals
                            pred_df = holdout_predictions[['ds', 'yhat']].copy()
                            pred_df['ds'] = pd.to_datetime(pred_df['ds'])
                            actual_df = holdout_df[['ds', 'y']].copy()

                            # DEBUG: Log date ranges
                            logger.info(f"      Prediction dates: {pred_df['ds'].min()} to {pred_df['ds'].max()} ({len(pred_df)} rows)")
                            logger.info(f"      Holdout dates: {actual_df['ds'].min()} to {actual_df['ds'].max()} ({len(actual_df)} rows)")

                            merged = actual_df.merge(pred_df, on='ds', how='inner')
                            logger.info(f"      Matched rows after merge: {len(merged)}/{len(actual_df)}")

                            if len(merged) > 0:
                                # Inverse transform holdout values if log transform was applied
                                # so MAPE is computed in original scale (meaningful %)
                                y_actual = merged['y'].values.copy()
                                y_pred = merged['yhat'].values.copy()
                                if log_transform_applied:
                                    y_actual = np.expm1(y_actual)
                                    y_pred = np.expm1(y_pred)

                                # Use safe_mape from utils for robust calculation
                                from backend.models.utils import safe_mape
                                holdout_mape = safe_mape(y_actual, y_pred)

                                # Log per-point details for debugging
                                logger.info(f"      Per-point comparison:")
                                for idx in range(len(y_actual)):
                                    dt = merged.iloc[idx]['ds']
                                    pct_err = abs(y_actual[idx] - y_pred[idx]) / max(abs(y_actual[idx]), 1e-10) * 100
                                    logger.info(f"         {dt.strftime('%Y-%m-%d')}: actual={y_actual[idx]:,.0f}, pred={y_pred[idx]:,.0f}, error={pct_err:.1f}%")

                                # holdout_mape already computed by safe_mape above
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

        # ========================================
        # INVERSE LOG TRANSFORM (expm1)
        # ========================================
        # If log transform was applied, convert everything back to original scale:
        # - History data (for chart visualization)
        # - All model forecasts (yhat, yhat_lower, yhat_upper)
        # - All model validation predictions
        if log_transform_applied:
            logger.info(f"")
            logger.info(f"{'='*60}")
            logger.info(f"📐 INVERSE LOG TRANSFORM (expm1)")
            logger.info(f"{'='*60}")

            # 1. Inverse transform history data
            target_col = request.target_col
            if target_col in history_df.columns:
                history_df[target_col] = np.expm1(history_df[target_col])
                logger.info(f"   History: inverse-transformed {len(history_df)} rows back to original scale")

            # 2. Inverse transform all model results (forecasts + validation)
            for res in model_results:
                model_label = res.model_name

                # Transform forecast records
                if res.forecast and isinstance(res.forecast, list):
                    for record in res.forecast:
                        for key in ('yhat', 'yhat_lower', 'yhat_upper'):
                            if key in record and record[key] is not None:
                                try:
                                    record[key] = float(np.expm1(record[key]))
                                except (ValueError, TypeError, OverflowError):
                                    pass
                    logger.info(f"   {model_label}: inverse-transformed {len(res.forecast)} forecast rows")

                # Transform validation records and recompute eval metrics in original scale
                if res.validation and isinstance(res.validation, list):
                    for record in res.validation:
                        for key in ('yhat', 'yhat_lower', 'yhat_upper', 'y'):
                            if key in record and record[key] is not None:
                                try:
                                    record[key] = float(np.expm1(record[key]))
                                except (ValueError, TypeError, OverflowError):
                                    pass

                    # Recompute eval MAPE in original scale
                    try:
                        from backend.models.utils import safe_mape
                        val_y = np.array([r.get('y', 0) for r in res.validation if r.get('y') is not None])
                        val_yhat = np.array([r.get('yhat', 0) for r in res.validation if r.get('yhat') is not None])
                        if len(val_y) == len(val_yhat) and len(val_y) > 0:
                            orig_mape = res.metrics.mape
                            new_mape = safe_mape(val_y, val_yhat)
                            res.metrics.mape = str(round(new_mape, 2))
                            new_rmse = float(np.sqrt(np.mean((val_y - val_yhat) ** 2)))
                            res.metrics.rmse = str(round(new_rmse, 2))
                            logger.info(f"   {model_label}: MAPE recalculated {orig_mape}% (log) -> {new_mape:.2f}% (original scale)")
                    except Exception as recomp_err:
                        logger.debug(f"   {model_label}: could not recompute metrics: {recomp_err}")

                    logger.info(f"   {model_label}: inverse-transformed {len(res.validation)} validation rows")

            # 3. Log transform params to MLflow
            try:
                with mlflow.start_run(run_id=parent_run_id):
                    mlflow.log_params({
                        "log_transform_applied": True,
                        "log_transform_mode": log_transform_mode,
                    })
            except Exception as mlflow_err:
                logger.debug(f"   Could not log transform params to MLflow: {mlflow_err}")

            # TRACE: Inverse transform step
            trace.add_step("INVERSE_LOG_TRANSFORM", details={
                "transform": "expm1",
                "n_models_transformed": len(model_results),
                "history_rows_transformed": len(history_df),
            })

            logger.info(f"   All values restored to original scale")
            logger.info(f"{'='*60}")

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

        # ========================================
        # AUTOML TRAINING SUMMARY
        # ========================================
        # Log summary of AutoML best practices applied during training
        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"📋 AUTOML TRAINING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"   Total models requested: {len(request.models)}")
        logger.info(f"   Models completed: {len([r for r in model_results if r.run_id and r.metrics.mape != 'N/A'])}")
        logger.info(f"   Models failed: {len([r for r in model_results if r.metrics.mape == 'N/A'])}")

        # Log data quality info
        if AUTOML_UTILS_AVAILABLE and data_quality_report:
            logger.info(f"   Data quality: {'✅ Valid' if data_quality_report.is_valid else '⚠️ Issues detected'}")
            if data_quality_report.issues:
                for issue in data_quality_report.issues[:3]:  # Show first 3 issues
                    logger.info(f"      - {issue}")

        # Log ensemble info
        ensemble_model = next((r for r in model_results if r.model_type == 'ensemble'), None)
        if ensemble_model:
            logger.info(f"   Ensemble created: ✅ Yes")
            logger.info(f"   Ensemble MAPE: {ensemble_model.metrics.mape}%")
        else:
            logger.info(f"   Ensemble created: ❌ No (insufficient valid models)")

        logger.info(f"   Best model: {best_model_name}")
        logger.info(f"   Best MAPE: {best_mape:.2f}%")
        logger.info(f"{'='*60}")

        # Ensure we have a valid best model before returning
        if best_model_name is None:
            # Check if any models were trained at all
            if model_results:
                # Pick the first model that succeeded as fallback
                for result in model_results:
                    if result.model_name:
                        best_model_name = result.model_name
                        logger.warning(f"No best model selected via holdout - falling back to: {best_model_name}")
                        break

            if best_model_name is None:
                # No models succeeded - raise meaningful error
                error_msgs = [f"{r.model_name}: {r.test_result.message if r.test_result else 'Unknown error'}"
                             for r in model_results if r.test_result and not r.test_result.passed]
                error_detail = f"All models failed training. Details: {'; '.join(error_msgs) if error_msgs else 'Unknown error'}"
                logger.error(error_detail)
                raise HTTPException(status_code=500, detail=error_detail)

        # TRACE Step 11: Final response
        trace.add_step("FINAL_RESPONSE", details={
            "best_model": best_model_name,
            "best_mape": float(best_mape) if best_mape != float('inf') else None,
            "n_models_returned": len(model_results),
            "n_models_succeeded": len([r for r in model_results if r.run_id and r.metrics.mape != 'N/A']),
            "n_models_failed": len([r for r in model_results if r.metrics.mape == 'N/A']),
            "history_rows": len(history_data),
            "model_summary": [{
                "name": r.model_name,
                "mape": r.metrics.mape,
                "is_best": r.is_best,
                "forecast_rows": len(r.forecast) if r.forecast else 0,
            } for r in model_results],
        })

        # Store trace for retrieval via /api/debug/pipeline-trace
        store_trace(trace)

        return TrainResponse(models=[m.dict() for m in model_results], best_model=best_model_name, artifact_uri=artifact_uri_ref or "N/A", history=history_data, trace_id=trace.trace_id, auto_optimize_info=auto_optimize_info)

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


# Maximum number of segments allowed in a single batch request
# Prevents resource exhaustion on Databricks Apps (4 vCPU, 12GB RAM limit)
MAX_BATCH_SIZE = 100


@app.post("/api/train-batch", response_model=BatchTrainResponse)
async def train_batch(request: BatchTrainRequest):
    """
    Train multiple forecasting models in parallel.

    This endpoint allows you to submit multiple training requests at once,
    processing them in parallel for faster batch forecasting across segments.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
    import traceback

    # Validate batch size to prevent resource exhaustion
    if len(request.requests) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.requests)} exceeds maximum allowed ({MAX_BATCH_SIZE}). "
                   f"Please split into smaller batches."
        )

    if len(request.requests) == 0:
        raise HTTPException(
            status_code=400,
            detail="Batch request must contain at least one segment."
        )

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

