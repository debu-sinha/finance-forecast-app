"""
FastAPI backend for Databricks Finance Forecasting Platform
"""
import os
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.models import (
    TrainRequest, TrainResponse, DeployRequest, DeployResponse,
    HealthResponse, ForecastMetrics, ModelResult, CovariateImpact,
    BatchTrainRequest, BatchTrainResponse, BatchResultItem,
    AggregateRequest, AggregateResponse
)
from backend.train_service import train_prophet_model, register_model_to_unity_catalog, prepare_prophet_data
from backend.models_training import train_arima_model, train_exponential_smoothing_model, train_sarimax_model, train_xgboost_model
from backend.deploy_service import (
    deploy_model_to_serving, get_endpoint_status, delete_endpoint,
    list_endpoints, get_databricks_client
)
from backend.ai_service import analyze_dataset, generate_forecast_insights, generate_executive_summary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

@app.get("/")
async def root():
    return FileResponse(f"{static_dir}/index.html") if os.path.exists(f"{static_dir}/index.html") else {"message": "API Running"}

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if full_path.startswith("api"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    return FileResponse(f"{static_dir}/index.html") if os.path.exists(f"{static_dir}/index.html") else JSONResponse(status_code=404, content={"error": "Frontend not found"})

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
    except Exception: pass
    
    try:
        import mlflow
        mlflow.set_tracking_uri("databricks")
        status_data["mlflow_enabled"] = True
    except Exception: pass
    return HealthResponse(**status_data)

@app.post("/api/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    try:
        logger.info(f"Training request: target={request.target_col}, horizon={request.horizon}")
        if not request.data or request.horizon <= 0:
            raise ValueError("Invalid data or horizon")
        
        # Set random seeds for reproducibility
        import random
        import numpy as np
        seed = request.random_seed if request.random_seed is not None else 42
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"🌱 Set random seed: {seed} for reproducibility")

        df = prepare_prophet_data(request.data, request.time_col, request.target_col, request.covariates)

        # Store pre-filter data for logging
        pre_filter_df = df.copy()

        # Apply date range filtering
        original_len = len(df)
        import pandas as pd
        
        # Default to_date to the end of the time series if not provided
        max_date = df['ds'].max()
        effective_to_date = request.to_date if request.to_date else str(max_date.date())
        
        # Apply from_date filter if provided
        if request.from_date:
            from_date = pd.to_datetime(request.from_date).normalize()
            df = df[df['ds'] >= from_date].copy()
            logger.info(f"📅 Filtered data: {original_len} -> {len(df)} rows (from_date >= {request.from_date})")
        
        # Always apply to_date filter (either provided or default to max date)
        before_to_filter = len(df)
        to_date = pd.to_datetime(effective_to_date).normalize()
        df = df[df['ds'] <= to_date].copy()
        if request.to_date:
            logger.info(f"📅 Filtered data: {before_to_filter} -> {len(df)} rows (to_date <= {request.to_date})")
        else:
            logger.info(f"📅 Applied default to_date filter: {before_to_filter} -> {len(df)} rows (to_date <= {effective_to_date}, max date in dataset)")
        
        if len(df) == 0:
            raise ValueError(f"No data remaining after date filtering (from_date: {request.from_date}, to_date: {effective_to_date})")
        
        if request.from_date or not request.to_date:
            logger.info(f"📅 Final filtered dataset: {len(df)} rows (from {df['ds'].min()} to {df['ds'].max()})")
        
        test_size = request.test_size or min(request.horizon, len(df) // 5)
        train_df, test_df = df.iloc[:-test_size].copy(), df.iloc[-test_size:].copy()

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
                
                # Add train/test split metadata
                split_metadata = {
                    "train_size": len(train_df),
                    "test_size": len(test_df),
                    "total_size": len(df),
                    "split_date": str(train_df['ds'].max()),
                    "test_percentage": round(len(test_df) / len(df) * 100, 2),
                    "train_date_range": {
                        "start": str(train_df['ds'].min()),
                        "end": str(train_df['ds'].max())
                    },
                    "test_date_range": {
                        "start": str(test_df['ds'].min()),
                        "end": str(test_df['ds'].max())
                    }
                }
                with open("/tmp/train_test_split.json", "w") as f:
                    json.dump(split_metadata, f, indent=2)
                mlflow.log_artifact("/tmp/train_test_split.json", "metadata")
                logger.info(f"Logged train/test split metadata to metadata/: train={len(train_df)}, test={len(test_df)}")
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
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("total_data_points", len(df))
            mlflow.log_param("training_data_points", len(train_df))
            mlflow.log_param("validation_data_points", len(test_df))
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
            
            for model_type in request.models:
                try:
                    result = None
                    if model_type == 'prophet':
                        run_id, _, metrics, val, fcst, uri, impacts = train_prophet_model(
                            request.data, request.time_col, request.target_col, request.covariates,
                            request.horizon, request.frequency, request.seasonality_mode, test_size,
                            request.regressor_method, request.country, seed, request.future_features
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
                            original_data=request.data, covariates=request.covariates
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
                            original_data=request.data, covariates=request.covariates
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
                            covariates=request.covariates, random_seed=seed, original_data=request.data,
                            country=request.country
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
                            covariates=request.covariates, random_seed=seed, original_data=request.data,
                            country=request.country
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
                        if metrics['mape'] < best_mape:
                            best_mape = metrics['mape']
                            best_model_name = result.model_name
                            best_run_id = run_id
                            artifact_uri_ref = uri
                        model_results.append(result)

                except Exception as e:
                    logger.error(f"{model_type} failed: {e}", exc_info=True)
                    # Add failed model to results with error info so frontend knows it was attempted
                    failed_result = ModelResult(
                        model_type=model_type,
                        model_name=f"{model_type.upper()} (Failed)",
                        run_id="",
                        metrics=ForecastMetrics(rmse="N/A", mape="N/A", r2="N/A"),
                        validation=[],
                        forecast=[],
                        covariate_impacts=[],
                        is_best=False
                    )
                    # Don't add failed models to results - just log the error
                    logger.warning(f"Model {model_type} skipped due to error: {str(e)[:200]}")

        if not model_results: raise Exception("All models failed")

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

        # Register ALL models to Unity Catalog so they are available for deployment
        logger.info(f"Attempting to register {len(model_results)} models to Unity Catalog...")
        for res in model_results:
            if not res.run_id:
                logger.error(f"Cannot register {res.model_name}: missing run_id")
                continue
            try:
                logger.info(f"Registering {res.model_name} (Run: {res.run_id})...")
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
                    f"{request.catalog_name}.{request.schema_name}.{request.model_name}",
                    tags_to_set
                )
                logger.info(f"Registered {res.model_name} as version {version} in Unity Catalog")
            except Exception as e: 
                logger.error(f"Auto-register failed for {res.model_name}: {e}", exc_info=True)

        return TrainResponse(models=[m.dict() for m in model_results], best_model=best_model_name, artifact_uri=artifact_uri_ref or "N/A")

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
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import traceback

    logger.info(f"Batch training request: {len(request.requests)} segments, max_workers={request.max_workers}")

    # Limit max_workers based on environment (Databricks Apps has 4 vCPU limit)
    max_workers = min(request.max_workers, int(os.environ.get('MLFLOW_MAX_WORKERS', '2')))
    logger.info(f"Using {max_workers} parallel workers")

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

    # Process requests in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(train_single, req, i): i
            for i, req in enumerate(request.requests)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results.append(result)
                status_icon = "" if result.status == "success" else ""
                logger.info(f"{status_icon} Completed segment {result.segment_id}")
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

    logger.info(f"Batch training complete: {successful} succeeded, {failed} failed")

    return BatchTrainResponse(
        total_requests=len(request.requests),
        successful=successful,
        failed=failed,
        results=results
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)

