# Databricks notebook source
# MAGIC %md
# MAGIC # Multi-User Training Notebook
# MAGIC
# MAGIC This notebook is executed on a dedicated Databricks cluster when training jobs are
# MAGIC submitted from the Finance Forecasting App. It integrates with Lakebase PostgreSQL
# MAGIC for state management and supports 30+ concurrent users.
# MAGIC
# MAGIC **Architecture:**
# MAGIC - App submits job via Databricks Jobs API
# MAGIC - Notebook runs on dedicated cluster (64 vCPU / 256 GB RAM)
# MAGIC - Results written to Lakebase PostgreSQL
# MAGIC - MLflow logs ONE parent run (no child runs for HP tuning)
# MAGIC
# MAGIC **Supported Models:**
# MAGIC - Prophet, ARIMA, ETS, XGBoost (legacy)
# MAGIC - StatsForecast (AutoARIMA, AutoETS, AutoTheta)
# MAGIC - Chronos (zero-shot foundation model)
# MAGIC - Ensemble (weighted average of top models)

# COMMAND ----------
# MAGIC %pip install asyncpg psycopg2-binary statsforecast chronos-forecasting torch mapie --quiet

# COMMAND ----------

import json
import os
import time
import tempfile
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID

import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters from Job Submission

# COMMAND ----------

# Job parameters passed from the App via Databricks Jobs API
dbutils.widgets.text("job_id", "", "Execution Job ID (UUID)")
dbutils.widgets.text("session_id", "", "User Session ID (UUID)")
dbutils.widgets.text("user_id", "", "User Identifier")
dbutils.widgets.text("data_upload_id", "", "Data Upload Reference (UUID)")
dbutils.widgets.text("request_json", "{}", "Complete TrainRequest (JSON)")
dbutils.widgets.text("lakebase_host", "", "Lakebase PostgreSQL Host")
dbutils.widgets.text("lakebase_database", "forecast", "Lakebase Database Name")

# Get parameter values
job_id = dbutils.widgets.get("job_id")
session_id = dbutils.widgets.get("session_id")
user_id = dbutils.widgets.get("user_id")
data_upload_id = dbutils.widgets.get("data_upload_id")
request_json = dbutils.widgets.get("request_json")
lakebase_host = dbutils.widgets.get("lakebase_host")
lakebase_database = dbutils.widgets.get("lakebase_database")

# Parse request
request = json.loads(request_json) if request_json else {}

logger.info(f"Job ID: {job_id}")
logger.info(f"User ID: {user_id}")
logger.info(f"Session ID: {session_id}")
logger.info(f"Data Upload ID: {data_upload_id}")
logger.info(f"Request keys: {list(request.keys())}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Lakebase Connection

# COMMAND ----------

import psycopg2
from psycopg2.extras import RealDictCursor, Json

def get_lakebase_connection():
    """Get PostgreSQL connection to Lakebase."""
    return psycopg2.connect(
        host=lakebase_host or os.getenv("LAKEBASE_HOST", "localhost"),
        port=int(os.getenv("LAKEBASE_PORT", "5432")),
        database=lakebase_database or os.getenv("LAKEBASE_DATABASE", "forecast"),
        user=os.getenv("LAKEBASE_USER", "forecast_app"),
        password=os.getenv("LAKEBASE_PASSWORD", ""),
        sslmode=os.getenv("LAKEBASE_SSL_MODE", "require"),
    )

def update_job_status(
    job_id: str,
    status: str,
    progress_percent: int = None,
    current_step: str = None,
    error_message: str = None,
    mlflow_run_id: str = None,
    best_model: str = None,
    best_mape: float = None,
):
    """Update job status in Lakebase."""
    try:
        conn = get_lakebase_connection()
        with conn.cursor() as cur:
            updates = ["status = %s", "updated_at = NOW()"]
            params = [status]

            # Timestamp updates based on status
            if status == "RUNNING":
                updates.append("started_at = NOW()")
            elif status in ("COMPLETED", "FAILED", "CANCELLED"):
                updates.append("completed_at = NOW()")

            # Optional fields
            if progress_percent is not None:
                updates.append("progress_percent = %s")
                params.append(progress_percent)
            if current_step is not None:
                updates.append("current_step = %s")
                params.append(current_step)
            if error_message is not None:
                updates.append("error_message = %s")
                params.append(error_message)
            if mlflow_run_id is not None:
                updates.append("mlflow_run_id = %s")
                params.append(mlflow_run_id)
            if best_model is not None:
                updates.append("best_model = %s")
                params.append(best_model)
            if best_mape is not None:
                updates.append("best_mape = %s")
                params.append(best_mape)

            params.append(job_id)

            cur.execute(
                f"""
                UPDATE forecast.execution_history
                SET {', '.join(updates)}
                WHERE job_id = %s
                """,
                params
            )
            conn.commit()
        conn.close()
        logger.info(f"Updated job {job_id} status to {status}")
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")

def save_forecast_result(
    job_id: str,
    model_name: str,
    forecast_dates: List[str],
    predictions: List[float],
    mape: float = None,
    rmse: float = None,
    **kwargs
):
    """Save forecast result to Lakebase."""
    try:
        conn = get_lakebase_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO forecast.forecast_results
                (result_id, job_id, model_name, forecast_dates, predictions,
                 lower_bounds, upper_bounds, mape, rmse, mae, r2, cv_mape,
                 model_params, training_time_seconds, mlflow_run_id)
                VALUES (
                    gen_random_uuid(), %s, %s, %s::jsonb, %s::jsonb,
                    %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s,
                    %s::jsonb, %s, %s
                )
                """,
                (
                    job_id,
                    model_name,
                    json.dumps(forecast_dates),
                    json.dumps(predictions),
                    json.dumps(kwargs.get("lower_bounds")),
                    json.dumps(kwargs.get("upper_bounds")),
                    mape,
                    rmse,
                    kwargs.get("mae"),
                    kwargs.get("r2"),
                    kwargs.get("cv_mape"),
                    json.dumps(kwargs.get("model_params")),
                    kwargs.get("training_time_seconds"),
                    kwargs.get("mlflow_run_id"),
                )
            )
            conn.commit()
        conn.close()
        logger.info(f"Saved result for {model_name} with MAPE={mape}")
    except Exception as e:
        logger.error(f"Failed to save result for {model_name}: {e}")

def update_model_rankings(job_id: str):
    """Update model rankings after all models complete."""
    try:
        conn = get_lakebase_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT forecast.update_model_rankings(%s)", (job_id,))
            conn.commit()
        conn.close()
        logger.info(f"Updated model rankings for job {job_id}")
    except Exception as e:
        logger.error(f"Failed to update rankings: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load Training Data

# COMMAND ----------

def load_data(request: Dict) -> pd.DataFrame:
    """Load training data from request."""
    if "data" not in request:
        raise ValueError("No data provided in request")

    df = pd.DataFrame(request["data"])

    # Parse date column
    time_col = request.get("time_col", "ds")
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df

# Load data
df = load_data(request)

# Extract parameters
time_col = request.get("time_col", "ds")
target_col = request.get("target_col", "y")
horizon = request.get("horizon", 12)
frequency = request.get("frequency", "weekly")
models_to_train = request.get("models", ["prophet"])
confidence_level = request.get("confidence_level", 0.95)
random_seed = request.get("random_seed", 42)
hyperparameter_filters = request.get("hyperparameter_filters", {})

logger.info(f"Horizon: {horizon}, Frequency: {frequency}")
logger.info(f"Models to train: {models_to_train}")
logger.info(f"Random seed: {random_seed}")

# Update status to RUNNING
update_job_status(job_id, "RUNNING", progress_percent=5, current_step="Loaded data")

# COMMAND ----------
# MAGIC %md
# MAGIC ## MLflow Setup

# COMMAND ----------

# Set experiment
experiment_name = request.get(
    "experiment_name",
    f"/Users/{user_id}/finance-forecasting"
)
mlflow.set_experiment(experiment_name)

logger.info(f"MLflow experiment: {experiment_name}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Model Training Functions

# COMMAND ----------

def train_prophet(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    frequency: str,
    random_seed: int,
    hyperparameter_filters: Dict = None,
) -> Dict[str, Any]:
    """Train Prophet model."""
    from prophet import Prophet

    start_time = time.time()

    # Prepare data
    prophet_train = train_df.rename(columns={time_col: "ds", target_col: "y"})

    # Configure model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=(frequency == "daily"),
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )

    # Fit
    model.fit(prophet_train)

    # Predict on test
    future = model.make_future_dataframe(
        periods=len(test_df),
        freq="W" if frequency == "weekly" else "D" if frequency == "daily" else "MS"
    )
    forecast = model.predict(future)

    # Get validation predictions
    val_start_idx = len(train_df)
    val_preds = forecast.iloc[val_start_idx:]["yhat"].values
    val_actuals = test_df[target_col].values

    # Metrics
    mape = np.mean(np.abs((val_actuals - val_preds) / val_actuals)) * 100
    rmse = np.sqrt(np.mean((val_actuals - val_preds) ** 2))

    # Future forecast
    future_dates = pd.date_range(
        start=test_df[time_col].max() + pd.Timedelta(days=7 if frequency == "weekly" else 1),
        periods=horizon,
        freq="W" if frequency == "weekly" else "D" if frequency == "daily" else "MS"
    )

    future_df = pd.DataFrame({"ds": future_dates})
    future_forecast = model.predict(future_df)

    training_time = time.time() - start_time

    return {
        "model_name": "prophet",
        "mape": mape,
        "rmse": rmse,
        "forecast_dates": future_forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
        "predictions": future_forecast["yhat"].tolist(),
        "lower_bounds": future_forecast["yhat_lower"].tolist(),
        "upper_bounds": future_forecast["yhat_upper"].tolist(),
        "validation_dates": test_df[time_col].dt.strftime("%Y-%m-%d").tolist(),
        "validation_predictions": val_preds.tolist(),
        "validation_actuals": val_actuals.tolist(),
        "training_time_seconds": training_time,
        "model_params": {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0},
    }


def train_statsforecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    frequency: str,
    random_seed: int,
    hyperparameter_filters: Dict = None,
) -> Dict[str, Any]:
    """Train StatsForecast models (AutoARIMA, AutoETS, AutoTheta)."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

    start_time = time.time()

    # Prepare data
    sf_df = train_df[[time_col, target_col]].copy()
    sf_df = sf_df.rename(columns={time_col: "ds", target_col: "y"})
    sf_df["unique_id"] = "series_1"
    sf_df = sf_df[["unique_id", "ds", "y"]]

    # Season length
    season_length = {"daily": 7, "weekly": 52, "monthly": 12}.get(frequency, 52)

    # Models
    models = [
        AutoARIMA(season_length=season_length),
        AutoETS(season_length=season_length),
        AutoTheta(season_length=season_length),
    ]

    # Fit
    sf = StatsForecast(models=models, freq="W" if frequency == "weekly" else "D", n_jobs=-1)
    sf.fit(sf_df)

    # Predict
    forecast = sf.predict(h=horizon, level=[90])

    training_time = time.time() - start_time

    # Best model (AutoARIMA by default)
    predictions = forecast["AutoARIMA"].values
    lower = forecast.get("AutoARIMA-lo-90", forecast["AutoARIMA"]).values
    upper = forecast.get("AutoARIMA-hi-90", forecast["AutoARIMA"]).values

    # Validate on test (simple evaluation)
    val_preds = predictions[:len(test_df)] if len(predictions) >= len(test_df) else predictions
    val_actuals = test_df[target_col].values[:len(val_preds)]

    mape = np.mean(np.abs((val_actuals - val_preds) / val_actuals)) * 100 if len(val_preds) > 0 else 0
    rmse = np.sqrt(np.mean((val_actuals - val_preds) ** 2)) if len(val_preds) > 0 else 0

    # Forecast dates
    last_date = train_df[time_col].max()
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=7 if frequency == "weekly" else 1),
        periods=horizon,
        freq="W" if frequency == "weekly" else "D"
    )

    return {
        "model_name": "statsforecast",
        "mape": mape,
        "rmse": rmse,
        "forecast_dates": forecast_dates.strftime("%Y-%m-%d").tolist(),
        "predictions": predictions.tolist(),
        "lower_bounds": lower.tolist(),
        "upper_bounds": upper.tolist(),
        "training_time_seconds": training_time,
        "model_params": {"season_length": season_length, "models": ["AutoARIMA", "AutoETS", "AutoTheta"]},
    }


def train_chronos(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    frequency: str,
    random_seed: int,
    hyperparameter_filters: Dict = None,
) -> Dict[str, Any]:
    """Train Chronos zero-shot foundation model."""
    import torch
    from chronos import ChronosBoltPipeline

    start_time = time.time()

    # Load model (small for speed)
    model_size = hyperparameter_filters.get("chronos_model_size", "small") if hyperparameter_filters else "small"
    model_name = f"amazon/chronos-bolt-{model_size}"

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    pipeline = ChronosBoltPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,
    )

    # Prepare context
    context = torch.tensor(train_df[target_col].values, dtype=torch.float32)

    # Predict
    forecast = pipeline.predict(
        context=context.unsqueeze(0),
        prediction_length=horizon,
        num_samples=100,
    )

    # Get quantiles
    predictions = np.median(forecast[0].numpy(), axis=0)
    lower = np.percentile(forecast[0].numpy(), 10, axis=0)
    upper = np.percentile(forecast[0].numpy(), 90, axis=0)

    training_time = time.time() - start_time

    # Validate on test
    val_preds = predictions[:len(test_df)] if len(predictions) >= len(test_df) else predictions
    val_actuals = test_df[target_col].values[:len(val_preds)]

    mape = np.mean(np.abs((val_actuals - val_preds) / val_actuals)) * 100 if len(val_preds) > 0 else 0
    rmse = np.sqrt(np.mean((val_actuals - val_preds) ** 2)) if len(val_preds) > 0 else 0

    # Forecast dates
    last_date = train_df[time_col].max()
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=7 if frequency == "weekly" else 1),
        periods=horizon,
        freq="W" if frequency == "weekly" else "D"
    )

    return {
        "model_name": "chronos",
        "mape": mape,
        "rmse": rmse,
        "forecast_dates": forecast_dates.strftime("%Y-%m-%d").tolist(),
        "predictions": predictions.tolist(),
        "lower_bounds": lower.tolist(),
        "upper_bounds": upper.tolist(),
        "training_time_seconds": training_time,
        "model_params": {"model_size": model_size, "device": device},
    }

# COMMAND ----------
# MAGIC %md
# MAGIC ## Train All Models

# COMMAND ----------

# Train/test split (80/20)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

# Model training functions
MODEL_TRAINERS = {
    "prophet": train_prophet,
    "statsforecast": train_statsforecast,
    "chronos": train_chronos,
}

# COMMAND ----------

# Start MLflow run (ONE parent run for entire job)
with mlflow.start_run(run_name=f"job_{job_id[:8]}") as parent_run:
    mlflow_run_id = parent_run.info.run_id
    experiment_url = mlflow.get_experiment(parent_run.info.experiment_id).name
    run_url = f"#mlflow/experiments/{parent_run.info.experiment_id}/runs/{mlflow_run_id}"

    logger.info(f"MLflow run ID: {mlflow_run_id}")

    # Log job-level params once
    mlflow.log_params({
        "job_id": job_id,
        "user_id": user_id,
        "horizon": horizon,
        "frequency": frequency,
        "confidence_level": confidence_level,
        "random_seed": random_seed,
        "models": ",".join(models_to_train),
        "data_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    })

    # Update Lakebase with MLflow info
    update_job_status(
        job_id,
        "RUNNING",
        progress_percent=10,
        current_step="Training models",
        mlflow_run_id=mlflow_run_id,
    )

    # Train models in parallel
    results = []
    total_models = len(models_to_train)

    # Use ThreadPoolExecutor for parallel training
    with ThreadPoolExecutor(max_workers=min(4, total_models)) as executor:
        futures = {}

        for model_name in models_to_train:
            if model_name in MODEL_TRAINERS:
                future = executor.submit(
                    MODEL_TRAINERS[model_name],
                    train_df,
                    test_df,
                    horizon,
                    frequency,
                    random_seed,
                    hyperparameter_filters,
                )
                futures[future] = model_name
            else:
                logger.warning(f"Unknown model: {model_name}")

        # Collect results
        for i, future in enumerate(as_completed(futures)):
            model_name = futures[future]
            try:
                result = future.result()
                results.append(result)

                # Log model metric to parent
                mlflow.log_metric(f"{model_name}_mape", result["mape"])
                mlflow.log_metric(f"{model_name}_rmse", result["rmse"])

                # Save to Lakebase
                save_forecast_result(
                    job_id=job_id,
                    **result,
                    mlflow_run_id=mlflow_run_id,
                )

                # Update progress
                progress = int(10 + (80 * (i + 1) / total_models))
                update_job_status(
                    job_id,
                    "RUNNING",
                    progress_percent=progress,
                    current_step=f"Completed {model_name}",
                )

                logger.info(f"Completed {model_name}: MAPE={result['mape']:.2f}%")

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                mlflow.log_param(f"{model_name}_error", str(e)[:200])

    # Find best model
    if results:
        best_result = min(results, key=lambda x: x["mape"])
        best_model = best_result["model_name"]
        best_mape = best_result["mape"]

        mlflow.log_metrics({
            "best_mape": best_mape,
            "models_trained": len(results),
        })

        logger.info(f"Best model: {best_model} with MAPE={best_mape:.2f}%")
    else:
        best_model = None
        best_mape = None
        logger.error("No models trained successfully")

    # Log summary artifact
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        summary = {
            "job_id": job_id,
            "best_model": best_model,
            "best_mape": best_mape,
            "models_trained": len(results),
            "results": [{"model": r["model_name"], "mape": r["mape"]} for r in results],
        }
        json.dump(summary, f, indent=2)
        mlflow.log_artifact(f.name, artifact_path="summary")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Finalize

# COMMAND ----------

# Update model rankings in Lakebase
update_model_rankings(job_id)

# Update final status
update_job_status(
    job_id,
    "COMPLETED",
    progress_percent=100,
    current_step="Complete",
    best_model=best_model,
    best_mape=best_mape,
)

logger.info(f"Job {job_id} completed successfully!")
logger.info(f"Best model: {best_model}, MAPE: {best_mape:.2f}%")

# COMMAND ----------

# Return results
output = json.dumps({
    "job_id": job_id,
    "status": "COMPLETED",
    "best_model": best_model,
    "best_mape": best_mape,
    "models_trained": len(results),
    "mlflow_run_id": mlflow_run_id,
}, default=str)

dbutils.notebook.exit(output)
