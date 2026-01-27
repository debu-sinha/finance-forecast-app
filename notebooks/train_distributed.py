# Databricks notebook source
# MAGIC %md
# MAGIC # Distributed Training Notebook
# MAGIC
# MAGIC This notebook is executed on a Databricks cluster when training jobs are submitted
# MAGIC from the Finance Forecasting App. It supports multiple AutoML frameworks.
# MAGIC
# MAGIC **Supported Training Modes:**
# MAGIC - `autogluon` - AutoGluon-TimeSeries (default)
# MAGIC - `statsforecast` - Nixtla StatsForecast
# MAGIC - `neuralforecast` - Nixtla NeuralForecast
# MAGIC - `mmf` - Databricks Many Model Forecasting

# COMMAND ----------
# MAGIC %pip install autogluon.timeseries==1.2.0 statsforecast==1.7.8 mlforecast==0.13.4 --quiet

# COMMAND ----------

import json
import os
import tempfile
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

# Job parameters passed from the App
dbutils.widgets.text("job_id", "", "Job ID")
dbutils.widgets.text("config", "{}", "Training Configuration (JSON)")
dbutils.widgets.text("data_table", "", "Unity Catalog table with training data (optional)")

job_id = dbutils.widgets.get("job_id")
config_json = dbutils.widgets.get("config")
data_table = dbutils.widgets.get("data_table")

# Parse configuration
config = json.loads(config_json) if config_json else {}

logger.info(f"Job ID: {job_id}")
logger.info(f"Config: {json.dumps(config, indent=2)}")
logger.info(f"Data Table: {data_table}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup MLflow Experiment

# COMMAND ----------

# Set up MLflow experiment
experiment_name = config.get("experiment_name", "/Shared/finance-forecasting")
mlflow.set_experiment(experiment_name)

logger.info(f"MLflow experiment: {experiment_name}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

def load_data(config: Dict, data_table: str) -> pd.DataFrame:
    """Load training data from config or Unity Catalog table."""

    if data_table:
        # Load from Unity Catalog
        logger.info(f"Loading data from table: {data_table}")
        df = spark.table(data_table).toPandas()
    elif "data" in config:
        # Load from config (passed as JSON)
        logger.info("Loading data from config")
        df = pd.DataFrame(config["data"])
    else:
        raise ValueError("No data provided. Either pass 'data' in config or 'data_table' parameter.")

    # Parse date column
    time_col = config.get("time_col", "ds")
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])

    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df


df = load_data(config, data_table)
display(df.head(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Training Functions

# COMMAND ----------

def train_autogluon(
    df: pd.DataFrame,
    config: Dict[str, Any],
    job_id: str
) -> Dict[str, Any]:
    """Train using AutoGluon-TimeSeries."""
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    logger.info("Training with AutoGluon-TimeSeries")

    # Configuration
    time_col = config.get("time_col", "ds")
    target_col = config.get("target_col", "y")
    id_col = config.get("id_col")
    horizon = config.get("horizon", 12)
    freq = config.get("frequency", "W")
    time_limit = config.get("time_limit", 600)
    presets = config.get("presets", "medium_quality")

    # Prepare TimeSeriesDataFrame
    if id_col and id_col in df.columns:
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column=id_col,
            timestamp_column=time_col,
        )
    else:
        # Single series - add dummy ID
        df["item_id"] = "series_1"
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column="item_id",
            timestamp_column=time_col,
        )

    # Train predictor
    with tempfile.TemporaryDirectory() as tmpdir:
        predictor_path = os.path.join(tmpdir, f"predictor_{job_id}")

        predictor = TimeSeriesPredictor(
            path=predictor_path,
            prediction_length=horizon,
            target=target_col,
            eval_metric="MAPE",
            freq=freq,
        ).fit(
            ts_df,
            presets=presets,
            time_limit=time_limit,
        )

        # Get results
        predictions = predictor.predict(ts_df)
        leaderboard = predictor.leaderboard()
        best_model = leaderboard.iloc[0]

        # Log to MLflow
        mlflow.log_params({
            "framework": "autogluon",
            "horizon": horizon,
            "frequency": freq,
            "presets": presets,
            "time_limit": time_limit,
        })

        mlflow.log_metric("mape", float(best_model["score_val"]))
        mlflow.log_metric("training_time", float(best_model["fit_time_marginal"]))

        # Log leaderboard
        leaderboard.to_csv(os.path.join(tmpdir, "leaderboard.csv"), index=False)
        mlflow.log_artifact(os.path.join(tmpdir, "leaderboard.csv"))

        # Log predictor
        mlflow.log_artifacts(predictor_path, artifact_path="predictor")

        # Prepare forecast output
        forecast_df = predictions.reset_index()
        forecast_df.to_csv(os.path.join(tmpdir, "forecast.csv"), index=False)
        mlflow.log_artifact(os.path.join(tmpdir, "forecast.csv"))

        return {
            "framework": "autogluon",
            "best_model": str(best_model["model"]),
            "mape": float(best_model["score_val"]),
            "leaderboard": leaderboard.to_dict(orient="records"),
            "forecast": forecast_df.to_dict(orient="records"),
        }


def train_statsforecast(
    df: pd.DataFrame,
    config: Dict[str, Any],
    job_id: str
) -> Dict[str, Any]:
    """Train using Nixtla StatsForecast."""
    from statsforecast import StatsForecast
    from statsforecast.models import (
        AutoARIMA, AutoETS, AutoTheta, AutoCES,
        SeasonalNaive, Naive
    )

    logger.info("Training with StatsForecast")

    # Configuration
    time_col = config.get("time_col", "ds")
    target_col = config.get("target_col", "y")
    id_col = config.get("id_col", "unique_id")
    horizon = config.get("horizon", 12)
    freq = config.get("frequency", "W")
    season_length = config.get("season_length", 52 if freq == "W" else 12)

    # Prepare data in StatsForecast format
    sf_df = df.rename(columns={
        time_col: "ds",
        target_col: "y",
    })

    if id_col not in sf_df.columns:
        sf_df["unique_id"] = "series_1"
    else:
        sf_df = sf_df.rename(columns={id_col: "unique_id"})

    sf_df = sf_df[["unique_id", "ds", "y"]].copy()
    sf_df["ds"] = pd.to_datetime(sf_df["ds"])

    # Define models
    models = [
        AutoARIMA(season_length=season_length),
        AutoETS(season_length=season_length),
        AutoTheta(season_length=season_length),
        SeasonalNaive(season_length=season_length),
        Naive(),
    ]

    # Train
    sf = StatsForecast(
        models=models,
        freq=freq,
        n_jobs=-1,
    )

    # Cross-validation for metrics
    cv_results = sf.cross_validation(
        df=sf_df,
        h=horizon,
        n_windows=3,
    )

    # Calculate MAPE per model
    model_names = [m.__class__.__name__ for m in models]
    metrics = {}
    for model in model_names:
        if model in cv_results.columns:
            actuals = cv_results["y"]
            preds = cv_results[model]
            mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
            metrics[model] = mape

    best_model = min(metrics, key=metrics.get)
    best_mape = metrics[best_model]

    # Generate forecast
    sf.fit(sf_df)
    forecast = sf.predict(h=horizon, level=[80, 95])

    # Log to MLflow
    mlflow.log_params({
        "framework": "statsforecast",
        "horizon": horizon,
        "frequency": freq,
        "season_length": season_length,
        "models": str(model_names),
    })

    mlflow.log_metric("mape", best_mape)

    for model, mape in metrics.items():
        mlflow.log_metric(f"mape_{model}", mape)

    # Log forecast
    with tempfile.TemporaryDirectory() as tmpdir:
        forecast_df = forecast.reset_index()
        forecast_df.to_csv(os.path.join(tmpdir, "forecast.csv"), index=False)
        mlflow.log_artifact(os.path.join(tmpdir, "forecast.csv"))

        # Log metrics
        pd.DataFrame([metrics]).to_csv(os.path.join(tmpdir, "metrics.csv"), index=False)
        mlflow.log_artifact(os.path.join(tmpdir, "metrics.csv"))

    return {
        "framework": "statsforecast",
        "best_model": best_model,
        "mape": best_mape,
        "model_metrics": metrics,
        "forecast": forecast_df.to_dict(orient="records"),
    }


def train_neuralforecast(
    df: pd.DataFrame,
    config: Dict[str, Any],
    job_id: str
) -> Dict[str, Any]:
    """Train using Nixtla NeuralForecast."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS, NBEATS, TFT
    from neuralforecast.losses.pytorch import MAE

    logger.info("Training with NeuralForecast")

    # Configuration
    time_col = config.get("time_col", "ds")
    target_col = config.get("target_col", "y")
    id_col = config.get("id_col", "unique_id")
    horizon = config.get("horizon", 12)
    freq = config.get("frequency", "W")
    max_steps = config.get("max_steps", 500)

    # Prepare data
    nf_df = df.rename(columns={
        time_col: "ds",
        target_col: "y",
    })

    if id_col not in nf_df.columns:
        nf_df["unique_id"] = "series_1"
    else:
        nf_df = nf_df.rename(columns={id_col: "unique_id"})

    nf_df = nf_df[["unique_id", "ds", "y"]].copy()
    nf_df["ds"] = pd.to_datetime(nf_df["ds"])

    # Define models (using smaller configs for speed)
    input_size = min(horizon * 3, len(nf_df) // 2)

    models = [
        NHITS(
            h=horizon,
            input_size=input_size,
            max_steps=max_steps,
            loss=MAE(),
        ),
        NBEATS(
            h=horizon,
            input_size=input_size,
            max_steps=max_steps,
            loss=MAE(),
        ),
    ]

    # Train
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(nf_df)

    # Cross-validation
    cv_results = nf.cross_validation(df=nf_df, n_windows=2)

    # Calculate metrics
    metrics = {}
    for model in ["NHITS", "NBEATS"]:
        if model in cv_results.columns:
            actuals = cv_results["y"]
            preds = cv_results[model]
            mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
            metrics[model] = mape

    best_model = min(metrics, key=metrics.get) if metrics else "NHITS"
    best_mape = metrics.get(best_model, 0)

    # Generate forecast
    forecast = nf.predict()

    # Log to MLflow
    mlflow.log_params({
        "framework": "neuralforecast",
        "horizon": horizon,
        "frequency": freq,
        "input_size": input_size,
        "max_steps": max_steps,
    })

    mlflow.log_metric("mape", best_mape)

    for model, mape in metrics.items():
        mlflow.log_metric(f"mape_{model}", mape)

    # Log forecast
    with tempfile.TemporaryDirectory() as tmpdir:
        forecast_df = forecast.reset_index()
        forecast_df.to_csv(os.path.join(tmpdir, "forecast.csv"), index=False)
        mlflow.log_artifact(os.path.join(tmpdir, "forecast.csv"))

    return {
        "framework": "neuralforecast",
        "best_model": best_model,
        "mape": best_mape,
        "model_metrics": metrics,
        "forecast": forecast_df.to_dict(orient="records"),
    }

# COMMAND ----------
# MAGIC %md
# MAGIC ## Execute Training

# COMMAND ----------

# Determine training mode
training_mode = config.get("training_mode", "autogluon").lower()

logger.info(f"Training mode: {training_mode}")

# Start MLflow run
with mlflow.start_run(run_name=f"job_{job_id[:8]}_{training_mode}") as run:
    mlflow_run_id = run.info.run_id
    logger.info(f"MLflow run ID: {mlflow_run_id}")

    # Log job metadata
    mlflow.log_params({
        "job_id": job_id,
        "training_mode": training_mode,
        "data_rows": len(df),
        "data_columns": len(df.columns),
    })

    # Log config as artifact
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f, indent=2)
        mlflow.log_artifact(f.name, artifact_path="config")

    # Train based on mode
    try:
        if training_mode == "autogluon":
            results = train_autogluon(df, config, job_id)
        elif training_mode == "statsforecast":
            results = train_statsforecast(df, config, job_id)
        elif training_mode == "neuralforecast":
            results = train_neuralforecast(df, config, job_id)
        else:
            raise ValueError(f"Unknown training mode: {training_mode}")

        # Add MLflow run ID to results
        results["mlflow_run_id"] = mlflow_run_id
        results["status"] = "success"

        logger.info(f"Training completed successfully!")
        logger.info(f"Best model: {results.get('best_model')}")
        logger.info(f"MAPE: {results.get('mape'):.2f}%")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        results = {
            "status": "failed",
            "error": str(e),
            "mlflow_run_id": mlflow_run_id,
        }
        mlflow.log_param("error", str(e)[:500])
        raise

# COMMAND ----------
# MAGIC %md
# MAGIC ## Return Results

# COMMAND ----------

# Output results for the job service to retrieve
output = json.dumps(results, default=str)
logger.info(f"Returning results: {len(output)} characters")

dbutils.notebook.exit(output)
