# Databricks notebook source
"""
Finance Forecasting with Prophet - Databricks Notebook Template
================================================================

This notebook demonstrates how to use the Finance Forecasting Platform
programmatically in Databricks notebooks.

You can use this alongside the web app or independently for automation.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup and Configuration

# COMMAND ----------

# Install required packages if not already available
%pip install mlflow prophet pandas numpy scikit-learn

# COMMAND ----------

import mlflow
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from datetime import datetime
import json

# Set MLflow tracking to Databricks
mlflow.set_tracking_uri("databricks")

# Configure experiment
EXPERIMENT_NAME = "/Users/your.email@company.com/finance-forecasting"
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"Using MLflow experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data

# COMMAND ----------

# Option 1: Load from Unity Catalog
# df = spark.table("main.finance_data.revenue_history").toPandas()

# Option 2: Load from DBFS
# df = pd.read_csv("/dbfs/FileStore/sample_data.csv")

# Option 3: Load from uploaded file
df = pd.read_csv("/dbfs/FileStore/tables/sample_data.csv")

# Display sample
display(df.head())

print(f"Loaded {len(df)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preparation

# COMMAND ----------

# Configuration
TIME_COL = "ds"          # Date column
TARGET_COL = "y"         # Target to forecast
COVARIATES = ["marketing_spend", "promotions"]  # Optional features
HORIZON = 12             # Forecast horizon
FREQUENCY = "MS"         # Month Start
SEASONALITY_MODE = "multiplicative"

# Prepare data in Prophet format
prophet_df = pd.DataFrame()
prophet_df['ds'] = pd.to_datetime(df[TIME_COL])
prophet_df['y'] = pd.to_numeric(df[TARGET_COL])

# Add covariates
for cov in COVARIATES:
    if cov in df.columns:
        prophet_df[cov] = pd.to_numeric(df[cov])

# Sort by date
prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)

print(f"Prepared {len(prophet_df)} rows for training")
print(f"   Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Train/Test Split

# COMMAND ----------

# Split data
test_size = min(HORIZON, len(prophet_df) // 5)
train_df = prophet_df.iloc[:-test_size].copy()
test_df = prophet_df.iloc[-test_size:].copy()

print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Train Prophet Model with MLflow

# COMMAND ----------

# Start MLflow run
with mlflow.start_run(run_name=f"prophet_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
    
    # Log parameters
    mlflow.log_param("time_column", TIME_COL)
    mlflow.log_param("target_column", TARGET_COL)
    mlflow.log_param("covariates", ",".join(COVARIATES))
    mlflow.log_param("horizon", HORIZON)
    mlflow.log_param("frequency", FREQUENCY)
    mlflow.log_param("seasonality_mode", SEASONALITY_MODE)
    mlflow.log_param("train_size", len(train_df))
    mlflow.log_param("test_size", len(test_df))
    
    # Initialize Prophet
    model = Prophet(
        seasonality_mode=SEASONALITY_MODE,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    # Add regressors (covariates)
    for cov in COVARIATES:
        if cov in train_df.columns:
            model.add_regressor(cov)
            print(f"   Added regressor: {cov}")
    
    # Fit model
    print("Training model...")
    model.fit(train_df)
    print("Training complete")
    
    # Validate on test set
    test_future = test_df[['ds'] + [c for c in COVARIATES if c in test_df.columns]].copy()
    validation_forecast = model.predict(test_future)
    
    # Merge actual and predicted
    validation_data = test_df.merge(
        validation_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
        on='ds'
    )
    
    # Compute metrics
    y_true = validation_data['y'].values
    y_pred = validation_data['yhat'].values
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("r2", r2)
    
    print(f"\nMetrics:")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   R²:   {r2:.4f}")
    
    # Create future forecast
    future = model.make_future_dataframe(periods=HORIZON, freq=FREQUENCY)
    
    # Fill covariate values (using mean of last 12 values)
    for cov in COVARIATES:
        if cov in prophet_df.columns:
            last_values = prophet_df[cov].tail(12)
            future[cov] = last_values.mean()
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Save plots
    fig1 = model.plot(forecast)
    mlflow.log_figure(fig1, "forecast_plot.png")
    
    fig2 = model.plot_components(forecast)
    mlflow.log_figure(fig2, "components_plot.png")
    
    # Log model
    mlflow.prophet.log_model(model, "model")
    
    # Save forecast data
    forecast_future = forecast.tail(HORIZON)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_future.to_csv("/tmp/forecast.csv", index=False)
    mlflow.log_artifact("/tmp/forecast.csv")
    
    run_id = run.info.run_id
    print(f"\nMLflow Run ID: {run_id}")
    print(f"   Model URI: runs:/{run_id}/model")

# COMMAND ----------

# MAGIC %md
# MAGIC # Display Results

# COMMAND ----------

# Display validation results
print("Validation Results (Last 6 predictions):")
display(validation_data[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))

# COMMAND ----------

# Display future forecast
print("🔮 Future Forecast:")
display(forecast_future)

# COMMAND ----------

# MAGIC %md
# MAGIC # Register Model to Unity Catalog

# COMMAND ----------

MODEL_NAME = "main.finance_forecast_model"

# Register model
result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=MODEL_NAME,
    tags={"source": "databricks_notebook"}
)

print(f"Model registered:")
print(f"   Name: {MODEL_NAME}")
print(f"   Version: {result.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy to Model Serving (Optional)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput
)

# Initialize client
w = WorkspaceClient()

ENDPOINT_NAME = "finance-forecast-endpoint"

try:
    # Create or update endpoint
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=MODEL_NAME,
                    entity_version=str(result.version),
                    workload_size="Small",
                    scale_to_zero_enabled=True
                )
            ]
        )
    )
    print(f"Endpoint '{ENDPOINT_NAME}' creation initiated")
    print(f"   This may take 5-10 minutes to provision")
except Exception as e:
    if "already exists" in str(e):
        print(f" Endpoint '{ENDPOINT_NAME}' already exists")
        print(f"   Update it manually in the Serving UI")
    else:
        print(f"Deployment failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Test Predictions (After Endpoint is Ready)

# COMMAND ----------

# Wait for endpoint to be ready, then test
import time

def wait_for_endpoint(endpoint_name, timeout=600):
    """Wait for endpoint to be ready"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            endpoint = w.serving_endpoints.get(endpoint_name)
            if endpoint.state.ready == "READY":
                return True
        except:
            pass
        time.sleep(30)
    return False

print("Waiting for endpoint to be ready...")
if wait_for_endpoint(ENDPOINT_NAME):
    print("Endpoint is ready!")
    
    # Make a test prediction
    test_input = {
        "ds": "2025-01-01",
        "marketing_spend": prophet_df["marketing_spend"].mean(),
        "promotions": prophet_df["promotions"].mean()
    }
    
    response = w.serving_endpoints.query(
        name=ENDPOINT_NAME,
        dataframe_records=[test_input]
    )
    
    print(f"\n🔮 Prediction for {test_input['ds']}:")
    print(response)
else:
    print("⏰ Endpoint not ready yet. Check Serving UI.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Summary

# COMMAND ----------

summary = f"""
================================
Forecasting Pipeline Summary
================================

Data:
   - Rows: {len(prophet_df)}
   - Time Range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}
   - Target: {TARGET_COL}
   - Covariates: {', '.join(COVARIATES)}

Model:
   - Type: Prophet
   - Seasonality: {SEASONALITY_MODE}
   - Horizon: {HORIZON} periods

Performance:
   - RMSE: {rmse:.2f}
   - MAPE: {mape:.2f}%
   - R²: {r2:.4f}

MLflow:
   - Run ID: {run_id}
   - Model URI: runs:/{run_id}/model
   - Experiment: {EXPERIMENT_NAME}

Deployment:
   - Model: {MODEL_NAME} v{result.version}
   - Endpoint: {ENDPOINT_NAME}

Pipeline complete!
================================
"""

print(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC # Next Steps
# MAGIC 
# MAGIC 1. **Review Forecast**: Check the plots and validation metrics
# MAGIC 2. **Tune Parameters**: Adjust seasonality_mode, changepoint_prior_scale, etc.
# MAGIC 3. **Add Features**: Include more covariates for better predictions
# MAGIC 4. **Schedule Retraining**: Set up a Databricks Job to run weekly/monthly
# MAGIC 5. **Monitor Performance**: Track endpoint metrics and retrain as needed
# MAGIC 
# MAGIC **For the web app**, visit: [Your App URL]
# MAGIC 
# MAGIC **For MLflow UI**, go to: Machine Learning > Experiments

