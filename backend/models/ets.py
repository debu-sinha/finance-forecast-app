import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import logging
import warnings
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from backend.models.utils import (
    compute_metrics, time_series_cross_validate, compute_prediction_intervals,
    detect_weekly_freq_code, detect_flat_forecast, get_fallback_ets_params
)

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Environment variable to control child run logging
# Set MLFLOW_SKIP_CHILD_RUNS=true to only log the best model (reduces MLflow overhead significantly)
SKIP_CHILD_RUNS = os.environ.get("MLFLOW_SKIP_CHILD_RUNS", "false").lower() == "true"


class ExponentialSmoothingModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow-compatible wrapper for Exponential Smoothing model

    Input format for serving endpoint:
    {
        "dataframe_records": [
            {"periods": 30, "start_date": "2025-01-01"}
        ]
    }

    - periods: Number of periods to forecast (required)
    - start_date: Date to start forecasting from (required)
    - frequency: Optional - uses training frequency if not specified
    """

    def __init__(self, fitted_model, params, frequency, seasonal_periods, weekly_freq_code=None):
        self.fitted_model = fitted_model
        self.params = params
        # Store frequency in human-readable format for consistency
        # Map pandas freq codes to human-readable if needed
        freq_to_human = {'MS': 'monthly', 'W': 'weekly', 'D': 'daily', 'YS': 'yearly'}
        self.frequency = freq_to_human.get(frequency, frequency)
        self.seasonal_periods = seasonal_periods
        # Store the exact weekly frequency code (e.g., 'W-MON') for date alignment
        self.weekly_freq_code = weekly_freq_code or 'W-MON'

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        # Extract parameters from input
        periods = int(model_input['periods'].iloc[0])
        start_date = pd.to_datetime(model_input['start_date'].iloc[0])

        # Map human-readable frequency to pandas freq code
        # Use stored weekly_freq_code for proper day-of-week alignment
        freq_map = {'daily': 'D', 'weekly': self.weekly_freq_code, 'monthly': 'MS', 'yearly': 'YS'}

        # Get frequency from input or use stored default
        if 'frequency' in model_input.columns:
            freq_str = str(model_input['frequency'].iloc[0]).lower()
            pandas_freq = freq_map.get(freq_str, freq_map.get(self.frequency, 'MS'))
        else:
            pandas_freq = freq_map.get(self.frequency, 'MS')

        # Generate forecast
        forecast_values = self.fitted_model.forecast(steps=periods)

        # Generate future dates starting from start_date
        future_dates = pd.date_range(start=start_date, periods=periods + 1, freq=pandas_freq)[1:]

        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values,
            'yhat_lower': forecast_values * 0.9,
            'yhat_upper': forecast_values * 1.1
        })

def generate_ets_training_code(
    params: Dict[str, Any], seasonal_periods: int, horizon: int, frequency: str,
    metrics: Dict[str, float], train_size: int, test_size: int
) -> str:
    """Generate reproducible Python code for Exponential Smoothing model training"""
    freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS', 'yearly': 'YS'}
    pd_freq = freq_map.get(frequency, 'MS')
    trend = params.get('trend', 'None')
    seasonal = params.get('seasonal', 'None')
    
    code = f'''"""
Reproducible Exponential Smoothing (ETS) Model Training Code
Generated for reproducibility
"""
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# ============================================================================
# DATA PREPARATION
# ============================================================================
# Load your data (replace with your actual data source)
# df = pd.read_csv("your_data.csv")
# Ensure you have a datetime column and target column

# Prepare time series data
# ts_data = df['y'].values  # Convert to numpy array

# Split into train/test (train_size={train_size}, test_size={test_size})
test_size = {test_size}
# train_data = ts_data[:-test_size]
# test_data = ts_data[-test_size:]

# ============================================================================
# MODEL INITIALIZATION & TRAINING FLOW
# ============================================================================
# ETS parameters:
# - trend: '{trend}'
# - seasonal: '{seasonal}'
# - seasonal_periods: {seasonal_periods}
# model = ExponentialSmoothing(
#     train_data,
#     seasonal_periods={seasonal_periods},
#     trend={f"'{trend}'" if trend else "None"},
#     seasonal={f"'{seasonal}'" if seasonal else "None"},
#     initialization_method='estimated'
# )
# fitted_model = model.fit(optimized=True)

# print("ETS Model Summary:")
# print(f"  Trend: {trend}")
# print(f"  Seasonal: {seasonal}")
# print(f"  Seasonal Periods: {seasonal_periods}")

# ============================================================================
# VALIDATION (on test set)
# ============================================================================
# test_predictions = fitted_model.forecast(steps=len(test_data))

# Calculate metrics
# rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
# mape = mean_absolute_percentage_error(test_data, test_predictions) * 100
# r2 = r2_score(test_data, test_predictions)

# print(f"\\nValidation Metrics:")
# print(f"  RMSE: {{rmse:.2f}}")
# print(f"  MAPE: {{mape:.2f}}%")
# print(f"  R²: {{r2:.4f}}")

# ============================================================================
# FORECASTING (future periods)
# ============================================================================
# Refit on full dataset for final forecast
# full_data = np.concatenate([train_data, test_data])
# final_model = ExponentialSmoothing(
#     full_data,
#     seasonal_periods={seasonal_periods},
#     trend={f"'{trend}'" if trend else "None"},
#     seasonal={f"'{seasonal}'" if seasonal else "None"},
#     initialization_method='estimated'
# )
# final_fitted_model = final_model.fit(optimized=True)

# Generate forecast for {horizon} periods
# forecast_values = final_fitted_model.forecast(steps={horizon})

# Create forecast dataframe with dates
# last_date = pd.to_datetime(df['ds'].max())
# future_dates = pd.date_range(start=last_date, periods={horizon} + 1, freq='{pd_freq}')[1:]

# forecast_df = pd.DataFrame({{
#     'ds': future_dates,
#     'yhat': forecast_values,
#     'yhat_lower': forecast_values * 0.9,
#     'yhat_upper': forecast_values * 1.1
# }})

# print(f"\\nForecast for {{len(forecast_df)}} periods:")
# print(forecast_df.head())
'''
    return code

def evaluate_ets_params(
    trend: Optional[str],
    seasonal: Optional[str],
    seasonal_periods: int,
    train_y: np.ndarray,
    test_y: np.ndarray,
    parent_run_id: str,
    experiment_id: str,
    skip_mlflow_logging: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a single ETS parameter combination (thread-safe).

    Args:
        skip_mlflow_logging: If True, skip creating MLflow child runs (reduces overhead)
    """
    client = MlflowClient() if not skip_mlflow_logging else None
    run_id = None

    if not skip_mlflow_logging:
        try:
            child_run = client.create_run(
                experiment_id=experiment_id,
                tags={"mlflow.parentRunId": parent_run_id}
            )
            run_id = child_run.info.run_id
        except Exception as e:
            logger.warning(f"Failed to create MLflow child run: {e}")
            skip_mlflow_logging = True

    try:
        if not skip_mlflow_logging and run_id:
            client.log_param(run_id, "model_type", "ExponentialSmoothing")
            client.log_param(run_id, "trend", str(trend))
            client.log_param(run_id, "seasonal", str(seasonal))

        # Train model
        model = ExponentialSmoothing(
            train_y,
            seasonal_periods=seasonal_periods,
            trend=trend,
            seasonal=seasonal,
            initialization_method='estimated'
        )
        fitted_model = model.fit(optimized=True)

        # Validate on test set
        test_predictions = fitted_model.forecast(steps=len(test_y))
        metrics = compute_metrics(test_y, test_predictions)

        if not skip_mlflow_logging and run_id:
            client.log_metric(run_id, "mape", metrics["mape"])
            client.log_metric(run_id, "rmse", metrics["rmse"])
            client.log_metric(run_id, "r2", metrics["r2"])
            client.set_tag(run_id, "mlflow.runName", f"ETS({trend}/{seasonal})")
            client.set_terminated(run_id, "FINISHED")

        logger.info(f"  ✓ ETS({trend}/{seasonal}): MAPE={metrics['mape']:.2f}%, RMSE={metrics['rmse']:.2f}")

        return {
            "params": {"trend": trend, "seasonal": seasonal},
            "metrics": metrics,
            "fitted_model": fitted_model
        }

    except Exception as e:
        if not skip_mlflow_logging and run_id:
            try:
                client.set_terminated(run_id, "FAILED")
            except:
                pass
        logger.warning(f"  ✗ ETS({trend}/{seasonal}) failed: {e}")
        return None

def train_exponential_smoothing_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    frequency: str = 'monthly',
    seasonal_periods: int = 12,
    random_seed: int = 42,
    original_data: List[Dict[str, Any]] = None,
    covariates: List[str] = None,  # Kept for API compatibility but NOT used - ETS is univariate
    hyperparameter_filters: Dict[str, Any] = None,
    forecast_start_date: pd.Timestamp = None  # User's specified end_date for forecast start
) -> Tuple[str, str, Dict[str, float], pd.DataFrame, pd.DataFrame, str, Dict[str, Any]]:
    """
    Train Exponential Smoothing model with hyperparameter tuning and MLflow logging
    """
    logger.info(f"Training ETS model (freq={frequency}, seasonal={seasonal_periods}, seed={random_seed})...")

    # Set random seeds for reproducibility
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    logger.info(f"ETS: Set random seed to {random_seed} for reproducibility")

    # Map frequency to pandas alias
    # For weekly, detect the actual day-of-week from training data
    pd_freq = detect_weekly_freq_code(train_df, frequency)

    best_model = None
    best_metrics = {"mape": float('inf'), "rmse": float('inf')}
    best_params = {}
    best_run_id = None
    best_artifact_uri = None

    # Apply hyperparameter filters from data analysis if provided
    ets_filters = (hyperparameter_filters or {}).get('ETS', {})

    # Extract confidence level for prediction intervals (default 0.95)
    global_filters = (hyperparameter_filters or {}).get('_global', {})
    confidence_level = global_filters.get('confidence_level', 0.95)

    # Use filtered values if provided, otherwise use defaults
    trend_options = ets_filters.get('trend', ['add', None])
    seasonal_options = ets_filters.get('seasonal', ['add', None])

    if ets_filters:
        logger.info(f"📊 Using data-driven ETS filters: trend={trend_options}, seasonal={seasonal_options}")

    # Limit ETS combinations for Databricks Apps
    param_combinations = list(set([(trend, seasonal) for trend in trend_options for seasonal in seasonal_options]))

    # CRITICAL: Filter out degenerate ETS combinations that produce flat/uninformative forecasts
    # (None, None) = simple exponential smoothing with no trend/seasonal = flat forecast
    original_count = len(param_combinations)
    param_combinations = [p for p in param_combinations if not (p[0] is None and p[1] is None)]

    if len(param_combinations) < original_count:
        logger.info(f"🚫 Excluded {original_count - len(param_combinations)} degenerate ETS combinations (flat forecast prevention)")

    # Ensure we have at least some valid combinations
    if len(param_combinations) == 0:
        param_combinations = [('add', None), (None, 'add'), ('add', 'add')]
        logger.warning("All ETS combinations were degenerate. Using fallback: trend or seasonal components.")

    max_ets_combinations = int(os.environ.get('ETS_MAX_COMBINATIONS', '4'))
    if len(param_combinations) > max_ets_combinations:
        # Prefer combinations with trend/seasonal components (non-flat)
        param_combinations.sort(key=lambda x: (x[0] is None, x[1] is None))
        param_combinations = param_combinations[:max_ets_combinations]
        logger.info(f"Limited ETS combinations to {max_ets_combinations}")

    with mlflow.start_run(run_name="ETS_Tuning", nested=True) as parent_run:
        parent_run_id = parent_run.info.run_id
        experiment_id = parent_run.info.experiment_id

        if original_data:
            try:
                original_df = pd.DataFrame(original_data)
                original_df.to_csv("/tmp/original_timeseries_data.csv", index=False)
                mlflow.log_artifact("/tmp/original_timeseries_data.csv", "datasets/raw")
            except Exception as e:
                logger.warning(f"Could not log original data for ETS: {e}")
        
        max_workers = int(os.environ.get('MLFLOW_MAX_WORKERS', '2'))
        if SKIP_CHILD_RUNS:
            logger.info(f"📊 MLFLOW_SKIP_CHILD_RUNS=true: Skipping child run logging to reduce MLflow overhead")
        logger.info(f"Running ETS hyperparameter tuning with {len(param_combinations)} combinations, {max_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    evaluate_ets_params, trend, seasonal, seasonal_periods,
                    train_df['y'].values, test_df['y'].values,
                    parent_run_id, experiment_id, skip_mlflow_logging=SKIP_CHILD_RUNS
                )
                for trend, seasonal in param_combinations
            ]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    metrics = result["metrics"]
                    is_better = False
                    if metrics["mape"] < best_metrics["mape"]: is_better = True
                    elif abs(metrics["mape"] - best_metrics["mape"]) < 0.5 and metrics["rmse"] < best_metrics["rmse"]: is_better = True
                    
                    if is_better:
                        best_metrics = metrics
                        best_model = result["fitted_model"]
                        best_params = result["params"]
                        logger.info(f"  ✨ New best ETS: MAPE={metrics['mape']:.2f}%")


        if best_model is None:
            error_msg = f"ETS training failed: Insufficient data for seasonal={seasonal_periods}. Need at least {seasonal_periods * 2} data points for seasonal models."
            logger.error(error_msg)
            raise Exception(error_msg)

        # Time Series Cross-Validation
        full_y = pd.concat([train_df, test_df])['y'].values
        logger.info(f"Running time series cross-validation for ETS...")

        def ets_fit_fn(y_train):
            model = ExponentialSmoothing(
                y_train,
                seasonal_periods=seasonal_periods if len(y_train) >= seasonal_periods * 2 else None,
                trend=best_params['trend'],
                seasonal=best_params['seasonal'] if len(y_train) >= seasonal_periods * 2 else None,
                initialization_method='estimated'
            )
            return model.fit(optimized=True)

        cv_results = time_series_cross_validate(
            y=full_y,
            model_fit_fn=ets_fit_fn,
            model_predict_fn=lambda fitted_model, steps: fitted_model.forecast(steps=steps),
            n_splits=3,
            horizon=min(horizon, len(test_df))
        )
        if cv_results["mean_mape"] is not None:
            logger.info(f"CV Results: Mean MAPE={cv_results['mean_mape']:.2f}% (±{cv_results['std_mape']:.2f}%), {cv_results['n_splits']} folds")
            best_metrics["cv_mape"] = cv_results["mean_mape"]
            best_metrics["cv_mape_std"] = cv_results["std_mape"]

        # Validation
        test_predictions = best_model.forecast(steps=len(test_df))
        train_predictions = best_model.fittedvalues
        if len(train_predictions) > 0:
            yhat_lower, yhat_upper = compute_prediction_intervals(
                y_train=train_df['y'].values,
                y_pred_train=train_predictions[-len(train_df):] if len(train_predictions) >= len(train_df) else train_predictions,
                forecast_values=test_predictions,
                confidence_level=confidence_level
            )
        else:
            yhat_lower = test_predictions * 0.9
            yhat_upper = test_predictions * 1.1

        validation_data = test_df[['ds', 'y']].copy()
        validation_data['yhat'] = test_predictions
        validation_data['yhat_lower'] = yhat_lower
        validation_data['yhat_upper'] = yhat_upper

        # ==========================================================================
        # DATA LEAKAGE FIX: Final model uses TRAIN DATA ONLY
        # ==========================================================================
        # Previous bug: Refit on full_data (train + test) leaked test info into model
        # Fix: Final model trained on train_df only. Test data is held out for
        # unbiased evaluation. Production deployment can retrain on all data later.
        # ==========================================================================
        # NOTE: For production, you may want to retrain on all available data AFTER
        # evaluation is complete and the model has been validated. This code preserves
        # proper train/test separation for evaluation integrity.
        # ==========================================================================
        final_model = ExponentialSmoothing(
            train_df['y'].values,  # TRAIN ONLY - no test data leakage
            seasonal_periods=seasonal_periods,
            trend=best_params['trend'],
            seasonal=best_params['seasonal'],
            initialization_method='estimated'
        )
        final_fitted_model = final_model.fit(optimized=True)

        # Forecast from end of TRAINING data (not test data)
        forecast_values = final_fitted_model.forecast(steps=horizon)

        # CRITICAL: Detect and handle flat forecasts
        flat_check = detect_flat_forecast(forecast_values, train_df['y'].values)
        if flat_check['is_flat']:
            logger.warning(f"🚨 ETS flat forecast detected with params trend={best_params['trend']}, seasonal={best_params['seasonal']}")
            logger.warning(f"   Reason: {flat_check['flat_reason']}")
            logger.warning(f"   Attempting fallback parameters...")

            # Try fallback parameter combinations
            for fallback_trend, fallback_seasonal in get_fallback_ets_params():
                try:
                    logger.info(f"   Trying fallback: ETS(trend={fallback_trend}, seasonal={fallback_seasonal})")
                    fallback_model = ExponentialSmoothing(
                        train_df['y'].values,
                        seasonal_periods=seasonal_periods,
                        trend=fallback_trend,
                        seasonal=fallback_seasonal,
                        initialization_method='estimated'
                    )
                    fallback_fitted = fallback_model.fit(optimized=True)
                    fallback_forecast = fallback_fitted.forecast(steps=horizon)

                    # Check if fallback is also flat
                    fallback_check = detect_flat_forecast(fallback_forecast, train_df['y'].values)
                    if not fallback_check['is_flat']:
                        logger.info(f"   ✅ Fallback params (trend={fallback_trend}, seasonal={fallback_seasonal}) produces non-flat forecast")
                        forecast_values = fallback_forecast
                        final_fitted_model = fallback_fitted
                        best_params = {'trend': fallback_trend, 'seasonal': fallback_seasonal}
                        break
                    else:
                        logger.warning(f"   ✗ Fallback params (trend={fallback_trend}, seasonal={fallback_seasonal}) also flat")
                except Exception as e:
                    logger.warning(f"   ✗ Fallback params (trend={fallback_trend}, seasonal={fallback_seasonal}) failed: {e}")
                    continue
            else:
                # All fallbacks failed - add noise to prevent identical values
                logger.error("   All fallback parameter combinations produced flat forecasts. Adding minimal variance.")
                forecast_std = np.std(train_df['y'].values) * 0.01
                forecast_values = forecast_values + np.random.normal(0, forecast_std, len(forecast_values))

        # ==========================================================================
        # DATA LEAKAGE FIX: Confidence intervals from TRAIN residuals only
        # ==========================================================================
        final_train_predictions = final_fitted_model.fittedvalues
        if len(final_train_predictions) > 0:
            forecast_lower, forecast_upper = compute_prediction_intervals(
                y_train=train_df['y'].values,  # TRAIN ONLY
                y_pred_train=final_train_predictions,
                forecast_values=forecast_values,
                confidence_level=confidence_level
            )
        else:
            forecast_lower = forecast_values * 0.9
            forecast_upper = forecast_values * 1.1

        # Use forecast_start_date if provided (user's to_date), otherwise use end of TRAINING data
        if forecast_start_date is not None:
            last_date = pd.to_datetime(forecast_start_date).normalize()
            logger.info(f"📅 Using user-specified forecast start: {last_date}")
        else:
            last_date = train_df['ds'].max()  # Use train data end (no test data leakage)
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=pd_freq)[1:]
        logger.info(f"📅 ETS forecast dates: {future_dates.min()} to {future_dates.max()}")

        forecast_data = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values,
            'yhat_lower': forecast_lower,
            'yhat_upper': forecast_upper
        })
        
        # Log datasets
        try:
            train_data_actual = pd.DataFrame({'ds': train_df['ds'], 'y': train_df['y']})
            train_data_actual.to_csv("/tmp/train.csv", index=False)
            mlflow.log_artifact("/tmp/train.csv", "datasets/training")

            eval_data_actual = pd.DataFrame({'ds': test_df['ds'], 'y': test_df['y']})
            eval_data_actual.to_csv("/tmp/eval.csv", index=False)
            mlflow.log_artifact("/tmp/eval.csv", "datasets/training")

            # Log combined data for reference (NOTE: model trained on train only, not this)
            combined_data = pd.concat([train_df, test_df]).sort_values('ds')
            combined_data_actual = pd.DataFrame({'ds': combined_data['ds'], 'y': combined_data['y']})
            combined_data_actual.to_csv("/tmp/full_merged_data.csv", index=False)
            mlflow.log_artifact("/tmp/full_merged_data.csv", "datasets/processed")
        except Exception as e:
            logger.warning(f"Could not log ETS datasets: {e}")

        # Log inference data
        try:
            inference_input = pd.DataFrame({
                'periods': [horizon],
                'start_date': [str(last_date.date()) if hasattr(last_date, 'date') else str(last_date)[:10]]
            })
            inference_input.to_csv("/tmp/input.csv", index=False)
            mlflow.log_artifact("/tmp/input.csv", "datasets/inference")
            forecast_data.to_csv("/tmp/output.csv", index=False)
            mlflow.log_artifact("/tmp/output.csv", "datasets/inference")
        except Exception as e:
            logger.warning(f"Could not log ETS inference data: {e}")

        # Log parameters and metrics
        mlflow.log_param("model_type", "ExponentialSmoothing")
        mlflow.log_param("trend", str(best_params.get('trend', 'None')))
        mlflow.log_param("seasonal", str(best_params.get('seasonal', 'None')))
        mlflow.log_param("seasonal_periods", seasonal_periods)
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("confidence_level", confidence_level)
        mlflow.log_metrics(best_metrics)
        
        # Log reproducible code
        training_code = generate_ets_training_code(
            best_params, seasonal_periods, horizon, frequency, best_metrics, len(train_df), len(test_df)
        )
        mlflow.log_text(training_code, "training_code.py")
        
        # Log as MLflow pyfunc model
        try:
            import sys
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            input_example = pd.DataFrame({
                'periods': [horizon],
                'start_date': [str(last_date.date()) if hasattr(last_date, 'date') else str(last_date)[:10]]
            })
            sample_output = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(1).copy()
            signature = infer_signature(input_example, sample_output)
            weekly_freq_code = detect_weekly_freq_code(train_df, frequency)
            model_wrapper = ExponentialSmoothingModelWrapper(final_fitted_model, best_params, frequency, seasonal_periods, weekly_freq_code)
            
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model_wrapper,
                signature=signature,
                input_example=input_example,
                code_paths=["backend"],
                conda_env={
                    "channels": ["defaults", "conda-forge"],
                    "dependencies": [
                        f"python={python_version}",
                        "pip",
                        {"pip": ["mlflow", "pandas", "numpy", "statsmodels", "scikit-learn", "holidays"]}
                    ],
                    "name": "ets_env"
                }
            )

            # Verify model was logged by checking artifact URI
            artifact_uri = mlflow.get_artifact_uri("model")
            logger.info(f"   ✅ ETS model logged successfully to: {artifact_uri}")

            # Also save a pickle backup for robustness
            model_backup_path = "/tmp/ets_model_backup.pkl"
            with open(model_backup_path, 'wb') as f:
                pickle.dump({
                    'model': final_fitted_model,
                    'wrapper': model_wrapper,
                    'params': best_params,
                    'frequency': frequency,
                    'seasonal_periods': seasonal_periods
                }, f)
            mlflow.log_artifact(model_backup_path, "model_backup")
            logger.info(f"   ✅ ETS model backup saved to model_backup/")

        except Exception as e:
            logger.error(f"   ❌ Failed to log ETS pyfunc model: {e}")
            try:
                model_path = "/tmp/ets_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': final_fitted_model,
                        'params': best_params,
                        'freq': pd_freq,
                        'frequency': frequency
                    }, f)
                mlflow.log_artifact(model_path, "model")
                logger.warning(f"   ⚠️ Logged ETS model as pickle fallback")
            except Exception as fallback_error:
                logger.error(f"   ❌ Fallback pickle also failed: {fallback_error}")
        
        best_run_id = parent_run_id
        best_artifact_uri = parent_run.info.artifact_uri
    
    return best_run_id, f"runs:/{best_run_id}/model", best_metrics, validation_data, forecast_data, best_artifact_uri, best_params
