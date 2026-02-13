import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import logging
import warnings
import itertools
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import holidays
from backend.models.utils import (
    compute_metrics, time_series_cross_validate, compute_prediction_intervals,
    detect_weekly_freq_code, detect_flat_forecast, get_fallback_arima_orders,
    get_fallback_sarimax_orders
)
from backend.utils.logging_utils import log_io

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Environment variable to control child run logging
# Set MLFLOW_SKIP_CHILD_RUNS=true to only log the best model (reduces MLflow overhead significantly)
SKIP_CHILD_RUNS = os.environ.get("MLFLOW_SKIP_CHILD_RUNS", "false").lower() == "true"


class ARIMAModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow-compatible wrapper for ARIMA model

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

    def __init__(self, fitted_model, order, frequency, weekly_freq_code=None, historical_max=None, historical_mean=None):
        self.fitted_model = fitted_model
        self.order = order
        # Store frequency in human-readable format for consistency
        # Map pandas freq codes to human-readable if needed
        freq_to_human = {'MS': 'monthly', 'W': 'weekly', 'D': 'daily', 'YS': 'yearly'}
        self.frequency = freq_to_human.get(frequency, frequency)
        # Store the exact weekly frequency code (e.g., 'W-MON') for date alignment
        self.weekly_freq_code = weekly_freq_code or 'W-MON'
        # Store historical bounds for explosion detection at inference time
        self.historical_max = historical_max or 0
        self.historical_mean = historical_mean or 0

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

        # FIX: Align start_date to the frequency anchor to avoid skipping periods
        if self.frequency == 'weekly' and '-' in pandas_freq:
            anchor_day_map = {'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 'FRI': 4, 'SAT': 5, 'SUN': 6}
            anchor_day_name = pandas_freq.split('-')[1]
            anchor_day = anchor_day_map.get(anchor_day_name, 0)

            days_since_anchor = (start_date.dayofweek - anchor_day) % 7
            if days_since_anchor != 0:
                start_date = start_date - pd.Timedelta(days=days_since_anchor)

        # Generate future dates starting from start_date
        future_dates = pd.date_range(start=start_date, periods=periods + 1, freq=pandas_freq)[1:]

        # CRITICAL: Detect numerical explosion (e.g., ARIMA producing extreme values)
        # Replace with naive forecast if values exceed 10x the historical max
        if np.any(np.isnan(forecast_values)) or np.any(np.isinf(forecast_values)):
            fallback = max(self.historical_mean, 0.0) if self.historical_mean else 0.0
            logger.error(f"ARIMA wrapper: NaN/Inf detected in forecast. Using fallback={fallback:,.0f}.")
            forecast_values = np.full(len(forecast_values), fallback)
        elif self.historical_max > 0 and np.max(np.abs(forecast_values)) > 10 * self.historical_max:
            fallback = max(self.historical_mean, 0.0)
            logger.error(f"ARIMA wrapper: numerical explosion detected (max={np.max(np.abs(forecast_values)):.2e}, hist_max={self.historical_max:,.0f}). Using fallback={fallback:,.0f}.")
            forecast_values = np.full(len(forecast_values), fallback)

        # CRITICAL: Clip negative forecasts - financial metrics cannot be negative
        forecast_values = np.maximum(forecast_values, 0.0)
        lower_bounds = np.maximum(forecast_values * 0.9, 0.0)
        upper_bounds = forecast_values * 1.1

        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values,
            'yhat_lower': lower_bounds,
            'yhat_upper': upper_bounds
        })

class SARIMAXModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow-compatible wrapper for SARIMAX model

    Input format for serving endpoint (Mode 1 - Simple, no covariates):
    {
        "dataframe_records": [
            {"periods": 30, "start_date": "2025-01-01"}
        ]
    }

    Input format for serving endpoint (Mode 2 - With covariates):
    {
        "dataframe_records": [
            {"ds": "2025-01-01", "Black Friday": 0, "Thanksgiving": 0},
            {"ds": "2025-01-02", "Black Friday": 0, "Thanksgiving": 0},
            ...
        ]
    }
    """

    def __init__(self, fitted_model, order, seasonal_order, frequency, covariates, covariate_means, weekly_freq_code=None, historical_max=None, historical_mean=None):
        self.fitted_model = fitted_model
        self.order = order
        self.seasonal_order = seasonal_order
        self.covariates = covariates or []
        self.covariate_means = covariate_means or {}
        # Store frequency in human-readable format
        freq_to_human = {'MS': 'monthly', 'W': 'weekly', 'D': 'daily', 'YS': 'yearly'}
        self.frequency = freq_to_human.get(frequency, frequency)
        # Store the exact weekly frequency code (e.g., 'W-MON') for date alignment
        self.weekly_freq_code = weekly_freq_code or 'W-MON'
        # Store historical bounds for explosion detection at inference time
        self.historical_max = historical_max or 0
        self.historical_mean = historical_mean or 0

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd
        import numpy as np

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        # Use stored weekly_freq_code for proper day-of-week alignment
        freq_map = {'daily': 'D', 'weekly': self.weekly_freq_code, 'monthly': 'MS', 'yearly': 'YS'}
        pandas_freq = freq_map.get(self.frequency, 'MS')

        # Mode 1: Simple forecast (periods + start_date)
        if 'periods' in model_input.columns:
            periods = int(model_input['periods'].iloc[0])
            start_date = pd.to_datetime(model_input['start_date'].iloc[0])

            # FIX: Align start_date to the frequency anchor to avoid skipping periods
            if self.frequency == 'weekly' and '-' in pandas_freq:
                anchor_day_map = {'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 'FRI': 4, 'SAT': 5, 'SUN': 6}
                anchor_day_name = pandas_freq.split('-')[1]
                anchor_day = anchor_day_map.get(anchor_day_name, 0)

                days_since_anchor = (start_date.dayofweek - anchor_day) % 7
                if days_since_anchor != 0:
                    start_date = start_date - pd.Timedelta(days=days_since_anchor)

            future_dates = pd.date_range(start=start_date, periods=periods + 1, freq=pandas_freq)[1:]

            # If model has covariates, use historical means
            if self.covariates:
                exog_future = pd.DataFrame(index=range(periods))
                for cov in self.covariates:
                    exog_future[cov] = self.covariate_means.get(cov, 0)
                forecast_values = self.fitted_model.forecast(steps=periods, exog=exog_future.values)
            else:
                forecast_values = self.fitted_model.forecast(steps=periods)

        # Mode 2: Advanced forecast with covariates
        else:
            model_input['ds'] = pd.to_datetime(model_input['ds'])
            future_dates = model_input['ds']
            periods = len(future_dates)

            if self.covariates:
                exog_future = model_input[self.covariates].copy()
                # Fill missing covariates with means
                for cov in self.covariates:
                    if cov not in exog_future.columns:
                        exog_future[cov] = self.covariate_means.get(cov, 0)
                    else:
                        exog_future[cov] = exog_future[cov].fillna(self.covariate_means.get(cov, 0))
                forecast_values = self.fitted_model.forecast(steps=periods, exog=exog_future.values)
            else:
                forecast_values = self.fitted_model.forecast(steps=periods)

        # CRITICAL: Detect numerical explosion (e.g., SARIMAX producing 4.8e+31)
        # Replace with naive forecast if values exceed 10x the historical max
        if hasattr(self, 'historical_max') and self.historical_max > 0:
            if np.any(np.isnan(forecast_values)) or np.any(np.isinf(forecast_values)) or np.max(np.abs(forecast_values)) > 10 * self.historical_max:
                logger.error(f"SARIMAX wrapper: numerical explosion detected (max={np.max(np.abs(forecast_values)):.2e}, hist_max={self.historical_max:,.0f}). Using historical mean as fallback.")
                forecast_values = np.full(len(forecast_values), self.historical_mean if hasattr(self, 'historical_mean') else self.historical_max * 0.5)

        # CRITICAL: Clip negative forecasts - financial metrics cannot be negative
        forecast_values = np.maximum(forecast_values, 0.0)
        lower_bounds = np.maximum(forecast_values * 0.9, 0.0)
        upper_bounds = forecast_values * 1.1

        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values,
            'yhat_lower': lower_bounds,
            'yhat_upper': upper_bounds
        })

@log_io
def generate_arima_training_code(
    order: Tuple[int, int, int], horizon: int, frequency: str,
    metrics: Dict[str, float], train_size: int, test_size: int
) -> str:
    """Generate reproducible Python code for ARIMA model training"""
    freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS', 'yearly': 'YS'}
    pd_freq = freq_map.get(frequency, 'MS')
    
    code = f'''"""
Reproducible ARIMA Model Training Code
Generated for reproducibility
"""
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# ============================================================================
# DATA PREPARATION
# ============================================================================
# Load your data (replace with your actual data source)
# df = pd.read_csv("your_data.csv")
# Ensure you have a datetime column and target column

# Prepare time series data
# Assuming 'ds' is datetime column and 'y' is target column
# ts_data = df['y'].values  # Convert to numpy array

# Split into train/test (train_size={train_size}, test_size={test_size})
test_size = {test_size}
# train_data = ts_data[:-test_size]
# test_data = ts_data[-test_size:]

# ============================================================================
# MODEL INITIALIZATION & TRAINING FLOW
# ============================================================================
# ARIMA order: ({order[0]}, {order[1]}, {order[2]})
# p={order[0]} (AR order), d={order[1]} (differencing), q={order[2]} (MA order)
# model = ARIMA(train_data, order=({order[0]}, {order[1]}, {order[2]}))
# fitted_model = model.fit()

# print("ARIMA Model Summary:")
# print(fitted_model.summary())

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
# final_model = ARIMA(full_data, order=({order[0]}, {order[1]}, {order[2]}))
# final_fitted_model = final_model.fit()

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

@log_io
def generate_sarimax_training_code(
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    horizon: int,
    frequency: str,
    covariates: List[str],
    metrics: Dict[str, float],
    train_size: int,
    test_size: int
) -> str:
    """Generate reproducible Python code for SARIMAX model training"""
    freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS', 'yearly': 'YS'}
    pd_freq = freq_map.get(frequency, 'MS')

    cov_str = str(covariates) if covariates else "[]"

    code = f'''"""
Reproducible SARIMAX Model Training Code
Generated for reproducibility - Supports external regressors (covariates)
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# =============================================================================
# Configuration
# =============================================================================
ORDER = {order}  # (p, d, q)
SEASONAL_ORDER = {seasonal_order}  # (P, D, Q, s)
FREQUENCY = '{frequency}'
HORIZON = {horizon}
COVARIATES = {cov_str}

# Metrics from training:
# MAPE: {metrics.get('mape', 'N/A'):.4f}%
# RMSE: {metrics.get('rmse', 'N/A'):.4f}
# R2: {metrics.get('r2', 'N/A'):.4f}

# Data splits:
# Training size: {train_size}
# Test size: {test_size}

# =============================================================================
# Load Data
# =============================================================================
# Replace with your data loading logic
# df = pd.read_csv('your_data.csv')
# df['ds'] = pd.to_datetime(df['your_date_column'])
# df['y'] = df['your_target_column']

# =============================================================================
# Prepare Exogenous Variables (if any)
# =============================================================================
exog_train = None
exog_test = None
# if COVARIATES:
#     exog_train = train_df[COVARIATES].values
#     exog_test = test_df[COVARIATES].values

# =============================================================================
# Train SARIMAX Model
# =============================================================================
# model = SARIMAX(
#     train_df['y'].values,
#     exog=exog_train,
#     order=ORDER,
#     seasonal_order=SEASONAL_ORDER,
#     enforce_stationarity=False,
#     enforce_invertibility=False
# )
# fitted_model = model.fit(disp=False)

# =============================================================================
# Validation
# =============================================================================
# test_predictions = fitted_model.forecast(steps=len(test_df), exog=exog_test)
# mape = mean_absolute_percentage_error(test_df['y'], test_predictions) * 100
# rmse = np.sqrt(mean_squared_error(test_df['y'], test_predictions))
# r2 = r2_score(test_df['y'], test_predictions)

# print(f"SARIMAX{order}x{seasonal_order} - MAPE: {{mape:.2f}}%, RMSE: {{rmse:.2f}}, R2: {{r2:.4f}}")

# =============================================================================
# Forecast
# =============================================================================
# For future forecasts, prepare exogenous variables for forecast horizon
# exog_future = future_df[COVARIATES].values if COVARIATES else None
# forecast = fitted_model.forecast(steps=HORIZON, exog=exog_future)
'''
    return code

@log_io
def evaluate_arima_params(
    order: Tuple[int, int, int],
    train_y: np.ndarray,
    test_y: np.ndarray,
    parent_run_id: str,
    experiment_id: str,
    skip_mlflow_logging: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a single ARIMA parameter combination (thread-safe).

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
            client.log_param(run_id, "model_type", "ARIMA")
            client.log_param(run_id, "order", str(order))
            client.log_param(run_id, "p", order[0])
            client.log_param(run_id, "d", order[1])
            client.log_param(run_id, "q", order[2])

        # Train model
        model = ARIMA(train_y, order=order)
        fitted_model = model.fit()

        # Get AIC for model selection (lower is better)
        aic = fitted_model.aic

        # Validate on test set
        test_predictions = fitted_model.forecast(steps=len(test_y))
        metrics = compute_metrics(test_y, test_predictions)

        # Penalize orders that produce flat test predictions
        flat_check = detect_flat_forecast(test_predictions, train_y)
        if flat_check['is_flat']:
            logger.warning(f"  ⚠️ ARIMA{order}: flat test predictions — penalizing")
            metrics["mape"] = float('inf')

        metrics["aic"] = aic

        if not skip_mlflow_logging and run_id:
            client.log_metric(run_id, "mape", metrics["mape"])
            client.log_metric(run_id, "rmse", metrics["rmse"])
            client.log_metric(run_id, "r2", metrics["r2"])
            client.log_metric(run_id, "aic", aic)
            client.set_tag(run_id, "mlflow.runName", f"ARIMA_{order}")
            client.set_terminated(run_id, "FINISHED")

        logger.info(f"  ✓ ARIMA{order}: MAPE={metrics['mape']:.2f}%, RMSE={metrics['rmse']:.2f}, AIC={aic:.1f}")

        return {
            "order": order,
            "metrics": metrics,
            "fitted_model": fitted_model,
            "aic": aic
        }

    except Exception as e:
        if not skip_mlflow_logging and run_id:
            try:
                client.set_terminated(run_id, "FAILED")
            except:
                pass
        logger.warning(f"  ✗ ARIMA{order} failed: {e}")
        return None

@log_io
def evaluate_sarimax_params(
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    train_y: np.ndarray,
    test_y: np.ndarray,
    train_exog: Optional[np.ndarray],
    test_exog: Optional[np.ndarray],
    parent_run_id: str,
    experiment_id: str,
    skip_mlflow_logging: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a single SARIMAX parameter combination (thread-safe).

    Args:
        skip_mlflow_logging: If True, skip creating MLflow child runs (reduces overhead)
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

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
            client.log_param(run_id, "model_type", "SARIMAX")
            client.log_param(run_id, "order", str(order))
            client.log_param(run_id, "seasonal_order", str(seasonal_order))

        model = SARIMAX(
            train_y,
            exog=train_exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False, maxiter=100)

        # Get AIC for model selection (lower is better)
        aic = fitted_model.aic

        test_predictions = fitted_model.forecast(steps=len(test_y), exog=test_exog)
        metrics = compute_metrics(test_y, test_predictions)

        # Penalize orders that produce flat test predictions
        flat_check = detect_flat_forecast(test_predictions, train_y)
        if flat_check['is_flat']:
            logger.warning(f"  ⚠️ SARIMAX{order}x{seasonal_order}: flat test predictions — penalizing")
            metrics["mape"] = float('inf')

        metrics["aic"] = aic

        if not skip_mlflow_logging and run_id:
            client.log_metric(run_id, "mape", metrics["mape"])
            client.log_metric(run_id, "rmse", metrics["rmse"])
            client.log_metric(run_id, "r2", metrics["r2"])
            client.log_metric(run_id, "aic", aic)
            client.set_tag(run_id, "mlflow.runName", f"SARIMAX{order}x{seasonal_order}")
            client.set_terminated(run_id, "FINISHED")

        logger.info(f"  ✓ SARIMAX{order}x{seasonal_order}: MAPE={metrics['mape']:.2f}%, RMSE={metrics['rmse']:.2f}, AIC={aic:.1f}")

        return {
            "order": order,
            "seasonal_order": seasonal_order,
            "metrics": metrics,
            "fitted_model": fitted_model,
            "aic": aic
        }

    except Exception as e:
        if not skip_mlflow_logging and run_id:
            try:
                client.set_terminated(run_id, "FAILED")
            except:
                pass
        logger.warning(f"  ✗ SARIMAX{order}x{seasonal_order} failed: {e}")
        return None

@log_io
def train_arima_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    frequency: str = 'monthly',
    order: Tuple[int, int, int] = None,
    random_seed: int = 42,
    original_data: List[Dict[str, Any]] = None,
    covariates: List[str] = None,  # Kept for API compatibility but NOT used - ARIMA is univariate
    hyperparameter_filters: Dict[str, Any] = None,
    forecast_start_date: pd.Timestamp = None  # User's specified end_date for forecast start
) -> Tuple[str, str, Dict[str, float], pd.DataFrame, pd.DataFrame, str, Tuple[int, int, int]]:
    """
    Train ARIMA model with hyperparameter tuning and MLflow logging
    """
    logger.info(f"Training ARIMA model (freq={frequency}, seed={random_seed})...")

    # Set random seeds for reproducibility
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    logger.info(f"ARIMA: Set random seed to {random_seed} for reproducibility")

    # Map frequency to pandas alias
    # For weekly, detect the actual day-of-week from training data
    pd_freq = detect_weekly_freq_code(train_df, frequency)

    best_model = None
    best_metrics = {"mape": float('inf'), "rmse": float('inf')}
    best_order = order
    best_fitted_model = None
    best_run_id = None
    best_artifact_uri = None
    
    # Define grid search space if order is not provided
    # Apply hyperparameter filters from data analysis if provided
    arima_filters = (hyperparameter_filters or {}).get('ARIMA', {})

    # Extract confidence level for prediction intervals (default 0.95)
    global_filters = (hyperparameter_filters or {}).get('_global', {})
    confidence_level = global_filters.get('confidence_level', 0.95)

    if order is None:
        max_arima_combinations = int(os.environ.get('ARIMA_MAX_COMBINATIONS', '6'))

        # Use filtered values if provided, otherwise use defaults
        p_values = arima_filters.get('p_values', [0, 1, 2])
        d_values = arima_filters.get('d_values', [0, 1])
        q_values = arima_filters.get('q_values', [0, 1])

        if arima_filters:
            logger.info(f"📊 Using data-driven ARIMA filters: p={p_values}, d={d_values}, q={q_values}")

        # Ensure all values are integers (filters may come as strings from JSON)
        p_values = [int(p) for p in p_values]
        d_values = [int(d) for d in d_values]
        q_values = [int(q) for q in q_values]

        # Generate all combinations as integer tuples
        all_orders = [(int(p), int(d), int(q)) for p, d, q in itertools.product(p_values, d_values, q_values)]
        all_orders = list(set(all_orders))  # Remove duplicates

        # CRITICAL: Filter out degenerate models that produce flat/uninformative forecasts
        # (0,0,0) = constant mean
        # (0,1,0) = random walk (flat forecast at last value)
        # (0,d,0) = pure differencing (no learning)
        degenerate_orders = {(0, 0, 0), (0, 1, 0), (0, 2, 0)}  # Use set for O(1) lookup
        original_count = len(all_orders)
        all_orders = [o for o in all_orders if o not in degenerate_orders]

        if len(all_orders) < original_count:
            excluded_count = original_count - len(all_orders)
            logger.info(f"🚫 Excluded {excluded_count} degenerate ARIMA orders: {[o for o in [(0,0,0), (0,1,0), (0,2,0)] if o in degenerate_orders]}")

        # Ensure we have at least some valid orders
        if len(all_orders) == 0:
            # Fall back to simple but informative models
            all_orders = [(1, 1, 0), (0, 1, 1), (1, 1, 1)]
            logger.warning("All ARIMA orders were degenerate. Using fallback: (1,1,0), (0,1,1), (1,1,1)")

        if len(all_orders) > max_arima_combinations:
            all_orders.sort(key=lambda x: sum(x))
            orders = all_orders[:max_arima_combinations]
            logger.info(f"Limited ARIMA combinations to {max_arima_combinations} (from {len(all_orders)} total)")
        else:
            orders = all_orders
    else:
        # User-specified order - warn if degenerate
        if order in [(0, 0, 0), (0, 1, 0), (0, 2, 0)]:
            logger.warning(f"⚠️ Specified ARIMA order {order} is a degenerate model (random walk). Consider using (1,1,0) or (1,1,1) instead.")
        orders = [order]
    
    with mlflow.start_run(run_name="ARIMA_Tuning", nested=True) as parent_run:
        parent_run_id = parent_run.info.run_id
        experiment_id = parent_run.info.experiment_id

        # Log original data
        if original_data:
            try:
                original_df = pd.DataFrame(original_data)
                original_df.to_csv("/tmp/original_timeseries_data.csv", index=False)
                mlflow.log_artifact("/tmp/original_timeseries_data.csv", "datasets/raw")
            except Exception as e:
                logger.warning(f"Could not log original data for ARIMA: {e}")
        
        max_workers = int(os.environ.get('MLFLOW_MAX_WORKERS', '1'))
        if SKIP_CHILD_RUNS:
            logger.info(f"📊 MLFLOW_SKIP_CHILD_RUNS=true: Skipping child run logging to reduce MLflow overhead")
        logger.info(f"Running ARIMA hyperparameter tuning with {len(orders)} combinations, {max_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    evaluate_arima_params, order, train_df['y'].values, test_df['y'].values,
                    parent_run_id, experiment_id, skip_mlflow_logging=SKIP_CHILD_RUNS
                )
                for order in orders
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
                        best_order = result["order"]
                        best_fitted_model = result["fitted_model"]
                        logger.info(f"  ✨ New best ARIMA{result['order']}: MAPE={metrics['mape']:.2f}%")

        if best_fitted_model is None: raise Exception("ARIMA training failed")

        # Time Series Cross-Validation
        full_y = pd.concat([train_df, test_df])['y'].values
        logger.info(f"Running time series cross-validation for ARIMA...")
        cv_results = time_series_cross_validate(
            y=full_y,
            model_fit_fn=lambda y_train: ARIMA(y_train, order=best_order).fit(),
            model_predict_fn=lambda fitted_model, steps: fitted_model.forecast(steps=steps),
            n_splits=3,
            horizon=min(horizon, len(test_df))
        )
        if cv_results["mean_mape"] is not None:
            logger.info(f"CV Results: Mean MAPE={cv_results['mean_mape']:.2f}% (±{cv_results['std_mape']:.2f}%), {cv_results['n_splits']} folds")
            best_metrics["cv_mape"] = cv_results["mean_mape"]
            best_metrics["cv_mape_std"] = cv_results["std_mape"]

        # Validation with proper confidence intervals
        test_predictions = best_fitted_model.forecast(steps=len(test_df))
        train_predictions = best_fitted_model.fittedvalues
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

        # Refit and Forecast
        full_data = pd.concat([train_df, test_df]).sort_values('ds')
        final_model = ARIMA(full_data['y'].values, order=best_order)
        final_fitted_model = final_model.fit()

        forecast_values = final_fitted_model.forecast(steps=horizon)

        # CRITICAL: Detect and handle flat forecasts
        flat_check = detect_flat_forecast(forecast_values, full_data['y'].values)
        if flat_check['is_flat']:
            logger.warning(f"🚨 ARIMA flat forecast detected with order {best_order}")
            logger.warning(f"   Reason: {flat_check['flat_reason']}")
            logger.warning(f"   Attempting fallback orders...")

            # Try fallback orders
            for fallback_order in get_fallback_arima_orders():
                try:
                    logger.info(f"   Trying fallback: ARIMA{fallback_order}")
                    fallback_model = ARIMA(full_data['y'].values, order=fallback_order)
                    fallback_fitted = fallback_model.fit()
                    fallback_forecast = fallback_fitted.forecast(steps=horizon)

                    # Check if fallback is also flat
                    fallback_check = detect_flat_forecast(fallback_forecast, full_data['y'].values)
                    if not fallback_check['is_flat']:
                        logger.info(f"   ✅ Fallback order {fallback_order} produces non-flat forecast")
                        forecast_values = fallback_forecast
                        final_fitted_model = fallback_fitted
                        best_order = fallback_order
                        break
                    else:
                        logger.warning(f"   ✗ Fallback order {fallback_order} also flat")
                except Exception as e:
                    logger.warning(f"   ✗ Fallback order {fallback_order} failed: {e}")
                    continue
            else:
                # All fallbacks failed - add noise to prevent identical values
                logger.error("   All fallback orders produced flat forecasts. Adding minimal variance.")
                forecast_std = np.std(full_data['y'].values) * 0.01
                forecast_values = forecast_values + np.random.normal(0, forecast_std, len(forecast_values))

        final_train_predictions = final_fitted_model.fittedvalues
        if len(final_train_predictions) > 0:
            forecast_lower, forecast_upper = compute_prediction_intervals(
                y_train=full_data['y'].values,
                y_pred_train=final_train_predictions,
                forecast_values=forecast_values,
                confidence_level=confidence_level
            )
        else:
            forecast_lower = forecast_values * 0.9
            forecast_upper = forecast_values * 1.1

        # Use forecast_start_date if provided (user's to_date), otherwise use end of data
        if forecast_start_date is not None:
            last_date = pd.to_datetime(forecast_start_date).normalize()
            logger.info(f"📅 Using user-specified forecast start: {last_date}")

            # FIX: Align last_date to the frequency anchor to avoid skipping periods
            # When last_date is not on the anchor day (e.g., Sunday for W-MON),
            # pd.date_range aligns to the NEXT anchor, then [1:] skips it, causing a gap.
            if frequency == 'weekly':
                anchor_day_map = {'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 'FRI': 4, 'SAT': 5, 'SUN': 6}
                anchor_day_name = pd_freq.split('-')[1] if '-' in pd_freq else 'MON'
                anchor_day = anchor_day_map.get(anchor_day_name, 0)

                days_since_anchor = (last_date.dayofweek - anchor_day) % 7
                if days_since_anchor != 0:
                    aligned_date = last_date - pd.Timedelta(days=days_since_anchor)
                    logger.info(f"📅 Aligned forecast start from {last_date.date()} ({last_date.day_name()}) to {aligned_date.date()} ({aligned_date.day_name()})")
                    last_date = aligned_date
        else:
            last_date = full_data['ds'].max()
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=pd_freq)[1:]
        logger.info(f"📅 ARIMA forecast dates: {future_dates.min()} to {future_dates.max()}")

        # Explosion guard — detect NaN/Inf/extreme values and fall back to naive
        hist_max = np.max(np.abs(full_data['y'].values)) if len(full_data) > 0 else 1
        hist_mean = np.mean(full_data['y'].values) if len(full_data) > 0 else 0
        has_explosion = (
            np.any(np.isnan(forecast_values)) or
            np.any(np.isinf(forecast_values)) or
            np.max(np.abs(forecast_values)) > 10 * hist_max
        )
        if has_explosion:
            logger.error(f"ARIMA training: explosion detected (max={np.max(np.abs(forecast_values)):.2e}, hist_max={hist_max:,.0f}). Using naive fallback.")
            naive_values = full_data['y'].values[-horizon:]
            if len(naive_values) < horizon:
                naive_values = np.full(horizon, hist_mean)
            forecast_values = naive_values.copy()
            forecast_lower = forecast_values * 0.9
            forecast_upper = forecast_values * 1.1

        # Clip negative forecasts — financial metrics cannot be negative
        forecast_values = np.maximum(forecast_values, 0.0)
        forecast_lower = np.maximum(forecast_lower, 0.0)
        forecast_upper = np.maximum(forecast_upper, 0.0)

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

            full_data_actual = pd.DataFrame({'ds': full_data['ds'], 'y': full_data['y']})
            full_data_actual.to_csv("/tmp/full_merged_data.csv", index=False)
            mlflow.log_artifact("/tmp/full_merged_data.csv", "datasets/processed")
        except Exception as e:
            logger.warning(f"Could not log ARIMA datasets: {e}")

        # Log inference input and output
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
            logger.warning(f"Could not log ARIMA inference data: {e}")
        
        mlflow.log_param("model_type", "ARIMA")
        mlflow.log_param("order", str(best_order))
        mlflow.log_param("p", best_order[0])
        mlflow.log_param("d", best_order[1])
        mlflow.log_param("q", best_order[2])
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("confidence_level", confidence_level)
        mlflow.log_metrics(best_metrics)
        
        training_code = generate_arima_training_code(
            best_order, horizon, frequency, best_metrics, len(train_df), len(test_df)
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
            model_wrapper = ARIMAModelWrapper(
                final_fitted_model, best_order, frequency, weekly_freq_code,
                historical_max=float(np.max(np.abs(full_data['y'].values))),
                historical_mean=float(np.mean(full_data['y'].values))
            )
            
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
                    "name": "arima_env"
                }
            )

            # Verify model was logged by checking artifact URI
            artifact_uri = mlflow.get_artifact_uri("model")
            logger.info(f"   ✅ ARIMA model logged successfully to: {artifact_uri}")

            # Also save a pickle backup for robustness
            model_backup_path = "/tmp/arima_model_backup.pkl"
            with open(model_backup_path, 'wb') as f:
                pickle.dump({
                    'model': final_fitted_model,
                    'wrapper': model_wrapper,
                    'order': best_order,
                    'frequency': frequency,
                    'weekly_freq_code': weekly_freq_code
                }, f)
            mlflow.log_artifact(model_backup_path, "model_backup")
            logger.info(f"   ✅ ARIMA model backup saved to model_backup/")

        except Exception as e:
            logger.error(f"   ❌ Failed to log ARIMA pyfunc model: {e}")
            try:
                model_path = "/tmp/arima_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': final_fitted_model,
                        'order': best_order,
                        'freq': pd_freq,
                        'frequency': frequency
                    }, f)
                mlflow.log_artifact(model_path, "model")
                logger.warning(f"   ⚠️ Logged ARIMA model as pickle fallback")
            except Exception as fallback_error:
                logger.error(f"   ❌ Fallback pickle also failed: {fallback_error}")
        
        best_run_id = parent_run_id
        best_artifact_uri = parent_run.info.artifact_uri
    
    return best_run_id, f"runs:/{best_run_id}/model", best_metrics, validation_data, forecast_data, best_artifact_uri, best_order

@log_io
def train_sarimax_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    frequency: str = 'monthly',
    covariates: List[str] = None,
    random_seed: int = 42,
    original_data: List[Dict[str, Any]] = None,
    country: str = 'US',
    hyperparameter_filters: Dict[str, Any] = None,
    forecast_start_date: pd.Timestamp = None  # User's specified end_date for forecast start
) -> Tuple[str, str, Dict[str, float], pd.DataFrame, pd.DataFrame, str, Dict[str, Any]]:
    """
    Train SARIMAX model with hyperparameter tuning and MLflow logging
    """
    logger.info(f"Training SARIMAX model (freq={frequency}, covariates={covariates}, country={country}, seed={random_seed})...")

    # Set random seeds for reproducibility
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)

    # For weekly, detect the actual day-of-week from training data
    pd_freq = detect_weekly_freq_code(train_df, frequency)

    # Determine seasonal period based on frequency
    seasonal_period_map = {'daily': 7, 'weekly': 52, 'monthly': 12, 'yearly': 1}
    seasonal_period = seasonal_period_map.get(frequency, 12)

    # Validate seasonal period is appropriate for data length
    # Need at least 2 complete seasonal cycles for meaningful seasonality estimation
    data_length = len(train_df)
    min_required = seasonal_period * 2
    if data_length < min_required:
        original_period = seasonal_period
        # Fall back to shorter seasonal period or disable seasonality
        if frequency == 'weekly' and data_length >= 26:
            seasonal_period = 26  # Use half-year seasonality
            logger.warning(
                f"⚠️ SARIMAX: Insufficient data for full seasonality (need {min_required} rows, have {data_length}). "
                f"Reduced seasonal period from {original_period} to {seasonal_period}."
            )
        elif frequency == 'weekly' and data_length >= 13:
            seasonal_period = 13  # Use quarterly seasonality
            logger.warning(
                f"⚠️ SARIMAX: Reduced seasonal period from {original_period} to {seasonal_period} (quarterly)."
            )
        else:
            # Disable seasonal component
            seasonal_period = 1
            logger.warning(
                f"⚠️ SARIMAX: Data too short for seasonal modeling ({data_length} rows). "
                f"Disabling seasonality (setting period=1). Need at least {min_required} rows for period={original_period}."
            )

    # Add country holidays as binary features
    train_df = train_df.copy()
    test_df = test_df.copy()
    try:
        country_holidays = holidays.country_holidays(country)
        train_df['is_holiday'] = train_df['ds'].apply(lambda x: 1 if x in country_holidays else 0)
        test_df['is_holiday'] = test_df['ds'].apply(lambda x: 1 if x in country_holidays else 0)
        holiday_count = train_df['is_holiday'].sum()
        logger.info(f"Added {country} holiday indicator: {holiday_count} holidays in training data")
    except Exception as e:
        logger.warning(f"Could not add holidays for country '{country}': {e}")
        train_df['is_holiday'] = 0
        test_df['is_holiday'] = 0

    # Prepare exogenous variables (user covariates + holiday indicator)
    covariates = covariates or []
    valid_covariates = [c for c in covariates if c in train_df.columns and c in test_df.columns]
    if 'is_holiday' not in valid_covariates:
        valid_covariates.append('is_holiday')

    if valid_covariates:
        train_exog = train_df[valid_covariates].values
        test_exog = test_df[valid_covariates].values
        covariate_means = {c: train_df[c].mean() for c in valid_covariates}
        logger.info(f"Using {len(valid_covariates)} covariates: {valid_covariates}")
    else:
        train_exog = None
        test_exog = None
        covariate_means = {}

    best_model = None
    best_metrics = {"mape": float('inf'), "rmse": float('inf')}
    best_order = None
    best_seasonal_order = None
    best_fitted_model = None
    best_run_id = None
    best_artifact_uri = None

    # Grid search space - apply hyperparameter filters from data analysis if provided
    sarimax_filters = (hyperparameter_filters or {}).get('SARIMAX', {})
    max_combinations = int(os.environ.get('SARIMAX_MAX_COMBINATIONS', '8'))

    # Extract confidence level for prediction intervals (default 0.95)
    global_filters = (hyperparameter_filters or {}).get('_global', {})
    confidence_level = global_filters.get('confidence_level', 0.95)

    # Use filtered values if provided, otherwise use defaults
    p_values = sarimax_filters.get('p_values', [0, 1, 2])
    d_values = sarimax_filters.get('d_values', [0, 1])
    q_values = sarimax_filters.get('q_values', [0, 1])
    P_values = sarimax_filters.get('P_values', [0, 1])
    D_values = sarimax_filters.get('D_values', [0, 1])
    Q_values = sarimax_filters.get('Q_values', [0, 1])

    if sarimax_filters:
        logger.info(f"📊 Using data-driven SARIMAX filters: p={p_values}, d={d_values}, q={q_values}, P={P_values}, D={D_values}, Q={Q_values}")

    all_orders = []
    for p, d, q in itertools.product(p_values, d_values, q_values):
        for P, D, Q in itertools.product(P_values, D_values, Q_values):
            all_orders.append(((p, d, q), (P, D, Q, seasonal_period)))

    # CRITICAL: Filter out degenerate SARIMAX orders that produce flat/uninformative forecasts
    # Same fix as ARIMA - (0,0,0), (0,1,0), (0,2,0) produce constant/random walk forecasts
    degenerate_orders = {(0, 0, 0), (0, 1, 0), (0, 2, 0)}
    original_count = len(all_orders)
    all_orders = [o for o in all_orders if o[0] not in degenerate_orders]

    if len(all_orders) < original_count:
        excluded_count = original_count - len(all_orders)
        logger.info(f"🚫 Excluded {excluded_count} degenerate SARIMAX orders (flat forecast prevention)")

    # Ensure we have at least some valid orders
    if len(all_orders) == 0:
        all_orders = [((1, 1, 0), (0, 0, 0, seasonal_period)),
                      ((0, 1, 1), (0, 0, 0, seasonal_period)),
                      ((1, 1, 1), (0, 0, 0, seasonal_period))]
        logger.warning("All SARIMAX orders were degenerate. Using fallback orders.")

    all_orders.sort(key=lambda x: sum(x[0]) + sum(x[1][:3]))
    orders = all_orders[:max_combinations]
    logger.info(f"Limited SARIMAX combinations to {len(orders)} (from {original_count} total, {original_count - len(all_orders)} degenerate excluded)")

    with mlflow.start_run(run_name="SARIMAX_Tuning", nested=True) as parent_run:
        parent_run_id = parent_run.info.run_id
        experiment_id = parent_run.info.experiment_id

        if original_data:
            try:
                original_df = pd.DataFrame(original_data)
                original_df.to_csv("/tmp/original_timeseries_data.csv", index=False)
                mlflow.log_artifact("/tmp/original_timeseries_data.csv", "datasets/raw")
            except Exception as e:
                logger.warning(f"Could not log original data: {e}")

        max_workers = int(os.environ.get('MLFLOW_MAX_WORKERS', '1'))
        if SKIP_CHILD_RUNS:
            logger.info(f"📊 MLFLOW_SKIP_CHILD_RUNS=true: Skipping child run logging to reduce MLflow overhead")
        logger.info(f"Running SARIMAX hyperparameter tuning with {len(orders)} combinations, {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    evaluate_sarimax_params, order, seasonal_order,
                    train_df['y'].values, test_df['y'].values,
                    train_exog, test_exog, parent_run_id, experiment_id,
                    skip_mlflow_logging=SKIP_CHILD_RUNS
                )
                for order, seasonal_order in orders
            ]

            for future in as_completed(futures):
                result = future.result()
                if result:
                    metrics = result["metrics"]
                    is_better = metrics["mape"] < best_metrics["mape"] or \
                               (abs(metrics["mape"] - best_metrics["mape"]) < 0.5 and metrics["rmse"] < best_metrics["rmse"])

                    if is_better:
                        best_metrics = metrics
                        best_order = result["order"]
                        best_seasonal_order = result["seasonal_order"]
                        best_fitted_model = result["fitted_model"]
                        logger.info(f"  ✨ New best SARIMAX{best_order}x{best_seasonal_order}: MAPE={metrics['mape']:.2f}%")

        if best_fitted_model is None:
            raise Exception("SARIMAX training failed - no successful model fits")

        # Time Series Cross-Validation
        full_y = pd.concat([train_df, test_df])['y'].values
        full_exog_cv = pd.concat([train_df, test_df])[valid_covariates].values if valid_covariates else None
        logger.info(f"Running time series cross-validation for SARIMAX...")

        def sarimax_fit_fn(y_train, exog_train=None):
            model = SARIMAX(
                y_train,
                exog=exog_train,
                order=best_order,
                seasonal_order=best_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            return model.fit(disp=False, maxiter=100)

        # Custom CV for SARIMAX with exogenous variables
        n = len(full_y)
        min_train_size = max(n // 2, 10)
        n_splits = 3
        available_for_cv = n - min_train_size
        if available_for_cv < n_splits * len(test_df):
            n_splits = max(1, available_for_cv // len(test_df))

        cv_scores = []
        fold_size = (n - min_train_size) // max(n_splits, 1)
        for i in range(n_splits):
            split_point = min_train_size + i * fold_size
            test_end = min(split_point + len(test_df), n)
            y_train_cv = full_y[:split_point]
            y_test_cv = full_y[split_point:test_end]
            exog_train_cv = full_exog_cv[:split_point] if full_exog_cv is not None else None
            exog_test_cv = full_exog_cv[split_point:test_end] if full_exog_cv is not None else None

            if len(y_test_cv) == 0:
                continue
            try:
                fitted = sarimax_fit_fn(y_train_cv, exog_train_cv)
                preds = fitted.forecast(steps=len(y_test_cv), exog=exog_test_cv)
                fold_mape = mean_absolute_percentage_error(y_test_cv, preds) * 100
                cv_scores.append(fold_mape)
            except Exception as e:
                logger.warning(f"  CV Fold {i+1} failed: {e}")

        if len(cv_scores) > 0:
            cv_mean = round(np.mean(cv_scores), 2)
            cv_std = round(np.std(cv_scores), 2)
            best_metrics["cv_mape"] = cv_mean
            best_metrics["cv_mape_std"] = cv_std

        # Validation
        test_predictions = best_fitted_model.forecast(steps=len(test_df), exog=test_exog)
        train_predictions = best_fitted_model.fittedvalues
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

        # Refit on full data and forecast
        full_data = pd.concat([train_df, test_df]).sort_values('ds')
        full_exog = full_data[valid_covariates].values if valid_covariates else None

        final_model = SARIMAX(
            full_data['y'].values,
            exog=full_exog,
            order=best_order,
            seasonal_order=best_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        final_fitted_model = final_model.fit(disp=False)

        if valid_covariates:
            future_exog = np.array([[covariate_means[c] for c in valid_covariates]] * horizon)
        else:
            future_exog = None

        forecast_values = final_fitted_model.forecast(steps=horizon, exog=future_exog)

        # CRITICAL: Detect and handle numerically explosive forecasts
        # SARIMAX can produce astronomically large values (e.g., 4.8e+31) when
        # the model is numerically unstable. Detect and replace with naive forecast.
        hist_max = np.max(np.abs(full_data['y'].values))
        hist_mean = np.mean(full_data['y'].values)
        forecast_abs_max = np.max(np.abs(forecast_values))
        has_explosion = (
            np.any(np.isnan(forecast_values)) or
            np.any(np.isinf(forecast_values)) or
            forecast_abs_max > 10 * hist_max
        )
        if has_explosion:
            logger.error(f"🚨 SARIMAX numerical explosion detected!")
            logger.error(f"   Forecast max: {forecast_abs_max:,.0f}, Historical max: {hist_max:,.0f}")
            logger.error(f"   Ratio: {forecast_abs_max / hist_max if hist_max > 0 else 'inf':.1f}x")
            logger.warning(f"   Replacing with naive seasonal forecast (last {horizon} values repeated)")
            # Use the last horizon values as a naive seasonal forecast
            naive_values = full_data['y'].values[-horizon:]
            if len(naive_values) < horizon:
                naive_values = np.full(horizon, hist_mean)
            forecast_values = naive_values.copy()
            logger.info(f"   Naive forecast range: {forecast_values.min():,.0f} - {forecast_values.max():,.0f}")

        # CRITICAL: Detect and handle flat forecasts
        flat_check = detect_flat_forecast(forecast_values, full_data['y'].values)
        if flat_check['is_flat']:
            logger.warning(f"🚨 SARIMAX flat forecast detected with order {best_order}x{best_seasonal_order}")
            logger.warning(f"   Reason: {flat_check['flat_reason']}")
            logger.warning(f"   Attempting fallback orders...")

            # Try fallback orders
            for fallback_order, fallback_seasonal in get_fallback_sarimax_orders():
                try:
                    # Adjust seasonal period to match data
                    adjusted_seasonal = (fallback_seasonal[0], fallback_seasonal[1],
                                        fallback_seasonal[2], seasonal_period)
                    logger.info(f"   Trying fallback: SARIMAX{fallback_order}x{adjusted_seasonal}")

                    fallback_model = SARIMAX(
                        full_data['y'].values,
                        exog=full_exog,
                        order=fallback_order,
                        seasonal_order=adjusted_seasonal,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fallback_fitted = fallback_model.fit(disp=False, maxiter=200)
                    fallback_forecast = fallback_fitted.forecast(steps=horizon, exog=future_exog)

                    # Check if fallback is also flat
                    fallback_check = detect_flat_forecast(fallback_forecast, full_data['y'].values)
                    if not fallback_check['is_flat']:
                        logger.info(f"   ✅ Fallback order {fallback_order}x{adjusted_seasonal} produces non-flat forecast")
                        forecast_values = fallback_forecast
                        final_fitted_model = fallback_fitted
                        best_order = fallback_order
                        best_seasonal_order = adjusted_seasonal
                        break
                    else:
                        logger.warning(f"   ✗ Fallback order {fallback_order}x{adjusted_seasonal} also flat")
                except Exception as e:
                    logger.warning(f"   ✗ Fallback order {fallback_order} failed: {e}")
                    continue
            else:
                # All fallbacks failed - add noise to prevent identical values
                logger.error("   All fallback orders produced flat forecasts. Adding minimal variance.")
                forecast_std = np.std(full_data['y'].values) * 0.01
                forecast_values = forecast_values + np.random.normal(0, forecast_std, len(forecast_values))

        final_train_predictions = final_fitted_model.fittedvalues
        if len(final_train_predictions) > 0:
            forecast_lower, forecast_upper = compute_prediction_intervals(
                y_train=full_data['y'].values,
                y_pred_train=final_train_predictions,
                forecast_values=forecast_values,
                confidence_level=confidence_level
            )
        else:
            forecast_lower = forecast_values * 0.9
            forecast_upper = forecast_values * 1.1

        # Use forecast_start_date if provided (user's to_date), otherwise use end of data
        if forecast_start_date is not None:
            last_date = pd.to_datetime(forecast_start_date).normalize()
            logger.info(f"📅 Using user-specified forecast start: {last_date}")

            # FIX: Align last_date to the frequency anchor to avoid skipping periods
            if frequency == 'weekly':
                anchor_day_map = {'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 'FRI': 4, 'SAT': 5, 'SUN': 6}
                anchor_day_name = pd_freq.split('-')[1] if '-' in pd_freq else 'MON'
                anchor_day = anchor_day_map.get(anchor_day_name, 0)

                days_since_anchor = (last_date.dayofweek - anchor_day) % 7
                if days_since_anchor != 0:
                    aligned_date = last_date - pd.Timedelta(days=days_since_anchor)
                    logger.info(f"📅 Aligned forecast start from {last_date.date()} ({last_date.day_name()}) to {aligned_date.date()} ({aligned_date.day_name()})")
                    last_date = aligned_date
        else:
            last_date = full_data['ds'].max()
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=pd_freq)[1:]
        logger.info(f"📅 SARIMAX forecast dates: {future_dates.min()} to {future_dates.max()}")

        forecast_data = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values,
            'yhat_lower': forecast_lower,
            'yhat_upper': forecast_upper
        })

        # Log datasets
        try:
            train_df.to_csv("/tmp/train.csv", index=False)
            mlflow.log_artifact("/tmp/train.csv", "datasets/training")
            test_df.to_csv("/tmp/eval.csv", index=False)
            mlflow.log_artifact("/tmp/eval.csv", "datasets/training")
            full_data.to_csv("/tmp/full_merged_data.csv", index=False)
            mlflow.log_artifact("/tmp/full_merged_data.csv", "datasets/processed")
        except Exception as e:
            logger.warning(f"Could not log datasets: {e}")

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
            logger.warning(f"Could not log inference data: {e}")

        # Log parameters and metrics
        mlflow.log_param("model_type", "SARIMAX")
        mlflow.log_param("order", str(best_order))
        mlflow.log_param("seasonal_order", str(best_seasonal_order))
        mlflow.log_param("covariates", str(valid_covariates))
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("confidence_level", confidence_level)
        mlflow.log_metrics(best_metrics)

        # Log reproducible code
        training_code = generate_sarimax_training_code(
            best_order, best_seasonal_order, horizon, frequency,
            valid_covariates, best_metrics, len(train_df), len(test_df)
        )
        mlflow.log_text(training_code, "training_code.py")

        # Log MLflow pyfunc model
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
            model_wrapper = SARIMAXModelWrapper(
                final_fitted_model, best_order, best_seasonal_order,
                frequency, valid_covariates, covariate_means, weekly_freq_code,
                historical_max=float(np.max(np.abs(full_data['y'].values))),
                historical_mean=float(np.mean(full_data['y'].values))
            )

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
                    "name": "sarimax_env"
                }
            )

            # Verify model was logged by checking artifact URI
            artifact_uri = mlflow.get_artifact_uri("model")
            logger.info(f"   ✅ SARIMAX model logged successfully to: {artifact_uri}")

            # Also save a pickle backup for robustness
            model_backup_path = "/tmp/sarimax_model_backup.pkl"
            with open(model_backup_path, 'wb') as f:
                pickle.dump({
                    'model': final_fitted_model,
                    'wrapper': model_wrapper,
                    'order': best_order,
                    'seasonal_order': best_seasonal_order,
                    'frequency': frequency,
                    'covariates': valid_covariates
                }, f)
            mlflow.log_artifact(model_backup_path, "model_backup")
            logger.info(f"   ✅ SARIMAX model backup saved to model_backup/")

        except Exception as e:
            logger.error(f"   ❌ Failed to log SARIMAX pyfunc model: {e}")
            try:
                model_path = "/tmp/sarimax_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': final_fitted_model,
                        'order': best_order,
                        'seasonal_order': best_seasonal_order,
                        'frequency': frequency
                    }, f)
                mlflow.log_artifact(model_path, "model")
                logger.warning(f"   ⚠️ Logged SARIMAX model as pickle fallback")
            except Exception as fallback_error:
                logger.error(f"   ❌ Fallback pickle also failed: {fallback_error}")

        best_run_id = parent_run_id
        best_artifact_uri = parent_run.info.artifact_uri

    params = {
        "order": best_order,
        "seasonal_order": best_seasonal_order,
        "covariates": valid_covariates
    }

    return best_run_id, f"runs:/{best_run_id}/model", best_metrics, validation_data, forecast_data, best_artifact_uri, params
