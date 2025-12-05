"""
Multiple model training implementations with hyperparameter tuning (Prophet, ARIMA, Exponential Smoothing)
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import pickle
import os
# Prophet import moved to lazy loading to avoid dependency issues when loading ARIMA/ETS models
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import logging
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


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

    def __init__(self, fitted_model, order, frequency):
        self.fitted_model = fitted_model
        self.order = order
        # Store frequency in human-readable format for consistency
        # Map pandas freq codes to human-readable if needed
        freq_to_human = {'MS': 'monthly', 'W': 'weekly', 'D': 'daily', 'YS': 'yearly'}
        self.frequency = freq_to_human.get(frequency, frequency)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        # Extract parameters from input
        periods = int(model_input['periods'].iloc[0])
        start_date = pd.to_datetime(model_input['start_date'].iloc[0])

        # Map human-readable frequency to pandas freq code
        freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS', 'yearly': 'YS'}

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

    def __init__(self, fitted_model, params, frequency, seasonal_periods):
        self.fitted_model = fitted_model
        self.params = params
        # Store frequency in human-readable format for consistency
        # Map pandas freq codes to human-readable if needed
        freq_to_human = {'MS': 'monthly', 'W': 'weekly', 'D': 'daily', 'YS': 'yearly'}
        self.frequency = freq_to_human.get(frequency, frequency)
        self.seasonal_periods = seasonal_periods

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        # Extract parameters from input
        periods = int(model_input['periods'].iloc[0])
        start_date = pd.to_datetime(model_input['start_date'].iloc[0])

        # Map human-readable frequency to pandas freq code
        freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS', 'yearly': 'YS'}

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
ts_data = df['y'].values  # Convert to numpy array

# Split into train/test (train_size={train_size}, test_size={test_size})
test_size = {test_size}
train_data = ts_data[:-test_size]
test_data = ts_data[-test_size:]

# ============================================================================
# MODEL INITIALIZATION & TRAINING FLOW
# ============================================================================
# ARIMA order: ({order[0]}, {order[1]}, {order[2]})
# p={order[0]} (AR order), d={order[1]} (differencing), q={order[2]} (MA order)
model = ARIMA(train_data, order=({order[0]}, {order[1]}, {order[2]}))
fitted_model = model.fit()

print("ARIMA Model Summary:")
print(fitted_model.summary())

# ============================================================================
# VALIDATION (on test set)
# ============================================================================
test_predictions = fitted_model.forecast(steps=len(test_data))

# Calculate metrics
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
mape = mean_absolute_percentage_error(test_data, test_predictions) * 100
r2 = r2_score(test_data, test_predictions)

print(f"\\nValidation Metrics:")
print(f"  RMSE: {{rmse:.2f}}")
print(f"  MAPE: {{mape:.2f}}%")
print(f"  R²: {{r2:.4f}}")

# ============================================================================
# FORECASTING (future periods)
# ============================================================================
# Refit on full dataset for final forecast
full_data = np.concatenate([train_data, test_data])
final_model = ARIMA(full_data, order=({order[0]}, {order[1]}, {order[2]}))
final_fitted_model = final_model.fit()

# Generate forecast for {horizon} periods
forecast_values = final_fitted_model.forecast(steps={horizon})

# Create forecast dataframe with dates
last_date = pd.to_datetime(df['ds'].max())
future_dates = pd.date_range(start=last_date, periods={horizon} + 1, freq='{pd_freq}')[1:]

forecast_df = pd.DataFrame({{
    'ds': future_dates,
    'yhat': forecast_values,
    'yhat_lower': forecast_values * 0.9,
    'yhat_upper': forecast_values * 1.1
}})

print(f"\\nForecast for {{len(forecast_df)}} periods:")
print(forecast_df.head())

# ============================================================================
# MLFLOW MODEL REGISTRATION
# ============================================================================
import mlflow
import mlflow.pyfunc
import sys
from mlflow.models.signature import infer_signature

# Set MLflow tracking URI (adjust as needed)
mlflow.set_tracking_uri("databricks")  # or "http://localhost:5000" for local
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/finance-forecasting"))

# Start MLflow run
with mlflow.start_run(run_name="ARIMA_Model") as run:
    # Log parameters
    mlflow.log_param("model_type", "ARIMA")
    mlflow.log_param("p", {order[0]})
    mlflow.log_param("d", {order[1]})
    mlflow.log_param("q", {order[2]})
    mlflow.log_param("horizon", {horizon})
    mlflow.log_param("frequency", "{frequency}")
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("r2", r2)
    
    # Create model wrapper class
    class ARIMAModelWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, fitted_model, order, freq):
            self.fitted_model = fitted_model
            self.order = order
            self.freq = freq

        def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
            import pandas as pd
            periods = model_input['periods'].iloc[0] if 'periods' in model_input.columns else 1
            forecast_values = self.fitted_model.forecast(steps=periods)
            
            if 'ds' in model_input.columns:
                future_dates = pd.to_datetime(model_input['ds'])
            else:
                last_date = pd.to_datetime(model_input['ds'].iloc[-1]) if 'ds' in model_input.columns else pd.Timestamp.now()
                freq_offset = pd.DateOffset(months=1) if self.freq == 'MS' else pd.DateOffset(weeks=1) if self.freq == 'W' else pd.DateOffset(days=1)
                future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq_offset)[1:]
            
            return pd.DataFrame({{
                'ds': future_dates,
                'yhat': forecast_values,
                'yhat_lower': forecast_values * 0.9,
                'yhat_upper': forecast_values * 1.1
            }})
    
    # Create input example
    input_example = pd.DataFrame({{'ds': [train_df['ds'].iloc[-1]], 'periods': [{horizon}]}})
    
    # Create model wrapper
    model_wrapper = ARIMAModelWrapper(final_fitted_model, ({order[0]}, {order[1]}, {order[2]}), '{pd_freq}')
    
    # Infer signature
    sample_output = model_wrapper.predict(None, input_example)
    signature = infer_signature(input_example, sample_output)
    
    # Log model to MLflow
    mlflow.pyfunc.log_model(
        name="model",
        python_model=model_wrapper,
        signature=signature,
        input_example=input_example,
        code_paths=["backend"] if os.path.exists("backend") else [],
        conda_env={{
            "channels": ["defaults", "conda-forge"],
            "dependencies": [
                f"python={{sys.version_info.major}}.{{sys.version_info.minor}}.{{sys.version_info.micro}}",
                "pip",
                {{"pip": ["mlflow", "pandas", "numpy", "statsmodels", "scikit-learn"]}}
            ],
            "name": "arima_env"
        }}
    )
    
    print("\\nModel logged to MLflow successfully!")
    print(f"   Run ID: {{run.info.run_id}}")
    print(f"   Model URI: runs:/{{run.info.run_id}}/model")

# ============================================================================
# NOTES
# ============================================================================
# - Training data: {train_size} rows
# - Test data: {test_size} rows
# - Forecast horizon: {horizon} periods
# - Frequency: {frequency} ({pd_freq})
# - ARIMA order: ({order[0]}, {order[1]}, {order[2]})
# - Best validation metrics:
#   * RMSE: {metrics.get('rmse', 'N/A')}
#   * MAPE: {metrics.get('mape', 'N/A')}%
#   * R²: {metrics.get('r2', 'N/A')}
'''
    return code


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
ts_data = df['y'].values  # Convert to numpy array

# Split into train/test (train_size={train_size}, test_size={test_size})
test_size = {test_size}
train_data = ts_data[:-test_size]
test_data = ts_data[-test_size:]

# ============================================================================
# MODEL INITIALIZATION & TRAINING FLOW
# ============================================================================
# ETS parameters:
# - trend: '{trend}'
# - seasonal: '{seasonal}'
# - seasonal_periods: {seasonal_periods}
model = ExponentialSmoothing(
    train_data,
    seasonal_periods={seasonal_periods},
    trend={f"'{trend}'" if trend else "None"},
    seasonal={f"'{seasonal}'" if seasonal else "None"},
    initialization_method='estimated'
)
fitted_model = model.fit(optimized=True)

print("ETS Model Summary:")
print(f"  Trend: {trend}")
print(f"  Seasonal: {seasonal}")
print(f"  Seasonal Periods: {seasonal_periods}")

# ============================================================================
# VALIDATION (on test set)
# ============================================================================
test_predictions = fitted_model.forecast(steps=len(test_data))

# Calculate metrics
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
mape = mean_absolute_percentage_error(test_data, test_predictions) * 100
r2 = r2_score(test_data, test_predictions)

print(f"\\nValidation Metrics:")
print(f"  RMSE: {{rmse:.2f}}")
print(f"  MAPE: {{mape:.2f}}%")
print(f"  R²: {{r2:.4f}}")

# ============================================================================
# FORECASTING (future periods)
# ============================================================================
# Refit on full dataset for final forecast
full_data = np.concatenate([train_data, test_data])
final_model = ExponentialSmoothing(
    full_data,
    seasonal_periods={seasonal_periods},
    trend={f"'{trend}'" if trend else "None"},
    seasonal={f"'{seasonal}'" if seasonal else "None"},
    initialization_method='estimated'
)
final_fitted_model = final_model.fit(optimized=True)

# Generate forecast for {horizon} periods
forecast_values = final_fitted_model.forecast(steps={horizon})

# Create forecast dataframe with dates
last_date = pd.to_datetime(df['ds'].max())
future_dates = pd.date_range(start=last_date, periods={horizon} + 1, freq='{pd_freq}')[1:]

forecast_df = pd.DataFrame({{
    'ds': future_dates,
    'yhat': forecast_values,
    'yhat_lower': forecast_values * 0.9,
    'yhat_upper': forecast_values * 1.1
}})

print(f"\\nForecast for {{len(forecast_df)}} periods:")
print(forecast_df.head())

# ============================================================================
# MLFLOW MODEL REGISTRATION
# ============================================================================
import mlflow
import mlflow.pyfunc
import sys
from mlflow.models.signature import infer_signature

# Set MLflow tracking URI (adjust as needed)
mlflow.set_tracking_uri("databricks")  # or "http://localhost:5000" for local
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/finance-forecasting"))

# Start MLflow run
with mlflow.start_run(run_name="ETS_Model") as run:
    # Log parameters
    mlflow.log_param("model_type", "ExponentialSmoothing")
    mlflow.log_param("trend", "{trend}")
    mlflow.log_param("seasonal", "{seasonal}")
    mlflow.log_param("seasonal_periods", {seasonal_periods})
    mlflow.log_param("horizon", {horizon})
    mlflow.log_param("frequency", "{frequency}")
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("r2", r2)
    
    # Create model wrapper class
    class ExponentialSmoothingModelWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, fitted_model, params, freq, seasonal_periods):
            self.fitted_model = fitted_model
            self.params = params
            self.freq = freq
            self.seasonal_periods = seasonal_periods

        def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
            import pandas as pd
            periods = model_input['periods'].iloc[0] if 'periods' in model_input.columns else 1
            forecast_values = self.fitted_model.forecast(steps=periods)
            
            if 'ds' in model_input.columns:
                future_dates = pd.to_datetime(model_input['ds'])
            else:
                last_date = pd.to_datetime(model_input['ds'].iloc[-1]) if 'ds' in model_input.columns else pd.Timestamp.now()
                freq_offset = pd.DateOffset(months=1) if self.freq == 'MS' else pd.DateOffset(weeks=1) if self.freq == 'W' else pd.DateOffset(days=1)
                future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq_offset)[1:]
            
            return pd.DataFrame({{
                'ds': future_dates,
                'yhat': forecast_values,
                'yhat_lower': forecast_values * 0.9,
                'yhat_upper': forecast_values * 1.1
            }})
    
    # Create input example
    input_example = pd.DataFrame({{'ds': [train_df['ds'].iloc[-1]], 'periods': [{horizon}]}})
    
    # Create model wrapper
    model_wrapper = ExponentialSmoothingModelWrapper(
        final_fitted_model,
        {{'trend': '{trend}', 'seasonal': '{seasonal}'}},
        '{pd_freq}',
        {seasonal_periods}
    )
    
    # Infer signature
    sample_output = model_wrapper.predict(None, input_example)
    signature = infer_signature(input_example, sample_output)
    
    # Log model to MLflow
    mlflow.pyfunc.log_model(
        name="model",
        python_model=model_wrapper,
        signature=signature,
        input_example=input_example,
        code_paths=["backend"] if os.path.exists("backend") else [],
        conda_env={{
            "channels": ["defaults", "conda-forge"],
            "dependencies": [
                f"python={{sys.version_info.major}}.{{sys.version_info.minor}}.{{sys.version_info.micro}}",
                "pip",
                {{"pip": ["mlflow", "pandas", "numpy", "statsmodels", "scikit-learn"]}}
            ],
            "name": "ets_env"
        }}
    )
    
    print("\\nModel logged to MLflow successfully!")
    print(f"   Run ID: {{run.info.run_id}}")
    print(f"   Model URI: runs:/{{run.info.run_id}}/model")

# ============================================================================
# NOTES
# ============================================================================
# - Training data: {train_size} rows
# - Test data: {test_size} rows
# - Forecast horizon: {horizon} periods
# - Frequency: {frequency} ({pd_freq})
# - Seasonal periods: {seasonal_periods}
# - Best parameters:
#   * Trend: {trend}
#   * Seasonal: {seasonal}
# - Best validation metrics:
#   * RMSE: {metrics.get('rmse', 'N/A')}
#   * MAPE: {metrics.get('mape', 'N/A')}%
#   * R²: {metrics.get('r2', 'N/A')}
'''
    return code


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute forecast accuracy metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = 0.0
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 4)
    }


def time_series_cross_validate(
    y: np.ndarray,
    model_fit_fn,
    model_predict_fn,
    n_splits: int = 3,
    min_train_size: int = None,
    horizon: int = 1
) -> Dict[str, Any]:
    """
    Perform time series cross-validation with expanding window.

    This is the gold standard for time series model evaluation:
    - Uses expanding window (not random splits)
    - Each fold uses all previous data for training
    - Tests on the next 'horizon' points
    - Returns average metrics across all folds

    Args:
        y: Target time series values
        model_fit_fn: Function that takes y_train and returns fitted model
        model_predict_fn: Function that takes (fitted_model, steps) and returns predictions
        n_splits: Number of cross-validation folds
        min_train_size: Minimum training size (defaults to 50% of data)
        horizon: Forecast horizon for each fold

    Returns:
        Dictionary with cv_scores (list of MAPE per fold), mean_mape, std_mape
    """
    n = len(y)
    if min_train_size is None:
        min_train_size = max(n // 2, 10)  # At least 50% or 10 points

    # Calculate fold boundaries
    # We need: min_train_size + n_splits * horizon <= n
    available_for_cv = n - min_train_size
    if available_for_cv < n_splits * horizon:
        # Not enough data for requested splits, reduce splits
        n_splits = max(1, available_for_cv // horizon)
        logger.warning(f"Reduced CV splits to {n_splits} due to limited data")

    if n_splits < 1:
        logger.warning("Not enough data for cross-validation, using simple holdout")
        return {"cv_scores": [], "mean_mape": None, "std_mape": None, "n_splits": 0}

    fold_size = (n - min_train_size) // n_splits
    cv_scores = []

    for i in range(n_splits):
        # Expanding window: train on all data up to split point
        split_point = min_train_size + i * fold_size
        test_end = min(split_point + horizon, n)

        y_train = y[:split_point]
        y_test = y[split_point:test_end]

        if len(y_test) == 0:
            continue

        try:
            fitted_model = model_fit_fn(y_train)
            predictions = model_predict_fn(fitted_model, len(y_test))

            # Calculate MAPE for this fold
            fold_mape = mean_absolute_percentage_error(y_test, predictions) * 100
            cv_scores.append(fold_mape)
            logger.info(f"  CV Fold {i+1}/{n_splits}: train={len(y_train)}, test={len(y_test)}, MAPE={fold_mape:.2f}%")
        except Exception as e:
            logger.warning(f"  CV Fold {i+1} failed: {e}")
            continue

    if len(cv_scores) == 0:
        return {"cv_scores": [], "mean_mape": None, "std_mape": None, "n_splits": 0}

    return {
        "cv_scores": cv_scores,
        "mean_mape": round(np.mean(cv_scores), 2),
        "std_mape": round(np.std(cv_scores), 2),
        "n_splits": len(cv_scores)
    }


def compute_prediction_intervals(
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    forecast_values: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute statistically valid prediction intervals based on residual distribution.

    Instead of using arbitrary ±10%, this calculates intervals based on
    the actual prediction error distribution from training data.

    Args:
        y_train: Actual training values
        y_pred_train: Predicted values for training period (in-sample)
        forecast_values: Future forecast values
        confidence_level: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bounds, upper_bounds) arrays
    """
    from scipy import stats

    # Calculate residuals from training period
    residuals = y_train - y_pred_train

    # Use residual standard deviation for interval width
    residual_std = np.std(residuals)

    # For confidence intervals, use t-distribution if small sample, normal otherwise
    n = len(residuals)
    if n < 30:
        # Use t-distribution for small samples
        alpha = 1 - confidence_level
        t_value = stats.t.ppf(1 - alpha/2, df=n-1)
        margin = t_value * residual_std
    else:
        # Use normal distribution for larger samples
        z_value = stats.norm.ppf(1 - (1 - confidence_level)/2)
        margin = z_value * residual_std

    # Apply margin to forecast (prediction intervals widen with horizon)
    # Simple approach: constant width. More sophisticated: widen with sqrt(horizon)
    lower_bounds = forecast_values - margin
    upper_bounds = forecast_values + margin

    # Ensure lower bounds are non-negative for positive-valued forecasts
    if np.all(forecast_values > 0):
        lower_bounds = np.maximum(lower_bounds, forecast_values * 0.1)  # At least 10% of forecast

    return lower_bounds, upper_bounds


def train_arima_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    frequency: str = 'monthly',
    order: Tuple[int, int, int] = None,
    random_seed: int = 42,
    original_data: List[Dict[str, Any]] = None,
    covariates: List[str] = None  # Kept for API compatibility but NOT used - ARIMA is univariate
) -> Tuple[str, str, Dict[str, float], pd.DataFrame, pd.DataFrame, str, Tuple[int, int, int]]:
    """
    Train ARIMA model with hyperparameter tuning and MLflow logging

    Note: ARIMA is a UNIVARIATE model - it only uses the target time series.
    Covariates are NOT used by ARIMA. The covariates parameter is kept for API
    compatibility but is ignored.

    Args:
        train_df: Training dataframe with 'ds' and 'y' columns
        test_df: Test dataframe with 'ds' and 'y' columns
        horizon: Number of periods to forecast
        frequency: Data frequency ('daily', 'weekly', 'monthly', 'yearly')
        order: Optional fixed ARIMA order (p, d, q). If None, grid search is performed.
        random_seed: Random seed for reproducibility tracking
        original_data: Original uploaded data (for logging to datasets/raw/)
        covariates: NOT USED - ARIMA is univariate. Kept for API compatibility.
    """
    logger.info(f"Training ARIMA model (freq={frequency}, seed={random_seed})...")

    # Set random seeds for reproducibility
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    logger.info(f"ARIMA: Set random seed to {random_seed} for reproducibility")

    # Map frequency to pandas alias
    freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS', 'yearly': 'YS'}
    pd_freq = freq_map.get(frequency, 'MS')

    best_model = None
    best_metrics = {"mape": float('inf'), "rmse": float('inf')}
    best_order = order
    best_fitted_model = None
    best_run_id = None
    best_artifact_uri = None
    
    # Define grid search space if order is not provided
    # Reduce combinations for Databricks Apps to avoid timeouts
    if order is None:
        # Limit ARIMA grid search to most common/useful combinations
        max_arima_combinations = int(os.environ.get('ARIMA_MAX_COMBINATIONS', '6'))  # Default to 6 for Databricks Apps
        p_values = [0, 1, 2]
        d_values = [0, 1]
        q_values = [0, 1]
        all_orders = list(set(itertools.product(p_values, d_values, q_values)))
        if len(all_orders) > max_arima_combinations:
            # Prioritize simpler models first (lower total p+d+q)
            all_orders.sort(key=lambda x: sum(x))
            orders = all_orders[:max_arima_combinations]
            logger.info(f"Limited ARIMA combinations to {max_arima_combinations} (from {len(all_orders)} total)")
        else:
            orders = all_orders
    else:
        orders = [order]
    
    with mlflow.start_run(run_name="ARIMA_Tuning", nested=True) as parent_run:
        parent_run_id = parent_run.info.run_id
        experiment_id = parent_run.info.experiment_id

        # Log original data to datasets/raw/ for reproducibility (matching Prophet structure)
        if original_data:
            try:
                original_df = pd.DataFrame(original_data)
                original_df.to_csv("/tmp/original_timeseries_data.csv", index=False)
                mlflow.log_artifact("/tmp/original_timeseries_data.csv", "datasets/raw")
                logger.info(f"Logged original time series data to datasets/raw/: {len(original_df)} rows with columns: {list(original_df.columns)}")
            except Exception as e:
                logger.warning(f"Could not log original data for ARIMA: {e}")
        
        # Reduce parallelism for Databricks Apps to avoid timeouts
        max_workers = int(os.environ.get('MLFLOW_MAX_WORKERS', '1'))  # Default to 1 for Databricks Apps
        logger.info(f"Running ARIMA hyperparameter tuning with {len(orders)} combinations, {max_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(evaluate_arima_params, order, train_df['y'].values, test_df['y'].values, parent_run_id, experiment_id)
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

        # Time Series Cross-Validation for more robust MAPE estimate
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

        # Get in-sample predictions for confidence interval calculation
        train_predictions = best_fitted_model.fittedvalues
        if len(train_predictions) > 0:
            yhat_lower, yhat_upper = compute_prediction_intervals(
                y_train=train_df['y'].values,
                y_pred_train=train_predictions[-len(train_df):] if len(train_predictions) >= len(train_df) else train_predictions,
                forecast_values=test_predictions,
                confidence_level=0.95
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

        # Compute proper prediction intervals for forecast
        final_train_predictions = final_fitted_model.fittedvalues
        if len(final_train_predictions) > 0:
            forecast_lower, forecast_upper = compute_prediction_intervals(
                y_train=full_data['y'].values,
                y_pred_train=final_train_predictions,
                forecast_values=forecast_values,
                confidence_level=0.95
            )
        else:
            forecast_lower = forecast_values * 0.9
            forecast_upper = forecast_values * 1.1

        last_date = full_data['ds'].max()
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=pd_freq)[1:]

        forecast_data = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values,
            'yhat_lower': forecast_lower,
            'yhat_upper': forecast_upper
        })
        
        # Log datasets to structured folders
        # ARIMA is univariate - log the actual data used for training/evaluation
        try:
            # Log the actual training data (y values as time series)
            train_data_actual = pd.DataFrame({
                'ds': train_df['ds'],
                'y': train_df['y']
            })
            train_data_actual.to_csv("/tmp/train.csv", index=False)
            mlflow.log_artifact("/tmp/train.csv", "datasets/training")
            logger.info(f"Logged ARIMA training data: {len(train_data_actual)} rows")

            # Log the actual evaluation data
            eval_data_actual = pd.DataFrame({
                'ds': test_df['ds'],
                'y': test_df['y']
            })
            eval_data_actual.to_csv("/tmp/eval.csv", index=False)
            mlflow.log_artifact("/tmp/eval.csv", "datasets/training")
            logger.info(f"Logged ARIMA evaluation data: {len(eval_data_actual)} rows")

            # Log full data used for final model refit
            full_data_actual = pd.DataFrame({
                'ds': full_data['ds'],
                'y': full_data['y']
            })
            full_data_actual.to_csv("/tmp/full_merged_data.csv", index=False)
            mlflow.log_artifact("/tmp/full_merged_data.csv", "datasets/processed")
            logger.info(f"Logged full data for final model: {len(full_data_actual)} rows")
        except Exception as e:
            logger.warning(f"Could not log ARIMA datasets: {e}")

        # Log inference input and output
        try:
            # For ARIMA, inference input matches what's sent to the serving endpoint
            # Only periods and start_date required - frequency is stored internally
            inference_input = pd.DataFrame({
                'periods': [horizon],
                'start_date': [str(last_date.date()) if hasattr(last_date, 'date') else str(last_date)[:10]]
            })
            inference_input.to_csv("/tmp/input.csv", index=False)
            mlflow.log_artifact("/tmp/input.csv", "datasets/inference")
            logger.info(f"Logged ARIMA inference input: periods={horizon}, start_date={last_date}")

            # Log forecast output
            forecast_data.to_csv("/tmp/output.csv", index=False)
            mlflow.log_artifact("/tmp/output.csv", "datasets/inference")
            logger.info(f"Logged ARIMA forecast output: {len(forecast_data)} rows")
        except Exception as e:
            logger.warning(f"Could not log ARIMA inference data: {e}")
        
        
        # Logging
        mlflow.log_param("model_type", "ARIMA")
        mlflow.log_param("order", str(best_order))
        mlflow.log_param("p", best_order[0])
        mlflow.log_param("d", best_order[1])
        mlflow.log_param("q", best_order[2])
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_metrics(best_metrics)
        
        # Log reproducible training code
        training_code = generate_arima_training_code(
            best_order, horizon, frequency, best_metrics, len(train_df), len(test_df)
        )
        mlflow.log_text(training_code, "training_code.py")
        logger.info("Logged reproducible ARIMA training code")
        
        # Log as MLflow pyfunc model
        try:
            import sys
            from mlflow.models.signature import infer_signature

            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            # Input example: periods and start_date required, frequency is optional (stored internally)
            input_example = pd.DataFrame({
                'periods': [horizon],
                'start_date': [str(last_date.date()) if hasattr(last_date, 'date') else str(last_date)[:10]]
            })
            # Output example matches actual forecast output
            sample_output = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(1).copy()
            signature = infer_signature(input_example, sample_output)
            # Pass human-readable frequency ('daily', 'weekly', 'monthly') for consistency
            model_wrapper = ARIMAModelWrapper(final_fitted_model, best_order, frequency)
            
            mlflow.pyfunc.log_model(
                name="model",
                python_model=model_wrapper,
                signature=signature,
                input_example=input_example,
                code_paths=["backend"],
                conda_env={
                    "channels": ["defaults", "conda-forge"],
                    "dependencies": [
                        f"python={python_version}",
                        "pip",
                        {"pip": ["mlflow", "pandas", "numpy", "statsmodels", "scikit-learn"]}
                    ],
                    "name": "arima_env"
                }
            )
            logger.info("Logged ARIMA pyfunc model")
        except Exception as e:
            logger.warning(f"Failed to log ARIMA pyfunc model: {e}")
            # Fallback: Save as pickle
            try:
                model_path = "/tmp/arima_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump({'model': final_fitted_model, 'order': best_order, 'freq': pd_freq}, f)
                mlflow.log_artifact(model_path, "model")
                logger.info("Logged ARIMA model as pickle artifact (fallback)")
            except Exception: pass
        
        best_run_id = parent_run_id
        best_artifact_uri = parent_run.info.artifact_uri
    
    return best_run_id, f"runs:/{best_run_id}/model", best_metrics, validation_data, forecast_data, best_artifact_uri, best_order


def train_exponential_smoothing_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    frequency: str = 'monthly',
    seasonal_periods: int = 12,
    random_seed: int = 42,
    original_data: List[Dict[str, Any]] = None,
    covariates: List[str] = None  # Kept for API compatibility but NOT used - ETS is univariate
) -> Tuple[str, str, Dict[str, float], pd.DataFrame, pd.DataFrame, str, Dict[str, Any]]:
    """
    Train Exponential Smoothing model with hyperparameter tuning and MLflow logging

    Note: ETS is a UNIVARIATE model - it only uses the target time series.
    Covariates are NOT used by ETS. The covariates parameter is kept for API
    compatibility but is ignored.

    Args:
        train_df: Training dataframe with 'ds' and 'y' columns
        test_df: Test dataframe with 'ds' and 'y' columns
        horizon: Number of periods to forecast
        frequency: Data frequency ('daily', 'weekly', 'monthly', 'yearly')
        seasonal_periods: Number of periods in a season (e.g., 12 for monthly)
        random_seed: Random seed for reproducibility tracking
        original_data: Original uploaded data (for logging to datasets/raw/)
        covariates: NOT USED - ETS is univariate. Kept for API compatibility.
    """
    logger.info(f"Training ETS model (freq={frequency}, seasonal={seasonal_periods}, seed={random_seed})...")

    # Set random seeds for reproducibility
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    logger.info(f"ETS: Set random seed to {random_seed} for reproducibility")

    # Map frequency to pandas alias
    freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS', 'yearly': 'YS'}
    pd_freq = freq_map.get(frequency, 'MS')

    best_model = None
    best_metrics = {"mape": float('inf'), "rmse": float('inf')}
    best_params = {}
    best_run_id = None
    best_artifact_uri = None

    trend_options = ['add', None]
    seasonal_options = ['add', None]
    
    # Limit ETS combinations for Databricks Apps
    param_combinations = list(set([(trend, seasonal) for trend in trend_options for seasonal in seasonal_options]))
    max_ets_combinations = int(os.environ.get('ETS_MAX_COMBINATIONS', '4'))  # Default to 4 for Databricks Apps
    if len(param_combinations) > max_ets_combinations:
        # Prioritize simpler models (None/None first, then add/None, etc.)
        param_combinations.sort(key=lambda x: (x[0] is not None, x[1] is not None))
        param_combinations = param_combinations[:max_ets_combinations]
        logger.info(f"Limited ETS combinations to {max_ets_combinations}")

    with mlflow.start_run(run_name="ETS_Tuning", nested=True) as parent_run:
        parent_run_id = parent_run.info.run_id
        experiment_id = parent_run.info.experiment_id

        # Log original data to datasets/raw/ for reproducibility (matching Prophet structure)
        if original_data:
            try:
                original_df = pd.DataFrame(original_data)
                original_df.to_csv("/tmp/original_timeseries_data.csv", index=False)
                mlflow.log_artifact("/tmp/original_timeseries_data.csv", "datasets/raw")
                logger.info(f"Logged original time series data to datasets/raw/: {len(original_df)} rows with columns: {list(original_df.columns)}")
            except Exception as e:
                logger.warning(f"Could not log original data for ETS: {e}")
        
        # Reduce parallelism for Databricks Apps to avoid timeouts
        max_workers = int(os.environ.get('MLFLOW_MAX_WORKERS', '2'))  # Default to 2 instead of 4
        logger.info(f"Running ETS hyperparameter tuning with {len(param_combinations)} combinations, {max_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(evaluate_ets_params, trend, seasonal, seasonal_periods, train_df['y'].values, test_df['y'].values, parent_run_id, experiment_id)
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

        # Time Series Cross-Validation for more robust MAPE estimate
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

        # Validation with proper confidence intervals
        test_predictions = best_model.forecast(steps=len(test_df))

        # Get in-sample predictions for confidence interval calculation
        train_predictions = best_model.fittedvalues
        if len(train_predictions) > 0:
            yhat_lower, yhat_upper = compute_prediction_intervals(
                y_train=train_df['y'].values,
                y_pred_train=train_predictions[-len(train_df):] if len(train_predictions) >= len(train_df) else train_predictions,
                forecast_values=test_predictions,
                confidence_level=0.95
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
        final_model = ExponentialSmoothing(
            full_data['y'].values,
            seasonal_periods=seasonal_periods,
            trend=best_params['trend'],
            seasonal=best_params['seasonal'],
            initialization_method='estimated'
        )
        final_fitted_model = final_model.fit(optimized=True)

        forecast_values = final_fitted_model.forecast(steps=horizon)

        # Compute proper prediction intervals for forecast
        final_train_predictions = final_fitted_model.fittedvalues
        if len(final_train_predictions) > 0:
            forecast_lower, forecast_upper = compute_prediction_intervals(
                y_train=full_data['y'].values,
                y_pred_train=final_train_predictions,
                forecast_values=forecast_values,
                confidence_level=0.95
            )
        else:
            forecast_lower = forecast_values * 0.9
            forecast_upper = forecast_values * 1.1

        last_date = full_data['ds'].max()
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=pd_freq)[1:]

        forecast_data = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values,
            'yhat_lower': forecast_lower,
            'yhat_upper': forecast_upper
        })
        
        # Log datasets to structured folders
        # ETS is univariate - log the actual data used for training/evaluation
        try:
            # Log the actual training data (y values as time series)
            train_data_actual = pd.DataFrame({
                'ds': train_df['ds'],
                'y': train_df['y']
            })
            train_data_actual.to_csv("/tmp/train.csv", index=False)
            mlflow.log_artifact("/tmp/train.csv", "datasets/training")
            logger.info(f"Logged ETS training data: {len(train_data_actual)} rows")

            # Log the actual evaluation data
            eval_data_actual = pd.DataFrame({
                'ds': test_df['ds'],
                'y': test_df['y']
            })
            eval_data_actual.to_csv("/tmp/eval.csv", index=False)
            mlflow.log_artifact("/tmp/eval.csv", "datasets/training")
            logger.info(f"Logged ETS evaluation data: {len(eval_data_actual)} rows")

            # Log full data used for final model refit
            full_data_actual = pd.DataFrame({
                'ds': full_data['ds'],
                'y': full_data['y']
            })
            full_data_actual.to_csv("/tmp/full_merged_data.csv", index=False)
            mlflow.log_artifact("/tmp/full_merged_data.csv", "datasets/processed")
            logger.info(f"Logged full data for final model: {len(full_data_actual)} rows")
        except Exception as e:
            logger.warning(f"Could not log ETS datasets: {e}")

        # Log inference input and output
        try:
            # For ETS, inference input matches what's sent to the serving endpoint
            # Only periods and start_date required - frequency is stored internally
            inference_input = pd.DataFrame({
                'periods': [horizon],
                'start_date': [str(last_date.date()) if hasattr(last_date, 'date') else str(last_date)[:10]]
            })
            inference_input.to_csv("/tmp/input.csv", index=False)
            mlflow.log_artifact("/tmp/input.csv", "datasets/inference")
            logger.info(f"Logged ETS inference input: periods={horizon}, start_date={last_date}")

            # Log forecast output
            forecast_data.to_csv("/tmp/output.csv", index=False)
            mlflow.log_artifact("/tmp/output.csv", "datasets/inference")
            logger.info(f"Logged ETS forecast output: {len(forecast_data)} rows")
        except Exception as e:
            logger.warning(f"Could not log ETS inference data: {e}")

        # Logging
        mlflow.log_param("model_type", "ExponentialSmoothing")
        mlflow.log_param("trend", str(best_params.get('trend', 'None')))
        mlflow.log_param("seasonal", str(best_params.get('seasonal', 'None')))
        mlflow.log_param("seasonal_periods", seasonal_periods)
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_metrics(best_metrics)
        
        # Log reproducible training code
        training_code = generate_ets_training_code(
            best_params, seasonal_periods, horizon, frequency, best_metrics, len(train_df), len(test_df)
        )
        mlflow.log_text(training_code, "training_code.py")
        logger.info("Logged reproducible ETS training code")
        
        # Log as MLflow pyfunc model
        try:
            import sys
            from mlflow.models.signature import infer_signature

            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            # Input example: periods and start_date required, frequency is optional (stored internally)
            input_example = pd.DataFrame({
                'periods': [horizon],
                'start_date': [str(last_date.date()) if hasattr(last_date, 'date') else str(last_date)[:10]]
            })
            # Output example matches actual forecast output
            sample_output = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(1).copy()
            signature = infer_signature(input_example, sample_output)
            # Pass human-readable frequency ('daily', 'weekly', 'monthly') for consistency
            model_wrapper = ExponentialSmoothingModelWrapper(final_fitted_model, best_params, frequency, seasonal_periods)
            
            mlflow.pyfunc.log_model(
                name="model",
                python_model=model_wrapper,
                signature=signature,
                input_example=input_example,
                code_paths=["backend"],
                conda_env={
                    "channels": ["defaults", "conda-forge"],
                    "dependencies": [
                        f"python={python_version}",
                        "pip",
                        {"pip": ["mlflow", "pandas", "numpy", "statsmodels", "scikit-learn"]}
                    ],
                    "name": "ets_env"
                }
            )
            logger.info("Logged ETS pyfunc model")
        except Exception as e:
            logger.warning(f"Failed to log ETS pyfunc model: {e}")
            # Fallback: Save as pickle
            try:
                model_path = "/tmp/ets_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump({'model': final_fitted_model, 'params': best_params, 'freq': pd_freq}, f)
                mlflow.log_artifact(model_path, "model")
                logger.info("Logged ETS model as pickle artifact (fallback)")
            except Exception: pass

        best_run_id = parent_run_id
        best_artifact_uri = parent_run.info.artifact_uri

    return best_run_id, f"runs:/{best_run_id}/model", best_metrics, validation_data, forecast_data, best_artifact_uri, best_params


def evaluate_ets_params(
    trend: Optional[str],
    seasonal: Optional[str],
    seasonal_periods: int,
    train_y: np.ndarray,
    test_y: np.ndarray,
    parent_run_id: str,
    experiment_id: str
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a single ETS parameter combination (thread-safe)
    """
    try:
        client = MlflowClient()
        
        # Create child run
        child_run = client.create_run(
            experiment_id=experiment_id,
            tags={"mlflow.parentRunId": parent_run_id}
        )
        run_id = child_run.info.run_id
        
        try:
            # Log parameters
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
            
            # Log metrics
            client.log_metric(run_id, "mape", metrics["mape"])
            client.log_metric(run_id, "rmse", metrics["rmse"])
            client.log_metric(run_id, "r2", metrics["r2"])
            
            # Set run name
            client.set_tag(run_id, "mlflow.runName", f"ETS({trend}/{seasonal})")
            
            # Terminate run
            client.set_terminated(run_id, "FINISHED")
            
            logger.info(f"  ✓ ETS({trend}/{seasonal}): MAPE={metrics['mape']:.2f}%, RMSE={metrics['rmse']:.2f}")
            
            return {
                "params": {"trend": trend, "seasonal": seasonal},
                "metrics": metrics,
                "fitted_model": fitted_model
            }
            
        except Exception as e:
            client.set_terminated(run_id, "FAILED")
            logger.warning(f"  ✗ ETS({trend}/{seasonal}) failed: {e}")
            return None
            
    except Exception as e:
        logger.warning(f"  ✗ ETS({trend}/{seasonal}) failed to create run: {e}")
        return None


def evaluate_arima_params(
    order: Tuple[int, int, int],
    train_y: np.ndarray,
    test_y: np.ndarray,
    parent_run_id: str,
    experiment_id: str
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a single ARIMA parameter combination (thread-safe)
    
    Args:
        order: ARIMA(p,d,q) order tuple
        train_y: Training target values
        test_y: Test target values
        parent_run_id: Parent MLflow run ID
        experiment_id: MLflow experiment ID
        
    Returns:
        Dictionary with metrics and order, or None if failed
    """
    try:
        client = MlflowClient()
        
        # Create child run
        child_run = client.create_run(
            experiment_id=experiment_id,
            tags={"mlflow.parentRunId": parent_run_id}
        )
        run_id = child_run.info.run_id
        
        try:
            # Log parameters
            client.log_param(run_id, "model_type", "ARIMA")
            client.log_param(run_id, "order", str(order))
            client.log_param(run_id, "p", order[0])
            client.log_param(run_id, "d", order[1])
            client.log_param(run_id, "q", order[2])
            
            # Train model
            model = ARIMA(train_y, order=order)
            fitted_model = model.fit()
            
            # Validate on test set
            test_predictions = fitted_model.forecast(steps=len(test_y))
            metrics = compute_metrics(test_y, test_predictions)
            
            # Log metrics
            client.log_metric(run_id, "mape", metrics["mape"])
            client.log_metric(run_id, "rmse", metrics["rmse"])
            client.log_metric(run_id, "r2", metrics["r2"])
            
            # Set run name
            client.set_tag(run_id, "mlflow.runName", f"ARIMA_{order}")
            
            # Terminate run
            client.set_terminated(run_id, "FINISHED")
            
            logger.info(f"  ✓ ARIMA{order}: MAPE={metrics['mape']:.2f}%, RMSE={metrics['rmse']:.2f}")
            
            return {
                "order": order,
                "metrics": metrics,
                "fitted_model": fitted_model
            }
            
        except Exception as e:
            client.set_terminated(run_id, "FAILED")
            logger.warning(f"  ✗ ARIMA{order} failed: {e}")
            return None
            
    except Exception as e:
        logger.warning(f"  ✗ ARIMA{order} failed to create run: {e}")
        return None


def analyze_covariate_impact(
    model,  # Prophet model (type hint removed for lazy import)
    train_df: pd.DataFrame,
    covariates: List[str]
) -> List[Dict[str, Any]]:
    """
    Analyze the impact of each covariate on the forecast (Prophet only)
    """
    from prophet import Prophet  # Lazy import
    
    impacts = []
    
    if not covariates:
        return impacts
    
    try:
        # Get regressor coefficients from Prophet model
        if hasattr(model, 'params') and 'beta' in model.params:
            # Prophet stores regressor coefficients in params['beta']
            beta = model.params['beta']
            
            # Handle potential 2D array (1, n_regressors)
            if len(beta.shape) > 1 and beta.shape[0] == 1:
                beta = beta[0]
            
            for i, cov in enumerate(covariates):
                if i < len(beta):
                    coef = beta[i]
                    
                    # Calculate importance score (normalized)
                    std = train_df[cov].std() if cov in train_df.columns else 1
                    impact_score = abs(coef * std)
                    
                    impacts.append({
                        'name': cov,
                        'coefficient': float(coef),
                        'impact_score': float(impact_score),
                        'direction': 'positive' if coef > 0 else 'negative'
                    })
        
        # Normalize scores to 0-100 scale
        if impacts:
            max_score = max(imp['impact_score'] for imp in impacts)
            if max_score > 0:
                for imp in impacts:
                    imp['score'] = round((imp['impact_score'] / max_score) * 100, 2)
            else:
                for imp in impacts:
                    imp['score'] = 0
        
        logger.info(f"Analyzed {len(impacts)} covariate impacts")

    except Exception as e:
        logger.warning(f"Could not analyze covariate impacts: {e}")

    return impacts


# =============================================================================
# SARIMAX MODEL - Seasonal ARIMA with eXogenous variables (supports covariates)
# =============================================================================

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

    def __init__(self, fitted_model, order, seasonal_order, frequency, covariates, covariate_means):
        self.fitted_model = fitted_model
        self.order = order
        self.seasonal_order = seasonal_order
        self.covariates = covariates or []
        self.covariate_means = covariate_means or {}
        # Store frequency in human-readable format
        freq_to_human = {'MS': 'monthly', 'W': 'weekly', 'D': 'daily', 'YS': 'yearly'}
        self.frequency = freq_to_human.get(frequency, frequency)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd
        import numpy as np

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS', 'yearly': 'YS'}
        pandas_freq = freq_map.get(self.frequency, 'MS')

        # Mode 1: Simple forecast (periods + start_date)
        if 'periods' in model_input.columns:
            periods = int(model_input['periods'].iloc[0])
            start_date = pd.to_datetime(model_input['start_date'].iloc[0])
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

        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values,
            'yhat_lower': forecast_values * 0.9,
            'yhat_upper': forecast_values * 1.1
        })


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
if COVARIATES:
    exog_train = train_df[COVARIATES].values
    exog_test = test_df[COVARIATES].values

# =============================================================================
# Train SARIMAX Model
# =============================================================================
model = SARIMAX(
    train_df['y'].values,
    exog=exog_train,
    order=ORDER,
    seasonal_order=SEASONAL_ORDER,
    enforce_stationarity=False,
    enforce_invertibility=False
)
fitted_model = model.fit(disp=False)

# =============================================================================
# Validation
# =============================================================================
test_predictions = fitted_model.forecast(steps=len(test_df), exog=exog_test)
mape = mean_absolute_percentage_error(test_df['y'], test_predictions) * 100
rmse = np.sqrt(mean_squared_error(test_df['y'], test_predictions))
r2 = r2_score(test_df['y'], test_predictions)

print(f"SARIMAX{order}x{seasonal_order} - MAPE: {{mape:.2f}}%, RMSE: {{rmse:.2f}}, R2: {{r2:.4f}}")

# =============================================================================
# Forecast
# =============================================================================
# For future forecasts, prepare exogenous variables for forecast horizon
# exog_future = future_df[COVARIATES].values if COVARIATES else None
# forecast = fitted_model.forecast(steps=HORIZON, exog=exog_future)
'''
    return code


def evaluate_sarimax_params(
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    train_y: np.ndarray,
    test_y: np.ndarray,
    train_exog: Optional[np.ndarray],
    test_exog: Optional[np.ndarray],
    parent_run_id: str,
    experiment_id: str
) -> Optional[Dict[str, Any]]:
    """Evaluate a single SARIMAX parameter combination (thread-safe)"""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    try:
        client = MlflowClient()

        child_run = client.create_run(
            experiment_id=experiment_id,
            tags={"mlflow.parentRunId": parent_run_id}
        )
        run_id = child_run.info.run_id

        try:
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

            test_predictions = fitted_model.forecast(steps=len(test_y), exog=test_exog)
            metrics = compute_metrics(test_y, test_predictions)

            client.log_metric(run_id, "mape", metrics["mape"])
            client.log_metric(run_id, "rmse", metrics["rmse"])
            client.log_metric(run_id, "r2", metrics["r2"])
            client.set_tag(run_id, "mlflow.runName", f"SARIMAX{order}x{seasonal_order}")
            client.set_terminated(run_id, "FINISHED")

            logger.info(f"  ✓ SARIMAX{order}x{seasonal_order}: MAPE={metrics['mape']:.2f}%, RMSE={metrics['rmse']:.2f}")

            return {
                "order": order,
                "seasonal_order": seasonal_order,
                "metrics": metrics,
                "fitted_model": fitted_model
            }

        except Exception as e:
            client.set_terminated(run_id, "FAILED")
            logger.warning(f"  ✗ SARIMAX{order}x{seasonal_order} failed: {e}")
            return None

    except Exception as e:
        logger.warning(f"  ✗ SARIMAX{order}x{seasonal_order} failed to create run: {e}")
        return None


def train_sarimax_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    frequency: str = 'monthly',
    covariates: List[str] = None,
    random_seed: int = 42,
    original_data: List[Dict[str, Any]] = None,
    country: str = 'US'
) -> Tuple[str, str, Dict[str, float], pd.DataFrame, pd.DataFrame, str, Dict[str, Any]]:
    """
    Train SARIMAX model with hyperparameter tuning and MLflow logging

    SARIMAX supports external regressors (covariates) unlike basic ARIMA.
    Automatically adds country-specific holiday indicators as features.

    Args:
        train_df: Training dataframe with 'ds', 'y', and covariate columns
        test_df: Test dataframe with 'ds', 'y', and covariate columns
        horizon: Number of periods to forecast
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        covariates: List of covariate column names to use as exogenous variables
        random_seed: Random seed for reproducibility
        original_data: Original uploaded data (for logging)
        country: Country code for holiday calendar (US, UK, CA, etc.)
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import holidays

    logger.info(f"Training SARIMAX model (freq={frequency}, covariates={covariates}, country={country}, seed={random_seed})...")

    # Set random seeds for reproducibility
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    logger.info(f"SARIMAX: Set random seed to {random_seed} for reproducibility")

    freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS', 'yearly': 'YS'}
    pd_freq = freq_map.get(frequency, 'MS')

    # Determine seasonal period based on frequency
    seasonal_period_map = {'daily': 7, 'weekly': 52, 'monthly': 12, 'yearly': 1}
    seasonal_period = seasonal_period_map.get(frequency, 12)

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
    # Add holiday to covariates if not already present
    if 'is_holiday' not in valid_covariates:
        valid_covariates.append('is_holiday')

    # Add promo-derived features for better holiday forecasting
    if valid_covariates:
        # Combined promo indicator
        promo_cols = [c for c in valid_covariates if c in train_df.columns and c != 'is_holiday']
        if promo_cols:
            for df in [train_df, test_df]:
                df['any_promo_active'] = df[promo_cols].max(axis=1)
                df['promo_window'] = df['any_promo_active'].rolling(window=5, center=True, min_periods=1).max()
                # Weekend near promo
                df['day_of_week'] = df['ds'].dt.dayofweek
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                df['is_promo_weekend'] = ((df['is_weekend'] == 1) & (df['promo_window'] == 1)).astype(int)

            # Add derived features to covariates
            for derived in ['any_promo_active', 'promo_window', 'is_promo_weekend']:
                if derived not in valid_covariates:
                    valid_covariates.append(derived)
            logger.info(f"SARIMAX: Added promo-derived features from {len(promo_cols)} promo columns")

    if valid_covariates:
        train_exog = train_df[valid_covariates].values
        test_exog = test_df[valid_covariates].values
        covariate_means = {c: train_df[c].mean() for c in valid_covariates}
        logger.info(f"Using {len(valid_covariates)} covariates: {valid_covariates}")
    else:
        train_exog = None
        test_exog = None
        covariate_means = {}
        if covariates:
            logger.warning(f"Requested covariates {covariates} not found in data, training without covariates")

    best_model = None
    best_metrics = {"mape": float('inf'), "rmse": float('inf')}
    best_order = None
    best_seasonal_order = None
    best_fitted_model = None
    best_run_id = None
    best_artifact_uri = None

    # Grid search space - reduced for efficiency
    max_combinations = int(os.environ.get('SARIMAX_MAX_COMBINATIONS', '8'))

    # ARIMA orders
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1]

    # Seasonal orders (simplified)
    P_values = [0, 1]
    D_values = [0, 1]
    Q_values = [0, 1]

    all_orders = []
    for p, d, q in itertools.product(p_values, d_values, q_values):
        for P, D, Q in itertools.product(P_values, D_values, Q_values):
            all_orders.append(((p, d, q), (P, D, Q, seasonal_period)))

    # Prioritize simpler models
    all_orders.sort(key=lambda x: sum(x[0]) + sum(x[1][:3]))
    orders = all_orders[:max_combinations]
    logger.info(f"Limited SARIMAX combinations to {len(orders)} (from {len(all_orders)} total)")

    with mlflow.start_run(run_name="SARIMAX_Tuning", nested=True) as parent_run:
        parent_run_id = parent_run.info.run_id
        experiment_id = parent_run.info.experiment_id

        # Log original data
        if original_data:
            try:
                original_df = pd.DataFrame(original_data)
                original_df.to_csv("/tmp/original_timeseries_data.csv", index=False)
                mlflow.log_artifact("/tmp/original_timeseries_data.csv", "datasets/raw")
                logger.info(f"Logged original data: {len(original_df)} rows")
            except Exception as e:
                logger.warning(f"Could not log original data: {e}")

        max_workers = int(os.environ.get('MLFLOW_MAX_WORKERS', '1'))
        logger.info(f"Running SARIMAX hyperparameter tuning with {len(orders)} combinations, {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    evaluate_sarimax_params, order, seasonal_order,
                    train_df['y'].values, test_df['y'].values,
                    train_exog, test_exog, parent_run_id, experiment_id
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

        # Time Series Cross-Validation for more robust MAPE estimate
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
                logger.info(f"  CV Fold {i+1}/{n_splits}: train={len(y_train_cv)}, test={len(y_test_cv)}, MAPE={fold_mape:.2f}%")
            except Exception as e:
                logger.warning(f"  CV Fold {i+1} failed: {e}")

        if len(cv_scores) > 0:
            cv_mean = round(np.mean(cv_scores), 2)
            cv_std = round(np.std(cv_scores), 2)
            logger.info(f"CV Results: Mean MAPE={cv_mean:.2f}% (±{cv_std:.2f}%), {len(cv_scores)} folds")
            best_metrics["cv_mape"] = cv_mean
            best_metrics["cv_mape_std"] = cv_std

        # Validation with proper confidence intervals
        test_predictions = best_fitted_model.forecast(steps=len(test_df), exog=test_exog)

        # Get in-sample predictions for confidence interval calculation
        train_predictions = best_fitted_model.fittedvalues
        if len(train_predictions) > 0:
            yhat_lower, yhat_upper = compute_prediction_intervals(
                y_train=train_df['y'].values,
                y_pred_train=train_predictions[-len(train_df):] if len(train_predictions) >= len(train_df) else train_predictions,
                forecast_values=test_predictions,
                confidence_level=0.95
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

        # For forecast, use mean covariate values (or user can provide actual future values)
        if valid_covariates:
            future_exog = np.array([[covariate_means[c] for c in valid_covariates]] * horizon)
        else:
            future_exog = None

        forecast_values = final_fitted_model.forecast(steps=horizon, exog=future_exog)

        # Compute proper prediction intervals for forecast
        final_train_predictions = final_fitted_model.fittedvalues
        if len(final_train_predictions) > 0:
            forecast_lower, forecast_upper = compute_prediction_intervals(
                y_train=full_data['y'].values,
                y_pred_train=final_train_predictions,
                forecast_values=forecast_values,
                confidence_level=0.95
            )
        else:
            forecast_lower = forecast_values * 0.9
            forecast_upper = forecast_values * 1.1

        last_date = full_data['ds'].max()
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=pd_freq)[1:]

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
            logger.info(f"Logged SARIMAX datasets")
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
        mlflow.log_metrics(best_metrics)

        # Log reproducible code
        training_code = generate_sarimax_training_code(
            best_order, best_seasonal_order, horizon, frequency,
            valid_covariates, best_metrics, len(train_df), len(test_df)
        )
        mlflow.log_text(training_code, "training_code.py")
        logger.info("Logged reproducible SARIMAX training code")

        # Log MLflow pyfunc model
        try:
            import sys
            from mlflow.models.signature import infer_signature

            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            input_example = pd.DataFrame({
                'periods': [horizon],
                'start_date': [str(last_date.date()) if hasattr(last_date, 'date') else str(last_date)[:10]]
            })
            sample_output = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(1).copy()
            signature = infer_signature(input_example, sample_output)

            model_wrapper = SARIMAXModelWrapper(
                final_fitted_model, best_order, best_seasonal_order,
                frequency, valid_covariates, covariate_means
            )

            mlflow.pyfunc.log_model(
                name="model",
                python_model=model_wrapper,
                signature=signature,
                input_example=input_example,
                code_paths=["backend"],
                conda_env={
                    "channels": ["defaults", "conda-forge"],
                    "dependencies": [
                        f"python={python_version}",
                        "pip",
                        {"pip": ["mlflow", "pandas", "numpy", "statsmodels", "scikit-learn"]}
                    ],
                    "name": "sarimax_env"
                }
            )
            logger.info("Logged SARIMAX pyfunc model")
        except Exception as e:
            logger.warning(f"Failed to log SARIMAX pyfunc model: {e}")

        best_run_id = parent_run_id
        best_artifact_uri = parent_run.info.artifact_uri

    params = {
        "order": best_order,
        "seasonal_order": best_seasonal_order,
        "covariates": valid_covariates
    }

    return best_run_id, f"runs:/{best_run_id}/model", best_metrics, validation_data, forecast_data, best_artifact_uri, params


# =============================================================================
# XGBOOST MODEL - Gradient Boosting for Time Series with full covariate support
# =============================================================================

class XGBoostModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow-compatible wrapper for XGBoost time series model

    Input format for serving endpoint (Mode 1 - Simple):
    {
        "dataframe_records": [
            {"periods": 30, "start_date": "2025-01-01"}
        ]
    }

    Input format for serving endpoint (Mode 2 - With features):
    {
        "dataframe_records": [
            {"ds": "2025-01-01", "Black Friday": 0, "Thanksgiving": 1, ...},
            {"ds": "2025-01-02", "Black Friday": 0, "Thanksgiving": 0, ...},
            ...
        ]
    }
    """

    def __init__(self, model, feature_columns, frequency, last_known_values, covariate_means, yoy_lag_values=None):
        self.model = model
        self.feature_columns = feature_columns
        self.last_known_values = last_known_values  # For short-term lag features
        self.yoy_lag_values = yoy_lag_values or {}  # For YoY lag features (date -> value mapping)
        self.covariate_means = covariate_means or {}
        freq_to_human = {'MS': 'monthly', 'W': 'weekly', 'D': 'daily', 'YS': 'yearly'}
        self.frequency = freq_to_human.get(frequency, frequency)
        # YoY lag period
        yoy_lag_map = {'daily': 364, 'weekly': 52, 'monthly': 12}
        self.yoy_lag = yoy_lag_map.get(self.frequency, 364)

    def _create_calendar_features(self, df):
        """Create enhanced calendar features from date column"""
        df = df.copy()
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['day_of_month'] = df['ds'].dt.day
        df['month'] = df['ds'].dt.month
        df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = (df['ds'].dt.dayofweek >= 5).astype(int)
        df['quarter'] = df['ds'].dt.quarter
        # Enhanced calendar features
        df['is_month_start'] = df['ds'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['ds'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['ds'].dt.is_quarter_end.astype(int)
        df['week_of_month'] = (df['ds'].dt.day - 1) // 7 + 1
        return df

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd
        import numpy as np

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS', 'yearly': 'YS'}
        pandas_freq = freq_map.get(self.frequency, 'MS')

        # Mode 1: Simple forecast
        if 'periods' in model_input.columns:
            periods = int(model_input['periods'].iloc[0])
            start_date = pd.to_datetime(model_input['start_date'].iloc[0])
            future_dates = pd.date_range(start=start_date, periods=periods + 1, freq=pandas_freq)[1:]
            future_df = pd.DataFrame({'ds': future_dates})
        else:
            # Mode 2: Advanced with features
            model_input['ds'] = pd.to_datetime(model_input['ds'])
            future_df = model_input.copy()
            periods = len(future_df)

        # Create calendar features
        future_df = self._create_calendar_features(future_df)

        # Add lag features (use last known values, then predictions recursively)
        predictions = []
        last_values = self.last_known_values.copy()
        hist_mean = np.mean(last_values) if last_values else 0

        for i in range(len(future_df)):
            row = future_df.iloc[[i]].copy()
            current_date = row['ds'].iloc[0]

            # Short-term lag features
            row['lag_1'] = last_values[-1] if len(last_values) >= 1 else 0
            row['lag_7'] = last_values[-7] if len(last_values) >= 7 else last_values[-1] if last_values else 0
            row['rolling_mean_7'] = np.mean(last_values[-7:]) if len(last_values) >= 7 else np.mean(last_values) if last_values else 0

            # YoY lag features - lookup from historical data
            yoy_lag_date = current_date - pd.Timedelta(days=self.yoy_lag if self.frequency == 'daily' else self.yoy_lag * 7 if self.frequency == 'weekly' else self.yoy_lag * 30)
            yoy_value = self.yoy_lag_values.get(yoy_lag_date.strftime('%Y-%m-%d'), hist_mean)
            row[f'lag_{self.yoy_lag}'] = yoy_value
            row[f'lag_{self.yoy_lag}_rolling_avg'] = yoy_value
            row['yoy_ratio'] = 1.0  # Default for prediction

            if self.frequency == 'daily':
                lag_365_date = current_date - pd.Timedelta(days=365)
                row['lag_365'] = self.yoy_lag_values.get(lag_365_date.strftime('%Y-%m-%d'), hist_mean)

            # Fill covariate columns with means if not provided
            for cov in self.covariate_means:
                if cov not in row.columns:
                    row[cov] = self.covariate_means[cov]
                elif pd.isna(row[cov].iloc[0]):
                    row[cov] = self.covariate_means[cov]

            # Promo-derived features (set defaults if not in input)
            for promo_feat in ['any_promo_active', 'promo_count', 'promo_window', 'is_promo_weekend', 'is_regular_weekend']:
                if promo_feat not in row.columns:
                    row[promo_feat] = 0

            # Select only the features used in training
            X = row[[c for c in self.feature_columns if c in row.columns]]

            # Add missing columns with 0
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0

            X = X[self.feature_columns]
            pred = self.model.predict(X)[0]
            predictions.append(pred)
            last_values.append(pred)

        forecast_values = np.array(predictions)

        return pd.DataFrame({
            'ds': future_df['ds'],
            'yhat': forecast_values,
            'yhat_lower': forecast_values * 0.9,
            'yhat_upper': forecast_values * 1.1
        })


def generate_xgboost_training_code(
    params: Dict[str, Any],
    feature_columns: List[str],
    horizon: int,
    frequency: str,
    covariates: List[str],
    metrics: Dict[str, float],
    train_size: int,
    test_size: int
) -> str:
    """Generate reproducible Python code for XGBoost time series model training"""

    code = f'''"""
Reproducible XGBoost Time Series Model Training Code
Generated for reproducibility - Full covariate and calendar feature support
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# =============================================================================
# Configuration
# =============================================================================
PARAMS = {params}
FEATURE_COLUMNS = {feature_columns}
FREQUENCY = '{frequency}'
HORIZON = {horizon}
COVARIATES = {covariates}

# Metrics from training:
# MAPE: {metrics.get('mape', 'N/A'):.4f}%
# RMSE: {metrics.get('rmse', 'N/A'):.4f}
# R2: {metrics.get('r2', 'N/A'):.4f}

# Data splits:
# Training size: {train_size}
# Test size: {test_size}

# =============================================================================
# Feature Engineering Functions
# =============================================================================
def create_calendar_features(df):
    """Create calendar features from date column"""
    df = df.copy()
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['day_of_month'] = df['ds'].dt.day
    df['month'] = df['ds'].dt.month
    df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['ds'].dt.dayofweek >= 5).astype(int)
    df['quarter'] = df['ds'].dt.quarter
    return df

def create_lag_features(df, target_col='y'):
    """Create lag and rolling features"""
    df = df.copy()
    df['lag_1'] = df[target_col].shift(1)
    df['lag_7'] = df[target_col].shift(7)
    df['rolling_mean_7'] = df[target_col].rolling(window=7, min_periods=1).mean()
    return df

# =============================================================================
# Load and Prepare Data
# =============================================================================
# df = pd.read_csv('your_data.csv')
# df['ds'] = pd.to_datetime(df['your_date_column'])
# df['y'] = df['your_target_column']
# df = df.sort_values('ds')

# Create features
# df = create_calendar_features(df)
# df = create_lag_features(df)

# =============================================================================
# Train XGBoost Model
# =============================================================================
# X_train = train_df[FEATURE_COLUMNS]
# y_train = train_df['y']

model = XGBRegressor(**PARAMS)
# model.fit(X_train, y_train)

# =============================================================================
# Feature Importance
# =============================================================================
# importance = pd.DataFrame({{
#     'feature': FEATURE_COLUMNS,
#     'importance': model.feature_importances_
# }}).sort_values('importance', ascending=False)
# print(importance)
'''
    return code


def create_xgboost_features(df: pd.DataFrame, target_col: str = 'y', covariates: List[str] = None, include_lags: bool = True, frequency: str = 'daily') -> pd.DataFrame:
    """
    Create all features for XGBoost time series model including YoY lag features
    for better holiday forecasting.

    Args:
        df: DataFrame with 'ds' column and optionally target_col
        target_col: Name of target column (default 'y')
        covariates: List of covariate column names
        include_lags: Whether to include lag features (set False for future prediction df without target)
        frequency: Data frequency for determining YoY lag periods ('daily', 'weekly', 'monthly')
    """
    df = df.copy()

    # Calendar features (always available since we have ds)
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['day_of_month'] = df['ds'].dt.day
    df['month'] = df['ds'].dt.month
    df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['ds'].dt.dayofweek >= 5).astype(int)
    df['quarter'] = df['ds'].dt.quarter

    # Enhanced calendar features for financial data
    df['is_month_start'] = df['ds'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['ds'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['ds'].dt.is_quarter_end.astype(int)
    df['week_of_month'] = (df['ds'].dt.day - 1) // 7 + 1

    # Determine YoY lag periods based on frequency
    yoy_lag_map = {'daily': 364, 'weekly': 52, 'monthly': 12}
    yoy_lag = yoy_lag_map.get(frequency, 364)

    # Lag features - only if target column exists and include_lags is True
    if include_lags and target_col in df.columns:
        # Short-term lags
        df['lag_1'] = df[target_col].shift(1)
        df['lag_7'] = df[target_col].shift(7)
        df['rolling_mean_7'] = df[target_col].rolling(window=7, min_periods=1).mean()

        # YoY lag features (CRITICAL for holiday patterns)
        # Same day/week/month last year is the best predictor for holidays
        df[f'lag_{yoy_lag}'] = df[target_col].shift(yoy_lag)
        if frequency == 'daily':
            df['lag_365'] = df[target_col].shift(365)  # Handle leap years

        # Smoothed YoY (handles slight date misalignment)
        df[f'lag_{yoy_lag}_rolling_avg'] = (
            df[target_col]
            .shift(yoy_lag - 3)
            .rolling(window=7, min_periods=1)
            .mean()
        )

        # YoY ratio - year-over-year growth indicator
        safe_lag = df[f'lag_{yoy_lag}'].replace(0, np.nan)
        df['yoy_ratio'] = (df[target_col] / safe_lag).clip(0.1, 10.0).fillna(1.0)

        # Fill NaN in lag features
        lag_cols = ['lag_1', 'lag_7', 'rolling_mean_7', f'lag_{yoy_lag}', f'lag_{yoy_lag}_rolling_avg']
        if frequency == 'daily':
            lag_cols.append('lag_365')
        for col in lag_cols:
            if col in df.columns:
                df[col] = df[col].bfill().ffill()
    elif include_lags:
        # Target column doesn't exist - initialize lag columns with 0 (will be filled later)
        df['lag_1'] = 0
        df['lag_7'] = 0
        df['rolling_mean_7'] = 0
        df[f'lag_{yoy_lag}'] = 0
        df[f'lag_{yoy_lag}_rolling_avg'] = 0
        df['yoy_ratio'] = 1.0
        if frequency == 'daily':
            df['lag_365'] = 0

    # Promo-derived features if covariates (promo columns) are present
    if covariates:
        valid_promo_cols = [c for c in covariates if c in df.columns]
        if valid_promo_cols:
            # Combined promo indicator
            df['any_promo_active'] = df[valid_promo_cols].max(axis=1)
            df['promo_count'] = df[valid_promo_cols].sum(axis=1)

            # Extended promo window (±2 days effect)
            df['promo_window'] = df['any_promo_active'].rolling(window=5, center=True, min_periods=1).max()

            # Is this a promo/holiday weekend?
            df['is_promo_weekend'] = ((df['is_weekend'] == 1) & (df['promo_window'] == 1)).astype(int)
            df['is_regular_weekend'] = ((df['is_weekend'] == 1) & (df['promo_window'] == 0)).astype(int)

            logger.info(f"XGBoost: Added promo-derived features from {len(valid_promo_cols)} promo columns")

    return df


def train_xgboost_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    frequency: str = 'monthly',
    covariates: List[str] = None,
    random_seed: int = 42,
    original_data: List[Dict[str, Any]] = None,
    country: str = 'US'
) -> Tuple[str, str, Dict[str, float], pd.DataFrame, pd.DataFrame, str, Dict[str, Any]]:
    """
    Train XGBoost model for time series forecasting with full covariate support

    XGBoost can learn non-linear holiday effects and feature interactions.
    Automatically creates calendar features (day_of_week, is_weekend, month, etc.)
    and lag features. Also adds country-specific holiday indicators.

    Args:
        train_df: Training dataframe with 'ds', 'y', and covariate columns
        test_df: Test dataframe with 'ds', 'y', and covariate columns
        horizon: Number of periods to forecast
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        covariates: List of covariate column names (e.g., holiday flags)
        random_seed: Random seed for reproducibility
        original_data: Original uploaded data (for logging)
        country: Country code for holiday calendar (US, UK, CA, etc.)
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    import holidays

    logger.info(f"Training XGBoost model (freq={frequency}, covariates={covariates}, country={country}, seed={random_seed})...")

    # Set random seeds for reproducibility (XGBRegressor also uses random_state param)
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    logger.info(f"XGBoost: Set random seed to {random_seed} for reproducibility")

    freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS', 'yearly': 'YS'}
    pd_freq = freq_map.get(frequency, 'MS')

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

    covariates = covariates or []
    valid_covariates = [c for c in covariates if c in train_df.columns]
    # Add holiday to covariates if not already present
    if 'is_holiday' not in valid_covariates:
        valid_covariates.append('is_holiday')

    if valid_covariates:
        logger.info(f"Using {len(valid_covariates)} covariates: {valid_covariates}")

    # Combine data for feature engineering (need continuous series for lag features)
    full_df = pd.concat([train_df, test_df]).sort_values('ds').reset_index(drop=True)
    full_df = create_xgboost_features(full_df, 'y', valid_covariates, include_lags=True, frequency=frequency)

    # Determine YoY lag column name based on frequency
    yoy_lag_map = {'daily': 364, 'weekly': 52, 'monthly': 12}
    yoy_lag = yoy_lag_map.get(frequency, 364)

    # Split back
    train_end_idx = len(train_df)
    train_featured = full_df.iloc[:train_end_idx].copy()
    test_featured = full_df.iloc[train_end_idx:].copy()

    # Define feature columns - now includes YoY lags and promo-derived features
    calendar_features = [
        'day_of_week', 'day_of_month', 'month', 'week_of_year', 'is_weekend', 'quarter',
        'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'week_of_month'
    ]
    lag_features = ['lag_1', 'lag_7', 'rolling_mean_7', f'lag_{yoy_lag}', f'lag_{yoy_lag}_rolling_avg', 'yoy_ratio']
    if frequency == 'daily':
        lag_features.append('lag_365')

    # Promo-derived features (added by create_xgboost_features if promo columns exist)
    promo_derived = ['any_promo_active', 'promo_count', 'promo_window', 'is_promo_weekend', 'is_regular_weekend']
    promo_derived = [c for c in promo_derived if c in full_df.columns]

    feature_columns = calendar_features + lag_features + valid_covariates + promo_derived
    logger.info(f"XGBoost using {len(feature_columns)} features including YoY lags and promo-derived features")

    # Ensure all feature columns exist
    for col in feature_columns:
        if col not in train_featured.columns:
            train_featured[col] = 0
            test_featured[col] = 0

    X_train = train_featured[feature_columns].fillna(0)
    y_train = train_featured['y']
    X_test = test_featured[feature_columns].fillna(0)
    y_test = test_featured['y']

    # Compute covariate means for future prediction
    covariate_means = {c: train_df[c].mean() for c in valid_covariates} if valid_covariates else {}

    best_model = None
    best_metrics = {"mape": float('inf'), "rmse": float('inf')}
    best_params = None
    best_run_id = None
    best_artifact_uri = None

    # Hyperparameter grid
    param_grid = [
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
        {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
        {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
    ]

    max_combinations = int(os.environ.get('XGBOOST_MAX_COMBINATIONS', '4'))
    param_grid = param_grid[:max_combinations]

    with mlflow.start_run(run_name="XGBoost_Tuning", nested=True) as parent_run:
        parent_run_id = parent_run.info.run_id
        experiment_id = parent_run.info.experiment_id

        # Log original data
        if original_data:
            try:
                original_df = pd.DataFrame(original_data)
                original_df.to_csv("/tmp/original_timeseries_data.csv", index=False)
                mlflow.log_artifact("/tmp/original_timeseries_data.csv", "datasets/raw")
                logger.info(f"Logged original data: {len(original_df)} rows")
            except Exception as e:
                logger.warning(f"Could not log original data: {e}")

        logger.info(f"Running XGBoost hyperparameter tuning with {len(param_grid)} combinations")

        for params in param_grid:
            try:
                client = MlflowClient()
                child_run = client.create_run(
                    experiment_id=experiment_id,
                    tags={"mlflow.parentRunId": parent_run_id}
                )
                run_id = child_run.info.run_id

                try:
                    for key, value in params.items():
                        client.log_param(run_id, key, value)

                    model = XGBRegressor(
                        **params,
                        random_state=random_seed,
                        objective='reg:squarederror',
                        verbosity=0
                    )
                    model.fit(X_train, y_train)

                    predictions = model.predict(X_test)
                    metrics = compute_metrics(y_test.values, predictions)

                    client.log_metric(run_id, "mape", metrics["mape"])
                    client.log_metric(run_id, "rmse", metrics["rmse"])
                    client.log_metric(run_id, "r2", metrics["r2"])
                    client.set_tag(run_id, "mlflow.runName", f"XGB_d{params['max_depth']}_n{params['n_estimators']}")
                    client.set_terminated(run_id, "FINISHED")

                    logger.info(f"  ✓ XGBoost(depth={params['max_depth']}, n={params['n_estimators']}): MAPE={metrics['mape']:.2f}%")

                    is_better = metrics["mape"] < best_metrics["mape"] or \
                               (abs(metrics["mape"] - best_metrics["mape"]) < 0.5 and metrics["rmse"] < best_metrics["rmse"])

                    if is_better:
                        best_metrics = metrics
                        best_model = model
                        best_params = params
                        logger.info(f"  ✨ New best XGBoost: MAPE={metrics['mape']:.2f}%")

                except Exception as e:
                    client.set_terminated(run_id, "FAILED")
                    logger.warning(f"  ✗ XGBoost params {params} failed: {e}")

            except Exception as e:
                logger.warning(f"  ✗ Failed to create run for params {params}: {e}")

        if best_model is None:
            raise Exception("XGBoost training failed - no successful model fits")

        # Time Series Cross-Validation for more robust MAPE estimate
        logger.info(f"Running time series cross-validation for XGBoost...")

        n = len(full_df)
        min_train_size = max(n // 2, 10)
        n_splits = 3
        test_size_cv = len(test_df)
        available_for_cv = n - min_train_size
        if available_for_cv < n_splits * test_size_cv:
            n_splits = max(1, available_for_cv // test_size_cv)

        cv_scores = []
        fold_size = (n - min_train_size) // max(n_splits, 1)

        for i in range(n_splits):
            split_point = min_train_size + i * fold_size
            test_end = min(split_point + test_size_cv, n)

            cv_train = full_df.iloc[:split_point].copy()
            cv_test = full_df.iloc[split_point:test_end].copy()

            if len(cv_test) == 0:
                continue

            try:
                # Create features for CV fold
                cv_combined = pd.concat([cv_train, cv_test]).sort_values('ds').reset_index(drop=True)
                cv_combined = create_xgboost_features(cv_combined, 'y', valid_covariates)
                cv_train_featured = cv_combined.iloc[:len(cv_train)].copy()
                cv_test_featured = cv_combined.iloc[len(cv_train):].copy()

                for col in feature_columns:
                    if col not in cv_train_featured.columns:
                        cv_train_featured[col] = 0
                        cv_test_featured[col] = 0

                X_train_cv = cv_train_featured[feature_columns].fillna(0)
                y_train_cv = cv_train_featured['y']
                X_test_cv = cv_test_featured[feature_columns].fillna(0)
                y_test_cv = cv_test_featured['y']

                cv_model = XGBRegressor(
                    **best_params,
                    random_state=random_seed,
                    objective='reg:squarederror',
                    verbosity=0
                )
                cv_model.fit(X_train_cv, y_train_cv)
                cv_preds = cv_model.predict(X_test_cv)

                fold_mape = mean_absolute_percentage_error(y_test_cv.values, cv_preds) * 100
                cv_scores.append(fold_mape)
                logger.info(f"  CV Fold {i+1}/{n_splits}: train={len(cv_train)}, test={len(cv_test)}, MAPE={fold_mape:.2f}%")
            except Exception as e:
                logger.warning(f"  CV Fold {i+1} failed: {e}")

        if len(cv_scores) > 0:
            cv_mean = round(np.mean(cv_scores), 2)
            cv_std = round(np.std(cv_scores), 2)
            logger.info(f"CV Results: Mean MAPE={cv_mean:.2f}% (±{cv_std:.2f}%), {len(cv_scores)} folds")
            best_metrics["cv_mape"] = cv_mean
            best_metrics["cv_mape_std"] = cv_std

        # Validation data with proper confidence intervals
        test_predictions = best_model.predict(X_test)
        train_predictions = best_model.predict(X_train)

        if len(train_predictions) > 0:
            yhat_lower, yhat_upper = compute_prediction_intervals(
                y_train=y_train.values,
                y_pred_train=train_predictions,
                forecast_values=test_predictions,
                confidence_level=0.95
            )
        else:
            yhat_lower = test_predictions * 0.9
            yhat_upper = test_predictions * 1.1

        validation_data = test_df[['ds', 'y']].copy()
        validation_data['yhat'] = test_predictions
        validation_data['yhat_lower'] = yhat_lower
        validation_data['yhat_upper'] = yhat_upper

        # Refit on full data
        full_featured = create_xgboost_features(full_df.copy(), 'y', valid_covariates, include_lags=True, frequency=frequency)
        for col in feature_columns:
            if col not in full_featured.columns:
                full_featured[col] = 0

        X_full = full_featured[feature_columns].fillna(0)
        y_full = full_featured['y']

        final_model = XGBRegressor(
            **best_params,
            random_state=random_seed,
            objective='reg:squarederror',
            verbosity=0
        )
        final_model.fit(X_full, y_full)

        # Store last known values for lag features during prediction
        last_known_values = list(full_df['y'].tail(30).values)

        # Store YoY lag values for the model wrapper (date -> value mapping)
        yoy_lag_values = {}
        for _, row in full_df.iterrows():
            date_str = row['ds'].strftime('%Y-%m-%d') if hasattr(row['ds'], 'strftime') else str(row['ds'])[:10]
            yoy_lag_values[date_str] = row['y']
        logger.info(f"XGBoost: Stored {len(yoy_lag_values)} historical values for YoY lag lookups")

        # Generate forecast
        last_date = full_df['ds'].max()
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=pd_freq)[1:]
        future_df = pd.DataFrame({'ds': future_dates})
        # Create features without lag columns (we'll fill them recursively during prediction)
        future_df = create_xgboost_features(future_df, 'y', valid_covariates, include_lags=False)

        # For lag features in forecast, use recursive prediction
        predictions = []
        temp_last_values = last_known_values.copy()

        for i in range(len(future_df)):
            row = future_df.iloc[[i]].copy()
            row['lag_1'] = temp_last_values[-1]
            row['lag_7'] = temp_last_values[-7] if len(temp_last_values) >= 7 else temp_last_values[-1]
            row['rolling_mean_7'] = np.mean(temp_last_values[-7:]) if len(temp_last_values) >= 7 else np.mean(temp_last_values)

            # Fill covariates - use 0 for binary flags (holidays/promos), mean for continuous
            for cov in valid_covariates:
                if cov not in row.columns or pd.isna(row[cov].iloc[0]):
                    # Check if this is a binary covariate (0/1 values only in training data)
                    cov_values = train_df[cov].dropna().unique() if cov in train_df.columns else []
                    is_binary = len(cov_values) <= 2 and all(v in [0, 1, 0.0, 1.0] for v in cov_values)
                    # For binary covariates (holidays, promos), default to 0 (no event)
                    # For continuous covariates, use mean
                    row[cov] = 0 if is_binary else covariate_means.get(cov, 0)

            for col in feature_columns:
                if col not in row.columns:
                    row[col] = 0

            X_pred = row[feature_columns].fillna(0)
            pred = final_model.predict(X_pred)[0]
            predictions.append(pred)
            temp_last_values.append(pred)

        forecast_values = np.array(predictions)

        # Compute proper prediction intervals for forecast
        # Use full training predictions for residual-based intervals
        full_train_predictions = final_model.predict(X_full)
        if len(full_train_predictions) > 0:
            forecast_lower, forecast_upper = compute_prediction_intervals(
                y_train=y_full.values,
                y_pred_train=full_train_predictions,
                forecast_values=forecast_values,
                confidence_level=0.95
            )
        else:
            forecast_lower = forecast_values * 0.9
            forecast_upper = forecast_values * 1.1

        forecast_data = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values,
            'yhat_lower': forecast_lower,
            'yhat_upper': forecast_upper
        })

        # Log datasets
        try:
            train_featured.to_csv("/tmp/train.csv", index=False)
            mlflow.log_artifact("/tmp/train.csv", "datasets/training")
            test_featured.to_csv("/tmp/eval.csv", index=False)
            mlflow.log_artifact("/tmp/eval.csv", "datasets/training")
            full_featured.to_csv("/tmp/full_merged_data.csv", index=False)
            mlflow.log_artifact("/tmp/full_merged_data.csv", "datasets/processed")
            logger.info(f"Logged XGBoost datasets")
        except Exception as e:
            logger.warning(f"Could not log datasets: {e}")

        # Log feature importance
        try:
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False)
            importance_df.to_csv("/tmp/feature_importance.csv", index=False)
            mlflow.log_artifact("/tmp/feature_importance.csv", "analysis")
            logger.info(f"Top 5 features: {importance_df.head(5)['feature'].tolist()}")
        except Exception as e:
            logger.warning(f"Could not log feature importance: {e}")

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
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", best_params['n_estimators'])
        mlflow.log_param("max_depth", best_params['max_depth'])
        mlflow.log_param("learning_rate", best_params['learning_rate'])
        mlflow.log_param("feature_columns", str(feature_columns))
        mlflow.log_param("covariates", str(valid_covariates))
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_metrics(best_metrics)

        # Log reproducible code
        training_code = generate_xgboost_training_code(
            best_params, feature_columns, horizon, frequency,
            valid_covariates, best_metrics, len(train_df), len(test_df)
        )
        mlflow.log_text(training_code, "training_code.py")
        logger.info("Logged reproducible XGBoost training code")

        # Log MLflow pyfunc model
        try:
            import sys
            from mlflow.models.signature import infer_signature

            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            input_example = pd.DataFrame({
                'periods': [horizon],
                'start_date': [str(last_date.date()) if hasattr(last_date, 'date') else str(last_date)[:10]]
            })
            sample_output = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(1).copy()
            signature = infer_signature(input_example, sample_output)

            model_wrapper = XGBoostModelWrapper(
                final_model, feature_columns, frequency,
                last_known_values, covariate_means, yoy_lag_values
            )

            mlflow.pyfunc.log_model(
                name="model",
                python_model=model_wrapper,
                signature=signature,
                input_example=input_example,
                code_paths=["backend"],
                conda_env={
                    "channels": ["defaults", "conda-forge"],
                    "dependencies": [
                        f"python={python_version}",
                        "pip",
                        {"pip": ["mlflow", "pandas", "numpy", "xgboost", "scikit-learn"]}
                    ],
                    "name": "xgboost_env"
                }
            )
            logger.info("Logged XGBoost pyfunc model")
        except Exception as e:
            logger.warning(f"Failed to log XGBoost pyfunc model: {e}")

        best_run_id = parent_run_id
        best_artifact_uri = parent_run.info.artifact_uri

    params_out = {
        **best_params,
        "feature_columns": feature_columns,
        "covariates": valid_covariates
    }

    return best_run_id, f"runs:/{best_run_id}/model", best_metrics, validation_data, forecast_data, best_artifact_uri, params_out
