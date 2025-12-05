"""
Training service for Prophet forecasting models with MLflow integration and Hyperparameter Tuning
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import mlflow
import mlflow.pyfunc
from datetime import datetime
import logging
import warnings
import itertools
import concurrent.futures
from mlflow.tracking import MlflowClient
from backend.preprocessing import enhance_features_for_forecasting, get_derived_feature_columns, prepare_future_features
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ProphetModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow-compatible wrapper for Prophet model

    Input format for serving endpoint (Mode 1 - Simple):
    {
        "dataframe_records": [
            {"ds": "2025-01-01", "periods": 30}
        ]
    }

    Input format for serving endpoint (Mode 2 - Advanced with covariates):
    {
        "dataframe_records": [
            {"ds": "2025-01-01", "Black Friday": 0, "Christmas": 0},
            {"ds": "2025-01-02", "Black Friday": 0, "Christmas": 0},
            ...
        ]
    }

    - frequency is stored internally (from training) and used for date generation
    - Supported frequencies: 'daily', 'weekly', 'monthly'
    """

    def __init__(self, model, time_col: str, target_col: str, covariates: list, frequency: str = 'monthly'):
        self.model = model
        self.time_col = time_col
        self.target_col = target_col
        self.covariates = covariates
        # Store frequency in human-readable format for consistency
        # Map pandas freq codes to human-readable if needed
        freq_to_human = {'MS': 'monthly', 'W': 'weekly', 'D': 'daily'}
        self.frequency = freq_to_human.get(frequency, frequency)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        # Map human-readable frequency to pandas freq code
        freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS'}
        pandas_freq = freq_map.get(self.frequency, 'MS')

        # MODE 1: Simple mode - just periods (and optionally start date)
        # Covariates are filled with historical mean - DO NOT accept covariate values in this mode
        # Use this for quick forecasts without specific event data
        if 'periods' in model_input.columns:
            periods = int(model_input['periods'].iloc[0])

            # Check if a start date ('ds') is provided
            if 'ds' in model_input.columns:
                start_date = pd.to_datetime(model_input['ds'].iloc[0]).normalize()
                future_dates = pd.date_range(start=start_date, periods=periods, freq=pandas_freq)
                future = pd.DataFrame({'ds': future_dates})
                future['ds'] = pd.to_datetime(future['ds']).dt.normalize()
                logger.info(f"🔮 Simple mode: Generating {periods} periods starting from {start_date} with frequency {self.frequency} ({pandas_freq})")
            else:
                future = self.model.make_future_dataframe(periods=periods, freq=pandas_freq, include_history=False)
                last_training_date = self.model.history['ds'].max()
                future = future[future['ds'] > last_training_date].copy()

            # In simple mode, ALWAYS use historical mean for covariates
            # This prevents incorrect usage where a single covariate value is applied to all periods
            if self.covariates:
                history = self.model.history
                for cov in self.covariates:
                    if cov in history.columns:
                        future[cov] = history[cov].tail(12).mean()
                        logger.info(f"   Covariate '{cov}': using historical mean = {future[cov].iloc[0]:.4f}")
            
            # Store our generated dates before Prophet predict (Prophet might modify them)
            original_dates = future['ds'].copy()
            
            forecast = self.model.predict(future)
            
            # Always use our generated dates, not Prophet's (Prophet might add timestamps or modify dates)
            if 'ds' in model_input.columns:
                # Replace forecast dates with our clean generated dates
                forecast['ds'] = original_dates.values
                logger.info(f"Using provided start date: {original_dates.iloc[0]} to {original_dates.iloc[-1]}")
            else:
                # Use our generated dates from make_future_dataframe
                forecast['ds'] = future['ds'].values
        else:
            # Mode 2: Specific dates provided
            df = model_input.copy()
            if self.time_col in df.columns and self.time_col != 'ds':
                df['ds'] = pd.to_datetime(df[self.time_col])
            elif 'ds' not in df.columns:
                raise ValueError("Either 'ds' or 'periods' must be provided")
            
            # Normalize input dates
            df['ds'] = pd.to_datetime(df['ds']).dt.normalize()
            
            if self.covariates:
                history = self.model.history
                for cov in self.covariates:
                    if cov not in df.columns and cov in history.columns:
                        df[cov] = history[cov].tail(12).mean()
            forecast = self.model.predict(df)
            # Use original input dates, not Prophet's modified dates
            forecast['ds'] = df['ds'].values
        
        # Return clean dates (ensure normalized, no time component)
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        result['ds'] = pd.to_datetime(result['ds']).dt.normalize()
        return result

def prepare_prophet_data(data: List[Dict[str, Any]], time_col: str, target_col: str, covariates: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(data)
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = pd.to_datetime(df[time_col])
    
    # Handle target column - it may not exist in future features data
    if target_col in df.columns:
        prophet_df['y'] = pd.to_numeric(df[target_col], errors='coerce')
    else:
        # For future features without target, set y to NaN
        prophet_df['y'] = np.nan
    
    for cov in covariates:
        if cov in df.columns:
            prophet_df[cov] = pd.to_numeric(df[cov], errors='coerce')
        else:
            # Initialize missing covariates with NaN so they can be filled during merge/update
            prophet_df[cov] = np.nan
    
    # Do not drop NaNs here, as they may contain future covariates
    return prophet_df.sort_values('ds').reset_index(drop=True)

def create_future_dataframe(model: Prophet, periods: int, freq: str, covariates: List[str], historical_data: pd.DataFrame, regressor_method: str = 'mean') -> pd.DataFrame:
    """
    Create future dataframe for forecasting, prioritizing provided future covariate values.
    
    Args:
        model: Fitted Prophet model
        periods: Number of periods to forecast
        freq: Frequency string (e.g., 'MS', 'W', 'D')
        covariates: List of covariate column names
        historical_data: DataFrame containing historical data AND future covariate values (may have NaN for 'y' in future rows)
        regressor_method: Method to fill missing future covariate values ('mean', 'last_value', 'linear_trend')
    
    Returns:
        DataFrame with 'ds' column and covariate columns for future periods
    """
    # Get the last date from training data
    last_training_date = model.history['ds'].max()
    
    # Find future dates with provided covariate values in historical_data
    # IMPORTANT: Only look at rows AFTER last_training_date to avoid including training data
    # And EXCLUDE the 'y' column entirely to prevent target leakage
    provided_future_data = historical_data[historical_data['ds'] > last_training_date].copy()
    
    # Drop 'y' column if it exists to prevent target leakage
    if 'y' in provided_future_data.columns:
        provided_future_data = provided_future_data.drop(columns=['y'])
        logger.info(f"🔒 Dropped 'y' column from future data to prevent target leakage")
    
    # Log what we found
    if len(provided_future_data) > 0:
        logger.info(f"Checking for provided promotions in original data:")
        logger.info(f"   Last training date: {last_training_date}")
        logger.info(f"   Found {len(provided_future_data)} rows with dates after training")
        logger.info(f"   Date range in provided future data: {provided_future_data['ds'].min()} to {provided_future_data['ds'].max()}")
        for cov in covariates:
            if cov in provided_future_data.columns:
                # Check for non-zero values (promotions are often 0/1)
                non_zero_count = (provided_future_data[cov] != 0).sum() if provided_future_data[cov].dtype in ['int64', 'float64'] else provided_future_data[cov].notna().sum()
                if non_zero_count > 0:
                    logger.info(f"   ✓ Found {non_zero_count} non-zero values for '{cov}' in future dates")
                    # Show which dates have promotions
                    promo_rows = provided_future_data[provided_future_data[cov] != 0] if provided_future_data[cov].dtype in ['int64', 'float64'] else provided_future_data[provided_future_data[cov].notna()]
                    promo_dates = promo_rows['ds'].tolist()
                    promo_values = promo_rows[cov].tolist()
                    logger.info(f"      Dates with promotions: {list(zip(promo_dates[:5], promo_values[:5]))}{'...' if len(promo_dates) > 5 else ''}")
                else:
                    logger.warning(f"   ⚠ No non-zero values found for '{cov}' in provided future data (all zeros)")
    
    # Generate standard future dates using Prophet's method
    future = model.make_future_dataframe(periods=periods, freq=freq)
    # Keep only future dates (after last training date)
    future = future[future['ds'] > last_training_date].copy()
    
    # If we have provided future dates with covariate values, use them
    if len(provided_future_data) > 0:
        logger.info(f"📅 Found {len(provided_future_data)} future dates with provided covariate data")
        logger.info(f"   Forecast horizon: {len(future)} periods from {future['ds'].min()} to {future['ds'].max()}")
        
        # Use string format YYYY-MM-DD for robust matching (avoids timezone/time issues)
        provided_future_data['ds_str'] = pd.to_datetime(provided_future_data['ds']).dt.strftime('%Y-%m-%d')
        future['ds_str'] = pd.to_datetime(future['ds']).dt.strftime('%Y-%m-%d')
        
        # For each covariate, map provided values to forecast dates
        for cov in covariates:
            if cov in provided_future_data.columns:
                # Create a mapping from provided future dates to covariate values
                # Include ALL values (including 0) to preserve actual feature values
                provided_map = {}
                for idx, row in provided_future_data.iterrows():
                    provided_map[row['ds_str']] = row[cov]
                
                if len(provided_map) > 0:
                    # DEBUG: Log what we're trying to map
                    logger.info(f"   Mapping '{cov}':")
                    logger.info(f"      Provided map has {len(provided_map)} dates")
                    logger.info(f"      Sample provided dates: {list(provided_map.keys())[:5]}")
                    logger.info(f"      Future dataframe has {len(future)} dates")
                    logger.info(f"      Sample future dates: {future['ds_str'].head(5).tolist()}")
                    
                    # Check for Black Friday specifically
                    if cov == 'Black Friday':
                        if '2025-11-27' in provided_map:
                            logger.info(f"      ✓ '2025-11-27' IS in provided_map with value: {provided_map['2025-11-27']}")
                        else:
                            logger.warning(f"      ✗ '2025-11-27' NOT in provided_map")
                            logger.warning(f"         All provided dates: {sorted(provided_map.keys())}")
                        
                        if '2025-11-27' in future['ds_str'].values:
                            logger.info(f"      ✓ '2025-11-27' IS in future dataframe")
                        else:
                            logger.warning(f"      ✗ '2025-11-27' NOT in future dataframe")
                            logger.warning(f"         Future date range: {future['ds_str'].min()} to {future['ds_str'].max()}")
                    
                    # Map provided values to future dataframe using string dates
                    future[cov] = future['ds_str'].map(provided_map)
                    
                    # Fill NaN with 0 only if we found some matches, otherwise warn
                    # But wait, if we map, unmatched becomes NaN. 
                    # If the user provided data for date X, we use it. If not, it stays NaN and gets filled by regressor_method later.
                    
                    provided_count = future[cov].notna().sum()
                    if provided_count > 0:
                        # Check for non-zero values
                        non_zero_count = (future[cov] != 0).sum() if future[cov].dtype in ['int64', 'float64'] else provided_count
                        logger.info(f"   ✓ Using {provided_count}/{len(future)} provided future values for covariate '{cov}' ({non_zero_count} non-zero)")
                        
                        # Debug Black Friday specifically if needed
                        if cov == 'Black Friday':
                            bf_rows = future[future['ds_str'] == '2025-11-27']
                            if len(bf_rows) > 0:
                                val = bf_rows.iloc[0][cov]
                                logger.info(f"      Black Friday (2025-11-27) value: {val}")
                    else:
                        logger.warning(f"   ⚠ No matching dates found for covariate '{cov}'")
                else:
                    logger.warning(f"   ⚠ No valid values found for covariate '{cov}' in provided future data")
        
        # Clean up temporary column
        future = future.drop(columns=['ds_str'])
        provided_future_data = provided_future_data.drop(columns=['ds_str'])
    
    # Fill missing covariate values using regressor method
    # IMPORTANT: Only fill where we DON'T have provided values (preserve provided values including 0)
    for cov in covariates:
        if cov not in future.columns:
            future[cov] = None
        
        # Calculate fill value from historical data (only rows with actual 'y' values)
        historical_only = historical_data.dropna(subset=['y'])
        if cov in historical_only.columns:
            # Count how many values we already have from provided data (including 0!)
            # We need to track which dates were actually mapped vs which are NaN
            provided_mask = future[cov].notna()
            provided_count = provided_mask.sum()
            
            if provided_count < len(future):
                # We need to fill some missing values (only where we have NaN, not where value is 0)
                missing_mask = future[cov].isna()
                
                if regressor_method == 'last_value':
                    fill_value = historical_only[cov].dropna().iloc[-1] if len(historical_only[cov].dropna()) > 0 else 0
                    future.loc[missing_mask, cov] = fill_value
                elif regressor_method == 'linear_trend':
                    valid_hist = historical_only.dropna(subset=[cov])
                    if len(valid_hist) > 1:
                        from sklearn.linear_model import LinearRegression
                        x = np.arange(len(valid_hist)).reshape(-1, 1)
                        y = valid_hist[cov].values
                        lr = LinearRegression().fit(x, y)
                        # Predict for missing future points using linear trend
                        # Only predict for the missing indices
                        missing_indices = future[missing_mask].index
                        future_dates_idx = np.arange(len(historical_only), len(historical_only) + len(missing_indices))
                        predicted = lr.predict(future_dates_idx.reshape(-1, 1))
                        future.loc[missing_mask, cov] = predicted
                    else:
                        fill_value = historical_only[cov].dropna().mean() if len(historical_only[cov].dropna()) > 0 else 0
                        future.loc[missing_mask, cov] = fill_value
                else:  # 'mean' or default
                    fill_value = historical_only[cov].dropna().tail(12).mean() if len(historical_only[cov].dropna()) > 0 else 0
                    future.loc[missing_mask, cov] = fill_value
                
                logger.info(f"   Filled {missing_mask.sum()} missing values for '{cov}' using {regressor_method} method (fill_value={fill_value})")
            else:
                logger.info(f"   All {len(future)} values for '{cov}' were provided, no filling needed")
                # Log sample of provided values to verify
                sample_values = future[cov].head(10).tolist()
                logger.info(f"      Sample values: {sample_values}")
    
    return future

def generate_prophet_training_code(
    time_col: str, target_col: str, covariates: List[str], horizon: int,
    frequency: str, best_params: Dict[str, Any], seasonality_mode: str,
    regressor_method: str, country: str, train_size: int, test_size: int,
    random_seed: int = 42, run_id: str = None, original_covariates: List[str] = None
) -> str:
    """Generate reproducible Python code for Prophet model training including preprocessing"""
    freq_code = {"weekly": "W", "monthly": "MS", "daily": "D"}.get(frequency, "MS")
    covariate_str = ", ".join([f"'{c}'" for c in covariates]) if covariates else ""
    original_cov_str = ", ".join([f"'{c}'" for c in (original_covariates or [])]) if original_covariates else ""

    code = f'''"""
Reproducible Prophet Model Training Code
Generated for reproducibility
Run ID: {run_id}

This code includes preprocessing steps that enhance features for better
holiday/weekend forecasting accuracy.
"""
import pandas as pd
import numpy as np
import os
import mlflow
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Set random seed for reproducibility
np.random.seed({random_seed})
import random
random.seed({random_seed})

# ============================================================================
# PREPROCESSING FUNCTIONS (for reproducibility)
# ============================================================================
def enhance_features_for_forecasting(df, date_col='ds', target_col='y', promo_cols=None, frequency='daily'):
    """
    Add generic features that improve forecasting for all algorithms.

    Features added:
    1. Calendar features (always): day_of_week, is_weekend, month, quarter
    2. Trend features (always): time_index, year
    3. YoY lag features (conditional): only if enough data exists
    """
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values(date_col).reset_index(drop=True)

    dates = pd.to_datetime(result[date_col])

    # Calendar features (always added)
    result['day_of_week'] = dates.dt.dayofweek
    result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
    result['month'] = dates.dt.month
    result['quarter'] = dates.dt.quarter
    result['day_of_month'] = dates.dt.day
    result['week_of_year'] = dates.dt.isocalendar().week.astype(int)

    # Trend features
    result['time_index'] = range(len(result))
    result['year'] = dates.dt.year

    # YoY lag features (only if enough data)
    lag_config = {{'daily': {{'lag': 364, 'min_rows': 400}}, 'weekly': {{'lag': 52, 'min_rows': 60}}, 'monthly': {{'lag': 12, 'min_rows': 15}}}}
    config = lag_config.get(frequency, lag_config['daily'])

    if target_col in result.columns and result[target_col].notna().sum() >= config['min_rows']:
        lag_col = f"lag_{{config['lag']}}"
        result[lag_col] = result[target_col].shift(config['lag'])
        if result[lag_col].notna().any():
            window = 7 if frequency == 'daily' else 4 if frequency == 'weekly' else 3
            result[f'{{lag_col}}_avg'] = result[target_col].shift(config['lag']).rolling(window=window, min_periods=1).mean()

    return result

# ============================================================================
# DATA PREPARATION
# ============================================================================
run_id = "{run_id}"
print(f"Downloading training data from Run ID: {{run_id}}...")

try:
    # Download the full merged data used for training (already preprocessed)
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="datasets/processed/full_merged_data.csv")
    df = pd.read_csv(local_path)
    print(f"Loaded preprocessed data from {{local_path}}")
except Exception as e:
    print(f"Could not download from MLflow: {{e}}")
    print("Please ensure you have 'datasets/processed/full_merged_data.csv' locally.")
    if os.path.exists("full_merged_data.csv"):
        df = pd.read_csv("full_merged_data.csv")
    else:
        raise FileNotFoundError("Could not find training data.")

# Prepare Prophet format
prophet_df = pd.DataFrame()
prophet_df['ds'] = pd.to_datetime(df['ds'])
prophet_df['y'] = pd.to_numeric(df['y'], errors='coerce')
'''
    
    if covariates:
        code += f'''
# Add covariates
'''
        for cov in covariates:
            code += f'''prophet_df['{cov}'] = pd.to_numeric(df['{cov}'], errors='coerce')
'''
    
    code += f'''
# Split into train/test (train_size={train_size}, test_size={test_size})
history_df = prophet_df.dropna(subset=['y']).copy()
test_size = {test_size}
train_df = history_df.iloc[:-test_size].copy()
test_df = history_df.iloc[-test_size:].copy()

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
model = Prophet(
    seasonality_mode='{best_params.get("seasonality_mode", seasonality_mode)}',
    yearly_seasonality={best_params.get("yearly_seasonality", True)},
    weekly_seasonality={best_params.get("weekly_seasonality", False)},
    daily_seasonality=False,
    changepoint_prior_scale={best_params.get("changepoint_prior_scale", 0.05)},
    seasonality_prior_scale={best_params.get("seasonality_prior_scale", 10.0)},
    growth='{best_params.get("growth", "linear")}',
    interval_width=0.95,
    uncertainty_samples=1000
)

# Add country holidays
try:
    model.add_country_holidays(country_name='{country}')
except:
    pass

# Add regressors (covariates)
'''
    
    if covariates:
        for cov in covariates:
            code += f'''model.add_regressor('{cov}')
'''
    else:
        code += '''# No covariates used
'''
    
    code += f'''
# ============================================================================
# MODEL TRAINING FLOW
# ============================================================================
model.fit(train_df)

# ============================================================================
# VALIDATION (on test set)
# ============================================================================
test_future_cols = ['ds']'''
    
    if covariates:
        code += " + ["
        code += ", ".join([f"'{c}'" for c in covariates])
        code += "]"
    code += '''
test_future = test_df[test_future_cols].copy()
test_forecast = model.predict(test_future)

# Calculate metrics
y_true = test_df['y'].values
y_pred = test_forecast['yhat'].values
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = mean_absolute_percentage_error(y_true, y_pred) * 100
r2 = r2_score(y_true, y_pred)

print(f"Validation Metrics:")
print(f"  RMSE: {{rmse:.2f}}")
print(f"  MAPE: {{mape:.2f}}%")
print(f"  R²: {{r2:.4f}}")

# ============================================================================
# FORECASTING (future periods)
# ============================================================================
# Option 1: Load prepared future dataframe from artifacts
# if os.path.exists("datasets/inference/input.csv"):
#     future = pd.read_csv("datasets/inference/input.csv")
#     future['ds'] = pd.to_datetime(future['ds'])

# Option 2: Generate from scratch
# Create future dataframe for {horizon} periods
future = model.make_future_dataframe(periods={horizon}, freq='{freq_code}')

# Add future covariate values (if provided)
# If you have future covariate values, add them here:
'''
    
    if covariates:
        code += f'''# Example: future['{covariates[0]}'] = [your_future_values]
# Or use regressor_method='{regressor_method}' to estimate:
'''
        if regressor_method == 'mean':
            code += f'''# future['{covariates[0]}'] = history_df['{covariates[0]}'].tail(12).mean()
'''
        elif regressor_method == 'last_value':
            code += f'''# future['{covariates[0]}'] = history_df['{covariates[0]}'].iloc[-1]
'''
        else:
            code += f'''# future['{covariates[0]}'] = [estimated_values]
'''
    else:
        code += '''# No covariates to add
'''
    
    code += f'''
# Generate forecast
forecast = model.predict(future)

# Extract future forecast (after last training date)
last_training_date = train_df['ds'].max()
future_forecast = forecast[forecast['ds'] > last_training_date][
    ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
].copy()

print(f"\\nForecast for {{len(future_forecast)}} periods:")
print(future_forecast.head())

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
with mlflow.start_run(run_name="Prophet_Model") as run:
    # Log parameters
    mlflow.log_param("model_type", "Prophet")
    mlflow.log_param("random_seed", {random_seed})
    mlflow.log_param("horizon", {horizon})
    mlflow.log_param("frequency", "{frequency}")
    mlflow.log_params(best_params)
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("r2", r2)
    
    # Create model wrapper class
    class ProphetModelWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, model, time_col, target_col, covariates, freq):
            self.model = model
            self.time_col = time_col
            self.target_col = target_col
            self.covariates = covariates
            self.freq = freq
        
        def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
            import pandas as pd
            if 'periods' in model_input.columns:
                periods = model_input['periods'].iloc[0]
                future = self.model.make_future_dataframe(periods=periods, freq=self.freq)
                if self.covariates:
                    for cov in self.covariates:
                        if cov in model_input.columns:
                            future[cov] = model_input[cov].iloc[0]
            else:
                future = model_input.copy()
                if self.time_col in future.columns and self.time_col != 'ds':
                    future = future.rename(columns={{self.time_col: 'ds'}})
            forecast = self.model.predict(future)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    # Create input example
    input_example_data = {{'ds': [train_df['ds'].iloc[-1]], 'periods': [{horizon}]}}
'''
    
    if covariates:
        code += f'''    input_example_data['{covariates[0]}'] = [train_df['{covariates[0]}'].iloc[-1]]
'''
    
    code += f'''    input_example = pd.DataFrame(input_example_data)
    
    # Create model wrapper
    model_wrapper = ProphetModelWrapper(model, '{time_col}', '{target_col}', {covariates}, '{freq_code}')
    
    # Infer signature
    sample_output = model_wrapper.predict(None, input_example)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    signature = infer_signature(input_example, sample_output)
    
    # Log model to MLflow
    mlflow.pyfunc.log_model(
        name="model",
        python_model=model_wrapper,
        signature=signature,
        input_example=input_example,
        code_paths=["backend"] if os.path.exists("backend") else [],
        metadata={{
            "description": "Prophet forecasting model",
            "covariates": "{str(covariates)}",
            "frequency": "{frequency}"
        }},
        conda_env={{
            "dependencies": [
                f"python={{sys.version_info.major}}.{{sys.version_info.minor}}.{{sys.version_info.micro}}",
                "pip",
                {{"pip": ["mlflow", "pandas", "numpy", "prophet", "holidays"]}}
            ]
        }}
    )
    
    print("\\nModel logged to MLflow successfully!")
    print(f"   Run ID: {{run.info.run_id}}")
    print(f"   Model URI: runs:/{{run.info.run_id}}/model")

# ============================================================================
# NOTES
# ============================================================================
# - Training data: {{len(train_df)}} rows
# - Test data: {{len(test_df)}} rows  
# - Forecast horizon: {horizon} periods
# - Frequency: {frequency} ({freq_code})
# - Best hyperparameters found via grid search:
#   * changepoint_prior_scale: {best_params.get("changepoint_prior_scale", "N/A")}
#   * seasonality_prior_scale: {best_params.get("seasonality_prior_scale", "N/A")}
#   * seasonality_mode: {best_params.get("seasonality_mode", seasonality_mode)}
#   * yearly_seasonality: {best_params.get("yearly_seasonality", True)}
#   * weekly_seasonality: {best_params.get("weekly_seasonality", False)}
'''
    
    return code


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = 0.0
    return {"rmse": round(rmse, 2), "mape": round(mape, 2), "r2": round(r2_score(y_true, y_pred), 4)}

def evaluate_param_set(params, country, covariates, train_df, test_df, time_col, target_col, experiment_id, parent_run_id):
    client = MlflowClient()
    run = client.create_run(experiment_id=experiment_id, tags={"mlflow.parentRunId": parent_run_id, "mlflow.runName": f"Prophet_{params}"})
    run_id = run.info.run_id
    
    try:
        for k, v in params.items(): client.log_param(run_id, k, str(v))
        
        model = Prophet(seasonality_mode=params['seasonality_mode'], yearly_seasonality=params['yearly_seasonality'],
                        weekly_seasonality=params['weekly_seasonality'], daily_seasonality=False,
                        changepoint_prior_scale=params['changepoint_prior_scale'], seasonality_prior_scale=params['seasonality_prior_scale'],
                        growth=params.get('growth', 'linear'), interval_width=0.95, uncertainty_samples=1000)
        try: model.add_country_holidays(country_name=country)
        except: pass
        
        # Track which regressors are actually added to ensure consistency between train and test
        added_regressors = []
        for cov in covariates:
            if cov in train_df.columns and cov in test_df.columns:
                # Only add regressor if it has non-NaN values in BOTH train and test
                if train_df[cov].notna().any() and test_df[cov].notna().any():
                    model.add_regressor(cov)
                    added_regressors.append(cov)
                else:
                    logger.warning(f"Skipping regressor '{cov}' - has NaN values in train or test data")

        model.fit(train_df)

        # Use only the regressors that were actually added to the model
        test_future = test_df[['ds'] + added_regressors].copy()
        metrics = compute_metrics(test_df['y'].values, model.predict(test_future)['yhat'].values)
        
        for k, v in metrics.items(): client.log_metric(run_id, k, v)
        client.set_terminated(run_id)
        
        return {"params": params, "metrics": metrics, "run_id": run_id, "model": model, "status": "SUCCESS"}
    except Exception as e:
        logger.error(f"Run {run_id} failed: {e}")
        client.set_terminated(run_id, status="FAILED")
        return {"params": params, "metrics": None, "run_id": run_id, "status": "FAILED", "error": str(e)}

def train_prophet_model(data, time_col, target_col, covariates, horizon, frequency, seasonality_mode="multiplicative", test_size=None, regressor_method='mean', country='US', random_seed=42, future_features=None):
    # Set global random seeds for reproducibility
    np.random.seed(random_seed)
    import random
    import copy
    random.seed(random_seed)
    logger.info(f"Set random seed to {random_seed} for reproducibility")
    
    freq_code = {"weekly": "W", "monthly": "MS", "daily": "D"}.get(frequency, "MS")
    # Store original data for logging (deep copy to ensure it's pristine)
    original_data = copy.deepcopy(data)
    # CRITICAL: Ensure target_col is NOT in covariates to prevent leakage
    # If the user selected the target as a feature, it would leak actuals into the future dataframe
    if target_col in covariates:
        logger.warning(f"🚨 Target column '{target_col}' found in covariates! Removing it to prevent leakage.")
        covariates = [c for c in covariates if c != target_col]
    
    df = prepare_prophet_data(data, time_col, target_col, covariates)
    
    # If future_features are provided, merge them with the dataframe
    if future_features:
        logger.info(f"🔮 Received {len(future_features)} future feature rows from frontend")
        future_df = prepare_prophet_data(future_features, time_col, target_col, covariates)
        
        # CRITICAL: Drop target 'y' from future features to ensure they are treated as future, not history
        # This prevents target leakage and ensures Prophet doesn't use them for training
        if 'y' in future_df.columns:
            future_df = future_df.drop(columns=['y'])
        logger.info("🔒 Dropped 'y' from future features to prevent leakage")
        
        # Merge future features with existing df using update logic
        # This ensures we keep 'y' from history but update covariates from future_features
        
        # Set ds as index for alignment
        df = df.set_index('ds')
        future_df = future_df.set_index('ds')
        
        # Initialize missing covariates in history with 0 before update
        # This handles the case where the user uploads "future" promotions that start LATER than history
        # and history itself didn't have these columns originally.
        for cov in covariates:
            if cov in df.columns:
                # If column exists (was added by prepare_prophet_data as NaN), fill NaNs with 0
                df[cov] = df[cov].fillna(0)
        
        # 1. Update existing history rows with covariate data from future_features
        # This allows users to upload a separate "promotions" file that covers history too
        df.update(future_df)
        
        # 2. Identify rows in future_df that are NOT in df (true future, or new history dates)
        # Note: df.index is DatetimeIndex, future_df.index is DatetimeIndex
        new_rows = future_df[~future_df.index.isin(df.index)]
        
        # 3. Concatenate
        df = pd.concat([df, new_rows])
        
        # Reset index
        df = df.reset_index().sort_values('ds').reset_index(drop=True)
        
        logger.info(f"Merged dataframe now has {len(df)} rows. Updated historical covariates and added {len(new_rows)} future rows.")

    # Apply preprocessing to enhance features for better holiday/weekend forecasting
    # This adds YoY lag features, promo-derived features, and enhanced calendar features
    original_covariates = covariates.copy() if covariates else []
    df = enhance_features_for_forecasting(
        df=df,
        date_col='ds',
        target_col='y',
        promo_cols=covariates,
        frequency=frequency
    )

    # Add derived feature columns to covariates list for Prophet to use as regressors
    derived_cols = get_derived_feature_columns(covariates)
    # Only add derived columns that exist in the dataframe and are numeric
    for col in derived_cols:
        if col in df.columns and col not in covariates:
            # Skip non-numeric columns and columns with all NaN
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32'] and df[col].notna().any():
                covariates.append(col)

    logger.info(f"Preprocessing added {len(covariates) - len(original_covariates)} derived features: {[c for c in covariates if c not in original_covariates]}")

    # Separate history (for training) and future (for covariates)
    # Keep ALL data in df (including future dates with covariate values but no 'y')
    history_df = df.dropna(subset=['y']).copy()
    
    # Check if we have future dates with provided covariate values
    if len(history_df) > 0:
        last_hist_date = history_df['ds'].max()
        future_rows = df[(df['y'].isna()) & (df['ds'] > last_hist_date)]
    else:
        future_rows = df[df['y'].isna()]
    
    if len(future_rows) > 0:
        future_dates_with_covs = []
        for cov in covariates:
            if cov in future_rows.columns:
                provided_count = future_rows[cov].notna().sum()
                if provided_count > 0:
                    future_dates_with_covs.append(f"{cov} ({provided_count} dates)")
        if future_dates_with_covs:
            logger.info(f"📅 Found future covariate values provided: {', '.join(future_dates_with_covs)}")
            logger.info(f"   These will be used for forecasting instead of estimating with '{regressor_method}' method")
    
    if test_size is None: test_size = min(horizon, len(history_df) // 5)
    train_df, test_df = history_df.iloc[:-test_size].copy(), history_df.iloc[-test_size:].copy()

    # Filter out covariates that have all NaN values in training data
    # This can happen with YoY lag features when there's < 1 year of data
    valid_covariates = []
    dropped_covariates = []
    for cov in covariates:
        if cov in train_df.columns:
            if train_df[cov].notna().any():
                valid_covariates.append(cov)
            else:
                dropped_covariates.append(cov)
        else:
            dropped_covariates.append(cov)

    if dropped_covariates:
        logger.warning(f"Dropped {len(dropped_covariates)} covariates with all-NaN values in training data: {dropped_covariates}")
    covariates = valid_covariates
    logger.info(f"Using {len(covariates)} valid covariates for Prophet: {covariates}")

    param_grid = {
        'changepoint_prior_scale': [0.001, 0.05, 0.5],
        'seasonality_prior_scale': [0.01, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'yearly_seasonality': [True, False],
        'weekly_seasonality': [True, False] if frequency in ['weekly', 'daily'] else [False],
        'growth': ['linear']
    }
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    # Deduplicate
    unique_params = []
    seen = set()
    for p in all_params:
        t = tuple(sorted((k, str(v)) for k, v in p.items()))
        if t not in seen: seen.add(t); unique_params.append(p)
    all_params = unique_params
    
    # Limit hyperparameter combinations for Databricks Apps to avoid timeouts
    max_combinations = int(os.environ.get('PROPHET_MAX_COMBINATIONS', '3'))  # Default to 3 for Databricks Apps
    if len(all_params) > max_combinations:
        import random
        random.seed(random_seed)  # Set seed for reproducible sampling
        all_params = random.sample(all_params, max_combinations)
        logger.info(f"Limited Prophet hyperparameter combinations to {max_combinations} (from {len(unique_params)} total) using seed {random_seed}")
    
    best_metrics = {'mape': float('inf'), 'rmse': float('inf')}
    best_params, best_model, best_run_id, best_artifact_uri = None, None, None, None
    
    with mlflow.start_run(run_name="Prophet_Tuning", nested=True) as parent_run:
        parent_run_id = parent_run.info.run_id
        experiment_id = parent_run.info.experiment_id
        
        # Log datasets to organized folders
        try:
            # Log original time series data
            original_df = pd.DataFrame(original_data)
            original_df.to_csv("/tmp/original_timeseries_data.csv", index=False)
            mlflow.log_artifact("/tmp/original_timeseries_data.csv", "datasets/raw")
            logger.info(f"Logged original time series data to datasets/raw/: {len(original_df)} rows with columns: {list(original_df.columns)}")

            # Log promotions dataset (future features) if available
            if future_features:
                try:
                    promotions_df = pd.DataFrame(future_features)
                    promotions_df.to_csv("/tmp/promotions_future_features.csv", index=False)
                    mlflow.log_artifact("/tmp/promotions_future_features.csv", "datasets/raw")
                    logger.info(f"Logged promotions data to datasets/raw/: {len(promotions_df)} rows with columns: {list(promotions_df.columns)}")
                except Exception as e:
                    logger.warning(f"Failed to log promotions data: {e}")
            
            # Check for promotions in original data
            if covariates:
                try:
                    original_data_df = pd.DataFrame(original_data)
                    for cov in covariates:
                        if cov in original_data_df.columns:
                            # Check both historical and future dates
                            if original_data_df[cov].dtype in ['int64', 'float64']:
                                cov_nonzero = (original_data_df[cov] != 0).sum()
                            else:
                                cov_nonzero = original_data_df[cov].notna().sum()

                            if cov_nonzero > 0:
                                logger.info(f"   ✓ Found {cov_nonzero} non-zero/non-null values for '{cov}' in original data")
                                # Show sample of dates with promotions
                                if original_data_df[cov].dtype in ['int64', 'float64']:
                                    promo_dates = original_data_df[original_data_df[cov] != 0][time_col].tolist()
                                else:
                                    promo_dates = original_data_df[original_data_df[cov].notna()][time_col].tolist()
                                
                                sample_dates = promo_dates[:5]
                                suffix = "..." if len(promo_dates) > 5 else ""
                                logger.info(f"      Sample dates with promotions: {sample_dates}{suffix}")
                except Exception as e:
                    logger.warning(f"Could not check promotions in original data: {e}")
            
            # Log Prophet's transformed data (after preprocessing)
            df.to_csv("/tmp/full_merged_data.csv", index=False)
            mlflow.log_artifact("/tmp/full_merged_data.csv", "datasets/processed")
            logger.info(f"Logged full merged data to datasets/processed/: {len(df)} rows with columns: {list(df.columns)}")

            # Log training and evaluation datasets
            train_df.to_csv("/tmp/train.csv", index=False)
            mlflow.log_artifact("/tmp/train.csv", "datasets/training")
            logger.info(f"Logged training data to datasets/training/: {len(train_df)} rows")

            test_df.to_csv("/tmp/eval.csv", index=False)
            mlflow.log_artifact("/tmp/eval.csv", "datasets/training")
            logger.info(f"Logged evaluation data to datasets/training/: {len(test_df)} rows")

            # Log which rows have future dates (y is NaN but covariates may be present)
            future_rows = df[df['y'].isna()].copy()
            if len(future_rows) > 0:
                logger.info(f"   Found {len(future_rows)} rows with future dates (y=NaN) that may contain promotion/covariate values")
                for cov in covariates:
                    if cov in future_rows.columns:
                        # Check for non-zero values (promotions are often 0/1)
                        non_zero_count = (future_rows[cov] != 0).sum() if future_rows[cov].dtype in ['int64', 'float64'] else future_rows[cov].notna().sum()
                        if non_zero_count > 0:
                            logger.info(f"   ✓ Future promotions available for '{cov}': {non_zero_count} non-zero values")
                            # Show which dates have promotions
                            promo_dates = future_rows[future_rows[cov] != 0]['ds'].tolist() if future_rows[cov].dtype in ['int64', 'float64'] else future_rows[future_rows[cov].notna()]['ds'].tolist()
                            logger.info(f"      Sample dates: {promo_dates[:5]}{'...' if len(promo_dates) > 5 else ''}")
        except Exception as e:
            logger.warning(f"Could not log processed data: {e}")
        
        # Reduce parallelism for Databricks Apps to avoid timeouts
        max_workers = int(os.environ.get('MLFLOW_MAX_WORKERS', '1'))  # Default to 1 for Databricks Apps
        logger.info(f"Running Prophet hyperparameter tuning with {len(all_params)} combinations, {max_workers} parallel workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_param_set, p, country, covariates, train_df, test_df, time_col, target_col, experiment_id, parent_run_id) for p in all_params]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result["status"] == "SUCCESS":
                    metrics = result["metrics"]
                    if metrics['mape'] < best_metrics['mape'] or (abs(metrics['mape'] - best_metrics['mape']) < 0.1 and metrics['rmse'] < best_metrics['rmse']):
                        best_metrics, best_params, best_model, best_run_id = metrics, result["params"], result["model"], result["run_id"]

        # Time Series Cross-Validation using Prophet's built-in method
        logger.info(f"Running Prophet time series cross-validation...")
        try:
            from prophet.diagnostics import cross_validation, performance_metrics as prophet_performance_metrics

            # Calculate appropriate initial and horizon for CV
            # Initial: at least 50% of data, Horizon: forecast horizon, Period: horizon/2
            total_days = (history_df['ds'].max() - history_df['ds'].min()).days
            cv_initial = f"{max(total_days // 2, 30)} days"
            cv_horizon = f"{min(horizon * 7 if frequency == 'weekly' else horizon * 30 if frequency == 'monthly' else horizon, total_days // 4)} days"
            cv_period = f"{max(7, (horizon * 7 if frequency == 'weekly' else horizon * 30 if frequency == 'monthly' else horizon) // 2)} days"

            # Run cross-validation on best model
            cv_df = cross_validation(best_model, initial=cv_initial, period=cv_period, horizon=cv_horizon)
            cv_metrics = prophet_performance_metrics(cv_df)

            if not cv_metrics.empty and 'mape' in cv_metrics.columns:
                cv_mape = round(cv_metrics['mape'].mean() * 100, 2)
                cv_mape_std = round(cv_metrics['mape'].std() * 100, 2)
                logger.info(f"Prophet CV Results: Mean MAPE={cv_mape:.2f}% (±{cv_mape_std:.2f}%)")
                best_metrics["cv_mape"] = cv_mape
                best_metrics["cv_mape_std"] = cv_mape_std
        except Exception as e:
            logger.warning(f"Prophet cross-validation failed: {e}. Using single holdout validation.")

        # Log best model to parent run (INSIDE the with block)
        import sys
        from mlflow.models.signature import infer_signature

        # Pass human-readable frequency ('daily', 'weekly', 'monthly') for consistency
        model_wrapper = ProphetModelWrapper(best_model, time_col, target_col, covariates, frequency)

        # Create input example - use ADVANCED mode (full dataframe with dates and covariates)
        # This is the recommended way to call Prophet when you have covariates/events
        #
        # Prophet supports two modes:
        # Mode 1 (Simple): {"ds": "2025-01-01", "periods": 90}
        #    - Generates N future periods, covariates filled with historical mean
        #    - Use when you don't have specific event data for future dates
        #
        # Mode 2 (Advanced): Full dataframe with specific ds dates and covariate values
        #    - Provide exact dates with covariate values for each date
        #    - Use when you know future events (Black Friday, promotions, etc.)
        #
        # Input example shows Mode 2 (Advanced) format - recommended for models with covariates
        last_date = train_df['ds'].max()
        freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS'}
        pandas_freq = freq_map.get(frequency, 'MS')
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=pandas_freq)[1:]  # Skip first (last_date)

        input_example_data = {'ds': [str(d.date()) for d in future_dates[:3]]}  # First 3 dates as example
        if covariates:
            for cov in covariates:
                # Set example values (0 for most, 1 for one to show it's a flag)
                input_example_data[cov] = [0, 0, 0]
        input_example = pd.DataFrame(input_example_data)
        
        # Log a note about Mode 2 (specific future dates) in model description
        model_description = f"Prophet model trained on {len(train_df)} samples. "
        model_description += f"Supports 2 prediction modes: "
        model_description += f"(1) Simple: provide 'periods' for N-step forecast, "
        model_description += f"(2) Advanced: provide specific 'ds' dates with covariate values for each future date."
        
        signature = infer_signature(input_example, model_wrapper.predict(None, input_example)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        
        mlflow.pyfunc.log_model(
            name="model", python_model=model_wrapper, signature=signature, input_example=input_example,
            code_paths=["backend"], 
            metadata={"description": model_description, "covariates": str(covariates), "frequency": frequency},
            conda_env={"dependencies": [f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}", "pip", {"pip": ["mlflow", "pandas", "numpy", "prophet", "holidays"]}]}
        )
        mlflow.log_metrics(best_metrics)
        mlflow.log_params(best_params)
        mlflow.log_param("random_seed", random_seed)
        
        # Log reproducible training code (includes preprocessing functions)
        training_code = generate_prophet_training_code(
            time_col, target_col, covariates, horizon, frequency,
            best_params, seasonality_mode, regressor_method, country,
            len(train_df), len(test_df), random_seed, run_id=parent_run_id,
            original_covariates=original_covariates
        )
        mlflow.log_text(training_code, "reproducibility/training_code.py")

        # Also log the preprocessing module for complete reproducibility
        try:
            import inspect
            from backend import preprocessing
            preprocessing_code = inspect.getsource(preprocessing)
            mlflow.log_text(preprocessing_code, "reproducibility/preprocessing.py")
            logger.info("Logged preprocessing module for reproducibility")
        except Exception as e:
            logger.warning(f"Could not log preprocessing module: {e}")
        logger.info("Logged reproducible training code")
        
        best_artifact_uri = mlflow.get_artifact_uri("model")
    
    # Refit on full HISTORY data
    final_model = Prophet(seasonality_mode=best_params['seasonality_mode'], yearly_seasonality=True, weekly_seasonality=(frequency == "weekly"),
                          daily_seasonality=False, changepoint_prior_scale=best_params['changepoint_prior_scale'], seasonality_prior_scale=best_params['seasonality_prior_scale'])
    try: final_model.add_country_holidays(country_name=country)
    except: pass

    # Only add regressors that have valid (non-NaN) values in history
    final_regressors = []
    for cov in covariates:
        if cov in history_df.columns:
            if history_df[cov].notna().any():
                final_model.add_regressor(cov)
                final_regressors.append(cov)
            else:
                logger.warning(f"Final model: Skipping regressor '{cov}' - all NaN in history")
    logger.info(f"Final model using {len(final_regressors)} regressors: {final_regressors}")
    final_model.fit(history_df)

    # Pass FULL df (including future covariates) to create_future_dataframe
    # Use final_regressors (not covariates) to ensure consistency with model
    future = create_future_dataframe(final_model, horizon, freq_code, final_regressors, df, regressor_method)
    
    # 4. Log the future dataframe used for prediction to the parent run
    # Note: We log to the parent_run_id since the nested run has ended
    try:
        # DEBUG: Check what's actually in the future dataframe
        logger.info(f"DEBUG: Future dataframe before logging:")
        logger.info(f"   Shape: {future.shape}")
        logger.info(f"   Columns: {list(future.columns)}")
        logger.info(f"   Date range: {future['ds'].min()} to {future['ds'].max()}")
        
        # Check Black Friday specifically
        if 'Black Friday' in future.columns:
            bf_rows = future[future['Black Friday'] == 1]
            logger.info(f"   Black Friday rows with value=1: {len(bf_rows)}")
            if len(bf_rows) > 0:
                logger.info(f"   Black Friday dates: {bf_rows['ds'].tolist()}")
            else:
                logger.warning(f"   NO Black Friday=1 values found!")
                logger.warning(f"   Sample Black Friday values: {future['Black Friday'].head(10).tolist()}")
                # Check if there are any dates around Thanksgiving 2025
                nov_2025 = future[(future['ds'] >= '2025-11-01') & (future['ds'] <= '2025-11-30')]
                if len(nov_2025) > 0:
                    logger.info(f"   November 2025 dates in future: {nov_2025['ds'].tolist()}")
                    logger.info(f"   November 2025 Black Friday values: {nov_2025['Black Friday'].tolist()}")
        
        # Log inference input - use ADVANCED format (full dataframe with dates and covariates)
        # This matches the input_example logged with the model
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        # Primary input.csv - Advanced mode (recommended for models with covariates)
        future.to_csv("/tmp/input.csv", index=False)
        client.log_artifact(parent_run_id, "/tmp/input.csv", "datasets/inference")
        logger.info(f"Logged Prophet inference input (advanced format): {len(future)} rows with covariates: {list(future.columns)}")

        # Also log simple mode example for reference
        simple_input = pd.DataFrame({
            'ds': [str(train_df['ds'].max().date()) if hasattr(train_df['ds'].max(), 'date') else str(train_df['ds'].max())[:10]],
            'periods': [horizon]
        })
        simple_input.to_csv("/tmp/input_simple.csv", index=False)
        client.log_artifact(parent_run_id, "/tmp/input_simple.csv", "datasets/inference")
        logger.info(f"Logged simple mode example: ds={simple_input['ds'].iloc[0]}, periods={horizon}")
        
        # Log which promotions were used from provided data vs estimated
        if covariates:
            for cov in covariates:
                if cov in future.columns:
                    provided_count = future[cov].notna().sum()
                    if provided_count > 0:
                        provided_dates = future[future[cov].notna()]['ds'].tolist()
                        logger.info(f"   ✓ Using {provided_count} provided promotion values for '{cov}' on dates: {provided_dates[:3]}{'...' if len(provided_dates) > 3 else ''}")
                    else:
                        logger.info(f"   ⚠ No provided values for '{cov}', using {regressor_method} method")
    except Exception as e:
        logger.warning(f"Could not log future dataframe with covariates: {e}")
    
    forecast = final_model.predict(future)

    # Log forecast output to predictions folder (use MlflowClient to log to parent run)
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        future_pred_df = forecast[forecast['ds'] > train_df['ds'].max()].copy()
        future_pred_df.to_csv("/tmp/output.csv", index=False)
        client.log_artifact(parent_run_id, "/tmp/output.csv", "datasets/inference")
        logger.info(f"Logged forecast output to datasets/inference/: {len(future_pred_df)} rows")
    except Exception as e:
        logger.warning(f"Could not log forecast output: {e}")
    
    # Generate and log evaluation graph
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot training data
        ax.plot(train_df['ds'], train_df['y'], 'o', markersize=3, label='Training Data', color='#1f77b4')
        
        # Plot test data
        ax.plot(test_df['ds'], test_df['y'], 'o', markersize=3, label='Validation Data', color='#ff7f0e')
        
        # Plot forecast
        forecast_future_plot = forecast[forecast['ds'] > train_df['ds'].max()]
        ax.plot(forecast_future_plot['ds'], forecast_future_plot['yhat'], '-', label='Forecast', color='#2ca02c', linewidth=2)
        ax.fill_between(forecast_future_plot['ds'], forecast_future_plot['yhat_lower'], forecast_future_plot['yhat_upper'], alpha=0.2, color='#2ca02c', label='Confidence Interval')
        
        # Plot validation predictions
        test_future = test_df[['ds'] + [c for c in covariates if c in test_df.columns]].copy()
        validation_forecast = best_model.predict(test_future)
        ax.plot(validation_forecast['ds'], validation_forecast['yhat'], '-', label='Validation Predictions', color='#d62728', linewidth=2)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(target_col, fontsize=12)
        ax.set_title(f'Model Evaluation: Training, Validation & Forecast', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save and log (use MlflowClient to log to parent run)
        plt.savefig("/tmp/evaluation_graph.png", dpi=150, bbox_inches='tight')
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.log_artifact(parent_run_id, "/tmp/evaluation_graph.png", "plots")
        plt.close()
        logger.info("Logged evaluation graph as artifact")
    except Exception as e:
        logger.warning(f"Could not log evaluation graph: {e}")
    
    # Validation data
    test_future = test_df[['ds'] + [c for c in covariates if c in test_df.columns]].copy()
    validation_forecast = best_model.predict(test_future)
    val_cols = ['ds', 'y'] + [c for c in covariates if c in test_df.columns]
    validation_data = test_df[val_cols].merge(validation_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left').rename(columns={'ds': time_col})
    
    # Forecast data
    forecast_future = forecast[forecast['ds'] > train_df['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper'] + [c for c in covariates if c in forecast.columns]].copy().rename(columns={'ds': time_col})
    
    from backend.models_training import analyze_covariate_impact
    covariate_impacts = analyze_covariate_impact(final_model, df, covariates)
    
    return parent_run_id, f"runs:/{parent_run_id}/model", best_metrics, validation_data.where(pd.notnull(validation_data), None).to_dict(orient='records'), forecast_future.where(pd.notnull(forecast_future), None).to_dict(orient='records'), best_artifact_uri, covariate_impacts

def register_model_to_unity_catalog(model_uri: str, model_name: str, tags: Optional[Dict[str, str]] = None) -> str:
    """Register a model to Unity Catalog with improved error handling"""
    try:
        run_id = tags.get("run_id") if tags else (model_uri.split("/")[1] if model_uri.startswith("runs:/") and len(model_uri.split("/")) > 1 else None)
        client = MlflowClient()
        
        # Check if this run is already registered
        if run_id:
            try:
                logger.info(f"Checking if run {run_id} is already registered as {model_name}...")
                for v in client.search_model_versions(f"name='{model_name}'"):
                    if v.run_id == run_id:
                        logger.info(f"♻️  Model already registered as version {v.version}, skipping...")
                        return str(v.version)
            except Exception as check_error:
                logger.warning(f"Could not check existing versions: {check_error}")

        # Register the model
        logger.info(f"Registering model from {model_uri} to {model_name}...")
        result = None
        try:
            # First try with tags
            result = mlflow.register_model(model_uri=model_uri, name=model_name, tags=tags)
            logger.info(f"Successfully registered as version {result.version} with tags")
            
            # Verify tags were set
            if tags and result.version:
                try:
                    version_str = str(result.version)
                    mv = client.get_model_version(name=model_name, version=version_str)
                    actual_tags = mv.tags if hasattr(mv, 'tags') and mv.tags else {}
                    logger.info(f"   Verified tags on version {version_str}: {dict(actual_tags) if actual_tags else 'No tags found'}")
                    if not actual_tags:
                        logger.warning(f"    Tags were not set via register_model, trying client API...")
                        # Try setting via client API as fallback
                        for tag_key, tag_value in tags.items():
                            try:
                                client.set_model_version_tag(name=model_name, version=version_str, key=tag_key, value=str(tag_value))
                                logger.info(f"   ✓ Set tag via client API: {tag_key}={tag_value}")
                            except Exception as e:
                                logger.warning(f"   ✗ Failed to set tag {tag_key}: {str(e)[:100]}")
                except Exception as verify_error:
                    logger.warning(f"   Could not verify tags: {verify_error}")
            
            return str(result.version)
        except Exception as reg_error:
            # If tags cause issues (tag policies, permissions), register without tags first
            error_str = str(reg_error).lower()
            if 'tag' in error_str or 'permission_denied' in error_str or 'tag assignment' in error_str:
                logger.info(f"ℹ️  Tag registration via register_model failed, registering without tags then adding tags via client API...")
                try:
                    result = mlflow.register_model(model_uri=model_uri, name=model_name)
                    logger.info(f"Successfully registered as version {result.version}")
                    
                    # Try to add tags via client API after registration (more permissive)
                    if tags and result.version:
                        tags_added = []
                        tags_failed = []
                        version_str = str(result.version)  # Ensure version is string
                        
                        # Skip 'source' tag if it's restricted by tag policies, but try others
                        for tag_key, tag_value in tags.items():
                            try:
                                # Use correct API with keyword arguments for clarity
                                # Signature: set_model_version_tag(name, version=None, key=None, value=None, stage=None)
                                client.set_model_version_tag(
                                    name=model_name,
                                    version=version_str,
                                    key=tag_key,
                                    value=str(tag_value)
                                )
                                tags_added.append(tag_key)
                                logger.info(f"   ✓ Added tag: {tag_key}={tag_value} to {model_name} version {version_str}")
                            except Exception as tag_error:
                                error_str = str(tag_error).lower()
                                # If it's a tag policy restriction, log as info (expected)
                                if 'tag assignment' in error_str or 'tag policy' in error_str or 'permission_denied' in error_str:
                                    logger.info(f"   ℹ️  Skipped tag {tag_key} (restricted by tag policies)")
                                else:
                                    tags_failed.append(f"{tag_key}")
                                    logger.warning(f"   ✗ Failed to add tag {tag_key}: {str(tag_error)[:100]}")
                        
                        # Verify tags were actually set
                        if tags_added:
                            try:
                                mv = client.get_model_version(model_name, version_str)
                                actual_tags = mv.tags if hasattr(mv, 'tags') else {}
                                logger.info(f"Added {len(tags_added)} tags via client API: {', '.join(tags_added)}")
                                logger.info(f"   Verified tags on version {version_str}: {list(actual_tags.keys())}")
                            except Exception as verify_error:
                                logger.warning(f"   Could not verify tags: {verify_error}")
                        
                        if tags_failed:
                            logger.warning(f" Could not add {len(tags_failed)} tags: {', '.join(tags_failed)}")
                    
                    return str(result.version)
                except Exception as retry_error:
                    logger.error(f"Registration failed even without tags: {retry_error}")
                    raise
            else:
                # Some other error, re-raise it
                raise
    except Exception as e:
        logger.error(f"Failed to register model {model_name} from {model_uri}: {e}")
        raise
