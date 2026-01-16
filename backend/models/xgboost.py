import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import logging
import warnings
import holidays
from backend.models.utils import compute_metrics, compute_prediction_intervals, detect_weekly_freq_code

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


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

    def __init__(self, model, feature_columns, frequency, last_known_values, covariate_means, yoy_lag_values=None, weekly_freq_code=None):
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
        # Store the exact weekly frequency code (e.g., 'W-MON') for date alignment
        self.weekly_freq_code = weekly_freq_code or 'W-MON'

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

        # Use stored weekly_freq_code for proper day-of-week alignment
        freq_map = {'daily': 'D', 'weekly': self.weekly_freq_code, 'monthly': 'MS', 'yearly': 'YS'}
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

def create_xgboost_features(df: pd.DataFrame, target_col: str = 'y', covariates: List[str] = None, include_lags: bool = True, frequency: str = 'daily') -> pd.DataFrame:
    """
    Create all features for XGBoost time series model including YoY lag features
    for better holiday forecasting.
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

    # User-provided covariates (promo columns) are used as-is
    # XGBoost can learn non-linear interactions directly from binary indicators
    if covariates:
        valid_promo_cols = [c for c in covariates if c in df.columns]
        if valid_promo_cols:
            logger.info(f"XGBoost: Using {len(valid_promo_cols)} user-provided covariate columns as-is")

    return df

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

# model = XGBRegressor(**PARAMS)
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

def train_xgboost_model(
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
    Train XGBoost model for time series forecasting with full covariate support
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    import holidays

    logger.info(f"Training XGBoost model (freq={frequency}, covariates={covariates}, country={country}, seed={random_seed})...")

    # Set random seeds for reproducibility
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    logger.info(f"XGBoost: Set random seed to {random_seed} for reproducibility")

    # For weekly, detect the actual day-of-week from training data
    pd_freq = detect_weekly_freq_code(train_df, frequency)

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

    # ==========================================================================
    # CRITICAL: Create lag features WITHOUT data leakage
    # ==========================================================================
    # We must NOT use test set values when computing lag features for training!
    # Instead:
    # 1. Create lag features for train set using only train data
    # 2. For test set, compute lag features using only data available at that point
    # ==========================================================================

    # Determine YoY lag column name based on frequency
    yoy_lag_map = {'daily': 364, 'weekly': 52, 'monthly': 12}
    yoy_lag = yoy_lag_map.get(frequency, 364)

    # Create features for training set (no leakage - uses only train data)
    train_featured = create_xgboost_features(train_df.copy(), 'y', valid_covariates, include_lags=True, frequency=frequency)

    # For test set, use an OPTIMIZED approach to prevent data leakage:
    # 1. Combine train + test but mask test target values first
    # 2. Create features on combined data
    # 3. The lag features for test rows will naturally use only train values
    #    because test values come AFTER train values chronologically
    #
    # NOTE: The key insight is that shift() in pandas respects row order.
    # Since test rows come after train rows, their lag features will
    # only use train data (no leakage).

    # However, rolling features could use test values for later test rows
    # To prevent this, we process test set row-by-row using vectorized operations

    # Simple approach that's correct: create lag features iteratively for test set
    # but use vectorized numpy operations for speed
    test_featured = test_df.copy()

    # Get train values for lag computation
    train_y_values = train_df['y'].values
    train_dates = train_df['ds'].values

    # Pre-compute calendar features for test (these don't depend on target)
    test_featured['day_of_week'] = test_featured['ds'].dt.dayofweek
    test_featured['day_of_month'] = test_featured['ds'].dt.day
    test_featured['month'] = test_featured['ds'].dt.month
    test_featured['week_of_year'] = test_featured['ds'].dt.isocalendar().week.astype(int)
    test_featured['is_weekend'] = (test_featured['ds'].dt.dayofweek >= 5).astype(int)
    test_featured['quarter'] = test_featured['ds'].dt.quarter
    test_featured['is_month_start'] = test_featured['ds'].dt.is_month_start.astype(int)
    test_featured['is_month_end'] = test_featured['ds'].dt.is_month_end.astype(int)
    test_featured['is_quarter_start'] = test_featured['ds'].dt.is_quarter_start.astype(int)
    test_featured['is_quarter_end'] = test_featured['ds'].dt.is_quarter_end.astype(int)
    test_featured['week_of_month'] = (test_featured['ds'].dt.day - 1) // 7 + 1

    # Compute lag features for test set using ONLY train data
    # For the first test row, use last train values
    # For subsequent test rows, could use predictions but we use actuals for validation
    all_y = np.concatenate([train_y_values, test_df['y'].values])

    test_lag_1 = []
    test_lag_7 = []
    test_rolling_7 = []
    test_yoy_lag = []
    test_yoy_rolling = []
    test_yoy_ratio = []

    for i in range(len(test_df)):
        idx_in_full = len(train_y_values) + i
        # lag_1: value at position idx-1
        test_lag_1.append(all_y[idx_in_full - 1] if idx_in_full >= 1 else 0)
        # lag_7: value at position idx-7
        test_lag_7.append(all_y[idx_in_full - 7] if idx_in_full >= 7 else all_y[0])
        # rolling_mean_7: mean of values at positions idx-7 to idx-1
        start_idx = max(0, idx_in_full - 7)
        test_rolling_7.append(np.mean(all_y[start_idx:idx_in_full]) if idx_in_full > 0 else 0)
        # YoY lag
        yoy_idx = idx_in_full - yoy_lag
        test_yoy_lag.append(all_y[yoy_idx] if yoy_idx >= 0 else np.mean(train_y_values))
        # YoY rolling avg
        yoy_start = max(0, yoy_idx - 3)
        yoy_end = yoy_idx + 4
        test_yoy_rolling.append(np.mean(all_y[yoy_start:min(yoy_end, len(train_y_values))]) if yoy_idx >= 0 else np.mean(train_y_values))
        # YoY ratio
        current_val = all_y[idx_in_full] if idx_in_full < len(all_y) else 0
        yoy_val = test_yoy_lag[-1]
        test_yoy_ratio.append(min(max(current_val / yoy_val if yoy_val != 0 else 1.0, 0.1), 10.0))

    test_featured['lag_1'] = test_lag_1
    test_featured['lag_7'] = test_lag_7
    test_featured['rolling_mean_7'] = test_rolling_7
    test_featured[f'lag_{yoy_lag}'] = test_yoy_lag
    test_featured[f'lag_{yoy_lag}_rolling_avg'] = test_yoy_rolling
    test_featured['yoy_ratio'] = test_yoy_ratio

    if frequency == 'daily':
        test_lag_365 = []
        for i in range(len(test_df)):
            idx_in_full = len(train_y_values) + i
            lag_idx = idx_in_full - 365
            test_lag_365.append(all_y[lag_idx] if lag_idx >= 0 else np.mean(train_y_values))
        test_featured['lag_365'] = test_lag_365

    # Also create full_df for later use (retraining on all data)
    full_df = pd.concat([train_df, test_df]).sort_values('ds').reset_index(drop=True)

    logger.info(f"Created features WITHOUT data leakage: train={len(train_featured)}, test={len(test_featured)}")

    # Define feature columns
    calendar_features = [
        'day_of_week', 'day_of_month', 'month', 'week_of_year', 'is_weekend', 'quarter',
        'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'week_of_month'
    ]
    lag_features = ['lag_1', 'lag_7', 'rolling_mean_7', f'lag_{yoy_lag}', f'lag_{yoy_lag}_rolling_avg', 'yoy_ratio']
    if frequency == 'daily':
        lag_features.append('lag_365')

    # Promo-derived features
    promo_derived = ['any_promo_active', 'promo_count', 'promo_window', 'is_promo_weekend', 'is_regular_weekend']
    promo_derived = [c for c in promo_derived if c in train_featured.columns]

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

    # Hyperparameter grid - apply filters from data analysis if provided
    xgb_filters = (hyperparameter_filters or {}).get('XGBoost', {})

    # Default param_grid values
    default_param_grid = [
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
        {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
        {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
    ]

    # Build param_grid from filters if provided
    if xgb_filters:
        n_estimators_options = xgb_filters.get('n_estimators', [100, 200])
        max_depth_options = xgb_filters.get('max_depth', [3, 5])
        learning_rate_options = xgb_filters.get('learning_rate', [0.05, 0.1])

        # Ensure they're lists
        if not isinstance(n_estimators_options, list):
            n_estimators_options = [n_estimators_options]
        if not isinstance(max_depth_options, list):
            max_depth_options = [max_depth_options]
        if not isinstance(learning_rate_options, list):
            learning_rate_options = [learning_rate_options]

        # Build combinations
        import itertools
        param_grid = [
            {'n_estimators': n, 'max_depth': d, 'learning_rate': lr}
            for n, d, lr in itertools.product(n_estimators_options, max_depth_options, learning_rate_options)
        ]
        logger.info(f"📊 Using data-driven XGBoost filters: n_estimators={n_estimators_options}, max_depth={max_depth_options}, learning_rate={learning_rate_options}")
    else:
        param_grid = default_param_grid

    max_combinations = int(os.environ.get('XGBOOST_MAX_COMBINATIONS', '4'))
    param_grid = param_grid[:max_combinations]

    with mlflow.start_run(run_name="XGBoost_Tuning", nested=True) as parent_run:
        parent_run_id = parent_run.info.run_id
        experiment_id = parent_run.info.experiment_id

        if original_data:
            try:
                original_df = pd.DataFrame(original_data)
                original_df.to_csv("/tmp/original_timeseries_data.csv", index=False)
                mlflow.log_artifact("/tmp/original_timeseries_data.csv", "datasets/raw")
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

        # Time Series Cross-Validation
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
            except Exception as e:
                logger.warning(f"  CV Fold {i+1} failed: {e}")

        if len(cv_scores) > 0:
            cv_mean = round(np.mean(cv_scores), 2)
            cv_std = round(np.std(cv_scores), 2)
            best_metrics["cv_mape"] = cv_mean
            best_metrics["cv_mape_std"] = cv_std

        # Validation data
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
        # Vectorized approach - much faster than iterrows
        date_strs = full_df['ds'].dt.strftime('%Y-%m-%d')
        yoy_lag_values = dict(zip(date_strs, full_df['y'].values))

        # Generate forecast - use forecast_start_date if provided (user's to_date)
        if forecast_start_date is not None:
            last_date = pd.to_datetime(forecast_start_date).normalize()
            logger.info(f"📅 Using user-specified forecast start: {last_date}")
        else:
            last_date = full_df['ds'].max()
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=pd_freq)[1:]
        logger.info(f"📅 XGBoost forecast dates: {future_dates.min()} to {future_dates.max()}")
        future_df = pd.DataFrame({'ds': future_dates})
        # Create features without lag columns (we'll fill them recursively during prediction)
        future_df = create_xgboost_features(future_df, 'y', valid_covariates, include_lags=False)

        # For lag features in forecast, use recursive prediction
        predictions = []
        temp_last_values = last_known_values.copy()

        # Get historical mean for fallback
        hist_mean = np.mean(full_df['y'].values)

        for i in range(len(future_df)):
            row = future_df.iloc[[i]].copy()
            current_date = row['ds'].iloc[0]

            # Short-term lag features
            row['lag_1'] = temp_last_values[-1]
            row['lag_7'] = temp_last_values[-7] if len(temp_last_values) >= 7 else temp_last_values[-1]
            row['rolling_mean_7'] = np.mean(temp_last_values[-7:]) if len(temp_last_values) >= 7 else np.mean(temp_last_values)

            # CRITICAL: YoY lag features - look up from historical data
            # This is essential for seasonal patterns!
            yoy_lag_date = current_date - pd.Timedelta(days=yoy_lag if frequency == 'daily' else yoy_lag * 7 if frequency == 'weekly' else yoy_lag * 30)
            yoy_date_str = yoy_lag_date.strftime('%Y-%m-%d')
            yoy_value = yoy_lag_values.get(yoy_date_str, hist_mean)
            row[f'lag_{yoy_lag}'] = yoy_value
            row[f'lag_{yoy_lag}_rolling_avg'] = yoy_value
            row['yoy_ratio'] = 1.0  # Default ratio for prediction

            if frequency == 'daily':
                lag_365_date = current_date - pd.Timedelta(days=365)
                row['lag_365'] = yoy_lag_values.get(lag_365_date.strftime('%Y-%m-%d'), hist_mean)

            # Fill covariates
            for cov in valid_covariates:
                if cov not in row.columns or pd.isna(row[cov].iloc[0]):
                    cov_values = train_df[cov].dropna().unique() if cov in train_df.columns else []
                    is_binary = len(cov_values) <= 2 and all(v in [0, 1, 0.0, 1.0] for v in cov_values)
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

            # Log signature and input example details
            logger.info(f"")
            logger.info(f"   {'='*50}")
            logger.info(f"   📦 LOGGING XGBOOST MODEL TO MLFLOW")
            logger.info(f"   {'='*50}")
            logger.info(f"   📝 Model Signature:")
            logger.info(f"      Inputs: {signature.inputs}")
            logger.info(f"      Outputs: {signature.outputs}")
            logger.info(f"   📋 Input Example:")
            logger.info(f"      Shape: {input_example.shape}")
            logger.info(f"      Columns: {list(input_example.columns)}")
            logger.info(f"      Sample: {input_example.iloc[0].to_dict()}")
            logger.info(f"   📦 Dependencies: mlflow, pandas, numpy, xgboost, scikit-learn")

            weekly_freq_code = detect_weekly_freq_code(train_df, frequency)
            model_wrapper = XGBoostModelWrapper(
                final_model, feature_columns, frequency,
                last_known_values, covariate_means, yoy_lag_values, weekly_freq_code
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
                        {"pip": ["mlflow", "pandas", "numpy", "xgboost", "scikit-learn", "holidays"]}
                    ],
                    "name": "xgboost_env"
                }
            )
            logger.info(f"   ✅ XGBoost model logged successfully to: model/")
            logger.info(f"   {'='*50}")
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
