"""
Preprocessing module for time series forecasting with holiday/promo enhancements.

This module automatically derives additional features from user-provided promo files
to improve forecasting accuracy on holidays and special events.

Applied to: Prophet, SARIMAX, XGBoost (models that support covariates)
NOT applied to: ARIMA, ETS (univariate models that can't use features)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def enhance_features_for_forecasting(
    df: pd.DataFrame,
    date_col: str = 'ds',
    target_col: str = 'y',
    promo_cols: Optional[List[str]] = None,
    frequency: str = 'daily'
) -> pd.DataFrame:
    """
    Auto-enhance dataframe with derived features that improve holiday/weekend
    forecasting accuracy.

    This function is called BEFORE model training to add features that help
    models understand holiday patterns without requiring users to modify their
    promo files.

    Features Added:
    1. lag_364/lag_365 - Same day last year (critical for holiday patterns)
    2. lag_52w_avg - Average of same week last year (smoothed version)
    3. days_to_nearest_promo - Anticipation effect before promotions
    4. days_since_last_promo - Post-event hangover effect
    5. is_promo_weekend - Holiday weekend vs regular weekend
    6. any_promo_active - Simplified "special day" indicator
    7. promo_window - Extended effect around promo days (±2 days)

    Args:
        df: DataFrame with date column, target column, and optionally promo columns
        date_col: Name of the date column (default 'ds')
        target_col: Name of the target column (default 'y')
        promo_cols: List of promo/holiday column names to use for derived features
        frequency: Data frequency ('daily', 'weekly', 'monthly')

    Returns:
        DataFrame with additional derived features
    """
    result = df.copy()

    # Ensure date column is datetime
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values(date_col).reset_index(drop=True)

    # Determine lag periods based on frequency
    lag_periods = _get_lag_periods(frequency)

    # 1. Add year-over-year lag features (CRITICAL for holiday patterns)
    if target_col in result.columns:
        _add_yoy_lag_features(result, target_col, lag_periods, frequency)

    # 2. Add promo-derived features if promo columns exist
    if promo_cols:
        valid_promo_cols = [c for c in promo_cols if c in result.columns]
        if valid_promo_cols:
            _add_promo_derived_features(result, date_col, valid_promo_cols)
            logger.info(f"Added promo-derived features from {len(valid_promo_cols)} promo columns")

    # 3. Add calendar features that help with weekend patterns
    _add_enhanced_calendar_features(result, date_col)

    return result


def _get_lag_periods(frequency: str) -> Dict[str, int]:
    """
    Get appropriate lag periods based on data frequency.

    For daily data: 364/365 days = same day last year
    For weekly data: 52 weeks = same week last year
    For monthly data: 12 months = same month last year
    """
    if frequency == 'daily':
        return {
            'yoy_lag': 364,      # Same day last year (52 weeks)
            'yoy_lag_alt': 365,  # Handle leap year edge cases
            'yoy_window': 7,     # Week window for averaging
        }
    elif frequency == 'weekly':
        return {
            'yoy_lag': 52,       # Same week last year
            'yoy_lag_alt': 53,   # Handle 53-week years
            'yoy_window': 4,     # 4-week window for averaging
        }
    elif frequency == 'monthly':
        return {
            'yoy_lag': 12,       # Same month last year
            'yoy_lag_alt': 12,   # No alternative needed
            'yoy_window': 3,     # 3-month window for averaging
        }
    else:
        # Default to daily
        return {
            'yoy_lag': 364,
            'yoy_lag_alt': 365,
            'yoy_window': 7,
        }


def _add_yoy_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lag_periods: Dict[str, int],
    frequency: str
) -> None:
    """
    Add year-over-year lag features to capture seasonal patterns.

    These are the most important features for holiday forecasting because
    the best predictor for Thanksgiving 2024 is Thanksgiving 2023.
    """
    yoy_lag = lag_periods['yoy_lag']
    yoy_lag_alt = lag_periods['yoy_lag_alt']
    yoy_window = lag_periods['yoy_window']

    # Primary YoY lag (e.g., 364 days for daily data)
    df[f'lag_{yoy_lag}'] = df[target_col].shift(yoy_lag)

    # Alternative YoY lag for leap year handling
    if yoy_lag != yoy_lag_alt:
        df[f'lag_{yoy_lag_alt}'] = df[target_col].shift(yoy_lag_alt)

    # Smoothed YoY average (handles slight date misalignment)
    # Take average of same week last year to be more robust
    df[f'lag_{yoy_lag}_rolling_avg'] = (
        df[target_col]
        .shift(yoy_lag - yoy_window // 2)
        .rolling(window=yoy_window, min_periods=1)
        .mean()
    )

    # YoY ratio - how much did same period grow/shrink?
    # This helps the model understand year-over-year trends
    if f'lag_{yoy_lag}' in df.columns:
        # Avoid division by zero
        safe_lag = df[f'lag_{yoy_lag}'].replace(0, np.nan)
        df['yoy_ratio'] = df[target_col] / safe_lag
        # Cap extreme ratios to avoid outlier influence
        df['yoy_ratio'] = df['yoy_ratio'].clip(0.1, 10.0)
        df['yoy_ratio'] = df['yoy_ratio'].fillna(1.0)

    logger.info(f"Added YoY lag features: lag_{yoy_lag}, lag_{yoy_lag}_rolling_avg, yoy_ratio")


def _add_promo_derived_features(
    df: pd.DataFrame,
    date_col: str,
    promo_cols: List[str]
) -> None:
    """
    Derive additional features from promo/holiday columns.

    These features capture:
    - Combined effect of any promotion being active
    - Extended "window" effect around promotions (±2 days)
    - Distance to/from nearest promotion (anticipation and hangover effects)
    - Whether a weekend falls near a holiday
    """
    # 1. Any promo active (simplified indicator)
    df['any_promo_active'] = df[promo_cols].max(axis=1)

    # 2. Count of active promos (some days may have overlapping events)
    df['promo_count'] = df[promo_cols].sum(axis=1)

    # 3. Promo window - extend promo effect ±2 days
    # This captures the "halo effect" around holidays
    window_size = 5  # 2 days before + day of + 2 days after
    df['promo_window'] = (
        df['any_promo_active']
        .rolling(window=window_size, center=True, min_periods=1)
        .max()
    )

    # 4. Days to nearest promo (anticipation effect)
    df['days_to_promo'] = _calculate_days_to_next_event(df, 'any_promo_active')

    # 5. Days since last promo (hangover effect)
    df['days_since_promo'] = _calculate_days_since_last_event(df, 'any_promo_active')

    # 6. Proximity indicator (within 3 days of any promo)
    df['near_promo'] = ((df['days_to_promo'] <= 3) | (df['days_since_promo'] <= 3)).astype(int)

    # 7. Holiday weekend indicator
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = pd.to_datetime(df[date_col]).dt.dayofweek

    is_weekend = df['day_of_week'] >= 5
    df['is_promo_weekend'] = (is_weekend & (df['near_promo'] == 1)).astype(int)
    df['is_regular_weekend'] = (is_weekend & (df['near_promo'] == 0)).astype(int)


def _calculate_days_to_next_event(df: pd.DataFrame, event_col: str) -> pd.Series:
    """
    Calculate days until the next event (promo=1).

    Returns a series where each row contains the number of days
    until the next occurrence of event_col=1.
    """
    result = pd.Series(index=df.index, dtype=float)

    # Find indices where event occurs
    event_indices = df[df[event_col] == 1].index.tolist()

    if not event_indices:
        return pd.Series(999, index=df.index)

    for idx in df.index:
        # Find next event after this index
        future_events = [e for e in event_indices if e > idx]
        if future_events:
            result[idx] = future_events[0] - idx
        else:
            result[idx] = 999  # No future event

    # On event days, distance is 0
    result[df[event_col] == 1] = 0

    return result


def _calculate_days_since_last_event(df: pd.DataFrame, event_col: str) -> pd.Series:
    """
    Calculate days since the last event (promo=1).

    Returns a series where each row contains the number of days
    since the last occurrence of event_col=1.
    """
    result = pd.Series(index=df.index, dtype=float)

    # Find indices where event occurs
    event_indices = df[df[event_col] == 1].index.tolist()

    if not event_indices:
        return pd.Series(999, index=df.index)

    for idx in df.index:
        # Find most recent event before or at this index
        past_events = [e for e in event_indices if e <= idx]
        if past_events:
            result[idx] = idx - past_events[-1]
        else:
            result[idx] = 999  # No past event

    return result


def _add_enhanced_calendar_features(df: pd.DataFrame, date_col: str) -> None:
    """
    Add enhanced calendar features that help differentiate weekends
    and capture monthly/quarterly patterns.
    """
    dates = pd.to_datetime(df[date_col])

    # Basic calendar features (may already exist, but ensure they do)
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = dates.dt.dayofweek

    if 'is_weekend' not in df.columns:
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Month start/end indicators (important for financial data)
    df['is_month_start'] = dates.dt.is_month_start.astype(int)
    df['is_month_end'] = dates.dt.is_month_end.astype(int)

    # Quarter indicators
    df['is_quarter_start'] = dates.dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = dates.dt.is_quarter_end.astype(int)

    # Week of month (1st week, 2nd week, etc.)
    df['week_of_month'] = (dates.dt.day - 1) // 7 + 1


def get_derived_feature_columns(promo_cols: Optional[List[str]] = None) -> List[str]:
    """
    Get the list of derived feature column names that will be added.

    This is useful for understanding what new columns are available
    after preprocessing.

    Args:
        promo_cols: Original promo column names (if any)

    Returns:
        List of derived feature column names
    """
    derived = [
        # YoY lag features
        'lag_364',
        'lag_365',
        'lag_364_rolling_avg',
        'yoy_ratio',
        # Calendar features
        'day_of_week',
        'is_weekend',
        'is_month_start',
        'is_month_end',
        'is_quarter_start',
        'is_quarter_end',
        'week_of_month',
    ]

    if promo_cols:
        derived.extend([
            # Promo-derived features
            'any_promo_active',
            'promo_count',
            'promo_window',
            'days_to_promo',
            'days_since_promo',
            'near_promo',
            'is_promo_weekend',
            'is_regular_weekend',
        ])

    return derived


def prepare_future_features(
    future_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    date_col: str = 'ds',
    target_col: str = 'y',
    promo_cols: Optional[List[str]] = None,
    frequency: str = 'daily'
) -> pd.DataFrame:
    """
    Prepare derived features for future forecasting periods.

    For future periods, we need to:
    1. Use historical lag values (lag_364 from last year's actuals)
    2. Calculate promo-derived features from future promo data
    3. Add calendar features

    Args:
        future_df: DataFrame with future dates and promo values
        historical_df: DataFrame with historical data (for lag values)
        date_col: Name of date column
        target_col: Name of target column
        promo_cols: List of promo column names
        frequency: Data frequency

    Returns:
        Future DataFrame with all derived features populated
    """
    result = future_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])

    # Combine historical and future for lag calculation
    hist = historical_df.copy()
    hist[date_col] = pd.to_datetime(hist[date_col])

    # For each future date, find the corresponding lag value from history
    lag_periods = _get_lag_periods(frequency)
    yoy_lag = lag_periods['yoy_lag']

    # Create a lookup from historical data
    hist_lookup = hist.set_index(date_col)[target_col].to_dict()

    # Map lag values for future dates
    def get_lag_value(future_date, lag_days):
        lag_date = future_date - pd.Timedelta(days=lag_days)
        return hist_lookup.get(lag_date, np.nan)

    result[f'lag_{yoy_lag}'] = result[date_col].apply(
        lambda x: get_lag_value(x, yoy_lag)
    )

    if lag_periods['yoy_lag'] != lag_periods['yoy_lag_alt']:
        result[f'lag_{lag_periods["yoy_lag_alt"]}'] = result[date_col].apply(
            lambda x: get_lag_value(x, lag_periods['yoy_lag_alt'])
        )

    # Fill missing lag values with historical mean as fallback
    hist_mean = hist[target_col].mean() if target_col in hist.columns else 0
    for col in result.columns:
        if col.startswith('lag_'):
            result[col] = result[col].fillna(hist_mean)

    # Add promo-derived features
    if promo_cols:
        valid_promo_cols = [c for c in promo_cols if c in result.columns]
        if valid_promo_cols:
            _add_promo_derived_features(result, date_col, valid_promo_cols)

    # Add calendar features
    _add_enhanced_calendar_features(result, date_col)

    # Set yoy_ratio to 1.0 for future (no actual value to compare)
    result['yoy_ratio'] = 1.0

    # Rolling avg uses lag value directly for future
    result[f'lag_{yoy_lag}_rolling_avg'] = result[f'lag_{yoy_lag}']

    return result
