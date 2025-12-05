"""
Preprocessing module for time series forecasting.

This module adds generic features that improve forecasting accuracy across
all covariate-supporting algorithms (Prophet, SARIMAX, XGBoost).

Design philosophy:
- Add features that are universally useful regardless of promo structure
- Conditionally add features based on data availability (e.g., YoY lags need 1+ year)
- Don't modify user's promo columns - they're already well-structured
- Keep it simple and avoid over-engineering

Applied to: Prophet, SARIMAX, XGBoost (models that support covariates)
NOT applied to: ARIMA, ETS (univariate models that can't use features)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
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
    Add generic features that improve forecasting for all algorithms.

    Features added:
    1. Calendar features (always): day_of_week, is_weekend, month, quarter
    2. Trend features (always): time_index (helps XGBoost capture trends)
    3. YoY lag features (conditional): only if 1+ year of data exists

    User's promo columns are preserved as-is without modification.

    Args:
        df: DataFrame with date column, target column, and optionally promo columns
        date_col: Name of the date column (default 'ds')
        target_col: Name of the target column (default 'y')
        promo_cols: List of promo/holiday column names (preserved as-is)
        frequency: Data frequency ('daily', 'weekly', 'monthly')

    Returns:
        DataFrame with additional features
    """
    result = df.copy()

    # Ensure date column is datetime
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values(date_col).reset_index(drop=True)

    # 1. Add calendar features (always useful)
    _add_calendar_features(result, date_col)

    # 2. Add trend features (helps XGBoost especially)
    _add_trend_features(result, date_col)

    # 3. Conditionally add YoY lag features if enough history exists
    if target_col in result.columns:
        _add_yoy_lag_features_if_available(result, target_col, frequency)

    # Log what promo columns are available (but don't modify them)
    if promo_cols:
        valid_promo_cols = [c for c in promo_cols if c in result.columns]
        if valid_promo_cols:
            logger.info(f"Using {len(valid_promo_cols)} user-provided promo columns as-is")

    return result


def _add_calendar_features(df: pd.DataFrame, date_col: str) -> None:
    """
    Add calendar features that help capture day/week/month patterns.
    These are universally useful for all time series.
    """
    dates = pd.to_datetime(df[date_col])

    # Day of week (0=Monday, 6=Sunday)
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = dates.dt.dayofweek

    # Weekend indicator
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Month (1-12) - useful for monthly seasonality
    if 'month' not in df.columns:
        df['month'] = dates.dt.month

    # Quarter (1-4) - useful for quarterly patterns
    if 'quarter' not in df.columns:
        df['quarter'] = dates.dt.quarter

    # Day of month - useful for monthly billing cycles, paydays, etc.
    if 'day_of_month' not in df.columns:
        df['day_of_month'] = dates.dt.day

    # Week of year (1-52) - useful for yearly seasonality
    if 'week_of_year' not in df.columns:
        df['week_of_year'] = dates.dt.isocalendar().week.astype(int)

    logger.info("Added calendar features: day_of_week, is_weekend, month, quarter, day_of_month, week_of_year")


def _add_trend_features(df: pd.DataFrame, date_col: str) -> None:
    """
    Add trend features that help tree-based models (XGBoost) capture trends.
    Prophet and SARIMAX handle trends internally, but these don't hurt.
    """
    # Time index (0, 1, 2, ...) - simple trend indicator
    if 'time_index' not in df.columns:
        df['time_index'] = range(len(df))

    # Year - helps capture year-over-year growth
    dates = pd.to_datetime(df[date_col])
    if 'year' not in df.columns:
        df['year'] = dates.dt.year

    logger.info("Added trend features: time_index, year")


def _add_yoy_lag_features_if_available(
    df: pd.DataFrame,
    target_col: str,
    frequency: str
) -> None:
    """
    Add year-over-year lag features ONLY if there's enough historical data.

    These features are powerful for capturing seasonal patterns, but require
    at least 1 year of data to be meaningful.
    """
    # Determine required lag based on frequency
    lag_config = {
        'daily': {'lag': 364, 'min_rows': 400},    # ~1 year + buffer
        'weekly': {'lag': 52, 'min_rows': 60},     # ~1 year + buffer
        'monthly': {'lag': 12, 'min_rows': 15},    # ~1 year + buffer
    }

    config = lag_config.get(frequency, lag_config['daily'])
    lag_periods = config['lag']
    min_rows = config['min_rows']

    # Check if we have enough data
    valid_rows = df[target_col].notna().sum()

    if valid_rows < min_rows:
        logger.info(f"Skipping YoY lag features: only {valid_rows} rows, need {min_rows}+ for {frequency} data")
        return

    # Add primary YoY lag
    lag_col = f'lag_{lag_periods}'
    df[lag_col] = df[target_col].shift(lag_periods)

    # Check if the lag feature has any non-NaN values
    non_null_count = df[lag_col].notna().sum()
    if non_null_count == 0:
        # Drop useless column
        df.drop(columns=[lag_col], inplace=True)
        logger.info(f"Skipping YoY lag features: lag_{lag_periods} would be all NaN")
        return

    # Add rolling average for smoothing (handles slight date misalignment)
    window = 7 if frequency == 'daily' else 4 if frequency == 'weekly' else 3
    df[f'{lag_col}_avg'] = (
        df[target_col]
        .shift(lag_periods)
        .rolling(window=window, min_periods=1)
        .mean()
    )

    logger.info(f"Added YoY lag features: {lag_col}, {lag_col}_avg ({non_null_count} non-null values)")


def get_derived_feature_columns(promo_cols: Optional[List[str]] = None) -> List[str]:
    """
    Get the list of derived feature column names that may be added.

    Note: YoY lag features are only added if enough data exists,
    so they may not all be present in the final dataframe.

    Args:
        promo_cols: Original promo column names (not modified)

    Returns:
        List of potentially derived feature column names
    """
    return [
        # Calendar features (always added)
        'day_of_week',
        'is_weekend',
        'month',
        'quarter',
        'day_of_month',
        'week_of_year',
        # Trend features (always added)
        'time_index',
        'year',
        # YoY lag features (conditionally added)
        'lag_364',
        'lag_364_avg',
        'lag_52',
        'lag_52_avg',
        'lag_12',
        'lag_12_avg',
    ]


def prepare_future_features(
    future_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    date_col: str = 'ds',
    target_col: str = 'y',
    promo_cols: Optional[List[str]] = None,
    frequency: str = 'daily'
) -> pd.DataFrame:
    """
    Prepare features for future forecasting periods.

    For future periods:
    - Calendar and trend features are calculated from dates
    - YoY lag values are looked up from historical data

    Args:
        future_df: DataFrame with future dates and promo values
        historical_df: DataFrame with historical data (for lag values)
        date_col: Name of date column
        target_col: Name of target column
        promo_cols: List of promo column names
        frequency: Data frequency

    Returns:
        Future DataFrame with all features
    """
    result = future_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])

    # Add calendar features
    _add_calendar_features(result, date_col)

    # Add trend features (continue from historical)
    hist_len = len(historical_df)
    result['time_index'] = range(hist_len, hist_len + len(result))
    result['year'] = pd.to_datetime(result[date_col]).dt.year

    # Add YoY lag values from historical data
    _add_future_yoy_lags(result, historical_df, date_col, target_col, frequency)

    return result


def _add_future_yoy_lags(
    future_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    frequency: str
) -> None:
    """
    Look up YoY lag values for future dates from historical data.
    """
    lag_config = {
        'daily': 364,
        'weekly': 52,
        'monthly': 12,
    }
    lag_periods = lag_config.get(frequency, 364)
    lag_col = f'lag_{lag_periods}'

    # Create lookup from historical data
    hist = historical_df.copy()
    hist[date_col] = pd.to_datetime(hist[date_col])
    hist_lookup = hist.set_index(date_col)[target_col].to_dict()

    # Map lag values for future dates
    def get_lag_value(future_date):
        if frequency == 'daily':
            lag_date = future_date - pd.Timedelta(days=lag_periods)
        elif frequency == 'weekly':
            lag_date = future_date - pd.Timedelta(weeks=lag_periods)
        else:  # monthly
            lag_date = future_date - pd.DateOffset(months=lag_periods)
        return hist_lookup.get(lag_date, np.nan)

    future_df[lag_col] = future_df[date_col].apply(get_lag_value)

    # Fill missing with historical mean as fallback
    hist_mean = hist[target_col].mean() if target_col in hist.columns else 0
    future_df[lag_col] = future_df[lag_col].fillna(hist_mean)
    future_df[f'{lag_col}_avg'] = future_df[lag_col]  # Same as lag for future

    non_null = future_df[lag_col].notna().sum()
    logger.info(f"Added future YoY lag values: {lag_col} ({non_null} values from history)")
