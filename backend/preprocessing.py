"""
Preprocessing module for time series forecasting.

This module adds calendar features to help models understand day-of-week
and month patterns. It is intentionally lightweight to avoid adding
noise from features that don't apply to all use cases.

Applied to: Prophet, SARIMAX, XGBoost (models that support covariates)
NOT applied to: ARIMA, ETS (univariate models that can't use features)

Design philosophy:
- User's promo/holiday files are already well-structured with binary indicators
- Don't create redundant derived features that duplicate user's data
- Only add calendar features that are universally useful
"""

import pandas as pd
import numpy as np
from typing import List, Optional
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
    Add calendar features to help models understand temporal patterns.

    This function adds lightweight calendar features that complement
    user-provided promo/holiday indicators. It does NOT modify or
    derive additional features from user's promo columns.

    Features Added:
    - day_of_week: 0=Monday, 6=Sunday
    - is_weekend: 1 if Saturday/Sunday, 0 otherwise

    Args:
        df: DataFrame with date column, target column, and optionally promo columns
        date_col: Name of the date column (default 'ds')
        target_col: Name of the target column (default 'y')
        promo_cols: List of promo/holiday column names (preserved as-is)
        frequency: Data frequency ('daily', 'weekly', 'monthly')

    Returns:
        DataFrame with calendar features added
    """
    result = df.copy()

    # Ensure date column is datetime
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values(date_col).reset_index(drop=True)

    # Add calendar features (useful for all datasets)
    _add_calendar_features(result, date_col, frequency)

    # Log what promo columns are available (but don't modify them)
    if promo_cols:
        valid_promo_cols = [c for c in promo_cols if c in result.columns]
        if valid_promo_cols:
            logger.info(f"Using {len(valid_promo_cols)} user-provided promo columns as-is: {valid_promo_cols}")

    return result


def _add_calendar_features(df: pd.DataFrame, date_col: str, frequency: str) -> None:
    """
    Add basic calendar features that help with day-of-week patterns.

    These features are universally useful and don't depend on promo structure.
    """
    dates = pd.to_datetime(df[date_col])

    # Day of week (0=Monday, 6=Sunday)
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = dates.dt.dayofweek

    # Weekend indicator
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    added_features = ['day_of_week', 'is_weekend']
    logger.info(f"Added calendar features: {added_features}")


def get_derived_feature_columns(promo_cols: Optional[List[str]] = None) -> List[str]:
    """
    Get the list of derived feature column names that will be added.

    This is intentionally minimal - only calendar features are added.
    User's promo columns are used as-is without modification.

    Args:
        promo_cols: Original promo column names (not used, kept for API compatibility)

    Returns:
        List of derived feature column names
    """
    return [
        'day_of_week',
        'is_weekend',
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

    Simply adds calendar features to future dates.
    User's promo columns for future dates should already be in future_df.

    Args:
        future_df: DataFrame with future dates and promo values
        historical_df: DataFrame with historical data (not used in simplified version)
        date_col: Name of date column
        target_col: Name of target column
        promo_cols: List of promo column names
        frequency: Data frequency

    Returns:
        Future DataFrame with calendar features added
    """
    result = future_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])

    # Add calendar features
    _add_calendar_features(result, date_col, frequency)

    return result
