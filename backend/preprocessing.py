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

Holiday Week Detection:
- For weekly data, holidays fall within a week rather than on a specific day
- This module auto-detects major US holidays and creates week-level indicators
- Helps models learn holiday-specific patterns (e.g., Thanksgiving week = +200%)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Major US holidays that significantly impact demand (especially for food delivery)
MAJOR_HOLIDAYS = {
    "New Year's Day": 'is_new_years_week',
    "Martin Luther King Jr. Day": 'is_mlk_week',
    "Presidents Day": 'is_presidents_week',
    "Memorial Day": 'is_memorial_week',
    "Independence Day": 'is_july4_week',
    "Labor Day": 'is_labor_week',
    "Columbus Day": 'is_columbus_week',
    "Veterans Day": 'is_veterans_week',
    "Thanksgiving": 'is_thanksgiving_week',
    "Christmas Day": 'is_christmas_week',
}

# Super Bowl Sunday (typically first Sunday of February) - huge for food delivery
# Black Friday (day after Thanksgiving)
# These are added separately as they're not in the holidays library

# Holiday proximity windows - how many weeks before/after to track
HOLIDAY_PROXIMITY_WEEKS = 2  # Track 2 weeks before and after major holidays


def get_thanksgiving_date(year: int) -> pd.Timestamp:
    """Get the date of Thanksgiving (4th Thursday of November) for a given year."""
    first_day = pd.Timestamp(year=year, month=11, day=1)
    first_thursday = first_day + pd.Timedelta(days=(3 - first_day.dayofweek) % 7)
    fourth_thursday = first_thursday + pd.Timedelta(weeks=3)
    return fourth_thursday


def get_christmas_date(year: int) -> pd.Timestamp:
    """Get Christmas date for a given year."""
    return pd.Timestamp(year=year, month=12, day=25)


def get_holiday_weeks_for_year(year: int, country: str = 'US') -> Dict[pd.Timestamp, str]:
    """
    Get a mapping of week start dates to holiday names for a given year.

    Args:
        year: The year to get holidays for
        country: Country code (default 'US')

    Returns:
        Dict mapping week start (Monday) to holiday column name
    """
    if not HOLIDAYS_AVAILABLE:
        return {}

    try:
        country_holidays = holidays.country_holidays(country, years=year)
    except Exception as e:
        logger.warning(f"Could not load holidays for {country} {year}: {e}")
        return {}

    week_holidays = {}

    for date, name in sorted(country_holidays.items()):
        # Find which major holiday this is
        col_name = None
        for holiday_name, col in MAJOR_HOLIDAYS.items():
            if holiday_name.lower() in name.lower():
                col_name = col
                break

        if col_name:
            # Get the Monday of the week containing this holiday
            holiday_date = pd.Timestamp(date)
            week_start = holiday_date - pd.Timedelta(days=holiday_date.dayofweek)
            week_holidays[week_start] = col_name

    return week_holidays


def _add_holiday_features(df: pd.DataFrame, date_col: str, frequency: str) -> List[str]:
    """
    Add holiday indicator features for both daily and weekly data.

    For daily data: creates binary indicator for holiday days
    For weekly data: creates binary indicators for weeks containing major holidays

    Args:
        df: DataFrame to add features to (modified in place)
        date_col: Name of the date column
        frequency: Data frequency ('daily', 'weekly', 'monthly')

    Returns:
        List of holiday column names that were added
    """
    if frequency == 'monthly':
        logger.info(f"Skipping holiday features for monthly data")
        return []

    if not HOLIDAYS_AVAILABLE:
        logger.warning("holidays package not installed. Run: pip install holidays")
        return []

    # For daily data, add simple holiday indicator
    if frequency == 'daily':
        return _add_daily_holiday_features(df, date_col)

    # For weekly data, add week-level indicators
    return _add_weekly_holiday_features(df, date_col)


def _add_daily_holiday_features(df: pd.DataFrame, date_col: str) -> List[str]:
    """
    Add holiday features for daily data.

    Creates:
    - is_holiday: binary indicator (1 if date is a US holiday)
    - days_to_holiday: days until next holiday (-ve if days after)
    - is_holiday_adjacent: 1 if within 1 day of a holiday
    """
    dates = pd.to_datetime(df[date_col])
    years = dates.dt.year.unique()

    # Get all holidays for all years in data (plus buffer years)
    all_holidays = set()
    for year in list(years) + [min(years) - 1, max(years) + 1]:
        try:
            year_holidays = holidays.US(years=int(year))
            all_holidays.update(year_holidays.keys())
        except Exception as e:
            logger.warning(f"Could not load holidays for {year}: {e}")

    added_cols = []

    # is_holiday: binary indicator
    if 'is_holiday' not in df.columns:
        df['is_holiday'] = dates.apply(lambda d: 1 if d.date() in all_holidays else 0).astype(int)
        added_cols.append('is_holiday')

    # is_holiday_adjacent: within 1 day of holiday
    if 'is_holiday_adjacent' not in df.columns:
        def is_adjacent(d):
            for offset in [-1, 0, 1]:
                check_date = (d + pd.Timedelta(days=offset)).date()
                if check_date in all_holidays:
                    return 1
            return 0
        df['is_holiday_adjacent'] = dates.apply(is_adjacent).astype(int)
        added_cols.append('is_holiday_adjacent')

    # Add specific major holiday indicators (without year - same pattern each year)
    # These use month/day patterns that repeat yearly
    major_holiday_patterns = {
        'is_new_years': lambda d: d.month == 1 and d.day <= 2,
        'is_july4': lambda d: d.month == 7 and 3 <= d.day <= 5,
        'is_christmas': lambda d: d.month == 12 and 24 <= d.day <= 26,
        'is_thanksgiving_period': lambda d: d.month == 11 and 22 <= d.day <= 28,  # Thanksgiving is 4th Thursday
        'is_super_bowl_period': lambda d: d.month == 2 and d.day <= 14 and d.weekday() == 6,  # First 2 Sundays of Feb
    }

    for col_name, pattern_fn in major_holiday_patterns.items():
        if col_name not in df.columns:
            df[col_name] = dates.apply(lambda d: 1 if pattern_fn(d) else 0).astype(int)
            added_cols.append(col_name)

    holiday_count = df['is_holiday'].sum() if 'is_holiday' in df.columns else 0
    logger.info(f"Added daily holiday features: {added_cols} ({holiday_count} holiday days found)")

    return added_cols


def _add_weekly_holiday_features(df: pd.DataFrame, date_col: str) -> List[str]:
    """
    Add holiday week indicator features for weekly data.

    Creates binary indicators for weeks containing major holidays.
    This helps models learn holiday-specific patterns.

    Args:
        df: DataFrame to add features to (modified in place)
        date_col: Name of the date column

    Returns:
        List of holiday column names that were added
    """

    dates = pd.to_datetime(df[date_col])
    years = dates.dt.year.unique()

    # Collect all holiday weeks across all years in the data
    all_holiday_weeks = {}
    for year in years:
        year_holidays = get_holiday_weeks_for_year(int(year))
        all_holiday_weeks.update(year_holidays)

    # Also check year before and after for edge cases
    for year in [min(years) - 1, max(years) + 1]:
        year_holidays = get_holiday_weeks_for_year(int(year))
        all_holiday_weeks.update(year_holidays)

    # Initialize all holiday columns to 0
    added_cols = []
    for col_name in set(MAJOR_HOLIDAYS.values()):
        if col_name not in df.columns:
            df[col_name] = 0
            added_cols.append(col_name)

    # Also add special days not in holidays library
    special_cols = ['is_super_bowl_week', 'is_black_friday_week']
    for col in special_cols:
        if col not in df.columns:
            df[col] = 0
            added_cols.append(col)

    # Mark holiday weeks using vectorized operations (much faster than iterrows)
    dates = pd.to_datetime(df[date_col])
    # Calculate week start (Monday) for each date vectorized
    week_starts = dates - pd.to_timedelta(dates.dt.dayofweek, unit='D')

    # Mark holiday weeks by checking against all_holiday_weeks dictionary
    for week_start_date, col_name in all_holiday_weeks.items():
        mask = week_starts == week_start_date
        if mask.any():
            df.loc[mask, col_name] = 1

    # Vectorized Black Friday week detection (4th Thursday of November)
    november_mask = dates.dt.month == 11
    if november_mask.any():
        nov_dates = dates[november_mask]
        nov_years = nov_dates.dt.year.unique()
        for year in nov_years:
            first_day = pd.Timestamp(year=year, month=11, day=1)
            first_thursday = first_day + pd.Timedelta(days=(3 - first_day.dayofweek) % 7)
            fourth_thursday = first_thursday + pd.Timedelta(weeks=3)
            thanksgiving_week_start = fourth_thursday - pd.Timedelta(days=fourth_thursday.dayofweek)
            bf_mask = (week_starts == thanksgiving_week_start)
            if bf_mask.any():
                df.loc[bf_mask, 'is_black_friday_week'] = 1

    # Vectorized Super Bowl week detection (first Sunday of February)
    february_mask = dates.dt.month == 2
    if february_mask.any():
        feb_dates = dates[february_mask]
        feb_years = feb_dates.dt.year.unique()
        for year in feb_years:
            first_day = pd.Timestamp(year=year, month=2, day=1)
            first_sunday = first_day + pd.Timedelta(days=(6 - first_day.dayofweek) % 7)
            super_bowl_week_start = first_sunday - pd.Timedelta(days=first_sunday.dayofweek)
            sb_mask = (week_starts == super_bowl_week_start)
            if sb_mask.any():
                df.loc[sb_mask, 'is_super_bowl_week'] = 1

    # Log which holidays were found
    found_holidays = []
    for col in added_cols:
        if df[col].sum() > 0:
            found_holidays.append(f"{col}({int(df[col].sum())})")

    if found_holidays:
        logger.info(f"Added holiday week features: {', '.join(found_holidays)}")
    else:
        logger.info("Added holiday week columns (no holidays found in date range)")

    # Add enhanced holiday proximity features for Thanksgiving and Christmas
    # These help models learn pre-holiday ramp-up and post-holiday patterns
    proximity_cols = _add_holiday_proximity_features(df, date_col, dates)
    added_cols.extend(proximity_cols)

    return added_cols


def _add_holiday_proximity_features(
    df: pd.DataFrame,
    date_col: str,
    dates: pd.Series
) -> List[str]:
    """
    Add holiday proximity features that help models learn:
    1. Pre-holiday ramp-up patterns (weeks_to_thanksgiving = 2, 1, 0)
    2. Post-holiday patterns (weeks_after_thanksgiving = 1, 2)
    3. Holiday magnitude hints (yoy_thanksgiving_ratio if historical data exists)

    These features are critical for accurate Thanksgiving/Christmas predictions.
    """
    added_cols = []
    years = dates.dt.year.unique()

    # Initialize proximity columns
    df['weeks_to_thanksgiving'] = -999  # -999 = not near Thanksgiving
    df['weeks_after_thanksgiving'] = -999
    df['weeks_to_christmas'] = -999
    df['weeks_after_christmas'] = -999
    df['is_pre_thanksgiving'] = 0  # 1-2 weeks before
    df['is_post_thanksgiving'] = 0  # 1-2 weeks after
    df['is_pre_christmas'] = 0
    df['is_post_christmas'] = 0

    added_cols.extend([
        'weeks_to_thanksgiving', 'weeks_after_thanksgiving',
        'weeks_to_christmas', 'weeks_after_christmas',
        'is_pre_thanksgiving', 'is_post_thanksgiving',
        'is_pre_christmas', 'is_post_christmas'
    ])

    # Calculate week start for each row (vectorized)
    week_starts = dates - pd.to_timedelta(dates.dt.dayofweek, unit='D')

    for year in years:
        try:
            # Thanksgiving proximity
            thanksgiving = get_thanksgiving_date(int(year))
            thanksgiving_week_start = thanksgiving - pd.Timedelta(days=thanksgiving.dayofweek)

            for weeks_offset in range(-HOLIDAY_PROXIMITY_WEEKS, HOLIDAY_PROXIMITY_WEEKS + 1):
                target_week = thanksgiving_week_start + pd.Timedelta(weeks=weeks_offset)
                mask = (week_starts == target_week)

                if mask.any():
                    if weeks_offset < 0:
                        # Weeks before Thanksgiving
                        df.loc[mask, 'weeks_to_thanksgiving'] = abs(weeks_offset)
                        df.loc[mask, 'is_pre_thanksgiving'] = 1
                    elif weeks_offset == 0:
                        df.loc[mask, 'weeks_to_thanksgiving'] = 0
                    else:
                        # Weeks after Thanksgiving
                        df.loc[mask, 'weeks_after_thanksgiving'] = weeks_offset
                        df.loc[mask, 'is_post_thanksgiving'] = 1

            # Christmas proximity
            christmas = get_christmas_date(int(year))
            christmas_week_start = christmas - pd.Timedelta(days=christmas.dayofweek)

            for weeks_offset in range(-HOLIDAY_PROXIMITY_WEEKS, HOLIDAY_PROXIMITY_WEEKS + 1):
                target_week = christmas_week_start + pd.Timedelta(weeks=weeks_offset)
                mask = (week_starts == target_week)

                if mask.any():
                    if weeks_offset < 0:
                        df.loc[mask, 'weeks_to_christmas'] = abs(weeks_offset)
                        df.loc[mask, 'is_pre_christmas'] = 1
                    elif weeks_offset == 0:
                        df.loc[mask, 'weeks_to_christmas'] = 0
                    else:
                        df.loc[mask, 'weeks_after_christmas'] = weeks_offset
                        df.loc[mask, 'is_post_christmas'] = 1

        except Exception as e:
            logger.warning(f"Could not calculate holiday proximity for year {year}: {e}")

    # Replace -999 with a neutral value (far from holiday)
    df['weeks_to_thanksgiving'] = df['weeks_to_thanksgiving'].replace(-999, 99)
    df['weeks_after_thanksgiving'] = df['weeks_after_thanksgiving'].replace(-999, 99)
    df['weeks_to_christmas'] = df['weeks_to_christmas'].replace(-999, 99)
    df['weeks_after_christmas'] = df['weeks_after_christmas'].replace(-999, 99)

    # Log what we added
    pre_thx_count = df['is_pre_thanksgiving'].sum()
    post_thx_count = df['is_post_thanksgiving'].sum()
    thx_week_count = (df['weeks_to_thanksgiving'] == 0).sum()

    logger.info(f"Added holiday proximity features: "
                f"Thanksgiving week={thx_week_count}, pre={pre_thx_count}, post={post_thx_count}")

    return added_cols


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

    # 3. Add holiday features (daily or weekly)
    _add_holiday_features(result, date_col, frequency)

    # 4. Conditionally add YoY lag features if enough history exists
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
    base_features = [
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

    # Holiday features
    # Daily: is_holiday, is_holiday_adjacent, is_new_years, is_july4, is_christmas, is_thanksgiving_period, is_super_bowl_period
    # Weekly: is_*_week for each major holiday
    daily_holiday_features = [
        'is_holiday',
        'is_holiday_adjacent',
        'is_new_years',
        'is_july4',
        'is_christmas',
        'is_thanksgiving_period',
        'is_super_bowl_period',
    ]
    weekly_holiday_features = list(MAJOR_HOLIDAYS.values()) + ['is_super_bowl_week', 'is_black_friday_week']

    # Holiday proximity features (for better Thanksgiving/Christmas predictions)
    holiday_proximity_features = [
        'weeks_to_thanksgiving', 'weeks_after_thanksgiving',
        'weeks_to_christmas', 'weeks_after_christmas',
        'is_pre_thanksgiving', 'is_post_thanksgiving',
        'is_pre_christmas', 'is_post_christmas',
    ]

    return base_features + daily_holiday_features + weekly_holiday_features + holiday_proximity_features


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
    - Holiday week features are calculated from dates
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

    # Add holiday features (daily or weekly)
    _add_holiday_features(result, date_col, frequency)

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
