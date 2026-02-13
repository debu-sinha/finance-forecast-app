"""
Preprocessing module for time series forecasting.

This module adds generic features that improve forecasting accuracy across
all covariate-supporting algorithms (Prophet, SARIMAX, XGBoost).

Implements best practices from:
- Greykite: Data quality checks, outlier detection, anomaly handling
- Nixtla MLForecast: Automatic lag feature engineering
- AutoGluon: Data validation and automatic transformations

Design philosophy:
- Add features that are universally useful regardless of promo structure
- Conditionally add features based on data availability (e.g., YoY lags need 1+ year)
- Don't modify user's promo columns - they're already well-structured
- Keep it simple and avoid over-engineering
- Validate data quality before training

Applied to: Prophet, SARIMAX, XGBoost (models that support covariates)
NOT applied to: ARIMA, ETS (univariate models that can't use features)

Holiday Week Detection:
- For weekly data, holidays fall within a week rather than on a specific day
- This module auto-detects major US holidays and creates week-level indicators
- Helps models learn holiday-specific patterns (e.g., Thanksgiving week = +200%)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from functools import lru_cache
from dataclasses import dataclass, field
import logging

from backend.utils.logging_utils import log_io

try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# DATA QUALITY CHECKS (Greykite, AutoTS patterns)
# =============================================================================

@dataclass
class DataQualityReport:
    """Report from data quality validation."""
    is_valid: bool = True
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    transformations_applied: List[str] = field(default_factory=list)


# Minimum data requirements by frequency
MIN_DATA_REQUIREMENTS = {
    'daily': {'min_rows': 90, 'recommended_rows': 365, 'seasonal_period': 7},
    'weekly': {'min_rows': 26, 'recommended_rows': 104, 'seasonal_period': 52},
    'monthly': {'min_rows': 12, 'recommended_rows': 36, 'seasonal_period': 12},
}


@log_io
def validate_data_quality(
    df: pd.DataFrame,
    date_col: str = 'ds',
    target_col: str = 'y',
    frequency: str = 'weekly',
    auto_fix: bool = True
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Comprehensive data quality validation (Greykite/AutoTS patterns).

    Checks performed:
    1. Minimum data requirement by frequency
    2. Missing values detection and handling
    3. Duplicate dates detection
    4. Irregular time intervals (gaps)
    5. Outlier detection using IQR method
    6. Negative value detection
    7. Variance check (too low = near constant)
    8. Seasonal cycle sufficiency

    Args:
        df: Input DataFrame
        date_col: Date column name
        target_col: Target column name
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        auto_fix: Whether to automatically fix issues where possible

    Returns:
        Tuple of (cleaned DataFrame, DataQualityReport)
    """
    report = DataQualityReport()
    df = df.copy()

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"📋 DATA QUALITY VALIDATION (Greykite/AutoTS patterns)")
    logger.info(f"{'='*60}")

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
        report.transformations_applied.append("Converted date column to datetime")

    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)

    # Get requirements for frequency
    freq_req = MIN_DATA_REQUIREMENTS.get(frequency, MIN_DATA_REQUIREMENTS['weekly'])
    n_rows = len(df)
    report.stats['n_rows'] = n_rows
    report.stats['frequency'] = frequency

    # 1. Check minimum data requirement
    logger.info(f"   1. Data size: {n_rows} rows")
    if n_rows < freq_req['min_rows']:
        msg = f"Insufficient data: {n_rows} rows, need at least {freq_req['min_rows']} for {frequency}"
        report.issues.append(msg)
        report.is_valid = False
        logger.warning(f"      ❌ {msg}")
    elif n_rows < freq_req['recommended_rows']:
        msg = f"Limited data ({n_rows} rows). Recommend {freq_req['recommended_rows']}+ for {frequency}"
        report.warnings.append(msg)
        report.recommendations.append("Consider using simpler models (ARIMA, ETS)")
        logger.warning(f"      ⚠️ {msg}")
    else:
        logger.info(f"      ✅ Sufficient data ({n_rows} >= {freq_req['recommended_rows']} recommended)")

    # 2. Check for missing values
    null_count = df[target_col].isnull().sum()
    null_pct = null_count / n_rows * 100 if n_rows > 0 else 0
    report.stats['null_count'] = null_count
    report.stats['null_pct'] = null_pct

    logger.info(f"   2. Missing values: {null_count} ({null_pct:.1f}%)")
    if null_count > 0:
        if null_pct > 20:
            msg = f"High missing rate: {null_pct:.1f}% of target values are null"
            report.issues.append(msg)
            report.is_valid = False
            logger.warning(f"      ❌ {msg}")
        elif null_pct > 5:
            msg = f"Moderate missing rate: {null_pct:.1f}%"
            report.warnings.append(msg)
            logger.warning(f"      ⚠️ {msg}")

        if auto_fix and null_pct <= 20:
            df[target_col] = df[target_col].ffill()
            remaining_nulls = df[target_col].isnull().sum()
            if remaining_nulls > 0:
                df[target_col] = df[target_col].fillna(0)
            report.transformations_applied.append(f"Filled {null_count} missing values with forward fill")
            logger.info(f"      🔧 Auto-fixed: filled {null_count} missing values")
    else:
        logger.info(f"      ✅ No missing values")

    # 3. Check for duplicates
    duplicate_dates = df[date_col].duplicated().sum()
    report.stats['duplicate_dates'] = duplicate_dates

    logger.info(f"   3. Duplicate dates: {duplicate_dates}")
    if duplicate_dates > 0:
        msg = f"Found {duplicate_dates} duplicate dates"
        report.issues.append(msg)
        logger.warning(f"      ⚠️ {msg}")
        if auto_fix:
            df = df.drop_duplicates(subset=[date_col], keep='last')
            report.transformations_applied.append(f"Removed {duplicate_dates} duplicate dates")
            logger.info(f"      🔧 Auto-fixed: removed duplicates")
    else:
        logger.info(f"      ✅ No duplicate dates")

    # 4. Check for gaps in time series
    date_diff = df[date_col].diff().dropna()
    expected_freq = {'daily': 1, 'weekly': 7, 'monthly': 30}
    expected_days = expected_freq.get(frequency, 7)

    if frequency == 'daily':
        gaps = (date_diff > pd.Timedelta(days=expected_days * 2)).sum()
    else:
        gaps = (date_diff > pd.Timedelta(days=expected_days * 1.5)).sum()

    report.stats['gaps_detected'] = gaps
    logger.info(f"   4. Time gaps detected: {gaps}")
    if gaps > 0:
        msg = f"Found {gaps} gaps in time series"
        report.warnings.append(msg)
        report.recommendations.append("Consider interpolating missing periods")
        logger.warning(f"      ⚠️ {msg}")
    else:
        logger.info(f"      ✅ No significant gaps")

    # 5. Check for outliers using IQR
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    outlier_mask = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
    n_outliers = outlier_mask.sum()
    outlier_pct = n_outliers / n_rows * 100 if n_rows > 0 else 0
    report.stats['n_outliers'] = n_outliers
    report.stats['outlier_pct'] = outlier_pct

    logger.info(f"   5. Outliers (3x IQR): {n_outliers} ({outlier_pct:.1f}%)")
    if n_outliers > 0:
        if outlier_pct > 10:
            msg = f"High outlier rate: {outlier_pct:.1f}%"
            report.warnings.append(msg)
            report.recommendations.append("Consider outlier treatment or robust models")
            logger.warning(f"      ⚠️ {msg}")
        else:
            logger.info(f"      ✅ Outlier rate acceptable")

        if auto_fix and outlier_pct <= 10:
            df.loc[df[target_col] < lower_bound, target_col] = lower_bound
            df.loc[df[target_col] > upper_bound, target_col] = upper_bound
            report.transformations_applied.append(f"Winsorized {n_outliers} outliers")
            logger.info(f"      🔧 Auto-fixed: winsorized {n_outliers} outliers")
    else:
        logger.info(f"      ✅ No outliers detected")

    # 6. Check for negative values
    n_negative = (df[target_col] < 0).sum()
    report.stats['n_negative'] = n_negative

    logger.info(f"   6. Negative values: {n_negative}")
    if n_negative > 0:
        report.warnings.append(f"Found {n_negative} negative values")
        logger.warning(f"      ⚠️ Found {n_negative} negative values - verify if valid")
    else:
        logger.info(f"      ✅ No negative values")

    # 7. Check variance
    mean_val = df[target_col].mean()
    std_val = df[target_col].std()
    cv = std_val / mean_val if mean_val != 0 else 0
    report.stats['mean'] = mean_val
    report.stats['std'] = std_val
    report.stats['cv'] = cv

    logger.info(f"   7. Coefficient of variation: {cv:.3f}")
    if cv < 0.01:
        msg = f"Very low variance (CV={cv:.4f}) - near-constant data"
        report.warnings.append(msg)
        report.recommendations.append("Consider if forecasting is needed for constant data")
        logger.warning(f"      ⚠️ {msg}")
    elif cv > 2.0:
        msg = f"Very high variance (CV={cv:.2f})"
        report.warnings.append(msg)
        report.recommendations.append("Consider log transformation")
        logger.warning(f"      ⚠️ {msg}")
    else:
        logger.info(f"      ✅ Variance is reasonable")

    # 8. Check seasonal cycles
    seasonal_period = freq_req['seasonal_period']
    n_cycles = n_rows / seasonal_period
    report.stats['seasonal_cycles'] = n_cycles

    logger.info(f"   8. Seasonal cycles: {n_cycles:.1f}")
    if n_cycles < 2:
        msg = f"Less than 2 seasonal cycles ({n_cycles:.1f})"
        report.warnings.append(msg)
        report.recommendations.append("Disable yearly_seasonality, use non-seasonal models")
        logger.warning(f"      ⚠️ {msg}")
    else:
        logger.info(f"      ✅ Sufficient seasonal cycles")

    # 9. Check for incomplete trailing data (critical for accurate forecasting)
    # This detects weeks/periods with anomalously low values at the end of the series
    # which are likely incomplete data that would cause models to predict false declines
    df, incomplete_report = detect_incomplete_trailing_data(
        df, date_col, target_col, frequency, auto_fix=auto_fix
    )

    if incomplete_report['n_dropped'] > 0:
        report.stats['incomplete_periods_dropped'] = incomplete_report['n_dropped']
        report.stats['incomplete_dates'] = incomplete_report['dropped_dates']
        report.warnings.append(incomplete_report['message'])
        report.transformations_applied.append(
            f"Dropped {incomplete_report['n_dropped']} incomplete trailing period(s)"
        )

    # Summary
    logger.info(f"")
    logger.info(f"   📊 SUMMARY:")
    logger.info(f"      Valid: {'✅ YES' if report.is_valid else '❌ NO'}")
    logger.info(f"      Issues: {len(report.issues)}")
    logger.info(f"      Warnings: {len(report.warnings)}")
    logger.info(f"      Auto-fixes applied: {len(report.transformations_applied)}")
    logger.info(f"{'='*60}")

    return df, report


@log_io
def detect_incomplete_trailing_data(
    df: pd.DataFrame,
    date_col: str = 'ds',
    target_col: str = 'y',
    frequency: str = 'weekly',
    auto_fix: bool = True,
    drop_threshold: float = 0.5,
    lookback_periods: int = 4
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect and optionally remove incomplete trailing data points.

    This is critical for accurate forecasting. Incomplete data at the end of a
    time series (e.g., partial week data) causes models like Prophet to detect
    a false "crash" and extrapolate severe declines.

    Detection logic:
    1. Compare the last N periods to the rolling median of prior periods
    2. If any trailing period has < drop_threshold * median, flag as incomplete
    3. Continue checking backwards until we find complete data

    Example: If typical weekly value is ~1.3B and last week shows 76M (6% of normal),
    that week is clearly incomplete and should be excluded.

    Args:
        df: Input DataFrame sorted by date
        date_col: Date column name
        target_col: Target column name
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        auto_fix: Whether to automatically drop incomplete periods
        drop_threshold: Drop if value < threshold * rolling_median (default 0.5 = 50%)
        lookback_periods: Number of periods to use for rolling median (default 4)

    Returns:
        Tuple of (cleaned DataFrame, report dict)
    """
    report = {
        'n_dropped': 0,
        'dropped_dates': [],
        'message': '',
        'details': []
    }

    if len(df) < lookback_periods + 2:
        logger.info(f"   9. Incomplete data check: Skipped (insufficient data)")
        return df, report

    df = df.copy()
    df = df.sort_values(date_col).reset_index(drop=True)

    # Calculate rolling median excluding the last few periods
    # We use median instead of mean for robustness to outliers
    values = np.array(df[target_col].values, dtype=float)
    n = len(values)

    # Get the median of the "stable" portion (excluding last lookback_periods)
    stable_values = values[:-lookback_periods] if n > lookback_periods else values[:-1]
    rolling_median = float(np.median(stable_values))
    rolling_std = float(np.std(stable_values))

    if rolling_median <= 0:
        logger.info(f"   9. Incomplete data check: Skipped (median <= 0)")
        return df, report

    logger.info(f"   9. Incomplete trailing data check:")
    logger.info(f"      Reference median (prior periods): {rolling_median:,.0f}")
    logger.info(f"      Drop threshold: {drop_threshold*100:.0f}% of median = {rolling_median * drop_threshold:,.0f}")

    # Check trailing periods from most recent backwards
    incomplete_indices = []
    for i in range(n - 1, max(n - lookback_periods - 1, lookback_periods), -1):
        value = values[i]
        ratio = value / rolling_median
        date = df[date_col].iloc[i]

        if ratio < drop_threshold:
            incomplete_indices.append(i)
            report['details'].append({
                'date': str(date),
                'value': value,
                'ratio': ratio,
                'threshold': drop_threshold
            })
            logger.warning(f"      ⚠️ {date}: {value:,.0f} = {ratio*100:.1f}% of median (INCOMPLETE)")
        else:
            # Found a complete period, stop checking
            logger.info(f"      ✅ {date}: {value:,.0f} = {ratio*100:.1f}% of median (OK)")
            break

    if incomplete_indices:
        n_incomplete = len(incomplete_indices)
        dropped_dates = [str(df[date_col].iloc[i]) for i in incomplete_indices]

        report['n_dropped'] = n_incomplete
        report['dropped_dates'] = dropped_dates
        report['message'] = (
            f"Detected {n_incomplete} incomplete trailing period(s): {dropped_dates}. "
            f"These periods have <{drop_threshold*100:.0f}% of normal values, "
            f"indicating incomplete/partial data."
        )

        if auto_fix:
            # Drop incomplete rows
            df = df.drop(incomplete_indices).reset_index(drop=True)
            logger.warning(f"      🔧 Auto-fixed: Dropped {n_incomplete} incomplete period(s)")
            logger.info(f"         Dropped dates: {dropped_dates}")
            logger.info(f"         New data range: {df[date_col].min()} to {df[date_col].max()}")
        else:
            logger.warning(f"      ❌ Found {n_incomplete} incomplete period(s) - set auto_fix=True to remove")
    else:
        logger.info(f"      ✅ No incomplete trailing data detected")

    return df, report


# =============================================================================
# DATA AGGREGATION FOR MULTI-DIMENSIONAL TIME SERIES
# =============================================================================

@log_io
def aggregate_time_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    group_by_cols: Optional[List[str]] = None,
    agg_method: str = 'sum',
    additional_agg: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Aggregate multi-dimensional time series data by specified dimensions.

    This function handles datasets with multiple dimensions (e.g., region, product,
    segment) and aggregates them to produce a single time series suitable for
    forecasting.

    Example use cases:
    - Aggregate sales across all stores: group_by_cols=None (total)
    - Aggregate by region: group_by_cols=['region']
    - Aggregate by region and product: group_by_cols=['region', 'product']

    Args:
        df: Input DataFrame with multi-dimensional data
        date_col: Name of the date column
        target_col: Name of the target column to aggregate
        group_by_cols: List of columns to group by. If None, aggregates to total.
        agg_method: Aggregation method for target ('sum', 'mean', 'median', 'max', 'min')
        additional_agg: Dict mapping column names to aggregation methods for other columns
                       e.g., {'volume': 'sum', 'price': 'mean'}

    Returns:
        Tuple of (aggregated DataFrame, aggregation report)
    """
    report = {
        'original_rows': len(df),
        'original_columns': list(df.columns),
        'group_by_cols': group_by_cols,
        'agg_method': agg_method,
        'aggregated_rows': 0,
        'dimension_values': {}
    }

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"📊 DATA AGGREGATION")
    logger.info(f"{'='*60}")
    logger.info(f"   Original shape: {df.shape}")
    logger.info(f"   Date column: {date_col}")
    logger.info(f"   Target column: {target_col}")
    logger.info(f"   Group by: {group_by_cols if group_by_cols else 'None (aggregate to total)'}")
    logger.info(f"   Aggregation method: {agg_method}")

    df = df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Clean numeric target column (handle comma-formatted numbers)
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].astype(str).str.replace(',', '', regex=False)
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        logger.info(f"   Cleaned target column: removed comma formatting")

    # Record unique values for each dimension
    if group_by_cols:
        for col in group_by_cols:
            if col in df.columns:
                unique_vals = df[col].unique().tolist()
                report['dimension_values'][col] = unique_vals
                logger.info(f"   Dimension '{col}': {len(unique_vals)} unique values")

    # Build aggregation dict
    agg_dict = {target_col: agg_method}

    # Add additional columns to aggregate
    if additional_agg:
        for col, method in additional_agg.items():
            if col in df.columns and col != target_col:
                agg_dict[col] = method

    # Perform aggregation
    if group_by_cols:
        # Aggregate by date AND specified dimensions
        grouping_cols = [date_col] + group_by_cols
        df_agg = df.groupby(grouping_cols, as_index=False).agg(agg_dict)
    else:
        # Aggregate to total (just by date)
        df_agg = df.groupby(date_col, as_index=False).agg(agg_dict)

    # Sort by date
    df_agg = df_agg.sort_values(date_col).reset_index(drop=True)

    report['aggregated_rows'] = len(df_agg)
    report['aggregated_columns'] = list(df_agg.columns)

    logger.info(f"")
    logger.info(f"   📈 AGGREGATION RESULTS:")
    logger.info(f"      Original rows: {report['original_rows']:,}")
    logger.info(f"      Aggregated rows: {report['aggregated_rows']:,}")
    logger.info(f"      Compression ratio: {report['original_rows']/max(1, report['aggregated_rows']):.1f}x")
    logger.info(f"      Date range: {df_agg[date_col].min()} to {df_agg[date_col].max()}")
    logger.info(f"      Target range: {df_agg[target_col].min():,.0f} to {df_agg[target_col].max():,.0f}")
    logger.info(f"{'='*60}")

    return df_agg, report


@log_io
def detect_dimension_columns(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    exclude_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Automatically detect dimension columns in a multi-dimensional dataset.

    Dimension columns are categorical columns that:
    - Are not the date or target column
    - Have a limited number of unique values (< 100)
    - Are likely to be used for grouping/filtering

    This helps users understand their data structure and select appropriate
    aggregation dimensions.

    Args:
        df: Input DataFrame
        date_col: Name of the date column
        target_col: Name of the target column
        exclude_cols: Additional columns to exclude from detection

    Returns:
        Dict with detected dimensions and their characteristics
    """
    exclude = {date_col, target_col}
    if exclude_cols:
        exclude.update(exclude_cols)

    dimensions = {}
    numeric_cols = []

    for col in df.columns:
        if col in exclude:
            continue

        n_unique = df[col].nunique()

        # Check if it's a likely dimension (categorical with reasonable cardinality)
        if n_unique <= 100:
            # Determine if categorical or numeric-looking categorical
            if df[col].dtype in ['object', 'category', 'bool']:
                dim_type = 'categorical'
            elif df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                # Numeric but few unique values - likely a flag or code
                if n_unique <= 10:
                    dim_type = 'flag/code'
                else:
                    dim_type = 'numeric_categorical'
            else:
                dim_type = 'other'

            dimensions[col] = {
                'n_unique': n_unique,
                'type': dim_type,
                'sample_values': df[col].unique()[:5].tolist(),
                'null_count': df[col].isnull().sum()
            }
        elif df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            # High cardinality numeric - likely a measure, not a dimension
            numeric_cols.append(col)

    # Log detection results
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"🔍 DIMENSION DETECTION")
    logger.info(f"{'='*60}")
    logger.info(f"   Found {len(dimensions)} potential dimension column(s):")
    for col, info in dimensions.items():
        logger.info(f"      • {col}: {info['n_unique']} unique values ({info['type']})")
        logger.info(f"        Sample: {info['sample_values']}")
    if numeric_cols:
        logger.info(f"   Found {len(numeric_cols)} numeric measure column(s): {numeric_cols}")
    logger.info(f"{'='*60}")

    return {
        'dimensions': dimensions,
        'numeric_measures': numeric_cols,
        'date_col': date_col,
        'target_col': target_col
    }


@log_io
def prepare_data_with_aggregation(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    group_by_cols: Optional[List[str]] = None,
    agg_method: str = 'sum',
    filter_dimensions: Optional[Dict[str, Any]] = None,
    auto_detect_incomplete: bool = True,
    frequency: str = 'weekly',
    covariate_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete data preparation pipeline with aggregation and quality checks.

    This is the main entry point for preparing multi-dimensional data for
    forecasting. It handles:
    1. Dimension filtering (if specified)
    2. Aggregation to specified granularity (preserving covariates)
    3. Incomplete data detection
    4. Data quality validation

    Args:
        df: Raw input DataFrame
        date_col: Name of the date column
        target_col: Name of the target column
        group_by_cols: Columns to group by (None = aggregate to total)
        agg_method: Aggregation method for target
        filter_dimensions: Dict of dimension filters, e.g., {'region': 'West', 'segment': ['A', 'B']}
        auto_detect_incomplete: Whether to detect and remove incomplete trailing data
        frequency: Data frequency for validation
        covariate_cols: List of covariate columns to preserve during aggregation

    Returns:
        Tuple of (prepared DataFrame, preparation report)
    """
    report = {
        'filtering': {},
        'aggregation': {},
        'incomplete_detection': {},
        'quality': {}
    }

    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 DATA PREPARATION PIPELINE")
    logger.info(f"{'='*70}")

    df = df.copy()

    # Step 1: Apply dimension filters
    if filter_dimensions:
        original_rows = len(df)
        for col, value in filter_dimensions.items():
            if col in df.columns:
                if isinstance(value, list):
                    df = df[df[col].isin(value)]
                else:
                    df = df[df[col] == value]
                logger.info(f"   Filtered {col} = {value}: {original_rows} → {len(df)} rows")
        report['filtering'] = {
            'filters_applied': filter_dimensions,
            'rows_before': original_rows,
            'rows_after': len(df)
        }

    # Step 2: Aggregate data (preserving covariates)
    # Build additional_agg dict for covariates - use 'max' for binary event flags
    # and 'sum' for numeric covariates that should be aggregated
    additional_agg = None
    if covariate_cols:
        additional_agg = {}
        for cov in covariate_cols:
            if cov in df.columns and cov != target_col:
                # For binary event flags (0/1), use 'max' to preserve if ANY row had the event
                # For continuous covariates, use 'sum' to aggregate
                col_values = df[cov].dropna()
                if len(col_values) > 0:
                    is_binary = set(col_values.unique()).issubset({0, 1, 0.0, 1.0})
                    additional_agg[cov] = 'max' if is_binary else 'sum'
                    logger.info(f"   Covariate '{cov}' will be aggregated using '{additional_agg[cov]}'")
        if additional_agg:
            logger.info(f"   📋 Preserving {len(additional_agg)} covariate(s) during aggregation")

    df_agg, agg_report = aggregate_time_series(
        df, date_col, target_col, group_by_cols, agg_method, additional_agg
    )
    report['aggregation'] = agg_report

    # Step 3: Detect and remove incomplete trailing data
    if auto_detect_incomplete:
        df_clean, incomplete_report = detect_incomplete_trailing_data(
            df_agg, date_col, target_col, frequency
        )
        report['incomplete_detection'] = incomplete_report
        df_agg = df_clean

    # Step 4: Data quality validation
    df_validated, quality_report = validate_data_quality(
        df_agg, date_col, target_col, frequency
    )
    report['quality'] = {
        'is_valid': quality_report.is_valid,
        'issues': quality_report.issues,
        'warnings': quality_report.warnings,
        'stats': quality_report.stats
    }

    logger.info(f"\n{'='*70}")
    logger.info(f"✅ DATA PREPARATION COMPLETE")
    logger.info(f"   Final shape: {df_validated.shape}")
    logger.info(f"   Ready for forecasting: {'Yes' if quality_report.is_valid else 'No - see issues'}")
    logger.info(f"{'='*70}\n")

    return df_validated, report


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
HOLIDAY_PROXIMITY_DAYS = 14  # Track days until/since for daily data

# Key holidays with known lift effects on business metrics
# Format: (name, month, day) for fixed-date holidays
# Thanksgiving/Easter handled separately (date varies)
KEY_HOLIDAYS_FIXED = [
    ("new_years", 1, 1),
    ("valentines", 2, 14),
    ("july4", 7, 4),
    ("halloween", 10, 31),
    ("christmas_eve", 12, 24),
    ("christmas", 12, 25),
    ("new_years_eve", 12, 31),
]

# Fiscal quarter-end dates - critical for finance forecasting
# These dates often have spending spikes as budgets close out
# Q4 end (Dec 31) is covered by new_years_eve above
FISCAL_QUARTER_ENDS = [
    ("fiscal_q1_end", 3, 31),   # Q1 close
    ("fiscal_q2_end", 6, 30),   # Q2 close (half-year)
    ("fiscal_q3_end", 9, 30),   # Q3 close
]

# Minimum data requirements by frequency for lag features
LAG_DATA_REQUIREMENTS = {
    'daily': {'lag': 364, 'min_rows': 400, 'seasonal_period': 7},
    'weekly': {'lag': 52, 'min_rows': 60, 'seasonal_period': 52},
    'monthly': {'lag': 12, 'min_rows': 15, 'seasonal_period': 12},
}


@log_io
def validate_lag_data_sufficiency(
    n_rows: int,
    frequency: str,
    raise_error: bool = False
) -> Dict[str, bool]:
    """
    Validate if there's sufficient data for lag features.

    This function provides early feedback before training to help users
    understand what features will be available based on their data size.

    Args:
        n_rows: Number of data rows
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        raise_error: If True, raise ValueError when data is insufficient

    Returns:
        Dict with validation results:
        - 'can_add_yoy_lag': Whether YoY lag features can be added
        - 'can_add_seasonal': Whether seasonal features are meaningful
        - 'recommended_min_rows': Minimum recommended rows
    """
    config = LAG_DATA_REQUIREMENTS.get(frequency, LAG_DATA_REQUIREMENTS['weekly'])
    min_rows = config['min_rows']
    seasonal_period = config['seasonal_period']

    can_add_yoy_lag = n_rows >= min_rows
    can_add_seasonal = n_rows >= seasonal_period * 2  # Need 2+ cycles

    result = {
        'can_add_yoy_lag': can_add_yoy_lag,
        'can_add_seasonal': can_add_seasonal,
        'recommended_min_rows': min_rows,
        'current_rows': n_rows,
        'seasonal_period': seasonal_period,
    }

    if raise_error and not can_add_yoy_lag:
        raise ValueError(
            f"Insufficient data for {frequency} frequency: {n_rows} rows provided, "
            f"need at least {min_rows} for meaningful lag features"
        )

    if not can_add_yoy_lag:
        logger.warning(
            f"Data sufficiency warning: {n_rows} rows for {frequency} data. "
            f"YoY lag features require {min_rows}+ rows for reliable results."
        )

    return result


@log_io
def get_thanksgiving_date(year: int) -> pd.Timestamp:
    """Get the date of Thanksgiving (4th Thursday of November) for a given year."""
    first_day = pd.Timestamp(year=year, month=11, day=1)
    first_thursday = first_day + pd.Timedelta(days=(3 - first_day.dayofweek) % 7)
    fourth_thursday = first_thursday + pd.Timedelta(weeks=3)
    return fourth_thursday


@log_io
def get_christmas_date(year: int) -> pd.Timestamp:
    """Get Christmas date for a given year."""
    return pd.Timestamp(year=year, month=12, day=25)


@log_io
def get_black_friday_date(year: int) -> pd.Timestamp:
    """Get Black Friday (day after Thanksgiving) for a given year."""
    return get_thanksgiving_date(year) + pd.Timedelta(days=1)


@log_io
def get_easter_date(year: int) -> pd.Timestamp:
    """
    Calculate Easter Sunday using the Anonymous Gregorian algorithm.
    Easter is important for retail and food industries.
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


@log_io
def get_super_bowl_date(year: int) -> pd.Timestamp:
    """
    Get Super Bowl Sunday for a given year.

    NFL scheduling:
    - 2020 and earlier: First Sunday of February
    - 2021: First Sunday of February (Feb 7)
    - 2022 onwards: Second Sunday of February (NFL added 17th game, pushed back schedule)
    """
    first_day = pd.Timestamp(year=year, month=2, day=1)
    days_until_sunday = (6 - first_day.dayofweek) % 7
    first_sunday = first_day + pd.Timedelta(days=days_until_sunday)

    # Starting 2022, Super Bowl moved to second Sunday of February
    if year >= 2022:
        return first_sunday + pd.Timedelta(weeks=1)
    return first_sunday


@log_io
def get_all_key_holiday_dates(year: int) -> Dict[str, pd.Timestamp]:
    """
    Get all key holiday dates for a given year.

    Returns dict mapping holiday name to date.
    Includes both fixed-date and variable-date holidays.
    """
    holidays_dict = {}

    # Fixed-date holidays
    for name, month, day in KEY_HOLIDAYS_FIXED:
        try:
            holidays_dict[name] = pd.Timestamp(year=year, month=month, day=day)
        except ValueError as e:
            logger.warning(f"Invalid holiday date {name} for {year}: {e}")

    # Fiscal quarter-end dates - critical for finance forecasting
    for name, month, day in FISCAL_QUARTER_ENDS:
        try:
            holidays_dict[name] = pd.Timestamp(year=year, month=month, day=day)
        except ValueError as e:
            logger.warning(f"Invalid fiscal date {name} for {year}: {e}")

    # Variable-date holidays
    try:
        holidays_dict['thanksgiving'] = get_thanksgiving_date(year)
        holidays_dict['black_friday'] = get_black_friday_date(year)
        holidays_dict['easter'] = get_easter_date(year)
        holidays_dict['super_bowl'] = get_super_bowl_date(year)

        # Mother's Day (2nd Sunday of May)
        may_first = pd.Timestamp(year=year, month=5, day=1)
        first_sunday = may_first + pd.Timedelta(days=(6 - may_first.dayofweek) % 7)
        holidays_dict['mothers_day'] = first_sunday + pd.Timedelta(weeks=1)

        # Father's Day (3rd Sunday of June)
        june_first = pd.Timestamp(year=year, month=6, day=1)
        first_sunday = june_first + pd.Timedelta(days=(6 - june_first.dayofweek) % 7)
        holidays_dict['fathers_day'] = first_sunday + pd.Timedelta(weeks=2)

        # Memorial Day (last Monday of May)
        may_last = pd.Timestamp(year=year, month=5, day=31)
        days_back = (may_last.dayofweek - 0) % 7  # Monday = 0
        holidays_dict['memorial_day'] = may_last - pd.Timedelta(days=days_back)

        # Labor Day (first Monday of September)
        sept_first = pd.Timestamp(year=year, month=9, day=1)
        days_until_monday = (7 - sept_first.dayofweek) % 7
        if sept_first.dayofweek == 0:
            days_until_monday = 0
        holidays_dict['labor_day'] = sept_first + pd.Timedelta(days=days_until_monday)

    except Exception as e:
        logger.warning(f"Could not calculate some variable-date holidays for {year}: {e}")

    return holidays_dict


@log_io
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


@log_io
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


@log_io
def _add_daily_holiday_features(df: pd.DataFrame, date_col: str) -> List[str]:
    """
    Add comprehensive holiday features for daily data.

    Creates:
    - is_holiday: binary indicator (1 if date is a US holiday)
    - is_holiday_adjacent: 1 if within 1 day of a holiday
    - days_to_<holiday>: days until the next occurrence of major holiday (0-365 scale)
    - days_since_<holiday>: days since the last occurrence of major holiday
    - is_<holiday>_window: 1 if within ±3 days of holiday (multi-day effect)
    - Specific major holiday indicators
    """
    dates = pd.to_datetime(df[date_col])
    years = list(dates.dt.year.unique())

    # Get all holidays for all years in data (plus buffer years)
    all_holidays = set()
    for year in years + [min(years) - 1, max(years) + 1]:
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
    major_holiday_patterns = {
        'is_new_years': lambda d: d.month == 1 and d.day <= 2,
        'is_july4': lambda d: d.month == 7 and 3 <= d.day <= 5,
        'is_christmas': lambda d: d.month == 12 and 24 <= d.day <= 26,
        'is_thanksgiving_period': lambda d: d.month == 11 and 22 <= d.day <= 28,
        'is_super_bowl_period': lambda d: d.month == 2 and d.day <= 14 and d.weekday() == 6,
    }

    for col_name, pattern_fn in major_holiday_patterns.items():
        if col_name not in df.columns:
            df[col_name] = dates.apply(lambda d: 1 if pattern_fn(d) else 0).astype(int)
            added_cols.append(col_name)

    # =========================================================================
    # ENHANCED: Days until/since major holidays + multi-day effect windows
    # These help models learn pre-holiday ramp-up and post-holiday patterns
    # =========================================================================
    key_holidays_to_track = ['thanksgiving', 'christmas', 'black_friday', 'easter', 'super_bowl']

    # Build a lookup of all holiday dates across years
    all_holiday_dates = {}
    for year in years + [min(years) - 1, max(years) + 1]:
        year_dates = get_all_key_holiday_dates(int(year))
        for name, date in year_dates.items():
            if name not in all_holiday_dates:
                all_holiday_dates[name] = []
            all_holiday_dates[name].append(date)

    # Sort each holiday's dates
    for name in all_holiday_dates:
        all_holiday_dates[name] = sorted(all_holiday_dates[name])

    # Calculate days_to and days_since for key holidays
    for holiday_name in key_holidays_to_track:
        if holiday_name not in all_holiday_dates:
            continue

        holiday_dates = all_holiday_dates[holiday_name]
        days_to_col = f'days_to_{holiday_name}'
        days_since_col = f'days_since_{holiday_name}'
        window_col = f'is_{holiday_name}_window'

        if days_to_col not in df.columns:
            def calc_days_to(d):
                """Calculate days until next occurrence of this holiday."""
                for hd in holiday_dates:
                    if hd >= d:
                        return (hd - d).days
                return 365  # Far away

            def calc_days_since(d):
                """Calculate days since last occurrence of this holiday."""
                for hd in reversed(holiday_dates):
                    if hd <= d:
                        return (d - hd).days
                return 365  # Far away

            def calc_window(d):
                """1 if within ±3 days of this holiday (multi-day effect)."""
                for hd in holiday_dates:
                    diff = abs((d - hd).days)
                    if diff <= 3:
                        return 1
                return 0

            df[days_to_col] = dates.apply(calc_days_to)
            df[days_since_col] = dates.apply(calc_days_since)
            df[window_col] = dates.apply(calc_window)

            # Cap at reasonable values to prevent extreme outliers
            df[days_to_col] = df[days_to_col].clip(0, 365)
            df[days_since_col] = df[days_since_col].clip(0, 365)

            added_cols.extend([days_to_col, days_since_col, window_col])

    # Add is_weekend (ensure it exists for all models to use)
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
        added_cols.append('is_weekend')

    holiday_count = df['is_holiday'].sum() if 'is_holiday' in df.columns else 0
    window_counts = {col: df[col].sum() for col in added_cols if col.endswith('_window')}
    logger.info(f"Added daily holiday features: {len(added_cols)} columns ({holiday_count} holiday days)")
    if window_counts:
        logger.info(f"  Holiday windows: {window_counts}")

    return added_cols


@log_io
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

    # Vectorized Super Bowl week detection using get_super_bowl_date()
    # which correctly handles the 2022+ move to the second Sunday of February
    february_mask = dates.dt.month == 2
    if february_mask.any():
        feb_dates = dates[february_mask]
        feb_years = feb_dates.dt.year.unique()
        for year in feb_years:
            sb_date = get_super_bowl_date(int(year))
            super_bowl_week_start = sb_date - pd.Timedelta(days=sb_date.dayofweek)
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


@log_io
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


@log_io
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


@log_io
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


@log_io
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


@log_io
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


@log_io
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
    # Weekly data:
    holiday_proximity_features_weekly = [
        'weeks_to_thanksgiving', 'weeks_after_thanksgiving',
        'weeks_to_christmas', 'weeks_after_christmas',
        'is_pre_thanksgiving', 'is_post_thanksgiving',
        'is_pre_christmas', 'is_post_christmas',
    ]

    # Daily data: days_to/days_since for key holidays + multi-day windows
    key_holidays = ['thanksgiving', 'christmas', 'black_friday', 'easter', 'super_bowl']
    holiday_proximity_features_daily = []
    for holiday in key_holidays:
        holiday_proximity_features_daily.extend([
            f'days_to_{holiday}',
            f'days_since_{holiday}',
            f'is_{holiday}_window',
        ])

    return (base_features +
            daily_holiday_features +
            weekly_holiday_features +
            holiday_proximity_features_weekly +
            holiday_proximity_features_daily)


@lru_cache(maxsize=32)
@log_io
def _build_prophet_holidays_cached(
    start_year: int,
    end_year: int,
    country: str
) -> Tuple[Tuple, ...]:
    """
    Cached internal function to build holiday data.
    Returns tuple of tuples for hashability, converted to DataFrame by wrapper.
    """
    holidays_list = []

    # Define holidays with their multi-day effect windows
    holiday_configs = [
        ('thanksgiving', -1, 3), ('christmas', -7, 1), ('black_friday', 0, 2),
        ('new_years', -1, 1), ('super_bowl', -1, 0), ('july4', -1, 1),
        ('labor_day', -2, 0), ('memorial_day', -2, 0), ('easter', -1, 1),
        ('valentines', -2, 0), ('halloween', -1, 0), ('mothers_day', -1, 0),
        ('fathers_day', -1, 0),
    ]

    for year in range(start_year, end_year + 1):
        year_dates = get_all_key_holiday_dates(year)
        for holiday_name, lower_window, upper_window in holiday_configs:
            if holiday_name in year_dates:
                holidays_list.append((
                    holiday_name,
                    str(year_dates[holiday_name]),
                    lower_window,
                    upper_window
                ))

    return tuple(holidays_list)


@log_io
def build_prophet_holidays_dataframe(
    start_year: int,
    end_year: int,
    country: str = 'US'
) -> pd.DataFrame:
    """
    Build a holidays DataFrame for Prophet with lower_window and upper_window
    for multi-day effects. Results are cached for performance.

    Prophet's holidays feature allows specifying windows around each holiday:
    - lower_window: days BEFORE the holiday that are affected (negative)
    - upper_window: days AFTER the holiday that are affected (positive)

    For example, Thanksgiving with lower_window=-1 and upper_window=2 means:
    - Day before Thanksgiving is affected (shopping/prep)
    - Thanksgiving day itself
    - Black Friday (day after)
    - Saturday after (weekend shopping)

    Returns:
        DataFrame with columns: holiday, ds, lower_window, upper_window
    """
    # Use cached internal function
    cached_data = _build_prophet_holidays_cached(start_year, end_year, country)

    if not cached_data:
        return pd.DataFrame(columns=['holiday', 'ds', 'lower_window', 'upper_window'])

    # Convert cached tuples to DataFrame
    holidays_list = [
        {'holiday': h[0], 'ds': h[1], 'lower_window': h[2], 'upper_window': h[3]}
        for h in cached_data
    ]

    holidays_df = pd.DataFrame(holidays_list)
    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])

    logger.info(f"Built Prophet holidays DataFrame: {len(holidays_df)} holiday entries "
                f"for years {start_year}-{end_year} (cached)")

    return holidays_df


@log_io
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


@log_io
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

    # Fill missing with recent historical mean as fallback (last lag_periods observations)
    # Using recent mean avoids bias from older data on trending series
    if target_col in hist.columns:
        recent_mean = hist[target_col].tail(lag_periods).mean()
        fill_val = recent_mean if not np.isnan(recent_mean) else 0
    else:
        fill_val = 0
    future_df[lag_col] = future_df[lag_col].fillna(fill_val)
    future_df[f'{lag_col}_avg'] = future_df[lag_col]  # Same as lag for future

    non_null = future_df[lag_col].notna().sum()
    logger.info(f"Added future YoY lag values: {lag_col} ({non_null} values from history)")
