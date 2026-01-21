import numpy as np
import pandas as pd
import logging
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Tuple, List, Optional
# Note: sklearn metrics inlined for performance - keeping import for fallback if needed
# from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy import stats
from functools import lru_cache

logger = logging.getLogger(__name__)

# =============================================================================
# PERFORMANCE: Pre-computed constants to avoid repeated calculations
# =============================================================================
_FREQ_MAP_HUMAN = {'daily': 'D', 'weekly': 'W-MON', 'monthly': 'MS', 'yearly': 'YS'}
_DAY_NAMES = ('MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN')

# Metric thresholds for validation
_MAX_REASONABLE_MAPE = 500.0  # Above this indicates model failure
_MIN_SAMPLE_SIZE = 3  # Minimum data points for meaningful metrics


@lru_cache(maxsize=128)
def _get_t_critical_value(n: int, confidence_level: float) -> float:
    """Cached t-distribution critical value computation."""
    alpha = 1 - confidence_level
    return float(stats.t.ppf(1 - alpha / 2, df=max(1, n - 1)))


@lru_cache(maxsize=32)
def _get_z_critical_value(confidence_level: float) -> float:
    """Cached z-distribution critical value computation."""
    return float(stats.norm.ppf(1 - (1 - confidence_level) / 2))


def detect_weekly_freq_code(df: pd.DataFrame, frequency: str) -> str:
    """
    Detect the appropriate weekly frequency code based on actual data.

    For weekly data, determines which day of week the data starts on
    (e.g., W-MON for Monday-based weeks, W-SUN for Sunday-based weeks).

    Args:
        df: DataFrame with date column (expects 'ds' or datetime column)
        frequency: Data frequency ('daily', 'weekly', 'monthly', 'yearly')

    Returns:
        Pandas frequency string (e.g., 'D', 'W-MON', 'MS')
    """
    if frequency != 'weekly':
        return _FREQ_MAP_HUMAN.get(frequency, 'MS')

    try:
        if 'ds' in df.columns:
            dates = pd.to_datetime(df['ds'])
        else:
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                dates = df[date_cols[0]]
            else:
                return 'W-MON'

        if len(dates) > 0:
            day_counts = dates.dt.dayofweek.value_counts()
            most_common_day = day_counts.idxmax()
            return f"W-{_DAY_NAMES[most_common_day]}"
    except Exception:
        pass
    return 'W-MON'


def validate_weekly_alignment(
    df: pd.DataFrame,
    expected_freq_code: Optional[str] = None,
    date_col: str = 'ds',
    auto_fix: bool = False
) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
    """
    Validate and optionally fix weekly date alignment.

    ==========================================================================
    CRITICAL FIX (P1): Weekly frequency misalignment causes merge failures
    ==========================================================================
    Problem: If historical data uses Monday-based weeks but future_df uses
    Sunday-based weeks, the date merge fails silently, resulting in NaN values
    that break forecasts.

    This function:
    1. Detects the dominant day-of-week in the data
    2. Identifies any misaligned dates
    3. Optionally realigns dates to the dominant day-of-week
    ==========================================================================

    Args:
        df: DataFrame with date column
        expected_freq_code: Expected frequency code (e.g., 'W-MON'). If None, auto-detect.
        date_col: Name of the date column
        auto_fix: If True, realign misaligned dates to nearest aligned date

    Returns:
        Tuple of (is_valid, fixed_df, diagnostics_dict)
        - is_valid: True if all dates are aligned (or were fixed)
        - fixed_df: DataFrame with potentially realigned dates
        - diagnostics: Dict with alignment statistics
    """
    if date_col not in df.columns:
        logger.warning(f"validate_weekly_alignment: date column '{date_col}' not found")
        return True, df, {"error": f"date column '{date_col}' not found"}

    dates = pd.to_datetime(df[date_col])
    n_dates = len(dates)

    if n_dates == 0:
        return True, df, {"n_dates": 0, "aligned": True}

    # Count day-of-week distribution
    day_counts = dates.dt.dayofweek.value_counts()
    most_common_day = day_counts.idxmax()

    # If expected_freq_code provided, use that day; otherwise use detected
    if expected_freq_code and expected_freq_code.startswith('W-'):
        expected_day_name = expected_freq_code.split('-')[1]
        expected_day_idx = _DAY_NAMES.index(expected_day_name) if expected_day_name in _DAY_NAMES else most_common_day
    else:
        expected_day_idx = most_common_day

    # Count misaligned dates
    misaligned_mask = dates.dt.dayofweek != expected_day_idx
    n_misaligned = misaligned_mask.sum()

    diagnostics = {
        "n_dates": n_dates,
        "expected_day": _DAY_NAMES[expected_day_idx],
        "expected_freq_code": f"W-{_DAY_NAMES[expected_day_idx]}",
        "n_aligned": n_dates - n_misaligned,
        "n_misaligned": n_misaligned,
        "pct_misaligned": round(n_misaligned / n_dates * 100, 2) if n_dates > 0 else 0,
        "day_distribution": {_DAY_NAMES[k]: int(v) for k, v in day_counts.items()},
        "aligned": n_misaligned == 0,
        "auto_fixed": False
    }

    # Log diagnostics
    if n_misaligned > 0:
        logger.warning(
            f"⚠️ WEEKLY ALIGNMENT ISSUE: {n_misaligned}/{n_dates} dates ({diagnostics['pct_misaligned']}%) "
            f"not aligned to {_DAY_NAMES[expected_day_idx]}. Distribution: {diagnostics['day_distribution']}"
        )

        if auto_fix:
            # Realign misaligned dates to the nearest expected day-of-week
            # This shifts dates forward to the next occurrence of the expected day
            fixed_df = df.copy()
            fixed_dates = dates.copy()

            for idx in dates[misaligned_mask].index:
                current_date = dates.loc[idx]
                current_dow = current_date.dayofweek
                # Calculate days to add to reach expected_day_idx
                days_ahead = expected_day_idx - current_dow
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                # Actually, shift to NEAREST (could be backward or forward)
                days_back = current_dow - expected_day_idx
                if days_back < 0:
                    days_back += 7
                # Choose closer direction
                if days_ahead <= days_back:
                    fixed_dates.loc[idx] = current_date + pd.Timedelta(days=days_ahead)
                else:
                    fixed_dates.loc[idx] = current_date - pd.Timedelta(days=days_back)

            fixed_df[date_col] = fixed_dates
            diagnostics["auto_fixed"] = True
            logger.info(f"✅ Auto-fixed {n_misaligned} misaligned dates to {_DAY_NAMES[expected_day_idx]}")
            return True, fixed_df, diagnostics
    else:
        logger.info(f"✅ Weekly alignment validated: all {n_dates} dates are on {_DAY_NAMES[expected_day_idx]}")

    return n_misaligned == 0, df, diagnostics


def align_forecast_dates_to_data(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    frequency: str,
    date_col: str = 'ds'
) -> pd.DataFrame:
    """
    Ensure forecast dates are aligned with historical data's date pattern.

    This is critical for weekly data where the day-of-week must match for
    proper merging between actuals and predictions.

    Args:
        forecast_df: DataFrame with forecast dates
        historical_df: DataFrame with historical dates (ground truth for alignment)
        frequency: Data frequency ('weekly', etc.)
        date_col: Name of the date column

    Returns:
        DataFrame with aligned dates
    """
    if frequency != 'weekly':
        return forecast_df

    if date_col not in historical_df.columns or date_col not in forecast_df.columns:
        return forecast_df

    # Detect alignment from historical data
    hist_freq_code = detect_weekly_freq_code(historical_df, frequency)

    # Validate and fix forecast dates
    is_valid, fixed_forecast, diagnostics = validate_weekly_alignment(
        forecast_df,
        expected_freq_code=hist_freq_code,
        date_col=date_col,
        auto_fix=True
    )

    if diagnostics.get("auto_fixed"):
        logger.info(f"📅 Aligned forecast dates to match historical data ({hist_freq_code})")

    return fixed_forecast

def safe_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Symmetric MAPE (SMAPE) - handles zero values gracefully.

    SMAPE = 2 * |y_true - y_pred| / (|y_true| + |y_pred| + epsilon) * 100

    Properties:
    - Bounded between 0% and 200% (we report 0-100 scale by dividing by 2)
    - Symmetric: error for predicting 100 when actual is 50 equals
      error for predicting 50 when actual is 100
    - Handles zeros: No division by zero issues
    - Comparable across segments with different scales

    Returns SMAPE as percentage (0-100), not decimal.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # SMAPE formula with epsilon to prevent division by zero
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    smape = np.abs(y_true - y_pred) / denominator * 100  # Scale to 0-100%

    # Cap extreme values
    smape = np.clip(smape, 0, 100)

    return float(np.mean(smape))


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute MAPE with protection against division by zero.

    When y_true contains zeros:
    - If ALL values are zero, uses SMAPE as fallback (maintains percentage scale)
    - If SOME values are zero, excludes those points from MAPE calculation
    - Uses epsilon safeguard for near-zero values

    Returns MAPE as percentage (0-100+), not decimal.

    NOTE: For cross-model comparison where segments may have zeros, consider
    using safe_smape() directly for consistent behavior.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Handle zero actuals - exclude from calculation to avoid inf
    non_zero_mask = np.abs(y_true) > epsilon

    if not np.any(non_zero_mask):
        # ==========================================================================
        # CRITICAL FIX: Use SMAPE instead of MAE for zero-actual fallback
        # ==========================================================================
        # Previous behavior: returned MAE which is in absolute units (not percentage)
        # This made cross-model/cross-segment comparison invalid when one segment
        # had zero actuals.
        #
        # New behavior: Use SMAPE which stays in percentage scale (0-100)
        # This ensures metric comparability across all segments/models.
        # ==========================================================================
        logger.warning("MAPE undefined (all actuals near zero) - falling back to SMAPE")
        return safe_smape(y_true, y_pred, epsilon)

    # Filter to non-zero actuals only
    y_true_safe = y_true[non_zero_mask]
    y_pred_safe = y_pred[non_zero_mask]

    # Calculate MAPE on valid points
    ape = np.abs((y_true_safe - y_pred_safe) / y_true_safe) * 100

    # Cap extreme values to prevent inf/nan propagation
    ape = np.clip(ape, 0, 1000)  # Cap at 1000% error

    mape = float(np.mean(ape))

    # Log warning if many points were excluded
    excluded_count = len(y_true) - len(y_true_safe)
    if excluded_count > 0:
        pct_excluded = (excluded_count / len(y_true)) * 100
        if pct_excluded > 20:
            # If more than 20% excluded, use SMAPE for more representative metric
            logger.warning(f"MAPE: {excluded_count}/{len(y_true)} ({pct_excluded:.1f}%) zero-value points - using SMAPE instead")
            return safe_smape(y_true, y_pred, epsilon)
        else:
            logger.warning(f"MAPE: excluded {excluded_count}/{len(y_true)} zero-value points from calculation")

    return mape


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute forecast accuracy metrics with robust handling.

    Handles edge cases:
    - Division by zero in MAPE (uses safe_mape)
    - NaN/Inf values (clips to valid range)
    - Empty arrays (returns default metrics)
    - Mismatched array lengths (truncates to shorter)

    Performance: Uses vectorized NumPy operations throughout.
    """
    # Convert to arrays with consistent dtype
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    # Handle length mismatch
    min_len = min(len(y_true), len(y_pred))
    if len(y_true) != len(y_pred):
        logger.warning(f"compute_metrics: Length mismatch (true={len(y_true)}, pred={len(y_pred)}). Truncating to {min_len}.")
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    # Handle empty arrays - fast path
    if min_len == 0:
        logger.warning("compute_metrics: Empty arrays provided")
        return {"rmse": 0.0, "mape": 100.0, "r2": 0.0}

    # Handle NaN/Inf values - vectorized mask
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid_mask.all():
        nan_count = (~valid_mask).sum()
        logger.warning(f"compute_metrics: Removing {nan_count} NaN/Inf values")
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        logger.warning("compute_metrics: No valid data points after filtering")
        return {"rmse": 0.0, "mape": 100.0, "r2": 0.0}

    # Warn if sample size is very small
    if len(y_true) < _MIN_SAMPLE_SIZE:
        logger.warning(f"compute_metrics: Only {len(y_true)} data points (< {_MIN_SAMPLE_SIZE}). Metrics may be unreliable.")

    # Compute RMSE - inline for performance (avoid sklearn overhead for simple case)
    diff = y_true - y_pred
    rmse = float(np.sqrt(np.mean(diff * diff)))

    # Compute MAPE with zero protection
    mape = safe_mape(y_true, y_pred)

    # Compute R2 - inline for performance
    ss_res = np.sum(diff * diff)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot > 0:
        r2 = float(1.0 - (ss_res / ss_tot))
        r2 = max(-1.0, min(1.0, r2))
    else:
        r2 = 0.0

    # Final sanitization - ensure valid output
    rmse = 0.0 if not np.isfinite(rmse) else round(rmse, 2)
    mape = 100.0 if not np.isfinite(mape) else round(mape, 2)
    r2 = 0.0 if not np.isfinite(r2) else round(r2, 4)

    # Warn if MAPE is unreasonably high (potential model failure)
    if mape > _MAX_REASONABLE_MAPE:
        logger.warning(f"compute_metrics: MAPE={mape:.2f}% exceeds threshold ({_MAX_REASONABLE_MAPE}%). Possible model failure.")
        mape = min(mape, _MAX_REASONABLE_MAPE)  # Cap at threshold for UI display

    return {
        "rmse": rmse,
        "mape": mape,
        "r2": r2
    }

def time_series_cross_validate(
    y: np.ndarray,
    model_fit_fn,
    model_predict_fn,
    n_splits: int = 3,
    min_train_size: Optional[int] = None,
    horizon: int = 1,
    gap: int = 0
) -> Dict[str, Any]:
    """
    Perform time series cross-validation with expanding window and optional gap.

    Args:
        y: Time series values
        model_fit_fn: Function to fit model, takes y_train returns fitted model
        model_predict_fn: Function to predict, takes (fitted_model, steps) returns predictions
        n_splits: Number of CV folds
        min_train_size: Minimum training set size
        horizon: Forecast horizon (test set size per fold)
        gap: Number of periods to skip between train and test (embargo/purging)
             Prevents data leakage from overlapping information sets

    Returns:
        Dict with cv_scores, mean_mape, std_mape, n_splits
    """
    n = len(y)
    if min_train_size is None:
        min_train_size = max(n // 2, 10)  # At least 50% or 10 points

    # Account for gap in available data calculation
    available_for_cv = n - min_train_size - gap
    if available_for_cv < n_splits * horizon:
        n_splits = max(1, available_for_cv // horizon)
        logger.warning(f"Reduced CV splits to {n_splits} due to limited data (gap={gap})")

    if n_splits < 1:
        logger.warning("Not enough data for cross-validation, using simple holdout")
        return {"cv_scores": [], "mean_mape": None, "std_mape": None, "n_splits": 0}

    fold_size = (n - min_train_size - gap) // n_splits
    cv_scores = []

    for i in range(n_splits):
        split_point = min_train_size + i * fold_size
        # Apply gap/embargo: skip 'gap' periods after training data
        test_start = split_point + gap
        test_end = min(test_start + horizon, n)

        y_train = y[:split_point]
        y_test = y[test_start:test_end]

        if len(y_test) == 0:
            continue

        try:
            fitted_model = model_fit_fn(y_train)
            predictions = model_predict_fn(fitted_model, len(y_test))

            # Use safe_mape to handle division by zero
            fold_mape = safe_mape(y_test, predictions)
            cv_scores.append(fold_mape)
            logger.info(f"  CV Fold {i+1}/{n_splits}: train={len(y_train)}, gap={gap}, test={len(y_test)}, MAPE={fold_mape:.2f}%")
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

    Uses cached critical values for performance when called repeatedly.
    """
    # Ensure arrays for vectorized operations
    y_train = np.asarray(y_train, dtype=np.float64)
    y_pred_train = np.asarray(y_pred_train, dtype=np.float64)
    forecast_values = np.asarray(forecast_values, dtype=np.float64)

    residuals = y_train - y_pred_train
    residual_std = np.std(residuals)
    n = len(residuals)

    # Use cached critical values for performance
    if n < 30:
        critical_value = _get_t_critical_value(n, confidence_level)
    else:
        critical_value = _get_z_critical_value(confidence_level)

    margin = critical_value * residual_std

    # Vectorized bounds computation
    lower_bounds = forecast_values - margin
    upper_bounds = forecast_values + margin

    # Ensure lower bounds are at least 10% of forecast for positive values
    if np.all(forecast_values > 0):
        lower_bounds = np.maximum(lower_bounds, forecast_values * 0.1)

    return lower_bounds, upper_bounds


def clip_forecast_non_negative(
    forecast_values: np.ndarray,
    lower_bounds: np.ndarray = None,
    upper_bounds: np.ndarray = None,
    min_value: float = 0.0
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Clip forecast values and confidence intervals to non-negative values.

    Financial metrics (revenue, volume, counts) cannot be negative.
    This function ensures all forecasts are >= min_value.

    Args:
        forecast_values: Array of forecast predictions
        lower_bounds: Optional lower confidence bounds
        upper_bounds: Optional upper confidence bounds
        min_value: Minimum allowed value (default 0.0)

    Returns:
        Tuple of (clipped_forecast, clipped_lower, clipped_upper)
    """
    forecast_values = np.asarray(forecast_values)

    # Count negative values for logging
    neg_count = np.sum(forecast_values < min_value)
    if neg_count > 0:
        logger.warning(f"Clipping {neg_count}/{len(forecast_values)} negative forecast values to {min_value}")

    # Clip forecast
    clipped_forecast = np.maximum(forecast_values, min_value)

    # Clip confidence bounds if provided
    clipped_lower = None
    clipped_upper = None

    if lower_bounds is not None:
        lower_bounds = np.asarray(lower_bounds)
        clipped_lower = np.maximum(lower_bounds, min_value)

    if upper_bounds is not None:
        upper_bounds = np.asarray(upper_bounds)
        clipped_upper = np.maximum(upper_bounds, clipped_forecast)  # Upper must be >= forecast

    return clipped_forecast, clipped_lower, clipped_upper


def sanitize_forecast_output(
    forecast_df: pd.DataFrame,
    clip_negative: bool = True,
    min_value: float = 0.0
) -> pd.DataFrame:
    """
    Sanitize forecast DataFrame to ensure valid, JSON-serializable output.

    Performs:
    1. Clips negative values to min_value (for financial data)
    2. Replaces NaN/Inf with interpolated or default values
    3. Ensures date column is string format
    4. Validates column types

    Args:
        forecast_df: DataFrame with 'ds', 'yhat', optionally 'yhat_lower', 'yhat_upper'
        clip_negative: Whether to clip negative values
        min_value: Minimum allowed value for clipping

    Returns:
        Sanitized DataFrame
    """
    df = forecast_df.copy()

    # Ensure ds is string
    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds']).dt.strftime('%Y-%m-%d')

    # Process numeric columns
    numeric_cols = ['yhat', 'yhat_lower', 'yhat_upper', 'forecast', 'lower', 'upper']
    for col in numeric_cols:
        if col in df.columns:
            # Replace NaN/Inf
            if df[col].isna().any() or np.isinf(df[col]).any():
                logger.warning(f"Replacing NaN/Inf values in {col}")
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

            # Clip negative values
            if clip_negative:
                neg_mask = df[col] < min_value
                if neg_mask.any():
                    logger.warning(f"Clipping {neg_mask.sum()} negative values in {col}")
                    df[col] = df[col].clip(lower=min_value)

    return df


def register_model_to_unity_catalog(model_uri: str, model_name: str, tags: Optional[Dict[str, str]] = None) -> str:
    """Register a model to Unity Catalog with improved error handling"""
    try:
        run_id = tags.get("run_id") if tags else (model_uri.split("/")[1] if model_uri.startswith("runs:/") and len(model_uri.split("/")) > 1 else None)
        client = MlflowClient()
        
        if run_id:
            try:
                logger.info(f"Checking if run {run_id} is already registered as {model_name}...")
                for v in client.search_model_versions(f"name='{model_name}'"):
                    if v.run_id == run_id:
                        logger.info(f"♻️  Model already registered as version {v.version}, skipping...")
                        return str(v.version)
            except Exception as check_error:
                logger.warning(f"Could not check existing versions: {check_error}")

        if "." in model_name:
            logger.info("Configuring MLflow to use Unity Catalog registry (databricks-uc)")
            mlflow.set_registry_uri("databricks-uc")
            
        logger.info(f"Registering model from {model_uri} to {model_name}...")
        result = None
        try:
            result = mlflow.register_model(model_uri=model_uri, name=model_name, tags=tags)
            logger.info(f"Successfully registered as version {result.version} with tags")
            
            if tags and result.version:
                try:
                    version_str = str(result.version)
                    mv = client.get_model_version(name=model_name, version=version_str)
                    actual_tags = mv.tags if hasattr(mv, 'tags') and mv.tags else {}
                    if not actual_tags:
                        logger.warning(f"    Tags were not set via register_model, trying client API...")
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
            error_str = str(reg_error).lower()
            if 'tag' in error_str or 'permission_denied' in error_str or 'tag assignment' in error_str:
                logger.info(f"ℹ️  Tag registration via register_model failed, registering without tags then adding tags via client API...")
                try:
                    result = mlflow.register_model(model_uri=model_uri, name=model_name)
                    logger.info(f"Successfully registered as version {result.version}")
                    
                    if tags and result.version:
                        version_str = str(result.version)
                        for tag_key, tag_value in tags.items():
                            try:
                                client.set_model_version_tag(
                                    name=model_name,
                                    version=version_str,
                                    key=tag_key,
                                    value=str(tag_value)
                                )
                                logger.info(f"   ✓ Added tag: {tag_key}={tag_value} to {model_name} version {version_str}")
                            except Exception as tag_error:
                                error_str = str(tag_error).lower()
                                if 'tag assignment' in error_str or 'tag policy' in error_str or 'permission_denied' in error_str:
                                    logger.info(f"   ℹ️  Skipped tag {tag_key} (restricted by tag policies)")
                                else:
                                    logger.warning(f"   ✗ Failed to add tag {tag_key}: {str(tag_error)[:100]}")
                    return str(result.version)
                except Exception as e:
                    logger.error(f"Registration failed even without tags: {e}")
                    raise e
            else:
                raise reg_error
                
    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        return "0"

def log_artifact_with_validation(artifact_path: str, artifact_dir: str, description: str) -> bool:
    """
    Log an artifact to MLflow with validation and detailed logging.

    Args:
        artifact_path: Local path to the artifact file
        artifact_dir: Directory in MLflow to store the artifact
        description: Human-readable description of the artifact

    Returns:
        True if successful, False otherwise
    """
    import os
    try:
        if not os.path.exists(artifact_path):
            logger.warning(f"   ⚠️ Artifact not found: {artifact_path}")
            return False

        file_size = os.path.getsize(artifact_path)
        file_name = os.path.basename(artifact_path)

        # Log the artifact
        mlflow.log_artifact(artifact_path, artifact_dir)

        # Log success with details
        logger.info(f"   ✅ Logged {description}: {artifact_dir}/{file_name} ({file_size:,} bytes)")
        return True

    except Exception as e:
        logger.error(f"   ❌ Failed to log {description}: {e}")
        return False


def log_model_with_validation(
    model_name: str,
    artifact_path: str,
    python_model,
    signature,
    input_example: pd.DataFrame,
    pip_requirements: List[str] = None
) -> bool:
    """
    Log a model to MLflow with validation and detailed logging of signature and input example.

    Args:
        model_name: Name for the model (e.g., "Prophet", "XGBoost")
        artifact_path: Path in MLflow to store the model
        python_model: The model wrapper/object to log
        signature: MLflow model signature
        input_example: Example input DataFrame
        pip_requirements: List of pip requirements

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"")
        logger.info(f"   {'='*50}")
        logger.info(f"   📦 LOGGING {model_name.upper()} MODEL TO MLFLOW")
        logger.info(f"   {'='*50}")

        # Log signature details
        if signature:
            logger.info(f"   📝 Model Signature:")
            if hasattr(signature, 'inputs') and signature.inputs:
                logger.info(f"      Inputs: {signature.inputs}")
            if hasattr(signature, 'outputs') and signature.outputs:
                logger.info(f"      Outputs: {signature.outputs}")
        else:
            logger.warning(f"   ⚠️ No signature provided")

        # Log input example details
        if input_example is not None and len(input_example) > 0:
            logger.info(f"   📋 Input Example:")
            logger.info(f"      Shape: {input_example.shape}")
            logger.info(f"      Columns: {list(input_example.columns)}")
            logger.info(f"      Dtypes: {dict(input_example.dtypes)}")
            logger.info(f"      Sample row: {input_example.iloc[0].to_dict()}")
        else:
            logger.warning(f"   ⚠️ No input example provided")

        # Log pip requirements
        if pip_requirements:
            logger.info(f"   📦 Pip Requirements: {pip_requirements}")

        # Perform the actual model logging
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=python_model,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements
        )

        logger.info(f"   ✅ Model logged successfully to: {artifact_path}")
        logger.info(f"   {'='*50}")
        return True

    except Exception as e:
        logger.error(f"   ❌ Failed to log model: {e}")
        return False


def validate_mlflow_run_artifacts(run_id: str) -> Dict[str, Any]:
    """
    Validate that all expected artifacts were logged to an MLflow run.

    Args:
        run_id: The MLflow run ID to validate

    Returns:
        Dictionary with validation results
    """
    try:
        client = MlflowClient()

        logger.info(f"")
        logger.info(f"   {'='*50}")
        logger.info(f"   🔍 VALIDATING MLFLOW RUN ARTIFACTS")
        logger.info(f"   {'='*50}")
        logger.info(f"   Run ID: {run_id}")

        # List all artifacts
        artifacts = client.list_artifacts(run_id)

        validation_result = {
            "run_id": run_id,
            "artifacts_found": [],
            "model_logged": False,
            "datasets_logged": False,
            "validation_passed": True,
            "issues": []
        }

        def list_artifacts_recursive(path=""):
            """Recursively list all artifacts"""
            items = client.list_artifacts(run_id, path)
            all_artifacts = []
            for item in items:
                if item.is_dir:
                    all_artifacts.extend(list_artifacts_recursive(item.path))
                else:
                    all_artifacts.append(item.path)
            return all_artifacts

        all_artifact_paths = list_artifacts_recursive()
        validation_result["artifacts_found"] = all_artifact_paths

        logger.info(f"   Found {len(all_artifact_paths)} artifacts:")
        for artifact_path in sorted(all_artifact_paths):
            logger.info(f"      - {artifact_path}")

        # Check for model
        model_artifacts = [a for a in all_artifact_paths if 'model' in a.lower() and ('MLmodel' in a or 'model.pkl' in a or 'python_model.pkl' in a)]
        if model_artifacts:
            validation_result["model_logged"] = True
            logger.info(f"   ✅ Model artifacts found: {model_artifacts}")
        else:
            validation_result["model_logged"] = False
            validation_result["issues"].append("No model artifacts found")
            logger.warning(f"   ⚠️ No model artifacts found")

        # Check for datasets
        dataset_artifacts = [a for a in all_artifact_paths if 'datasets' in a.lower() or 'data' in a.lower()]
        if dataset_artifacts:
            validation_result["datasets_logged"] = True
            logger.info(f"   ✅ Dataset artifacts found: {len(dataset_artifacts)} files")
        else:
            validation_result["datasets_logged"] = False
            validation_result["issues"].append("No dataset artifacts found")
            logger.warning(f"   ⚠️ No dataset artifacts found")

        # Check for signature
        mlmodel_path = [a for a in all_artifact_paths if 'MLmodel' in a]
        if mlmodel_path:
            logger.info(f"   ✅ MLmodel file found (contains signature)")

        validation_result["validation_passed"] = len(validation_result["issues"]) == 0

        if validation_result["validation_passed"]:
            logger.info(f"   ✅ VALIDATION PASSED - All expected artifacts found")
        else:
            logger.warning(f"   ⚠️ VALIDATION ISSUES: {validation_result['issues']}")

        logger.info(f"   {'='*50}")

        return validation_result

    except Exception as e:
        logger.error(f"   ❌ Artifact validation failed: {e}")
        return {
            "run_id": run_id,
            "validation_passed": False,
            "error": str(e)
        }


def analyze_covariate_impact(
    model,  # Prophet model (type hint removed for lazy import)
    df: pd.DataFrame,
    covariates: List[str]
) -> List[Dict[str, Any]]:
    """
    Analyze the impact of each covariate on the forecast (Prophet only)
    """
    # Lazy import to avoid dependency issues if Prophet is not installed/used
    try:
        from prophet import Prophet
    except ImportError:
        return []
    
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
                    std = df[cov].std() if cov in df.columns else 1
                    impact_score = abs(coef * std)

                    impacts.append({
                        'name': cov,
                        'coefficient': float(coef),
                        'impact_score': float(impact_score),
                        'direction': 'positive' if coef > 0 else 'negative'
                    })

        # Sort by impact score
        impacts.sort(key=lambda x: x['impact_score'], reverse=True)

        # Add normalized score (0-100) based on relative impact
        if impacts:
            max_impact = max(i['impact_score'] for i in impacts) if impacts else 1
            for impact in impacts:
                impact['score'] = float(min(100, (impact['impact_score'] / max_impact * 100) if max_impact > 0 else 0))

    except Exception as e:
        logger.warning(f"Could not analyze covariate impacts: {e}")

    return impacts


def compute_segment_mape_summary(
    segment_results: List[Dict[str, Any]],
    mape_threshold_good: float = 10.0,
    mape_threshold_acceptable: float = 20.0
) -> Dict[str, Any]:
    """
    Compute segment-level MAPE summary for batch training results.

    ==========================================================================
    P2 FIX: Segment-level MAPE tracking for identifying worst performers
    ==========================================================================
    Finance teams need to know which segments (regions, products, etc.) have
    the worst forecast accuracy so they can:
    1. Prioritize manual review for high-error segments
    2. Identify segments that may need more data or different models
    3. Set appropriate confidence levels for downstream decisions
    ==========================================================================

    Args:
        segment_results: List of dicts with 'segment_id', 'mape', and optionally
                        'filters', 'model_name', 'rmse', 'r2'
        mape_threshold_good: MAPE below this is considered "good" (default 10%)
        mape_threshold_acceptable: MAPE below this is "acceptable" (default 20%)

    Returns:
        Dict with:
        - segments_ranked: List of segments sorted by MAPE (worst first)
        - aggregate_stats: Mean, median, std, min, max MAPE across segments
        - quality_distribution: Count of good/acceptable/poor segments
        - worst_segments: Top 5 worst performing segments
        - best_segments: Top 5 best performing segments
    """
    if not segment_results:
        return {
            "segments_ranked": [],
            "aggregate_stats": {},
            "quality_distribution": {"good": 0, "acceptable": 0, "poor": 0},
            "worst_segments": [],
            "best_segments": []
        }

    # Extract segments with valid MAPE values
    valid_segments = []
    for seg in segment_results:
        mape = seg.get('mape')
        if mape is not None and not np.isnan(mape) and not np.isinf(mape):
            valid_segments.append({
                'segment_id': seg.get('segment_id', 'unknown'),
                'mape': float(mape),
                'rmse': seg.get('rmse'),
                'r2': seg.get('r2'),
                'model_name': seg.get('model_name', 'unknown'),
                'filters': seg.get('filters', {}),
                'data_points': seg.get('data_points'),
            })

    if not valid_segments:
        logger.warning("compute_segment_mape_summary: No valid MAPE values found")
        return {
            "segments_ranked": [],
            "aggregate_stats": {},
            "quality_distribution": {"good": 0, "acceptable": 0, "poor": 0},
            "worst_segments": [],
            "best_segments": []
        }

    # Sort by MAPE (worst first)
    segments_ranked = sorted(valid_segments, key=lambda x: x['mape'], reverse=True)

    # Add rank and quality category
    for i, seg in enumerate(segments_ranked):
        seg['rank'] = i + 1
        if seg['mape'] <= mape_threshold_good:
            seg['quality'] = 'good'
        elif seg['mape'] <= mape_threshold_acceptable:
            seg['quality'] = 'acceptable'
        else:
            seg['quality'] = 'poor'

    # Compute aggregate statistics
    mape_values = [s['mape'] for s in valid_segments]
    aggregate_stats = {
        'mean_mape': round(np.mean(mape_values), 2),
        'median_mape': round(np.median(mape_values), 2),
        'std_mape': round(np.std(mape_values), 2),
        'min_mape': round(np.min(mape_values), 2),
        'max_mape': round(np.max(mape_values), 2),
        'total_segments': len(valid_segments)
    }

    # Quality distribution
    quality_distribution = {
        'good': sum(1 for s in segments_ranked if s['quality'] == 'good'),
        'acceptable': sum(1 for s in segments_ranked if s['quality'] == 'acceptable'),
        'poor': sum(1 for s in segments_ranked if s['quality'] == 'poor')
    }

    # Log summary
    logger.info(f"")
    logger.info(f"📊 SEGMENT-LEVEL MAPE SUMMARY")
    logger.info(f"   Total segments: {aggregate_stats['total_segments']}")
    logger.info(f"   Mean MAPE: {aggregate_stats['mean_mape']}%")
    logger.info(f"   Median MAPE: {aggregate_stats['median_mape']}%")
    logger.info(f"   Range: {aggregate_stats['min_mape']}% - {aggregate_stats['max_mape']}%")
    logger.info(f"   Quality: {quality_distribution['good']} good, {quality_distribution['acceptable']} acceptable, {quality_distribution['poor']} poor")

    if quality_distribution['poor'] > 0:
        logger.warning(f"   ⚠️ {quality_distribution['poor']} segments have MAPE > {mape_threshold_acceptable}%")
        for seg in segments_ranked[:min(3, quality_distribution['poor'])]:
            if seg['quality'] == 'poor':
                logger.warning(f"      - {seg['segment_id']}: {seg['mape']}% MAPE")

    return {
        "segments_ranked": segments_ranked,
        "aggregate_stats": aggregate_stats,
        "quality_distribution": quality_distribution,
        "worst_segments": segments_ranked[:5],  # Top 5 worst
        "best_segments": segments_ranked[-5:][::-1] if len(segments_ranked) >= 5 else segments_ranked[::-1]  # Top 5 best
    }


def format_segment_mape_report(summary: Dict[str, Any]) -> str:
    """
    Format segment MAPE summary as a human-readable report.

    Args:
        summary: Output from compute_segment_mape_summary()

    Returns:
        Formatted string report
    """
    if not summary.get('segments_ranked'):
        return "No segment data available for MAPE report."

    lines = [
        "=" * 60,
        "SEGMENT FORECAST ACCURACY REPORT",
        "=" * 60,
        "",
        f"Total Segments Analyzed: {summary['aggregate_stats']['total_segments']}",
        "",
        "AGGREGATE STATISTICS:",
        f"  Mean MAPE:   {summary['aggregate_stats']['mean_mape']}%",
        f"  Median MAPE: {summary['aggregate_stats']['median_mape']}%",
        f"  Std Dev:     {summary['aggregate_stats']['std_mape']}%",
        f"  Range:       {summary['aggregate_stats']['min_mape']}% - {summary['aggregate_stats']['max_mape']}%",
        "",
        "QUALITY DISTRIBUTION:",
        f"  Good (≤10%):       {summary['quality_distribution']['good']} segments",
        f"  Acceptable (≤20%): {summary['quality_distribution']['acceptable']} segments",
        f"  Poor (>20%):       {summary['quality_distribution']['poor']} segments",
        "",
    ]

    if summary['worst_segments']:
        lines.extend([
            "WORST PERFORMING SEGMENTS (needs attention):",
            "-" * 40,
        ])
        for seg in summary['worst_segments'][:5]:
            lines.append(f"  {seg['rank']:3d}. {seg['segment_id'][:40]:<40} MAPE: {seg['mape']:6.2f}%  [{seg['quality'].upper()}]")

    if summary['best_segments']:
        lines.extend([
            "",
            "BEST PERFORMING SEGMENTS:",
            "-" * 40,
        ])
        for i, seg in enumerate(summary['best_segments'][:5]):
            lines.append(f"  {i+1:3d}. {seg['segment_id'][:40]:<40} MAPE: {seg['mape']:6.2f}%  [{seg['quality'].upper()}]")

    lines.extend(["", "=" * 60])

    return "\n".join(lines)


def compute_data_quality_summary(
    df: pd.DataFrame,
    date_col: str = 'ds',
    target_col: str = 'y',
    frequency: str = 'monthly'
) -> Dict[str, Any]:
    """
    Compute data quality summary with quick wins recommendations.

    Provides insights about:
    - Data volume and date range
    - Missing value analysis
    - Zero value analysis (important for MAPE)
    - Seasonality indicators
    - Quick wins recommendations

    Args:
        df: DataFrame with time series data
        date_col: Name of the date column
        target_col: Name of the target column
        frequency: Data frequency ('daily', 'weekly', 'monthly')

    Returns:
        Dict with data quality metrics and recommendations
    """
    summary = {
        'volume': {},
        'date_range': {},
        'missing_values': {},
        'zero_values': {},
        'statistics': {},
        'quick_wins': [],
        'warnings': []
    }

    # Volume metrics
    n_rows = len(df)
    summary['volume'] = {
        'total_rows': n_rows,
        'is_sufficient': n_rows >= 24,  # At least 2 years monthly or equivalent
        'recommended_min': 24 if frequency == 'monthly' else (52 if frequency == 'weekly' else 365)
    }

    # Date range
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
        summary['date_range'] = {
            'start': dates.min().strftime('%Y-%m-%d'),
            'end': dates.max().strftime('%Y-%m-%d'),
            'span_days': (dates.max() - dates.min()).days,
            'span_years': round((dates.max() - dates.min()).days / 365.25, 2)
        }

        # Check for gaps
        if frequency == 'monthly':
            expected_periods = ((dates.max().year - dates.min().year) * 12 +
                               dates.max().month - dates.min().month + 1)
        elif frequency == 'weekly':
            expected_periods = (dates.max() - dates.min()).days // 7 + 1
        else:
            expected_periods = (dates.max() - dates.min()).days + 1

        actual_periods = len(dates.unique())
        gap_count = expected_periods - actual_periods
        summary['date_range']['expected_periods'] = expected_periods
        summary['date_range']['actual_periods'] = actual_periods
        summary['date_range']['gap_count'] = max(0, gap_count)

        if gap_count > 0:
            summary['warnings'].append(f"⚠️ {gap_count} missing time periods detected")
            summary['quick_wins'].append({
                'issue': 'Missing time periods',
                'impact': 'Models may struggle with gaps in data',
                'fix': 'Fill missing periods with interpolation or explicit handling'
            })

    # Missing values
    if target_col in df.columns:
        target_series = df[target_col]
        missing_count = target_series.isna().sum()
        missing_pct = (missing_count / n_rows) * 100

        summary['missing_values'] = {
            'count': int(missing_count),
            'percentage': round(missing_pct, 2),
            'is_critical': missing_pct > 10
        }

        if missing_pct > 10:
            summary['warnings'].append(f"⚠️ {missing_pct:.1f}% missing target values")
            summary['quick_wins'].append({
                'issue': 'High missing value rate',
                'impact': 'Reduced training data, potential bias',
                'fix': 'Investigate cause of missing values, consider imputation'
            })

        # Zero values (affects MAPE)
        zero_count = (target_series == 0).sum()
        zero_pct = (zero_count / n_rows) * 100

        summary['zero_values'] = {
            'count': int(zero_count),
            'percentage': round(zero_pct, 2),
            'affects_mape': zero_pct > 5
        }

        if zero_pct > 5:
            summary['warnings'].append(f"⚠️ {zero_pct:.1f}% zero values in target (affects MAPE calculation)")
            summary['quick_wins'].append({
                'issue': 'High zero-value rate',
                'impact': 'MAPE becomes unreliable, consider SMAPE instead',
                'fix': 'Use SMAPE metric or segment data to separate zero-heavy segments'
            })

        # Basic statistics
        non_null = target_series.dropna()
        if len(non_null) > 0:
            summary['statistics'] = {
                'mean': round(float(non_null.mean()), 2),
                'std': round(float(non_null.std()), 2),
                'min': round(float(non_null.min()), 2),
                'max': round(float(non_null.max()), 2),
                'cv': round(float(non_null.std() / non_null.mean()), 4) if non_null.mean() != 0 else None
            }

            # High variance warning
            cv = summary['statistics']['cv']
            if cv and cv > 1.0:
                summary['warnings'].append(f"⚠️ High coefficient of variation ({cv:.2f}) - data is highly variable")
                summary['quick_wins'].append({
                    'issue': 'High data variability',
                    'impact': 'May need wider confidence intervals, harder to forecast',
                    'fix': 'Consider log transformation or segment by high/low variance periods'
                })

    # Log summary
    logger.info(f"")
    logger.info(f"📊 DATA QUALITY SUMMARY")
    logger.info(f"   Total rows: {summary['volume']['total_rows']}")
    logger.info(f"   Date range: {summary['date_range'].get('start', 'N/A')} to {summary['date_range'].get('end', 'N/A')}")
    logger.info(f"   Missing values: {summary['missing_values'].get('percentage', 0)}%")
    logger.info(f"   Zero values: {summary['zero_values'].get('percentage', 0)}%")

    if summary['warnings']:
        for warning in summary['warnings']:
            logger.warning(f"   {warning}")

    if summary['quick_wins']:
        logger.info(f"   Quick wins identified: {len(summary['quick_wins'])}")

    return summary


def format_data_quality_report(summary: Dict[str, Any]) -> str:
    """
    Format data quality summary as a human-readable report.

    Args:
        summary: Output from compute_data_quality_summary()

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "DATA QUALITY REPORT",
        "=" * 60,
        "",
        "VOLUME:",
        f"  Total rows: {summary['volume'].get('total_rows', 'N/A')}",
        f"  Sufficient: {'Yes' if summary['volume'].get('is_sufficient') else 'No (need more data)'}",
        "",
        "DATE RANGE:",
        f"  Start: {summary['date_range'].get('start', 'N/A')}",
        f"  End: {summary['date_range'].get('end', 'N/A')}",
        f"  Span: {summary['date_range'].get('span_years', 'N/A')} years",
        f"  Gaps: {summary['date_range'].get('gap_count', 0)} missing periods",
        "",
        "DATA QUALITY:",
        f"  Missing values: {summary['missing_values'].get('percentage', 0)}%",
        f"  Zero values: {summary['zero_values'].get('percentage', 0)}%",
        "",
    ]

    if summary.get('statistics'):
        lines.extend([
            "STATISTICS:",
            f"  Mean: {summary['statistics'].get('mean', 'N/A')}",
            f"  Std Dev: {summary['statistics'].get('std', 'N/A')}",
            f"  Range: {summary['statistics'].get('min', 'N/A')} - {summary['statistics'].get('max', 'N/A')}",
            f"  CV: {summary['statistics'].get('cv', 'N/A')}",
            "",
        ])

    if summary.get('warnings'):
        lines.extend([
            "WARNINGS:",
        ])
        for warning in summary['warnings']:
            lines.append(f"  {warning}")
        lines.append("")

    if summary.get('quick_wins'):
        lines.extend([
            "QUICK WINS (Recommendations):",
            "-" * 40,
        ])
        for i, win in enumerate(summary['quick_wins'], 1):
            lines.extend([
                f"  {i}. {win['issue']}",
                f"     Impact: {win['impact']}",
                f"     Fix: {win['fix']}",
                ""
            ])

    lines.append("=" * 60)
    return "\n".join(lines)


# =============================================================================
# DATA LEAKAGE DETECTION UTILITIES
# =============================================================================

def validate_train_test_separation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = 'ds'
) -> Dict[str, Any]:
    """
    Validate proper temporal separation between train and test sets.

    Checks for:
    1. No overlapping dates between train and test
    2. Test dates are strictly after train dates
    3. No gaps between train end and test start (optional warning)

    Args:
        train_df: Training DataFrame
        test_df: Test/validation DataFrame
        date_col: Name of date column

    Returns:
        Dict with validation results and any violations found
    """
    result = {
        'is_valid': True,
        'violations': [],
        'warnings': [],
        'train_date_range': None,
        'test_date_range': None
    }

    if date_col not in train_df.columns or date_col not in test_df.columns:
        result['is_valid'] = False
        result['violations'].append(f"Date column '{date_col}' not found in both DataFrames")
        return result

    train_dates = pd.to_datetime(train_df[date_col])
    test_dates = pd.to_datetime(test_df[date_col])

    train_min, train_max = train_dates.min(), train_dates.max()
    test_min, test_max = test_dates.min(), test_dates.max()

    result['train_date_range'] = (train_min.strftime('%Y-%m-%d'), train_max.strftime('%Y-%m-%d'))
    result['test_date_range'] = (test_min.strftime('%Y-%m-%d'), test_max.strftime('%Y-%m-%d'))

    # Check 1: No overlapping dates
    train_date_set = set(train_dates.dt.normalize())
    test_date_set = set(test_dates.dt.normalize())
    overlap = train_date_set & test_date_set

    if overlap:
        result['is_valid'] = False
        result['violations'].append(
            f"DATA LEAKAGE: {len(overlap)} dates appear in both train and test sets. "
            f"First overlapping date: {min(overlap).strftime('%Y-%m-%d')}"
        )
        logger.error(f"🚨 DATA LEAKAGE DETECTED: {len(overlap)} overlapping dates between train and test!")

    # Check 2: Test dates after train dates
    if test_min <= train_max:
        result['is_valid'] = False
        result['violations'].append(
            f"DATA LEAKAGE: Test set starts ({test_min.strftime('%Y-%m-%d')}) "
            f"before or at train end ({train_max.strftime('%Y-%m-%d')})"
        )
        logger.error(f"🚨 DATA LEAKAGE DETECTED: Test dates not strictly after train dates!")

    # Check 3: Warn about gaps (optional, not a violation)
    gap_days = (test_min - train_max).days
    if gap_days > 1:
        result['warnings'].append(
            f"Gap of {gap_days} days between train end and test start. "
            f"This is normal for forecasting but verify it's intentional."
        )

    if result['is_valid']:
        logger.info(f"✅ Train/test temporal separation validated: "
                   f"Train [{result['train_date_range'][0]} to {result['train_date_range'][1]}], "
                   f"Test [{result['test_date_range'][0]} to {result['test_date_range'][1]}]")

    return result


def validate_no_future_leakage(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    date_col: str = 'ds'
) -> Dict[str, Any]:
    """
    Check for features that might contain future information.

    Detects potential leakage from:
    1. Features with suspiciously high correlation with target
    2. Features that appear to "know" future values
    3. Lag features that use insufficient lag periods

    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        feature_cols: List of feature column names to check
        date_col: Name of date column

    Returns:
        Dict with leakage risk assessment for each feature
    """
    result = {
        'high_risk_features': [],
        'medium_risk_features': [],
        'low_risk_features': [],
        'details': {}
    }

    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found")
        return result

    target = df[target_col].dropna()

    for col in feature_cols:
        if col not in df.columns:
            continue

        feature = df[col].dropna()
        risk_level = 'low'
        risk_reasons = []

        # Check 1: Suspiciously high correlation
        try:
            if len(target) == len(feature) and len(target) > 10:
                common_idx = target.index.intersection(feature.index)
                if len(common_idx) > 10:
                    corr = target.loc[common_idx].corr(feature.loc[common_idx])
                    if abs(corr) > 0.95:
                        risk_level = 'high'
                        risk_reasons.append(f"Very high correlation ({corr:.3f}) - possible leakage")
                    elif abs(corr) > 0.85:
                        risk_level = 'medium'
                        risk_reasons.append(f"High correlation ({corr:.3f}) - verify this is expected")
        except Exception:
            pass

        # Check 2: Lag features with insufficient lag
        if col.startswith('lag_'):
            try:
                lag_period = int(col.split('_')[1])
                if lag_period < 1:
                    risk_level = 'high'
                    risk_reasons.append(f"Lag period {lag_period} < 1 - definite leakage")
            except (ValueError, IndexError):
                pass

        # Check 3: Features named with future-looking terms
        future_terms = ['future', 'next', 'forward', 'ahead', 'predict', 'forecast']
        if any(term in col.lower() for term in future_terms):
            if risk_level == 'low':
                risk_level = 'medium'
            risk_reasons.append(f"Feature name suggests future information")

        result['details'][col] = {
            'risk_level': risk_level,
            'reasons': risk_reasons
        }

        if risk_level == 'high':
            result['high_risk_features'].append(col)
            logger.warning(f"🚨 HIGH LEAKAGE RISK: {col} - {'; '.join(risk_reasons)}")
        elif risk_level == 'medium':
            result['medium_risk_features'].append(col)
            logger.warning(f"⚠️ MEDIUM LEAKAGE RISK: {col} - {'; '.join(risk_reasons)}")
        else:
            result['low_risk_features'].append(col)

    return result


def assert_no_data_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = 'ds',
    raise_on_violation: bool = True
) -> bool:
    """
    Assert that there is no data leakage between train and test sets.

    Use this as a guard before training to ensure data integrity.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        date_col: Date column name
        raise_on_violation: If True, raises ValueError on leakage detection

    Returns:
        True if no leakage detected, False otherwise

    Raises:
        ValueError: If leakage detected and raise_on_violation is True
    """
    validation = validate_train_test_separation(train_df, test_df, date_col)

    if not validation['is_valid']:
        error_msg = "DATA LEAKAGE DETECTED:\n" + "\n".join(validation['violations'])
        logger.error(error_msg)

        if raise_on_violation:
            raise ValueError(error_msg)
        return False

    return True


def validate_forecast_output(
    forecast_df: pd.DataFrame,
    expected_horizon: int,
    historical_df: Optional[pd.DataFrame] = None,
    date_col: str = 'ds',
    value_col: str = 'yhat'
) -> Dict[str, Any]:
    """
    Validate forecast output for common issues.

    Checks:
    1. Forecast has expected number of periods
    2. Forecast dates are in the future (after historical data)
    3. Forecast values are within reasonable bounds
    4. No NaN or Inf values
    5. Confidence intervals are properly ordered (lower < forecast < upper)

    Args:
        forecast_df: DataFrame with forecast results
        expected_horizon: Expected number of forecast periods
        historical_df: Optional historical data to validate dates against
        date_col: Name of date column
        value_col: Name of forecast value column

    Returns:
        Dict with validation results
    """
    result: Dict[str, Any] = {
        'is_valid': True,
        'issues': [],
        'warnings': [],
        'stats': {}
    }

    # Input validation
    if forecast_df is None or len(forecast_df) == 0:
        result['is_valid'] = False
        result['issues'].append("Forecast DataFrame is empty or None")
        logger.error("validate_forecast_output: Empty forecast provided")
        return result

    if expected_horizon <= 0:
        result['warnings'].append(f"Invalid expected_horizon ({expected_horizon}), using actual length")
        expected_horizon = len(forecast_df)

    # Check 1: Expected horizon
    actual_horizon = len(forecast_df)
    result['stats']['actual_horizon'] = actual_horizon
    if actual_horizon != expected_horizon:
        result['warnings'].append(
            f"Forecast has {actual_horizon} periods, expected {expected_horizon}"
        )

    # Check 2: Future dates validation
    if historical_df is not None and date_col in historical_df.columns and date_col in forecast_df.columns:
        hist_max = pd.to_datetime(historical_df[date_col]).max()
        forecast_min = pd.to_datetime(forecast_df[date_col]).min()

        if forecast_min <= hist_max:
            result['is_valid'] = False
            result['issues'].append(
                f"INVALID: Forecast starts ({forecast_min.strftime('%Y-%m-%d')}) "
                f"at or before historical data ends ({hist_max.strftime('%Y-%m-%d')})"
            )

    # Check 3: Value bounds and NaN/Inf
    if value_col in forecast_df.columns:
        values = forecast_df[value_col]

        # NaN check
        nan_count = values.isna().sum()
        if nan_count > 0:
            result['is_valid'] = False
            result['issues'].append(f"Forecast contains {nan_count} NaN values")

        # Inf check
        inf_count = np.isinf(values).sum()
        if inf_count > 0:
            result['is_valid'] = False
            result['issues'].append(f"Forecast contains {inf_count} Inf values")

        # Negative values for financial data
        neg_count = (values < 0).sum()
        if neg_count > 0:
            result['warnings'].append(
                f"Forecast contains {neg_count} negative values (unusual for financial data)"
            )

        # Use any() instead of all() for proper boolean evaluation
        has_valid_values = values.notna().any()
        result['stats']['min'] = float(values.min()) if has_valid_values else None
        result['stats']['max'] = float(values.max()) if has_valid_values else None
        result['stats']['mean'] = float(values.mean()) if has_valid_values else None

    # Check 4: Confidence interval ordering
    if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
        lower = forecast_df['yhat_lower']
        upper = forecast_df['yhat_upper']
        yhat = forecast_df.get(value_col, pd.Series([0]))

        # Lower should be <= yhat
        lower_violations = (lower > yhat).sum()
        if lower_violations > 0:
            result['warnings'].append(
                f"Lower CI exceeds forecast in {lower_violations} periods"
            )

        # Upper should be >= yhat
        upper_violations = (upper < yhat).sum()
        if upper_violations > 0:
            result['warnings'].append(
                f"Upper CI below forecast in {upper_violations} periods"
            )

        # CI should have reasonable width
        ci_width = (upper - lower).mean()
        if ci_width <= 0:
            result['warnings'].append(
                f"Confidence interval has zero or negative width"
            )

    if result['is_valid']:
        logger.info(f"✅ Forecast validation passed: {actual_horizon} periods, "
                   f"range [{result['stats'].get('min', 'N/A'):.2f} - {result['stats'].get('max', 'N/A'):.2f}]")
    else:
        logger.error(f"❌ Forecast validation FAILED: {result['issues']}")

    return result


# =============================================================================
# RESIDUAL DIAGNOSTICS
# =============================================================================

def compute_residual_acf_diagnostics(
    residuals: np.ndarray,
    max_lags: int = 10,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Compute residual autocorrelation diagnostics to verify model quality.

    A well-fitted model should have residuals that are white noise (no autocorrelation).
    Significant autocorrelation in residuals indicates the model missed patterns.

    Based on: Hyndman & Athanasopoulos "Forecasting: Principles and Practice"

    Args:
        residuals: Model residuals (y_actual - y_predicted)
        max_lags: Maximum number of lags to check (default 10)
        significance_level: Alpha for Ljung-Box test (default 0.05)

    Returns:
        Dict with:
        - is_white_noise: True if residuals pass white noise test
        - acf_values: Autocorrelation at each lag
        - significant_lags: Lags with significant autocorrelation
        - ljung_box_p: P-value from Ljung-Box test
        - diagnostics: Human-readable diagnostic messages
    """
    residuals = np.asarray(residuals, dtype=np.float64)

    # Remove NaN/Inf
    valid_mask = np.isfinite(residuals)
    residuals = residuals[valid_mask]

    result = {
        'is_white_noise': True,
        'acf_values': [],
        'significant_lags': [],
        'ljung_box_p': None,
        'ljung_box_statistic': None,
        'diagnostics': [],
        'n_residuals': len(residuals)
    }

    if len(residuals) < max_lags + 5:
        result['diagnostics'].append(f"Insufficient residuals ({len(residuals)}) for ACF analysis")
        return result

    # Compute ACF manually (avoid statsmodels dependency for portability)
    n = len(residuals)
    mean_r = np.mean(residuals)
    var_r = np.var(residuals)

    if var_r == 0:
        result['diagnostics'].append("Zero variance in residuals - model may be constant")
        result['is_white_noise'] = False
        return result

    # ACF at each lag
    acf_values = []
    for lag in range(1, max_lags + 1):
        if lag >= n:
            break
        acf = np.sum((residuals[lag:] - mean_r) * (residuals[:-lag] - mean_r)) / (n * var_r)
        acf_values.append(float(acf))

    result['acf_values'] = acf_values

    # Bartlett's approximation for significance bound: ±1.96/sqrt(n)
    sig_bound = stats.norm.ppf(1 - significance_level / 2) / np.sqrt(n)

    # Find significant lags
    significant_lags = []
    for i, acf in enumerate(acf_values):
        if abs(acf) > sig_bound:
            significant_lags.append({
                'lag': i + 1,
                'acf': round(acf, 4),
                'bound': round(sig_bound, 4)
            })

    result['significant_lags'] = significant_lags

    # Ljung-Box test for overall white noise
    # Q = n(n+2) * sum(acf[k]^2 / (n-k)) for k=1..m
    Q = 0.0
    for k, acf in enumerate(acf_values, start=1):
        Q += (acf ** 2) / (n - k)
    Q *= n * (n + 2)

    # Q follows chi-squared with max_lags degrees of freedom
    p_value = float(1.0 - stats.chi2.cdf(Q, df=len(acf_values)))
    result['ljung_box_statistic'] = round(Q, 4)
    result['ljung_box_p'] = round(p_value, 4)

    # Determine if white noise
    if p_value < significance_level:
        result['is_white_noise'] = False
        result['diagnostics'].append(
            f"Ljung-Box test FAILED (p={p_value:.4f} < {significance_level}): "
            f"Residuals have significant autocorrelation"
        )
        logger.warning(f"⚠️ Residual ACF diagnostic: Model residuals are NOT white noise (p={p_value:.4f})")
    else:
        result['diagnostics'].append(
            f"Ljung-Box test PASSED (p={p_value:.4f}): Residuals are white noise"
        )
        logger.info(f"✅ Residual ACF diagnostic: Residuals are white noise (p={p_value:.4f})")

    if significant_lags:
        result['diagnostics'].append(
            f"Significant autocorrelation at lags: {[s['lag'] for s in significant_lags]}"
        )
        if not result['is_white_noise']:
            result['diagnostics'].append(
                "Consider: (1) Adding seasonal terms, (2) Increasing model order, "
                "(3) Using differencing"
            )

    return result


# =============================================================================
# CONFORMAL PREDICTION INTERVALS
# =============================================================================

def compute_conformal_prediction_intervals(
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    forecast_values: np.ndarray,
    coverage: float = 0.95,
    method: str = 'split'
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute distribution-free conformal prediction intervals.

    Conformal prediction provides prediction intervals with guaranteed coverage
    without assuming normality of residuals.

    Based on: Barber et al. "Conformal Prediction Under Covariate Shift" (2019)
              and "Distribution-Free Predictive Inference for Regression" (2018)

    Args:
        y_train: Actual training values
        y_pred_train: Predicted values for training set (from CV or holdout)
        forecast_values: Point forecasts for future periods
        coverage: Desired coverage probability (default 0.95)
        method: 'split' for split conformal, 'quantile' for quantile-based

    Returns:
        Tuple of (lower_bounds, upper_bounds, diagnostics)
        - Intervals have guaranteed coverage if calibration set is exchangeable
    """
    y_train = np.asarray(y_train, dtype=np.float64)
    y_pred_train = np.asarray(y_pred_train, dtype=np.float64)
    forecast_values = np.asarray(forecast_values, dtype=np.float64)

    diagnostics = {
        'method': method,
        'coverage': coverage,
        'n_calibration': len(y_train),
        'conformity_scores': None,
        'quantile_threshold': None
    }

    # Compute conformity scores (absolute residuals)
    residuals = y_train - y_pred_train
    conformity_scores = np.abs(residuals)

    # Remove NaN/Inf
    valid_mask = np.isfinite(conformity_scores)
    conformity_scores = conformity_scores[valid_mask]

    if len(conformity_scores) < 5:
        logger.warning("Insufficient calibration data for conformal intervals, using parametric fallback")
        # Fallback to parametric
        residual_std = np.std(residuals[np.isfinite(residuals)])
        z = _get_z_critical_value(coverage)
        margin = z * residual_std
        return forecast_values - margin, forecast_values + margin, diagnostics

    diagnostics['n_calibration'] = len(conformity_scores)

    if method == 'split':
        # Split conformal: use quantile of conformity scores
        # Finite sample correction: (1 + 1/n) * coverage quantile
        n = len(conformity_scores)
        adjusted_coverage = min(1.0, (1 + 1/n) * coverage)
        quantile_threshold = float(np.quantile(conformity_scores, adjusted_coverage))

        diagnostics['quantile_threshold'] = round(quantile_threshold, 4)
        diagnostics['adjusted_coverage'] = round(adjusted_coverage, 4)

        # Constant width intervals
        lower_bounds = forecast_values - quantile_threshold
        upper_bounds = forecast_values + quantile_threshold

    elif method == 'quantile':
        # Quantile-based: separate lower and upper
        alpha = 1 - coverage
        lower_residuals = residuals[np.isfinite(residuals)]
        upper_residuals = residuals[np.isfinite(residuals)]

        lower_q = float(np.quantile(lower_residuals, alpha / 2))
        upper_q = float(np.quantile(upper_residuals, 1 - alpha / 2))

        diagnostics['lower_quantile'] = round(lower_q, 4)
        diagnostics['upper_quantile'] = round(upper_q, 4)

        lower_bounds = forecast_values + lower_q  # lower_q is negative
        upper_bounds = forecast_values + upper_q

    else:
        raise ValueError(f"Unknown conformal method: {method}")

    # Ensure lower < upper
    lower_bounds = np.minimum(lower_bounds, forecast_values)
    upper_bounds = np.maximum(upper_bounds, forecast_values)

    # Log diagnostics
    avg_width = np.mean(upper_bounds - lower_bounds)
    logger.info(f"✅ Conformal {coverage:.0%} PI: avg width={avg_width:.2f}, "
               f"method={method}, n_cal={diagnostics['n_calibration']}")

    return lower_bounds, upper_bounds, diagnostics


def validate_forecast_reasonableness(
    forecast_values: np.ndarray,
    historical_values: np.ndarray,
    max_change_ratio: float = 5.0
) -> Dict[str, Any]:
    """
    Check if forecast values are reasonable compared to historical data.

    Detects potential issues like:
    1. Forecasts dramatically higher/lower than historical range
    2. Sudden jumps at forecast boundary
    3. Unrealistic growth rates

    Args:
        forecast_values: Array of forecast values
        historical_values: Array of historical values
        max_change_ratio: Maximum acceptable ratio of forecast/historical mean

    Returns:
        Dict with reasonableness assessment
    """
    result = {
        'is_reasonable': True,
        'concerns': [],
        'stats': {}
    }

    hist_clean = historical_values[~np.isnan(historical_values)]
    fc_clean = forecast_values[~np.isnan(forecast_values)]

    if len(hist_clean) == 0 or len(fc_clean) == 0:
        result['is_reasonable'] = False
        result['concerns'].append("Empty historical or forecast data")
        return result

    hist_mean = np.mean(hist_clean)
    hist_std = np.std(hist_clean)
    fc_mean = np.mean(fc_clean)

    result['stats'] = {
        'historical_mean': float(hist_mean),
        'historical_std': float(hist_std),
        'forecast_mean': float(fc_mean),
        'change_ratio': float(fc_mean / hist_mean) if hist_mean != 0 else None
    }

    # Check 1: Forecast mean vs historical mean
    if hist_mean != 0:
        change_ratio = fc_mean / hist_mean
        if change_ratio > max_change_ratio or change_ratio < (1 / max_change_ratio):
            result['is_reasonable'] = False
            result['concerns'].append(
                f"Forecast mean ({fc_mean:.2f}) differs significantly from historical mean "
                f"({hist_mean:.2f}) - ratio: {change_ratio:.2f}x"
            )

    # Check 2: Boundary jump (last historical vs first forecast)
    if len(hist_clean) > 0 and len(fc_clean) > 0:
        last_hist = hist_clean[-1]
        first_fc = fc_clean[0]

        if last_hist != 0:
            boundary_jump = abs(first_fc - last_hist) / abs(last_hist)
            if boundary_jump > 0.5:  # More than 50% jump
                result['concerns'].append(
                    f"Large jump at forecast boundary: {last_hist:.2f} -> {first_fc:.2f} ({boundary_jump:.1%} change)"
                )

    # Check 3: Forecast variance vs historical
    if hist_std > 0:
        fc_std = np.std(fc_clean)
        std_ratio = fc_std / hist_std
        if std_ratio > 3 or std_ratio < 0.1:
            result['concerns'].append(
                f"Forecast variability ({fc_std:.2f}) differs significantly from historical ({hist_std:.2f})"
            )

    if result['is_reasonable'] and not result['concerns']:
        logger.info(f"✅ Forecast reasonableness check passed")
    elif result['concerns']:
        for concern in result['concerns']:
            logger.warning(f"⚠️ Forecast concern: {concern}")

    return result


# =============================================================================
# FLAT FORECAST DETECTION
# =============================================================================

def detect_flat_forecast(
    forecast_values: np.ndarray,
    historical_values: np.ndarray,
    relative_variance_threshold: float = 0.01,
    absolute_variance_threshold: float = 1e-6
) -> Dict[str, Any]:
    """
    Detect if a forecast is flat (constant or near-constant values).

    A flat forecast indicates model failure - the model is essentially predicting
    the same value for all future periods, which is uninformative.

    Common causes:
    1. Degenerate model orders (0,0,0), (0,1,0) for ARIMA/SARIMAX
    2. Model coefficients converged to near-zero
    3. Simple exponential smoothing with no trend/seasonal (ETS)
    4. Insufficient data for parameter estimation

    Args:
        forecast_values: Array of forecast values
        historical_values: Array of historical values (for context)
        relative_variance_threshold: If forecast_var/historical_var < this, flat
        absolute_variance_threshold: If forecast_var < this, definitely flat

    Returns:
        Dict with:
        - is_flat: True if forecast is flat/constant
        - forecast_variance: Variance of forecast values
        - historical_variance: Variance of historical values
        - variance_ratio: forecast_var / historical_var
        - unique_values: Number of unique forecast values
        - recommendation: What to do if flat
    """
    forecast_values = np.asarray(forecast_values, dtype=np.float64)
    historical_values = np.asarray(historical_values, dtype=np.float64)

    # Clean arrays
    fc_clean = forecast_values[np.isfinite(forecast_values)]
    hist_clean = historical_values[np.isfinite(historical_values)]

    result = {
        'is_flat': False,
        'forecast_variance': 0.0,
        'historical_variance': 0.0,
        'variance_ratio': 0.0,
        'unique_values': 0,
        'flat_reason': None,
        'recommendation': None
    }

    if len(fc_clean) < 2:
        result['flat_reason'] = "Insufficient forecast points"
        return result

    # Calculate variances
    fc_var = float(np.var(fc_clean))
    hist_var = float(np.var(hist_clean)) if len(hist_clean) > 1 else 1.0

    result['forecast_variance'] = fc_var
    result['historical_variance'] = hist_var
    result['unique_values'] = len(np.unique(np.round(fc_clean, 4)))

    # Check 1: Absolute variance near zero
    if fc_var < absolute_variance_threshold:
        result['is_flat'] = True
        result['flat_reason'] = f"Near-zero forecast variance ({fc_var:.2e})"
        result['recommendation'] = "Model coefficients may have converged to zero. Try different orders or add differencing."
        logger.error(f"🚨 FLAT FORECAST DETECTED: {result['flat_reason']}")
        return result

    # Check 2: Relative variance compared to historical
    if hist_var > 0:
        variance_ratio = fc_var / hist_var
        result['variance_ratio'] = variance_ratio

        if variance_ratio < relative_variance_threshold:
            result['is_flat'] = True
            result['flat_reason'] = f"Forecast variance ({fc_var:.4f}) is {variance_ratio:.4f}x historical variance ({hist_var:.4f})"
            result['recommendation'] = "Forecast has much less variability than historical data. Consider adding trend/seasonal components."
            logger.error(f"🚨 FLAT FORECAST DETECTED: {result['flat_reason']}")
            return result

    # Check 3: All values nearly identical
    if result['unique_values'] == 1:
        result['is_flat'] = True
        result['flat_reason'] = f"All {len(fc_clean)} forecast values are identical ({fc_clean[0]:.2f})"
        result['recommendation'] = "Model is producing constant predictions. Try a different model type or order."
        logger.error(f"🚨 FLAT FORECAST DETECTED: {result['flat_reason']}")
        return result

    # Check 4: Very few unique values for long forecasts
    if len(fc_clean) > 5 and result['unique_values'] <= 2:
        result['is_flat'] = True
        result['flat_reason'] = f"Only {result['unique_values']} unique values in {len(fc_clean)} period forecast"
        result['recommendation'] = "Forecast is oscillating between very few values. Check for model degeneracy."
        logger.warning(f"⚠️ NEAR-FLAT FORECAST: {result['flat_reason']}")
        return result

    logger.debug(f"✅ Forecast variance check passed: var_ratio={result['variance_ratio']:.4f}, unique={result['unique_values']}")
    return result


def get_fallback_arima_orders() -> List[Tuple[int, int, int]]:
    """
    Get fallback ARIMA orders known to produce non-flat forecasts.

    These orders include AR and/or MA components that capture dynamics.
    """
    return [
        (1, 1, 1),  # Basic ARIMA with differencing
        (2, 1, 1),  # AR(2) for more dynamics
        (1, 1, 2),  # MA(2) for moving average
        (2, 1, 2),  # Full ARIMA
        (1, 0, 1),  # ARMA without differencing
    ]


def get_fallback_sarimax_orders() -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]]:
    """
    Get fallback SARIMAX orders known to produce non-flat forecasts.

    These orders include AR and/or MA components that capture dynamics.
    """
    return [
        ((1, 1, 1), (0, 0, 0, 12)),  # Basic ARIMA with differencing
        ((2, 1, 1), (0, 0, 0, 12)),  # AR(2) for more dynamics
        ((1, 1, 2), (0, 0, 0, 12)),  # MA(2) for moving average
        ((1, 0, 1), (1, 0, 1, 12)),  # With seasonal components
        ((2, 1, 2), (1, 0, 1, 12)),  # Full seasonal ARIMA
    ]


def get_fallback_ets_params() -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Get fallback ETS parameters known to produce non-flat forecasts.

    These combinations include trend and/or seasonal components.
    """
    return [
        ('add', 'add'),    # Additive trend and seasonal
        ('add', 'mul'),    # Additive trend, multiplicative seasonal
        ('mul', 'add'),    # Multiplicative trend, additive seasonal
        ('add', None),     # Trend only (Holt's linear)
        (None, 'add'),     # Seasonal only
    ]
