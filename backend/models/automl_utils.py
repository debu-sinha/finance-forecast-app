"""
AutoML Best Practices Utilities

This module implements best practices from top open-source forecasting AutoML frameworks:
- AutoGluon-TimeSeries: Model validation, ensemble strategies, presets
- Nixtla (StatsForecast, MLForecast): Efficient CV, lag features, speed optimizations
- Greykite/Silverkite: Data quality checks, interpretability, anomaly detection
- GluonTS: Probabilistic forecasting, Monte Carlo intervals
- Darts: Unified interface patterns, stacking ensembles
- Chronos: Zero-shot baselines

Author: debu-sinha
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
from backend.utils.logging_utils import log_io

logger = logging.getLogger(__name__)


# =============================================================================
# TRAINING PRESETS (AutoGluon-style)
# =============================================================================

class TrainingPreset(Enum):
    """Training presets inspired by AutoGluon's preset system."""
    FAST = "fast"
    MEDIUM = "medium"
    HIGH_QUALITY = "high_quality"
    BEST = "best"


TRAINING_PRESETS = {
    TrainingPreset.FAST: {
        'models': ['arima', 'statsforecast'],
        'max_time_per_model': 60,
        'cv_folds': 2,
        'hyperparameter_tuning': 'minimal',
        'ensemble': False,
        'max_combinations': 5,
        'description': 'Quick baseline - 2-3 minutes total',
    },
    TrainingPreset.MEDIUM: {
        'models': ['prophet', 'arima', 'statsforecast', 'ets'],
        'max_time_per_model': 180,
        'cv_folds': 3,
        'hyperparameter_tuning': 'standard',
        'ensemble': True,
        'max_combinations': 10,
        'description': 'Balanced speed and accuracy - 5-10 minutes',
    },
    TrainingPreset.HIGH_QUALITY: {
        'models': ['prophet', 'arima', 'sarimax', 'xgboost', 'statsforecast', 'ets'],
        'max_time_per_model': 600,
        'cv_folds': 5,
        'hyperparameter_tuning': 'extensive',
        'ensemble': True,
        'max_combinations': 20,
        'description': 'Production quality - 15-30 minutes',
    },
    TrainingPreset.BEST: {
        'models': ['prophet', 'arima', 'sarimax', 'xgboost', 'statsforecast', 'ets', 'chronos'],
        'max_time_per_model': None,  # No limit
        'cv_folds': 5,
        'hyperparameter_tuning': 'exhaustive',
        'ensemble': True,
        'max_combinations': 50,
        'description': 'Maximum accuracy - 30+ minutes',
    },
}


@log_io
def get_preset_config(preset: Union[str, TrainingPreset]) -> Dict[str, Any]:
    """Get configuration for a training preset."""
    if isinstance(preset, str):
        preset = TrainingPreset(preset.lower())
    return TRAINING_PRESETS[preset].copy()


# =============================================================================
# DATA QUALITY CHECKS (Greykite, AutoTS patterns)
# =============================================================================

@dataclass
class DataQualityReport:
    """Report from data quality checks."""
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    transformations_applied: List[str] = field(default_factory=list)


# Minimum data requirements by frequency (from best practices)
MIN_DATA_REQUIREMENTS = {
    'daily': {'min_rows': 90, 'recommended_rows': 365, 'seasonal_period': 7},
    'weekly': {'min_rows': 26, 'recommended_rows': 104, 'seasonal_period': 52},
    'monthly': {'min_rows': 12, 'recommended_rows': 36, 'seasonal_period': 12},
}


@log_io
def check_data_quality(
    df: pd.DataFrame,
    date_col: str = 'ds',
    target_col: str = 'y',
    frequency: str = 'weekly',
    auto_fix: bool = True
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Comprehensive data quality checks inspired by Greykite and AutoTS.

    Args:
        df: Input DataFrame
        date_col: Date column name
        target_col: Target column name
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        auto_fix: Whether to automatically fix issues where possible

    Returns:
        Tuple of (cleaned DataFrame, DataQualityReport)
    """
    report = DataQualityReport(is_valid=True)
    df = df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
        report.transformations_applied.append("Converted date column to datetime")

    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)

    # Get requirements for frequency
    freq_req = MIN_DATA_REQUIREMENTS.get(frequency, MIN_DATA_REQUIREMENTS['weekly'])

    # 1. Check minimum data requirement
    n_rows = len(df)
    report.stats['n_rows'] = n_rows
    report.stats['frequency'] = frequency

    if n_rows < freq_req['min_rows']:
        report.issues.append(
            f"Insufficient data: {n_rows} rows, need at least {freq_req['min_rows']} for {frequency} data"
        )
        report.is_valid = False
    elif n_rows < freq_req['recommended_rows']:
        report.warnings.append(
            f"Limited data: {n_rows} rows. Recommend {freq_req['recommended_rows']}+ for {frequency} forecasting"
        )
        report.recommendations.append("Consider using simpler models (ARIMA, ETS) with limited data")

    # 2. Check for missing values
    null_count = df[target_col].isnull().sum()
    null_pct = null_count / n_rows * 100
    report.stats['null_count'] = null_count
    report.stats['null_pct'] = null_pct

    if null_count > 0:
        if null_pct > 20:
            report.issues.append(f"High missing rate: {null_pct:.1f}% of target values are null")
            report.is_valid = False
        elif null_pct > 5:
            report.warnings.append(f"Missing values: {null_pct:.1f}% of target values are null")

        if auto_fix and null_pct <= 20:
            # Forward fill for small gaps (best practice from Nixtla)
            df[target_col] = df[target_col].ffill().bfill()
            report.transformations_applied.append(f"Filled {null_count} missing values with forward/backward fill")

    # 3. Check for duplicates
    duplicate_dates = df[date_col].duplicated().sum()
    report.stats['duplicate_dates'] = duplicate_dates

    if duplicate_dates > 0:
        report.issues.append(f"Found {duplicate_dates} duplicate dates")
        if auto_fix:
            # Keep last value for duplicates (common practice)
            df = df.drop_duplicates(subset=[date_col], keep='last')
            report.transformations_applied.append(f"Removed {duplicate_dates} duplicate dates (kept last)")

    # 4. Check for irregular time intervals (gaps)
    date_diff = df[date_col].diff().dropna()
    expected_freq = {'daily': 1, 'weekly': 7, 'monthly': 30}
    expected_days = expected_freq.get(frequency, 7)

    # Check for gaps (missing periods)
    if frequency == 'daily':
        gaps = (date_diff > pd.Timedelta(days=expected_days * 2)).sum()
    elif frequency == 'weekly':
        gaps = (date_diff > pd.Timedelta(days=expected_days * 1.5)).sum()
    else:
        gaps = (date_diff > pd.Timedelta(days=expected_days * 1.5)).sum()

    report.stats['gaps_detected'] = gaps
    if gaps > 0:
        report.warnings.append(f"Found {gaps} gaps in time series (missing periods)")
        report.recommendations.append("Consider interpolating missing periods for better model performance")

    # 5. Check for outliers using IQR method (Greykite pattern)
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    outlier_mask = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
    n_outliers = outlier_mask.sum()
    report.stats['n_outliers'] = n_outliers
    report.stats['outlier_pct'] = n_outliers / n_rows * 100

    if n_outliers > 0:
        outlier_pct = n_outliers / n_rows * 100
        if outlier_pct > 10:
            report.warnings.append(f"High outlier rate: {n_outliers} outliers ({outlier_pct:.1f}%)")
            report.recommendations.append("Consider outlier treatment or robust models")
        else:
            report.warnings.append(f"Found {n_outliers} outliers ({outlier_pct:.1f}%)")

        if auto_fix and outlier_pct <= 10:
            # Winsorize outliers (Greykite pattern)
            df.loc[df[target_col] < lower_bound, target_col] = lower_bound
            df.loc[df[target_col] > upper_bound, target_col] = upper_bound
            report.transformations_applied.append(f"Winsorized {n_outliers} outliers to IQR bounds")

    # 6. Check for negative values (if unexpected)
    n_negative = (df[target_col] < 0).sum()
    report.stats['n_negative'] = n_negative
    if n_negative > 0:
        report.warnings.append(f"Found {n_negative} negative values in target")
        report.recommendations.append("Verify if negative values are valid for your use case")

    # 7. Check variance/coefficient of variation
    mean_val = df[target_col].mean()
    std_val = df[target_col].std()
    cv = std_val / mean_val if mean_val != 0 else 0
    report.stats['mean'] = mean_val
    report.stats['std'] = std_val
    report.stats['cv'] = cv

    if cv < 0.01:
        report.warnings.append(f"Very low variance (CV={cv:.4f}) - models may not learn meaningful patterns")
        report.recommendations.append("Consider if forecasting is needed for near-constant data")
    elif cv > 2.0:
        report.warnings.append(f"Very high variance (CV={cv:.2f}) - may indicate data quality issues")
        report.recommendations.append("Consider data transformation (log, sqrt) or outlier treatment")

    # 8. Check for sufficient seasonal cycles
    seasonal_period = freq_req['seasonal_period']
    n_cycles = n_rows / seasonal_period
    report.stats['seasonal_cycles'] = n_cycles

    if n_cycles < 2:
        report.warnings.append(f"Less than 2 seasonal cycles ({n_cycles:.1f}) - seasonal models may not work well")
        report.recommendations.append("Disable yearly_seasonality in Prophet, use non-seasonal ARIMA")

    # Log summary
    if report.issues:
        logger.warning(f"Data quality issues: {report.issues}")
    if report.warnings:
        logger.warning(f"Data quality warnings: {report.warnings}")
    if report.transformations_applied:
        logger.info(f"Data transformations applied: {report.transformations_applied}")

    return df, report


# =============================================================================
# FORECAST VALIDATION (AutoGluon patterns)
# =============================================================================

@dataclass
class ForecastValidationResult:
    """Result of forecast validation."""
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@log_io
def validate_forecast(
    predictions: np.ndarray,
    historical_mean: float,
    historical_std: float,
    model_name: str = "Unknown",
    max_ratio: float = 10.0,
    min_ratio: float = 0.1,
    max_absolute_value: float = 1e30
) -> ForecastValidationResult:
    """
    Validate forecast predictions (AutoGluon-style).

    Args:
        predictions: Array of predictions
        historical_mean: Mean of historical target values
        historical_std: Std of historical target values
        model_name: Name of the model for logging
        max_ratio: Maximum allowed ratio of prediction mean to historical mean
        min_ratio: Minimum allowed ratio
        max_absolute_value: Maximum absolute value allowed

    Returns:
        ForecastValidationResult
    """
    result = ForecastValidationResult(is_valid=True)
    predictions = np.array(predictions).flatten()

    # 1. Check for NaN
    nan_count = np.isnan(predictions).sum()
    if nan_count > 0:
        result.is_valid = False
        result.issues.append(f"Contains {nan_count} NaN values")
        result.stats['nan_count'] = nan_count

    # 2. Check for Inf
    inf_count = np.isinf(predictions).sum()
    if inf_count > 0:
        result.is_valid = False
        result.issues.append(f"Contains {inf_count} infinite values")
        result.stats['inf_count'] = inf_count

    # Early exit if NaN/Inf
    if not result.is_valid:
        logger.warning(f"🚫 {model_name}: Invalid predictions - {result.issues}")
        return result

    # 3. Check for extreme absolute values
    max_abs = np.max(np.abs(predictions))
    if max_abs > max_absolute_value:
        result.is_valid = False
        result.issues.append(f"Extreme values detected (max abs: {max_abs:.2e})")
        result.stats['max_abs_value'] = max_abs

    # 4. Check ratio to historical mean
    pred_mean = np.mean(predictions)
    ratio = pred_mean / historical_mean if historical_mean != 0 else float('inf')
    result.stats['pred_mean'] = pred_mean
    result.stats['historical_mean'] = historical_mean
    result.stats['mean_ratio'] = ratio

    if ratio > max_ratio:
        result.is_valid = False
        result.issues.append(f"Predictions {ratio:.1f}x higher than historical mean")
    elif ratio < min_ratio:
        result.is_valid = False
        result.issues.append(f"Predictions {ratio:.1f}x lower than historical mean (only {ratio*100:.1f}%)")
    elif ratio > 3.0 or ratio < 0.33:
        result.warnings.append(f"Predictions differ significantly from historical ({ratio:.2f}x)")

    # 5. Check for flat/constant predictions
    pred_std = np.std(predictions)
    result.stats['pred_std'] = pred_std

    if pred_std < 1e-6:
        result.is_valid = False
        result.issues.append("Flat predictions (zero variance)")
    elif pred_std < historical_std * 0.01:
        result.warnings.append("Very low variance in predictions compared to historical")

    # 6. Check for all zeros
    if np.all(predictions == 0):
        result.is_valid = False
        result.issues.append("All predictions are zero")

    # 7. Check for negative values when historical is all positive
    if historical_mean > 0 and np.any(predictions < 0):
        n_negative = (predictions < 0).sum()
        if n_negative > len(predictions) * 0.5:
            result.is_valid = False
            result.issues.append(f"Majority negative predictions ({n_negative}/{len(predictions)}) but historical is positive")
        else:
            result.warnings.append(f"{n_negative} negative predictions (historical mean is positive)")

    # Log result
    if result.is_valid:
        if result.warnings:
            logger.warning(f"⚠️ {model_name}: Valid with warnings - {result.warnings}")
        else:
            logger.info(f"✅ {model_name}: Predictions validated successfully")
    else:
        logger.warning(f"🚫 {model_name}: Invalid predictions - {result.issues}")

    return result


@log_io
def validate_prediction_intervals(
    lower: np.ndarray,
    upper: np.ndarray,
    point_forecast: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[bool, List[str]]:
    """
    Validate prediction intervals.

    Args:
        lower: Lower bound predictions
        upper: Upper bound predictions
        point_forecast: Point predictions
        confidence_level: Confidence level (e.g., 0.95)

    Returns:
        Tuple of (is_valid, issues_list)
    """
    issues = []

    # Check lower <= point <= upper
    if np.any(lower > point_forecast):
        issues.append("Lower bound exceeds point forecast")
    if np.any(upper < point_forecast):
        issues.append("Upper bound below point forecast")
    if np.any(lower > upper):
        issues.append("Lower bound exceeds upper bound")

    # Check for reasonable interval width
    interval_width = upper - lower
    relative_width = interval_width / np.abs(point_forecast + 1e-10)

    if np.any(relative_width > 10):
        issues.append("Extremely wide intervals (>10x point forecast)")
    if np.any(interval_width < 0):
        issues.append("Negative interval width detected")

    # Check for NaN/Inf in intervals
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        issues.append("NaN in prediction intervals")
    if np.any(np.isinf(lower)) or np.any(np.isinf(upper)):
        issues.append("Infinite values in prediction intervals")

    return len(issues) == 0, issues


# =============================================================================
# OVERFITTING DETECTION (Enhanced from multiple frameworks)
# =============================================================================

@dataclass
class OverfittingReport:
    """Report from overfitting detection."""
    is_overfitting: bool
    severity: str  # 'none', 'low', 'medium', 'high', 'severe'
    train_metric: float
    eval_metric: float
    ratio: float
    recommendations: List[str] = field(default_factory=list)


@log_io
def detect_overfitting(
    train_mape: float,
    eval_mape: float,
    model_name: str = "Unknown",
    thresholds: Dict[str, float] = None
) -> OverfittingReport:
    """
    Detect overfitting based on train/eval metric gap.

    Inspired by patterns from AutoGluon, Greykite, and general ML best practices.

    Args:
        train_mape: MAPE on training data
        eval_mape: MAPE on evaluation/validation data
        model_name: Name of model for logging
        thresholds: Custom thresholds dict with keys 'low', 'medium', 'high', 'severe'

    Returns:
        OverfittingReport
    """
    if thresholds is None:
        thresholds = {
            'low': 2.0,      # 2x ratio - mild concern
            'medium': 5.0,   # 5x ratio - significant overfitting
            'high': 10.0,    # 10x ratio - severe overfitting
            'severe': 20.0,  # 20x ratio - model is memorizing
        }

    # Handle edge cases
    if train_mape <= 0:
        train_mape = 0.01  # Avoid division by zero

    ratio = eval_mape / train_mape

    # Determine severity
    if ratio >= thresholds['severe']:
        severity = 'severe'
        is_overfitting = True
    elif ratio >= thresholds['high']:
        severity = 'high'
        is_overfitting = True
    elif ratio >= thresholds['medium']:
        severity = 'medium'
        is_overfitting = True
    elif ratio >= thresholds['low']:
        severity = 'low'
        is_overfitting = True
    else:
        severity = 'none'
        is_overfitting = False

    # Generate recommendations
    recommendations = []
    if is_overfitting:
        recommendations.append(f"Model shows {severity} overfitting (train/eval ratio: {ratio:.1f}x)")

        if severity in ['high', 'severe']:
            recommendations.append("Consider: Reduce model complexity (fewer parameters)")
            recommendations.append("Consider: Increase regularization (higher changepoint_prior_scale)")
            recommendations.append("Consider: Use simpler model (ARIMA instead of Prophet)")
            recommendations.append("Consider: Add more training data if available")
        elif severity == 'medium':
            recommendations.append("Consider: Increase regularization slightly")
            recommendations.append("Consider: Reduce number of regressors/covariates")
        elif severity == 'low':
            recommendations.append("Monitor: Slight overfitting detected, may still be acceptable")

    report = OverfittingReport(
        is_overfitting=is_overfitting,
        severity=severity,
        train_metric=train_mape,
        eval_metric=eval_mape,
        ratio=ratio,
        recommendations=recommendations
    )

    # Log
    if is_overfitting:
        logger.warning(
            f"⚠️ {model_name}: Overfitting detected ({severity}) - "
            f"Train MAPE: {train_mape:.2f}%, Eval MAPE: {eval_mape:.2f}%, Ratio: {ratio:.1f}x"
        )
    else:
        logger.info(f"✅ {model_name}: No significant overfitting (ratio: {ratio:.1f}x)")

    return report


# =============================================================================
# ENHANCED CROSS-VALIDATION (Nixtla, sktime patterns)
# =============================================================================

@log_io
def weighted_temporal_cv_score(
    fold_scores: List[float],
    fold_weights: Optional[List[float]] = None,
    recency_weight: float = 1.5
) -> Tuple[float, float]:
    """
    Calculate weighted CV score with emphasis on recent folds.

    Inspired by Nixtla's approach where recent performance matters more.

    Args:
        fold_scores: List of scores from each CV fold
        fold_weights: Optional custom weights for each fold
        recency_weight: How much to weight later folds (1.0 = equal, >1 = more weight)

    Returns:
        Tuple of (weighted_mean, weighted_std)
    """
    n_folds = len(fold_scores)

    if fold_weights is None:
        # Generate weights that increase for later folds
        fold_weights = [1.0 + (i / n_folds) * (recency_weight - 1.0) for i in range(n_folds)]

    # Normalize weights
    total_weight = sum(fold_weights)
    fold_weights = [w / total_weight for w in fold_weights]

    # Weighted mean
    weighted_mean = sum(s * w for s, w in zip(fold_scores, fold_weights))

    # Weighted standard deviation
    variance = sum(w * (s - weighted_mean) ** 2 for s, w in zip(fold_scores, fold_weights))
    weighted_std = np.sqrt(variance)

    return weighted_mean, weighted_std


# =============================================================================
# MODEL SELECTION FOR ENSEMBLE (AutoGluon patterns)
# =============================================================================

@log_io
def select_models_for_ensemble(
    model_results: List[Dict[str, Any]],
    max_mape_ratio: float = 3.0,
    min_models: int = 2,
    max_models: int = 5,
    historical_mean: float = None,
    historical_std: float = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Select valid models for ensemble creation (AutoGluon-style).

    Args:
        model_results: List of model result dicts with 'model_name', 'mape', 'predictions', etc.
        max_mape_ratio: Maximum ratio of model MAPE to best model MAPE
        min_models: Minimum models required for ensemble
        max_models: Maximum models to include in ensemble
        historical_mean: Historical mean for validation
        historical_std: Historical std for validation

    Returns:
        Tuple of (selected_models, exclusion_reasons)
    """
    if not model_results:
        return [], ["No model results provided"]

    exclusion_reasons = []
    valid_models = []

    # First pass: validate predictions
    for model in model_results:
        model_name = model.get('model_name') or model.get('model_type') or 'Unknown'

        # Skip if no predictions
        if 'predictions' not in model and 'fcst_df' not in model:
            exclusion_reasons.append(f"{model_name}: No predictions available")
            continue

        # Get predictions
        if 'predictions' in model:
            preds = model['predictions']
        else:
            preds = model['fcst_df']['yhat'].values if 'yhat' in model['fcst_df'].columns else None

        if preds is None:
            exclusion_reasons.append(f"{model_name}: Could not extract predictions")
            continue

        # Validate predictions
        if historical_mean is not None and historical_std is not None:
            validation = validate_forecast(
                preds, historical_mean, historical_std, model_name
            )
            if not validation.is_valid:
                exclusion_reasons.append(f"{model_name}: {', '.join(validation.issues)}")
                continue

        # Check for valid MAPE
        mape = model.get('mape') or model.get('metrics', {}).get('mape')
        if mape is None or mape == 'N/A':
            exclusion_reasons.append(f"{model_name}: No valid MAPE metric")
            continue

        try:
            mape = float(mape)
            if np.isnan(mape) or np.isinf(mape):
                exclusion_reasons.append(f"{model_name}: Invalid MAPE value ({mape})")
                continue
            model['_mape_float'] = mape
            valid_models.append(model)
        except (ValueError, TypeError):
            exclusion_reasons.append(f"{model_name}: Could not parse MAPE ({mape})")
            continue

    if not valid_models:
        return [], exclusion_reasons + ["No valid models remaining after validation"]

    # Second pass: filter by relative performance
    best_mape = min(m['_mape_float'] for m in valid_models)
    mape_threshold = best_mape * max_mape_ratio

    filtered_models = []
    for model in valid_models:
        model_name = model.get('model_name') or model.get('model_type') or 'Unknown'
        mape = model['_mape_float']

        if mape <= mape_threshold:
            filtered_models.append(model)
        else:
            exclusion_reasons.append(
                f"{model_name}: MAPE {mape:.2f}% exceeds threshold {mape_threshold:.2f}% "
                f"({mape/best_mape:.1f}x worse than best)"
            )

    # Sort by MAPE and take top models
    filtered_models.sort(key=lambda m: m['_mape_float'])
    selected = filtered_models[:max_models]

    # Log summary
    logger.info(f"Model selection: {len(selected)}/{len(model_results)} models selected for ensemble")
    for reason in exclusion_reasons:
        logger.warning(f"   Excluded: {reason}")

    # Clean up temporary field
    for model in selected:
        model.pop('_mape_float', None)

    return selected, exclusion_reasons


# =============================================================================
# ENSEMBLE WEIGHTS (Multiple framework patterns)
# =============================================================================

@log_io
def calculate_ensemble_weights(
    model_results: List[Dict[str, Any]],
    method: str = 'inverse_mape',
    min_weight: float = 0.05,
    max_weight: float = 0.40,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Calculate ensemble weights using various methods.

    Methods:
        - 'inverse_mape': Weight inversely proportional to MAPE (AutoGluon)
        - 'inverse_mape_squared': Stronger penalty for high MAPE
        - 'softmax': Softmax of negative MAPE (smoother distribution)
        - 'equal': Equal weights for all models
        - 'rank': Based on rank rather than absolute MAPE

    Args:
        model_results: List of model results with 'model_name' and 'mape'
        method: Weighting method
        min_weight: Minimum weight for any model
        max_weight: Maximum weight for any single model (prevents domination)
        normalize: Whether to normalize weights to sum to 1

    Returns:
        Dict mapping model_name to weight
    """
    if not model_results:
        return {}

    # Extract model names and MAPE values
    models_mape = []
    for model in model_results:
        name = model.get('model_name') or model.get('model_type') or 'Unknown'
        mape = model.get('mape') or model.get('metrics', {}).get('mape')
        try:
            mape = float(mape)
            if not np.isnan(mape) and not np.isinf(mape) and mape > 0:
                models_mape.append((name, mape))
        except (ValueError, TypeError):
            continue

    if not models_mape:
        return {}

    weights = {}

    if method == 'equal':
        weight = 1.0 / len(models_mape)
        weights = {name: weight for name, _ in models_mape}

    elif method == 'inverse_mape':
        for name, mape in models_mape:
            weights[name] = 1.0 / (mape + 1e-10)

    elif method == 'inverse_mape_squared':
        for name, mape in models_mape:
            weights[name] = 1.0 / ((mape + 1e-10) ** 2)

    elif method == 'softmax':
        mapes = np.array([mape for _, mape in models_mape])
        # Use negative MAPE (lower is better)
        exp_scores = np.exp(-mapes / np.mean(mapes))
        for (name, _), w in zip(models_mape, exp_scores):
            weights[name] = w

    elif method == 'rank':
        # Sort by MAPE
        sorted_models = sorted(models_mape, key=lambda x: x[1])
        n = len(sorted_models)
        for rank, (name, _) in enumerate(sorted_models):
            # Higher rank = lower MAPE = higher weight
            weights[name] = n - rank

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Apply minimum weight
    for name in weights:
        weights[name] = max(weights[name], min_weight)

    # Normalize first pass
    if normalize:
        total = sum(weights.values())
        weights = {name: w / total for name, w in weights.items()}

    # Apply maximum weight cap to prevent single-model domination
    # A model with very low eval MAPE (e.g., 0.69%) can get 53% weight via
    # inverse_mape, which contaminates the ensemble if the model over-extrapolates.
    if max_weight < 1.0 and len(weights) > 1:
        # Iterate cap+normalize until stable — a single pass can push weights
        # above the cap after renormalization redistributes the excess.
        for _ in range(10):  # converges in 2-3 iterations
            capped = False
            for name in weights:
                if weights[name] > max_weight:
                    weights[name] = max_weight
                    capped = True
            if not capped:
                break
            if normalize:
                total = sum(weights.values())
                weights = {name: w / total for name, w in weights.items()}

    return weights


# =============================================================================
# FEATURE ENGINEERING (MLForecast patterns)
# =============================================================================

@log_io
def create_lag_features(
    df: pd.DataFrame,
    target_col: str,
    frequency: str = 'weekly',
    custom_lags: List[int] = None
) -> pd.DataFrame:
    """
    Create lag features based on frequency (MLForecast pattern).

    Args:
        df: DataFrame with target column
        target_col: Name of target column
        frequency: Data frequency
        custom_lags: Optional custom lag periods

    Returns:
        DataFrame with lag features added
    """
    df = df.copy()

    # Default lag configurations by frequency
    LAG_CONFIGS = {
        'daily': {
            'lags': [1, 7, 14, 28, 364],
            'rolling': [(7, 'mean'), (7, 'std'), (28, 'mean')],
        },
        'weekly': {
            'lags': [1, 4, 13, 26, 52],
            'rolling': [(4, 'mean'), (4, 'std'), (13, 'mean')],
        },
        'monthly': {
            'lags': [1, 3, 6, 12],
            'rolling': [(3, 'mean'), (3, 'std'), (6, 'mean')],
        },
    }

    config = LAG_CONFIGS.get(frequency, LAG_CONFIGS['weekly'])
    lags = custom_lags if custom_lags else config['lags']

    # Create lag features
    for lag in lags:
        if len(df) > lag:
            col_name = f'lag_{lag}'
            df[col_name] = df[target_col].shift(lag)
            logger.debug(f"Created lag feature: {col_name}")

    # Create rolling features (shifted to avoid leakage)
    for window, agg in config['rolling']:
        if len(df) > window:
            col_name = f'rolling_{agg}_{window}'
            df[col_name] = df[target_col].shift(1).rolling(window, min_periods=1).agg(agg)
            logger.debug(f"Created rolling feature: {col_name}")

    return df


# =============================================================================
# MONTE CARLO PREDICTION INTERVALS (GluonTS pattern)
# =============================================================================

@log_io
def generate_monte_carlo_intervals(
    point_forecasts: np.ndarray,
    residuals: np.ndarray,
    n_samples: int = 1000,
    confidence_levels: List[float] = [0.5, 0.8, 0.95]
) -> Dict[str, np.ndarray]:
    """
    Generate prediction intervals using Monte Carlo sampling (GluonTS pattern).

    Args:
        point_forecasts: Array of point predictions
        residuals: Historical residuals (actuals - predictions)
        n_samples: Number of Monte Carlo samples
        confidence_levels: List of confidence levels (e.g., [0.5, 0.8, 0.95])

    Returns:
        Dict with 'yhat' and interval bounds for each confidence level
    """
    horizon = len(point_forecasts)

    # Sample residuals and add to forecasts
    samples = []
    for _ in range(n_samples):
        # Bootstrap residuals
        sampled_residuals = np.random.choice(residuals, size=horizon, replace=True)
        sample = point_forecasts + sampled_residuals
        samples.append(sample)

    samples = np.array(samples)

    # Compute quantiles
    result = {'yhat': point_forecasts}

    for level in confidence_levels:
        alpha = 1 - level
        lower_q = alpha / 2 * 100
        upper_q = (1 - alpha / 2) * 100

        level_str = int(level * 100)
        result[f'lower_{level_str}'] = np.percentile(samples, lower_q, axis=0)
        result[f'upper_{level_str}'] = np.percentile(samples, upper_q, axis=0)

    return result


# =============================================================================
# MODEL REGISTRY VALIDATION (MLflow patterns)
# =============================================================================

@dataclass
class RegistrationValidation:
    """Validation result for model registration."""
    should_register: bool
    checks_passed: Dict[str, bool]
    reasons: List[str]


@log_io
def validate_for_registration(
    model_result: Dict[str, Any],
    metrics: Dict[str, float],
    validation_result: ForecastValidationResult,
    overfitting_report: OverfittingReport,
    min_mape_threshold: float = 50.0,
    max_overfitting_severity: str = 'high'
) -> RegistrationValidation:
    """
    Validate model before registering to model registry.

    Args:
        model_result: Model result dict
        metrics: Model metrics dict
        validation_result: Forecast validation result
        overfitting_report: Overfitting detection report
        min_mape_threshold: Maximum acceptable MAPE
        max_overfitting_severity: Maximum acceptable overfitting severity

    Returns:
        RegistrationValidation result
    """
    checks = {}
    reasons = []

    # Check 1: Predictions are valid
    checks['predictions_valid'] = validation_result.is_valid
    if not validation_result.is_valid:
        reasons.append(f"Invalid predictions: {validation_result.issues}")

    # Check 2: MAPE is reasonable
    mape = metrics.get('mape', float('inf'))
    checks['mape_reasonable'] = mape < min_mape_threshold
    if mape >= min_mape_threshold:
        reasons.append(f"MAPE too high: {mape:.2f}% (threshold: {min_mape_threshold}%)")

    # Check 3: Not severely overfitting
    severity_order = ['none', 'low', 'medium', 'high', 'severe']
    max_severity_idx = severity_order.index(max_overfitting_severity)
    actual_severity_idx = severity_order.index(overfitting_report.severity)
    checks['no_severe_overfitting'] = actual_severity_idx <= max_severity_idx
    if actual_severity_idx > max_severity_idx:
        reasons.append(f"Overfitting too severe: {overfitting_report.severity}")

    # Check 4: Has required artifacts
    has_run_id = model_result.get('run_id') is not None
    checks['has_artifacts'] = has_run_id
    if not has_run_id:
        reasons.append("Missing MLflow run ID")

    # Determine if should register
    should_register = all(checks.values())

    return RegistrationValidation(
        should_register=should_register,
        checks_passed=checks,
        reasons=reasons
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@log_io
def log_automl_summary(
    data_quality: DataQualityReport,
    model_results: List[Dict[str, Any]],
    selected_models: List[Dict[str, Any]],
    best_model: str,
    preset_used: Optional[str] = None
):
    """Log a summary of the AutoML training run."""
    logger.info("=" * 60)
    logger.info("AutoML Training Summary")
    logger.info("=" * 60)

    if preset_used:
        logger.info(f"Preset used: {preset_used}")

    logger.info(f"Data quality: {'PASS' if data_quality.is_valid else 'ISSUES FOUND'}")
    if data_quality.warnings:
        logger.info(f"  Warnings: {len(data_quality.warnings)}")

    logger.info(f"Models trained: {len(model_results)}")
    logger.info(f"Models selected for ensemble: {len(selected_models)}")
    logger.info(f"Best model: {best_model}")

    logger.info("=" * 60)
