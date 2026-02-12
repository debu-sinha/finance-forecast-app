"""
Intelligent Data Analysis Module for Time Series Forecasting

This module analyzes training data characteristics and provides recommendations for:
1. Which models are most suitable for the data
2. Which hyperparameters to include/exclude based on data size and patterns
3. Whether the data has enough history for reliable forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from backend.utils.logging_utils import log_io

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INSUFFICIENT = "insufficient"


class TrendType(Enum):
    """Types of trend in time series"""
    STRONG_UP = "strong_upward"
    MODERATE_UP = "moderate_upward"
    FLAT = "flat"
    MODERATE_DOWN = "moderate_downward"
    STRONG_DOWN = "strong_downward"


class SeasonalityType(Enum):
    """Types of seasonality detected"""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


@dataclass
class DataCharacteristics:
    """Container for all analyzed data characteristics"""
    # Basic stats
    n_observations: int = 0
    n_years: float = 0.0
    frequency: str = "unknown"

    # Time coverage
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    has_gaps: bool = False
    gap_count: int = 0

    # Target statistics
    mean_value: float = 0.0
    std_value: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    cv: float = 0.0  # Coefficient of variation
    has_negative_values: bool = False
    has_zero_values: bool = False

    # Pattern detection
    trend_type: TrendType = TrendType.FLAT
    trend_strength: float = 0.0
    seasonality_type: SeasonalityType = SeasonalityType.NONE
    seasonality_strength: float = 0.0
    has_outliers: bool = False
    outlier_percentage: float = 0.0

    # Autocorrelation
    has_autocorrelation: bool = False
    acf_lag1: float = 0.0

    # Data quality
    quality: DataQuality = DataQuality.FAIR
    quality_score: float = 0.0

    # Warnings and notes
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class ModelRecommendation:
    """Recommendation for a specific model"""
    model_name: str
    recommended: bool
    confidence: float  # 0-1 scale
    reason: str
    hyperparameter_filter: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete analysis result with characteristics and recommendations"""
    characteristics: DataCharacteristics
    model_recommendations: List[ModelRecommendation]
    recommended_models: List[str]
    excluded_models: List[str]
    hyperparameter_filters: Dict[str, Dict[str, Any]]
    overall_recommendation: str


@log_io
def analyze_time_series(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    frequency: str = "auto"
) -> AnalysisResult:
    """
    Analyze time series data and provide model/hyperparameter recommendations.

    Args:
        df: DataFrame with time series data
        time_col: Name of the datetime column
        target_col: Name of the target column to forecast
        frequency: Expected frequency ('daily', 'weekly', 'monthly', 'auto')

    Returns:
        AnalysisResult with characteristics and recommendations
    """
    logger.info(f"📊 Analyzing time series data: {len(df)} rows, target={target_col}")

    # Prepare data
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    # Extract characteristics
    chars = _analyze_characteristics(df, time_col, target_col, frequency)

    # Generate model recommendations based on characteristics
    recommendations = _generate_model_recommendations(chars)

    # Generate hyperparameter filters
    hp_filters = _generate_hyperparameter_filters(chars)

    # Compile results
    recommended = [r.model_name for r in recommendations if r.recommended]
    excluded = [r.model_name for r in recommendations if not r.recommended]

    overall = _generate_overall_recommendation(chars, recommendations)

    result = AnalysisResult(
        characteristics=chars,
        model_recommendations=recommendations,
        recommended_models=recommended,
        excluded_models=excluded,
        hyperparameter_filters=hp_filters,
        overall_recommendation=overall
    )

    _log_analysis_summary(result)

    return result


@log_io
def _analyze_characteristics(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    frequency: str
) -> DataCharacteristics:
    """Extract all characteristics from the data."""
    chars = DataCharacteristics()

    # Basic stats
    chars.n_observations = len(df)
    values = df[target_col].dropna().values

    if len(values) == 0:
        chars.quality = DataQuality.INSUFFICIENT
        chars.warnings.append("No valid target values found")
        return chars

    # Detect frequency
    if frequency == "auto":
        frequency = _detect_frequency(df, time_col)
    chars.frequency = frequency

    # Time coverage
    chars.start_date = str(df[time_col].min().date())
    chars.end_date = str(df[time_col].max().date())

    date_range = (df[time_col].max() - df[time_col].min()).days
    chars.n_years = date_range / 365.25

    # Check for gaps
    chars.has_gaps, chars.gap_count = _check_for_gaps(df, time_col, frequency)

    # Target statistics
    chars.mean_value = float(np.mean(values))
    chars.std_value = float(np.std(values))
    chars.min_value = float(np.min(values))
    chars.max_value = float(np.max(values))
    chars.cv = chars.std_value / abs(chars.mean_value) if chars.mean_value != 0 else 0
    chars.has_negative_values = bool(np.any(values < 0))
    chars.has_zero_values = bool(np.any(values == 0))

    # Detect trend
    chars.trend_type, chars.trend_strength = _detect_trend(values)

    # Detect seasonality
    chars.seasonality_type, chars.seasonality_strength = _detect_seasonality(
        values, frequency
    )

    # Detect outliers
    chars.has_outliers, chars.outlier_percentage = _detect_outliers(values)

    # Check autocorrelation
    chars.has_autocorrelation, chars.acf_lag1 = _check_autocorrelation(values)

    # Calculate quality score
    chars.quality, chars.quality_score = _calculate_quality(chars)

    # Add warnings
    _add_warnings(chars)

    return chars


@log_io
def _detect_frequency(df: pd.DataFrame, time_col: str) -> str:
    """Auto-detect the frequency of the time series."""
    if len(df) < 2:
        return "unknown"

    diffs = df[time_col].diff().dropna()
    median_diff = diffs.median().days

    if median_diff <= 1:
        return "daily"
    elif 5 <= median_diff <= 9:
        return "weekly"
    elif 25 <= median_diff <= 35:
        return "monthly"
    elif 85 <= median_diff <= 95:
        return "quarterly"
    elif 350 <= median_diff <= 380:
        return "yearly"
    else:
        return "irregular"


@log_io
def _check_for_gaps(df: pd.DataFrame, time_col: str, frequency: str) -> Tuple[bool, int]:
    """Check for missing time periods in the data."""
    if len(df) < 2:
        return False, 0

    expected_diff_days = {
        "daily": 1,
        "weekly": 7,
        "monthly": 30,
        "quarterly": 91,
        "yearly": 365
    }

    expected = expected_diff_days.get(frequency, 7)
    tolerance = expected * 0.5  # 50% tolerance

    diffs = df[time_col].diff().dropna().dt.days
    gaps = diffs[diffs > (expected + tolerance)]

    return len(gaps) > 0, len(gaps)


@log_io
def _detect_trend(values: np.ndarray) -> Tuple[TrendType, float]:
    """Detect trend in the time series using linear regression."""
    if len(values) < 3:
        return TrendType.FLAT, 0.0

    x = np.arange(len(values))

    # Use polyfit for trend
    try:
        slope, _ = np.polyfit(x, values, 1)

        # Normalize slope by mean value to get relative trend
        mean_val = np.mean(np.abs(values)) + 1e-10
        relative_slope = slope * len(values) / mean_val

        # Calculate R² for trend strength
        y_pred = slope * x + _
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2) + 1e-10
        r2 = 1 - (ss_res / ss_tot)

        strength = abs(r2) if r2 > 0 else 0

        if relative_slope > 0.3:
            trend_type = TrendType.STRONG_UP
        elif relative_slope > 0.1:
            trend_type = TrendType.MODERATE_UP
        elif relative_slope < -0.3:
            trend_type = TrendType.STRONG_DOWN
        elif relative_slope < -0.1:
            trend_type = TrendType.MODERATE_DOWN
        else:
            trend_type = TrendType.FLAT

        return trend_type, float(strength)

    except Exception:
        return TrendType.FLAT, 0.0


@log_io
def _detect_seasonality(values: np.ndarray, frequency: str) -> Tuple[SeasonalityType, float]:
    """Detect seasonality using autocorrelation analysis."""
    if len(values) < 4:
        return SeasonalityType.NONE, 0.0

    # Expected seasonal period
    periods = {
        "daily": 7,      # Weekly seasonality
        "weekly": 52,    # Yearly seasonality
        "monthly": 12,   # Yearly seasonality
        "quarterly": 4,  # Yearly seasonality
    }

    period = periods.get(frequency, 12)

    if len(values) < period * 2:
        # Not enough data to detect seasonality
        return SeasonalityType.NONE, 0.0

    try:
        # Calculate autocorrelation at seasonal lag
        mean_val = np.mean(values)
        var_val = np.var(values) + 1e-10

        n = len(values)
        if n <= period:
            return SeasonalityType.NONE, 0.0

        acf = np.sum((values[:n-period] - mean_val) * (values[period:] - mean_val))
        acf /= (n - period) * var_val

        strength = abs(acf)

        if strength > 0.7:
            return SeasonalityType.STRONG, strength
        elif strength > 0.4:
            return SeasonalityType.MODERATE, strength
        elif strength > 0.2:
            return SeasonalityType.WEAK, strength
        else:
            return SeasonalityType.NONE, strength

    except Exception:
        return SeasonalityType.NONE, 0.0


@log_io
def _detect_seasonality_mode(chars: DataCharacteristics) -> str:
    """
    Determine whether multiplicative or additive seasonality is more appropriate.

    Multiplicative seasonality: seasonal amplitude grows/shrinks proportionally with trend.
        - Common in financial data (revenue, sales) where seasonal peaks scale with growth.
        - Use when: data has positive trend AND seasonal amplitude increases over time.

    Additive seasonality: seasonal amplitude stays constant regardless of trend.
        - Use when: seasonal patterns are consistent in absolute terms.
        - Required when: data has zero or negative values (multiplicative can't handle these).

    Returns:
        'multiplicative' or 'additive'
    """
    # If data has negative or zero values, must use additive
    if chars.has_negative_values or chars.has_zero_values:
        logger.info("📊 Seasonality mode: additive (data has zero/negative values)")
        return 'additive'

    # If no clear trend, default to additive (simpler model)
    if chars.trend_type == TrendType.FLAT:
        logger.info("📊 Seasonality mode: additive (no significant trend detected)")
        return 'additive'

    # If we have a trend and significant seasonality, multiplicative often works better
    # for financial/business data where seasonal effects scale with the level
    if chars.trend_strength > 0.3 and chars.seasonality_strength > 0.3:
        logger.info(f"📊 Seasonality mode: multiplicative (trend_strength={chars.trend_strength:.2f}, seasonality_strength={chars.seasonality_strength:.2f})")
        return 'multiplicative'

    # For moderate patterns with positive values, multiplicative is generally safer for finance
    if chars.trend_type in [TrendType.STRONG_UP, TrendType.MODERATE_UP] and chars.cv > 0.2:
        logger.info(f"📊 Seasonality mode: multiplicative (upward trend with CV={chars.cv:.2f})")
        return 'multiplicative'

    # Default to additive for simpler cases
    logger.info("📊 Seasonality mode: additive (default for simpler patterns)")
    return 'additive'


@log_io
def _detect_outliers(values: np.ndarray) -> Tuple[bool, float]:
    """Detect outliers using IQR method."""
    if len(values) < 4:
        return False, 0.0

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = (values < lower_bound) | (values > upper_bound)
    outlier_pct = np.sum(outliers) / len(values) * 100

    return outlier_pct > 1, float(outlier_pct)


@log_io
def _check_autocorrelation(values: np.ndarray) -> Tuple[bool, float]:
    """Check for lag-1 autocorrelation."""
    if len(values) < 3:
        return False, 0.0

    try:
        mean_val = np.mean(values)
        var_val = np.var(values) + 1e-10

        n = len(values)
        acf = np.sum((values[:-1] - mean_val) * (values[1:] - mean_val))
        acf /= (n - 1) * var_val

        return abs(acf) > 0.3, float(acf)
    except Exception:
        return False, 0.0


@log_io
def _calculate_quality(chars: DataCharacteristics) -> Tuple[DataQuality, float]:
    """Calculate overall data quality score."""
    score = 100.0

    # Penalize for insufficient data
    if chars.n_observations < 12:
        score -= 50
    elif chars.n_observations < 24:
        score -= 30
    elif chars.n_observations < 52:
        score -= 15

    # Penalize for short history
    if chars.n_years < 1:
        score -= 20
    elif chars.n_years < 2:
        score -= 10

    # Penalize for gaps
    if chars.has_gaps:
        score -= min(20, chars.gap_count * 2)

    # Penalize for high variance (but some variance is normal)
    if chars.cv > 1.0:
        score -= 10

    # Penalize for outliers
    if chars.outlier_percentage > 5:
        score -= 15
    elif chars.outlier_percentage > 2:
        score -= 5

    # Bonus for detected patterns (easier to forecast)
    if chars.seasonality_type in [SeasonalityType.STRONG, SeasonalityType.MODERATE]:
        score += 5
    if chars.trend_strength > 0.5:
        score += 5

    score = max(0, min(100, score))

    if score >= 80:
        quality = DataQuality.EXCELLENT
    elif score >= 60:
        quality = DataQuality.GOOD
    elif score >= 40:
        quality = DataQuality.FAIR
    elif score >= 20:
        quality = DataQuality.POOR
    else:
        quality = DataQuality.INSUFFICIENT

    return quality, score


@log_io
def _add_warnings(chars: DataCharacteristics) -> None:
    """Add relevant warnings based on characteristics."""
    if chars.n_observations < 52:
        chars.warnings.append(
            f"Limited data: Only {chars.n_observations} observations. "
            "Consider using simpler models."
        )

    if chars.n_years < 1:
        chars.warnings.append(
            f"Short history: Only {chars.n_years:.1f} years of data. "
            "Yearly seasonality may not be reliable."
        )

    if chars.has_gaps:
        chars.warnings.append(
            f"Data gaps detected: {chars.gap_count} missing periods found."
        )

    if chars.has_negative_values:
        chars.warnings.append(
            "Negative values present. Some models may have issues."
        )

    if chars.outlier_percentage > 5:
        chars.warnings.append(
            f"High outlier rate: {chars.outlier_percentage:.1f}% of values are outliers."
        )

    if chars.cv > 1.5:
        chars.warnings.append(
            f"High volatility: Coefficient of variation is {chars.cv:.2f}."
        )


@log_io
def _generate_model_recommendations(chars: DataCharacteristics) -> List[ModelRecommendation]:
    """Generate model recommendations based on data characteristics.

    Model defaults based on 12-slice benchmark testing (TOT_VOL + TOT_SUB,
    Oct 2025 cutoff, 12-week horizon). ARIMA, SARIMAX, and XGBoost are disabled
    by default due to reliability issues. See Forecast_Data_Quality_Analysis.md.
    """
    recommendations = []

    # Prophet recommendation
    prophet_rec = _recommend_prophet(chars)
    recommendations.append(prophet_rec)

    # ARIMA recommendation
    arima_rec = _recommend_arima(chars)
    recommendations.append(arima_rec)

    # SARIMAX recommendation (ARIMA with exogenous variables)
    sarimax_rec = _recommend_sarimax(chars)
    recommendations.append(sarimax_rec)

    # ETS recommendation
    ets_rec = _recommend_ets(chars)
    recommendations.append(ets_rec)

    # XGBoost recommendation
    xgb_rec = _recommend_xgboost(chars)
    recommendations.append(xgb_rec)

    # StatsForecast recommendation (AutoARIMA, AutoETS, AutoTheta)
    statsforecast_rec = _recommend_statsforecast(chars)
    recommendations.append(statsforecast_rec)

    # Chronos recommendation (Zero-shot foundation model)
    chronos_rec = _recommend_chronos(chars)
    recommendations.append(chronos_rec)

    return recommendations


@log_io
def _recommend_prophet(chars: DataCharacteristics) -> ModelRecommendation:
    """Generate Prophet model recommendation."""
    confidence = 0.7  # Base confidence
    reasons = []
    hp_filter = {}

    # Prophet excels with seasonal data
    if chars.seasonality_type in [SeasonalityType.STRONG, SeasonalityType.MODERATE]:
        confidence += 0.2
        reasons.append("Strong/moderate seasonality detected - Prophet handles this well")

    # Prophet needs reasonable data volume
    if chars.n_observations < 52:
        confidence -= 0.3
        reasons.append("Limited data points may reduce Prophet accuracy")
        hp_filter['changepoint_prior_scale'] = [0.05, 0.1]  # Less flexible

    if chars.n_years < 2:
        hp_filter['yearly_seasonality'] = [False]  # Disable yearly
        reasons.append("Disabling yearly seasonality due to short history")

    # Prophet handles outliers reasonably well
    if chars.has_outliers and chars.outlier_percentage > 5:
        confidence -= 0.1
        reasons.append("Some outliers may affect changepoint detection")

    # Trend handling
    if chars.trend_type in [TrendType.STRONG_UP, TrendType.STRONG_DOWN]:
        reasons.append("Prophet can model the detected trend")

    recommended = confidence >= 0.5

    return ModelRecommendation(
        model_name="Prophet",
        recommended=recommended,
        confidence=min(1.0, max(0.0, confidence)),
        reason="; ".join(reasons) if reasons else "Standard Prophet recommendation",
        hyperparameter_filter=hp_filter
    )


@log_io
def _recommend_arima(chars: DataCharacteristics) -> ModelRecommendation:
    """Generate ARIMA model recommendation.

    Disabled by default based on 12-slice benchmark testing (TOT_VOL + TOT_SUB,
    Oct 2025 cutoff, 12-week horizon). StatsForecast AutoARIMA provides equivalent
    functionality with better reliability. ARIMA frequently fails or produces
    degenerate (0,1,0) flat forecasts.
    """
    reasons = [
        "Disabled by default: StatsForecast AutoARIMA provides equivalent "
        "functionality with better reliability. ARIMA frequently fails or "
        "produces degenerate (0,1,0) flat forecasts."
    ]
    hp_filter = {}

    # Keep hyperparameter logic for users who force-enable ARIMA
    if chars.n_observations < 30:
        hp_filter['max_p'] = 2
        hp_filter['max_q'] = 2
    elif chars.n_observations < 100:
        hp_filter['max_p'] = 3
        hp_filter['max_q'] = 3

    if chars.trend_type != TrendType.FLAT:
        hp_filter['d_values'] = [1, 2]

    return ModelRecommendation(
        model_name="ARIMA",
        recommended=False,
        confidence=0.0,
        reason="; ".join(reasons),
        hyperparameter_filter=hp_filter
    )


@log_io
def _recommend_sarimax(chars: DataCharacteristics) -> ModelRecommendation:
    """Generate SARIMAX (Seasonal ARIMA with eXogenous variables) recommendation.

    Disabled by default based on 12-slice benchmark testing (TOT_VOL + TOT_SUB,
    Oct 2025 cutoff, 12-week horizon). SARIMAX is numerically unstable and has
    produced forecast explosions exceeding +/-100 billion in testing. Use Prophet
    with covariates instead for seasonal modeling with exogenous variables.
    """
    reasons = [
        "Disabled by default: numerically unstable — observed forecast explosions "
        "exceeding +/-100 billion in benchmark testing. Use Prophet with covariates instead."
    ]
    hp_filter = {}

    # Keep hyperparameter logic for users who force-enable SARIMAX
    if chars.n_observations < 52:
        hp_filter['p_values'] = [0, 1, 2]
        hp_filter['d_values'] = [0, 1]
        hp_filter['q_values'] = [0, 1]
    elif chars.n_observations < 104:
        hp_filter['p_values'] = [0, 1, 2, 3]
        hp_filter['d_values'] = [0, 1, 2]
        hp_filter['q_values'] = [0, 1, 2]

    return ModelRecommendation(
        model_name="SARIMAX",
        recommended=False,
        confidence=0.0,
        reason="; ".join(reasons),
        hyperparameter_filter=hp_filter
    )


@log_io
def _recommend_ets(chars: DataCharacteristics) -> ModelRecommendation:
    """Generate ETS (Exponential Smoothing) recommendation."""
    confidence = 0.6  # Base confidence
    reasons = []
    hp_filter = {}

    # ETS works well with clear patterns
    if chars.seasonality_type in [SeasonalityType.STRONG, SeasonalityType.MODERATE]:
        confidence += 0.15
        reasons.append("Seasonal patterns suit ETS models")
        hp_filter['seasonal'] = ['add', 'mul']
    else:
        hp_filter['seasonal'] = [None]
        reasons.append("No strong seasonality - using non-seasonal ETS")

    if chars.trend_type != TrendType.FLAT:
        hp_filter['trend'] = ['add', 'mul']
        reasons.append("Trend component will be modeled")
    else:
        hp_filter['trend'] = [None, 'add']

    # ETS needs reasonable data
    if chars.n_observations < 24:
        confidence -= 0.2
        reasons.append("Limited data may affect ETS reliability")

    # ETS struggles with negative values in multiplicative mode
    if chars.has_negative_values:
        hp_filter['trend'] = [None, 'add']
        hp_filter['seasonal'] = [None, 'add']
        reasons.append("Negative values - using additive components only")

    recommended = confidence >= 0.5

    return ModelRecommendation(
        model_name="ETS",
        recommended=recommended,
        confidence=min(1.0, max(0.0, confidence)),
        reason="; ".join(reasons) if reasons else "Standard ETS recommendation",
        hyperparameter_filter=hp_filter
    )


@log_io
def _recommend_xgboost(chars: DataCharacteristics) -> ModelRecommendation:
    """Generate XGBoost recommendation.

    Disabled by default based on 12-slice benchmark testing (TOT_VOL + TOT_SUB,
    Oct 2025 cutoff, 12-week horizon). XGBoost cannot extrapolate beyond the
    training data range, causing systematic under-prediction on trending data
    (-48% error observed). Use StatsForecast or Prophet for trend-following.
    """
    reasons = [
        "Disabled by default: cannot extrapolate beyond training range, causing "
        "systematic under-prediction on trending data (-48% error observed). "
        "Use StatsForecast or Prophet for trends."
    ]
    hp_filter = {}

    # Keep hyperparameter logic for users who force-enable XGBoost
    if chars.n_observations < 100:
        hp_filter['n_estimators'] = [50, 100]
        hp_filter['max_depth'] = [3]
    elif chars.n_observations > 500:
        hp_filter['n_estimators'] = [100, 200, 300]
        hp_filter['max_depth'] = [3, 5, 7]

    if chars.cv > 1.0:
        hp_filter['learning_rate'] = [0.05, 0.1]

    return ModelRecommendation(
        model_name="XGBoost",
        recommended=False,
        confidence=0.0,
        reason="; ".join(reasons),
        hyperparameter_filter=hp_filter
    )


@log_io
def _recommend_statsforecast(chars: DataCharacteristics) -> ModelRecommendation:
    """Generate StatsForecast (AutoARIMA, AutoETS, AutoTheta) recommendation."""
    confidence = 0.75  # Base confidence - StatsForecast is generally reliable
    reasons = []
    hp_filter = {}

    # StatsForecast is 10-100x faster than traditional implementations
    reasons.append("StatsForecast provides 10-100x faster statistical models")

    # Works well with autocorrelated data (like ARIMA)
    if chars.has_autocorrelation:
        confidence += 0.1
        reasons.append("AutoARIMA will capture autocorrelation patterns effectively")

    # Handles seasonality well
    if chars.seasonality_type in [SeasonalityType.STRONG, SeasonalityType.MODERATE]:
        confidence += 0.1
        reasons.append("AutoETS and AutoTheta handle seasonal patterns well")

    # Needs reasonable data volume
    if chars.n_observations < 24:
        confidence -= 0.2
        reasons.append("Limited observations - simpler models will be selected")
        hp_filter['model_type'] = ['auto']  # Let it auto-select
    elif chars.n_observations < 52:
        confidence -= 0.1
        reasons.append("Moderate data - seasonal models may be limited")
        hp_filter['model_type'] = ['auto', 'autoarima']

    # Set season length based on frequency
    if chars.frequency == 'daily':
        hp_filter['season_length'] = 7
    elif chars.frequency == 'weekly':
        hp_filter['season_length'] = 52
    elif chars.frequency == 'monthly':
        hp_filter['season_length'] = 12
    elif chars.frequency == 'quarterly':
        hp_filter['season_length'] = 4

    # StatsForecast handles trends automatically
    if chars.trend_type != TrendType.FLAT:
        reasons.append("Automatic differencing for non-stationary trends")

    recommended = confidence >= 0.5

    return ModelRecommendation(
        model_name="StatsForecast",
        recommended=recommended,
        confidence=min(1.0, max(0.0, confidence)),
        reason="; ".join(reasons) if reasons else "Fast statistical forecasting with AutoARIMA, AutoETS, AutoTheta",
        hyperparameter_filter=hp_filter
    )


@log_io
def _recommend_chronos(chars: DataCharacteristics) -> ModelRecommendation:
    """Generate Chronos (Amazon's zero-shot foundation model) recommendation."""
    confidence = 0.70  # Base confidence - Chronos is a solid baseline
    reasons = []
    hp_filter = {}

    # Chronos is zero-shot - no training required
    reasons.append("Zero-shot foundation model - no training needed, instant forecasts")

    # Works well as a baseline for any data
    reasons.append("Pre-trained on diverse time series - generalizes well")

    # Small datasets are fine - Chronos doesn't need training
    if chars.n_observations < 52:
        confidence += 0.1
        reasons.append("Excellent for limited data - leverages pre-training")
        hp_filter['model_size'] = 'small'  # Use smaller model for speed
    elif chars.n_observations >= 104:
        # With more data, trained models may outperform zero-shot
        confidence -= 0.1
        reasons.append("Sufficient data for trained models - use Chronos as baseline")
        hp_filter['model_size'] = 'base'  # Use larger model for accuracy

    # Handles various patterns without explicit configuration
    if chars.seasonality_type in [SeasonalityType.STRONG, SeasonalityType.MODERATE]:
        reasons.append("Captures seasonal patterns from pre-training")

    if chars.trend_type != TrendType.FLAT:
        reasons.append("Handles trends without explicit differencing")

    # Chronos struggles less with outliers due to robust pre-training
    if chars.has_outliers:
        confidence += 0.05
        reasons.append("Robust to outliers due to diverse pre-training data")

    # Model size recommendation based on compute needs
    if chars.n_observations > 500:
        hp_filter['model_size'] = 'base'
        reasons.append("Larger model recommended for complex patterns")
    elif chars.n_observations < 100:
        hp_filter['model_size'] = 'small'
        reasons.append("Smaller model sufficient and faster")
    else:
        hp_filter['model_size'] = 'small'  # Default to small for balance

    recommended = confidence >= 0.5

    return ModelRecommendation(
        model_name="Chronos",
        recommended=recommended,
        confidence=min(1.0, max(0.0, confidence)),
        reason="; ".join(reasons) if reasons else "Amazon's pre-trained foundation model for zero-shot forecasting",
        hyperparameter_filter=hp_filter
    )


@log_io
def _generate_hyperparameter_filters(chars: DataCharacteristics) -> Dict[str, Dict[str, Any]]:
    """Generate hyperparameter filters for all models."""
    filters = {}

    # Prophet hyperparameters
    prophet_hp = {}
    if chars.n_observations < 52:
        prophet_hp['changepoint_prior_scale'] = [0.01, 0.05, 0.1]
        prophet_hp['seasonality_prior_scale'] = [0.1, 1.0]
    elif chars.n_observations < 104:
        prophet_hp['changepoint_prior_scale'] = [0.01, 0.05, 0.1, 0.5]
        prophet_hp['seasonality_prior_scale'] = [0.1, 1.0, 10.0]
    # Full grid for larger datasets

    if chars.n_years < 2:
        prophet_hp['yearly_seasonality'] = [False]

    if chars.frequency == 'daily':
        prophet_hp['weekly_seasonality'] = [True]
    elif chars.frequency == 'monthly':
        prophet_hp['weekly_seasonality'] = [False]

    # Determine seasonality_mode based on data characteristics
    # Multiplicative: seasonal amplitude grows with trend (common in financial data)
    # Additive: seasonal amplitude stays constant
    seasonality_mode = _detect_seasonality_mode(chars)
    prophet_hp['seasonality_mode'] = [seasonality_mode]

    filters['Prophet'] = prophet_hp

    # ARIMA hyperparameters
    arima_hp = {}
    if chars.n_observations < 50:
        arima_hp['p_values'] = [0, 1, 2]
        arima_hp['d_values'] = [0, 1]
        arima_hp['q_values'] = [0, 1, 2]
    elif chars.n_observations < 100:
        arima_hp['p_values'] = [0, 1, 2, 3]
        arima_hp['d_values'] = [0, 1, 2]
        arima_hp['q_values'] = [0, 1, 2, 3]
    # Full grid for larger datasets

    filters['ARIMA'] = arima_hp

    # SARIMAX hyperparameters (similar to ARIMA but used with covariates)
    sarimax_hp = {}
    if chars.n_observations < 50:
        sarimax_hp['p_values'] = [0, 1, 2]
        sarimax_hp['d_values'] = [0, 1]
        sarimax_hp['q_values'] = [0, 1, 2]
    elif chars.n_observations < 100:
        sarimax_hp['p_values'] = [0, 1, 2, 3]
        sarimax_hp['d_values'] = [0, 1, 2]
        sarimax_hp['q_values'] = [0, 1, 2, 3]
    # Full grid for larger datasets

    filters['SARIMAX'] = sarimax_hp

    # ETS hyperparameters
    ets_hp = {}
    if chars.has_negative_values:
        ets_hp['trend'] = [None, 'add']
        ets_hp['seasonal'] = [None, 'add']
    elif chars.seasonality_type == SeasonalityType.NONE:
        ets_hp['seasonal'] = [None]

    filters['ETS'] = ets_hp

    # XGBoost hyperparameters
    xgb_hp = {}
    if chars.n_observations < 100:
        xgb_hp['n_estimators'] = [50, 100]
        xgb_hp['max_depth'] = [3]
        xgb_hp['learning_rate'] = [0.1]
    elif chars.n_observations < 500:
        xgb_hp['n_estimators'] = [100, 200]
        xgb_hp['max_depth'] = [3, 5]
        xgb_hp['learning_rate'] = [0.05, 0.1]
    # Full grid for larger datasets

    if chars.cv > 1.0:
        xgb_hp['learning_rate'] = [0.01, 0.05]

    filters['XGBoost'] = xgb_hp

    # StatsForecast hyperparameters
    statsforecast_hp = {}

    # Set season length based on frequency
    if chars.frequency == 'daily':
        statsforecast_hp['season_length'] = 7
    elif chars.frequency == 'weekly':
        statsforecast_hp['season_length'] = 52
    elif chars.frequency == 'monthly':
        statsforecast_hp['season_length'] = 12
    elif chars.frequency == 'quarterly':
        statsforecast_hp['season_length'] = 4
    else:
        statsforecast_hp['season_length'] = 12  # Default

    # Model type selection based on data characteristics
    if chars.n_observations < 24:
        # Limited data - use simpler auto selection
        statsforecast_hp['model_type'] = ['auto']
    elif chars.n_observations < 52:
        # Moderate data - focus on faster models
        statsforecast_hp['model_type'] = ['auto', 'autoarima']
    else:
        # Sufficient data - allow all model types
        statsforecast_hp['model_type'] = ['auto', 'autoarima', 'autoets', 'autotheta']

    filters['StatsForecast'] = statsforecast_hp

    # Chronos hyperparameters
    chronos_hp = {}

    # Model size selection based on data and compute needs
    if chars.n_observations < 100:
        # Smaller model for faster inference on smaller datasets
        chronos_hp['model_size'] = ['tiny', 'small']
    elif chars.n_observations < 500:
        # Medium-sized model for balanced speed/accuracy
        chronos_hp['model_size'] = ['small', 'base']
    else:
        # Larger model for complex patterns
        chronos_hp['model_size'] = ['base']

    # Chronos doesn't need much tuning - it's zero-shot
    # But we can suggest number of samples for prediction intervals
    if chars.has_outliers or chars.cv > 0.5:
        chronos_hp['num_samples'] = 200  # More samples for uncertain data
    else:
        chronos_hp['num_samples'] = 100  # Standard samples

    filters['Chronos'] = chronos_hp

    return filters


@log_io
def _generate_overall_recommendation(
    chars: DataCharacteristics,
    recommendations: List[ModelRecommendation]
) -> str:
    """Generate an overall recommendation summary."""
    recommended = [r for r in recommendations if r.recommended]

    if chars.quality == DataQuality.INSUFFICIENT:
        return (
            "⚠️ Data quality is insufficient for reliable forecasting. "
            f"Only {chars.n_observations} observations available. "
            "Consider collecting more historical data."
        )

    if chars.quality == DataQuality.POOR:
        return (
            f"⚠️ Data quality is poor (score: {chars.quality_score:.0f}/100). "
            "Forecast accuracy may be limited. "
            f"Recommended models: {', '.join(r.model_name for r in recommended[:2])}."
        )

    best_models = sorted(recommended, key=lambda x: x.confidence, reverse=True)[:2]

    pattern_desc = []
    if chars.trend_type != TrendType.FLAT:
        pattern_desc.append(f"{chars.trend_type.value} trend")
    if chars.seasonality_type != SeasonalityType.NONE:
        pattern_desc.append(f"{chars.seasonality_type.value} seasonality")

    pattern_str = " with " + " and ".join(pattern_desc) if pattern_desc else ""

    return (
        f"✅ Data quality: {chars.quality.value} ({chars.quality_score:.0f}/100). "
        f"{chars.n_observations} observations spanning {chars.n_years:.1f} years{pattern_str}. "
        f"Top models: {', '.join(m.model_name for m in best_models)}."
    )


@log_io
def _log_analysis_summary(result: AnalysisResult) -> None:
    """Log a summary of the analysis."""
    chars = result.characteristics

    logger.info("=" * 60)
    logger.info("📊 DATA ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"   Observations: {chars.n_observations}")
    logger.info(f"   Time span: {chars.n_years:.1f} years ({chars.start_date} to {chars.end_date})")
    logger.info(f"   Frequency: {chars.frequency}")
    logger.info(f"   Quality: {chars.quality.value} ({chars.quality_score:.0f}/100)")
    logger.info(f"   Trend: {chars.trend_type.value} (strength: {chars.trend_strength:.2f})")
    logger.info(f"   Seasonality: {chars.seasonality_type.value} (strength: {chars.seasonality_strength:.2f})")

    logger.info("   Model Recommendations:")
    for rec in result.model_recommendations:
        status = "✅" if rec.recommended else "❌"
        logger.info(f"     {status} {rec.model_name}: {rec.confidence:.2f} - {rec.reason[:60]}...")

    if chars.warnings:
        logger.info("   Warnings:")
        for warning in chars.warnings:
            logger.info(f"     ⚠️ {warning}")

    logger.info("=" * 60)


@log_io
def get_analysis_summary_for_ui(result: AnalysisResult) -> Dict[str, Any]:
    """
    Convert analysis result to a format suitable for the UI.

    Returns a dictionary that can be JSON serialized and displayed in the frontend.
    Note: We explicitly convert numpy types to native Python types for JSON serialization.
    """
    chars = result.characteristics

    return {
        "dataQuality": {
            "level": chars.quality.value,
            "score": float(chars.quality_score),
            "description": _get_quality_description(chars.quality)
        },
        "dataStats": {
            "observations": int(chars.n_observations),
            "yearsOfData": float(round(chars.n_years, 1)),
            "dateRange": f"{chars.start_date} to {chars.end_date}",
            "frequency": str(chars.frequency),
            "hasGaps": bool(chars.has_gaps),
            "gapCount": int(chars.gap_count)
        },
        "patterns": {
            "trend": {
                "type": chars.trend_type.value,
                "strength": float(round(chars.trend_strength, 2))
            },
            "seasonality": {
                "type": chars.seasonality_type.value,
                "strength": float(round(chars.seasonality_strength, 2))
            },
            "hasOutliers": bool(chars.has_outliers),
            "outlierPercentage": float(round(chars.outlier_percentage, 1))
        },
        "modelRecommendations": [
            {
                "model": str(rec.model_name),
                "recommended": bool(rec.recommended),
                "confidence": float(round(rec.confidence, 2)),
                "reason": str(rec.reason)
            }
            for rec in result.model_recommendations
        ],
        "recommendedModels": [str(m) for m in result.recommended_models],
        "excludedModels": [str(m) for m in result.excluded_models],
        "warnings": [str(w) for w in chars.warnings],
        "notes": [str(n) for n in chars.notes],
        "overallRecommendation": str(result.overall_recommendation),
        "hyperparameterFilters": result.hyperparameter_filters
    }


@log_io
def _get_quality_description(quality: DataQuality) -> str:
    """Get a user-friendly description for data quality levels."""
    descriptions = {
        DataQuality.EXCELLENT: "Excellent data quality. All models should perform well.",
        DataQuality.GOOD: "Good data quality. Most models should perform reliably.",
        DataQuality.FAIR: "Fair data quality. Some limitations may affect accuracy.",
        DataQuality.POOR: "Poor data quality. Consider using simpler models.",
        DataQuality.INSUFFICIENT: "Insufficient data for reliable forecasting."
    }
    return descriptions.get(quality, "Unknown quality level")
