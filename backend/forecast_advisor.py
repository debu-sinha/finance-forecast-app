"""
Smart Forecast Advisor using research-backed time series features.

Analyzes all slices using forecastability metrics (spectral entropy, STL-based
trend/seasonal strength), recommends aggregations, auto-selects models and
training windows.

Research basis:
- tsfeatures (Hyndman et al. 2015) canonical feature set
- FFORMA (Montero-Manso et al. 2020) feature-based model selection
- Spectral entropy (Goerg 2013, ForeCA) as primary forecastability measure
- STL decomposition (Cleveland et al. 1990) for trend/seasonal strength
- Armstrong (1992) rule-based pre-filtering for hard constraints
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import fft as scipy_fft
from scipy.stats import entropy as scipy_entropy
from statsmodels.tsa.seasonal import STL


logger = logging.getLogger(__name__)


def _clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    Clean a column that may contain string-formatted numbers.

    Handles CSV exports from Excel with comma separators, currency symbols,
    whitespace, etc.  Returns a numeric Series with non-parseable values as NaN.
    """
    if series.dtype == "object" or series.dtype.name == "string":
        cleaned = series.astype(str).str.replace(",", "", regex=False)
        cleaned = cleaned.str.replace("$", "", regex=False)
        cleaned = cleaned.str.replace(" ", "", regex=False)
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


# Seasonal period by frequency
SEASONAL_PERIODS = {
    "daily": 7,
    "weekly": 52,
    "monthly": 12,
}

# Signal weights for composite forecastability score.
# These are heuristic weights chosen to prioritize the most informative signals:
#   - spectral_entropy (0.30): Primary forecastability indicator per Goerg 2013 ForeCA.
#   - trend/seasonal strength (0.20 each): Core STL-derived signals from tsfeatures (Hyndman 2015).
#   - linearity, spikiness, acf1 (0.10 each): Secondary discriminators for edge cases.
# Note: FFORMA (Montero-Manso 2020) uses tsfeatures as *inputs* to a gradient boosting
# classifier, not a fixed weighted combination. These weights approximate the relative
# importance found in FFORMA's feature importance analysis but are not trained on our data.
# Future calibration: regress actual MAPE on signals using benchmark results.
SIGNAL_WEIGHTS = {
    "spectral_entropy": 0.30,
    "trend_strength": 0.20,
    "seasonal_strength": 0.20,
    "linearity": 0.10,
    "spikiness": 0.10,
    "acf1_remainder": 0.10,
}


class ForecastabilityGrade(str, Enum):
    EXCELLENT = "excellent"        # 80-100
    GOOD = "good"                  # 60-79
    FAIR = "fair"                  # 40-59
    POOR = "poor"                  # 20-39
    UNFORECASTABLE = "unforecastable"  # 0-19


@dataclass
class TrainingConfig:
    """Auto-configured training settings based on data analysis.

    Generic and dataset-agnostic — works with any time series data.
    All recommendations are derived from statistical properties of the series,
    not hardcoded to any specific dataset or domain.
    """
    # Training window
    from_date: Optional[str]               # ISO date to start training from (None = use all)
    training_window_weeks: Optional[int]    # Weeks of data to use (None = all)
    training_window_reason: str

    # Model selection
    recommended_models: List[str]
    excluded_models: List[str]
    model_exclusion_reasons: Dict[str, str]

    # Transform settings
    log_transform: str                     # "auto", "always", "never"
    log_transform_reason: str

    # Horizon guidance
    recommended_horizon: Optional[int]     # Suggested horizon (periods)
    max_reliable_horizon: int              # Max horizon before accuracy degrades
    horizon_reason: str

    # Forecastability assessment
    forecastability_score: float            # 0-100
    grade: str                             # excellent/good/fair/poor/unforecastable
    expected_mape_range: Tuple[float, float]

    # Data characteristics detected
    growth_pct: float
    trend_strength: float
    seasonal_strength: float
    spectral_entropy: float
    n_observations: int
    data_warnings: List[str]

    # Human-readable summary
    summary: str


@dataclass
class DataQualityCheck:
    """Data quality prerequisites, separated from forecastability."""
    n_observations: int
    has_sufficient_history: bool
    missing_pct: float
    gap_count: int
    anomalous_week_count: int
    anomalous_weeks: List[str]
    volume_level: str
    weekly_mean: float
    warnings: List[str]


@dataclass
class SliceAnalysis:
    """Complete analysis for a single data slice."""
    slice_name: str
    filters: Dict[str, str]

    # Forecastability (research-backed signals)
    forecastability_score: float
    grade: ForecastabilityGrade
    spectral_entropy: float
    trend_strength: float
    seasonal_strength: float
    linearity: float
    spikiness: float
    acf1_remainder: float

    # Growth metrics
    total_growth_pct: float
    recent_growth_pct: float

    # Data quality (separated from forecastability)
    data_quality: DataQualityCheck

    # Recommendations
    recommended_models: List[str]
    excluded_models: List[str]
    model_exclusion_reasons: Dict[str, str]
    recommended_training_window: Optional[int]
    training_window_reason: str
    expected_mape_range: Tuple[float, float]


@dataclass
class AggregationRecommendation:
    """Recommendation to merge POOR/UNFORECASTABLE sibling slices."""
    from_slices: List[str]
    to_slice: str
    reason: str
    combined_forecastability_score: float
    improvement_pct: float


@dataclass
class AdvisorResult:
    """Complete advisor analysis result."""
    slice_analyses: List[SliceAnalysis]
    aggregation_recommendations: List[AggregationRecommendation]
    summary: str
    overall_data_quality: str
    total_slices: int
    forecastable_slices: int
    problematic_slices: int


class ForecastAdvisor:
    """
    Smart Forecast Advisor using research-backed time series features.

    Uses the tsfeatures canonical set (Hyndman et al. 2015, FFORMA 2020)
    with spectral entropy (Goerg 2013) as the primary forecastability measure.
    All signals computed from STL decomposition for independence.
    """

    # Models proven reliable in 12-slice benchmark testing
    RECOMMENDED_MODELS = ["prophet", "exponential_smoothing", "statsforecast", "chronos"]

    # Models disabled by default based on empirical evidence
    DISABLED_MODELS = {
        "arima": (
            "StatsForecast AutoARIMA provides equivalent functionality with better "
            "reliability. ARIMA frequently fails or produces degenerate (0,1,0) flat forecasts."
        ),
        "sarimax": (
            "Numerically unstable — observed forecast explosions exceeding +/-100 billion "
            "in benchmark testing. Use Prophet with covariates instead."
        ),
        "xgboost": (
            "Cannot extrapolate beyond training range, causing systematic under-prediction "
            "on trending data (-48% error observed). Use StatsForecast or Prophet for trends."
        ),
    }

    def analyze_all_slices(
        self,
        df: pd.DataFrame,
        time_col: str,
        target_col: str,
        dimension_cols: List[str],
        frequency: str = "weekly",
        horizon: int = 12,
    ) -> AdvisorResult:
        """
        Main entry point. Groups df by dimension_cols, analyzes each slice.

        Args:
            df: Full dataset with all dimensions
            time_col: Date column name
            target_col: Target value column name
            dimension_cols: Columns to slice by (e.g., ["IS_CGNA", "ORDER_PROTOCOL", "BIZ_SIZE"])
            frequency: Data frequency ('daily', 'weekly', 'monthly')
            horizon: Forecast horizon in periods

        Returns:
            AdvisorResult with per-slice analyses, aggregation recs, and summary
        """
        logger.info(
            f"Forecast Advisor: analyzing {len(df)} rows, target={target_col}, "
            f"dimensions={dimension_cols}, frequency={frequency}"
        )

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        # Clean target column — handle string-formatted numbers ("29,031" etc.)
        df[target_col] = _clean_numeric_column(df[target_col])
        df = df.sort_values(time_col).reset_index(drop=True)

        period = SEASONAL_PERIODS.get(frequency, 52)

        # Group by dimensions
        if dimension_cols:
            groups = df.groupby(dimension_cols, dropna=False)
        else:
            groups = [("All", df)]

        # Collect peer means for relative volume classification
        peer_means: List[float] = []
        slice_data: List[Tuple[str, Dict[str, str], pd.DataFrame]] = []

        for group_key, group_df in groups:
            if isinstance(group_key, tuple):
                filters = {col: str(val) for col, val in zip(dimension_cols, group_key)}
                slice_name = "/".join(f"{col}={val}" for col, val in zip(dimension_cols, group_key))
            else:
                filters = {dimension_cols[0]: str(group_key)} if dimension_cols else {}
                slice_name = f"{dimension_cols[0]}={group_key}" if dimension_cols else "All"

            values = group_df[target_col].dropna().values.astype(float)
            if len(values) > 0:
                peer_means.append(float(np.mean(values)))
                slice_data.append((slice_name, filters, group_df))

        # Analyze each slice
        analyses: List[SliceAnalysis] = []
        for slice_name, filters, group_df in slice_data:
            analysis = self._analyze_single_slice(
                group_df, time_col, target_col, slice_name, filters,
                frequency, period, horizon, peer_means
            )
            analyses.append(analysis)

        # Sort by forecastability score (best first)
        analyses.sort(key=lambda a: a.forecastability_score, reverse=True)

        # Cross-slice aggregation recommendations
        agg_recs = self._recommend_aggregations(analyses, dimension_cols, df, time_col, target_col, period)

        # Build summary
        forecastable = sum(1 for a in analyses if a.forecastability_score >= 40)
        problematic = sum(1 for a in analyses if a.forecastability_score < 40)

        # Overall data quality
        if all(a.data_quality.has_sufficient_history for a in analyses) and forecastable >= len(analyses) * 0.7:
            overall_quality = "EXCELLENT" if problematic == 0 else "GOOD"
        elif forecastable >= len(analyses) * 0.5:
            overall_quality = "FAIR"
        else:
            overall_quality = "POOR"

        summary = self._build_summary(analyses, agg_recs, forecastable, problematic)

        return AdvisorResult(
            slice_analyses=analyses,
            aggregation_recommendations=agg_recs,
            summary=summary,
            overall_data_quality=overall_quality,
            total_slices=len(analyses),
            forecastable_slices=forecastable,
            problematic_slices=problematic,
        )

    def _analyze_single_slice(
        self,
        df: pd.DataFrame,
        time_col: str,
        target_col: str,
        slice_name: str,
        filters: Dict[str, str],
        frequency: str,
        period: int,
        horizon: int,
        peer_means: List[float],
    ) -> SliceAnalysis:
        """Analyze a single data slice: forecastability + data quality + recommendations."""
        values = df[target_col].dropna().values.astype(float)
        dates = df[time_col]
        n = len(values)

        logger.info(f"Analyzing slice '{slice_name}': {n} observations")

        # Data quality check (separated from forecastability)
        dq = self._data_quality_check(values, dates, peer_means, frequency, period)

        # Compute growth metrics
        total_growth = self._compute_growth(values, n)
        recent_growth = self._compute_recent_growth(values, period)

        # Compute forecastability score
        if n >= period * 2:
            score, signals = self._compute_forecastability_score(values, period)
        else:
            # Insufficient data for STL — assign low score based on available info
            score = max(10, min(30, n / (period * 2) * 40))
            signals = {
                "spectral_entropy": 0.8,
                "trend_strength": 0.0,
                "seasonal_strength": 0.0,
                "linearity": 0.0,
                "spikiness": 1.0,
                "acf1_remainder": 0.0,
            }
            dq.warnings.append(
                f"Insufficient data for full forecastability analysis "
                f"(need {period * 2}, have {n})"
            )

        grade = self._score_to_grade(score)

        # Model recommendations
        rec_models, exc_models, exc_reasons = self._recommend_models(
            score, total_growth, signals["spectral_entropy"],
            signals["trend_strength"], n
        )

        # Training window recommendation
        training_window, window_reason = self._recommend_training_window(total_growth, n)

        # Expected MAPE range
        mape_range = self._estimate_expected_mape(score, dq.volume_level)

        return SliceAnalysis(
            slice_name=slice_name,
            filters=filters,
            forecastability_score=round(score, 1),
            grade=grade,
            spectral_entropy=round(signals["spectral_entropy"], 4),
            trend_strength=round(signals["trend_strength"], 4),
            seasonal_strength=round(signals["seasonal_strength"], 4),
            linearity=round(signals["linearity"], 4),
            spikiness=round(signals["spikiness"], 4),
            acf1_remainder=round(signals["acf1_remainder"], 4),
            total_growth_pct=round(total_growth, 1),
            recent_growth_pct=round(recent_growth, 1),
            data_quality=dq,
            recommended_models=rec_models,
            excluded_models=exc_models,
            model_exclusion_reasons=exc_reasons,
            recommended_training_window=training_window,
            training_window_reason=window_reason,
            expected_mape_range=mape_range,
        )

    def _compute_forecastability_score(
        self, values: np.ndarray, period: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute 0-100 forecastability score from 6 STL/spectral signals.

        Uses statsmodels STL for decomposition and scipy FFT for spectral entropy.
        Each signal is mapped to a 0-100 subscale, then weighted.
        """
        # STL decomposition
        trend, seasonal, remainder = self._stl_decompose(values, period)

        # 1. Spectral entropy on raw series (0-1, lower = more forecastable)
        # Per Goerg 2013 ForeCA: measures how concentrated the power spectrum is.
        # Raw series is correct — trend/seasonal contribute to concentrated spectrum,
        # and their contribution is already separated via trend_strength/seasonal_strength signals.
        se = self._compute_spectral_entropy(values)

        # 2. Trend strength from STL: 1 - Var(remainder) / Var(trend + remainder)
        var_tr = np.var(trend + remainder)
        ts = max(0.0, 1.0 - np.var(remainder) / var_tr) if var_tr > 0 else 0.0

        # 3. Seasonal strength from STL: 1 - Var(remainder) / Var(seasonal + remainder)
        var_sr = np.var(seasonal + remainder)
        ss = max(0.0, 1.0 - np.var(remainder) / var_sr) if var_sr > 0 else 0.0

        # 4. Linearity: R² of linear fit on STL trend component
        lin = self._compute_linearity(trend)

        # 5. Spikiness: variance of leave-one-out variances of remainder
        spk = self._compute_spikiness(remainder)

        # 6. ACF1 of remainder
        acf1 = self._compute_acf1(remainder)

        # Map each signal to 0-100 subscale
        se_score = (1.0 - se) * 100            # Lower entropy = higher score
        ts_score = ts * 100                     # Higher trend strength = higher score
        ss_score = ss * 100                     # Higher seasonal strength = higher score
        lin_score = min(100, max(0, lin * 100)) # Higher linearity = higher score
        spk_score = 100.0 / (1.0 + spk * 50)    # Sigmoid mapping: lower spikiness = higher score
        acf1_score = max(0, (1.0 - abs(acf1)) * 100)  # Low |acf1| = white noise remainder = good

        # Weighted combination
        composite = (
            SIGNAL_WEIGHTS["spectral_entropy"] * se_score
            + SIGNAL_WEIGHTS["trend_strength"] * ts_score
            + SIGNAL_WEIGHTS["seasonal_strength"] * ss_score
            + SIGNAL_WEIGHTS["linearity"] * lin_score
            + SIGNAL_WEIGHTS["spikiness"] * spk_score
            + SIGNAL_WEIGHTS["acf1_remainder"] * acf1_score
        )

        composite = max(0.0, min(100.0, composite))

        signals = {
            "spectral_entropy": se,
            "trend_strength": ts,
            "seasonal_strength": ss,
            "linearity": lin,
            "spikiness": spk,
            "acf1_remainder": acf1,
        }

        logger.info(
            f"Forecastability signals: SE={se:.3f}, TS={ts:.3f}, SS={ss:.3f}, "
            f"Lin={lin:.3f}, Spk={spk:.4f}, ACF1r={acf1:.3f} → score={composite:.1f}"
        )

        return composite, signals

    def _stl_decompose(
        self, values: np.ndarray, period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run STL decomposition with robust estimation."""
        try:
            stl = STL(values, period=period, robust=True)
            result = stl.fit()
            return result.trend, result.seasonal, result.resid
        except Exception as e:
            logger.warning(f"STL decomposition failed: {e}. Using simple decomposition fallback.")
            # Simple fallback: rolling mean for trend, zero seasonal
            window = min(period, len(values) // 3)
            if window < 3:
                window = 3
            trend = pd.Series(values).rolling(window, center=True, min_periods=1).mean().values
            seasonal = np.zeros_like(values)
            remainder = values - trend
            return trend, seasonal, remainder

    def _compute_spectral_entropy(self, values: np.ndarray) -> float:
        """
        Compute spectral entropy of the series.

        Lower entropy = more concentrated power spectrum = more forecastable.
        Returns 0-1 (0 = perfectly predictable, 1 = white noise).

        Based on Goerg (2013) ForeCA framework.
        """
        n = len(values)
        if n < 4:
            return 1.0  # Assume white noise with insufficient data

        # Detrend (remove mean)
        centered = values - np.mean(values)

        # FFT -> power spectral density
        fft_vals = scipy_fft.rfft(centered)
        psd = np.abs(fft_vals) ** 2

        # Normalize to probability distribution
        total_power = np.sum(psd)
        if total_power == 0:
            return 1.0
        psd_normalized = psd / total_power

        # Remove zeros for entropy calculation
        psd_normalized = psd_normalized[psd_normalized > 0]

        # Shannon entropy, normalized by log(N) to get 0-1 range
        h = scipy_entropy(psd_normalized, base=2)
        max_entropy = np.log2(len(psd_normalized))

        if max_entropy == 0:
            return 1.0

        return float(min(1.0, h / max_entropy))

    def _compute_linearity(self, trend: np.ndarray) -> float:
        """
        Compute linearity of the STL trend component.

        Uses R² of linear regression on the trend. Higher R² means the trend
        is well-explained by a linear model (more predictable).
        """
        n = len(trend)
        if n < 3:
            return 0.0

        x = np.arange(n)
        try:
            coeffs = np.polyfit(x, trend, 1)
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((trend - y_pred) ** 2)
            ss_tot = np.sum((trend - np.mean(trend)) ** 2)
            if ss_tot == 0:
                return 1.0  # Perfectly flat trend is perfectly linear
            r2 = 1.0 - ss_res / ss_tot
            return max(0.0, float(r2))
        except Exception:
            return 0.0

    def _compute_spikiness(self, remainder: np.ndarray) -> float:
        """
        Compute spikiness: variance of leave-one-out variances of STL remainder.

        Lower spikiness = more uniform noise = more forecastable.
        Based on tsfeatures (Hyndman et al. 2015).
        """
        n = len(remainder)
        if n < 4:
            return 1.0

        total_var = np.var(remainder)
        if total_var == 0:
            return 0.0

        # Leave-one-out variances
        loo_vars = np.zeros(n)
        sum_r = np.sum(remainder)
        sum_r2 = np.sum(remainder ** 2)

        for i in range(n):
            loo_mean = (sum_r - remainder[i]) / (n - 1)
            loo_var = (sum_r2 - remainder[i] ** 2) / (n - 1) - loo_mean ** 2
            loo_vars[i] = max(0.0, loo_var)

        spikiness = float(np.var(loo_vars))

        # Normalize by total variance squared to make it scale-invariant
        if total_var > 0:
            spikiness = spikiness / (total_var ** 2)

        return spikiness

    def _compute_acf1(self, remainder: np.ndarray) -> float:
        """Compute first autocorrelation of STL remainder."""
        n = len(remainder)
        if n < 3:
            return 0.0

        mean_r = np.mean(remainder)
        var_r = np.var(remainder)
        if var_r == 0:
            return 0.0

        acf1 = np.sum((remainder[:-1] - mean_r) * (remainder[1:] - mean_r)) / ((n - 1) * var_r)
        return float(np.clip(acf1, -1.0, 1.0))

    def _data_quality_check(
        self,
        values: np.ndarray,
        dates: pd.Series,
        peer_means: List[float],
        frequency: str,
        period: int,
    ) -> DataQualityCheck:
        """
        Separate data quality assessment (prerequisites, not forecastability).
        Volume classification is relative to peer slices (percentile-based).
        """
        n = len(values)
        weekly_mean = float(np.mean(values)) if n > 0 else 0.0
        sufficient = n >= period * 2  # 104 weeks for weekly

        # Missing data
        total_possible = n  # Approximate; could check for gaps
        null_count = int(np.sum(np.isnan(values))) if n > 0 else 0
        missing_pct = (null_count / total_possible * 100) if total_possible > 0 else 0.0

        # Gap detection
        gap_count = 0
        if n >= 2:
            expected_days = {"daily": 1, "weekly": 7, "monthly": 30}.get(frequency, 7)
            diffs = dates.diff().dropna().dt.days
            tolerance = expected_days * 0.5
            gap_count = int((diffs > expected_days + tolerance).sum())

        # Anomalous weeks (IQR-based)
        anomalous_weeks: List[str] = []
        if n >= 10:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower = q1 - 2.0 * iqr
            upper = q3 + 2.0 * iqr
            for i in range(n):
                if values[i] < lower or values[i] > upper:
                    date_str = dates.iloc[i].strftime("%Y-%m-%d")
                    anomalous_weeks.append(date_str)

        # Volume level relative to peers
        volume_level = self._classify_volume(weekly_mean, peer_means)

        # Warnings
        warnings: List[str] = []
        if not sufficient:
            warnings.append(
                f"Insufficient history: {n} observations, recommend {period * 2}+ "
                f"({frequency})"
            )
        if gap_count > 0:
            warnings.append(f"{gap_count} gaps detected in time series")
        if missing_pct > 5:
            warnings.append(f"High missing data rate: {missing_pct:.1f}%")
        if len(anomalous_weeks) > n * 0.05:
            warnings.append(
                f"{len(anomalous_weeks)} anomalous weeks detected ({len(anomalous_weeks)/n*100:.1f}%)"
            )

        return DataQualityCheck(
            n_observations=n,
            has_sufficient_history=sufficient,
            missing_pct=round(missing_pct, 1),
            gap_count=gap_count,
            anomalous_week_count=len(anomalous_weeks),
            anomalous_weeks=anomalous_weeks[:10],  # Limit to first 10
            volume_level=volume_level,
            weekly_mean=round(weekly_mean, 2),
            warnings=warnings,
        )

    def _classify_volume(self, mean_val: float, peer_means: List[float]) -> str:
        """Classify volume level relative to peer slices."""
        if not peer_means or len(peer_means) < 2:
            return "medium"

        sorted_peers = sorted(peer_means)
        n = len(sorted_peers)

        p20 = sorted_peers[int(n * 0.2)]
        p40 = sorted_peers[int(n * 0.4)]
        p60 = sorted_peers[min(int(n * 0.6), n - 1)]
        p80 = sorted_peers[min(int(n * 0.8), n - 1)]

        if mean_val <= p20:
            return "very_low"
        elif mean_val <= p40:
            return "low"
        elif mean_val <= p60:
            return "medium"
        elif mean_val <= p80:
            return "high"
        else:
            return "very_high"

    def _compute_growth(self, values: np.ndarray, n: int) -> float:
        """Compute total growth as percentage change from first to last quarter.

        Uses median instead of mean for robustness to outliers and partial data weeks.
        """
        if n < 8:
            return 0.0
        quarter = max(1, n // 4)
        first_q = float(np.median(values[:quarter]))
        last_q = float(np.median(values[-quarter:]))
        if first_q == 0:
            return 0.0
        return ((last_q - first_q) / abs(first_q)) * 100

    def _compute_recent_growth(self, values: np.ndarray, period: int) -> float:
        """Compute recent growth: last half-period vs prior half-period."""
        half = period // 2
        n = len(values)
        if n < period:
            return 0.0
        recent = float(np.mean(values[-half:]))
        prior = float(np.mean(values[-period:-half]))
        if prior == 0:
            return 0.0
        return ((recent - prior) / abs(prior)) * 100

    def _score_to_grade(self, score: float) -> ForecastabilityGrade:
        """Convert numeric score to grade."""
        if score >= 80:
            return ForecastabilityGrade.EXCELLENT
        elif score >= 60:
            return ForecastabilityGrade.GOOD
        elif score >= 40:
            return ForecastabilityGrade.FAIR
        elif score >= 20:
            return ForecastabilityGrade.POOR
        else:
            return ForecastabilityGrade.UNFORECASTABLE

    def _recommend_models(
        self,
        score: float,
        growth_pct: float,
        spectral_entropy: float,
        trend_strength: float,
        n_observations: int,
    ) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Select models based on forecastability score and data characteristics.

        Hard constraints (always applied):
        - ARIMA, SARIMAX, XGBoost always excluded (DISABLED_MODELS)
        - High growth (>200%) drops Chronos (tends to flat-line)
        - Short history (<104 obs) drops Chronos (needs 2+ years for context)
        - UNFORECASTABLE slices use minimal model set
        """
        excluded = dict(self.DISABLED_MODELS)  # Always exclude these
        recommended = list(self.RECOMMENDED_MODELS)

        # Chronos constraints
        if growth_pct > 200:
            excluded["chronos"] = (
                f"Series has {growth_pct:.0f}% growth. Chronos tends to produce "
                f"flat forecasts on high-growth data."
            )
            if "chronos" in recommended:
                recommended.remove("chronos")

        if n_observations < 104:
            excluded["chronos"] = (
                f"Only {n_observations} observations. Chronos needs 2+ years of "
                f"weekly data for reliable context."
            )
            if "chronos" in recommended:
                recommended.remove("chronos")

        # Unforecastable: minimal set only
        if score < 20:
            recommended = ["prophet", "exponential_smoothing"]
            if "statsforecast" not in excluded:
                excluded["statsforecast"] = (
                    "Series is unforecastable (score < 20). Using minimal model set."
                )
            if "chronos" not in excluded:
                excluded["chronos"] = (
                    "Series is unforecastable (score < 20). Using minimal model set."
                )

        return recommended, list(excluded.keys()), excluded

    def _recommend_training_window(
        self, total_growth_pct: float, n_observations: int
    ) -> Tuple[Optional[int], str]:
        """
        Recommend shorter training window for high-growth series.

        Based on empirical finding: high-growth slices perform better with recent
        data only (older data is misleading due to regime change).
        """
        if total_growth_pct > 500 and n_observations > 78:
            return 52, (
                f"Very high growth ({total_growth_pct:.0f}%). Using last 52 weeks only — "
                f"older data reflects a different regime."
            )
        elif total_growth_pct > 200 and n_observations > 104:
            return 78, (
                f"High growth ({total_growth_pct:.0f}%). Using last 78 weeks — "
                f"balances recency with seasonal coverage."
            )
        elif total_growth_pct > 100 and n_observations > 130:
            return 104, (
                f"Moderate growth ({total_growth_pct:.0f}%). Using last 104 weeks (2 years) — "
                f"captures recent trend while preserving two seasonal cycles."
            )
        else:
            return None, "Growth is within normal range. Use all available data for training."

    def _estimate_expected_mape(
        self, score: float, volume_level: str
    ) -> Tuple[float, float]:
        """
        Estimate expected MAPE range based on forecastability score and volume.

        Calibrated from our 12-slice benchmark (TOT_VOL + TOT_SUB, Oct 2025 cutoff).
        """
        # Base MAPE range from score (calibrated from 12-slice benchmark results)
        if score >= 80:
            low, high = 3.0, 8.0
        elif score >= 60:
            low, high = 8.0, 15.0
        elif score >= 40:
            low, high = 15.0, 25.0
        elif score >= 20:
            low, high = 25.0, 40.0
        else:
            low, high = 35.0, 60.0

        # Volume adjustment: low volume series have higher error
        if volume_level in ("very_low", "low"):
            low *= 1.5
            high *= 1.5

        return (round(low, 1), round(high, 1))

    def _recommend_aggregations(
        self,
        analyses: List[SliceAnalysis],
        dimension_cols: List[str],
        df: pd.DataFrame,
        time_col: str,
        target_col: str,
        period: int,
    ) -> List[AggregationRecommendation]:
        """
        Find POOR/UNFORECASTABLE slices sharing a parent dimension and recommend merging.

        Algorithm:
        1. Find all slices with score < 40
        2. Group by shared dimensions (e.g., both are CGNA=1/Pickup/*)
        3. Simulate combined series (sum the targets)
        4. Compute combined forecastability score
        5. If combined score improves by >= 15 points -> recommend merging
        """
        if len(dimension_cols) < 2:
            return []

        poor_slices = [a for a in analyses if a.forecastability_score < 40]
        if len(poor_slices) < 2:
            return []

        recommendations: List[AggregationRecommendation] = []

        # Try merging slices that share all but one dimension
        for dim_idx in range(len(dimension_cols)):
            # Group poor slices by all dimensions except dim_idx
            parent_groups: Dict[str, List[SliceAnalysis]] = {}
            for analysis in poor_slices:
                parent_key_parts = []
                for j, col in enumerate(dimension_cols):
                    if j != dim_idx and col in analysis.filters:
                        parent_key_parts.append(f"{col}={analysis.filters[col]}")
                parent_key = "/".join(parent_key_parts)
                if parent_key not in parent_groups:
                    parent_groups[parent_key] = []
                parent_groups[parent_key].append(analysis)

            # For groups with 2+ poor slices, simulate merged series
            for parent_key, group_analyses in parent_groups.items():
                if len(group_analyses) < 2:
                    continue

                # Build filter for combined data
                from_slices = [a.slice_name for a in group_analyses]
                varied_dim = dimension_cols[dim_idx]
                to_slice = f"{parent_key}/All_{varied_dim}"

                # Combine the data
                combined_values = self._simulate_combined_series(
                    df, time_col, target_col, group_analyses, dimension_cols
                )

                if combined_values is None or len(combined_values) < period * 2:
                    continue

                # Compute combined forecastability
                combined_score, _ = self._compute_forecastability_score(combined_values, period)

                # Average score of individual slices
                avg_individual = np.mean([a.forecastability_score for a in group_analyses])

                improvement = combined_score - avg_individual

                if improvement >= 15:
                    recommendations.append(AggregationRecommendation(
                        from_slices=from_slices,
                        to_slice=to_slice,
                        reason=(
                            f"Merging {len(from_slices)} low-forecastability slices "
                            f"(varying {varied_dim}) improves score by {improvement:.0f} points. "
                            f"Individual avg: {avg_individual:.0f}, combined: {combined_score:.0f}."
                        ),
                        combined_forecastability_score=round(combined_score, 1),
                        improvement_pct=round(improvement, 1),
                    ))

        # Sort by improvement
        recommendations.sort(key=lambda r: r.improvement_pct, reverse=True)
        return recommendations

    def _simulate_combined_series(
        self,
        df: pd.DataFrame,
        time_col: str,
        target_col: str,
        group_analyses: List[SliceAnalysis],
        dimension_cols: List[str],
    ) -> Optional[np.ndarray]:
        """Combine multiple slices into one by summing the target values per time period."""
        combined_data: Optional[pd.DataFrame] = None

        for analysis in group_analyses:
            mask = pd.Series([True] * len(df), index=df.index)
            for col, val in analysis.filters.items():
                if col in df.columns:
                    mask = mask & (df[col].astype(str) == val)

            slice_df = df.loc[mask, [time_col, target_col]].copy()
            if len(slice_df) == 0:
                continue

            slice_agg = slice_df.groupby(time_col)[target_col].sum().reset_index()

            if combined_data is None:
                combined_data = slice_agg.rename(columns={target_col: "combined"})
            else:
                merged = combined_data.merge(slice_agg, on=time_col, how="outer")
                merged["combined"] = merged["combined"].fillna(0) + merged[target_col].fillna(0)
                combined_data = merged[[time_col, "combined"]]

        if combined_data is None or len(combined_data) == 0:
            return None

        combined_data = combined_data.sort_values(time_col)
        return combined_data["combined"].dropna().values.astype(float)

    def _build_summary(
        self,
        analyses: List[SliceAnalysis],
        agg_recs: List[AggregationRecommendation],
        forecastable: int,
        problematic: int,
    ) -> str:
        """Build a human-readable summary of the advisor analysis."""
        total = len(analyses)
        parts = []

        parts.append(f"Analyzed {total} data slices.")

        if forecastable == total:
            parts.append("All slices are forecastable (score >= 40).")
        elif problematic > 0:
            parts.append(
                f"{forecastable} slices are forecastable, "
                f"{problematic} have low forecastability scores."
            )

        # Best and worst slices
        if analyses:
            best = analyses[0]
            worst = analyses[-1]
            parts.append(
                f"Best: {best.slice_name} (score {best.forecastability_score:.0f}, "
                f"grade {best.grade.value})."
            )
            if worst.forecastability_score < 40:
                parts.append(
                    f"Worst: {worst.slice_name} (score {worst.forecastability_score:.0f}, "
                    f"grade {worst.grade.value})."
                )

        if agg_recs:
            parts.append(
                f"{len(agg_recs)} aggregation recommendation(s) to improve forecastability "
                f"of low-scoring slices."
            )

        return " ".join(parts)

    # =========================================================================
    # AUTO-CONFIGURATION: Generic entry point for any dataset
    # =========================================================================

    def auto_configure_training(
        self,
        values: np.ndarray,
        dates: pd.Series,
        frequency: str = "weekly",
        requested_horizon: Optional[int] = None,
    ) -> TrainingConfig:
        """
        Analyze a single time series and return recommended training settings.

        This is the generic auto-configuration entry point. It works with any
        time series dataset — no domain-specific hardcoding. All recommendations
        are derived from statistical properties of the series.

        Args:
            values: Target column values (numeric array, cleaned of NaN)
            dates: Corresponding datetime Series
            frequency: Data frequency ('daily', 'weekly', 'monthly')
            requested_horizon: User's requested horizon (None = let advisor decide)

        Returns:
            TrainingConfig with all recommended settings
        """
        period = SEASONAL_PERIODS.get(frequency, 52)
        n = len(values)

        # Ensure dates is a pandas Series (not DatetimeIndex)
        if not isinstance(dates, pd.Series):
            dates = pd.Series(dates)
        dates = pd.to_datetime(dates)

        logger.info(
            f"Auto-configure: {n} observations, frequency={frequency}, "
            f"period={period}, requested_horizon={requested_horizon}"
        )

        # Compute growth
        total_growth = self._compute_growth(values, n)
        recent_growth = self._compute_recent_growth(values, period)

        # Data quality (use self as peer for single-series)
        mean_val = float(np.mean(values)) if n > 0 else 0.0
        dq = self._data_quality_check(values, dates, [mean_val], frequency, period)

        # Forecastability score
        if n >= period * 2:
            score, signals = self._compute_forecastability_score(values, period)
        else:
            score = max(10, min(30, n / (period * 2) * 40))
            signals = {
                "spectral_entropy": 0.8,
                "trend_strength": 0.0,
                "seasonal_strength": 0.0,
                "linearity": 0.0,
                "spikiness": 1.0,
                "acf1_remainder": 0.0,
            }

        grade = self._score_to_grade(score)

        # Model selection
        rec_models, exc_models, exc_reasons = self._recommend_models(
            score, total_growth, signals["spectral_entropy"],
            signals["trend_strength"], n
        )

        # Training window
        training_window, window_reason = self._recommend_training_window(total_growth, n)

        # Compute from_date if window is recommended
        from_date = None
        if training_window is not None and n > training_window:
            dates_sorted = dates.sort_values()
            cutoff_idx = max(0, len(dates_sorted) - training_window)
            from_date = dates_sorted.iloc[cutoff_idx].strftime("%Y-%m-%d")

        # Log transform recommendation
        log_transform, log_reason = self._recommend_log_transform(
            values, total_growth, score
        )

        # Horizon recommendation
        rec_horizon, max_horizon, horizon_reason = self._recommend_horizon(
            score, n, period, frequency, requested_horizon
        )

        # Expected MAPE
        mape_range = self._estimate_expected_mape(score, dq.volume_level)

        # Build summary
        summary = self._build_auto_config_summary(
            score, grade, total_growth, training_window,
            rec_models, log_transform, rec_horizon, max_horizon, dq.warnings
        )

        return TrainingConfig(
            from_date=from_date,
            training_window_weeks=training_window,
            training_window_reason=window_reason,
            recommended_models=rec_models,
            excluded_models=exc_models,
            model_exclusion_reasons=exc_reasons,
            log_transform=log_transform,
            log_transform_reason=log_reason,
            recommended_horizon=rec_horizon,
            max_reliable_horizon=max_horizon,
            horizon_reason=horizon_reason,
            forecastability_score=round(score, 1),
            grade=grade.value,
            expected_mape_range=mape_range,
            growth_pct=round(total_growth, 1),
            trend_strength=round(signals["trend_strength"], 4),
            seasonal_strength=round(signals["seasonal_strength"], 4),
            spectral_entropy=round(signals["spectral_entropy"], 4),
            n_observations=n,
            data_warnings=dq.warnings,
            summary=summary,
        )

    def _recommend_log_transform(
        self, values: np.ndarray, growth_pct: float, score: float
    ) -> Tuple[str, str]:
        """
        Recommend log transform strategy based on data characteristics.

        Log transform (log1p) is useful when:
        - Data spans multiple orders of magnitude (e.g., combining slices)
        - Variance is proportional to level (heteroscedastic)

        Log transform is HARMFUL when:
        - Data has a strong upward/downward trend — compresses the growth signal,
          causing models (especially ARIMA/StatsForecast) to produce flat forecasts
        - Data range is within a single order of magnitude

        Generic rules:
        - Negative values -> 'never'
        - Single-series with strong trend -> 'never' (preserves growth signal)
        - Multi-order-of-magnitude range (>10x) with low trend -> 'always'
        - Moderate range -> 'auto'
        """
        if (values < 0).any():
            return "never", "Data contains negative values; log transform not applicable."

        # Check if data spans multiple orders of magnitude
        recent_n = min(len(values), 52)
        recent_mean = float(np.mean(values[-recent_n:]))
        early_mean = float(np.mean(values[:recent_n])) if len(values) > recent_n else recent_mean
        magnitude_ratio = max(recent_mean, early_mean) / max(min(recent_mean, early_mean), 1)

        # Direct heteroscedasticity test: compare variance of first half vs second half.
        # If variance grows proportionally with level, log transform stabilizes it.
        n = len(values)
        is_heteroscedastic = False
        variance_ratio = 1.0
        if n >= 20:
            mid = n // 2
            var_first = float(np.var(values[:mid]))
            var_second = float(np.var(values[mid:]))
            if var_first > 0:
                variance_ratio = var_second / var_first
                is_heteroscedastic = variance_ratio > 4.0  # 4x variance increase

        # For trending data, log transform compresses the signal that models need.
        # Only apply when data truly spans multiple orders of magnitude AND
        # the concern is heteroscedasticity, not when it's just growth.
        if magnitude_ratio > 10 and growth_pct > 500:
            return "always", (
                f"Data spans {magnitude_ratio:.0f}x range with {growth_pct:.0f}% growth. "
                f"Log transform stabilizes extreme variance differences."
            )
        elif is_heteroscedastic and growth_pct <= 100:
            # Multiplicative seasonality / heteroscedastic variance without extreme trend.
            # Log transform helps without compressing the growth signal.
            return "always", (
                f"Heteroscedastic variance detected (variance ratio: {variance_ratio:.1f}x). "
                f"Log transform stabilizes variance proportional to level."
            )
        elif growth_pct > 100:
            # High growth but NOT multi-order-of-magnitude — log transform would
            # compress the trend signal and cause flat forecasts from ARIMA/ETS
            return "never", (
                f"High growth ({growth_pct:.0f}%) but data stays within {magnitude_ratio:.1f}x range. "
                f"Log transform would compress the trend signal, causing flat forecasts. "
                f"Prophet and StatsForecast handle trends directly."
            )
        elif growth_pct > 50:
            return "auto", (
                f"Moderate growth ({growth_pct:.0f}%). Auto-detect will apply "
                f"log transform if warranted by recent trend acceleration."
            )
        else:
            return "never", (
                f"Growth is moderate or low ({growth_pct:.0f}%). "
                f"Log transform not needed."
            )

    def _recommend_horizon(
        self,
        score: float,
        n_observations: int,
        period: int,
        frequency: str,
        requested_horizon: Optional[int],
    ) -> Tuple[Optional[int], int, str]:
        """
        Recommend forecast horizon based on forecastability and data volume.

        Generic rules based on time series theory:
        - More forecastable series support longer horizons
        - Horizon should not exceed ~25% of training data length
        - Low forecastability -> shorter horizon more reliable
        - Never exceed one full seasonal cycle for poor series

        Returns:
            (recommended_horizon, max_reliable_horizon, reason)
        """
        # Max reliable horizon: ~25% of data, capped by forecastability
        data_based_max = max(1, n_observations // 4)

        if score >= 80:
            # Excellent: can forecast up to a full seasonal cycle
            max_horizon = min(data_based_max, period)
            default = min(period // 4, max_horizon)  # ~13 for weekly
            reason_prefix = "Highly forecastable series"
        elif score >= 60:
            max_horizon = min(data_based_max, period // 2)
            default = min(period // 4, max_horizon)  # ~13 for weekly
            reason_prefix = "Good forecastability"
        elif score >= 40:
            max_horizon = min(data_based_max, period // 4)
            default = min(8, max_horizon)  # ~8 for weekly
            reason_prefix = "Fair forecastability"
        elif score >= 20:
            max_horizon = min(data_based_max, period // 6)
            default = min(4, max_horizon)  # ~4 for weekly
            reason_prefix = "Poor forecastability — shorter horizon more reliable"
        else:
            max_horizon = min(data_based_max, 4)
            default = min(4, max_horizon)
            reason_prefix = "Very low forecastability — keep horizon short"

        max_horizon = max(1, max_horizon)
        default = max(1, default)

        # If user requested a specific horizon, validate it
        if requested_horizon is not None:
            if requested_horizon <= max_horizon:
                return (
                    requested_horizon, max_horizon,
                    f"{reason_prefix}. Requested horizon of {requested_horizon} "
                    f"{frequency} periods is within reliable range (max {max_horizon})."
                )
            else:
                return (
                    requested_horizon, max_horizon,
                    f"{reason_prefix}. Requested horizon of {requested_horizon} "
                    f"exceeds reliable range (max {max_horizon} {frequency} periods). "
                    f"Accuracy will degrade beyond {max_horizon} periods."
                )

        return (
            default, max_horizon,
            f"{reason_prefix}. Recommended {default} {frequency} periods "
            f"(max reliable: {max_horizon})."
        )

    def _build_auto_config_summary(
        self,
        score: float,
        grade: ForecastabilityGrade,
        growth_pct: float,
        training_window: Optional[int],
        models: List[str],
        log_transform: str,
        rec_horizon: Optional[int],
        max_horizon: int,
        warnings: List[str],
    ) -> str:
        """Build human-readable summary of auto-configuration decisions."""
        parts = []

        parts.append(
            f"Forecastability: {grade.value} (score {score:.0f}/100)."
        )

        if abs(growth_pct) > 50:
            direction = "growth" if growth_pct > 0 else "decline"
            parts.append(f"Detected {abs(growth_pct):.0f}% {direction} in the series.")

        if training_window:
            parts.append(f"Using last {training_window} weeks for training (recent data is more representative).")
        else:
            parts.append("Using all available data for training.")

        parts.append(f"Models: {', '.join(models)}.")

        if log_transform in ("always", "auto"):
            parts.append(f"Log transform: {log_transform} (stabilizes high-growth variance).")

        if rec_horizon:
            parts.append(f"Recommended horizon: {rec_horizon} periods (max reliable: {max_horizon}).")

        if warnings:
            parts.append(f"Warnings: {'; '.join(warnings[:3])}")

        return " ".join(parts)
