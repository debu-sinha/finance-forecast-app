"""
Conformal Prediction Intervals for distribution-free uncertainty quantification.

Provides calibrated prediction intervals with guaranteed coverage using
residual-based conformal prediction methods.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
import warnings
from backend.utils.logging_utils import log_io

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@log_io
def calculate_conformal_intervals(
    residuals: np.ndarray,
    point_forecasts: np.ndarray,
    coverage: float = 0.90
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate conformal prediction intervals from validation residuals.

    Uses the quantile of absolute residuals to create symmetric intervals
    around point forecasts with guaranteed coverage.

    Args:
        residuals: Array of validation residuals (actual - predicted)
        point_forecasts: Array of point forecasts
        coverage: Desired coverage level (e.g., 0.90 for 90% intervals)

    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    if len(residuals) == 0:
        logger.warning("No residuals provided for conformal intervals")
        return point_forecasts * 0.9, point_forecasts * 1.1

    # Calculate the quantile of absolute residuals
    # For coverage α, we need the (1-α) quantile
    abs_residuals = np.abs(residuals)

    # Apply finite sample correction for conformal prediction
    # Ref: Vovk et al. (2005) - Algorithmic Learning in a Random World
    n = len(abs_residuals)
    adjusted_quantile = min(1.0, (1 - coverage) * (n + 1) / n)
    quantile_value = np.quantile(abs_residuals, 1 - adjusted_quantile)

    logger.info(f"Conformal interval width: ±{quantile_value:.2f} (coverage={coverage*100:.0f}%)")

    # Create symmetric intervals
    lower = point_forecasts - quantile_value
    upper = point_forecasts + quantile_value

    return lower, upper


@log_io
def calculate_asymmetric_conformal_intervals(
    residuals: np.ndarray,
    point_forecasts: np.ndarray,
    coverage: float = 0.90
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate asymmetric conformal prediction intervals.

    Uses separate quantiles for negative and positive residuals to handle
    asymmetric error distributions (common in financial forecasting).

    Args:
        residuals: Array of validation residuals (actual - predicted)
        point_forecasts: Array of point forecasts
        coverage: Desired coverage level

    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    if len(residuals) == 0:
        logger.warning("No residuals provided for conformal intervals")
        return point_forecasts * 0.9, point_forecasts * 1.1

    # Separate positive and negative residuals
    alpha = 1 - coverage
    lower_alpha = alpha / 2
    upper_alpha = 1 - alpha / 2

    # Calculate quantiles of signed residuals
    lower_quantile = np.quantile(residuals, lower_alpha)
    upper_quantile = np.quantile(residuals, upper_alpha)

    # Create asymmetric intervals
    # If residual = actual - predicted, then:
    # actual = predicted + residual
    # lower bound: predicted + lower_quantile (usually negative)
    # upper bound: predicted + upper_quantile (usually positive)
    lower = point_forecasts + lower_quantile
    upper = point_forecasts + upper_quantile

    logger.info(f"Asymmetric conformal interval: [{lower_quantile:.2f}, +{upper_quantile:.2f}] (coverage={coverage*100:.0f}%)")

    return lower, upper


@log_io
def calculate_growing_intervals(
    residuals: np.ndarray,
    point_forecasts: np.ndarray,
    coverage: float = 0.90,
    growth_rate: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate conformal prediction intervals that grow with forecast horizon.

    Prediction uncertainty naturally increases as we forecast further into
    the future. This method applies a growth factor to interval widths.

    Args:
        residuals: Array of validation residuals
        point_forecasts: Array of point forecasts
        coverage: Desired coverage level
        growth_rate: Per-period growth rate for interval width (e.g., 0.02 = 2% per period)

    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    # Get base conformal intervals
    base_lower, base_upper = calculate_conformal_intervals(
        residuals, point_forecasts, coverage
    )

    # Calculate base width
    base_width = base_upper - base_lower
    horizon = len(point_forecasts)

    # Apply growing factor
    growth_factors = 1 + growth_rate * np.arange(horizon)
    adjusted_width = base_width * growth_factors

    # Recalculate bounds from center
    center = point_forecasts
    lower = center - adjusted_width / 2
    upper = center + adjusted_width / 2

    logger.info(f"Growing intervals: base_width={base_width[0]:.2f}, final_width={adjusted_width[-1]:.2f}")

    return lower, upper


@log_io
def add_conformal_intervals(
    validation_residuals: np.ndarray,
    forecast_df: pd.DataFrame,
    coverage: float = 0.90,
    method: str = 'symmetric',
    growth_rate: Optional[float] = None
) -> pd.DataFrame:
    """
    Add conformal prediction intervals to a forecast DataFrame.

    This function post-processes an existing forecast to add
    distribution-free prediction intervals with guaranteed coverage.

    Args:
        validation_residuals: Residuals from validation set (actual - predicted)
        forecast_df: DataFrame with 'ds' and 'yhat' columns
        coverage: Desired coverage level (e.g., 0.90 for 90% intervals)
        method: 'symmetric', 'asymmetric', or 'growing'
        growth_rate: Growth rate for 'growing' method (e.g., 0.02)

    Returns:
        DataFrame with updated yhat_lower and yhat_upper columns
    """
    if 'yhat' not in forecast_df.columns:
        raise ValueError("forecast_df must have 'yhat' column")

    forecast_df = forecast_df.copy()
    point_forecasts = forecast_df['yhat'].values

    # Calculate conformal intervals based on method
    if method == 'asymmetric':
        lower, upper = calculate_asymmetric_conformal_intervals(
            validation_residuals, point_forecasts, coverage
        )
    elif method == 'growing':
        lower, upper = calculate_growing_intervals(
            validation_residuals, point_forecasts, coverage, growth_rate or 0.02
        )
    else:  # symmetric
        lower, upper = calculate_conformal_intervals(
            validation_residuals, point_forecasts, coverage
        )

    # CRITICAL: Clip negative intervals - financial metrics cannot be negative
    lower = np.maximum(lower, 0.0)
    upper = np.maximum(upper, point_forecasts)

    forecast_df['yhat_lower'] = lower
    forecast_df['yhat_upper'] = upper

    return forecast_df


@log_io
def evaluate_interval_coverage(
    actuals: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    target_coverage: float = 0.90
) -> Dict[str, Any]:
    """
    Evaluate the actual coverage of prediction intervals.

    Args:
        actuals: Array of actual values
        lower_bounds: Array of lower interval bounds
        upper_bounds: Array of upper interval bounds
        target_coverage: Target coverage level

    Returns:
        Dict with coverage statistics
    """
    n = len(actuals)
    if n == 0:
        return {'actual_coverage': 0, 'target_coverage': target_coverage}

    # Count points within intervals
    within_interval = (actuals >= lower_bounds) & (actuals <= upper_bounds)
    actual_coverage = np.mean(within_interval)

    # Calculate interval widths
    widths = upper_bounds - lower_bounds
    mean_width = np.mean(widths)
    mean_width_pct = np.mean(widths / np.maximum(actuals, 0.01)) * 100

    # Coverage gap
    coverage_gap = actual_coverage - target_coverage

    results = {
        'actual_coverage': float(actual_coverage),
        'target_coverage': float(target_coverage),
        'coverage_gap': float(coverage_gap),
        'n_samples': int(n),
        'n_within': int(np.sum(within_interval)),
        'mean_interval_width': float(mean_width),
        'mean_interval_width_pct': float(mean_width_pct),
        'is_calibrated': abs(coverage_gap) < 0.05  # Within 5% of target
    }

    logger.info(f"Interval coverage: {actual_coverage*100:.1f}% (target: {target_coverage*100:.0f}%)")
    if results['is_calibrated']:
        logger.info("  ✓ Intervals are well-calibrated")
    else:
        logger.warning(f"  ⚠️ Intervals under/over-cover by {coverage_gap*100:.1f}%")

    return results


@log_io
def calibrate_intervals(
    validation_actuals: np.ndarray,
    validation_predictions: np.ndarray,
    validation_lower: np.ndarray,
    validation_upper: np.ndarray,
    target_coverage: float = 0.90,
    max_iterations: int = 10
) -> float:
    """
    Find the scaling factor to achieve target coverage.

    Iteratively adjusts interval width to achieve desired coverage
    on the validation set.

    Args:
        validation_actuals: Actual values on validation set
        validation_predictions: Predicted values on validation set
        validation_lower: Current lower bounds
        validation_upper: Current upper bounds
        target_coverage: Target coverage level
        max_iterations: Maximum calibration iterations

    Returns:
        Scaling factor to apply to interval widths
    """
    current_lower = validation_lower.copy()
    current_upper = validation_upper.copy()
    center = validation_predictions

    scale = 1.0

    for i in range(max_iterations):
        # Evaluate current coverage
        within = (validation_actuals >= current_lower) & (validation_actuals <= current_upper)
        current_coverage = np.mean(within)

        # Check if we've achieved target
        if abs(current_coverage - target_coverage) < 0.01:
            logger.info(f"Calibration converged at scale={scale:.3f}, coverage={current_coverage*100:.1f}%")
            break

        # Adjust scale
        if current_coverage < target_coverage:
            scale *= 1.1  # Widen intervals
        else:
            scale *= 0.95  # Narrow intervals

        # Apply scale to intervals
        base_width = (validation_upper - validation_lower) / 2
        current_lower = center - base_width * scale
        current_upper = center + base_width * scale

    return scale


class ConformalPredictor:
    """
    Conformal predictor that can be fitted on validation data
    and applied to new forecasts.
    """

    def __init__(
        self,
        coverage: float = 0.90,
        method: str = 'symmetric',
        growth_rate: Optional[float] = None,
        calibrate: bool = True
    ):
        """
        Initialize conformal predictor.

        Args:
            coverage: Target coverage level
            method: 'symmetric', 'asymmetric', or 'growing'
            growth_rate: Growth rate for 'growing' method
            calibrate: Whether to calibrate intervals on validation set
        """
        self.coverage = coverage
        self.method = method
        self.growth_rate = growth_rate
        self.calibrate = calibrate

        self._residuals = None
        self._scale = 1.0
        self._quantile_value = None

    @log_io
    def fit(
        self,
        validation_actuals: np.ndarray,
        validation_predictions: np.ndarray,
        validation_lower: Optional[np.ndarray] = None,
        validation_upper: Optional[np.ndarray] = None
    ) -> 'ConformalPredictor':
        """
        Fit the conformal predictor on validation data.

        Args:
            validation_actuals: Actual values
            validation_predictions: Predicted values
            validation_lower: Optional existing lower bounds (for calibration)
            validation_upper: Optional existing upper bounds (for calibration)

        Returns:
            Self
        """
        # Calculate residuals
        self._residuals = validation_actuals - validation_predictions

        # Calculate base quantile
        abs_residuals = np.abs(self._residuals)
        n = len(abs_residuals)
        adjusted_quantile = min(1.0, (1 - self.coverage) * (n + 1) / n)
        self._quantile_value = np.quantile(abs_residuals, 1 - adjusted_quantile)

        # Calibrate if requested and existing bounds provided
        if self.calibrate and validation_lower is not None and validation_upper is not None:
            self._scale = calibrate_intervals(
                validation_actuals,
                validation_predictions,
                validation_lower,
                validation_upper,
                self.coverage
            )

        logger.info(f"Conformal predictor fitted: quantile={self._quantile_value:.2f}, scale={self._scale:.3f}")

        return self

    @log_io
    def predict_intervals(
        self,
        point_forecasts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals for new forecasts.

        Args:
            point_forecasts: Array of point forecasts

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if self._residuals is None:
            raise ValueError("Conformal predictor not fitted. Call fit() first.")

        # Calculate intervals based on method
        if self.method == 'asymmetric':
            lower, upper = calculate_asymmetric_conformal_intervals(
                self._residuals, point_forecasts, self.coverage
            )
        elif self.method == 'growing':
            lower, upper = calculate_growing_intervals(
                self._residuals, point_forecasts, self.coverage, self.growth_rate or 0.02
            )
        else:
            lower, upper = calculate_conformal_intervals(
                self._residuals, point_forecasts, self.coverage
            )

        # Apply calibration scale
        if self._scale != 1.0:
            width = upper - lower
            center = point_forecasts
            lower = center - (width / 2) * self._scale
            upper = center + (width / 2) * self._scale

        # Clip negative intervals
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, point_forecasts)

        return lower, upper

    @log_io
    def transform(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add conformal intervals to a forecast DataFrame.

        Args:
            forecast_df: DataFrame with 'yhat' column

        Returns:
            DataFrame with updated yhat_lower, yhat_upper
        """
        forecast_df = forecast_df.copy()
        point_forecasts = forecast_df['yhat'].values

        lower, upper = self.predict_intervals(point_forecasts)

        forecast_df['yhat_lower'] = lower
        forecast_df['yhat_upper'] = upper

        return forecast_df
