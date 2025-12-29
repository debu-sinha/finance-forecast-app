"""
Forecast Explainer for Simple Mode.

Provides Excel-level transparency for ML forecasts.
Users can understand and explain every number.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class PeriodBreakdown:
    """Breakdown of forecast for a single period."""
    period: date
    forecast: float
    lower_bound: float
    upper_bound: float

    # Component breakdown
    base: float
    trend: float
    seasonal: float
    holiday: float

    # Human-readable explanation
    explanation: str


@dataclass
class ForecastComponents:
    """Decomposition of forecast into explainable components."""

    # Overall formula
    formula: str = "Forecast = Base + Trend + Seasonality + Holiday Effect"

    # Aggregate components
    total_base: float = 0.0
    total_trend: float = 0.0
    total_seasonal: float = 0.0
    total_holiday: float = 0.0

    # Per-period breakdown
    period_breakdown: List[PeriodBreakdown] = field(default_factory=list)


@dataclass
class ConfidenceAssessment:
    """Assessment of forecast confidence."""

    level: str  # "high", "medium", "low"
    score: float  # 0-100
    mape: float  # Mean Absolute Percentage Error

    # Factors affecting confidence
    factors: List[Dict[str, Any]] = field(default_factory=list)

    # Plain English explanation
    explanation: str = ""


@dataclass
class AuditTrail:
    """Complete audit trail for compliance/reproducibility."""

    # When
    run_id: str
    run_timestamp: str

    # What data
    input_data_hash: str
    input_row_count: int
    input_date_range: Tuple[date, date]

    # What configuration
    config_hash: str
    config_snapshot: Dict[str, Any]

    # What model
    model_type: str
    model_version: str

    # What output
    output_hash: str

    # Reproducibility guarantee
    reproducibility_token: str

    # Optional fields (must come after required fields)
    model_uri: Optional[str] = None
    mlflow_run_id: Optional[str] = None


@dataclass
class ForecastExplanation:
    """Complete explanation of a forecast result."""

    # Summary (for executives)
    summary: str

    # Component breakdown (like Excel formula)
    components: ForecastComponents

    # Confidence assessment
    confidence: ConfidenceAssessment

    # Caveats and warnings
    caveats: List[str]

    # Full audit trail
    audit_trail: AuditTrail


class ForecastExplainer:
    """
    Provides Excel-level transparency for ML forecasts.
    Users can understand and explain every number.
    """

    def explain(
        self,
        forecast_result: Dict[str, Any],
        config: Dict[str, Any],
        data_profile: Dict[str, Any]
    ) -> ForecastExplanation:
        """
        Generate human-readable explanation of forecast.

        Args:
            forecast_result: Results from model training
            config: ForecastConfig used
            data_profile: DataProfile of input data

        Returns:
            ForecastExplanation with all transparency layers
        """
        logger.info("Generating forecast explanation...")

        # Generate each component
        summary = self._generate_summary(forecast_result, config)
        components = self._decompose_forecast(forecast_result)
        confidence = self._assess_confidence(forecast_result, data_profile)
        caveats = self._generate_caveats(forecast_result, data_profile)
        audit_trail = self._build_audit_trail(forecast_result, config, data_profile)

        return ForecastExplanation(
            summary=summary,
            components=components,
            confidence=confidence,
            caveats=caveats,
            audit_trail=audit_trail,
        )

    def _generate_summary(
        self, result: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Generate plain English summary for executives."""

        # Extract key metrics
        forecast_values = result.get('forecast', [])
        total = sum(forecast_values) if forecast_values else 0
        best_model = result.get('best_model', 'Unknown')
        mape = result.get('metrics', {}).get('mape', 0)
        horizon = config.get('horizon', len(forecast_values))

        # Determine trend direction
        if len(forecast_values) >= 2:
            first_half = np.mean(forecast_values[:len(forecast_values)//2])
            second_half = np.mean(forecast_values[len(forecast_values)//2:])
            trend_pct = ((second_half - first_half) / first_half * 100) if first_half != 0 else 0
            trend_direction = "increasing" if trend_pct > 1 else "decreasing" if trend_pct < -1 else "stable"
        else:
            trend_pct = 0
            trend_direction = "stable"

        # Confidence level (stricter thresholds for financial forecasting)
        if mape <= 3:
            confidence = "High"
        elif mape <= 5:
            confidence = "Medium"
        else:
            confidence = "Low"

        summary = f"""FORECAST SUMMARY
================

Total Forecast ({horizon} periods): ${total:,.0f}
Trend: {trend_direction.capitalize()} ({'+' if trend_pct > 0 else ''}{trend_pct:.1f}%)
Confidence: {confidence} (MAPE: {mape:.1f}%)
Best Model: {best_model}

This forecast was generated automatically based on your historical data.
All parameters were optimized for your specific data patterns."""

        return summary

    def _decompose_forecast(self, result: Dict[str, Any]) -> ForecastComponents:
        """Break down forecast into explainable components like Excel formula."""

        forecast_values = result.get('forecast', [])
        forecast_dates = result.get('dates', [])
        lower_bounds = result.get('lower', forecast_values)
        upper_bounds = result.get('upper', forecast_values)

        # Try to get component breakdown from model
        components_data = result.get('components', {})

        # If model provides decomposition, use it
        base_values = components_data.get('base', [np.mean(forecast_values)] * len(forecast_values))
        trend_values = components_data.get('trend', [0] * len(forecast_values))
        seasonal_values = components_data.get('seasonal', [0] * len(forecast_values))
        holiday_values = components_data.get('holiday', [0] * len(forecast_values))

        # If no decomposition, estimate from forecast values
        if not components_data and len(forecast_values) > 0:
            base_values, trend_values, seasonal_values, holiday_values = \
                self._estimate_components(forecast_values, forecast_dates)

        # Build period-by-period breakdown
        period_breakdown = []
        for i, val in enumerate(forecast_values):
            date_val = forecast_dates[i] if i < len(forecast_dates) else None
            base = base_values[i] if i < len(base_values) else 0
            trend = trend_values[i] if i < len(trend_values) else 0
            seasonal = seasonal_values[i] if i < len(seasonal_values) else 0
            holiday = holiday_values[i] if i < len(holiday_values) else 0
            lower = lower_bounds[i] if i < len(lower_bounds) else val
            upper = upper_bounds[i] if i < len(upper_bounds) else val

            # Generate explanation
            explanation = self._generate_period_explanation(
                val, base, trend, seasonal, holiday
            )

            period_breakdown.append(PeriodBreakdown(
                period=date_val,
                forecast=round(val, 2),
                lower_bound=round(lower, 2),
                upper_bound=round(upper, 2),
                base=round(base, 2),
                trend=round(trend, 2),
                seasonal=round(seasonal, 2),
                holiday=round(holiday, 2),
                explanation=explanation,
            ))

        return ForecastComponents(
            formula="Forecast = Base + Trend + Seasonality + Holiday Effect",
            total_base=round(sum(base_values), 2),
            total_trend=round(sum(trend_values), 2),
            total_seasonal=round(sum(seasonal_values), 2),
            total_holiday=round(sum(holiday_values), 2),
            period_breakdown=period_breakdown,
        )

    def _estimate_components(
        self, forecast_values: List[float], forecast_dates: List
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Estimate components when model doesn't provide them."""

        n = len(forecast_values)
        mean_val = np.mean(forecast_values)

        # Base: average value
        base_values = [mean_val] * n

        # Trend: linear component
        if n > 1:
            x = np.arange(n)
            slope = np.polyfit(x, forecast_values, 1)[0]
            trend_values = [slope * i for i in range(n)]
        else:
            trend_values = [0] * n

        # Seasonal: deviation from trend-adjusted base
        seasonal_values = []
        for i, val in enumerate(forecast_values):
            residual = val - mean_val - trend_values[i]
            seasonal_values.append(residual)

        # Holiday: zero if not detected (would need date analysis)
        holiday_values = [0] * n

        return base_values, trend_values, seasonal_values, holiday_values

    def _generate_period_explanation(
        self, forecast: float, base: float, trend: float,
        seasonal: float, holiday: float
    ) -> str:
        """Generate human-readable explanation for a single period."""

        parts = [f"${base:,.0f} base"]

        if abs(trend) > 0.01:
            sign = "+" if trend >= 0 else ""
            parts.append(f"{sign}${trend:,.0f} trend")

        if abs(seasonal) > 0.01:
            sign = "+" if seasonal >= 0 else ""
            parts.append(f"{sign}${seasonal:,.0f} seasonal")

        if abs(holiday) > 0.01:
            sign = "+" if holiday >= 0 else ""
            parts.append(f"{sign}${holiday:,.0f} holiday")

        return " ".join(parts) + f" = ${forecast:,.0f}"

    def _assess_confidence(
        self, result: Dict[str, Any], profile: Dict[str, Any]
    ) -> ConfidenceAssessment:
        """Assess forecast confidence based on metrics and data quality."""

        mape = result.get('metrics', {}).get('mape', 15)
        history_months = profile.get('history_months', 12)
        quality_score = profile.get('data_quality_score', 70)
        holiday_coverage = profile.get('holiday_coverage_score', 50)

        # Calculate confidence score (0-100)
        factors = []

        # MAPE factor (40% weight)
        # Stricter thresholds aligned with financial forecasting requirements
        # Target: Under 1% MAPE for production-grade forecasts
        if mape <= 1:
            mape_score = 100
            factors.append({'factor': 'Accuracy', 'score': 100, 'note': 'Excellent (MAPE ≤1%)'})
        elif mape <= 3:
            mape_score = 90
            factors.append({'factor': 'Accuracy', 'score': 90, 'note': 'Very Good (MAPE 1-3%)'})
        elif mape <= 5:
            mape_score = 75
            factors.append({'factor': 'Accuracy', 'score': 75, 'note': 'Good (MAPE 3-5%)'})
        elif mape <= 10:
            mape_score = 50
            factors.append({'factor': 'Accuracy', 'score': 50, 'note': 'Fair (MAPE 5-10%)'})
        else:
            mape_score = 25
            factors.append({'factor': 'Accuracy', 'score': 25, 'note': f'Low (MAPE {mape:.1f}%)'})

        # History length factor (30% weight)
        if history_months >= 36:
            history_score = 100
            factors.append({'factor': 'History Length', 'score': 100, 'note': '3+ years'})
        elif history_months >= 24:
            history_score = 80
            factors.append({'factor': 'History Length', 'score': 80, 'note': '2-3 years'})
        elif history_months >= 12:
            history_score = 60
            factors.append({'factor': 'History Length', 'score': 60, 'note': '1-2 years'})
        else:
            history_score = 30
            factors.append({'factor': 'History Length', 'score': 30, 'note': '<1 year'})

        # Data quality factor (20% weight)
        factors.append({'factor': 'Data Quality', 'score': quality_score, 'note': f'{quality_score}/100'})

        # Holiday coverage factor (10% weight)
        factors.append({'factor': 'Holiday Coverage', 'score': holiday_coverage, 'note': f'{holiday_coverage}/100'})

        # Weighted score
        score = (
            mape_score * 0.4 +
            history_score * 0.3 +
            quality_score * 0.2 +
            holiday_coverage * 0.1
        )

        # Determine level
        if score >= 80:
            level = "high"
            explanation = "High confidence - accurate model with sufficient historical data."
        elif score >= 60:
            level = "medium"
            explanation = "Medium confidence - reasonable accuracy but some limitations in data."
        else:
            level = "low"
            explanation = "Low confidence - limited data or high forecast variance. Use with caution."

        return ConfidenceAssessment(
            level=level,
            score=round(score, 1),
            mape=mape,
            factors=factors,
            explanation=explanation,
        )

    def _generate_caveats(
        self, result: Dict[str, Any], profile: Dict[str, Any]
    ) -> List[str]:
        """Generate caveats and warnings about the forecast."""

        caveats = []

        # Data-based caveats
        history_months = profile.get('history_months', 12)
        if history_months < 24:
            caveats.append(
                f"Based on {history_months:.0f} months of data. "
                "Holiday patterns may not be fully captured."
            )

        holiday_coverage = profile.get('holiday_coverage_score', 50)
        if holiday_coverage < 70:
            caveats.append(
                "Limited holiday data. Thanksgiving/Christmas forecasts "
                "may be less accurate."
            )

        # Model-based caveats
        mape = result.get('metrics', {}).get('mape', 10)
        if mape > 10:
            caveats.append(
                f"Model accuracy (MAPE {mape:.1f}%) indicates some uncertainty. "
                "Consider the confidence intervals."
            )

        # General caveat
        caveats.append(
            "Forecasts assume historical patterns continue. "
            "Unexpected events may affect actual results."
        )

        return caveats

    def _build_audit_trail(
        self, result: Dict[str, Any], config: Dict[str, Any], profile: Dict[str, Any]
    ) -> AuditTrail:
        """Build complete audit trail for compliance/reproducibility."""

        import hashlib
        from datetime import datetime

        # Generate output hash
        forecast_str = str(result.get('forecast', []))
        output_hash = hashlib.sha256(forecast_str.encode()).hexdigest()[:16]

        # Get config hash
        config_hash = config.get('config_hash', 'unknown')
        data_hash = profile.get('data_hash', 'unknown')

        # Reproducibility token
        model_version = result.get('model_version', '1.0')
        reproducibility_token = f"{data_hash}:{config_hash}:{model_version}"

        return AuditTrail(
            run_id=result.get('run_id', 'unknown'),
            run_timestamp=datetime.now().isoformat(),

            input_data_hash=data_hash,
            input_row_count=profile.get('row_count', 0),
            input_date_range=profile.get('date_range', (None, None)),

            config_hash=config_hash,
            config_snapshot=config,

            model_type=result.get('best_model', 'unknown'),
            model_version=model_version,
            model_uri=result.get('model_uri'),
            mlflow_run_id=result.get('mlflow_run_id'),

            output_hash=output_hash,
            reproducibility_token=reproducibility_token,
        )


def format_explanation_for_display(explanation: ForecastExplanation) -> Dict[str, Any]:
    """Format explanation for frontend display."""

    return {
        'summary': explanation.summary,
        'components': {
            'formula': explanation.components.formula,
            'totals': {
                'base': explanation.components.total_base,
                'trend': explanation.components.total_trend,
                'seasonal': explanation.components.total_seasonal,
                'holiday': explanation.components.total_holiday,
            },
            'periods': [
                {
                    'date': str(p.period) if p.period else None,
                    'forecast': p.forecast,
                    'lower': p.lower_bound,
                    'upper': p.upper_bound,
                    'base': p.base,
                    'trend': p.trend,
                    'seasonal': p.seasonal,
                    'holiday': p.holiday,
                    'explanation': p.explanation,
                }
                for p in explanation.components.period_breakdown
            ],
        },
        'confidence': {
            'level': explanation.confidence.level,
            'score': explanation.confidence.score,
            'mape': explanation.confidence.mape,
            'factors': explanation.confidence.factors,
            'explanation': explanation.confidence.explanation,
        },
        'caveats': explanation.caveats,
        'audit': {
            'run_id': explanation.audit_trail.run_id,
            'timestamp': explanation.audit_trail.run_timestamp,
            'data_hash': explanation.audit_trail.input_data_hash,
            'config_hash': explanation.audit_trail.config_hash,
            'model': explanation.audit_trail.model_type,
            'reproducibility_token': explanation.audit_trail.reproducibility_token,
        },
    }
