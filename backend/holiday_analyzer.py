"""
Holiday and Event Impact Analyzer for Time Series Data.

Detects and quantifies holiday/event impacts using STL remainder analysis,
then matches anomalous weeks to known holidays from the preprocessing module.

Research basis:
- STAHL (Haller et al. 2025): STL with explicit holiday component
- Prophet decomposition (Taylor & Letham 2018): Holiday effect quantification
- Intervention analysis (Box & Tiao 1975): Event impact measurement
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from backend.preprocessing import (
    KEY_HOLIDAYS_FIXED,
    FISCAL_QUARTER_ENDS,
    get_all_key_holiday_dates,
    get_thanksgiving_date,
    get_black_friday_date,
)

logger = logging.getLogger(__name__)

# Seasonal period by frequency
SEASONAL_PERIODS = {
    "daily": 7,
    "weekly": 52,
    "monthly": 12,
}


@dataclass
class HolidayImpact:
    """Quantified impact of a known holiday on the time series."""
    holiday_name: str
    affected_weeks: List[str]
    avg_lift_pct: float
    consistency: float          # 0-1, higher = more consistent across years
    direction: str              # "increase" or "decrease"
    confidence: str             # "high", "medium", "low"
    yearly_impacts: Dict[int, float]
    recommendation: str


@dataclass
class AnomalousEvent:
    """An unexplained anomalous week detected via STL remainder analysis."""
    week_date: str
    deviation_pct: float
    direction: str              # "spike" or "dip"
    matched_holiday: Optional[str]
    is_recurring: bool
    note: str


@dataclass
class HolidayAnalysisResult:
    """Complete holiday/event analysis result."""
    holiday_impacts: List[HolidayImpact]
    anomalous_events: List[AnomalousEvent]
    summary: str
    training_recommendations: List[str]
    detected_partial_weeks: List[str]


class HolidayAnalyzer:
    """
    Detects and quantifies holiday/event impacts using STL decomposition.

    Uses existing holiday infrastructure from backend/preprocessing.py
    (KEY_HOLIDAYS_FIXED, get_all_key_holiday_dates, etc.)

    Approach:
    1. STL decompose -> isolate remainder (what's left after trend + seasonality)
    2. Flag weeks where |remainder| > threshold * std(remainder)
    3. Match anomalous weeks to known holidays
    4. Quantify impact per holiday across years
    5. Surface unexplained events
    """

    def __init__(self, anomaly_threshold: float = 2.0):
        self.anomaly_threshold = anomaly_threshold

    def analyze(
        self,
        df: pd.DataFrame,
        time_col: str,
        target_col: str,
        frequency: str = "weekly",
        country: str = "US",
    ) -> HolidayAnalysisResult:
        """
        Main entry point for holiday/event impact analysis.

        Args:
            df: DataFrame with time series data
            time_col: Name of the date column
            target_col: Name of the target column
            frequency: Data frequency ('daily', 'weekly', 'monthly')
            country: Country code for holiday calendar

        Returns:
            HolidayAnalysisResult with impacts, anomalies, and recommendations
        """
        logger.info(f"Analyzing holiday/event impacts: {len(df)} rows, target={target_col}")

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        # Clean target column — handle string-formatted numbers ("29,031" etc.)
        if df[target_col].dtype == "object" or df[target_col].dtype.name == "string":
            cleaned = df[target_col].astype(str).str.replace(",", "", regex=False)
            cleaned = cleaned.str.replace("$", "", regex=False)
            cleaned = cleaned.str.replace(" ", "", regex=False)
            df[target_col] = pd.to_numeric(cleaned, errors="coerce")
        df = df.sort_values(time_col).reset_index(drop=True)

        values = df[target_col].dropna().values.astype(float)
        dates = df[time_col]

        # Get seasonal period
        period = SEASONAL_PERIODS.get(frequency, 52)

        if len(values) < period * 2:
            logger.warning(f"Insufficient data for STL decomposition: {len(values)} rows, need {period * 2}")
            return HolidayAnalysisResult(
                holiday_impacts=[],
                anomalous_events=[],
                summary=f"Insufficient data for holiday analysis (need at least {period * 2} {frequency} observations).",
                training_recommendations=[],
                detected_partial_weeks=[],
            )

        # Step 1: STL decomposition
        trend, seasonal, remainder = self._stl_decompose(values, period)

        # Step 2: Find anomalous weeks from remainder
        anomalous_weeks = self._find_anomalous_weeks(remainder, values, dates)

        # Step 3: Get year range for holiday matching
        years = sorted(dates.dt.year.unique().tolist())

        # Step 4: Match anomalous weeks to known holidays
        matched, unmatched = self._match_to_holidays(anomalous_weeks, dates, years, country)

        # Step 5: Quantify holiday impacts
        holiday_impacts = []
        for holiday_name, week_data in matched.items():
            impact = self._quantify_holiday_impact(holiday_name, week_data)
            if impact is not None:
                holiday_impacts.append(impact)

        # Sort by absolute impact
        holiday_impacts.sort(key=lambda h: abs(h.avg_lift_pct), reverse=True)

        # Step 6: Build anomalous events list
        anomalous_events = self._build_anomalous_events(unmatched, dates, years)

        # Step 7: Detect partial data weeks
        partial_weeks = self._detect_partial_data_weeks(values, dates)

        # Step 8: Generate recommendations
        recommendations = self._generate_training_recommendations(
            holiday_impacts, anomalous_events, partial_weeks
        )

        # Build summary
        summary = self._build_summary(holiday_impacts, anomalous_events, partial_weeks)

        return HolidayAnalysisResult(
            holiday_impacts=holiday_impacts,
            anomalous_events=anomalous_events,
            summary=summary,
            training_recommendations=recommendations,
            detected_partial_weeks=partial_weeks,
        )

    def _stl_decompose(
        self, values: np.ndarray, period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run STL decomposition with robust estimation."""
        stl = STL(values, period=period, robust=True)
        result = stl.fit()
        return result.trend, result.seasonal, result.resid

    def _find_anomalous_weeks(
        self,
        remainder: np.ndarray,
        values: np.ndarray,
        dates: pd.Series,
    ) -> List[Tuple[str, float, int]]:
        """
        Find weeks where STL remainder exceeds threshold * std.

        Returns list of (iso_date, deviation_pct, index) tuples.
        Deviation is expressed as % of the expected value (trend + seasonal).
        """
        std_remainder = np.std(remainder)
        if std_remainder == 0:
            return []

        anomalous = []
        expected = values - remainder  # trend + seasonal

        for i in range(len(remainder)):
            if abs(remainder[i]) > self.anomaly_threshold * std_remainder:
                exp_val = expected[i]
                if exp_val != 0:
                    dev_pct = (remainder[i] / abs(exp_val)) * 100
                else:
                    dev_pct = 0.0
                date_str = dates.iloc[i].strftime("%Y-%m-%d")
                anomalous.append((date_str, dev_pct, i))

        return anomalous

    def _match_to_holidays(
        self,
        anomalous_weeks: List[Tuple[str, float, int]],
        dates: pd.Series,
        years: List[int],
        country: str,
    ) -> Tuple[Dict[str, List[Tuple[str, float]]], List[Tuple[str, float]]]:
        """
        Match anomalous weeks to known holidays.

        For weekly data, checks if the anomalous week contains or is adjacent
        to a known holiday (within ±7 days).

        Returns:
            matched: {holiday_name: [(date, deviation_pct), ...]}
            unmatched: [(date, deviation_pct), ...]
        """
        # Build holiday lookup: date -> holiday name
        holiday_lookup = {}
        for year in years:
            try:
                year_holidays = get_all_key_holiday_dates(year)
                for name, date in year_holidays.items():
                    holiday_lookup[date] = name
            except Exception:
                continue

        matched: Dict[str, List[Tuple[str, float]]] = {}
        unmatched: List[Tuple[str, float]] = []

        for date_str, dev_pct, _idx in anomalous_weeks:
            anomaly_date = pd.Timestamp(date_str)
            found_holiday = None

            # Check if any holiday falls within ±7 days of this week
            for holiday_date, holiday_name in holiday_lookup.items():
                day_diff = abs((anomaly_date - holiday_date).days)
                if day_diff <= 7:
                    found_holiday = holiday_name
                    break

            if found_holiday:
                if found_holiday not in matched:
                    matched[found_holiday] = []
                matched[found_holiday].append((date_str, dev_pct))
            else:
                unmatched.append((date_str, dev_pct))

        return matched, unmatched

    def _quantify_holiday_impact(
        self, holiday_name: str, week_data: List[Tuple[str, float]]
    ) -> Optional[HolidayImpact]:
        """Compute average lift, consistency, and confidence for a matched holiday."""
        if not week_data:
            return None

        # Group by year
        yearly_impacts: Dict[int, float] = {}
        for date_str, dev_pct in week_data:
            year = pd.Timestamp(date_str).year
            # If multiple matches in same year, take the one with largest absolute deviation
            if year not in yearly_impacts or abs(dev_pct) > abs(yearly_impacts[year]):
                yearly_impacts[year] = dev_pct

        if not yearly_impacts:
            return None

        impacts = list(yearly_impacts.values())
        avg_lift = float(np.mean(impacts))
        n_years = len(impacts)

        # Consistency: 1 - normalized std (higher = more consistent)
        if n_years > 1 and avg_lift != 0:
            consistency = max(0.0, 1.0 - abs(float(np.std(impacts)) / abs(avg_lift)))
        else:
            consistency = 0.5  # Unknown with single observation

        # Confidence based on years observed
        if n_years >= 3:
            confidence = "high"
        elif n_years == 2:
            confidence = "medium"
        else:
            confidence = "low"

        direction = "increase" if avg_lift > 0 else "decrease"

        # Generate recommendation
        recommendation = self._holiday_recommendation(holiday_name, avg_lift, consistency, n_years)

        # Pretty-print holiday name
        display_name = holiday_name.replace("_", " ").title()

        return HolidayImpact(
            holiday_name=display_name,
            affected_weeks=[d for d, _ in week_data],
            avg_lift_pct=round(avg_lift, 1),
            consistency=round(consistency, 2),
            direction=direction,
            confidence=confidence,
            yearly_impacts={y: round(v, 1) for y, v in yearly_impacts.items()},
            recommendation=recommendation,
        )

    def _holiday_recommendation(
        self, holiday_name: str, avg_lift: float, consistency: float, n_years: int
    ) -> str:
        """Generate an actionable recommendation for a holiday."""
        display = holiday_name.replace("_", " ").title()
        abs_lift = abs(avg_lift)

        if abs_lift > 30 and consistency > 0.6:
            return (
                f"{display} causes a strong {'+' if avg_lift > 0 else ''}{avg_lift:.0f}% effect "
                f"with high consistency. Prophet holiday windows are configured to handle this."
            )
        elif abs_lift > 15:
            return (
                f"{display} shows a moderate {'+' if avg_lift > 0 else ''}{avg_lift:.0f}% effect. "
                f"Ensure holiday features are enabled in model configuration."
            )
        elif abs_lift > 5:
            return (
                f"{display} has a mild {'+' if avg_lift > 0 else ''}{avg_lift:.0f}% effect. "
                f"Standard seasonal models should capture this."
            )
        else:
            return f"{display} has minimal impact ({avg_lift:+.0f}%). No special handling needed."

    def _build_anomalous_events(
        self,
        unmatched: List[Tuple[str, float]],
        dates: pd.Series,
        years: List[int],
    ) -> List[AnomalousEvent]:
        """Build anomalous event objects for unmatched anomalies."""
        events = []

        for date_str, dev_pct in unmatched:
            anomaly_date = pd.Timestamp(date_str)
            direction = "spike" if dev_pct > 0 else "dip"

            # Check if recurring (same week-of-year in other years)
            week_of_year = anomaly_date.isocalendar()[1]
            is_recurring = False
            for other_date_str, _ in unmatched:
                other_date = pd.Timestamp(other_date_str)
                if (other_date.year != anomaly_date.year and
                        other_date.isocalendar()[1] == week_of_year):
                    is_recurring = True
                    break

            note = (
                f"{'Spike' if direction == 'spike' else 'Dip'} of {dev_pct:+.1f}% "
                f"on week of {date_str}."
            )
            if is_recurring:
                note += f" Similar pattern seen in other years at week {week_of_year}."

            events.append(AnomalousEvent(
                week_date=date_str,
                deviation_pct=round(dev_pct, 1),
                direction=direction,
                matched_holiday=None,
                is_recurring=is_recurring,
                note=note,
            ))

        # Sort by absolute deviation
        events.sort(key=lambda e: abs(e.deviation_pct), reverse=True)
        return events

    def _detect_partial_data_weeks(
        self, values: np.ndarray, dates: pd.Series
    ) -> List[str]:
        """
        Detect weeks that appear to be partial data rather than genuine low demand.

        Pattern: week with value < 50% of the mean of ±2 surrounding weeks,
        AND near a month/quarter/year boundary.
        """
        partial = []
        n = len(values)

        for i in range(2, n - 2):
            surrounding = np.concatenate([values[max(0, i-2):i], values[i+1:min(n, i+3)]])
            surrounding_mean = np.mean(surrounding)

            if surrounding_mean > 0 and values[i] < 0.5 * surrounding_mean:
                date = dates.iloc[i]
                # Check if near a boundary (last or first week of month)
                day = date.day
                days_in_month = pd.Timestamp(year=date.year, month=date.month, day=1).days_in_month

                is_near_boundary = day <= 7 or day >= days_in_month - 6

                if is_near_boundary:
                    drop_pct = ((values[i] / surrounding_mean) - 1) * 100
                    date_str = date.strftime("%Y-%m-%d")
                    logger.info(
                        f"Potential partial data week: {date_str} "
                        f"(value={values[i]:.0f}, surrounding mean={surrounding_mean:.0f}, "
                        f"drop={drop_pct:.1f}%)"
                    )
                    partial.append(date_str)

        return partial

    def _generate_training_recommendations(
        self,
        impacts: List[HolidayImpact],
        anomalies: List[AnomalousEvent],
        partial_weeks: List[str],
    ) -> List[str]:
        """Generate actionable recommendations for training configuration."""
        recs = []

        # Partial data weeks
        if partial_weeks:
            weeks_str = ", ".join(partial_weeks)
            recs.append(
                f"Potential partial data detected on {weeks_str}. "
                f"Consider excluding these weeks from training or setting training "
                f"end date before these weeks."
            )

        # Strong holiday effects
        strong_holidays = [h for h in impacts if abs(h.avg_lift_pct) > 20 and h.confidence == "high"]
        if strong_holidays:
            names = ", ".join(h.holiday_name for h in strong_holidays)
            recs.append(
                f"Strong holiday effects detected for: {names}. "
                f"Prophet holiday windows are configured to handle these automatically. "
                f"Ensure 'prophet' is included in model selection."
            )

        # Moderate holiday effects not well covered
        moderate_holidays = [h for h in impacts if 10 < abs(h.avg_lift_pct) <= 20]
        if moderate_holidays:
            names = ", ".join(h.holiday_name for h in moderate_holidays)
            recs.append(
                f"Moderate holiday effects detected for: {names}. "
                f"Consider enabling holiday features in model configuration."
            )

        # Unexplained anomalies
        significant_anomalies = [a for a in anomalies if abs(a.deviation_pct) > 20]
        if significant_anomalies:
            recs.append(
                f"{len(significant_anomalies)} unexplained anomalous weeks with >20% deviation detected. "
                f"Review for one-time events (promotions, outages, data pipeline issues). "
                f"Consider excluding if they represent data quality issues."
            )

        # Recurring unexplained patterns
        recurring = [a for a in anomalies if a.is_recurring]
        if recurring:
            recs.append(
                f"{len(recurring)} recurring anomalous patterns detected at the same week-of-year "
                f"across multiple years. These may represent unmodeled events. Consider adding "
                f"them as custom holidays in Prophet configuration."
            )

        if not recs:
            recs.append("No significant holiday effects or anomalies detected. Standard model configuration is appropriate.")

        return recs

    def _build_summary(
        self,
        impacts: List[HolidayImpact],
        anomalies: List[AnomalousEvent],
        partial_weeks: List[str],
    ) -> str:
        """Build a human-readable summary."""
        parts = []

        if impacts:
            strong = [h for h in impacts if abs(h.avg_lift_pct) > 15]
            if strong:
                names = ", ".join(h.holiday_name for h in strong[:3])
                parts.append(f"{len(strong)} holidays with significant impact detected ({names})")
            else:
                parts.append(f"{len(impacts)} holidays detected with mild impact")
        else:
            parts.append("No holiday impacts detected")

        if anomalies:
            parts.append(f"{len(anomalies)} unexplained anomalous weeks found")

        if partial_weeks:
            parts.append(f"{len(partial_weeks)} potential partial data weeks identified")

        return ". ".join(parts) + "."
