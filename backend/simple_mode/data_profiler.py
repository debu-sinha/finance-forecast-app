"""
Data Profiler for Simple Mode.

Automatically analyzes uploaded data and extracts configuration,
eliminating the need for users to specify parameters manually.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, date
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class Warning:
    """User-friendly warning about data issues."""
    level: str  # "low", "medium", "high"
    message: str
    recommendation: str


@dataclass
class DataProfile:
    """Complete profile of uploaded data."""

    # Auto-detected configuration
    frequency: str  # "daily", "weekly", "monthly"
    date_column: str
    target_column: str

    # Date range info
    date_range: Tuple[date, date]
    total_periods: int
    history_months: float

    # Data quality metrics
    missing_values: int
    missing_periods: List[date]
    outliers: List[Dict[str, Any]]
    data_quality_score: float  # 0-100

    # Holiday coverage
    holidays_in_data: List[str]
    holiday_coverage_score: float  # 0-100

    # Detected patterns
    has_trend: bool
    has_seasonality: bool
    seasonality_period: Optional[int]

    # Covariate columns (optional)
    covariate_columns: List[str]

    # Warnings for user
    warnings: List[Warning]

    # Data fingerprint for reproducibility
    data_hash: str
    row_count: int

    # Recommendations
    recommended_horizon: int
    recommended_models: List[str]


class DataProfiler:
    """
    Automatically analyzes uploaded data and extracts configuration.
    Eliminates need for user to specify parameters.
    """

    # Common date column names
    DATE_COLUMN_PATTERNS = [
        'date', 'ds', 'time', 'timestamp', 'period', 'week', 'month', 'day',
        'week_start', 'week_end', 'week_starting', 'week_ending',
        'report_date', 'transaction_date', 'order_date'
    ]

    # Common target column names
    TARGET_COLUMN_PATTERNS = [
        'y', 'value', 'target', 'amount', 'revenue', 'sales', 'quantity',
        'total', 'count', 'volume', 'actual', 'demand', 'orders'
    ]

    # Columns to exclude from target detection
    EXCLUDE_FROM_TARGET = [
        'id', 'index', 'row', 'year', 'month', 'day', 'week', 'quarter',
        'day_of_week', 'is_weekend', 'is_holiday'
    ]

    def profile(self, df: pd.DataFrame) -> DataProfile:
        """
        Analyze data and generate complete profile.

        Args:
            df: Input DataFrame

        Returns:
            DataProfile with all detected settings and recommendations
        """
        logger.info(f"Profiling data: {len(df)} rows, {len(df.columns)} columns")

        # Step 1: Detect date column
        date_column = self._detect_date_column(df)
        logger.info(f"Detected date column: {date_column}")

        # Ensure date column is datetime
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)

        # Step 2: Detect frequency
        frequency = self._detect_frequency(df, date_column)
        logger.info(f"Detected frequency: {frequency}")

        # Step 3: Detect target column
        target_column = self._detect_target_column(df, date_column)
        logger.info(f"Detected target column: {target_column}")

        # Step 4: Analyze date range
        date_range = self._get_date_range(df, date_column)
        history_months = self._calculate_history_months(date_range)

        # Step 5: Find data quality issues
        missing_values = df[target_column].isna().sum()
        missing_periods = self._find_missing_periods(df, date_column, frequency)
        outliers = self._detect_outliers(df, target_column)

        # Step 6: Analyze holiday coverage
        holidays_in_data, holiday_coverage = self._analyze_holiday_coverage(
            df, date_column, frequency
        )

        # Step 7: Detect patterns
        has_trend, has_seasonality, seasonality_period = self._detect_patterns(
            df, target_column, frequency
        )

        # Step 8: Find covariate columns
        covariate_columns = self._find_covariate_columns(
            df, date_column, target_column
        )

        # Step 9: Calculate quality score
        quality_score = self._calculate_quality_score(
            df, target_column, missing_values, len(missing_periods), len(outliers)
        )

        # Step 10: Generate warnings
        warnings = self._generate_warnings(
            history_months, holiday_coverage, missing_values,
            len(missing_periods), quality_score, frequency
        )

        # Step 11: Generate recommendations
        recommended_horizon = self._recommend_horizon(frequency)
        recommended_models = self._recommend_models(
            history_months, has_seasonality, len(covariate_columns) > 0
        )

        # Step 12: Compute data hash for reproducibility
        data_hash = self._compute_data_hash(df)

        return DataProfile(
            # Auto-detected config
            frequency=frequency,
            date_column=date_column,
            target_column=target_column,

            # Date range
            date_range=date_range,
            total_periods=len(df),
            history_months=history_months,

            # Quality metrics
            missing_values=missing_values,
            missing_periods=missing_periods,
            outliers=outliers,
            data_quality_score=quality_score,

            # Holiday coverage
            holidays_in_data=holidays_in_data,
            holiday_coverage_score=holiday_coverage,

            # Patterns
            has_trend=has_trend,
            has_seasonality=has_seasonality,
            seasonality_period=seasonality_period,

            # Covariates
            covariate_columns=covariate_columns,

            # Warnings
            warnings=warnings,

            # Reproducibility
            data_hash=data_hash,
            row_count=len(df),

            # Recommendations
            recommended_horizon=recommended_horizon,
            recommended_models=recommended_models,
        )

    def _detect_date_column(self, df: pd.DataFrame) -> str:
        """Detect the date column from data."""

        # First, try exact matches with common names
        for col in df.columns:
            if col.lower() in self.DATE_COLUMN_PATTERNS:
                return col

        # Second, try partial matches
        for col in df.columns:
            col_lower = col.lower()
            for pattern in self.DATE_COLUMN_PATTERNS:
                if pattern in col_lower:
                    return col

        # Third, try to find datetime columns
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                valid_ratio = parsed.notna().sum() / len(df)
                if valid_ratio > 0.9:  # 90% valid dates
                    return col
            except:
                continue

        # Fourth, check dtype
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            return datetime_cols[0]

        raise ValueError(
            "Could not detect date column. Please ensure your data has a column "
            "named 'date', 'ds', 'week_start', or similar."
        )

    def _detect_frequency(self, df: pd.DataFrame, date_column: str) -> str:
        """Detect if data is daily, weekly, or monthly."""

        dates = pd.to_datetime(df[date_column]).sort_values()

        if len(dates) < 2:
            return "daily"  # Default

        # Calculate median gap between consecutive dates
        gaps = dates.diff().dropna()
        median_gap_days = gaps.median().days

        if median_gap_days <= 1:
            return "daily"
        elif median_gap_days <= 8:
            return "weekly"
        elif median_gap_days <= 32:
            return "monthly"
        else:
            return "monthly"  # Default for longer gaps

    def _detect_target_column(self, df: pd.DataFrame, date_column: str) -> str:
        """Detect the target (value) column."""

        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove date column if numeric
        if date_column in numeric_cols:
            numeric_cols.remove(date_column)

        # Remove excluded columns
        numeric_cols = [
            c for c in numeric_cols
            if c.lower() not in self.EXCLUDE_FROM_TARGET
        ]

        if not numeric_cols:
            raise ValueError(
                "Could not detect target column. Please ensure your data has "
                "a numeric column for forecasting (e.g., 'revenue', 'sales', 'value')."
            )

        # First, try exact matches with common names
        for col in numeric_cols:
            if col.lower() in self.TARGET_COLUMN_PATTERNS:
                return col

        # Second, try partial matches
        for col in numeric_cols:
            col_lower = col.lower()
            for pattern in self.TARGET_COLUMN_PATTERNS:
                if pattern in col_lower:
                    return col

        # Third, pick the column with highest variance (likely the target)
        variances = {col: df[col].var() for col in numeric_cols}
        return max(variances, key=variances.get)

    def _get_date_range(
        self, df: pd.DataFrame, date_column: str
    ) -> Tuple[date, date]:
        """Get the date range of the data."""
        dates = pd.to_datetime(df[date_column])
        return (dates.min().date(), dates.max().date())

    def _calculate_history_months(self, date_range: Tuple[date, date]) -> float:
        """Calculate how many months of history we have."""
        start, end = date_range
        days = (end - start).days
        return round(days / 30.44, 1)  # Average days per month

    def _find_missing_periods(
        self, df: pd.DataFrame, date_column: str, frequency: str
    ) -> List[date]:
        """Find gaps in the time series."""

        dates = pd.to_datetime(df[date_column]).sort_values()

        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'MS'
        }

        # Generate expected date range
        expected = pd.date_range(
            start=dates.min(),
            end=dates.max(),
            freq=freq_map.get(frequency, 'D')
        )

        # Find missing dates
        actual_dates = set(dates.dt.normalize())
        expected_dates = set(expected.normalize())

        missing = expected_dates - actual_dates
        return sorted([d.date() for d in missing])[:10]  # Return first 10

    def _detect_outliers(
        self, df: pd.DataFrame, target_column: str
    ) -> List[Dict[str, Any]]:
        """Detect outliers using IQR method."""

        values = df[target_column].dropna()

        if len(values) < 10:
            return []

        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = []
        for idx, val in values.items():
            if val < lower_bound or val > upper_bound:
                outliers.append({
                    'index': idx,
                    'value': val,
                    'type': 'low' if val < lower_bound else 'high'
                })

        return outliers[:10]  # Return first 10

    def _analyze_holiday_coverage(
        self, df: pd.DataFrame, date_column: str, frequency: str
    ) -> Tuple[List[str], float]:
        """Analyze how many holidays are covered in the data."""

        try:
            import holidays as holidays_lib
        except ImportError:
            return [], 0.0

        dates = pd.to_datetime(df[date_column])
        years = dates.dt.year.unique()

        # Get all US holidays in the date range
        all_holidays = []
        for year in years:
            us_holidays = holidays_lib.US(years=int(year))
            all_holidays.extend(us_holidays.items())

        # Major holidays to check
        major_holidays = [
            'Thanksgiving', 'Christmas', 'New Year',
            'Independence Day', 'Memorial Day', 'Labor Day'
        ]

        found_holidays = []
        for holiday_date, holiday_name in all_holidays:
            for major in major_holidays:
                if major.lower() in holiday_name.lower():
                    # Check if this holiday is in our data
                    if frequency == 'weekly':
                        # Check if holiday week is covered
                        holiday_ts = pd.Timestamp(holiday_date)
                        week_start = holiday_ts - pd.Timedelta(days=holiday_ts.dayofweek)
                        if any(abs((dates - week_start).dt.days) <= 7):
                            found_holidays.append(holiday_name)
                    else:
                        if holiday_date in dates.dt.date.values:
                            found_holidays.append(holiday_name)

        # Calculate coverage score
        # Need at least 2 of each major holiday for good coverage
        holiday_counts = {}
        for h in found_holidays:
            for major in major_holidays:
                if major.lower() in h.lower():
                    holiday_counts[major] = holiday_counts.get(major, 0) + 1

        # Score: 100 if 2+ of each, proportionally less otherwise
        total_score = 0
        for major in major_holidays:
            count = holiday_counts.get(major, 0)
            total_score += min(count / 2, 1) * (100 / len(major_holidays))

        return list(set(found_holidays)), round(total_score, 1)

    def _detect_patterns(
        self, df: pd.DataFrame, target_column: str, frequency: str
    ) -> Tuple[bool, bool, Optional[int]]:
        """Detect trend and seasonality in the data."""

        values = df[target_column].dropna().values

        if len(values) < 10:
            return False, False, None

        # Trend detection: simple linear regression slope
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        has_trend = abs(slope) > (np.std(values) * 0.01)

        # Seasonality detection using autocorrelation
        has_seasonality = False
        seasonality_period = None

        expected_period = {
            'daily': 7,    # Weekly seasonality
            'weekly': 52,  # Yearly seasonality
            'monthly': 12  # Yearly seasonality
        }.get(frequency)

        if expected_period and len(values) > expected_period * 2:
            # Calculate autocorrelation at expected lag
            mean = np.mean(values)
            var = np.var(values)

            if var > 0:
                shifted = np.roll(values, expected_period)
                autocorr = np.mean((values - mean) * (shifted - mean)) / var

                if autocorr > 0.3:  # Significant autocorrelation
                    has_seasonality = True
                    seasonality_period = expected_period

        return has_trend, has_seasonality, seasonality_period

    def _find_covariate_columns(
        self, df: pd.DataFrame, date_column: str, target_column: str
    ) -> List[str]:
        """Find potential covariate columns."""

        covariates = []

        for col in df.columns:
            # Skip date and target
            if col in [date_column, target_column]:
                continue

            # Check if it's a valid covariate (numeric or binary)
            if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                unique_vals = df[col].nunique()

                # Binary indicators (0/1) or low-cardinality
                if unique_vals <= 20:
                    covariates.append(col)
                # Or if it's a continuous variable that's not mostly NaN
                elif df[col].notna().sum() / len(df) > 0.5:
                    covariates.append(col)

        return covariates

    def _calculate_quality_score(
        self, df: pd.DataFrame, target_column: str,
        missing_values: int, missing_periods: int, outlier_count: int
    ) -> float:
        """Calculate overall data quality score (0-100)."""

        total_rows = len(df)

        # Start at 100, deduct for issues
        score = 100.0

        # Missing values penalty (up to 30 points)
        missing_pct = missing_values / total_rows
        score -= min(missing_pct * 100, 30)

        # Missing periods penalty (up to 20 points)
        if total_rows > 0:
            missing_period_pct = missing_periods / total_rows
            score -= min(missing_period_pct * 100, 20)

        # Outlier penalty (up to 10 points)
        outlier_pct = outlier_count / total_rows if total_rows > 0 else 0
        score -= min(outlier_pct * 50, 10)

        # Bonus for having enough data
        if total_rows >= 104:  # 2 years weekly
            score = min(score + 10, 100)
        elif total_rows >= 52:  # 1 year weekly
            score = min(score + 5, 100)

        return round(max(score, 0), 1)

    def _generate_warnings(
        self, history_months: float, holiday_coverage: float,
        missing_values: int, missing_periods: int,
        quality_score: float, frequency: str
    ) -> List[Warning]:
        """Generate user-friendly warnings about data issues."""

        warnings = []

        # History length warnings
        if history_months < 12:
            warnings.append(Warning(
                level="high",
                message=f"Only {history_months:.0f} months of data. Cannot learn yearly patterns.",
                recommendation="Provide at least 12 months of historical data for accurate forecasts."
            ))
        elif history_months < 24:
            warnings.append(Warning(
                level="medium",
                message=f"Only {history_months:.0f} months of data. Holiday forecasts may be less accurate.",
                recommendation="Provide 2+ years of data for best holiday accuracy."
            ))

        # Holiday coverage warnings
        if holiday_coverage < 50:
            warnings.append(Warning(
                level="medium",
                message="Limited holiday data coverage.",
                recommendation="Include data spanning multiple Thanksgivings and Christmases for holiday accuracy."
            ))

        # Missing data warnings
        if missing_values > 0:
            warnings.append(Warning(
                level="low" if missing_values < 5 else "medium",
                message=f"{missing_values} missing values in target column.",
                recommendation="Missing values will be interpolated. Consider filling them manually for better accuracy."
            ))

        if missing_periods > 0:
            warnings.append(Warning(
                level="low" if missing_periods < 3 else "medium",
                message=f"{missing_periods} gaps in the time series.",
                recommendation="Ensure continuous data without gaps for best results."
            ))

        # Quality score warning
        if quality_score < 70:
            warnings.append(Warning(
                level="high",
                message=f"Data quality score is {quality_score}/100.",
                recommendation="Review data for issues before forecasting."
            ))

        return warnings

    def _recommend_horizon(self, frequency: str) -> int:
        """Recommend forecast horizon based on frequency."""
        defaults = {
            'daily': 30,    # 1 month ahead
            'weekly': 12,   # 3 months ahead (12 weeks)
            'monthly': 6,   # 6 months ahead
        }
        return defaults.get(frequency, 12)

    def _recommend_models(
        self, history_months: float, has_seasonality: bool, has_covariates: bool
    ) -> List[str]:
        """Recommend models based on data characteristics."""

        models = []

        # Prophet: Good default, handles holidays
        models.append('prophet')

        # XGBoost: Good with covariates and enough data
        if history_months >= 12 and has_covariates:
            models.append('xgboost')

        # ARIMA: Good for trend/seasonality
        if has_seasonality:
            models.append('arima')

        # ETS: Robust with limited data
        if history_months < 24:
            models.append('ets')

        return models

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute deterministic hash of input data for reproducibility."""

        # Sort columns for consistency
        df_sorted = df[sorted(df.columns)]

        # Convert to CSV string (deterministic)
        csv_string = df_sorted.to_csv(index=False)

        # Compute SHA256
        return hashlib.sha256(csv_string.encode()).hexdigest()[:16]
