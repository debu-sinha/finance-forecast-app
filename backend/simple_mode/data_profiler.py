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

from backend.utils.logging_utils import log_io

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
    total_periods: int  # Unique time periods in the data
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
    row_count: int  # Total rows in the file (may include duplicates from slices)

    # Recommendations
    recommended_horizon: int
    recommended_models: List[str]

    # Multi-slice data detection
    unique_periods: int = 0  # Number of unique time periods
    has_multiple_slices: bool = False  # True if duplicate dates exist (multiple segments/stores)
    slice_count: int = 1  # Estimated number of slices/segments

    # Future covariate rows (dates with covariates but no target value)
    future_rows_count: int = 0
    future_rows_date_range: Optional[Tuple[date, date]] = None
    has_future_covariates: bool = False
    future_covariates_valid: bool = True
    future_covariates_issues: List[str] = field(default_factory=list)

    # Data leakage detection
    leaky_covariates: List[str] = field(default_factory=list)
    correlation_details: List[Dict[str, Any]] = field(default_factory=list)
    safe_covariates: List[str] = field(default_factory=list)  # Covariates after removing leaky ones

    # Incomplete final period detection
    has_incomplete_final_period: bool = False
    incomplete_period_details: Dict[str, Any] = field(default_factory=dict)


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

    # Common target column names (ordered by specificity - most specific first)
    # Note: 'y' is a common ML convention but matches too broadly in partial matching
    TARGET_COLUMN_PATTERNS = [
        'revenue', 'sales', 'quantity', 'amount', 'volume', 'demand', 'orders',
        'target', 'actual', 'value', 'total', 'count', 'tot_sub', 'tot_vol',
        'subtotal', 'subscription'
    ]

    # Exact match only patterns (avoid partial matching issues)
    TARGET_COLUMN_EXACT = ['y', 'Y']

    # Columns to exclude from target detection
    EXCLUDE_FROM_TARGET = [
        'id', 'index', 'row', 'year', 'month', 'day', 'week', 'quarter',
        'day_of_week', 'is_weekend', 'is_holiday'
    ]

    # Patterns that indicate a column is likely a COVARIATE, not a target
    # These should be de-prioritized in target detection
    COVARIATE_COLUMN_PATTERNS = [
        'spend', 'cost', 'expense', 'rate', 'margin', 'percent', 'pct', 'ratio',
        'flag', 'indicator', 'is_', 'has_', 'avg_', 'average', 'mean', 'median',
        'fee', 'discount', 'price', 'budget', 'marketing'
    ]

    @log_io
    def profile(self, df: pd.DataFrame) -> DataProfile:
        """
        Analyze data and generate complete profile.

        Args:
            df: Input DataFrame

        Returns:
            DataProfile with all detected settings and recommendations
        """
        logger.info("=" * 70)
        logger.info("📊 DATA PROFILER - START")
        logger.info("=" * 70)
        logger.info(f"[INPUT] DataFrame shape: {df.shape}")
        logger.info(f"[INPUT] Columns: {list(df.columns)}")
        logger.info(f"[INPUT] Dtypes:\n{df.dtypes.to_string()}")
        logger.info(f"[INPUT] First 3 rows:\n{df.head(3).to_string()}")
        logger.info(f"[INPUT] Last 3 rows:\n{df.tail(3).to_string()}")
        logger.info("-" * 70)

        # Step 1: Detect date column
        logger.info("[STEP 1] Detecting date column...")
        date_column = self._detect_date_column(df)
        logger.info(f"[STEP 1 OUTPUT] date_column = '{date_column}'")
        logger.info(f"[STEP 1 OUTPUT] Sample values: {df[date_column].head(5).tolist()}")

        # Ensure date column is datetime
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
        logger.info(f"[STEP 1 OUTPUT] After datetime conversion - min: {df[date_column].min()}, max: {df[date_column].max()}")

        # Step 2: Detect frequency
        logger.info("[STEP 2] Detecting frequency...")
        frequency = self._detect_frequency(df, date_column)
        logger.info(f"[STEP 2 OUTPUT] frequency = '{frequency}'")

        # Step 3: Detect target column
        logger.info("[STEP 3] Detecting target column...")
        target_column = self._detect_target_column(df, date_column)
        logger.info(f"[STEP 3 OUTPUT] target_column = '{target_column}'")
        logger.info(f"[STEP 3 OUTPUT] Target stats - min: {df[target_column].min()}, max: {df[target_column].max()}, mean: {df[target_column].mean():.2f}")

        # Step 4: Analyze date range and detect multi-slice data
        logger.info("[STEP 4] Analyzing date range and multi-slice detection...")
        date_range = self._get_date_range(df, date_column)
        history_months = self._calculate_history_months(date_range)
        logger.info(f"[STEP 4 OUTPUT] date_range = {date_range}")
        logger.info(f"[STEP 4 OUTPUT] history_months = {history_months:.2f}")

        # Detect if data has multiple slices (duplicate dates)
        unique_dates = df[date_column].nunique()
        total_rows = len(df)
        has_multiple_slices = unique_dates < total_rows
        slice_count = total_rows // unique_dates if unique_dates > 0 else 1
        logger.info(f"[STEP 4 OUTPUT] unique_dates = {unique_dates}, total_rows = {total_rows}")
        logger.info(f"[STEP 4 OUTPUT] has_multiple_slices = {has_multiple_slices}, slice_count = {slice_count}")

        # Step 5: Find data quality issues
        logger.info("[STEP 5] Finding data quality issues...")
        missing_values = df[target_column].isna().sum()
        missing_periods = self._find_missing_periods(df, date_column, frequency)
        outliers = self._detect_outliers(df, target_column)
        logger.info(f"[STEP 5 OUTPUT] missing_values = {missing_values}")
        logger.info(f"[STEP 5 OUTPUT] missing_periods count = {len(missing_periods)}")
        if missing_periods:
            logger.info(f"[STEP 5 OUTPUT] missing_periods (first 10): {missing_periods[:10]}")
        logger.info(f"[STEP 5 OUTPUT] outliers count = {len(outliers)}")

        # Step 6: Analyze holiday coverage
        logger.info("[STEP 6] Analyzing holiday coverage...")
        holidays_in_data, holiday_coverage = self._analyze_holiday_coverage(
            df, date_column, frequency
        )
        logger.info(f"[STEP 6 OUTPUT] holidays_in_data = {holidays_in_data}")
        logger.info(f"[STEP 6 OUTPUT] holiday_coverage = {holiday_coverage}")

        # Step 7: Detect patterns
        logger.info("[STEP 7] Detecting patterns (trend, seasonality)...")
        has_trend, has_seasonality, seasonality_period = self._detect_patterns(
            df, target_column, frequency
        )
        logger.info(f"[STEP 7 OUTPUT] has_trend = {has_trend}, has_seasonality = {has_seasonality}, seasonality_period = {seasonality_period}")

        # Step 8: Find covariate columns
        logger.info("[STEP 8] Finding covariate columns...")
        covariate_columns = self._find_covariate_columns(
            df, date_column, target_column
        )
        logger.info(f"[STEP 8 OUTPUT] covariate_columns = {covariate_columns}")

        # Step 8b: Detect data leakage (highly correlated covariates)
        logger.info("[STEP 8b] Detecting data leakage (high correlation covariates)...")
        leaky_covariates, correlation_details = self._detect_leaky_covariates(
            df, target_column, covariate_columns
        )
        # Create safe covariates list by removing leaky ones
        safe_covariates = [c for c in covariate_columns if c not in leaky_covariates]
        logger.info(f"[STEP 8b OUTPUT] leaky_covariates = {leaky_covariates}")
        logger.info(f"[STEP 8b OUTPUT] safe_covariates = {safe_covariates}")

        # Step 9: Calculate quality score (use unique_dates for accurate calculation)
        logger.info("[STEP 9] Calculating quality score...")
        quality_score = self._calculate_quality_score(
            df, target_column, missing_values, len(missing_periods), len(outliers),
            unique_periods=unique_dates
        )
        logger.info(f"[STEP 9 OUTPUT] quality_score = {quality_score}")

        # Step 10: Generate warnings
        logger.info("[STEP 10] Generating warnings...")
        warnings = self._generate_warnings(
            history_months, holiday_coverage, missing_values,
            missing_periods, quality_score, frequency
        )
        logger.info(f"[STEP 10 OUTPUT] warnings count = {len(warnings)}")
        for w in warnings:
            logger.info(f"[STEP 10 OUTPUT] Warning: [{w.level}] {w.message}")

        # Add warning for multi-slice data
        if has_multiple_slices:
            warnings.append(Warning(
                level="medium",
                message=f"Data contains multiple segments (~{slice_count} slices with {unique_dates} unique dates each). Total rows: {total_rows}.",
                recommendation="Select a specific segment/slice to forecast, or use batch forecasting for all segments."
            ))
            logger.info(f"[STEP 10 OUTPUT] Added multi-slice warning")

        # Add HIGH PRIORITY warning for data leakage
        if leaky_covariates:
            leaky_list = ', '.join(leaky_covariates)
            warnings.insert(0, Warning(  # Insert at beginning for high visibility
                level="high",
                message=f"⚠️ DATA LEAKAGE DETECTED: Column(s) [{leaky_list}] have >90% correlation with target '{target_column}'. "
                        f"Using these as covariates will cause severe overfitting and poor forecast accuracy.",
                recommendation=f"UNSELECT these columns from covariates: {leaky_list}. "
                              f"They likely contain target-derived values that won't be available for future predictions."
            ))
            logger.warning(f"[STEP 10 OUTPUT] Added DATA LEAKAGE warning for: {leaky_covariates}")

        # Step 10b: Detect incomplete final period
        logger.info("[STEP 10b] Detecting incomplete final period...")
        has_incomplete_final, incomplete_details = self._detect_incomplete_final_period(
            df, date_column, target_column, frequency, has_multiple_slices
        )
        logger.info(f"[STEP 10b OUTPUT] has_incomplete_final_period = {has_incomplete_final}")
        if has_incomplete_final:
            logger.info(f"[STEP 10b OUTPUT] incomplete_details = {incomplete_details}")

        # Add HIGH PRIORITY warning for incomplete final period
        if has_incomplete_final:
            warnings.insert(0, Warning(  # Insert at beginning for high visibility
                level="high",
                message=f"🚨 INCOMPLETE FINAL PERIOD DETECTED: Last period ({incomplete_details.get('last_date', 'N/A')}) "
                        f"has value {incomplete_details.get('last_value', 0):,.0f} which is "
                        f"{incomplete_details.get('drop_pct', 0):.0f}% lower than the median ({incomplete_details.get('median_value', 0):,.0f}). "
                        f"This appears to be partial/incomplete data.",
                recommendation="EXCLUDE this final period from training data, or wait until the period is complete. "
                              "Training on incomplete data will cause models to learn incorrect patterns."
            ))
            logger.warning(f"[STEP 10b OUTPUT] Added INCOMPLETE FINAL PERIOD warning")

        # Step 11: Generate recommendations
        logger.info("[STEP 11] Generating recommendations...")
        recommended_horizon = self._recommend_horizon(frequency)
        recommended_models = self._recommend_models(
            history_months, has_seasonality, len(covariate_columns) > 0
        )
        logger.info(f"[STEP 11 OUTPUT] recommended_horizon = {recommended_horizon}")
        logger.info(f"[STEP 11 OUTPUT] recommended_models = {recommended_models}")

        # Step 12: Compute data hash for reproducibility
        logger.info("[STEP 12] Computing data hash...")
        data_hash = self._compute_data_hash(df)
        logger.info(f"[STEP 12 OUTPUT] data_hash = {data_hash}")

        # Step 13: Detect and validate future covariate rows
        logger.info("[STEP 13] Detecting future covariate rows...")
        future_rows_count, future_rows_date_range, has_future_covariates, future_valid, future_issues = self._detect_future_rows(
            df, date_column, target_column, covariate_columns
        )
        logger.info(f"[STEP 13 OUTPUT] future_rows_count = {future_rows_count}")
        logger.info(f"[STEP 13 OUTPUT] has_future_covariates = {has_future_covariates}")
        logger.info(f"[STEP 13 OUTPUT] future_valid = {future_valid}")

        # Add warning if future covariates detected
        if has_future_covariates:
            if future_valid:
                warnings.append(Warning(
                    level="low",
                    message=f"Detected {future_rows_count} rows with future covariate values (dates with predictors but no actuals).",
                    recommendation="These will be used for more accurate forecasting using your planned/known future values."
                ))
            else:
                warnings.append(Warning(
                    level="high",
                    message=f"Future covariate data has validation issues: {'; '.join([i for i in future_issues if not i.startswith('Warning')])}",
                    recommendation="Please fix the issues in your future covariate rows before forecasting."
                ))

            # Add any warnings from validation
            for issue in future_issues:
                if issue.startswith("Warning:"):
                    warnings.append(Warning(
                        level="medium",
                        message=issue.replace("Warning: ", ""),
                        recommendation="Review your future covariate data."
                    ))

        # Final output summary
        logger.info("=" * 70)
        logger.info("📊 DATA PROFILER - FINAL OUTPUT SUMMARY")
        logger.info("=" * 70)
        logger.info(f"[OUTPUT] frequency: {frequency}")
        logger.info(f"[OUTPUT] date_column: {date_column}")
        logger.info(f"[OUTPUT] target_column: {target_column}")
        logger.info(f"[OUTPUT] date_range: {date_range}")
        logger.info(f"[OUTPUT] total_periods (unique): {unique_dates}")
        logger.info(f"[OUTPUT] row_count (total): {total_rows}")
        logger.info(f"[OUTPUT] history_months: {history_months:.2f}")
        logger.info(f"[OUTPUT] data_quality_score: {quality_score}")
        logger.info(f"[OUTPUT] holiday_coverage_score: {holiday_coverage}")
        logger.info(f"[OUTPUT] has_trend: {has_trend}, has_seasonality: {has_seasonality}")
        logger.info(f"[OUTPUT] covariate_columns: {covariate_columns}")
        logger.info(f"[OUTPUT] leaky_covariates: {leaky_covariates}")
        logger.info(f"[OUTPUT] safe_covariates: {safe_covariates}")
        logger.info(f"[OUTPUT] recommended_horizon: {recommended_horizon}")
        logger.info(f"[OUTPUT] recommended_models: {recommended_models}")
        logger.info(f"[OUTPUT] has_multiple_slices: {has_multiple_slices}, slice_count: {slice_count}")
        logger.info(f"[OUTPUT] warnings count: {len(warnings)}")
        logger.info(f"[OUTPUT] data_hash: {data_hash}")
        logger.info("=" * 70)

        return DataProfile(
            # Auto-detected config
            frequency=frequency,
            date_column=date_column,
            target_column=target_column,

            # Date range
            date_range=date_range,
            total_periods=int(unique_dates),  # Unique time periods, not total rows
            history_months=float(history_months),

            # Quality metrics - convert numpy types to native Python types
            missing_values=int(missing_values),
            missing_periods=missing_periods,
            outliers=outliers,
            data_quality_score=float(quality_score),

            # Holiday coverage
            holidays_in_data=holidays_in_data,
            holiday_coverage_score=float(holiday_coverage),

            # Patterns - ensure native Python bool
            has_trend=bool(has_trend),
            has_seasonality=bool(has_seasonality),
            seasonality_period=int(seasonality_period) if seasonality_period else None,

            # Covariates
            covariate_columns=covariate_columns,

            # Warnings
            warnings=warnings,

            # Reproducibility
            data_hash=data_hash,
            row_count=int(total_rows),  # Total rows in the file

            # Recommendations
            recommended_horizon=int(recommended_horizon),
            recommended_models=recommended_models,

            # Multi-slice detection
            unique_periods=int(unique_dates),
            has_multiple_slices=bool(has_multiple_slices),
            slice_count=int(slice_count),

            # Future covariate rows
            future_rows_count=future_rows_count,
            future_rows_date_range=future_rows_date_range,
            has_future_covariates=has_future_covariates,
            future_covariates_valid=future_valid,
            future_covariates_issues=future_issues,

            # Data leakage detection
            leaky_covariates=leaky_covariates,
            correlation_details=correlation_details,
            safe_covariates=safe_covariates,

            # Incomplete final period detection
            has_incomplete_final_period=has_incomplete_final,
            incomplete_period_details=incomplete_details,
        )

    @log_io
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

    @log_io
    def _detect_frequency(self, df: pd.DataFrame, date_column: str) -> str:
        """Detect if data is daily, weekly, or monthly."""

        dates = pd.to_datetime(df[date_column]).sort_values()

        # Use UNIQUE dates to handle multi-slice data correctly
        unique_dates = dates.drop_duplicates().sort_values()

        if len(unique_dates) < 2:
            return "daily"  # Default

        # Calculate median gap between consecutive UNIQUE dates
        gaps = unique_dates.diff().dropna()
        median_gap_days = gaps.median().days

        logger.info(f"Frequency detection: {len(unique_dates)} unique dates, median gap = {median_gap_days} days")

        if median_gap_days <= 1:
            return "daily"
        elif median_gap_days <= 8:
            return "weekly"
        elif median_gap_days <= 32:
            return "monthly"
        else:
            return "monthly"  # Default for longer gaps

    @log_io
    def _detect_target_column(self, df: pd.DataFrame, date_column: str) -> str:
        """
        Detect the target (value) column.

        Priority order:
        1. Exact match with known target patterns (e.g., 'y', 'revenue')
        2. Partial match with target patterns, excluding covariate-like columns
        3. Highest variance column that isn't covariate-like
        4. Highest variance column as fallback
        """

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

        def is_covariate_like(col_name: str) -> bool:
            """Check if column name suggests it's a covariate rather than target."""
            col_lower = col_name.lower()
            return any(pattern in col_lower for pattern in self.COVARIATE_COLUMN_PATTERNS)

        # Separate columns into likely targets vs likely covariates
        likely_targets = [c for c in numeric_cols if not is_covariate_like(c)]
        likely_covariates = [c for c in numeric_cols if is_covariate_like(c)]

        logger.info(f"[TARGET DETECTION] likely_targets: {likely_targets}")
        logger.info(f"[TARGET DETECTION] likely_covariates: {likely_covariates}")

        # Prioritize likely_targets in pattern matching
        cols_to_check = likely_targets + likely_covariates  # Check targets first

        # Step 1: Try exact matches with short patterns (like 'y')
        for col in cols_to_check:
            if col in self.TARGET_COLUMN_EXACT:
                logger.info(f"[TARGET DETECTION] Exact match found: {col}")
                return col

        # Step 2: Try exact matches with common names (case-insensitive)
        # Prioritize columns that aren't covariate-like
        for col in cols_to_check:
            if col.lower() in self.TARGET_COLUMN_PATTERNS:
                logger.info(f"[TARGET DETECTION] Pattern exact match found: {col}")
                return col

        # Step 3: Try partial matches with longer patterns
        # Only match if pattern is a substantial part of the column name
        # First check likely_targets, then likely_covariates
        for col in cols_to_check:
            col_lower = col.lower()
            for pattern in self.TARGET_COLUMN_PATTERNS:
                # Require pattern to be at least 3 chars to avoid false positives
                if len(pattern) >= 3 and pattern in col_lower:
                    logger.info(f"[TARGET DETECTION] Pattern partial match found: {col} (pattern: {pattern})")
                    return col

        # Step 4: Pick the column with highest variance
        # Prefer columns that aren't covariate-like
        if likely_targets:
            variances = {col: df[col].var() for col in likely_targets}
            best_target = max(variances, key=variances.get)
            logger.info(f"[TARGET DETECTION] Highest variance among likely targets: {best_target}")
            return best_target

        # Step 5: Fallback - highest variance among all numeric columns
        variances = {col: df[col].var() for col in numeric_cols}
        best_fallback = max(variances, key=variances.get)
        logger.info(f"[TARGET DETECTION] Fallback to highest variance: {best_fallback}")
        return best_fallback

    @log_io
    def _get_date_range(
        self, df: pd.DataFrame, date_column: str
    ) -> Tuple[date, date]:
        """Get the date range of the data."""
        dates = pd.to_datetime(df[date_column])
        return (dates.min().date(), dates.max().date())

    @log_io
    def _calculate_history_months(self, date_range: Tuple[date, date]) -> float:
        """Calculate how many months of history we have."""
        start, end = date_range
        days = (end - start).days
        return round(days / 30.44, 1)  # Average days per month

    @log_io
    def _find_missing_periods(
        self, df: pd.DataFrame, date_column: str, frequency: str
    ) -> List[date]:
        """Find gaps in the time series."""

        dates = pd.to_datetime(df[date_column]).sort_values()
        unique_dates = dates.drop_duplicates()

        if len(unique_dates) < 2:
            return []

        # Determine the appropriate frequency code
        if frequency == 'daily':
            freq_code = 'D'
        elif frequency == 'weekly':
            # Detect which day of week the data uses (e.g., Monday, Sunday)
            # by looking at the most common day of week in the data
            day_of_week = unique_dates.dt.dayofweek.mode()
            if len(day_of_week) > 0:
                dow = day_of_week.iloc[0]
                # Map to pandas weekly frequency anchored to that day
                # 0=Monday -> W-MON, 1=Tuesday -> W-TUE, etc.
                dow_map = {0: 'W-MON', 1: 'W-TUE', 2: 'W-WED', 3: 'W-THU',
                          4: 'W-FRI', 5: 'W-SAT', 6: 'W-SUN'}
                freq_code = dow_map.get(dow, 'W-MON')
                logger.info(f"Detected weekly data on day {dow} ({freq_code})")
            else:
                freq_code = 'W-MON'  # Default to Monday
        elif frequency == 'monthly':
            freq_code = 'MS'
        else:
            freq_code = 'D'

        # Generate expected date range
        try:
            expected = pd.date_range(
                start=unique_dates.min(),
                end=unique_dates.max(),
                freq=freq_code
            )
        except Exception as e:
            logger.warning(f"Failed to generate expected date range with freq={freq_code}: {e}")
            return []

        # Find missing dates by comparing unique actual dates to expected
        actual_dates_set = set(unique_dates.dt.normalize())
        expected_dates_set = set(expected.normalize())

        missing = expected_dates_set - actual_dates_set
        # Return all missing dates (sorted), we'll limit display elsewhere
        return sorted([d.date() for d in missing])

    @log_io
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

    @log_io
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

    @log_io
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
        has_trend = bool(abs(slope) > (np.std(values) * 0.01))

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
                    seasonality_period = int(expected_period)

        return bool(has_trend), bool(has_seasonality), seasonality_period

    @log_io
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

    @log_io
    def _detect_leaky_covariates(
        self, df: pd.DataFrame, target_column: str, covariate_columns: List[str],
        correlation_threshold: float = 0.90
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Detect covariates that are highly correlated with target (potential data leakage).

        Data leakage occurs when covariates contain information about the target that
        wouldn't be available at prediction time. This often manifests as very high
        correlation (>0.9) between a covariate and the target.

        Args:
            df: DataFrame with data
            target_column: Name of target column
            covariate_columns: List of covariate column names
            correlation_threshold: Correlation above which to flag as leaky (default 0.90)

        Returns:
            Tuple of (list of leaky columns, list of correlation details)
        """
        leaky_columns = []
        correlation_details = []

        target_values = df[target_column].dropna()
        if len(target_values) < 10:
            return [], []

        for cov in covariate_columns:
            if cov not in df.columns:
                continue

            try:
                cov_values = df[cov].dropna()
                if len(cov_values) < 10:
                    continue

                # Align indices for correlation
                common_idx = target_values.index.intersection(cov_values.index)
                if len(common_idx) < 10:
                    continue

                correlation = target_values.loc[common_idx].corr(cov_values.loc[common_idx])

                if pd.isna(correlation):
                    continue

                abs_corr = abs(correlation)

                if abs_corr >= correlation_threshold:
                    leaky_columns.append(cov)
                    correlation_details.append({
                        'column': cov,
                        'correlation': round(correlation, 4),
                        'abs_correlation': round(abs_corr, 4),
                        'is_leaky': True,
                        'reason': f"Very high correlation ({abs_corr:.1%}) with target suggests data leakage"
                    })
                    logger.warning(
                        f"⚠️ POTENTIAL DATA LEAKAGE: '{cov}' has {abs_corr:.1%} correlation "
                        f"with target '{target_column}'. This may cause poor forecast accuracy."
                    )
                elif abs_corr >= 0.7:
                    # High but not extreme - just log for awareness
                    correlation_details.append({
                        'column': cov,
                        'correlation': round(correlation, 4),
                        'abs_correlation': round(abs_corr, 4),
                        'is_leaky': False,
                        'reason': f"High correlation ({abs_corr:.1%}) - monitor for potential issues"
                    })
                    logger.info(f"High correlation detected: '{cov}' has {abs_corr:.1%} correlation with target")

            except Exception as e:
                logger.warning(f"Could not compute correlation for '{cov}': {e}")

        return leaky_columns, correlation_details

    @log_io
    def _calculate_quality_score(
        self, df: pd.DataFrame, target_column: str,
        missing_values: int, missing_periods: int, outlier_count: int,
        unique_periods: int = None
    ) -> float:
        """Calculate overall data quality score (0-100).

        Args:
            df: DataFrame
            target_column: Name of target column
            missing_values: Number of missing values in target
            missing_periods: Number of missing time periods
            outlier_count: Number of outliers detected
            unique_periods: Number of unique time periods (for multi-slice data)
        """
        total_rows = len(df)

        # Use unique_periods if provided (for multi-slice data), otherwise use total_rows
        actual_periods = unique_periods if unique_periods is not None else total_rows

        # Expected total periods = actual unique periods + missing periods
        expected_periods = actual_periods + missing_periods

        # Start at 100, deduct for issues
        score = 100.0

        # Missing values penalty (up to 30 points)
        # For multi-slice data, divide by total_rows (since missing values spans all slices)
        if total_rows > 0:
            missing_pct = missing_values / total_rows
            score -= min(missing_pct * 100, 30)

        # Missing periods penalty (up to 30 points) - based on expected vs actual unique periods
        # This is a significant data quality issue
        if expected_periods > 0:
            missing_period_pct = missing_periods / expected_periods
            # Apply heavier penalty: 5 missing periods = ~10 points, 10+ = 20+ points
            score -= min(missing_period_pct * 150, 30)

        # Outlier penalty (up to 10 points)
        outlier_pct = outlier_count / total_rows if total_rows > 0 else 0
        score -= min(outlier_pct * 50, 10)

        # Bonus for having enough unique time periods (up to 10 points)
        if actual_periods >= 104:  # 2 years weekly
            score = min(score + 10, 100)
        elif actual_periods >= 52:  # 1 year weekly
            score = min(score + 5, 100)

        return round(max(score, 0), 1)

    @log_io
    def _generate_warnings(
        self, history_months: float, holiday_coverage: float,
        missing_values: int, missing_periods_list: List[date],
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

        # Missing periods warning - NOW WITH ACTUAL DATES
        missing_count = len(missing_periods_list)
        if missing_count > 0:
            # Determine severity based on percentage of missing data
            level = "low"
            if missing_count >= 10:
                level = "high"
            elif missing_count >= 5:
                level = "medium"

            # Show actual missing dates (up to 10)
            if missing_count <= 10:
                dates_str = ", ".join([str(d) for d in missing_periods_list])
                message = f"{missing_count} missing date(s) in the time series: {dates_str}"
            else:
                # Show first 10 and indicate more
                dates_str = ", ".join([str(d) for d in missing_periods_list[:10]])
                message = f"{missing_count} missing dates in the time series. First 10: {dates_str} (and {missing_count - 10} more)"

            warnings.append(Warning(
                level=level,
                message=message,
                recommendation="These dates are missing from your data. Consider adding rows for these dates, or the model will interpolate them."
            ))

        # Quality score warning
        if quality_score < 70:
            warnings.append(Warning(
                level="high",
                message=f"Data quality score is {quality_score}/100.",
                recommendation="Review data for issues before forecasting."
            ))

        return warnings

    @log_io
    def _recommend_horizon(self, frequency: str) -> int:
        """Recommend forecast horizon based on frequency."""
        defaults = {
            'daily': 30,    # 1 month ahead
            'weekly': 12,   # 3 months ahead (12 weeks)
            'monthly': 6,   # 6 months ahead
        }
        return defaults.get(frequency, 12)

    @log_io
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

    @log_io
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute deterministic hash of input data for reproducibility."""

        # Sort columns for consistency
        df_sorted = df[sorted(df.columns)]

        # Convert to CSV string (deterministic)
        csv_string = df_sorted.to_csv(index=False)

        # Compute SHA256
        return hashlib.sha256(csv_string.encode()).hexdigest()[:16]

    @log_io
    def _detect_future_rows(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        covariate_columns: List[str]
    ) -> Tuple[int, Optional[Tuple[date, date]], bool, bool, List[str]]:
        """
        Detect and validate rows that have future dates with covariates but no target values.

        This is common in forecasting when users know future values of predictors
        (e.g., planned promotions, scheduled events) but not the actual target.

        Args:
            df: DataFrame with data
            date_column: Name of date column
            target_column: Name of target column
            covariate_columns: List of covariate column names

        Returns:
            Tuple of (count, date_range, has_future_covariates, is_valid, issues_list)
        """
        issues = []

        if not covariate_columns:
            return 0, None, False, True, []

        # Find rows where target is NaN/null but at least one covariate has a value
        target_missing = df[target_column].isna() | df[target_column].isnull()

        # Check if any covariate has non-null value in these rows
        has_covariate_value = pd.Series(False, index=df.index)
        for cov in covariate_columns:
            if cov in df.columns:
                has_covariate_value |= df[cov].notna()

        # Future rows: target is missing BUT covariates have values
        future_mask = target_missing & has_covariate_value

        future_rows = df[future_mask]
        count = len(future_rows)

        if count == 0:
            return 0, None, False, True, []

        # ============================================================
        # VALIDATION CHECKS
        # ============================================================
        is_valid = True

        # Parse dates for validation
        try:
            df_dates = pd.to_datetime(df[date_column])
            future_dates = pd.to_datetime(future_rows[date_column])
            historical_dates = df_dates[~future_mask]

            min_future_date = future_dates.min()
            max_future_date = future_dates.max()
            max_historical_date = historical_dates.max() if len(historical_dates) > 0 else None

            date_range = (min_future_date.date(), max_future_date.date())

        except Exception as e:
            issues.append(f"Could not parse dates in future rows: {str(e)}")
            return count, None, True, False, issues

        # VALIDATION 1: Future rows should come AFTER historical data
        if max_historical_date and min_future_date < max_historical_date:
            # Check if there are interspersed rows
            interspersed_count = (future_dates < max_historical_date).sum()
            if interspersed_count > 0:
                issues.append(
                    f"Found {interspersed_count} future covariate rows with dates BEFORE the last historical date "
                    f"({max_historical_date.date()}). Future rows should come after historical data."
                )
                is_valid = False

        # VALIDATION 2: Future rows should have valid (parseable) dates
        invalid_dates = future_dates.isna().sum()
        if invalid_dates > 0:
            issues.append(f"Found {invalid_dates} future rows with invalid/unparseable dates.")
            is_valid = False

        # VALIDATION 3: Check for gaps in future dates (warn, not error)
        if len(future_dates) > 1:
            sorted_dates = future_dates.sort_values()
            date_diffs = sorted_dates.diff().dropna()
            if len(date_diffs) > 0:
                median_diff = date_diffs.median()
                large_gaps = date_diffs[date_diffs > median_diff * 3]
                if len(large_gaps) > 0:
                    issues.append(
                        f"Warning: Found {len(large_gaps)} large gaps in future dates. "
                        "This may indicate missing future periods."
                    )
                    # This is a warning, not an error - still valid

        # VALIDATION 4: Check covariate completeness in future rows
        for cov in covariate_columns:
            if cov in future_rows.columns:
                null_count = future_rows[cov].isna().sum()
                if null_count > 0 and null_count < count:
                    issues.append(
                        f"Warning: Covariate '{cov}' has {null_count} missing values in future rows. "
                        "Missing values will be imputed with historical mean."
                    )

        # VALIDATION 5: Check that future covariates have reasonable values
        for cov in covariate_columns:
            if cov in future_rows.columns and cov in df.columns:
                future_vals = future_rows[cov].dropna()
                hist_vals = df[~future_mask][cov].dropna()

                if len(future_vals) > 0 and len(hist_vals) > 0:
                    hist_min, hist_max = hist_vals.min(), hist_vals.max()
                    hist_mean, hist_std = hist_vals.mean(), hist_vals.std()

                    # Check for values far outside historical range
                    if hist_std > 0:
                        extreme_low = future_vals < (hist_mean - 5 * hist_std)
                        extreme_high = future_vals > (hist_mean + 5 * hist_std)
                        extreme_count = extreme_low.sum() + extreme_high.sum()

                        if extreme_count > 0:
                            issues.append(
                                f"Warning: Covariate '{cov}' has {extreme_count} extreme values in future rows "
                                f"(>5 std from historical mean). Verify these are intentional."
                            )

        # VALIDATION 6: Check horizon alignment
        if max_historical_date:
            days_into_future = (max_future_date - max_historical_date).days
            if days_into_future > 365 * 2:  # More than 2 years
                issues.append(
                    f"Warning: Future covariates extend {days_into_future} days ({days_into_future // 365} years) "
                    "beyond historical data. Long-range forecasts have higher uncertainty."
                )

        logger.info(f"Detected {count} future covariate rows with date range {date_range}")
        if issues:
            logger.info(f"Future covariate validation issues: {issues}")

        return count, date_range, True, is_valid, issues

    @log_io
    def _detect_incomplete_final_period(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        frequency: str,
        has_multiple_slices: bool
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if the final period in the data appears incomplete.

        This is critical for forecasting accuracy - incomplete final periods
        (e.g., partial week data) can severely skew model training and cause
        negative or unrealistic forecasts.

        Detection criteria:
        1. Final period target value is >50% below median (aggregated data)
        2. Final period has significantly fewer rows than median (multi-slice data)
        3. Sudden drop from previous period

        Args:
            df: DataFrame with data
            date_column: Name of date column
            target_column: Name of target column
            frequency: Data frequency (daily, weekly, monthly)
            has_multiple_slices: Whether data has multiple slices per period

        Returns:
            Tuple of (is_incomplete, details_dict)
        """
        details = {}

        try:
            # Get unique dates sorted
            df_dates = pd.to_datetime(df[date_column])
            unique_dates = df_dates.drop_duplicates().sort_values()

            if len(unique_dates) < 5:
                return False, {}

            last_date = unique_dates.iloc[-1]
            second_last_date = unique_dates.iloc[-2]

            details['last_date'] = str(last_date.date())
            details['second_last_date'] = str(second_last_date.date())

            # For multi-slice data, aggregate by date first
            if has_multiple_slices:
                agg_df = df.groupby(df_dates.dt.normalize())[target_column].agg(['sum', 'count']).reset_index()
                agg_df.columns = ['date', 'total_value', 'row_count']
                agg_df = agg_df.sort_values('date')

                # Get values
                last_row = agg_df.iloc[-1]
                last_value = last_row['total_value']
                last_row_count = last_row['row_count']

                # Calculate statistics excluding last period
                history = agg_df.iloc[:-1]
                median_value = history['total_value'].median()
                median_row_count = history['row_count'].median()
                prev_value = history.iloc[-1]['total_value'] if len(history) > 0 else median_value

                details['last_value'] = float(last_value)
                details['median_value'] = float(median_value)
                details['prev_value'] = float(prev_value)
                details['last_row_count'] = int(last_row_count)
                details['median_row_count'] = int(median_row_count)

                # Detection 1: Value significantly below median
                if median_value > 0:
                    drop_pct = (1 - last_value / median_value) * 100
                    details['drop_pct'] = float(drop_pct)

                    if drop_pct > 50:  # More than 50% below median
                        details['reason'] = f"Final period value is {drop_pct:.0f}% below median"
                        return True, details

                # Detection 2: Row count significantly below median (for multi-slice)
                if median_row_count > 0:
                    row_drop_pct = (1 - last_row_count / median_row_count) * 100
                    details['row_drop_pct'] = float(row_drop_pct)

                    if row_drop_pct > 30:  # More than 30% fewer rows
                        details['reason'] = f"Final period has {row_drop_pct:.0f}% fewer rows than median"
                        return True, details

                # Detection 3: Sudden drop from previous period
                if prev_value > 0:
                    sudden_drop_pct = (1 - last_value / prev_value) * 100
                    details['sudden_drop_pct'] = float(sudden_drop_pct)

                    if sudden_drop_pct > 70:  # More than 70% drop from previous
                        details['reason'] = f"Final period dropped {sudden_drop_pct:.0f}% from previous period"
                        return True, details

            else:
                # Single slice data - simpler analysis
                values = df.set_index(df_dates)[target_column].sort_index()
                last_value = values.iloc[-1]
                prev_values = values.iloc[:-1]

                median_value = prev_values.median()
                prev_value = prev_values.iloc[-1] if len(prev_values) > 0 else median_value

                details['last_value'] = float(last_value)
                details['median_value'] = float(median_value)
                details['prev_value'] = float(prev_value)

                # Detection: Value significantly below median
                if median_value > 0:
                    drop_pct = (1 - last_value / median_value) * 100
                    details['drop_pct'] = float(drop_pct)

                    if drop_pct > 50:
                        details['reason'] = f"Final period value is {drop_pct:.0f}% below median"
                        return True, details

                # Detection: Sudden drop from previous
                if prev_value > 0:
                    sudden_drop_pct = (1 - last_value / prev_value) * 100
                    details['sudden_drop_pct'] = float(sudden_drop_pct)

                    if sudden_drop_pct > 70:
                        details['reason'] = f"Final period dropped {sudden_drop_pct:.0f}% from previous period"
                        return True, details

            return False, details

        except Exception as e:
            logger.warning(f"Error detecting incomplete final period: {e}")
            return False, {'error': str(e)}
