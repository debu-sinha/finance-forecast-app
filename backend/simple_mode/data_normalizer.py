"""
Data Normalizer - Robust date and column data standardization.

Handles various date formats and data representations consistently,
with optional LLM assistance for ambiguous cases.
"""

import re
import logging
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from dateutil import parser as dateutil_parser
from dateutil.parser import ParserError
import pandas as pd

from backend.utils.logging_utils import log_io

logger = logging.getLogger(__name__)


class DateNormalizer:
    """
    Robust date parser that handles multiple formats.

    Priority order:
    1. Explicit format patterns (fastest, most reliable)
    2. dateutil parser (handles many formats automatically)
    3. Heuristic detection for ambiguous formats
    """

    # Common date patterns with their strptime formats
    # Order matters - more specific patterns first
    KNOWN_PATTERNS = [
        # ISO formats (unambiguous)
        (r'^\d{4}-\d{2}-\d{2}T', '%Y-%m-%dT%H:%M:%S'),  # ISO with time
        (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),           # ISO date
        (r'^\d{4}/\d{2}/\d{2}$', '%Y/%m/%d'),           # YYYY/MM/DD

        # US formats (MM/DD/YYYY, MM-DD-YYYY)
        (r'^\d{1,2}/\d{1,2}/\d{4}$', '%m/%d/%Y'),
        (r'^\d{1,2}-\d{1,2}-\d{4}$', '%m-%d-%Y'),

        # European formats (DD/MM/YYYY, DD-MM-YYYY) - handled by heuristics

        # 2-digit year formats - ASSUME 2000s
        (r'^\d{1,2}/\d{1,2}/\d{2}$', 'MM/DD/YY'),       # Special handling
        (r'^\d{1,2}-\d{1,2}-\d{2}$', 'MM-DD-YY'),       # Special handling

        # Written formats
        (r'^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$', '%B %d, %Y'),  # January 15, 2024
        (r'^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$', '%d %B %Y'),     # 15 January 2024

        # Week/Quarter formats (need special handling)
        (r'^W\d{1,2}\s+\d{4}$', 'WEEK'),               # W52 2024
        (r'^Q[1-4]\s+\d{4}$', 'QUARTER'),              # Q1 2024
        (r'^\d{4}\s*W\d{1,2}$', 'YEAR_WEEK'),          # 2024 W52
    ]

    def __init__(self, dayfirst: Optional[bool] = None):
        """
        Initialize normalizer.

        Args:
            dayfirst: If True, prefer DD/MM/YYYY. If False, prefer MM/DD/YYYY.
                     If None, auto-detect from data.
        """
        self.dayfirst = dayfirst
        self._detected_format = None

    @log_io
    def parse(self, date_str: str) -> Optional[datetime]:
        """
        Parse a date string to datetime.

        Args:
            date_str: Date string in any supported format

        Returns:
            datetime object or None if parsing fails
        """
        if not date_str or pd.isna(date_str):
            return None

        date_str = str(date_str).strip()

        # Try explicit patterns first (fastest)
        for pattern, fmt in self.KNOWN_PATTERNS:
            if re.match(pattern, date_str):
                try:
                    return self._parse_with_format(date_str, fmt)
                except ValueError:
                    continue

        # Fall back to dateutil parser
        try:
            return dateutil_parser.parse(
                date_str,
                dayfirst=self.dayfirst if self.dayfirst is not None else False
            )
        except (ParserError, ValueError, OverflowError):
            pass

        # Last resort: try common transformations
        return self._fuzzy_parse(date_str)

    @log_io
    def _parse_with_format(self, date_str: str, fmt: str) -> datetime:
        """Parse with specific format, handling special cases."""

        # Handle 2-digit year formats
        if fmt in ('MM/DD/YY', 'MM-DD-YY'):
            sep = '/' if '/' in date_str else '-'
            parts = date_str.split(sep)
            month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
            # Assume 2000s for 2-digit years
            year = 2000 + year if year < 100 else year
            return datetime(year, month, day)

        # Handle week formats
        if fmt == 'WEEK':
            match = re.match(r'^W(\d{1,2})\s+(\d{4})$', date_str)
            if match:
                week, year = int(match.group(1)), int(match.group(2))
                # Return Monday of that week
                return datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w')

        if fmt == 'YEAR_WEEK':
            match = re.match(r'^(\d{4})\s*W(\d{1,2})$', date_str)
            if match:
                year, week = int(match.group(1)), int(match.group(2))
                return datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w')

        # Handle quarter formats
        if fmt == 'QUARTER':
            match = re.match(r'^Q([1-4])\s+(\d{4})$', date_str)
            if match:
                quarter, year = int(match.group(1)), int(match.group(2))
                month = (quarter - 1) * 3 + 1  # Q1=Jan, Q2=Apr, etc.
                return datetime(year, month, 1)

        # Standard strptime
        return datetime.strptime(date_str, fmt)

    @log_io
    def _fuzzy_parse(self, date_str: str) -> Optional[datetime]:
        """Try fuzzy matching for unusual formats."""

        # Remove common noise
        cleaned = date_str.replace('  ', ' ').strip()

        # Try with different separators
        for old, new in [('.', '/'), ('_', '-'), (',', '')]:
            if old in cleaned:
                try:
                    return dateutil_parser.parse(cleaned.replace(old, new))
                except (ParserError, ValueError):
                    pass

        return None

    @log_io
    def detect_format_from_column(
        self,
        values: List[str],
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze a column to detect the predominant date format.

        Returns format info and whether dates are ambiguous.
        """
        sample = values[:sample_size] if len(values) > sample_size else values

        format_counts = {}
        ambiguous_count = 0
        parsed_dates = []

        for val in sample:
            if not val or pd.isna(val):
                continue

            val_str = str(val).strip()

            # Check for ambiguous formats (could be MM/DD or DD/MM)
            match = re.match(r'^(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})$', val_str)
            if match:
                first, second = int(match.group(1)), int(match.group(2))
                if first <= 12 and second <= 12:
                    ambiguous_count += 1

            # Try to parse and categorize
            parsed = self.parse(val_str)
            if parsed:
                parsed_dates.append(parsed)

                # Categorize format
                for pattern, fmt in self.KNOWN_PATTERNS:
                    if re.match(pattern, val_str):
                        format_counts[fmt] = format_counts.get(fmt, 0) + 1
                        break

        # Determine if ambiguous (could be US or European)
        is_ambiguous = ambiguous_count > len(sample) * 0.5

        # Detect if dayfirst based on impossible values (first number > 12 means day)
        def _first_num_over_12(v: Any) -> bool:
            match = re.match(r'^(\d{1,2})[/\-]', str(v))
            return match is not None and int(match.group(1)) > 12

        has_day_over_12 = any(
            _first_num_over_12(v)
            for v in sample if v and not pd.isna(v)
        )

        return {
            'detected_formats': format_counts,
            'is_ambiguous': is_ambiguous,
            'likely_dayfirst': has_day_over_12,
            'parse_success_rate': len(parsed_dates) / max(len(sample), 1),
            'sample_parsed': parsed_dates[:5]
        }


class ColumnNormalizer:
    """
    Normalize column data to consistent formats.

    Handles:
    - Numeric formats (1,000 vs 1000 vs 1.000)
    - Currency symbols
    - Percentage formats
    - Boolean representations
    - Whitespace and encoding issues
    """

    # Currency symbols to strip
    CURRENCY_SYMBOLS = ['$', '€', '£', '¥', '₹', 'USD', 'EUR', 'GBP']

    # Boolean representations
    BOOL_TRUE = {'true', 'yes', 'y', '1', 'on', 'active', 'enabled'}
    BOOL_FALSE = {'false', 'no', 'n', '0', 'off', 'inactive', 'disabled'}

    @log_io
    def normalize_numeric(self, value: Any) -> Optional[float]:
        """
        Normalize numeric values handling various formats.

        Handles:
        - 1,000.50 (US format)
        - 1.000,50 (European format)
        - $1,000 (with currency)
        - 50% (percentages)
        - (100) (accounting negative)
        """
        if value is None or pd.isna(value):
            return None

        val_str = str(value).strip()

        # Empty string
        if not val_str:
            return None

        # Already numeric
        if isinstance(value, (int, float)) and not pd.isna(value):
            return float(value)

        # Remove currency symbols
        for sym in self.CURRENCY_SYMBOLS:
            val_str = val_str.replace(sym, '')

        val_str = val_str.strip()

        # Handle percentage
        is_percent = val_str.endswith('%')
        if is_percent:
            val_str = val_str[:-1].strip()

        # Handle accounting format (negative in parentheses)
        if val_str.startswith('(') and val_str.endswith(')'):
            val_str = '-' + val_str[1:-1]

        # Detect number format
        # Count commas and periods to determine format
        comma_count = val_str.count(',')
        period_count = val_str.count('.')

        try:
            if comma_count > 0 and period_count > 0:
                # Both present - determine which is decimal
                last_comma = val_str.rfind(',')
                last_period = val_str.rfind('.')

                if last_period > last_comma:
                    # US format: 1,000.50
                    val_str = val_str.replace(',', '')
                else:
                    # European format: 1.000,50
                    val_str = val_str.replace('.', '').replace(',', '.')

            elif comma_count == 1 and period_count == 0:
                # Could be 1,000 (thousands) or 1,5 (decimal)
                parts = val_str.split(',')
                if len(parts[1]) == 3:
                    # Likely thousands separator
                    val_str = val_str.replace(',', '')
                else:
                    # Likely decimal
                    val_str = val_str.replace(',', '.')

            elif comma_count > 1:
                # Multiple commas = thousands separators
                val_str = val_str.replace(',', '')

            elif period_count > 1:
                # Multiple periods = thousands separators (European)
                val_str = val_str.replace('.', '')

            result = float(val_str)

            if is_percent:
                result = result / 100.0

            return result

        except ValueError:
            return None

    @log_io
    def normalize_boolean(self, value: Any) -> Optional[bool]:
        """Normalize boolean values from various representations."""
        if value is None or pd.isna(value):
            return None

        val_str = str(value).strip().lower()

        if val_str in self.BOOL_TRUE:
            return True
        elif val_str in self.BOOL_FALSE:
            return False

        return None

    @log_io
    def normalize_string(self, value: Any) -> Optional[str]:
        """Clean and normalize string values."""
        if value is None or pd.isna(value):
            return None

        val_str = str(value)

        # Fix common encoding issues
        replacements = {
            '\xa0': ' ',   # Non-breaking space
            '\u200b': '',  # Zero-width space
            '\ufeff': '',  # BOM
            '\r\n': '\n',  # Windows line endings
            '\r': '\n',    # Old Mac line endings
        }

        for old, new in replacements.items():
            val_str = val_str.replace(old, new)

        # Normalize whitespace
        val_str = ' '.join(val_str.split())

        return val_str.strip() if val_str.strip() else None

    @log_io
    def detect_column_type(
        self,
        values: List[Any],
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Detect the likely type and format of a column.

        Returns type info and normalization suggestions.
        """
        sample = values[:sample_size] if len(values) > sample_size else values
        non_null = [v for v in sample if v is not None and not pd.isna(v)]

        if not non_null:
            return {'type': 'empty', 'confidence': 1.0}

        # Try to parse as different types
        numeric_success = sum(1 for v in non_null if self.normalize_numeric(v) is not None)
        bool_success = sum(1 for v in non_null if self.normalize_boolean(v) is not None)

        numeric_rate = numeric_success / len(non_null)
        bool_rate = bool_success / len(non_null)

        # Check for dates
        date_normalizer = DateNormalizer()
        date_success = sum(1 for v in non_null if date_normalizer.parse(str(v)) is not None)
        date_rate = date_success / len(non_null)

        # Determine type
        if date_rate > 0.8:
            return {
                'type': 'datetime',
                'confidence': date_rate,
                'format_info': date_normalizer.detect_format_from_column([str(v) for v in non_null])
            }
        elif bool_rate > 0.9:
            return {'type': 'boolean', 'confidence': bool_rate}
        elif numeric_rate > 0.8:
            # Check for currency/percentage
            has_currency = any(
                any(sym in str(v) for sym in self.CURRENCY_SYMBOLS)
                for v in non_null
            )
            has_percent = any('%' in str(v) for v in non_null)

            return {
                'type': 'numeric',
                'subtype': 'currency' if has_currency else 'percentage' if has_percent else 'number',
                'confidence': numeric_rate
            }
        else:
            return {'type': 'string', 'confidence': 1.0 - max(numeric_rate, date_rate, bool_rate)}


class DataFrameNormalizer:
    """
    Normalize an entire DataFrame with consistent formatting.
    """

    def __init__(self):
        self.date_normalizer = DateNormalizer()
        self.column_normalizer = ColumnNormalizer()

    @log_io
    def normalize(
        self,
        df: pd.DataFrame,
        date_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        infer_types: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalize a DataFrame.

        Args:
            df: Input DataFrame
            date_columns: Columns to treat as dates (auto-detected if None)
            numeric_columns: Columns to treat as numeric (auto-detected if None)
            infer_types: Whether to auto-detect column types

        Returns:
            Tuple of (normalized DataFrame, normalization report)
        """
        df_normalized = df.copy()
        report = {
            'columns': {},
            'date_columns_detected': [],
            'numeric_columns_detected': [],
            'warnings': []
        }

        for col in df.columns:
            col_values = df[col].tolist()

            # Detect or use specified type
            if date_columns and col in date_columns:
                col_type = 'datetime'
            elif numeric_columns and col in numeric_columns:
                col_type = 'numeric'
            elif infer_types:
                type_info = self.column_normalizer.detect_column_type(col_values)
                col_type = type_info['type']
                report['columns'][col] = type_info
            else:
                col_type = 'string'

            # Normalize based on type
            if col_type == 'datetime':
                df_normalized[col] = df[col].apply(
                    lambda x: self.date_normalizer.parse(str(x)) if pd.notna(x) else None
                )
                report['date_columns_detected'].append(col)

                # Check for parsing failures
                failed = df_normalized[col].isna().sum() - df[col].isna().sum()
                if failed > 0:
                    report['warnings'].append(
                        f"Column '{col}': {failed} dates could not be parsed"
                    )

            elif col_type == 'numeric':
                df_normalized[col] = df[col].apply(
                    lambda x: self.column_normalizer.normalize_numeric(x)
                )
                report['numeric_columns_detected'].append(col)

            elif col_type == 'boolean':
                df_normalized[col] = df[col].apply(
                    lambda x: self.column_normalizer.normalize_boolean(x)
                )

            else:
                # String normalization
                df_normalized[col] = df[col].apply(
                    lambda x: self.column_normalizer.normalize_string(x)
                )

        return df_normalized, report


@log_io
def normalize_dataframe(
    df: pd.DataFrame,
    date_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to normalize a DataFrame.

    Args:
        df: Input DataFrame
        date_columns: Columns to treat as dates
        numeric_columns: Columns to treat as numeric

    Returns:
        Tuple of (normalized DataFrame, report)
    """
    normalizer = DataFrameNormalizer()
    return normalizer.normalize(df, date_columns, numeric_columns)
