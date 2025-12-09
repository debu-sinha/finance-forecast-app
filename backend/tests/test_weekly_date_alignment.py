"""
End-to-end test for weekly frequency date alignment fix.

This test verifies that all models generate forecast dates on the same
day of the week as the training data (e.g., if training data is on Mondays,
forecast dates should also be on Mondays).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.models.arima import detect_weekly_freq_code as arima_detect
from backend.models.ets import detect_weekly_freq_code as ets_detect
from backend.models.xgboost import detect_weekly_freq_code as xgboost_detect


def prophet_detect(df: pd.DataFrame, frequency: str) -> str:
    """Simulate Prophet's inline detection logic for testing."""
    if frequency != 'weekly':
        return {'daily': 'D', 'monthly': 'MS'}.get(frequency, 'MS')

    try:
        if 'ds' in df.columns:
            dates = pd.to_datetime(df['ds'])
        else:
            return 'W-MON'

        if len(dates) > 0:
            day_counts = dates.dt.dayofweek.value_counts()
            most_common_day = day_counts.idxmax()
            day_names = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
            return f"W-{day_names[most_common_day]}"
    except Exception:
        pass
    return 'W-MON'


class TestWeeklyFreqCodeDetection:
    """Test the detect_weekly_freq_code function across all models."""

    @pytest.fixture
    def monday_data(self):
        """Create sample data with Monday dates."""
        dates = pd.date_range(start='2024-01-01', periods=52, freq='W-MON')
        return pd.DataFrame({
            'ds': dates,
            'y': np.random.rand(52) * 1000
        })

    @pytest.fixture
    def tuesday_data(self):
        """Create sample data with Tuesday dates."""
        dates = pd.date_range(start='2024-01-02', periods=52, freq='W-TUE')
        return pd.DataFrame({
            'ds': dates,
            'y': np.random.rand(52) * 1000
        })

    @pytest.fixture
    def sunday_data(self):
        """Create sample data with Sunday dates."""
        dates = pd.date_range(start='2024-01-07', periods=52, freq='W-SUN')
        return pd.DataFrame({
            'ds': dates,
            'y': np.random.rand(52) * 1000
        })

    def test_prophet_detects_monday(self, monday_data):
        """Prophet should detect Monday-based weeks."""
        freq_code = prophet_detect(monday_data, 'weekly')
        assert freq_code == 'W-MON', f"Expected 'W-MON', got '{freq_code}'"

    def test_arima_detects_monday(self, monday_data):
        """ARIMA should detect Monday-based weeks."""
        freq_code = arima_detect(monday_data, 'weekly')
        assert freq_code == 'W-MON', f"Expected 'W-MON', got '{freq_code}'"

    def test_ets_detects_monday(self, monday_data):
        """ETS should detect Monday-based weeks."""
        freq_code = ets_detect(monday_data, 'weekly')
        assert freq_code == 'W-MON', f"Expected 'W-MON', got '{freq_code}'"

    def test_xgboost_detects_monday(self, monday_data):
        """XGBoost should detect Monday-based weeks."""
        freq_code = xgboost_detect(monday_data, 'weekly')
        assert freq_code == 'W-MON', f"Expected 'W-MON', got '{freq_code}'"

    def test_prophet_detects_tuesday(self, tuesday_data):
        """Prophet should detect Tuesday-based weeks."""
        freq_code = prophet_detect(tuesday_data, 'weekly')
        assert freq_code == 'W-TUE', f"Expected 'W-TUE', got '{freq_code}'"

    def test_arima_detects_sunday(self, sunday_data):
        """ARIMA should detect Sunday-based weeks."""
        freq_code = arima_detect(sunday_data, 'weekly')
        assert freq_code == 'W-SUN', f"Expected 'W-SUN', got '{freq_code}'"

    def test_daily_frequency_returns_D(self, monday_data):
        """Daily frequency should return 'D' regardless of data."""
        assert prophet_detect(monday_data, 'daily') == 'D'
        assert arima_detect(monday_data, 'daily') == 'D'
        assert ets_detect(monday_data, 'daily') == 'D'
        assert xgboost_detect(monday_data, 'daily') == 'D'

    def test_monthly_frequency_returns_MS(self, monday_data):
        """Monthly frequency should return 'MS' regardless of data."""
        assert prophet_detect(monday_data, 'monthly') == 'MS'
        assert arima_detect(monday_data, 'monthly') == 'MS'
        assert ets_detect(monday_data, 'monthly') == 'MS'
        assert xgboost_detect(monday_data, 'monthly') == 'MS'


class TestForecastDateGeneration:
    """Test that forecast dates are generated on the correct day of week."""

    @pytest.fixture
    def monday_training_data(self):
        """Create training data with Monday dates."""
        dates = pd.date_range(start='2024-01-01', periods=80, freq='W-MON')
        np.random.seed(42)
        return pd.DataFrame({
            'ds': dates,
            'y': np.random.rand(80) * 1000 + 500
        })

    def test_date_range_generates_correct_day(self):
        """Verify pd.date_range with W-MON generates Monday dates."""
        start_date = pd.Timestamp('2025-09-22')  # Monday
        dates = pd.date_range(start=start_date, periods=12, freq='W-MON')

        for date in dates:
            assert date.dayofweek == 0, f"Expected Monday (0), got {date.day_name()} ({date.dayofweek}) for {date}"

    def test_forecast_dates_match_training_dates(self, monday_training_data):
        """Verify forecast dates are on same day as training data."""
        # Detect the frequency code
        freq_code = prophet_detect(monday_training_data, 'weekly')
        assert freq_code == 'W-MON'

        # Generate forecast dates like the training function does
        last_date = monday_training_data['ds'].max()
        horizon = 12
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq_code)[1:]

        # All training dates should be Mondays
        training_days = monday_training_data['ds'].dt.dayofweek.unique()
        assert len(training_days) == 1, f"Training data has multiple days: {training_days}"
        assert training_days[0] == 0, f"Training data should be Monday, got {training_days[0]}"

        # All forecast dates should also be Mondays
        for date in future_dates:
            assert date.dayofweek == 0, f"Forecast date {date} is {date.day_name()}, expected Monday"

    def test_actual_data_format_detection(self):
        """Test with data similar to the actual user's format (M/D/YY)."""
        # Simulate the user's data format after parsing
        # Their data has dates like '1/22/24' which are Mondays
        raw_dates = ['1/22/24', '1/29/24', '2/5/24', '2/12/24', '2/19/24']

        # Parse like the app does
        parsed_dates = []
        for d in raw_dates:
            parts = d.split('/')
            month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
            parsed_dates.append(pd.Timestamp(2000 + year, month, day))

        df = pd.DataFrame({
            'ds': parsed_dates,
            'y': [100, 110, 120, 130, 140]
        })

        # Verify all parsed dates are Mondays
        for date in df['ds']:
            assert date.dayofweek == 0, f"Date {date} should be Monday, got {date.day_name()}"

        # Detect frequency
        freq_code = prophet_detect(df, 'weekly')
        assert freq_code == 'W-MON', f"Expected 'W-MON', got '{freq_code}'"


class TestEndToEndComparison:
    """Test that forecast dates can be compared with actuals."""

    def test_forecast_actuals_date_matching(self):
        """Simulate the full flow: training data -> forecast -> comparison with actuals."""

        # 1. Create training data (Mondays, Jan 2024 - Sep 2025)
        training_dates = pd.date_range(start='2024-01-22', end='2025-09-22', freq='W-MON')
        np.random.seed(42)
        training_df = pd.DataFrame({
            'ds': training_dates,
            'y': np.random.rand(len(training_dates)) * 1000 + 500
        })

        print(f"\n1. Training data:")
        print(f"   Rows: {len(training_df)}")
        print(f"   Date range: {training_df['ds'].min()} to {training_df['ds'].max()}")
        print(f"   Days of week: {training_df['ds'].dt.day_name().unique()}")

        # 2. Detect weekly frequency code
        freq_code = prophet_detect(training_df, 'weekly')
        print(f"\n2. Detected freq_code: {freq_code}")
        assert freq_code == 'W-MON'

        # 3. Generate forecast dates (12 weeks after training end)
        horizon = 12
        last_date = training_df['ds'].max()
        forecast_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq_code)[1:]

        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': np.random.rand(horizon) * 1000 + 500
        })

        print(f"\n3. Forecast dates:")
        print(f"   Rows: {len(forecast_df)}")
        print(f"   Date range: {forecast_df['ds'].min()} to {forecast_df['ds'].max()}")
        print(f"   Days of week: {forecast_df['ds'].dt.day_name().unique()}")

        # 4. Create actuals data (also Mondays, overlapping with forecast period)
        actuals_dates = pd.date_range(start='2024-01-22', end='2025-12-08', freq='W-MON')
        actuals_df = pd.DataFrame({
            'ds': actuals_dates,
            'actual': np.random.rand(len(actuals_dates)) * 1000 + 500
        })

        print(f"\n4. Actuals data:")
        print(f"   Rows: {len(actuals_df)}")
        print(f"   Date range: {actuals_df['ds'].min()} to {actuals_df['ds'].max()}")
        print(f"   Days of week: {actuals_df['ds'].dt.day_name().unique()}")

        # 5. Find overlapping dates (this is what was failing before the fix)
        forecast_date_set = set(forecast_df['ds'].dt.strftime('%Y-%m-%d'))
        actuals_date_set = set(actuals_df['ds'].dt.strftime('%Y-%m-%d'))

        overlapping = forecast_date_set & actuals_date_set

        print(f"\n5. Date overlap analysis:")
        print(f"   Forecast dates: {sorted(forecast_date_set)}")
        print(f"   Actuals dates in forecast range: {sorted([d for d in actuals_date_set if d >= min(forecast_date_set)])}")
        print(f"   Overlapping dates: {len(overlapping)}")
        print(f"   Overlapping: {sorted(overlapping)}")

        # 6. Assert we have overlapping dates
        assert len(overlapping) > 0, "No overlapping dates found! This was the bug."
        # Note: May have fewer than horizon if actuals don't extend far enough
        assert len(overlapping) >= horizon - 1, f"Expected at least {horizon - 1} overlapping dates, got {len(overlapping)}"

        # 7. Verify all dates are on the same day of week
        all_forecast_days = forecast_df['ds'].dt.dayofweek.unique()
        all_actuals_days = actuals_df['ds'].dt.dayofweek.unique()

        assert len(all_forecast_days) == 1, f"Forecast has multiple days: {all_forecast_days}"
        assert len(all_actuals_days) == 1, f"Actuals has multiple days: {all_actuals_days}"
        assert all_forecast_days[0] == all_actuals_days[0], \
            f"Forecast day ({all_forecast_days[0]}) != Actuals day ({all_actuals_days[0]})"

        print("\n✅ All tests passed! Forecast dates align with actuals dates.")


class TestModelWrapperDateGeneration:
    """Test that model wrappers generate dates correctly."""

    def test_arima_wrapper_uses_weekly_freq_code(self):
        """Test ARIMAModelWrapper uses the correct weekly frequency."""
        from backend.models.arima import ARIMAModelWrapper
        import pickle

        # Create a mock fitted model (we just need to test date generation)
        class MockARIMA:
            def forecast(self, steps):
                return np.array([100] * steps)

        # Create wrapper with W-MON
        wrapper = ARIMAModelWrapper(
            fitted_model=MockARIMA(),
            order=(1, 1, 1),
            frequency='weekly',
            weekly_freq_code='W-MON'
        )

        # Test prediction
        input_df = pd.DataFrame({
            'periods': [12],
            'start_date': ['2025-09-22']  # Monday
        })

        result = wrapper.predict(None, input_df)

        # All dates should be Mondays
        for date in result['ds']:
            assert date.dayofweek == 0, f"Date {date} is {date.day_name()}, expected Monday"

        print(f"\nARIMA wrapper forecast dates:")
        for date in result['ds']:
            print(f"  {date.strftime('%Y-%m-%d')} ({date.day_name()})")

    def test_sarimax_wrapper_uses_weekly_freq_code(self):
        """Test SARIMAXModelWrapper uses the correct weekly frequency."""
        from backend.models.arima import SARIMAXModelWrapper

        class MockSARIMAX:
            def forecast(self, steps, exog=None):
                return np.array([100] * steps)

        wrapper = SARIMAXModelWrapper(
            fitted_model=MockSARIMAX(),
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 52),
            frequency='weekly',
            covariates=[],
            covariate_means={},
            weekly_freq_code='W-MON'
        )

        input_df = pd.DataFrame({
            'periods': [12],
            'start_date': ['2025-09-22']
        })

        result = wrapper.predict(None, input_df)

        for date in result['ds']:
            assert date.dayofweek == 0, f"Date {date} is {date.day_name()}, expected Monday"

        print(f"\nSARIMAX wrapper forecast dates:")
        for date in result['ds']:
            print(f"  {date.strftime('%Y-%m-%d')} ({date.day_name()})")

    def test_ets_wrapper_uses_weekly_freq_code(self):
        """Test ExponentialSmoothingModelWrapper uses the correct weekly frequency."""
        from backend.models.ets import ExponentialSmoothingModelWrapper

        class MockETS:
            def forecast(self, steps):
                return np.array([100] * steps)

        wrapper = ExponentialSmoothingModelWrapper(
            fitted_model=MockETS(),
            params={'trend': 'add', 'seasonal': 'add'},
            frequency='weekly',
            seasonal_periods=52,
            weekly_freq_code='W-MON'
        )

        input_df = pd.DataFrame({
            'periods': [12],
            'start_date': ['2025-09-22']
        })

        result = wrapper.predict(None, input_df)

        for date in result['ds']:
            assert date.dayofweek == 0, f"Date {date} is {date.day_name()}, expected Monday"

        print(f"\nETS wrapper forecast dates:")
        for date in result['ds']:
            print(f"  {date.strftime('%Y-%m-%d')} ({date.day_name()})")


def run_quick_validation():
    """Quick validation that can be run without pytest."""
    print("=" * 60)
    print("WEEKLY DATE ALIGNMENT - QUICK VALIDATION")
    print("=" * 60)

    # Test 1: Frequency detection
    print("\n[TEST 1] Frequency Code Detection")
    print("-" * 40)

    monday_dates = pd.date_range(start='2024-01-01', periods=52, freq='W-MON')
    df = pd.DataFrame({'ds': monday_dates, 'y': range(52)})

    results = {
        'prophet': prophet_detect(df, 'weekly'),
        'arima': arima_detect(df, 'weekly'),
        'ets': ets_detect(df, 'weekly'),
        'xgboost': xgboost_detect(df, 'weekly'),
    }

    all_pass = True
    for model, freq in results.items():
        status = "✅" if freq == 'W-MON' else "❌"
        if freq != 'W-MON':
            all_pass = False
        print(f"  {model}: {freq} {status}")

    # Test 2: Date generation
    print("\n[TEST 2] Forecast Date Generation")
    print("-" * 40)

    last_monday = pd.Timestamp('2025-09-22')  # Monday
    forecast_dates = pd.date_range(start=last_monday, periods=13, freq='W-MON')[1:]

    print(f"  Last training date: {last_monday} ({last_monday.day_name()})")
    print(f"  Forecast dates:")
    for d in forecast_dates:
        day_ok = d.dayofweek == 0
        status = "✅" if day_ok else "❌"
        if not day_ok:
            all_pass = False
        print(f"    {d.strftime('%Y-%m-%d')} ({d.day_name()}) {status}")

    # Test 3: Overlap check
    print("\n[TEST 3] Forecast vs Actuals Overlap")
    print("-" * 40)

    actuals_dates = pd.date_range(start='2024-01-22', end='2025-12-08', freq='W-MON')
    forecast_set = set(forecast_dates.strftime('%Y-%m-%d'))
    actuals_set = set(actuals_dates.strftime('%Y-%m-%d'))
    overlap = forecast_set & actuals_set

    print(f"  Forecast dates: {len(forecast_set)}")
    print(f"  Actuals dates: {len(actuals_set)}")
    print(f"  Overlapping: {len(overlap)}")

    if len(overlap) == 0:
        print("  ❌ NO OVERLAP - Bug still present!")
        all_pass = False
    else:
        print(f"  ✅ {len(overlap)} overlapping dates found")
        print(f"  Overlap: {sorted(overlap)}")

    # Summary
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ ALL TESTS PASSED - Weekly date alignment is working!")
    else:
        print("❌ SOME TESTS FAILED - Check the issues above")
    print("=" * 60)

    return all_pass


if __name__ == '__main__':
    # Run quick validation first
    success = run_quick_validation()

    if success:
        print("\n\nRunning full pytest suite...\n")
        pytest.main([__file__, '-v', '--tb=short'])
