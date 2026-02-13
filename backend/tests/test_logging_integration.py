"""Integration tests — verify decorated modules still work correctly."""
import logging

import pytest


class TestDecoratedModulesImport:
    """Verify that importing decorated modules doesn't break anything."""

    def test_import_logging_utils(self):
        from backend.utils.logging_utils import log_io, _truncate_value, _Lazy
        assert callable(log_io)
        assert callable(_truncate_value)

    def test_import_preprocessing(self):
        from backend import preprocessing
        assert hasattr(preprocessing, "enhance_features_for_forecasting")

    def test_import_data_analyzer(self):
        from backend import data_analyzer
        assert hasattr(data_analyzer, "analyze_time_series")

    def test_import_models_utils(self):
        from backend.models import utils
        assert hasattr(utils, "compute_metrics")

    def test_import_models_prophet(self):
        from backend.models import prophet
        assert hasattr(prophet, "train_prophet_model")

    def test_import_models_arima(self):
        from backend.models import arima
        assert hasattr(arima, "train_arima_model")

    def test_import_models_ets(self):
        from backend.models import ets
        assert hasattr(ets, "train_exponential_smoothing_model")

    def test_import_models_xgboost(self):
        from backend.models import xgboost
        assert hasattr(xgboost, "train_xgboost_model")

    def test_import_models_ensemble(self):
        from backend.models import ensemble
        assert hasattr(ensemble, "train_ensemble_model")


class TestDecoratedFunctionsPreserveBehavior:
    """Verify that decorated functions produce the same results."""

    def test_compute_metrics(self):
        np = pytest.importorskip("numpy")
        from backend.models.utils import compute_metrics

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        metrics = compute_metrics(y_true, y_pred)

        assert "mape" in metrics
        assert "rmse" in metrics
        assert "wape" in metrics
        assert metrics["mape"] < 10  # should be close

    def test_truncate_value_with_real_dataframe(self):
        pd = pytest.importorskip("pandas")
        from backend.utils.logging_utils import _truncate_value

        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100),
            "value": range(100),
        })
        result = _truncate_value(df)
        assert "DataFrame" in result
        assert "100" in result


class TestFastAPIApp:
    """Verify the FastAPI app still starts and responds to health checks."""

    def test_health_endpoint(self):
        httpx = pytest.importorskip("httpx")
        from starlette.testclient import TestClient
        from backend.main import app

        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"
