"""
Chronos Foundation Model integration for zero-shot time series forecasting.

Provides zero-shot forecasting using Amazon's pretrained Chronos-Bolt models.
No training required - the model generates forecasts from historical context alone.

IMPORTANT: This model requires heavy dependencies (torch, chronos-forecasting).
If these are not available, a naive fallback is used instead.
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import logging
import warnings
import pickle

from backend.models.utils import (
    compute_metrics, detect_weekly_freq_code, detect_flat_forecast, compute_prediction_intervals
)
from backend.utils.logging_utils import log_io

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Check if Chronos is available
CHRONOS_AVAILABLE = False
CHRONOS_ERROR = None
try:
    from chronos import ChronosPipeline
    import torch
    CHRONOS_AVAILABLE = True
    logger.info("Chronos foundation model is available")
except ImportError as e:
    CHRONOS_ERROR = str(e)
    logger.warning(f"Chronos import failed: {e}. If version mismatch, try: pip install 'huggingface-hub<1.0'. Using naive fallback.")

# Model size configurations
_CHRONOS_MODELS = {
    'tiny': 'amazon/chronos-t5-tiny',        # 8M params - Fastest
    'mini': 'amazon/chronos-t5-mini',         # 20M params - Fast
    'small': 'amazon/chronos-t5-small',       # 46M params - Balanced
    'base': 'amazon/chronos-t5-base',         # 200M params - Best accuracy
    'large': 'amazon/chronos-t5-large',       # 710M params - Highest accuracy
}

# Bolt models (faster inference) - reserved for future use when ChronosBoltPipeline is stable
# _CHRONOS_BOLT_MODELS = {
#     'tiny': 'amazon/chronos-bolt-tiny',
#     'mini': 'amazon/chronos-bolt-mini',
#     'small': 'amazon/chronos-bolt-small',
#     'base': 'amazon/chronos-bolt-base',
# }


@log_io
def _get_device():
    """Detect the best available device for Chronos inference."""
    if not CHRONOS_AVAILABLE:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("Chronos: Using CUDA GPU")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Chronos: Using Apple MPS")
            return "mps"
        else:
            logger.info("Chronos: Using CPU")
            return "cpu"
    except ImportError:
        logger.warning("Chronos: PyTorch not available, defaulting to CPU")
        return "cpu"


@log_io
def _naive_forecast(history: np.ndarray, horizon: int, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate naive seasonal forecast when Chronos is not available.

    Uses seasonal naive method: forecast = value from same period last year.
    Falls back to drift method if insufficient history.
    """
    n = len(history)

    # Determine seasonal period based on history length
    if n >= 52:
        season = 52  # Weekly data, use yearly seasonality
    elif n >= 12:
        season = 12  # Monthly data, use yearly seasonality
    else:
        season = n  # Use all available history

    # Generate forecast using seasonal naive + drift
    forecast = np.zeros(horizon)
    for i in range(horizon):
        if n >= season:
            # Use value from same period last season
            idx = n - season + (i % season)
            if idx >= 0 and idx < n:
                forecast[i] = history[idx]
            else:
                forecast[i] = history[-1]
        else:
            # Drift method: last value + average change
            avg_change = (history[-1] - history[0]) / max(n - 1, 1) if n > 1 else 0
            forecast[i] = history[-1] + avg_change * (i + 1)

    # Compute prediction intervals based on historical variance
    if n > 1:
        residuals = np.diff(history)
        std = np.std(residuals) if len(residuals) > 0 else np.std(history) * 0.1
    else:
        std = abs(history[-1]) * 0.1 if len(history) > 0 else 1.0

    # Widen intervals for longer horizons
    # Calculate z-score for given confidence level
    try:
        from scipy.stats import norm
        z = norm.ppf(1 - (1 - confidence_level) / 2)
    except ImportError:
        # Fallback: approximate z-scores for common levels
        z = 1.96 if confidence_level >= 0.95 else (1.645 if confidence_level >= 0.90 else 1.28)
    widening = np.sqrt(np.arange(1, horizon + 1))  # Intervals widen with sqrt(h)
    margin = z * std * widening

    lower = forecast - margin
    upper = forecast + margin

    return forecast, lower, upper


class ChronosModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow-compatible wrapper for Chronos foundation model.

    Input format for serving endpoint:
    {
        "dataframe_records": [
            {"periods": 30, "start_date": "2025-01-01"}
        ]
    }

    Note: Chronos requires the historical context to generate forecasts.
    This wrapper stores the training context for inference.
    """

    def __init__(
        self,
        model_size: str,
        frequency: str,
        context_values: np.ndarray,
        weekly_freq_code: str = None,
        confidence_level: float = 0.95
    ):
        self.model_size = model_size
        self.model_path = _CHRONOS_MODELS.get(model_size, _CHRONOS_MODELS['small'])
        # Store frequency in human-readable format
        freq_to_human = {'MS': 'monthly', 'W': 'weekly', 'D': 'daily', 'YS': 'yearly'}
        self.frequency = freq_to_human.get(frequency, frequency)
        self.context_values = context_values
        self.weekly_freq_code = weekly_freq_code or 'W-MON'
        self.confidence_level = confidence_level
        self._pipeline = None

    def __getstate__(self):
        """Exclude loaded pipeline from pickling to avoid serializing 180MB+ model."""
        state = self.__dict__.copy()
        state['_pipeline'] = None
        return state

    @log_io
    def _load_pipeline(self):
        """Lazy load the Chronos pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        if not CHRONOS_AVAILABLE:
            logger.warning("Chronos not available, will use naive fallback")
            return None

        try:
            from chronos import ChronosPipeline
            import torch

            device = _get_device()

            # Try to load the model
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_path,
                device_map=device if device != "cpu" else "auto",
                torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            )
            return self._pipeline
        except Exception as e:
            logger.warning(f"Failed to load Chronos pipeline: {e}")
            return None

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd
        import numpy as np

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        # Extract parameters from input
        periods = int(model_input['periods'].iloc[0])
        start_date = pd.to_datetime(model_input['start_date'].iloc[0])

        # Map human-readable frequency to pandas freq code
        freq_map = {
            'daily': 'D',
            'weekly': self.weekly_freq_code,
            'monthly': 'MS',
            'yearly': 'YS'
        }

        # Get frequency from input or use stored default
        if 'frequency' in model_input.columns:
            freq_str = str(model_input['frequency'].iloc[0]).lower()
            pandas_freq = freq_map.get(freq_str, freq_map.get(self.frequency, 'MS'))
        else:
            pandas_freq = freq_map.get(self.frequency, 'MS')

        # Try Chronos pipeline first, fall back to naive if unavailable
        pipeline = self._load_pipeline()

        if pipeline is not None:
            try:
                import torch

                # Prepare context tensor
                context_tensor = torch.tensor(self.context_values, dtype=torch.float32)

                # Generate forecasts using Chronos
                forecast = pipeline.predict(
                    context_tensor.unsqueeze(0),
                    prediction_length=periods,
                    num_samples=200  # Generate samples for reliable prediction intervals
                )

                # Extract median forecast and quantiles
                forecast_np = forecast[0].numpy()  # Shape: (num_samples, horizon)
                forecast_values = np.median(forecast_np, axis=0)

                # Calculate prediction intervals from samples
                alpha = 1 - self.confidence_level
                lower_bounds = np.percentile(forecast_np, alpha / 2 * 100, axis=0)
                upper_bounds = np.percentile(forecast_np, (1 - alpha / 2) * 100, axis=0)

            except Exception as e:
                logger.warning(f"Chronos prediction failed, using naive fallback: {e}")
                forecast_values, lower_bounds, upper_bounds = _naive_forecast(
                    self.context_values, periods, self.confidence_level
                )
        else:
            # Use naive fallback
            logger.info("Using naive seasonal forecast (Chronos not available)")
            forecast_values, lower_bounds, upper_bounds = _naive_forecast(
                self.context_values, periods, self.confidence_level
            )

        # Generate future dates starting from start_date
        future_dates = pd.date_range(start=start_date, periods=periods + 1, freq=pandas_freq)[1:]

        # CRITICAL: Clip negative forecasts - financial metrics cannot be negative
        forecast_values = np.maximum(forecast_values, 0.0)
        lower_bounds = np.maximum(lower_bounds, 0.0)
        upper_bounds = np.maximum(upper_bounds, forecast_values)

        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values,
            'yhat_lower': lower_bounds,
            'yhat_upper': upper_bounds
        })


@log_io
def train_chronos_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    frequency: str = 'monthly',
    random_seed: int = 42,
    original_data: Optional[List[Dict[str, Any]]] = None,
    covariates: Optional[List[str]] = None,  # Kept for API compatibility
    hyperparameter_filters: Optional[Dict[str, Any]] = None,
    forecast_start_date: Optional[pd.Timestamp] = None,
    model_size: str = 'small'  # tiny, mini, small, base
) -> Tuple[str, str, Dict[str, float], pd.DataFrame, pd.DataFrame, str, Dict[str, Any]]:
    """
    Generate forecasts using Chronos foundation model (zero-shot, no training).

    Chronos is a pretrained time series foundation model that provides forecasts
    without any training on your specific data. It uses the historical context
    to generate predictions.

    If Chronos is not installed, falls back to naive seasonal forecasting.

    Args:
        train_df: Training DataFrame with 'ds' and 'y' columns (used as context)
        test_df: Test DataFrame for validation
        horizon: Number of periods to forecast
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        random_seed: Random seed for reproducibility
        original_data: Original data for logging
        covariates: Not used (Chronos is univariate)
        hyperparameter_filters: Not used (Chronos has no hyperparameters to tune)
        forecast_start_date: Date to start forecast from
        model_size: Model size ('tiny', 'mini', 'small', 'base')

    Returns:
        Tuple of (run_id, model_uri, metrics, validation_df, forecast_df, artifact_uri, model_info)
    """
    # Extract confidence level for prediction intervals (default 0.95)
    global_filters = (hyperparameter_filters or {}).get('_global', {})
    confidence_level = global_filters.get('confidence_level', 0.95)

    # Set random seeds for reproducibility
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Detect weekly frequency code for proper date alignment
    pd_freq = detect_weekly_freq_code(train_df, frequency)

    # Handle model_size being a list (from hyperparameter filters)
    if isinstance(model_size, list):
        model_size = model_size[0] if model_size else 'small'

    # Get model path
    model_path = _CHRONOS_MODELS.get(model_size, _CHRONOS_MODELS['small'])

    # Prepare context from training data
    context_values = train_df['y'].values.astype(np.float32)

    # Determine if we can use Chronos or need fallback
    use_chronos = CHRONOS_AVAILABLE
    pipeline = None
    device = "cpu"

    if use_chronos:
        logger.info(f"Running Chronos {model_size} model (freq={frequency}, seed={random_seed})...")
        logger.info("Note: Chronos is a zero-shot model - no training, only inference")
        logger.info(f"  Confidence level for prediction intervals: {confidence_level*100:.0f}%")
        logger.info(f"  Loading Chronos model: {model_path}")

        try:
            from chronos import ChronosPipeline
            import torch

            device = _get_device()
            pipeline = ChronosPipeline.from_pretrained(
                model_path,
                device_map=device if device != "cpu" else "auto",
                torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            )
            logger.info(f"  ✓ Chronos model loaded on {device}")
        except Exception as e:
            logger.warning(f"  ✗ Failed to load Chronos: {e}. Using naive fallback.")
            use_chronos = False
            pipeline = None
    else:
        logger.info(f"Running Chronos-Naive fallback (Chronos not available)")
        logger.info(f"  Using seasonal naive forecast with confidence level: {confidence_level*100:.0f}%")

    with mlflow.start_run(run_name=f"Chronos_{model_size}_{'Inference' if use_chronos else 'Naive'}", nested=True) as parent_run:
        parent_run_id = parent_run.info.run_id

        # Log original data if provided
        if original_data:
            try:
                original_df = pd.DataFrame(original_data)
                original_df.to_csv("/tmp/original_timeseries_data.csv", index=False)
                mlflow.log_artifact("/tmp/original_timeseries_data.csv", "datasets/raw")
            except Exception as e:
                logger.warning(f"Could not log original data: {e}")

        # Generate validation predictions (on test set)
        test_len = len(test_df)
        logger.info(f"  Generating validation predictions for {test_len} periods...")

        if use_chronos and pipeline is not None:
            try:
                import torch
                context_tensor = torch.tensor(context_values, dtype=torch.float32)

                # Generate forecasts with samples for uncertainty
                forecast_samples = pipeline.predict(
                    context_tensor.unsqueeze(0),
                    prediction_length=test_len,
                    num_samples=200
                )

                forecast_np = forecast_samples[0].numpy()
                val_predictions = np.median(forecast_np, axis=0)

                # Calculate prediction intervals from samples
                alpha = 1 - confidence_level
                val_lower = np.percentile(forecast_np, alpha / 2 * 100, axis=0)
                val_upper = np.percentile(forecast_np, (1 - alpha / 2) * 100, axis=0)

            except Exception as e:
                logger.warning(f"  Chronos validation failed: {e}. Using naive fallback.")
                val_predictions, val_lower, val_upper = _naive_forecast(
                    context_values, test_len, confidence_level
                )
        else:
            val_predictions, val_lower, val_upper = _naive_forecast(
                context_values, test_len, confidence_level
            )

        # Compute validation metrics
        actuals = np.array(test_df['y'].values[:len(val_predictions)])
        val_predictions = val_predictions[:len(actuals)]
        metrics = compute_metrics(actuals, val_predictions)

        logger.info(f"  ✓ Chronos {model_size}: MAPE={metrics['mape']:.2f}%, RMSE={metrics['rmse']:.2f}")

        # No cross-validation for zero-shot model
        metrics["cv_mape"] = None  # Zero-shot model: no cross-validation performed
        metrics["cv_mape_std"] = None

        # Create validation DataFrame
        validation_data = pd.DataFrame({
            'ds': test_df['ds'].values,
            'y': test_df['y'].values,
            'yhat': val_predictions,
            'yhat_lower': val_lower[:len(test_df)],
            'yhat_upper': val_upper[:len(test_df)]
        })

        # Generate future forecast
        logger.info(f"  Generating future forecast for {horizon} periods...")

        if use_chronos and pipeline is not None:
            try:
                import torch
                context_tensor = torch.tensor(context_values, dtype=torch.float32)

                forecast_samples = pipeline.predict(
                    context_tensor.unsqueeze(0),
                    prediction_length=horizon,
                    num_samples=200
                )

                forecast_np = forecast_samples[0].numpy()
                fcst_predictions = np.median(forecast_np, axis=0)

                alpha = 1 - confidence_level
                fcst_lower = np.percentile(forecast_np, alpha / 2 * 100, axis=0)
                fcst_upper = np.percentile(forecast_np, (1 - alpha / 2) * 100, axis=0)

            except Exception as e:
                logger.warning(f"  Chronos forecast failed: {e}. Using naive fallback.")
                fcst_predictions, fcst_lower, fcst_upper = _naive_forecast(
                    context_values, horizon, confidence_level
                )
        else:
            fcst_predictions, fcst_lower, fcst_upper = _naive_forecast(
                context_values, horizon, confidence_level
            )

        # Check for flat forecast
        flat_check = detect_flat_forecast(fcst_predictions, np.array(train_df['y'].values))
        if flat_check['is_flat']:
            logger.warning(f"Chronos flat forecast detected: {flat_check['flat_reason']}")

        # CRITICAL: Clip negative forecasts
        fcst_predictions = np.maximum(fcst_predictions, 0.0)
        fcst_lower = np.maximum(fcst_lower, 0.0)
        fcst_upper = np.maximum(fcst_upper, fcst_predictions)

        # Use forecast_start_date if provided, otherwise use end of training data
        if forecast_start_date is not None:
            last_date = pd.to_datetime(forecast_start_date).normalize()
            logger.info(f"📅 Using user-specified forecast start: {last_date}")
        else:
            last_date = train_df['ds'].max()

        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=pd_freq)[1:]
        logger.info(f"📅 Chronos forecast dates: {future_dates.min()} to {future_dates.max()}")

        forecast_data = pd.DataFrame({
            'ds': future_dates,
            'yhat': fcst_predictions[:len(future_dates)],
            'yhat_lower': fcst_lower[:len(future_dates)],
            'yhat_upper': fcst_upper[:len(future_dates)]
        })

        # Log datasets
        try:
            train_df[['ds', 'y']].to_csv("/tmp/train.csv", index=False)
            mlflow.log_artifact("/tmp/train.csv", "datasets/training")
            test_df[['ds', 'y']].to_csv("/tmp/eval.csv", index=False)
            mlflow.log_artifact("/tmp/eval.csv", "datasets/training")
        except Exception as e:
            logger.warning(f"Could not log datasets: {e}")

        # Log parameters and metrics
        mlflow.log_param("model_type", "Chronos")
        mlflow.log_param("model_size", model_size)
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("device", device)
        mlflow.log_param("context_length", len(context_values))
        mlflow.log_param("frequency", frequency)
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("zero_shot", True)
        mlflow.log_param("confidence_level", confidence_level)
        mlflow.log_metrics(metrics)

        # Log model as MLflow pyfunc
        try:
            import sys
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

            input_example = pd.DataFrame({
                'periods': [horizon],
                'start_date': [str(last_date.date()) if hasattr(last_date, 'date') else str(last_date)[:10]]
            })

            sample_output = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(1).copy()
            signature = infer_signature(input_example, sample_output)

            weekly_freq_code = detect_weekly_freq_code(train_df, frequency)

            model_wrapper = ChronosModelWrapper(
                model_size=model_size,
                frequency=frequency,
                context_values=context_values,
                weekly_freq_code=weekly_freq_code,
                confidence_level=confidence_level
            )

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model_wrapper,
                signature=signature,
                input_example=input_example,
                code_paths=["backend"],
                conda_env={
                    "channels": ["defaults", "conda-forge", "pytorch"],
                    "dependencies": [
                        f"python={python_version}",
                        "pip",
                        {"pip": ["mlflow", "pandas", "numpy", "chronos-forecasting", "torch"]}
                    ],
                    "name": "chronos_env"
                }
            )

            artifact_uri = mlflow.get_artifact_uri("model")
            logger.info(f"   ✅ Chronos model logged to: {artifact_uri}")

            # Save context as backup
            model_backup_path = "/tmp/chronos_model_backup.pkl"
            with open(model_backup_path, 'wb') as f:
                pickle.dump({
                    'model_size': model_size,
                    'model_path': model_path,
                    'context_values': context_values,
                    'frequency': frequency,
                    'weekly_freq_code': weekly_freq_code,
                    'confidence_level': confidence_level
                }, f)
            mlflow.log_artifact(model_backup_path, "model_backup")

        except Exception as e:
            logger.error(f"   ❌ Failed to log Chronos pyfunc model: {e}")
            try:
                model_path_pkl = "/tmp/chronos_model.pkl"
                with open(model_path_pkl, 'wb') as f:
                    pickle.dump({
                        'model_size': model_size,
                        'context_values': context_values,
                        'frequency': frequency,
                        'confidence_level': confidence_level
                    }, f)
                mlflow.log_artifact(model_path_pkl, "model")
                logger.warning("   ⚠️ Logged Chronos model as pickle fallback")
            except Exception as fallback_error:
                logger.error(f"   ❌ Fallback pickle also failed: {fallback_error}")

        best_artifact_uri = parent_run.info.artifact_uri or f"runs:/{parent_run_id}/artifacts"

    model_info = {
        'model_size': model_size,
        'model_path': model_path,
        'device': device,
        'context_length': len(context_values),
        'zero_shot': True,
        'confidence_level': confidence_level
    }

    return parent_run_id, f"runs:/{parent_run_id}/model", metrics, validation_data, forecast_data, best_artifact_uri, model_info
