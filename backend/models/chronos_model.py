"""
Chronos Foundation Model integration for zero-shot time series forecasting.

Provides zero-shot forecasting using Amazon's pretrained Chronos-Bolt models.
No training required - the model generates forecasts from historical context alone.
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

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Model size configurations
_CHRONOS_MODELS = {
    'tiny': 'amazon/chronos-bolt-tiny',      # 9M params - Fastest
    'mini': 'amazon/chronos-bolt-mini',       # 21M params - Fast
    'small': 'amazon/chronos-bolt-small',     # 48M params - Balanced
    'base': 'amazon/chronos-bolt-base',       # 205M params - Best accuracy
}


def _get_device():
    """Detect the best available device for Chronos inference."""
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

    def _load_pipeline(self):
        """Lazy load the Chronos pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from chronos import ChronosBoltPipeline
            import torch

            device = _get_device()
            self._pipeline = ChronosBoltPipeline.from_pretrained(
                self.model_path,
                device_map=device,
                torch_dtype=torch.float32
            )
            return self._pipeline
        except ImportError:
            raise ImportError(
                "Chronos not installed. Install with: pip install chronos-forecasting torch"
            )

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

        try:
            import torch
            pipeline = self._load_pipeline()

            # Prepare context tensor
            context_tensor = torch.tensor(self.context_values, dtype=torch.float32)

            # Calculate quantile levels based on confidence level
            # For 95% confidence: alpha=0.05, so quantiles are [0.025, 0.5, 0.975]
            # For 80% confidence: alpha=0.20, so quantiles are [0.10, 0.5, 0.90]
            alpha = 1 - self.confidence_level
            lower_quantile = alpha / 2
            upper_quantile = 1 - alpha / 2
            quantile_levels = [lower_quantile, 0.5, upper_quantile]

            # Generate quantile forecasts
            quantiles, mean = pipeline.predict_quantiles(
                context_tensor.unsqueeze(0),
                prediction_length=periods,
                quantile_levels=quantile_levels
            )

            # Extract forecast values
            forecast_values = mean[0].numpy()
            lower_bounds = quantiles[0, :, 0].numpy()  # lower quantile
            upper_bounds = quantiles[0, :, 2].numpy()  # upper quantile

        except Exception as e:
            logger.warning(f"Chronos prediction failed, using naive fallback: {e}")
            # Fallback: use last value as naive forecast
            if len(self.context_values) > 0:
                last_val = self.context_values[-1]
                forecast_values = np.full(periods, last_val)
            else:
                forecast_values = np.zeros(periods)
            lower_bounds = forecast_values * 0.9
            upper_bounds = forecast_values * 1.1

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


def train_chronos_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    frequency: str = 'monthly',
    random_seed: int = 42,
    original_data: List[Dict[str, Any]] = None,
    covariates: List[str] = None,  # Kept for API compatibility
    hyperparameter_filters: Dict[str, Any] = None,
    forecast_start_date: pd.Timestamp = None,
    model_size: str = 'small'  # tiny, mini, small, base
) -> Tuple[str, str, Dict[str, float], pd.DataFrame, pd.DataFrame, str, Dict[str, Any]]:
    """
    Generate forecasts using Chronos foundation model (zero-shot, no training).

    Chronos is a pretrained time series foundation model that provides forecasts
    without any training on your specific data. It uses the historical context
    to generate predictions.

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
    logger.info(f"Running Chronos {model_size} model (freq={frequency}, seed={random_seed})...")
    logger.info("Note: Chronos is a zero-shot model - no training, only inference")

    # Extract confidence level for prediction intervals (default 0.95)
    global_filters = (hyperparameter_filters or {}).get('_global', {})
    confidence_level = global_filters.get('confidence_level', 0.95)
    logger.info(f"  Confidence level for prediction intervals: {confidence_level*100:.0f}%")

    # Set random seeds for reproducibility
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Import Chronos
    try:
        from chronos import ChronosBoltPipeline
        import torch
    except ImportError as e:
        raise ImportError(
            "Chronos not installed. Install with: pip install chronos-forecasting torch"
        ) from e

    # Detect weekly frequency code for proper date alignment
    pd_freq = detect_weekly_freq_code(train_df, frequency)

    # Handle model_size being a list (from hyperparameter filters)
    if isinstance(model_size, list):
        model_size = model_size[0] if model_size else 'small'
        logger.info(f"  Model size was a list, using first value: {model_size}")

    # Get model path
    model_path = _CHRONOS_MODELS.get(model_size, _CHRONOS_MODELS['small'])
    logger.info(f"  Loading Chronos model: {model_path}")

    # Load Chronos pipeline
    device = _get_device()
    try:
        pipeline = ChronosBoltPipeline.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float32
        )
        logger.info(f"  ✓ Chronos model loaded on {device}")
    except Exception as e:
        logger.error(f"  ✗ Failed to load Chronos model: {e}")
        raise

    # Prepare context from training data
    context_values = train_df['y'].values.astype(np.float32)
    context_tensor = torch.tensor(context_values, dtype=torch.float32)

    with mlflow.start_run(run_name=f"Chronos_{model_size}_Inference", nested=True) as parent_run:
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

        # Calculate quantile levels based on confidence level
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        quantile_levels = [lower_quantile, 0.5, upper_quantile]

        try:
            val_quantiles, val_mean = pipeline.predict_quantiles(
                context_tensor.unsqueeze(0),
                prediction_length=test_len,
                quantile_levels=quantile_levels
            )

            val_predictions = val_mean[0].numpy()
            val_lower = val_quantiles[0, :, 0].numpy()
            val_upper = val_quantiles[0, :, 2].numpy()

        except Exception as e:
            logger.error(f"  Chronos validation inference failed: {e}")
            raise

        # Compute validation metrics
        actuals = test_df['y'].values[:len(val_predictions)]
        val_predictions = val_predictions[:len(actuals)]
        metrics = compute_metrics(actuals, val_predictions)

        logger.info(f"  ✓ Chronos {model_size}: MAPE={metrics['mape']:.2f}%, RMSE={metrics['rmse']:.2f}")

        # No cross-validation for zero-shot model (would be identical results)
        metrics["cv_mape"] = None
        metrics["cv_mape_std"] = None

        # Create validation DataFrame
        validation_data = test_df[['ds', 'y']].copy()
        validation_data['yhat'] = val_predictions
        validation_data['yhat_lower'] = val_lower[:len(validation_data)]
        validation_data['yhat_upper'] = val_upper[:len(validation_data)]

        # Generate future forecast
        logger.info(f"  Generating future forecast for {horizon} periods...")

        try:
            fcst_quantiles, fcst_mean = pipeline.predict_quantiles(
                context_tensor.unsqueeze(0),
                prediction_length=horizon,
                quantile_levels=quantile_levels  # Use same quantile levels as validation
            )

            fcst_predictions = fcst_mean[0].numpy()
            fcst_lower = fcst_quantiles[0, :, 0].numpy()
            fcst_upper = fcst_quantiles[0, :, 2].numpy()

        except Exception as e:
            logger.error(f"  Chronos forecast inference failed: {e}")
            raise

        # Check for flat forecast
        flat_check = detect_flat_forecast(fcst_predictions, train_df['y'].values)
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

        best_artifact_uri = parent_run.info.artifact_uri

    model_info = {
        'model_size': model_size,
        'model_path': model_path,
        'device': device,
        'context_length': len(context_values),
        'zero_shot': True,
        'confidence_level': confidence_level
    }

    return parent_run_id, f"runs:/{parent_run_id}/model", metrics, validation_data, forecast_data, best_artifact_uri, model_info
