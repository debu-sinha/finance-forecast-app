"""
Model Ensemble module for combining predictions from multiple forecasting models.

Provides weighted averaging of model predictions based on validation performance
(inverse MAPE weighting) to create more robust forecasts.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import logging
import warnings
import pickle

from backend.models.utils import compute_metrics, detect_weekly_freq_code

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def calculate_inverse_mape_weights(
    validation_errors: Dict[str, float],
    min_weight: float = 0.05
) -> Dict[str, float]:
    """
    Calculate model weights inversely proportional to MAPE.

    Models with lower MAPE get higher weights. A minimum weight threshold
    ensures no model is completely excluded.

    Args:
        validation_errors: Dict mapping model_name -> MAPE value
        min_weight: Minimum weight for any model (prevents complete exclusion)

    Returns:
        Dict mapping model_name -> normalized weight
    """
    if not validation_errors:
        return {}

    # Calculate inverse errors (lower MAPE = higher inverse)
    inverse_errors = {
        model: 1.0 / max(mape, 0.001)  # Avoid division by zero
        for model, mape in validation_errors.items()
    }

    # Normalize to sum to 1
    total = sum(inverse_errors.values())
    weights = {model: w / total for model, w in inverse_errors.items()}

    # Apply minimum weight threshold
    for model in weights:
        weights[model] = max(weights[model], min_weight)

    # Renormalize after applying minimum
    total = sum(weights.values())
    weights = {model: w / total for model, w in weights.items()}

    return weights


def calculate_equal_weights(model_names: List[str]) -> Dict[str, float]:
    """
    Calculate equal weights for all models.

    Args:
        model_names: List of model names

    Returns:
        Dict mapping model_name -> equal weight (1/n)
    """
    if not model_names:
        return {}

    n = len(model_names)
    return {name: 1.0 / n for name in model_names}


class EnsembleModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow-compatible wrapper for ensemble model.

    The ensemble model stores predictions from all component models
    and combines them using pre-computed weights.

    Input format for serving endpoint:
    {
        "dataframe_records": [
            {"periods": 30, "start_date": "2025-01-01"}
        ]
    }
    """

    def __init__(
        self,
        model_weights: Dict[str, float],
        model_forecasts: Dict[str, pd.DataFrame],
        frequency: str,
        weekly_freq_code: str = None
    ):
        self.model_weights = model_weights
        self.model_forecasts = model_forecasts
        # Store frequency in human-readable format
        freq_to_human = {'MS': 'monthly', 'W': 'weekly', 'D': 'daily', 'YS': 'yearly'}
        self.frequency = freq_to_human.get(frequency, frequency)
        self.weekly_freq_code = weekly_freq_code or 'W-MON'

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

        # Generate future dates
        future_dates = pd.date_range(start=start_date, periods=periods + 1, freq=pandas_freq)[1:]

        # Combine forecasts using weights
        ensemble_yhat = np.zeros(periods)
        ensemble_lower = np.zeros(periods)
        ensemble_upper = np.zeros(periods)

        for model_name, weight in self.model_weights.items():
            if model_name in self.model_forecasts:
                fcst = self.model_forecasts[model_name]
                # Align lengths
                n = min(periods, len(fcst))
                ensemble_yhat[:n] += weight * fcst['yhat'].values[:n]
                ensemble_lower[:n] += weight * fcst['yhat_lower'].values[:n]
                ensemble_upper[:n] += weight * fcst['yhat_upper'].values[:n]

        # CRITICAL: Clip negative forecasts
        ensemble_yhat = np.maximum(ensemble_yhat, 0.0)
        ensemble_lower = np.maximum(ensemble_lower, 0.0)
        ensemble_upper = np.maximum(ensemble_upper, ensemble_yhat)

        return pd.DataFrame({
            'ds': future_dates,
            'yhat': ensemble_yhat,
            'yhat_lower': ensemble_lower,
            'yhat_upper': ensemble_upper
        })


def create_ensemble_forecast(
    model_results: List[Dict[str, Any]],
    weighting_method: str = 'inverse_mape',
    min_weight: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Create ensemble forecast by combining predictions from multiple models.

    Args:
        model_results: List of dicts, each containing:
            - 'model_name': Name of the model
            - 'metrics': Dict with 'mape' key
            - 'val_df': Validation DataFrame with yhat, yhat_lower, yhat_upper
            - 'fcst_df': Forecast DataFrame with yhat, yhat_lower, yhat_upper
        weighting_method: 'inverse_mape' or 'equal'
        min_weight: Minimum weight for any model

    Returns:
        Tuple of:
            - ensemble_val_df: Weighted validation predictions
            - ensemble_fcst_df: Weighted forecast predictions
            - ensemble_metrics: MAPE, RMSE, MAE, R2 for ensemble
            - model_weights: Dict of model_name -> weight
    """
    if not model_results:
        raise ValueError("No model results provided for ensemble")

    if len(model_results) < 2:
        logger.warning("Ensemble requires at least 2 models. Returning single model result.")
        result = model_results[0]
        return (
            result['val_df'],
            result['fcst_df'],
            result['metrics'],
            {result['model_name']: 1.0}
        )

    logger.info(f"Creating ensemble from {len(model_results)} models using {weighting_method} weighting")

    # Calculate weights
    if weighting_method == 'inverse_mape':
        validation_errors = {
            r['model_name']: r['metrics']['mape']
            for r in model_results
            if r['metrics'].get('mape') is not None
        }
        weights = calculate_inverse_mape_weights(validation_errors, min_weight)
    else:  # equal
        weights = calculate_equal_weights([r['model_name'] for r in model_results])

    logger.info(f"  Model weights: {weights}")

    # Get reference DataFrames for structure
    ref_val_df = model_results[0]['val_df']
    ref_fcst_df = model_results[0]['fcst_df']

    # Initialize ensemble arrays
    val_len = len(ref_val_df)
    fcst_len = len(ref_fcst_df)

    ensemble_val_yhat = np.zeros(val_len)
    ensemble_val_lower = np.zeros(val_len)
    ensemble_val_upper = np.zeros(val_len)

    ensemble_fcst_yhat = np.zeros(fcst_len)
    ensemble_fcst_lower = np.zeros(fcst_len)
    ensemble_fcst_upper = np.zeros(fcst_len)

    # Combine predictions with weights
    for result in model_results:
        model_name = result['model_name']
        weight = weights.get(model_name, 0)

        if weight == 0:
            continue

        # Validation data
        val_df = result['val_df']
        n_val = min(val_len, len(val_df))
        ensemble_val_yhat[:n_val] += weight * val_df['yhat'].values[:n_val]
        ensemble_val_lower[:n_val] += weight * val_df['yhat_lower'].values[:n_val]
        ensemble_val_upper[:n_val] += weight * val_df['yhat_upper'].values[:n_val]

        # Forecast data
        fcst_df = result['fcst_df']
        n_fcst = min(fcst_len, len(fcst_df))
        ensemble_fcst_yhat[:n_fcst] += weight * fcst_df['yhat'].values[:n_fcst]
        ensemble_fcst_lower[:n_fcst] += weight * fcst_df['yhat_lower'].values[:n_fcst]
        ensemble_fcst_upper[:n_fcst] += weight * fcst_df['yhat_upper'].values[:n_fcst]

    # Create ensemble DataFrames
    ensemble_val_df = ref_val_df[['ds', 'y']].copy()
    ensemble_val_df['yhat'] = ensemble_val_yhat
    ensemble_val_df['yhat_lower'] = ensemble_val_lower
    ensemble_val_df['yhat_upper'] = ensemble_val_upper

    ensemble_fcst_df = ref_fcst_df[['ds']].copy()
    ensemble_fcst_df['yhat'] = ensemble_fcst_yhat
    ensemble_fcst_df['yhat_lower'] = ensemble_fcst_lower
    ensemble_fcst_df['yhat_upper'] = ensemble_fcst_upper

    # CRITICAL: Clip negative forecasts
    ensemble_val_df['yhat'] = np.maximum(ensemble_val_df['yhat'], 0.0)
    ensemble_val_df['yhat_lower'] = np.maximum(ensemble_val_df['yhat_lower'], 0.0)
    ensemble_val_df['yhat_upper'] = np.maximum(ensemble_val_df['yhat_upper'], ensemble_val_df['yhat'])

    ensemble_fcst_df['yhat'] = np.maximum(ensemble_fcst_df['yhat'], 0.0)
    ensemble_fcst_df['yhat_lower'] = np.maximum(ensemble_fcst_df['yhat_lower'], 0.0)
    ensemble_fcst_df['yhat_upper'] = np.maximum(ensemble_fcst_df['yhat_upper'], ensemble_fcst_df['yhat'])

    # Compute ensemble metrics
    actuals = ensemble_val_df['y'].values
    predictions = ensemble_val_df['yhat'].values
    ensemble_metrics = compute_metrics(actuals, predictions)

    logger.info(f"  Ensemble MAPE: {ensemble_metrics['mape']:.2f}%")

    # Compare to individual models
    best_individual_mape = min(r['metrics']['mape'] for r in model_results if r['metrics'].get('mape'))
    if ensemble_metrics['mape'] < best_individual_mape:
        logger.info(f"  ✨ Ensemble beats best individual model ({best_individual_mape:.2f}%)")
    else:
        logger.info(f"  ℹ️ Best individual model ({best_individual_mape:.2f}%) beats ensemble")

    return ensemble_val_df, ensemble_fcst_df, ensemble_metrics, weights


def train_ensemble_model(
    model_results: List[Dict[str, Any]],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    frequency: str = 'monthly',
    random_seed: int = 42,
    weighting_method: str = 'inverse_mape',
    min_weight: float = 0.05,
    forecast_start_date: pd.Timestamp = None
) -> Tuple[str, str, Dict[str, float], pd.DataFrame, pd.DataFrame, str, Dict[str, Any]]:
    """
    Create and log an ensemble model to MLflow.

    Args:
        model_results: List of dicts from individual model training
        train_df: Training DataFrame (for reference)
        test_df: Test DataFrame (for validation metrics)
        horizon: Forecast horizon
        frequency: Data frequency
        random_seed: Random seed
        weighting_method: 'inverse_mape' or 'equal'
        min_weight: Minimum weight for any model
        forecast_start_date: Date to start forecast from

    Returns:
        Tuple of (run_id, model_uri, metrics, validation_df, forecast_df, artifact_uri, model_info)
    """
    logger.info(f"Training Ensemble model (method={weighting_method}, min_weight={min_weight})...")

    # Set random seeds
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create ensemble
    ensemble_val_df, ensemble_fcst_df, ensemble_metrics, weights = create_ensemble_forecast(
        model_results=model_results,
        weighting_method=weighting_method,
        min_weight=min_weight
    )

    # Detect weekly frequency code
    pd_freq = detect_weekly_freq_code(train_df, frequency)

    with mlflow.start_run(run_name="Ensemble_Model", nested=True) as parent_run:
        parent_run_id = parent_run.info.run_id

        # Log parameters
        mlflow.log_param("model_type", "Ensemble")
        mlflow.log_param("weighting_method", weighting_method)
        mlflow.log_param("min_weight", min_weight)
        mlflow.log_param("num_models", len(model_results))
        mlflow.log_param("component_models", str(list(weights.keys())))
        mlflow.log_param("frequency", frequency)
        mlflow.log_param("random_seed", random_seed)

        # Log weights
        for model_name, weight in weights.items():
            mlflow.log_param(f"weight_{model_name}", round(weight, 4))

        # Log metrics
        mlflow.log_metrics(ensemble_metrics)

        # Log datasets
        try:
            train_df[['ds', 'y']].to_csv("/tmp/train.csv", index=False)
            mlflow.log_artifact("/tmp/train.csv", "datasets/training")
            test_df[['ds', 'y']].to_csv("/tmp/eval.csv", index=False)
            mlflow.log_artifact("/tmp/eval.csv", "datasets/training")
        except Exception as e:
            logger.warning(f"Could not log datasets: {e}")

        # Prepare forecast dict for wrapper
        model_forecasts = {
            r['model_name']: r['fcst_df']
            for r in model_results
        }

        # Use forecast_start_date if provided
        if forecast_start_date is not None:
            last_date = pd.to_datetime(forecast_start_date).normalize()
        else:
            last_date = train_df['ds'].max()

        # Log model as MLflow pyfunc
        try:
            import sys
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

            input_example = pd.DataFrame({
                'periods': [horizon],
                'start_date': [str(last_date.date()) if hasattr(last_date, 'date') else str(last_date)[:10]]
            })

            sample_output = ensemble_fcst_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(1).copy()
            signature = infer_signature(input_example, sample_output)

            weekly_freq_code = detect_weekly_freq_code(train_df, frequency)

            model_wrapper = EnsembleModelWrapper(
                model_weights=weights,
                model_forecasts=model_forecasts,
                frequency=frequency,
                weekly_freq_code=weekly_freq_code
            )

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model_wrapper,
                signature=signature,
                input_example=input_example,
                code_paths=["backend"],
                conda_env={
                    "channels": ["defaults", "conda-forge"],
                    "dependencies": [
                        f"python={python_version}",
                        "pip",
                        {"pip": ["mlflow", "pandas", "numpy"]}
                    ],
                    "name": "ensemble_env"
                }
            )

            artifact_uri = mlflow.get_artifact_uri("model")
            logger.info(f"   ✅ Ensemble model logged to: {artifact_uri}")

            # Save backup
            model_backup_path = "/tmp/ensemble_model_backup.pkl"
            with open(model_backup_path, 'wb') as f:
                pickle.dump({
                    'weights': weights,
                    'forecasts': model_forecasts,
                    'frequency': frequency,
                    'weighting_method': weighting_method
                }, f)
            mlflow.log_artifact(model_backup_path, "model_backup")

        except Exception as e:
            logger.error(f"   ❌ Failed to log Ensemble pyfunc model: {e}")

        best_artifact_uri = parent_run.info.artifact_uri

    model_info = {
        'weights': weights,
        'weighting_method': weighting_method,
        'num_models': len(model_results),
        'component_models': list(weights.keys())
    }

    return parent_run_id, f"runs:/{parent_run_id}/model", ensemble_metrics, ensemble_val_df, ensemble_fcst_df, best_artifact_uri, model_info
