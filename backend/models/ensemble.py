"""
Model Ensemble module for combining predictions from multiple forecasting models.

Provides weighted averaging of model predictions based on validation performance
(inverse MAPE weighting) to create more robust forecasts.

Implements best practices from:
- AutoGluon: Automatic model filtering, weighted ensembles
- Nixtla: Efficient combination strategies
- Greykite: Robust validation checks
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
from backend.models.automl_utils import (
    validate_forecast,
    detect_overfitting,
    calculate_ensemble_weights,
    select_models_for_ensemble,
    ForecastValidationResult,
    OverfittingReport,
)
from backend.utils.logging_utils import log_io

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@log_io
def _get_result_dataframe(result: Dict[str, Any], df_type: str) -> Optional[pd.DataFrame]:
    """
    Get validation or forecast DataFrame from a model result dict.

    Handles both naming conventions:
    - val_df/fcst_df (DataFrame)
    - validation/forecast (list of dicts)

    Args:
        result: Model result dictionary
        df_type: Either 'val' or 'fcst'

    Returns:
        DataFrame or None if not found
    """
    if df_type == 'val':
        keys = ['val_df', 'validation']
    elif df_type == 'fcst':
        keys = ['fcst_df', 'forecast']
    else:
        return None

    for key in keys:
        df = result.get(key)
        if df is not None:
            # Convert list of dicts to DataFrame if needed
            if isinstance(df, list):
                df = pd.DataFrame(df)
            return df

    return None


@log_io
def _get_model_name(result: Dict[str, Any]) -> str:
    """
    Get model name from a result dict.

    Handles both 'model_name' and 'model_type' keys.

    Args:
        result: Model result dictionary

    Returns:
        Model name string or 'Unknown' if not found
    """
    return result.get('model_name') or result.get('model_type') or 'Unknown'


@log_io
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


@log_io
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


@log_io
def _validate_model_predictions(
    model_result: Dict[str, Any],
    historical_mean: float = None,
    historical_std: float = None,
    max_value_threshold: float = 1e30,
    max_mape_ratio: float = 5.0
) -> Tuple[bool, str, Optional[ForecastValidationResult]]:
    """
    Validate model predictions using AutoGluon-style comprehensive checks.

    Args:
        model_result: Dict containing 'val_df'/'fcst_df' DataFrames OR
                     'validation'/'forecast' list of dicts
        historical_mean: Historical mean for ratio validation
        historical_std: Historical std for variance validation
        max_value_threshold: Maximum absolute value allowed (default 1e30)
        max_mape_ratio: Maximum ratio to historical mean (default 5x)

    Returns:
        Tuple of (is_valid, reason_if_invalid, validation_result)
    """
    model_name = _get_model_name(model_result)

    # Support both DataFrame keys (val_df/fcst_df) and list-of-dict keys (validation/forecast)
    key_mapping = [
        ('val_df', 'validation'),
        ('fcst_df', 'forecast')
    ]

    for df_key, alt_key in key_mapping:
        df = model_result.get(df_key)

        # Try alternative key if primary key not found
        if df is None:
            df = model_result.get(alt_key)

        # Convert list of dicts to DataFrame if needed
        if df is not None and isinstance(df, list):
            df = pd.DataFrame(df)

        if df is None:
            return False, f"{df_key} is None", None

        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            if col not in df.columns:
                continue

            values = df[col].values

            # Check for NaN
            if np.any(np.isnan(values)):
                return False, f"{df_key}[{col}] contains NaN values", None

            # Check for inf
            if np.any(np.isinf(values)):
                return False, f"{df_key}[{col}] contains infinite values", None

            # Check for extreme values
            if np.any(np.abs(values) > max_value_threshold):
                max_abs = np.max(np.abs(values))
                return False, f"{df_key}[{col}] contains extreme values (max abs: {max_abs:.2e})", None

            # Check for all zeros (model loading/inference issue)
            if np.all(values == 0):
                return False, f"{df_key}[{col}] contains all zeros (inference failure)", None

    # Use automl_utils for comprehensive validation if historical stats provided
    validation_result = None
    if historical_mean is not None and historical_std is not None:
        fcst_df = _get_result_dataframe(model_result, 'fcst')
        if fcst_df is not None and 'yhat' in fcst_df.columns:
            validation_result = validate_forecast(
                predictions=fcst_df['yhat'].values,
                historical_mean=historical_mean,
                historical_std=historical_std,
                model_name=model_name,
                max_ratio=max_mape_ratio,
                min_ratio=1.0 / max_mape_ratio
            )
            if not validation_result.is_valid:
                return False, "; ".join(validation_result.issues), validation_result

    return True, "", validation_result


@log_io
def _check_model_overfitting(
    model_result: Dict[str, Any],
    max_severity: str = 'high'
) -> Tuple[bool, Optional[OverfittingReport]]:
    """
    Check if model shows severe overfitting.

    Args:
        model_result: Dict containing metrics
        max_severity: Maximum acceptable overfitting severity ('low', 'medium', 'high', 'severe')

    Returns:
        Tuple of (should_include, overfitting_report)
    """
    model_name = _get_model_name(model_result)
    metrics = model_result.get('metrics', {})

    # Need both train and eval metrics to detect overfitting
    train_mape = metrics.get('train_mape')
    eval_mape = metrics.get('mape')

    if train_mape is None or eval_mape is None:
        # Can't detect overfitting without both metrics
        return True, None

    try:
        train_mape = float(train_mape)
        eval_mape = float(eval_mape)
    except (ValueError, TypeError):
        return True, None

    # Use automl_utils overfitting detection
    overfitting_report = detect_overfitting(
        train_mape=train_mape,
        eval_mape=eval_mape,
        model_name=model_name
    )

    # Determine if severity exceeds threshold
    severity_order = ['none', 'low', 'medium', 'high', 'severe']
    max_severity_idx = severity_order.index(max_severity)
    actual_severity_idx = severity_order.index(overfitting_report.severity)

    should_include = actual_severity_idx <= max_severity_idx

    if not should_include:
        logger.warning(
            f"🚫 Excluding {model_name} from ensemble: "
            f"Overfitting severity '{overfitting_report.severity}' exceeds threshold '{max_severity}'"
        )

    return should_include, overfitting_report


@log_io
def create_ensemble_forecast(
    model_results: List[Dict[str, Any]],
    weighting_method: str = 'inverse_mape',
    min_weight: float = 0.05,
    historical_mean: float = None,
    historical_std: float = None,
    max_mape_ratio: float = 3.0,
    filter_overfitting: bool = True,
    max_overfitting_severity: str = 'high'
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Create ensemble forecast by combining predictions from multiple models.

    Implements AutoGluon-style model filtering:
    1. Filters models with invalid predictions (NaN, inf, extreme values, all zeros)
    2. Filters models with severe overfitting
    3. Filters models with MAPE far worse than best model
    4. Combines remaining models using weighted average

    Args:
        model_results: List of dicts, each containing:
            - 'model_name': Name of the model
            - 'metrics': Dict with 'mape' key
            - 'val_df': Validation DataFrame with yhat, yhat_lower, yhat_upper
            - 'fcst_df': Forecast DataFrame with yhat, yhat_lower, yhat_upper
        weighting_method: 'inverse_mape', 'inverse_mape_squared', 'softmax', 'equal', 'rank'
        min_weight: Minimum weight for any model
        historical_mean: Historical mean for validation (optional)
        historical_std: Historical std for validation (optional)
        max_mape_ratio: Maximum ratio of model MAPE to best model MAPE
        filter_overfitting: Whether to filter severely overfitting models
        max_overfitting_severity: Maximum acceptable overfitting severity

    Returns:
        Tuple of:
            - ensemble_val_df: Weighted validation predictions
            - ensemble_fcst_df: Weighted forecast predictions
            - ensemble_metrics: MAPE, RMSE, MAE, R2 for ensemble
            - model_weights: Dict of model_name -> weight
    """
    if not model_results:
        raise ValueError("No model results provided for ensemble")

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"🔀 AUTOML-STYLE ENSEMBLE CREATION")
    logger.info(f"{'='*60}")
    logger.info(f"   Input models: {len(model_results)}")
    logger.info(f"   Weighting method: {weighting_method}")
    logger.info(f"   Max MAPE ratio: {max_mape_ratio}x")
    logger.info(f"   Filter overfitting: {filter_overfitting} (max severity: {max_overfitting_severity})")

    # STEP 1: Filter out models with invalid predictions (NaN, inf, extreme values, zeros)
    logger.info(f"")
    logger.info(f"📋 Step 1: Validating model predictions...")
    valid_model_results = []
    for result in model_results:
        model_name = _get_model_name(result)
        is_valid, reason, validation_result = _validate_model_predictions(
            result,
            historical_mean=historical_mean,
            historical_std=historical_std,
            max_mape_ratio=max_mape_ratio * 2  # More lenient for individual validation
        )
        if is_valid:
            valid_model_results.append(result)
            logger.info(f"   ✅ {model_name}: Valid predictions")
        else:
            logger.warning(f"   🚫 {model_name}: EXCLUDED - {reason}")

    if not valid_model_results:
        raise ValueError("No valid model results after filtering invalid predictions")

    # STEP 2: Filter out severely overfitting models
    if filter_overfitting:
        logger.info(f"")
        logger.info(f"📋 Step 2: Checking for overfitting...")
        non_overfitting_results = []
        for result in valid_model_results:
            model_name = _get_model_name(result)
            should_include, overfitting_report = _check_model_overfitting(
                result,
                max_severity=max_overfitting_severity
            )
            if should_include:
                non_overfitting_results.append(result)
                if overfitting_report and overfitting_report.is_overfitting:
                    logger.info(f"   ⚠️ {model_name}: Mild overfitting ({overfitting_report.severity}), included")
                else:
                    logger.info(f"   ✅ {model_name}: No significant overfitting")
            else:
                logger.warning(f"   🚫 {model_name}: EXCLUDED - Severe overfitting")
        valid_model_results = non_overfitting_results

    if not valid_model_results:
        raise ValueError("No valid model results after filtering overfitting models")

    # STEP 3: Filter by relative MAPE performance
    logger.info(f"")
    logger.info(f"📋 Step 3: Filtering by relative performance...")
    mape_values = []
    for result in valid_model_results:
        mape = result.get('metrics', {}).get('mape')
        if mape is not None and mape != 'N/A':
            try:
                mape_values.append((result, float(mape)))
            except (ValueError, TypeError):
                pass

    if not mape_values:
        raise ValueError("No models with valid MAPE metrics")

    best_mape = min(m[1] for m in mape_values)
    mape_threshold = best_mape * max_mape_ratio

    filtered_results = []
    for result, mape in mape_values:
        model_name = _get_model_name(result)
        if mape <= mape_threshold:
            filtered_results.append(result)
            logger.info(f"   ✅ {model_name}: MAPE={mape:.2f}% (within {mape/best_mape:.1f}x of best)")
        else:
            logger.warning(f"   🚫 {model_name}: EXCLUDED - MAPE={mape:.2f}% ({mape/best_mape:.1f}x worse than best)")

    model_results = filtered_results

    if not model_results:
        raise ValueError("No valid model results after MAPE filtering")

    # STEP 3b: Filter by forecast divergence
    # Exclude models whose forecast mean diverges > 50% from the median forecast.
    # This catches models that have low eval MAPE but produce unreasonable extrapolations.
    logger.info(f"")
    logger.info(f"📋 Step 3b: Filtering by forecast divergence...")
    forecast_means = []
    for result in model_results:
        fcst_df = _get_result_dataframe(result, 'fcst')
        if fcst_df is not None and 'yhat' in fcst_df.columns:
            fcst_mean = fcst_df['yhat'].mean()
            if not np.isnan(fcst_mean) and not np.isinf(fcst_mean):
                forecast_means.append((_get_model_name(result), fcst_mean))

    if len(forecast_means) >= 3:
        median_forecast = np.median([m[1] for m in forecast_means])
        divergence_filtered = []
        for result in model_results:
            model_name = _get_model_name(result)
            fcst_df = _get_result_dataframe(result, 'fcst')
            if fcst_df is not None and 'yhat' in fcst_df.columns:
                fcst_mean = fcst_df['yhat'].mean()
                divergence_ratio = fcst_mean / median_forecast if median_forecast != 0 else 1.0
                if 0.5 <= divergence_ratio <= 1.5:
                    divergence_filtered.append(result)
                    logger.info(f"   ✅ {model_name}: forecast mean={fcst_mean:,.0f} ({divergence_ratio:.2f}x median)")
                else:
                    logger.warning(f"   🚫 {model_name}: EXCLUDED - forecast mean={fcst_mean:,.0f} ({divergence_ratio:.2f}x median, diverges > 50%)")
            else:
                divergence_filtered.append(result)
        # Only apply filter if we still have >= 2 models
        if len(divergence_filtered) >= 2:
            model_results = divergence_filtered
        else:
            logger.warning(f"   ⚠️ Divergence filter would leave < 2 models, skipping")
    else:
        logger.info(f"   Skipping (need >= 3 models for median-based filtering)")

    if len(model_results) < 2:
        logger.warning("Ensemble requires at least 2 valid models. Returning single model result.")
        result = model_results[0]
        return (
            _get_result_dataframe(result, 'val'),
            _get_result_dataframe(result, 'fcst'),
            result['metrics'],
            {_get_model_name(result): 1.0}
        )

    logger.info(f"")
    logger.info(f"📋 Step 4: Creating weighted ensemble from {len(model_results)} models...")

    # Calculate weights using AutoML-style methods
    # max_weight=0.40 prevents any single model from dominating the ensemble
    # (e.g., Prophet with 0.69% eval MAPE getting 53% weight while over-extrapolating)
    weights = calculate_ensemble_weights(
        model_results=model_results,
        method=weighting_method,
        min_weight=min_weight,
        max_weight=0.40,
        normalize=True
    )

    logger.info(f"   Ensemble weights:")
    for model_name, weight in sorted(weights.items(), key=lambda x: -x[1]):
        logger.info(f"      {model_name}: {weight:.1%}")

    # Get reference DataFrames for structure (use first model for dates and actuals)
    ref_val_df = _get_result_dataframe(model_results[0], 'val')
    ref_fcst_df = _get_result_dataframe(model_results[0], 'fcst')

    # FIX: Sort all val/fcst DataFrames by date before combining to ensure alignment
    ref_val_df = ref_val_df.copy()
    ref_val_df['ds'] = pd.to_datetime(ref_val_df['ds'])
    ref_val_df = ref_val_df.sort_values('ds').reset_index(drop=True)

    ref_fcst_df = ref_fcst_df.copy()
    ref_fcst_df['ds'] = pd.to_datetime(ref_fcst_df['ds'])
    ref_fcst_df = ref_fcst_df.sort_values('ds').reset_index(drop=True)

    # Initialize ensemble arrays
    val_len = len(ref_val_df)
    fcst_len = len(ref_fcst_df)

    ensemble_val_yhat = np.zeros(val_len)
    ensemble_val_lower = np.zeros(val_len)
    ensemble_val_upper = np.zeros(val_len)

    ensemble_fcst_yhat = np.zeros(fcst_len)
    ensemble_fcst_lower = np.zeros(fcst_len)
    ensemble_fcst_upper = np.zeros(fcst_len)

    # Combine predictions with weights (sort by date first to ensure alignment)
    for result in model_results:
        model_name = _get_model_name(result)
        weight = weights.get(model_name, 0)

        if weight == 0:
            continue

        # Validation data - sort by date before combining
        val_df = _get_result_dataframe(result, 'val')
        if val_df is not None and 'yhat' in val_df.columns:
            val_df = val_df.copy()
            val_df['ds'] = pd.to_datetime(val_df['ds'])
            val_df = val_df.sort_values('ds').reset_index(drop=True)
            n_val = min(val_len, len(val_df))
            ensemble_val_yhat[:n_val] += weight * val_df['yhat'].values[:n_val]
            ensemble_val_lower[:n_val] += weight * val_df['yhat_lower'].values[:n_val]
            ensemble_val_upper[:n_val] += weight * val_df['yhat_upper'].values[:n_val]

        # Forecast data - sort by date before combining
        fcst_df = _get_result_dataframe(result, 'fcst')
        if fcst_df is not None and 'yhat' in fcst_df.columns:
            fcst_df = fcst_df.copy()
            fcst_df['ds'] = pd.to_datetime(fcst_df['ds'])
            fcst_df = fcst_df.sort_values('ds').reset_index(drop=True)
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


@log_io
def train_ensemble_model(
    model_results: List[Dict[str, Any]],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    frequency: str = 'monthly',
    random_seed: int = 42,
    weighting_method: str = 'inverse_mape',
    min_weight: float = 0.05,
    forecast_start_date: pd.Timestamp = None,
    # AutoML best practices parameters
    historical_mean: float = None,
    historical_std: float = None,
    max_mape_ratio: float = 3.0,
    filter_overfitting: bool = True,
    max_overfitting_severity: str = 'high'
) -> Tuple[str, str, Dict[str, float], pd.DataFrame, pd.DataFrame, str, Dict[str, Any]]:
    """
    Create and log an ensemble model to MLflow.

    Implements AutoML best practices from AutoGluon, Nixtla, and Greykite:
    - Model validation (NaN, inf, extreme values, all-zeros detection)
    - Overfitting detection (train vs eval MAPE ratio)
    - MAPE-based filtering (only include models close to best)
    - Weighted ensemble with multiple weighting methods

    Args:
        model_results: List of dicts from individual model training
        train_df: Training DataFrame (for reference)
        test_df: Test DataFrame (for validation metrics)
        horizon: Forecast horizon
        frequency: Data frequency
        random_seed: Random seed
        weighting_method: 'inverse_mape', 'inverse_mape_squared', 'softmax', 'equal', 'rank'
        min_weight: Minimum weight for any model
        forecast_start_date: Date to start forecast from
        historical_mean: Mean of training data (for extreme value detection)
        historical_std: Std of training data (for extreme value detection)
        max_mape_ratio: Maximum MAPE ratio vs best model to include
        filter_overfitting: Whether to filter overfitting models
        max_overfitting_severity: 'low', 'medium', 'high' - threshold for overfitting

    Returns:
        Tuple of (run_id, model_uri, metrics, validation_df, forecast_df, artifact_uri, model_info)
    """
    logger.info(f"Training Ensemble model (method={weighting_method}, min_weight={min_weight})...")

    # Set random seeds
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create ensemble with AutoML best practices filtering
    ensemble_val_df, ensemble_fcst_df, ensemble_metrics, weights = create_ensemble_forecast(
        model_results=model_results,
        weighting_method=weighting_method,
        min_weight=min_weight,
        historical_mean=historical_mean,
        historical_std=historical_std,
        max_mape_ratio=max_mape_ratio,
        filter_overfitting=filter_overfitting,
        max_overfitting_severity=max_overfitting_severity
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

        # Log weights (sanitize model names for valid MLflow parameter keys)
        import re
        for model_name, weight in weights.items():
            # Replace invalid chars with underscores (MLflow allows: alphanumeric, _, -, ., :, /, space)
            sanitized_name = re.sub(r'[^a-zA-Z0-9_\-\. :/]', '_', model_name)
            mlflow.log_param(f"weight_{sanitized_name}", round(weight, 4))

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
            _get_model_name(r): _get_result_dataframe(r, 'fcst')
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
