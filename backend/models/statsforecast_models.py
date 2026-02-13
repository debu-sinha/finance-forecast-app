"""
StatsForecast integration for high-performance statistical forecasting.

Provides 10-100x faster alternatives to statsmodels ARIMA/ETS using Nixtla's
StatsForecast library. Includes AutoARIMA, AutoETS, and AutoTheta models.
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import logging
import warnings
import pickle

from backend.models.utils import (
    compute_metrics, time_series_cross_validate, compute_prediction_intervals,
    detect_weekly_freq_code, detect_flat_forecast
)
from backend.utils.logging_utils import log_io

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Season length mapping for StatsForecast
_SEASON_LENGTH_MAP = {
    'daily': 7,      # Weekly seasonality
    'weekly': 52,    # Yearly seasonality
    'monthly': 12,   # Yearly seasonality
    'yearly': 1      # No seasonality
}

# Frequency mapping for StatsForecast
_FREQ_MAP = {
    'daily': 'D',
    'weekly': 'W',
    'monthly': 'MS',
    'yearly': 'YS'
}


class StatsForecastModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow-compatible wrapper for StatsForecast models.

    Input format for serving endpoint:
    {
        "dataframe_records": [
            {"periods": 30, "start_date": "2025-01-01"}
        ]
    }

    - periods: Number of periods to forecast (required)
    - start_date: Date to start forecasting from (required)
    - frequency: Optional - uses training frequency if not specified
    """

    def __init__(
        self,
        fitted_model,
        model_name: str,
        frequency: str,
        season_length: int,
        weekly_freq_code: str = None,
        last_values: np.ndarray = None,
        confidence_level: float = 0.90
    ):
        self.fitted_model = fitted_model
        self.model_name = model_name
        # Store frequency in human-readable format
        freq_to_human = {'MS': 'monthly', 'W': 'weekly', 'D': 'daily', 'YS': 'yearly'}
        self.frequency = freq_to_human.get(frequency, frequency)
        self.season_length = season_length
        self.weekly_freq_code = weekly_freq_code or 'W-MON'
        # Store last values from training for prediction
        self.last_values = last_values
        # Store confidence level for prediction intervals
        self.confidence_level = confidence_level

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

        # Generate forecast using the fitted model
        try:
            # StatsForecast models are already fitted - use predict
            forecast_df = self.fitted_model.predict(h=periods, level=[round(self.confidence_level * 100)])

            # Extract forecast values - column name varies by model
            # Use dynamic confidence level suffix (e.g., -lo-95, -hi-80) instead of hardcoded -lo-90
            conf_int = round(self.confidence_level * 100)
            forecast_cols = [c for c in forecast_df.columns if c not in ['unique_id', 'ds']]
            main_col = [c for c in forecast_cols
                        if not c.endswith(f'-lo-{conf_int}') and not c.endswith(f'-hi-{conf_int}')]

            if main_col:
                forecast_values = forecast_df[main_col[0]].values
            else:
                forecast_values = forecast_df.iloc[:, -1].values

            # Get prediction intervals
            lo_cols = [c for c in forecast_cols if c.endswith(f'-lo-{conf_int}')]
            hi_cols = [c for c in forecast_cols if c.endswith(f'-hi-{conf_int}')]

            if lo_cols and hi_cols:
                lower_bounds = forecast_df[lo_cols[0]].values
                upper_bounds = forecast_df[hi_cols[0]].values
            else:
                # Fallback: use percentage-based intervals
                lower_bounds = forecast_values * 0.9
                upper_bounds = forecast_values * 1.1

        except Exception as e:
            logger.warning(f"StatsForecast predict failed, using fallback: {e}")
            # Fallback: use naive forecast from last values
            if self.last_values is not None and len(self.last_values) > 0:
                last_val = self.last_values[-1]
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


@log_io
def _prepare_statsforecast_data(
    train_df: pd.DataFrame,
    unique_id: str = 'series_1'
) -> pd.DataFrame:
    """Convert training data to StatsForecast format.

    StatsForecast expects columns: unique_id, ds, y
    """
    sf_df = pd.DataFrame({
        'unique_id': unique_id,
        'ds': pd.to_datetime(train_df['ds']),
        'y': train_df['y'].astype(float)
    })
    return sf_df.sort_values('ds').reset_index(drop=True)


@log_io
def train_statsforecast_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    frequency: str = 'monthly',
    random_seed: int = 42,
    original_data: List[Dict[str, Any]] = None,
    covariates: List[str] = None,  # Kept for API compatibility
    hyperparameter_filters: Dict[str, Any] = None,
    forecast_start_date: pd.Timestamp = None,
    model_type: str = 'auto'  # 'auto', 'autoarima', 'autoets', 'autotheta', or 'all'
) -> Tuple[str, str, Dict[str, float], pd.DataFrame, pd.DataFrame, str, Dict[str, Any]]:
    """
    Train StatsForecast models (AutoARIMA, AutoETS, AutoTheta) with MLflow logging.

    StatsForecast provides 10-100x faster training compared to pmdarima/statsmodels.

    Args:
        train_df: Training DataFrame with 'ds' and 'y' columns
        test_df: Test DataFrame for validation
        horizon: Number of periods to forecast
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        random_seed: Random seed for reproducibility
        original_data: Original data for logging
        covariates: Not used (StatsForecast models are univariate)
        hyperparameter_filters: Optional hyperparameter constraints
        forecast_start_date: Date to start forecast from
        model_type: Which model(s) to train

    Returns:
        Tuple of (run_id, model_uri, metrics, validation_df, forecast_df, artifact_uri, model_info)
    """
    logger.info(f"Training StatsForecast model (freq={frequency}, seed={random_seed}, type={model_type})...")

    # Set random seeds for reproducibility
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Extract confidence level for prediction intervals (default 0.95)
    global_filters = (hyperparameter_filters or {}).get('_global', {})
    confidence_level = global_filters.get('confidence_level', 0.95)

    # Import StatsForecast
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
    except ImportError as e:
        raise ImportError(
            f"StatsForecast import failed: {e}. Install/reinstall with: pip install statsforecast>=1.7.0"
        ) from e

    # Detect weekly frequency code for proper date alignment
    pd_freq = detect_weekly_freq_code(train_df, frequency)
    season_length = _SEASON_LENGTH_MAP.get(frequency, 12)

    # Prepare data in StatsForecast format
    sf_train_df = _prepare_statsforecast_data(train_df)

    # Select models based on model_type
    if model_type == 'autoarima':
        models = [AutoARIMA(season_length=season_length)]
        model_names = ['AutoARIMA']
    elif model_type == 'autoets':
        models = [AutoETS(season_length=season_length)]
        model_names = ['AutoETS']
    elif model_type == 'autotheta':
        models = [AutoTheta(season_length=season_length)]
        model_names = ['AutoTheta']
    elif model_type == 'all':
        models = [
            AutoARIMA(season_length=season_length),
            AutoETS(season_length=season_length),
            AutoTheta(season_length=season_length)
        ]
        model_names = ['AutoARIMA', 'AutoETS', 'AutoTheta']
    else:  # 'auto' - use AutoARIMA as default
        models = [AutoARIMA(season_length=season_length)]
        model_names = ['AutoARIMA']

    best_model = None
    best_sf = None
    best_metrics = {"mape": float('inf'), "rmse": float('inf')}
    best_model_name = None

    with mlflow.start_run(run_name="StatsForecast_Training", nested=True) as parent_run:
        parent_run_id = parent_run.info.run_id
        experiment_id = parent_run.info.experiment_id

        # Log original data if provided
        if original_data:
            try:
                original_df = pd.DataFrame(original_data)
                original_df.to_csv("/tmp/original_timeseries_data.csv", index=False)
                mlflow.log_artifact("/tmp/original_timeseries_data.csv", "datasets/raw")
            except Exception as e:
                logger.warning(f"Could not log original data: {e}")

        # Get number of parallel workers from environment
        n_jobs = int(os.environ.get('MLFLOW_MAX_WORKERS', '1'))

        # Train each model and evaluate
        for model, name in zip(models, model_names):
            try:
                logger.info(f"  Training {name}...")

                # Create StatsForecast instance with this model
                sf = StatsForecast(
                    models=[model],
                    freq=_FREQ_MAP.get(frequency, 'MS'),
                    n_jobs=n_jobs
                )

                # Fit the model
                sf.fit(sf_train_df)

                # Predict on test set length for validation
                test_len = len(test_df)
                val_forecast = sf.predict(h=test_len, level=[round(confidence_level * 100)])

                # Extract predictions
                # Dynamically match confidence level columns (e.g., -lo-95, -hi-80)
                conf_int = round(confidence_level * 100)
                pred_cols = [c for c in val_forecast.columns
                             if c not in ['unique_id', 'ds']
                             and not c.endswith(f'-lo-{conf_int}')
                             and not c.endswith(f'-hi-{conf_int}')]

                if pred_cols:
                    predictions = val_forecast[pred_cols[0]].values
                else:
                    predictions = val_forecast.iloc[:, -1].values

                # Compute metrics
                actuals = test_df['y'].values[:len(predictions)]
                predictions = predictions[:len(actuals)]
                metrics = compute_metrics(actuals, predictions)

                logger.info(f"  ✓ {name}: MAPE={metrics['mape']:.2f}%, RMSE={metrics['rmse']:.2f}")

                # Track best model
                if metrics['mape'] < best_metrics['mape']:
                    best_metrics = metrics
                    best_model = model
                    best_sf = sf
                    best_model_name = name

                # Log metrics for this model
                mlflow.log_metrics({
                    f"{name}_mape": metrics['mape'],
                    f"{name}_rmse": metrics['rmse'],
                    f"{name}_r2": metrics['r2']
                })

            except Exception as e:
                logger.warning(f"  ✗ {name} failed: {e}")
                continue

        if best_sf is None:
            raise Exception("All StatsForecast models failed to train")

        logger.info(f"  ✨ Best model: {best_model_name} (MAPE={best_metrics['mape']:.2f}%)")

        # Time Series Cross-Validation
        try:
            full_y = pd.concat([train_df, test_df])['y'].values

            def sf_fit_fn(y_train):
                temp_df = pd.DataFrame({
                    'unique_id': 'series_1',
                    'ds': pd.date_range(end='2024-01-01', periods=len(y_train), freq=_FREQ_MAP.get(frequency, 'MS')),
                    'y': y_train
                })
                temp_sf = StatsForecast(
                    models=[type(best_model)(season_length=season_length)],
                    freq=_FREQ_MAP.get(frequency, 'MS'),
                    n_jobs=1
                )
                temp_sf.fit(temp_df)
                return temp_sf

            def sf_predict_fn(fitted_sf, steps):
                pred = fitted_sf.predict(h=steps)
                pred_cols = [c for c in pred.columns if c not in ['unique_id', 'ds']]
                return pred[pred_cols[0]].values if pred_cols else pred.iloc[:, -1].values

            cv_results = time_series_cross_validate(
                y=full_y,
                model_fit_fn=sf_fit_fn,
                model_predict_fn=sf_predict_fn,
                n_splits=3,
                horizon=min(horizon, len(test_df))
            )

            if cv_results["mean_mape"] is not None:
                logger.info(f"CV Results: Mean MAPE={cv_results['mean_mape']:.2f}% (±{cv_results['std_mape']:.2f}%)")
                best_metrics["cv_mape"] = cv_results["mean_mape"]
                best_metrics["cv_mape_std"] = cv_results["std_mape"]
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            best_metrics["cv_mape"] = None
            best_metrics["cv_mape_std"] = None

        # Generate validation data
        test_len = len(test_df)
        val_forecast = best_sf.predict(h=test_len, level=[round(confidence_level * 100)])

        conf_int = round(confidence_level * 100)
        pred_cols = [c for c in val_forecast.columns
                     if c not in ['unique_id', 'ds']
                     and not c.endswith(f'-lo-{conf_int}')
                     and not c.endswith(f'-hi-{conf_int}')]
        lo_cols = [c for c in val_forecast.columns if c.endswith(f'-lo-{conf_int}')]
        hi_cols = [c for c in val_forecast.columns if c.endswith(f'-hi-{conf_int}')]

        val_predictions = val_forecast[pred_cols[0]].values if pred_cols else val_forecast.iloc[:, -1].values
        val_lower = val_forecast[lo_cols[0]].values if lo_cols else val_predictions * 0.9
        val_upper = val_forecast[hi_cols[0]].values if hi_cols else val_predictions * 1.1

        validation_data = test_df[['ds', 'y']].copy()
        validation_data['yhat'] = val_predictions[:len(validation_data)]
        validation_data['yhat_lower'] = val_lower[:len(validation_data)]
        validation_data['yhat_upper'] = val_upper[:len(validation_data)]

        # Generate future forecast
        fcst_result = best_sf.predict(h=horizon, level=[round(confidence_level * 100)])

        fcst_predictions = fcst_result[pred_cols[0]].values if pred_cols else fcst_result.iloc[:, -1].values
        fcst_lower = fcst_result[lo_cols[0]].values if lo_cols else fcst_predictions * 0.9
        fcst_upper = fcst_result[hi_cols[0]].values if hi_cols else fcst_predictions * 1.1

        # Check for flat forecast
        flat_check = detect_flat_forecast(fcst_predictions, train_df['y'].values)
        if flat_check['is_flat']:
            logger.warning(f"StatsForecast flat forecast detected: {flat_check['flat_reason']}")

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
        logger.info(f"📅 StatsForecast forecast dates: {future_dates.min()} to {future_dates.max()}")

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
        mlflow.log_param("model_type", "StatsForecast")
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("season_length", season_length)
        mlflow.log_param("frequency", frequency)
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("confidence_level", confidence_level)
        mlflow.log_metrics(best_metrics)

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

            model_wrapper = StatsForecastModelWrapper(
                fitted_model=best_sf,
                model_name=best_model_name,
                frequency=frequency,
                season_length=season_length,
                weekly_freq_code=weekly_freq_code,
                last_values=train_df['y'].values,
                confidence_level=confidence_level
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
                        {"pip": ["mlflow", "pandas", "numpy", "statsforecast>=1.7.0"]}
                    ],
                    "name": "statsforecast_env"
                }
            )

            artifact_uri = mlflow.get_artifact_uri("model")
            logger.info(f"   ✅ StatsForecast model logged to: {artifact_uri}")

            # Save backup
            model_backup_path = "/tmp/statsforecast_model_backup.pkl"
            with open(model_backup_path, 'wb') as f:
                pickle.dump({
                    'model': best_sf,
                    'wrapper': model_wrapper,
                    'model_name': best_model_name,
                    'frequency': frequency,
                    'season_length': season_length
                }, f)
            mlflow.log_artifact(model_backup_path, "model_backup")

        except Exception as e:
            logger.error(f"   ❌ Failed to log StatsForecast pyfunc model: {e}")
            try:
                model_path = "/tmp/statsforecast_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': best_sf,
                        'model_name': best_model_name,
                        'frequency': frequency,
                        'season_length': season_length
                    }, f)
                mlflow.log_artifact(model_path, "model")
                logger.warning("   ⚠️ Logged StatsForecast model as pickle fallback")
            except Exception as fallback_error:
                logger.error(f"   ❌ Fallback pickle also failed: {fallback_error}")

        best_artifact_uri = parent_run.info.artifact_uri

    model_info = {
        'model_name': best_model_name,
        'season_length': season_length,
        'frequency': frequency
    }

    return parent_run_id, f"runs:/{parent_run_id}/model", best_metrics, validation_data, forecast_data, best_artifact_uri, model_info
