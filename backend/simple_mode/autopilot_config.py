"""
Autopilot Configuration Generator.

Generates optimal configuration based on data profile.
User doesn't need to set any parameters.
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .data_profiler import DataProfile

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    model_type: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastConfig:
    """Complete forecast configuration (auto-generated or manual)."""

    # Data configuration (auto-detected)
    frequency: str
    date_column: str
    target_column: str
    covariate_columns: List[str]

    # Forecast settings
    horizon: int

    # Model selection
    models: List[str]
    model_configs: Dict[str, ModelConfig]

    # Reproducibility settings
    random_seed: int = 42

    # Metadata
    config_version: str = "1.0"
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_mode: str = "autopilot"  # "autopilot" or "expert"

    # Hash for reproducibility
    config_hash: str = ""

    def __post_init__(self):
        """Compute config hash after initialization."""
        if not self.config_hash:
            self.config_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of this configuration."""
        config_str = (
            f"{self.frequency}:{self.date_column}:{self.target_column}:"
            f"{sorted(self.covariate_columns)}:{self.horizon}:"
            f"{sorted(self.models)}:{self.random_seed}"
        )
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""

        def _convert_value(val):
            """Convert numpy types to native Python types for JSON serialization."""
            import numpy as np
            if isinstance(val, np.bool_):
                return bool(val)
            elif isinstance(val, np.integer):
                return int(val)
            elif isinstance(val, np.floating):
                return float(val)
            elif isinstance(val, dict):
                return {k: _convert_value(v) for k, v in val.items()}
            elif isinstance(val, (list, tuple)):
                return [_convert_value(v) for v in val]
            return val

        return {
            'frequency': self.frequency,
            'date_column': self.date_column,
            'target_column': self.target_column,
            'covariate_columns': self.covariate_columns,
            'horizon': int(self.horizon),
            'models': self.models,
            'model_configs': {
                k: {
                    'model_type': v.model_type,
                    'enabled': bool(v.enabled),
                    'params': _convert_value(v.params)
                }
                for k, v in self.model_configs.items()
            },
            'random_seed': int(self.random_seed),
            'config_version': self.config_version,
            'generated_at': self.generated_at,
            'generation_mode': self.generation_mode,
            'config_hash': self.config_hash,
        }


class AutopilotConfig:
    """
    Generates optimal configuration based on data profile.
    User doesn't need to set any parameters.
    """

    def generate(
        self,
        profile: DataProfile,
        horizon: Optional[int] = None
    ) -> ForecastConfig:
        """
        Generate best configuration for this data.

        Args:
            profile: DataProfile from DataProfiler
            horizon: Optional override for forecast horizon

        Returns:
            ForecastConfig with optimal settings
        """
        logger.info("Generating autopilot configuration...")

        # Use profile recommendations or override
        final_horizon = horizon if horizon else profile.recommended_horizon
        models = profile.recommended_models

        # Generate optimal config for each model
        model_configs = {}
        for model in models:
            model_configs[model] = self._get_optimal_params(model, profile)

        config = ForecastConfig(
            # Auto-determined settings
            frequency=profile.frequency,
            date_column=profile.date_column,
            target_column=profile.target_column,
            covariate_columns=profile.covariate_columns,

            # Forecast settings
            horizon=final_horizon,

            # Model selection
            models=models,
            model_configs=model_configs,

            # Reproducibility (fixed seed)
            random_seed=42,

            # Metadata
            generation_mode="autopilot",
        )

        logger.info(f"Generated config: {len(models)} models, horizon={final_horizon}")
        logger.info(f"Config hash: {config.config_hash}")

        return config

    def _get_optimal_params(
        self, model_type: str, profile: DataProfile
    ) -> ModelConfig:
        """Get optimal parameters for a specific model type."""

        if model_type == 'prophet':
            return self._get_prophet_params(profile)
        elif model_type == 'arima':
            return self._get_arima_params(profile)
        elif model_type == 'xgboost':
            return self._get_xgboost_params(profile)
        elif model_type == 'ets':
            return self._get_ets_params(profile)
        elif model_type == 'sarimax':
            return self._get_sarimax_params(profile)
        else:
            return ModelConfig(model_type=model_type, params={})

    def _get_prophet_params(self, profile: DataProfile) -> ModelConfig:
        """Optimal Prophet parameters based on data profile."""

        params = {
            # Seasonality mode: multiplicative for data with varying magnitude
            'seasonality_mode': 'multiplicative',

            # Yearly seasonality: enable if we have enough data
            'yearly_seasonality': profile.history_months >= 12,

            # Weekly seasonality: enable for daily data
            'weekly_seasonality': profile.frequency == 'daily',

            # Growth: linear is usually sufficient
            'growth': 'linear',

            # Country holidays: enable for US
            'country': 'US',

            # Changepoint prior: higher = more flexible trend
            'changepoint_prior_scale': 0.05 if profile.has_trend else 0.01,

            # Seasonality prior: higher = more flexible seasonality
            'seasonality_prior_scale': 10.0 if profile.has_seasonality else 1.0,
        }

        return ModelConfig(model_type='prophet', params=params)

    def _get_arima_params(self, profile: DataProfile) -> ModelConfig:
        """Optimal ARIMA parameters based on data profile."""

        # ARIMA orders (p, d, q) - let auto_arima find optimal
        params = {
            'auto_arima': True,
            'max_p': 5,
            'max_q': 5,
            'max_d': 2,
            'seasonal': profile.has_seasonality,
            'stepwise': True,  # Faster search
        }

        if profile.has_seasonality and profile.seasonality_period:
            params['m'] = profile.seasonality_period

        return ModelConfig(model_type='arima', params=params)

    def _get_xgboost_params(self, profile: DataProfile) -> ModelConfig:
        """Optimal XGBoost parameters based on data profile."""

        # XGBoost benefits from more data
        n_estimators = 100 if profile.history_months < 24 else 200

        params = {
            'n_estimators': n_estimators,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,

            # Enable categorical covariates
            'enable_categorical': len(profile.covariate_columns) > 0,
        }

        return ModelConfig(model_type='xgboost', params=params)

    def _get_ets_params(self, profile: DataProfile) -> ModelConfig:
        """Optimal ETS parameters based on data profile."""

        params = {
            'error': 'add',  # Additive error
            'trend': 'add' if profile.has_trend else None,
            'seasonal': 'add' if profile.has_seasonality else None,
            'damped_trend': True if profile.has_trend else False,
        }

        if profile.has_seasonality and profile.seasonality_period:
            params['seasonal_periods'] = profile.seasonality_period

        return ModelConfig(model_type='ets', params=params)

    def _get_sarimax_params(self, profile: DataProfile) -> ModelConfig:
        """Optimal SARIMAX parameters based on data profile."""

        params = {
            'order': (1, 1, 1),  # Default ARIMA order
            'enforce_stationarity': False,
            'enforce_invertibility': False,
        }

        if profile.has_seasonality and profile.seasonality_period:
            params['seasonal_order'] = (1, 1, 1, profile.seasonality_period)

        return ModelConfig(model_type='sarimax', params=params)


def generate_reproducibility_token(
    data_hash: str, config_hash: str, model_version: str = "1.0"
) -> str:
    """
    Generate a reproducibility token.
    Same token = guaranteed same output.
    """
    return f"{data_hash}:{config_hash}:{model_version}"


def generate_hyperparameter_filters(profile: DataProfile) -> Dict[str, Dict[str, Any]]:
    """
    Generate intelligent hyperparameter filters based on data profile.

    These filters are passed to model training functions to reduce the search space
    based on data characteristics. This makes training faster and more targeted.

    Args:
        profile: DataProfile from DataProfiler with detected data characteristics

    Returns:
        Dictionary of model name -> hyperparameter constraints
        Format matches what backend/models/*.py expect
    """
    filters = {}

    # Calculate derived metrics
    n_observations = profile.total_periods
    n_years = profile.history_months / 12.0

    logger.info(f"Generating hyperparameter filters for {n_observations} observations, {n_years:.1f} years")

    # ============================================================
    # PROPHET HYPERPARAMETERS
    # ============================================================
    prophet_hp = {}

    # Adjust search space based on data size
    if n_observations < 52:
        # Small dataset: use simpler models to avoid overfitting
        prophet_hp['changepoint_prior_scale'] = [0.01, 0.05, 0.1]
        prophet_hp['seasonality_prior_scale'] = [0.1, 1.0]
    elif n_observations < 104:
        # Medium dataset: moderate search space
        prophet_hp['changepoint_prior_scale'] = [0.01, 0.05, 0.1, 0.5]
        prophet_hp['seasonality_prior_scale'] = [0.1, 1.0, 10.0]
    else:
        # Large dataset: can handle more flexibility
        prophet_hp['changepoint_prior_scale'] = [0.01, 0.05, 0.1, 0.5, 1.0]
        prophet_hp['seasonality_prior_scale'] = [0.1, 1.0, 10.0]

    # Yearly seasonality: only enable if we have enough data
    if n_years < 2:
        prophet_hp['yearly_seasonality'] = [False]

    # Weekly seasonality based on frequency
    if profile.frequency == 'daily':
        prophet_hp['weekly_seasonality'] = [True]
    elif profile.frequency == 'monthly':
        prophet_hp['weekly_seasonality'] = [False]

    # Seasonality mode based on detected patterns
    # Multiplicative: seasonal amplitude grows with trend (common in financial data)
    # Additive: seasonal amplitude stays constant
    if profile.has_trend and profile.has_seasonality:
        prophet_hp['seasonality_mode'] = ['multiplicative']
    elif profile.has_seasonality:
        prophet_hp['seasonality_mode'] = ['additive', 'multiplicative']
    else:
        prophet_hp['seasonality_mode'] = ['additive']

    filters['Prophet'] = prophet_hp

    # ============================================================
    # ARIMA HYPERPARAMETERS
    # ============================================================
    arima_hp = {}

    if n_observations < 50:
        # Small dataset: constrain search space
        arima_hp['p_values'] = [0, 1, 2]
        arima_hp['d_values'] = [0, 1]
        arima_hp['q_values'] = [0, 1, 2]
    elif n_observations < 100:
        arima_hp['p_values'] = [0, 1, 2, 3]
        arima_hp['d_values'] = [0, 1, 2]
        arima_hp['q_values'] = [0, 1, 2, 3]
    else:
        # Larger dataset: allow fuller search
        arima_hp['p_values'] = [0, 1, 2, 3, 4]
        arima_hp['d_values'] = [0, 1, 2]
        arima_hp['q_values'] = [0, 1, 2, 3, 4]

    filters['ARIMA'] = arima_hp

    # ============================================================
    # SARIMAX HYPERPARAMETERS (for models with covariates)
    # ============================================================
    sarimax_hp = {}

    if n_observations < 50:
        sarimax_hp['p_values'] = [0, 1, 2]
        sarimax_hp['d_values'] = [0, 1]
        sarimax_hp['q_values'] = [0, 1, 2]
    elif n_observations < 100:
        sarimax_hp['p_values'] = [0, 1, 2, 3]
        sarimax_hp['d_values'] = [0, 1, 2]
        sarimax_hp['q_values'] = [0, 1, 2, 3]
    else:
        sarimax_hp['p_values'] = [0, 1, 2, 3]
        sarimax_hp['d_values'] = [0, 1, 2]
        sarimax_hp['q_values'] = [0, 1, 2, 3]

    filters['SARIMAX'] = sarimax_hp

    # ============================================================
    # ETS (EXPONENTIAL SMOOTHING) HYPERPARAMETERS
    # ============================================================
    ets_hp = {}

    # ETS trend options based on detected trend
    if profile.has_trend:
        ets_hp['trend'] = ['add', 'mul']  # Try both additive and multiplicative
        ets_hp['damped_trend'] = [True, False]
    else:
        ets_hp['trend'] = [None, 'add']  # Simpler options
        ets_hp['damped_trend'] = [False]

    # ETS seasonal options based on detected seasonality
    if profile.has_seasonality:
        ets_hp['seasonal'] = ['add', 'mul']
    else:
        ets_hp['seasonal'] = [None]

    # Seasonal periods if detected
    if profile.seasonality_period:
        ets_hp['seasonal_periods'] = [profile.seasonality_period]

    filters['ETS'] = ets_hp

    # ============================================================
    # XGBOOST HYPERPARAMETERS
    # ============================================================
    xgb_hp = {}

    if n_observations < 100:
        # Small dataset: simpler models
        xgb_hp['n_estimators'] = [50, 100]
        xgb_hp['max_depth'] = [3]
        xgb_hp['learning_rate'] = [0.1]
    elif n_observations < 500:
        # Medium dataset
        xgb_hp['n_estimators'] = [100, 200]
        xgb_hp['max_depth'] = [3, 5]
        xgb_hp['learning_rate'] = [0.05, 0.1]
    else:
        # Large dataset: can handle more complexity
        xgb_hp['n_estimators'] = [100, 200, 300]
        xgb_hp['max_depth'] = [3, 5, 7]
        xgb_hp['learning_rate'] = [0.01, 0.05, 0.1]

    # If data has high variance, use lower learning rate
    # (We don't have CV directly, but we can infer from data quality)
    if profile.data_quality_score < 70:
        xgb_hp['learning_rate'] = [0.01, 0.05]

    filters['XGBoost'] = xgb_hp

    logger.info(f"Generated hyperparameter filters: {list(filters.keys())}")
    for model, params in filters.items():
        logger.info(f"  {model}: {list(params.keys())}")

    return filters
