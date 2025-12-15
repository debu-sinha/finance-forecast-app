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
        return {
            'frequency': self.frequency,
            'date_column': self.date_column,
            'target_column': self.target_column,
            'covariate_columns': self.covariate_columns,
            'horizon': self.horizon,
            'models': self.models,
            'model_configs': {
                k: {'model_type': v.model_type, 'enabled': v.enabled, 'params': v.params}
                for k, v in self.model_configs.items()
            },
            'random_seed': self.random_seed,
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
