"""
Training Presets for Forecasting Models

Implements AutoGluon-style presets for automatic model selection and configuration.
Provides pre-configured combinations of models, hyperparameters, and training settings
optimized for different use cases (speed vs accuracy).

Author: debu-sinha
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from backend.utils.logging_utils import log_io

logger = logging.getLogger(__name__)


class PresetLevel(Enum):
    """Training preset levels from fastest to most accurate."""
    FAST = "fast"
    MEDIUM = "medium"
    HIGH_QUALITY = "high_quality"
    BEST = "best"


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    enabled: bool = True
    max_combinations: int = 10
    cv_folds: int = 3
    hyperparameter_grid: Dict[str, List[Any]] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None


@dataclass
class PresetConfig:
    """Configuration for a training preset."""
    name: str
    description: str
    models: List[str]
    cv_folds: int
    max_time_per_model: Optional[int]  # seconds, None = unlimited
    hyperparameter_tuning: str  # 'minimal', 'standard', 'extensive', 'exhaustive'
    ensemble_enabled: bool
    max_combinations_per_model: int
    model_configs: Dict[str, ModelConfig] = field(default_factory=dict)

    # Data quality settings
    auto_fix_data_quality: bool = True
    filter_overfitting: bool = True
    max_overfitting_severity: str = 'high'

    # Ensemble settings
    ensemble_weighting: str = 'inverse_mape'
    ensemble_min_weight: float = 0.05
    ensemble_max_mape_ratio: float = 3.0


# =============================================================================
# PRESET DEFINITIONS
# =============================================================================

PRESETS: Dict[PresetLevel, PresetConfig] = {
    PresetLevel.FAST: PresetConfig(
        name="fast",
        description="Quick baseline forecasts in 2-3 minutes. Best for rapid prototyping.",
        models=['arima', 'statsforecast'],
        cv_folds=2,
        max_time_per_model=60,
        hyperparameter_tuning='minimal',
        ensemble_enabled=False,
        max_combinations_per_model=5,
        model_configs={
            'arima': ModelConfig(
                max_combinations=5,
                cv_folds=2,
                hyperparameter_grid={
                    'p_values': [0, 1],
                    'd_values': [0, 1],
                    'q_values': [0, 1],
                }
            ),
            'statsforecast': ModelConfig(
                max_combinations=3,
                cv_folds=2,
                hyperparameter_grid={
                    'model_type': ['auto'],
                }
            ),
        },
        filter_overfitting=False,  # Skip for speed
    ),

    PresetLevel.MEDIUM: PresetConfig(
        name="medium",
        description="Balanced speed and accuracy. Good for most use cases. ~5-10 minutes.",
        models=['prophet', 'arima', 'statsforecast', 'ets'],
        cv_folds=3,
        max_time_per_model=180,
        hyperparameter_tuning='standard',
        ensemble_enabled=True,
        max_combinations_per_model=10,
        model_configs={
            'prophet': ModelConfig(
                max_combinations=8,
                cv_folds=3,
                hyperparameter_grid={
                    'changepoint_prior_scale': [0.01, 0.1, 0.5],
                    'seasonality_prior_scale': [0.1, 1.0, 10.0],
                    'seasonality_mode': ['additive', 'multiplicative'],
                }
            ),
            'arima': ModelConfig(
                max_combinations=10,
                cv_folds=3,
                hyperparameter_grid={
                    'p_values': [0, 1, 2],
                    'd_values': [0, 1],
                    'q_values': [0, 1, 2],
                }
            ),
            'statsforecast': ModelConfig(
                max_combinations=5,
                cv_folds=3,
            ),
            'ets': ModelConfig(
                max_combinations=5,
                cv_folds=3,
            ),
        },
        ensemble_max_mape_ratio=3.0,
    ),

    PresetLevel.HIGH_QUALITY: PresetConfig(
        name="high_quality",
        description="Production-quality forecasts with thorough tuning. ~15-30 minutes.",
        models=['prophet', 'arima', 'sarimax', 'xgboost', 'statsforecast', 'ets'],
        cv_folds=5,
        max_time_per_model=600,
        hyperparameter_tuning='extensive',
        ensemble_enabled=True,
        max_combinations_per_model=20,
        model_configs={
            'prophet': ModelConfig(
                max_combinations=16,
                cv_folds=5,
                hyperparameter_grid={
                    'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
                    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                    'seasonality_mode': ['additive', 'multiplicative'],
                    'yearly_seasonality': [True, False],
                }
            ),
            'arima': ModelConfig(
                max_combinations=20,
                cv_folds=5,
                hyperparameter_grid={
                    'p_values': [0, 1, 2, 3],
                    'd_values': [0, 1, 2],
                    'q_values': [0, 1, 2, 3],
                }
            ),
            'sarimax': ModelConfig(
                max_combinations=15,
                cv_folds=5,
                hyperparameter_grid={
                    'p_values': [0, 1, 2],
                    'd_values': [0, 1],
                    'q_values': [0, 1, 2],
                    'P_values': [0, 1],
                    'D_values': [0, 1],
                    'Q_values': [0, 1],
                }
            ),
            'xgboost': ModelConfig(
                max_combinations=15,
                cv_folds=5,
                hyperparameter_grid={
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                }
            ),
            'statsforecast': ModelConfig(
                max_combinations=8,
                cv_folds=5,
            ),
            'ets': ModelConfig(
                max_combinations=8,
                cv_folds=5,
            ),
        },
        filter_overfitting=True,
        max_overfitting_severity='high',
        ensemble_max_mape_ratio=2.5,
    ),

    PresetLevel.BEST: PresetConfig(
        name="best",
        description="Maximum accuracy with exhaustive tuning. 30+ minutes. Best for final production models.",
        models=['prophet', 'arima', 'sarimax', 'xgboost', 'statsforecast', 'ets', 'chronos'],
        cv_folds=5,
        max_time_per_model=None,  # No limit
        hyperparameter_tuning='exhaustive',
        ensemble_enabled=True,
        max_combinations_per_model=50,
        model_configs={
            'prophet': ModelConfig(
                max_combinations=32,
                cv_folds=5,
                hyperparameter_grid={
                    'changepoint_prior_scale': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'seasonality_mode': ['additive', 'multiplicative'],
                    'yearly_seasonality': [True, False],
                    'weekly_seasonality': [True, False],
                }
            ),
            'arima': ModelConfig(
                max_combinations=40,
                cv_folds=5,
                hyperparameter_grid={
                    'p_values': [0, 1, 2, 3, 4],
                    'd_values': [0, 1, 2],
                    'q_values': [0, 1, 2, 3, 4],
                }
            ),
            'sarimax': ModelConfig(
                max_combinations=30,
                cv_folds=5,
            ),
            'xgboost': ModelConfig(
                max_combinations=25,
                cv_folds=5,
                hyperparameter_grid={
                    'n_estimators': [50, 100, 200, 500],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'subsample': [0.8, 1.0],
                }
            ),
            'statsforecast': ModelConfig(
                max_combinations=10,
                cv_folds=5,
            ),
            'ets': ModelConfig(
                max_combinations=10,
                cv_folds=5,
            ),
            'chronos': ModelConfig(
                max_combinations=3,
                cv_folds=3,
                hyperparameter_grid={
                    'model_size': ['small', 'base'],
                }
            ),
        },
        filter_overfitting=True,
        max_overfitting_severity='medium',  # Stricter filtering
        ensemble_max_mape_ratio=2.0,  # Only include good models
    ),
}


# =============================================================================
# PRESET FUNCTIONS
# =============================================================================

@log_io
def get_preset(preset: str) -> PresetConfig:
    """
    Get preset configuration by name.

    Args:
        preset: Preset name ('fast', 'medium', 'high_quality', 'best')

    Returns:
        PresetConfig object
    """
    try:
        preset_level = PresetLevel(preset.lower())
        return PRESETS[preset_level]
    except (ValueError, KeyError):
        logger.warning(f"Unknown preset '{preset}', using 'medium'")
        return PRESETS[PresetLevel.MEDIUM]


@log_io
def get_preset_models(preset: str) -> List[str]:
    """Get list of models for a preset."""
    return get_preset(preset).models


@log_io
def get_preset_description(preset: str) -> str:
    """Get description for a preset."""
    return get_preset(preset).description


@log_io
def get_model_config(preset: str, model_name: str) -> ModelConfig:
    """
    Get model-specific configuration for a preset.

    Args:
        preset: Preset name
        model_name: Model name (e.g., 'prophet', 'arima')

    Returns:
        ModelConfig for the model, or default if not specified
    """
    preset_config = get_preset(preset)
    return preset_config.model_configs.get(
        model_name.lower(),
        ModelConfig()  # Default config
    )


@log_io
def apply_preset_to_hyperparameters(
    preset: str,
    model_name: str,
    base_hyperparameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply preset-specific hyperparameter overrides.

    Args:
        preset: Preset name
        model_name: Model name
        base_hyperparameters: Base hyperparameters from user/default

    Returns:
        Modified hyperparameters dict
    """
    model_config = get_model_config(preset, model_name)

    # Merge preset hyperparameters with base (preset takes precedence)
    result = base_hyperparameters.copy()
    for key, values in model_config.hyperparameter_grid.items():
        result[key] = values

    return result


@log_io
def recommend_preset(
    n_rows: int,
    frequency: str,
    time_budget_minutes: Optional[int] = None
) -> str:
    """
    Recommend a preset based on data characteristics and time budget.

    Args:
        n_rows: Number of rows in dataset
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        time_budget_minutes: Optional time budget in minutes

    Returns:
        Recommended preset name
    """
    # Time-based recommendation
    if time_budget_minutes is not None:
        if time_budget_minutes < 5:
            return 'fast'
        elif time_budget_minutes < 15:
            return 'medium'
        elif time_budget_minutes < 45:
            return 'high_quality'
        else:
            return 'best'

    # Data-size based recommendation
    if n_rows < 50:
        return 'fast'  # Limited data, complex models won't help
    elif n_rows < 100:
        return 'medium'
    elif n_rows < 200:
        return 'high_quality'
    else:
        return 'best'  # Enough data for complex models


@log_io
def log_preset_info(preset: str):
    """Log information about the selected preset."""
    config = get_preset(preset)
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"📋 TRAINING PRESET: {config.name.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"   Description: {config.description}")
    logger.info(f"   Models: {', '.join(config.models)}")
    logger.info(f"   CV Folds: {config.cv_folds}")
    logger.info(f"   Max combinations per model: {config.max_combinations_per_model}")
    logger.info(f"   Ensemble enabled: {config.ensemble_enabled}")
    logger.info(f"   Hyperparameter tuning: {config.hyperparameter_tuning}")
    if config.max_time_per_model:
        logger.info(f"   Max time per model: {config.max_time_per_model}s")
    logger.info(f"{'='*60}")
