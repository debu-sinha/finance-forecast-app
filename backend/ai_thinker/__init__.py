"""
AI Thinker Module - Opus 4.5 powered intelligent analysis for forecasting.

This module provides deep reasoning capabilities for:
- Pre-training data intelligence
- Forecast interpretation and explanation
- Interactive Q&A about forecasts
- Anomaly root cause analysis
"""

from .service import ForecastThinker
from .api import thinker_router
from .models import (
    DataInsights,
    ForecastExplanation,
    AnomalyAnalysis,
    ThinkerResponse
)

__all__ = [
    'ForecastThinker',
    'thinker_router',
    'DataInsights',
    'ForecastExplanation',
    'AnomalyAnalysis',
    'ThinkerResponse'
]
