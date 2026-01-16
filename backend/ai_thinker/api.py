"""
API endpoints for AI Thinker - Opus 4.5 powered intelligent analysis.

Provides endpoints for:
- Pre-training data analysis
- Forecast explanation
- Interactive Q&A
- Anomaly investigation
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .service import ForecastThinker

logger = logging.getLogger(__name__)

# Create router with /api/thinker prefix
thinker_router = APIRouter(prefix="/api/thinker", tags=["AI Thinker"])

# Initialize thinker service (singleton)
_thinker: Optional[ForecastThinker] = None


def get_thinker() -> ForecastThinker:
    """Get or create the thinker instance."""
    global _thinker
    if _thinker is None:
        _thinker = ForecastThinker()
    return _thinker


# =============================================================================
# Request/Response Models
# =============================================================================

class DataAnalysisRequest(BaseModel):
    """Request for pre-training data analysis."""
    profile: dict = Field(..., description="Data profile from profiler")
    sample_data: List[dict] = Field(..., description="Sample data rows")
    target_col: str = Field(..., description="Target column name")
    date_col: str = Field(..., description="Date column name")
    covariates: List[str] = Field(default_factory=list, description="Covariate columns")


class ForecastExplanationRequest(BaseModel):
    """Request for forecast explanation."""
    forecast_result: dict = Field(..., description="Forecast result from training")
    training_result: dict = Field(..., description="Training result metadata")
    profile: dict = Field(..., description="Data profile")
    target_col: str = Field(..., description="Target column name")
    covariates: List[str] = Field(default_factory=list, description="Covariates used")


class QuestionRequest(BaseModel):
    """Request for Q&A."""
    question: str = Field(..., description="Analyst question")
    forecast_context: dict = Field(..., description="Current forecast context")
    data_context: dict = Field(..., description="Data context")
    conversation_history: Optional[List[dict]] = Field(
        default=None, description="Previous Q&A in this session"
    )


class AnomalyRequest(BaseModel):
    """Request for anomaly investigation."""
    anomaly: dict = Field(..., description="Anomaly details")
    historical_data: List[dict] = Field(..., description="Historical data around anomaly")
    forecast_data: List[dict] = Field(..., description="Forecast data")
    covariates: List[str] = Field(default_factory=list, description="Covariates")


class ThinkerStatusResponse(BaseModel):
    """Response for thinker status check."""
    available: bool
    model: str
    endpoint: str
    cache_size: int


# =============================================================================
# Endpoints
# =============================================================================

@thinker_router.get("/status", response_model=ThinkerStatusResponse)
async def get_thinker_status() -> ThinkerStatusResponse:
    """Check if the Opus 4.5 thinker service is available."""
    thinker = get_thinker()
    status = thinker.get_status()
    return ThinkerStatusResponse(**status)


@thinker_router.post("/analyze-data")
async def analyze_data_for_training(request: DataAnalysisRequest):
    """
    Deep analysis of data before model training.

    Uses Opus 4.5 extended thinking to analyze:
    - Data quality and patterns
    - Seasonality and trends
    - Holiday effects
    - Model recommendations

    Returns comprehensive DataInsights with model recommendations.
    """
    thinker = get_thinker()

    try:
        result = await thinker.analyze_data_for_training(
            profile=request.profile,
            sample_data=request.sample_data,
            target_col=request.target_col,
            date_col=request.date_col,
            covariates=request.covariates
        )

        return {
            "success": result.success,
            "response_type": result.response_type,
            "data": result.data,
            "thinking_time_ms": result.thinking_time_ms,
            "model_used": result.model_used,
            "tokens_used": result.tokens_used,
            "error": result.error
        }

    except Exception as e:
        logger.error(f"Data analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@thinker_router.post("/explain-forecast")
async def explain_forecast(request: ForecastExplanationRequest):
    """
    Generate comprehensive forecast explanation for analysts.

    Uses Opus 4.5 to explain:
    - What the forecast shows
    - Why it shows that (key drivers)
    - Risk factors and assumptions
    - Actionable recommendations

    Returns ForecastExplanation with executive summary and details.
    """
    thinker = get_thinker()

    try:
        result = await thinker.explain_forecast(
            forecast_result=request.forecast_result,
            training_result=request.training_result,
            profile=request.profile,
            target_col=request.target_col,
            covariates=request.covariates
        )

        return {
            "success": result.success,
            "response_type": result.response_type,
            "data": result.data,
            "thinking_time_ms": result.thinking_time_ms,
            "model_used": result.model_used,
            "tokens_used": result.tokens_used,
            "error": result.error
        }

    except Exception as e:
        logger.error(f"Forecast explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@thinker_router.post("/ask")
async def answer_question(request: QuestionRequest):
    """
    Answer analyst questions about the forecast.

    Uses Opus 4.5 to provide:
    - Contextual, data-backed answers
    - Supporting evidence
    - Follow-up question suggestions

    Supports conversation history for multi-turn Q&A.
    """
    thinker = get_thinker()

    try:
        result = await thinker.answer_question(
            question=request.question,
            forecast_context=request.forecast_context,
            data_context=request.data_context,
            conversation_history=request.conversation_history
        )

        return {
            "success": result.success,
            "response_type": result.response_type,
            "data": result.data,
            "thinking_time_ms": result.thinking_time_ms,
            "model_used": result.model_used,
            "tokens_used": result.tokens_used,
            "error": result.error
        }

    except Exception as e:
        logger.error(f"Q&A failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@thinker_router.post("/investigate-anomaly")
async def investigate_anomaly(request: AnomalyRequest):
    """
    Investigate root cause of detected anomaly.

    Uses Opus 4.5 to analyze:
    - Likely causes with probabilities
    - Supporting and contradicting evidence
    - Similar historical events
    - Recommended actions

    Returns AnomalyAnalysis with root cause investigation.
    """
    thinker = get_thinker()

    try:
        result = await thinker.investigate_anomaly(
            anomaly=request.anomaly,
            historical_data=request.historical_data,
            forecast_data=request.forecast_data,
            covariates=request.covariates
        )

        return {
            "success": result.success,
            "response_type": result.response_type,
            "data": result.data,
            "thinking_time_ms": result.thinking_time_ms,
            "model_used": result.model_used,
            "tokens_used": result.tokens_used,
            "error": result.error
        }

    except Exception as e:
        logger.error(f"Anomaly investigation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@thinker_router.post("/clear-cache")
async def clear_thinker_cache():
    """Clear the thinker response cache."""
    thinker = get_thinker()
    thinker.clear_cache()
    return {"success": True, "message": "Cache cleared"}
