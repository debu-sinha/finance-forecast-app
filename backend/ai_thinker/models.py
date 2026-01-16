"""
Data models for AI Thinker responses.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InsightType(str, Enum):
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    RECOMMENDATION = "recommendation"
    WARNING = "warning"
    OPPORTUNITY = "opportunity"


@dataclass
class Insight:
    """Single insight from the thinker."""
    type: InsightType
    title: str
    description: str
    confidence: ConfidenceLevel
    evidence: List[str] = field(default_factory=list)
    actionable: bool = False
    action: Optional[str] = None


@dataclass
class ModelRecommendation:
    """Model recommendation with reasoning."""
    model: str
    score: float  # 0-100
    reasoning: str
    strengths: List[str]
    weaknesses: List[str]
    best_for: str


@dataclass
class DataInsights:
    """Deep analysis of uploaded data before training."""
    summary: str
    data_quality_assessment: str
    pattern_analysis: str
    seasonality_insights: str
    holiday_insights: str
    trend_analysis: str
    insights: List[Insight]
    model_recommendations: List[ModelRecommendation]
    suggested_horizon: int
    suggested_covariates: List[str]
    warnings: List[str]
    opportunities: List[str]
    confidence: ConfidenceLevel
    thinking_process: Optional[str] = None  # Show reasoning chain


@dataclass
class ForecastComponent:
    """Breakdown of a forecast component."""
    name: str
    contribution: float  # Percentage or absolute
    description: str
    direction: str  # "positive", "negative", "neutral"


@dataclass
class PeriodInsight:
    """Insight for a specific forecast period."""
    period: str
    date: str
    forecast_value: float
    key_drivers: List[str]
    confidence: ConfidenceLevel
    notable_events: List[str]


@dataclass
class ForecastExplanation:
    """Comprehensive forecast explanation for analysts."""
    executive_summary: str
    methodology_explanation: str
    key_findings: List[str]
    components: List[ForecastComponent]
    period_insights: List[PeriodInsight]
    risk_factors: List[str]
    assumptions: List[str]
    confidence_assessment: str
    recommendations: List[str]
    caveats: List[str]
    thinking_process: Optional[str] = None


@dataclass
class AnomalyAnalysis:
    """Root cause analysis for detected anomalies."""
    anomaly_description: str
    severity: str  # "critical", "warning", "info"
    likely_causes: List[Dict[str, Any]]  # [{cause, probability, evidence}]
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    recommended_actions: List[str]
    similar_historical_events: List[Dict[str, Any]]
    confidence: ConfidenceLevel
    thinking_process: Optional[str] = None


@dataclass
class QAResponse:
    """Response to an analyst's question."""
    answer: str
    supporting_data: List[Dict[str, Any]]
    related_insights: List[str]
    follow_up_questions: List[str]
    confidence: ConfidenceLevel
    sources: List[str]  # What data was used to answer
    thinking_process: Optional[str] = None


@dataclass
class ThinkerResponse:
    """Generic wrapper for thinker responses."""
    success: bool
    response_type: str
    data: Any
    thinking_time_ms: int
    model_used: str
    tokens_used: Optional[int] = None
    error: Optional[str] = None
