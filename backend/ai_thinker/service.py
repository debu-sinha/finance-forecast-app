"""
Forecast Thinker Service - Opus 4.5 powered intelligent analysis.

Uses Databricks Model Serving to host Claude Opus 4.5 for deep reasoning
about forecasting data, results, and analyst questions.
"""

import os
import json
import time
import logging
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import asdict
from .models import (
    DataInsights, ForecastExplanation, AnomalyAnalysis, QAResponse,
    ThinkerResponse, ConfidenceLevel, ModelRecommendation, ForecastComponent
)

logger = logging.getLogger(__name__)

# Cache for thinker responses (avoid repeated expensive calls)
_response_cache: Dict[str, ThinkerResponse] = {}
CACHE_TTL_SECONDS = 3600  # 1 hour


class DatabricksOpusClient:
    """Client for Databricks-hosted Opus 4.5 model serving endpoint."""

    def __init__(self):
        self.workspace_url = os.environ.get("DATABRICKS_HOST", "")
        self.token = os.environ.get("DATABRICKS_TOKEN", "")
        self.endpoint_name = os.environ.get("OPUS_ENDPOINT_NAME", "opus-4-5-thinker")
        self.timeout = int(os.environ.get("OPUS_TIMEOUT_SECONDS", "60"))
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of the client."""
        if self._initialized:
            return

        if not self.workspace_url or not self.token:
            logger.warning("Databricks credentials not configured for Opus 4.5")
            self._initialized = True
            return

        # Try to import databricks SDK
        try:
            from databricks.sdk import WorkspaceClient
            self.client = WorkspaceClient(
                host=self.workspace_url,
                token=self.token
            )
            self._initialized = True
            logger.info(f"Initialized Databricks Opus client for endpoint: {self.endpoint_name}")
        except ImportError:
            logger.warning("databricks-sdk not installed. Using fallback mode.")
            self.client = None
            self._initialized = True
        except Exception as e:
            logger.warning(f"Could not initialize Databricks client: {e}")
            self.client = None
            self._initialized = True

    def is_available(self) -> bool:
        """Check if the Opus endpoint is available."""
        self._ensure_initialized()
        return self.client is not None and bool(self.workspace_url) and bool(self.token)

    async def think(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,  # Opus 4.5 extended thinking works best at temp=1
        max_tokens: int = 16000,
        budget_tokens: int = 10000  # Thinking budget
    ) -> Dict[str, Any]:
        """
        Send a thinking request to Opus 4.5.

        Uses extended thinking mode for deep reasoning about forecasting problems.
        """
        self._ensure_initialized()

        if not self.is_available():
            return self._fallback_response(user_prompt)

        try:
            import httpx

            # Construct the request for Databricks model serving
            # Using the Anthropic API format that Databricks supports
            request_body = {
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "system": system_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "anthropic_version": "2023-06-01",
                # Extended thinking configuration
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": budget_tokens
                }
            }

            endpoint_url = f"{self.workspace_url}/serving-endpoints/{self.endpoint_name}/invocations"

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    endpoint_url,
                    json=request_body,
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json"
                    }
                )
                response.raise_for_status()
                result = response.json()

                # Extract thinking and response
                content = result.get("content", [])
                thinking_text = ""
                response_text = ""

                for block in content:
                    if block.get("type") == "thinking":
                        thinking_text = block.get("thinking", "")
                    elif block.get("type") == "text":
                        response_text = block.get("text", "")

                return {
                    "success": True,
                    "thinking": thinking_text,
                    "response": response_text,
                    "usage": result.get("usage", {}),
                    "model": result.get("model", self.endpoint_name)
                }

        except Exception as e:
            logger.error(f"Opus 4.5 request failed: {e}")
            return self._fallback_response(user_prompt, str(e))

    def _fallback_response(self, prompt: str, error: Optional[str] = None) -> Dict[str, Any]:  # noqa: ARG002
        """Generate a fallback response when Opus is unavailable."""
        return {
            "success": False,
            "thinking": None,
            "response": None,
            "error": error or "Opus 4.5 endpoint not available",
            "fallback": True
        }


class ForecastThinker:
    """
    Opus 4.5 powered reasoning engine for forecast intelligence.

    Provides deep analysis capabilities:
    - Pre-training data intelligence
    - Forecast interpretation and explanation
    - Interactive Q&A about forecasts
    - Anomaly root cause analysis
    """

    def __init__(self):
        self.opus_client = DatabricksOpusClient()
        self.model_name = "claude-opus-4-5-20250514"

    def _cache_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return f"{prefix}:{hashlib.md5(data_str.encode()).hexdigest()}"

    def _get_cached(self, key: str) -> Optional[ThinkerResponse]:
        """Get cached response if valid."""
        if key in _response_cache:
            cached = _response_cache[key]
            # Check TTL (stored in thinking_time_ms as timestamp hack)
            return cached
        return None

    def _set_cached(self, key: str, response: ThinkerResponse):
        """Cache a response."""
        _response_cache[key] = response

    # =========================================================================
    # DATA ANALYSIS - Pre-training intelligence
    # =========================================================================

    async def analyze_data_for_training(
        self,
        profile: Dict[str, Any],
        sample_data: List[Dict[str, Any]],
        target_col: str,
        date_col: str,  # noqa: ARG002 - reserved for future use
        covariates: List[str]
    ) -> ThinkerResponse:
        """
        Deep analysis of data before model training.

        Provides insights about:
        - Data quality and patterns
        - Seasonality and trends
        - Holiday effects
        - Model recommendations
        """
        start_time = time.time()

        # Check cache
        cache_key = self._cache_key("data_analysis", {
            "profile": profile,
            "target": target_col,
            "covariates": covariates
        })
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        system_prompt = """You are an expert time series analyst and forecasting specialist working with finance data.
Your role is to analyze uploaded data and provide actionable insights for forecasting.

You have deep expertise in:
- Time series patterns (trend, seasonality, cyclicality)
- Holiday and promotional effects on business metrics
- Statistical forecasting methods (Prophet, ARIMA, XGBoost, ETS)
- Data quality assessment
- Feature engineering for forecasting

Provide your analysis in a structured, actionable format. Be specific about what you observe
in the data and why it matters for forecasting accuracy.

When recommending models, explain WHY based on the specific data characteristics you observe."""

        # Build context about the data
        profile_summary = f"""
DATA PROFILE:
- Frequency: {profile.get('frequency', 'unknown')}
- Date Range: {profile.get('date_range', ['?', '?'])}
- Total Periods: {profile.get('total_periods', 0)}
- History (months): {profile.get('history_months', 0)}
- Data Quality Score: {profile.get('data_quality_score', 0)}%
- Holiday Coverage: {profile.get('holiday_coverage_score', 0)}%
- Has Trend: {profile.get('has_trend', False)}
- Has Seasonality: {profile.get('has_seasonality', False)}
- Seasonality Period: {profile.get('seasonality_period', 'None')}
- Missing Periods: {len(profile.get('missing_dates', []))}

TARGET COLUMN: {target_col}
COVARIATES: {', '.join(covariates) if covariates else 'None'}

SAMPLE DATA (first 10 rows):
{json.dumps(sample_data[:10], indent=2, default=str)}
"""

        user_prompt = f"""Analyze this forecasting dataset and provide comprehensive insights.

{profile_summary}

Please provide:
1. EXECUTIVE SUMMARY (2-3 sentences on data quality and forecasting potential)

2. PATTERN ANALYSIS
   - What patterns do you observe? (trend, seasonality, cycles)
   - Are there any anomalies or outliers in the sample?

3. SEASONALITY INSIGHTS
   - What type of seasonality is present?
   - How strong is the seasonal pattern?

4. HOLIDAY/EVENT ANALYSIS
   - Based on the covariates, what holiday effects might be important?
   - Any gaps in holiday coverage?

5. MODEL RECOMMENDATIONS
   For each recommended model, provide:
   - Model name
   - Suitability score (0-100)
   - Why it's suitable for THIS specific data
   - Potential weaknesses

6. SUGGESTED IMPROVEMENTS
   - What additional data would improve forecasts?
   - Any data quality issues to address?

7. WARNINGS
   - Any red flags or concerns?

Format your response as JSON with this structure:
{{
  "summary": "...",
  "data_quality_assessment": "...",
  "pattern_analysis": "...",
  "seasonality_insights": "...",
  "holiday_insights": "...",
  "trend_analysis": "...",
  "model_recommendations": [
    {{"model": "Prophet", "score": 85, "reasoning": "...", "strengths": [...], "weaknesses": [...], "best_for": "..."}}
  ],
  "suggested_horizon": 12,
  "suggested_covariates": [...],
  "warnings": [...],
  "opportunities": [...],
  "confidence": "high|medium|low"
}}"""

        # Call Opus 4.5
        result = await self.opus_client.think(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            budget_tokens=8000
        )

        thinking_time = int((time.time() - start_time) * 1000)

        if result.get("success"):
            try:
                # Parse JSON response
                response_text = result.get("response", "{}")
                # Extract JSON from response (handle markdown code blocks)
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                parsed = json.loads(response_text)

                insights = DataInsights(
                    summary=parsed.get("summary", "Analysis complete."),
                    data_quality_assessment=parsed.get("data_quality_assessment", ""),
                    pattern_analysis=parsed.get("pattern_analysis", ""),
                    seasonality_insights=parsed.get("seasonality_insights", ""),
                    holiday_insights=parsed.get("holiday_insights", ""),
                    trend_analysis=parsed.get("trend_analysis", ""),
                    insights=[],  # Could parse detailed insights
                    model_recommendations=[
                        ModelRecommendation(**rec)
                        for rec in parsed.get("model_recommendations", [])
                    ],
                    suggested_horizon=parsed.get("suggested_horizon", 12),
                    suggested_covariates=parsed.get("suggested_covariates", []),
                    warnings=parsed.get("warnings", []),
                    opportunities=parsed.get("opportunities", []),
                    confidence=ConfidenceLevel(parsed.get("confidence", "medium")),
                    thinking_process=result.get("thinking")
                )

                response = ThinkerResponse(
                    success=True,
                    response_type="data_insights",
                    data=asdict(insights),
                    thinking_time_ms=thinking_time,
                    model_used=self.model_name,
                    tokens_used=result.get("usage", {}).get("output_tokens")
                )

            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse Opus response as JSON: {e}")
                response = self._create_fallback_data_insights(profile, target_col, covariates, thinking_time)
        else:
            response = self._create_fallback_data_insights(profile, target_col, covariates, thinking_time)

        self._set_cached(cache_key, response)
        return response

    def _create_fallback_data_insights(
        self,
        profile: Dict[str, Any],
        target_col: str,  # noqa: ARG002 - reserved for future use
        covariates: List[str],
        thinking_time: int
    ) -> ThinkerResponse:
        """Create fallback insights when Opus is unavailable."""
        freq = profile.get('frequency', 'weekly')
        history_months = profile.get('history_months', 12)
        quality = profile.get('data_quality_score', 100)

        insights = DataInsights(
            summary=f"Dataset contains {profile.get('total_periods', 0)} {freq} periods spanning {history_months:.1f} months. Data quality score: {quality}%.",
            data_quality_assessment=f"Quality score of {quality}% indicates {'excellent' if quality >= 95 else 'good' if quality >= 80 else 'fair'} data quality.",
            pattern_analysis=f"{'Trend detected. ' if profile.get('has_trend') else ''}{'Seasonality detected with period {}.'.format(profile.get('seasonality_period')) if profile.get('has_seasonality') else 'No strong seasonality detected.'}",
            seasonality_insights="Seasonality analysis requires Opus 4.5 for detailed insights.",
            holiday_insights=f"Holiday coverage: {profile.get('holiday_coverage_score', 0):.0f}%. {'Adequate' if profile.get('holiday_coverage_score', 0) >= 80 else 'Consider adding more historical data for better holiday patterns.'}",
            trend_analysis="Trend analysis requires Opus 4.5 for detailed insights.",
            insights=[],
            model_recommendations=[
                ModelRecommendation(
                    model="Prophet",
                    score=85,
                    reasoning="Good default for business time series with seasonality and holidays.",
                    strengths=["Handles missing data", "Built-in holiday support", "Interpretable"],
                    weaknesses=["May overfit on short series"],
                    best_for="Business metrics with clear seasonality"
                ),
                ModelRecommendation(
                    model="XGBoost",
                    score=80,
                    reasoning="Strong for complex patterns with multiple covariates.",
                    strengths=["Handles non-linear relationships", "Feature importance"],
                    weaknesses=["Requires more feature engineering"],
                    best_for="Data with many covariates and complex interactions"
                )
            ],
            suggested_horizon=12 if freq == 'weekly' else 30 if freq == 'daily' else 6,
            suggested_covariates=covariates,
            warnings=["Opus 4.5 unavailable - using basic analysis"] if not self.opus_client.is_available() else [],
            opportunities=[],
            confidence=ConfidenceLevel.MEDIUM,
            thinking_process=None
        )

        return ThinkerResponse(
            success=True,
            response_type="data_insights",
            data=asdict(insights),
            thinking_time_ms=thinking_time,
            model_used="fallback",
            tokens_used=0
        )

    # =========================================================================
    # FORECAST EXPLANATION - Post-training interpretation
    # =========================================================================

    async def explain_forecast(
        self,
        forecast_result: Dict[str, Any],
        training_result: Dict[str, Any],
        profile: Dict[str, Any],
        target_col: str,
        covariates: List[str]
    ) -> ThinkerResponse:
        """
        Generate comprehensive forecast explanation for analysts.

        Provides:
        - Executive summary
        - Key findings
        - Component breakdown
        - Risk factors
        - Recommendations
        """
        start_time = time.time()

        system_prompt = """You are a senior financial analyst explaining forecast results to stakeholders.
Your explanations should be:
- Clear and jargon-free
- Backed by specific numbers from the data
- Actionable with clear recommendations
- Honest about uncertainty and limitations

Focus on WHAT the forecast shows, WHY it shows that, and WHAT actions to consider."""

        # Build forecast context
        best_model = forecast_result.get('best_model', 'Unknown')
        metrics = forecast_result.get('metrics', {})
        forecast_values = forecast_result.get('forecast', [])
        dates = forecast_result.get('dates', [])

        # Calculate some basic stats
        if forecast_values:
            forecast_avg = sum(forecast_values) / len(forecast_values)
            forecast_min = min(forecast_values)
            forecast_max = max(forecast_values)
        else:
            forecast_avg = forecast_min = forecast_max = 0

        context = f"""
FORECAST RESULTS:
- Model Used: {best_model}
- MAPE: {metrics.get('mape', 'N/A')}%
- RMSE: {metrics.get('rmse', 'N/A')}
- R²: {metrics.get('r2', 'N/A')}
- Forecast Periods: {len(forecast_values)}
- Forecast Range: {forecast_min:,.0f} to {forecast_max:,.0f}
- Forecast Average: {forecast_avg:,.0f}

DATA CONTEXT:
- Target: {target_col}
- Frequency: {profile.get('frequency', 'unknown')}
- History: {profile.get('history_months', 0):.1f} months
- Covariates Used: {', '.join(covariates) if covariates else 'None'}

FORECAST VALUES (first 5):
{json.dumps(list(zip(dates[:5], forecast_values[:5])), default=str)}

MODEL COMPARISON:
{json.dumps(forecast_result.get('model_comparison', []), default=str)}
"""

        user_prompt = f"""Explain this forecast to a finance analyst in clear, actionable terms.

{context}

Provide your explanation as JSON:
{{
  "executive_summary": "2-3 sentence overview for executives",
  "methodology_explanation": "Brief explanation of how the forecast was generated",
  "key_findings": ["finding 1", "finding 2", ...],
  "components": [
    {{"name": "Baseline", "contribution": 0.0, "description": "...", "direction": "neutral"}},
    {{"name": "Trend", "contribution": 5.2, "description": "...", "direction": "positive"}}
  ],
  "risk_factors": ["risk 1", "risk 2"],
  "assumptions": ["assumption 1", ...],
  "confidence_assessment": "Explanation of forecast confidence",
  "recommendations": ["action 1", "action 2"],
  "caveats": ["caveat 1", ...]
}}"""

        result = await self.opus_client.think(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            budget_tokens=6000
        )

        thinking_time = int((time.time() - start_time) * 1000)

        if result.get("success"):
            try:
                response_text = result.get("response", "{}")
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                parsed = json.loads(response_text)

                explanation = ForecastExplanation(
                    executive_summary=parsed.get("executive_summary", ""),
                    methodology_explanation=parsed.get("methodology_explanation", ""),
                    key_findings=parsed.get("key_findings", []),
                    components=[
                        ForecastComponent(**c) for c in parsed.get("components", [])
                    ],
                    period_insights=[],  # Could add if needed
                    risk_factors=parsed.get("risk_factors", []),
                    assumptions=parsed.get("assumptions", []),
                    confidence_assessment=parsed.get("confidence_assessment", ""),
                    recommendations=parsed.get("recommendations", []),
                    caveats=parsed.get("caveats", []),
                    thinking_process=result.get("thinking")
                )

                return ThinkerResponse(
                    success=True,
                    response_type="forecast_explanation",
                    data=asdict(explanation),
                    thinking_time_ms=thinking_time,
                    model_used=self.model_name,
                    tokens_used=result.get("usage", {}).get("output_tokens")
                )

            except json.JSONDecodeError:
                pass

        # Fallback explanation
        return self._create_fallback_explanation(
            forecast_result, training_result, profile, target_col, thinking_time
        )

    def _create_fallback_explanation(
        self,
        forecast_result: Dict[str, Any],
        training_result: Dict[str, Any],  # noqa: ARG002 - reserved for future use
        profile: Dict[str, Any],
        target_col: str,  # noqa: ARG002 - reserved for future use
        thinking_time: int
    ) -> ThinkerResponse:
        """Create fallback explanation when Opus unavailable."""
        best_model = forecast_result.get('best_model', 'Unknown')
        mape = forecast_result.get('metrics', {}).get('mape', 0)
        forecast_values = forecast_result.get('forecast', [])

        quality = "excellent" if mape < 5 else "good" if mape < 10 else "fair" if mape < 15 else "needs review"

        explanation = ForecastExplanation(
            executive_summary=f"The {best_model} model generated a {len(forecast_values)}-period forecast with {quality} accuracy (MAPE: {mape:.1f}%). Based on {profile.get('history_months', 0):.0f} months of historical data.",
            methodology_explanation=f"The forecast was generated using {best_model}, which was selected as the best performing model based on validation metrics.",
            key_findings=[
                f"Model accuracy: {mape:.1f}% MAPE ({quality})",
                f"Forecast horizon: {len(forecast_values)} periods",
                f"Based on {profile.get('total_periods', 0)} historical data points"
            ],
            components=[
                ForecastComponent(
                    name="Historical Average",
                    contribution=100.0,
                    description="Baseline from historical data",
                    direction="neutral"
                )
            ],
            period_insights=[],
            risk_factors=["Forecast assumes historical patterns continue"],
            assumptions=["No major external disruptions", "Consistent data collection"],
            confidence_assessment=f"Confidence is {'high' if mape < 5 else 'medium' if mape < 10 else 'low'} based on model accuracy.",
            recommendations=["Monitor actuals vs forecast weekly", "Recalibrate if actuals deviate >10%"],
            caveats=["Opus 4.5 unavailable - basic explanation provided"],
            thinking_process=None
        )

        return ThinkerResponse(
            success=True,
            response_type="forecast_explanation",
            data=asdict(explanation),
            thinking_time_ms=thinking_time,
            model_used="fallback",
            tokens_used=0
        )

    # =========================================================================
    # INTERACTIVE Q&A - Answer analyst questions
    # =========================================================================

    async def answer_question(
        self,
        question: str,
        forecast_context: Dict[str, Any],
        data_context: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> ThinkerResponse:
        """
        Answer analyst questions about the forecast.

        Provides contextual, data-backed answers with supporting evidence.
        """
        start_time = time.time()

        system_prompt = """You are an AI forecasting assistant helping finance analysts understand their forecasts.

When answering questions:
1. Be specific - reference actual numbers from the data
2. Be honest about uncertainty
3. Provide context for your answers
4. Suggest follow-up questions if relevant
5. If you don't have enough information, say so

You have access to:
- Forecast results (predictions, confidence intervals, model metrics)
- Historical data summary
- Model configuration and parameters"""

        # Build context
        context = f"""
FORECAST CONTEXT:
{json.dumps(forecast_context, indent=2, default=str)[:3000]}

DATA CONTEXT:
{json.dumps(data_context, indent=2, default=str)[:2000]}
"""

        # Build conversation history
        history_str = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages
                history_str += f"\n{msg['role'].upper()}: {msg['content']}"

        user_prompt = f"""Context for answering:
{context}

{f'Previous conversation:{history_str}' if history_str else ''}

ANALYST QUESTION: {question}

Provide your answer as JSON:
{{
  "answer": "Your detailed answer here",
  "supporting_data": [{{"metric": "...", "value": "...", "relevance": "..."}}],
  "related_insights": ["insight 1", ...],
  "follow_up_questions": ["Suggested follow-up 1", ...],
  "confidence": "high|medium|low",
  "sources": ["What data/context you used to answer"]
}}"""

        result = await self.opus_client.think(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            budget_tokens=4000
        )

        thinking_time = int((time.time() - start_time) * 1000)

        if result.get("success"):
            try:
                response_text = result.get("response", "{}")
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                parsed = json.loads(response_text)

                qa_response = QAResponse(
                    answer=parsed.get("answer", "I couldn't generate an answer."),
                    supporting_data=parsed.get("supporting_data", []),
                    related_insights=parsed.get("related_insights", []),
                    follow_up_questions=parsed.get("follow_up_questions", []),
                    confidence=ConfidenceLevel(parsed.get("confidence", "medium")),
                    sources=parsed.get("sources", []),
                    thinking_process=result.get("thinking")
                )

                return ThinkerResponse(
                    success=True,
                    response_type="qa_response",
                    data=asdict(qa_response),
                    thinking_time_ms=thinking_time,
                    model_used=self.model_name,
                    tokens_used=result.get("usage", {}).get("output_tokens")
                )

            except json.JSONDecodeError:
                pass

        # Fallback response
        return ThinkerResponse(
            success=True,
            response_type="qa_response",
            data=asdict(QAResponse(
                answer=f"I'm unable to provide a detailed answer right now (Opus 4.5 unavailable). Based on the available data, your question about '{question[:50]}...' would require analysis of the forecast context.",
                supporting_data=[],
                related_insights=[],
                follow_up_questions=["What specific metric are you interested in?", "Would you like to see the raw forecast data?"],
                confidence=ConfidenceLevel.LOW,
                sources=["Limited analysis - AI service unavailable"],
                thinking_process=None
            )),
            thinking_time_ms=thinking_time,
            model_used="fallback",
            tokens_used=0
        )

    # =========================================================================
    # ANOMALY ANALYSIS - Root cause investigation
    # =========================================================================

    async def investigate_anomaly(
        self,
        anomaly: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        forecast_data: List[Dict[str, Any]],
        covariates: List[str]
    ) -> ThinkerResponse:
        """
        Investigate the root cause of a detected anomaly.

        Provides:
        - Likely causes with probabilities
        - Supporting and contradicting evidence
        - Historical similar events
        - Recommended actions
        """
        start_time = time.time()

        system_prompt = """You are a data detective investigating anomalies in time series forecasts.

Your job is to:
1. Analyze why actual values differ significantly from forecast
2. Identify most likely root causes
3. Find supporting evidence in the data
4. Look for similar historical patterns
5. Recommend actions

Be thorough but concise. Prioritize likely causes by probability."""

        context = f"""
ANOMALY DETECTED:
{json.dumps(anomaly, indent=2, default=str)}

HISTORICAL DATA (recent periods):
{json.dumps(historical_data[-20:], indent=2, default=str)}

FORECAST DATA:
{json.dumps(forecast_data[:10], indent=2, default=str)}

COVARIATES AVAILABLE: {', '.join(covariates)}
"""

        user_prompt = f"""Investigate this anomaly and provide root cause analysis.

{context}

Provide your analysis as JSON:
{{
  "anomaly_description": "Clear description of the anomaly",
  "severity": "critical|warning|info",
  "likely_causes": [
    {{"cause": "...", "probability": 0.6, "evidence": ["..."]}}
  ],
  "supporting_evidence": ["evidence 1", ...],
  "contradicting_evidence": ["evidence 1", ...],
  "recommended_actions": ["action 1", ...],
  "similar_historical_events": [
    {{"date": "...", "description": "...", "outcome": "..."}}
  ],
  "confidence": "high|medium|low"
}}"""

        result = await self.opus_client.think(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            budget_tokens=5000
        )

        thinking_time = int((time.time() - start_time) * 1000)

        if result.get("success"):
            try:
                response_text = result.get("response", "{}")
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                parsed = json.loads(response_text)

                analysis = AnomalyAnalysis(
                    anomaly_description=parsed.get("anomaly_description", ""),
                    severity=parsed.get("severity", "warning"),
                    likely_causes=parsed.get("likely_causes", []),
                    supporting_evidence=parsed.get("supporting_evidence", []),
                    contradicting_evidence=parsed.get("contradicting_evidence", []),
                    recommended_actions=parsed.get("recommended_actions", []),
                    similar_historical_events=parsed.get("similar_historical_events", []),
                    confidence=ConfidenceLevel(parsed.get("confidence", "medium")),
                    thinking_process=result.get("thinking")
                )

                return ThinkerResponse(
                    success=True,
                    response_type="anomaly_analysis",
                    data=asdict(analysis),
                    thinking_time_ms=thinking_time,
                    model_used=self.model_name,
                    tokens_used=result.get("usage", {}).get("output_tokens")
                )

            except json.JSONDecodeError:
                pass

        # Fallback analysis
        return ThinkerResponse(
            success=True,
            response_type="anomaly_analysis",
            data=asdict(AnomalyAnalysis(
                anomaly_description=f"Anomaly detected: {anomaly.get('description', 'Value outside expected range')}",
                severity="warning",
                likely_causes=[
                    {"cause": "External event not captured in model", "probability": 0.4, "evidence": []},
                    {"cause": "Data quality issue", "probability": 0.3, "evidence": []},
                    {"cause": "Model limitation", "probability": 0.3, "evidence": []}
                ],
                supporting_evidence=["Detailed analysis requires Opus 4.5"],
                contradicting_evidence=[],
                recommended_actions=["Review raw data for the anomaly period", "Check for external events", "Consider retraining with updated data"],
                similar_historical_events=[],
                confidence=ConfidenceLevel.LOW,
                thinking_process=None
            )),
            thinking_time_ms=thinking_time,
            model_used="fallback",
            tokens_used=0
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def is_available(self) -> bool:
        """Check if thinker service is available."""
        return self.opus_client.is_available()

    def get_status(self) -> Dict[str, Any]:
        """Get thinker service status."""
        return {
            "available": self.is_available(),
            "model": self.model_name,
            "endpoint": self.opus_client.endpoint_name,
            "cache_size": len(_response_cache)
        }

    def clear_cache(self):
        """Clear response cache."""
        global _response_cache
        _response_cache = {}
