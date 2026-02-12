"""
Service for dataset analysis using Databricks Foundation Models
"""
import os
import requests
import json
import logging
from typing import Dict, Any, List

from backend.utils.logging_utils import log_io

logger = logging.getLogger(__name__)

MODELS_TO_TRY = ['databricks-gpt-5-1', 'databricks-gemini-3-pro', 'databricks-meta-llama-3-3-70b-instruct']

@log_io(log_result=False)
def call_databricks_model(prompt: str, system_prompt: str = "You are a helpful assistant.", temperature: float = 0.3, max_tokens: int = 2000) -> Dict[str, Any]:
    from databricks.sdk import WorkspaceClient
    from mlflow.utils.databricks_utils import get_databricks_host_creds
    
    client = None
    try:
        creds = get_databricks_host_creds("databricks")
        if creds.host and creds.host != 'inherit':
            client = WorkspaceClient(host=creds.host, token=creds.token)
        else:
            host = os.environ.get('DATABRICKS_HOST', '')
            if host == 'inherit': del os.environ['DATABRICKS_HOST']
            client = WorkspaceClient()
            if client.config.host == 'inherit' and os.environ.get('DATABRICKS_HOST') and os.environ.get('DATABRICKS_HOST') != 'inherit':
                 client = WorkspaceClient(host=os.environ['DATABRICKS_HOST'])
    except Exception as e:
        logger.warning(f"SDK init failed: {e}")

    last_error = None
    for model_name in MODELS_TO_TRY:
        payload = {
            "model": model_name,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            if client:
                response = client.api_client.do(method="POST", path="/serving-endpoints/chat/completions", body=payload)
                content = (response.get('choices', [{}])[0].get('message', {}).get('content', '{}') if isinstance(response, dict) else
                           getattr(getattr(getattr(response, 'choices', [])[0], 'message', None), 'content', '{}'))
            else:
                host, token = os.environ.get('DATABRICKS_HOST', ''), os.environ.get('DATABRICKS_TOKEN', '')
                if not host or not token or host == 'inherit': raise ValueError("Invalid manual creds")
                resp = requests.post(f"{host}/serving-endpoints/chat/completions", json=payload, headers={"Authorization": f"Bearer {token}"}, timeout=60)
                resp.raise_for_status()
                content = resp.json().get('choices', [{}])[0].get('message', {}).get('content', '{}')
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                import re
                match = re.search(r'```json\s*([\s\S]*?)\s*```', content) or re.search(r'\{[\s\S]*\}', content)
                if match: return json.loads(match.group(1) if match.lastindex else match.group(0))
                raise ValueError(f"JSON parse failed: {content[:200]}")
                
        except Exception as e:
            last_error = e
            continue
    
    raise Exception(f"All models failed. Last error: {last_error}")

@log_io
def analyze_dataset(sample_data: List[Dict[str, Any]], columns: List[str]) -> Dict[str, Any]:
    try:
        def json_serial(obj):
            return obj.isoformat() if hasattr(obj, 'isoformat') else obj.tolist() if hasattr(obj, 'tolist') else str(obj)

        prompt = f"""
        Analyze this dataset sample (first 5 rows) and columns.
        Columns: {', '.join(columns)}
        Sample Data: {json.dumps(sample_data[:5], default=json_serial)}

        Goal: Identify Date/Time column, Target column (numeric metric), Group columns, Covariates, Summary, Seasonality.
        Return ONLY valid JSON: {{ "summary": "str", "suggestedTimeColumn": "str", "suggestedTargetColumn": "str", "suggestedGroupColumns": ["str"], "suggestedCovariates": ["str"], "seasonality": "str" }}
        """
        return call_databricks_model(prompt, "You are a senior data scientist. Return JSON only.")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        date_col = next((c for c in columns if any(kw in c.lower() for kw in ['date', 'time', 'ds', 'period'])), columns[0] if columns else '')
        target_col = next((c for c in columns if c != date_col and any(kw in c.lower() for kw in ['revenue', 'sales', 'cost', 'amount'])), '')
        return {
            "summary": "Heuristic analysis used.",
            "suggestedTimeColumn": date_col,
            "suggestedTargetColumn": target_col,
            "suggestedGroupColumns": [c for c in columns if c not in [date_col, target_col]][:3],
            "suggestedCovariates": [c for c in columns if c not in [date_col, target_col]][:5],
            "seasonality": "Unknown"
        }

@log_io
def generate_forecast_insights(data_summary, target_col, time_col, covariates, filters, seasonality_mode, winning_model, frequency) -> Dict[str, Any]:
    prompt = f"""
    Context: Forecasting results.
    Config: Target={target_col}, Time={time_col}, Covariates={covariates}, Model={winning_model}, Seasonality={seasonality_mode}, Freq={frequency}
    Summary: {data_summary}
    Task: Executive Summary & Python code snippet.
    Return JSON: {{ "explanation": "str", "pythonCode": "str" }}
    """
    try:
        return call_databricks_model(prompt, "Generate insights. Return JSON only.", temperature=0.5)
    except Exception:
        return {"explanation": f"Model: {winning_model}", "pythonCode": "# Prophet code\nmodel = Prophet()\nmodel.fit(df)"}

@log_io
def generate_executive_summary(best_model_name, best_model_metrics, all_models, target_col, time_col, covariates, forecast_horizon, frequency, actuals_comparison=None) -> str:
    # Format model comparison with full hyperparameter details and ranking
    sorted_models = sorted(all_models, key=lambda m: float(m['metrics']['mape']))
    model_comp_lines = []
    for i, m in enumerate(sorted_models, 1):
        is_best = m['modelName'] == best_model_name
        model_comp_lines.append(
            f"- #{i} **{m['modelName']}**: MAPE={m['metrics']['mape']}%, RMSE={m['metrics']['rmse']}{' [SELECTED]' if is_best else ''}"
        )
    model_comp = "\n".join(model_comp_lines)

    # Build actuals comparison section if available
    actuals_section = ""
    anomaly_analysis_prompt = ""

    if actuals_comparison:
        total = actuals_comparison.get('totalPeriods', 0)
        excellent = actuals_comparison.get('excellentCount', 0)
        good = actuals_comparison.get('goodCount', 0)
        acceptable = actuals_comparison.get('acceptableCount', 0)
        review = actuals_comparison.get('reviewCount', 0)
        deviation = actuals_comparison.get('deviationCount', 0)
        overall_mape = actuals_comparison.get('overallMAPE', 0)
        overall_bias = actuals_comparison.get('overallBias', 0)

        # Format worst periods for deep analysis
        worst_periods = actuals_comparison.get('worstPeriods', [])
        worst_periods_detail = ""
        if worst_periods:
            worst_periods_detail = "\n    ANOMALOUS PERIODS REQUIRING INVESTIGATION:\n"
            for i, wp in enumerate(worst_periods[:5], 1):
                error_direction = "UNDER-FORECAST (actual exceeded prediction)" if wp.get('error', 0) > 0 else "OVER-FORECAST (prediction exceeded actual)"
                variance = wp.get('actual', 0) - wp.get('predicted', 0)
                variance_pct = (variance / wp.get('predicted', 1)) * 100 if wp.get('predicted', 0) != 0 else 0
                worst_periods_detail += f"""
    ANOMALY #{i}: {wp.get('date')}
      - Predicted: {wp.get('predicted', 0):,.2f}
      - Actual: {wp.get('actual', 0):,.2f}
      - Variance: {variance:+,.2f} ({variance_pct:+.1f}%)
      - MAPE: {wp.get('mape', 0):.1f}%
      - Direction: {error_direction}
"""

        # Calculate bias pattern - industry agnostic
        bias_interpretation = ""
        if overall_bias > 0:
            bias_interpretation = f"SYSTEMATIC UNDER-FORECASTING: The model consistently predicted {abs(overall_bias):,.2f} units LOWER than actuals on average. This suggests the model is missing upward drivers or growth factors."
        else:
            bias_interpretation = f"SYSTEMATIC OVER-FORECASTING: The model consistently predicted {abs(overall_bias):,.2f} units HIGHER than actuals on average. This suggests the model is overestimating or missing downward factors."

        # Calculate failure rate
        failure_rate = 100 * (review + deviation) / total if total > 0 else 0
        success_rate = 100 * (excellent + good + acceptable) / total if total > 0 else 0

        actuals_section = f"""

=== FORECAST ACCURACY AUDIT: {best_model_name} vs. ACTUALS ===

NOTE: This comparison uses predictions from the SELECTED model ({best_model_name}) against uploaded actuals.
The other models were used only for model selection (best MAPE on validation set).

OVERALL METRICS:
- Overall MAPE: {overall_mape:.2f}%
- Overall Bias: {overall_bias:+,.2f}
- Bias Analysis: {bias_interpretation}

PERFORMANCE BREAKDOWN ({total} periods analyzed):
- SUCCESS RATE (within 15% error): {success_rate:.1f}% ({excellent + good + acceptable} periods)
  * Excellent (≤5%): {excellent} periods ({100*excellent/total:.1f}%)
  * Good (5-10%): {good} periods ({100*good/total:.1f}%)
  * Acceptable (10-15%): {acceptable} periods ({100*acceptable/total:.1f}%)

- FAILURE RATE (>15% error): {failure_rate:.1f}% ({review + deviation} periods)
  * Needs Review (15-25%): {review} periods ({100*review/total:.1f}%)
  * CRITICAL DEVIATION (>25%): {deviation} periods ({100*deviation/total:.1f}%)

{worst_periods_detail}
"""

        # Enhanced anomaly analysis prompt
        anomaly_analysis_prompt = f"""
CRITICAL ANALYSIS REQUIRED:

For each anomalous period listed above, you MUST:

1. **Date Pattern Analysis**:
   - Is this date near a major holiday (Thanksgiving week, Christmas, Black Friday, Cyber Monday, New Year, Easter, Memorial Day, Labor Day, July 4th, Super Bowl Sunday)?
   - Is it a weekend vs weekday pattern?
   - Is it month-end/quarter-end?
   - Is it near a typical payday cycle (1st/15th of month)?

2. **Seasonal Pattern Analysis**:
   - Does the date fall in a known high/low season?
   - Are there weather-related factors (storms, extreme heat/cold)?
   - Summer vs winter patterns? Back-to-school timing?

3. **Business Event Hypothesis**:
   - Could there have been a promotion or marketing campaign?
   - New feature launches or service changes?
   - Competitor actions or market dynamics?
   - Service disruptions or operational issues?

4. **External Factors**:
   - Economic events (stimulus, inflation, consumer sentiment)?
   - Major sporting events (Super Bowl, playoffs, World Cup)?
   - Weather events affecting behavior?
   - News events or viral moments?

5. **Model Limitation Assessment**:
   - Does this error pattern suggest missing covariates?
   - Is the model failing to capture non-linear relationships?
   - Are there data quality issues for this period?

BE SPECIFIC AND CRITICAL. Don't give generic advice. For a date like "2025-11-19" near Thanksgiving, explicitly state "This date is 8 days before Thanksgiving - likely impacted by holiday ordering patterns."
"""

    system_prompt = """You are a CRITICAL financial forecast analyst. Your job is to:

1. IDENTIFY PROBLEMS - Don't sugarcoat. If the model is failing, say so clearly.
2. EXPLAIN ROOT CAUSES - For every anomaly, provide specific hypotheses based on the date and context.
3. QUANTIFY IMPACT - Use actual numbers and percentages, not vague language.
4. RECOMMEND ACTIONS - Give concrete next steps, not generic advice.

You are skeptical and thorough. You look for patterns others miss. You connect dates to real-world events.
When you see a date with high error, you immediately think: "What was happening on or around this date that could explain this?"

IMPORTANT CONTEXT:
- Keep business impact descriptions generic (e.g., "resource allocation", "planning", "budgeting")
- DO NOT assume specific industries - avoid terms like "inventory", "stockouts", "labor scheduling", "materials", "logistics"
- Focus on forecast accuracy implications rather than operational specifics

FORMATTING RULES:
- Use markdown headers (###) and bullet points (-)
- DO NOT use markdown tables (they don't render properly in the UI)
- Instead of tables, use bullet points with bold labels like: **MAPE:** 6.42% - Within target
- Keep sections visually separated with horizontal rules (---)
- Be direct and analytical."""

    prompt = f"""
FORECAST PERFORMANCE REPORT

Target Variable: {target_col}
Forecast Horizon: {forecast_horizon} {frequency} periods
Covariates Used: {', '.join(covariates) if covariates else 'None'}

MODEL SELECTION RESULTS:
Best Model: {best_model_name}
- MAPE: {best_model_metrics.get('mape')}%
- RMSE: {best_model_metrics.get('rmse')}

All Models Evaluated:
{model_comp}
{actuals_section}
{anomaly_analysis_prompt}

Generate a CRITICAL executive summary with the following sections:

### Executive Overview
(2-3 sentences on overall forecast reliability and key concerns)

### Model Performance Assessment
(Analyze why this model won, and critically assess if the winning metrics are actually acceptable for business use)

{"### Forecast vs Actuals: Critical Findings" if actuals_comparison else ""}
(If actuals provided: Analyze the success/failure rates. Is this model production-ready?)

{"### Anomaly Deep-Dive & Root Cause Analysis" if actuals_comparison and (review > 0 or deviation > 0) else ""}
(For EACH anomalous period: Provide specific date-based hypotheses. What likely happened on or around these dates?)

### Risk Assessment
(What are the business risks of using this forecast? Where might it fail next?)

### Actionable Recommendations
(Specific steps to improve forecast accuracy - not generic advice. Reference the specific anomalies found.)
"""

    try:
        # Use a simplified direct call since we need text, not JSON
        from databricks.sdk import WorkspaceClient
        client = WorkspaceClient()
        resp = client.api_client.do(method="POST", path="/serving-endpoints/chat/completions", body={
            "model": MODELS_TO_TRY[0],
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            "temperature": 0.4,  # Lower temperature for more focused, analytical responses
            "max_tokens": 3000   # Allow longer responses for detailed analysis
        })
        return (resp.get('choices', [{}])[0].get('message', {}).get('content', '') if isinstance(resp, dict) else
                getattr(getattr(getattr(resp, 'choices', [])[0], 'message', None), 'content', ''))
    except Exception as e:
        logger.warning(f"AI summary generation failed: {e}")
        # Enhanced fallback summary with critical analysis
        mape_val = float(best_model_metrics.get('mape', 0))
        rmse_val = float(best_model_metrics.get('rmse', 0))

        # Critical quality assessment
        if mape_val <= 5:
            quality = "EXCELLENT"
            quality_desc = "Industry gold standard"
            risk_level = "LOW"
            recommendation = "Model is production-ready. Implement monitoring for drift detection."
        elif mape_val <= 10:
            quality = "GOOD"
            quality_desc = "Acceptable for most business decisions"
            risk_level = "LOW-MEDIUM"
            recommendation = "Consider adding external regressors to capture remaining variance."
        elif mape_val <= 15:
            quality = "ACCEPTABLE WITH CAUTION"
            quality_desc = "Use for directional guidance only"
            risk_level = "MEDIUM"
            recommendation = "Not recommended for precise budgeting. Investigate high-error periods before production use."
        elif mape_val <= 25:
            quality = "NEEDS IMPROVEMENT"
            quality_desc = "Significant forecast risk"
            risk_level = "HIGH"
            recommendation = "DO NOT use for critical decisions without manual review. Root cause analysis required."
        else:
            quality = "CRITICAL - UNRELIABLE"
            quality_desc = "Model is not fit for purpose"
            risk_level = "CRITICAL"
            recommendation = "STOP - This model should not be deployed. Fundamental issues with data or approach."

        fallback = f"""### Executive Overview

The **{best_model_name}** model achieved **{mape_val:.2f}% MAPE** for forecasting **{target_col}**. Risk Level: **{risk_level}**

---

### Model Performance Assessment

**Quality Rating: {quality}** ({quality_desc})

**Key Metrics:**
- **MAPE:** {mape_val:.2f}% - {"Within target" if mape_val <= 10 else "Above target" if mape_val <= 15 else "Unacceptable"}
- **RMSE:** {rmse_val:,.2f} - Absolute error magnitude

**All Models Evaluated (Ranked by MAPE):**
{model_comp}

*Note: Models were tuned via hyperparameter grid search. The best configuration for each model type is shown above.*

---

### Risk Assessment

**Production Readiness:** {"READY" if mape_val <= 10 else "CONDITIONAL" if mape_val <= 15 else "NOT READY"}

**Key Risks:**
- {"Low risk of material forecast errors" if mape_val <= 10 else "Medium risk - expect periodic significant misses" if mape_val <= 15 else "High risk - forecast errors may significantly impact business decisions"}
- Model may underperform during holidays, promotions, or unusual market conditions
- {"Consider manual override capability for critical periods" if mape_val > 10 else "Standard monitoring sufficient"}

---

### Actionable Recommendations

1. **Immediate:** {recommendation}

2. **Short-term:** Compare forecast vs actuals weekly to detect model degradation early.

3. **Long-term:** Retrain monthly with fresh data; evaluate adding covariates for known events."""

        if actuals_comparison:
            total = actuals_comparison.get('totalPeriods', 1)
            excellent = actuals_comparison.get('excellentCount', 0)
            good = actuals_comparison.get('goodCount', 0)
            acceptable = actuals_comparison.get('acceptableCount', 0)
            review = actuals_comparison.get('reviewCount', 0)
            deviation = actuals_comparison.get('deviationCount', 0)
            overall_mape = actuals_comparison.get('overallMAPE', 0)
            overall_bias = actuals_comparison.get('overallBias', 0)

            # Critical bias analysis - industry agnostic
            if overall_bias > 0:
                bias_analysis = f"**SYSTEMATIC UNDER-FORECASTING**: Model predicted {abs(overall_bias):,.2f} units LOW on average. Business impact: missed opportunities, under-allocation of resources, potential revenue loss."
            else:
                bias_analysis = f"**SYSTEMATIC OVER-FORECASTING**: Model predicted {abs(overall_bias):,.2f} units HIGH on average. Business impact: over-allocation of resources, inflated expectations, budget misalignment."

            # Calculate rates
            success_rate = 100 * (excellent + good + acceptable) / total if total > 0 else 0
            failure_rate = 100 * (review + deviation) / total if total > 0 else 0

            # Production readiness based on actuals
            if failure_rate > 20:
                prod_status = "**NOT PRODUCTION READY** - Failure rate exceeds 20%"
            elif failure_rate > 10:
                prod_status = "**CONDITIONAL** - Review anomalous periods before deployment"
            else:
                prod_status = "**PRODUCTION READY** - Acceptable error distribution"

            fallback += f"""

---

### Forecast vs Actuals: Critical Findings

**Model Under Evaluation:** {best_model_name}
*(Actuals are compared against predictions from the selected best model only)*

**Overall Assessment:** {prod_status}

**Key Metrics:**
- **MAPE vs Actuals:** {overall_mape:.2f}%
- **Success Rate (≤15% error):** {success_rate:.1f}% ({excellent + good + acceptable}/{total} periods)
- **Failure Rate (>15% error):** {failure_rate:.1f}% ({review + deviation}/{total} periods)

**Bias Analysis:**
{bias_analysis}

**Error Distribution:**
- Excellent (≤5%): {excellent} periods ({100*excellent/total:.1f}%)
- Good (5-10%): {good} periods ({100*good/total:.1f}%)
- Acceptable (10-15%): {acceptable} periods ({100*acceptable/total:.1f}%)
- Needs Review (15-25%): {review} periods ({100*review/total:.1f}%)
- Critical (>25%): {deviation} periods ({100*deviation/total:.1f}%)"""

            # Add worst periods with date-specific analysis
            worst_periods = actuals_comparison.get('worstPeriods', [])
            if worst_periods:
                fallback += "\n\n---\n\n### Anomaly Deep-Dive\n\n**Periods with Largest Errors (Requires Investigation):**\n"
                for i, wp in enumerate(worst_periods[:5], 1):
                    error_dir = "UNDER-FORECAST" if wp.get('error', 0) > 0 else "OVER-FORECAST"
                    variance = wp.get('actual', 0) - wp.get('predicted', 0)

                    # Try to identify date context
                    date_str = wp.get('date', '')
                    date_context = ""
                    try:
                        from datetime import datetime
                        dt = datetime.strptime(date_str, '%Y-%m-%d')
                        month_day = dt.strftime('%B %d')
                        weekday = dt.strftime('%A')

                        # Check for known events
                        if dt.month == 11 and 20 <= dt.day <= 30:
                            date_context = "**Thanksgiving Week** - High likelihood of holiday shopping impact"
                        elif dt.month == 12 and dt.day >= 15:
                            date_context = "**Holiday Season** - Christmas/year-end shopping surge"
                        elif dt.month == 12 and dt.day <= 5:
                            date_context = "**Post-Holiday** - Returns, gift card redemptions"
                        elif dt.month == 1 and dt.day <= 7:
                            date_context = "**New Year Period** - Post-holiday slowdown typical"
                        elif weekday in ['Saturday', 'Sunday']:
                            date_context = f"**Weekend ({weekday})** - Different demand pattern than weekdays"
                        elif dt.day == 1 or dt.day == 15:
                            date_context = "**Payday Pattern** - Potential consumer spending spike"
                        else:
                            date_context = f"{weekday}, {month_day}"
                    except:
                        date_context = ""

                    fallback += f"""
**Anomaly #{i}: {date_str}** ({error_dir})
- Predicted: {wp.get('predicted', 0):,.2f}
- Actual: {wp.get('actual', 0):,.2f}
- Variance: {variance:+,.2f} ({wp.get('mape', 0):.1f}% error)
- {date_context}
- **Action Required:** Investigate what business event or external factor caused this deviation.
"""

                fallback += """
---

### Root Cause Investigation Checklist

For each anomaly above, verify:
- [ ] Was there a promotion, sale, or marketing campaign?
- [ ] Did a holiday or event fall on/near this date?
- [ ] Were there supply chain issues (stockouts/overstock)?
- [ ] Did competitors take significant actions?
- [ ] Were there weather or external disruptions?
- [ ] Is this a data quality issue (reporting delay, correction)?
"""

        return fallback
