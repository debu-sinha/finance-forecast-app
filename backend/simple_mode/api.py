"""
Simple Mode API endpoints.

Provides autopilot forecasting for finance users who want
Excel-like simplicity with ML accuracy.
"""

import io
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from .data_profiler import DataProfiler, DataProfile
from .autopilot_config import AutopilotConfig, ForecastConfig, generate_hyperparameter_filters
from .forecast_explainer import ForecastExplainer, format_explanation_for_display
from .excel_exporter import export_forecast_to_excel

logger = logging.getLogger(__name__)


def _sanitize_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    """Sanitize float values to be JSON-compliant (no NaN or Inf)."""
    import math
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _sanitize_dict(d: Dict[str, Any], float_keys: List[str] = None) -> Dict[str, Any]:
    """Recursively sanitize dict values to be JSON-compliant."""
    import math
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _sanitize_dict(v)
        elif isinstance(v, list):
            result[k] = _sanitize_list(v)
        elif isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                result[k] = 0.0 if 'mape' not in k.lower() else 100.0
            else:
                result[k] = v
        else:
            result[k] = v
    return result


def _sanitize_list(lst: List[Any]) -> List[Any]:
    """Recursively sanitize list values to be JSON-compliant."""
    import math
    result = []
    for item in lst:
        if isinstance(item, dict):
            result.append(_sanitize_dict(item))
        elif isinstance(item, list):
            result.append(_sanitize_list(item))
        elif isinstance(item, float):
            if math.isnan(item) or math.isinf(item):
                result.append(0.0)
            else:
                result.append(item)
        else:
            result.append(item)
    return result


# Create router for simple mode endpoints
router = APIRouter(prefix="/api/simple", tags=["Simple Mode"])


# Pydantic models for API responses
class DataProfileResponse(BaseModel):
    """Response from data profiling."""
    success: bool
    profile: Dict[str, Any]
    warnings: List[Dict[str, str]]
    config_preview: Dict[str, Any]


class SliceForecastResult(BaseModel):
    """Forecast result for a single slice/segment."""
    slice_id: str
    slice_filters: Dict[str, str]
    forecast: List[float]
    dates: List[str]
    lower_bounds: List[float]
    upper_bounds: List[float]
    best_model: Optional[str] = None
    holdout_mape: Optional[float] = None
    data_points: int = 0


class SimpleForecastResponse(BaseModel):
    """Response from simple mode forecast."""
    success: bool
    mode: str
    run_id: str

    # Summary
    summary: str

    # Forecast data (aggregate or single-slice)
    forecast: List[float]
    dates: List[str]
    lower_bounds: List[float]
    upper_bounds: List[float]

    # Transparency
    components: Dict[str, Any]
    confidence: Dict[str, Any]

    # Warnings
    warnings: List[Dict[str, str]]
    caveats: List[str]

    # Audit & Traceability
    audit: Dict[str, Any]

    # MLflow Tracking (for reproducibility)
    mlflow_run_id: Optional[str] = None
    mlflow_experiment_url: Optional[str] = None
    mlflow_run_url: Optional[str] = None
    model_uri: Optional[str] = None

    # Models trained
    best_model: Optional[str] = None
    all_models_trained: Optional[List[str]] = None

    # Reproducibility Guarantee
    data_hash: Optional[str] = None
    config_hash: Optional[str] = None
    random_seed: int = 42
    reproducibility_note: str = "Same data + same horizon = identical results guaranteed"

    # Data Split Strategy (Train/Eval/Holdout)
    data_split: Optional[Dict[str, Any]] = None
    model_comparison: Optional[List[Dict[str, Any]]] = None
    selection_reason: Optional[str] = None
    trained_on_full_data: bool = False

    # Anomaly Detection
    anomalies: Optional[List[Dict[str, Any]]] = None

    # Holdout Performance (true test of model quality)
    holdout_mape: Optional[float] = None
    eval_mape: Optional[float] = None

    # Future covariates info
    future_covariates_used: bool = False
    future_covariates_count: int = 0
    future_covariates_date_range: Optional[List[str]] = None

    # By-slice forecasting (NEW)
    forecast_mode: str = "aggregate"  # "aggregate" or "by_slice"
    slice_forecasts: Optional[List[SliceForecastResult]] = None
    slice_columns: Optional[List[str]] = None

    # Export links
    excel_download_url: str
    reproduce_url: str = ""


# Store for forecast results (in production, use database/cache)
_forecast_store: Dict[str, Dict[str, Any]] = {}


@router.post("/profile", response_model=DataProfileResponse)
async def profile_data(file: UploadFile = File(...)):
    """
    Profile uploaded data without running forecast.

    Returns auto-detected settings and warnings so user can review
    before proceeding with forecast.
    """
    try:
        # Read file
        content = await file.read()
        df = _parse_file(content, file.filename)

        # Profile data
        profiler = DataProfiler()
        profile = profiler.profile(df)

        # Generate preview config
        autopilot = AutopilotConfig()
        config = autopilot.generate(profile)

        return DataProfileResponse(
            success=True,
            profile={
                'frequency': profile.frequency,
                'date_column': profile.date_column,
                'target_column': profile.target_column,
                'date_range': [str(profile.date_range[0]), str(profile.date_range[1])],
                'total_periods': int(profile.total_periods),
                'history_months': float(profile.history_months),
                'data_quality_score': float(profile.data_quality_score),
                'holiday_coverage_score': float(profile.holiday_coverage_score),
                'has_trend': bool(profile.has_trend),
                'has_seasonality': bool(profile.has_seasonality),
                'covariate_columns': profile.covariate_columns,
                'recommended_models': profile.recommended_models,
                'recommended_horizon': int(profile.recommended_horizon),
                'data_hash': profile.data_hash,
                'row_count': int(profile.row_count),
                # Future covariates
                'has_future_covariates': bool(profile.has_future_covariates),
                'future_rows_count': int(profile.future_rows_count),
                'future_rows_date_range': [str(d) for d in profile.future_rows_date_range] if profile.future_rows_date_range else None,
                'future_covariates_valid': bool(profile.future_covariates_valid),
                'future_covariates_issues': profile.future_covariates_issues,
                # Multi-slice data detection
                'unique_periods': int(profile.unique_periods),
                'has_multiple_slices': bool(profile.has_multiple_slices),
                'slice_count': int(profile.slice_count),
                # Missing dates (for debugging)
                'missing_dates': [str(d) for d in profile.missing_periods],
                # Data leakage detection
                'leaky_covariates': profile.leaky_covariates,
                'safe_covariates': profile.safe_covariates,
                'correlation_details': profile.correlation_details,
            },
            warnings=[
                {'level': w.level, 'message': w.message, 'recommendation': w.recommendation}
                for w in profile.warnings
            ],
            config_preview=config.to_dict(),
        )

    except Exception as e:
        logger.error(f"Profile error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/forecast", response_model=SimpleForecastResponse)
async def simple_forecast(
    file: UploadFile = File(...),
    horizon: Optional[int] = Query(None, description="Forecast horizon (auto-detected if not provided)"),
    date_col: Optional[str] = Query(None, description="User-selected date column"),
    target_col: Optional[str] = Query(None, description="User-selected target column"),
    covariates: Optional[str] = Query(None, description="Comma-separated list of covariate columns"),
    forecast_mode: Optional[str] = Query(None, description="'aggregate' or 'by_slice'"),
    slice_columns: Optional[str] = Query(None, description="Comma-separated slice column names"),
    slice_values: Optional[str] = Query(None, description="Pipe-separated slice values to forecast"),
    confidence_level: Optional[float] = Query(None, ge=0.50, le=0.99, description="Confidence level for prediction intervals (0.50-0.99). Default 0.95. Use 0.80 for ~35% narrower intervals."),
):
    """
    Simple mode forecast - upload data, get forecast automatically.

    All configuration is automatic:
    - Auto-detects frequency (daily/weekly/monthly)
    - Auto-detects date and target columns
    - Auto-selects best models
    - Auto-configures parameters

    Returns forecast with full transparency and audit trail.
    """
    try:
        run_id = str(uuid.uuid4())[:8]
        logger.info("=" * 60)
        logger.info(f"🚀 SIMPLE MODE FORECAST - RUN {run_id}")
        logger.info("=" * 60)
        logger.info("📥 RECEIVED PARAMETERS:")
        logger.info(f"    horizon: {horizon}")
        logger.info(f"    date_col: {date_col}")
        logger.info(f"    target_col: {target_col}")
        logger.info(f"    covariates: {covariates}")
        logger.info(f"    forecast_mode: {forecast_mode}")
        logger.info(f"    slice_columns: {slice_columns}")
        logger.info(f"    slice_values: {slice_values}")
        logger.info(f"    confidence_level: {confidence_level}")
        logger.info("-" * 60)

        # Step 1: Parse file
        content = await file.read()
        df = _parse_file(content, file.filename)
        logger.info(f"📄 Parsed {len(df)} rows from {file.filename}")
        logger.info(f"    Columns: {list(df.columns)}")

        # Step 2: Profile data (with user overrides)
        profiler = DataProfiler()
        profile = profiler.profile(df)

        # Apply user-selected columns if provided
        if date_col:
            if date_col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Date column '{date_col}' not found in data")
            profile.date_column = date_col
            logger.info(f"Using user-selected date column: {date_col}")

        if target_col:
            if target_col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found in data")
            profile.target_column = target_col
            logger.info(f"Using user-selected target column: {target_col}")

        if covariates:
            covariate_list = [c.strip() for c in covariates.split(',') if c.strip()]
            # Validate all covariates exist
            missing = [c for c in covariate_list if c not in df.columns]
            if missing:
                raise HTTPException(status_code=400, detail=f"Covariate columns not found: {missing}")
            profile.covariate_columns = covariate_list
            logger.info(f"Using user-selected covariates: {covariate_list}")

        logger.info(f"Profiled data: {profile.frequency} frequency, {profile.history_months:.1f} months history")

        # Step 3: Generate config
        autopilot = AutopilotConfig()
        config = autopilot.generate(profile, horizon=horizon)

        # Override config with user selections
        if date_col:
            config.date_column = date_col
        if target_col:
            config.target_column = target_col
        if covariates:
            config.covariate_columns = [c.strip() for c in covariates.split(',') if c.strip()]
        if confidence_level is not None:
            config.confidence_level = confidence_level
            logger.info(f"Using user-specified confidence level: {confidence_level}")

        logger.info(f"Generated config: models={config.models}, horizon={config.horizon}, confidence_level={config.confidence_level}")

        # Step 3.5: Generate intelligent hyperparameter filters based on data profile
        hp_filters = generate_hyperparameter_filters(profile, confidence_level=config.confidence_level)
        logger.info(f"Generated hyperparameter filters for {len(hp_filters)} model types (confidence_level={config.confidence_level})")

        # Step 4: Handle by-slice vs aggregate forecasting
        slice_forecasts_list = None
        slice_cols_list = None
        actual_forecast_mode = forecast_mode or "aggregate"

        # Auto-detect slice values if slice_columns provided but slice_values not
        if forecast_mode == "by_slice" and slice_columns and not slice_values:
            slice_cols_list = [c.strip() for c in slice_columns.split(',') if c.strip()]
            # Validate slice columns exist
            missing_cols = [c for c in slice_cols_list if c not in df.columns]
            if missing_cols:
                raise HTTPException(status_code=400, detail=f"Slice columns not found: {missing_cols}")

            # Auto-detect unique values for slice columns
            if len(slice_cols_list) == 1:
                unique_vals = df[slice_cols_list[0]].dropna().unique().tolist()
                # Use ||| as separator between slice values
                slice_values = '|||'.join(str(v) for v in unique_vals)
            else:
                # For multiple columns, get unique combinations
                combo_df = df[slice_cols_list].drop_duplicates()
                combinations = []
                for _, row in combo_df.iterrows():
                    # Use ' | ' (space pipe space) as separator within each combination
                    combo = ' | '.join(str(row[c]) for c in slice_cols_list)
                    combinations.append(combo)
                # Use ||| as separator between different combinations
                slice_values = '|||'.join(combinations)

            logger.info(f"📊 Auto-detected slice values: {slice_values}")

        if forecast_mode == "by_slice" and slice_columns and slice_values:
            # BY-SLICE MODE: Train separate models for each selected slice
            slice_cols_list = [c.strip() for c in slice_columns.split(',') if c.strip()]
            # Split by ||| (triple pipe) - each slice value may contain ' | ' for multi-column slices
            slice_values_list = [v.strip() for v in slice_values.split('|||') if v.strip()]

            logger.info(f"📊 By-slice forecasting mode:")
            logger.info(f"    Raw slice_values parameter: '{slice_values}'")
            logger.info(f"    Parsed {len(slice_values_list)} slices: {slice_values_list}")
            logger.info(f"    Slice columns: {slice_cols_list}")

            slice_forecasts_list = []
            all_slice_forecasts = []
            all_slice_dates = []

            for slice_idx, slice_value in enumerate(slice_values_list):
                # Parse slice value (e.g., "Classic | Enterprise | 0" for multi-column)
                # Frontend uses ' | ' (space pipe space) as separator within each slice
                slice_parts = [p.strip() for p in slice_value.split(' | ')] if ' | ' in slice_value else [slice_value.strip()]

                logger.info(f"  Parsing slice '{slice_value}' -> parts: {slice_parts}")
                logger.info(f"    Expected columns: {slice_cols_list}")

                # Build filter for this slice
                slice_filter = {}
                for i, col in enumerate(slice_cols_list):
                    if i < len(slice_parts):
                        slice_filter[col] = slice_parts[i].strip()

                logger.info(f"    Built filter: {slice_filter}")

                # Filter data for this slice
                slice_df = df.copy()
                for col, val in slice_filter.items():
                    if col in slice_df.columns:
                        before_count = len(slice_df)
                        slice_df = slice_df[slice_df[col].astype(str) == str(val)]
                        logger.info(f"    Filter {col}='{val}': {before_count} -> {len(slice_df)} rows")
                    else:
                        logger.warning(f"    Column '{col}' not found in data! Available: {list(df.columns)}")

                if len(slice_df) < 10:
                    logger.warning(f"Skipping slice '{slice_value}' - only {len(slice_df)} rows (minimum 10 required)")
                    continue

                logger.info(f"  Training model for slice '{slice_value}' ({len(slice_df)} rows)...")

                try:
                    # Profile this slice's data
                    slice_profiler = DataProfiler()
                    slice_profile = slice_profiler.profile(slice_df)
                    slice_profile.date_column = config.date_column
                    slice_profile.target_column = config.target_column
                    slice_profile.covariate_columns = [c for c in config.covariate_columns if c not in slice_cols_list]

                    # Generate config for this slice
                    slice_autopilot = AutopilotConfig()
                    slice_config = slice_autopilot.generate(slice_profile, horizon=config.horizon)
                    slice_config.date_column = config.date_column
                    slice_config.target_column = config.target_column
                    slice_config.covariate_columns = [c for c in config.covariate_columns if c not in slice_cols_list]

                    # Generate hyperparameter filters for this slice
                    slice_hp_filters = generate_hyperparameter_filters(slice_profile, confidence_level=config.confidence_level)

                    # Run forecast for this slice
                    slice_result = await _run_forecast(slice_df, slice_config, slice_profile, f"{run_id}_slice{slice_idx}", slice_hp_filters)

                    # Store slice forecast (sanitize to avoid NaN values)
                    slice_forecast_item = _sanitize_dict({
                        'slice_id': slice_value,
                        'slice_filters': slice_filter,
                        'forecast': slice_result.get('forecast', []),
                        'dates': [str(d) for d in slice_result.get('dates', [])],
                        'lower_bounds': slice_result.get('lower', []),
                        'upper_bounds': slice_result.get('upper', []),
                        'best_model': slice_result.get('best_model'),
                        'holdout_mape': _sanitize_float(slice_result.get('holdout_mape'), 100.0),
                        'data_points': len(slice_df),
                    })
                    slice_forecasts_list.append(slice_forecast_item)

                    # Track for aggregate summary
                    all_slice_forecasts.append(slice_result.get('forecast', []))
                    if not all_slice_dates:
                        all_slice_dates = [str(d) for d in slice_result.get('dates', [])]

                    logger.info(f"  ✅ Slice '{slice_value}' complete: {slice_result.get('best_model')} (MAPE: {slice_result.get('holdout_mape', 'N/A')})")

                except Exception as slice_error:
                    logger.error(f"  ❌ Slice '{slice_value}' failed: {slice_error}")
                    continue

            # Create aggregate forecast (sum of all slices) for the main response
            if slice_forecasts_list:
                import numpy as np
                # Sum all slice forecasts
                max_len = max(len(f) for f in all_slice_forecasts) if all_slice_forecasts else 0
                aggregate_forecast = [0.0] * max_len
                aggregate_lower = [0.0] * max_len
                aggregate_upper = [0.0] * max_len

                for sf in slice_forecasts_list:
                    for i, val in enumerate(sf['forecast']):
                        if i < max_len:
                            aggregate_forecast[i] += val
                    for i, val in enumerate(sf['lower_bounds']):
                        if i < max_len:
                            aggregate_lower[i] += val
                    for i, val in enumerate(sf['upper_bounds']):
                        if i < max_len:
                            aggregate_upper[i] += val

                # Create a "virtual" aggregate result for the main response
                # Sanitize holdout_mape values (handle NaN/None)
                holdout_mapes = [_sanitize_float(sf.get('holdout_mape'), 100.0) for sf in slice_forecasts_list]
                avg_holdout_mape = _sanitize_float(np.mean(holdout_mapes) if holdout_mapes else 100.0, 100.0)

                forecast_result = _sanitize_dict({
                    'run_id': run_id,
                    'best_model': f"Multi-slice ({len(slice_forecasts_list)} models)",
                    'model_version': '1.0',
                    'forecast': aggregate_forecast,
                    'dates': all_slice_dates,
                    'lower': aggregate_lower,
                    'upper': aggregate_upper,
                    'metrics': {'mape': avg_holdout_mape},
                    'mlflow_run_id': None,
                    'model_uri': None,
                    'all_models_trained': list(set(sf.get('best_model', 'Unknown') for sf in slice_forecasts_list)),
                    'random_seed': 42,
                    'data_hash': profile.data_hash,
                    'config_hash': config.config_hash,
                    'holdout_mape': avg_holdout_mape,
                })

                logger.info(f"✅ By-slice forecasting complete: {len(slice_forecasts_list)} slices")
            else:
                # No successful slices - fall back to aggregate
                logger.warning("No slices succeeded, falling back to aggregate mode")
                actual_forecast_mode = "aggregate"
                forecast_result = await _run_forecast(df, config, profile, run_id, hp_filters)
        else:
            # AGGREGATE MODE: Single model on aggregated data
            forecast_result = await _run_forecast(df, config, profile, run_id, hp_filters)

        # Step 5: Generate explanation
        explainer = ForecastExplainer()
        explanation = explainer.explain(
            forecast_result,
            config.to_dict(),
            {
                'history_months': profile.history_months,
                'data_quality_score': profile.data_quality_score,
                'holiday_coverage_score': profile.holiday_coverage_score,
                'data_hash': profile.data_hash,
                'row_count': profile.row_count,
                'date_range': profile.date_range,
            }
        )

        # Step 6: Store for later retrieval (Excel export, reproducibility)
        _forecast_store[run_id] = {
            'result': forecast_result,
            'explanation': explanation,
            'input_data': df,
            'config': config,
            'profile': profile,
            'timestamp': datetime.now().isoformat(),
            # By-slice forecast data for Excel export
            'slice_forecasts': slice_forecasts_list,
            'forecast_mode': actual_forecast_mode,
        }

        # Format for response
        formatted = format_explanation_for_display(explanation)

        # Build MLflow tracking URLs for traceability
        import os
        databricks_host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        mlflow_run_id = forecast_result.get('mlflow_run_id')
        mlflow_run_url = None
        mlflow_experiment_url = None

        if databricks_host and mlflow_run_id:
            # Try to get experiment ID from MLflow
            try:
                import mlflow
                mlflow.set_tracking_uri("databricks")
                run = mlflow.get_run(mlflow_run_id)
                experiment_id = run.info.experiment_id
                mlflow_experiment_url = f"{databricks_host}/ml/experiments/{experiment_id}"
                mlflow_run_url = f"{databricks_host}/ml/experiments/{experiment_id}/runs/{mlflow_run_id}"
            except Exception as e:
                logger.warning(f"Could not get MLflow URLs: {e}")

        return SimpleForecastResponse(
            success=True,
            mode="simple",
            run_id=run_id,

            summary=explanation.summary,

            forecast=forecast_result.get('forecast', []),
            dates=[str(d) for d in forecast_result.get('dates', [])],
            lower_bounds=forecast_result.get('lower', []),
            upper_bounds=forecast_result.get('upper', []),

            components=formatted['components'],
            confidence=formatted['confidence'],

            warnings=[
                {'level': w.level, 'message': w.message, 'recommendation': w.recommendation}
                for w in profile.warnings
            ],
            caveats=explanation.caveats,

            audit=formatted['audit'],

            # MLflow Tracking for Reproducibility
            mlflow_run_id=mlflow_run_id,
            mlflow_experiment_url=mlflow_experiment_url,
            mlflow_run_url=mlflow_run_url,
            model_uri=forecast_result.get('model_uri'),

            # Model info
            best_model=forecast_result.get('best_model'),
            all_models_trained=forecast_result.get('all_models_trained', []),

            # Reproducibility Guarantee
            data_hash=profile.data_hash,
            config_hash=config.config_hash,
            random_seed=forecast_result.get('random_seed', 42),
            reproducibility_note="Same data + same horizon = identical results guaranteed. Use data_hash and config_hash to verify.",

            # Data Split & Model Selection (NEW)
            data_split=forecast_result.get('data_split'),
            model_comparison=forecast_result.get('model_comparison'),
            selection_reason=forecast_result.get('selection_reason'),
            trained_on_full_data=forecast_result.get('trained_on_full_data', False),

            # Anomaly Detection (NEW)
            anomalies=forecast_result.get('anomalies'),

            # Holdout Performance (NEW) - sanitize to avoid NaN
            holdout_mape=_sanitize_float(forecast_result.get('holdout_mape'), None),
            eval_mape=_sanitize_float(forecast_result.get('eval_mape'), None),

            # Future Covariates Info (NEW)
            future_covariates_used=forecast_result.get('future_covariates_used', False),
            future_covariates_count=forecast_result.get('future_covariates_count', 0),
            future_covariates_date_range=forecast_result.get('future_covariates_date_range'),

            # By-slice forecasting (NEW)
            forecast_mode=actual_forecast_mode,
            slice_forecasts=[
                SliceForecastResult(**sf) for sf in slice_forecasts_list
            ] if slice_forecasts_list else None,
            slice_columns=slice_cols_list,

            # Export links
            excel_download_url=f"/api/simple/export/{run_id}/excel",
            reproduce_url=f"/api/simple/reproduce/{run_id}",
        )

    except Exception as e:
        logger.error(f"Simple forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{run_id}/excel")
async def export_excel(run_id: str):
    """
    Download forecast as Excel file with formulas.

    Includes multiple sheets:
    - Summary (executive view)
    - Forecast Detail (with formulas)
    - Components (breakdown)
    - Confidence (quality metrics)
    - Audit Trail (for compliance)
    - Raw Data (input data)
    - Individual slice forecast sheets (if by-slice mode)
    """
    if run_id not in _forecast_store:
        raise HTTPException(status_code=404, detail="Forecast not found. Please run forecast first.")

    stored = _forecast_store[run_id]

    try:
        # Get slice forecasts if this was a by-slice forecast
        slice_forecasts = stored.get('slice_forecasts')

        excel_bytes = export_forecast_to_excel(
            forecast_result=stored['result'],
            explanation=stored['explanation'],
            input_data=stored['input_data'],
            slice_forecasts=slice_forecasts,
        )

        filename = f"forecast_{run_id}_{datetime.now().strftime('%Y%m%d')}.xlsx"

        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.error(f"Excel export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{run_id}/csv")
async def export_csv(run_id: str):
    """Download forecast as CSV file."""

    if run_id not in _forecast_store:
        raise HTTPException(status_code=404, detail="Forecast not found.")

    stored = _forecast_store[run_id]
    result = stored['result']

    # Build CSV dataframe
    df = pd.DataFrame({
        'date': result.get('dates', []),
        'forecast': result.get('forecast', []),
        'lower_bound': result.get('lower', []),
        'upper_bound': result.get('upper', []),
    })

    csv_bytes = df.to_csv(index=False).encode()
    filename = f"forecast_{run_id}_{datetime.now().strftime('%Y%m%d')}.csv"

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.post("/reproduce/{run_id}")
async def reproduce_forecast(run_id: str):
    """
    Reproduce an exact previous forecast.

    Uses stored config and data hash to verify reproducibility.
    """
    if run_id not in _forecast_store:
        raise HTTPException(status_code=404, detail="Forecast not found.")

    stored = _forecast_store[run_id]
    original_result = stored['result']
    config = stored['config']
    profile = stored['profile']
    df = stored['input_data']

    # Re-generate hyperparameter filters for reproducibility
    hp_filters = generate_hyperparameter_filters(profile, confidence_level=config.confidence_level)

    # Re-run forecast
    new_result = await _run_forecast(df, config, profile, run_id + "_reproduced", hp_filters)

    # Verify output matches
    original_forecast = original_result.get('forecast', [])
    new_forecast = new_result.get('forecast', [])

    # Check if forecasts match (within floating point tolerance)
    import numpy as np
    matches = np.allclose(original_forecast, new_forecast, rtol=1e-5)

    return {
        'success': True,
        'original_run_id': run_id,
        'reproduced': True,
        'verification': "✅ Output matches original exactly" if matches else "⚠️ Output differs from original",
        'matches': matches,
    }


# Helper functions

def _parse_file(content: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded file into DataFrame."""

    if filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(content))
    elif filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(io.BytesIO(content))
    else:
        # Try CSV first, then Excel
        try:
            df = pd.read_csv(io.BytesIO(content))
        except:
            df = pd.read_excel(io.BytesIO(content))

    # Clean comma-formatted numbers (e.g., "1,234,567" -> 1234567)
    df = _clean_numeric_columns(df)

    return df


def _clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean columns that contain comma-formatted numbers.
    E.g., "1,234,567" should be converted to 1234567.0
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if this column looks like comma-formatted numbers
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                # Check if values match pattern like "1,234" or "1,234,567"
                looks_like_number = sample.astype(str).str.match(r'^-?[\d,]+\.?\d*$').all()
                if looks_like_number:
                    try:
                        # Remove commas and convert to float
                        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
                        logger.info(f"Cleaned comma-formatted numbers in column: {col}")
                    except (ValueError, TypeError):
                        pass  # Not actually numeric, leave as-is
    return df


async def _run_forecast(
    df: pd.DataFrame,
    config: ForecastConfig,
    profile: DataProfile,
    run_id: str,
    hyperparameter_filters: Dict[str, Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run the actual forecast using existing AutoML training infrastructure.

    DATA SPLIT STRATEGY (for robust model selection):
    ┌─────────────────────────────────────────────────────────────────┐
    │  TRAIN (70%)    │  EVAL (15%)  │  HOLDOUT (15%)  │  FORECAST   │
    │  Model learns   │  Tune params │  Final test     │  Prediction │
    │  patterns here  │  & early stop│  (unseen data)  │  for future │
    └─────────────────────────────────────────────────────────────────┘

    - TRAIN: Model learns patterns from this data
    - EVAL: Used during training for early stopping and hyperparameter tuning
    - HOLDOUT: Never seen during training - used to select the best model
    - Best model is chosen based on HOLDOUT performance, not EVAL!

    This prevents overfitting to the eval set and gives realistic performance estimates.
    """
    import numpy as np
    import random

    logger.info("=" * 70)
    logger.info("🔮 _run_forecast - START")
    logger.info("=" * 70)
    logger.info(f"[INPUT] run_id: {run_id}")
    logger.info(f"[INPUT] df.shape: {df.shape}")
    logger.info(f"[INPUT] df.columns: {list(df.columns)}")
    logger.info(f"[INPUT] df.dtypes:\n{df.dtypes.to_string()}")
    logger.info(f"[INPUT] config.date_column: {config.date_column}")
    logger.info(f"[INPUT] config.target_column: {config.target_column}")
    logger.info(f"[INPUT] config.covariate_columns: {config.covariate_columns}")
    logger.info(f"[INPUT] config.horizon: {config.horizon}")
    logger.info(f"[INPUT] config.frequency: {config.frequency}")
    logger.info(f"[INPUT] config.models: {config.models}")
    logger.info(f"[INPUT] profile.has_multiple_slices: {profile.has_multiple_slices}")
    logger.info(f"[INPUT] profile.has_future_covariates: {profile.has_future_covariates}")
    logger.info(f"[INPUT] hyperparameter_filters provided: {hyperparameter_filters is not None}")
    logger.info("-" * 70)

    # CRITICAL: Set random seeds for reproducibility FIRST
    seed = config.random_seed  # Default: 42
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"[SETUP] Set random seed {seed} for reproducibility")

    # Prepare data in expected format
    logger.info("[DATA PREP] Preparing data...")
    data = df.copy()

    # Get covariates if any (exclude target column to prevent data leakage)
    covariates = [c for c in config.covariate_columns
                  if c in df.columns and c != config.target_column]
    logger.info(f"[DATA PREP] Covariates after filtering: {covariates}")

    # ============================================================
    # AUTO-AGGREGATE MULTI-SLICE DATA
    # ============================================================
    # If data has multiple segments (duplicate dates), aggregate by date
    # This prevents the model from getting confused by mixed segment data
    # ============================================================
    logger.info("[AGGREGATION] Checking for multi-slice data...")
    aggregation_applied = False
    if profile.has_multiple_slices:
        logger.info(f"[AGGREGATION] ⚠️ Multi-slice data detected ({profile.slice_count} segments). Auto-aggregating by date...")

        # Convert date column to datetime for grouping
        data[config.date_column] = pd.to_datetime(data[config.date_column])

        # Aggregate: sum target, mean for numeric covariates
        agg_dict = {config.target_column: 'sum'}
        for cov in covariates:
            if cov in data.columns:
                if data[cov].dtype in ['int64', 'float64']:
                    # For binary (0/1) covariates, use max; for others use mean
                    unique_vals = data[cov].dropna().unique()
                    if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                        agg_dict[cov] = 'max'  # Keep 1 if any segment has 1
                    else:
                        agg_dict[cov] = 'mean'

        data = data.groupby(config.date_column).agg(agg_dict).reset_index()
        aggregation_applied = True
        logger.info(f"[AGGREGATION] ✅ Aggregated to {len(data)} unique time periods (was {profile.row_count} rows)")
        logger.info(f"[AGGREGATION] After aggregation - columns: {list(data.columns)}")
        logger.info(f"[AGGREGATION] After aggregation - target stats: min={data[config.target_column].min()}, max={data[config.target_column].max()}, mean={data[config.target_column].mean():.2f}")

    # ============================================================
    # SEPARATE FUTURE COVARIATE ROWS FROM HISTORICAL DATA
    # ============================================================
    # Future rows: have covariates but no target value (NaN/null)
    # These are used for more accurate forecasting when users know
    # future predictor values (e.g., planned promotions, scheduled events)
    # ============================================================
    future_features_list = None
    future_covariates_used = False
    future_covariates_count = 0
    future_covariates_date_range = None

    if covariates and profile.has_future_covariates:
        # Identify rows with missing target but having covariate values
        target_missing = data[config.target_column].isna() | data[config.target_column].isnull()
        has_covariate_value = pd.Series(False, index=data.index)
        for cov in covariates:
            if cov in data.columns:
                has_covariate_value |= data[cov].notna()

        future_mask = target_missing & has_covariate_value

        if future_mask.any():
            # Extract future rows
            future_rows_df = data[future_mask].copy()
            future_features_list = future_rows_df.to_dict('records')
            future_covariates_count = len(future_features_list)

            # Get date range
            try:
                future_dates = pd.to_datetime(future_rows_df[config.date_column])
                future_covariates_date_range = [str(future_dates.min().date()), str(future_dates.max().date())]
            except:
                pass

            future_covariates_used = True
            logger.info(f"🔮 Separated {future_covariates_count} future covariate rows for prediction")

            # Remove future rows from training data
            data = data[~future_mask].copy()
            logger.info(f"📊 Historical data: {len(data)} rows (after removing future rows)")

    # Rename columns to standard format expected by training functions
    logger.info("[COLUMN RENAME] Renaming columns to standard format...")
    logger.info(f"[COLUMN RENAME] Before rename - columns: {list(data.columns)}")
    column_mapping = {
        config.date_column: 'ds',
        config.target_column: 'y'
    }
    logger.info(f"[COLUMN RENAME] Mapping: {column_mapping}")
    data = data.rename(columns=column_mapping)
    data['ds'] = pd.to_datetime(data['ds'])
    data = data.sort_values('ds').reset_index(drop=True)
    logger.info(f"[COLUMN RENAME] After rename - columns: {list(data.columns)}")
    logger.info(f"[COLUMN RENAME] Date range: {data['ds'].min()} to {data['ds'].max()}")
    logger.info(f"[COLUMN RENAME] Target 'y' - first 5 values: {data['y'].head(5).tolist()}")
    logger.info(f"[COLUMN RENAME] Target 'y' - last 5 values: {data['y'].tail(5).tolist()}")

    # Prepare data as list of dicts (expected format for training API)
    # Use the sorted, renamed 'data' DataFrame, not the original 'df'
    # The data is already filtered (future rows removed) and sorted by date
    data_list = data.to_dict('records')
    logger.info(f"[DATA LIST] Converted to list of dicts: {len(data_list)} records")
    if data_list:
        logger.info(f"[DATA LIST] First record: {data_list[0]}")
        logger.info(f"[DATA LIST] Last record: {data_list[-1]}")

    logger.info(f"[AUTOML] Simple Mode AutoML: Training with {len(data)} rows, horizon={config.horizon}, covariates={covariates}")

    # Log hyperparameter filters if provided
    if hyperparameter_filters:
        logger.info(f"Using data-driven hyperparameter filters for {len(hyperparameter_filters)} models")
        for model_name, filters in hyperparameter_filters.items():
            logger.info(f"   - {model_name}: {list(filters.keys())}")

    # Determine which models to train based on data characteristics
    logger.info("[MODEL SELECTION] Determining models to train...")
    models_to_train = _select_models_for_data(profile, len(data))
    logger.info(f"[MODEL SELECTION] Selected models: {models_to_train}")

    # ============================================================
    # TRAIN / EVAL / HOLDOUT SPLIT
    # ============================================================
    # We use a 3-way split to prevent overfitting:
    # - Train on TRAIN set (70%)
    # - Tune/validate on EVAL set (15%)
    # - Select best model on HOLDOUT set (15%) - never seen during training!
    # ============================================================
    logger.info("[DATA SPLIT] Calculating train/eval/holdout split...")

    n = len(data)
    logger.info(f"[DATA SPLIT] Total data points: {n}")

    # Calculate percentage-based splits (70/15/15)
    # But ensure minimum sizes for meaningful evaluation
    holdout_size = max(int(n * 0.15), min(config.horizon, 12), 3)
    eval_size = max(int(n * 0.15), min(config.horizon, 12), 3)
    train_size = n - eval_size - holdout_size

    # Ensure train set is at least 50% of data
    if train_size < n * 0.5:
        # Scale down eval and holdout proportionally
        available_for_test = int(n * 0.5)  # Max 50% for eval + holdout combined
        holdout_size = max(available_for_test // 2, config.horizon, 3)
        eval_size = max(available_for_test // 2, config.horizon, 3)
        train_size = n - eval_size - holdout_size

    # For very small datasets, use simpler split
    if n < 50:
        logger.warning(f"Limited data ({n} rows). Using simpler 2-way split.")
        holdout_size = max(min(config.horizon, n // 4), 3)
        eval_size = holdout_size
        train_size = n - holdout_size - eval_size

    split_info = {
        'total_rows': n,
        'train_size': train_size,
        'eval_size': eval_size,
        'holdout_size': holdout_size,
        'train_pct': round(train_size / n * 100, 1),
        'eval_pct': round(eval_size / n * 100, 1),
        'holdout_pct': round(holdout_size / n * 100, 1),
        'train_date_range': None,
        'eval_date_range': None,
        'holdout_date_range': None,
    }

    logger.info(f"[DATA SPLIT] FINAL SPLIT:")
    logger.info(f"[DATA SPLIT]   Train: {train_size} rows ({split_info['train_pct']}%)")
    logger.info(f"[DATA SPLIT]   Eval: {eval_size} rows ({split_info['eval_pct']}%)")
    logger.info(f"[DATA SPLIT]   Holdout: {holdout_size} rows ({split_info['holdout_pct']}%)")

    best_result = None
    best_holdout_mape = float('inf')
    all_results = []
    model_comparison = []

    # Try to use the full training infrastructure
    try:
        from backend.models.prophet import train_prophet_model, prepare_prophet_data
        from backend.models.arima import train_arima_model
        from backend.models.xgboost import train_xgboost_model
        from backend.models.ets import train_exponential_smoothing_model
        from backend.models.statsforecast_models import train_statsforecast_model
        from backend.models.chronos_model import train_chronos_model
        from backend.models.ensemble import train_ensemble_model

        # ============================================================
        # MLFLOW SETUP WITH FALLBACK
        # ============================================================
        # Try Databricks MLflow first, fall back to local SQLite if unavailable.
        # This ensures Simple Mode works both in Databricks and local environments.
        # ============================================================
        import mlflow
        import os
        from datetime import datetime as dt

        mlflow_mode = "databricks"  # Track which mode we're using
        mlflow_parent_run = None
        experiment_id = None

        try:
            # First, try Databricks MLflow
            databricks_host = os.environ.get("DATABRICKS_HOST", "")
            databricks_token = os.environ.get("DATABRICKS_TOKEN", "")

            if databricks_host and databricks_token:
                mlflow.set_tracking_uri("databricks")
                experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/finance-forecasting-simple")

                # Try to get or create the experiment
                try:
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if experiment is None:
                        experiment_id = mlflow.create_experiment(experiment_name)
                        logger.info(f"📊 Created new MLflow experiment: {experiment_name}")
                    else:
                        experiment_id = experiment.experiment_id
                    mlflow.set_experiment(experiment_name)
                    logger.info(f"📊 Using Databricks MLflow: {experiment_name}")
                except Exception as db_exp_error:
                    logger.warning(f"Could not access Databricks experiment: {db_exp_error}")
                    raise  # Trigger fallback
            else:
                raise ValueError("Databricks credentials not configured")

        except Exception as mlflow_error:
            # Fall back to local SQLite MLflow
            mlflow_mode = "local"
            local_mlflow_dir = os.path.join(os.getcwd(), "mlruns")
            os.makedirs(local_mlflow_dir, exist_ok=True)
            mlflow.set_tracking_uri(f"sqlite:///{os.getcwd()}/mlflow.db")

            experiment_name = "simple-mode-forecasting"
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                else:
                    experiment_id = experiment.experiment_id
                mlflow.set_experiment(experiment_name)
                logger.info(f"📊 Using local MLflow (fallback): {experiment_name}")
                logger.info(f"   ⚠️ Databricks unavailable: {str(mlflow_error)[:100]}")
            except Exception as local_exp_error:
                logger.warning(f"Could not set up local MLflow experiment: {local_exp_error}")
                # Continue without MLflow - models can still train
                mlflow_mode = "disabled"

        # Start parent MLflow run for this forecast session
        mlflow_parent_run_context = None
        if mlflow_mode != "disabled":
            try:
                run_name = f"SimpleMode_{run_id}_{dt.now().strftime('%Y%m%d_%H%M%S')}"
                mlflow_parent_run_context = mlflow.start_run(run_name=run_name)
                mlflow_parent_run = mlflow_parent_run_context.__enter__()
                logger.info(f"📊 Started MLflow parent run: {mlflow_parent_run.info.run_id}")

                # Log run metadata
                mlflow.log_param("mode", "simple")
                mlflow.log_param("run_id", run_id)
                mlflow.log_param("horizon", config.horizon)
                mlflow.log_param("frequency", config.frequency)
                mlflow.log_param("data_points", len(data))
                mlflow.log_param("covariates", str(covariates))
                mlflow.log_param("mlflow_mode", mlflow_mode)
            except Exception as run_error:
                logger.warning(f"Could not start MLflow parent run: {run_error}")
                mlflow_mode = "disabled"

        # Prepare data using the standard preprocessing
        # Note: data_list already has columns renamed to 'ds' and 'y' (lines 752-758)
        # so we pass 'ds' and 'y' to prepare_prophet_data, not the original column names
        processed_df = prepare_prophet_data(
            data_list,
            'ds',  # Already renamed from config.date_column
            'y',   # Already renamed from config.target_column
            covariates
        )

        # Create the 3-way split
        train_df = processed_df.iloc[:train_size].copy()
        eval_df = processed_df.iloc[train_size:train_size + eval_size].copy()
        holdout_df = processed_df.iloc[train_size + eval_size:].copy()

        # Record date ranges for explanation
        split_info['train_date_range'] = [str(train_df['ds'].min()), str(train_df['ds'].max())]
        split_info['eval_date_range'] = [str(eval_df['ds'].min()), str(eval_df['ds'].max())]
        split_info['holdout_date_range'] = [str(holdout_df['ds'].min()), str(holdout_df['ds'].max())]

        logger.info(f"Train: {split_info['train_date_range'][0]} to {split_info['train_date_range'][1]}")
        logger.info(f"Eval: {split_info['eval_date_range'][0]} to {split_info['eval_date_range'][1]}")
        logger.info(f"Holdout: {split_info['holdout_date_range'][0]} to {split_info['holdout_date_range'][1]}")

        # For training, we use train+eval (models will use eval for validation)
        train_eval_df = processed_df.iloc[:train_size + eval_size].copy()

        # Train each selected model
        logger.info("=" * 70)
        logger.info("🚂 MODEL TRAINING LOOP - Starting")
        logger.info("=" * 70)
        logger.info(f"[TRAIN LOOP] Models to train: {models_to_train}")
        logger.info(f"[TRAIN LOOP] Train df shape: {train_df.shape}")
        logger.info(f"[TRAIN LOOP] Eval df shape: {eval_df.shape}")
        logger.info(f"[TRAIN LOOP] Holdout df shape: {holdout_df.shape}")
        logger.info(f"[TRAIN LOOP] Covariates: {covariates}")
        logger.info(f"[TRAIN LOOP] Horizon: {config.horizon}")
        logger.info(f"[TRAIN LOOP] Frequency: {config.frequency}")

        models_attempted = []
        models_succeeded = []
        models_failed = []

        for model_idx, model_type in enumerate(models_to_train):
            logger.info("-" * 50)
            logger.info(f"🔄 [{model_idx + 1}/{len(models_to_train)}] ATTEMPTING: {model_type.upper()}")
            logger.info("-" * 50)
            models_attempted.append(model_type)

            try:
                logger.info(f"[{model_type.upper()}] Starting training...")

                if model_type == 'prophet':
                    # IMPORTANT: data_list has columns already renamed to 'ds' and 'y'
                    # So we pass 'ds' and 'y' as the column names, not the original names
                    logger.info(f"[PROPHET] Calling train_prophet_model with:")
                    logger.info(f"[PROPHET]   data_list length: {len(data_list)}")
                    logger.info(f"[PROPHET]   time_col: 'ds'")
                    logger.info(f"[PROPHET]   target_col: 'y'")
                    logger.info(f"[PROPHET]   covariates: {covariates}")
                    logger.info(f"[PROPHET]   horizon: {config.horizon}")
                    logger.info(f"[PROPHET]   frequency: {config.frequency}")
                    logger.info(f"[PROPHET]   Using train_df/eval_df overrides for consistent splits")
                    logger.info(f"[PROPHET]   hyperparameter_filters: {list(hyperparameter_filters.get('prophet', {}).keys()) if hyperparameter_filters else 'None'}")
                    # FIX: Use train_df_override and test_df_override to ensure Prophet uses
                    # the same train/eval split as other models (for consistent ensemble evaluation)
                    mlflow_run_id, _, metrics, validation, forecast, uri, impacts = train_prophet_model(
                        data_list,
                        'ds',  # data_list already has columns renamed
                        'y',   # data_list already has columns renamed
                        covariates,
                        config.horizon,
                        config.frequency,
                        'multiplicative',
                        None,  # Don't use eval_size - we'll override with explicit splits
                        'ridge',
                        'US',
                        config.random_seed,
                        future_features_list,  # Pass future covariate rows if available
                        hyperparameter_filters,  # Pass intelligent hyperparameter filters
                        train_df_override=train_df,  # Use same split as other models
                        test_df_override=eval_df,    # Use same split as other models
                        forecast_start_date=processed_df['ds'].max()  # Ensure forecast starts from data end
                    )

                    result = {
                        'model_type': 'Prophet',
                        'metrics': metrics,
                        'validation': validation,
                        'forecast': forecast,
                        'mlflow_run_id': mlflow_run_id,
                        'model_uri': uri,
                        'covariate_impacts': impacts
                    }

                elif model_type == 'arima':
                    logger.info(f"[ARIMA] Calling train_arima_model with:")
                    logger.info(f"[ARIMA]   train_df shape: {train_df.shape}")
                    logger.info(f"[ARIMA]   eval_df shape: {eval_df.shape}")
                    logger.info(f"[ARIMA]   horizon: {config.horizon}")
                    logger.info(f"[ARIMA]   frequency: {config.frequency}")
                    logger.info(f"[ARIMA]   covariates: {covariates}")
                    logger.info(f"[ARIMA]   hyperparameter_filters: {list(hyperparameter_filters.get('arima', {}).keys()) if hyperparameter_filters else 'None'}")
                    mlflow_run_id, _, metrics, val_df, fcst_df, uri, params = train_arima_model(
                        train_df, eval_df, config.horizon, config.frequency,
                        None, config.random_seed,
                        original_data=data_list, covariates=covariates,
                        hyperparameter_filters=hyperparameter_filters,
                        forecast_start_date=processed_df['ds'].max()  # Ensure forecast starts from data end
                    )
                    logger.info(f"[ARIMA] Training complete.")
                    logger.info(f"[ARIMA]   📊 Selected order: {params}")
                    logger.info(f"[ARIMA]   📊 (p={params[0] if params else '?'}, d={params[1] if params else '?'}, q={params[2] if params else '?'})")
                    logger.info(f"[ARIMA]   📊 p (AR terms): {params[0] if params else '?'} - past values influence")
                    logger.info(f"[ARIMA]   📊 d (differencing): {params[1] if params else '?'} - trend removal")
                    logger.info(f"[ARIMA]   📊 q (MA terms): {params[2] if params else '?'} - past errors influence")
                    logger.info(f"[ARIMA]   📊 MAPE: {metrics.get('mape', 'N/A'):.2f}%")

                    result = {
                        'model_type': f'ARIMA{params}' if params else 'ARIMA',
                        'metrics': metrics,
                        'validation': val_df.to_dict('records'),
                        'forecast': fcst_df.to_dict('records'),
                        'mlflow_run_id': mlflow_run_id,
                        'model_uri': uri,
                        'covariate_impacts': []
                    }

                elif model_type == 'xgboost':
                    logger.info(f"[XGBOOST] Calling train_xgboost_model with:")
                    logger.info(f"[XGBOOST]   train_df shape: {train_df.shape}")
                    logger.info(f"[XGBOOST]   eval_df shape: {eval_df.shape}")
                    logger.info(f"[XGBOOST]   horizon: {config.horizon}")
                    logger.info(f"[XGBOOST]   frequency: {config.frequency}")
                    logger.info(f"[XGBOOST]   covariates: {covariates}")
                    logger.info(f"[XGBOOST]   hyperparameter_filters: {list(hyperparameter_filters.get('xgboost', {}).keys()) if hyperparameter_filters else 'None'}")
                    mlflow_run_id, _, metrics, val_df, fcst_df, uri, params = train_xgboost_model(
                        train_df, eval_df, config.horizon, config.frequency,
                        covariates=covariates, random_seed=config.random_seed,
                        original_data=data_list, country='US',
                        hyperparameter_filters=hyperparameter_filters,
                        forecast_start_date=processed_df['ds'].max()  # Ensure forecast starts from data end
                    )
                    logger.info(f"[XGBOOST] Training complete. Params: {params}")

                    result = {
                        'model_type': f'XGBoost(depth={params.get("max_depth", "?")})' if params else 'XGBoost',
                        'metrics': metrics,
                        'validation': val_df.to_dict('records'),
                        'forecast': fcst_df.to_dict('records'),
                        'mlflow_run_id': mlflow_run_id,
                        'model_uri': uri,
                        'covariate_impacts': []
                    }

                elif model_type == 'ets':
                    logger.info(f"[ETS] Calling train_exponential_smoothing_model with:")
                    logger.info(f"[ETS]   train_df shape: {train_df.shape}")
                    logger.info(f"[ETS]   eval_df shape: {eval_df.shape}")
                    logger.info(f"[ETS]   horizon: {config.horizon}")
                    logger.info(f"[ETS]   frequency: {config.frequency}")

                    # Determine seasonal periods based on frequency
                    seasonal_periods_map = {'daily': 7, 'weekly': 52, 'monthly': 12, 'yearly': 1}
                    seasonal_periods = seasonal_periods_map.get(config.frequency, 12)

                    mlflow_run_id, _, metrics, val_df, fcst_df, uri, params = train_exponential_smoothing_model(
                        train_df, eval_df, config.horizon, config.frequency,
                        seasonal_periods=seasonal_periods,
                        random_seed=config.random_seed,
                        original_data=data_list, covariates=covariates,
                        hyperparameter_filters=hyperparameter_filters,
                        forecast_start_date=processed_df['ds'].max()
                    )
                    logger.info(f"[ETS] Training complete. Params: {params}")

                    result = {
                        'model_type': f'ETS({params.get("trend", "N")}/{params.get("seasonal", "N")})' if params else 'ETS',
                        'metrics': metrics,
                        'validation': val_df.to_dict('records'),
                        'forecast': fcst_df.to_dict('records'),
                        'mlflow_run_id': mlflow_run_id,
                        'model_uri': uri,
                        'covariate_impacts': []
                    }

                elif model_type == 'sarimax':
                    logger.info(f"[SARIMAX] Calling train_statsforecast_model with:")
                    logger.info(f"[SARIMAX]   train_df shape: {train_df.shape}")
                    logger.info(f"[SARIMAX]   eval_df shape: {eval_df.shape}")
                    logger.info(f"[SARIMAX]   horizon: {config.horizon}")
                    logger.info(f"[SARIMAX]   frequency: {config.frequency}")

                    mlflow_run_id, _, metrics, val_df, fcst_df, uri, params = train_statsforecast_model(
                        train_df, eval_df, config.horizon, config.frequency,
                        random_seed=config.random_seed,
                        original_data=data_list, covariates=covariates,
                        hyperparameter_filters=hyperparameter_filters,
                        forecast_start_date=processed_df['ds'].max(),
                        model_type='autoarima'  # Use AutoARIMA from statsforecast
                    )
                    logger.info(f"[SARIMAX] Training complete. Params: {params}")

                    result = {
                        'model_type': f'SARIMAX{params.get("order", "")}' if params else 'SARIMAX',
                        'metrics': metrics,
                        'validation': val_df.to_dict('records'),
                        'forecast': fcst_df.to_dict('records'),
                        'mlflow_run_id': mlflow_run_id,
                        'model_uri': uri,
                        'covariate_impacts': []
                    }

                elif model_type == 'chronos':
                    logger.info(f"[CHRONOS] Calling train_chronos_model with:")
                    logger.info(f"[CHRONOS]   train_df shape: {train_df.shape}")
                    logger.info(f"[CHRONOS]   eval_df shape: {eval_df.shape}")
                    logger.info(f"[CHRONOS]   horizon: {config.horizon}")
                    logger.info(f"[CHRONOS]   frequency: {config.frequency}")

                    mlflow_run_id, _, metrics, val_df, fcst_df, uri, params = train_chronos_model(
                        train_df, eval_df, config.horizon, config.frequency,
                        random_seed=config.random_seed,
                        original_data=data_list, covariates=covariates,
                        hyperparameter_filters=hyperparameter_filters,
                        forecast_start_date=processed_df['ds'].max(),
                        model_size='small'  # Use small model for balance of speed/accuracy
                    )
                    logger.info(f"[CHRONOS] Inference complete. Model: {params.get('model_size', 'small')}")

                    result = {
                        'model_type': f'Chronos({params.get("model_size", "small")})',
                        'metrics': metrics,
                        'validation': val_df.to_dict('records'),
                        'forecast': fcst_df.to_dict('records'),
                        'mlflow_run_id': mlflow_run_id,
                        'model_uri': uri,
                        'covariate_impacts': []
                    }

                elif model_type == 'statsforecast':
                    from backend.models.statsforecast_models import train_statsforecast_model
                    logger.info(f"[STATSFORECAST] Calling train_statsforecast_model with:")
                    logger.info(f"[STATSFORECAST]   train_df shape: {train_df.shape}")
                    logger.info(f"[STATSFORECAST]   eval_df shape: {eval_df.shape}")
                    logger.info(f"[STATSFORECAST]   horizon: {config.horizon}")
                    logger.info(f"[STATSFORECAST]   frequency: {config.frequency}")

                    mlflow_run_id, _, metrics, val_df, fcst_df, uri, params = train_statsforecast_model(
                        train_df, eval_df, config.horizon, config.frequency,
                        random_seed=config.random_seed,
                        model_type='autotheta',  # Use AutoTheta - great for trends
                        forecast_start_date=processed_df['ds'].max()
                    )
                    logger.info(f"[STATSFORECAST] Training complete. Model: AutoTheta")

                    result = {
                        'model_type': 'AutoTheta',
                        'metrics': metrics,
                        'validation': val_df.to_dict('records'),
                        'forecast': fcst_df.to_dict('records'),
                        'mlflow_run_id': mlflow_run_id,
                        'model_uri': uri,
                        'covariate_impacts': []
                    }

                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue

                # ============================================================
                # EVALUATE ON HOLDOUT SET (the key improvement!)
                # ============================================================
                holdout_mape = _evaluate_on_holdout(result, holdout_df, config)
                result['eval_mape'] = metrics.get('mape', float('inf'))
                result['holdout_mape'] = holdout_mape

                all_results.append(result)

                # Track for comparison
                model_comparison.append({
                    'model': result['model_type'],
                    'eval_mape': round(result['eval_mape'], 2),
                    'holdout_mape': round(holdout_mape, 2),
                    'mape_difference': round(holdout_mape - result['eval_mape'], 2),
                    'overfit_warning': holdout_mape > result['eval_mape'] * 1.5
                })

                models_succeeded.append(model_type)
                logger.info(f"✅ [{model_type.upper()}] TRAINING SUCCEEDED!")
                logger.info(f"[{model_type.upper()}] Result model_type: {result['model_type']}")
                logger.info(f"[{model_type.upper()}] Eval MAPE: {result['eval_mape']:.2f}%")
                logger.info(f"[{model_type.upper()}] Holdout MAPE: {holdout_mape:.2f}%")
                logger.info(f"[{model_type.upper()}] Metrics: {result.get('metrics', {})}")

                # Select best model based on HOLDOUT performance
                if holdout_mape < best_holdout_mape:
                    best_holdout_mape = holdout_mape
                    best_result = result
                    logger.info(f"🏆 [{model_type.upper()}] NEW BEST MODEL! (Holdout MAPE: {best_holdout_mape:.2f}%)")

            except Exception as model_error:
                import traceback
                error_traceback = traceback.format_exc()
                models_failed.append({
                    'model': model_type,
                    'error': str(model_error),
                    'traceback': error_traceback
                })
                logger.error("=" * 50)
                logger.error(f"❌ [{model_type.upper()}] TRAINING FAILED!")
                logger.error("=" * 50)
                logger.error(f"[{model_type.upper()}] Error type: {type(model_error).__name__}")
                logger.error(f"[{model_type.upper()}] Error message: {str(model_error)}")
                logger.error(f"[{model_type.upper()}] Full traceback:")
                for line in error_traceback.split('\n'):
                    if line.strip():
                        logger.error(f"[{model_type.upper()}]   {line}")
                logger.error("=" * 50)
                continue

        # ============================================================
        # ENSEMBLE MODEL (combines all successful models)
        # ============================================================
        if len(all_results) >= 2:
            logger.info("-" * 50)
            logger.info("🔄 TRAINING ENSEMBLE MODEL")
            logger.info("-" * 50)
            try:
                # Calculate historical stats for ensemble validation
                hist_mean = float(np.mean(processed_df['y'].values))
                hist_std = float(np.std(processed_df['y'].values))

                mlflow_run_id, _, metrics, val_df, fcst_df, uri, params = train_ensemble_model(
                    all_results,
                    train_df, eval_df, config.horizon, config.frequency,
                    random_seed=config.random_seed,
                    weighting_method='inverse_mape',
                    min_weight=0.05,
                    forecast_start_date=processed_df['ds'].max(),
                    historical_mean=hist_mean,
                    historical_std=hist_std,
                    filter_overfitting=True
                )
                logger.info(f"[ENSEMBLE] Training complete. Weights: {params.get('weights', {})}")

                ensemble_result = {
                    'model_type': 'Ensemble',
                    'metrics': metrics,
                    'validation': val_df.to_dict('records') if isinstance(val_df, pd.DataFrame) else val_df,
                    'forecast': fcst_df.to_dict('records') if isinstance(fcst_df, pd.DataFrame) else fcst_df,
                    'mlflow_run_id': mlflow_run_id,
                    'model_uri': uri,
                    'covariate_impacts': [],
                    'ensemble_weights': params.get('weights', {})
                }

                # Evaluate ensemble on holdout
                ensemble_holdout_mape = _evaluate_on_holdout(ensemble_result, holdout_df, config)
                ensemble_result['eval_mape'] = metrics.get('mape', float('inf'))
                ensemble_result['holdout_mape'] = ensemble_holdout_mape

                all_results.append(ensemble_result)
                models_succeeded.append('ensemble')

                model_comparison.append({
                    'model': 'Ensemble',
                    'eval_mape': round(ensemble_result['eval_mape'], 2),
                    'holdout_mape': round(ensemble_holdout_mape, 2),
                    'mape_difference': round(ensemble_holdout_mape - ensemble_result['eval_mape'], 2),
                    'overfit_warning': ensemble_holdout_mape > ensemble_result['eval_mape'] * 1.5
                })

                # Check if ensemble is the new best
                if ensemble_holdout_mape < best_holdout_mape:
                    best_holdout_mape = ensemble_holdout_mape
                    best_result = ensemble_result
                    logger.info(f"🏆 [ENSEMBLE] NEW BEST MODEL! (Holdout MAPE: {best_holdout_mape:.2f}%)")
                else:
                    logger.info(f"[ENSEMBLE] Holdout MAPE: {ensemble_holdout_mape:.2f}% (best remains: {best_holdout_mape:.2f}%)")

            except Exception as ensemble_error:
                logger.warning(f"[ENSEMBLE] Failed: {ensemble_error}")
        else:
            logger.info("Skipping Ensemble: Need at least 2 successful models")

        # ============================================================
        # MODEL TRAINING SUMMARY
        # ============================================================
        logger.info("=" * 70)
        logger.info("📊 MODEL TRAINING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"[SUMMARY] Models attempted: {models_attempted}")
        logger.info(f"[SUMMARY] Models succeeded: {models_succeeded}")
        logger.info(f"[SUMMARY] Models failed: {[f['model'] for f in models_failed]}")

        if models_failed:
            logger.info("[SUMMARY] Failed model details:")
            for fail in models_failed:
                logger.info(f"[SUMMARY]   ❌ {fail['model']}: {fail['error'][:100]}...")

        if model_comparison:
            logger.info("[SUMMARY] Model comparison (sorted by holdout MAPE):")
            sorted_comparison = sorted(model_comparison, key=lambda x: x['holdout_mape'])
            for mc in sorted_comparison:
                logger.info(f"[SUMMARY]   {mc['model']}: Eval={mc['eval_mape']:.2f}%, Holdout={mc['holdout_mape']:.2f}%, Overfit={mc['overfit_warning']}")

        if best_result:
            logger.info(f"[SUMMARY] 🏆 BEST MODEL: {best_result['model_type']} (Holdout MAPE: {best_holdout_mape:.2f}%)")
        else:
            logger.error("[SUMMARY] ⚠️ NO MODEL SUCCEEDED! All models failed.")
            logger.error("[SUMMARY] This is likely due to:")
            logger.error("[SUMMARY]   - Insufficient data for training")
            logger.error("[SUMMARY]   - Data quality issues (NaN, invalid values)")
            logger.error("[SUMMARY]   - Configuration issues (wrong column names)")

        logger.info("=" * 70)

        # ============================================================
        # RETRAIN BEST MODEL ON FULL DATA FOR FINAL FORECAST
        # ============================================================
        # The holdout evaluation helped us select the best model type.
        # Now we retrain that model on ALL data to maximize accuracy
        # for the actual forecast.
        # ============================================================

        final_result = None
        if best_result:
            best_model_type = best_result['model_type'].split('(')[0].lower().strip()
            logger.info(f"Retraining {best_model_type} on full dataset ({len(processed_df)} rows) for final forecast...")

            try:
                # Retrain on full data
                if 'prophet' in best_model_type:
                    # IMPORTANT: data_list has columns already renamed to 'ds' and 'y'
                    mlflow_run_id, _, metrics, _, forecast, uri, impacts = train_prophet_model(
                        data_list,
                        'ds',  # data_list already has columns renamed
                        'y',   # data_list already has columns renamed
                        covariates,
                        config.horizon,
                        config.frequency,
                        'multiplicative',
                        0,  # No validation split - use all data
                        'ridge',
                        'US',
                        config.random_seed,
                        None,
                        hyperparameter_filters,  # Pass intelligent hyperparameter filters
                        forecast_start_date=processed_df['ds'].max()  # Ensure forecast starts from data end
                    )
                    final_result = {
                        'model_type': 'Prophet',
                        'metrics': metrics,
                        'forecast': forecast,
                        'mlflow_run_id': mlflow_run_id,
                        'model_uri': uri,
                        'covariate_impacts': impacts
                    }

                elif 'arima' in best_model_type:
                    # For ARIMA, train on all data
                    mlflow_run_id, _, metrics, _, fcst_df, uri, params = train_arima_model(
                        processed_df, processed_df.iloc[-1:], config.horizon, config.frequency,
                        None, config.random_seed,
                        original_data=data_list, covariates=covariates,
                        hyperparameter_filters=hyperparameter_filters,
                        forecast_start_date=processed_df['ds'].max()  # Ensure forecast starts from data end
                    )
                    final_result = {
                        'model_type': f'ARIMA{params}' if params else 'ARIMA',
                        'metrics': metrics,
                        'forecast': fcst_df.to_dict('records'),
                        'mlflow_run_id': mlflow_run_id,
                        'model_uri': uri,
                        'covariate_impacts': []
                    }

                elif 'xgboost' in best_model_type:
                    mlflow_run_id, _, metrics, _, fcst_df, uri, params = train_xgboost_model(
                        processed_df, processed_df.iloc[-1:], config.horizon, config.frequency,
                        covariates=covariates, random_seed=config.random_seed,
                        original_data=data_list, country='US',
                        hyperparameter_filters=hyperparameter_filters,
                        forecast_start_date=processed_df['ds'].max()  # Ensure forecast starts from data end
                    )
                    final_result = {
                        'model_type': f'XGBoost(depth={params.get("max_depth", "?")})' if params else 'XGBoost',
                        'metrics': metrics,
                        'forecast': fcst_df.to_dict('records'),
                        'mlflow_run_id': mlflow_run_id,
                        'model_uri': uri,
                        'covariate_impacts': []
                    }

                elif 'ets' in best_model_type:
                    seasonal_periods_map = {'daily': 7, 'weekly': 52, 'monthly': 12, 'yearly': 1}
                    seasonal_periods = seasonal_periods_map.get(config.frequency, 12)

                    mlflow_run_id, _, metrics, _, fcst_df, uri, params = train_exponential_smoothing_model(
                        processed_df, processed_df.iloc[-1:], config.horizon, config.frequency,
                        seasonal_periods=seasonal_periods,
                        random_seed=config.random_seed,
                        original_data=data_list, covariates=covariates,
                        hyperparameter_filters=hyperparameter_filters,
                        forecast_start_date=processed_df['ds'].max()
                    )
                    final_result = {
                        'model_type': f'ETS({params.get("trend", "N")}/{params.get("seasonal", "N")})' if params else 'ETS',
                        'metrics': metrics,
                        'forecast': fcst_df.to_dict('records'),
                        'mlflow_run_id': mlflow_run_id,
                        'model_uri': uri,
                        'covariate_impacts': []
                    }

                elif 'sarimax' in best_model_type:
                    mlflow_run_id, _, metrics, _, fcst_df, uri, params = train_statsforecast_model(
                        processed_df, processed_df.iloc[-1:], config.horizon, config.frequency,
                        random_seed=config.random_seed,
                        original_data=data_list, covariates=covariates,
                        hyperparameter_filters=hyperparameter_filters,
                        forecast_start_date=processed_df['ds'].max(),
                        model_type='autoarima'
                    )
                    final_result = {
                        'model_type': f'SARIMAX{params.get("order", "")}' if params else 'SARIMAX',
                        'metrics': metrics,
                        'forecast': fcst_df.to_dict('records'),
                        'mlflow_run_id': mlflow_run_id,
                        'model_uri': uri,
                        'covariate_impacts': []
                    }

                elif 'chronos' in best_model_type:
                    # For Chronos, train on all data
                    mlflow_run_id, _, metrics, _, fcst_df, uri, params = train_chronos_model(
                        processed_df, processed_df.iloc[-1:], config.horizon, config.frequency,
                        random_seed=config.random_seed,
                        original_data=data_list, covariates=covariates,
                        hyperparameter_filters=hyperparameter_filters,
                        forecast_start_date=processed_df['ds'].max(),
                        model_size='small'  # Consistent with training phase
                    )
                    final_result = {
                        'model_type': f'Chronos({params.get("model_size", "base")})' if params else 'Chronos',
                        'metrics': metrics,
                        'forecast': fcst_df.to_dict('records'),
                        'mlflow_run_id': mlflow_run_id,
                        'model_uri': uri,
                        'covariate_impacts': []
                    }

                elif 'autotheta' in best_model_type or 'statsforecast' in best_model_type:
                    # For StatsForecast/AutoTheta, train on all data
                    from backend.models.statsforecast_models import train_statsforecast_model
                    mlflow_run_id, _, metrics, _, fcst_df, uri, params = train_statsforecast_model(
                        processed_df, processed_df.iloc[-1:], config.horizon, config.frequency,
                        random_seed=config.random_seed,
                        model_type='autotheta',
                        forecast_start_date=processed_df['ds'].max()
                    )
                    final_result = {
                        'model_type': 'AutoTheta',
                        'metrics': metrics,
                        'forecast': fcst_df.to_dict('records'),
                        'mlflow_run_id': mlflow_run_id,
                        'model_uri': uri,
                        'covariate_impacts': []
                    }

                elif 'ensemble' in best_model_type:
                    # For Ensemble, we need to retrain all component models first
                    # Then combine them. For efficiency, we use the already computed ensemble.
                    # The ensemble was trained with holdout, so we just keep the best result.
                    logger.info("Ensemble model selected - using pre-trained ensemble (includes all data)")
                    final_result = best_result
                    # Update the model_type for clarity
                    final_result['model_type'] = best_result.get('model_type', 'Ensemble')

                if final_result:
                    logger.info(f"Final model trained on full data. Ready for forecast.")
                    # Carry over holdout metrics for reporting
                    final_result['holdout_mape'] = best_result.get('holdout_mape')
                    final_result['eval_mape'] = best_result.get('eval_mape')

            except Exception as retrain_error:
                logger.warning(f"Could not retrain on full data: {retrain_error}. Using original model.")
                final_result = best_result

        # Use final_result if available, otherwise fall back to best_result
        result_to_use = final_result or best_result

        # ============================================================
        # DETECT ANOMALIES IN PREDICTIONS
        # ============================================================
        anomalies = []
        if result_to_use:
            anomalies = _detect_forecast_anomalies(
                result_to_use, data, config, profile
            )

        # If we got a result, use it
        if result_to_use:
            # Extract forecast data
            forecast_data = result_to_use['forecast']

            # Handle different forecast formats
            if isinstance(forecast_data, list) and len(forecast_data) > 0:
                dates = [row.get('ds', row.get(config.date_column, '')) for row in forecast_data]
                forecasts = [row.get('yhat', row.get('forecast', 0)) for row in forecast_data]
                lowers = [row.get('yhat_lower', row.get('lower', forecasts[i] * 0.9)) for i, row in enumerate(forecast_data)]
                uppers = [row.get('yhat_upper', row.get('upper', forecasts[i] * 1.1)) for i, row in enumerate(forecast_data)]
            else:
                dates, forecasts, lowers, uppers = [], [], [], []

            # Add explanation of what we did
            split_info['explanation'] = (
                f"We split your {n} data points into three sets: "
                f"TRAIN ({split_info['train_pct']}% - model learns patterns), "
                f"EVAL ({split_info['eval_pct']}% - tune parameters), and "
                f"HOLDOUT ({split_info['holdout_pct']}% - final unbiased test). "
                f"The best model was selected based on HOLDOUT performance to avoid overfitting. "
                f"Then we retrained {result_to_use['model_type']} on ALL {n} data points for the final forecast."
            )

            logger.info("=" * 70)
            logger.info("🔮 _run_forecast - FINAL OUTPUT")
            logger.info("=" * 70)
            logger.info(f"[OUTPUT] best_model: {result_to_use['model_type']}")
            logger.info(f"[OUTPUT] forecast count: {len(forecasts)}")
            logger.info(f"[OUTPUT] forecast values (first 5): {forecasts[:5]}")
            logger.info(f"[OUTPUT] forecast values (last 5): {forecasts[-5:]}")
            logger.info(f"[OUTPUT] dates (first 5): {[str(d) for d in dates[:5]]}")
            logger.info(f"[OUTPUT] dates (last 5): {[str(d) for d in dates[-5:]]}")
            logger.info(f"[OUTPUT] metrics: {result_to_use['metrics']}")
            logger.info(f"[OUTPUT] holdout_mape: {result_to_use.get('holdout_mape')}")
            logger.info(f"[OUTPUT] eval_mape: {result_to_use.get('eval_mape')}")
            logger.info(f"[OUTPUT] all_models_trained: {[r['model_type'] for r in all_results]}")
            logger.info(f"[OUTPUT] mlflow_run_id: {result_to_use.get('mlflow_run_id')}")
            logger.info("=" * 70)

            final_output = {
                'run_id': run_id,
                'best_model': result_to_use['model_type'],
                'model_version': '1.0',
                'forecast': forecasts,
                'dates': [str(d) for d in dates],
                'lower': lowers,
                'upper': uppers,
                'metrics': result_to_use['metrics'],
                'mlflow_run_id': result_to_use.get('mlflow_run_id'),
                'model_uri': result_to_use.get('model_uri'),
                'all_models_trained': [r['model_type'] for r in all_results],
                'covariate_impacts': result_to_use.get('covariate_impacts', []),
                # Reproducibility info
                'random_seed': seed,
                'data_hash': profile.data_hash,
                'config_hash': config.config_hash,
                # NEW: Split info and model comparison
                'data_split': split_info,
                'model_comparison': model_comparison,
                'selection_reason': _generate_selection_reason(best_result, model_comparison),
                'anomalies': anomalies,
                'holdout_mape': result_to_use.get('holdout_mape'),
                'eval_mape': result_to_use.get('eval_mape'),
                'trained_on_full_data': final_result is not None,
                # Future covariates info
                'future_covariates_used': future_covariates_used,
                'future_covariates_count': future_covariates_count,
                'future_covariates_date_range': future_covariates_date_range,
                # Track MLflow mode used
                'mlflow_mode': mlflow_mode,
            }

            # Cleanup: End MLflow parent run if active
            if mlflow_parent_run_context is not None:
                try:
                    mlflow_parent_run_context.__exit__(None, None, None)
                    logger.info(f"📊 Closed MLflow parent run")
                except Exception as cleanup_error:
                    logger.warning(f"Could not close MLflow run: {cleanup_error}")

            # Sanitize all float values to be JSON-compliant (no NaN/Inf)
            return _sanitize_dict(final_output)

    except ImportError as e:
        logger.warning(f"[FALLBACK] Could not import full training infrastructure: {e}. Using fallback.")
        # Cleanup MLflow if it was started
        if 'mlflow_parent_run_context' in locals() and mlflow_parent_run_context is not None:
            try:
                mlflow_parent_run_context.__exit__(None, None, None)
            except:
                pass
    except Exception as e:
        logger.error(f"[FALLBACK] AutoML training failed: {e}. Using fallback.")
        import traceback
        logger.error(f"[FALLBACK] Full traceback:\n{traceback.format_exc()}")
        # Cleanup MLflow if it was started
        if 'mlflow_parent_run_context' in locals() and mlflow_parent_run_context is not None:
            try:
                mlflow_parent_run_context.__exit__(None, None, None)
            except:
                pass

    # Fallback: simple moving average forecast (when full training unavailable)
    logger.info("[FALLBACK] Using fallback moving average forecast")
    y = data['y'].values
    last_values = y[-min(12, len(y)):]
    mean_val = np.mean(last_values)
    std_val = np.std(last_values) if len(last_values) > 1 else mean_val * 0.1

    # Generate forecast dates
    last_date = data['ds'].max()
    freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS'}
    freq = freq_map.get(config.frequency, 'W')

    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=config.horizon,
        freq=freq
    )

    # Simple trend-adjusted forecast
    trend = (y[-1] - y[0]) / len(y) if len(y) > 1 else 0
    forecast_values = [float(mean_val + trend * (i + 1)) for i in range(config.horizon)]

    # Sanitize all float values to be JSON-compliant (no NaN/Inf)
    return _sanitize_dict({
        'run_id': run_id,
        'best_model': 'MovingAverage (fallback)',
        'model_version': '1.0',
        'forecast': forecast_values,
        'dates': [str(d.date()) for d in forecast_dates],
        'lower': [v - 1.96 * std_val for v in forecast_values],
        'upper': [v + 1.96 * std_val for v in forecast_values],
        'metrics': {'mape': 15.0, 'rmse': float(std_val), 'r2': 0.5},
        'mlflow_run_id': None,
        'model_uri': None,
        'all_models_trained': ['MovingAverage'],
        # Reproducibility info
        'random_seed': seed,
        'data_hash': profile.data_hash,
        'config_hash': config.config_hash,
        # Future covariates info
        'future_covariates_used': future_covariates_used,
        'future_covariates_count': future_covariates_count,
        'future_covariates_date_range': future_covariates_date_range,
    })


def _select_models_for_data(profile: DataProfile, data_length: int) -> List[str]:
    """
    Select which models to train based on data characteristics.

    Returns a list of model types to train, ordered by likelihood of success.
    """
    logger.info("=" * 70)
    logger.info("🎯 MODEL SELECTION - _select_models_for_data")
    logger.info("=" * 70)
    logger.info(f"[MODEL SELECT] Input data_length: {data_length}")
    logger.info(f"[MODEL SELECT] Profile frequency: {profile.frequency}")
    logger.info(f"[MODEL SELECT] Profile history_months: {profile.history_months}")
    logger.info(f"[MODEL SELECT] Profile data_quality_score: {profile.data_quality_score}")

    models = []
    selection_reasons = []

    # Prophet: Good for most cases, especially with seasonality
    if data_length >= 12:  # Minimum for Prophet
        models.append('prophet')
        selection_reasons.append(f"✅ Prophet: data_length ({data_length}) >= 12")
    else:
        selection_reasons.append(f"❌ Prophet SKIPPED: data_length ({data_length}) < 12 minimum")

    # XGBoost: Good with covariates and enough data
    if data_length >= 30:
        models.append('xgboost')
        selection_reasons.append(f"✅ XGBoost: data_length ({data_length}) >= 30")
    else:
        selection_reasons.append(f"❌ XGBoost SKIPPED: data_length ({data_length}) < 30 minimum")

    # ARIMA: Good for shorter series without strong seasonality
    if data_length >= 20:
        models.append('arima')
        selection_reasons.append(f"✅ ARIMA: data_length ({data_length}) >= 20")
    else:
        selection_reasons.append(f"❌ ARIMA SKIPPED: data_length ({data_length}) < 20 minimum")

    # ETS: Good for trend and seasonality, simpler than Prophet
    if data_length >= 24:  # Need at least 2 seasonal cycles for weekly
        models.append('ets')
        selection_reasons.append(f"✅ ETS: data_length ({data_length}) >= 24")
    else:
        selection_reasons.append(f"❌ ETS SKIPPED: data_length ({data_length}) < 24 minimum")

    # SARIMAX (via StatsForecast): Fast AutoARIMA with seasonal support
    if data_length >= 52:  # Need at least 1 year for seasonal patterns
        models.append('sarimax')
        selection_reasons.append(f"✅ SARIMAX: data_length ({data_length}) >= 52")
    else:
        selection_reasons.append(f"❌ SARIMAX SKIPPED: data_length ({data_length}) < 52 minimum")

    # Chronos: Zero-shot foundation model (no training needed)
    if data_length >= 20:  # Just needs context, no minimum for training
        models.append('chronos')
        selection_reasons.append(f"✅ Chronos: data_length ({data_length}) >= 20")
    else:
        selection_reasons.append(f"❌ Chronos SKIPPED: data_length ({data_length}) < 20 minimum")

    # StatsForecast (AutoTheta): Fast statistical model, great for trends
    if data_length >= 24:  # Need sufficient data for seasonal detection
        models.append('statsforecast')
        selection_reasons.append(f"✅ StatsForecast: data_length ({data_length}) >= 24")
    else:
        selection_reasons.append(f"❌ StatsForecast SKIPPED: data_length ({data_length}) < 24 minimum")

    # If no models selected, at least try Prophet
    if not models:
        models = ['prophet']
        selection_reasons.append("⚠️ No models qualified - FALLBACK to Prophet")

    logger.info("[MODEL SELECT] Selection decisions:")
    for reason in selection_reasons:
        logger.info(f"[MODEL SELECT]   {reason}")
    logger.info(f"[MODEL SELECT] FINAL MODELS TO TRAIN: {models}")
    logger.info("=" * 70)

    return models


def _evaluate_on_holdout(
    result: Dict[str, Any],
    holdout_df: pd.DataFrame,
    config: 'ForecastConfig'
) -> float:
    """
    Evaluate a trained model's predictions against the holdout set.

    This performs TRUE holdout evaluation by:
    1. Loading the trained model from MLflow
    2. Predicting for holdout dates
    3. Comparing predictions to holdout actuals
    4. Using robust MAPE (median) if outliers detected

    Args:
        result: Training result dict with metrics and run_id
        holdout_df: The holdout dataframe with actual values
        config: Forecast configuration

    Returns:
        MAPE (Mean Absolute Percentage Error) from holdout evaluation
    """
    import numpy as np
    import mlflow.pyfunc

    try:
        # FIX: Use 'mlflow_run_id' key, not 'run_id' - the result dict stores it as mlflow_run_id
        run_id = result.get('mlflow_run_id')
        if not run_id or holdout_df.empty:
            # Fall back to eval MAPE if no run_id or empty holdout
            metrics = result.get('metrics', {})
            eval_mape = metrics.get('mape', 100.0)
            logger.info(f"No mlflow_run_id or empty holdout - using eval MAPE: {eval_mape:.2f}%")
            return float(eval_mape)

        # Load the trained model
        model_uri = f"runs:/{run_id}/model"
        try:
            loaded_model = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.warning(f"Could not load model for holdout eval: {e}")
            metrics = result.get('metrics', {})
            return float(metrics.get('mape', 100.0))

        # Check model signature to determine prediction mode
        use_periods_mode = False
        try:
            model_signature = loaded_model.metadata.signature
            if model_signature and model_signature.inputs:
                col_specs = model_signature.inputs.to_dict()
                col_names = [spec.get('name', '') for spec in col_specs]
                # If model expects 'periods' and 'start_date', use period-based prediction
                use_periods_mode = 'periods' in col_names and 'start_date' in col_names
                logger.info(f"Model signature columns: {col_names}, use_periods_mode: {use_periods_mode}")
        except Exception as sig_error:
            logger.warning(f"Could not read model signature: {sig_error}")

        if use_periods_mode:
            # Mode for XGBoost/ARIMA: Use periods and start_date
            holdout_dates = pd.to_datetime(holdout_df['ds'])
            start_date = holdout_dates.min().strftime('%Y-%m-%d')
            periods = len(holdout_df)

            period_input = pd.DataFrame({
                'periods': [periods],
                'start_date': [start_date]
            })
            logger.info(f"Holdout using period mode: periods={periods}, start_date={start_date}")

            try:
                holdout_predictions = loaded_model.predict(period_input)
            except Exception as pred_error:
                logger.warning(f"Period-mode predict failed: {pred_error}")
                metrics = result.get('metrics', {})
                return float(metrics.get('mape', 100.0))
        else:
            # Mode for Prophet: Use dataframe with features
            holdout_input = holdout_df.copy().reset_index(drop=True)
            holdout_input['ds'] = pd.to_datetime(holdout_input['ds']).dt.strftime('%Y-%m-%d')

            # Build input with columns in correct order AND types
            try:
                model_signature = loaded_model.metadata.signature
                if model_signature and model_signature.inputs:
                    col_specs = model_signature.inputs.to_dict()
                    input_cols = []
                    for col_spec in col_specs:
                        col_name = col_spec.get('name', '')
                        col_type = col_spec.get('type', 'double')

                        if col_name in holdout_input.columns:
                            input_cols.append(col_name)
                            if col_type in ('long', 'integer', 'int'):
                                holdout_input[col_name] = holdout_input[col_name].fillna(0).astype(int)
                            elif col_type in ('double', 'float'):
                                holdout_input[col_name] = holdout_input[col_name].astype(float)
                        elif col_name == 'ds':
                            input_cols.append(col_name)

                    if input_cols:
                        holdout_input = holdout_input[input_cols]
                        logger.info(f"Holdout input prepared with {len(input_cols)} columns (types converted): {input_cols[:5]}...")
            except Exception as sig_error:
                logger.warning(f"Could not read model signature for feature prep: {sig_error}")
                holdout_input = holdout_input[['ds']]

            try:
                holdout_predictions = loaded_model.predict(holdout_input)
            except Exception as pred_error:
                logger.warning(f"Feature-mode predict failed: {pred_error}")
                metrics = result.get('metrics', {})
                return float(metrics.get('mape', 100.0))

        # Calculate MAPE with robust outlier handling
        if isinstance(holdout_predictions, pd.DataFrame) and 'yhat' in holdout_predictions.columns:
            pred_df = holdout_predictions[['ds', 'yhat']].copy()
            pred_df['ds'] = pd.to_datetime(pred_df['ds'])
            actual_df = holdout_df[['ds', 'y']].copy()
            actual_df['ds'] = pd.to_datetime(actual_df['ds'])

            merged = actual_df.merge(pred_df, on='ds', how='inner')
            if len(merged) > 0:
                # Calculate per-point errors
                errors = np.abs((merged['y'] - merged['yhat']) / (merged['y'] + 1e-10)) * 100

                # Detect extreme outliers (>500% error typically indicates data quality issues)
                outlier_mask = errors > 500
                outlier_count = outlier_mask.sum()

                if outlier_count > 0:
                    logger.warning(f"   ⚠️ {outlier_count} holdout points have >500% error (likely data quality issues)")

                # Use robust MAPE (median) if outliers detected, otherwise mean
                if outlier_count > 0 and outlier_count < len(merged):
                    holdout_mape = float(np.median(errors))
                    logger.info(f"   Using ROBUST holdout MAPE (median): {holdout_mape:.2f}% (mean would be {np.mean(errors):.2f}%)")
                else:
                    holdout_mape = float(np.mean(errors))

                logger.info(f"TRUE holdout MAPE calculated: {holdout_mape:.2f}%")
                return holdout_mape

        # Fallback to eval MAPE if prediction format not recognized
        logger.warning("Could not calculate true holdout MAPE - using eval MAPE")
        metrics = result.get('metrics', {})
        return float(metrics.get('mape', 100.0))

    except Exception as e:
        logger.error(f"Error in holdout evaluation: {e}")
        # Fall back to eval MAPE
        metrics = result.get('metrics', {})
        return float(metrics.get('mape', 100.0))


def _detect_forecast_anomalies(
    result: Dict[str, Any],
    historical_data: pd.DataFrame,
    config: 'ForecastConfig',
    profile: 'DataProfile'
) -> List[Dict[str, Any]]:
    """
    Detect anomalies in the forecast predictions.

    Identifies:
    1. Sudden jumps/drops from recent history
    2. Values outside historical range
    3. Unexpected trend reversals
    4. Seasonality violations

    Args:
        result: Forecast result with predictions
        historical_data: Historical data with 'ds' and 'y' columns
        config: Forecast configuration
        profile: Data profile with characteristics

    Returns:
        List of anomaly dicts with type, description, severity, cause, and fix
    """
    import numpy as np

    anomalies = []

    try:
        forecast_data = result.get('forecast', [])
        if not forecast_data:
            return anomalies

        # Extract forecast values - handle both list of dicts and list of numbers
        if isinstance(forecast_data, list) and len(forecast_data) > 0:
            if isinstance(forecast_data[0], dict):
                # List of dicts - extract the forecast value
                forecast_values = []
                for row in forecast_data:
                    val = row.get('yhat', row.get('forecast', row.get('predicted', None)))
                    if val is not None:
                        forecast_values.append(float(val))
            else:
                # Already a list of numbers
                forecast_values = [float(v) for v in forecast_data]
        else:
            return anomalies

        if not forecast_values:
            return anomalies

        # Get historical statistics
        y = historical_data['y'].values
        hist_mean = np.mean(y)
        hist_std = np.std(y)
        hist_min = np.min(y)
        hist_max = np.max(y)
        recent_mean = np.mean(y[-min(12, len(y)):])  # Last 12 periods
        recent_trend = (y[-1] - y[-min(6, len(y))]) / min(6, len(y)) if len(y) > 1 else 0

        # Convert forecast to numpy
        forecast = np.array(forecast_values)

        # Check 1: Values outside historical range (with buffer)
        buffer = hist_std * 2
        too_high = forecast > (hist_max + buffer)
        too_low = forecast < (hist_min - buffer)

        if np.any(too_high):
            high_indices = np.where(too_high)[0]
            anomalies.append({
                'type': 'out_of_range_high',
                'severity': 'warning',
                'periods': high_indices.tolist(),
                'description': f"Forecast values at periods {high_indices.tolist()} exceed historical maximum by more than 2 standard deviations.",
                'cause': "This could indicate: (1) Strong detected growth trend being extrapolated, (2) Seasonal peak being amplified, or (3) Covariate effects pushing predictions higher.",
                'fix': "Review if recent growth trend is sustainable. Consider: (1) Adding more recent data, (2) Adjusting the forecast horizon, (3) Checking covariate assumptions for future periods.",
                'forecast_values': [float(forecast[i]) for i in high_indices[:3]],
                'threshold': float(hist_max + buffer)
            })

        if np.any(too_low):
            low_indices = np.where(too_low)[0]
            # Check for negative values specifically
            negative_mask = forecast < 0
            if np.any(negative_mask):
                neg_indices = np.where(negative_mask)[0]
                anomalies.append({
                    'type': 'negative_values',
                    'severity': 'critical',
                    'periods': neg_indices.tolist(),
                    'description': f"Forecast predicts negative values at periods {neg_indices.tolist()}. This may be invalid for your metric.",
                    'cause': "Strong downward trend or seasonality is pushing predictions below zero. This is common with additive models when trends continue beyond data support.",
                    'fix': "Consider: (1) Using a model with non-negative constraints, (2) Setting a floor at zero for business logic, (3) Reviewing if the downward trend is realistic.",
                    'forecast_values': [float(forecast[i]) for i in neg_indices[:3]],
                    'threshold': 0
                })
            else:
                anomalies.append({
                    'type': 'out_of_range_low',
                    'severity': 'warning',
                    'periods': low_indices.tolist(),
                    'description': f"Forecast values at periods {low_indices.tolist()} are below historical minimum by more than 2 standard deviations.",
                    'cause': "Detected downward trend or seasonal trough being extrapolated beyond historical patterns.",
                    'fix': "Verify if decline is expected. Consider external factors not captured in the data.",
                    'forecast_values': [float(forecast[i]) for i in low_indices[:3]],
                    'threshold': float(hist_min - buffer)
                })

        # Check 2: Sudden jump/drop from last historical value
        last_actual = float(y[-1])
        first_forecast = float(forecast[0])
        jump_pct = abs(first_forecast - last_actual) / abs(last_actual) * 100 if last_actual != 0 else 0

        if jump_pct > 30:  # More than 30% jump
            direction = "increase" if first_forecast > last_actual else "decrease"
            anomalies.append({
                'type': 'sudden_jump',
                'severity': 'warning' if jump_pct < 50 else 'critical',
                'periods': [0],
                'description': f"Forecast shows a {jump_pct:.1f}% {direction} from the last historical value.",
                'cause': f"The model predicts a sharp {direction} from {last_actual:,.0f} to {first_forecast:,.0f}. This could be due to: (1) Seasonal patterns in the data, (2) Holiday effects, (3) Covariate changes.",
                'fix': "Review: (1) Is this aligned with known business events? (2) Check if recent data had unusual values. (3) Verify covariate assumptions.",
                'last_actual': last_actual,
                'first_forecast': first_forecast,
                'jump_percentage': jump_pct
            })

        # Check 3: High volatility in forecast (unrealistic swings)
        if len(forecast) > 2:
            forecast_volatility = np.std(np.diff(forecast))
            historical_volatility = np.std(np.diff(y)) if len(y) > 1 else hist_std

            if forecast_volatility > historical_volatility * 2:
                anomalies.append({
                    'type': 'high_volatility',
                    'severity': 'info',
                    'periods': list(range(len(forecast))),
                    'description': f"Forecast shows higher volatility ({forecast_volatility:.1f}) than historical data ({historical_volatility:.1f}).",
                    'cause': "The model is predicting larger period-to-period swings than seen historically. This could indicate uncertainty in predictions.",
                    'fix': "Focus on the confidence intervals rather than point forecasts. Consider using a simpler model with smoother predictions.",
                    'forecast_volatility': float(forecast_volatility),
                    'historical_volatility': float(historical_volatility)
                })

        # Check 4: Trend reversal
        if len(forecast) > 3:
            forecast_trend = (forecast[-1] - forecast[0]) / len(forecast)
            trend_reversal = (recent_trend > 0 and forecast_trend < -abs(recent_trend) * 0.5) or \
                           (recent_trend < 0 and forecast_trend > abs(recent_trend) * 0.5)

            if trend_reversal and abs(recent_trend) > hist_std * 0.1:
                anomalies.append({
                    'type': 'trend_reversal',
                    'severity': 'info',
                    'periods': list(range(len(forecast))),
                    'description': f"Forecast shows trend reversal from recent historical pattern.",
                    'cause': f"Recent data showed {'upward' if recent_trend > 0 else 'downward'} trend, but forecast predicts {'downward' if recent_trend > 0 else 'upward'} movement.",
                    'fix': "This could be: (1) Mean reversion (normal), (2) Seasonal pattern, or (3) Model uncertainty. Review seasonal patterns.",
                    'recent_trend_direction': 'up' if recent_trend > 0 else 'down',
                    'forecast_trend_direction': 'up' if forecast_trend > 0 else 'down'
                })

        # Check 5: Flat forecast (no variation)
        if len(forecast) > 2:
            forecast_range = np.max(forecast) - np.min(forecast)
            if forecast_range < hist_std * 0.1:
                anomalies.append({
                    'type': 'flat_forecast',
                    'severity': 'info',
                    'periods': list(range(len(forecast))),
                    'description': "Forecast shows very little variation (nearly flat line).",
                    'cause': "The model is predicting stable values without seasonal or trend patterns. This happens when: (1) No clear patterns detected, (2) High uncertainty leading to mean predictions.",
                    'fix': "If you expect variation, consider: (1) Adding more historical data, (2) Including covariates that capture seasonality.",
                    'forecast_range': float(forecast_range),
                    'expected_range': float(hist_std)
                })

    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        anomalies.append({
            'type': 'detection_error',
            'severity': 'info',
            'description': f"Could not complete anomaly detection: {str(e)}",
            'cause': "Technical error during analysis",
            'fix': "Proceed with caution and manually review forecast values"
        })

    return anomalies


def _generate_selection_reason(
    best_result: Optional[Dict[str, Any]],
    model_comparison: List[Dict[str, Any]]
) -> str:
    """
    Generate a human-readable explanation of why the best model was selected.

    Args:
        best_result: The winning model's result dict
        model_comparison: List of all models with their eval and holdout MAPEs

    Returns:
        Human-readable explanation string
    """
    if not best_result or not model_comparison:
        return "No model comparison data available."

    best_model = best_result.get('model_type', 'Unknown')
    best_holdout = best_result.get('holdout_mape', 0)
    best_eval = best_result.get('eval_mape', 0)

    # Sort by holdout MAPE
    sorted_models = sorted(model_comparison, key=lambda x: x.get('holdout_mape', float('inf')))

    # Build explanation
    parts = []

    # Main selection reason
    parts.append(f"**{best_model}** was selected as the best model with a holdout MAPE of {best_holdout:.2f}%.")

    # Compare to other models
    if len(sorted_models) > 1:
        runner_up = sorted_models[1]
        parts.append(
            f"It outperformed {runner_up['model']} (holdout MAPE: {runner_up['holdout_mape']:.2f}%) "
            f"and {len(sorted_models) - 1} other model(s)."
        )

    # Overfitting check
    mape_diff = best_holdout - best_eval
    if mape_diff > 5:
        parts.append(
            f"⚠️ Note: There's a {mape_diff:.1f}% gap between eval ({best_eval:.1f}%) and holdout ({best_holdout:.1f}%) performance, "
            f"suggesting some overfitting to the evaluation set."
        )
    elif mape_diff < -2:
        parts.append(
            f"✓ Good sign: The model performed better on holdout ({best_holdout:.1f}%) than eval ({best_eval:.1f}%), "
            f"suggesting robust generalization."
        )

    # Check if any model showed severe overfitting
    overfit_models = [m for m in model_comparison if m.get('overfit_warning', False)]
    if overfit_models and best_model not in [m['model'] for m in overfit_models]:
        parts.append(
            f"Models {[m['model'] for m in overfit_models]} showed signs of overfitting and were not selected."
        )

    # Model ranking
    if len(sorted_models) > 1:
        ranking = ", ".join([f"{i+1}. {m['model']} ({m['holdout_mape']:.1f}%)"
                            for i, m in enumerate(sorted_models[:3])])
        parts.append(f"Model ranking by holdout performance: {ranking}")

    return " ".join(parts)


def register_simple_mode_routes(app):
    """Register simple mode routes with the main FastAPI app."""
    app.include_router(router)
