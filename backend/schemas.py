"""
Pydantic models for request/response validation
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator


import os

# Minimum data points required for reliable forecasting
# 52 weeks = 1 year of weekly data, which is minimum for seasonal pattern detection
MIN_DATA_POINTS = 52

class DataRow(BaseModel):
    """Represents a single row of data"""
    data: Dict[str, Any]
    
    class Config:
        extra = "allow"


class AggregationConfig(BaseModel):
    """Configuration for data aggregation before forecasting"""
    enabled: bool = Field(default=False, description="Whether to aggregate multi-dimensional data")
    group_by_cols: Optional[List[str]] = Field(default=None, description="Columns to group by. None = aggregate to total.")
    agg_method: Literal['sum', 'mean', 'median', 'max', 'min'] = Field(default='sum', description="Aggregation method for target column")
    filter_dimensions: Optional[Dict[str, Any]] = Field(default=None, description="Dimension filters, e.g., {'region': 'West', 'segment': ['A', 'B']}")
    auto_detect_incomplete: bool = Field(default=True, description="Automatically detect and exclude incomplete trailing data")

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "group_by_cols": None,
                "agg_method": "sum",
                "filter_dimensions": {"BUSINESS_SEGMENT": "Classic"},
                "auto_detect_incomplete": True
            }
        }


class TrainRequest(BaseModel):
    """Request model for training endpoint"""
    data: List[Dict[str, Any]] = Field(..., description="Time series data rows (minimum 52 points required)")
    time_col: str = Field(..., description="Name of the time/date column")
    target_col: str = Field(..., description="Name of the target column to forecast")
    covariates: List[str] = Field(default=[], description="List of covariate column names")
    horizon: int = Field(default=12, ge=1, le=104, description="Number of periods to forecast (1-104)")
    frequency: Literal['daily', 'weekly', 'monthly'] = Field(default="monthly", description="Data frequency: 'daily', 'weekly', or 'monthly'")
    seasonality_mode: Literal['additive', 'multiplicative'] = Field(default="multiplicative", description="Seasonality mode: 'additive' or 'multiplicative'")
    # Data aggregation options for multi-dimensional data
    aggregation: Optional[AggregationConfig] = Field(default=None, description="Aggregation config for multi-dimensional data. If provided, data will be aggregated before forecasting.")

    @field_validator('data')
    @classmethod
    def validate_minimum_data_points(cls, v):
        """Ensure minimum data points for reliable forecasting."""
        if len(v) < MIN_DATA_POINTS:
            raise ValueError(
                f"Minimum {MIN_DATA_POINTS} data points required for reliable forecasting. "
                f"Received only {len(v)} rows. For seasonal pattern detection, at least "
                f"1 year of data (52 weekly points, 12 monthly points, or 365 daily points) is recommended."
            )
        return v
    test_size: Optional[int] = Field(default=None, description="Size of test set for validation")
    regressor_method: str = Field(default="mean", description="How to fill future covariates: 'mean', 'last_value', 'linear_trend'")
    models: List[str] = Field(default=["prophet"], description="Models to train. Recommended: 'prophet', 'exponential_smoothing', 'statsforecast', 'chronos'. Available but not recommended: 'arima', 'sarimax', 'xgboost' (see /api/analyze-data for reasons).")
    catalog_name: str = Field(default=os.getenv("UC_CATALOG_NAME", "main"), description="Unity Catalog catalog name")
    schema_name: str = Field(default=os.getenv("UC_SCHEMA_NAME", "default"), description="Unity Catalog schema name")
    model_name: str = Field(default=os.getenv("UC_MODEL_NAME_ONLY", "finance_forecast_model"), description="Model name in Unity Catalog")
    country: str = Field(default="US", description="Country code for holiday calendar (US, UK, CA, etc.)")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata about data filters applied (e.g., {'store': 'Store A', 'region': 'West'})")
    from_date: Optional[str] = Field(default=None, description="Start date for filtering training data (YYYY-MM-DD format). If provided, only data from this date onwards will be used.")
    to_date: Optional[str] = Field(default=None, description="End date for filtering training data (YYYY-MM-DD format). If provided, only data up to this date will be used.")
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducibility. Set to ensure consistent results across runs.")
    confidence_level: Optional[float] = Field(default=0.95, ge=0.50, le=0.99, description="Confidence level for prediction intervals (0.50-0.99). Lower values = narrower intervals. Default 0.95 (95%). Use 0.80 for ~35% narrower intervals.")
    future_features: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional future feature data for prediction horizon. If provided, actual feature values will be used instead of imputation.")
    # Batch training context - enables grouped MLflow tracking
    batch_id: Optional[str] = Field(default=None, description="Unique batch training session ID. When provided, all segments use the same MLflow experiment.")
    batch_segment_id: Optional[str] = Field(default=None, description="Human-readable segment identifier (e.g., 'region=US | product=Widget')")
    batch_segment_index: Optional[int] = Field(default=None, description="Segment index within the batch (1-based)")
    batch_total_segments: Optional[int] = Field(default=None, description="Total number of segments in the batch")
    # Hyperparameter guidance from data analysis - reduces search space
    hyperparameter_filters: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Data-driven hyperparameter filters per model. Keys are model names (Prophet, ARIMA, ETS, XGBoost), values are dicts of param_name -> allowed_values."
    )
    # Log transform for high-growth series
    log_transform: Optional[str] = Field(
        default="auto",
        description="Log transform strategy for the target column. "
                    "'auto' = detect high-growth series (>100% growth) and apply automatically, "
                    "'always' = always apply log1p transform, "
                    "'never' = never apply. Log transform linearizes exponential growth, "
                    "improving accuracy on fast-growing segments."
    )
    # Smart auto-optimization
    auto_optimize: bool = Field(
        default=True,
        description="When true, automatically analyze data characteristics and optimize "
                    "training settings before model fitting. Uses spectral entropy, "
                    "STL decomposition, and growth analysis to select optimal training "
                    "window, models, log transform, and horizon. Set to false to use "
                    "your explicit settings without modification."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"ds": "2023-01-01", "y": 1000, "marketing_spend": 500},
                    {"ds": "2023-02-01", "y": 1100, "marketing_spend": 550}
                ],
                "time_col": "ds",
                "target_col": "y",
                "covariates": ["marketing_spend"],
                "horizon": 12,
                "frequency": "monthly",
                "seasonality_mode": "multiplicative"
            }
        }


class DimensionInfo(BaseModel):
    """Information about a detected dimension column"""
    n_unique: int = Field(..., description="Number of unique values")
    type: str = Field(..., description="Dimension type: categorical, flag/code, numeric_categorical")
    sample_values: List[Any] = Field(..., description="Sample of unique values")
    null_count: int = Field(default=0, description="Number of null values")


class DetectDimensionsRequest(BaseModel):
    """Request model for dimension detection"""
    data: List[Dict[str, Any]] = Field(..., description="Data rows to analyze")
    time_col: str = Field(..., description="Name of the date/time column")
    target_col: str = Field(..., description="Name of the target column")


class DetectDimensionsResponse(BaseModel):
    """Response model for dimension detection"""
    dimensions: Dict[str, DimensionInfo] = Field(..., description="Detected dimension columns")
    numeric_measures: List[str] = Field(default=[], description="Numeric columns that could be aggregated")
    date_col: str = Field(..., description="Date column name")
    target_col: str = Field(..., description="Target column name")
    row_count: int = Field(..., description="Total row count")
    unique_dates: int = Field(..., description="Number of unique dates")
    aggregation_recommended: bool = Field(..., description="Whether aggregation is recommended")
    recommendation: str = Field(default="", description="Aggregation recommendation message")


class ForecastMetrics(BaseModel):
    """Model performance metrics"""
    rmse: str = Field(..., description="Root Mean Square Error")
    mape: str = Field(..., description="Mean Absolute Percentage Error")
    r2: str = Field(..., description="R-squared score")
    cv_mape: Optional[str] = Field(None, description="Cross-validation MAPE (more robust estimate)")
    cv_mape_std: Optional[str] = Field(None, description="Cross-validation MAPE standard deviation")


class ForecastDataPoint(BaseModel):
    """Single forecast data point"""
    ds: str = Field(..., description="Date/time value")
    yhat: float = Field(..., description="Predicted value")
    yhat_lower: Optional[float] = Field(None, description="Lower confidence bound")
    yhat_upper: Optional[float] = Field(None, description="Upper confidence bound")


class CovariateImpact(BaseModel):
    """Covariate impact analysis"""
    name: str
    coefficient: float
    impact_score: float
    score: float = Field(..., description="Normalized score 0-100")
    direction: str = Field(..., description="'positive' or 'negative'")


class ModelTestResult(BaseModel):
    """Result from pre-deployment model testing"""
    test_passed: bool = Field(..., description="Whether the model passed inference testing")
    message: str = Field(..., description="Test result message")
    load_time_seconds: Optional[float] = Field(None, description="Time taken to load model")
    inference_time_seconds: Optional[float] = Field(None, description="Time taken for inference")
    error_details: Optional[str] = Field(None, description="Error details if test failed")


class ModelResult(BaseModel):
    """Result for a single model"""
    model_type: str
    model_name: str
    run_id: str
    metrics: ForecastMetrics
    validation: List[Dict[str, Any]]
    forecast: List[Dict[str, Any]]
    covariate_impacts: List[CovariateImpact] = Field(default=[])
    is_best: bool = False
    experiment_url: Optional[str] = Field(None, description="URL to MLflow experiment")
    run_url: Optional[str] = Field(None, description="URL to MLflow run")
    error: Optional[str] = Field(None, description="Error message if model failed")
    # Model version registered in Unity Catalog
    registered_version: Optional[str] = Field(None, description="Version registered in Unity Catalog")
    # Pre-deployment test results - only models with test_result.test_passed=True should be deployed
    test_result: Optional[ModelTestResult] = Field(None, description="Pre-deployment test result. Only deploy models where test_passed=True")


class AutoOptimizeInfo(BaseModel):
    """Information about auto-optimization decisions applied before training."""
    enabled: bool = Field(..., description="Whether auto-optimization was applied")
    forecastability_score: Optional[float] = Field(default=None, description="Forecastability score 0-100")
    grade: Optional[str] = Field(default=None, description="Forecastability grade")
    training_window_weeks: Optional[int] = Field(default=None, description="Training window applied (weeks)")
    from_date_applied: Optional[str] = Field(default=None, description="Training start date applied")
    log_transform: Optional[str] = Field(default=None, description="Log transform setting applied")
    models_selected: Optional[List[str]] = Field(default=None, description="Models selected by advisor")
    models_excluded: Optional[List[str]] = Field(default=None, description="Models excluded by advisor")
    recommended_horizon: Optional[int] = Field(default=None, description="Recommended horizon")
    max_reliable_horizon: Optional[int] = Field(default=None, description="Max reliable horizon")
    expected_mape_range: Optional[List[float]] = Field(default=None, description="Expected MAPE range [low, high]")
    growth_pct: Optional[float] = Field(default=None, description="Detected growth percentage")
    summary: Optional[str] = Field(default=None, description="Human-readable summary of decisions")


class TrainResponse(BaseModel):
    """Response model for training endpoint"""
    models: List[ModelResult] = Field(..., description="Results for each trained model")
    best_model: str = Field(..., description="Best performing model name")
    artifact_uri: str = Field(..., description="MLflow artifact location")
    history: List[Dict[str, Any]] = Field(default=[], description="Historical data (actuals) used for training")
    trace_id: Optional[str] = Field(default=None, description="Pipeline trace ID for debugging")
    auto_optimize_info: Optional[AutoOptimizeInfo] = Field(default=None, description="Auto-optimization decisions applied before training. None if auto_optimize was disabled.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "abc123",
                "model_uri": "runs:/abc123/model",
                "metrics": {
                    "rmse": "125.43",
                    "mape": "8.32",
                    "r2": "0.92"
                },
                "validation": [],
                "forecast": [],
                "artifact_uri": "dbfs:/databricks/mlflow/abc123/artifacts"
            }
        }


class DeployRequest(BaseModel):
    """Request model for model deployment"""
    model_name: str = Field(..., description="Name of the model in Unity Catalog")
    model_version: Optional[str] = Field(None, description="Version of the model to deploy (optional if run_id is provided)")
    run_id: Optional[str] = Field(None, description="MLflow Run ID to register and deploy (optional)")
    endpoint_name: str = Field(..., description="Name of the serving endpoint")
    workload_size: str = Field(default="Small", description="Workload size: Small, Medium, Large")
    scale_to_zero: bool = Field(default=True, description="Enable scale to zero")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "main.finance_forecast_model",
                "model_version": "1",
                "endpoint_name": "finance-forecast-endpoint",
                "workload_size": "Small",
                "scale_to_zero": True
            }
        }


class DeployResponse(BaseModel):
    """Response model for deployment endpoint"""
    endpoint_name: str
    status: str
    message: str
    endpoint_url: Optional[str] = None
    deployed_version: Optional[str] = Field(None, description="The actual model version that was deployed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "endpoint_name": "finance-forecast-endpoint",
                "status": "pending",
                "message": "Endpoint creation initiated",
                "endpoint_url": "https://your-workspace.cloud.databricks.com/serving-endpoints/finance-forecast-endpoint"
            }
        }


class TestModelRequest(BaseModel):
    """Request model for pre-deployment model testing"""
    model_name: str = Field(..., description="Full model name in Unity Catalog (e.g., main.default.finance_forecast_model)")
    model_version: str = Field(..., description="Version of the model to test")
    test_periods: int = Field(default=5, description="Number of periods to forecast for testing")
    start_date: Optional[str] = Field(None, description="Start date for test forecast (YYYY-MM-DD). If not provided, uses model's last training date + 1 period")
    frequency: str = Field(default="daily", description="Forecast frequency: daily, weekly, monthly")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "main.default.finance_forecast_model",
                "model_version": "373",
                "test_periods": 5,
                "start_date": "2025-10-06",
                "frequency": "daily"
            }
        }


class TestModelResponse(BaseModel):
    """Response model for pre-deployment model testing"""
    model_name: str
    model_version: str
    test_passed: bool
    message: str
    load_time_seconds: float = Field(..., description="Time taken to load the model")
    inference_time_seconds: float = Field(..., description="Time taken for inference")
    sample_predictions: List[Dict[str, Any]] = Field(..., description="Sample predictions from test inference")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="Model's expected input schema")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Model's output schema")
    error_details: Optional[str] = Field(None, description="Error details if test failed")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "main.default.finance_forecast_model",
                "model_version": "373",
                "test_passed": True,
                "message": "Model loaded and inference successful",
                "load_time_seconds": 2.34,
                "inference_time_seconds": 0.15,
                "sample_predictions": [
                    {"ds": "2025-10-06", "yhat": 150000000.0, "yhat_lower": 140000000.0, "yhat_upper": 160000000.0},
                    {"ds": "2025-10-07", "yhat": 151000000.0, "yhat_lower": 141000000.0, "yhat_upper": 161000000.0}
                ],
                "input_schema": {"columns": ["periods", "start_date"]},
                "output_schema": {"columns": ["ds", "yhat", "yhat_lower", "yhat_upper"]}
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    message: str = "Backend is operational"
    databricks_connected: bool = False
    mlflow_enabled: bool = False


class BatchTrainRequest(BaseModel):
    """Request model for batch training endpoint"""
    requests: List[TrainRequest] = Field(..., description="List of training requests to process in parallel")
    max_workers: int = Field(default=4, ge=1, le=16, description="Maximum number of parallel workers (1-16)")

    class Config:
        json_schema_extra = {
            "example": {
                "requests": [
                    {
                        "data": [{"ds": "2023-01-01", "y": 1000}],
                        "time_col": "ds",
                        "target_col": "y",
                        "horizon": 12,
                        "filters": {"segment": "US"}
                    },
                    {
                        "data": [{"ds": "2023-01-01", "y": 2000}],
                        "time_col": "ds",
                        "target_col": "y",
                        "horizon": 12,
                        "filters": {"segment": "EU"}
                    }
                ],
                "max_workers": 4
            }
        }


class BatchResultItem(BaseModel):
    """Result for a single batch item"""
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters that identify this segment")
    segment_id: Optional[str] = Field(None, description="Identifier for this segment")
    status: str = Field(..., description="'success' or 'error'")
    result: Optional[TrainResponse] = Field(None, description="Training result if successful")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchTrainResponse(BaseModel):
    """Response model for batch training endpoint"""
    total_requests: int = Field(..., description="Total number of requests processed")
    successful: int = Field(..., description="Number of successful requests")
    failed: int = Field(..., description="Number of failed requests")
    results: List[BatchResultItem] = Field(..., description="Results for each request")


class AggregateRequest(BaseModel):
    """Request model for data aggregation endpoint"""
    data: List[Dict[str, Any]] = Field(..., description="Time series data rows")
    time_col: str = Field(..., description="Name of the time/date column")
    target_col: str = Field(..., description="Name of the target column to aggregate")
    covariates: List[str] = Field(default=[], description="List of covariate column names")
    source_frequency: str = Field(..., description="Source frequency: 'daily', 'weekly', 'monthly'")
    target_frequency: str = Field(..., description="Target frequency: 'weekly', 'monthly'")
    aggregation_method: str = Field(default="sum", description="How to aggregate target: 'sum', 'mean', 'last'")
    covariate_aggregation: Dict[str, str] = Field(
        default={},
        description="How to aggregate each covariate: {'col_name': 'sum'|'mean'|'max'|'last'}. Defaults to 'max' for binary, 'mean' for continuous."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"date": "2023-01-01", "sales": 100, "promo": 1},
                    {"date": "2023-01-02", "sales": 150, "promo": 0}
                ],
                "time_col": "date",
                "target_col": "sales",
                "covariates": ["promo"],
                "source_frequency": "daily",
                "target_frequency": "weekly",
                "aggregation_method": "sum"
            }
        }


class AggregateResponse(BaseModel):
    """Response model for data aggregation endpoint"""
    data: List[Dict[str, Any]] = Field(..., description="Aggregated data rows")
    original_rows: int = Field(..., description="Number of rows before aggregation")
    aggregated_rows: int = Field(..., description="Number of rows after aggregation")
    source_frequency: str
    target_frequency: str
    aggregation_methods: Dict[str, str] = Field(..., description="Aggregation method used for each column")


class BatchSegmentInfo(BaseModel):
    """Information about a segment for batch deployment"""
    segment_id: str = Field(..., description="Unique segment identifier (e.g., 'region=US | product=Widget')")
    filters: Dict[str, Any] = Field(..., description="Filter values that identify this segment")
    model_version: str = Field(..., description="Model version to deploy for this segment")
    run_id: Optional[str] = Field(None, description="MLflow run ID if available")


class BatchDeployRequest(BaseModel):
    """Request model for batch deployment endpoint"""
    segments: List[BatchSegmentInfo] = Field(..., description="List of segments with their model versions")
    endpoint_name: str = Field(..., description="Name of the serving endpoint to create")
    catalog_name: str = Field(default="main", description="Unity Catalog catalog name")
    schema_name: str = Field(default="default", description="Unity Catalog schema name")
    model_name: str = Field(default="finance_forecast_model", description="Base model name in Unity Catalog")
    workload_size: str = Field(default="Small", description="Workload size: Small, Medium, Large")
    scale_to_zero: bool = Field(default=True, description="Enable scale to zero")

    class Config:
        json_schema_extra = {
            "example": {
                "segments": [
                    {"segment_id": "region=US", "filters": {"region": "US"}, "model_version": "1"},
                    {"segment_id": "region=EU", "filters": {"region": "EU"}, "model_version": "2"}
                ],
                "endpoint_name": "batch-forecast-endpoint",
                "catalog_name": "main",
                "schema_name": "default",
                "model_name": "finance_forecast_model"
            }
        }


class BatchDeployResponse(BaseModel):
    """Response model for batch deployment endpoint"""
    status: str = Field(..., description="'success' or 'error'")
    message: str = Field(..., description="Status message")
    endpoint_name: Optional[str] = Field(None, description="Name of the created/updated endpoint")
    endpoint_url: Optional[str] = Field(None, description="URL to the endpoint")
    deployed_segments: Optional[int] = Field(None, description="Number of segments deployed")
    router_model_version: Optional[str] = Field(None, description="Version of the router model")


class DataAnalysisRequest(BaseModel):
    """Request model for data analysis endpoint"""
    data: List[Dict[str, Any]] = Field(..., description="Time series data rows")
    time_col: str = Field(..., description="Name of the time/date column")
    target_col: str = Field(..., description="Name of the target column to forecast")
    frequency: str = Field(default="auto", description="Frequency: 'daily', 'weekly', 'monthly', or 'auto'")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"ds": "2023-01-01", "y": 1000},
                    {"ds": "2023-02-01", "y": 1100}
                ],
                "time_col": "ds",
                "target_col": "y",
                "frequency": "monthly"
            }
        }


class ModelRecommendationResponse(BaseModel):
    """Model recommendation from data analysis"""
    model: str = Field(..., description="Model name")
    recommended: bool = Field(..., description="Whether this model is recommended")
    confidence: float = Field(..., description="Confidence score 0-1")
    reason: str = Field(..., description="Reason for recommendation")


class DataAnalysisResponse(BaseModel):
    """Response model for data analysis endpoint"""
    dataQuality: Dict[str, Any] = Field(..., description="Data quality assessment")
    dataStats: Dict[str, Any] = Field(..., description="Data statistics")
    patterns: Dict[str, Any] = Field(..., description="Detected patterns (trend, seasonality)")
    modelRecommendations: List[ModelRecommendationResponse] = Field(..., description="Model recommendations")
    recommendedModels: List[str] = Field(..., description="List of recommended model names")
    excludedModels: List[str] = Field(..., description="List of excluded model names")
    warnings: List[str] = Field(..., description="Data quality warnings")
    notes: List[str] = Field(..., description="Additional notes")
    overallRecommendation: str = Field(..., description="Overall recommendation summary")
    hyperparameterFilters: Dict[str, Dict[str, Any]] = Field(..., description="Hyperparameter filters per model")


# =============================================================================
# SMART FORECAST ADVISOR SCHEMAS
# =============================================================================

class ForecastAdvisorRequest(BaseModel):
    """Smart forecast advisor — analyzes all slices with research-backed metrics.

    Uses spectral entropy (Goerg 2013), STL-based trend/seasonal strength
    (Hyndman et al. 2015), and FFORMA-inspired feature profiling to score
    forecastability, recommend models, training windows, and aggregations.
    """
    data: List[Dict[str, Any]] = Field(..., description="Full dataset with all dimensions")
    time_col: str = Field(..., description="Date column name")
    target_col: str = Field(..., description="Target column to forecast")
    dimension_cols: List[str] = Field(..., description="Dimension columns to slice by (e.g., ['IS_CGNA', 'BUSINESS_SEGMENT', 'MX_TYPE'])")
    frequency: str = Field(default="weekly", description="Data frequency: 'daily', 'weekly', 'monthly'")
    horizon: int = Field(default=12, ge=1, le=52, description="Forecast horizon in periods")


class DataQualityCheckResponse(BaseModel):
    """Data quality assessment for a single slice (prerequisites, not forecastability)."""
    n_observations: int = Field(..., description="Number of time series observations")
    has_sufficient_history: bool = Field(..., description="Whether there's enough data (>= 104 weeks for weekly)")
    missing_pct: float = Field(..., description="Percentage of missing values")
    gap_count: int = Field(..., description="Number of gaps in the time series")
    anomalous_week_count: int = Field(..., description="Number of detected anomalous weeks")
    anomalous_weeks: List[str] = Field(default=[], description="ISO dates of anomalous weeks")
    volume_level: str = Field(..., description="Volume classification: very_low, low, medium, high, very_high")
    weekly_mean: float = Field(..., description="Mean weekly value")
    warnings: List[str] = Field(default=[], description="Data quality warnings")


class SliceAnalysisResponse(BaseModel):
    """Forecastability analysis for a single data slice."""
    slice_name: str = Field(..., description="Slice identifier (e.g., 'CGNA=1/Pickup/Enterprise')")
    filters: Dict[str, str] = Field(..., description="Dimension filter values for this slice")
    forecastability_score: float = Field(..., ge=0, le=100, description="Forecastability score 0-100 based on spectral entropy + STL features")
    grade: str = Field(..., description="Grade: excellent, good, fair, poor, unforecastable")
    spectral_entropy: float = Field(..., description="Spectral entropy 0-1 (lower = more forecastable)")
    trend_strength: float = Field(..., description="STL trend strength 0-1")
    seasonal_strength: float = Field(..., description="STL seasonal strength 0-1")
    total_growth_pct: float = Field(..., description="Total growth % over training period")
    recent_growth_pct: float = Field(..., description="Recent growth % (last 26 vs prior 26 weeks)")
    data_quality: DataQualityCheckResponse = Field(..., description="Data quality assessment")
    recommended_models: List[str] = Field(..., description="Models recommended for this slice")
    excluded_models: List[str] = Field(..., description="Models excluded for this slice")
    model_exclusion_reasons: Dict[str, str] = Field(default={}, description="Reason each model was excluded")
    recommended_training_window: Optional[int] = Field(default=None, description="Recommended training window in weeks (None = use all)")
    training_window_reason: str = Field(default="", description="Why this training window was recommended")
    expected_mape_range: List[float] = Field(default=[], description="Expected MAPE range [low, high] based on forecastability")


class AggregationRecommendationResponse(BaseModel):
    """Recommendation to merge underperforming slices."""
    from_slices: List[str] = Field(..., description="Slice names to merge")
    to_slice: str = Field(..., description="Suggested combined slice name")
    reason: str = Field(..., description="Why merging helps")
    combined_forecastability_score: float = Field(..., description="Projected forecastability score after merging")
    improvement_pct: float = Field(..., description="How much the score improves vs individual slices")


class ForecastAdvisorResponse(BaseModel):
    """Response from the smart forecast advisor."""
    slice_analyses: List[SliceAnalysisResponse] = Field(..., description="Per-slice forecastability analysis")
    aggregation_recommendations: List[AggregationRecommendationResponse] = Field(default=[], description="Recommendations to merge underperforming slices")
    summary: str = Field(..., description="Human-readable summary of all findings")
    overall_data_quality: str = Field(..., description="Overall data quality: excellent, good, fair, poor")
    total_slices: int = Field(..., description="Total number of slices analyzed")
    forecastable_slices: int = Field(..., description="Number of slices with score >= 40")
    problematic_slices: int = Field(..., description="Number of slices with score < 40")


# =============================================================================
# HOLIDAY / EVENT ANALYSIS SCHEMAS
# =============================================================================

class HolidayAnalysisRequest(BaseModel):
    """Request for holiday/event impact analysis using STL remainder decomposition."""
    data: List[Dict[str, Any]] = Field(..., description="Time series data")
    time_col: str = Field(..., description="Date column name")
    target_col: str = Field(..., description="Target column to analyze")
    frequency: str = Field(default="weekly", description="Data frequency")
    country: str = Field(default="US", description="Country code for holiday calendar")


class HolidayImpactResponse(BaseModel):
    """Quantified impact of a known holiday on the time series."""
    holiday_name: str = Field(..., description="Holiday name (e.g., 'Thanksgiving')")
    avg_lift_pct: float = Field(..., description="Average % deviation from expected (positive = spike, negative = dip)")
    consistency: float = Field(..., ge=0, le=1, description="How consistent the effect is across years (1 = identical every year)")
    direction: str = Field(..., description="'increase' or 'decrease'")
    confidence: str = Field(..., description="'high' (3+ years), 'medium' (2 years), 'low' (1 year)")
    yearly_impacts: Dict[str, float] = Field(default={}, description="Per-year impact percentages")
    recommendation: str = Field(default="", description="Actionable recommendation for training")


class AnomalousEventResponse(BaseModel):
    """An unexplained anomalous week detected in the time series."""
    week_date: str = Field(..., description="ISO date of the anomalous week")
    deviation_pct: float = Field(..., description="% deviation from STL expected value")
    direction: str = Field(..., description="'spike' or 'dip'")
    matched_holiday: Optional[str] = Field(default=None, description="Matched holiday name, if any")
    is_recurring: bool = Field(default=False, description="Whether similar anomaly appears in same week across years")
    note: str = Field(default="", description="Human-readable description")


class HolidayAnalysisResponse(BaseModel):
    """Response from holiday/event impact analysis."""
    holiday_impacts: List[HolidayImpactResponse] = Field(default=[], description="Known holidays with measured impact")
    anomalous_events: List[AnomalousEventResponse] = Field(default=[], description="Unexplained anomalous weeks")
    summary: str = Field(..., description="Human-readable summary")
    training_recommendations: List[str] = Field(default=[], description="Actionable recommendations for training configuration")
    detected_partial_weeks: List[str] = Field(default=[], description="Weeks that appear to be partial data")


# =============================================================================
# MULTI-USER ARCHITECTURE SCHEMAS (Lakebase PostgreSQL)
# =============================================================================

class SessionCreateRequest(BaseModel):
    """Request model for creating a new user session"""
    user_id: str = Field(..., description="Unique user identifier")
    user_email: Optional[str] = Field(None, description="User's email address")
    session_config: Optional[Dict[str, Any]] = Field(default={}, description="Session configuration/preferences")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user@example.com",
                "user_email": "user@example.com",
                "session_config": {"theme": "dark", "default_horizon": 12}
            }
        }


class SessionResponse(BaseModel):
    """Response model for session operations"""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User identifier")
    user_email: Optional[str] = Field(None, description="User's email address")
    created_at: str = Field(..., description="Session creation timestamp (ISO format)")
    last_active_at: str = Field(..., description="Last activity timestamp (ISO format)")
    expires_at: str = Field(..., description="Session expiration timestamp (ISO format)")
    is_active: bool = Field(..., description="Whether the session is active")
    session_config: Dict[str, Any] = Field(default={}, description="Session configuration")


class TrainAsyncRequest(BaseModel):
    """Request model for async training endpoint (delegates to Databricks Jobs)"""
    data: List[Dict[str, Any]] = Field(..., description="Time series data rows")
    time_col: str = Field(..., description="Name of the time/date column")
    target_col: str = Field(..., description="Name of the target column to forecast")
    covariates: List[str] = Field(default=[], description="List of covariate column names")
    horizon: int = Field(default=12, ge=1, le=104, description="Number of periods to forecast (1-104)")
    frequency: Literal['daily', 'weekly', 'monthly'] = Field(default="monthly", description="Data frequency")
    seasonality_mode: Literal['additive', 'multiplicative'] = Field(default="multiplicative", description="Seasonality mode")
    models: List[str] = Field(default=["prophet"], description="Models to train")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    confidence_level: float = Field(default=0.95, ge=0.50, le=0.99, description="Confidence level for intervals")
    hyperparameter_filters: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Hyperparameter filters per model")
    session_id: str = Field(..., description="Session ID from /api/session/create")
    user_id: str = Field(..., description="User identifier")
    priority: Literal['LOW', 'NORMAL', 'HIGH'] = Field(default="NORMAL", description="Job priority")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [{"ds": "2023-01-01", "y": 1000}],
                "time_col": "ds",
                "target_col": "y",
                "horizon": 12,
                "models": ["prophet", "statsforecast"],
                "session_id": "abc-123-def",
                "user_id": "user@example.com"
            }
        }


class TrainAsyncResponse(BaseModel):
    """Response model for async training submission"""
    job_id: str = Field(..., description="Unique job identifier for tracking")
    databricks_run_id: Optional[int] = Field(None, description="Databricks Jobs run ID")
    status: str = Field(..., description="Initial job status (SUBMITTED, QUEUED)")
    message: str = Field(..., description="Status message")
    poll_url: str = Field(..., description="URL to poll for job status")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "abc-123-def-456",
                "databricks_run_id": 12345678,
                "status": "SUBMITTED",
                "message": "Training job submitted to dedicated cluster",
                "poll_url": "/api/job/abc-123-def-456/status"
            }
        }


class JobStatusResponse(BaseModel):
    """Response model for job status polling"""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status: PENDING, QUEUED, RUNNING, COMPLETED, FAILED")
    databricks_run_id: Optional[int] = Field(None, description="Databricks Jobs run ID")
    submitted_at: str = Field(..., description="Submission timestamp (ISO format)")
    started_at: Optional[str] = Field(None, description="Job start timestamp (ISO format)")
    completed_at: Optional[str] = Field(None, description="Job completion timestamp (ISO format)")
    duration_seconds: Optional[int] = Field(None, description="Job duration in seconds")
    progress_message: Optional[str] = Field(None, description="Current progress message")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    # Results summary (available when COMPLETED)
    best_model: Optional[str] = Field(None, description="Best performing model")
    best_mape: Optional[float] = Field(None, description="Best model MAPE")
    mlflow_run_id: Optional[str] = Field(None, description="MLflow run ID for results")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "abc-123-def-456",
                "status": "RUNNING",
                "databricks_run_id": 12345678,
                "submitted_at": "2025-01-27T10:30:00Z",
                "started_at": "2025-01-27T10:31:00Z",
                "completed_at": None,
                "progress_message": "Training Prophet model (2/5)"
            }
        }


class JobResultsResponse(BaseModel):
    """Response model for completed job results"""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status (should be COMPLETED)")
    models: List[Dict[str, Any]] = Field(..., description="Results for each trained model")
    best_model: str = Field(..., description="Best performing model name")
    forecast: List[Dict[str, Any]] = Field(..., description="Best model's forecast")
    validation: List[Dict[str, Any]] = Field(..., description="Best model's validation results")
    mlflow_run_id: str = Field(..., description="MLflow run ID")
    mlflow_experiment_url: Optional[str] = Field(None, description="URL to MLflow experiment")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "abc-123-def-456",
                "status": "COMPLETED",
                "best_model": "prophet",
                "models": [{"model_name": "prophet", "mape": 5.2}],
                "forecast": [{"ds": "2025-02-01", "yhat": 1100}],
                "validation": [{"ds": "2024-12-01", "y": 1000, "yhat": 1010}],
                "mlflow_run_id": "run-abc-123"
            }
        }


class UserHistoryItem(BaseModel):
    """Single item in user's execution history"""
    job_id: str = Field(..., description="Job identifier")
    submitted_at: str = Field(..., description="Submission timestamp")
    status: str = Field(..., description="Job status")
    best_model: Optional[str] = Field(None, description="Best model (if completed)")
    best_mape: Optional[float] = Field(None, description="Best MAPE (if completed)")
    duration_seconds: Optional[int] = Field(None, description="Job duration")
    horizon: int = Field(..., description="Forecast horizon")
    frequency: str = Field(..., description="Data frequency")
    models: List[str] = Field(..., description="Models trained")


class UserHistoryResponse(BaseModel):
    """Response model for user execution history"""
    user_id: str = Field(..., description="User identifier")
    total_executions: int = Field(..., description="Total number of executions")
    executions: List[UserHistoryItem] = Field(..., description="Execution history items")


class ReproduceJobRequest(BaseModel):
    """Request model for reproducing a previous job"""
    session_id: str = Field(..., description="Current session ID")
    user_id: str = Field(..., description="User identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "new-session-id",
                "user_id": "user@example.com"
            }
        }


class ReproduceJobResponse(BaseModel):
    """Response model for job reproduction"""
    new_job_id: str = Field(..., description="New job identifier")
    reproduced_from: str = Field(..., description="Original job identifier")
    original_params: Dict[str, Any] = Field(..., description="Original request parameters")
    data_hash_match: bool = Field(..., description="Whether data hash matches original")
    status: str = Field(..., description="New job status")
    poll_url: str = Field(..., description="URL to poll for new job status")
