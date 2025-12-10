"""
Pydantic models for request/response validation
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


import os

class DataRow(BaseModel):
    """Represents a single row of data"""
    data: Dict[str, Any]
    
    class Config:
        extra = "allow"


class TrainRequest(BaseModel):
    """Request model for training endpoint"""
    data: List[Dict[str, Any]] = Field(..., description="Time series data rows")
    time_col: str = Field(..., description="Name of the time/date column")
    target_col: str = Field(..., description="Name of the target column to forecast")
    covariates: List[str] = Field(default=[], description="List of covariate column names")
    horizon: int = Field(default=12, description="Number of periods to forecast")
    frequency: str = Field(default="monthly", description="Frequency: 'weekly', 'monthly', or 'daily'")
    seasonality_mode: str = Field(default="multiplicative", description="'additive' or 'multiplicative'")
    test_size: Optional[int] = Field(default=None, description="Size of test set for validation")
    regressor_method: str = Field(default="mean", description="How to fill future covariates: 'mean', 'last_value', 'linear_trend'")
    models: List[str] = Field(default=["prophet"], description="Models to train: 'prophet', 'arima', 'exponential_smoothing', 'sarimax', 'xgboost'")
    catalog_name: str = Field(default=os.getenv("UC_CATALOG_NAME", "main"), description="Unity Catalog catalog name")
    schema_name: str = Field(default=os.getenv("UC_SCHEMA_NAME", "default"), description="Unity Catalog schema name")
    model_name: str = Field(default=os.getenv("UC_MODEL_NAME_ONLY", "finance_forecast_model"), description="Model name in Unity Catalog")
    country: str = Field(default="US", description="Country code for holiday calendar (US, UK, CA, etc.)")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata about data filters applied (e.g., {'store': 'Store A', 'region': 'West'})")
    from_date: Optional[str] = Field(default=None, description="Start date for filtering training data (YYYY-MM-DD format). If provided, only data from this date onwards will be used.")
    to_date: Optional[str] = Field(default=None, description="End date for filtering training data (YYYY-MM-DD format). If provided, only data up to this date will be used.")
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducibility. Set to ensure consistent results across runs.")
    future_features: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional future feature data for prediction horizon. If provided, actual feature values will be used instead of imputation.")
    # Batch training context - enables grouped MLflow tracking
    batch_id: Optional[str] = Field(default=None, description="Unique batch training session ID. When provided, all segments use the same MLflow experiment.")
    batch_segment_id: Optional[str] = Field(default=None, description="Human-readable segment identifier (e.g., 'region=US | product=Widget')")
    batch_segment_index: Optional[int] = Field(default=None, description="Segment index within the batch (1-based)")
    batch_total_segments: Optional[int] = Field(default=None, description="Total number of segments in the batch")
    
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


class TrainResponse(BaseModel):
    """Response model for training endpoint"""
    models: List[ModelResult] = Field(..., description="Results for each trained model")
    best_model: str = Field(..., description="Best performing model name")
    artifact_uri: str = Field(..., description="MLflow artifact location")
    history: List[Dict[str, Any]] = Field(default=[], description="Historical data (actuals) used for training")
    
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
    max_workers: int = Field(default=4, description="Maximum number of parallel workers")

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
