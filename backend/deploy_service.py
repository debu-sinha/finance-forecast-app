"""
Deployment service for Databricks Model Serving
"""
import os
import logging
from typing import Optional, Dict, Any
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, AutoCaptureConfigInput

logger = logging.getLogger(__name__)

def get_databricks_client() -> WorkspaceClient:
    try:
        return WorkspaceClient()
    except Exception as e:
        logger.warning(f"Auto-auth failed: {e}. Trying env vars.")
        host, token = os.environ.get("DATABRICKS_HOST"), os.environ.get("DATABRICKS_TOKEN")
        if not host or not token: raise ValueError("DATABRICKS_HOST/TOKEN required")
        return WorkspaceClient(host=host, token=token)

def deploy_model_to_serving(model_name: str, model_version: str, endpoint_name: str, workload_size: str = "Small", scale_to_zero: bool = True) -> Dict[str, Any]:
    try:
        client = get_databricks_client()
        
        if model_version == 'latest':
            try:
                all_versions = set(v.version for v in client.model_versions.list(full_name=model_name))
                if all_versions: model_version = str(max(all_versions))
            except Exception:
                try:
                    import mlflow
                    model_version = str(mlflow.MlflowClient().get_latest_versions(name=model_name, stages=[])[0].version)
                except: model_version = "1"
        
        logger.info(f"Deploying {model_name} v{model_version} to {endpoint_name}")
        
        try:
            client.serving_endpoints.get(endpoint_name)
            client.serving_endpoints.update_config(
                name=endpoint_name,
                served_entities=[ServedEntityInput(entity_name=model_name, entity_version=model_version, workload_size=workload_size, scale_to_zero_enabled=scale_to_zero)]
            )
            status, msg = "updating", f"Updating {endpoint_name} with {model_name} v{model_version}"
        except Exception:
            client.serving_endpoints.create(
                name=endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=[ServedEntityInput(entity_name=model_name, entity_version=model_version, workload_size=workload_size, scale_to_zero_enabled=scale_to_zero)],
                    auto_capture_config=AutoCaptureConfigInput(catalog_name=os.getenv("UC_CATALOG_NAME", "main"), schema_name=os.getenv("UC_SCHEMA_NAME", "default"), table_name_prefix=f"{endpoint_name}_payload")
                )
            )
            status, msg = "creating", f"Creating {endpoint_name} with {model_name} v{model_version}"
            
        return {"endpoint_name": endpoint_name, "status": status, "message": msg, "endpoint_url": f"{client.config.host}/serving-endpoints/{endpoint_name}", "deployed_version": model_version}
            
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise Exception(f"Failed to deploy: {str(e)}")

def get_endpoint_status(endpoint_name: str) -> Dict[str, Any]:
    try:
        client = get_databricks_client()
        endpoint = client.serving_endpoints.get(endpoint_name)
        return {"endpoint_name": endpoint_name, "state": endpoint.state.config_update if endpoint.state else "unknown", "ready": endpoint.state.ready if endpoint.state else "unknown", "endpoint_url": f"{client.config.host}/serving-endpoints/{endpoint_name}"}
    except Exception as e:
        return {"endpoint_name": endpoint_name, "state": "error", "error": str(e)}

def delete_endpoint(endpoint_name: str) -> Dict[str, str]:
    try:
        get_databricks_client().serving_endpoints.delete(endpoint_name)
        return {"endpoint_name": endpoint_name, "status": "deleted", "message": f"Endpoint {endpoint_name} deleted"}
    except Exception as e:
        raise Exception(f"Failed to delete: {str(e)}")

def invoke_endpoint(endpoint_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return {"predictions": get_databricks_client().serving_endpoints.query(name=endpoint_name, dataframe_records=[data]).predictions, "endpoint_name": endpoint_name}
    except Exception as e:
        raise Exception(f"Failed to invoke: {str(e)}")

def list_endpoints() -> Dict[str, Any]:
    try:
        endpoints = [{"name": e.name, "state": e.state.config_update if e.state else "unknown", "creator": e.creator} for e in get_databricks_client().serving_endpoints.list()]
        return {"endpoints": endpoints, "count": len(endpoints)}
    except Exception as e:
        return {"endpoints": [], "count": 0, "error": str(e)}


def test_model_inference(
    model_name: str,
    model_version: str,
    test_periods: int = 5,
    start_date: Optional[str] = None,
    frequency: str = "daily"
) -> Dict[str, Any]:
    """
    Test a registered model by loading it as pyfunc and running inference.
    This validates the model can be loaded and produces valid predictions
    before deploying to a serving endpoint.

    Args:
        model_name: Full Unity Catalog model name (e.g., main.default.finance_forecast_model)
        model_version: Version number to test
        test_periods: Number of periods to forecast for testing
        start_date: Optional start date (YYYY-MM-DD). If not provided, defaults to today.
        frequency: Forecast frequency (daily, weekly, monthly)

    Returns:
        Dictionary with test results including predictions, timing, and schema info
    """
    import time
    import mlflow
    import pandas as pd
    from datetime import datetime, timedelta

    result = {
        "model_name": model_name,
        "model_version": model_version,
        "test_passed": False,
        "message": "",
        "load_time_seconds": 0.0,
        "inference_time_seconds": 0.0,
        "sample_predictions": [],
        "input_schema": None,
        "output_schema": None,
        "error_details": None
    }

    try:
        # Configure MLflow for Unity Catalog if needed
        if "." in model_name:
            mlflow.set_registry_uri("databricks-uc")

        model_uri = f"models:/{model_name}/{model_version}"
        logger.info(f"🧪 Testing model: {model_uri}")

        # Step 1: Load the model and measure time
        load_start = time.time()
        try:
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            load_time = time.time() - load_start
            result["load_time_seconds"] = round(load_time, 3)
            logger.info(f"   ✅ Model loaded in {load_time:.3f}s")
        except Exception as load_error:
            result["message"] = f"Failed to load model: {str(load_error)}"
            result["error_details"] = str(load_error)
            logger.error(f"   ❌ Model load failed: {load_error}")
            return result

        # Step 2: Extract schema information and input_example
        logged_input_example = None
        try:
            if hasattr(loaded_model, 'metadata') and loaded_model.metadata:
                if hasattr(loaded_model.metadata, 'signature') and loaded_model.metadata.signature:
                    sig = loaded_model.metadata.signature
                    if sig.inputs:
                        result["input_schema"] = {"schema": str(sig.inputs)}
                    if sig.outputs:
                        result["output_schema"] = {"schema": str(sig.outputs)}
                    logger.info(f"   📋 Input schema: {sig.inputs}")
                    logger.info(f"   📋 Output schema: {sig.outputs}")

                # Try to get the input_example from the model artifacts
                if hasattr(loaded_model.metadata, 'saved_input_example_info'):
                    try:
                        # Load the input example from MLflow artifacts
                        import mlflow
                        client = mlflow.tracking.MlflowClient()
                        artifact_path = loaded_model.metadata.saved_input_example_info.get('artifact_path', 'input_example.json')
                        run_id = loaded_model.metadata.run_id if hasattr(loaded_model.metadata, 'run_id') else None

                        if run_id:
                            local_path = client.download_artifacts(run_id, artifact_path)
                            logged_input_example = pd.read_json(local_path, orient='split')
                            logger.info(f"   📋 Retrieved logged input_example from artifacts")
                            logger.info(f"      Columns: {list(logged_input_example.columns)}")
                            logger.info(f"      Sample: {logged_input_example.iloc[0].to_dict() if len(logged_input_example) > 0 else 'empty'}")
                    except Exception as input_ex_error:
                        logger.debug(f"   Could not load input_example from artifacts: {input_ex_error}")

        except Exception as schema_error:
            logger.warning(f"   ⚠️ Could not extract schema: {schema_error}")

        # Step 3: Prepare test input based on model type
        # Prefer using the logged input_example if available
        test_input = None

        if logged_input_example is not None and len(logged_input_example) > 0:
            # Use the logged input_example as the test input
            test_input = logged_input_example.copy()
            logger.info(f"   🔮 Using logged input_example for testing:")
            logger.info(f"      Columns: {list(test_input.columns)}")
            logger.info(f"      Values: {test_input.iloc[0].to_dict()}")

            # Optionally update the start_date/ds if provided
            if start_date:
                if 'start_date' in test_input.columns:
                    test_input['start_date'] = start_date
                if 'ds' in test_input.columns:
                    test_input['ds'] = start_date
                logger.info(f"      Updated start_date to: {start_date}")

            # Update periods if different from logged example
            if 'periods' in test_input.columns and test_input['periods'].iloc[0] != test_periods:
                test_input['periods'] = test_periods
                logger.info(f"      Updated periods to: {test_periods}")
        else:
            # Fall back to constructing test input manually
            if start_date:
                test_start = pd.to_datetime(start_date)
            else:
                test_start = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)

            # Check if model requires covariates from the signature
            required_covariates = []
            if result["input_schema"]:
                schema_str = result["input_schema"].get("schema", "")
                # Parse covariate names from schema string
                import re
                # Match patterns like "'Super Bowl': long" or "\"Valentine's Day\": long"
                # Handle both single and double quotes, and names with apostrophes
                # Use a more robust parsing approach
                parts = schema_str.replace('[', '').replace(']', '').split(', ')
                for part in parts:
                    # Match 'name': type or "name": type
                    match = re.match(r"['\"](.+?)['\"]:\s*\w+", part.strip())
                    if match:
                        col_name = match.group(1)
                        # Filter out non-covariate columns
                        non_covariates = {'ds', 'periods', 'start_date', 'y', 'yhat', 'yhat_lower', 'yhat_upper'}
                        if col_name not in non_covariates:
                            required_covariates.append(col_name)
                if required_covariates:
                    logger.info(f"   📋 Model requires covariates: {required_covariates[:5]}{'...' if len(required_covariates) > 5 else ''}")

            # Try simple mode first (works for XGBoost, ARIMA, ETS, SARIMAX)
            test_input = pd.DataFrame({
                "periods": [test_periods],
                "start_date": [test_start.strftime("%Y-%m-%d")],
                "ds": [test_start.strftime("%Y-%m-%d")]  # Some models use 'ds' instead of 'start_date'
            })

            # Add required covariates with default values (0 for binary covariates)
            for cov in required_covariates:
                test_input[cov] = 0

            logger.info(f"   🔮 Test input (constructed): periods={test_periods}, start_date={test_start.strftime('%Y-%m-%d')}, frequency={frequency}")

        # Step 4: Run inference and measure time
        inference_start = time.time()
        try:
            predictions = loaded_model.predict(test_input)
            inference_time = time.time() - inference_start
            result["inference_time_seconds"] = round(inference_time, 3)
            logger.info(f"   ✅ Inference completed in {inference_time:.3f}s")
        except Exception as pred_error:
            # If simple mode fails, try alternative input format
            logger.warning(f"   ⚠️ Simple mode failed: {pred_error}, trying date-based mode...")
            try:
                # Generate date range for forecast
                freq_map = {"daily": "D", "weekly": "W", "monthly": "MS"}
                pandas_freq = freq_map.get(frequency, "D")
                future_dates = pd.date_range(start=test_start, periods=test_periods, freq=pandas_freq)
                test_input = pd.DataFrame({"ds": future_dates})

                # Add required covariates with default values
                for cov in required_covariates:
                    test_input[cov] = 0

                inference_start = time.time()
                predictions = loaded_model.predict(test_input)
                inference_time = time.time() - inference_start
                result["inference_time_seconds"] = round(inference_time, 3)
                logger.info(f"   ✅ Inference (date mode) completed in {inference_time:.3f}s")
            except Exception as alt_error:
                result["message"] = f"Inference failed: {str(pred_error)} | Alt mode: {str(alt_error)}"
                result["error_details"] = f"Simple mode: {str(pred_error)}\nDate mode: {str(alt_error)}"
                logger.error(f"   ❌ Inference failed: {alt_error}")
                return result

        # Step 5: Validate predictions thoroughly
        if predictions is None or (isinstance(predictions, pd.DataFrame) and predictions.empty):
            result["message"] = "Model returned empty predictions"
            result["error_details"] = "Predictions DataFrame is empty or None"
            logger.error("   ❌ Empty predictions returned")
            return result

        # Validate prediction quality
        if isinstance(predictions, pd.DataFrame):
            # Check for required columns
            yhat_col = None
            for col in ['yhat', 'prediction', 'forecast', 'y_pred']:
                if col in predictions.columns:
                    yhat_col = col
                    break

            if yhat_col:
                yhat_values = predictions[yhat_col]

                # Check for NaN values
                nan_count = yhat_values.isna().sum()
                if nan_count > 0:
                    nan_pct = (nan_count / len(yhat_values)) * 100
                    if nan_pct > 50:
                        result["message"] = f"Model predictions contain too many NaN values ({nan_pct:.1f}%)"
                        result["error_details"] = f"{nan_count}/{len(yhat_values)} predictions are NaN"
                        logger.error(f"   ❌ Too many NaN predictions: {nan_count}/{len(yhat_values)}")
                        return result
                    else:
                        logger.warning(f"   ⚠️ Some NaN predictions: {nan_count}/{len(yhat_values)} ({nan_pct:.1f}%)")

                # Check for infinite values
                inf_count = (~yhat_values.isna() & ~pd.to_numeric(yhat_values, errors='coerce').between(-1e15, 1e15)).sum()
                if inf_count > 0:
                    logger.warning(f"   ⚠️ Some extreme/infinite predictions: {inf_count}/{len(yhat_values)}")

                # Check if all predictions are the same (might indicate a problem)
                if len(yhat_values.dropna().unique()) == 1 and len(yhat_values) > 1:
                    logger.warning(f"   ⚠️ All predictions are identical: {yhat_values.iloc[0]}")

                logger.info(f"   ✅ Prediction validation passed: {len(yhat_values)} values, {nan_count} NaN, range [{yhat_values.min():.2f}, {yhat_values.max():.2f}]")

            # Check for expected output columns (ds, yhat, yhat_lower, yhat_upper)
            expected_cols = ['ds', 'yhat']
            optional_cols = ['yhat_lower', 'yhat_upper']
            missing_cols = [col for col in expected_cols if col not in predictions.columns]

            if missing_cols:
                logger.warning(f"   ⚠️ Missing expected columns: {missing_cols}. Available: {list(predictions.columns)}")
            else:
                logger.info(f"   ✅ Output structure validated: {[c for c in expected_cols + optional_cols if c in predictions.columns]}")

            # Validate number of predictions matches requested periods
            if len(predictions) != test_periods:
                logger.warning(f"   ⚠️ Expected {test_periods} predictions, got {len(predictions)}")
            else:
                logger.info(f"   ✅ Correct number of predictions: {len(predictions)}")

        # Convert predictions to list of dicts for response
        if isinstance(predictions, pd.DataFrame):
            # Handle different column names
            pred_df = predictions.copy()

            # Standardize date column
            date_col = None
            for col in ['ds', 'date', 'time', 'timestamp']:
                if col in pred_df.columns:
                    date_col = col
                    break

            if date_col and pred_df[date_col].dtype != 'object':
                pred_df[date_col] = pd.to_datetime(pred_df[date_col]).dt.strftime('%Y-%m-%d')

            # Convert to records
            sample_preds = pred_df.head(min(5, len(pred_df))).to_dict('records')

            # Round numeric values for cleaner output
            for pred in sample_preds:
                for key, val in pred.items():
                    if isinstance(val, float):
                        pred[key] = round(val, 2)

            result["sample_predictions"] = sample_preds
            logger.info(f"   📊 Sample predictions: {len(sample_preds)} rows")
            for i, pred in enumerate(sample_preds[:3]):
                logger.info(f"      [{i}] {pred}")
        else:
            result["sample_predictions"] = [{"prediction": float(p) if hasattr(p, '__float__') else str(p)} for p in predictions[:5]]

        # Step 6: All tests passed
        result["test_passed"] = True
        result["message"] = f"Model loaded and inference successful. Generated {len(predictions) if hasattr(predictions, '__len__') else 'N/A'} predictions."
        logger.info(f"   ✅ TEST PASSED: {result['message']}")

        return result

    except Exception as e:
        result["message"] = f"Test failed: {str(e)}"
        result["error_details"] = str(e)
        logger.error(f"   ❌ Test failed with exception: {e}")
        return result
