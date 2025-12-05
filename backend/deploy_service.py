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
