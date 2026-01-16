import numpy as np
import pandas as pd
import logging
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy import stats

logger = logging.getLogger(__name__)


def detect_weekly_freq_code(df: pd.DataFrame, frequency: str) -> str:
    """
    Detect the appropriate weekly frequency code based on actual data.

    For weekly data, determines which day of week the data starts on
    (e.g., W-MON for Monday-based weeks, W-SUN for Sunday-based weeks).

    Args:
        df: DataFrame with date column (expects 'ds' or datetime column)
        frequency: Data frequency ('daily', 'weekly', 'monthly', 'yearly')

    Returns:
        Pandas frequency string (e.g., 'D', 'W-MON', 'MS')
    """
    if frequency != 'weekly':
        return {'daily': 'D', 'monthly': 'MS', 'yearly': 'YS'}.get(frequency, 'MS')

    try:
        if 'ds' in df.columns:
            dates = pd.to_datetime(df['ds'])
        else:
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                dates = df[date_cols[0]]
            else:
                return 'W-MON'

        if len(dates) > 0:
            day_counts = dates.dt.dayofweek.value_counts()
            most_common_day = day_counts.idxmax()
            day_names = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
            return f"W-{day_names[most_common_day]}"
    except Exception:
        pass
    return 'W-MON'

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute forecast accuracy metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = 0.0
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 4)
    }

def time_series_cross_validate(
    y: np.ndarray,
    model_fit_fn,
    model_predict_fn,
    n_splits: int = 3,
    min_train_size: int = None,
    horizon: int = 1
) -> Dict[str, Any]:
    """
    Perform time series cross-validation with expanding window.
    """
    n = len(y)
    if min_train_size is None:
        min_train_size = max(n // 2, 10)  # At least 50% or 10 points

    available_for_cv = n - min_train_size
    if available_for_cv < n_splits * horizon:
        n_splits = max(1, available_for_cv // horizon)
        logger.warning(f"Reduced CV splits to {n_splits} due to limited data")

    if n_splits < 1:
        logger.warning("Not enough data for cross-validation, using simple holdout")
        return {"cv_scores": [], "mean_mape": None, "std_mape": None, "n_splits": 0}

    fold_size = (n - min_train_size) // n_splits
    cv_scores = []

    for i in range(n_splits):
        split_point = min_train_size + i * fold_size
        test_end = min(split_point + horizon, n)

        y_train = y[:split_point]
        y_test = y[split_point:test_end]

        if len(y_test) == 0:
            continue

        try:
            fitted_model = model_fit_fn(y_train)
            predictions = model_predict_fn(fitted_model, len(y_test))

            fold_mape = mean_absolute_percentage_error(y_test, predictions) * 100
            cv_scores.append(fold_mape)
            logger.info(f"  CV Fold {i+1}/{n_splits}: train={len(y_train)}, test={len(y_test)}, MAPE={fold_mape:.2f}%")
        except Exception as e:
            logger.warning(f"  CV Fold {i+1} failed: {e}")
            continue

    if len(cv_scores) == 0:
        return {"cv_scores": [], "mean_mape": None, "std_mape": None, "n_splits": 0}

    return {
        "cv_scores": cv_scores,
        "mean_mape": round(np.mean(cv_scores), 2),
        "std_mape": round(np.std(cv_scores), 2),
        "n_splits": len(cv_scores)
    }

def compute_prediction_intervals(
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    forecast_values: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute statistically valid prediction intervals based on residual distribution.
    """
    residuals = y_train - y_pred_train
    residual_std = np.std(residuals)
    n = len(residuals)
    
    if n < 30:
        alpha = 1 - confidence_level
        t_value = stats.t.ppf(1 - alpha/2, df=n-1)
        margin = t_value * residual_std
    else:
        z_value = stats.norm.ppf(1 - (1 - confidence_level)/2)
        margin = z_value * residual_std

    lower_bounds = forecast_values - margin
    upper_bounds = forecast_values + margin

    if np.all(forecast_values > 0):
        lower_bounds = np.maximum(lower_bounds, forecast_values * 0.1)

    return lower_bounds, upper_bounds

def register_model_to_unity_catalog(model_uri: str, model_name: str, tags: Optional[Dict[str, str]] = None) -> str:
    """Register a model to Unity Catalog with improved error handling"""
    try:
        run_id = tags.get("run_id") if tags else (model_uri.split("/")[1] if model_uri.startswith("runs:/") and len(model_uri.split("/")) > 1 else None)
        client = MlflowClient()
        
        if run_id:
            try:
                logger.info(f"Checking if run {run_id} is already registered as {model_name}...")
                for v in client.search_model_versions(f"name='{model_name}'"):
                    if v.run_id == run_id:
                        logger.info(f"♻️  Model already registered as version {v.version}, skipping...")
                        return str(v.version)
            except Exception as check_error:
                logger.warning(f"Could not check existing versions: {check_error}")

        if "." in model_name:
            logger.info("Configuring MLflow to use Unity Catalog registry (databricks-uc)")
            mlflow.set_registry_uri("databricks-uc")
            
        logger.info(f"Registering model from {model_uri} to {model_name}...")
        result = None
        try:
            result = mlflow.register_model(model_uri=model_uri, name=model_name, tags=tags)
            logger.info(f"Successfully registered as version {result.version} with tags")
            
            if tags and result.version:
                try:
                    version_str = str(result.version)
                    mv = client.get_model_version(name=model_name, version=version_str)
                    actual_tags = mv.tags if hasattr(mv, 'tags') and mv.tags else {}
                    if not actual_tags:
                        logger.warning(f"    Tags were not set via register_model, trying client API...")
                        for tag_key, tag_value in tags.items():
                            try:
                                client.set_model_version_tag(name=model_name, version=version_str, key=tag_key, value=str(tag_value))
                                logger.info(f"   ✓ Set tag via client API: {tag_key}={tag_value}")
                            except Exception as e:
                                logger.warning(f"   ✗ Failed to set tag {tag_key}: {str(e)[:100]}")
                except Exception as verify_error:
                    logger.warning(f"   Could not verify tags: {verify_error}")
            
            return str(result.version)
        except Exception as reg_error:
            error_str = str(reg_error).lower()
            if 'tag' in error_str or 'permission_denied' in error_str or 'tag assignment' in error_str:
                logger.info(f"ℹ️  Tag registration via register_model failed, registering without tags then adding tags via client API...")
                try:
                    result = mlflow.register_model(model_uri=model_uri, name=model_name)
                    logger.info(f"Successfully registered as version {result.version}")
                    
                    if tags and result.version:
                        version_str = str(result.version)
                        for tag_key, tag_value in tags.items():
                            try:
                                client.set_model_version_tag(
                                    name=model_name,
                                    version=version_str,
                                    key=tag_key,
                                    value=str(tag_value)
                                )
                                logger.info(f"   ✓ Added tag: {tag_key}={tag_value} to {model_name} version {version_str}")
                            except Exception as tag_error:
                                error_str = str(tag_error).lower()
                                if 'tag assignment' in error_str or 'tag policy' in error_str or 'permission_denied' in error_str:
                                    logger.info(f"   ℹ️  Skipped tag {tag_key} (restricted by tag policies)")
                                else:
                                    logger.warning(f"   ✗ Failed to add tag {tag_key}: {str(tag_error)[:100]}")
                    return str(result.version)
                except Exception as e:
                    logger.error(f"Registration failed even without tags: {e}")
                    raise e
            else:
                raise reg_error
                
    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        return "0"

def log_artifact_with_validation(artifact_path: str, artifact_dir: str, description: str) -> bool:
    """
    Log an artifact to MLflow with validation and detailed logging.

    Args:
        artifact_path: Local path to the artifact file
        artifact_dir: Directory in MLflow to store the artifact
        description: Human-readable description of the artifact

    Returns:
        True if successful, False otherwise
    """
    import os
    try:
        if not os.path.exists(artifact_path):
            logger.warning(f"   ⚠️ Artifact not found: {artifact_path}")
            return False

        file_size = os.path.getsize(artifact_path)
        file_name = os.path.basename(artifact_path)

        # Log the artifact
        mlflow.log_artifact(artifact_path, artifact_dir)

        # Log success with details
        logger.info(f"   ✅ Logged {description}: {artifact_dir}/{file_name} ({file_size:,} bytes)")
        return True

    except Exception as e:
        logger.error(f"   ❌ Failed to log {description}: {e}")
        return False


def log_model_with_validation(
    model_name: str,
    artifact_path: str,
    python_model,
    signature,
    input_example: pd.DataFrame,
    pip_requirements: List[str] = None
) -> bool:
    """
    Log a model to MLflow with validation and detailed logging of signature and input example.

    Args:
        model_name: Name for the model (e.g., "Prophet", "XGBoost")
        artifact_path: Path in MLflow to store the model
        python_model: The model wrapper/object to log
        signature: MLflow model signature
        input_example: Example input DataFrame
        pip_requirements: List of pip requirements

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"")
        logger.info(f"   {'='*50}")
        logger.info(f"   📦 LOGGING {model_name.upper()} MODEL TO MLFLOW")
        logger.info(f"   {'='*50}")

        # Log signature details
        if signature:
            logger.info(f"   📝 Model Signature:")
            if hasattr(signature, 'inputs') and signature.inputs:
                logger.info(f"      Inputs: {signature.inputs}")
            if hasattr(signature, 'outputs') and signature.outputs:
                logger.info(f"      Outputs: {signature.outputs}")
        else:
            logger.warning(f"   ⚠️ No signature provided")

        # Log input example details
        if input_example is not None and len(input_example) > 0:
            logger.info(f"   📋 Input Example:")
            logger.info(f"      Shape: {input_example.shape}")
            logger.info(f"      Columns: {list(input_example.columns)}")
            logger.info(f"      Dtypes: {dict(input_example.dtypes)}")
            logger.info(f"      Sample row: {input_example.iloc[0].to_dict()}")
        else:
            logger.warning(f"   ⚠️ No input example provided")

        # Log pip requirements
        if pip_requirements:
            logger.info(f"   📦 Pip Requirements: {pip_requirements}")

        # Perform the actual model logging
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=python_model,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements
        )

        logger.info(f"   ✅ Model logged successfully to: {artifact_path}")
        logger.info(f"   {'='*50}")
        return True

    except Exception as e:
        logger.error(f"   ❌ Failed to log model: {e}")
        return False


def validate_mlflow_run_artifacts(run_id: str) -> Dict[str, Any]:
    """
    Validate that all expected artifacts were logged to an MLflow run.

    Args:
        run_id: The MLflow run ID to validate

    Returns:
        Dictionary with validation results
    """
    try:
        client = MlflowClient()

        logger.info(f"")
        logger.info(f"   {'='*50}")
        logger.info(f"   🔍 VALIDATING MLFLOW RUN ARTIFACTS")
        logger.info(f"   {'='*50}")
        logger.info(f"   Run ID: {run_id}")

        # List all artifacts
        artifacts = client.list_artifacts(run_id)

        validation_result = {
            "run_id": run_id,
            "artifacts_found": [],
            "model_logged": False,
            "datasets_logged": False,
            "validation_passed": True,
            "issues": []
        }

        def list_artifacts_recursive(path=""):
            """Recursively list all artifacts"""
            items = client.list_artifacts(run_id, path)
            all_artifacts = []
            for item in items:
                if item.is_dir:
                    all_artifacts.extend(list_artifacts_recursive(item.path))
                else:
                    all_artifacts.append(item.path)
            return all_artifacts

        all_artifact_paths = list_artifacts_recursive()
        validation_result["artifacts_found"] = all_artifact_paths

        logger.info(f"   Found {len(all_artifact_paths)} artifacts:")
        for artifact_path in sorted(all_artifact_paths):
            logger.info(f"      - {artifact_path}")

        # Check for model
        model_artifacts = [a for a in all_artifact_paths if 'model' in a.lower() and ('MLmodel' in a or 'model.pkl' in a or 'python_model.pkl' in a)]
        if model_artifacts:
            validation_result["model_logged"] = True
            logger.info(f"   ✅ Model artifacts found: {model_artifacts}")
        else:
            validation_result["model_logged"] = False
            validation_result["issues"].append("No model artifacts found")
            logger.warning(f"   ⚠️ No model artifacts found")

        # Check for datasets
        dataset_artifacts = [a for a in all_artifact_paths if 'datasets' in a.lower() or 'data' in a.lower()]
        if dataset_artifacts:
            validation_result["datasets_logged"] = True
            logger.info(f"   ✅ Dataset artifacts found: {len(dataset_artifacts)} files")
        else:
            validation_result["datasets_logged"] = False
            validation_result["issues"].append("No dataset artifacts found")
            logger.warning(f"   ⚠️ No dataset artifacts found")

        # Check for signature
        mlmodel_path = [a for a in all_artifact_paths if 'MLmodel' in a]
        if mlmodel_path:
            logger.info(f"   ✅ MLmodel file found (contains signature)")

        validation_result["validation_passed"] = len(validation_result["issues"]) == 0

        if validation_result["validation_passed"]:
            logger.info(f"   ✅ VALIDATION PASSED - All expected artifacts found")
        else:
            logger.warning(f"   ⚠️ VALIDATION ISSUES: {validation_result['issues']}")

        logger.info(f"   {'='*50}")

        return validation_result

    except Exception as e:
        logger.error(f"   ❌ Artifact validation failed: {e}")
        return {
            "run_id": run_id,
            "validation_passed": False,
            "error": str(e)
        }


def analyze_covariate_impact(
    model,  # Prophet model (type hint removed for lazy import)
    df: pd.DataFrame,
    covariates: List[str]
) -> List[Dict[str, Any]]:
    """
    Analyze the impact of each covariate on the forecast (Prophet only)
    """
    # Lazy import to avoid dependency issues if Prophet is not installed/used
    try:
        from prophet import Prophet
    except ImportError:
        return []
    
    impacts = []
    
    if not covariates:
        return impacts
    
    try:
        # Get regressor coefficients from Prophet model
        if hasattr(model, 'params') and 'beta' in model.params:
            # Prophet stores regressor coefficients in params['beta']
            beta = model.params['beta']
            
            # Handle potential 2D array (1, n_regressors)
            if len(beta.shape) > 1 and beta.shape[0] == 1:
                beta = beta[0]
            
            for i, cov in enumerate(covariates):
                if i < len(beta):
                    coef = beta[i]

                    # Calculate importance score (normalized)
                    std = df[cov].std() if cov in df.columns else 1
                    impact_score = abs(coef * std)

                    impacts.append({
                        'name': cov,
                        'coefficient': float(coef),
                        'impact_score': float(impact_score),
                        'direction': 'positive' if coef > 0 else 'negative'
                    })

        # Sort by impact score
        impacts.sort(key=lambda x: x['impact_score'], reverse=True)

        # Add normalized score (0-100) based on relative impact
        if impacts:
            max_impact = max(i['impact_score'] for i in impacts) if impacts else 1
            for impact in impacts:
                impact['score'] = float(min(100, (impact['impact_score'] / max_impact * 100) if max_impact > 0 else 0))
        
    except Exception as e:
        logger.warning(f"Could not analyze covariate impacts: {e}")
        
    return impacts
