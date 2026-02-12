"""
History Service for Finance Forecasting Platform.

Provides execution history tracking with full reproducibility support.
Enables users to:
- View past executions with parameters and results
- Reproduce any previous forecast with exact same configuration
- Compare results across different runs
- Track model performance over time

Key Features:
- Complete parameter capture for reproducibility
- Data hash verification for integrity
- MLflow integration for model lineage
- Results storage with metrics and predictions
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from backend.utils.logging_utils import log_io

from backend.services.lakebase_client import (
    LakebaseClient,
    compute_data_hash,
    get_lakebase_client,
)

logger = logging.getLogger(__name__)


class HistoryService:
    """
    Manages execution history and reproducibility.

    Usage:
        service = HistoryService()

        # Create execution record
        job_id = await service.create_execution(
            session_id=session_id,
            user_id=user_id,
            request=train_request.dict(),
            data=uploaded_data,
        )

        # Get reproduction parameters
        repro = await service.get_reproduction_params(job_id)

        # Compare runs
        comparison = await service.compare_executions([job_id_1, job_id_2])
    """

    def __init__(self, client: Optional[LakebaseClient] = None):
        """
        Initialize HistoryService.

        Args:
            client: Optional LakebaseClient instance.
        """
        self._client = client

    @log_io
    async def _get_client(self) -> LakebaseClient:
        """Get Lakebase client."""
        if self._client is None:
            self._client = await get_lakebase_client()
        return self._client

    @log_io
    async def create_execution(
        self,
        session_id: str,
        user_id: str,
        request: Dict[str, Any],
        data: List[Dict[str, Any]],
        data_upload_id: Optional[str] = None,
    ) -> str:
        """
        Create execution history record when job is submitted.

        Captures all parameters needed for exact reproduction.

        Args:
            session_id: User session ID
            user_id: User identifier
            request: Complete TrainRequest dict
            data: Training data for hash computation
            data_upload_id: Optional reference to stored upload

        Returns:
            job_id: UUID string of created execution
        """
        client = await self._get_client()

        # Compute data hash for verification
        data_hash = compute_data_hash(data)

        # Extract key parameters
        time_col = request.get("time_col", "")
        target_col = request.get("target_col", "")
        horizon = request.get("horizon", 12)
        frequency = request.get("frequency", "weekly")
        models = request.get("models", ["prophet"])
        confidence_level = request.get("confidence_level", 0.95)
        random_seed = request.get("random_seed", 42)
        hyperparameter_filters = request.get("hyperparameter_filters", {})

        job_id = await client.create_execution(
            session_id=UUID(session_id),
            user_id=user_id,
            request_params=request,
            time_col=time_col,
            target_col=target_col,
            horizon=horizon,
            frequency=frequency,
            models=models,
            confidence_level=confidence_level,
            random_seed=random_seed,
            hyperparameter_filters=hyperparameter_filters,
            data_upload_id=UUID(data_upload_id) if data_upload_id else None,
            data_row_count=len(data),
            data_hash=data_hash,
        )

        # Audit log
        await client.log_audit(
            action="EXECUTION_CREATE",
            resource_type="execution",
            user_id=user_id,
            session_id=UUID(session_id),
            resource_id=job_id,
            request_data={
                "horizon": horizon,
                "frequency": frequency,
                "models": models,
                "data_rows": len(data),
            },
        )

        logger.info(f"Created execution {job_id} for user {user_id}")
        return str(job_id)

    @log_io
    async def get_execution(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get execution details by job ID.

        Args:
            job_id: Execution job ID

        Returns:
            Execution details dict or None if not found
        """
        client = await self._get_client()

        try:
            uuid_job_id = UUID(job_id)
        except ValueError:
            return None

        execution = await client.get_execution(uuid_job_id)

        if execution:
            # Convert UUIDs to strings
            execution["job_id"] = str(execution["job_id"])
            execution["session_id"] = str(execution["session_id"])
            if execution.get("data_upload_id"):
                execution["data_upload_id"] = str(execution["data_upload_id"])

        return execution

    @log_io
    async def update_status(
        self,
        job_id: str,
        status: str,
        progress_percent: Optional[int] = None,
        current_step: Optional[str] = None,
        error_message: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_experiment_url: Optional[str] = None,
        mlflow_run_url: Optional[str] = None,
        best_model: Optional[str] = None,
        best_mape: Optional[float] = None,
    ) -> None:
        """
        Update execution status and related fields.

        Args:
            job_id: Execution job ID
            status: New status (PENDING, QUEUED, RUNNING, COMPLETED, FAILED)
            progress_percent: Optional progress (0-100)
            current_step: Optional current step description
            error_message: Optional error message if failed
            mlflow_run_id: Optional MLflow run ID
            mlflow_experiment_url: Optional MLflow experiment URL
            mlflow_run_url: Optional MLflow run URL
            best_model: Optional best model name
            best_mape: Optional best MAPE score
        """
        client = await self._get_client()

        await client.update_execution_status(
            job_id=UUID(job_id),
            status=status,
            progress_percent=progress_percent,
            current_step=current_step,
            error_message=error_message,
            mlflow_run_id=mlflow_run_id,
            best_model=best_model,
            best_mape=best_mape,
        )

        # Update URLs separately if provided (not in base method)
        if mlflow_experiment_url or mlflow_run_url:
            await client.execute(
                """
                UPDATE forecast.execution_history
                SET mlflow_experiment_url = COALESCE($2, mlflow_experiment_url),
                    mlflow_run_url = COALESCE($3, mlflow_run_url)
                WHERE job_id = $1
                """,
                UUID(job_id),
                mlflow_experiment_url,
                mlflow_run_url,
            )

    @log_io
    async def save_result(
        self,
        job_id: str,
        model_name: str,
        forecast_dates: List[str],
        predictions: List[float],
        lower_bounds: Optional[List[float]] = None,
        upper_bounds: Optional[List[float]] = None,
        validation_dates: Optional[List[str]] = None,
        validation_actuals: Optional[List[float]] = None,
        validation_predictions: Optional[List[float]] = None,
        mape: Optional[float] = None,
        rmse: Optional[float] = None,
        mae: Optional[float] = None,
        r2: Optional[float] = None,
        cv_mape: Optional[float] = None,
        model_params: Optional[Dict] = None,
        training_time_seconds: Optional[float] = None,
        mlflow_run_id: Optional[str] = None,
    ) -> str:
        """
        Save forecast result for a model.

        Args:
            job_id: Parent execution job ID
            model_name: Name of the model (prophet, arima, etc.)
            forecast_dates: List of forecast date strings
            predictions: List of predicted values
            lower_bounds: Optional lower prediction interval bounds
            upper_bounds: Optional upper prediction interval bounds
            validation_dates: Optional validation date strings
            validation_actuals: Optional actual values for validation
            validation_predictions: Optional predicted values for validation
            mape: Optional MAPE score
            rmse: Optional RMSE score
            mae: Optional MAE score
            r2: Optional R² score
            cv_mape: Optional cross-validation MAPE
            model_params: Optional fitted model parameters
            training_time_seconds: Optional training duration
            mlflow_run_id: Optional MLflow run ID for this model

        Returns:
            result_id: UUID string of saved result
        """
        client = await self._get_client()

        result_id = await client.save_forecast_result(
            job_id=UUID(job_id),
            model_name=model_name,
            forecast_dates=forecast_dates,
            predictions=predictions,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            validation_dates=validation_dates,
            validation_actuals=validation_actuals,
            validation_predictions=validation_predictions,
            mape=mape,
            rmse=rmse,
            mae=mae,
            r2=r2,
            cv_mape=cv_mape,
            model_params=model_params,
            training_time_seconds=training_time_seconds,
            mlflow_run_id=mlflow_run_id,
        )

        return str(result_id)

    @log_io
    async def get_results(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Get all forecast results for a job.

        Args:
            job_id: Execution job ID

        Returns:
            List of result dicts ordered by MAPE (best first)
        """
        client = await self._get_client()

        results = await client.get_job_results(UUID(job_id))

        # Convert UUIDs to strings
        for result in results:
            result["result_id"] = str(result["result_id"])
            result["job_id"] = str(result["job_id"])

        return results

    @log_io
    async def finalize_execution(self, job_id: str) -> None:
        """
        Finalize execution after all models complete.

        Updates model rankings and execution summary.

        Args:
            job_id: Execution job ID
        """
        client = await self._get_client()

        # Update model rankings
        await client.update_model_rankings(UUID(job_id))

        # Get updated execution
        execution = await client.get_execution(UUID(job_id))

        # Log completion
        await client.log_audit(
            action="EXECUTION_COMPLETE",
            resource_type="execution",
            user_id=execution["user_id"] if execution else None,
            resource_id=UUID(job_id),
            response_data={
                "best_model": execution.get("best_model") if execution else None,
                "best_mape": execution.get("best_mape") if execution else None,
                "models_trained": execution.get("models_trained") if execution else None,
            },
        )

    @log_io
    async def get_reproduction_params(self, job_id: str) -> Dict[str, Any]:
        """
        Get all parameters needed to reproduce an execution.

        Returns complete TrainRequest with data reference for exact reproduction.

        Args:
            job_id: Execution job ID to reproduce

        Returns:
            Dict with original_request, data_upload_id, random_seed, etc.

        Raises:
            ValueError: If execution not found
        """
        client = await self._get_client()

        params = await client.get_reproduction_params(UUID(job_id))

        # Add helpful metadata
        params["reproduction_notes"] = (
            "To reproduce this execution exactly:\n"
            "1. Load data from data_upload_id (verify data_hash matches)\n"
            "2. Use request_params as TrainRequest\n"
            "3. Ensure random_seed is set to original value\n"
            "4. Results should match within floating-point tolerance"
        )

        return params

    @log_io
    async def verify_data_integrity(
        self,
        job_id: str,
        current_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify current data matches original execution data.

        Args:
            job_id: Execution job ID
            current_data: Data to verify against original

        Returns:
            Dict with matches (bool) and details
        """
        execution = await self.get_execution(job_id)
        if not execution:
            raise ValueError(f"Execution {job_id} not found")

        original_hash = execution.get("data_hash")
        current_hash = compute_data_hash(current_data)

        matches = original_hash == current_hash

        return {
            "matches": matches,
            "original_hash": original_hash,
            "current_hash": current_hash,
            "original_row_count": execution.get("data_row_count"),
            "current_row_count": len(current_data),
            "warning": None if matches else "Data has changed since original execution",
        }

    @log_io
    async def get_user_history(
        self,
        user_id: str,
        limit: int = 50,
        status_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get execution history for a user.

        Args:
            user_id: User identifier
            limit: Maximum records to return
            status_filter: Optional list of statuses to filter by

        Returns:
            List of execution summaries ordered by submission time
        """
        client = await self._get_client()

        history = await client.get_user_history(user_id, limit, status_filter)

        # Convert UUIDs
        for record in history:
            record["job_id"] = str(record["job_id"])

        return history

    @log_io
    async def compare_executions(
        self,
        job_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple executions.

        Useful for comparing different model configurations or data versions.

        Args:
            job_ids: List of job IDs to compare

        Returns:
            Dict with comparison summary and detailed breakdown
        """
        executions = []
        all_results = []

        for job_id in job_ids:
            execution = await self.get_execution(job_id)
            results = await self.get_results(job_id)

            if execution:
                executions.append(execution)
                all_results.append({
                    "job_id": job_id,
                    "results": results,
                })

        if not executions:
            return {"error": "No executions found"}

        # Build comparison
        comparison = {
            "execution_count": len(executions),
            "executions": [
                {
                    "job_id": e["job_id"],
                    "submitted_at": e["submitted_at"].isoformat() if e.get("submitted_at") else None,
                    "status": e["status"],
                    "best_model": e.get("best_model"),
                    "best_mape": e.get("best_mape"),
                    "horizon": e["horizon"],
                    "frequency": e["frequency"],
                    "models": e["models"],
                    "data_row_count": e.get("data_row_count"),
                }
                for e in executions
            ],
            "best_overall": None,
            "model_comparison": {},
        }

        # Find best overall
        completed = [e for e in executions if e.get("best_mape") is not None]
        if completed:
            best = min(completed, key=lambda x: x["best_mape"])
            comparison["best_overall"] = {
                "job_id": best["job_id"],
                "model": best["best_model"],
                "mape": best["best_mape"],
            }

        # Compare models across executions
        model_scores = {}
        for job_result in all_results:
            for result in job_result["results"]:
                model = result["model_name"]
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append({
                    "job_id": job_result["job_id"],
                    "mape": result.get("mape"),
                    "rmse": result.get("rmse"),
                })

        comparison["model_comparison"] = model_scores

        return comparison

    @log_io
    async def get_model_performance_trend(
        self,
        user_id: str,
        model_name: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get performance trend for a specific model over time.

        Args:
            user_id: User identifier
            model_name: Model to track (prophet, arima, etc.)
            limit: Number of recent executions to include

        Returns:
            List of performance records ordered by time
        """
        client = await self._get_client()

        trend = await client.fetch(
            """
            SELECT
                eh.job_id,
                eh.submitted_at,
                fr.mape,
                fr.rmse,
                fr.cv_mape,
                fr.is_best_model,
                fr.training_time_seconds
            FROM forecast.execution_history eh
            JOIN forecast.forecast_results fr ON eh.job_id = fr.job_id
            WHERE eh.user_id = $1
              AND fr.model_name = $2
              AND eh.status = 'COMPLETED'
            ORDER BY eh.submitted_at DESC
            LIMIT $3
            """,
            user_id,
            model_name,
            limit,
        )

        for record in trend:
            record["job_id"] = str(record["job_id"])

        return trend


# Singleton instance
_history_service: Optional[HistoryService] = None


@log_io
async def get_history_service() -> HistoryService:
    """Get or create singleton HistoryService."""
    global _history_service
    if _history_service is None:
        _history_service = HistoryService()
    return _history_service
