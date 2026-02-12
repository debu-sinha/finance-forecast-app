"""
Job Service for Finance Forecasting Platform.

Handles Databricks Jobs submission and tracking for distributed training.
Implements the recommended patterns from Databricks Jobs API:
- Job queueing (48-hour queue time)
- max_concurrent_runs for controlled parallelism
- Dedicated cluster for heavy ML training workloads

References:
- https://docs.databricks.com/aws/en/jobs/
- https://docs.databricks.com/aws/en/jobs/advanced (queueing)
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from backend.utils.logging_utils import log_io

try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service import jobs
    from databricks.sdk.service.jobs import (
        RunLifeCycleState,
        RunResultState,
        RunState,
    )
    DATABRICKS_SDK_AVAILABLE = True
except ImportError:
    DATABRICKS_SDK_AVAILABLE = False
    WorkspaceClient = None
    jobs = None

from backend.services.lakebase_client import LakebaseClient, get_lakebase_client

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class JobConfig:
    """Configuration for training job submission."""

    # Cluster configuration
    cluster_id: str  # Dedicated cluster ID
    notebook_path: str = "/Workspace/finance-forecasting/notebooks/train_models"

    # Job settings
    timeout_seconds: int = 7200  # 2 hours max
    max_retries: int = 1
    retry_on_timeout: bool = True

    # Queue settings (Databricks Jobs feature)
    enable_queueing: bool = True
    max_queue_duration_seconds: int = 172800  # 48 hours

    # Concurrency
    max_concurrent_runs: int = 10  # Per-job concurrency limit

    @classmethod
    def from_env(cls) -> "JobConfig":
        """Create config from environment variables."""
        return cls(
            cluster_id=os.getenv("DEDICATED_CLUSTER_ID", ""),
            notebook_path=os.getenv(
                "TRAINING_NOTEBOOK_PATH",
                "/Workspace/finance-forecasting/notebooks/train_models"
            ),
            timeout_seconds=int(os.getenv("JOB_TIMEOUT_SECONDS", "7200")),
            max_retries=int(os.getenv("JOB_MAX_RETRIES", "1")),
            max_concurrent_runs=int(os.getenv("JOB_MAX_CONCURRENT_RUNS", "10")),
        )


class JobService:
    """
    Manages Databricks Jobs submission and tracking.

    Implements distributed job pattern:
    1. App receives training request
    2. JobService submits to Databricks Jobs (dedicated cluster)
    3. App polls for status via Lakebase
    4. Job writes results to Lakebase on completion

    Usage:
        service = JobService()

        # Submit job
        run_id = await service.submit_training_job(
            job_id=job_id,
            session_id=session_id,
            user_id=user_id,
            data_upload_id=data_upload_id,
            train_request=request.dict(),
        )

        # Poll status
        status = await service.get_job_status(run_id)
    """

    def __init__(
        self,
        config: Optional[JobConfig] = None,
        client: Optional[LakebaseClient] = None
    ):
        """
        Initialize JobService.

        Args:
            config: Job configuration. If None, loads from environment.
            client: Optional LakebaseClient for state persistence.
        """
        self.config = config or JobConfig.from_env()
        self._lakebase = client
        self._workspace_client: Optional[WorkspaceClient] = None

        if not DATABRICKS_SDK_AVAILABLE:
            logger.warning("Databricks SDK not available - job submission disabled")

    @log_io
    def _get_workspace_client(self) -> WorkspaceClient:
        """Get or create Databricks WorkspaceClient."""
        if self._workspace_client is None:
            if not DATABRICKS_SDK_AVAILABLE:
                raise RuntimeError("Databricks SDK not installed")

            # WorkspaceClient auto-configures from environment or Databricks Apps context
            self._workspace_client = WorkspaceClient()

        return self._workspace_client

    @log_io
    async def _get_lakebase(self) -> LakebaseClient:
        """Get Lakebase client."""
        if self._lakebase is None:
            self._lakebase = await get_lakebase_client()
        return self._lakebase

    @log_io
    async def submit_training_job(
        self,
        job_id: UUID,
        session_id: UUID,
        user_id: str,
        data_upload_id: UUID,
        train_request: Dict[str, Any],
        priority: str = "NORMAL",
    ) -> int:
        """
        Submit training job to dedicated Databricks cluster.

        Args:
            job_id: UUID for this execution (created by caller)
            session_id: User session ID
            user_id: User identifier
            data_upload_id: Reference to uploaded data
            train_request: Complete TrainRequest dict for reproducibility
            priority: Queue priority (LOW, NORMAL, HIGH, URGENT)

        Returns:
            run_id: Databricks run ID for tracking

        Raises:
            RuntimeError: If SDK not available or submission fails
        """
        if not DATABRICKS_SDK_AVAILABLE:
            raise RuntimeError("Databricks SDK not available")

        if not self.config.cluster_id:
            raise RuntimeError("DEDICATED_CLUSTER_ID not configured")

        workspace = self._get_workspace_client()
        lakebase = await self._get_lakebase()

        # Serialize request for reproducibility
        request_json = json.dumps(train_request, default=str)

        # Build job parameters
        parameters = {
            "job_id": str(job_id),
            "session_id": str(session_id),
            "user_id": user_id,
            "data_upload_id": str(data_upload_id),
            "request_json": request_json,
            "lakebase_host": os.getenv("LAKEBASE_HOST", ""),
            "lakebase_database": os.getenv("LAKEBASE_DATABASE", "forecast"),
        }

        # Submit run
        try:
            run = workspace.jobs.submit(
                run_name=f"forecast_{str(job_id)[:8]}_{user_id}",
                tasks=[
                    jobs.SubmitTask(
                        task_key="train_all_models",
                        existing_cluster_id=self.config.cluster_id,
                        notebook_task=jobs.NotebookTask(
                            notebook_path=self.config.notebook_path,
                            base_parameters=parameters,
                        ),
                        timeout_seconds=self.config.timeout_seconds,
                    )
                ],
                queue=jobs.QueueSettings(enabled=self.config.enable_queueing),
            )

            run_id = run.run_id
            logger.info(f"Submitted job {job_id} as Databricks run {run_id}")

            # Update execution history with run ID
            await lakebase.update_execution_status(
                job_id=job_id,
                status="QUEUED",
                databricks_run_id=run_id,
            )

            # Add to job queue
            await lakebase.enqueue_job(
                job_id=job_id,
                user_id=user_id,
                priority=priority,
            )

            # Log audit event
            await lakebase.log_audit(
                action="JOB_SUBMIT",
                resource_type="execution",
                user_id=user_id,
                session_id=session_id,
                resource_id=job_id,
                request_data={"databricks_run_id": run_id},
            )

            return run_id

        except Exception as e:
            logger.error(f"Failed to submit job {job_id}: {e}")

            # Update status to FAILED
            await lakebase.update_execution_status(
                job_id=job_id,
                status="FAILED",
                error_message=str(e),
            )

            raise

    @log_io
    async def get_job_status(self, run_id: int) -> Dict[str, Any]:
        """
        Get job execution status from Databricks.

        Args:
            run_id: Databricks run ID

        Returns:
            Dict with state, result_state, start_time, end_time, run_page_url
        """
        if not DATABRICKS_SDK_AVAILABLE:
            return {"state": "UNKNOWN", "error": "SDK not available"}

        workspace = self._get_workspace_client()

        try:
            run = workspace.jobs.get_run(run_id=run_id)

            state = run.state.life_cycle_state.value if run.state.life_cycle_state else "UNKNOWN"
            result_state = run.state.result_state.value if run.state.result_state else None

            return {
                "run_id": run_id,
                "state": state,
                "result_state": result_state,
                "state_message": run.state.state_message if run.state else None,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "run_page_url": run.run_page_url,
                "setup_duration_ms": run.setup_duration,
                "execution_duration_ms": run.execution_duration,
            }
        except Exception as e:
            logger.error(f"Failed to get status for run {run_id}: {e}")
            return {"run_id": run_id, "state": "UNKNOWN", "error": str(e)}

    @log_io
    async def sync_job_status(self, job_id: UUID) -> Dict[str, Any]:
        """
        Sync job status from Databricks to Lakebase.

        Checks Databricks run status and updates Lakebase if changed.

        Args:
            job_id: Execution job ID

        Returns:
            Updated execution details
        """
        lakebase = await self._get_lakebase()

        # Get execution record
        execution = await lakebase.get_execution(job_id)
        if not execution:
            raise ValueError(f"Execution {job_id} not found")

        # If already terminal, no need to sync
        if execution["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return execution

        # Check Databricks status
        run_id = execution.get("databricks_run_id")
        if not run_id:
            return execution

        db_status = await self.get_job_status(run_id)

        # Map Databricks states to our states
        new_status = self._map_databricks_state(
            db_status["state"],
            db_status.get("result_state")
        )

        # Update if changed
        if new_status != execution["status"]:
            error_message = None
            if new_status == "FAILED":
                error_message = db_status.get("state_message", "Job failed")

            await lakebase.update_execution_status(
                job_id=job_id,
                status=new_status,
                error_message=error_message,
            )

            # Dequeue if terminal
            if new_status in ("COMPLETED", "FAILED", "CANCELLED"):
                await lakebase.dequeue_job(job_id)

            logger.info(f"Job {job_id} status updated: {execution['status']} -> {new_status}")

        # Return updated execution
        return await lakebase.get_execution(job_id)

    @log_io
    def _map_databricks_state(
        self,
        lifecycle_state: str,
        result_state: Optional[str]
    ) -> str:
        """Map Databricks run states to our JobStatus."""
        # Terminal states
        if lifecycle_state == "TERMINATED":
            if result_state == "SUCCESS":
                return JobStatus.COMPLETED.value
            elif result_state in ("FAILED", "TIMEDOUT"):
                return JobStatus.FAILED.value
            elif result_state == "CANCELED":
                return JobStatus.CANCELLED.value
            else:
                return JobStatus.FAILED.value

        # Active states
        if lifecycle_state in ("PENDING", "QUEUED"):
            return JobStatus.QUEUED.value
        elif lifecycle_state == "RUNNING":
            return JobStatus.RUNNING.value
        elif lifecycle_state in ("TERMINATING", "SKIPPED", "INTERNAL_ERROR"):
            return JobStatus.FAILED.value

        return JobStatus.PENDING.value

    @log_io
    async def cancel_job(self, job_id: UUID, user_id: str) -> bool:
        """
        Cancel a running or queued job.

        Args:
            job_id: Execution job ID
            user_id: User requesting cancellation (for audit)

        Returns:
            True if cancelled successfully
        """
        lakebase = await self._get_lakebase()

        execution = await lakebase.get_execution(job_id)
        if not execution:
            raise ValueError(f"Execution {job_id} not found")

        # Can only cancel pending/queued/running jobs
        if execution["status"] not in ("PENDING", "QUEUED", "RUNNING"):
            return False

        # Cancel in Databricks
        run_id = execution.get("databricks_run_id")
        if run_id and DATABRICKS_SDK_AVAILABLE:
            try:
                workspace = self._get_workspace_client()
                workspace.jobs.cancel_run(run_id=run_id)
                logger.info(f"Cancelled Databricks run {run_id}")
            except Exception as e:
                logger.warning(f"Failed to cancel Databricks run {run_id}: {e}")

        # Update Lakebase
        await lakebase.update_execution_status(
            job_id=job_id,
            status="CANCELLED",
        )
        await lakebase.dequeue_job(job_id)

        # Audit
        await lakebase.log_audit(
            action="JOB_CANCEL",
            resource_type="execution",
            user_id=user_id,
            resource_id=job_id,
        )

        return True

    @log_io
    async def get_user_active_jobs(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get active (running/pending/queued) jobs for a user.

        Args:
            user_id: User identifier

        Returns:
            List of active job details
        """
        lakebase = await self._get_lakebase()
        return await lakebase.get_active_jobs(user_id)

    @log_io
    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get job queue statistics.

        Returns:
            Dict with queue length, estimated wait time, etc.
        """
        lakebase = await self._get_lakebase()

        stats = await lakebase.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE is_active) as queue_length,
                COUNT(*) FILTER (WHERE is_active AND priority = 'URGENT') as urgent_count,
                COUNT(*) FILTER (WHERE is_active AND priority = 'HIGH') as high_count,
                AVG(EXTRACT(EPOCH FROM (NOW() - enqueued_at)))
                    FILTER (WHERE is_active) as avg_wait_seconds,
                MAX(EXTRACT(EPOCH FROM (NOW() - enqueued_at)))
                    FILTER (WHERE is_active) as max_wait_seconds
            FROM forecast.job_queue
            """
        )

        return dict(stats) if stats else {}

    @log_io
    async def estimate_queue_position(self, job_id: UUID) -> Dict[str, Any]:
        """
        Estimate queue position and wait time for a job.

        Args:
            job_id: Execution job ID

        Returns:
            Dict with position, estimated_wait_seconds
        """
        lakebase = await self._get_lakebase()

        position = await lakebase.get_queue_position(job_id)

        # Estimate based on average job duration
        avg_duration = await lakebase.fetchval(
            """
            SELECT AVG(duration_seconds)
            FROM forecast.execution_history
            WHERE status = 'COMPLETED'
              AND completed_at > NOW() - INTERVAL '24 hours'
            """
        )

        estimated_wait = (position or 0) * (avg_duration or 300)  # Default 5 min

        return {
            "position": position,
            "estimated_wait_seconds": int(estimated_wait),
        }


# Singleton instance
_job_service: Optional[JobService] = None


@log_io
async def get_job_service() -> JobService:
    """Get or create singleton JobService."""
    global _job_service
    if _job_service is None:
        _job_service = JobService()
    return _job_service
