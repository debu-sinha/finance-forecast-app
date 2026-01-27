"""
Job Delegation Service - Delegates training to Databricks clusters.

Handles job submission, status tracking, cancellation, and result retrieval.
"""

import os
import json
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List

from .job_state_store import JobStateStore, TrainingJob, JobStatus

logger = logging.getLogger(__name__)

# Environment configuration
DEDICATED_CLUSTER_ID = os.getenv("DEDICATED_CLUSTER_ID")
ENABLE_CLUSTER_DELEGATION = os.getenv("ENABLE_CLUSTER_DELEGATION", "false").lower() == "true"

# Concurrency limits
MAX_CONCURRENT_JOBS_PER_USER = int(os.getenv("MAX_JOBS_PER_USER", "3"))
MAX_TOTAL_CONCURRENT_JOBS = int(os.getenv("MAX_TOTAL_JOBS", "10"))


class TrainingMode(str, Enum):
    """Supported distributed training modes."""
    AUTOGLUON = "autogluon"
    STATSFORECAST = "statsforecast"
    NEURALFORECAST = "neuralforecast"
    MMF = "mmf"
    LEGACY = "legacy"


# Notebook paths per training mode
NOTEBOOK_PATHS = {
    TrainingMode.AUTOGLUON: os.getenv(
        "AUTOGLUON_NOTEBOOK_PATH",
        "/Workspace/Shared/finance-forecast/notebooks/train_distributed"
    ),
    TrainingMode.STATSFORECAST: os.getenv(
        "STATSFORECAST_NOTEBOOK_PATH",
        "/Workspace/Shared/finance-forecast/notebooks/train_distributed"
    ),
    TrainingMode.NEURALFORECAST: os.getenv(
        "NEURALFORECAST_NOTEBOOK_PATH",
        "/Workspace/Shared/finance-forecast/notebooks/train_distributed"
    ),
    TrainingMode.MMF: os.getenv(
        "MMF_NOTEBOOK_PATH",
        "/Workspace/Shared/finance-forecast/notebooks/train_mmf"
    ),
    TrainingMode.LEGACY: os.getenv(
        "LEGACY_NOTEBOOK_PATH",
        "/Workspace/Shared/finance-forecast/notebooks/train_legacy"
    ),
}


class JobDelegationService:
    """
    Manages the lifecycle of training jobs delegated to Databricks clusters.

    Responsibilities:
    - Submit jobs to dedicated cluster via Databricks SDK
    - Track job status and progress
    - Handle cancellation
    - Persist state for recovery
    - Retrieve results from completed jobs
    """

    def __init__(self, state_store: JobStateStore):
        self.state_store = state_store
        self._workspace_client = None

    @property
    def workspace_client(self):
        """Lazy-load Databricks WorkspaceClient."""
        if self._workspace_client is None:
            try:
                from databricks.sdk import WorkspaceClient
                self._workspace_client = WorkspaceClient()
                logger.info("Databricks WorkspaceClient initialized")
            except Exception as e:
                logger.error(f"Failed to initialize WorkspaceClient: {e}")
                raise
        return self._workspace_client

    async def create_job(
        self,
        user_id: str,
        config: Dict[str, Any]
    ) -> TrainingJob:
        """
        Create a new training job (but don't submit yet).
        Returns the job for user review before starting.

        Enforces concurrency limits per user and globally.
        """
        # Check user concurrency limit
        user_running_jobs = await self.state_store.count_running_jobs(user_id=user_id)
        if user_running_jobs >= MAX_CONCURRENT_JOBS_PER_USER:
            raise ValueError(
                f"Maximum concurrent jobs ({MAX_CONCURRENT_JOBS_PER_USER}) reached for user. "
                "Please wait for existing jobs to complete."
            )

        # Check global concurrency limit
        total_running_jobs = await self.state_store.count_running_jobs()
        if total_running_jobs >= MAX_TOTAL_CONCURRENT_JOBS:
            raise ValueError(
                "System is at maximum capacity. Please try again later."
            )

        # Validate training mode
        training_mode = config.get("training_mode", "autogluon")
        if training_mode not in [m.value for m in TrainingMode]:
            raise ValueError(f"Invalid training mode: {training_mode}")

        job = TrainingJob(
            job_id=str(uuid.uuid4()),
            user_id=user_id,
            config=config,
            cluster_id=DEDICATED_CLUSTER_ID,
            status=JobStatus.PENDING,
            progress=0,
            current_step="Job created, ready to submit",
            created_at=datetime.now(timezone.utc)
        )

        await self.state_store.save(job)
        logger.info(f"Created job {job.job_id} for user {user_id} with mode {training_mode}")
        return job

    async def submit_job(self, job_id: str) -> TrainingJob:
        """
        Submit a pending job to the Databricks cluster.
        """
        job = await self.state_store.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status != JobStatus.PENDING:
            raise ValueError(f"Job {job_id} is not in PENDING state (current: {job.status.value})")

        if not DEDICATED_CLUSTER_ID:
            raise ValueError("DEDICATED_CLUSTER_ID not configured")

        job.status = JobStatus.SUBMITTING
        job.current_step = "Submitting to cluster..."
        await self.state_store.save(job)

        try:
            from databricks.sdk.service import jobs

            # Create the job run on the dedicated cluster
            run = self.workspace_client.jobs.submit(
                run_name=f"forecast_{job.job_id[:8]}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                tasks=[
                    jobs.SubmitTask(
                        task_key="training",
                        existing_cluster_id=DEDICATED_CLUSTER_ID,
                        python_wheel_task=None,
                        notebook_task=jobs.NotebookTask(
                            notebook_path=self._get_training_notebook_path(
                                job.config.get("training_mode", "autogluon")
                            ),
                            base_parameters={
                                "job_id": job.job_id,
                                "config": json.dumps(job.config)
                            }
                        ),
                        timeout_seconds=3600  # 1 hour max
                    )
                ]
            )

            # Wait for submission confirmation
            run_result = run.result()
            job.run_id = str(run_result.run_id)
            job.status = JobStatus.RUNNING
            job.submitted_at = datetime.now(timezone.utc)
            job.current_step = "Training started on cluster"
            job.progress = 5

            logger.info(f"Job {job.job_id} submitted successfully, run_id: {job.run_id}")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.current_step = f"Submission failed: {str(e)[:100]}"
            job.completed_at = datetime.now(timezone.utc)
            logger.error(f"Job {job.job_id} submission failed: {e}")

        await self.state_store.save(job)
        return job

    async def get_job_status(self, job_id: str) -> TrainingJob:
        """
        Get current job status, syncing with Databricks if running.
        """
        job = await self.state_store.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Sync with Databricks for running jobs
        if job.status == JobStatus.RUNNING and job.run_id:
            try:
                await self._sync_job_status(job)
            except Exception as e:
                logger.warning(f"Failed to sync job status for {job_id}: {e}")

        return job

    async def _sync_job_status(self, job: TrainingJob) -> None:
        """Sync job status with Databricks."""
        from databricks.sdk.service.jobs import RunLifeCycleState, RunResultState

        run = self.workspace_client.jobs.get_run(int(job.run_id))

        # Update progress based on run state
        if run.state.life_cycle_state == RunLifeCycleState.RUNNING:
            # Estimate progress based on time (simple heuristic)
            if job.submitted_at:
                elapsed = (datetime.now(timezone.utc) - job.submitted_at).total_seconds()
                # Assume ~5 minutes average training time
                estimated_progress = min(int(5 + (elapsed / 300) * 85), 90)
                job.progress = estimated_progress
                job.current_step = "Training in progress..."

        elif run.state.life_cycle_state == RunLifeCycleState.TERMINATED:
            job.completed_at = datetime.now(timezone.utc)

            if run.state.result_state == RunResultState.SUCCESS:
                job.status = JobStatus.COMPLETED
                job.progress = 100
                job.current_step = "Training completed successfully"

                # Fetch results
                try:
                    job.results = await self._fetch_results(job, run)
                except Exception as e:
                    logger.warning(f"Failed to fetch results for {job.job_id}: {e}")

            elif run.state.result_state == RunResultState.CANCELED:
                job.status = JobStatus.CANCELLED
                job.current_step = "Job was cancelled"

            else:
                job.status = JobStatus.FAILED
                job.error = run.state.state_message or "Unknown error"
                job.current_step = f"Training failed: {job.error[:50]}"

            await self.state_store.save(job)

        elif run.state.life_cycle_state in [RunLifeCycleState.PENDING, RunLifeCycleState.QUEUED]:
            job.current_step = "Waiting for cluster resources..."

    async def _fetch_results(self, job: TrainingJob, run) -> Dict[str, Any]:
        """Fetch training results from the completed run."""
        results = {
            "run_id": job.run_id,
            "duration_seconds": (job.completed_at - job.submitted_at).total_seconds() if job.completed_at and job.submitted_at else None
        }

        # Try to get task output with MLflow run ID
        try:
            for task in run.tasks:
                if task.state and task.state.result_state:
                    # Task values might contain MLflow run ID
                    task_run = self.workspace_client.jobs.get_run_output(task.run_id)
                    if task_run.notebook_output and task_run.notebook_output.result:
                        output = json.loads(task_run.notebook_output.result)
                        results.update(output)
        except Exception as e:
            logger.warning(f"Could not parse task output: {e}")

        return results

    async def cancel_job(self, job_id: str) -> TrainingJob:
        """Cancel a running or pending job."""
        job = await self.state_store.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status not in [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SUBMITTING]:
            raise ValueError(f"Cannot cancel job in {job.status.value} state")

        job.status = JobStatus.CANCELLING
        job.current_step = "Cancelling..."
        await self.state_store.save(job)

        if job.run_id:
            try:
                self.workspace_client.jobs.cancel_run(int(job.run_id))
                logger.info(f"Cancelled Databricks run {job.run_id}")
            except Exception as e:
                logger.warning(f"Failed to cancel Databricks run: {e}")

        job.status = JobStatus.CANCELLED
        job.current_step = "Cancelled by user"
        job.completed_at = datetime.now(timezone.utc)
        await self.state_store.save(job)

        logger.info(f"Job {job.job_id} cancelled")
        return job

    async def list_jobs(
        self,
        user_id: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50
    ) -> List[TrainingJob]:
        """List jobs with optional filters."""
        return await self.state_store.list_jobs(
            user_id=user_id,
            status=status,
            limit=limit
        )

    async def delete_job(self, job_id: str) -> None:
        """Delete a job from the state store."""
        job = await self.state_store.get(job_id)
        if job and job.status in [JobStatus.RUNNING, JobStatus.SUBMITTING]:
            raise ValueError("Cannot delete a running job. Cancel it first.")

        await self.state_store.delete(job_id)
        logger.info(f"Deleted job {job_id}")

    async def sync_running_jobs(self) -> None:
        """
        Sync status of all running jobs with Databricks.
        Call this periodically or on app startup.
        """
        running_jobs = await self.state_store.get_running_jobs()
        for job in running_jobs:
            try:
                await self._sync_job_status(job)
            except Exception as e:
                logger.error(f"Failed to sync job {job.job_id}: {e}")

    def _get_training_notebook_path(self, training_mode: str = "autogluon") -> str:
        """Get the path to the training notebook based on training mode."""
        try:
            mode = TrainingMode(training_mode)
        except ValueError:
            mode = TrainingMode.AUTOGLUON

        return NOTEBOOK_PATHS.get(mode, NOTEBOOK_PATHS[TrainingMode.AUTOGLUON])


# Singleton instance management
_job_service: Optional[JobDelegationService] = None


def get_job_service() -> JobDelegationService:
    """Get or create the singleton JobDelegationService instance."""
    global _job_service
    if _job_service is None:
        from .job_state_store import SQLiteJobStateStore
        state_store = SQLiteJobStateStore(
            db_path=os.getenv("JOB_STATE_DB_PATH", "training_jobs.db")
        )
        _job_service = JobDelegationService(state_store)
    return _job_service


def is_delegation_enabled() -> bool:
    """Check if cluster delegation is enabled."""
    return ENABLE_CLUSTER_DELEGATION and DEDICATED_CLUSTER_ID is not None
