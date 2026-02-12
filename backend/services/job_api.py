"""
Job API endpoints for training job management.

Provides REST endpoints for job creation, submission, status, and cancellation.
"""

import logging
import os
from enum import Enum
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.utils.logging_utils import log_io

from .job_delegation import get_job_service, is_delegation_enabled
from .job_state_store import JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/jobs", tags=["jobs"])


# Request/Response Models
class TrainingMode(str, Enum):
    """Supported distributed training modes."""
    AUTOGLUON = "autogluon"
    STATSFORECAST = "statsforecast"
    NEURALFORECAST = "neuralforecast"
    MMF = "mmf"
    LEGACY = "legacy"


class JobConfig(BaseModel):
    """Training job configuration."""
    data: List[dict] = Field(..., description="Training data")
    time_col: str = Field(..., description="Time column name")
    target_col: str = Field(..., description="Target column name")
    id_col: Optional[str] = Field(None, description="Series ID column for multi-series")
    covariates: List[str] = Field(default_factory=list, description="Covariate columns")
    horizon: int = Field(default=12, description="Forecast horizon")
    frequency: str = Field(default="W", description="Data frequency (D, W, M)")
    training_mode: TrainingMode = Field(default=TrainingMode.AUTOGLUON, description="AutoML framework to use")
    models: List[str] = Field(default=["prophet"], description="Models to train (legacy mode)")
    seasonality_mode: str = Field(default="multiplicative", description="Seasonality mode")
    time_limit: int = Field(default=600, description="Training time limit in seconds")
    presets: str = Field(default="medium_quality", description="AutoGluon presets")
    season_length: Optional[int] = Field(None, description="Seasonal period length")


class CreateJobRequest(BaseModel):
    """Request to create a new training job."""
    config: JobConfig


class JobResponse(BaseModel):
    """Training job response."""
    job_id: str
    status: str
    progress: int
    current_step: str
    run_id: Optional[str] = None
    created_at: Optional[str] = None
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    results: Optional[dict] = None
    error: Optional[str] = None


class JobListResponse(BaseModel):
    """List of jobs response."""
    jobs: List[JobResponse]
    total: int


@router.get("/delegation-status")
async def get_delegation_status():
    """Check if cluster delegation is enabled and configured."""
    return {
        "enabled": is_delegation_enabled(),
        "cluster_id": os.getenv("DEDICATED_CLUSTER_ID"),
        "message": "Cluster delegation is enabled" if is_delegation_enabled() else "Cluster delegation is disabled",
        "training_modes": [mode.value for mode in TrainingMode],
    }


@router.get("/training-modes")
async def get_training_modes():
    """Get available training modes with descriptions."""
    return {
        "modes": [
            {
                "value": TrainingMode.AUTOGLUON.value,
                "name": "AutoGluon-TimeSeries",
                "description": "Best accuracy with automatic ensembling. Includes Chronos foundation model.",
                "speed": "medium",
                "recommended": True,
            },
            {
                "value": TrainingMode.STATSFORECAST.value,
                "name": "StatsForecast (Nixtla)",
                "description": "Lightning-fast statistical models. 500x faster than Prophet.",
                "speed": "fast",
                "recommended": False,
            },
            {
                "value": TrainingMode.NEURALFORECAST.value,
                "name": "NeuralForecast (Nixtla)",
                "description": "Deep learning models (NHITS, NBEATS, TFT). Best for complex patterns.",
                "speed": "slow",
                "recommended": False,
            },
            {
                "value": TrainingMode.MMF.value,
                "name": "Many Model Forecasting",
                "description": "Databricks solution with 40+ models. Best for production scale.",
                "speed": "variable",
                "recommended": False,
            },
            {
                "value": TrainingMode.LEGACY.value,
                "name": "Legacy (Prophet/ARIMA)",
                "description": "Original implementation. Runs in App container.",
                "speed": "medium",
                "recommended": False,
            },
        ]
    }


@router.post("", response_model=JobResponse)
async def create_job(request: CreateJobRequest):
    """
    Create a new training job (does not start it).

    Returns the job ID for subsequent operations.
    """
    if not is_delegation_enabled():
        raise HTTPException(
            status_code=503,
            detail="Cluster delegation is not enabled. Set ENABLE_CLUSTER_DELEGATION=true"
        )

    try:
        service = get_job_service()
        job = await service.create_job(
            user_id="default_user",  # TODO: Get from auth context
            config=request.config.model_dump()
        )
        return _job_to_response(job)
    except Exception as e:
        logger.error(f"Failed to create job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{job_id}/submit", response_model=JobResponse)
async def submit_job(job_id: str):
    """
    Submit a pending job to the Databricks cluster.

    The job will start executing on the dedicated cluster.
    """
    if not is_delegation_enabled():
        raise HTTPException(
            status_code=503,
            detail="Cluster delegation is not enabled"
        )

    try:
        service = get_job_service()
        job = await service.submit_job(job_id)
        return _job_to_response(job)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to submit job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """
    Get the current status of a training job.

    For running jobs, status is synced with Databricks.
    """
    try:
        service = get_job_service()
        job = await service.get_job_status(job_id)
        return _job_to_response(job)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(job_id: str):
    """
    Cancel a running or pending job.

    The job will be terminated on the cluster.
    """
    try:
        service = get_job_service()
        job = await service.cancel_job(job_id)
        return _job_to_response(job)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job from the state store.

    Cannot delete running jobs - cancel them first.
    """
    try:
        service = get_job_service()
        await service.delete_job(job_id)
        return {"message": f"Job {job_id} deleted"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of jobs to return")
):
    """
    List training jobs with optional filters.
    """
    try:
        service = get_job_service()
        job_status = JobStatus(status) if status else None
        jobs = await service.list_jobs(
            user_id=None,  # TODO: Get from auth context
            status=job_status,
            limit=limit
        )
        return JobListResponse(
            jobs=[_job_to_response(job) for job in jobs],
            total=len(jobs)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}/results")
async def get_job_results(job_id: str):
    """
    Get the results of a completed job.

    Returns training metrics, forecasts, and MLflow run information.
    """
    try:
        service = get_job_service()
        job = await service.get_job_status(job_id)

        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Job is not completed (status: {job.status.value})"
            )

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "mlflow_run_id": job.mlflow_run_id,
            "results": job.results or {},
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get results for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@log_io
def _job_to_response(job) -> JobResponse:
    """Convert TrainingJob to JobResponse."""
    return JobResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        current_step=job.current_step,
        run_id=job.run_id,
        created_at=job.created_at.isoformat() if job.created_at else None,
        submitted_at=job.submitted_at.isoformat() if job.submitted_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        results=job.results,
        error=job.error
    )
