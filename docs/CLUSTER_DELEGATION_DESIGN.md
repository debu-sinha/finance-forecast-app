# Cluster Delegation Architecture Design

> **Version:** 1.0.0
> **Status:** Design
> **Author:** Debu Sinha
> **Date:** January 2026

---

## Executive Summary

This document describes the architecture for delegating compute-intensive forecasting workloads from the Databricks App (4 vCPU/12GB RAM limit) to on-demand clusters. The design prioritizes:

1. **State Persistence** - No data loss on page refresh
2. **Cancellation Support** - Users can cancel running jobs
3. **Resume Capability** - Continue processing after interruption
4. **Scalability** - Handle concurrent users and large workloads
5. **Cost Optimization** - Use spot instances, auto-terminate clusters

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Design](#2-component-design)
3. [State Management](#3-state-management)
4. [Job Lifecycle](#4-job-lifecycle)
5. [API Contracts](#5-api-contracts)
6. [Data Flow](#6-data-flow)
7. [Error Handling](#7-error-handling)
8. [Frontend Integration](#8-frontend-integration)
9. [Local Development](#9-local-development)
10. [Security Considerations](#10-security-considerations)
11. [Cost Optimization](#11-cost-optimization)
12. [Migration Path](#12-migration-path)

---

## 1. Architecture Overview

### Current Architecture (Blocking)

```
┌─────────────────────────────────────────────────────────────┐
│                    Databricks App (4 vCPU)                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │   FastAPI   │───▶│   Training  │───▶│     MLflow      │ │
│  │   Backend   │    │   (Inline)  │    │    Tracking     │ │
│  └─────────────┘    └─────────────┘    └─────────────────┘ │
│         ▲                                                   │
│         │                                                   │
│  ┌─────────────┐                                           │
│  │   React UI  │  ← Blocks until training completes        │
│  └─────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

**Problems:**
- Training blocks the app (unresponsive UI)
- Limited to 4 vCPU (slow for large datasets)
- Page refresh loses all progress
- Cannot scale for multiple concurrent users

### Proposed Architecture (Async Delegation)

```
┌─────────────────────────────────────────────────────────────┐
│                    Databricks App (4 vCPU)                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │   FastAPI   │───▶│    Job      │───▶│   State Store   │ │
│  │   Backend   │    │  Delegation │    │   (SQLite/UC)   │ │
│  └─────────────┘    │   Service   │    └─────────────────┘ │
│         ▲           └──────┬──────┘                        │
│         │                  │                               │
│  ┌─────────────┐           │ Submit Job                    │
│  │   React UI  │           ▼                               │
│  │  (Polling)  │    ┌─────────────┐                        │
│  └─────────────┘    │  Jobs API   │                        │
└─────────────────────┴──────┬──────┴─────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              On-Demand Cluster (Auto-scaling)               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │  Training   │───▶│   MLflow    │───▶│   Artifacts     │ │
│  │  Notebook   │    │   Logging   │    │   (Results)     │ │
│  └─────────────┘    └─────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Benefits:**
- App remains responsive
- Heavy compute on dedicated cluster
- State persisted - survives page refresh
- Supports cancellation and resume
- Scales to multiple concurrent users

---

## 2. Component Design

### 2.1 Job Delegation Service

**Location:** `backend/services/job_delegation.py`

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

class JobStatus(Enum):
    PENDING = "pending"           # Created, not yet submitted
    SUBMITTING = "submitting"     # Being submitted to cluster
    RUNNING = "running"           # Executing on cluster
    CANCELLING = "cancelling"     # Cancel requested
    CANCELLED = "cancelled"       # Successfully cancelled
    COMPLETED = "completed"       # Finished successfully
    FAILED = "failed"             # Failed with error

@dataclass
class TrainingJob:
    """Represents a training job with full lifecycle tracking."""

    # Identity
    job_id: str                   # Our internal UUID
    run_id: Optional[str]         # Databricks run_id (after submission)
    user_id: str                  # User who created the job

    # Configuration
    config: Dict[str, Any]        # Training configuration
    cluster_config: Dict[str, Any]  # Cluster specification

    # State
    status: JobStatus
    progress: int                 # 0-100 percentage
    current_step: str             # Human-readable current step

    # Timing
    created_at: datetime
    submitted_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    # Results
    mlflow_run_id: Optional[str]  # MLflow tracking run
    results: Optional[Dict[str, Any]]  # Final results
    error: Optional[str]          # Error message if failed

    # Resume support
    checkpoint_path: Optional[str]  # For resumable training
    retry_count: int = 0

class JobDelegationService:
    """
    Manages the lifecycle of training jobs delegated to Databricks clusters.

    Responsibilities:
    - Submit jobs to on-demand clusters
    - Track job status and progress
    - Handle cancellation
    - Persist state for recovery
    - Retrieve results from completed jobs
    """

    def __init__(self, state_store: 'JobStateStore'):
        self.state_store = state_store
        self.w = WorkspaceClient()

    async def create_job(
        self,
        user_id: str,
        config: Dict[str, Any],
        data_path: str
    ) -> TrainingJob:
        """
        Create a new training job (but don't submit yet).
        This allows the user to review before starting.
        """
        job = TrainingJob(
            job_id=str(uuid.uuid4()),
            run_id=None,
            user_id=user_id,
            config=config,
            cluster_config=self._get_cluster_config(config),
            status=JobStatus.PENDING,
            progress=0,
            current_step="Job created, ready to submit",
            created_at=datetime.utcnow(),
            submitted_at=None,
            started_at=None,
            completed_at=None,
            mlflow_run_id=None,
            results=None,
            error=None,
            checkpoint_path=None,
            retry_count=0
        )

        await self.state_store.save(job)
        return job

    async def submit_job(self, job_id: str) -> TrainingJob:
        """
        Submit a pending job to Databricks cluster.
        """
        job = await self.state_store.get(job_id)

        if job.status != JobStatus.PENDING:
            raise ValueError(f"Job {job_id} is not in PENDING state")

        job.status = JobStatus.SUBMITTING
        job.current_step = "Submitting to cluster..."
        await self.state_store.save(job)

        try:
            # Submit to Databricks
            run = self.w.jobs.submit(
                run_name=f"forecast_{job.job_id[:8]}",
                tasks=[
                    jobs.SubmitTask(
                        task_key="training",
                        new_cluster=job.cluster_config,
                        notebook_task=jobs.NotebookTask(
                            notebook_path="/Workspace/ML/training_notebook",
                            base_parameters={
                                "job_id": job.job_id,
                                "config": json.dumps(job.config)
                            }
                        ),
                        timeout_seconds=3600
                    )
                ]
            )

            job.run_id = str(run.run_id)
            job.status = JobStatus.RUNNING
            job.submitted_at = datetime.utcnow()
            job.current_step = "Training started on cluster"

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.current_step = "Submission failed"

        await self.state_store.save(job)
        return job

    async def get_status(self, job_id: str) -> TrainingJob:
        """
        Get current job status, syncing with Databricks if running.
        """
        job = await self.state_store.get(job_id)

        if job.status == JobStatus.RUNNING and job.run_id:
            # Sync with Databricks
            run = self.w.jobs.get_run(int(job.run_id))

            if run.state.life_cycle_state == "TERMINATED":
                if run.state.result_state == "SUCCESS":
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.utcnow()
                    job.current_step = "Training completed"
                    job.results = await self._fetch_results(job)
                else:
                    job.status = JobStatus.FAILED
                    job.error = run.state.state_message
                    job.current_step = "Training failed"

                await self.state_store.save(job)

        return job

    async def cancel_job(self, job_id: str) -> TrainingJob:
        """
        Cancel a running job.
        """
        job = await self.state_store.get(job_id)

        if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
            raise ValueError(f"Cannot cancel job in {job.status} state")

        job.status = JobStatus.CANCELLING
        job.current_step = "Cancelling..."
        await self.state_store.save(job)

        if job.run_id:
            try:
                self.w.jobs.cancel_run(int(job.run_id))
                job.status = JobStatus.CANCELLED
                job.current_step = "Cancelled by user"
            except Exception as e:
                job.error = f"Cancel failed: {e}"
        else:
            job.status = JobStatus.CANCELLED
            job.current_step = "Cancelled before submission"

        job.completed_at = datetime.utcnow()
        await self.state_store.save(job)
        return job

    async def list_jobs(
        self,
        user_id: str,
        status: Optional[JobStatus] = None,
        limit: int = 50
    ) -> List[TrainingJob]:
        """
        List jobs for a user, optionally filtered by status.
        """
        return await self.state_store.list(
            user_id=user_id,
            status=status,
            limit=limit
        )

    def _get_cluster_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine appropriate cluster configuration based on training config.
        """
        # Scale cluster based on data size and model complexity
        num_models = len(config.get("models", ["prophet"]))
        data_rows = config.get("data_rows", 1000)

        if data_rows > 50000 or num_models > 3:
            # Large workload
            return {
                "spark_version": "15.3.x-scala2.12",
                "node_type_id": "i3.2xlarge",
                "num_workers": 4,
                "autoscale": {"min_workers": 2, "max_workers": 8},
                "aws_attributes": {
                    "availability": "SPOT_WITH_FALLBACK"
                }
            }
        else:
            # Standard workload
            return {
                "spark_version": "15.3.x-scala2.12",
                "node_type_id": "i3.xlarge",
                "num_workers": 2,
                "aws_attributes": {
                    "availability": "SPOT_WITH_FALLBACK"
                }
            }

    async def _fetch_results(self, job: TrainingJob) -> Dict[str, Any]:
        """
        Fetch training results from MLflow after job completes.
        """
        # Results are stored in MLflow by the training notebook
        # The notebook logs the mlflow_run_id as a task value
        run = self.w.jobs.get_run(int(job.run_id))

        # Get MLflow run ID from task output
        mlflow_run_id = None
        for task in run.tasks:
            if task.state.result_state == "SUCCESS":
                # Task values contain the MLflow run ID
                mlflow_run_id = task.state.state_message  # Or from task values

        if mlflow_run_id:
            job.mlflow_run_id = mlflow_run_id

            # Fetch metrics and artifacts from MLflow
            client = mlflow.tracking.MlflowClient()
            run_data = client.get_run(mlflow_run_id)

            return {
                "mlflow_run_id": mlflow_run_id,
                "metrics": run_data.data.metrics,
                "params": run_data.data.params,
                "artifact_uri": run_data.info.artifact_uri
            }

        return {}
```

### 2.2 Job State Store

**Location:** `backend/services/job_state_store.py`

```python
from abc import ABC, abstractmethod
from typing import Optional, List
import sqlite3
import json
from pathlib import Path

class JobStateStore(ABC):
    """Abstract base class for job state persistence."""

    @abstractmethod
    async def save(self, job: TrainingJob) -> None:
        """Save or update a job."""
        pass

    @abstractmethod
    async def get(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID."""
        pass

    @abstractmethod
    async def list(
        self,
        user_id: str,
        status: Optional[JobStatus] = None,
        limit: int = 50
    ) -> List[TrainingJob]:
        """List jobs for a user."""
        pass

    @abstractmethod
    async def delete(self, job_id: str) -> None:
        """Delete a job."""
        pass


class SQLiteJobStateStore(JobStateStore):
    """
    SQLite-based state store for local development and single-instance deployment.

    For production multi-instance deployment, use Unity Catalog tables or
    a shared database like PostgreSQL.
    """

    def __init__(self, db_path: str = "jobs.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_jobs (
                job_id TEXT PRIMARY KEY,
                run_id TEXT,
                user_id TEXT NOT NULL,
                config TEXT NOT NULL,
                cluster_config TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER DEFAULT 0,
                current_step TEXT,
                created_at TEXT NOT NULL,
                submitted_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                mlflow_run_id TEXT,
                results TEXT,
                error TEXT,
                checkpoint_path TEXT,
                retry_count INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_status
            ON training_jobs(user_id, status)
        """)
        conn.commit()
        conn.close()

    async def save(self, job: TrainingJob) -> None:
        """Save or update a job."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO training_jobs (
                job_id, run_id, user_id, config, cluster_config,
                status, progress, current_step, created_at, submitted_at,
                started_at, completed_at, mlflow_run_id, results, error,
                checkpoint_path, retry_count, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job.job_id,
            job.run_id,
            job.user_id,
            json.dumps(job.config),
            json.dumps(job.cluster_config),
            job.status.value,
            job.progress,
            job.current_step,
            job.created_at.isoformat(),
            job.submitted_at.isoformat() if job.submitted_at else None,
            job.started_at.isoformat() if job.started_at else None,
            job.completed_at.isoformat() if job.completed_at else None,
            job.mlflow_run_id,
            json.dumps(job.results) if job.results else None,
            job.error,
            job.checkpoint_path,
            job.retry_count,
            datetime.utcnow().isoformat()
        ))
        conn.commit()
        conn.close()

    async def get(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM training_jobs WHERE job_id = ?",
            (job_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_job(row)
        return None

    async def list(
        self,
        user_id: str,
        status: Optional[JobStatus] = None,
        limit: int = 50
    ) -> List[TrainingJob]:
        """List jobs for a user."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        if status:
            cursor = conn.execute(
                """SELECT * FROM training_jobs
                   WHERE user_id = ? AND status = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, status.value, limit)
            )
        else:
            cursor = conn.execute(
                """SELECT * FROM training_jobs
                   WHERE user_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, limit)
            )

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_job(row) for row in rows]

    async def delete(self, job_id: str) -> None:
        """Delete a job."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM training_jobs WHERE job_id = ?", (job_id,))
        conn.commit()
        conn.close()

    def _row_to_job(self, row: sqlite3.Row) -> TrainingJob:
        """Convert a database row to a TrainingJob object."""
        return TrainingJob(
            job_id=row["job_id"],
            run_id=row["run_id"],
            user_id=row["user_id"],
            config=json.loads(row["config"]),
            cluster_config=json.loads(row["cluster_config"]),
            status=JobStatus(row["status"]),
            progress=row["progress"],
            current_step=row["current_step"],
            created_at=datetime.fromisoformat(row["created_at"]),
            submitted_at=datetime.fromisoformat(row["submitted_at"]) if row["submitted_at"] else None,
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            mlflow_run_id=row["mlflow_run_id"],
            results=json.loads(row["results"]) if row["results"] else None,
            error=row["error"],
            checkpoint_path=row["checkpoint_path"],
            retry_count=row["retry_count"]
        )


class UnityCatalogJobStateStore(JobStateStore):
    """
    Unity Catalog-based state store for production deployment.
    Uses Delta tables for ACID compliance and multi-instance support.

    Table: {catalog}.{schema}.training_jobs
    """

    def __init__(
        self,
        catalog: str = "main",
        schema: str = "forecasting",
        table: str = "training_jobs"
    ):
        self.table_path = f"{catalog}.{schema}.{table}"
        self._init_table()

    def _init_table(self):
        """Create table if not exists using Spark SQL."""
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.table_path} (
                job_id STRING,
                run_id STRING,
                user_id STRING,
                config STRING,
                cluster_config STRING,
                status STRING,
                progress INT,
                current_step STRING,
                created_at TIMESTAMP,
                submitted_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                mlflow_run_id STRING,
                results STRING,
                error STRING,
                checkpoint_path STRING,
                retry_count INT,
                updated_at TIMESTAMP
            )
            USING DELTA
            PARTITIONED BY (user_id)
            TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
        """)

    async def save(self, job: TrainingJob) -> None:
        """Save using MERGE for upsert behavior."""
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

        # Create temp view with new data
        # Then MERGE into target table
        # (Implementation details omitted for brevity)
        pass

    # ... similar implementations for get, list, delete
```

---

## 3. State Management

### 3.1 State Persistence Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                     State Layers                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │  Frontend State  │◀──▶│   localStorage   │              │
│  │   (React)        │    │   (UI cache)     │              │
│  └────────┬─────────┘    └──────────────────┘              │
│           │                                                  │
│           │ API calls                                        │
│           ▼                                                  │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │  Backend State   │◀──▶│   SQLite/UC      │              │
│  │  (Job Service)   │    │   (Persistent)   │              │
│  └────────┬─────────┘    └──────────────────┘              │
│           │                                                  │
│           │ Jobs API                                         │
│           ▼                                                  │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │ Databricks Jobs  │◀──▶│  Cluster State   │              │
│  │    (Source of    │    │  (Ephemeral)     │              │
│  │     Truth)       │    │                  │              │
│  └──────────────────┘    └──────────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 State Recovery Flow

When user refreshes the page:

```
1. Frontend loads from localStorage (instant UI)
2. Frontend calls GET /api/jobs (user's jobs)
3. Backend returns persisted job states
4. For RUNNING jobs, backend syncs with Databricks
5. Frontend updates with authoritative state
```

```typescript
// Frontend: useJobRecovery.ts
export function useJobRecovery() {
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 1. Load from localStorage immediately
    const cached = localStorage.getItem('training_jobs');
    if (cached) {
      setJobs(JSON.parse(cached));
    }

    // 2. Fetch authoritative state from backend
    fetchJobs().then(serverJobs => {
      setJobs(serverJobs);
      localStorage.setItem('training_jobs', JSON.stringify(serverJobs));
      setLoading(false);
    });
  }, []);

  // 3. Poll for updates on running jobs
  useEffect(() => {
    const runningJobs = jobs.filter(j => j.status === 'running');
    if (runningJobs.length === 0) return;

    const interval = setInterval(async () => {
      for (const job of runningJobs) {
        const updated = await fetchJobStatus(job.job_id);
        setJobs(prev => prev.map(j =>
          j.job_id === updated.job_id ? updated : j
        ));
      }
    }, 5000); // Poll every 5 seconds

    return () => clearInterval(interval);
  }, [jobs]);

  return { jobs, loading };
}
```

---

## 4. Job Lifecycle

### 4.1 State Machine

```
                              ┌───────────────┐
                              │    PENDING    │
                              │  (Created,    │
                              │   not sent)   │
                              └───────┬───────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │ submit()        │                 │ cancel()
                    ▼                 │                 ▼
           ┌───────────────┐         │         ┌───────────────┐
           │  SUBMITTING   │         │         │   CANCELLED   │
           │ (Sending to   │         │         │               │
           │  cluster)     │         │         └───────────────┘
           └───────┬───────┘         │
                   │                 │
     ┌─────────────┼─────────────┐   │
     │ success     │             │ error
     ▼             │             ▼
┌───────────────┐  │    ┌───────────────┐
│    RUNNING    │  │    │    FAILED     │
│ (Executing    │──┼───▶│               │
│  on cluster)  │  │    └───────────────┘
└───────┬───────┘  │              ▲
        │          │              │
        │ cancel() │              │
        ▼          │              │
┌───────────────┐  │              │
│  CANCELLING   │  │              │
│               │──┼──────────────┘
└───────┬───────┘  │
        │          │
        │          │
        ▼          │
┌───────────────┐  │    ┌───────────────┐
│   CANCELLED   │  │    │   COMPLETED   │
│               │  │    │  (Success)    │
└───────────────┘  │    └───────────────┘
                   │              ▲
                   │              │
                   └──────────────┘
                       completed
```

### 4.2 Lifecycle Events

| Event | Trigger | Actions |
|-------|---------|---------|
| `JOB_CREATED` | User configures training | Save to state store, return job_id |
| `JOB_SUBMITTED` | User clicks "Start Training" | Submit to Jobs API, update status |
| `JOB_STARTED` | Cluster reports execution began | Update started_at, set progress=5% |
| `JOB_PROGRESS` | Notebook reports progress | Update progress %, current_step |
| `JOB_COMPLETED` | Cluster reports success | Fetch results, update status |
| `JOB_FAILED` | Cluster reports error | Capture error, update status |
| `JOB_CANCELLED` | User requests cancel | Call cancel API, update status |

---

## 5. API Contracts

### 5.1 Job Management Endpoints

```yaml
# Create a new training job
POST /api/v2/jobs
Request:
  {
    "config": {
      "data_path": "/dbfs/data/training.parquet",
      "target_col": "sales",
      "time_col": "date",
      "horizon": 12,
      "models": ["prophet", "xgboost"],
      "covariates": ["is_promo", "marketing_spend"]
    }
  }
Response:
  {
    "job_id": "a1b2c3d4-e5f6-...",
    "status": "pending",
    "created_at": "2026-01-20T15:30:00Z",
    "cluster_config": { ... }
  }

# Submit a pending job
POST /api/v2/jobs/{job_id}/submit
Response:
  {
    "job_id": "a1b2c3d4-e5f6-...",
    "run_id": "12345678",
    "status": "running",
    "submitted_at": "2026-01-20T15:31:00Z"
  }

# Get job status
GET /api/v2/jobs/{job_id}
Response:
  {
    "job_id": "a1b2c3d4-e5f6-...",
    "run_id": "12345678",
    "status": "running",
    "progress": 45,
    "current_step": "Training XGBoost model (2/5)...",
    "created_at": "2026-01-20T15:30:00Z",
    "submitted_at": "2026-01-20T15:31:00Z",
    "started_at": "2026-01-20T15:32:00Z"
  }

# Cancel a job
POST /api/v2/jobs/{job_id}/cancel
Response:
  {
    "job_id": "a1b2c3d4-e5f6-...",
    "status": "cancelled",
    "completed_at": "2026-01-20T15:35:00Z"
  }

# Get job results (only for completed jobs)
GET /api/v2/jobs/{job_id}/results
Response:
  {
    "job_id": "a1b2c3d4-e5f6-...",
    "status": "completed",
    "mlflow_run_id": "abc123...",
    "results": {
      "best_model": "prophet",
      "metrics": {
        "prophet": { "mape": 5.2, "rmse": 1200 },
        "xgboost": { "mape": 6.8, "rmse": 1400 }
      },
      "forecast": [ ... ],
      "artifact_uri": "dbfs:/mlflow/..."
    }
  }

# List user's jobs
GET /api/v2/jobs?status=running&limit=10
Response:
  {
    "jobs": [ ... ],
    "total": 25,
    "has_more": true
  }
```

### 5.2 Legacy Endpoint (Backward Compatibility)

The existing `/api/train` endpoint will be modified to use delegation internally while maintaining the same response format:

```python
@app.post("/api/train")
async def train_model(request: TrainRequest):
    """
    Legacy endpoint - now delegates to cluster.
    Maintains backward compatibility with existing frontend.
    """
    if os.getenv("ENABLE_CLUSTER_DELEGATION", "false").lower() == "true":
        # New async flow
        job = await job_service.create_job(
            user_id=get_current_user(),
            config=request.dict()
        )
        job = await job_service.submit_job(job.job_id)

        # Poll until complete (with timeout)
        while job.status == JobStatus.RUNNING:
            await asyncio.sleep(5)
            job = await job_service.get_status(job.job_id)

        if job.status == JobStatus.COMPLETED:
            return job.results  # Same format as before
        else:
            raise HTTPException(500, job.error)
    else:
        # Original inline training (for local development)
        return await train_model_inline(request)
```

---

## 6. Data Flow

### 6.1 Training Data Flow

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ User    │───▶│ Frontend │───▶│ Backend  │───▶│ DBFS/UC  │
│ Uploads │    │ (React)  │    │ (FastAPI)│    │ Storage  │
│ CSV     │    └──────────┘    └──────────┘    └────┬─────┘
└─────────┘                                         │
                                                    │ data_path
                                                    ▼
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Results │◀───│ Frontend │◀───│ Backend  │◀───│ Training │
│ Display │    │ (Polls)  │    │ (State)  │    │ Cluster  │
└─────────┘    └──────────┘    └──────────┘    └──────────┘
```

### 6.2 Data Upload Strategy

For large datasets, data should be uploaded to DBFS/Unity Catalog before job submission:

```python
@app.post("/api/v2/data/upload")
async def upload_training_data(file: UploadFile):
    """
    Upload training data to DBFS.
    Returns a data_path for use in job submission.
    """
    # Generate unique path
    data_id = str(uuid.uuid4())[:8]
    data_path = f"/dbfs/training_data/{data_id}/data.parquet"

    # Read and validate
    df = pd.read_csv(file.file)

    # Convert to parquet and upload
    df.to_parquet(data_path, index=False)

    return {
        "data_path": data_path,
        "rows": len(df),
        "columns": list(df.columns)
    }
```

---

## 7. Error Handling

### 7.1 Error Categories

| Category | Examples | Recovery Strategy |
|----------|----------|-------------------|
| **Transient** | Network timeout, cluster busy | Auto-retry with backoff |
| **Cluster** | OOM, timeout, spot termination | Retry with larger cluster |
| **Data** | Invalid format, missing columns | Return clear error to user |
| **Config** | Invalid model, bad parameters | Validate before submission |
| **Permanent** | Quota exceeded, auth failed | Alert user, no retry |

### 7.2 Retry Logic

```python
class JobDelegationService:
    MAX_RETRIES = 3
    RETRY_DELAYS = [30, 60, 120]  # seconds

    async def submit_with_retry(self, job_id: str) -> TrainingJob:
        """Submit job with automatic retry on transient failures."""
        job = await self.state_store.get(job_id)

        for attempt in range(self.MAX_RETRIES):
            try:
                return await self.submit_job(job_id)

            except TransientError as e:
                job.retry_count = attempt + 1
                job.current_step = f"Retry {attempt + 1}/{self.MAX_RETRIES}..."
                await self.state_store.save(job)

                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAYS[attempt])
                else:
                    job.status = JobStatus.FAILED
                    job.error = f"Failed after {self.MAX_RETRIES} retries: {e}"
                    await self.state_store.save(job)
                    raise
```

### 7.3 Cluster Failure Recovery

```python
async def handle_spot_termination(self, job: TrainingJob) -> TrainingJob:
    """
    Handle spot instance termination by resubmitting with fallback.
    """
    if job.retry_count < self.MAX_RETRIES:
        # Upgrade to on-demand for reliability
        job.cluster_config["aws_attributes"]["availability"] = "ON_DEMAND"
        job.retry_count += 1
        job.status = JobStatus.PENDING
        job.current_step = "Resubmitting with on-demand cluster..."

        await self.state_store.save(job)
        return await self.submit_job(job.job_id)
    else:
        job.status = JobStatus.FAILED
        job.error = "Cluster terminated, max retries exceeded"
        await self.state_store.save(job)
        return job
```

---

## 8. Frontend Integration

### 8.1 React Hook for Job Management

```typescript
// hooks/useTrainingJob.ts
import { useState, useEffect, useCallback } from 'react';

interface TrainingJob {
  job_id: string;
  status: 'pending' | 'submitting' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  current_step: string;
  results?: any;
  error?: string;
}

export function useTrainingJob(jobId: string | null) {
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch job status
  const fetchStatus = useCallback(async () => {
    if (!jobId) return;

    try {
      const response = await fetch(`/api/v2/jobs/${jobId}`);
      const data = await response.json();
      setJob(data);

      // Persist to localStorage for recovery
      localStorage.setItem(`job_${jobId}`, JSON.stringify(data));

      return data;
    } catch (e) {
      setError(e.message);
    }
  }, [jobId]);

  // Poll for updates when running
  useEffect(() => {
    if (!job || !['running', 'submitting'].includes(job.status)) {
      return;
    }

    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [job?.status, fetchStatus]);

  // Submit job
  const submit = useCallback(async () => {
    if (!jobId) return;

    setLoading(true);
    try {
      const response = await fetch(`/api/v2/jobs/${jobId}/submit`, {
        method: 'POST'
      });
      const data = await response.json();
      setJob(data);
      return data;
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  // Cancel job
  const cancel = useCallback(async () => {
    if (!jobId) return;

    setLoading(true);
    try {
      const response = await fetch(`/api/v2/jobs/${jobId}/cancel`, {
        method: 'POST'
      });
      const data = await response.json();
      setJob(data);
      return data;
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  // Recovery from localStorage on mount
  useEffect(() => {
    if (jobId) {
      const cached = localStorage.getItem(`job_${jobId}`);
      if (cached) {
        setJob(JSON.parse(cached));
      }
      fetchStatus();
    }
  }, [jobId, fetchStatus]);

  return { job, loading, error, submit, cancel, refresh: fetchStatus };
}
```

### 8.2 Training Panel Component

```typescript
// components/TrainingPanel.tsx
import { useTrainingJob } from '../hooks/useTrainingJob';

export function TrainingPanel({ jobId }: { jobId: string }) {
  const { job, loading, error, submit, cancel, refresh } = useTrainingJob(jobId);

  if (!job) {
    return <div>Loading...</div>;
  }

  return (
    <div className="training-panel">
      {/* Status Header */}
      <div className="status-header">
        <StatusBadge status={job.status} />
        <span className="current-step">{job.current_step}</span>
      </div>

      {/* Progress Bar */}
      {['running', 'submitting'].includes(job.status) && (
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${job.progress}%` }}
          />
          <span>{job.progress}%</span>
        </div>
      )}

      {/* Action Buttons */}
      <div className="actions">
        {job.status === 'pending' && (
          <button onClick={submit} disabled={loading}>
            Start Training
          </button>
        )}

        {['running', 'submitting'].includes(job.status) && (
          <button onClick={cancel} disabled={loading} className="cancel">
            Cancel
          </button>
        )}

        {job.status === 'completed' && (
          <button onClick={() => window.location.href = `/results/${jobId}`}>
            View Results
          </button>
        )}

        {job.status === 'failed' && (
          <>
            <div className="error-message">{job.error}</div>
            <button onClick={submit}>Retry</button>
          </>
        )}
      </div>

      {/* Results Preview */}
      {job.status === 'completed' && job.results && (
        <ResultsPreview results={job.results} />
      )}
    </div>
  );
}
```

---

## 9. Local Development

### 9.1 Environment Detection

```python
# backend/config.py
import os

class Config:
    # Deployment mode
    IS_DATABRICKS = os.getenv("DATABRICKS_RUNTIME_VERSION") is not None
    ENABLE_CLUSTER_DELEGATION = os.getenv("ENABLE_CLUSTER_DELEGATION", "false").lower() == "true"

    # State store configuration
    STATE_STORE_TYPE = os.getenv("STATE_STORE_TYPE", "sqlite")  # sqlite, unity_catalog
    SQLITE_PATH = os.getenv("SQLITE_PATH", "jobs.db")
    UC_CATALOG = os.getenv("UC_CATALOG", "main")
    UC_SCHEMA = os.getenv("UC_SCHEMA", "forecasting")

    @classmethod
    def get_state_store(cls) -> JobStateStore:
        if cls.STATE_STORE_TYPE == "unity_catalog":
            return UnityCatalogJobStateStore(
                catalog=cls.UC_CATALOG,
                schema=cls.UC_SCHEMA
            )
        else:
            return SQLiteJobStateStore(cls.SQLITE_PATH)

    @classmethod
    def should_delegate(cls) -> bool:
        """Determine if jobs should be delegated to clusters."""
        return cls.IS_DATABRICKS and cls.ENABLE_CLUSTER_DELEGATION
```

### 9.2 Local Mode (No Delegation)

When running locally without Databricks:

```python
# backend/services/job_delegation.py

class LocalJobExecutor:
    """
    Execute training jobs inline for local development.
    Same interface as JobDelegationService but runs synchronously.
    """

    def __init__(self, state_store: JobStateStore):
        self.state_store = state_store

    async def submit_job(self, job_id: str) -> TrainingJob:
        """Execute training inline instead of delegating."""
        job = await self.state_store.get(job_id)

        job.status = JobStatus.RUNNING
        job.submitted_at = datetime.utcnow()
        job.started_at = datetime.utcnow()
        await self.state_store.save(job)

        try:
            # Run training inline (same as current implementation)
            results = await self._run_training(job.config)

            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.results = results
            job.progress = 100
            job.current_step = "Training completed"

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.current_step = "Training failed"

        await self.state_store.save(job)
        return job

    async def _run_training(self, config: dict) -> dict:
        """Run the actual training (reuse existing training code)."""
        from backend.models.prophet import train_prophet_model
        from backend.models.xgboost import train_xgboost_model
        # ... existing training logic
```

### 9.3 Feature Flags

```bash
# .env.local (Local Development)
ENABLE_CLUSTER_DELEGATION=false
STATE_STORE_TYPE=sqlite

# .env.databricks (Databricks Deployment)
ENABLE_CLUSTER_DELEGATION=true
STATE_STORE_TYPE=unity_catalog
UC_CATALOG=main
UC_SCHEMA=forecasting
```

---

## 10. Security Considerations

### 10.1 Authentication & Authorization

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)) -> str:
    """
    Validate token and return user ID.
    On Databricks, uses workspace authentication.
    """
    if Config.IS_DATABRICKS:
        # Use Databricks workspace auth
        w = WorkspaceClient()
        user = w.current_user.me()
        return user.user_name
    else:
        # Local development - use header or default
        return "local_user"

@app.post("/api/v2/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    current_user: str = Depends(get_current_user)
):
    """Only allow users to cancel their own jobs."""
    job = await job_service.get_status(job_id)

    if job.user_id != current_user:
        raise HTTPException(403, "Cannot cancel another user's job")

    return await job_service.cancel_job(job_id)
```

### 10.2 Data Isolation

```python
# Ensure users can only access their own jobs
async def list_jobs(
    current_user: str = Depends(get_current_user),
    status: Optional[str] = None,
    limit: int = 50
):
    """List only the current user's jobs."""
    return await job_service.list_jobs(
        user_id=current_user,  # Always filter by user
        status=JobStatus(status) if status else None,
        limit=limit
    )
```

### 10.3 Cluster Permissions

```python
def _get_cluster_config(self, config: dict) -> dict:
    """
    Configure cluster with appropriate permissions.
    Uses instance profiles for AWS resource access.
    """
    return {
        "spark_version": "15.3.x-scala2.12",
        "node_type_id": "i3.xlarge",
        "num_workers": 2,
        "aws_attributes": {
            "availability": "SPOT_WITH_FALLBACK",
            "instance_profile_arn": os.getenv("CLUSTER_INSTANCE_PROFILE")
        },
        "spark_conf": {
            "spark.databricks.cluster.profile": "serverless"
        }
    }
```

---

## 11. Cost Optimization

### 11.1 Cluster Configuration

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| Spot instances | 60-70% | May be interrupted |
| Auto-scaling | 40-50% | Slower scale-up |
| Auto-termination | Variable | Cold start on next job |
| Smaller nodes | Variable | Longer runtime |

### 11.2 Implementation

```python
def _get_cost_optimized_cluster(self, config: dict) -> dict:
    """
    Select cluster configuration optimized for cost.
    """
    data_size = config.get("data_rows", 1000)
    model_count = len(config.get("models", []))

    # Start small, scale if needed
    if data_size < 10000 and model_count <= 2:
        # Small workload - use minimal resources
        return {
            "spark_version": "15.3.x-scala2.12",
            "node_type_id": "m5.large",  # Cheaper instance
            "num_workers": 1,
            "aws_attributes": {
                "availability": "SPOT",  # Spot only for cost
                "zone_id": "auto"
            },
            "autotermination_minutes": 10  # Quick termination
        }
    else:
        # Larger workload - balanced approach
        return {
            "spark_version": "15.3.x-scala2.12",
            "node_type_id": "i3.xlarge",
            "autoscale": {
                "min_workers": 1,
                "max_workers": 4
            },
            "aws_attributes": {
                "availability": "SPOT_WITH_FALLBACK"
            },
            "autotermination_minutes": 15
        }
```

### 11.3 Job Pooling (Future Enhancement)

For high-volume usage, implement a warm cluster pool:

```python
class ClusterPool:
    """
    Maintain a pool of warm clusters for instant job submission.
    Reduces cold start time from ~5 minutes to ~30 seconds.
    """

    def __init__(self, pool_size: int = 2):
        self.pool_size = pool_size
        self.available_clusters = []
        self.busy_clusters = {}

    async def acquire_cluster(self, job: TrainingJob) -> str:
        """Get an available cluster or create one."""
        if self.available_clusters:
            cluster_id = self.available_clusters.pop()
            self.busy_clusters[job.job_id] = cluster_id
            return cluster_id
        else:
            # No available clusters - create new one
            return None  # Will create on-demand

    async def release_cluster(self, job_id: str) -> None:
        """Return cluster to pool after job completes."""
        if job_id in self.busy_clusters:
            cluster_id = self.busy_clusters.pop(job_id)
            if len(self.available_clusters) < self.pool_size:
                self.available_clusters.append(cluster_id)
            else:
                # Pool full - terminate cluster
                await self._terminate_cluster(cluster_id)
```

---

## 12. Deployment Architecture

### 12.1 Deployment Prerequisites

| Requirement | Description |
|-------------|-------------|
| Databricks Workspace | With Unity Catalog enabled |
| MLflow Experiment | Pre-created experiment path |
| Training Notebook | Deployed to `/Workspace/ML/training_notebook` |

### 12.2 Post-Deployment Configuration Flow

The app service principal (SP) is created automatically when the app is deployed. We configure permissions **after** deployment:

```
┌─────────────────────────────────────────────────────────────┐
│                    Deployment Flow                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Deploy App (no SP exists yet)                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  databricks bundle deploy                            │   │
│  │  → Creates Databricks App                            │   │
│  │  → Auto-creates App Service Principal                │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  Step 2: Retrieve App Service Principal                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  databricks apps get finance-forecast-app            │   │
│  │  → Returns: service_principal_id                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  Step 3: Create Single-User Cluster for App SP              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Create cluster with:                                │   │
│  │  - single_user_name: <app_sp_id>                    │   │
│  │  - data_security_mode: SINGLE_USER                  │   │
│  │  - auto_termination_minutes: 30                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  Step 4: Grant Permissions to App SP                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  - CAN_ATTACH_TO on cluster                         │   │
│  │  - CAN_MANAGE on cluster (start/stop/restart)       │   │
│  │  - CAN_RUN on Jobs                                  │   │
│  │  - USE_CATALOG on Unity Catalog                     │   │
│  │  - USE_SCHEMA on forecasting schema                 │   │
│  │  - SELECT/MODIFY on training_jobs table             │   │
│  │  - READ_WRITE on MLflow experiment                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 12.3 Deployment Script

**Location:** `scripts/deploy_to_databricks.py`

```python
#!/usr/bin/env python3
"""
Post-deployment configuration script.
Run after `databricks bundle deploy` to configure cluster and permissions.
"""

import os
import json
import subprocess
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute, iam

def main():
    app_name = os.getenv("APP_NAME", "finance-forecast-app")
    cluster_name = f"{app_name}-compute"

    w = WorkspaceClient()

    # Step 1: Get App Service Principal
    print(f"Getting service principal for app: {app_name}")
    app = w.apps.get(app_name)
    app_sp_id = app.service_principal_id
    app_sp_name = app.service_principal_name
    print(f"  Service Principal ID: {app_sp_id}")
    print(f"  Service Principal Name: {app_sp_name}")

    # Step 2: Create or Get Single-User Cluster
    print(f"\nConfiguring cluster: {cluster_name}")
    existing_clusters = list(w.clusters.list())
    cluster = next((c for c in existing_clusters if c.cluster_name == cluster_name), None)

    if cluster:
        print(f"  Cluster exists: {cluster.cluster_id}")
        # Update to ensure correct config
        w.clusters.edit(
            cluster_id=cluster.cluster_id,
            cluster_name=cluster_name,
            spark_version="15.3.x-scala2.12",
            node_type_id="i3.xlarge",
            num_workers=0,  # Single node for cost
            autotermination_minutes=30,
            data_security_mode=compute.DataSecurityMode.SINGLE_USER,
            single_user_name=app_sp_name,
            spark_conf={
                "spark.databricks.cluster.profile": "singleNode",
                "spark.master": "local[*]"
            },
            custom_tags={
                "app": app_name,
                "managed_by": "forecast-app"
            }
        )
    else:
        print("  Creating new cluster...")
        cluster = w.clusters.create(
            cluster_name=cluster_name,
            spark_version="15.3.x-scala2.12",
            node_type_id="i3.xlarge",
            num_workers=0,
            autotermination_minutes=30,
            data_security_mode=compute.DataSecurityMode.SINGLE_USER,
            single_user_name=app_sp_name,
            spark_conf={
                "spark.databricks.cluster.profile": "singleNode",
                "spark.master": "local[*]"
            },
            custom_tags={
                "app": app_name,
                "managed_by": "forecast-app"
            }
        ).result()
        print(f"  Created cluster: {cluster.cluster_id}")

    cluster_id = cluster.cluster_id

    # Step 3: Grant Cluster Permissions to App SP
    print(f"\nGranting cluster permissions to {app_sp_name}...")
    w.permissions.set(
        object_type="clusters",
        object_id=cluster_id,
        access_control_list=[
            iam.AccessControlRequest(
                service_principal_name=app_sp_name,
                permission_level=iam.PermissionLevel.CAN_MANAGE
            )
        ]
    )
    print("  CAN_MANAGE granted on cluster")

    # Step 4: Grant Jobs Permissions
    print(f"\nGranting jobs permissions...")
    # App SP needs ability to create and run jobs
    # This is typically workspace-level or handled by cluster access

    # Step 5: Grant Unity Catalog Permissions
    print(f"\nGranting Unity Catalog permissions...")
    catalog = os.getenv("UC_CATALOG", "main")
    schema = os.getenv("UC_SCHEMA", "forecasting")

    # Grant USE CATALOG
    w.grants.update(
        securable_type="catalog",
        full_name=catalog,
        changes=[
            iam.PermissionsChange(
                add=[iam.Privilege.USE_CATALOG],
                principal=app_sp_name
            )
        ]
    )
    print(f"  USE_CATALOG granted on {catalog}")

    # Grant USE SCHEMA + CREATE TABLE
    w.grants.update(
        securable_type="schema",
        full_name=f"{catalog}.{schema}",
        changes=[
            iam.PermissionsChange(
                add=[
                    iam.Privilege.USE_SCHEMA,
                    iam.Privilege.CREATE_TABLE,
                    iam.Privilege.SELECT,
                    iam.Privilege.MODIFY
                ],
                principal=app_sp_name
            )
        ]
    )
    print(f"  Schema permissions granted on {catalog}.{schema}")

    # Step 6: Grant MLflow Permissions
    print(f"\nGranting MLflow experiment permissions...")
    experiment_path = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/finance-forecasting")

    try:
        experiment = w.experiments.get_by_name(experiment_path)
        w.experiments.set_permissions(
            experiment_id=experiment.experiment_id,
            access_control_list=[
                iam.AccessControlRequest(
                    service_principal_name=app_sp_name,
                    permission_level=iam.PermissionLevel.CAN_MANAGE
                )
            ]
        )
        print(f"  CAN_MANAGE granted on experiment {experiment_path}")
    except Exception as e:
        print(f"  Warning: Could not set experiment permissions: {e}")

    # Step 7: Store cluster_id in app config
    print(f"\nStoring cluster ID in app environment...")
    # Update app.yaml or environment variable

    print(f"\n{'='*60}")
    print("DEPLOYMENT COMPLETE")
    print(f"{'='*60}")
    print(f"App Name:           {app_name}")
    print(f"Service Principal:  {app_sp_name}")
    print(f"Cluster ID:         {cluster_id}")
    print(f"Cluster Name:       {cluster_name}")
    print(f"\nNext steps:")
    print(f"1. Set DEDICATED_CLUSTER_ID={cluster_id} in app environment")
    print(f"2. Restart the app to pick up the cluster configuration")

if __name__ == "__main__":
    main()
```

### 12.4 Cluster Configuration

The dedicated cluster for the app should be:

| Setting | Value | Reason |
|---------|-------|--------|
| `data_security_mode` | `SINGLE_USER` | Only app SP can use |
| `single_user_name` | App SP name | Exclusive access |
| `autotermination_minutes` | `30` | Cost optimization |
| `num_workers` | `0` (single node) or `2` | Start small |
| `node_type_id` | `i3.xlarge` | Good for ML workloads |

### 12.5 Required Permissions Matrix

| Permission | Resource | Level | Purpose |
|------------|----------|-------|---------|
| `CAN_MANAGE` | Cluster | Cluster | Start/stop/restart cluster |
| `CAN_ATTACH_TO` | Cluster | Cluster | Run code on cluster |
| `CAN_RUN` | Jobs | Workspace | Submit and manage jobs |
| `USE_CATALOG` | Unity Catalog | Catalog | Access catalog |
| `USE_SCHEMA` | Schema | Schema | Access schema |
| `CREATE_TABLE` | Schema | Schema | Create state tables |
| `SELECT/MODIFY` | Tables | Schema | Read/write job state |
| `CAN_MANAGE` | MLflow Experiment | Experiment | Log runs and artifacts |

### 12.6 Updated app.yaml

```yaml
name: finance-forecast-app
description: Finance Forecasting Platform with Cluster Delegation

command:
  - python
  - -m
  - uvicorn
  - backend.main:app
  - --host
  - 0.0.0.0
  - --port
  - "8000"

env:
  - name: DATABRICKS_HOST
    value: "inherit"
  - name: ENABLE_CLUSTER_DELEGATION
    value: "true"
  - name: DEDICATED_CLUSTER_ID
    value: "${CLUSTER_ID}"  # Set after deployment
  - name: STATE_STORE_TYPE
    value: "unity_catalog"
  - name: UC_CATALOG
    value: "main"
  - name: UC_SCHEMA
    value: "forecasting"
  - name: MLFLOW_EXPERIMENT_NAME
    value: "/Shared/finance-forecasting"

permissions:
  - workspace_access
  - cluster_access
  - model_serving_access
  - unity_catalog_access
```

### 12.7 Full Deployment Commands

```bash
#!/bin/bash
# deploy.sh - Complete deployment script

set -e

APP_NAME="finance-forecast-app"
CLUSTER_NAME="${APP_NAME}-compute"

echo "=== Step 1: Build Frontend ==="
npm run build

echo "=== Step 2: Deploy App Bundle ==="
databricks bundle deploy --target prod

echo "=== Step 3: Wait for App to be Ready ==="
sleep 30  # Wait for SP creation

echo "=== Step 4: Configure Cluster and Permissions ==="
python scripts/deploy_to_databricks.py

echo "=== Step 5: Update App with Cluster ID ==="
CLUSTER_ID=$(databricks clusters list --output json | jq -r ".[] | select(.cluster_name==\"$CLUSTER_NAME\") | .cluster_id")
databricks apps update $APP_NAME --env "DEDICATED_CLUSTER_ID=$CLUSTER_ID"

echo "=== Step 6: Restart App ==="
databricks apps restart $APP_NAME

echo "=== Deployment Complete ==="
databricks apps get $APP_NAME
```

---

## 13. Migration Path

### Phase 1: Backend Infrastructure (Week 1)

1. Create `backend/services/job_delegation.py`
2. Create `backend/services/job_state_store.py`
3. Add SQLite state store for local development
4. Add feature flag `ENABLE_CLUSTER_DELEGATION`
5. Create `/api/v2/jobs` endpoints

### Phase 2: Training Notebook (Week 1)

1. Create Databricks notebook at `/Workspace/ML/training_notebook`
2. Notebook reads config from parameters
3. Runs existing training code
4. Logs results to MLflow
5. Reports progress via task values

### Phase 3: Frontend Integration (Week 2)

1. Create `useTrainingJob` React hook
2. Update `SimpleModePanel.tsx` to use new flow
3. Add job status polling
4. Implement localStorage persistence
5. Add cancel button and error handling

### Phase 4: Testing & Rollout (Week 2)

1. Test locally with `ENABLE_CLUSTER_DELEGATION=false`
2. Test on Databricks with delegation enabled
3. Verify state persistence across page refresh
4. Verify cancellation works
5. Performance test with concurrent users

### Phase 5: Production Deployment

1. Deploy to Databricks Apps
2. Monitor job queue and cluster usage
3. Tune cluster configurations
4. Enable Unity Catalog state store for multi-instance

---

## Appendix A: Training Notebook Template

```python
# /Workspace/ML/training_notebook

# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Forecasting Training Job
# MAGIC This notebook is executed by the Job Delegation Service.

# COMMAND ----------

import json
import mlflow
from datetime import datetime

# Get parameters from job
job_id = dbutils.widgets.get("job_id")
config_json = dbutils.widgets.get("config")
config = json.loads(config_json)

print(f"Starting training job: {job_id}")
print(f"Config: {config}")

# COMMAND ----------

# Report progress function
def report_progress(progress: int, step: str):
    """Report progress back to the app."""
    dbutils.jobs.taskValues.set(key="progress", value=progress)
    dbutils.jobs.taskValues.set(key="current_step", value=step)

report_progress(5, "Loading training data...")

# COMMAND ----------

# Load data
import pandas as pd

data_path = config["data_path"]
df = pd.read_parquet(data_path)
print(f"Loaded {len(df)} rows")

report_progress(10, "Data loaded, starting preprocessing...")

# COMMAND ----------

# Preprocessing
from backend.preprocessing import enhance_features_for_forecasting

df = enhance_features_for_forecasting(
    df,
    date_col=config["time_col"],
    target_col=config["target_col"],
    promo_cols=config.get("covariates", []),
    frequency=config["frequency"]
)

report_progress(20, "Preprocessing complete, starting training...")

# COMMAND ----------

# Train models
from backend.models.prophet import train_prophet_model
from backend.models.xgboost import train_xgboost_model

models_to_train = config.get("models", ["prophet"])
results = {}

for i, model_name in enumerate(models_to_train):
    progress = 20 + int((i / len(models_to_train)) * 60)
    report_progress(progress, f"Training {model_name}...")

    if model_name == "prophet":
        result = train_prophet_model(df, config)
    elif model_name == "xgboost":
        result = train_xgboost_model(df, config)
    # ... other models

    results[model_name] = result

report_progress(85, "Training complete, logging results...")

# COMMAND ----------

# Log to MLflow
with mlflow.start_run(run_name=f"job_{job_id[:8]}") as run:
    # Log metrics
    for model_name, result in results.items():
        mlflow.log_metrics({
            f"{model_name}_mape": result["metrics"]["mape"],
            f"{model_name}_rmse": result["metrics"]["rmse"]
        })

    # Log artifacts
    mlflow.log_dict(results, "results.json")

    # Store MLflow run ID for retrieval
    dbutils.jobs.taskValues.set(key="mlflow_run_id", value=run.info.run_id)

report_progress(95, "Results logged to MLflow...")

# COMMAND ----------

# Final output
report_progress(100, "Job completed successfully")

# Return results as notebook output
dbutils.notebook.exit(json.dumps({
    "status": "success",
    "mlflow_run_id": run.info.run_id,
    "best_model": min(results, key=lambda m: results[m]["metrics"]["mape"])
}))
```

---

## Appendix B: Environment Variables

```bash
# Required for cluster delegation
ENABLE_CLUSTER_DELEGATION=true
STATE_STORE_TYPE=unity_catalog  # or sqlite
UC_CATALOG=main
UC_SCHEMA=forecasting

# Cluster configuration
DEFAULT_CLUSTER_NODE_TYPE=i3.xlarge
DEFAULT_CLUSTER_MIN_WORKERS=1
DEFAULT_CLUSTER_MAX_WORKERS=4
CLUSTER_SPOT_FALLBACK=true
CLUSTER_AUTO_TERMINATE_MINUTES=15

# Training notebook
TRAINING_NOTEBOOK_PATH=/Workspace/ML/training_notebook

# Timeouts
JOB_SUBMIT_TIMEOUT_SECONDS=60
JOB_EXECUTION_TIMEOUT_SECONDS=3600
JOB_POLL_INTERVAL_SECONDS=5

# Retry configuration
MAX_JOB_RETRIES=3
RETRY_DELAY_SECONDS=30
```

---

**End of Design Document**
