"""
Job State Store - Persistent storage for training job state.

Supports page refresh recovery, cancellation, and resume.
"""

import sqlite3
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

from backend.utils.logging_utils import log_io

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Training job status states."""
    PENDING = "pending"
    SUBMITTING = "submitting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingJob:
    """Represents a training job with full lifecycle tracking."""

    # Identity
    job_id: str
    user_id: str
    run_id: Optional[str] = None  # Databricks run_id after submission

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    cluster_id: Optional[str] = None

    # State
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    current_step: str = "Job created"

    # Timing
    created_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    mlflow_run_id: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Resume support
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'job_id': self.job_id,
            'user_id': self.user_id,
            'run_id': self.run_id,
            'config': self.config,
            'cluster_id': self.cluster_id,
            'status': self.status.value,
            'progress': self.progress,
            'current_step': self.current_step,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'mlflow_run_id': self.mlflow_run_id,
            'results': self.results,
            'error': self.error,
            'retry_count': self.retry_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingJob':
        """Create from dictionary."""
        return cls(
            job_id=data['job_id'],
            user_id=data['user_id'],
            run_id=data.get('run_id'),
            config=data.get('config', {}),
            cluster_id=data.get('cluster_id'),
            status=JobStatus(data.get('status', 'pending')),
            progress=data.get('progress', 0),
            current_step=data.get('current_step', ''),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            submitted_at=datetime.fromisoformat(data['submitted_at']) if data.get('submitted_at') else None,
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            mlflow_run_id=data.get('mlflow_run_id'),
            results=data.get('results'),
            error=data.get('error'),
            retry_count=data.get('retry_count', 0)
        )


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
    async def list_jobs(
        self,
        user_id: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50
    ) -> List[TrainingJob]:
        """List jobs, optionally filtered by user and/or status."""
        pass

    @abstractmethod
    async def delete(self, job_id: str) -> None:
        """Delete a job."""
        pass

    @abstractmethod
    async def count_running_jobs(self, user_id: Optional[str] = None) -> int:
        """Count jobs in running/submitting state."""
        pass


class SQLiteJobStateStore(JobStateStore):
    """
    SQLite-based state store for job persistence.

    Thread-safe and supports concurrent access.
    Suitable for single-instance deployment (Databricks Apps).
    """

    def __init__(self, db_path: str = "jobs.db"):
        self.db_path = db_path
        self._init_db()

    @log_io
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    run_id TEXT,
                    config TEXT NOT NULL,
                    cluster_id TEXT,
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
                    retry_count INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_status
                ON training_jobs(user_id, status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON training_jobs(status)
            """)
            conn.commit()
            logger.info(f"SQLite job state store initialized at {self.db_path}")
        finally:
            conn.close()

    @log_io
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @log_io
    async def save(self, job: TrainingJob) -> None:
        """Save or update a job."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO training_jobs (
                    job_id, user_id, run_id, config, cluster_id,
                    status, progress, current_step, created_at, submitted_at,
                    started_at, completed_at, mlflow_run_id, results, error,
                    retry_count, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id,
                job.user_id,
                job.run_id,
                json.dumps(job.config),
                job.cluster_id,
                job.status.value,
                job.progress,
                job.current_step,
                job.created_at.isoformat() if job.created_at else datetime.utcnow().isoformat(),
                job.submitted_at.isoformat() if job.submitted_at else None,
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.mlflow_run_id,
                json.dumps(job.results) if job.results else None,
                job.error,
                job.retry_count,
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            logger.debug(f"Saved job {job.job_id} with status {job.status.value}")
        finally:
            conn.close()

    @log_io
    async def get(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM training_jobs WHERE job_id = ?",
                (job_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_job(row)
            return None
        finally:
            conn.close()

    @log_io
    async def list_jobs(
        self,
        user_id: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50
    ) -> List[TrainingJob]:
        """List jobs with optional filters."""
        conn = self._get_connection()
        try:
            conditions = []
            params = []

            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            if status:
                conditions.append("status = ?")
                params.append(status.value)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            params.append(limit)

            cursor = conn.execute(
                f"""SELECT * FROM training_jobs
                    WHERE {where_clause}
                    ORDER BY created_at DESC LIMIT ?""",
                params
            )
            rows = cursor.fetchall()
            return [self._row_to_job(row) for row in rows]
        finally:
            conn.close()

    @log_io
    async def delete(self, job_id: str) -> None:
        """Delete a job."""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM training_jobs WHERE job_id = ?", (job_id,))
            conn.commit()
            logger.info(f"Deleted job {job_id}")
        finally:
            conn.close()

    @log_io
    async def get_running_jobs(self) -> List[TrainingJob]:
        """Get all jobs currently in running state."""
        return await self.list_jobs(status=JobStatus.RUNNING, limit=100)

    @log_io
    async def count_running_jobs(self, user_id: Optional[str] = None) -> int:
        """Count jobs in running/submitting state, optionally filtered by user."""
        conn = self._get_connection()
        try:
            if user_id:
                cursor = conn.execute(
                    """SELECT COUNT(*) FROM training_jobs
                       WHERE user_id = ? AND status IN (?, ?)""",
                    (user_id, JobStatus.RUNNING.value, JobStatus.SUBMITTING.value)
                )
            else:
                cursor = conn.execute(
                    """SELECT COUNT(*) FROM training_jobs
                       WHERE status IN (?, ?)""",
                    (JobStatus.RUNNING.value, JobStatus.SUBMITTING.value)
                )
            return cursor.fetchone()[0]
        finally:
            conn.close()

    @log_io
    def _row_to_job(self, row: sqlite3.Row) -> TrainingJob:
        """Convert a database row to TrainingJob."""
        return TrainingJob(
            job_id=row["job_id"],
            user_id=row["user_id"],
            run_id=row["run_id"],
            config=json.loads(row["config"]) if row["config"] else {},
            cluster_id=row["cluster_id"],
            status=JobStatus(row["status"]),
            progress=row["progress"],
            current_step=row["current_step"] or "",
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            submitted_at=datetime.fromisoformat(row["submitted_at"]) if row["submitted_at"] else None,
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            mlflow_run_id=row["mlflow_run_id"],
            results=json.loads(row["results"]) if row["results"] else None,
            error=row["error"],
            retry_count=row["retry_count"] or 0
        )
