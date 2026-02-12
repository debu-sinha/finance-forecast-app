"""
Lakebase PostgreSQL Client for Finance Forecasting Platform.

Provides async database connectivity to Databricks Lakebase (managed PostgreSQL)
with connection pooling, automatic retries, and observability.

Lakebase Features Used:
- Scale-to-zero: Auto-suspends after inactivity
- <10ms latency: Fast reads for session validation
- >10K QPS: High concurrency for 30+ users
- ACID transactions: Reliable state management

References:
- https://docs.databricks.com/aws/en/oltp/
- https://www.databricks.com/product/lakebase
"""

import asyncio
import hashlib
import json
import logging
import os

from backend.utils.logging_utils import log_io
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union
from uuid import UUID, uuid4

try:
    import asyncpg
    from asyncpg import Pool, Connection
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None
    Pool = None
    Connection = None

try:
    import psycopg2
    from psycopg2 import pool as psycopg2_pool
    from psycopg2.extras import RealDictCursor, execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None
    psycopg2_pool = None

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass(frozen=True)
class LakebaseConfig:
    """Configuration for Lakebase PostgreSQL connection."""

    host: str = field(default_factory=lambda: os.getenv("LAKEBASE_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("LAKEBASE_PORT", "5432")))
    database: str = field(default_factory=lambda: os.getenv("LAKEBASE_DATABASE", "forecast"))
    user: str = field(default_factory=lambda: os.getenv("LAKEBASE_USER", "forecast_app"))
    password: str = field(default_factory=lambda: os.getenv("LAKEBASE_PASSWORD", ""))

    # Connection pool settings
    min_pool_size: int = 2
    max_pool_size: int = 20
    connection_timeout: float = 30.0
    command_timeout: float = 60.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # SSL for Lakebase (always enabled in production)
    ssl_mode: str = field(default_factory=lambda: os.getenv("LAKEBASE_SSL_MODE", "require"))

    @property
    def dsn(self) -> str:
        """Generate PostgreSQL DSN string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def asyncpg_dsn(self) -> str:
        """DSN for asyncpg (includes SSL)."""
        ssl_param = f"?sslmode={self.ssl_mode}" if self.ssl_mode else ""
        return f"{self.dsn}{ssl_param}"


class LakebaseClient:
    """
    Async PostgreSQL client for Databricks Lakebase.

    Provides connection pooling, automatic retries, and CRUD operations
    for the forecast schema tables.

    Usage:
        async with LakebaseClient() as client:
            sessions = await client.get_active_sessions()

        # Or manual lifecycle
        client = LakebaseClient()
        await client.connect()
        try:
            result = await client.execute("SELECT * FROM forecast.sessions")
        finally:
            await client.close()
    """

    def __init__(self, config: Optional[LakebaseConfig] = None):
        self.config = config or LakebaseConfig()
        self._pool: Optional[Pool] = None
        self._sync_pool = None
        self._connected = False

    @log_io
    async def connect(self) -> None:
        """Establish connection pool to Lakebase."""
        if self._connected:
            return

        if not ASYNCPG_AVAILABLE:
            logger.warning("asyncpg not available, falling back to sync mode")
            self._setup_sync_pool()
            self._connected = True
            return

        try:
            self._pool = await asyncpg.create_pool(
                dsn=self.config.asyncpg_dsn,
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
                command_timeout=self.config.command_timeout,
                timeout=self.config.connection_timeout,
            )
            self._connected = True
            logger.info(
                f"Connected to Lakebase at {self.config.host}:{self.config.port}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Lakebase: {e}")
            raise

    @log_io
    def _setup_sync_pool(self) -> None:
        """Setup synchronous connection pool as fallback."""
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("Neither asyncpg nor psycopg2 available")

        self._sync_pool = psycopg2_pool.ThreadedConnectionPool(
            minconn=self.config.min_pool_size,
            maxconn=self.config.max_pool_size,
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
            sslmode=self.config.ssl_mode,
        )

    @log_io
    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

        if self._sync_pool:
            self._sync_pool.closeall()
            self._sync_pool = None

        self._connected = False
        logger.info("Disconnected from Lakebase")

    async def __aenter__(self) -> "LakebaseClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    @asynccontextmanager
    @log_io
    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self._connected:
            await self.connect()

        if self._pool:
            async with self._pool.acquire() as conn:
                yield conn
        else:
            # Sync fallback
            conn = self._sync_pool.getconn()
            try:
                yield conn
            finally:
                self._sync_pool.putconn(conn)

    @log_io
    async def execute(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> str:
        """Execute a query without returning results."""
        async with self.acquire() as conn:
            if self._pool:
                return await conn.execute(query, *args, timeout=timeout)
            else:
                with conn.cursor() as cur:
                    cur.execute(query, args)
                    conn.commit()
                    return cur.statusmessage

    @log_io
    async def fetch(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Execute a query and return all results as list of dicts."""
        async with self.acquire() as conn:
            if self._pool:
                rows = await conn.fetch(query, *args, timeout=timeout)
                return [dict(row) for row in rows]
            else:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, args)
                    return cur.fetchall()

    @log_io
    async def fetchrow(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute a query and return first result as dict."""
        async with self.acquire() as conn:
            if self._pool:
                row = await conn.fetchrow(query, *args, timeout=timeout)
                return dict(row) if row else None
            else:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, args)
                    row = cur.fetchone()
                    return dict(row) if row else None

    @log_io
    async def fetchval(
        self,
        query: str,
        *args,
        column: int = 0,
        timeout: Optional[float] = None
    ) -> Any:
        """Execute a query and return a single value."""
        async with self.acquire() as conn:
            if self._pool:
                return await conn.fetchval(query, *args, column=column, timeout=timeout)
            else:
                with conn.cursor() as cur:
                    cur.execute(query, args)
                    row = cur.fetchone()
                    return row[column] if row else None

    @asynccontextmanager
    @log_io
    async def transaction(self):
        """Execute operations within a transaction."""
        async with self.acquire() as conn:
            if self._pool:
                async with conn.transaction():
                    yield conn
            else:
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

    # =========================================================================
    # Session Operations
    # =========================================================================

    @log_io
    async def create_session(
        self,
        user_id: str,
        user_email: Optional[str] = None,
        session_config: Optional[Dict[str, Any]] = None,
        expires_hours: int = 24
    ) -> UUID:
        """
        Create a new user session.

        Args:
            user_id: Unique user identifier
            user_email: Optional email for audit
            session_config: Optional UI preferences
            expires_hours: Session expiration in hours

        Returns:
            session_id: UUID of created session
        """
        session_id = uuid4()
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
        config_json = json.dumps(session_config or {})

        await self.execute(
            """
            INSERT INTO forecast.sessions
            (session_id, user_id, user_email, session_config, expires_at)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            """,
            session_id,
            user_id,
            user_email,
            config_json,
            expires_at,
        )

        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id

    @log_io
    async def get_session(self, session_id: UUID) -> Optional[Dict[str, Any]]:
        """Get session by ID if active and not expired."""
        return await self.fetchrow(
            """
            SELECT *
            FROM forecast.sessions
            WHERE session_id = $1
              AND is_active = TRUE
              AND expires_at > NOW()
            """,
            session_id,
        )

    @log_io
    async def update_session_activity(self, session_id: UUID) -> None:
        """Update last_active_at timestamp for a session."""
        await self.execute(
            """
            UPDATE forecast.sessions
            SET last_active_at = NOW(),
                request_count = request_count + 1
            WHERE session_id = $1
            """,
            session_id,
        )

    @log_io
    async def invalidate_session(self, session_id: UUID) -> None:
        """Mark a session as inactive."""
        await self.execute(
            """
            UPDATE forecast.sessions
            SET is_active = FALSE
            WHERE session_id = $1
            """,
            session_id,
        )

    @log_io
    async def get_user_sessions(
        self,
        user_id: str,
        include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """Get all sessions for a user."""
        if include_inactive:
            return await self.fetch(
                """
                SELECT *
                FROM forecast.sessions
                WHERE user_id = $1
                ORDER BY created_at DESC
                """,
                user_id,
            )
        else:
            return await self.fetch(
                """
                SELECT *
                FROM forecast.sessions
                WHERE user_id = $1
                  AND is_active = TRUE
                  AND expires_at > NOW()
                ORDER BY last_active_at DESC
                """,
                user_id,
            )

    @log_io
    async def cleanup_expired_sessions(self) -> int:
        """Mark expired sessions as inactive. Returns count."""
        result = await self.execute(
            """
            UPDATE forecast.sessions
            SET is_active = FALSE
            WHERE is_active = TRUE
              AND expires_at < NOW()
            """
        )
        count = int(result.split()[-1]) if result else 0
        logger.info(f"Cleaned up {count} expired sessions")
        return count

    # =========================================================================
    # Execution History Operations
    # =========================================================================

    @log_io
    async def create_execution(
        self,
        session_id: UUID,
        user_id: str,
        request_params: Dict[str, Any],
        time_col: str,
        target_col: str,
        horizon: int,
        frequency: str,
        models: List[str],
        confidence_level: float = 0.95,
        random_seed: int = 42,
        hyperparameter_filters: Optional[Dict] = None,
        data_upload_id: Optional[UUID] = None,
        data_row_count: Optional[int] = None,
        data_hash: Optional[str] = None,
    ) -> UUID:
        """
        Create execution history record when job is submitted.

        Returns:
            job_id: UUID of created execution record
        """
        job_id = uuid4()

        await self.execute(
            """
            INSERT INTO forecast.execution_history
            (job_id, session_id, user_id, request_params, time_col, target_col,
             horizon, frequency, models, confidence_level, random_seed,
             hyperparameter_filters, data_upload_id, data_row_count, data_hash,
             status)
            VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9, $10, $11,
                    $12::jsonb, $13, $14, $15, 'PENDING')
            """,
            job_id,
            session_id,
            user_id,
            json.dumps(request_params, default=str),
            time_col,
            target_col,
            horizon,
            frequency,
            models,
            confidence_level,
            random_seed,
            json.dumps(hyperparameter_filters or {}),
            data_upload_id,
            data_row_count,
            data_hash,
        )

        logger.info(f"Created execution {job_id} for user {user_id}")
        return job_id

    @log_io
    async def get_execution(self, job_id: UUID) -> Optional[Dict[str, Any]]:
        """Get execution details by job ID."""
        return await self.fetchrow(
            "SELECT * FROM forecast.execution_history WHERE job_id = $1",
            job_id,
        )

    @log_io
    async def update_execution_status(
        self,
        job_id: UUID,
        status: str,
        databricks_run_id: Optional[int] = None,
        progress_percent: Optional[int] = None,
        current_step: Optional[str] = None,
        error_message: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        best_model: Optional[str] = None,
        best_mape: Optional[float] = None,
    ) -> None:
        """Update execution status and related fields."""
        updates = ["status = $2", "updated_at = NOW()"]
        params = [job_id, status]
        param_idx = 3

        # Timestamp updates based on status
        if status == "QUEUED":
            updates.append("queued_at = NOW()")
        elif status == "RUNNING":
            updates.append("started_at = NOW()")
        elif status in ("COMPLETED", "FAILED", "CANCELLED"):
            updates.append("completed_at = NOW()")

        # Optional field updates
        optional_fields = [
            ("databricks_run_id", databricks_run_id),
            ("progress_percent", progress_percent),
            ("current_step", current_step),
            ("error_message", error_message),
            ("mlflow_run_id", mlflow_run_id),
            ("best_model", best_model),
            ("best_mape", best_mape),
        ]

        for field_name, value in optional_fields:
            if value is not None:
                updates.append(f"{field_name} = ${param_idx}")
                params.append(value)
                param_idx += 1

        query = f"""
            UPDATE forecast.execution_history
            SET {', '.join(updates)}
            WHERE job_id = $1
        """

        await self.execute(query, *params)

    @log_io
    async def get_user_history(
        self,
        user_id: str,
        limit: int = 50,
        status_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get execution history for a user."""
        if status_filter:
            return await self.fetch(
                """
                SELECT job_id, submitted_at, status, best_model, best_mape,
                       duration_seconds, horizon, frequency, models,
                       mlflow_run_url, progress_percent, current_step
                FROM forecast.execution_history
                WHERE user_id = $1 AND status = ANY($2)
                ORDER BY submitted_at DESC
                LIMIT $3
                """,
                user_id,
                status_filter,
                limit,
            )
        else:
            return await self.fetch(
                """
                SELECT job_id, submitted_at, status, best_model, best_mape,
                       duration_seconds, horizon, frequency, models,
                       mlflow_run_url, progress_percent, current_step
                FROM forecast.execution_history
                WHERE user_id = $1
                ORDER BY submitted_at DESC
                LIMIT $2
                """,
                user_id,
                limit,
            )

    @log_io
    async def get_active_jobs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get currently running or pending jobs."""
        if user_id:
            return await self.fetch(
                """
                SELECT * FROM forecast.v_active_jobs
                WHERE user_id = $1
                """,
                user_id,
            )
        else:
            return await self.fetch("SELECT * FROM forecast.v_active_jobs")

    @log_io
    async def get_reproduction_params(self, job_id: UUID) -> Dict[str, Any]:
        """
        Get all parameters needed to reproduce an execution.

        Returns complete TrainRequest with data reference for exact reproduction.
        """
        execution = await self.get_execution(job_id)
        if not execution:
            raise ValueError(f"Execution {job_id} not found")

        return {
            "original_job_id": str(job_id),
            "request_params": execution["request_params"],
            "data_upload_id": str(execution["data_upload_id"]) if execution["data_upload_id"] else None,
            "data_hash": execution["data_hash"],
            "random_seed": execution["random_seed"],
            "confidence_level": execution["confidence_level"],
            "mlflow_run_id": execution["mlflow_run_id"],
            "original_submitted_at": execution["submitted_at"].isoformat() if execution["submitted_at"] else None,
        }

    # =========================================================================
    # Forecast Results Operations
    # =========================================================================

    @log_io
    async def save_forecast_result(
        self,
        job_id: UUID,
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
    ) -> UUID:
        """Save forecast result for a model."""
        result_id = uuid4()

        await self.execute(
            """
            INSERT INTO forecast.forecast_results
            (result_id, job_id, model_name, forecast_dates, predictions,
             lower_bounds, upper_bounds, validation_dates, validation_actuals,
             validation_predictions, mape, rmse, mae, r2, cv_mape,
             model_params, training_time_seconds, mlflow_run_id)
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7::jsonb,
                    $8::jsonb, $9::jsonb, $10::jsonb, $11, $12, $13, $14, $15,
                    $16::jsonb, $17, $18)
            """,
            result_id,
            job_id,
            model_name,
            json.dumps(forecast_dates),
            json.dumps(predictions),
            json.dumps(lower_bounds) if lower_bounds else None,
            json.dumps(upper_bounds) if upper_bounds else None,
            json.dumps(validation_dates) if validation_dates else None,
            json.dumps(validation_actuals) if validation_actuals else None,
            json.dumps(validation_predictions) if validation_predictions else None,
            mape,
            rmse,
            mae,
            r2,
            cv_mape,
            json.dumps(model_params) if model_params else None,
            training_time_seconds,
            mlflow_run_id,
        )

        return result_id

    @log_io
    async def get_job_results(self, job_id: UUID) -> List[Dict[str, Any]]:
        """Get all forecast results for a job, ordered by MAPE."""
        return await self.fetch(
            """
            SELECT *
            FROM forecast.forecast_results
            WHERE job_id = $1
            ORDER BY mape ASC NULLS LAST
            """,
            job_id,
        )

    @log_io
    async def update_model_rankings(self, job_id: UUID) -> None:
        """Update model rankings for a completed job."""
        await self.execute(
            "SELECT forecast.update_model_rankings($1)",
            job_id,
        )

    # =========================================================================
    # User Uploads Operations
    # =========================================================================

    @log_io
    async def create_upload(
        self,
        user_id: str,
        session_id: UUID,
        file_name: str,
        file_size_bytes: int,
        columns: List[Dict[str, Any]],
        row_count: int,
        data_hash: str,
        storage_path: Optional[str] = None,
        detected_time_col: Optional[str] = None,
        detected_target_col: Optional[str] = None,
        detected_frequency: Optional[str] = None,
        profile_json: Optional[Dict] = None,
    ) -> UUID:
        """Create record for uploaded data file."""
        upload_id = uuid4()

        await self.execute(
            """
            INSERT INTO forecast.user_uploads
            (upload_id, user_id, session_id, file_name, file_size_bytes,
             columns, row_count, data_hash, storage_path, detected_time_col,
             detected_target_col, detected_frequency, profile_json)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11, $12, $13::jsonb)
            """,
            upload_id,
            user_id,
            session_id,
            file_name,
            file_size_bytes,
            json.dumps(columns),
            row_count,
            data_hash,
            storage_path,
            detected_time_col,
            detected_target_col,
            detected_frequency,
            json.dumps(profile_json) if profile_json else None,
        )

        return upload_id

    @log_io
    async def get_upload(self, upload_id: UUID) -> Optional[Dict[str, Any]]:
        """Get upload by ID."""
        return await self.fetchrow(
            """
            SELECT *
            FROM forecast.user_uploads
            WHERE upload_id = $1 AND is_deleted = FALSE
            """,
            upload_id,
        )

    @log_io
    async def get_upload_by_hash(
        self,
        user_id: str,
        data_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Find existing upload with same data hash (deduplication)."""
        return await self.fetchrow(
            """
            SELECT *
            FROM forecast.user_uploads
            WHERE user_id = $1
              AND data_hash = $2
              AND is_deleted = FALSE
            ORDER BY uploaded_at DESC
            LIMIT 1
            """,
            user_id,
            data_hash,
        )

    # =========================================================================
    # Job Queue Operations
    # =========================================================================

    @log_io
    async def enqueue_job(
        self,
        job_id: UUID,
        user_id: str,
        priority: str = "NORMAL",
        estimated_duration: Optional[int] = None,
    ) -> UUID:
        """Add job to queue."""
        queue_id = uuid4()

        await self.execute(
            """
            INSERT INTO forecast.job_queue
            (queue_id, job_id, user_id, priority, estimated_duration_seconds)
            VALUES ($1, $2, $3, $4::forecast.queue_priority, $5)
            """,
            queue_id,
            job_id,
            user_id,
            priority,
            estimated_duration,
        )

        return queue_id

    @log_io
    async def dequeue_job(self, job_id: UUID) -> None:
        """Mark job as dequeued."""
        await self.execute(
            """
            UPDATE forecast.job_queue
            SET is_active = FALSE, dequeued_at = NOW()
            WHERE job_id = $1
            """,
            job_id,
        )

    @log_io
    async def get_queue_position(self, job_id: UUID) -> Optional[int]:
        """Get position in queue for a job."""
        return await self.fetchval(
            """
            SELECT COUNT(*) + 1
            FROM forecast.job_queue
            WHERE is_active = TRUE
              AND (priority > (SELECT priority FROM forecast.job_queue WHERE job_id = $1)
                   OR (priority = (SELECT priority FROM forecast.job_queue WHERE job_id = $1)
                       AND enqueued_at < (SELECT enqueued_at FROM forecast.job_queue WHERE job_id = $1)))
            """,
            job_id,
        )

    # =========================================================================
    # Audit Log Operations
    # =========================================================================

    @log_io
    async def log_audit(
        self,
        action: str,
        resource_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[UUID] = None,
        resource_id: Optional[UUID] = None,
        request_data: Optional[Dict] = None,
        response_data: Optional[Dict] = None,
        error_data: Optional[Dict] = None,
        duration_ms: Optional[int] = None,
    ) -> UUID:
        """Log an audit event."""
        log_id = uuid4()

        await self.execute(
            """
            INSERT INTO forecast.audit_log
            (log_id, user_id, session_id, action, resource_type, resource_id,
             request_data, response_data, error_data, duration_ms)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9::jsonb, $10)
            """,
            log_id,
            user_id,
            session_id,
            action,
            resource_type,
            resource_id,
            json.dumps(request_data) if request_data else None,
            json.dumps(response_data) if response_data else None,
            json.dumps(error_data) if error_data else None,
            duration_ms,
        )

        return log_id

    # =========================================================================
    # Utility Functions
    # =========================================================================

    @log_io
    async def health_check(self) -> Dict[str, Any]:
        """Check database connectivity and return status."""
        try:
            start = datetime.utcnow()
            result = await self.fetchval("SELECT 1")
            latency_ms = (datetime.utcnow() - start).total_seconds() * 1000

            pool_size = self._pool.get_size() if self._pool else 0
            pool_free = self._pool.get_idle_size() if self._pool else 0

            return {
                "status": "healthy",
                "connected": True,
                "latency_ms": round(latency_ms, 2),
                "pool_size": pool_size,
                "pool_free": pool_free,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
            }

    @log_io
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics for monitoring."""
        stats = await self.fetchrow(
            """
            SELECT
                (SELECT COUNT(*) FROM forecast.sessions WHERE is_active = TRUE) as active_sessions,
                (SELECT COUNT(*) FROM forecast.execution_history WHERE status = 'RUNNING') as running_jobs,
                (SELECT COUNT(*) FROM forecast.execution_history WHERE status = 'PENDING') as pending_jobs,
                (SELECT COUNT(*) FROM forecast.job_queue WHERE is_active = TRUE) as queued_jobs,
                (SELECT COUNT(*) FROM forecast.execution_history WHERE submitted_at > NOW() - INTERVAL '24 hours') as jobs_24h,
                (SELECT AVG(duration_seconds) FROM forecast.execution_history WHERE status = 'COMPLETED' AND completed_at > NOW() - INTERVAL '24 hours') as avg_duration_24h
            """
        )
        return dict(stats) if stats else {}


# Singleton instance for global access
_client: Optional[LakebaseClient] = None


@log_io
async def get_lakebase_client() -> LakebaseClient:
    """Get or create singleton Lakebase client."""
    global _client
    if _client is None:
        _client = LakebaseClient()
        await _client.connect()
    return _client


@log_io
async def close_lakebase_client() -> None:
    """Close singleton Lakebase client."""
    global _client
    if _client:
        await _client.close()
        _client = None


@log_io
def compute_data_hash(data: List[Dict[str, Any]]) -> str:
    """Compute SHA-256 hash of data for versioning."""
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()
