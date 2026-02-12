"""
Session Manager for Finance Forecasting Platform.

Manages user sessions with Lakebase PostgreSQL persistence.
Provides session creation, validation, activity tracking, and cleanup.

Key Features:
- Session creation with configurable expiration
- Activity tracking for session keepalive
- Automatic cleanup of expired sessions
- User history retrieval across sessions
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from backend.utils.logging_utils import log_io

from backend.services.lakebase_client import LakebaseClient, get_lakebase_client

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages user sessions with Lakebase persistence.

    Usage:
        manager = SessionManager()
        await manager.initialize()

        # Create session
        session_id = await manager.create_session("user123", "user@example.com")

        # Validate session
        session = await manager.get_session(session_id)
        if session:
            await manager.update_activity(session_id)

        # Get user history
        history = await manager.get_user_history("user123")
    """

    def __init__(self, client: Optional[LakebaseClient] = None):
        """
        Initialize SessionManager.

        Args:
            client: Optional LakebaseClient instance. If not provided,
                   will use the global singleton on first operation.
        """
        self._client = client
        self._initialized = False

    @log_io
    async def _get_client(self) -> LakebaseClient:
        """Get or initialize Lakebase client."""
        if self._client is None:
            self._client = await get_lakebase_client()
        return self._client

    @log_io
    async def initialize(self) -> None:
        """Initialize the session manager and verify database connectivity."""
        client = await self._get_client()
        health = await client.health_check()

        if health["status"] != "healthy":
            raise RuntimeError(f"Lakebase connection unhealthy: {health}")

        self._initialized = True
        logger.info("SessionManager initialized successfully")

    @log_io
    async def create_session(
        self,
        user_id: str,
        user_email: Optional[str] = None,
        session_config: Optional[Dict[str, Any]] = None,
        expires_hours: int = 24
    ) -> str:
        """
        Create a new user session.

        Args:
            user_id: Unique user identifier (e.g., Databricks username)
            user_email: Optional email for audit trail
            session_config: Optional UI preferences and settings
            expires_hours: Session expiration time in hours (default: 24)

        Returns:
            session_id: String UUID of the created session
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")

        client = await self._get_client()

        session_id = await client.create_session(
            user_id=user_id.strip(),
            user_email=user_email,
            session_config=session_config,
            expires_hours=expires_hours,
        )

        # Log audit event
        await client.log_audit(
            action="SESSION_CREATE",
            resource_type="session",
            user_id=user_id,
            resource_id=session_id,
            request_data={"expires_hours": expires_hours},
        )

        logger.info(f"Created session {session_id} for user {user_id}")
        return str(session_id)

    @log_io
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session by ID if active and not expired.

        Args:
            session_id: UUID string of the session

        Returns:
            Session data dict or None if not found/expired
        """
        try:
            uuid_session_id = UUID(session_id)
        except ValueError:
            logger.warning(f"Invalid session_id format: {session_id}")
            return None

        client = await self._get_client()
        session = await client.get_session(uuid_session_id)

        if session:
            # Convert UUID fields to strings for JSON serialization
            session["session_id"] = str(session["session_id"])
            if session.get("last_request_id"):
                session["last_request_id"] = str(session["last_request_id"])

        return session

    @log_io
    async def validate_session(self, session_id: str) -> bool:
        """
        Quick validation check if session exists and is active.

        Args:
            session_id: UUID string of the session

        Returns:
            True if session is valid, False otherwise
        """
        session = await self.get_session(session_id)
        return session is not None

    @log_io
    async def update_activity(self, session_id: str) -> None:
        """
        Update last_active_at timestamp for a session.

        Call this on every user request to keep session alive.

        Args:
            session_id: UUID string of the session
        """
        try:
            uuid_session_id = UUID(session_id)
        except ValueError:
            logger.warning(f"Invalid session_id format: {session_id}")
            return

        client = await self._get_client()
        await client.update_session_activity(uuid_session_id)

    @log_io
    async def update_config(
        self,
        session_id: str,
        config_updates: Dict[str, Any]
    ) -> None:
        """
        Update session configuration (UI preferences, etc.).

        Args:
            session_id: UUID string of the session
            config_updates: Dict of config keys to update
        """
        try:
            uuid_session_id = UUID(session_id)
        except ValueError:
            raise ValueError(f"Invalid session_id format: {session_id}")

        client = await self._get_client()

        # Merge with existing config
        await client.execute(
            """
            UPDATE forecast.sessions
            SET session_config = session_config || $2::jsonb,
                last_active_at = NOW()
            WHERE session_id = $1
            """,
            uuid_session_id,
            config_updates,
        )

    @log_io
    async def invalidate_session(self, session_id: str) -> None:
        """
        Mark a session as inactive (logout).

        Args:
            session_id: UUID string of the session
        """
        try:
            uuid_session_id = UUID(session_id)
        except ValueError:
            logger.warning(f"Invalid session_id format: {session_id}")
            return

        client = await self._get_client()
        await client.invalidate_session(uuid_session_id)

        # Log audit event
        await client.log_audit(
            action="SESSION_INVALIDATE",
            resource_type="session",
            resource_id=uuid_session_id,
        )

        logger.info(f"Invalidated session {session_id}")

    @log_io
    async def get_user_sessions(
        self,
        user_id: str,
        include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all sessions for a user.

        Args:
            user_id: User identifier
            include_inactive: Include expired/inactive sessions

        Returns:
            List of session dicts
        """
        client = await self._get_client()
        sessions = await client.get_user_sessions(user_id, include_inactive)

        # Convert UUIDs to strings
        for session in sessions:
            session["session_id"] = str(session["session_id"])
            if session.get("last_request_id"):
                session["last_request_id"] = str(session["last_request_id"])

        return sessions

    @log_io
    async def get_user_history(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get execution history for a user across all sessions.

        Args:
            user_id: User identifier
            limit: Maximum number of records to return

        Returns:
            List of execution history dicts ordered by submission time
        """
        client = await self._get_client()
        history = await client.get_user_history(user_id, limit)

        # Convert UUIDs to strings
        for record in history:
            record["job_id"] = str(record["job_id"])

        return history

    @log_io
    async def get_active_job_count(self, user_id: str) -> int:
        """
        Get count of active (running/pending) jobs for a user.

        Useful for rate limiting or showing user their active jobs.

        Args:
            user_id: User identifier

        Returns:
            Count of active jobs
        """
        client = await self._get_client()
        count = await client.fetchval(
            """
            SELECT COUNT(*)
            FROM forecast.execution_history
            WHERE user_id = $1
              AND status IN ('PENDING', 'QUEUED', 'RUNNING')
            """,
            user_id,
        )
        return count or 0

    @log_io
    async def cleanup_expired_sessions(self) -> int:
        """
        Mark all expired sessions as inactive.

        Should be called periodically (e.g., every hour) to clean up
        expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        client = await self._get_client()
        count = await client.cleanup_expired_sessions()

        if count > 0:
            await client.log_audit(
                action="SESSION_CLEANUP",
                resource_type="session",
                response_data={"cleaned_count": count},
            )

        return count

    @log_io
    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics for monitoring.

        Returns:
            Dict with active_sessions, total_users, etc.
        """
        client = await self._get_client()

        stats = await client.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE is_active AND expires_at > NOW()) as active_sessions,
                COUNT(DISTINCT user_id) FILTER (WHERE is_active AND expires_at > NOW()) as active_users,
                COUNT(*) as total_sessions,
                AVG(request_count) FILTER (WHERE is_active) as avg_requests_per_session,
                MAX(last_active_at) as most_recent_activity
            FROM forecast.sessions
            """
        )

        return dict(stats) if stats else {}


# Singleton instance
_session_manager: Optional[SessionManager] = None


@log_io
async def get_session_manager() -> SessionManager:
    """Get or create singleton SessionManager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
        await _session_manager.initialize()
    return _session_manager
