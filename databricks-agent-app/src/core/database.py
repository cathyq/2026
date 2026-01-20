"""
Database connection management for Databricks Lakebase.
Handles OAuth token refresh and connection pooling.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

from databricks.sdk import WorkspaceClient
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from .config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections to Databricks Lakebase with automatic
    OAuth token refresh and connection pooling.
    """

    def __init__(self):
        self._pool: Optional[AsyncConnectionPool] = None
        self._workspace_client: Optional[WorkspaceClient] = None
        self._token_expiry: Optional[datetime] = None
        self._refresh_task: Optional[asyncio.Task] = None

    @property
    def workspace_client(self) -> WorkspaceClient:
        """Get or create the Databricks workspace client."""
        if self._workspace_client is None:
            self._workspace_client = WorkspaceClient()
        return self._workspace_client

    def _get_connection_string(self) -> str:
        """
        Generate PostgreSQL connection string for Lakebase.
        Uses OAuth token for authentication.
        """
        # Get Lakebase instance details
        instance = self.workspace_client.lakebase.get(
            instance_name=settings.lakebase_instance_name
        )

        # Get OAuth token for database authentication
        token_response = self.workspace_client.lakebase.get_database_access_token(
            instance_name=settings.lakebase_instance_name,
            database_name=settings.lakebase_database_name,
        )

        # Build connection string
        conn_string = (
            f"postgresql://{token_response.username}:{token_response.token}"
            f"@{instance.hostname}:{settings.databricks_database_port}"
            f"/{settings.lakebase_database_name}"
            f"?sslmode=require"
        )

        # Track token expiry for refresh
        self._token_expiry = datetime.now() + timedelta(
            seconds=settings.token_refresh_interval
        )

        return conn_string

    async def initialize(self) -> None:
        """Initialize the database connection pool."""
        logger.info("Initializing database connection pool...")

        conn_string = self._get_connection_string()

        self._pool = AsyncConnectionPool(
            conninfo=conn_string,
            min_size=1,
            max_size=settings.db_pool_size,
            max_idle=settings.db_pool_recycle_interval,
            kwargs={"row_factory": dict_row, "autocommit": True},
        )

        await self._pool.open()

        # Start background token refresh task
        self._refresh_task = asyncio.create_task(self._token_refresh_loop())

        logger.info("Database connection pool initialized successfully")

    async def _token_refresh_loop(self) -> None:
        """Background task to refresh OAuth token before expiry."""
        while True:
            try:
                # Sleep until 5 minutes before token expiry
                if self._token_expiry:
                    sleep_seconds = max(
                        (self._token_expiry - datetime.now()).total_seconds() - 300,
                        60,
                    )
                else:
                    sleep_seconds = settings.token_refresh_interval - 300

                await asyncio.sleep(sleep_seconds)

                logger.info("Refreshing database OAuth token...")
                await self._refresh_pool()

            except asyncio.CancelledError:
                logger.info("Token refresh task cancelled")
                break
            except Exception as e:
                logger.error(f"Error refreshing token: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error

    async def _refresh_pool(self) -> None:
        """Refresh the connection pool with new OAuth token."""
        if self._pool:
            # Close existing pool
            await self._pool.close()

        # Create new pool with fresh token
        conn_string = self._get_connection_string()
        self._pool = AsyncConnectionPool(
            conninfo=conn_string,
            min_size=1,
            max_size=settings.db_pool_size,
            max_idle=settings.db_pool_recycle_interval,
            kwargs={"row_factory": dict_row, "autocommit": True},
        )
        await self._pool.open()

        logger.info("Database connection pool refreshed with new token")

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        if not self._pool:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")

        async with self._pool.connection() as conn:
            yield conn

    def get_sync_connection_string(self) -> str:
        """
        Get a synchronous connection string for LangGraph checkpointer.
        This is needed because langgraph-checkpoint-postgres uses sync connections.
        """
        instance = self.workspace_client.lakebase.get(
            instance_name=settings.lakebase_instance_name
        )

        token_response = self.workspace_client.lakebase.get_database_access_token(
            instance_name=settings.lakebase_instance_name,
            database_name=settings.lakebase_database_name,
        )

        return (
            f"postgresql://{token_response.username}:{token_response.token}"
            f"@{instance.hostname}:{settings.databricks_database_port}"
            f"/{settings.lakebase_database_name}"
            f"?sslmode=require"
        )

    async def close(self) -> None:
        """Close the database connection pool and cleanup."""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

        if self._pool:
            await self._pool.close()
            logger.info("Database connection pool closed")

    async def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            async with self.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    result = await cur.fetchone()
                    return result is not None
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()
