"""
Health check endpoints for the Databricks Agent App.
"""

from datetime import datetime
from fastapi import APIRouter, Depends

from ...core.database import db_manager
from ...models.chat import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        database=False,  # Quick check without DB
        agent=True,
        timestamp=datetime.now().isoformat(),
    )


@router.get("/database", response_model=HealthResponse)
async def database_health_check() -> HealthResponse:
    """Check database connectivity."""
    db_healthy = await db_manager.health_check()

    return HealthResponse(
        status="healthy" if db_healthy else "degraded",
        database=db_healthy,
        agent=True,
        timestamp=datetime.now().isoformat(),
    )


@router.get("/ready")
async def readiness_check() -> dict:
    """Kubernetes-style readiness probe."""
    db_healthy = await db_manager.health_check()

    if db_healthy:
        return {"status": "ready"}
    else:
        return {"status": "not_ready", "reason": "database_unavailable"}


@router.get("/live")
async def liveness_check() -> dict:
    """Kubernetes-style liveness probe."""
    return {"status": "alive"}
