"""
Databricks Agent App - Main FastAPI Application

A production-ready AI agent application with long-term memory capabilities
using Databricks Lakebase for persistence.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .core.config import settings
from .core.database import db_manager
from .agent import create_agent
from .routers.v1 import chat_router, health_router
from .routers.v1.chat import set_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Application startup initiated...")

    try:
        # Initialize database connection pool
        await db_manager.initialize()
        logger.info("Database connection pool initialized")

        # Create and configure the agent
        agent = create_agent()
        set_agent(agent)
        logger.info("AI Agent initialized with Lakebase memory")

        logger.info("Application startup completed successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Application shutdown initiated...")
    await db_manager.close()
    logger.info("Application shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="Databricks Agent App",
    description=(
        "AI Agent with Long-Term Memory using Databricks Lakebase. "
        "Features short-term memory (within conversation) and long-term memory "
        "(across conversations) powered by LangGraph and PostgreSQL."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."},
    )


# Include routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Databricks Agent App",
        "version": "1.0.0",
        "description": "AI Agent with Long-Term Memory",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# Simple health check at root level
@app.get("/health")
async def simple_health():
    """Simple health check for load balancers."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
