"""
Entry point for the MLflow Agent Server.

This script initializes and starts the MLflow Agent Server with the
Databricks Memory Agent. It supports:
- Automatic request/response validation for ResponsesAgent schema
- Streaming support via SSE
- MLflow tracing integration
- Hot reloading for development

Usage:
    # Basic start
    python start_server.py

    # With custom port
    python start_server.py --port 8080

    # With multiple workers (production)
    python start_server.py --workers 4

    # Development mode with reload
    python start_server.py --reload
"""

import os
import sys
import logging
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import agent_server module to register @invoke and @stream functions
import agent_server  # noqa: F401

from mlflow.genai.agent_server import AgentServer

# Optional: Enable MLflow git-based version tracking
try:
    from mlflow.genai.agent_server import setup_mlflow_git_based_version_tracking

    setup_mlflow_git_based_version_tracking()
    logger.info("MLflow git-based version tracking enabled")
except ImportError:
    logger.warning("MLflow git-based version tracking not available")
except Exception as e:
    logger.warning(f"Could not enable git-based version tracking: {e}")


def create_app():
    """
    Create the MLflow Agent Server application.

    Returns:
        FastAPI application instance
    """
    # Create AgentServer with ResponsesAgent type for automatic
    # input/output validation and streaming tracing aggregation
    agent_server_instance = AgentServer(agent_type="ResponsesAgent")
    return agent_server_instance.app


# Create the app instance for uvicorn
app = create_app()


def main():
    """Main entry point for starting the server."""
    parser = argparse.ArgumentParser(
        description="Start the MLflow Agent Server for Databricks Memory Agent"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)",
    )

    args = parser.parse_args()

    logger.info(f"Starting MLflow Agent Server on {args.host}:{args.port}")
    logger.info("Endpoints:")
    logger.info(f"  - POST /invocations - Chat endpoint (streaming with stream=true)")
    logger.info(f"  - GET /health - Health check")
    logger.info(f"  - GET /version - Version info")

    # Create and run the server
    agent_server_instance = AgentServer(agent_type="ResponsesAgent")
    agent_server_instance.run(
        app_import_string="start_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
