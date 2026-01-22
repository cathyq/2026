"""
MLflow Agent Server implementation for Databricks Agent App.

This module defines the agent endpoints using MLflow's @invoke and @stream
decorators for seamless integration with MLflow Agent Server.

Usage:
    # Start the server
    python start_server.py

    # Or with uvicorn directly
    uvicorn start_server:app --reload --port 8000
"""

import os
import sys
import logging
from typing import Generator

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from src.agent.responses_agent import get_responses_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@invoke()
async def handle_invoke(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """
    Handle non-streaming invocation requests.

    This function is registered with MLflow's Agent Server and will be
    called for requests to the /invocations endpoint without streaming.

    Args:
        request: The ResponsesAgentRequest containing user input and context

    Returns:
        ResponsesAgentResponse with the agent's response
    """
    logger.info("Received invoke request")
    agent = get_responses_agent()
    return agent.invoke(request)


@stream()
async def handle_stream(
    request: ResponsesAgentRequest,
) -> Generator[ResponsesAgentStreamEvent, None, None]:
    """
    Handle streaming invocation requests.

    This function is registered with MLflow's Agent Server and will be
    called for requests to the /invocations endpoint with stream=true.

    Args:
        request: The ResponsesAgentRequest containing user input and context

    Yields:
        ResponsesAgentStreamEvent objects with streamed content chunks
    """
    logger.info("Received stream request")
    agent = get_responses_agent()
    for event in agent.stream(request):
        yield event
