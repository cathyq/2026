"""
MLflow ResponsesAgent implementation for the Databricks Agent App.

This module wraps the MemoryAgent to be compatible with MLflow's
ResponsesAgent interface for serving via MLflow Agent Server.
"""

import logging
from typing import Any, Generator, Optional
from datetime import datetime
import uuid

from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from mlflow.types.responses_helpers import (
    ResponseOutputMessage,
    ResponseOutputText,
)

from .memory_agent import MemoryAgent, create_agent

logger = logging.getLogger(__name__)


class ResponsesAgentWrapper:
    """
    Wrapper that adapts the MemoryAgent to work with MLflow's ResponsesAgent schema.

    This enables serving the agent via MLflow Agent Server with:
    - Automatic request/response validation
    - Streaming support
    - MLflow tracing integration
    """

    def __init__(self, agent: Optional[MemoryAgent] = None):
        """
        Initialize the ResponsesAgent wrapper.

        Args:
            agent: Optional MemoryAgent instance. If not provided,
                   one will be created when first needed.
        """
        self._agent = agent
        self._initialized = False

    @property
    def agent(self) -> MemoryAgent:
        """Get or create the MemoryAgent instance."""
        if self._agent is None:
            logger.info("Initializing MemoryAgent...")
            self._agent = create_agent()
            self._initialized = True
            logger.info("MemoryAgent initialized successfully")
        return self._agent

    def _extract_message_content(self, request: ResponsesAgentRequest) -> str:
        """
        Extract the user message content from a ResponsesAgentRequest.

        Args:
            request: The incoming request object

        Returns:
            The extracted message string
        """
        if not request.input:
            return ""

        # Get the last user message from input
        for item in reversed(request.input):
            if hasattr(item, "role") and item.role == "user":
                if hasattr(item, "content"):
                    content = item.content
                    # Content can be a string or a list of content items
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        # Extract text from content items
                        text_parts = []
                        for part in content:
                            if hasattr(part, "text"):
                                text_parts.append(part.text)
                            elif isinstance(part, dict) and "text" in part:
                                text_parts.append(part["text"])
                        return " ".join(text_parts)
            # Handle dict-style input
            elif isinstance(item, dict):
                if item.get("role") == "user":
                    content = item.get("content", "")
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and "text" in part:
                                text_parts.append(part["text"])
                        return " ".join(text_parts)

        return ""

    def _extract_context(self, request: ResponsesAgentRequest) -> tuple[str, str]:
        """
        Extract thread_id and user_id from request context.

        Args:
            request: The incoming request object

        Returns:
            Tuple of (thread_id, user_id)
        """
        thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        user_id = "default"

        if request.context:
            if hasattr(request.context, "conversation_id") and request.context.conversation_id:
                thread_id = request.context.conversation_id
            if hasattr(request.context, "user_id") and request.context.user_id:
                user_id = request.context.user_id

        # Also check custom_inputs for backward compatibility
        if request.custom_inputs:
            thread_id = request.custom_inputs.get("thread_id", thread_id)
            user_id = request.custom_inputs.get("user_id", user_id)

        return thread_id, user_id

    def _create_response(
        self,
        content: str,
        message_id: Optional[str] = None,
    ) -> ResponsesAgentResponse:
        """
        Create a ResponsesAgentResponse from the agent's output.

        Args:
            content: The response text content
            message_id: Optional message ID

        Returns:
            A properly formatted ResponsesAgentResponse
        """
        if message_id is None:
            message_id = f"msg_{uuid.uuid4().hex[:12]}"

        output_message = {
            "type": "message",
            "id": message_id,
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": content,
                }
            ],
        }

        return ResponsesAgentResponse(output=[output_message])

    def invoke(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Handle a non-streaming invocation request.

        Args:
            request: The ResponsesAgentRequest containing the user input

        Returns:
            ResponsesAgentResponse with the agent's response
        """
        # Extract message and context
        message = self._extract_message_content(request)
        thread_id, user_id = self._extract_context(request)

        logger.info(f"Invoking agent for thread={thread_id}, user={user_id}")

        if not message:
            return self._create_response(
                "I didn't receive a message. How can I help you?"
            )

        try:
            # Call the underlying MemoryAgent
            response = self.agent.invoke(
                message=message,
                thread_id=thread_id,
                user_id=user_id,
            )
            return self._create_response(response)

        except Exception as e:
            logger.error(f"Error invoking agent: {e}", exc_info=True)
            return self._create_response(
                f"I encountered an error processing your request: {str(e)}"
            )

    def stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Handle a streaming invocation request.

        Args:
            request: The ResponsesAgentRequest containing the user input

        Yields:
            ResponsesAgentStreamEvent objects with streamed content
        """
        # Extract message and context
        message = self._extract_message_content(request)
        thread_id, user_id = self._extract_context(request)
        message_id = f"msg_{uuid.uuid4().hex[:12]}"

        logger.info(f"Streaming agent response for thread={thread_id}, user={user_id}")

        if not message:
            # Yield a single response for empty input
            yield ResponsesAgentStreamEvent(
                type="response.output_text.delta",
                output_index=0,
                content_index=0,
                delta="I didn't receive a message. How can I help you?",
            )
            yield ResponsesAgentStreamEvent(
                type="response.output_text.done",
                output_index=0,
                content_index=0,
                text="I didn't receive a message. How can I help you?",
            )
            yield ResponsesAgentStreamEvent(
                type="response.completed",
                response=self._create_response(
                    "I didn't receive a message. How can I help you?",
                    message_id=message_id,
                ),
            )
            return

        try:
            # Stream from the underlying MemoryAgent
            full_response = ""
            for chunk in self.agent.stream(
                message=message,
                thread_id=thread_id,
                user_id=user_id,
            ):
                full_response += chunk
                yield ResponsesAgentStreamEvent(
                    type="response.output_text.delta",
                    output_index=0,
                    content_index=0,
                    delta=chunk,
                )

            # Send completion events
            yield ResponsesAgentStreamEvent(
                type="response.output_text.done",
                output_index=0,
                content_index=0,
                text=full_response,
            )
            yield ResponsesAgentStreamEvent(
                type="response.completed",
                response=self._create_response(full_response, message_id=message_id),
            )

        except Exception as e:
            logger.error(f"Error streaming agent response: {e}", exc_info=True)
            error_message = f"I encountered an error: {str(e)}"
            yield ResponsesAgentStreamEvent(
                type="response.output_text.delta",
                output_index=0,
                content_index=0,
                delta=error_message,
            )
            yield ResponsesAgentStreamEvent(
                type="response.output_text.done",
                output_index=0,
                content_index=0,
                text=error_message,
            )
            yield ResponsesAgentStreamEvent(
                type="response.completed",
                response=self._create_response(error_message, message_id=message_id),
            )


# Global wrapper instance for the agent server
_responses_agent: Optional[ResponsesAgentWrapper] = None


def get_responses_agent() -> ResponsesAgentWrapper:
    """Get or create the global ResponsesAgentWrapper instance."""
    global _responses_agent
    if _responses_agent is None:
        _responses_agent = ResponsesAgentWrapper()
    return _responses_agent
