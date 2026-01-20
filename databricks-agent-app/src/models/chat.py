"""
Pydantic models for chat API requests and responses.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single message in a conversation."""

    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., description="The user's message")
    thread_id: str = Field(
        default="default",
        description="Unique identifier for the conversation thread",
    )
    user_id: str = Field(
        default="default",
        description="Unique identifier for the user (for long-term memory)",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response",
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    message: str = Field(..., description="The assistant's response")
    thread_id: str = Field(..., description="The conversation thread ID")
    user_id: str = Field(..., description="The user ID")


class ConversationHistory(BaseModel):
    """Response model for conversation history."""

    thread_id: str
    messages: list[Message]


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    database: bool
    agent: bool
    timestamp: str
