"""
Chat API endpoints for the Databricks Agent App.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from ...models.chat import ChatRequest, ChatResponse, ConversationHistory, Message, MessageRole
from ...agent import MemoryAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

# Agent instance - will be set during app startup
_agent: Optional[MemoryAgent] = None


def get_agent() -> MemoryAgent:
    """Get the global agent instance."""
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Please wait for startup to complete.",
        )
    return _agent


def set_agent(agent: MemoryAgent) -> None:
    """Set the global agent instance."""
    global _agent
    _agent = agent


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the agent and get a response.

    The agent maintains both short-term memory (within the thread) and
    long-term memory (across threads for the same user).
    """
    agent = get_agent()

    try:
        if request.stream:
            # For streaming, redirect to the stream endpoint
            raise HTTPException(
                status_code=400,
                detail="For streaming responses, use the /chat/stream endpoint",
            )

        response = agent.invoke(
            message=request.message,
            thread_id=request.thread_id,
            user_id=request.user_id,
        )

        return ChatResponse(
            message=response,
            thread_id=request.thread_id,
            user_id=request.user_id,
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Send a message and stream the response using Server-Sent Events.
    """
    agent = get_agent()

    async def event_generator():
        try:
            for chunk in agent.stream(
                message=request.message,
                thread_id=request.thread_id,
                user_id=request.user_id,
            ):
                yield {"event": "message", "data": chunk}

            yield {"event": "done", "data": "[DONE]"}

        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())


@router.get("/history/{thread_id}", response_model=ConversationHistory)
async def get_history(thread_id: str) -> ConversationHistory:
    """
    Get the conversation history for a specific thread.
    """
    agent = get_agent()

    try:
        messages = agent.get_conversation_history(thread_id)

        return ConversationHistory(
            thread_id=thread_id,
            messages=[
                Message(role=MessageRole(m["role"]), content=m["content"])
                for m in messages
            ],
        )

    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{thread_id}")
async def clear_history(thread_id: str) -> dict:
    """
    Clear the conversation history for a specific thread.

    Note: This doesn't affect long-term memories stored for the user.
    """
    # In a full implementation, you would clear the checkpointer state
    # For now, we just acknowledge the request
    return {
        "status": "success",
        "message": f"Conversation history for thread '{thread_id}' will be cleared",
    }
