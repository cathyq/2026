"""
LangGraph Agent with Long-Term Memory using Lakebase.
Implements both short-term (thread-level) and long-term (cross-thread) memory.
"""

import logging
from typing import Annotated, Any, Generator, Optional
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_databricks import ChatDatabricks
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore

from ..core.config import settings
from ..core.database import db_manager

logger = logging.getLogger(__name__)


class AgentState(MessagesState):
    """State for the memory agent."""

    # User context for long-term memory
    user_id: str = ""
    # Memories retrieved from long-term storage
    memories: list[dict] = []


class MemoryAgent:
    """
    AI Agent with both short-term and long-term memory capabilities.

    Short-term memory: Persisted per conversation thread using PostgresSaver
    Long-term memory: Persisted across threads using PostgresStore with namespaces
    """

    def __init__(
        self,
        checkpointer: Optional[PostgresSaver] = None,
        store: Optional[PostgresStore] = None,
    ):
        self.checkpointer = checkpointer
        self.store = store
        self.graph: Optional[CompiledStateGraph] = None
        self._llm: Optional[ChatDatabricks] = None

    @property
    def llm(self) -> ChatDatabricks:
        """Get or create the LLM instance."""
        if self._llm is None:
            self._llm = ChatDatabricks(
                endpoint=settings.databricks_model_endpoint,
                temperature=0.7,
                max_tokens=2048,
            )
        return self._llm

    def _retrieve_memories(
        self, state: AgentState, config: RunnableConfig, *, store: BaseStore
    ) -> dict:
        """Retrieve relevant memories from long-term storage."""
        user_id = state.get("user_id", config.get("configurable", {}).get("user_id", "default"))

        if not store or not user_id:
            return {"memories": []}

        namespace = ("memories", user_id)

        try:
            # Get the last user message for semantic search
            last_message = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_message = msg.content
                    break

            # Search for relevant memories
            memories = store.search(namespace, query=last_message, limit=5)
            memory_list = [
                {"key": m.key, "value": m.value, "score": getattr(m, "score", 0)}
                for m in memories
            ]

            logger.info(f"Retrieved {len(memory_list)} memories for user {user_id}")
            return {"memories": memory_list}

        except Exception as e:
            logger.warning(f"Error retrieving memories: {e}")
            return {"memories": []}

    def _call_model(
        self, state: AgentState, config: RunnableConfig, *, store: BaseStore
    ) -> dict:
        """Call the LLM with context from memories."""
        messages = list(state["messages"])
        memories = state.get("memories", [])

        # Build system message with memory context
        system_content = settings.agent_system_prompt

        if memories:
            memory_context = "\n\nRelevant information from previous conversations:\n"
            for mem in memories:
                memory_context += f"- {mem['value'].get('content', '')}\n"
            system_content += memory_context

        # Prepend system message
        full_messages = [SystemMessage(content=system_content)] + messages

        # Call the LLM
        response = self.llm.invoke(full_messages)

        return {"messages": [response]}

    def _store_memory(
        self, state: AgentState, config: RunnableConfig, *, store: BaseStore
    ) -> dict:
        """Extract and store important information as long-term memory."""
        user_id = state.get("user_id", config.get("configurable", {}).get("user_id", "default"))

        if not store or not user_id:
            return {}

        namespace = ("memories", user_id)

        try:
            # Get the conversation context
            messages = state["messages"]
            if len(messages) < 2:
                return {}

            # Get the last exchange
            last_human = None
            last_ai = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and last_ai is None:
                    last_ai = msg
                elif isinstance(msg, HumanMessage) and last_human is None:
                    last_human = msg
                if last_human and last_ai:
                    break

            if not last_human or not last_ai:
                return {}

            # Simple heuristic: store if user shares personal info or preferences
            indicators = [
                "my name is",
                "i am",
                "i like",
                "i prefer",
                "i work",
                "i live",
                "remember",
                "don't forget",
                "important",
            ]

            content_lower = last_human.content.lower()
            should_store = any(ind in content_lower for ind in indicators)

            if should_store:
                memory_key = f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                memory_value = {
                    "content": last_human.content,
                    "response": last_ai.content[:500],  # Truncate long responses
                    "timestamp": datetime.now().isoformat(),
                }

                store.put(namespace, key=memory_key, value=memory_value)
                logger.info(f"Stored memory '{memory_key}' for user {user_id}")

        except Exception as e:
            logger.warning(f"Error storing memory: {e}")

        return {}

    def build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph workflow."""
        builder = StateGraph(AgentState)

        # Add nodes
        builder.add_node("retrieve_memories", self._retrieve_memories)
        builder.add_node("call_model", self._call_model)
        builder.add_node("store_memory", self._store_memory)

        # Define edges
        builder.add_edge(START, "retrieve_memories")
        builder.add_edge("retrieve_memories", "call_model")
        builder.add_edge("call_model", "store_memory")
        builder.add_edge("store_memory", END)

        # Compile with checkpointer and store
        self.graph = builder.compile(
            checkpointer=self.checkpointer,
            store=self.store,
        )

        return self.graph

    def invoke(
        self,
        message: str,
        thread_id: str,
        user_id: str = "default",
    ) -> str:
        """
        Invoke the agent with a message.

        Args:
            message: The user's message
            thread_id: Unique identifier for the conversation thread
            user_id: Unique identifier for the user (for long-term memory)

        Returns:
            The agent's response
        """
        if not self.graph:
            self.build_graph()

        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            }
        }

        input_state = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
        }

        result = self.graph.invoke(input_state, config)

        # Extract the last AI message
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content

        return "I apologize, but I couldn't generate a response."

    def stream(
        self,
        message: str,
        thread_id: str,
        user_id: str = "default",
    ) -> Generator[str, None, None]:
        """
        Stream the agent's response.

        Args:
            message: The user's message
            thread_id: Unique identifier for the conversation thread
            user_id: Unique identifier for the user

        Yields:
            Chunks of the agent's response
        """
        if not self.graph:
            self.build_graph()

        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            }
        }

        input_state = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
        }

        for event in self.graph.stream(input_state, config, stream_mode="messages"):
            if event and len(event) > 0:
                msg = event[0]
                if isinstance(msg, AIMessage) and msg.content:
                    yield msg.content

    def get_conversation_history(self, thread_id: str) -> list[dict]:
        """Get the conversation history for a thread."""
        if not self.checkpointer:
            return []

        config = {"configurable": {"thread_id": thread_id}}

        try:
            state = self.graph.get_state(config)
            if state and state.values:
                messages = state.values.get("messages", [])
                return [
                    {
                        "role": "user" if isinstance(m, HumanMessage) else "assistant",
                        "content": m.content,
                    }
                    for m in messages
                ]
        except Exception as e:
            logger.warning(f"Error getting conversation history: {e}")

        return []


def create_agent() -> MemoryAgent:
    """
    Factory function to create a MemoryAgent with Lakebase persistence.

    Returns:
        Configured MemoryAgent instance
    """
    # Get connection string from database manager
    conn_string = db_manager.get_sync_connection_string()

    # Create checkpointer for short-term memory
    checkpointer = PostgresSaver.from_conn_string(conn_string)

    # Create store for long-term memory
    store = PostgresStore.from_conn_string(conn_string)

    # Initialize tables (safe to call multiple times)
    try:
        checkpointer.setup()
        store.setup()
        logger.info("Database tables initialized for checkpointer and store")
    except Exception as e:
        logger.warning(f"Table setup warning (may already exist): {e}")

    # Create and return agent
    agent = MemoryAgent(checkpointer=checkpointer, store=store)
    agent.build_graph()

    return agent
