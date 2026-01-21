"""
Local Development Streamlit Chat Application.

This version uses in-memory storage instead of Lakebase,
making it suitable for local testing and development.
"""

import os
import sys
import uuid
import logging
from typing import Optional
from datetime import datetime

import streamlit as st

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# Try to import Databricks LLM, fall back to a simple mock
try:
    from langchain_databricks import ChatDatabricks
    HAS_DATABRICKS = True
except ImportError:
    HAS_DATABRICKS = False

try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(MessagesState):
    """State for the memory agent."""
    user_id: str = ""
    memories: list = []


class LocalMemoryAgent:
    """Local agent with in-memory storage for development."""

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.store = InMemoryStore()
        self.graph = None
        self._llm = None
        self._build_graph()

    @property
    def llm(self):
        """Get or create the LLM instance."""
        if self._llm is None:
            # Try different LLM options
            if HAS_DATABRICKS and os.getenv("DATABRICKS_HOST"):
                self._llm = ChatDatabricks(
                    endpoint=os.getenv("DATABRICKS_MODEL_ENDPOINT", "databricks-claude-3-5-sonnet"),
                    temperature=0.7,
                )
                logger.info("Using Databricks LLM")
            elif HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
                self._llm = ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=0.7,
                )
                logger.info("Using OpenAI LLM")
            else:
                # Mock LLM for demo
                self._llm = None
                logger.warning("No LLM configured - using mock responses")
        return self._llm

    def _retrieve_memories(self, state: AgentState) -> dict:
        """Retrieve relevant memories from in-memory store."""
        user_id = state.get("user_id", "default")
        namespace = ("memories", user_id)

        try:
            memories = list(self.store.search(namespace))
            memory_list = [
                {"key": m.key, "value": m.value}
                for m in memories
            ][:5]
            return {"memories": memory_list}
        except Exception as e:
            logger.warning(f"Error retrieving memories: {e}")
            return {"memories": []}

    def _call_model(self, state: AgentState) -> dict:
        """Call the LLM with context from memories."""
        messages = list(state["messages"])
        memories = state.get("memories", [])

        system_content = (
            "You are a helpful AI assistant with long-term memory capabilities. "
            "You can remember information from previous conversations."
        )

        if memories:
            memory_context = "\n\nRelevant information from previous conversations:\n"
            for mem in memories:
                memory_context += f"- {mem['value'].get('content', '')}\n"
            system_content += memory_context

        full_messages = [SystemMessage(content=system_content)] + messages

        if self.llm:
            response = self.llm.invoke(full_messages)
            return {"messages": [response]}
        else:
            # Mock response for demo
            last_msg = messages[-1].content if messages else ""
            mock_response = self._generate_mock_response(last_msg, memories)
            return {"messages": [AIMessage(content=mock_response)]}

    def _generate_mock_response(self, user_message: str, memories: list) -> str:
        """Generate a mock response for demo purposes."""
        user_lower = user_message.lower()

        # Check if asking about memories
        if any(q in user_lower for q in ["what do you know", "what's my name", "who am i", "remember"]):
            if memories:
                memory_info = "\n".join([f"- {m['value'].get('content', '')}" for m in memories])
                return f"Based on our previous conversations, I remember:\n{memory_info}"
            else:
                return "I don't have any stored memories about you yet. Tell me something about yourself!"

        # Check for personal info sharing
        if "my name is" in user_lower:
            name = user_lower.split("my name is")[-1].strip().split()[0].title()
            return f"Nice to meet you, {name}! I'll remember that. How can I help you today?"

        if "i work at" in user_lower or "i work for" in user_lower:
            return "That's great! I've noted your workplace. What would you like to know?"

        if "i like" in user_lower or "i prefer" in user_lower:
            return "Thanks for sharing your preferences! I'll keep that in mind for our future conversations."

        # Default response
        return (
            f"I received your message: '{user_message}'\n\n"
            "This is a demo mode without a real LLM. To enable full functionality:\n"
            "1. Set DATABRICKS_HOST and DATABRICKS_TOKEN for Databricks LLM\n"
            "2. Or set OPENAI_API_KEY for OpenAI\n\n"
            "Try telling me your name or preferences - I'll still remember them!"
        )

    def _store_memory(self, state: AgentState) -> dict:
        """Store important information as long-term memory."""
        user_id = state.get("user_id", "default")
        namespace = ("memories", user_id)

        messages = state["messages"]
        if len(messages) < 2:
            return {}

        # Get the last human message
        last_human = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human = msg
                break

        if not last_human:
            return {}

        # Check for memory triggers
        indicators = [
            "my name is", "i am", "i like", "i prefer",
            "i work", "i live", "remember", "don't forget", "important"
        ]

        content_lower = last_human.content.lower()
        should_store = any(ind in content_lower for ind in indicators)

        if should_store:
            memory_key = f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            memory_value = {
                "content": last_human.content,
                "timestamp": datetime.now().isoformat(),
            }
            self.store.put(namespace, key=memory_key, value=memory_value)
            logger.info(f"Stored memory '{memory_key}' for user {user_id}")

        return {}

    def _build_graph(self):
        """Build the LangGraph workflow."""
        builder = StateGraph(AgentState)

        builder.add_node("retrieve_memories", self._retrieve_memories)
        builder.add_node("call_model", self._call_model)
        builder.add_node("store_memory", self._store_memory)

        builder.add_edge(START, "retrieve_memories")
        builder.add_edge("retrieve_memories", "call_model")
        builder.add_edge("call_model", "store_memory")
        builder.add_edge("store_memory", END)

        self.graph = builder.compile(
            checkpointer=self.checkpointer,
        )

    def invoke(self, message: str, thread_id: str, user_id: str = "default") -> str:
        """Invoke the agent with a message."""
        config = {"configurable": {"thread_id": thread_id}}

        # Manually handle store since in-memory store doesn't auto-bind
        state = self.graph.get_state(config)

        input_state = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
        }

        result = self.graph.invoke(input_state, config)

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content

        return "I apologize, but I couldn't generate a response."


def generate_thread_id() -> str:
    """Generate a unique thread ID."""
    return f"thread_{uuid.uuid4().hex[:12]}"


@st.cache_resource
def get_agent() -> LocalMemoryAgent:
    """Get or create the agent instance."""
    return LocalMemoryAgent()


def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = generate_thread_id()
    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo_user"


def clear_conversation():
    """Clear the conversation."""
    st.session_state.messages = []
    st.session_state.thread_id = generate_thread_id()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Databricks Agent Chat (Local)",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # Custom CSS
    st.markdown("""
        <style>
        .status-box {
            padding: 0.75rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .status-ok { background-color: #d4edda; border: 1px solid #c3e6cb; }
        .status-warn { background-color: #fff3cd; border: 1px solid #ffeeba; }
        </style>
    """, unsafe_allow_html=True)

    agent = get_agent()

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        # Status indicator
        if agent.llm:
            st.markdown('<div class="status-box status-ok">✅ LLM Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box status-warn">⚠️ Demo Mode (No LLM)</div>', unsafe_allow_html=True)
            st.info("Set OPENAI_API_KEY or DATABRICKS_HOST to enable full LLM responses.")

        st.divider()

        # User ID
        new_user_id = st.text_input(
            "👤 User ID",
            value=st.session_state.user_id,
            help="Your unique identifier for long-term memory.",
        )
        if new_user_id != st.session_state.user_id:
            st.session_state.user_id = new_user_id
            st.rerun()

        # Thread ID
        st.text_input(
            "🧵 Thread ID",
            value=st.session_state.thread_id,
            disabled=True,
        )

        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Chat", use_container_width=True):
                clear_conversation()
                st.rerun()
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                clear_conversation()
                st.cache_resource.clear()
                st.rerun()

        st.divider()

        with st.expander("💡 Try these prompts", expanded=True):
            st.markdown("""
            - *"My name is Alex"*
            - *"I prefer Python over JavaScript"*
            - *"I work at Acme Corporation"*

            Then click **New Chat** and ask:
            - *"What's my name?"*
            - *"What do you know about me?"*
            """)

    # Main area
    st.title("🤖 Databricks Agent Chat")
    st.caption(f"User: `{st.session_state.user_id}` | Thread: `{st.session_state.thread_id}`")

    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = agent.invoke(
                        message=prompt,
                        thread_id=st.session_state.thread_id,
                        user_id=st.session_state.user_id,
                    )
                    st.markdown(response)
                except Exception as e:
                    response = f"Error: {str(e)}"
                    st.error(response)
                    logger.error(f"Agent error: {e}", exc_info=True)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
