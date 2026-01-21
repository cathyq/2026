"""
Standalone Streamlit Chat Application for Databricks Agent.

This app directly integrates with the LangGraph agent and Lakebase,
making it suitable for deployment as a single Databricks App.
"""

import os
import sys
import uuid
import logging
from typing import Optional

import streamlit as st

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.config import settings
from src.core.database import db_manager
from src.agent.memory_agent import MemoryAgent, create_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_thread_id() -> str:
    """Generate a unique thread ID for new conversations."""
    return f"thread_{uuid.uuid4().hex[:12]}"


@st.cache_resource
def get_agent() -> Optional[MemoryAgent]:
    """
    Get or create the agent instance.
    Uses Streamlit's caching to ensure only one agent is created.
    """
    try:
        logger.info("Initializing agent...")
        agent = create_agent()
        logger.info("Agent initialized successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return None


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = generate_thread_id()
    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo_user"


def clear_conversation():
    """Clear the conversation and start fresh."""
    st.session_state.messages = []
    st.session_state.thread_id = generate_thread_id()


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Databricks Agent Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    init_session_state()

    # Custom CSS
    st.markdown(
        """
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .status-box {
            padding: 0.75rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .status-ok {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
        }
        .status-error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize agent
    agent = get_agent()

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        # Connection status
        if agent:
            st.markdown(
                '<div class="status-box status-ok">✅ Agent Connected</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-box status-error">❌ Agent Unavailable</div>',
                unsafe_allow_html=True,
            )
            st.error(
                "Could not connect to Lakebase. Please check your configuration."
            )

        st.divider()

        # User ID input
        new_user_id = st.text_input(
            "👤 User ID",
            value=st.session_state.user_id,
            help="Your unique identifier for long-term memory. Memories are stored per user.",
        )
        if new_user_id != st.session_state.user_id:
            st.session_state.user_id = new_user_id
            st.rerun()

        # Thread ID display
        st.text_input(
            "🧵 Thread ID",
            value=st.session_state.thread_id,
            disabled=True,
            help="Current conversation thread. Each thread has its own short-term memory.",
        )

        # New conversation button
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

        # Memory information
        st.subheader("🧠 Memory System")

        with st.expander("How it works", expanded=False):
            st.markdown(
                """
                **Short-term Memory** (per thread):
                - Remembers context within this conversation
                - Stored in PostgresSaver checkpoints
                - Cleared when you start a new chat

                **Long-term Memory** (per user):
                - Remembers important info across all chats
                - Stored in PostgresStore by user namespace
                - Persists forever until manually cleared
                """
            )

        with st.expander("Try these prompts", expanded=True):
            st.markdown(
                """
                Share information the agent will remember:

                - *"My name is Alex"*
                - *"I prefer Python over JavaScript"*
                - *"I work at Acme Corporation"*
                - *"Remember that my favorite color is blue"*

                Then start a **new conversation** and ask:
                - *"What's my name?"*
                - *"What do you know about me?"*
                """
            )

        st.divider()

        # Configuration info
        with st.expander("📋 Configuration"):
            st.code(
                f"""
Lakebase Instance: {settings.lakebase_instance_name or 'Not set'}
Database: {settings.lakebase_database_name or 'Not set'}
Model: {settings.databricks_model_endpoint}
Pool Size: {settings.db_pool_size}
                """.strip()
            )

    # Main chat area
    st.title("🤖 Databricks Agent Chat")
    st.caption(
        f"AI Assistant with Long-Term Memory | User: `{st.session_state.user_id}` | Thread: `{st.session_state.thread_id}`"
    )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here...", disabled=not agent):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            if agent:
                with st.spinner("Thinking..."):
                    try:
                        response = agent.invoke(
                            message=prompt,
                            thread_id=st.session_state.thread_id,
                            user_id=st.session_state.user_id,
                        )
                        st.markdown(response)
                    except Exception as e:
                        response = f"I encountered an error: {str(e)}"
                        st.error(response)
                        logger.error(f"Agent error: {e}")
            else:
                response = "Agent is not available. Please check the configuration."
                st.error(response)

        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Footer
    st.divider()
    st.caption(
        "Built with Databricks Apps, LangGraph, and Lakebase | "
        "[Documentation](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/)"
    )


if __name__ == "__main__":
    main()
