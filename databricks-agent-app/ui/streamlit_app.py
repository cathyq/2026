"""
Streamlit Chat UI for Databricks Agent App.

A modern chat interface for interacting with the AI agent
that has both short-term and long-term memory capabilities.
"""

import os
import uuid
import requests
import streamlit as st
from typing import Generator

# API endpoint configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def generate_thread_id() -> str:
    """Generate a unique thread ID for new conversations."""
    return f"thread_{uuid.uuid4().hex[:12]}"


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = generate_thread_id()
    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo_user"


def send_message(message: str) -> str:
    """Send a message to the agent API and get a response."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat",
            json={
                "message": message,
                "thread_id": st.session_state.thread_id,
                "user_id": st.session_state.user_id,
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["message"]
    except requests.exceptions.RequestException as e:
        return f"Error communicating with the agent: {str(e)}"


def stream_message(message: str) -> Generator[str, None, None]:
    """Stream a message response from the agent API."""
    try:
        with requests.post(
            f"{API_BASE_URL}/api/v1/chat/stream",
            json={
                "message": message,
                "thread_id": st.session_state.thread_id,
                "user_id": st.session_state.user_id,
            },
            stream=True,
            timeout=60,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: ") and "[DONE]" not in line:
                        yield line[6:]
    except requests.exceptions.RequestException as e:
        yield f"Error: {str(e)}"


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

    # Custom CSS for better chat appearance
    st.markdown(
        """
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .sidebar-info {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .memory-tip {
            font-size: 0.9rem;
            color: #666;
            padding: 0.5rem;
            background-color: #e8f4f8;
            border-radius: 0.25rem;
            margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        # User ID input
        new_user_id = st.text_input(
            "User ID",
            value=st.session_state.user_id,
            help="Your unique identifier for long-term memory",
        )
        if new_user_id != st.session_state.user_id:
            st.session_state.user_id = new_user_id

        # Thread ID display
        st.text_input(
            "Thread ID",
            value=st.session_state.thread_id,
            disabled=True,
            help="Current conversation thread identifier",
        )

        # New conversation button
        if st.button("🔄 New Conversation", use_container_width=True):
            clear_conversation()
            st.rerun()

        st.divider()

        # Memory information
        st.subheader("💡 Memory Tips")
        st.markdown(
            """
            The agent remembers information you share:

            - **Your name**: *"My name is Alex"*
            - **Preferences**: *"I prefer Python"*
            - **Work info**: *"I work at Acme Corp"*
            - **Explicit requests**: *"Remember that..."*

            This information persists across conversations!
            """
        )

        st.divider()

        # About section
        st.subheader("ℹ️ About")
        st.markdown(
            """
            **Databricks Agent App**

            An AI assistant with:
            - 🧠 Short-term memory (within conversation)
            - 💾 Long-term memory (across conversations)
            - ⚡ Powered by LangGraph & Lakebase
            """
        )

    # Main chat area
    st.title("🤖 Databricks Agent Chat")
    st.caption("AI Assistant with Long-Term Memory")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # Option 1: Non-streaming response
            response = send_message(prompt)
            message_placeholder.markdown(response)

            # Option 2: Streaming response (uncomment to use)
            # full_response = ""
            # for chunk in stream_message(prompt):
            #     full_response += chunk
            #     message_placeholder.markdown(full_response + "▌")
            # message_placeholder.markdown(full_response)
            # response = full_response

        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
