"""
Streamlit Chat UI for MLflow Agent Server.

This UI connects to the MLflow Agent Server's /invocations endpoint,
providing a visual interface for the ResponsesAgent API.

Usage:
    # First, start the agent server in one terminal:
    python start_server.py

    # Then, run this UI in another terminal:
    streamlit run ui/agent_server_chat.py
"""

import os
import uuid
import json
import logging
from typing import Generator

import streamlit as st
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default server URL
DEFAULT_SERVER_URL = os.getenv("AGENT_SERVER_URL", "http://localhost:8000")


def generate_ids() -> tuple[str, str]:
    """Generate unique thread and user IDs."""
    return f"thread_{uuid.uuid4().hex[:12]}", f"user_{uuid.uuid4().hex[:8]}"


def send_message(
    server_url: str,
    message: str,
    thread_id: str,
    user_id: str,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """
    Send a message to the MLflow Agent Server.

    Args:
        server_url: Base URL of the agent server
        message: User message to send
        thread_id: Conversation thread ID
        user_id: User ID for memory persistence
        stream: Whether to stream the response

    Returns:
        Response text or generator for streaming
    """
    url = f"{server_url}/invocations"

    payload = {
        "input": [{"role": "user", "content": message}],
        "context": {
            "conversation_id": thread_id,
            "user_id": user_id,
        },
        "stream": stream,
    }

    headers = {"Content-Type": "application/json"}

    if stream:
        return _stream_response(url, payload, headers)
    else:
        return _send_request(url, payload, headers)


def _send_request(url: str, payload: dict, headers: dict) -> str:
    """Send a non-streaming request."""
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()

        data = response.json()
        return _extract_response_text(data)

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the agent server. Make sure it's running with `python start_server.py`"
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The server might be overloaded."
    except requests.exceptions.HTTPError as e:
        return f"Error: Server returned {e.response.status_code}: {e.response.text}"
    except Exception as e:
        logger.error(f"Request error: {e}", exc_info=True)
        return f"Error: {str(e)}"


def _stream_response(url: str, payload: dict, headers: dict) -> Generator[str, None, None]:
    """Stream a response from the server."""
    try:
        with requests.post(
            url, json=payload, headers=headers, stream=True, timeout=120
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            event = json.loads(data_str)
                            if event.get("type") == "response.output_text.delta":
                                delta = event.get("delta", "")
                                if delta:
                                    yield delta
                        except json.JSONDecodeError:
                            continue

    except requests.exceptions.ConnectionError:
        yield "Error: Could not connect to the agent server."
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        yield f"Error: {str(e)}"


def _extract_response_text(data: dict) -> str:
    """Extract text from ResponsesAgentResponse."""
    try:
        output = data.get("output", [])
        if not output:
            return "No response received."

        for item in output:
            if item.get("type") == "message" and item.get("role") == "assistant":
                content = item.get("content", [])
                for content_item in content:
                    if content_item.get("type") == "output_text":
                        return content_item.get("text", "")

        return "Could not parse response."
    except Exception as e:
        logger.error(f"Parse error: {e}")
        return f"Error parsing response: {str(e)}"


def check_server_health(server_url: str) -> tuple[bool, str]:
    """Check if the agent server is running."""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            return True, "Connected"
        return False, f"Status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Not running"
    except Exception as e:
        return False, str(e)


def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        thread_id, user_id = generate_ids()
        st.session_state.thread_id = thread_id
        st.session_state.user_id = user_id
    if "server_url" not in st.session_state:
        st.session_state.server_url = DEFAULT_SERVER_URL
    if "use_streaming" not in st.session_state:
        st.session_state.use_streaming = True


def new_conversation():
    """Start a new conversation."""
    st.session_state.messages = []
    thread_id, _ = generate_ids()
    st.session_state.thread_id = thread_id


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Agent Server Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # Custom CSS
    st.markdown("""
        <style>
        .status-connected { color: #28a745; font-weight: bold; }
        .status-disconnected { color: #dc3545; font-weight: bold; }
        .server-info {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.85em;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("Settings")

        # Server configuration
        st.subheader("Server")
        server_url = st.text_input(
            "Agent Server URL",
            value=st.session_state.server_url,
            help="URL of the MLflow Agent Server",
        )
        if server_url != st.session_state.server_url:
            st.session_state.server_url = server_url

        # Check server health
        is_healthy, status_msg = check_server_health(st.session_state.server_url)
        if is_healthy:
            st.markdown(f'Status: <span class="status-connected">{status_msg}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'Status: <span class="status-disconnected">{status_msg}</span>', unsafe_allow_html=True)
            st.warning("Start the server with:\n```\npython start_server.py\n```")

        st.divider()

        # Streaming toggle
        st.session_state.use_streaming = st.toggle(
            "Enable Streaming",
            value=st.session_state.use_streaming,
            help="Stream responses token by token",
        )

        st.divider()

        # Session info
        st.subheader("Session")
        new_user_id = st.text_input(
            "User ID",
            value=st.session_state.user_id,
            help="Your ID for long-term memory",
        )
        if new_user_id != st.session_state.user_id:
            st.session_state.user_id = new_user_id

        st.text_input(
            "Thread ID",
            value=st.session_state.thread_id,
            disabled=True,
            help="Current conversation thread",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("New Chat", use_container_width=True):
                new_conversation()
                st.rerun()
        with col2:
            if st.button("Clear All", use_container_width=True):
                new_conversation()
                st.session_state.user_id = generate_ids()[1]
                st.rerun()

        st.divider()

        # Tips
        with st.expander("Tips", expanded=True):
            st.markdown("""
            **Try these prompts:**
            - "My name is Alex"
            - "I prefer Python over Java"
            - "Remember that I work at Acme"

            **Then start a new chat and ask:**
            - "What's my name?"
            - "What do you know about me?"
            """)

    # Main chat area
    st.title("MLflow Agent Server Chat")
    st.caption(f"Connected to: `{st.session_state.server_url}`")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message...", disabled=not is_healthy):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            if st.session_state.use_streaming:
                # Streaming response
                response_placeholder = st.empty()
                full_response = ""

                for chunk in send_message(
                    server_url=st.session_state.server_url,
                    message=prompt,
                    thread_id=st.session_state.thread_id,
                    user_id=st.session_state.user_id,
                    stream=True,
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                response = full_response
            else:
                # Non-streaming response
                with st.spinner("Thinking..."):
                    response = send_message(
                        server_url=st.session_state.server_url,
                        message=prompt,
                        thread_id=st.session_state.thread_id,
                        user_id=st.session_state.user_id,
                        stream=False,
                    )
                st.markdown(response)

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
