"""
Gradio Chat UI for Databricks Agent App.

This module provides a web-based chat interface for interacting with
the AI agent. Can be run standalone or mounted in the FastAPI app.
"""

import os
import uuid
import httpx
import gradio as gr
from typing import Generator


# API endpoint configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def generate_thread_id() -> str:
    """Generate a unique thread ID for new conversations."""
    return f"thread_{uuid.uuid4().hex[:12]}"


def chat_with_agent(
    message: str,
    history: list[tuple[str, str]],
    thread_id: str,
    user_id: str,
) -> Generator[tuple[list[tuple[str, str]], str], None, None]:
    """
    Send a message to the agent and stream the response.

    Args:
        message: The user's message
        history: Conversation history as list of (user, assistant) tuples
        thread_id: The conversation thread ID
        user_id: The user ID for long-term memory

    Yields:
        Updated history and thread_id
    """
    if not message.strip():
        yield history, thread_id
        return

    # Add user message to history
    history = history + [(message, "")]

    try:
        # Call the chat API
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{API_BASE_URL}/api/v1/chat",
                json={
                    "message": message,
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Update history with assistant response
            history[-1] = (message, data["message"])
            yield history, thread_id

    except httpx.HTTPStatusError as e:
        error_msg = f"API Error: {e.response.status_code}"
        history[-1] = (message, error_msg)
        yield history, thread_id

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history[-1] = (message, error_msg)
        yield history, thread_id


def clear_conversation() -> tuple[list, str]:
    """Clear the conversation and generate a new thread ID."""
    return [], generate_thread_id()


def create_chat_interface() -> gr.Blocks:
    """Create the Gradio chat interface."""
    with gr.Blocks(
        title="Databricks Agent Chat",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 900px; margin: auto; }
        .header { text-align: center; margin-bottom: 20px; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # Databricks Agent Chat
            ### AI Assistant with Long-Term Memory

            This agent remembers information from your conversations. Try telling it your
            name or preferences - it will remember them in future conversations!

            **Features:**
            - **Short-term memory**: Remembers context within the current conversation
            - **Long-term memory**: Remembers important information across conversations
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_label=False,
                    avatar_images=(None, "https://www.databricks.com/favicon.ico"),
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### Settings")

                user_id = gr.Textbox(
                    label="User ID",
                    value="demo_user",
                    info="Your unique identifier for long-term memory",
                )

                thread_id = gr.Textbox(
                    label="Thread ID",
                    value=generate_thread_id(),
                    info="Current conversation thread",
                    interactive=False,
                )

                clear_btn = gr.Button("New Conversation", variant="secondary")

                gr.Markdown(
                    """
                    ---
                    ### Tips

                    - Share your name: *"My name is Alex"*
                    - Share preferences: *"I prefer Python over Java"*
                    - Ask it to remember: *"Remember that I work at Acme Corp"*

                    The agent will recall this information in future conversations!
                    """
                )

        # Event handlers
        def submit_message(message, history, thread_id, user_id):
            for result in chat_with_agent(message, history, thread_id, user_id):
                yield result[0], "", result[1]

        msg.submit(
            submit_message,
            inputs=[msg, chatbot, thread_id, user_id],
            outputs=[chatbot, msg, thread_id],
        )

        send_btn.click(
            submit_message,
            inputs=[msg, chatbot, thread_id, user_id],
            outputs=[chatbot, msg, thread_id],
        )

        clear_btn.click(
            clear_conversation,
            outputs=[chatbot, thread_id],
        )

    return demo


# Create the interface
chat_interface = create_chat_interface()


if __name__ == "__main__":
    # Run standalone for development
    chat_interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
