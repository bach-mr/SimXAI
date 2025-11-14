"""Streamlit chat application for interacting with ToolACEAgent."""

import html
import json
from datetime import datetime
import os
import streamlit as st
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from model_wrapper import ToolACEModel
from agent import ToolACEAgent
from utils.function_parser import get_available_tools
from tools import functions as tool_functions
import config


@st.cache_resource
def initialize_agent():
    """Initialize the ToolACE agent (cached to avoid reloading)."""
    with st.spinner("Loading model... This may take a few minutes on first run."):
        # Initialize model
        model = ToolACEModel(config.MODEL_NAME)
        
        # Get available tools
        tools = get_available_tools(tool_functions)
        
        # Initialize agent
        agent = ToolACEAgent(
            model=model,
            tools_module=tool_functions,
            system_prompt=config.SYSTEM_PROMPT,
            tools=tools
        )
        
        return agent, tools


def save_conversation(messages: list) -> None:
    """Save conversation to a timestamped JSON file."""
    # Create conversations directory if it doesn't exist
    conversations_dir = Path("conversations")
    conversations_dir.mkdir(exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = conversations_dir / f"conversation_{timestamp}.json"
    
    # Prepare conversation data
    conversation_data = {
        "timestamp": datetime.now().isoformat(),
        "messages": messages
    }
    
    # Save to JSON file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    
    print(f"Conversation saved to {filename}")


def format_chat_bubble(text: str, role: str) -> str:
    """Create HTML snippet for a chat bubble with proper escaping."""
    safe_text = html.escape(text).replace("\n", "<br>")
    icon = "🧑" if role == "user" else "🤖"
    bubble_role = "user" if role == "user" else "assistant"

    if bubble_role == "user":
        inner_html = (
            f"<div class='chat-bubble {bubble_role}'>{safe_text}</div>"
            f"<div class='chat-icon {bubble_role}'>{icon}</div>"
        )
    else:
        inner_html = (
            f"<div class='chat-icon {bubble_role}'>{icon}</div>"
            f"<div class='chat-bubble {bubble_role}'>{safe_text}</div>"
        )

    return (
        f"<div class='chat-row {bubble_role}'>"
        f"  <div class='chat-inner {bubble_role}'>"
        f"    {inner_html}"
        f"  </div>"
        "</div>"
    )


def render_chat_bubble(text: str, role: str) -> None:
    """Render chat bubble HTML via Streamlit markdown."""
    st.markdown(format_chat_bubble(text, role), unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Lottery Agent",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS for chat styling
    st.markdown("""
    <style>
    [data-testid="stChatMessage"] > div:first-child,
    [data-testid="stChatMessageAvatar"],
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {
        display: none !important;
    }

    .chat-row {
        display: flex;
        margin-bottom: 0.75rem;
    }

    .chat-row.user {
        justify-content: flex-end;
    }

    .chat-row.assistant {
        justify-content: flex-start;
    }

    .chat-inner {
        display: flex;
        align-items: flex-end;
        gap: 0.5rem;
        max-width: 75%;
    }

    .chat-inner.user {
        justify-content: flex-end;
    }

    .chat-inner.assistant {
        justify-content: flex-start;
    }

    .chat-bubble {
        padding: 10px 16px;
        border-radius: 18px;
        line-height: 1.5;
        word-break: break-word;
        background-color: #f0f0f0;
        color: #1f1f1f;
    }

    .chat-bubble.user {
        background-color: #0084ff;
        color: #ffffff;
    }

    .chat-icon {
        font-size: 1.6rem;
        line-height: 1;
    }

    /* Typing animation */
    @keyframes typing {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }

    .typing-indicator {
        display: inline-block;
        animation: typing 1.5s infinite;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🤖 Lottery Agent  Interface")
    st.markdown("Chat with the Lottery Agent that can use various tools to help you win the lottery!")
    
    # Initialize agent
    try:
        agent, tools = initialize_agent()
        
        # Display available tools in sidebar
        with st.sidebar:
            st.header("Available Tools")
            st.markdown(f"**Total Tools:** {len(tools)}")
            
            with st.expander("View Tool Details"):
                for tool in tools:
                    st.markdown(f"**{tool['name']}**")
                    if 'description' in tool:
                        st.caption(tool['description'])
                    st.markdown("---")
            
            # Add clear conversation button
            if st.button("Clear Conversation", type="secondary"):
                st.session_state.messages = []
                st.session_state.conversation_messages = []
                st.rerun()
        
        # Initialize session state for messages
        if 'messages' not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Welcome to the lottery! I’m here to help you understand the model that determines the prize for your ticket. Each ticket contains three numbers (from 1 to 9), separated by commas, for example: 1,2,3.\nYour task is to explore how the model works by talking with me. \
                                          I will help you retrieve and understand information from the model. You can start with creating your ticket by entering three numbers separated by commas."}]
        
        if 'conversation_messages' not in st.session_state:
            st.session_state.conversation_messages = agent.create_initial_messages()
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=None):
                render_chat_bubble(message["content"], message["role"])
                if message.get("tools_used"):
                    with st.expander("🔧 Tools Used"):
                        for tool in message["tools_used"]:
                            st.code(tool, language="text")
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user", avatar=None):
                render_chat_bubble(prompt, "user")
            
            # Get agent response
            with st.chat_message("assistant", avatar=None):
                # Show typing indicator
                typing_placeholder = st.empty()
                typing_placeholder.markdown('<span class="typing-indicator">● ● ●</span> Thinking...', unsafe_allow_html=True)
                
                try:
                    response, tool_names, updated_messages = agent.process_user_message(
                        st.session_state.conversation_messages,
                        prompt
                    )
                    
                    # Update conversation messages
                    st.session_state.conversation_messages = updated_messages
                    
                    # Clear typing indicator
                    typing_placeholder.empty()
                    
                    # Display response with typing effect
                    response_placeholder = st.empty()
                    displayed_text = ""
                    
                    # Simulate typing effect (adjust speed as needed)
                    words = response.split()
                    for word in words:
                        displayed_text += word + " "
                        response_placeholder.markdown(
                            format_chat_bubble(displayed_text.strip(), "assistant"),
                            unsafe_allow_html=True,
                        )
                        time.sleep(0.05)  # Adjust typing speed here (0.05 seconds per word)
                    
                    # Clear streaming placeholder so rerender shows single bubble
                    response_placeholder.empty()
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "tools_used": tool_names
                    })
                    
                    # Auto-save conversation after each exchange
                    save_conversation(st.session_state.messages)
                    
                    # Force rerun to render consolidated history (prevents duplicates)
                    st.rerun()
                    
                except Exception as e:
                    typing_placeholder.empty()
                    error_msg = f"Error processing message: {str(e)}"
                    render_chat_bubble(error_msg, "assistant")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "tools_used": None
                    })
                    # Auto-save on error too
                    save_conversation(st.session_state.messages)
        
        # Add information section at the bottom
        with st.expander("ℹ️ About this Application"):
            st.markdown("""
            ### ToolACE Chat Interface
            
            This is an interactive chat interface for the ToolACE agent. The agent can:
            - Understand your questions and requests
            - Automatically invoke appropriate tools to help answer your questions
            - Provide explanations based on tool results
            
            **How to use:**
            1. Type your question or request in the chat input
            2. The agent will analyze your message and decide which tools to use
            3. Tool results will be processed and presented in a natural response
            
            **Tips:**
            - You can ask questions about predictions, explanations, or any other tasks the tools support
            - Check the sidebar to see all available tools
            - Use the "Clear Conversation" button to start fresh
            """)
            
            st.markdown("---")
            st.caption(f"Model: {config.MODEL_NAME}")
    
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        st.markdown("""
        ### Troubleshooting
        - Ensure all dependencies are installed: `pip install streamlit transformers torch`
        - Check that the model and tools are properly configured
        - Verify that you have sufficient memory and GPU resources
        """)


if __name__ == "__main__":
    main()
