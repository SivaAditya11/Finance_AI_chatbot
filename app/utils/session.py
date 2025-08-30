"""Session state management utilities"""

from datetime import datetime
import streamlit as st


def initialize_session_state():
    """Initialize default session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Hello! I'm your AI finance advisor. How can I help?",
                "timestamp": datetime.now(),
            }
        ]
    if "use_ai_model" not in st.session_state:
        st.session_state.use_ai_model = False


def add_message(role, content):
    """Add a message to session history"""
    st.session_state.messages.append(
        {"role": role, "content": content, "timestamp": datetime.now()}
    )


def get_model_status(available: bool, loaded: bool):
    """Return model status indicator"""
    if not available:
        return "🔴 AI Unavailable - Rule Mode Active"
    return "🟢 AI Model Ready" if loaded else "⚪ Rule-Based Mode"
