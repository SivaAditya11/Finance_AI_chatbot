"""Utilities for exporting chat history"""

import json
from datetime import datetime
import streamlit as st


def export_chat():
    """Export chat as JSON"""
    chat_data = {
        "exported_at": datetime.now().isoformat(),
        "messages": st.session_state.messages,
    }
    st.download_button(
        "📥 Download Chat JSON",
        json.dumps(chat_data, indent=2, default=str),
        f"finance_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        "application/json",
    )
