"""
============================================================
💰 FinanceAI - Smart Money Assistant
============================================================

FinanceAI is an interactive personal finance chatbot built
with Streamlit. It helps users with budgeting, saving,
investing, debt management, and retirement planning.

This package provides:
    - Chatbot logic with both rule-based & AI-driven responses
    - Visualization tools for budgeting and savings
    - Streamlit components for the web app
    - Export and utility functions for session data

------------------------------------------------------------
Usage Example
------------------------------------------------------------
from finance_ai import FinanceAIApp

app = FinanceAIApp()
app.run()

------------------------------------------------------------
Author:  Polisetty Siva Aditya
License: MIT
============================================================
"""

# Package Metadata
__title__ = "FinanceAI"
__version__ = "1.0.0"
__author__ = "Polisetty Siva Aditya"
__license__ = "MIT"
__all__ = [
    "load_model",
    "get_response",
    "get_fallback_response",
    "create_spending_chart",
    "create_savings_progress",
    "initialize_session_state",
    "FinanceAIApp",
]

# Import core functions from the main module
from .finance_chatbot_fixed1 import (
    load_model,
    get_response,
    get_fallback_response,
    create_spending_chart,
    create_savings_progress,
    initialize_session_state,
)


# Optional: Wrapper class for clean usage
class FinanceAIApp:
    """
    Wrapper class for the FinanceAI Streamlit application.
    Provides a clean entry point for running the chatbot.
    """

    def __init__(self):
        self.title = __title__
        self.version = __version__
        self.author = __author__

    def run(self):
        """Run the FinanceAI Streamlit app"""
        from . import finance_chatbot_fixed1

        finance_chatbot_fixed1.main()

    def info(self):
        """Return project metadata"""
        return {
            "name": self.title,
            "version": self.version,
            "author": self.author,
            "license": __license__,
        }
