"""
Utility package for FinanceAI

Contains helper functions for:
- Input validation
- Generating responses (rule-based/AI fallback)
- Chart creation
- Session state management
- Exporting chat history
"""

from .input_validation import validate_input
from .responses import get_fallback_response, get_ai_response, get_response
from .charts import create_spending_chart, create_savings_progress
from .session import initialize_session_state, add_message, get_model_status
from .export import export_chat
