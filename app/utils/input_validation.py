"""Utility for validating user inputs"""

MAX_INPUT_LENGTH = 300


def validate_input(user_input: str):
    """Validate user input before sending to the chatbot"""
    if not user_input or user_input.strip() == "":
        return False, "Please enter a message"

    if len(user_input) > MAX_INPUT_LENGTH:
        return False, f"Message too long. Keep it under {MAX_INPUT_LENGTH} characters"

    return True, ""
