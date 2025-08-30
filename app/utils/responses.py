"""Response generation utilities for FinanceAI"""

import random


def get_fallback_response(user_input: str) -> str:
    """Rule-based responses for common finance queries"""
    responses = {
        "budget": [
            "💡 Try the 50/30/20 rule: 50% needs, 30% wants, 20% savings.",
            "📊 Track expenses with apps like Mint or YNAB.",
        ],
        "save": [
            "💰 Build an emergency fund first (3-6 months expenses).",
            "🎯 Use high-yield savings accounts & automate transfers.",
        ],
        "invest": [
            "📈 Start with index funds (VTI, VTSAX). Time > timing!",
            "💎 Max out 401(k) match, then Roth IRA.",
        ],
    }

    user_input_lower = user_input.lower()
    for keyword, options in responses.items():
        if keyword in user_input_lower:
            return random.choice(options)

    return "🤖 I can help with budgeting, saving, investing, or debt. What do you want to focus on?"


def get_ai_response(user_input: str, tokenizer, model):
    """AI model response with Hugging Face"""
    if not tokenizer or not model:
        return get_fallback_response(user_input)

    inputs = tokenizer.encode(
        f"Financial Advisor: {user_input}\nAdvice:",
        return_tensors="pt",
        truncation=True,
    )
    outputs = model.generate(inputs, max_length=200, temperature=0.7, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response.split("Advice:")[-1].strip() or get_fallback_response(user_input)


def get_response(user_input: str, use_ai: bool, tokenizer=None, model=None):
    """Choose AI or fallback mode based on user preference"""
    if use_ai and tokenizer and model:
        return get_ai_response(user_input, tokenizer, model)
    return get_fallback_response(user_input)
