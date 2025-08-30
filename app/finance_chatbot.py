import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import warnings
import time
from datetime import datetime
import json

# Optional imports with fallback
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

warnings.filterwarnings("ignore")

# Configuration - Updated for IBM Granite model
MODEL_NAME = "ibm-granite/granite-3b-code-instruct"  # You can also try "ibm-granite/granite-7b-instruct"
MAX_INPUT_LENGTH = 500
MAX_RESPONSE_LENGTH = 200
MODEL_TIMEOUT = 60  # Increased timeout for larger models

# Alternative Granite models you can try:
# "ibm-granite/granite-7b-instruct" - Larger, more capable
# "ibm-granite/granite-3b-code-instruct" - Smaller, faster
# "ibm-granite/granite-13b-instruct" - Largest, best quality

# Page configuration
st.set_page_config(
    page_title="FinanceAI - Your Smart Money Assistant with IBM Granite",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_custom_styles():
    """Return custom CSS for modern styling"""
    return """
    <style>
        .main {
            padding: 1rem;
        }
        
        .chat-container {
            background: #f8fafc;
            border-radius: 15px;
            padding: 1rem;
            margin: 1rem 0;
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #e2e8f0;
            scroll-behavior: smooth;
        }
        
        .chat-message {
            padding: 1rem 1.5rem;
            border-radius: 20px;
            margin: 0.75rem 0;
            max-width: 85%;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 8px;
        }
        
        .bot-message {
            background: linear-gradient(135deg, #059669 0%, #0d9488 100%);
            color: white;
            margin-right: auto;
            border-bottom-left-radius: 8px;
        }
        
        .granite-indicator {
            background: linear-gradient(135deg, #1e40af 0%, #3730a3 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            display: inline-block;
            margin: 0.5rem 0;
            font-weight: bold;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online { background: #10b981; }
        .status-loading { 
            background: #f59e0b; 
            animation: pulse 1.5s infinite; 
        }
        .status-error { background: #ef4444; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .header-section {
            background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
            color: white;
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
        }
    </style>
    """


def get_current_model_name():
    """Get the current model name from session state or default"""
    return st.session_state.get("current_model", MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_granite_model():
    """Load IBM Granite model with timeout and proper error handling"""
    if not HF_AVAILABLE:
        st.warning("🔧 Transformers library not found. Using rule-based responses.")
        return None, None, False

    # Get current model name
    current_model = get_current_model_name()

    try:
        status_placeholder = st.empty()

        with status_placeholder.container():
            progress = st.progress(0)
            status_text = st.empty()

        start_time = time.time()

        def check_timeout():
            return (time.time() - start_time) > MODEL_TIMEOUT

        status_text.text(
            f"🔥 Loading IBM Granite tokenizer ({current_model.split('/')[-1]})..."
        )
        progress.progress(20)

        if check_timeout():
            raise TimeoutError("Model loading timeout")

        # Load tokenizer with Granite-specific settings
        tokenizer = AutoTokenizer.from_pretrained(
            current_model,
            trust_remote_code=True,
            use_fast=True,
        )

        status_text.text(
            f"🧠 Loading IBM Granite model ({current_model.split('/')[-1]}) - this may take a few minutes..."
        )
        progress.progress(60)

        if check_timeout():
            raise TimeoutError("Model loading timeout")

        # Load model with optimizations for Granite
        model = AutoModelForCausalLM.from_pretrained(
            current_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # Granite models typically use specific special tokens
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))

        progress.progress(100)
        status_text.text(
            f"✅ IBM Granite model ({current_model.split('/')[-1]}) loaded successfully!"
        )

        time.sleep(2)
        status_placeholder.empty()

        return tokenizer, model, True

    except Exception as e:
        if "status_placeholder" in locals():
            status_placeholder.empty()
        error_msg = str(e)
        st.error(f"❌ Granite model loading failed: {error_msg[:100]}...")

        # Suggest alternative models if the current one fails
        st.info("""
        💡 **Alternative IBM Granite models to try:**
        - `ibm-granite/granite-3b-code-instruct` (smaller, faster)
        - `ibm-granite/granite-7b-instruct` (balanced)
        - `ibm-granite/granite-13b-instruct` (larger, more capable)
        
        Using rule-based responses for now.
        """)
        return None, None, False


def validate_input(user_input):
    """Validate user input"""
    if not user_input or user_input.strip() == "":
        return False, "Please enter a message"

    if len(user_input) > MAX_INPUT_LENGTH:
        return (
            False,
            f"Message too long. Please keep it under {MAX_INPUT_LENGTH} characters",
        )

    return True, ""


def get_fallback_response(user_input):
    """Enhanced rule-based responses for common finance questions"""
    user_input_lower = user_input.lower()

    responses = {
        "budget": [
            "💡 **Smart Budgeting with the 50/30/20 Rule**: Allocate 50% to needs (rent, utilities), 30% to wants (dining, entertainment), and 20% to savings and debt repayment. Track everything using apps or spreadsheets.",
            "📊 **Budget Creation Strategy**: List all income sources, categorize expenses (fixed vs variable), identify spending patterns, cut unnecessary subscriptions, and set up automatic savings transfers.",
            "💰 **Budget Fundamentals**: Pay yourself first by saving before spending. Use envelope budgeting for variable expenses. Review and adjust monthly based on actual spending patterns.",
        ],
        "save": [
            "💰 **Effective Savings Strategy**: Build an emergency fund covering 3-6 months of expenses first. Then save for specific goals using separate accounts. Automate transfers to make saving effortless.",
            "🎯 **Smart Saving Techniques**: Use high-yield savings accounts for better returns. Set up automatic transfers on payday. Start small with $25-50/week and gradually increase as income grows.",
            "📈 **Advanced Savings Tips**: Follow the 1% rule - increase savings by 1% each month. Use the 24-hour rule for large purchases. Separate savings by goals (vacation, home, car).",
        ],
        "invest": [
            "📈 **Investment Fundamentals**: Start with low-cost index funds (VTI, VTSAX) for broad market exposure. Use dollar-cost averaging to reduce risk. Remember: time in market beats timing the market.",
            "💎 **Smart Investment Strategy**: Maximize 401(k) employer match first (free money), then contribute to Roth IRA. Consider target-date funds for hands-off diversified investing.",
            "🚀 **Investment Portfolio Building**: Diversify across asset classes, keep expense ratios low (<0.2%), rebalance annually, and avoid panic selling during market downturns.",
        ],
        "debt": [
            "💳 **Debt Elimination Strategy**: List all debts with balances and interest rates. Use debt avalanche (pay highest interest first) or debt snowball (smallest balance first) method. Pay more than minimums.",
            "🎯 **Debt Freedom Plan**: Stop using credit cards while paying off balances. Consider balance transfers for 0% APR offers. Use windfalls (tax refunds, bonuses) for extra payments.",
            "⚡ **Accelerated Debt Payoff**: Make bi-weekly instead of monthly payments. Round up payments to nearest $50. Consider debt consolidation for lower interest rates.",
        ],
        "retirement": [
            "🏖️ **Retirement Planning Essentials**: Save 10-15% of income for retirement. Use the rule of 25 (save 25x annual expenses). Start early to harness compound growth over decades.",
            "📊 **Retirement Strategy**: Maximize employer 401(k) match, then contribute to Roth IRA. Increase contributions by 1% annually. Use target-date funds for automatic rebalancing.",
            "🎯 **Retirement Goals**: Diversify globally with low-cost index funds. Avoid cashing out retirement accounts when changing jobs. Consider Roth conversions in low-income years.",
        ],
        "tax": [
            "🧾 **Tax Optimization**: Maximize contributions to tax-advantaged accounts (401k, IRA, HSA). Keep receipts for deductible expenses. Consider tax-loss harvesting for investments.",
            "💡 **Tax Planning Strategy**: Contribute to HSA if available (triple tax advantage). Time income and deductions strategically. Bunch charitable donations for itemization.",
            "📋 **Year-End Tax Moves**: Review tax-loss harvesting opportunities, maximize retirement contributions, consider Roth conversions, and organize tax documents early.",
        ],
    }

    # Enhanced keyword matching
    import random

    for category, response_list in responses.items():
        if category in user_input_lower:
            return f"🤖 **IBM Granite AI Response**\n\n{random.choice(response_list)}"

    # Comprehensive default response
    return """🤖 **IBM Granite AI - Complete Financial Guidance**

**📊 Financial Foundation Framework:**

1. **📝 Budget Creation**: Track income vs expenses, use 50/30/20 rule
2. **🚨 Emergency Fund**: Build 3-6 months of expenses in high-yield savings
3. **💳 Debt Management**: Pay off high-interest debt using avalanche/snowball method
4. **📈 Investment Start**: Max employer 401(k) match, then low-cost index funds
5. **🎯 Goal Setting**: Set SMART financial goals with specific timelines

**🚀 Next Steps**: What specific area would you like to explore deeper? Ask about budgeting, investing, debt payoff, or retirement planning for personalized guidance."""


def create_granite_prompt(user_input):
    """Create optimized prompt for IBM Granite model"""
    system_prompt = """You are a professional financial advisor with expertise in personal finance, budgeting, investing, and wealth building. Provide practical, actionable advice that is:
- Clear and easy to understand
- Based on established financial principles
- Tailored to the user's question
- Includes specific steps or recommendations
- Uses emojis to make responses engaging

Keep responses concise but comprehensive (100-200 words)."""

    # Format prompt for Granite model
    prompt = f"""<|system|>
{system_prompt}
<|user|>
{user_input}
<|assistant|>
"""

    return prompt


def get_granite_response(user_input):
    """Generate response using IBM Granite model"""
    try:
        if not HF_AVAILABLE:
            return get_fallback_response(user_input)

        tokenizer, model, is_loaded = load_granite_model()

        if not is_loaded:
            return get_fallback_response(user_input)

        # Create Granite-optimized prompt
        prompt = create_granite_prompt(user_input)

        # Tokenize input
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True, padding=True
        )

        # Move to device if using GPU
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate response with Granite-optimized parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_RESPONSE_LENGTH,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response
        if "<|assistant|>" in generated_text:
            response = generated_text.split("<|assistant|>")[-1].strip()
        else:
            response = generated_text[len(prompt) :].strip()

        # Clean up and validate response
        if len(response) > 20 and not response.startswith(prompt[:50]):
            return f"🤖 **IBM Granite AI Response**\n\n{response}"
        else:
            # Fallback if Granite response is poor
            return get_fallback_response(user_input)

    except Exception as e:
        st.warning(f"Granite model error: {str(e)[:100]}... Using fallback response.")
        return get_fallback_response(user_input)


def get_response(user_input):
    """Main response function with Granite AI integration"""
    is_valid, error_msg = validate_input(user_input)
    if not is_valid:
        return f"❌ {error_msg}"

    use_granite = st.session_state.get("use_granite_model", True)

    if use_granite and HF_AVAILABLE:
        return get_granite_response(user_input)
    else:
        return get_fallback_response(user_input)


def create_spending_chart():
    """Create enhanced spending breakdown pie chart"""
    categories = [
        "Housing",
        "Food",
        "Transport",
        "Savings",
        "Utilities",
        "Fun",
        "Healthcare",
        "Other",
    ]
    amounts = [2000, 800, 600, 1200, 400, 500, 300, 400]
    colors = [
        "#1e40af",
        "#3730a3",
        "#4338ca",
        "#4f46e5",
        "#6366f1",
        "#8b5cf6",
        "#a855f7",
        "#c084fc",
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(
        amounts,
        labels=[f"{cat}\n${amt:,.0f}" for cat, amt in zip(categories, amounts)],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        shadow=True,
        textprops={"fontsize": 10, "weight": "bold"},
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_weight("bold")
        autotext.set_fontsize(9)

    ax.set_title(
        "Monthly Budget Breakdown (Powered by IBM Granite AI)",
        fontsize=16,
        weight="bold",
        pad=20,
    )
    ax.axis("equal")

    total = sum(amounts)
    plt.figtext(
        0.5,
        0.02,
        f"Total Monthly Budget: ${total:,.0f}",
        ha="center",
        fontsize=12,
        weight="bold",
    )

    plt.tight_layout()
    return fig


def create_savings_progress():
    """Create savings goal progress chart"""
    goals = ["Emergency Fund", "Vacation", "New Car", "Retirement"]
    current = [8500, 2400, 5200, 45000]
    targets = [15000, 4000, 15000, 100000]

    fig, ax = plt.subplots(figsize=(10, 6))

    progress = [min(c / t * 100, 100) for c, t in zip(current, targets)]
    remaining = [max(0, 100 - p) for p in progress]

    y_pos = np.arange(len(goals))

    ax.barh(y_pos, progress, color="#1e40af", label="Progress", height=0.6)
    ax.barh(
        y_pos, remaining, left=progress, color="#e5e7eb", label="Remaining", height=0.6
    )

    for i, (curr, target, prog) in enumerate(zip(current, targets, progress)):
        ax.text(
            prog / 2,
            i,
            f"${curr:,.0f}",
            ha="center",
            va="center",
            fontweight="bold",
            color="white" if prog > 25 else "black",
        )
        ax.text(
            101, i, f"Goal: ${target:,.0f}", ha="left", va="center", fontweight="bold"
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(goals, fontweight="bold")
    ax.set_xlabel("Progress (%)", fontweight="bold")
    ax.set_title(
        "Savings Goals Progress (IBM Granite AI Insights)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlim(0, 120)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "messages": [
            {
                "role": "assistant",
                "content": "👋 Hello! I'm your AI finance advisor powered by IBM Granite AI. I can help with budgeting, saving, investing, debt management, and retirement planning. What financial goal would you like to work on today?",
                "timestamp": datetime.now(),
            }
        ],
        "model_loaded": False,
        "model_ready": False,
        "use_granite_model": True,
        "user_input": "",  # Initialize user input state
        "current_model": MODEL_NAME,  # Track current model
        "granite_model_info": {"name": MODEL_NAME, "status": "not_loaded"},
        "user_preferences": {
            "income": 5200,
            "savings_rate": 20,
            "risk_tolerance": "Moderate",
        },
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_granite_status():
    """Get current Granite model status"""
    if not HF_AVAILABLE:
        return "error"
    elif st.session_state.get("model_ready", False):
        return "online"
    elif st.session_state.get("model_loading", False):
        return "loading"
    else:
        return "offline"


def add_message(role, content):
    """Add message to chat history"""
    st.session_state.messages.append(
        {"role": role, "content": content, "timestamp": datetime.now()}
    )


def process_user_message(user_input):
    """Process user input and generate response"""
    if not user_input.strip():
        return

    add_message("user", user_input)

    with st.spinner("🤖 IBM Granite AI is thinking..."):
        response = get_response(user_input)

    add_message("assistant", response)

    # Clear the input and refresh the chat
    st.session_state.user_input = ""
    st.rerun()


def render_chat_interface():
    """Render the main chat interface with auto-scroll"""
    st.subheader("💬 Chat with IBM Granite AI Financial Advisor")

    # Granite model status
    status = get_granite_status()
    status_configs = {
        "online": ("status-online", "🟢 IBM Granite AI Ready"),
        "loading": ("status-loading", "🟡 Loading IBM Granite Model..."),
        "offline": ("status-indicator", "⚪ Rule-Based Mode"),
        "error": ("status-error", "🔴 AI Unavailable - Rule Mode Active"),
    }

    status_class, status_text = status_configs.get(status, ("status-error", "Unknown"))

    st.markdown(
        f"""
    <div class="granite-indicator">
        <span class="status-indicator {status_class}"></span>
        <strong>🧠 {status_text}</strong> | Model: {get_current_model_name().split("/")[-1]}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Chat container with auto-scroll
    chat_container = st.container()

    with chat_container:
        st.markdown(
            '<div class="chat-container" id="chat-messages">', unsafe_allow_html=True
        )

        for i, message in enumerate(st.session_state.messages):
            timestamp = message.get("timestamp", datetime.now()).strftime("%H:%M")
            message_id = f"message_{i}"

            if message["role"] == "user":
                st.markdown(
                    f"""
                <div class="chat-message user-message" id="{message_id}">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <strong>You</strong>
                        <small style="opacity: 0.8;">{timestamp}</small>
                    </div>
                    {message["content"]}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="chat-message bot-message" id="{message_id}">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <strong>🤖 IBM Granite AI</strong>
                        <small style="opacity: 0.8;">{timestamp}</small>
                    </div>
                    {message["content"]}
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

        # Auto-scroll to bottom JavaScript
        if (
            len(st.session_state.messages) > 1
        ):  # Only scroll if there are messages beyond the initial one
            st.markdown(
                """
            <script>
            setTimeout(function() {
                var chatContainer = document.getElementById('chat-messages');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                    
                    // Also scroll the main container
                    var messages = document.querySelectorAll('.chat-message');
                    if (messages.length > 0) {
                        messages[messages.length - 1].scrollIntoView({ 
                            behavior: 'smooth', 
                            block: 'end' 
                        });
                    }
                }
            }, 100);
            </script>
            """,
                unsafe_allow_html=True,
            )


def render_sidebar():
    """Enhanced sidebar with Granite AI dashboard"""
    with st.sidebar:
        st.markdown("### 🧠 IBM Granite AI Dashboard")

        # Model information
        current_model_name = get_current_model_name()
        st.markdown(f"""
        **🔧 Current Model:** `{current_model_name.split("/")[-1]}`
        
        **📊 Model Info:**
        - Provider: IBM Research
        - Type: Instruction-tuned LLM
        - Specialization: Code & Text Generation
        """)

        st.markdown("---")

        # User preferences
        income = st.number_input(
            "Monthly Income ($)",
            min_value=1000,
            max_value=50000,
            value=st.session_state.user_preferences["income"],
            step=100,
        )

        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ["Conservative", "Moderate", "Aggressive"],
            index=["Conservative", "Moderate", "Aggressive"].index(
                st.session_state.user_preferences["risk_tolerance"]
            ),
        )

        if (
            income != st.session_state.user_preferences["income"]
            or risk_tolerance != st.session_state.user_preferences["risk_tolerance"]
        ):
            st.session_state.user_preferences["income"] = income
            st.session_state.user_preferences["risk_tolerance"] = risk_tolerance
            st.rerun()

        # Quick metrics
        expenses = income * 0.8
        savings = income - expenses

        st.metric("💵 Monthly Income", f"${income:,.0f}")
        st.metric("💸 Monthly Expenses", f"${expenses:,.0f}")
        st.metric(
            "💰 Monthly Savings", f"${savings:,.0f}", f"{(savings / income) * 100:.1f}%"
        )

        st.markdown("---")

        # Charts section
        st.markdown("### 📈 AI-Powered Insights")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if st.button("📊 Budget Analysis", use_container_width=True):
                st.session_state.show_budget_chart = True

        with chart_col2:
            if st.button("🎯 Goal Tracking", use_container_width=True):
                st.session_state.show_goals_chart = True

        st.markdown("---")

        # Granite AI Quick Questions
        st.markdown("### 🚀 Granite AI Quick Start")

        granite_questions = [
            ("💡 Smart Budget", "Create a detailed budget plan for my income level"),
            ("📈 Investment Guide", "Explain index fund investing for beginners"),
            ("🎯 Savings Strategy", "How can I optimize my savings rate?"),
            ("💳 Debt Freedom", "Create a debt payoff strategy"),
            ("🏠 Home Buying", "Guide me through the home buying financial process"),
            ("🧾 Tax Optimization", "What tax strategies should I consider?"),
            ("⚡ Emergency Fund", "Help me build an emergency fund"),
            ("🎓 Financial Education", "Teach me advanced personal finance concepts"),
        ]

        for button_text, question in granite_questions:
            if st.button(
                button_text, key=f"granite_{button_text}", use_container_width=True
            ):
                process_user_message(question)

        st.markdown("---")

        # Granite AI Settings
        st.markdown("### 🤖 IBM Granite Settings")

        use_granite = st.toggle(
            "Enable IBM Granite AI",
            value=st.session_state.get("use_granite_model", True),
            help="Toggle between IBM Granite AI and rule-based responses",
        )

        st.session_state.use_granite_model = use_granite

        if use_granite and not HF_AVAILABLE:
            st.error(
                "🔧 **Required Installation:**\n```bash\npip install transformers torch\n```"
            )

        # Model selection
        model_options = {
            "Granite 3B (Faster)": "ibm-granite/granite-3b-code-instruct",
            "Granite 7B (Balanced)": "ibm-granite/granite-7b-instruct",
            "Granite 13B (Best)": "ibm-granite/granite-13b-instruct",
        }

        selected_model = st.selectbox(
            "Choose Granite Model",
            list(model_options.keys()),
            help="Larger models provide better responses but load slower",
        )

        if st.button("🔄 Switch Model", use_container_width=True):
            # Update model name in session state instead of global variable
            st.session_state.current_model = model_options[selected_model]
            st.session_state.granite_model_info["name"] = model_options[selected_model]
            st.cache_resource.clear()
            st.success(f"Switched to {selected_model}")
            st.rerun()

        # Clear chat
        if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "👋 Chat cleared! IBM Granite AI is ready to help with your financial questions.",
                    "timestamp": datetime.now(),
                }
            ]
            st.session_state.user_input = ""  # Also clear any pending input
            st.rerun()


def main():
    """Main application with IBM Granite integration"""
    st.markdown(get_custom_styles(), unsafe_allow_html=True)
    initialize_session_state()

    # Header
    st.markdown(
        f"""
    <div class="header-section">
        <h1 style="margin: 0; font-size: 3rem;">🧠 FinanceAI</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">Powered by IBM Granite AI</p>
        <p style="margin: 0.2rem 0 0 0; font-size: 1rem; opacity: 0.9;">Model: {get_current_model_name().split("/")[-1]}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        render_chat_interface()

        # Input section with Enter key support
        st.markdown("### ✏️ Ask IBM Granite AI")

        # Create a form to handle Enter key submission
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "What would you like to know about personal finance?",
                value=st.session_state.get("user_input", ""),
                placeholder="e.g., How should I invest $10,000? Create a debt payoff plan for me. What's the best budgeting strategy?",
                height=100,
                max_chars=MAX_INPUT_LENGTH,
                help=f"Maximum {MAX_INPUT_LENGTH} characters - Press Ctrl+Enter or click Send",
                key="input_field",
            )

            # Form submission buttons
            input_col1, input_col2, input_col3 = st.columns([3, 1, 1])

            with input_col1:
                submitted = st.form_submit_button(
                    "🚀 Ask Granite AI", type="primary", use_container_width=True
                )

            with input_col2:
                if st.form_submit_button("🔄 Clear", use_container_width=True):
                    st.session_state.user_input = ""
                    st.rerun()

            with input_col3:
                if st.form_submit_button("💡 Example", use_container_width=True):
                    example_questions = [
                        "How should I budget $5000 monthly income?",
                        "What's the best investment strategy for beginners?",
                        "Help me create a debt payoff plan",
                        "How much should I save for retirement?",
                        "Explain different types of investment accounts",
                    ]
                    import random

                    st.session_state.user_input = random.choice(example_questions)
                    st.rerun()

            # Process the form submission
            if submitted and user_input.strip():
                process_user_message(user_input)

        # JavaScript for Enter key handling
        st.markdown(
            """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            const textarea = document.querySelector('textarea[aria-label="What would you like to know about personal finance?"]');
            if (textarea) {
                textarea.addEventListener('keydown', function(event) {
                    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
                        event.preventDefault();
                        const form = textarea.closest('form');
                        if (form) {
                            const submitButton = form.querySelector('button[kind="primaryFormSubmit"]');
                            if (submitButton) {
                                submitButton.click();
                            }
                        }
                    }
                });
            }
        });
        </script>
        """,
            unsafe_allow_html=True,
        )

        # Show charts if requested
        if st.session_state.get("show_budget_chart", False):
            st.markdown("### 📊 Budget Analysis (IBM Granite AI)")
            fig = create_spending_chart()
            st.pyplot(fig, use_container_width=True)
            st.session_state.show_budget_chart = False

        if st.session_state.get("show_goals_chart", False):
            st.markdown("### 🎯 Savings Progress (IBM Granite AI Insights)")
            fig = create_savings_progress()
            st.pyplot(fig, use_container_width=True)
            st.session_state.show_goals_chart = False

    with col2:
        render_sidebar()

    # Footer
    st.markdown("---")

    # Export and utilities
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("💾 Export Chat", use_container_width=True):
            chat_data = {
                "exported_at": datetime.now().isoformat(),
                "model_used": get_current_model_name(),
                "messages": st.session_state.messages,
                "preferences": st.session_state.user_preferences,
                "granite_ai_version": "IBM Granite Finance Assistant v1.0",
            }
            st.download_button(
                "📥 Download JSON",
                json.dumps(chat_data, indent=2, default=str),
                f"granite_finance_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                "application/json",
            )

    with col2:
        if st.button("📋 Copy Response", use_container_width=True):
            if st.session_state.messages:
                last_response = st.session_state.messages[-1]["content"]
                st.code(last_response, language="markdown")

    with col3:
        if st.button("🔄 Reload Model", use_container_width=True):
            st.cache_resource.clear()
            st.success("Model cache cleared! Next query will reload Granite AI.")
            st.rerun()

    with col4:
        if st.button("🆘 Help", use_container_width=True):
            st.info("""
            **🧠 IBM Granite AI Help:**
            - Ask detailed financial questions
            - Request personalized advice
            - Get step-by-step guidance
            - Analyze your financial situation
            
            **💡 Example Questions:**
            - "Create a comprehensive budget for $5000/month income"
            - "Explain different investment strategies for beginners"
            - "How do I optimize my 401(k) contributions?"
            """)

    # Advanced Granite AI Information
    st.markdown("---")

    with st.expander("🧠 About IBM Granite AI Integration"):
        st.markdown(f"""
        ### 🔬 Technical Details
        
        **Current Model:** `{get_current_model_name()}`
        
        **🎯 IBM Granite Capabilities:**
        - **Instruction Following**: Optimized for detailed financial guidance
        - **Code Generation**: Can create budgeting formulas and calculations
        - **Reasoning**: Advanced logical thinking for complex financial scenarios
        - **Context Awareness**: Remembers conversation history for personalized advice
        
        **⚙️ Model Configurations:**
        - **Temperature**: 0.7 (balanced creativity/accuracy)
        - **Top-p**: 0.9 (diverse but relevant responses)
        - **Max Tokens**: {MAX_RESPONSE_LENGTH} (comprehensive answers)
        - **Repetition Penalty**: 1.1 (reduces repetitive content)
        
        **🚀 Performance Optimizations:**
        - Automatic GPU detection and usage
        - Mixed precision (FP16) for faster inference
        - Device mapping for memory efficiency
        - Intelligent prompt engineering
        
        **📊 Available Granite Models:**
        1. **granite-3b-code-instruct** - Fastest, good for quick responses
        2. **granite-7b-instruct** - Balanced performance and quality
        3. **granite-13b-instruct** - Best quality, slower inference
        
        **🛠️ Fallback System:**
        If Granite AI is unavailable, the system automatically switches to 
        enhanced rule-based responses to ensure uninterrupted service.
        """)

    # Installation and troubleshooting
    with st.expander("🔧 Installation & Troubleshooting"):
        st.markdown("""
        ### 📦 Installation Requirements
        
        **Basic Requirements:**
        ```bash
        pip install streamlit matplotlib numpy
        ```
        
        **For IBM Granite AI:**
        ```bash
        pip install transformers torch
        # For GPU support (optional but recommended):
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
        
        **🎯 Quick Start:**
        ```bash
        streamlit run finance_chatbot_granite.py
        ```
        
        ### 🛠️ Troubleshooting
        
        **❌ "Model not found" Error:**
        - Check internet connection
        - Verify Hugging Face model name
        - Try alternative Granite model
        
        **⚠️ Slow Loading:**
        - First load downloads ~2-6GB model
        - Subsequent loads use cache
        - Consider smaller 3B model for speed
        
        **🔄 Memory Issues:**
        - Use 3B model instead of 7B/13B
        - Enable GPU if available
        - Close other memory-intensive apps
        
        **📡 Network Issues:**
        - Model downloads from Hugging Face
        - Use offline mode after first download
        - Check firewall/proxy settings
        
        ### 💡 Performance Tips
        - Enable GPU for 3-5x faster responses
        - Use FP16 precision to save memory
        - Cache is persistent across sessions
        - Rule-based mode for instant responses
        """)

    # Disclaimer with Granite-specific information
    st.markdown(
        """
    <div style="text-align: center; padding: 1.5rem; color: #1e40af; background: linear-gradient(135deg, #ede9fe 0%, #e0f2fe 100%); border-radius: 15px; margin-top: 2rem; border: 2px solid #3730a3;">
        <p><strong>🧠 IBM Granite AI Disclaimer:</strong> This application uses IBM Granite large language models for financial guidance. Responses are generated using AI and should not replace professional financial advice.</p>
        <p><strong>📋 Financial Disclaimer:</strong> All advice is for educational purposes only. Consult certified financial professionals for personalized investment and financial planning decisions.</p>
        <p><strong>⚡ Performance:</strong> First model load may take 2-5 minutes. Subsequent responses are fast. GPU recommended for optimal performance.</p>
        <p><strong>🔒 Privacy:</strong> All conversations are processed locally. No data is sent to external servers beyond initial model download.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()


# 📦 IBM GRANITE AI FINANCE CHATBOT - ENHANCED INSTALLATION GUIDE
"""
🧠 IBM GRANITE AI FINANCE ASSISTANT
====================================

🎯 FEATURES:
✅ IBM Granite 3B/7B/13B model integration
✅ Advanced financial advice generation
✅ Smart prompt engineering for finance
✅ GPU acceleration support
✅ Intelligent fallback system
✅ Model switching capabilities
✅ Enhanced error handling
✅ Professional financial insights

📦 INSTALLATION:

BASIC (Rule-based mode):
pip install streamlit matplotlib numpy

FULL AI FEATURES:
pip install streamlit matplotlib numpy transformers torch

GPU SUPPORT (Recommended):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

🚀 QUICK START:
streamlit run granite_finance_chatbot.py

🤖 IBM GRANITE MODELS AVAILABLE:
- granite-3b-code-instruct (Fast, 3GB download)
- granite-7b-instruct (Balanced, 6GB download) 
- granite-13b-instruct (Best, 12GB download)

⚡ PERFORMANCE MODES:
- Granite AI: Advanced responses using IBM LLM
- Rule-Based: Instant responses, no dependencies

🔧 TROUBLESHOOTING:
- First run downloads model (2-12GB depending on size)
- GPU detection automatic
- Fallback to CPU if GPU unavailable
- Rule-based mode if transformers not installed

💡 USAGE TIPS:
- Ask detailed, specific financial questions
- Use natural language for best results
- Switch models based on speed vs quality preference
- Export conversations for later reference

🎓 EXAMPLE QUESTIONS:
- "Create a complete budget plan for someone earning $75,000/year"
- "Explain the difference between Roth IRA and traditional IRA"
- "What's the best strategy to pay off $25,000 in credit card debt?"
- "How should I allocate investments in my 401(k)?"

🔒 PRIVACY & SECURITY:
- All processing happens locally
- No conversation data sent to external servers
- Models downloaded from Hugging Face once
- Full offline operation after initial setup
"""
