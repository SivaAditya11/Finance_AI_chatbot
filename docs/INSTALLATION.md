🚀 FinanceAI Installation Guide

This guide explains how to install and run the FinanceAI – Smart Money Assistant chatbot.

📦 Prerequisites

Python 3.9 or later

Git (optional, for cloning repo)

Virtual environment (recommended)

⚙️ Step 1: Clone the Repository
git clone https://github.com/your-username/finance-ai.git
cd finance-ai


If you don’t use Git, simply download the ZIP from GitHub and extract it.

⚙️ Step 2: Set Up Virtual Environment

It is recommended to use a virtual environment:

python -m venv venv


Activate it:

Linux / macOS:

source venv/bin/activate


Windows:

venv\Scripts\activate

⚙️ Step 3: Install Dependencies
🔹 Option A: Rule-Based Mode (lightweight)

This mode works without AI models (fast & lightweight):

pip install streamlit matplotlib numpy

🔹 Option B: Full Installation (AI-Enhanced)

For AI-powered responses (Hugging Face DialoGPT):

pip install -r requirements.txt

⚙️ Step 4: Run the Application
streamlit run finance_chatbot_fixed1.py


The app will open at:
👉 http://localhost:8501

📊 Modes of Operation

Rule-Based Mode (default) → Instant, pre-defined finance tips.

AI-Enhanced Mode → Toggle "Enable AI Model" in sidebar (requires Hugging Face + PyTorch).

🛠 Troubleshooting

ModuleNotFoundError → Run pip install <missing-package>

Model loading too slow → Disable AI mode (fallback to rule-based).

Memory issues → Restart app or clear Streamlit cache.

CUDA/GPU issues with PyTorch → Install a GPU-compatible version of torch from PyTorch.org
.