# 💰 FinanceAI – Your Smart Money Assistant  

FinanceAI is an interactive **personal finance chatbot** built with **Streamlit**.  
It helps users with budgeting, saving, investing, debt management, and retirement planning.  
You can chat with the bot, view financial insights, and even enable an **AI-powered advisor** using Hugging Face models.  

---

## 🚀 Features  

✅ **Interactive Chat Interface** – Ask finance-related questions and get instant responses.  
✅ **Rule-based Mode** – Fast, reliable responses without heavy dependencies.  
✅ **AI Mode (Optional)** – Dynamic answers powered by Hugging Face’s `DialoGPT`.  We can also switch the models based on our requirements.
✅ **Financial Dashboard** – Side panel with income, expenses, and savings metrics.  
✅ **Visual Insights** – Charts for budget breakdown and savings progress.  
✅ **Quick Actions** – One-click finance FAQs (budgeting, debt, investments, taxes, etc.).  
✅ **Export & Copy** – Download chat history or copy responses.  
✅ **Customizable Preferences** – Adjust monthly income and risk tolerance.  
✅ **Modern UI/UX** – Clean design with animations, gradients, and interactive sidebar.  

---

## 📦 Installation  

### 1. Clone Repository  
```bash
git clone https://github.com/your-username/finance-ai.git
cd finance-ai
```

### 2. Install Requirements  
Minimal setup (rule-based only):  
 AI features (for Hugging Face model support):
```bash
pip install transformers torch
```

---

## ▶️ Running the App  

```bash
streamlit run finance_chatbot.py
```

The app will launch in your default browser at **http://localhost:8501**.  

---

## ⚙️ Modes of Operation  

### 🔹 Rule-Based Mode (default)  
- No heavy dependencies.  
- Instant responses using predefined finance tips.  
- Ideal for fast performance.  

### 🔹 AI-Enhanced Mode  
- Toggle "Enable AI Model" in the sidebar.  
- Uses Hugging Face’s **DialoGPT-small** for contextual replies.  
- Requires `transformers` and `torch`.  

---

## 📊 Dashboard & Charts  

- **Budget Breakdown** → Pie chart of monthly expenses.  
- **Savings Goals** → Progress bar chart for long-term goals (retirement, emergency fund, vacation, etc.).  
- **Metrics** → Monthly income, expenses, and savings rate.  

---

## 🛠️ Troubleshooting  

- **ModuleNotFoundError** → Run `pip install <missing-package>`.  
- **Slow startup** → Use rule-based mode instead of AI.  
- **Memory issues** → Restart Streamlit and clear cache.  
- **Model not loading** → Check internet connection or use fallback rule-based mode.  

---

## 📂 Project Structure  

```
finance-ai/
│── finance_chatbot.py          # Main Streamlit app
│── README.md                   # Documentation
│── requirements.txt            # Dependencies (optional)
```

---

## ⚠️ Disclaimer  

This chatbot provides **general financial education only**.  
It is **not a substitute for professional financial advice**.  
Consult certified professionals for personalized investment or tax planning.  
