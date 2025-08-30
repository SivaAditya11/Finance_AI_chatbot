"""Chart generation utilities"""

import matplotlib.pyplot as plt
import numpy as np


def create_spending_chart():
    """Generate pie chart for monthly budget breakdown"""
    categories = ["Housing", "Food", "Transport", "Savings", "Utilities", "Fun"]
    amounts = [2000, 800, 600, 1200, 400, 500]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(amounts, labels=categories, autopct="%1.1f%%", startangle=90)
    ax.set_title("Monthly Budget Breakdown")
    return fig


def create_savings_progress():
    """Generate savings goal progress chart"""
    goals = ["Emergency Fund", "Vacation", "Retirement"]
    current = [5000, 2000, 30000]
    targets = [10000, 5000, 100000]

    progress = [min(c / t * 100, 100) for c, t in zip(current, targets)]
    y_pos = np.arange(len(goals))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y_pos, progress, color="green")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(goals)
    ax.set_xlabel("Progress (%)")
    ax.set_title("Savings Goals Progress")
    return fig
