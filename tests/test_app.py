"""
============================================================
🧪 Test Script for FinanceAI
============================================================

This script verifies:
    - Required packages are installed
    - Core functions from finance_chatbot_fixed1.py work
    - Utils folder functions run without errors
    - Streamlit app entry point loads

Run:
    python test_app.py
============================================================
"""

import importlib
import sys


def check_import(package):
    try:
        importlib.import_module(package)
        print(f"✅ {package} is installed")
        return True
    except ImportError:
        print(f"❌ {package} is NOT installed")
        return False


def test_core_functions():
    try:
        import finance_chatbot_fixed1 as app

        # Test input validation
        valid, msg = app.validate_input("Test message")
        assert valid, "Validation failed for valid input"

        # Test fallback response
        response = app.get_fallback_response("budget")
        assert isinstance(response, str) and len(response) > 0

        # Test chart functions
        fig1 = app.create_spending_chart()
        fig2 = app.create_savings_progress()
        assert fig1 is not None and fig2 is not None

        print("✅ Core functions work correctly")
    except Exception as e:
        print(f"❌ Core function tests failed: {e}")


def test_utils_functions():
    try:
        from utils import (
            validate_input,
            get_fallback_response,
            create_spending_chart,
            create_savings_progress,
            initialize_session_state,
        )

        # Run sample tests
        valid, _ = validate_input("Hello finance")
        assert valid

        response = get_fallback_response("save money")
        assert isinstance(response, str)

        fig1 = create_spending_chart()
        fig2 = create_savings_progress()
        assert fig1 is not None and fig2 is not None

        print("✅ Utils functions work correctly")
    except Exception as e:
        print(f"❌ Utils tests failed: {e}")


def test_app_entry():
    try:
        import finance_chatbot_fixed1

        assert hasattr(finance_chatbot_fixed1, "main")
        print("✅ App entry point exists (main function)")
    except Exception as e:
        print(f"❌ App entry test failed: {e}")


if __name__ == "__main__":
    print("🚀 Running FinanceAI Tests...\n")

    # Check core dependencies
    dependencies = ["streamlit", "matplotlib", "numpy"]
    optional = ["transformers", "torch"]

    print("📦 Checking required packages...")
    for dep in dependencies + optional:
        check_import(dep)

    print("\n🔍 Testing core functions...")
    test_core_functions()

    print("\n🔍 Testing utils functions...")
    test_utils_functions()

    print("\n🔍 Testing app entry...")
    test_app_entry()

    print("\n✅ All tests completed.")
