#!/bin/bash
# Quick Setup Script for Predictive Maintenance System
# Compatible with both Windows (Git Bash) and Unix-like systems

echo "🔧 Predictive Maintenance System - Quick Setup"
echo "=============================================="

# Check Python version
echo "📋 Checking Python version..."
python --version

# Create virtual environment
echo "🌐 Creating virtual environment..."
python -m venv pm_env

# Activate virtual environment (platform-specific)
echo "⚡ Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source pm_env/Scripts/activate
else
    # macOS/Linux
    source pm_env/bin/activate
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "🚀 To start the system:"
echo "1. Activate environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source pm_env/Scripts/activate    # Windows Git Bash"
    echo "   .\\pm_env\\Scripts\\Activate.ps1     # Windows PowerShell"
else
    echo "   source pm_env/bin/activate        # macOS/Linux"
fi
echo ""
echo "2. Start backend API:"
echo "   uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "3. Start dashboard (in new terminal):"
echo "   streamlit run app.py"
echo ""
echo "📖 For detailed instructions, see README.md"