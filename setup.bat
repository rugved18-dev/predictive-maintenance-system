@echo off
REM Quick Setup Script for Predictive Maintenance System - Windows Version

echo 🔧 Predictive Maintenance System - Quick Setup
echo ==============================================

REM Check Python version
echo 📋 Checking Python version...
python --version

REM Create virtual environment
echo 🌐 Creating virtual environment...
python -m venv pm_env

REM Activate virtual environment
echo ⚡ Activating virtual environment...
call pm_env\Scripts\activate.bat

REM Install dependencies
echo 📦 Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ✅ Setup complete!
echo.
echo 🚀 To start the system:
echo 1. Activate environment:
echo    pm_env\Scripts\activate.bat        # Command Prompt
echo    .\pm_env\Scripts\Activate.ps1      # PowerShell
echo.
echo 2. Start backend API:
echo    uvicorn main:app --reload --host 0.0.0.0 --port 8000
echo.
echo 3. Start dashboard (in new terminal):
echo    streamlit run app.py
echo.
echo 📖 For detailed instructions, see README.md

pause