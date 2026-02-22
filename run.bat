@echo off
title Amodal Segmentation Analysis Tool — Setup
echo.
echo ============================================================
echo   Amodal Instance Segmentation Analysis Tool
echo   Setup ^& Launch Script for Windows
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not on PATH.
    echo        Download from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python found.

:: Create venv if not exists
if not exist "venv\" (
    echo.
    echo Creating virtual environment ...
    python -m venv venv
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment exists.
)

:: Activate
call venv\Scripts\activate.bat

:: Check PyTorch
python -c "import torch; print(torch.__version__)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ============================================================
    echo   PyTorch is NOT installed in the virtual environment.
    echo.
    echo   Please install it manually with CUDA support:
    echo.
    echo     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    echo.
    echo   For other CUDA versions, see https://pytorch.org/get-started/locally/
    echo   Then run this script again.
    echo ============================================================
    pause
    exit /b 1
)
echo [OK] PyTorch installed.

:: Install remaining deps
echo.
echo Installing dependencies from requirements.txt ...
pip install -r requirements.txt --quiet
echo [OK] Dependencies installed.

:: Launch
echo.
echo ============================================================
echo   Launching application ...
echo ============================================================
echo.
python app.py

pause
