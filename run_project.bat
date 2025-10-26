@echo off
REM Dynamic Pricing RL Project - Automated Setup and Run Script (Windows)
REM This script will:
REM 1. Create virtual environment
REM 2. Install dependencies
REM 3. Run all notebooks in order
REM 4. Launch Streamlit dashboard

echo ==========================================
echo Dynamic Pricing RL - Automated Setup
echo ==========================================
echo.

REM Step 1: Check Python installation
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [OK] Found: %PYTHON_VERSION%
echo.

REM Step 2: Create and activate virtual environment
echo [2/5] Setting up virtual environment...
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Step 3: Install dependencies
echo [3/5] Installing dependencies...
echo This may take a few minutes...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
echo [OK] All dependencies installed
echo.

REM Step 4: Run notebooks in order
echo [4/5] Running Jupyter Notebooks...
echo This will generate data, create environment, train model, and evaluate results.
echo.

REM Install jupyter if not already installed
pip install jupyter nbconvert >nul 2>&1

echo   Running notebook 1/4: 1_data_generation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/1_data_generation.ipynb >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Completed: 1_data_generation.ipynb
) else (
    echo   [WARNING] Failed: 1_data_generation.ipynb
)

echo   Running notebook 2/4: 2_environment_creation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/2_environment_creation.ipynb >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Completed: 2_environment_creation.ipynb
) else (
    echo   [WARNING] Failed: 2_environment_creation.ipynb
)

echo   Running notebook 3/4: 3_agent_training.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/3_agent_training.ipynb >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Completed: 3_agent_training.ipynb
) else (
    echo   [WARNING] Failed: 3_agent_training.ipynb
)

echo   Running notebook 4/4: 4_evaluation_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/4_evaluation_analysis.ipynb >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Completed: 4_evaluation_analysis.ipynb
) else (
    echo   [WARNING] Failed: 4_evaluation_analysis.ipynb
)

echo.
echo [OK] All notebooks executed
echo.

REM Step 5: Launch Streamlit Dashboard
echo [5/5] Launching Streamlit Dashboard...
echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo The Streamlit dashboard will open in your browser.
echo Dashboard URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard.
echo.
echo ==========================================
echo.

REM Launch Streamlit
streamlit run streamlit_app.py
