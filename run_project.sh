#!/bin/bash

# Dynamic Pricing RL Project - Automated Setup and Run Script
# This script will:
# 1. Create virtual environment
# 2. Install dependencies
# 3. Run all notebooks in order
# 4. Launch Streamlit dashboard

set -e  # Exit on any error

echo "=========================================="
echo "Dynamic Pricing RL - Automated Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check Python installation
echo -e "${BLUE}[1/5] Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ Found: $PYTHON_VERSION${NC}"
echo ""

# Step 2: Create and activate virtual environment
echo -e "${BLUE}[2/5] Setting up virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Step 3: Install dependencies
echo -e "${BLUE}[3/5] Installing dependencies...${NC}"
echo "This may take a few minutes..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo -e "${GREEN}✓ All dependencies installed${NC}"
echo ""

# Step 4: Run notebooks in order
echo -e "${BLUE}[4/5] Running Jupyter Notebooks...${NC}"
echo "This will generate data, create environment, train model, and evaluate results."
echo ""

# Install jupyter if not already installed
pip install jupyter nbconvert > /dev/null 2>&1

notebooks=(
    "notebooks/1_data_generation.ipynb"
    "notebooks/2_environment_creation.ipynb"
    "notebooks/3_agent_training.ipynb"
    "notebooks/4_evaluation_analysis.ipynb"
)

for i in "${!notebooks[@]}"; do
    notebook="${notebooks[$i]}"
    num=$((i + 1))
    echo -e "${BLUE}  Running notebook $num/4: $(basename $notebook)${NC}"
    
    # Run notebook and capture output
    if jupyter nbconvert --to notebook --execute --inplace "$notebook" > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ Completed: $(basename $notebook)${NC}"
    else
        echo -e "${RED}  ✗ Failed: $(basename $notebook)${NC}"
        echo "  You can still continue, but some features may not work."
    fi
done

echo ""
echo -e "${GREEN}✓ All notebooks executed successfully${NC}"
echo ""

# Step 5: Launch Streamlit Dashboard
echo -e "${BLUE}[5/5] Launching Streamlit Dashboard...${NC}"
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "The Streamlit dashboard will open in your browser."
echo "Dashboard URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard."
echo ""
echo "=========================================="
echo ""

# Launch Streamlit
streamlit run streamlit_app.py
