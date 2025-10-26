# Quick Start Guide

## Automated One-Command Setup

This project includes automated scripts that will handle everything for you!

### For macOS/Linux Users

Simply run:

```bash
./run_project.sh
```

### For Windows Users

Simply run:

```bash
run_project.bat
```

## What the Script Does Automatically

The script will perform all setup steps in order:

1. **Check Python Installation** - Verifies Python 3.8+ is installed
2. **Create Virtual Environment** - Sets up isolated Python environment
3. **Install Dependencies** - Installs all required packages from `requirements.txt`
4. **Run Notebooks** (in order):
   - `1_data_generation.ipynb` - Generate synthetic demand data
   - `2_environment_creation.ipynb` - Create RL environment
   - `3_agent_training.ipynb` - Train PPO agent (takes ~5-10 minutes)
   - `4_evaluation_analysis.ipynb` - Evaluate and compare strategies
5. **Launch Streamlit Dashboard** - Opens interactive visualization at http://localhost:8501

## Estimated Time

- **First run**: 15-20 minutes (includes dependency installation and model training)
- **Subsequent runs**: 5-10 minutes (if model is already trained)

## What You'll See

The script provides colorful progress output:
- ✓ Green checkmarks for successful steps
- ✗ Red warnings if something fails (but continues anyway)
- Blue progress indicators for each phase

## Manual Setup (Optional)

If you prefer to run steps manually, see the main [README.md](README.md) for detailed instructions.

## Troubleshooting

### Script won't run on macOS/Linux
```bash
# Make sure the script is executable
chmod +x run_project.sh
```

### Python not found
Make sure Python 3.8 or higher is installed:
```bash
python3 --version
```

### Permission denied
On macOS, you might need to allow the script in System Preferences > Security & Privacy.

### Notebooks fail to execute
You can skip the notebook execution and just run the dashboard:
```bash
# Activate virtual environment first
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Then run dashboard directly
streamlit run streamlit_app.py
```

## Stopping the Dashboard

Press `Ctrl+C` in the terminal to stop the Streamlit server.

## Re-running the Project

Just run the same command again:
```bash
./run_project.sh        # macOS/Linux
run_project.bat         # Windows
```

The script will:
- Reuse the existing virtual environment
- Skip installing dependencies if already installed
- Re-run notebooks with fresh data
- Launch the dashboard

---

**Note:** The first time you run this, it will take longer because it needs to:
- Install ~20 Python packages
- Train the RL agent for 100,000 timesteps (~5-10 minutes)

Subsequent runs will be much faster!
