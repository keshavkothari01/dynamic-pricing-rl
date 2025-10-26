# 🎯 Dynamic Pricing using Reinforcement Learning

**A complete academic project demonstrating AI-powered dynamic pricing optimization**

---

## 📋 Project Overview

### Title
**Dynamic Pricing using Reinforcement Learning**

### Objective
To build a Reinforcement Learning (RL) based agent that learns to set optimal prices dynamically for a product or service, in order to maximize total revenue or profit. The agent interacts with a simulated market environment built using synthetic data, where customer demand depends on factors like price, competitor pricing, and seasonality.

### Problem Statement
Traditional businesses use static or rule-based pricing models. However, in real-world scenarios like e-commerce, airline ticketing, and ride-hailing, demand constantly changes, and so should prices.

This project aims to simulate such an environment and train an AI agent that:
- Observes the market (demand trends, competitor price)
- Chooses prices dynamically
- Learns pricing strategy through reward feedback (profit)
- Outperforms fixed or random pricing methods

---

## 🏗️ Project Structure

```
dynamic_pricing_rl_project/
│
├── 📁 notebooks/
│   ├── 1_data_generation.ipynb         # Simulate demand dataset
│   ├── 2_environment_creation.ipynb    # Define & test Gym environment
│   ├── 3_agent_training.ipynb          # Train RL agent (PPO)
│   ├── 4_evaluation_analysis.ipynb     # Evaluate, visualize, compare baselines
│   └── 5_report_summary.ipynb          # Final analysis & presentation
│
├── 📁 saved_models/
│   └── pricing_agent_ppo.zip           # Trained PPO model
│
├── 📁 data/
│   └── simulated_demand.csv            # Synthetic demand data
│
├── 📁 visuals/
│   ├── price_vs_demand.png             # Price-demand relationship
│   ├── reward_curve.png                # Learning progress
│   └── performance_comparison.png      # Strategy comparison
│
├── streamlit_app.py                    # 🌟 Interactive dashboard
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── SETUP_GUIDE.md                      # Installation instructions
├── STREAMLIT_GUIDE.md                  # Dashboard usage guide
├── PROJECT_COMPLETE.md                 # Success summary
└── STEP_BY_STEP_GUIDE.md               # 📚 Detailed file-by-file guide
```

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Programming Language** | Python 3.x |
| **RL Framework** | Stable-Baselines3 |
| **Environment** | Gymnasium (OpenAI Gym) |
| **Algorithm** | PPO (Proximal Policy Optimization) |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Interactive Dashboard** | Streamlit |
| **Deep Learning Backend** | PyTorch |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project**
   ```bash
   cd dynamic_pricing_rl_project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import gymnasium, stable_baselines3; print('✅ Installation successful!')"
   ```

---

## 🚀 Usage

### Option 1: Interactive Streamlit Dashboard 🌟 (Recommended!)

Run the real-time interactive visualization:

```bash
streamlit run streamlit_app.py
```

**Features:**
- 🎮 Real-time simulation controls
- 📊 Live charts comparing RL vs baselines
- 💰 Revenue tracking and performance metrics
- 🏆 Strategy comparison and winner announcement
- ⚙️ Configurable simulation parameters

See [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md) for detailed instructions.

### Option 2: Jupyter Notebooks

Execute notebooks in order:

1. **Data Generation** (`1_data_generation.ipynb`)
   - Generates synthetic demand data
   - Simulates price elasticity, seasonality, and competitor pricing
   - Saves data to `data/simulated_demand.csv`

2. **Environment Creation** (`2_environment_creation.ipynb`)
   - Defines custom `DynamicPricingEnv` Gym environment
   - Sets up observation and action spaces
   - Tests environment with random actions

3. **Agent Training** (`3_agent_training.ipynb`)
   - Initializes PPO agent
   - Trains for 100,000 timesteps
   - Saves trained model to `saved_models/`
   - Visualizes learning curve

4. **Evaluation & Analysis** (`4_evaluation_analysis.ipynb`)
   - Loads trained model
   - Compares RL agent vs. baselines (Fixed, Random)
   - Generates performance visualizations
   - Analyzes pricing behavior

5. **Report Summary** (`5_report_summary.ipynb`)
   - Complete project summary
   - Key findings and conclusions
   - Future work recommendations

---

## 📊 Key Results

### Performance Comparison

The RL agent (PPO) demonstrates superior performance:

- **✅ Revenue Maximization**: Highest total revenue among all strategies
- **✅ Adaptive Pricing**: Dynamically adjusts to market conditions
- **✅ Learning Efficiency**: Converges within 100K timesteps
- **✅ Robustness**: Maintains performance across seasons

### Baseline Comparisons

| Strategy | Description | Performance |
|----------|-------------|-------------|
| **RL Agent (PPO)** | Learned dynamic pricing | 🏆 Best |
| **Fixed (₹100)** | Constant price at ₹100 | Baseline |
| **Fixed (₹120)** | Constant price at ₹120 | Below optimal |
| **Random** | Random price selection | Worst |

---

## 🎯 Real-World Applications

This approach can be applied to:

1. **E-commerce**: Dynamic product pricing based on demand
2. **Airlines**: Ticket pricing optimization
3. **Ride-hailing**: Surge pricing (Uber, Lyft)
4. **Hotels**: Room rate optimization
5. **Cloud Services**: Resource pricing
6. **Energy**: Electricity pricing (peak/off-peak)

---

## 🧠 Technical Details

### Environment Specifications

- **Observation Space**: `[day_normalized, last_price_normalized, competitor_price_normalized]`
- **Action Space**: Discrete(11) - prices from ₹50 to ₹150 (step: ₹10)
- **Reward Function**: `Revenue = Price × Demand` (normalized)
- **Episode Length**: 365 days (1 year)

### Demand Function

```python
demand = base_demand × price_elasticity × seasonal_factor × competitor_effect × noise
```

Where:
- **Price Elasticity**: -1.5 (demand decreases with price)
- **Seasonal Factor**: Sine wave (peaks during festive seasons)
- **Competitor Effect**: Reduces demand if competitor price is lower
- **Noise**: Random market uncertainty (±10%)

---

## 📈 Future Improvements

1. **Real Data Integration**: Use actual sales and market data
2. **Multi-Product Pricing**: Handle product portfolios
3. **Advanced Algorithms**: Experiment with A3C, SAC, TD3
4. **Constraint Handling**: Add business rules (minimum margins)
5. **Deployment**: Create API for real-time recommendations
6. **Multi-Agent**: Simulate competing pricing agents
7. **Explainability**: Add interpretability to pricing decisions

---

## 📚 References

### Papers
1. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms"
2. Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction"
3. Den Boer, A. V. (2015). "Dynamic Pricing and Learning: Historical Origins, Current Research, and New Directions"

### Documentation
- [Gymnasium (OpenAI Gym)](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [PyTorch](https://pytorch.org/)

---

## 👨‍🎓 Academic Use

This project is suitable for:
- ✅ Reinforcement Learning course projects
- ✅ Machine Learning applications
- ✅ Operations Research demonstrations
- ✅ AI/Business optimization case studies

### Learning Outcomes
- Understanding of RL fundamentals (MDP, rewards, policies)
- Practical implementation with Stable-Baselines3
- Custom environment creation with Gymnasium
- Performance evaluation methodologies
- Real-world AI/ML application

---

## 📝 License

This project is created for educational purposes.

---

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Extend the project

---

## 📧 Contact

For questions or suggestions, please create an issue in the project repository.

---

**🎉 Happy Learning!**

*Demonstrating the power of Reinforcement Learning in solving real-world business optimization problems.*
