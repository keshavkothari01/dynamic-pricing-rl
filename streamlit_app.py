"""
Dynamic Pricer using RL - Interactive Dashboard
Real-time visualization of the RL agent's pricing decisions
"""

import streamlit as st
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
import time
import os

# Set page config
st.set_page_config(
    page_title="Dynamic Pricer using RL",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Theme Adaptive
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Metric styling - works on both themes */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    /* Metric containers with colored backgrounds - visible on both themes */
    div[data-testid="metric-container"] {
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        border: 2px solid !important;
    }
    
    /* Individual metric box colors */
    [data-testid="column"]:nth-child(1) [data-testid="metric-container"] {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
        border-color: #3b82f6 !important;
    }
    
    [data-testid="column"]:nth-child(1) [data-testid="stMetricValue"],
    [data-testid="column"]:nth-child(1) [data-testid="stMetricLabel"],
    [data-testid="column"]:nth-child(1) [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    
    [data-testid="column"]:nth-child(2) [data-testid="metric-container"] {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        border-color: #10b981 !important;
    }
    
    [data-testid="column"]:nth-child(2) [data-testid="stMetricValue"],
    [data-testid="column"]:nth-child(2) [data-testid="stMetricLabel"],
    [data-testid="column"]:nth-child(2) [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    
    [data-testid="column"]:nth-child(3) [data-testid="metric-container"] {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        border-color: #ef4444 !important;
    }
    
    [data-testid="column"]:nth-child(3) [data-testid="stMetricValue"],
    [data-testid="column"]:nth-child(3) [data-testid="stMetricLabel"],
    [data-testid="column"]:nth-child(3) [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    
    [data-testid="column"]:nth-child(4) [data-testid="metric-container"] {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
        border-color: #8b5cf6 !important;
    }
    
    [data-testid="column"]:nth-child(4) [data-testid="stMetricValue"],
    [data-testid="column"]:nth-child(4) [data-testid="stMetricLabel"],
    [data-testid="column"]:nth-child(4) [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    
    .stProgress > div > div > div > div {
        background-color: #4a90e2 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Environment Definition
class DynamicPricingEnv(gym.Env):
    """Custom Gym Environment for Dynamic Pricing"""
    
    def __init__(self, max_days=365):
        super(DynamicPricingEnv, self).__init__()
        
        self.max_days = max_days
        self.current_day = 0
        self.last_price = 100
        
        self.min_price = 50
        self.max_price = 150
        self.price_step = 10
        self.prices = np.arange(self.min_price, self.max_price + 1, self.price_step)
        
        self.base_demand = 1000
        self.optimal_price = 100
        self.price_elasticity = -1.5
        
        self.action_space = spaces.Discrete(len(self.prices))
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )
        
    def _get_competitor_price(self):
        competitor_price = self.optimal_price + np.random.normal(0, 15)
        return np.clip(competitor_price, self.min_price, self.max_price)
    
    def _get_seasonal_factor(self):
        return 1 + 0.3 * np.sin(2 * np.pi * self.current_day / 365)
    
    def _calculate_demand(self, price, competitor_price):
        price_ratio = price / self.optimal_price
        elasticity_effect = np.power(price_ratio, self.price_elasticity)
        seasonal_factor = self._get_seasonal_factor()
        competitor_effect = 1 - 0.2 * (competitor_price < price)
        demand = self.base_demand * elasticity_effect * seasonal_factor * competitor_effect
        noise = np.random.normal(1, 0.1)
        demand = demand * noise
        return max(0, demand)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_day = 0
        self.last_price = 100
        observation = self._get_observation()
        info = {}
        return observation, info
    
    def _get_observation(self):
        competitor_price = self._get_competitor_price()
        obs = np.array([
            self.current_day / self.max_days,
            (self.last_price - self.min_price) / (self.max_price - self.min_price),
            (competitor_price - self.min_price) / (self.max_price - self.min_price)
        ], dtype=np.float32)
        return obs
    
    def step(self, action):
        price = self.prices[action]
        competitor_price = self._get_competitor_price()
        demand = self._calculate_demand(price, competitor_price)
        revenue = price * demand
        reward = revenue / 1000
        
        self.last_price = price
        self.current_day += 1
        
        terminated = self.current_day >= self.max_days
        truncated = False
        observation = self._get_observation()
        
        info = {
            'price': price,
            'demand': demand,
            'revenue': revenue,
            'competitor_price': competitor_price,
            'day': self.current_day
        }
        
        return observation, reward, terminated, truncated, info


# Initialize session state
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = {
        'days': [],
        'rl_prices': [],
        'rl_demands': [],
        'rl_revenues': [],
        'fixed_prices': [],
        'fixed_demands': [],
        'fixed_revenues': [],
        'random_prices': [],
        'random_demands': [],
        'random_revenues': [],
        'competitor_prices': []
    }
    st.session_state.simulation_running = False
    st.session_state.current_day = 0

# Header
st.markdown('<div class="main-header">Dynamic Pricer using RL</div>', unsafe_allow_html=True)
st.markdown("### Real-time Reinforcement Learning Agent vs Traditional Pricing Strategies")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Configuration Panel")
    
    # Model selection
    model_exists = os.path.exists('saved_models/pricing_agent_ppo.zip')
    
    if model_exists:
        st.success("Trained PPO model loaded")
        use_trained_model = st.checkbox("Use trained PPO model", value=True)
    else:
        st.warning("No trained model found. Using random policy.")
        use_trained_model = False
    
    st.markdown("---")
    
    # Simulation parameters
    st.subheader("Simulation Parameters")
    num_days = st.slider("Number of days", 10, 365, 100, help="Total days to simulate")
    simulation_speed = st.slider("Speed (days/sec)", 1, 20, 5, help="Simulation speed")
    fixed_price_value = st.slider("Fixed price baseline", 50, 150, 100, 10, help="Fixed pricing strategy value")
    
    st.markdown("---")
    
    # Control buttons
    st.subheader("Simulation Controls")
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("Start", use_container_width=True, type="primary")
        if start_btn:
            st.session_state.simulation_running = True
            st.session_state.current_day = 0
            # Reset data
            for key in st.session_state.simulation_data:
                st.session_state.simulation_data[key] = []
    
    with col2:
        pause_btn = st.button("Pause", use_container_width=True)
        if pause_btn:
            st.session_state.simulation_running = False
    
    reset_btn = st.button("Reset", use_container_width=True)
    if reset_btn:
        st.session_state.simulation_running = False
        st.session_state.current_day = 0
        for key in st.session_state.simulation_data:
            st.session_state.simulation_data[key] = []
        st.rerun()
    
    st.markdown("---")
    
    # Information
    st.subheader("About")
    st.markdown("""
    <div class="info-box">
    <b>Strategies:</b><br>
    • <b>RL Agent (PPO):</b> Learns optimal pricing<br>
    • <b>Fixed Price:</b> Constant pricing<br>
    • <b>Random Price:</b> Random selection<br><br>
    The RL agent adapts to market conditions in real-time.
    </div>
    """, unsafe_allow_html=True)

# Main content area
if not st.session_state.simulation_running and st.session_state.current_day == 0:
    # Welcome screen
    st.info("Configure settings in the sidebar and click **Start** to begin simulation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### RL Agent (PPO)")
        st.write("Learns optimal pricing through reinforcement learning")
        st.write("• Adapts to seasonal demand patterns")
        st.write("• Responds to competitor pricing")
        st.write("• Maximizes long-term revenue")
    
    with col2:
        st.markdown("### Fixed Price Strategy")
        st.write("Maintains constant pricing")
        st.write("• No market adaptation")
        st.write("• Ignores demand fluctuations")
        st.write("• Simple baseline for comparison")
    
    with col3:
        st.markdown("### Random Price Strategy")
        st.write("Random price selection")
        st.write("• No strategic decision-making")
        st.write("• High variance in performance")
        st.write("• Worst-case baseline")

else:
    # Load model if available
    if use_trained_model and model_exists:
        try:
            @st.cache_resource
            def load_model():
                return PPO.load('saved_models/pricing_agent_ppo')
            model = load_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            use_trained_model = False
    
    # Create environments
    env_rl = DynamicPricingEnv(max_days=num_days)
    env_fixed = DynamicPricingEnv(max_days=num_days)
    env_random = DynamicPricingEnv(max_days=num_days)
    
    # Real-time metrics
    metrics_container = st.container()
    with metrics_container:
        st.subheader("Performance Metrics")
        
        # Add progress bar
        if st.session_state.current_day > 0:
            progress = st.session_state.current_day / num_days
            st.progress(progress, text=f"Simulation Progress: {st.session_state.current_day}/{num_days} days")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_day_val = st.session_state.current_day if st.session_state.current_day > 0 else 0
            st.metric(
                "Current Day",
                f"{current_day_val}/{num_days}",
                delta=None
            )
        
        with col2:
            if st.session_state.simulation_data['rl_revenues']:
                total_rl = sum(st.session_state.simulation_data['rl_revenues'])
                current_rl_price = st.session_state.simulation_data['rl_prices'][-1] if st.session_state.simulation_data['rl_prices'] else 0
                st.metric(
                    "RL Total Revenue",
                    f"₹{total_rl:,.0f}",
                    delta=f"Current: ₹{current_rl_price:.0f}",
                    help="Total revenue generated by RL agent. Delta shows current price."
                )
        
        with col3:
            if st.session_state.simulation_data['fixed_revenues']:
                total_fixed = sum(st.session_state.simulation_data['fixed_revenues'])
                st.metric(
                    "Fixed Total Revenue",
                    f"₹{total_fixed:,.0f}",
                    delta=f"Fixed: ₹{fixed_price_value}",
                    help="Total revenue from fixed pricing"
                )
        
        with col4:
            if st.session_state.simulation_data['rl_revenues'] and st.session_state.simulation_data['fixed_revenues']:
                improvement = ((total_rl - total_fixed) / total_fixed) * 100
                st.metric(
                    "RL Performance",
                    f"{improvement:+.1f}%",
                    delta=f"vs Fixed Price",
                    delta_color="normal" if improvement > 0 else "inverse"
                )
        
        # Add current pricing display
        if st.session_state.current_day > 0 and st.session_state.simulation_data['rl_prices']:
            st.markdown("---")
            st.markdown("#### Current Pricing Decisions")
            price_col1, price_col2, price_col3 = st.columns(3)
            
            with price_col1:
                rl_current = st.session_state.simulation_data['rl_prices'][-1]
                st.markdown(f"**RL Agent:** ₹{rl_current:.0f}")
            
            with price_col2:
                fixed_current = st.session_state.simulation_data['fixed_prices'][-1]
                st.markdown(f"**Fixed Price:** ₹{fixed_current:.0f}")
            
            with price_col3:
                random_current = st.session_state.simulation_data['random_prices'][-1]
                st.markdown(f"**Random Price:** ₹{random_current:.0f}")
    
    # Charts container
    charts_container = st.container()
    
    # Simulation loop
    if st.session_state.simulation_running and st.session_state.current_day < num_days:
        # Initialize environments if first run
        if st.session_state.current_day == 0:
            obs_rl, _ = env_rl.reset(seed=42)
            obs_fixed, _ = env_fixed.reset(seed=42)
            obs_random, _ = env_random.reset(seed=42)
            st.session_state.obs_rl = obs_rl
            st.session_state.obs_fixed = obs_fixed
            st.session_state.obs_random = obs_random
        
        # Get observations
        obs_rl = st.session_state.get('obs_rl')
        obs_fixed = st.session_state.get('obs_fixed')
        obs_random = st.session_state.get('obs_random')
        
        # RL Agent action
        if use_trained_model and model_exists:
            # Use deterministic=False to show actual adaptive behavior
            # This allows the agent to explore and adapt to different states
            action_rl, _ = model.predict(obs_rl, deterministic=False)
        else:
            action_rl = env_rl.action_space.sample()
        
        # Fixed price action
        action_fixed = np.argmin(np.abs(env_fixed.prices - fixed_price_value))
        
        # Random action
        action_random = env_random.action_space.sample()
        
        # Step environments
        obs_rl, reward_rl, done_rl, _, info_rl = env_rl.step(action_rl)
        obs_fixed, reward_fixed, done_fixed, _, info_fixed = env_fixed.step(action_fixed)
        obs_random, reward_random, done_random, _, info_random = env_random.step(action_random)
        
        # Store observations
        st.session_state.obs_rl = obs_rl
        st.session_state.obs_fixed = obs_fixed
        st.session_state.obs_random = obs_random
        
        # Update data
        st.session_state.simulation_data['days'].append(st.session_state.current_day + 1)
        
        st.session_state.simulation_data['rl_prices'].append(info_rl['price'])
        st.session_state.simulation_data['rl_demands'].append(info_rl['demand'])
        st.session_state.simulation_data['rl_revenues'].append(info_rl['revenue'])
        
        st.session_state.simulation_data['fixed_prices'].append(info_fixed['price'])
        st.session_state.simulation_data['fixed_demands'].append(info_fixed['demand'])
        st.session_state.simulation_data['fixed_revenues'].append(info_fixed['revenue'])
        
        st.session_state.simulation_data['random_prices'].append(info_random['price'])
        st.session_state.simulation_data['random_demands'].append(info_random['demand'])
        st.session_state.simulation_data['random_revenues'].append(info_random['revenue'])
        
        st.session_state.simulation_data['competitor_prices'].append(info_rl['competitor_price'])
        
        st.session_state.current_day += 1
        
        # Auto-advance
        time.sleep(1.0 / simulation_speed)
        st.rerun()
    
    elif st.session_state.current_day >= num_days:
        st.session_state.simulation_running = False
        st.success(f"Simulation completed ({num_days} days)")
    
    # Display charts
    with charts_container:
        if st.session_state.simulation_data['days']:
            # Create DataFrame
            df = pd.DataFrame({
                'Day': st.session_state.simulation_data['days'],
                'RL Price': st.session_state.simulation_data['rl_prices'],
                'Fixed Price': st.session_state.simulation_data['fixed_prices'],
                'Random Price': st.session_state.simulation_data['random_prices'],
                'RL Revenue': st.session_state.simulation_data['rl_revenues'],
                'Fixed Revenue': st.session_state.simulation_data['fixed_revenues'],
                'Random Revenue': st.session_state.simulation_data['random_revenues'],
                'Competitor Price': st.session_state.simulation_data['competitor_prices'],
            })
            
            # Tabs for better organization
            tab1, tab2, tab3 = st.tabs(["Pricing Strategies", "Revenue Analysis", "Performance Summary"])
            
            with tab1:
                # Price comparison chart
                st.subheader("Price Strategies Over Time")
                fig1, ax1 = plt.subplots(figsize=(12, 4))
                ax1.plot(df['Day'], df['RL Price'], label='RL Agent', linewidth=2, color='#1f77b4')
                ax1.plot(df['Day'], df['Fixed Price'], label='Fixed Price', linewidth=2, color='#ff7f0e', linestyle='--')
                ax1.plot(df['Day'], df['Random Price'], label='Random Price', linewidth=1, color='#d62728', alpha=0.5)
                ax1.plot(df['Day'], df['Competitor Price'], label='Competitor', linewidth=1, color='gray', linestyle=':')
                ax1.set_xlabel('Day', fontweight='bold')
                ax1.set_ylabel('Price (₹)', fontweight='bold')
                ax1.legend(loc='upper right')
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
                plt.close()
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Daily Revenue Comparison")
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    ax2.plot(df['Day'], df['RL Revenue'], label='RL Agent', linewidth=2, color='#2ca02c')
                    ax2.plot(df['Day'], df['Fixed Revenue'], label='Fixed Price', linewidth=2, color='#ff7f0e')
                    ax2.plot(df['Day'], df['Random Revenue'], label='Random', linewidth=1, color='#d62728', alpha=0.5)
                    ax2.set_xlabel('Day', fontweight='bold')
                    ax2.set_ylabel('Revenue (₹)', fontweight='bold')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2)
                    plt.close()
                
                with col2:
                    st.subheader("Cumulative Revenue")
                    fig3, ax3 = plt.subplots(figsize=(6, 4))
                    ax3.plot(df['Day'], df['RL Revenue'].cumsum(), label='RL Agent', linewidth=2, color='#2ca02c')
                    ax3.plot(df['Day'], df['Fixed Revenue'].cumsum(), label='Fixed Price', linewidth=2, color='#ff7f0e')
                    ax3.plot(df['Day'], df['Random Revenue'].cumsum(), label='Random', linewidth=2, color='#d62728')
                    ax3.set_xlabel('Day', fontweight='bold')
                    ax3.set_ylabel('Cumulative Revenue (₹)', fontweight='bold')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    st.pyplot(fig3)
                    plt.close()
            
            with tab3:
                # Summary statistics (shown during and after simulation)
                st.subheader("Summary Statistics")
                
                summary_data = {
                    'Strategy': ['RL Agent (PPO)', 'Fixed Price', 'Random Price'],
                    'Total Revenue': [
                        df['RL Revenue'].sum(),
                        df['Fixed Revenue'].sum(),
                        df['Random Revenue'].sum()
                    ],
                    'Avg Daily Revenue': [
                        df['RL Revenue'].mean(),
                        df['Fixed Revenue'].mean(),
                        df['Random Revenue'].mean()
                    ],
                    'Avg Price': [
                        df['RL Price'].mean(),
                        df['Fixed Price'].mean(),
                        df['Random Price'].mean()
                    ],
                    'Price Volatility': [
                        df['RL Price'].std(),
                        df['Fixed Price'].std(),
                        df['Random Price'].std()
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df['Total Revenue'] = summary_df['Total Revenue'].apply(lambda x: f"₹{x:,.0f}")
                summary_df['Avg Daily Revenue'] = summary_df['Avg Daily Revenue'].apply(lambda x: f"₹{x:,.0f}")
                summary_df['Avg Price'] = summary_df['Avg Price'].apply(lambda x: f"₹{x:.2f}")
                summary_df['Price Volatility'] = summary_df['Price Volatility'].apply(lambda x: f"₹{x:.2f}")
                
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                if st.session_state.current_day >= num_days:
                    # Performance comparison KPIs
                    st.subheader("Performance Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    rl_total = df['RL Revenue'].sum()
                    fixed_total = df['Fixed Revenue'].sum()
                    random_total = df['Random Revenue'].sum()
                    
                    with col1:
                        improvement = ((rl_total - fixed_total) / fixed_total) * 100
                        st.metric(
                            "RL vs Fixed Price",
                            f"{improvement:+.1f}%",
                            delta="Revenue Improvement" if improvement > 0 else "Revenue Loss"
                        )
                    
                    with col2:
                        vs_random = ((rl_total - random_total) / random_total) * 100
                        st.metric(
                            "RL vs Random",
                            f"{vs_random:+.1f}%",
                            delta="Revenue Improvement" if vs_random > 0 else "Revenue Loss"
                        )
                    
                    with col3:
                        best_strategy = summary_data['Strategy'][
                            [df['RL Revenue'].sum(), df['Fixed Revenue'].sum(), df['Random Revenue'].sum()].index(
                                max([df['RL Revenue'].sum(), df['Fixed Revenue'].sum(), df['Random Revenue'].sum()])
                            )
                        ]
                        st.metric(
                            "Best Performing Strategy",
                            best_strategy,
                            delta="Winner"
                        )
                
                # Performance comparison
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    rl_total = df['RL Revenue'].sum()
                    fixed_total = df['Fixed Revenue'].sum()
                    improvement = ((rl_total - fixed_total) / fixed_total) * 100
                    st.metric(
                        "RL vs Fixed",
                        f"{improvement:+.1f}%",
                        delta="Better" if improvement > 0 else "Worse"
                    )
                
                with col2:
                    random_total = df['Random Revenue'].sum()
                    vs_random = ((rl_total - random_total) / random_total) * 100
                    st.metric(
                        "RL vs Random",
                        f"{vs_random:+.1f}%",
                        delta="Better" if vs_random > 0 else "Worse"
                    )
                
                with col3:
                    best_strategy = summary_data['Strategy'][
                        [df['RL Revenue'].sum(), df['Fixed Revenue'].sum(), df['Random Revenue'].sum()].index(
                            max([df['RL Revenue'].sum(), df['Fixed Revenue'].sum(), df['Random Revenue'].sum()])
                        )
                    ]
                    st.metric(
                        "Best Strategy",
                        best_strategy,
                        delta="Winner 🏆"
                    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0 1rem 0;'>
    <p style='font-size: 0.9rem; margin-bottom: 0.5rem;'><b>Dynamic Pricer using RL</b> | Reinforcement Learning Project 2025</p>
    <p style='font-size: 0.85rem; color: #888;'>Built with Streamlit, Stable-Baselines3, and Gymnasium</p>
</div>
""", unsafe_allow_html=True)
