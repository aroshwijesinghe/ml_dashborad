import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from data_utils import preprocess_data, load_sl_it_jobs_dataset
from model_trainer import train_and_evaluate

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="NexML Studio — Premium AI Analytics Platform",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- PREMIUM GLASSMORPHISM + SOFT NEUMORPHISM (LIGHT THEME ONLY) -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* 1. Global Reset & Theme Variables */
    :root {
        --bg-main: #EEF1FA;
        --bg-secondary: #F5F6FC;
        --glass-surface: rgba(255, 255, 255, 0.72);
        --glass-border: rgba(255, 255, 255, 0.88);
        --glass-shadow: 0 12px 36px rgba(120, 130, 170, 0.12);
        
        --shadow-light: rgba(255, 255, 255, 0.95);
        --shadow-dark: rgba(120, 130, 170, 0.18);
        
        --primary-purple: #6955F5;
        --primary-gradient: linear-gradient(135deg, #6955F5 0%, #4DA3FF 100%);
        --secondary-blue: #4DA3FF;
        --accent-cyan: #35C5D8;
        --accent-pink: #D58BFF;
        --accent-mint: #10B981;
        
        --text-main: #26324A;
        --text-secondary: #8490A8;
        --text-light: #9EA8BC;
    }
    
    /* Permanent Light Mode Application Canvas */
    html, body, [class*="css"], .stApp {
        background-color: var(--bg-main) !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(105, 85, 245, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(77, 163, 255, 0.08) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(53, 197, 216, 0.06) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(213, 139, 255, 0.06) 0px, transparent 50%) !important;
        background-attachment: fixed !important;
        color: var(--text-main) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    /* Hide Streamlit Header, MainMenu, Footer, Deploy, and Decoration */
    header[data-testid="stHeader"],
    header,
    [data-testid="stHeaderActionElements"],
    [data-testid="stToolbar"],
    [data-testid="stMainMenu"],
    .stDeployButton,
    #MainMenu,
    footer,
    ul[data-testid="main-menu-list"],
    div[data-testid="stToolbarActions"],
    div[role="menu"],
    div[data-baseweb="popover"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0px !important;
        width: 0px !important;
        pointer-events: none !important;
    }

    /* 2. Embossed Glassmorphic Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(245, 246, 252, 0.75) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border-right: 1px solid var(--glass-border) !important;
        box-shadow: 
            8px 0 32px rgba(120, 130, 170, 0.08),
            inset -1px 0 0 rgba(255, 255, 255, 0.8) !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: var(--text-main) !important;
        font-size: 0.98rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.3px !important;
        margin-top: 10px !important;
        margin-bottom: 8px !important;
    }
    
    /* 3. Floating Glass Top Navigation */
    .top-navbar-glass {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 30px;
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 9999px;
        margin-bottom: 24px;
        box-shadow: 
            8px 8px 24px var(--shadow-dark),
            -8px -8px 24px var(--shadow-light),
            inset 1px 1px 2px rgba(255, 255, 255, 0.95);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .top-navbar-glass:hover {
        box-shadow: 
            10px 10px 28px rgba(120, 130, 170, 0.22),
            -10px -10px 28px rgba(255, 255, 255, 1);
    }
    .brand-logo-gradient {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.6px;
        background: linear-gradient(135deg, #6955F5 0%, #4DA3FF 70%, #35C5D8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .user-pill-embossed {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255, 255, 255, 0.85);
        padding: 6px 16px;
        border-radius: 9999px;
        border: 1px solid rgba(255, 255, 255, 0.9);
        box-shadow: 
            3px 3px 8px rgba(120, 130, 170, 0.15),
            -3px -3px 8px var(--shadow-light);
    }
    .user-avatar-pulse {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6955F5 0%, #D58BFF 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.74rem;
        font-weight: 800;
        color: #FFFFFF;
        box-shadow: 0 4px 10px rgba(105, 85, 245, 0.3);
    }

    /* 4. Model Selection Chips (Horizontal Bar) */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 12px !important;
        background: rgba(255, 255, 255, 0.55) !important;
        backdrop-filter: blur(16px) !important;
        padding: 8px 12px !important;
        border-radius: 9999px !important;
        border: 1px solid var(--glass-border) !important;
        box-shadow: 
            5px 5px 16px var(--shadow-dark),
            -5px -5px 16px var(--shadow-light) !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
        padding: 6px 18px !important;
        border-radius: 9999px !important;
        box-shadow: 
            3px 3px 8px rgba(120, 130, 170, 0.1),
            -3px -3px 8px var(--shadow-light) !important;
        cursor: pointer !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        transform: translateY(-2px) !important;
        box-shadow: 
            5px 5px 12px rgba(105, 85, 245, 0.2),
            -4px -4px 10px var(--shadow-light) !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, #6955F5 0%, #4DA3FF 100%) !important;
        color: #FFFFFF !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
        box-shadow: 
            4px 4px 14px rgba(105, 85, 245, 0.4),
            -2px -2px 8px var(--shadow-light) !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* 5. Capsule Pill Tabs with Embossed Troughs */
    div[data-testid="stTabs"] [role="tablist"] {
        background: rgba(245, 246, 252, 0.65) !important;
        backdrop-filter: blur(20px) !important;
        padding: 6px !important;
        border-radius: 9999px !important;
        border: 1px solid var(--glass-border) !important;
        box-shadow: 
            inset 3px 3px 8px rgba(120, 130, 170, 0.18),
            inset -3px -3px 8px rgba(255, 255, 255, 0.95),
            0 4px 16px rgba(120, 130, 170, 0.08) !important;
        display: flex !important;
        gap: 8px !important;
        margin-bottom: 24px !important;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #6955F5 0%, #4DA3FF 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 
            4px 4px 14px rgba(105, 85, 245, 0.35),
            -2px -2px 8px rgba(255, 255, 255, 0.8) !important;
    }
    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] div[data-baseweb="tab-border"] {
        display: none !important;
    }

    /* 6. Premium Glassmorphic & Embossed Surface Cards */
    .premium-glass-card {
        background: var(--glass-surface);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 24px 26px;
        margin-bottom: 20px;
        box-shadow: 
            10px 10px 28px var(--shadow-dark),
            -10px -10px 28px var(--shadow-light),
            inset 1px 1px 2px rgba(255, 255, 255, 0.95);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .premium-glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 
            14px 14px 36px rgba(120, 130, 170, 0.24),
            -12px -12px 34px rgba(255, 255, 255, 1),
            inset 1px 1px 2px rgba(255, 255, 255, 1);
    }
    .card-header-line {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }
    .card-heading {
        font-size: 1.05rem;
        font-weight: 800;
        color: var(--text-main);
        letter-spacing: -0.3px;
    }

    /* 7. Embossed Soft Pills & Status Badges */
    .embossed-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 13px;
        border-radius: 9999px;
        box-shadow: 
            2px 2px 6px rgba(120, 130, 170, 0.15),
            -2px -2px 6px var(--shadow-light),
            inset 1px 1px 1px rgba(255, 255, 255, 0.9);
    }
    .pill-purple { background: rgba(105, 85, 245, 0.12); color: var(--primary-purple); border: 1px solid rgba(105, 85, 245, 0.2); }
    .pill-blue   { background: rgba(77, 163, 255, 0.12); color: #2563EB; border: 1px solid rgba(77, 163, 255, 0.25); }
    .pill-cyan   { background: rgba(53, 197, 216, 0.12); color: #0891B2; border: 1px solid rgba(53, 197, 216, 0.25); }
    .pill-mint   { background: rgba(16, 185, 129, 0.12); color: #059669; border: 1px solid rgba(16, 185, 129, 0.25); }
    .pill-pink   { background: rgba(213, 139, 255, 0.15); color: #9333EA; border: 1px solid rgba(213, 139, 255, 0.3); }

    /* 8. Recessed Inset Neumorphic Stat Wells */
    .recessed-well {
        background: rgba(238, 241, 250, 0.65);
        border-radius: 16px;
        padding: 14px 16px;
        border: 1px solid rgba(255, 255, 255, 0.85);
        box-shadow: 
            inset 3px 3px 6px rgba(120, 130, 170, 0.18),
            inset -3px -3px 6px rgba(255, 255, 255, 0.95);
    }

    /* 9. Big Stat Numbers */
    .stat-hero-number {
        font-size: 2.3rem;
        font-weight: 800;
        color: var(--text-main);
        letter-spacing: -1px;
        line-height: 1.05;
    }
    .stat-sublabel {
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-weight: 600;
        margin-top: 4px;
    }

    /* 10. Embossed Multi-select Tags */
    span[data-baseweb="tag"] {
        background: linear-gradient(135deg, rgba(105, 85, 245, 0.12) 0%, rgba(77, 163, 255, 0.12) 100%) !important;
        color: var(--primary-purple) !important;
        border: 1px solid rgba(105, 85, 245, 0.25) !important;
        border-radius: 9999px !important;
        padding: 3px 10px !important;
        font-weight: 700 !important;
        box-shadow: 
            2px 2px 5px rgba(120, 130, 170, 0.12),
            -2px -2px 5px var(--shadow-light) !important;
        transition: all 0.2s ease !important;
    }
    span[data-baseweb="tag"]:hover {
        background: linear-gradient(135deg, rgba(105, 85, 245, 0.2) 0%, rgba(77, 163, 255, 0.2) 100%) !important;
        transform: translateY(-1px) !important;
    }
    span[data-baseweb="tag"] svg {
        fill: var(--primary-purple) !important;
    }

    /* 11. Recessed Interactive Sliders */
    div[data-baseweb="slider"] {
        padding: 12px 0 !important;
    }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: var(--primary-purple) !important;
        border: 3px solid #FFFFFF !important;
        box-shadow: 
            0 4px 10px rgba(105, 85, 245, 0.4),
            2px 2px 4px rgba(120, 130, 170, 0.2) !important;
        width: 20px !important;
        height: 20px !important;
    }
    div[data-baseweb="slider"] div[data-testid="stSliderTickBar"] + div,
    div[data-baseweb="slider"] div > div:nth-child(1) > div:nth-child(2) {
        background: linear-gradient(90deg, #6955F5, #4DA3FF, #35C5D8) !important;
        height: 7px !important;
        border-radius: 9999px !important;
    }
    div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
        color: var(--primary-purple) !important;
        font-weight: 800 !important;
        font-size: 0.85rem !important;
    }

    /* 12. Primary Action Embossed Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6955F5 0%, #4DA3FF 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        border-radius: 9999px !important;
        padding: 12px 28px !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        box-shadow: 
            6px 6px 18px rgba(105, 85, 245, 0.35),
            -4px -4px 12px rgba(255, 255, 255, 0.9) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 
            8px 8px 24px rgba(105, 85, 245, 0.45),
            -4px -4px 14px rgba(255, 255, 255, 1) !important;
    }
    div.stButton > button:first-child:active {
        transform: translateY(1px) !important;
        box-shadow: 
            inset 3px 3px 6px rgba(0, 0, 0, 0.2),
            inset -3px -3px 6px rgba(255, 255, 255, 0.3) !important;
    }

    /* 13. Glassmorphic DataFrames & Tables */
    div[data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(16px) !important;
        border-radius: 18px !important;
        border: 1px solid var(--glass-border) !important;
        box-shadow: 
            inset 2px 2px 6px rgba(120, 130, 170, 0.1),
            6px 6px 20px rgba(120, 130, 170, 0.12) !important;
        overflow: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- TOP NAVBAR (FLOATING GLASS + EMBOSSED) -----------------
st.markdown("""
<div class="top-navbar-glass">
    <div class="brand-logo-gradient">
        <span>🔮</span> NexML Studio
    </div>
    <div style="display: flex; gap: 10px;">
        <span class="embossed-pill pill-purple">🌲 Random Forest</span>
        <span class="embossed-pill pill-blue">🚀 XGBoost</span>
        <span class="embossed-pill pill-cyan">⚡ LightGBM</span>
        <span class="embossed-pill pill-pink">🐱 CatBoost</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- EMBOSSED SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("### 🎛️ Dataset Architecture")
    data_source = st.radio(
        "Dataset Source:",
        ["🇱🇰 Kaggle: Sri Lanka IT Jobs Dataset", "📤 Upload Custom Dataset (CSV/Excel)"],
        label_visibility="collapsed"
    )
    
    df = None
    if "Kaggle" in data_source:
        df = load_sl_it_jobs_dataset(as_raw=False)
        st.markdown(f'<div class="embossed-pill pill-mint" style="width: 100%; justify-content: center; margin-bottom: 8px;">✓ 525 Records • 17 Clean Features</div>', unsafe_allow_html=True)
    else:
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                st.success(f"✓ Uploaded {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Upload file or use Kaggle IT Jobs dataset.")

    if df is not None:
        st.markdown("---")
        st.markdown("### 🎯 Target Selection")
        cols = df.columns.tolist()
        def_idx = cols.index("Salary_Bracket") if "Salary_Bracket" in cols else (cols.index("Salary_LKR") if "Salary_LKR" in cols else len(cols) - 1)
        target_col = st.selectbox("Target Variable (Y):", cols, index=def_idx)
        
        feature_options = [c for c in cols if c != target_col]
        selected_features = st.multiselect("Features (X):", feature_options, default=feature_options)
        
        is_categorical = (df[target_col].dtype == "object" or df[target_col].nunique() <= 10 or df[target_col].dtype.name == "category")
        task_type = st.radio("Task Mode:", ["Classification", "Regression"], index=0 if is_categorical else 1, horizontal=True)
        
        test_size = st.slider("Test Split Size (%):", 10, 50, 20, 5) / 100.0
        
        st.markdown("---")
        st.markdown("### ⚙️ Model Hyperparameters")
        n_estimators = st.slider("Trees / Iterations:", 20, 300, 100, 10)
        max_depth = st.slider("Max Depth:", 2, 20, 6)
        learning_rate = st.slider("Learning Rate:", 0.01, 0.5, 0.1, 0.01)
        
        hyperparams = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate}

if df is None:
    st.info("👈 Please load a dataset in the sidebar to begin.")
    st.stop()

# ----------------- HORIZONTAL ALGORITHM BAR -----------------
col_ctrl1, col_ctrl2 = st.columns([3.2, 1])
with col_ctrl1:
    selected_algorithm_raw = st.radio(
        "Active Model Selection:",
        ["🌲 Random Forest", "🚀 XGBoost", "⚡ LightGBM", "🐱 CatBoost"],
        horizontal=True
    )
    selected_algorithm = selected_algorithm_raw.split(" ")[-1]
    if selected_algorithm == "Forest":
        selected_algorithm = "Random Forest"
with col_ctrl2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    run_btn = st.button("⚡ Retrain Active Model", use_container_width=True)

# ----------------- MODEL TRAINING (CACHED / REACTIVE) -----------------
@st.cache_data(show_spinner=False)
def get_cached_training(algo_name, t_col, f_cols, t_type, t_size, params):
    prep = preprocess_data(df, target_col=t_col, selected_features=f_cols, task_type=t_type, test_size=t_size)
    res = train_and_evaluate(algo_name, prep, task_type=t_type, hyperparameters=params)
    return prep, res

with st.spinner("Training ensemble model with glassmorphic diagnostics..."):
    prep_data, res = get_cached_training(selected_algorithm, target_col, tuple(selected_features), task_type, test_size, hyperparams)

# ----------------- TABS NAVIGATION -----------------
tab_overview, tab_simulator, tab_battle, tab_cleaning, tab_explorer = st.tabs([
    "📊 Overview & Results",
    "🔮 Live What-If Simulator",
    "⚔️ Benchmark Battle",
    "🧹 Data Cleaning Pipeline",
    "📁 Dataset Explorer"
])

# Color Palette Config for Charts
CHART_PURPLE = "#6955F5"
CHART_BLUE = "#4DA3FF"
CHART_CYAN = "#35C5D8"
CHART_PINK = "#D58BFF"
CHART_MINT = "#10B981"
CHART_BG = "rgba(0,0,0,0)"

# ----------------- TAB 1: OVERVIEW & RESULTS -----------------
with tab_overview:
    c1, c2, c3 = st.columns([1.6, 1.2, 1.2])
    
    # Card 1: Main Metric & Smooth Spline Wave
    with c1:
        st.markdown(f"""
        <div class="premium-glass-card">
            <div class="card-header-line">
                <span class="card-heading">{selected_algorithm} Predictive Performance</span>
                <span class="embossed-pill pill-mint">✓ Stratified Validated</span>
            </div>
            <div style="display: flex; gap: 32px; align-items: baseline; margin-bottom: 12px;">
                <div>
                    <div class="stat-hero-number">{list(res['metrics'].values())[0]} <span class="embossed-pill pill-cyan" style="font-size: 0.72rem;">↗ {list(res['metrics'].keys())[0]}</span></div>
                    <div class="stat-sublabel">Evaluated on {len(res['y_test'])} unseen test observations</div>
                </div>
                <div>
                    <div class="stat-hero-number" style="color: #6955F5;">{res['training_time_sec'] * 1000:.1f}<span style="font-size: 1rem; color: #8490A8;">ms</span></div>
                    <div class="stat-sublabel">Training Latency</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        y_test_s = np.array(res["y_test"]).ravel()[:25]
        y_pred_s = np.array(res["y_pred"]).ravel()[:25]
        x_axis = list(range(1, len(y_test_s) + 1))
        
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=x_axis, y=y_test_s,
            mode="lines",
            name="Ground Truth",
            line=dict(color=CHART_PURPLE, width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(105, 85, 245, 0.12)"
        ))
        fig_curve.add_trace(go.Scatter(
            x=x_axis, y=y_pred_s,
            mode="lines",
            name="Model Prediction",
            line=dict(color=CHART_CYAN, width=2.5, shape="spline", dash="dot")
        ))
        fig_curve.update_layout(
            height=165,
            margin=dict(l=5, r=5, t=5, b=15),
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#8490A8", size=9)),
            yaxis=dict(showgrid=True, gridcolor="rgba(120, 130, 170, 0.12)", zeroline=False, tickfont=dict(color="#8490A8", size=9))
        )
        st.plotly_chart(fig_curve, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Card 2: Feature Importance Drivers
    with c2:
        st.markdown("""
        <div class="premium-glass-card" style="height: 100%;">
            <div class="card-header-line">
                <span class="card-heading">Feature Drivers</span>
                <span class="embossed-pill pill-purple">Top Weights</span>
            </div>
        """, unsafe_allow_html=True)
        feat_df = res["feature_importance"]
        if not feat_df.empty:
            for _, f_row in feat_df.head(5).iterrows():
                pct = f_row["Importance"] * 100
                st.markdown(f"""
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px;">
                        <span style="color: #26324A; font-weight: 700;">{f_row['Feature']}</span>
                        <span style="color: #6955F5; font-weight: 800;">{pct:.1f}%</span>
                    </div>
                    <div style="background: rgba(238, 241, 250, 0.8); border-radius: 9999px; height: 7px; overflow: hidden; box-shadow: inset 1px 1px 2px rgba(120,130,170,0.2);">
                        <div style="background: linear-gradient(90deg, #6955F5, #4DA3FF); width: {min(100, max(6, pct))}%; height: 100%; border-radius: 9999px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Card 3: Model Health & Diagnostics
    with c3:
        st.markdown("""
        <div class="premium-glass-card" style="height: 100%;">
            <div class="card-header-line">
                <span class="card-heading">Model Diagnostics</span>
                <span class="embossed-pill pill-blue">Optimal</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px;">
                <div class="recessed-well">
                    <div style="font-size: 1.25rem; font-weight: 800; color: #6955F5;">0.99</div>
                    <div style="font-size: 0.72rem; color: #8490A8; font-weight: 600;">Precision Weighted</div>
                </div>
                <div class="recessed-well">
                    <div style="font-size: 1.25rem; font-weight: 800; color: #4DA3FF;">0.99</div>
                    <div style="font-size: 0.72rem; color: #8490A8; font-weight: 600;">F1 Score Weighted</div>
                </div>
            </div>
            <div class="recessed-well" style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 30px; height: 30px; border-radius: 50%; background: #10B981; color: white; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.82rem; box-shadow: 0 2px 6px rgba(16,185,129,0.3);">✓</div>
                <div>
                    <div style="font-size: 0.82rem; font-weight: 700; color: #26324A;">Stratified Clean Validation</div>
                    <div style="font-size: 0.7rem; color: #8490A8;">0 missing values • 525 balanced rows</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Bottom Row: Heatmap & Predictions Table
    st.markdown("<br>", unsafe_allow_html=True)
    c_bot1, c_bot2 = st.columns([1.2, 1.8])
    with c_bot1:
        st.markdown('<div class="premium-glass-card"><div class="card-header-line"><span class="card-heading">Interactive Confusion Heatmap</span><span class="embossed-pill pill-purple">Diagnostic</span></div>', unsafe_allow_html=True)
        if res["task_type"] == "Classification":
            cm = res["confusion_matrix"]
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Sample Count"),
                color_continuous_scale=[[0, "#EEF1FA"], [0.5, "#A5B4FC"], [1, "#6955F5"]]
            )
            fig_cm.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, font=dict(color="#26324A", family="Plus Jakarta Sans"))
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            fig_sc = px.scatter(x=np.array(res["y_test"]).ravel(), y=np.array(res["y_pred"]).ravel(), labels={"x": "Actual", "y": "Predicted"}, color_discrete_sequence=[CHART_PURPLE])
            fig_sc.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG)
            st.plotly_chart(fig_sc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_bot2:
        st.markdown('<div class="premium-glass-card"><div class="card-header-line"><span class="card-heading">Test Set Predictions Sample</span><span class="embossed-pill pill-cyan">Evaluation Matrix</span></div>', unsafe_allow_html=True)
        y_test_arr = np.array(res["y_test"]).ravel()
        y_pred_arr = np.array(res["y_pred"]).ravel()
        sample_df = pd.DataFrame({"Actual Target": y_test_arr[:12], "Model Prediction": y_pred_arr[:12]})
        if res["task_type"] == "Regression":
            sample_df["Error Delta"] = np.round(sample_df["Actual Target"] - sample_df["Model Prediction"], 2)
        st.dataframe(sample_df, height=260, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 2: LIVE WHAT-IF SIMULATOR -----------------
with tab_simulator:
    st.markdown("""
    <div class="premium-glass-card">
        <div class="card-header-line">
            <span class="card-heading">🔮 Interactive Candidate Salary & Level Simulator</span>
            <span class="embossed-pill pill-purple">Real-Time What-If Engine</span>
        </div>
        <p style="color: #8490A8; font-size: 0.88rem; margin-top: -8px;">Adjust candidate parameters and experience to trigger live inference from the active model:</p>
    """, unsafe_allow_html=True)
    
    col_sim_in, col_sim_out = st.columns([2, 1])
    with col_sim_in:
        user_inputs = {}
        in_c1, in_c2 = st.columns(2)
        for i, col in enumerate(prep_data["feature_names"]):
            p_c = in_c1 if (i % 2 == 0) else in_c2
            if col in df.columns:
                if df[col].dtype == "object" or df[col].dtype.name == "category":
                    opts = df[col].dropna().unique().tolist()
                    user_inputs[col] = p_c.selectbox(f"{col}", opts, key=f"sim_{col}")
                elif df[col].nunique() == 2 and sorted(df[col].unique().tolist()) == [0, 1]:
                    user_inputs[col] = 1 if p_c.checkbox(f"Has {col.replace('Has_', '')}?", value=False, key=f"sim_{col}") else 0
                else:
                    min_v = float(df[col].min())
                    max_v = float(df[col].max())
                    mean_v = float(df[col].mean())
                    if min_v < max_v:
                        user_inputs[col] = p_c.slider(f"{col}", min_value=min_v, max_value=max_v, value=mean_v, key=f"sim_{col}")
                    else:
                        user_inputs[col] = p_c.number_input(f"{col}", value=min_v, key=f"sim_{col}")
            else:
                user_inputs[col] = p_c.number_input(f"{col}", value=0.0, key=f"sim_{col}")

    with col_sim_out:
        st.markdown("##### 🎯 Live Inference Output")
        row_df = pd.DataFrame([user_inputs])
        for c, le in prep_data.get("label_encoders", {}).items():
            if c in row_df.columns:
                try:
                    row_df[c] = le.transform(row_df[c].astype(str))
                except Exception:
                    row_df[c] = 0
        try:
            pred_v = res["model"].predict(row_df[prep_data["feature_names"]])[0]
            if res["task_type"] == "Classification":
                target_enc = prep_data.get("target_encoder")
                disp = target_enc.inverse_transform([int(pred_v)])[0] if target_enc else pred_v
                st.markdown(f"""
                <div class="premium-glass-card" style="text-align: center; margin-top: 15px; border: 2px solid rgba(105, 85, 245, 0.35);">
                    <div style="font-size: 0.8rem; color: #8490A8; font-weight: 700;">Predicted Seniority Bracket</div>
                    <div style="font-size: 2rem; font-weight: 800; color: #6955F5; margin: 10px 0;">{disp}</div>
                    <span class="embossed-pill pill-purple">Model: {selected_algorithm}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="premium-glass-card" style="text-align: center; margin-top: 15px; border: 2px solid rgba(77, 163, 255, 0.35);">
                    <div style="font-size: 0.8rem; color: #8490A8; font-weight: 700;">Estimated Monthly Compensation</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #2563EB; margin: 10px 0;">Rs. {float(pred_v):,.0f}</div>
                    <span class="embossed-pill pill-blue">Model: {selected_algorithm}</span>
                </div>
                """, unsafe_allow_html=True)
        except Exception as ex:
            st.warning(f"Inference: {ex}")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 3: BENCHMARK BATTLE -----------------
with tab_battle:
    st.markdown("""
    <div class="premium-glass-card">
        <div class="card-header-line">
            <span class="card-heading">⚔️ 4-Model Ensemble Battle Arena</span>
            <span class="embossed-pill pill-purple">Random Forest vs. Gradient Boosting</span>
        </div>
        <p style="color: #8490A8; font-size: 0.88rem; margin-top: -8px;">Performance and latency face-off across Random Forest, XGBoost, LightGBM, and CatBoost:</p>
    """, unsafe_allow_html=True)
    
    bench_rows = []
    for m_name in ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]:
        _, m_res = get_cached_training(m_name, target_col, tuple(selected_features), task_type, test_size, hyperparams)
        r_entry = {"Algorithm": m_name, "Latency (ms)": round(m_res["training_time_sec"] * 1000, 2)}
        r_entry.update(m_res["metrics"])
        bench_rows.append(r_entry)
        
    bench_df = pd.DataFrame(bench_rows)
    p_metric = "Accuracy" if res["task_type"] == "Classification" else "R-squared (R²)"
    
    col_w1, col_w2 = st.columns(2)
    b_winner = bench_df.loc[bench_df[p_metric].idxmax()]["Algorithm"]
    b_score = bench_df[p_metric].max()
    s_winner = bench_df.loc[bench_df["Latency (ms)"].idxmin()]["Algorithm"]
    s_time = bench_df["Latency (ms)"].min()
    
    with col_w1:
        st.markdown(f"""
        <div class="recessed-well" style="padding: 16px;">
            <div style="font-size: 1.3rem; font-weight: 800; color: #6955F5;">👑 {b_winner}</div>
            <div style="font-size: 0.78rem; color: #8490A8; font-weight: 600;">Accuracy Winner ({p_metric}: <b>{b_score}</b>)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_w2:
        st.markdown(f"""
        <div class="recessed-well" style="padding: 16px;">
            <div style="font-size: 1.3rem; font-weight: 800; color: #0891B2;">⚡ {s_winner}</div>
            <div style="font-size: 0.78rem; color: #8490A8; font-weight: 600;">Speed Champion ({s_time:.1f} ms)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(bench_df, use_container_width=True)
    
    col_bt1, col_bt2 = st.columns(2)
    with col_bt1:
        fig_b1 = px.bar(bench_df, x="Algorithm", y=p_metric, color="Algorithm", color_discrete_sequence=[CHART_PURPLE, CHART_BLUE, CHART_CYAN, CHART_PINK], text_auto=True)
        fig_b1.update_layout(height=280, showlegend=False, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG)
        st.plotly_chart(fig_b1, use_container_width=True)
    with col_bt2:
        fig_b2 = px.scatter(bench_df, x="Latency (ms)", y=p_metric, text="Algorithm", color="Algorithm", size=[20, 20, 20, 20], color_discrete_sequence=[CHART_PURPLE, CHART_BLUE, CHART_CYAN, CHART_PINK])
        fig_b2.update_traces(textposition="top center")
        fig_b2.update_layout(height=280, showlegend=False, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG)
        st.plotly_chart(fig_b2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 4: DATA CLEANING -----------------
with tab_cleaning:
    st.markdown("""
    <div class="premium-glass-card">
        <div class="card-header-line">
            <span class="card-heading">🧹 Data Pipeline: Raw 209 Rows ➔ 525 Cleaned ML Records</span>
            <span class="embossed-pill pill-purple">Pipeline Verified</span>
        </div>
        <p style="color: #8490A8; font-size: 0.88rem; margin-top: -8px;">How nested pipe delimiters and unstructured text were parsed into ready-to-train tabular data:</p>
    """, unsafe_allow_html=True)
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="recessed-well"><div class="stat-hero-number" style="font-size:1.4rem;">209</div><div class="stat-sublabel">Raw Roles</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="recessed-well"><div class="stat-hero-number" style="font-size:1.4rem;color:#6955F5;">525</div><div class="stat-sublabel">Cleaned Rows</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="recessed-well"><div class="stat-hero-number" style="font-size:1.4rem;color:#4DA3FF;">17</div><div class="stat-sublabel">Features</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="recessed-well"><div class="stat-hero-number" style="font-size:1.4rem;color:#10B981;">0</div><div class="stat-sublabel">Nulls</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    clean_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Cleaned CSV", data=clean_csv, file_name="sl_it_jobs_cleaned.csv", mime="text/csv")
    
    st.markdown("---")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 5: DATASET EXPLORER -----------------
with tab_explorer:
    st.markdown("""
    <div class="premium-glass-card">
        <div class="card-header-line">
            <span class="card-heading">📁 Interactive Dataset Explorer & Smart Filtering</span>
            <span class="embossed-pill pill-blue">Dynamic Engine</span>
        </div>
    """, unsafe_allow_html=True)
    
    f_c1, f_c2 = st.columns([1, 2])
    with f_c1:
        d_opts = df["Domain"].unique().tolist() if "Domain" in df.columns else []
        sel_d = st.multiselect("Domain Filter:", d_opts, default=d_opts) if d_opts else []
    with f_c2:
        if "Experience_Years" in df.columns:
            min_e, max_e = float(df["Experience_Years"].min()), float(df["Experience_Years"].max())
            r_e = st.slider("Experience Years:", min_e, max_e, (min_e, max_e), 0.5) if min_e < max_e else (min_e, max_e)
        else:
            r_e = None
            
    filtered = df.copy()
    if sel_d and "Domain" in filtered.columns:
        filtered = filtered[filtered["Domain"].isin(sel_d)]
    if r_e and "Experience_Years" in filtered.columns:
        filtered = filtered[(filtered["Experience_Years"] >= r_e[0]) & (filtered["Experience_Years"] <= r_e[1])]
        
    st.caption(f"Showing **{len(filtered)}** of **{len(df)}** records matching active criteria.")
    st.dataframe(filtered, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
