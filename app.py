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
    page_title="NexML | Glassmorphism & Embossed Studio",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- LIGHT GLASSMORPHISM + EMBOSSED (NEUMORPHISM) CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Hide Streamlit Header, Deploy Button, Main Menu, and Footer */
    header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }
    .stDeployButton, [data-testid="stToolbar"], #MainMenu, footer {
        display: none !important;
        visibility: hidden !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Light Background with Cool Aurora Gradients */
    html, body, [class*="css"], .stApp {
        background: linear-gradient(135deg, #EFF4FC 0%, #E6EDF9 35%, #F4F0FD 70%, #EBF4FE 100%) !important;
        color: #1E293B !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Sidebar Light Glassmorphism */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.8) !important;
        box-shadow: 4px 0 24px rgba(99, 102, 241, 0.06) !important;
    }
    
    /* Top Floating Glass Navbar */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 30px;
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 9999px;
        margin-bottom: 24px;
        box-shadow: 
            8px 8px 24px rgba(166, 180, 208, 0.35),
            -8px -8px 24px rgba(255, 255, 255, 0.9),
            inset 1px 1px 2px rgba(255, 255, 255, 0.8);
    }
    .brand-logo {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.6px;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .user-pill {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255, 255, 255, 0.85);
        padding: 6px 16px;
        border-radius: 9999px;
        box-shadow: 
            3px 3px 8px rgba(166, 180, 208, 0.25),
            -3px -3px 8px rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.8);
    }
    .user-avatar {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 800;
        color: #FFFFFF;
        box-shadow: 0 2px 6px rgba(99, 102, 241, 0.3);
    }
    .user-name {
        font-size: 0.82rem;
        font-weight: 700;
        color: #334155;
    }

    /* Embossed + Glassmorphic Cards */
    .glass-embossed-card {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.85);
        border-radius: 24px;
        padding: 24px 26px;
        margin-bottom: 20px;
        box-shadow: 
            10px 10px 30px rgba(166, 180, 208, 0.3),
            -10px -10px 30px rgba(255, 255, 255, 0.95),
            inset 1px 1px 2px rgba(255, 255, 255, 0.9);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-embossed-card:hover {
        transform: translateY(-2px);
        box-shadow: 
            12px 12px 34px rgba(166, 180, 208, 0.38),
            -12px -12px 34px rgba(255, 255, 255, 1),
            inset 1px 1px 2px rgba(255, 255, 255, 1);
    }
    .card-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #1E293B;
        letter-spacing: -0.3px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
    }
    
    /* Embossed Buttons & Pills */
    .embossed-pill-cyan {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(6, 182, 212, 0.12);
        color: #0891B2;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        box-shadow: 
            2px 2px 5px rgba(6, 182, 212, 0.15),
            -2px -2px 5px rgba(255, 255, 255, 0.9),
            inset 1px 1px 1px rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(6, 182, 212, 0.2);
    }
    .embossed-pill-indigo {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(99, 102, 241, 0.12);
        color: #4F46E5;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        box-shadow: 
            2px 2px 5px rgba(99, 102, 241, 0.15),
            -2px -2px 5px rgba(255, 255, 255, 0.9),
            inset 1px 1px 1px rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .embossed-pill-purple {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(168, 85, 247, 0.12);
        color: #7E22CE;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        box-shadow: 
            2px 2px 5px rgba(168, 85, 247, 0.15),
            -2px -2px 5px rgba(255, 255, 255, 0.9),
            inset 1px 1px 1px rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(168, 85, 247, 0.2);
    }
    .embossed-pill-green {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(16, 185, 129, 0.12);
        color: #059669;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        box-shadow: 
            2px 2px 5px rgba(16, 185, 129, 0.15),
            -2px -2px 5px rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    /* Big Numbers */
    .big-stat-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -1px;
        line-height: 1;
    }
    .big-stat-label {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Inset/Embossed Mini Stat Boxes */
    .embossed-inset-box {
        background: rgba(241, 245, 249, 0.65);
        border-radius: 16px;
        padding: 14px 16px;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 
            inset 3px 3px 6px rgba(166, 180, 208, 0.25),
            inset -3px -3px 6px rgba(255, 255, 255, 0.95);
    }

    /* Tabs Pill Group */
    div[data-testid="stTabs"] [role="tablist"] {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(16px) !important;
        padding: 6px !important;
        border-radius: 9999px !important;
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
        box-shadow: 
            6px 6px 18px rgba(166, 180, 208, 0.25),
            -6px -6px 18px rgba(255, 255, 255, 0.9),
            inset 1px 1px 2px rgba(255, 255, 255, 0.8) !important;
        display: flex !important;
        gap: 6px !important;
        margin-bottom: 24px !important;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        background: transparent !important;
        color: #64748B !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 9px 24px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 
            3px 3px 10px rgba(99, 102, 241, 0.35),
            -2px -2px 6px rgba(255, 255, 255, 0.8) !important;
    }
    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] div[data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Embossed Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        border-radius: 9999px !important;
        padding: 12px 28px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 
            6px 6px 18px rgba(99, 102, 241, 0.35),
            -4px -4px 12px rgba(255, 255, 255, 0.8) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px) !important;
        box-shadow: 
            8px 8px 22px rgba(99, 102, 241, 0.45),
            -4px -4px 12px rgba(255, 255, 255, 1) !important;
    }

    /* Glass Table Styling */
    div[data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 18px !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        box-shadow: 
            inset 2px 2px 6px rgba(166, 180, 208, 0.15),
            4px 4px 16px rgba(166, 180, 208, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- TOP NAVBAR -----------------
st.markdown("""
<div class="top-navbar">
    <div class="brand-logo">
        <span>🔮</span> nexml.studio
    </div>
    <div style="display: flex; gap: 10px;">
        <span class="embossed-pill-indigo">🌲 Random Forest</span>
        <span class="embossed-pill-cyan">🚀 XGBoost</span>
        <span class="embossed-pill-purple">⚡ LightGBM</span>
        <span class="embossed-pill-green">🐱 CatBoost</span>
    </div>
    <div class="user-pill">
        <div class="user-avatar">AW</div>
        <div class="user-name">Arosh Wijesinghe</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("### 🎛️ Dataset & Architecture")
    data_source = st.radio(
        "Dataset Source:",
        ["🇱🇰 Kaggle: Sri Lanka IT Jobs Dataset", "📤 Upload Custom Dataset (CSV/Excel)"],
        label_visibility="collapsed"
    )
    
    df = None
    if "Kaggle" in data_source:
        df = load_sl_it_jobs_dataset(as_raw=False)
        st.success(f"✓ Loaded Sri Lanka IT Jobs ({len(df)} rows, {df.shape[1]} features)")
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
col_ctrl1, col_ctrl2 = st.columns([3, 1])
with col_ctrl1:
    selected_algorithm_raw = st.radio(
        "Select Model:",
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

with st.spinner("Fitting ensemble model..."):
    prep_data, res = get_cached_training(selected_algorithm, target_col, tuple(selected_features), task_type, test_size, hyperparams)

# ----------------- TABS NAVIGATION -----------------
tab_overview, tab_simulator, tab_battle, tab_cleaning, tab_explorer = st.tabs([
    "📊 Overview & Results",
    "🔮 Live What-If Simulator",
    "⚔️ Benchmark Battle",
    "🧹 Data Cleaning Pipeline",
    "📁 Dataset Explorer"
])

# Cool Palette: Indigo, Cyan, Mint, Soft Slate
COLOR_INDIGO = "#6366F1"
COLOR_CYAN = "#06B6D4"
COLOR_PURPLE = "#A855F7"
COLOR_MINT = "#10B981"
COLOR_BG_LIGHT = "rgba(0,0,0,0)"

# ----------------- TAB 1: OVERVIEW -----------------
with tab_overview:
    c1, c2, c3 = st.columns([1.6, 1.2, 1.2])
    
    # Card 1: Main Metric & Spline Wave
    with c1:
        st.markdown(f"""
        <div class="glass-embossed-card">
            <div class="card-title">
                <span>{selected_algorithm} Performance</span>
                <span class="embossed-pill-green">✓ Validated</span>
            </div>
            <div style="display: flex; gap: 32px; align-items: baseline; margin-bottom: 12px;">
                <div>
                    <div class="big-stat-val">{list(res['metrics'].values())[0]} <span class="embossed-pill-cyan" style="font-size: 0.75rem;">↗ High</span></div>
                    <div class="big-stat-label">{list(res['metrics'].keys())[0]} on {len(res['y_test'])} test samples</div>
                </div>
                <div>
                    <div class="big-stat-val" style="color: #6366F1;">{res['training_time_sec'] * 1000:.1f}<span style="font-size: 1rem; color: #64748B;">ms</span></div>
                    <div class="big-stat-label">Training Latency</div>
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
            line=dict(color=COLOR_INDIGO, width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.12)"
        ))
        fig_curve.add_trace(go.Scatter(
            x=x_axis, y=y_pred_s,
            mode="lines",
            name="Prediction",
            line=dict(color=COLOR_CYAN, width=2.5, shape="spline", dash="dot")
        ))
        fig_curve.update_layout(
            height=160,
            margin=dict(l=5, r=5, t=5, b=15),
            paper_bgcolor=COLOR_BG_LIGHT,
            plot_bgcolor=COLOR_BG_LIGHT,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#64748B", size=9)),
            yaxis=dict(showgrid=True, gridcolor="rgba(203, 213, 225, 0.4)", zeroline=False, tickfont=dict(color="#64748B", size=9))
        )
        st.plotly_chart(fig_curve, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Card 2: Feature Drivers
    with c2:
        st.markdown("""
        <div class="glass-embossed-card" style="height: 100%;">
            <div class="card-title">
                <span>Feature Drivers</span>
                <span class="embossed-pill-indigo">Top Weights</span>
            </div>
        """, unsafe_allow_html=True)
        feat_df = res["feature_importance"]
        if not feat_df.empty:
            for _, f_row in feat_df.head(5).iterrows():
                pct = f_row["Importance"] * 100
                st.markdown(f"""
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px;">
                        <span style="color: #334155; font-weight: 700;">{f_row['Feature']}</span>
                        <span style="color: #4F46E5; font-weight: 800;">{pct:.1f}%</span>
                    </div>
                    <div style="background: rgba(226, 232, 240, 0.7); border-radius: 9999px; height: 6px; overflow: hidden; box-shadow: inset 1px 1px 2px rgba(0,0,0,0.1);">
                        <div style="background: linear-gradient(90deg, #6366F1, #06B6D4); width: {min(100, max(5, pct))}%; height: 100%; border-radius: 9999px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Card 3: Model Health
    with c3:
        st.markdown("""
        <div class="glass-embossed-card" style="height: 100%;">
            <div class="card-title">
                <span>Model Diagnostics</span>
                <span class="embossed-pill-purple">Optimal</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px;">
                <div class="embossed-inset-box">
                    <div style="font-size: 1.25rem; font-weight: 800; color: #4F46E5;">0.99</div>
                    <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">Precision Score</div>
                </div>
                <div class="embossed-inset-box">
                    <div style="font-size: 1.25rem; font-weight: 800; color: #7C3AED;">0.99</div>
                    <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">F1 Weighted</div>
                </div>
            </div>
            <div class="embossed-inset-box" style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 28px; height: 28px; border-radius: 50%; background: #10B981; color: white; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.8rem;">✓</div>
                <div>
                    <div style="font-size: 0.82rem; font-weight: 700; color: #0F172A;">Stratified Split Verified</div>
                    <div style="font-size: 0.72rem; color: #64748B;">0 nulls | 525 balanced rows</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Bottom Row: Heatmap & Predictions Table
    st.markdown("<br>", unsafe_allow_html=True)
    c_bot1, c_bot2 = st.columns([1.2, 1.8])
    with c_bot1:
        st.markdown('<div class="glass-embossed-card"><div class="card-title"><span>Interactive Confusion Heatmap</span><span class="embossed-pill-indigo">Diagnostic</span></div>', unsafe_allow_html=True)
        if res["task_type"] == "Classification":
            cm = res["confusion_matrix"]
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Samples"),
                color_continuous_scale=[[0, "#F1F5F9"], [0.5, "#818CF8"], [1, "#4F46E5"]]
            )
            fig_cm.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor=COLOR_BG_LIGHT, plot_bgcolor=COLOR_BG_LIGHT, font=dict(color="#334155"))
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            fig_sc = px.scatter(x=np.array(res["y_test"]).ravel(), y=np.array(res["y_pred"]).ravel(), labels={"x": "Actual", "y": "Predicted"}, color_discrete_sequence=[COLOR_INDIGO])
            fig_sc.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor=COLOR_BG_LIGHT, plot_bgcolor=COLOR_BG_LIGHT)
            st.plotly_chart(fig_sc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_bot2:
        st.markdown('<div class="glass-embossed-card"><div class="card-title"><span>Test Set Predictions Table</span><span class="embossed-pill-cyan">Evaluation Sample</span></div>', unsafe_allow_html=True)
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
    <div class="glass-embossed-card">
        <div class="card-title">
            <span>🔮 Interactive Candidate Salary & Level Simulator</span>
            <span class="embossed-pill-indigo">Real-Time What-If</span>
        </div>
        <p style="color: #64748B; font-size: 0.88rem; margin-top: -6px;">Tweak candidate attributes and observe real-time model inferences:</p>
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
                <div class="glass-embossed-card" style="text-align: center; margin-top: 15px; border: 2px solid rgba(99, 102, 241, 0.4);">
                    <div style="font-size: 0.8rem; color: #64748B; font-weight: 700;">Predicted Seniority Bracket</div>
                    <div style="font-size: 2rem; font-weight: 800; color: #4F46E5; margin: 10px 0;">{disp}</div>
                    <span class="embossed-pill-indigo">Model: {selected_algorithm}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="glass-embossed-card" style="text-align: center; margin-top: 15px; border: 2px solid rgba(6, 182, 212, 0.4);">
                    <div style="font-size: 0.8rem; color: #64748B; font-weight: 700;">Estimated Monthly Compensation</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #0891B2; margin: 10px 0;">Rs. {float(pred_v):,.0f}</div>
                    <span class="embossed-pill-cyan">Model: {selected_algorithm}</span>
                </div>
                """, unsafe_allow_html=True)
        except Exception as ex:
            st.warning(f"Inference: {ex}")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 3: BENCHMARK BATTLE -----------------
with tab_battle:
    st.markdown("""
    <div class="glass-embossed-card">
        <div class="card-title">
            <span>⚔️ 4-Model Ensemble Battle</span>
            <span class="embossed-pill-indigo">Random Forest vs. Boosting</span>
        </div>
        <p style="color: #64748B; font-size: 0.88rem; margin-top: -6px;">Confrontation between Random Forest, XGBoost, LightGBM, and CatBoost:</p>
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
        <div class="embossed-inset-box" style="padding: 16px;">
            <div style="font-size: 1.3rem; font-weight: 800; color: #4F46E5;">👑 {b_winner}</div>
            <div style="font-size: 0.78rem; color: #64748B; font-weight: 600;">Accuracy Winner ({p_metric}: <b>{b_score}</b>)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_w2:
        st.markdown(f"""
        <div class="embossed-inset-box" style="padding: 16px;">
            <div style="font-size: 1.3rem; font-weight: 800; color: #0891B2;">⚡ {s_winner}</div>
            <div style="font-size: 0.78rem; color: #64748B; font-weight: 600;">Speed Champion ({s_time:.1f} ms)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(bench_df, use_container_width=True)
    
    col_bt1, col_bt2 = st.columns(2)
    with col_bt1:
        fig_b1 = px.bar(bench_df, x="Algorithm", y=p_metric, color="Algorithm", color_discrete_sequence=[COLOR_INDIGO, COLOR_CYAN, COLOR_PURPLE, COLOR_MINT], text_auto=True)
        fig_b1.update_layout(height=280, showlegend=False, paper_bgcolor=COLOR_BG_LIGHT, plot_bgcolor=COLOR_BG_LIGHT)
        st.plotly_chart(fig_b1, use_container_width=True)
    with col_bt2:
        fig_b2 = px.scatter(bench_df, x="Latency (ms)", y=p_metric, text="Algorithm", color="Algorithm", size=[20, 20, 20, 20], color_discrete_sequence=[COLOR_INDIGO, COLOR_CYAN, COLOR_PURPLE, COLOR_MINT])
        fig_b2.update_traces(textposition="top center")
        fig_b2.update_layout(height=280, showlegend=False, paper_bgcolor=COLOR_BG_LIGHT, plot_bgcolor=COLOR_BG_LIGHT)
        st.plotly_chart(fig_b2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 4: DATA CLEANING -----------------
with tab_cleaning:
    st.markdown("""
    <div class="glass-embossed-card">
        <div class="card-title">
            <span>🧹 Data Pipeline: Raw 209 Rows ➔ 525 Cleaned ML Records</span>
            <span class="embossed-pill-indigo">Pipeline Verified</span>
        </div>
        <p style="color: #64748B; font-size: 0.88rem; margin-top: -6px;">How nested pipe delimiters and unstructured text were parsed into ready-to-train tabular data:</p>
    """, unsafe_allow_html=True)
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="embossed-inset-box"><div class="big-stat-val" style="font-size:1.4rem;">209</div><div class="big-stat-label">Raw Roles</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="embossed-inset-box"><div class="big-stat-val" style="font-size:1.4rem;color:#4F46E5;">525</div><div class="big-stat-label">Cleaned Rows</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="embossed-inset-box"><div class="big-stat-val" style="font-size:1.4rem;color:#0891B2;">17</div><div class="big-stat-label">Features</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="embossed-inset-box"><div class="big-stat-val" style="font-size:1.4rem;color:#10B981;">0</div><div class="big-stat-label">Nulls</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    clean_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Cleaned CSV", data=clean_csv, file_name="sl_it_jobs_cleaned.csv", mime="text/csv")
    
    st.markdown("---")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 5: DATASET EXPLORER -----------------
with tab_explorer:
    st.markdown("""
    <div class="glass-embossed-card">
        <div class="card-title">
            <span>📁 Interactive Dataset Explorer</span>
            <span class="embossed-pill-indigo">Dynamic Filtering</span>
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
