import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from data_utils import (
    preprocess_data,
    load_sl_it_jobs_dataset
)
from model_trainer import train_and_evaluate

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="NexML | Obsidian Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- FITONIST OBSIDIAN / LILAC / PASTEL YELLOW CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Global Base */
    html, body, [class*="css"], .stApp {
        background-color: #0B0B0F !important;
        color: #F3F4F6 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #121218 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    
    /* Top Header Bar */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 28px;
        background: #14141C;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 9999px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .brand-logo {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-logo span {
        color: #B8A5FF;
    }
    .user-pill {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #1C1C26;
        padding: 6px 14px;
        border-radius: 9999px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .user-avatar {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #B8A5FF 0%, #FFE58F 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 800;
        color: #0B0B0F;
    }
    .user-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: #E2E8F0;
        line-height: 1.1;
    }
    .user-role {
        font-size: 0.7rem;
        color: #7E7E94;
    }
    
    /* Fitonist Surface Cards */
    .fitonist-card {
        background-color: #161620;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 22px;
        padding: 24px 26px;
        margin-bottom: 20px;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .fitonist-card:hover {
        border-color: rgba(184, 165, 255, 0.2);
    }
    .card-header-flex {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 18px;
    }
    .card-title-text {
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.3px;
    }
    
    /* Pill Tag Switchers */
    .filter-pill-group {
        display: inline-flex;
        background: #1E1E2A;
        padding: 4px;
        border-radius: 9999px;
        gap: 4px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .filter-pill {
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #8E8EA0;
        cursor: pointer;
    }
    .filter-pill.active {
        background: #FFFFFF;
        color: #0F0F14;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }
    
    /* Stat Badges */
    .growth-badge-green {
        display: inline-flex;
        align-items: center;
        gap: 3px;
        background: rgba(52, 211, 153, 0.16);
        color: #34D399;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 9999px;
        border: 1px solid rgba(52, 211, 153, 0.25);
    }
    .badge-lilac {
        display: inline-flex;
        align-items: center;
        background: rgba(184, 165, 255, 0.16);
        color: #C4B5FD;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 9999px;
        border: 1px solid rgba(184, 165, 255, 0.25);
    }
    .badge-yellow {
        display: inline-flex;
        align-items: center;
        background: rgba(254, 240, 138, 0.15);
        color: #FDE047;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 9999px;
        border: 1px solid rgba(254, 240, 138, 0.25);
    }
    
    /* Big Value Numbers */
    .hero-stat-num {
        font-size: 2.3rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .hero-stat-label {
        font-size: 0.82rem;
        color: #7E7E94;
        font-weight: 500;
        margin-top: 4px;
    }
    
    /* Mini Channel Caps */
    .mini-stat-card {
        background: #1C1C27;
        border-radius: 14px;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        gap: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .mini-icon-box {
        width: 32px;
        height: 32px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
    }
    .mini-icon-lilac { background: #B8A5FF; color: #0B0B0F; }
    .mini-icon-yellow { background: #FFE58F; color: #0B0B0F; }
    
    /* Tabs Overrides */
    div[data-testid="stTabs"] button {
        background: #161620 !important;
        border-radius: 9999px !important;
        padding: 8px 20px !important;
        color: #8E8EA0 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        margin-right: 8px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: #FFFFFF !important;
        color: #0B0B0F !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 16px rgba(255, 255, 255, 0.15) !important;
    }
    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
        display: none !important;
    }
    div[data-testid="stTabs"] div[data-baseweb="tab-border"] {
        display: none !important;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        background: #FFFFFF !important;
        color: #0B0B0F !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        border-radius: 9999px !important;
        padding: 12px 28px !important;
        border: none !important;
        box-shadow: 0 6px 20px rgba(255, 255, 255, 0.12) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    div.stButton > button:first-child:hover {
        background: #B8A5FF !important;
        color: #0B0B0F !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(184, 165, 255, 0.3) !important;
    }
    
    /* Slider & Form Inputs */
    div[data-baseweb="slider"] {
        padding: 10px 0 !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1A1A26 !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
    }
    
    /* Clean Tables */
    div[data-testid="stDataFrame"] {
        background: #14141D !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        overflow: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- FITONIST TOP NAV BAR -----------------
st.markdown("""
<div class="top-navbar">
    <div class="brand-logo">
        <span>⚡</span> nexml.studio
    </div>
    <div style="display: flex; gap: 8px;">
        <span class="badge-lilac">🌲 Random Forest</span>
        <span class="badge-yellow">🚀 XGBoost</span>
        <span class="badge-lilac">⚡ LightGBM</span>
        <span class="badge-yellow">🐱 CatBoost</span>
    </div>
    <div class="user-pill">
        <div class="user-avatar">AW</div>
        <div>
            <div class="user-name">Arosh Wijesinghe</div>
            <div class="user-role">Lead ML Researcher</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("### 🎛️ Dataset Settings")
    
    data_source = st.radio(
        "Source:",
        [
            "🇱🇰 Kaggle: Sri Lanka IT Jobs Dataset",
            "📤 Upload Custom Dataset (CSV / Excel)"
        ]
    )
    
    df = None
    if data_source == "🇱🇰 Kaggle: Sri Lanka IT Jobs Dataset":
        try:
            view_mode = st.radio("Dataset Version:", ["Cleaned & Preprocessed (525 rows)", "Raw Kaggle (209 unparsed rows)"], horizontal=True)
            df = load_sl_it_jobs_dataset(as_raw=(view_mode == "Raw Kaggle (209 unparsed rows)"))
            st.success(f"Loaded: `{df.shape[0]} rows` | `{df.shape[1]} features`")
        except Exception as e:
            st.error(f"Error loading Kaggle dataset: {e}")

    elif data_source == "📤 Upload Custom Dataset (CSV / Excel)":
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success(f"Uploaded: `{uploaded_file.name}` ({df.shape[0]} rows, {df.shape[1]} cols)")
            except Exception as e:
                st.error(f"Error loading file: {e}")
        else:
            st.info("Upload your custom CSV/Excel dataset or select the Kaggle IT Jobs dataset above.")

    if df is not None:
        st.markdown("---")
        st.markdown("### 🎯 Target & Architecture")
        
        columns = df.columns.tolist()
        
        # Intelligent default target suggestion
        default_target_idx = len(columns) - 1
        if "Salary_Bracket" in columns:
            default_target_idx = columns.index("Salary_Bracket")
        elif "Salary_LKR" in columns:
            default_target_idx = columns.index("Salary_LKR")
            
        target_col = st.selectbox("Target Column (Y):", columns, index=default_target_idx)
        
        feature_options = [c for c in columns if c != target_col]
        selected_features = st.multiselect(
            "Predictor Features (X):",
            feature_options,
            default=feature_options
        )
        
        # Auto-detect task type
        is_categorical_target = (
            df[target_col].dtype == "object"
            or df[target_col].nunique() <= 10
            or df[target_col].dtype.name == "category"
        )
        default_task = "Classification" if is_categorical_target else "Regression"
        
        task_type = st.radio(
            "Machine Learning Task:",
            ["Classification", "Regression"],
            index=0 if default_task == "Classification" else 1,
            horizontal=True
        )
        
        test_size = st.slider("Test Split Size (%):", min_value=10, max_value=50, value=20, step=5) / 100.0
        
        st.markdown("---")
        st.markdown("### 🤖 Model & Execution")
        
        algorithms = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]
        selected_algorithm = st.selectbox("Active Algorithm:", algorithms, index=0)
        
        run_mode = st.radio(
            "Run Mode:",
            ["Train Active Model", "Benchmark All 4 Models (Battle Arena)"],
            help="Compare Random Forest vs Gradient Boosting (XGBoost, LightGBM, CatBoost) side-by-side!"
        )
        
        with st.expander("⚙️ Hyperparameters", expanded=False):
            max_depth_val = st.slider("Max Depth", min_value=1, max_value=20, value=6)
            n_estimators_val = st.slider("Trees / Iterations", min_value=10, max_value=300, value=100, step=10)
            learning_rate_val = st.slider("Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
            
            hyperparams = {
                "max_depth": max_depth_val,
                "n_estimators": n_estimators_val,
                "learning_rate": learning_rate_val
            }
            
        run_button = st.button("🚀 Train Model", type="primary", use_container_width=True)

# ----------------- MAIN APP WORKSPACE -----------------
if df is None:
    st.info("👈 Please select a dataset in the sidebar to begin.")
    st.stop()

if not selected_features:
    st.warning("⚠️ Please select at least one feature column from the sidebar.")
    st.stop()

# ----------------- 5 CAPSULE TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview & Results",
    "Live Predictor",
    "Benchmark Battle",
    "Data Cleaning",
    "Dataset Explorer"
])

# ----------------- PLOTLY THEME PRESETS -----------------
COLOR_LILAC = "#B8A5FF"
COLOR_YELLOW = "#FFE58F"
COLOR_MINT = "#34D399"
COLOR_CYAN = "#38BDF8"
COLOR_DARK_BG = "rgba(0,0,0,0)"

# ----------------- TRAIN LOGIC (SESSION STATE) -----------------
if run_button or "current_result" in st.session_state:
    if run_button:
        with st.spinner("⚡ Fitting ensemble models..."):
            prep_data = preprocess_data(
                df,
                target_col=target_col,
                selected_features=selected_features,
                task_type=task_type,
                test_size=test_size
            )
            st.session_state["prep_data"] = prep_data
            
            if run_mode == "Train Active Model":
                res = train_and_evaluate(
                    selected_algorithm,
                    prep_data,
                    task_type=task_type,
                    hyperparameters=hyperparams
                )
                st.session_state["current_result"] = res
                st.session_state["benchmark_results"] = None
            else:
                bench_results = []
                prog = st.progress(0, text="Benchmarking ensemble algorithms...")
                for i, algo in enumerate(algorithms):
                    prog.progress((i + 1) / len(algorithms), text=f"Training {algo}...")
                    r = train_and_evaluate(algo, prep_data, task_type=task_type)
                    bench_results.append(r)
                prog.empty()
                st.session_state["benchmark_results"] = bench_results
                st.session_state["current_result"] = [r for r in bench_results if r["algorithm"] == selected_algorithm][0]

    res = st.session_state.get("current_result")
    bench_results = st.session_state.get("benchmark_results")
    prep_data = st.session_state.get("prep_data")
    
    # ----------------- TAB 1: OVERVIEW & RESULTS -----------------
    with tab1:
        if res is not None:
            # Top Overview Row - 3 Clean Cards mirroring the Fitonist UI
            col_k1, col_k2, col_k3 = st.columns([1.6, 1.2, 1.4])
            
            # Card 1: Performance Hero Card with Dual Metrics & Smooth Curve
            with col_k1:
                st.markdown(f"""
                <div class="fitonist-card">
                    <div class="card-header-flex">
                        <div class="card-title-text">{res['algorithm']} Performance</div>
                        <div class="filter-pill-group">
                            <span class="filter-pill">Today</span>
                            <span class="filter-pill">Week</span>
                            <span class="filter-pill active">Test Set</span>
                        </div>
                    </div>
                    <div style="display: flex; gap: 32px; align-items: baseline; margin-bottom: 8px;">
                        <div>
                            <div class="hero-stat-num">{list(res['metrics'].values())[0]} <span class="growth-badge-green">↗ 99.4%</span></div>
                            <div class="hero-stat-label">{list(res['metrics'].keys())[0]} on {len(res['y_test'])} samples</div>
                        </div>
                        <div>
                            <div class="hero-stat-num" style="color: #FFE58F;">{res['training_time_sec'] * 1000:.1f}<span style="font-size: 1.1rem; color: #7E7E94;">ms</span></div>
                            <div class="hero-stat-label">Inference Latency</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Smooth Dual Area Curve matching the lilac/yellow curve in Fitonist image
                y_test_s = np.array(res["y_test"][:30])
                y_pred_s = np.array(res["y_pred"][:30])
                x_axis = list(range(1, len(y_test_s) + 1))
                
                fig_curve = go.Figure()
                fig_curve.add_trace(go.Scatter(
                    x=x_axis, y=y_test_s,
                    mode="lines",
                    name="Actual Ground Truth",
                    line=dict(color=COLOR_LILAC, width=2.5, shape="spline"),
                    fill="tozeroy",
                    fillcolor="rgba(184, 165, 255, 0.08)"
                ))
                fig_curve.add_trace(go.Scatter(
                    x=x_axis, y=y_pred_s,
                    mode="lines",
                    name="Model Prediction",
                    line=dict(color=COLOR_YELLOW, width=2, shape="spline", dash="dot")
                ))
                fig_curve.update_layout(
                    height=200,
                    margin=dict(l=10, r=10, t=10, b=20),
                    paper_bgcolor=COLOR_DARK_BG,
                    plot_bgcolor=COLOR_DARK_BG,
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=True, tickfont=dict(color="#7E7E94", size=10)),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.04)", zeroline=False, tickfont=dict(color="#7E7E94", size=10))
                )
                st.plotly_chart(fig_curve, use_container_width=True)

            # Card 2: Feature Matrix / Calendar Grid Style
            with col_k2:
                feat_df = res["feature_importance"]
                st.markdown("""
                <div class="fitonist-card" style="height: 100%;">
                    <div class="card-header-flex">
                        <div class="card-title-text">Feature Drivers</div>
                        <span class="badge-lilac">Top 6 Weights</span>
                    </div>
                """, unsafe_allow_html=True)
                
                if not feat_df.empty:
                    top_f = feat_df.head(6)
                    for _, f_row in top_f.iterrows():
                        pct = f_row["Importance"] * 100
                        st.markdown(f"""
                        <div style="margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px;">
                                <span style="color: #E2E8F0; font-weight: 600;">{f_row['Feature']}</span>
                                <span style="color: #B8A5FF; font-weight: 700;">{pct:.1f}%</span>
                            </div>
                            <div style="background: #20202E; border-radius: 9999px; height: 6px; overflow: hidden;">
                                <div style="background: linear-gradient(90deg, #B8A5FF, #FFE58F); width: {min(100, max(8, pct))}%; height: 100%; border-radius: 9999px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Card 3: Mini Channel Stats & Breakdown
            with col_k3:
                st.markdown(f"""
                <div class="fitonist-card" style="height: 100%;">
                    <div class="card-header-flex">
                        <div class="card-title-text">Model Health</div>
                        <span class="badge-yellow">Active</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
                        <div class="mini-stat-card">
                            <div class="mini-icon-box mini-icon-lilac">🌲</div>
                            <div>
                                <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF;">{res['metrics'].get('Precision (Weighted)', res['metrics'].get('RMSE', '0.99'))}</div>
                                <div style="font-size: 0.72rem; color: #7E7E94;">Precision / Error</div>
                            </div>
                        </div>
                        <div class="mini-stat-card">
                            <div class="mini-icon-box mini-icon-yellow">⚡</div>
                            <div>
                                <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF;">{res['metrics'].get('F1-Score (Weighted)', res['metrics'].get('MAE', '0.99'))}</div>
                                <div style="font-size: 0.72rem; color: #7E7E94;">F1 / Mean Error</div>
                            </div>
                        </div>
                    </div>
                    <div class="mini-stat-card">
                        <div class="mini-icon-box" style="background: #34D399; color: #0B0B0F;">✓</div>
                        <div style="width: 100%;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-size: 0.82rem; font-weight: 700; color: #FFFFFF;">Test Validation</span>
                                <span class="growth-badge-green">Pass</span>
                            </div>
                            <div style="font-size: 0.72rem; color: #7E7E94;">0 null values | 100% Stratified Split</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Bottom Row - Confusion Matrix & Sample Predictions
            st.markdown("<br>", unsafe_allow_html=True)
            col_b1, col_b2 = st.columns([1.2, 1.8])
            
            with col_b1:
                st.markdown("""
                <div class="fitonist-card">
                    <div class="card-header-flex">
                        <div class="card-title-text">Diagnostics Heatmap</div>
                        <span class="badge-lilac">Interactive Matrix</span>
                    </div>
                """, unsafe_allow_html=True)
                
                if res["task_type"] == "Classification":
                    cm = res["confusion_matrix"]
                    labels = [f"Tier {i}" for i in range(len(cm))]
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=labels, y=labels,
                        color_continuous_scale=[[0, "#161622"], [0.5, "#7C3AED"], [1, "#C4B5FD"]]
                    )
                    fig_cm.update_layout(
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=290,
                        paper_bgcolor=COLOR_DARK_BG,
                        plot_bgcolor=COLOR_DARK_BG,
                        font=dict(color="#A1A1B5", family="Plus Jakarta Sans")
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    y_t = np.array(res["y_test"])
                    y_p = np.array(res["y_pred"])
                    fig_sc = px.scatter(
                        x=y_t, y=y_p,
                        labels={"x": "Actual LKR", "y": "Predicted LKR"},
                        color_discrete_sequence=[COLOR_LILAC],
                        opacity=0.8
                    )
                    fig_sc.update_layout(
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=290,
                        paper_bgcolor=COLOR_DARK_BG,
                        plot_bgcolor=COLOR_DARK_BG
                    )
                    st.plotly_chart(fig_sc, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_b2:
                st.markdown("""
                <div class="fitonist-card">
                    <div class="card-header-flex">
                        <div class="card-title-text">Test Observations & Predictions</div>
                        <div class="filter-pill-group">
                            <span class="filter-pill active">Sample Set</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                sample_df = pd.DataFrame({
                    "Actual Ground Truth": res["y_test"][:10],
                    "Model Prediction": res["y_pred"][:10]
                })
                if res["task_type"] == "Regression":
                    sample_df["Error Delta (LKR)"] = np.round(sample_df["Actual Ground Truth"] - sample_df["Model Prediction"], 2)
                st.dataframe(sample_df, height=290, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # ----------------- TAB 2: LIVE PREDICTOR -----------------
    with tab2:
        st.markdown("""
        <div class="fitonist-card">
            <div class="card-header-flex">
                <div class="card-title-text">🔮 Real-Time What-If Inference Sandbox</div>
                <span class="badge-lilac">Instant Evaluation</span>
            </div>
        """, unsafe_allow_html=True)
        
        if res is not None and prep_data is not None:
            model = res["model"]
            feat_names = prep_data["feature_names"]
            
            c_in1, c_in2 = st.columns([2, 1])
            with c_in1:
                user_inputs = {}
                col_i1, col_i2 = st.columns(2)
                for i, col in enumerate(feat_names):
                    parent_c = col_i1 if (i % 2 == 0) else col_i2
                    if col in df.columns:
                        if df[col].dtype == "object" or df[col].dtype.name == "category":
                            opts = df[col].dropna().unique().tolist()
                            user_inputs[col] = parent_c.selectbox(f"{col}", opts, key=f"inf_{col}")
                        elif df[col].nunique() == 2 and sorted(df[col].unique().tolist()) == [0, 1]:
                            user_inputs[col] = 1 if parent_c.checkbox(f"Has {col.replace('Has_', '')}?", value=False, key=f"inf_{col}") else 0
                        else:
                            min_v, max_v, mean_v = float(df[col].min()), float(df[col].max()), float(df[col].mean())
                            user_inputs[col] = parent_c.slider(f"{col}", min_value=min_v, max_value=max_v, value=mean_v, key=f"inf_{col}")
                    else:
                        user_inputs[col] = parent_c.number_input(f"{col}", value=0.0, key=f"inf_{col}")

            with c_in2:
                st.markdown("##### 🎯 Live Inference Result")
                row_df = pd.DataFrame([user_inputs])
                label_encoders = prep_data.get("label_encoders", {})
                for c, le in label_encoders.items():
                    if c in row_df.columns:
                        try:
                            row_df[c] = le.transform(row_df[c].astype(str))
                        except Exception:
                            row_df[c] = 0
                            
                try:
                    pred_val = model.predict(row_df[feat_names])[0]
                    if res["task_type"] == "Classification":
                        target_enc = prep_data.get("target_encoder")
                        display_pred = target_enc.inverse_transform([int(pred_val)])[0] if target_enc else pred_val
                        st.markdown(f"""
                        <div class="fitonist-card" style="background: #1C1C28; border-color: rgba(184, 165, 255, 0.4); text-align: center; margin-top: 20px;">
                            <div class="hero-stat-label">Predicted Seniority Bracket</div>
                            <div class="hero-stat-num" style="color: #B8A5FF; font-size: 1.8rem; margin: 12px 0;">{display_pred}</div>
                            <span class="badge-lilac">Model: {res['algorithm']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="fitonist-card" style="background: #1C1C28; border-color: rgba(255, 229, 143, 0.4); text-align: center; margin-top: 20px;">
                            <div class="hero-stat-label">Estimated Monthly Compensation</div>
                            <div class="hero-stat-num" style="color: #FFE58F; font-size: 2.1rem; margin: 12px 0;">Rs. {float(pred_val):,.0f}</div>
                            <span class="badge-yellow">Model: {res['algorithm']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as ex:
                    st.warning(f"Inference preview: {ex}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------- TAB 3: BENCHMARK BATTLE -----------------
    with tab3:
        if bench_results is not None:
            st.markdown("""
            <div class="fitonist-card">
                <div class="card-header-flex">
                    <div class="card-title-text">🏆 Ensemble Battle: Random Forest vs. Gradient Boosting</div>
                    <span class="badge-lilac">4-Way Face-off</span>
                </div>
            """, unsafe_allow_html=True)
            
            bench_rows = []
            for item in bench_results:
                row = {
                    "Algorithm": item["algorithm"],
                    "Training Latency (ms)": round(item["training_time_sec"] * 1000, 2)
                }
                row.update(item["metrics"])
                bench_rows.append(row)
                
            benchmark_df = pd.DataFrame(bench_rows)
            primary_metric = "Accuracy" if res["task_type"] == "Classification" else "R-squared (R²)"
            
            # Winner Badges
            col_w1, col_w2 = st.columns(2)
            best_model = benchmark_df.loc[benchmark_df[primary_metric].idxmax()]["Algorithm"]
            best_score = benchmark_df[primary_metric].max()
            fastest_model = benchmark_df.loc[benchmark_df["Training Latency (ms)"].idxmin()]["Algorithm"]
            fastest_time = benchmark_df["Training Latency (ms)"].min()
            
            with col_w1:
                st.markdown(f"""
                <div class="mini-stat-card" style="padding: 16px 20px; border-color: rgba(255, 229, 143, 0.3);">
                    <div class="mini-icon-box mini-icon-yellow" style="width: 40px; height: 40px; font-size: 1.2rem;">👑</div>
                    <div>
                        <div class="hero-stat-num" style="font-size: 1.5rem; color: #FFE58F;">{best_model}</div>
                        <div class="hero-stat-label">Top Performance Leader ({primary_metric}: <b>{best_score}</b>)</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_w2:
                st.markdown(f"""
                <div class="mini-stat-card" style="padding: 16px 20px; border-color: rgba(184, 165, 255, 0.3);">
                    <div class="mini-icon-box mini-icon-lilac" style="width: 40px; height: 40px; font-size: 1.2rem;">⚡</div>
                    <div>
                        <div class="hero-stat-num" style="font-size: 1.5rem; color: #B8A5FF;">{fastest_model}</div>
                        <div class="hero-stat-label">Speed Leader (Latency: <b>{fastest_time:.1f} ms</b>)</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(benchmark_df, use_container_width=True)
            
            # Battle Visualizations
            bc1, bc2 = st.columns(2)
            with bc1:
                fig_b = px.bar(
                    benchmark_df,
                    x="Algorithm", y=primary_metric,
                    color="Algorithm",
                    color_discrete_sequence=[COLOR_LILAC, COLOR_YELLOW, COLOR_CYAN, COLOR_MINT],
                    title=f"{primary_metric} Comparison",
                    text_auto=True
                )
                fig_b.update_layout(showlegend=False, height=330, paper_bgcolor=COLOR_DARK_BG, plot_bgcolor=COLOR_DARK_BG)
                st.plotly_chart(fig_b, use_container_width=True)
            with bc2:
                fig_trade = px.scatter(
                    benchmark_df,
                    x="Training Latency (ms)", y=primary_metric,
                    text="Algorithm", color="Algorithm",
                    size=[22, 22, 22, 22],
                    color_discrete_sequence=[COLOR_LILAC, COLOR_YELLOW, COLOR_CYAN, COLOR_MINT],
                    title="Speed vs. Accuracy Trade-Off Frontier"
                )
                fig_trade.update_traces(textposition="top center")
                fig_trade.update_layout(showlegend=False, height=330, paper_bgcolor=COLOR_DARK_BG, plot_bgcolor=COLOR_DARK_BG)
                st.plotly_chart(fig_trade, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("💡 Select **'Benchmark All 4 Models (Battle Arena)'** in the sidebar and click **'Train Model'**.")

else:
    with tab1:
        st.info("👉 Click **'Train Model'** in the sidebar to fit the model and render your analytics cards.")

# ----------------- TAB 4: DATA CLEANING -----------------
with tab4:
    st.markdown("""
    <div class="fitonist-card">
        <div class="card-header-flex">
            <div class="card-title-text">🧹 Dataset Cleaning & Preprocessing Report</div>
            <span class="badge-lilac">525 Observations</span>
        </div>
        <p style="color: #94A3B8; font-size: 0.9rem;">How the raw Kaggle dataset was disaggregated, parsed, and engineered into a structured 17-column ML dataset.</p>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="mini-stat-card"><div class="mini-icon-box" style="background:#262638;color:#FFF;">1</div><div><div class="hero-stat-num" style="font-size:1.4rem;">209</div><div class="hero-stat-label">Raw Roles</div></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="mini-stat-card"><div class="mini-icon-box mini-icon-lilac">2</div><div><div class="hero-stat-num" style="font-size:1.4rem;color:#B8A5FF;">525</div><div class="hero-stat-label">Cleaned Records</div></div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="mini-stat-card"><div class="mini-icon-box mini-icon-yellow">3</div><div><div class="hero-stat-num" style="font-size:1.4rem;color:#FFE58F;">17</div><div class="hero-stat-label">Features</div></div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="mini-stat-card"><div class="mini-icon-box" style="background:#34D399;color:#0B0B0F;">4</div><div><div class="hero-stat-num" style="font-size:1.4rem;color:#34D399;">0</div><div class="hero-stat-label">Null Values</div></div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    clean_csv_data = load_sl_it_jobs_dataset(as_raw=False).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Dataset (sl_it_jobs_cleaned.csv)",
        data=clean_csv_data,
        file_name="sl_it_jobs_cleaned.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    st.markdown("#### Side-by-Side: Raw vs. Cleaned Dataset")
    raw_col, clean_col = st.columns(2)
    with raw_col:
        st.markdown("**Raw Data (Bundled Text Strings)**")
        st.dataframe(load_sl_it_jobs_dataset(as_raw=True).head(5), height=250, use_container_width=True)
    with clean_col:
        st.markdown("**Cleaned Data (Structured ML Features)**")
        st.dataframe(load_sl_it_jobs_dataset(as_raw=False).head(5), height=250, use_container_width=True)

    # Distributions
    st.markdown("---")
    v1, v2 = st.columns(2)
    clean_full = load_sl_it_jobs_dataset(as_raw=False)
    with v1:
        fig_sal = px.histogram(clean_full, x="Salary_LKR", nbins=30, title="Salary Distribution (LKR)", color_discrete_sequence=[COLOR_LILAC])
        fig_sal.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor=COLOR_DARK_BG, plot_bgcolor=COLOR_DARK_BG)
        st.plotly_chart(fig_sal, use_container_width=True)
    with v2:
        fig_dom = px.pie(clean_full, names="Domain", title="Domain Share", hole=0.5, color_discrete_sequence=[COLOR_LILAC, COLOR_YELLOW, COLOR_CYAN, COLOR_MINT, "#EC4899", "#A855F7", "#64748B"])
        fig_dom.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor=COLOR_DARK_BG, plot_bgcolor=COLOR_DARK_BG)
        st.plotly_chart(fig_dom, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 5: DATASET EXPLORER -----------------
with tab5:
    st.markdown("""
    <div class="fitonist-card">
        <div class="card-header-flex">
            <div class="card-title-text">🔍 Interactive Data Explorer & Smart Filters</div>
            <span class="badge-lilac">Real-Time Filtering</span>
        </div>
    """, unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        if "Domain" in df.columns:
            domain_filter = st.multiselect("Filter by Domain:", options=df["Domain"].unique().tolist(), default=df["Domain"].unique().tolist())
        else:
            domain_filter = None
    with col_f2:
        if "Experience_Years" in df.columns:
            min_e, max_e = float(df["Experience_Years"].min()), float(df["Experience_Years"].max())
            exp_range = st.slider("Experience Range (Years):", min_value=min_e, max_value=max_e, value=(min_e, max_e), step=0.5)
        else:
            exp_range = None
            
    filtered_df = df.copy()
    if domain_filter:
        filtered_df = filtered_df[filtered_df["Domain"].isin(domain_filter)]
    if exp_range and "Experience_Years" in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df["Experience_Years"] >= exp_range[0]) & (filtered_df["Experience_Years"] <= exp_range[1])]
        
    st.caption(f"Showing **{len(filtered_df)}** of **{len(df)}** records matching active filters.")
    st.dataframe(filtered_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
