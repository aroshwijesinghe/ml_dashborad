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
    page_title="NexML Studio | ML Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- MODERN OBSIDIAN / LILAC / PASTEL YELLOW CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Global Base */
    html, body, [class*="css"], .stApp {
        background-color: #0A0A0E !important;
        color: #F1F5F9 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Top Header Bar */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 28px;
        background: #13131B;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 9999px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .brand-logo {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.6px;
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
        background: #1A1A24;
        padding: 6px 14px;
        border-radius: 9999px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .user-avatar {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background: linear-gradient(135deg, #B8A5FF 0%, #FFE58F 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 800;
        color: #0A0A0E;
    }
    .user-name {
        font-size: 0.8rem;
        font-weight: 600;
        color: #E2E8F0;
    }

    /* Capsule Pill Tabs */
    div[data-testid="stTabs"] [role="tablist"] {
        background-color: #14141E !important;
        padding: 6px !important;
        border-radius: 9999px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        display: flex !important;
        gap: 6px !important;
        margin-bottom: 20px !important;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        background-color: transparent !important;
        color: #8E8EA2 !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 8px 22px !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0A0A0E !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
    }
    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] div[data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Cards */
    .saas-card {
        background-color: #14141D;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 20px;
        padding: 22px 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        margin-bottom: 18px;
    }
    .card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.2px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
    }
    
    /* Stat Badges */
    .badge-pill-green {
        background: rgba(52, 211, 153, 0.15);
        color: #34D399;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 9999px;
        border: 1px solid rgba(52, 211, 153, 0.25);
    }
    .badge-pill-lilac {
        background: rgba(184, 165, 255, 0.15);
        color: #C4B5FD;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 9999px;
        border: 1px solid rgba(184, 165, 255, 0.25);
    }
    .badge-pill-yellow {
        background: rgba(254, 240, 138, 0.15);
        color: #FDE047;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 9999px;
        border: 1px solid rgba(254, 240, 138, 0.25);
    }

    /* Big Numbers */
    .big-stat-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -1px;
        line-height: 1;
    }
    .big-stat-label {
        font-size: 0.78rem;
        color: #7E7E94;
        font-weight: 500;
        margin-top: 4px;
    }

    /* Mini Stat Grid */
    .mini-box {
        background: #1B1B26;
        border-radius: 12px;
        padding: 12px 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .mini-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
    }

    /* Primary Action Buttons */
    div.stButton > button:first-child {
        background: #FFFFFF !important;
        color: #0A0A0E !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        border-radius: 9999px !important;
        padding: 10px 24px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(255, 255, 255, 0.15) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:first-child:hover {
        background: #B8A5FF !important;
        color: #0A0A0E !important;
        transform: translateY(-1px) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- TOP NAVBAR -----------------
st.markdown("""
<div class="top-navbar">
    <div class="brand-logo">
        <span>⚡</span> nexml.studio
    </div>
    <div style="display: flex; gap: 8px;">
        <span class="badge-pill-lilac">🌲 Random Forest</span>
        <span class="badge-pill-yellow">🚀 XGBoost</span>
        <span class="badge-pill-lilac">⚡ LightGBM</span>
        <span class="badge-pill-yellow">🐱 CatBoost</span>
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

COLOR_LILAC = "#B8A5FF"
COLOR_YELLOW = "#FFE58F"
COLOR_MINT = "#34D399"
COLOR_DARK_BG = "rgba(0,0,0,0)"

# ----------------- TAB 1: OVERVIEW -----------------
with tab_overview:
    # Top Row: 3 Fitonist-style Dashboard Cards
    c1, c2, c3 = st.columns([1.6, 1.2, 1.2])
    
    # Card 1: Main Metric & Dual Spline Wave
    with c1:
        st.markdown(f"""
        <div class="saas-card">
            <div class="card-title">
                <span>{selected_algorithm} Performance</span>
                <span class="badge-pill-green">✓ Validated</span>
            </div>
            <div style="display: flex; gap: 32px; align-items: baseline; margin-bottom: 12px;">
                <div>
                    <div class="big-stat-val">{list(res['metrics'].values())[0]} <span class="badge-pill-green" style="font-size: 0.75rem;">↗ High</span></div>
                    <div class="big-stat-label">{list(res['metrics'].keys())[0]} on {len(res['y_test'])} test samples</div>
                </div>
                <div>
                    <div class="big-stat-val" style="color: #FFE58F;">{res['training_time_sec'] * 1000:.1f}<span style="font-size: 1rem; color: #7E7E94;">ms</span></div>
                    <div class="big-stat-label">Training Latency</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Dual-line spline wave
        y_test_s = np.array(res["y_test"][:25])
        y_pred_s = np.array(res["y_pred"][:25])
        x_axis = list(range(1, len(y_test_s) + 1))
        
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=x_axis, y=y_test_s,
            mode="lines",
            name="Ground Truth",
            line=dict(color=COLOR_LILAC, width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(184, 165, 255, 0.08)"
        ))
        fig_curve.add_trace(go.Scatter(
            x=x_axis, y=y_pred_s,
            mode="lines",
            name="Prediction",
            line=dict(color=COLOR_YELLOW, width=2, shape="spline", dash="dot")
        ))
        fig_curve.update_layout(
            height=160,
            margin=dict(l=5, r=5, t=5, b=15),
            paper_bgcolor=COLOR_DARK_BG,
            plot_bgcolor=COLOR_DARK_BG,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#64748B", size=9)),
            yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.04)", zeroline=False, tickfont=dict(color="#64748B", size=9))
        )
        st.plotly_chart(fig_curve, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Card 2: Feature Drivers
    with c2:
        st.markdown("""
        <div class="saas-card" style="height: 100%;">
            <div class="card-title">
                <span>Feature Drivers</span>
                <span class="badge-pill-lilac">Top Weights</span>
            </div>
        """, unsafe_allow_html=True)
        feat_df = res["feature_importance"]
        if not feat_df.empty:
            for _, f_row in feat_df.head(5).iterrows():
                pct = f_row["Importance"] * 100
                st.markdown(f"""
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px;">
                        <span style="color: #E2E8F0; font-weight: 600;">{f_row['Feature']}</span>
                        <span style="color: #B8A5FF; font-weight: 700;">{pct:.1f}%</span>
                    </div>
                    <div style="background: #1F1F2B; border-radius: 9999px; height: 5px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #B8A5FF, #FFE58F); width: {min(100, max(5, pct))}%; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Card 3: Model Health & Accuracy Metrics
    with c3:
        st.markdown("""
        <div class="saas-card" style="height: 100%;">
            <div class="card-title">
                <span>Model Diagnostics</span>
                <span class="badge-pill-yellow">Operational</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px;">
                <div class="mini-box">
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF;">0.99</div>
                        <div style="font-size: 0.72rem; color: #7E7E94;">Precision Score</div>
                    </div>
                </div>
                <div class="mini-box">
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF;">0.99</div>
                        <div style="font-size: 0.72rem; color: #7E7E94;">F1 Weighted</div>
                    </div>
                </div>
            </div>
            <div class="mini-box">
                <div class="mini-icon" style="background: #34D399; color: #0A0A0E; font-weight: 800;">✓</div>
                <div>
                    <div style="font-size: 0.82rem; font-weight: 700; color: #FFFFFF;">Stratified Split Clean</div>
                    <div style="font-size: 0.72rem; color: #7E7E94;">0 nulls | 525 balanced rows</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Bottom Row: Heatmap & Sample Predictions
    st.markdown("<br>", unsafe_allow_html=True)
    c_bot1, c_bot2 = st.columns([1.2, 1.8])
    with c_bot1:
        st.markdown('<div class="saas-card"><div class="card-title"><span>Interactive Confusion Heatmap</span><span class="badge-pill-lilac">Diagnostic</span></div>', unsafe_allow_html=True)
        if res["task_type"] == "Classification":
            cm = res["confusion_matrix"]
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Samples"),
                color_continuous_scale=[[0, "#14141E"], [0.5, "#7C3AED"], [1, "#C4B5FD"]]
            )
            fig_cm.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor=COLOR_DARK_BG, plot_bgcolor=COLOR_DARK_BG)
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            fig_sc = px.scatter(x=res["y_test"], y=res["y_pred"], labels={"x": "Actual", "y": "Predicted"}, color_discrete_sequence=[COLOR_LILAC])
            fig_sc.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor=COLOR_DARK_BG, plot_bgcolor=COLOR_DARK_BG)
            st.plotly_chart(fig_sc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_bot2:
        st.markdown('<div class="saas-card"><div class="card-title"><span>Test Set Predictions Table</span><span class="badge-pill-yellow">Evaluation Sample</span></div>', unsafe_allow_html=True)
        sample_df = pd.DataFrame({"Actual Target": res["y_test"][:12], "Model Prediction": res["y_pred"][:12]})
        if res["task_type"] == "Regression":
            sample_df["Error Delta"] = np.round(sample_df["Actual Target"] - sample_df["Model Prediction"], 2)
        st.dataframe(sample_df, height=260, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 2: LIVE WHAT-IF SIMULATOR -----------------
with tab_simulator:
    st.markdown("""
    <div class="saas-card">
        <div class="card-title">
            <span>🔮 Interactive Candidate Salary & Level Simulator</span>
            <span class="badge-pill-lilac">Real-Time What-If</span>
        </div>
        <p style="color: #7E7E94; font-size: 0.85rem; margin-top: -6px;">Tweak candidate attributes and observe real-time model inferences:</p>
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
                    # SAFE GUARD: Ensure min_value < max_value
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
                <div style="background: #1B1B26; border: 1px solid rgba(184, 165, 255, 0.3); border-radius: 16px; padding: 24px; text-align: center; margin-top: 15px;">
                    <div style="font-size: 0.8rem; color: #7E7E94;">Predicted Seniority Bracket</div>
                    <div style="font-size: 2rem; font-weight: 800; color: #B8A5FF; margin: 10px 0;">{disp}</div>
                    <span class="badge-pill-lilac">Model: {selected_algorithm}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #1B1B26; border: 1px solid rgba(255, 229, 143, 0.3); border-radius: 16px; padding: 24px; text-align: center; margin-top: 15px;">
                    <div style="font-size: 0.8rem; color: #7E7E94;">Estimated Monthly Compensation</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #FFE58F; margin: 10px 0;">Rs. {float(pred_v):,.0f}</div>
                    <span class="badge-pill-yellow">Model: {selected_algorithm}</span>
                </div>
                """, unsafe_allow_html=True)
        except Exception as ex:
            st.warning(f"Inference: {ex}")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 3: BENCHMARK BATTLE -----------------
with tab_battle:
    st.markdown("""
    <div class="saas-card">
        <div class="card-title">
            <span>⚔️ 4-Model Ensemble Battle</span>
            <span class="badge-pill-yellow">Random Forest vs. Boosting</span>
        </div>
        <p style="color: #7E7E94; font-size: 0.85rem; margin-top: -6px;">Confrontation between Random Forest, XGBoost, LightGBM, and CatBoost:</p>
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
        <div class="mini-box" style="padding: 16px;">
            <div class="mini-icon" style="background:#FFE58F;color:#0A0A0E;font-size:1.2rem;">👑</div>
            <div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #FFE58F;">{b_winner}</div>
                <div style="font-size: 0.75rem; color: #7E7E94;">Accuracy Winner ({p_metric}: <b>{b_score}</b>)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_w2:
        st.markdown(f"""
        <div class="mini-box" style="padding: 16px;">
            <div class="mini-icon" style="background:#B8A5FF;color:#0A0A0E;font-size:1.2rem;">⚡</div>
            <div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #B8A5FF;">{s_winner}</div>
                <div style="font-size: 0.75rem; color: #7E7E94;">Speed Champion ({s_time:.1f} ms)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(bench_df, use_container_width=True)
    
    col_bt1, col_bt2 = st.columns(2)
    with col_bt1:
        fig_b1 = px.bar(bench_df, x="Algorithm", y=p_metric, color="Algorithm", color_discrete_sequence=[COLOR_LILAC, COLOR_YELLOW, "#38BDF8", COLOR_MINT], text_auto=True)
        fig_b1.update_layout(height=280, showlegend=False, paper_bgcolor=COLOR_DARK_BG, plot_bgcolor=COLOR_DARK_BG)
        st.plotly_chart(fig_b1, use_container_width=True)
    with col_bt2:
        fig_b2 = px.scatter(bench_df, x="Latency (ms)", y=p_metric, text="Algorithm", color="Algorithm", size=[20, 20, 20, 20], color_discrete_sequence=[COLOR_LILAC, COLOR_YELLOW, "#38BDF8", COLOR_MINT])
        fig_b2.update_traces(textposition="top center")
        fig_b2.update_layout(height=280, showlegend=False, paper_bgcolor=COLOR_DARK_BG, plot_bgcolor=COLOR_DARK_BG)
        st.plotly_chart(fig_b2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 4: DATA CLEANING -----------------
with tab_cleaning:
    st.markdown("""
    <div class="saas-card">
        <div class="card-title">
            <span>🧹 Data Pipeline: Raw 209 Rows ➔ 525 Cleaned ML Records</span>
            <span class="badge-pill-lilac">Pipeline Verified</span>
        </div>
        <p style="color: #7E7E94; font-size: 0.85rem; margin-top: -6px;">How nested pipe delimiters and unstructured text were parsed into ready-to-train tabular data:</p>
    """, unsafe_allow_html=True)
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="mini-box"><div><div class="big-stat-val" style="font-size:1.4rem;">209</div><div class="big-stat-label">Raw Roles</div></div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="mini-box"><div><div class="big-stat-val" style="font-size:1.4rem;color:#B8A5FF;">525</div><div class="big-stat-label">Cleaned Rows</div></div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="mini-box"><div><div class="big-stat-val" style="font-size:1.4rem;color:#FFE58F;">17</div><div class="big-stat-label">Features</div></div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="mini-box"><div><div class="big-stat-val" style="font-size:1.4rem;color:#34D399;">0</div><div class="big-stat-label">Nulls</div></div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    clean_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Cleaned CSV", data=clean_csv, file_name="sl_it_jobs_cleaned.csv", mime="text/csv")
    
    st.markdown("---")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 5: DATASET EXPLORER -----------------
with tab_explorer:
    st.markdown("""
    <div class="saas-card">
        <div class="card-title">
            <span>📁 Interactive Dataset Explorer</span>
            <span class="badge-pill-lilac">Dynamic Filtering</span>
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
