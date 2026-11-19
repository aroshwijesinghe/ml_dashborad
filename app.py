import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from data_utils import (
    preprocess_data,
    get_sample_telecom_churn_data,
    get_sample_patient_data,
    load_sl_it_jobs_dataset
)
from model_trainer import train_and_evaluate

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="NexML | Model Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- COOL MODERN CSS THEME -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hero Gradient Title */
    .hero-title {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 35%, #EC4899 70%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 500;
        margin-top: 4px;
        margin-bottom: 24px;
    }
    
    /* Glowing Metric Cards */
    .metric-box {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-box-top-border {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366F1, #EC4899);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 4px;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #10B981;
        margin-top: 2px;
        font-weight: 500;
    }
    
    /* Algorithm Pills */
    .algo-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 6px;
    }
    .badge-rf { background: rgba(6, 182, 212, 0.15); color: #22D3EE; border: 1px solid rgba(6, 182, 212, 0.3); }
    .badge-xgb { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-lgbm { background: rgba(99, 102, 241, 0.15); color: #818CF8; border: 1px solid rgba(99, 102, 241, 0.3); }
    .badge-cb { background: rgba(236, 72, 153, 0.15); color: #F472B6; border: 1px solid rgba(236, 72, 153, 0.3); }
    
    /* Custom Info Callouts */
    .custom-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
    }
    
    /* Interactive Button styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- HERO HEADER -----------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="hero-title">⚡ NexML Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">High-performance evaluation, diagnostics & live inference for <b>Random Forest, XGBoost, LightGBM, and CatBoost</b>.</div>', unsafe_allow_html=True)
with col_h2:
    st.markdown("""
    <div style="text-align: right; padding-top: 10px;">
        <span class="algo-badge badge-rf">🌲 Random Forest</span>
        <span class="algo-badge badge-xgb">🚀 XGBoost</span>
        <span class="algo-badge badge-lgbm">⚡ LightGBM</span>
        <span class="algo-badge badge-cb">🐱 CatBoost</span>
    </div>
    """, unsafe_allow_html=True)

# ----------------- SIDEBAR CONFIGURATION -----------------
with st.sidebar:
    st.markdown("### 🎛️ Dataset Control")
    
    data_source = st.radio(
        "Choose Data Source:",
        [
            "🇱🇰 Kaggle: Sri Lanka IT Jobs Dataset",
            "📤 Upload Dataset (CSV / Excel)",
            "📞 Sample: Telecom Churn (Classification)",
            "🩺 Sample: Patient Health (Regression)"
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

    elif data_source == "📤 Upload Dataset (CSV / Excel)":
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
            st.info("Upload your custom CSV or pick the pre-loaded Kaggle dataset.")

    elif data_source == "📞 Sample: Telecom Churn (Classification)":
        df = get_sample_telecom_churn_data()
        st.info("Loaded Telecom Churn dataset (1,000 records).")

    elif data_source == "🩺 Sample: Patient Health (Regression)":
        df = get_sample_patient_data()
        st.info("Loaded Patient Disease Progression dataset (442 records).")

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
        elif "Churn" in columns:
            default_target_idx = columns.index("Churn")
        elif "DiseaseProgression" in columns:
            default_target_idx = columns.index("DiseaseProgression")
            
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
        
        with st.expander("⚙️ Hyperparameter Sandbox", expanded=False):
            max_depth_val = st.slider("Max Depth", min_value=1, max_value=20, value=6, help="Maximum tree depth.")
            n_estimators_val = st.slider("Trees / Iterations", min_value=10, max_value=300, value=100, step=10, help="Number of boosting/bagging trees.")
            learning_rate_val = st.slider("Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Step size shrinkage for gradient boosting.")
            
            hyperparams = {
                "max_depth": max_depth_val,
                "n_estimators": n_estimators_val,
                "learning_rate": learning_rate_val
            }
            
        run_button = st.button("🚀 Train & Analyze Model", type="primary", use_container_width=True)

# ----------------- MAIN APP WORKSPACE -----------------
if df is None:
    st.info("👈 Please select a dataset in the sidebar to begin.")
    st.stop()

if not selected_features:
    st.warning("⚠️ Please select at least one feature column from the sidebar.")
    st.stop()

# ----------------- 5 DETAILED TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model Performance & Diagnostics",
    "🔮 Live Prediction Sandbox",
    "🏆 Benchmark Arena (Battle Mode)",
    "🧹 Data Cleaning Report",
    "🔍 Interactive Data Explorer"
])

# ----------------- TRAIN LOGIC (SESSION STATE) -----------------
if run_button or "current_result" in st.session_state:
    if run_button:
        with st.spinner("⚡ Training model pipeline..."):
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
                prog = st.progress(0, text="Benchmarking models...")
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
    
    # ----------------- TAB 1: MODEL PERFORMANCE -----------------
    with tab1:
        if res is not None:
            # Header with badges
            algo_name = res["algorithm"]
            st.markdown(f"### Performance Dashboard: `{algo_name}` ({res['task_type']})")
            st.caption(f"⏱️ Training Time: **{res['training_time_sec'] * 1000:.1f} ms** | Test Partition: **{len(res['y_test'])} samples**")
            
            # Glowing Metric Cards
            metrics = res["metrics"]
            m_cols = st.columns(len(metrics) + 1)
            
            for i, (m_name, m_val) in enumerate(metrics.items()):
                with m_cols[i]:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-box-top-border"></div>
                        <div class="metric-title">{m_name}</div>
                        <div class="metric-val">{m_val}</div>
                        <div class="metric-sub">✓ Validated on test set</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with m_cols[-1]:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-box-top-border" style="background: linear-gradient(90deg, #10B981, #06B6D4);"></div>
                    <div class="metric-title">Speed / Latency</div>
                    <div class="metric-val">{res['training_time_sec'] * 1000:.1f}<span style="font-size: 1rem; color: #94A3B8;">ms</span></div>
                    <div class="metric-sub">🚀 Fast fit time</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Interactive Charts
            if res["task_type"] == "Classification":
                col_c1, col_c2 = st.columns([1, 1])
                
                with col_c1:
                    st.markdown("#### 🎯 Interactive Confusion Matrix")
                    cm = res["confusion_matrix"]
                    labels = [f"Class {i}" for i in range(len(cm))]
                    
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        labels=dict(x="Predicted Class", y="Ground Truth Class", color="Sample Count"),
                        x=labels,
                        y=labels,
                        color_continuous_scale="Purples"
                    )
                    fig_cm.update_layout(
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=360,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                with col_c2:
                    st.markdown("#### 📊 Top Feature Importances")
                    feat_df = res["feature_importance"]
                    if not feat_df.empty:
                        top_n = st.slider("Features to show:", min_value=3, max_value=min(15, len(feat_df)), value=min(8, len(feat_df)), key="feat_slider")
                        fig_feat = px.bar(
                            feat_df.head(top_n),
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            color="Importance",
                            color_continuous_scale="Sunset"
                        )
                        fig_feat.update_layout(
                            yaxis=dict(autorange="reversed"),
                            margin=dict(l=20, r=20, t=30, b=20),
                            height=360,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig_feat, use_container_width=True)

                st.markdown("#### 📑 Comprehensive Classification Breakdown")
                rep_df = pd.DataFrame(res["classification_report"]).transpose()
                st.dataframe(rep_df.style.background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]).format(precision=3), use_container_width=True)

            else:
                col_r1, col_r2 = st.columns([1, 1])
                
                with col_r1:
                    st.markdown("#### 🎯 Actual vs. Predicted Values")
                    y_test_arr = np.array(res["y_test"])
                    y_pred_arr = np.array(res["y_pred"])
                    
                    fig_scatter = px.scatter(
                        x=y_test_arr,
                        y=y_pred_arr,
                        labels={"x": "Actual Values", "y": "Predicted Values"},
                        opacity=0.75,
                        color=np.abs(y_test_arr - y_pred_arr),
                        color_continuous_scale="Viridis"
                    )
                    min_val = min(y_test_arr.min(), y_pred_arr.min())
                    max_val = max(y_test_arr.max(), y_pred_arr.max())
                    fig_scatter.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode="lines", name="Perfect Fit (y = x)",
                        line=dict(color="#EF4444", dash="dash", width=2)
                    ))
                    fig_scatter.update_layout(
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=360,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                with col_r2:
                    st.markdown("#### 📊 Top Feature Importances")
                    feat_df = res["feature_importance"]
                    if not feat_df.empty:
                        fig_feat = px.bar(
                            feat_df.head(8),
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            color="Importance",
                            color_continuous_scale="Tealgrn"
                        )
                        fig_feat.update_layout(
                            yaxis=dict(autorange="reversed"),
                            margin=dict(l=20, r=20, t=30, b=20),
                            height=360,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig_feat, use_container_width=True)

            # Sample Predictions
            st.markdown("#### 🔍 Test Set Prediction Samples")
            sample_df = pd.DataFrame({
                "Actual Ground Truth": res["y_test"][:12],
                "Model Prediction": res["y_pred"][:12]
            })
            if res["task_type"] == "Regression":
                sample_df["Error Delta"] = np.round(sample_df["Actual Ground Truth"] - sample_df["Model Prediction"], 2)
            st.dataframe(sample_df, use_container_width=True)

    # ----------------- TAB 2: LIVE PREDICTION SANDBOX -----------------
    with tab2:
        st.subheader("🔮 Interactive Live Inference Sandbox")
        st.markdown("Simulate a custom candidate or scenario in real-time and observe model predictions:")
        
        if res is not None and prep_data is not None:
            model = res["model"]
            feat_names = prep_data["feature_names"]
            
            col_inf1, col_inf2 = st.columns([2, 1])
            
            with col_inf1:
                st.markdown("##### 📝 Input Feature Values")
                user_inputs = {}
                
                # Split inputs into two columns
                input_c1, input_c2 = st.columns(2)
                for i, col in enumerate(feat_names):
                    parent_c = input_c1 if (i % 2 == 0) else input_c2
                    
                    if col in df.columns:
                        if df[col].dtype == "object" or df[col].dtype.name == "category":
                            options = df[col].dropna().unique().tolist()
                            user_inputs[col] = parent_c.selectbox(f"{col}", options, key=f"inf_{col}")
                        elif df[col].nunique() == 2 and sorted(df[col].unique().tolist()) == [0, 1]:
                            user_inputs[col] = 1 if parent_c.checkbox(f"Has {col.replace('Has_', '')}?", value=False, key=f"inf_{col}") else 0
                        else:
                            min_v = float(df[col].min())
                            max_v = float(df[col].max())
                            mean_v = float(df[col].mean())
                            user_inputs[col] = parent_c.slider(f"{col}", min_value=min_v, max_value=max_v, value=mean_v, key=f"inf_{col}")
                    else:
                        user_inputs[col] = parent_c.number_input(f"{col}", value=0.0, key=f"inf_{col}")
            
            with col_inf2:
                st.markdown("##### 🎯 Real-Time Prediction Output")
                
                # Build test row
                row_df = pd.DataFrame([user_inputs])
                # Encode categorical features
                label_encoders = prep_data.get("label_encoders", {})
                for c, le in label_encoders.items():
                    if c in row_df.columns:
                        try:
                            row_df[c] = le.transform(row_df[c].astype(str))
                        except Exception:
                            row_df[c] = 0
                            
                # Predict
                try:
                    pred_val = model.predict(row_df[feat_names])[0]
                    
                    if res["task_type"] == "Classification":
                        target_enc = prep_data.get("target_encoder")
                        display_pred = target_enc.inverse_transform([int(pred_val)])[0] if target_enc else pred_val
                        
                        st.markdown(f"""
                        <div class="metric-box" style="margin-top: 15px; border-color: rgba(99, 102, 241, 0.5);">
                            <div class="metric-box-top-border" style="background: linear-gradient(90deg, #6366F1, #8B5CF6);"></div>
                            <div class="metric-title">Predicted Classification</div>
                            <div class="metric-val" style="color: #A78BFA; font-size: 1.6rem;">{display_pred}</div>
                            <div class="metric-sub">Generated by {res['algorithm']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-box" style="margin-top: 15px; border-color: rgba(16, 185, 129, 0.5);">
                            <div class="metric-box-top-border" style="background: linear-gradient(90deg, #10B981, #06B6D4);"></div>
                            <div class="metric-title">Predicted Value</div>
                            <div class="metric-val" style="color: #34D399; font-size: 1.8rem;">{float(pred_val):,.2f}</div>
                            <div class="metric-sub">Estimated Monthly LKR</div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as ex:
                    st.warning(f"Live prediction preview: {ex}")
        else:
            st.info("Train a model first to unlock live prediction sandbox.")

    # ----------------- TAB 3: BENCHMARK ARENA -----------------
    with tab3:
        if bench_results is not None:
            st.subheader("🏆 Model Benchmark Arena: Random Forest vs. Gradient Boosting")
            st.caption("Side-by-side confrontation across Random Forest, XGBoost, LightGBM, and CatBoost.")
            
            bench_rows = []
            for item in bench_results:
                row = {
                    "Algorithm": item["algorithm"],
                    "Training Latency (ms)": round(item["training_time_sec"] * 1000, 2)
                }
                row.update(item["metrics"])
                bench_rows.append(row)
                
            benchmark_df = pd.DataFrame(bench_rows)
            
            # Winner badge callout
            primary_metric = "Accuracy" if res["task_type"] == "Classification" else "R-squared (R²)"
            if primary_metric in benchmark_df.columns:
                best_model = benchmark_df.loc[benchmark_df[primary_metric].idxmax()]["Algorithm"]
                best_score = benchmark_df[primary_metric].max()
                fastest_model = benchmark_df.loc[benchmark_df["Training Latency (ms)"].idxmin()]["Algorithm"]
                fastest_time = benchmark_df["Training Latency (ms)"].min()
                
                col_win1, col_win2 = st.columns(2)
                with col_win1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-box-top-border" style="background: linear-gradient(90deg, #F59E0B, #EF4444);"></div>
                        <div class="metric-title">👑 Top Accuracy Winner</div>
                        <div class="metric-val" style="color: #FBBF24;">{best_model}</div>
                        <div class="metric-sub">{primary_metric}: <b>{best_score}</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_win2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-box-top-border" style="background: linear-gradient(90deg, #10B981, #06B6D4);"></div>
                        <div class="metric-title">⚡ Speed Champion</div>
                        <div class="metric-val" style="color: #34D399;">{fastest_model}</div>
                        <div class="metric-sub">Latency: <b>{fastest_time:.1f} ms</b></div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(benchmark_df.style.highlight_max(color="#1e3a5f", subset=[c for c in benchmark_df.columns if c not in ["Algorithm", "RMSE", "MAE", "MAPE", "Training Latency (ms)"]]), use_container_width=True)
            
            # Visual Benchmark Charts
            b_c1, b_c2 = st.columns(2)
            with b_c1:
                fig_bench = px.bar(
                    benchmark_df,
                    x="Algorithm",
                    y=primary_metric,
                    color="Algorithm",
                    color_discrete_sequence=["#06B6D4", "#F59E0B", "#6366F1", "#EC4899"],
                    title=f"Comparative {primary_metric} by Algorithm",
                    text_auto=True
                )
                fig_bench.update_layout(showlegend=False, height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bench, use_container_width=True)

            with b_c2:
                # Speed vs Accuracy trade-off bubble chart
                fig_tradeoff = px.scatter(
                    benchmark_df,
                    x="Training Latency (ms)",
                    y=primary_metric,
                    text="Algorithm",
                    color="Algorithm",
                    size=[20, 20, 20, 20],
                    title="Speed vs. Accuracy Trade-Off Frontier",
                    color_discrete_sequence=["#06B6D4", "#F59E0B", "#6366F1", "#EC4899"]
                )
                fig_tradeoff.update_traces(textposition="top center")
                fig_tradeoff.update_layout(showlegend=False, height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_tradeoff, use_container_width=True)

        else:
            st.info("💡 Select **'Benchmark All 4 Models (Battle Arena)'** in the sidebar and click **'Train & Analyze Model'** to unlock the multi-model comparison.")

else:
    with tab1:
        st.info("👉 Click **'Train & Analyze Model'** in the sidebar to run the model and explore metrics, confusion matrix, and feature importances.")

# ----------------- TAB 4: DATA CLEANING REPORT -----------------
with tab4:
    st.subheader("🧹 Dataset Cleaning & Preprocessing Report")
    st.markdown("""
    The raw **Sri Lanka IT Jobs Dataset** (`miyurusanjana/sl-it-jobs-dataset`) originally contained **209 rows and 5 string columns**, 
    with nested data, multiple job seniority levels bundled into single cells, and formatted salary strings.
    Here is how we transformed it into a clean **525-row, 17-column tabular dataset** ready for machine learning.
    """)

    # Interactive Before vs After Metrics
    c_clean1, c_clean2, c_clean3, c_clean4 = st.columns(4)
    with c_clean1:
        st.markdown("""<div class="metric-box"><div class="metric-title">Raw Observations</div><div class="metric-val">209</div><div class="metric-sub">Bundled roles</div></div>""", unsafe_allow_html=True)
    with c_clean2:
        st.markdown("""<div class="metric-box"><div class="metric-title">Cleaned Observations</div><div class="metric-val" style="color: #34D399;">525</div><div class="metric-sub">+316 expanded rows</div></div>""", unsafe_allow_html=True)
    with c_clean3:
        st.markdown("""<div class="metric-box"><div class="metric-title">Raw Columns</div><div class="metric-val">5</div><div class="metric-sub">Unstructured strings</div></div>""", unsafe_allow_html=True)
    with c_clean4:
        st.markdown("""<div class="metric-box"><div class="metric-title">Engineered Features</div><div class="metric-val" style="color: #818CF8;">17</div><div class="metric-sub">ML features & targets</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Download Button
    clean_csv_data = load_sl_it_jobs_dataset(as_raw=False).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Dataset (sl_it_jobs_cleaned.csv)",
        data=clean_csv_data,
        file_name="sl_it_jobs_cleaned.csv",
        mime="text/csv",
        help="Download the verified 525-row cleaned tabular dataset."
    )
    
    st.markdown("---")
    st.markdown("### 📋 Step-by-Step Data Cleaning Pipeline")
    
    col_step1, col_step2 = st.columns(2)
    with col_step1:
        st.markdown("""
        #### 1. Row Disaggregation & Level Expansion
        * **The Problem**: Each raw record compressed multiple seniority levels (e.g., `Entry`, `Mid`, `Senior`, or `Unpaid`, `Paid internship`) into a single row using pipe `|` delimiters.
        * **How We Cleaned It**: Deconstructed each bundled role into individual job postings, yielding **525 distinct, valid observations**.
        
        #### 2. Numeric Extraction & Normalization
        * **The Problem**: Salary ranges were text strings with commas and plus signs (e.g., `'120,000-180,000 (Entry) | 300,000-500,000+ (Senior)'`).
        * **How We Cleaned It**: 
          - Stripped currency formatting, commas, and plus symbols with regular expressions.
          - Extracted `Min_Salary_LKR` and `Max_Salary_LKR`.
          - Calculated the central midpoint `Salary_LKR = (Min + Max) / 2` as the **continuous target for Regression**.
        """)
    with col_step2:
        st.markdown("""
        #### 3. Experience Standardization
        * **The Problem**: Experience contained qualitative labels (e.g., `'Fresh graduate / Final year student'`) mixed with numerical ranges (`'0-2'`, `'3-5'`, `'6+'`).
        * **How We Cleaned It**: Mapped textual descriptions to quantitative years (e.g., fresh graduate $\\to$ 0.0 yrs, 0-2 yrs $\\to$ 1.0 yr, 3-5 yrs $\\to$ 4.0 yrs, 6+ yrs $\\to$ 7.5 yrs) producing the continuous feature `Experience_Years`.

        #### 4. Feature Engineering & Domain Classification
        * **Job Domain**: Categorized roles into 7 industry sectors (`Software Engineering`, `Data Science & AI`, `Cloud & DevOps`, `Cybersecurity`, `QA & Testing`, `UI/UX`, `Management`).
        * **Skill Indicators**: Created binary flags for in-demand technologies: `Has_Python`, `Has_JavaScript`, `Has_Java`, `Has_SQL`, `Has_Cloud`, `Has_AI_ML`, `Has_Docker_K8s`.
        * **Skill Counts**: Derived `Tech_Count` and `Soft_Count` to measure skill breadth.
        """)

    # Visual Before vs After preview
    st.markdown("---")
    st.markdown("### 🔍 Side-by-Side: Raw vs. Cleaned Dataset")
    raw_col, clean_col = st.columns(2)
    with raw_col:
        st.markdown("**Raw Data Sample (Unstructured Text)**")
        raw_preview = load_sl_it_jobs_dataset(as_raw=True).head(5)
        st.dataframe(raw_preview, height=260, use_container_width=True)
    with clean_col:
        st.markdown("**Cleaned Data Sample (Structured Tabular Features)**")
        clean_preview = load_sl_it_jobs_dataset(as_raw=False).head(5)
        st.dataframe(clean_preview, height=260, use_container_width=True)

    # Visualizations of cleaned distributions
    st.markdown("---")
    st.markdown("### 📊 Distribution of Cleaned Variables")
    clean_full = load_sl_it_jobs_dataset(as_raw=False)
    v1, v2 = st.columns(2)
    with v1:
        fig_sal = px.histogram(
            clean_full,
            x="Salary_LKR",
            nbins=30,
            title="Cleaned Monthly Salary Distribution (LKR)",
            color_discrete_sequence=["#6366F1"]
        )
        fig_sal.update_layout(margin=dict(l=20, r=20, t=35, b=20), height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sal, use_container_width=True)
    with v2:
        fig_dom = px.pie(
            clean_full,
            names="Domain",
            title="Job Distribution by Domain",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        fig_dom.update_layout(margin=dict(l=20, r=20, t=35, b=20), height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dom, use_container_width=True)

# ----------------- TAB 5: DATA EXPLORER & FILTERS -----------------
with tab5:
    st.subheader("🔍 Interactive Data Explorer & Smart Filtering")
    
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
    
    with st.expander("📊 Statistical Summary (Describe)", expanded=False):
        st.dataframe(filtered_df.describe(include="all"), use_container_width=True)
