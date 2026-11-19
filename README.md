# ⚡ NexML Intelligence Dashboard

An interactive, high-performance Machine Learning web dashboard built with **Streamlit**, **Scikit-Learn**, **XGBoost**, **LightGBM**, **CatBoost**, and **Plotly** to train, benchmark, and deploy tree-based ensemble models on tabular datasets.

---

## 🤖 Supported ML Algorithms

1. 🌲 **Random Forest (RF)** — Bagging ensemble of randomized decision trees
2. 🚀 **XGBoost** — Scalable, distributed gradient-boosted decision trees
3. ⚡ **LightGBM** — Fast histogram-based gradient boosting framework
4. 🐱 **CatBoost** — Categorical feature-optimized gradient boosting

---

## 🇱🇰 Integrated Dataset: Sri Lanka IT Jobs Dataset

* **Source**: Kaggle (`miyurusanjana/sl-it-jobs-dataset`)
* **Raw vs. Cleaned**:
  * Raw dataset: 209 multi-level text rows
  * Cleaned dataset: 525 structured observations with 17 extracted features and 0 nulls.
* **Supported Tasks**:
  * **Classification**: Predict `Salary_Bracket` (`<50k`, `50k-150k`, `150k-300k`, `300k-500k`, `>500k`) or `Job_Level`.
  * **Regression**: Predict `Salary_LKR` (monthly salary in Sri Lankan Rupees).
* **Engineered Features**: `Experience_Years`, `Domain`, `Tech_Count`, `Soft_Count`, `Has_Python`, `Has_JavaScript`, `Has_Java`, `Has_SQL`, `Has_Cloud`, `Has_AI_ML`, `Has_Docker_K8s`.

---

## ✨ Dashboard Highlights

* **Glowing Glassmorphism KPI Cards**: Accuracy, Precision, Recall, F1, ROC-AUC, $R^2$, RMSE, MAE, and inference latency.
* **Interactive Diagnostics**: Interactive Confusion Matrix heatmap, custom gradient Feature Importance rankings, and Actual vs Predicted scatter plots.
* **🔮 Live Prediction Sandbox**: Real-time "What-If" simulator predicting salary and seniority for custom candidates.
* **🏆 Benchmark Battle Arena**: Side-by-side comparative leaderboard with **Speed vs. Accuracy Trade-Off Frontier** bubble charts.
* **🧹 Data Cleaning Report**: In-depth methodology breakdown with side-by-side raw vs. cleaned tables and a one-click CSV download.
* **🔍 Data Explorer**: Dynamic multi-filter engine by domain and experience ranges.

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.
