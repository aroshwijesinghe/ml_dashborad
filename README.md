# 🔮 NexML Intelligence Studio | Glassmorphic & Embossed ML Dashboard

A Machine Learning web dashboard featuring a **Light Glassmorphism & Embossed (Neumorphism)** design system built with **Streamlit**, **Scikit-Learn**, **XGBoost**, **LightGBM**, **CatBoost**, and **Plotly**.

---

## 🎨 UI & Aesthetic Highlights

* **Light Glassmorphism**: Translucent frosted-glass surfaces (`backdrop-filter: blur(20px)`), soft aurora gradients (`#EFF4FC` to `#F4F0FD`), and glossy borders.
* **Embossed Neumorphic Elements**: Dual-directional inner & outer drop shadows simulating physical tactile cards, embossed pills, and inset statistical containers.
* **Cool Color Pattern**: Electric indigo (`#4F46E5`), vibrant cyan (`#06B6D4`), violet (`#7C3AED`), and fresh emerald (`#10B981`).
* **Interactive Architecture**:
  * Floating top glass navbar with algorithm status pills and user badge.
  * Instant horizontal model selector: **Random Forest**, **XGBoost**, **LightGBM**, and **CatBoost**.
  * 5 capsule pill tabs: Overview & Spline Waves, Live What-If Simulator, Benchmark Battle Arena, Data Cleaning Pipeline, and Dataset Explorer.

---

## 🤖 Supported ML Models

1. 🌲 **Random Forest (RF)**: Bagging ensemble of randomized decision trees.
2. 🚀 **XGBoost**: Scalable gradient tree boosting.
3. ⚡ **LightGBM**: Fast histogram-based gradient boosting.
4. 🐱 **CatBoost**: Categorical feature-optimized gradient boosting.

---

## 🇱🇰 Integrated Dataset: Sri Lanka IT Jobs Dataset

* **Source**: Kaggle (`miyurusanjana/sl-it-jobs-dataset`)
* **Cleaning Pipeline**: Disaggregated 209 multi-level text rows into **525 structured, ML-ready records** with 17 features and 0 null values.
* **Supported Tasks**:
  * **Classification**: Predict `Salary_Bracket` (`<50k`, `50k-150k`, `150k-300k`, `300k-500k`, `>500k`) or `Job_Level`.
  * **Regression**: Predict `Salary_LKR` (monthly salary in LKR).

---

## 🚀 Step-by-Step: How to Run the Dashboard

### Prerequisites
Make sure Python 3.10+ is installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/aroshwijesinghe/ml_dashborad.git
cd ml_dashborad
```

### Step 2: Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### Step 3: Launch the Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```

### Step 4: Open in Your Browser
The dashboard will open automatically. If not, open your web browser and navigate to:
```
http://localhost:8501
```

---

## 📁 Repository Structure

```
ml_dashborad/
├── app.py              # Main Streamlit dashboard with Light Glassmorphism + Embossed UI
├── model_trainer.py    # Training & evaluation engine for RF, XGBoost, LightGBM, CatBoost
├── clean_dataset.py    # 5-step data cleaning and transformation pipeline
├── data_utils.py       # Data loading, preprocessing, and encoding utilities
├── test_pipeline.py    # Automated test suite verifying all 4 models
├── data/
│   ├── sri_lanka_it_jobs_dataset.csv   # Raw Kaggle dataset
│   └── sl_it_jobs_cleaned.csv         # Cleaned 525-row tabular dataset
├── requirements.txt    # Project dependencies
└── README.md           # Instructions and documentation
```
