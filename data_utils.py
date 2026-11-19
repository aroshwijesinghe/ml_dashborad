import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_diabetes

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SL_JOBS_RAW_PATH = os.path.join(DATA_DIR, "sri_lanka_it_jobs_dataset.csv")
SL_JOBS_PROCESSED_PATH = os.path.join(DATA_DIR, "sl_it_jobs_cleaned.csv")


def load_sl_it_jobs_dataset(as_raw=False):
    """
    Loads the Sri Lanka IT Jobs dataset downloaded from Kaggle (miyurusanjana/sl-it-jobs-dataset).
    If as_raw=True, returns the original 209-row unparsed dataframe.
    If as_raw=False, returns the thoroughly cleaned 525-row tabular ML dataset.
    """
    if as_raw:
        if not os.path.exists(SL_JOBS_RAW_PATH):
            import kagglehub
            path = kagglehub.dataset_download("miyurusanjana/sl-it-jobs-dataset")
            raw_file = os.path.join(path, "sri_lanka_it_jobs_dataset.csv")
            os.makedirs(DATA_DIR, exist_ok=True)
            df_raw = pd.read_csv(raw_file)
            df_raw.to_csv(SL_JOBS_RAW_PATH, index=False)
            return df_raw
        return pd.read_csv(SL_JOBS_RAW_PATH)

    if os.path.exists(SL_JOBS_PROCESSED_PATH):
        return pd.read_csv(SL_JOBS_PROCESSED_PATH)

    from clean_dataset import clean_sl_it_jobs_dataset
    return clean_sl_it_jobs_dataset()


def get_sample_telecom_churn_data():
    """Generates a realistic synthetic telecom customer churn dataset."""
    np.random.seed(42)
    n_samples = 1000

    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 20, size=n_samples)
    total_charges = np.maximum(total_charges, 20.0)

    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.4, 0.45, 0.15])
    payment_method = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_samples)
    tech_support = np.random.choice(["Yes", "No", "No internet"], size=n_samples, p=[0.3, 0.55, 0.15])

    churn_prob = (
        0.3
        + (contract == "Month-to-month") * 0.25
        + (internet_service == "Fiber optic") * 0.15
        + (tech_support == "No") * 0.15
        - (tenure / 72.0) * 0.35
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    churn = (np.random.rand(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "TenureMonths": tenure,
        "MonthlyCharges": np.round(monthly_charges, 2),
        "TotalCharges": np.round(total_charges, 2),
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "TechSupport": tech_support,
        "Churn": churn
    })
    return df


def get_sample_patient_data():
    """Patient health data for regression."""
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={"target": "DiseaseProgression"}, inplace=True)
    return df


def preprocess_data(df, target_col, selected_features=None, task_type="Classification", test_size=0.2, random_state=42):
    """
    Cleans data, encodes categorical features, handles missing values,
    and returns train/test splits along with feature metadata.
    """
    data = df.copy()

    # Drop rows where target is missing
    data = data.dropna(subset=[target_col])

    if selected_features:
        cols_to_keep = [col for col in selected_features if col != target_col]
        X = data[cols_to_keep].copy()
    else:
        X = data.drop(columns=[target_col]).copy()

    y = data[target_col].copy()

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Impute missing values
    for col in numerical_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    for col in categorical_cols:
        if X[col].isnull().any():
            mode_val = X[col].mode()[0] if not X[col].mode().empty else "Unknown"
            X[col] = X[col].fillna(mode_val)

    # Encode categorical features for tree-based models
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Process target
    target_encoder = None
    if task_type == "Classification":
        if y.dtype == "object" or y.dtype.name == "category" or not np.issubdtype(y.dtype, np.number):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
        else:
            y = y.astype(int)
            unique_classes = np.unique(y)
            if not np.array_equal(unique_classes, np.arange(len(unique_classes))):
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y)
    else:
        y = y.astype(float)
        if y.isnull().any():
            y = y.fillna(y.median())

    # Split train and test
    stratify = y if task_type == "Classification" and len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": X.columns.tolist(),
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "label_encoders": label_encoders,
        "target_encoder": target_encoder,
        "num_classes": len(np.unique(y)) if task_type == "Classification" else None
    }
