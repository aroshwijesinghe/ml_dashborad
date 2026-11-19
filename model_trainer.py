import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
)

def get_model_instance(algorithm_name, task_type="Classification", hyperparameters=None):
    """
    Returns an initialized model instance based on algorithm name, task type, and hyperparams.
    """
    params = hyperparameters or {}
    
    if algorithm_name == "Random Forest":
        n_estimators = params.get("n_estimators", 100)
        max_depth = params.get("max_depth", None)
        if task_type == "Classification":
            return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        else:
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)

    elif algorithm_name == "XGBoost":
        n_estimators = params.get("n_estimators", 100)
        max_depth = params.get("max_depth", None)
        if task_type == "Classification":
            return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        else:
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)

    elif algorithm_name == "XGBoost":
        n_estimators = params.get("n_estimators", 100)
        max_depth = params.get("max_depth", 6)
        learning_rate = params.get("learning_rate", 0.1)
        if task_type == "Classification":
            return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, eval_metric="logloss")
        else:
            return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)

    elif algorithm_name == "LightGBM":
        n_estimators = params.get("n_estimators", 100)
        max_depth = params.get("max_depth", -1)
        learning_rate = params.get("learning_rate", 0.1)
        if task_type == "Classification":
            return LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, verbose=-1)
        else:
            return LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, verbose=-1)

    elif algorithm_name == "CatBoost":
        iterations = params.get("n_estimators", 100)
        depth = params.get("max_depth", 6)
        learning_rate = params.get("learning_rate", 0.1)
        if task_type == "Classification":
            return CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=learning_rate, random_seed=42, verbose=0)
        else:
            return CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=learning_rate, random_seed=42, verbose=0)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def train_and_evaluate(algorithm_name, preprocessed_data, task_type="Classification", hyperparameters=None):
    """
    Trains the chosen model and computes evaluation metrics, feature importances, and predictions.
    """
    X_train = preprocessed_data["X_train"]
    X_test = preprocessed_data["X_test"]
    y_train = preprocessed_data["y_train"]
    y_test = preprocessed_data["y_test"]
    feature_names = preprocessed_data["feature_names"]
    
    model = get_model_instance(algorithm_name, task_type=task_type, hyperparameters=hyperparameters)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    
    # Feature importances extraction
    feature_importance_df = pd.DataFrame()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    elif hasattr(model, "get_feature_importance"):
        importances = model.get_feature_importance()
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        
    results = {
        "algorithm": algorithm_name,
        "task_type": task_type,
        "training_time_sec": round(training_time, 4),
        "model": model,
        "feature_importance": feature_importance_df,
        "y_pred": y_pred,
        "y_test": y_test
    }
    
    if task_type == "Classification":
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        auc = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
            except Exception:
                auc = None
                
        results["metrics"] = {
            "Accuracy": round(float(acc), 4),
            "Precision (Weighted)": round(float(prec), 4),
            "Recall (Weighted)": round(float(rec), 4),
            "F1-Score (Weighted)": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4) if auc is not None else "N/A"
        }
        results["confusion_matrix"] = cm
        results["classification_report"] = report
        
    else:  # Regression
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        try:
            mape = mean_absolute_percentage_error(y_test, y_pred)
        except Exception:
            mape = None
            
        results["metrics"] = {
            "R-squared (R²)": round(float(r2), 4),
            "RMSE": round(float(rmse), 4),
            "MAE": round(float(mae), 4),
            "MAPE": round(float(mape), 4) if mape is not None else "N/A"
        }
        
    return results
