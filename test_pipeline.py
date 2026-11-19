from data_utils import load_sl_it_jobs_dataset, preprocess_data
from model_trainer import train_and_evaluate

def main():
    print("Testing Sri Lanka IT Jobs Dataset with Random Forest, XGBoost, LightGBM, and CatBoost...")
    df = load_sl_it_jobs_dataset(as_raw=False)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    models = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]
    
    print("\n--- 1. Classification (Target: Salary_Bracket) ---")
    prep_clf = preprocess_data(df, target_col="Salary_Bracket", task_type="Classification")
    for algo in models:
        res = train_and_evaluate(algo, prep_clf, task_type="Classification")
        print(f"  [OK] {algo:15s} | Accuracy: {res['metrics']['Accuracy']} | Latency: {res['training_time_sec']*1000:.1f}ms")

    print("\n--- 2. Regression (Target: Salary_LKR) ---")
    prep_reg = preprocess_data(df, target_col="Salary_LKR", task_type="Regression")
    for algo in models:
        res_reg = train_and_evaluate(algo, prep_reg, task_type="Regression")
        print(f"  [OK] {algo:15s} | R²: {res_reg['metrics']['R-squared (R²)']} | RMSE: {res_reg['metrics']['RMSE']}")

    print("\n🎉 All 4 ensemble algorithms successfully trained and verified!")

if __name__ == "__main__":
    main()
