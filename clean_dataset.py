import os
import re
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_PATH = os.path.join(DATA_DIR, "sri_lanka_it_jobs_dataset.csv")
CLEANED_PATH = os.path.join(DATA_DIR, "sl_it_jobs_cleaned.csv")

def clean_sl_it_jobs_dataset(raw_df=None):
    """
    Cleans and transforms the raw Sri Lanka IT Jobs dataset into a high-quality,
    tabular dataset ready for ML models (Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost).
    
    Data Cleaning Pipeline:
    -----------------------
    1. Row Disaggregation:
       - The raw dataset packages multiple job levels (Internship, Entry, Mid, Senior) inside single cells
         delimited by '|' in salary and ',' in experience.
       - Each level is parsed into an individual, discrete observation.
    
    2. Text & Numeric Cleaning:
       - Strips whitespace and commas from salary figures ('120,000' -> 120000.0).
       - Removes '+' signs ('500,000+' -> 500000.0).
       - Computes average monthly salary in LKR as continuous target: (min + max) / 2.
    
    3. Experience Standardization:
       - Converts textual ranges ('0-2', '3-5', '6+') and labels ('Fresh graduate / Final year student')
         into standardized continuous numerical values (Experience_Years).
    
    4. Categorical Target Formulation:
       - Constructs standardized job seniority tiers ('Internship', 'Entry Level', 'Mid Level', 'Senior Level').
       - Creates structured salary classification brackets ('<50k', '50k-150k', '150k-300k', '300k-500k', '>500k').
    
    5. Feature Engineering:
       - Role domain classification (Software Engineering, Data Science & AI, Cloud & DevOps, QA, etc.).
       - Binary indicator flags for in-demand technologies (Python, JavaScript, Java, SQL, Cloud, AI/ML, Docker).
       - Total tech skills count and soft skills count.
    """
    if raw_df is None:
        if not os.path.exists(RAW_PATH):
            raise FileNotFoundError(f"Raw dataset not found at {RAW_PATH}")
        raw_df = pd.read_csv(RAW_PATH)
        
    cleaned_rows = []
    
    for idx, row in raw_df.iterrows():
        role = str(row.get("Role", "")).strip()
        soft = str(row.get("Soft Skills", "")).strip()
        tech = str(row.get("Tech Stack", "")).strip()
        exp_raw = str(row.get("Experience (Years)", "")).strip()
        sal_raw = str(row.get("Salary Range (LKR/Month)", "")).strip()

        # Step 1: Parse experience tiers mapping
        exp_tiers = {}
        if "fresh graduate" in exp_raw.lower() or "final year" in exp_raw.lower():
            exp_tiers["unpaid"] = 0.0
            exp_tiers["paid internship"] = 0.5
            exp_tiers["internship"] = 0.5
            exp_tiers["entry"] = 1.0
        else:
            for item in exp_raw.split(","):
                m = re.search(r"([\d\+\-]+)\s*\(([^)]+)\)", item.strip())
                if m:
                    range_str = m.group(1).strip()
                    tier_name = m.group(2).strip().lower()
                    if "-" in range_str:
                        lo, hi = range_str.split("-")
                        avg_e = (float(lo) + float(hi)) / 2.0
                    elif "+" in range_str:
                        avg_e = float(range_str.replace("+", "")) + 1.5
                    else:
                        try:
                            avg_e = float(range_str)
                        except Exception:
                            avg_e = 1.0
                    exp_tiers[tier_name] = avg_e

        # Step 2: Parse salary ranges per tier
        for part in sal_raw.split("|"):
            part = part.strip()
            m = re.search(r"([\d,]+)\s*-\s*([\d,]+)\+?\s*\(([^)]+)\)", part)
            if m:
                min_s = float(m.group(1).replace(",", ""))
                max_s = float(m.group(2).replace(",", ""))
                tier = m.group(3).strip()
                tier_l = tier.lower()
                avg_s = (min_s + max_s) / 2.0

                # Match standardized experience
                exp_val = exp_tiers.get(tier_l)
                if exp_val is None:
                    if "unpaid" in tier_l:
                        exp_val = 0.0
                    elif "intern" in tier_l:
                        exp_val = 0.5
                    elif "entry" in tier_l or "junior" in tier_l:
                        exp_val = 1.0
                    elif "mid" in tier_l:
                        exp_val = 4.0
                    elif "senior" in tier_l or "lead" in tier_l:
                        exp_val = 7.0
                    else:
                        exp_val = 2.0

                # Step 3: Standardize Job Level category
                if "unpaid" in tier_l or "intern" in tier_l:
                    clean_level = "Internship"
                elif "entry" in tier_l:
                    clean_level = "Entry Level"
                elif "mid" in tier_l:
                    clean_level = "Mid Level"
                elif "senior" in tier_l:
                    clean_level = "Senior Level"
                else:
                    clean_level = tier

                # Step 4: Structured Salary Bracket
                if avg_s <= 50000:
                    salary_bracket = "1. Intern (<50k)"
                elif avg_s <= 150000:
                    salary_bracket = "2. Entry (50k-150k)"
                elif avg_s <= 300000:
                    salary_bracket = "3. Mid (150k-300k)"
                elif avg_s <= 500000:
                    salary_bracket = "4. Senior (300k-500k)"
                else:
                    salary_bracket = "5. Lead/Executive (>500k)"

                # Step 5: Domain Classification
                rl = role.lower()
                if any(k in rl for k in ["data", "ai", "ml", "machine learning", "deep learning", "nlp", "bi", "analytics"]):
                    domain = "Data Science & AI"
                elif any(k in rl for k in ["devops", "cloud", "sre", "infrastructure", "sysadmin", "network"]):
                    domain = "Cloud & DevOps"
                elif any(k in rl for k in ["security", "cyber", "soc", "penetration", "infosec"]):
                    domain = "Cybersecurity"
                elif any(k in rl for k in ["qa", "test", "quality", "automation engineer"]):
                    domain = "QA & Testing"
                elif any(k in rl for k in ["ui", "ux", "product design", "graphic"]):
                    domain = "UI/UX & Design"
                elif any(k in rl for k in ["architect", "manager", "lead", "director", "cto", "scrum", "product owner"]):
                    domain = "Management & Architecture"
                else:
                    domain = "Software Engineering"

                tl = tech.lower()
                cleaned_rows.append({
                    "Role": role,
                    "Domain": domain,
                    "Job_Level": clean_level,
                    "Experience_Years": exp_val,
                    "Min_Salary_LKR": min_s,
                    "Max_Salary_LKR": max_s,
                    "Salary_LKR": avg_s,
                    "Salary_Bracket": salary_bracket,
                    "Tech_Count": len([x for x in tech.split(",") if x.strip()]),
                    "Soft_Count": len([x for x in soft.split(",") if x.strip()]),
                    "Has_Python": int("python" in tl),
                    "Has_JavaScript": int("javascript" in tl or "typescript" in tl or "node" in tl or "react" in tl),
                    "Has_Java": int("java" in tl and "javascript" not in tl or "java," in tl),
                    "Has_SQL": int("sql" in tl or "postgres" in tl or "mysql" in tl or "database" in tl),
                    "Has_Cloud": int("aws" in tl or "azure" in tl or "cloud" in tl or "gcp" in tl),
                    "Has_AI_ML": int("ml" in tl or "ai" in tl or "tensorflow" in tl or "pytorch" in tl),
                    "Has_Docker_K8s": int("docker" in tl or "kubernetes" in tl or "ci/cd" in tl)
                })

    clean_df = pd.DataFrame(cleaned_rows)
    os.makedirs(DATA_DIR, exist_ok=True)
    clean_df.to_csv(CLEANED_PATH, index=False)
    return clean_df

if __name__ == "__main__":
    df_clean = clean_sl_it_jobs_dataset()
    print(f"Dataset successfully cleaned and saved to: {CLEANED_PATH}")
    print(f"Cleaned dimensions: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")
