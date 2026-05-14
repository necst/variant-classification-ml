import os
import sys

# --- MUST BE SET BEFORE IMPORTING NUMPY/SHAP/SKLEARN ---
# Set this to the number of physical cores you want to use
cores = "32"

os.environ["OMP_NUM_THREADS"] = cores 
os.environ["OPENBLAS_NUM_THREADS"] = cores 
os.environ["MKL_NUM_THREADS"] = cores 
os.environ["VECLIB_MAXIMUM_THREADS"] = cores 
os.environ["NUMEXPR_NUM_THREADS"] = cores 

import shap
from pathlib import Path
import joblib
from omegaconf import OmegaConf
import pandas as pd
from load_preprocess import apply_preprocessing
import time
import numpy as np

outputs_dir = Path("outputs/")

df_results = []
for run_dir in outputs_dir.iterdir():
    if not run_dir.is_dir():
        continue

    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        continue

    print(f"Processing: {run_dir.name}")

    start_time = time.time()

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    model = joblib.load(model_path)
    le = joblib.load(run_dir / "label_encoder.joblib")

    scaler_path = run_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    imputer_path = run_dir / "imputer.joblib"
    imputer = joblib.load(imputer_path) if imputer_path.exists() else None

    if "5c" in run_dir.name:
        df = pd.read_csv("datasets_new/dbnfs_5c_test.csv", low_memory=False)
    elif "4c" in run_dir.name:
        df = pd.read_csv("datasets_new/dbnfs_4c_test.csv", low_memory=False)
    elif "3c" in run_dir.name:
        df = pd.read_csv("datasets_new/dbnfs_3c_test.csv", low_memory=False)
    elif "2c" in run_dir.name:
        df = pd.read_csv("datasets_new/dbnfs_2c_test.csv", low_memory=False)
    target_col = cfg.dataset.target_column

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # for speed reasons
    if len(df) > 10000:
        df = df[:10000]

    X = df.drop(columns=[target_col])
    X_processed = apply_preprocessing(X, cfg)

    if imputer:
        numeric_cols = X_processed.select_dtypes(include=["number"]).columns
        X_processed[numeric_cols] = imputer.transform(X_processed[numeric_cols])

    if scaler:
        X_scaled = scaler.transform(X_processed)
        X_processed = pd.DataFrame(
            X_scaled, columns=X_processed.columns, index=X_processed.index
        )

    shap_samples = 200
    X_test = X_processed[:shap_samples]
    X_train = X_processed[shap_samples:]

    X_train_summary = shap.kmeans(X_train, 100)
    feature_names = X_train.columns.tolist()
    print(f"Shap of {cfg.model.name}, using KernelExplainer")
    # --- THE FIX ---
    # We define a wrapper function. SHAP sees this as a "generic function"
    # and won't try to access/modify 'feature_names_in_' on the model object.
    def predict_wrapper(X_input):
        # SHAP usually passes numpy arrays here. 
        # We must convert back to DataFrame because XGBoost is picky about column names.
        if isinstance(X_input, np.ndarray):
            X_df = pd.DataFrame(X_input, columns=feature_names)
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X_df)
            else:
                return model.predict(X_df)
        # Handle case if it's already a DataFrame
        else:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X_input)
            else:
                return model.predict(X_input)
    explainer = shap.KernelExplainer(predict_wrapper, X_train_summary)

    print("Computing shapley values...")
    shap_values = explainer.shap_values(X_test)
    
    elapsed = int(time.time() - start_time)
    print(f"Computing took {int(elapsed/60)}m{elapsed%60}s, adding shapley values to previous results")

    local_dfs = []

    if isinstance(shap_values, list):
        for class_idx, class_values in enumerate(shap_values):
            temp_df = pd.DataFrame(class_values, columns=feature_names)
            temp_df['sample_index'] = temp_df.index
            temp_df['class'] = class_idx
            
            melted = temp_df.melt(
                id_vars=['sample_index', 'class'], 
                var_name='feature', 
                value_name='value'
            )
            local_dfs.append(melted)
    elif isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 3:
            n_classes = shap_values.shape[2]
            
            for class_idx in range(n_classes):
                class_slice = shap_values[:, :, class_idx]
                
                temp_df = pd.DataFrame(class_slice, columns=feature_names)
                temp_df['sample_index'] = temp_df.index
                temp_df['class'] = class_idx
                
                melted = temp_df.melt(
                    id_vars=['sample_index', 'class'], 
                    var_name='feature', 
                    value_name='value'
                )
                local_dfs.append(melted)
        else:
            temp_df = pd.DataFrame(shap_values, columns=feature_names)
            temp_df['sample_index'] = temp_df.index
            temp_df['class'] = 1 # Default for binary/regression
            
            melted = temp_df.melt(
                id_vars=['sample_index', 'class'], 
                var_name='feature', 
                value_name='value'
            )
            local_dfs.append(melted)

    model_df = pd.concat(local_dfs, ignore_index=True)

    model_df["model"] = cfg.model.name
    ds_name = cfg.dataset.name
    parts = ds_name.split("_")
    model_df["ds_size"] = int(parts[-1])
    model_df["ds_classes"] = parts[1]
    model_df["preprocessing"] = cfg.preprocessing.name

    df_results.append(model_df[["model", "ds_size", "ds_classes", "preprocessing", 
                            "sample_index", "feature", "class", "value"]])

final_csv = pd.concat(df_results, ignore_index=True)
final_csv.to_csv("experiments/shapleys.csv", index=False)