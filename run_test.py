import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from load_preprocess import apply_preprocessing
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
import csv

def main():
    outputs_dir = Path("outputs")

    test_csv_rows = []

    for run_dir in outputs_dir.iterdir():
        
        if not run_dir.is_dir():
            continue
        
        model_path = run_dir / "model.joblib"
        if not model_path.exists():
            continue

        print(f"Processing: {run_dir.name}")

        try:
            cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

            model = joblib.load(model_path)
            le = joblib.load(run_dir / "label_encoder.joblib")
            
            scaler_path = run_dir / "scaler.joblib"
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None

            imputer_path = run_dir / "imputer.joblib"
            imputer = joblib.load(imputer_path) if imputer_path.exists() else None

            if "5c" in run_dir.name:
                df_test = pd.read_csv("datasets_new/dbnfs_5c_test.csv", low_memory=False)
                df_unc = None
            elif "4c" in run_dir.name:
                df_test = pd.read_csv("datasets_new/dbnfs_4c_test.csv", low_memory=False)
                df_unc = pd.read_csv("datasets_new/dbnfs_uncertain.csv")
            elif "3c" in run_dir.name:
                df_test = pd.read_csv("datasets_new/dbnfs_3c_test.csv", low_memory=False)
                df_unc = None
            elif "2c" in run_dir.name:
                df_test = pd.read_csv("datasets_new/dbnfs_2c_test.csv", low_memory=False)
                df_unc = pd.read_csv("datasets_new/dbnfs_uncertain.csv")
            
            target_col = cfg.dataset.target_column
            y_test = le.transform(df_test[target_col])
            X = df_test.drop(columns=[target_col])

            X_processed = apply_preprocessing(X, cfg)

            if imputer:
                numeric_cols = X_processed.select_dtypes(include=["number"]).columns
                X_processed[numeric_cols] = imputer.transform(X_processed[numeric_cols])

            if scaler:
                X_scaled = scaler.transform(X_processed)
                X_processed = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)

            y_pred = model.predict(X_processed)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            cm_path = Path(run_dir / "test_confusion_matrix.png")
            cm = confusion_matrix(y_test, y_pred)
            class_names = le.classes_.astype(str)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="g",  # Use integer formatting for annotations
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
            )
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title("Confusion Matrix")

            plt.savefig(cm_path)
            plt.close()

            dataset = cfg.dataset.name
            ds_parts = dataset.split("_")
            ds_size = int(ds_parts[-1])
            ds_classes = ds_parts[1]

            test_csv_rows.append({
                "model": cfg.model.name,
                "DS": ds_classes,
                "size": ds_size,
                "preprocessing": cfg.preprocessing.name,
                "F1": f1,
                "Accuracy": accuracy
            })

            if df_unc is not None:
                print(f"Testing {run_dir.name} on uncertains")
                target_col = cfg.dataset.target_column
                X = df_unc.drop(columns=[target_col])
                X_processed = apply_preprocessing(X, cfg)

                if imputer:
                    numeric_cols = X_processed.select_dtypes(include=["number"]).columns
                    X_processed[numeric_cols] = imputer.transform(X_processed[numeric_cols])
                
                if scaler:
                    X_scaled = scaler.transform(X_processed)
                    X_processed = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)

                preds = model.predict(X_processed)
                probs = model.predict_proba(X_processed)
                results = pd.DataFrame({
                    "original_index": df_unc.index,
                    "predicted_label": le.inverse_transform(preds),
                    "confidence": np.max(probs, axis=1)
                })

                output_file = run_dir / "uncertain_predictions.csv"
                results.to_csv(output_file, index=False)

        except Exception as e:
            print(f"  -> Error: {e}")
    
    fieldnames = ["model", "DS", "size", "preprocessing", "F1", "Accuracy"]
    with open("experiments/test_results.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_csv_rows)
    
    print("Wrote test results to experiments/test_results.csv")

if __name__ == "__main__":
    main()