import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import combinations
import numpy as np
import os


RESULTS_DIR = "experiments/uncertain_predictions_comparison" 

MODELS = {
    "XGBoost_2c_fullfeatures": "outputs/xgboost_dbnfs_2c_bal_full_218142_all_features/uncertain_predictions.csv",
    "XGBoost_4c_fullfeatures": "outputs/xgboost_dbnfs_4c_bal_full_123932_all_features/uncertain_predictions.csv",
    "RandFor_2c_fullfeatures": "outputs/randfor_dbnfs_2c_bal_full_218142_all_features/uncertain_predictions.csv",
    "RandFor_4c_fullfeatures": "outputs/randfor_dbnfs_4c_bal_full_123932_all_features/uncertain_predictions.csv"
}

def load_and_merge_predictions(models_dict):
    merged_df = None
    print(f"Loading {len(models_dict)} models...")

    for name, path in models_dict.items():
        if not os.path.exists(path):
            print(f"  [WARNING] File not found: {path}")
            continue

        try:
            df = pd.read_csv(path, usecols=["original_index", "predicted_label"])
            df = df.rename(columns={"predicted_label": name})
            df = df.set_index("original_index")
            
            if merged_df is None:
                merged_df = df
            else:
                merged_df = merged_df.join(df, how="inner")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    if merged_df is not None:
        print(f"Successfully aligned {len(merged_df)} variants across {len(merged_df.columns)} models.")
    
    return merged_df

def plot_overall_agreement(df):
    models = df.columns
    n = len(models)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            agreement = (df[models[i]] == df[models[j]]).mean()
            matrix[i, j] = agreement

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2%", cmap="Greens", 
                xticklabels=models, yticklabels=models, vmin=0.5, vmax=1.0)
    plt.title("Overall Model Agreement % (Strict)")
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, "overall_agreement.png")
    plt.savefig(save_path)
    plt.close() 
    print(f"Saved: {save_path}")

def plot_relaxed_agreement(df):
    mapping = {
        "Likely benign": "Benign",
        "Likely pathogenic": "Pathogenic"
    }
    
    df_mapped = df.replace(mapping)
    
    models = df_mapped.columns
    n = len(models)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            agreement = (df_mapped[models[i]] == df_mapped[models[j]]).mean()
            matrix[i, j] = agreement

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2%", cmap="Blues", 
                xticklabels=models, yticklabels=models, vmin=0.5, vmax=1.0)
    plt.title("Overall Model Agreement % (Relaxed: Likely X == X)")
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, "relaxed_agreement.png")
    plt.savefig(save_path)
    plt.close() 
    print(f"Saved: {save_path}")

def plot_pairwise_matrix(df, model_a, model_b, labels):
    cm = confusion_matrix(df[model_a], df[model_b], labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    
    plt.ylabel(f"{model_a} Predictions")
    plt.xlabel(f"{model_b} Predictions")
    plt.title(f"Concordance: {model_a} vs {model_b}")
    plt.tight_layout()

    filename = f"concordance_{model_a}_vs_{model_b}.png"
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Output directory created: {RESULTS_DIR}")

    df_compare = load_and_merge_predictions(MODELS)
    
    if df_compare is None or df_compare.empty:
        print("No overlapping data found.")
        return

    unique_labels = sorted(pd.unique(df_compare.values.ravel()))
    print(f"Classes found: {unique_labels}")

    plot_overall_agreement(df_compare)
    plot_relaxed_agreement(df_compare)

    for model_a, model_b in combinations(df_compare.columns, 2):
        plot_pairwise_matrix(df_compare, model_a, model_b, unique_labels)
    
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()