"""
Variant Classification Pipeline with Hydra Configuration
"""

import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from load_preprocess import load_and_preprocess_data


def save_results(
    cfg: DictConfig, report: str, accuracy: float, y_test, y_pred, class_names: list, model, scaler, label_encoder, imputer
):
    """
    Saves the experiment results into the run's dedicated output directory.
    """

    output_dir = HydraConfig.get().runtime.output_dir
    print(f"\n--- Saving Results to Hydra's output directory: {output_dir} ---")

    # 1. Save the combined configuration and results report into one file.
    report_path = os.path.join(output_dir, "experiment_report.txt")
    with open(report_path, "w") as f:
        f.write("--- CONFIGURATION USED FOR THIS RUN ---\n\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n\n--- MODEL PERFORMANCE ---\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Condensed experiment report saved to: {report_path}")

    # 2. Calculate, plot, and save the confusion matrix as a separate image file.
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    cm = confusion_matrix(y_test, y_pred)

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
    plt.close()  # Close the plot to free up memory
    print(f"Confusion matrix plot saved to: {cm_path}")

    if cfg.training.save_model:
        print(f">>> [Flag is TRUE] Saving Model and Artifacts to {output_dir}...")
        
        joblib.dump(model, os.path.join(output_dir, "model.joblib"))
        
        if scaler is not None:
            joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
        
        joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.joblib"))

        if imputer is not None:
            joblib.dump(imputer, os.path.join(output_dir, "imputer.joblib"))
        
        print(">>> Save complete.")
    else:
        print(">>> [Flag is FALSE] Model was NOT saved.")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point with Hydra configuration."""

    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler, imputer = load_and_preprocess_data(
        cfg
    )

    if X_train is None:
        print("Halting execution due to data loading failure.")
        return

    print(f"\nInstantiating model: {cfg.model._target_}")

    model_cfg = cfg.model.copy()
    if "sweep_params" in model_cfg:
        del model_cfg.sweep_params

    # Special handling for MLP: build hidden_layer_sizes from n_hidden_layers and hidden_layer_size
    if cfg.model._target_ == "sklearn.neural_network.MLPClassifier":
        if "n_hidden_layers" in model_cfg and "hidden_layer_size" in model_cfg:
            n_layers = model_cfg.n_hidden_layers
            layer_size = model_cfg.hidden_layer_size
            model_cfg.hidden_layer_sizes = tuple([layer_size] * n_layers)
            del model_cfg.n_hidden_layers
            del model_cfg.hidden_layer_size
            print(
                f"Using MLP with {n_layers} hidden layers of size {layer_size}: {model_cfg.hidden_layer_sizes}"
            )

    # Special handling for LogReg
    if cfg.model._target_ == "sklearn.linear_model.LogisticRegression":
        if cfg.model.class_weight == "null":
            cfg.model.class_weight = None
        if cfg.model.penalty == "null":
            cfg.model.penalty = None

    # Special handling for SVM
    if cfg.model._target_ == "sklearn.svm.SVC":
        if cfg.model.class_weight == "null":
            cfg.model.class_weight = None

    del model_cfg.name


    model = instantiate(model_cfg)
    print(f"Model instantiated: {model}\n\n")

    print("--- Training Model ---")
    # Special handling for xgboost early stop
    if X_val is not None and y_val is not None:
        print(f"Training with early stopping")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=200
        )
    else:
        model.fit(X_train, y_train)
    print("Model training complete.")
    print("----------------------\n\n")

    print("--- Evaluating Model ---")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    class_names = label_encoder.classes_.astype(str)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("------------------------")

    save_results(cfg, report, accuracy, y_test, y_pred, class_names, model, scaler, label_encoder, imputer)

    return f1


if __name__ == "__main__":
    main()
