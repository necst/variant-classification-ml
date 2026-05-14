import matplotlib

matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
    }
)

RESULTS_DIR = "../experiments/uncertain_predictions_comparison"

MODELS = {
    "MLP_2c_drop_raw_metadata_keep_clinical_context": "../outputs/mlp_dbnfs_2c_bal_full_218142_drop_raw_metadata_keep_clinical_context/uncertain_predictions.csv",
    "LogReg_4c_drop_raw_metadata_and_labs": "../outputs/logreg_dbnfs_4c_bal_full_123932_drop_raw_metadata_and_labs/uncertain_predictions.csv",
    "XGBoost_2c_10k_all_features": "../outputs/xgboost_dbnfs_2c_bal_10k_10000_all_features/uncertain_predictions.csv",
    "RandFor_4c_100k_all_features": "../outputs/randfor_dbnfs_4c_bal_100k_100000_all_features/uncertain_predictions.csv",
}


def abbreviate_model_name(name):
    arch = (
        "MLP"
        if "MLP" in name
        else "LR"
        if "LogReg" in name
        else "XGB"
        if "XGBoost" in name
        else "RF"
    )
    size = "-100k" if "100k" in name else "-10k" if "10k" in name else ""
    config = (
        "-NR"
        if "drop_raw_metadata_and_labs" in name
        else "-CC"
        if "drop_raw_metadata_keep_clinical_context" in name
        else "-Full"
    )
    return f"{arch}{size}{config}"


def load_data(models_dict):
    merged_df = None
    for name, path in models_dict.items():
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=["original_index", "predicted_label"])
        df = df.rename(columns={"predicted_label": name}).set_index("original_index")
        merged_df = df if merged_df is None else merged_df.join(df, how="inner")
    return merged_df


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_compare = load_data(MODELS)
    if df_compare is None:
        return

    mapping = {"Likely benign": "Benign", "Likely pathogenic": "Pathogenic"}
    df_relaxed = df_compare.replace(mapping)
    df_strict = df_compare

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), constrained_layout=True)

    for i, ax in enumerate(axes):
        if i < 2:
            df = df_strict if i == 0 else df_relaxed
            title = "Strict Agreement" if i == 0 else "Relaxed Agreement"

            models = df.columns
            n = len(models)
            matrix = np.zeros((n, n))
            for r in range(n):
                for c in range(n):
                    matrix[r, c] = (df[models[r]] == df[models[c]]).mean()

            labels = [abbreviate_model_name(m) for m in models]
            matrix = matrix * 100
            annot_labels = np.vectorize(lambda x: f"{x:.0f}%")(matrix)
            sns.heatmap(
                matrix,
                fmt="",
                annot=annot_labels,
                annot_kws={"size": 6, "weight": "bold"},
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                vmin=40,
                vmax=100,
                cbar=(i == 1),
                cbar_kws={"label": "%"} if i == 1 else {},
                linewidths=0,
                square=True,
                ax=ax,
            )

            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_title(title, fontsize=10)
        else:
            # --- PLOT 3: FEATURE STABILITY (Replaces Placeholder) ---
            ax = axes[2]

            # 1. PREPARE DATA (Local calculation)
            # Assuming your raw dataframe variable is named 'test_results'
            df_stab = pd.read_csv("../experiments/shapleys.csv")
            # Calculate Ranks
            global_importance = (
                df_stab.groupby(
                    ["model", "ds_size", "ds_classes", "preprocessing", "feature"]
                )["value"]
                .apply(lambda x: x.abs().mean())
                .reset_index(name="score")
            )
            global_importance["rank"] = global_importance.groupby(
                ["model", "ds_size", "ds_classes", "preprocessing"]
            )["score"].rank(ascending=False, method="min")

            # Calculate Spearman Correlations
            stability_scores = []
            group_cols = ["model", "preprocessing"]

            for (model, prep), group in global_importance.groupby(group_cols):
                class_stabilities = []
                for ds_class, subgroup in group.groupby("ds_classes"):
                    pivot = subgroup.pivot(
                        index="feature", columns="ds_size", values="rank"
                    )
                    if pivot.shape[1] < 2:
                        continue
                    corr_mat = pivot.corr(method="spearman")
                    # Get upper triangle off-diagonals
                    unique_corrs = (
                        corr_mat.where(
                            np.triu(np.ones(corr_mat.shape), k=1).astype(bool)
                        )
                        .stack()
                        .values
                    )
                    if len(unique_corrs) > 0:
                        class_stabilities.append(np.mean(unique_corrs))
                if class_stabilities:
                    stability_scores.append(
                        {
                            "model": model,
                            "preprocessing": prep,
                            "score": np.mean(class_stabilities),
                        }
                    )

            plot_df = pd.DataFrame(stability_scores)

            # 2. CONFIGURATION
            model_order = ["logreg", "randfor", "xgboost", "mlp"]
            model_labels = ["LogReg", "RF", "XGB", "MLP"]
            model_colors = {
                "logreg": "#2ca02c",
                "randfor": "#ff7f0e",
                "xgboost": "#1f77b4",
                "mlp": "#d62728",
            }
            # Order and Alphas from your snippet
            prep_order = [
                "drop_raw_metadata_and_labs",  # Lightest (0.6)
                "drop_raw_metadata_keep_clinical_context",  # Medium (0.8)
                "all_features",  # Darkest (1.0)
            ]
            prep_alphas = {
                "all_features": 1.0,
                "drop_raw_metadata_keep_clinical_context": 0.8,
                "drop_raw_metadata_and_labs": 0.6,
            }

            # 3. PLOTTING
            x_indices = np.arange(len(model_order))
            bar_width = 0.25

            for i, model_name in enumerate(model_order):
                base_color = model_colors[model_name]
                for j, prep_name in enumerate(prep_order):
                    val = plot_df[
                        (plot_df["model"] == model_name)
                        & (plot_df["preprocessing"] == prep_name)
                    ]["score"].values

                    if len(val) > 0:
                        score = val[0]
                        # Center the group of 3 bars around the tick
                        # j=0 -> -0.25, j=1 -> 0, j=2 -> +0.25
                        pos = i + (j - 1) * bar_width

                        ax.bar(
                            pos,
                            score,
                            bar_width,
                            color=base_color,
                            alpha=prep_alphas[prep_name],
                            edgecolor="black",
                            linewidth=0.5,
                        )

            # 4. FORMATTING
            ax.set_title("Rank stability vs Features", fontsize=10)
            ax.set_ylabel("Spearman coeff.", fontsize=9)
            # Only set x-label if it doesn't crowd the bottom; usually 'Model' is implied
            ax.set_xlabel("Model", fontsize=9)

            ax.set_xticks(x_indices)
            ax.set_xticklabels(model_labels, fontsize=9)
            ax.set_ylim(0, 1.10)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
    labels = ["a)", "b)", "c)"]
    for ax, label in zip(axes, labels):
        ax.text(
            -0.15,
            1.10,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="right",
        )

    base_path = os.path.join(RESULTS_DIR, "combined_results")
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
