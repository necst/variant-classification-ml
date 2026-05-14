import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import os

RESULTS_DIR = "experiments/shapley_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv("../experiments/shapleys.csv")


def plot_importance_by_ds_size(df, model, ds_classes, preprocessing):
    df_examine = df[
        (df["model"] == model)
        & (df["ds_classes"] == ds_classes)
        & (df["preprocessing"] == preprocessing)
    ].copy()
    df_examine["value"] = df_examine["value"].abs()

    global_importance = (
        df_examine.groupby(["ds_size", "feature"])["value"].mean().reset_index()
    )

    max_size = global_importance["ds_size"].max()

    top_features = (
        global_importance[global_importance["ds_size"] == max_size]
        .sort_values(by="value", ascending=False)
        .head(100)["feature"]
        .tolist()
    )

    plot_data = global_importance[
        global_importance["feature"].isin(top_features)
    ].copy()
    plot_data["ds_size"] = plot_data["ds_size"].astype(str)

    plt.figure(figsize=(12, 20))
    sns.barplot(
        data=plot_data,
        y="feature",
        x="value",
        hue="ds_size",
        order=top_features,
        palette="viridis",
    )
    plt.title(
        f"Feature Importance Evolution by Dataset Size of {model} for {ds_classes}"
    )
    plt.xlabel("Mean |SHAP Value| (Global Importance)")
    plt.ylabel("Feature")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.legend(title="Dataset Size")
    plt.tight_layout()
    filename = f"importance_by_ds_size_{model}_{ds_classes}_{preprocessing}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


def spearman_corr_sizes(df, model, ds_classes, preprocessing):
    df_examine = df[
        (df["model"] == model)
        & (df["ds_classes"] == ds_classes)
        & (df["preprocessing"] == preprocessing)
    ].copy()
    df_examine["value"] = df_examine["value"].abs()

    global_importance = (
        df_examine.groupby(["ds_size", "feature"])["value"].mean().reset_index()
    )
    pivot_df = global_importance.pivot(
        index="feature", columns="ds_size", values="value"
    )
    pivot_df.fillna(0, inplace=True)
    rank_df = pivot_df.rank(ascending=False, method="min")
    stability_matrix = rank_df.corr(method="spearman")

    plt.figure(figsize=(6, 5))
    plt.title(f"Feature importance correlation of {model} for {ds_classes}")
    sns.heatmap(stability_matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.tight_layout()
    filename = f"spearman_corr_sizes_{model}_{ds_classes}_{preprocessing}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


def spearman_corr_classes(df, model, ds_size, preprocessing):
    df_examine = df[
        (df["model"] == model)
        & (df["ds_size"] == ds_size)
        & (df["preprocessing"] == preprocessing)
    ].copy()
    df_examine["value"] = df_examine["value"].abs()

    global_importance = (
        df_examine.groupby(["ds_classes", "feature"])["value"].mean().reset_index()
    )
    pivot_df = global_importance.pivot(
        index="feature", columns="ds_classes", values="value"
    )
    pivot_df.fillna(0, inplace=True)
    rank_df = pivot_df.rank(ascending=False, method="min")
    correlation_matrix = rank_df.corr(method="spearman")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1, fmt=".2f"
    )
    plt.title(f"Feature Importance Consistency: {model} (Size: {ds_size})")
    plt.xlabel("Dataset Version / Classes")
    plt.ylabel("Dataset Version / Classes")
    plt.tight_layout()
    filename = f"spearman_corr_classes_{model}_{ds_size}_{preprocessing}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


def plot_relative_importance_variance(
    df, topn=30, model=None, ds_size=None, ds_classes=None, preprocessing=None
):
    df_filtered = df.copy()
    if model is not None:
        df_filtered = df_filtered[df_filtered["model"] == model]
    if ds_size is not None:
        df_filtered = df_filtered[df_filtered["ds_size"] == ds_size]
    if ds_classes is not None:
        df_filtered = df_filtered[df_filtered["ds_classes"] == ds_classes]
    if preprocessing is not None:
        df_filtered = df_filtered[df_filtered["preprocessing"] == preprocessing]

    group_cols = ["model", "ds_size", "ds_classes", "preprocessing"]
    config_importance = (
        df_filtered.groupby(group_cols + ["feature"])["value"]
        .apply(lambda x: x.abs().mean())
        .reset_index(name="mean_abs_shap")
    )
    config_importance["rank"] = config_importance.groupby(group_cols)[
        "mean_abs_shap"
    ].rank(ascending=False, method="min")
    config_importance["total_features"] = config_importance.groupby(group_cols)[
        "feature"
    ].transform("count")
    config_importance["relative_rank"] = (
        config_importance["rank"] / config_importance["total_features"]
    )

    feature_stats = (
        config_importance.groupby("feature")["relative_rank"]
        .median()
        .sort_values(ascending=True)
    )
    top_20_features = feature_stats.head(topn).index.tolist()
    plot_data = config_importance[config_importance["feature"].isin(top_20_features)]

    plt.figure(figsize=(12, 10))
    sns.boxplot(
        data=plot_data,
        x="relative_rank",
        y="feature",
        order=top_20_features,
        palette="viridis_r",
        showfliers=False,
        linewidth=1.2,
    )
    sns.stripplot(
        data=plot_data,
        x="relative_rank",
        y="feature",
        order=top_20_features,
        size=3,
        color=".3",
        alpha=0.6,
    )
    plt.title("Universal Feature Importance (Normalized by Model Size)")
    plt.xlabel("Relative Rank (0.0 = Top Feature, 1.0 = Worst Feature)")
    plt.ylabel("Feature")
    plt.axvline(x=0.1, color="green", linestyle="--", alpha=0.3, label="Top 10%")
    plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.3, label="Median")
    plt.legend(loc="lower right")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    filename = f"relative_importance_variance_top{topn}.png"
    if model:
        filename = f"relative_importance_variance_{model}_top{topn}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


def plot_configuration_stability(df):
    print("Computing ranks...")
    global_importance = (
        df.groupby(["model", "ds_size", "ds_classes", "preprocessing", "feature"])[
            "value"
        ]
        .apply(lambda x: x.abs().mean())
        .reset_index(name="score")
    )
    global_importance["rank"] = global_importance.groupby(
        ["model", "ds_size", "ds_classes", "preprocessing"]
    )["score"].rank(ascending=False, method="min")

    stability_scores = []
    group_cols = ["model", "preprocessing"]

    for (model, prep), group in global_importance.groupby(group_cols):
        class_stabilities = []
        for ds_class, subgroup in group.groupby("ds_classes"):
            pivot = subgroup.pivot(index="feature", columns="ds_size", values="rank")
            if pivot.shape[1] < 2:
                continue
            corr_mat = pivot.corr(method="spearman")
            unique_corrs = (
                corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
                .stack()
                .values
            )
            if len(unique_corrs) > 0:
                class_stabilities.append(np.mean(unique_corrs))
        if class_stabilities:
            final_avg_stability = np.mean(class_stabilities)
            stability_scores.append(
                {
                    "model": model,
                    "preprocessing": prep,
                    "label": f"{model}\n({prep})",
                    "avg_stability": final_avg_stability,
                }
            )

    plot_df = pd.DataFrame(stability_scores)
    plot_df = plot_df.sort_values(by="avg_stability", ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=plot_df,
        x="avg_stability",
        y="label",
        hue="model",
        dodge=False,
        palette="viridis",
    )
    plt.title(
        "Explanation Stability by Configuration\n(Avg Auto-Correlation across Dataset Sizes & Classes)"
    )
    plt.xlabel("Average Spearman Correlation (Higher = More Stable Logic)")
    plt.ylabel("Configuration")
    plt.xlim(0, 1.0)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.legend(title="Model Family", loc="lower right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "configuration_stability.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print("--- Most Stable Configurations ---")
    print(plot_df[["model", "preprocessing", "avg_stability"]].head())


def plot_inter_model_correlation(df):
    group_cols = ["model", "ds_size", "ds_classes", "preprocessing"]
    config_importance = (
        df.groupby(group_cols + ["feature"])["value"]
        .apply(lambda x: x.abs().mean())
        .reset_index(name="mean_abs_shap")
    )
    config_importance["rank"] = config_importance.groupby(group_cols)[
        "mean_abs_shap"
    ].rank(ascending=False, method="min")
    config_importance["total_features"] = config_importance.groupby(group_cols)[
        "feature"
    ].transform("count")
    config_importance["relative_rank"] = (
        config_importance["rank"] / config_importance["total_features"]
    )

    model_archetypes = (
        config_importance.groupby(["model", "feature"])["relative_rank"]
        .mean()
        .reset_index()
    )
    pivot_df = model_archetypes.pivot(
        index="feature", columns="model", values="relative_rank"
    )
    pivot_df.fillna(1.0, inplace=True)
    corr_matrix = pivot_df.corr(method="spearman")

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        fmt=".2f",
        square=True,
    )
    plt.title("Inter-Model Agreement\n(Correlation of Feature Rankings)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "inter_model_correlation.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    plot_importance_by_ds_size(df, "xgboost", "3c", "all_features")
    plot_relative_importance_variance(df, topn=50)
    spearman_corr_sizes(df, "mlp", "2c", "all_features")
    spearman_corr_classes(df, "mlp", 100000, "all_features")
    plot_inter_model_correlation(df)
    plot_configuration_stability(df)
