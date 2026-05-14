import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import os

RESULTS_DIR = "../experiments/shapley_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv("../experiments/shapleys.csv")


def plot_relative_importance_variance_column(
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
    top_n_features = feature_stats.head(topn).index.tolist()
    plot_data = config_importance[config_importance["feature"].isin(top_n_features)]

    plt.figure(figsize=(3.4, 2.8))

    # Ensure we only order by the number of features requested
    current_order = top_n_features[:topn]

    # 2. Plot Vertical Boxplot (Swapped x and y from original)
    sns.boxplot(
        data=plot_data,
        x="feature",  # Features on X-axis now
        y="relative_rank",  # Rank on Y-axis now
        order=current_order,
        palette="viridis_r",
        showfliers=False,
        linewidth=1.0,  # Slightly thinner lines for smaller plot
        width=0.6,  # Adjust box width for aesthetics
    )

    sns.stripplot(
        data=plot_data,
        x="feature",
        y="relative_rank",
        order=current_order,
        size=2.5,  # Smaller dots for smaller plot
        color=".3",
        alpha=0.6,
    )

    # 3. Formatting
    plt.title(f"Top {topn} Features by importance", fontsize=10)  # Shortened title
    plt.ylabel("Relative Rank\n(0.0 = Top Feature, 1.0 = Worst)", fontsize=9)
    plt.xlabel("")  # X-label is redundant since labels are feature names

    # 5. Rotate Feature Names (Inclined)
    plt.xticks(rotation=50, ha="right", fontsize=7)
    plt.yticks(fontsize=9)
    plt.ylim((0.0, 0.5))

    # Grid on Y-axis now
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # 6. Save as PDF
    filename = f"relative_importance_variance_top{topn}_column.pdf"
    if model:
        filename = f"relative_importance_variance_{model}_top{topn}_column.pdf"

    # bbox_inches="tight" is crucial here to prevent the rotated labels from being cut off
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_relative_importance_variance_column(df, topn=7)
