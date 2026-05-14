import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


test_results = pd.read_csv("../experiments/test_results.csv")


def get_best_for_model(df, model):
    mod_df = df[df["model"] == model]
    sizes = ["2c", "3c", "4c", "5c"]
    perfs = []
    for s in sizes:
        df_s = mod_df[mod_df["DS"] == s]
        f1 = df_s["F1"].max()
        perfs.append(f1)
    return perfs


xgb_f1 = get_best_for_model(test_results, "xgboost")
rnf_f1 = get_best_for_model(test_results, "randfor")
log_f1 = get_best_for_model(test_results, "logreg")
mlp_f1 = get_best_for_model(test_results, "mlp")


def get_avg_drop_for_model(df, model):
    mod_df = df[df["model"] == model]
    sizes = ["2c", "3c", "4c", "5c"]
    preprocessing = [
        "all_features",
        "drop_raw_metadata_keep_clinical_context",
        "drop_raw_metadata_and_labs",
    ]
    dps = []
    for s in sizes:
        df_s = mod_df[mod_df["DS"] == s]
        dropoffs = []
        for p in preprocessing:
            df_p = df_s[df_s["preprocessing"] == p]
            dropoff = 100 * (df_p["F1"].max() - df_p["F1"].min()) / df_p["F1"].max()
            dropoffs.append(dropoff)
        dps.append(sum(dropoffs) / len(dropoffs))
    dps_no10k = []
    for s in sizes:
        df_s = mod_df[mod_df["DS"] == s]
        df_s = df_s[df_s["size"] > 10000]
        dropoffs = []
        for p in preprocessing:
            df_p = df_s[df_s["preprocessing"] == p]
            dropoff = 100 * (df_p["F1"].max() - df_p["F1"].min()) / df_p["F1"].max()
            dropoffs.append(dropoff)
        dps_no10k.append(sum(dropoffs) / len(dropoffs))
    return dps, dps_no10k


xgb_size, xgb_size_no10k = get_avg_drop_for_model(test_results, "xgboost")
rnf_size, rnf_size_no10k = get_avg_drop_for_model(test_results, "randfor")
log_size, log_size_no10k = get_avg_drop_for_model(test_results, "logreg")
mlp_size, mlp_size_no10k = get_avg_drop_for_model(test_results, "mlp")


def get_avg_drop_for_model_feat(df, model):
    mod_df = df[df["model"] == model]
    sizes = ["2c", "3c", "4c", "5c"]
    dps = []
    for s in sizes:
        df_s = mod_df[mod_df["DS"] == s]
        train_sizes = df_s["size"].unique()
        dropoffs = []
        for ts in train_sizes:
            df_ts = df_s[df_s["size"] == ts]
            dropoff = 100 * (df_ts["F1"].max() - df_ts["F1"].min()) / df_ts["F1"].max()
            dropoffs.append(dropoff)
        dps.append(sum(dropoffs) / len(dropoffs))

    dps_sub = []
    for s in sizes:
        df_s = mod_df[mod_df["DS"] == s]
        df_s = df_s[df_s["preprocessing"] != "drop_raw_metadata_and_labs"]
        train_sizes = df_s["size"].unique()
        dropoffs = []
        for ts in train_sizes:
            df_ts = df_s[df_s["size"] == ts]
            dropoff = 100 * (df_ts["F1"].max() - df_ts["F1"].min()) / df_ts["F1"].max()
            dropoffs.append(dropoff)
        dps_sub.append(sum(dropoffs) / len(dropoffs))
    return dps, dps_sub


xgb_feat, xgb_feat_noleast = get_avg_drop_for_model_feat(test_results, "xgboost")
rnf_feat, rnf_feat_noleast = get_avg_drop_for_model_feat(test_results, "randfor")
log_feat, log_feat_noleast = get_avg_drop_for_model_feat(test_results, "logreg")
mlp_feat, mlp_feat_noleast = get_avg_drop_for_model_feat(test_results, "mlp")


x_axis = [2, 3, 4, 5]
x_labels = ["2c", "3c", "4c", "5c"]

# --- 1. DATA PREPARATION ---
model_config = {
    "XGB": {
        "color": "#1f77b4",
        "f1": xgb_f1,
        "size": (xgb_size, xgb_size_no10k),
        "feat": (xgb_feat, xgb_feat_noleast),
    },
    "RF": {
        "color": "#ff7f0e",
        "f1": rnf_f1,
        "size": (rnf_size, rnf_size_no10k),
        "feat": (rnf_feat, rnf_feat_noleast),
    },
    "LogReg": {
        "color": "#2ca02c",
        "f1": log_f1,
        "size": (log_size, log_size_no10k),
        "feat": (log_feat, log_feat_noleast),
    },
    "MLP": {
        "color": "#d62728",
        "f1": mlp_f1,
        "size": (mlp_size, mlp_size_no10k),
        "feat": (mlp_feat, mlp_feat_noleast),
    },
}

x_labels = ["2c", "3c", "4c", "5c"]
x_indices = np.arange(len(x_labels))
bar_width = 0.18

# --- 2. FIGURE SETUP ---
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), constrained_layout=True)

# --- 3. PLOT 1: LINE CHART ---
ax0 = axes[0]
markers = {"XGB": "o", "RF": "s", "LogReg": "^", "MLP": "d"}

for name, config in model_config.items():
    ax0.plot(
        x_indices,
        config["f1"],
        marker=markers[name],
        linestyle="-",
        linewidth=1.5,
        markersize=5,
        color=config["color"],
        label=name,
    )

ax0.set_title("Best F1 Performance", fontsize=10)
ax0.set_ylabel("F1 Score", fontsize=9)
ax0.set_ylim(0.65, 1.05)
ax0.grid(True, linestyle="--", alpha=0.5)


# --- 4. HELPER FOR BAR CHARTS ---
def plot_overlay_bars(ax, data_key, y_label):
    model_names = list(model_config.keys())
    for i, name in enumerate(model_names):
        config = model_config[name]
        main_data, sub_data = config[data_key]
        pos = x_indices + (i - 1.5) * bar_width

        # Main Bar (Darker)
        ax.bar(
            pos,
            main_data,
            bar_width,
            color=config["color"],
            edgecolor="black",
            linewidth=0.5,
        )
        # Subset Bar (Lighter Overlay)
        ax.bar(pos, sub_data, bar_width, color="white", alpha=0.6, edgecolor="none")

    ax.set_ylabel(y_label, fontsize=9)
    ax.set_ylim(0, 19)
    ax.grid(axis="y", linestyle="--", alpha=0.5)


# --- 5. PLOT 2 & 3 ---
plot_overlay_bars(axes[1], "size", "Avg Drop (%)")
axes[1].set_title("Stability vs Train Size", fontsize=10)

plot_overlay_bars(axes[2], "feat", "Avg Drop (%)")
axes[2].set_title("Stability vs Features", fontsize=10)

# --- 6. COMMON FORMATTING ---
for ax in axes:
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Classes", fontsize=9)

# --- 7. ADD SUBPLOT LABELS (a, b, c) ---
labels = ["a)", "b)", "c)"]
for ax, label in zip(axes, labels):
    # transform=ax.transAxes makes (0,0) bottom-left and (1,1) top-right of the axis
    # (-0.15, 1.05) places the label to the left and slightly above the top corner
    ax.text(
        -0.15,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="right",
    )

# --- 8. LEGEND & SAVE ---
handles = [
    plt.Line2D([0], [0], color=model_config[m]["color"], lw=2, label=m)
    for m in model_config
]

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=4,
    frameon=False,
    fontsize=9,
)

plt.savefig("../experiments/combined_performance.pdf", dpi=300, bbox_inches="tight")
plt.show()
