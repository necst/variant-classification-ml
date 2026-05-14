import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

RESULTS_DIR = "../experiments/performance_plots"
os.makedirs(RESULTS_DIR, exist_ok=True)

file_path = '../experiments/test_results.csv'

try:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    print(f"Error: '{file_path}' not found.")
    exit()

df['size'] = pd.to_numeric(df['size'], errors='coerce')
df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
df = df.sort_values(by=['model', 'DS', 'preprocessing', 'size'])

g = sns.relplot(
    data=df,
    x="size", 
    y="Accuracy",
    hue="preprocessing", 
    row="model",
    col="DS",
    kind="line",
    marker="o",
    height=3.5,
    aspect=1.2,
    facet_kws={'sharex': False, 'sharey': True} 
)

g.set(ylim=(0.55, 1))

for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

g.fig.subplots_adjust(top=0.9, right=0.82, wspace=0.2, hspace=0.3)
sns.move_legend(g, "upper left", bbox_to_anchor=(0.83, 0.98))

g.fig.suptitle("Accuracy vs Dataset Size", fontsize=16)
g.set_axis_labels("Dataset Size", "Accuracy")

plt.savefig(os.path.join(RESULTS_DIR, "accuracy_results.png"), dpi=150, bbox_inches='tight')
plt.close()