# %%
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=UserWarning)

import yaml
from tqdm import tqdm

# load the alias -> feature map
with open("../data/feature_map.yaml", "r") as f:
    alias_to_feature = yaml.safe_load(f)

# (optional) reverse map
feature_to_alias = {v: k for k, v in alias_to_feature.items()}

# pick aliases you want to use (start with d1, d2, d3, etc.)
aliases = ["c1", "c2", "c3", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
           "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21"]

# resolve to real feature names
input_features = [alias_to_feature[a] for a in aliases]

# load data and select columns
df = pd.read_parquet("../data/data.parquet", engine="pyarrow")

# -----------------------------
# Load CSVs
# -----------------------------
df_lin = pd.read_csv("../data/sfs_linear.csv").sort_values("k")
df_rf  = pd.read_csv("../data/sfs_rf.csv").sort_values("k")

# (optional) mapping real feature names to aliases
# assumes feature_to_alias already defined in your notebook
df_lin["feature_alias"] = df_lin["feature"].map(feature_to_alias)
df_rf["feature_alias"]  = df_rf["feature"].map(feature_to_alias)

# -----------------------------
# Figure configuration
# -----------------------------
fontsize = 22
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': fontsize})

fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=160)
plt.subplots_adjust(wspace=0.35)

# ==========================================================
# (a) SFS performance curves
# ==========================================================
ax = axes[0]

ks_lin = np.arange(1, len(df_lin["cv_r2_at_k"]) + 1)
ks_rf  = np.arange(1, len(df_rf["cv_r2_at_k"]) + 1)

ax.plot(ks_lin, df_lin["cv_r2_at_k"], marker="o", linewidth=2.5,
        alpha=0.9, label="Linear (Lasso)")
ax.plot(ks_rf,  df_rf["cv_r2_at_k"],  marker="s", linewidth=2.5,
        alpha=0.9, label="Random Forest")

ax.set_xlabel("Number of selected features (k)")
ax.set_ylabel("Cross-validated $R^2$")
ax.legend()
# ax.grid(alpha=0.3)
# ax.set_title("(a) SFS performance curves", loc="left", fontsize=fontsize)

# ==========================================================
# (b) Linear feature gains
# ==========================================================
ax = axes[1]
df_lin_sorted = df_lin.sort_values("k").head(10)
ax.barh(df_lin_sorted["feature_alias"], df_lin_sorted["gain"], alpha=0.85, color="tab:blue")

ax.set_xlabel("Gain in CV $R^2$")
ax.set_ylabel("Features (selection order)")
ax.invert_yaxis()
ax.set_xlim(left=min(0, df_lin_sorted["gain"].min()) * 1.1,
            right=max(0, df_lin_sorted["gain"].max()) * 1.15)

# Annotate gains beside bars
for i, (gain, feature) in enumerate(zip(df_lin_sorted["gain"], df_lin_sorted["feature_alias"])):
    ax.text(gain + 0.002, i, f"{gain:.3f}", va="center", ha="left", fontsize=fontsize - 4)

# ax.set_title("Linear model (Lasso)", loc="center", fontsize=fontsize)
ax.title.set_position([0.5, -0.3])  # Adjust the position to move the title below the plot

# ==========================================================
# (c) RF feature gains
# ==========================================================
ax = axes[2]
df_rf_sorted = df_rf.sort_values("k").head(10)
ax.barh(df_rf_sorted["feature_alias"], df_rf_sorted["gain"], alpha=0.85, color="tab:orange")

ax.set_xlabel("Gain in CV $R^2$")
ax.set_ylabel("Features (selection order)")
ax.invert_yaxis()
ax.set_xlim(left=min(0, df_rf_sorted["gain"].min()) * 1.1,
            right=max(0, df_rf_sorted["gain"].max()) * 1.15)

# Annotate gains beside bars
for i, (gain, feature) in enumerate(zip(df_rf_sorted["gain"], df_rf_sorted["feature_alias"])):
    ax.text(gain + 0.002, i, f"{gain:.3f}", va="center", ha="left", fontsize=fontsize - 4)

# ax.set_title("(c) Random Forest model", loc="left", fontsize=fontsize)

fig.text(0.19, -0.05, "(a) SFS performance curve", ha='center', fontsize=fontsize)
fig.text(0.52, -0.05, "(b) Feature importantace (Linear)", ha='center', fontsize=fontsize)
fig.text(0.85, -0.05, "(c) Feature importance (RF)", ha='center', fontsize=fontsize)

# ==========================================================
# Save figure
# ==========================================================
OUT_FIG = "../figures/sfs.pdf"
plt.tight_layout()
fig.savefig(OUT_FIG, bbox_inches="tight", dpi=600)
print(f"Saved figure -> {OUT_FIG}")


# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

# -----------------------------
# Load feature map (alias -> full name)
# -----------------------------
with open("../data/feature_map.yaml", "r") as f:
    alias_to_feature = yaml.safe_load(f)
feature_to_alias = {v: k for k, v in alias_to_feature.items()}

# -----------------------------
# Load data
# -----------------------------
df = pd.read_parquet("../data/data.parquet", engine="pyarrow")

# Select features that exist in the dataframe
input_features = [f for f in alias_to_feature.values() if f in df.columns]
df_sub = df[input_features].copy()

# -----------------------------
# Compute correlation matrix
# -----------------------------
corr = df_sub.corr(method="pearson")

# Rename rows/columns using aliases
corr_alias = corr.rename(index=feature_to_alias, columns=feature_to_alias)

# -----------------------------
# Plot heatmap
# -----------------------------
fontsize = 18
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': fontsize})

fig, ax = plt.subplots(figsize=(15, 12), dpi=160)

sns.heatmap(
    corr_alias,
    annot=True,
    cmap="vlag",
    fmt=".2f",
    annot_kws={"size": fontsize - 6},
    square=True,
    ax=ax,
    cbar_kws={"shrink": 0.99}
)

# ax.set_title("Correlation matrix of input features (alias view)", fontsize=fontsize + 2, pad=16)
plt.xticks(rotation=0, fontsize=fontsize - 2)
plt.yticks(rotation=0, fontsize=fontsize - 2)
plt.tight_layout()

out_fig = "../figures/heatmap.pdf"
fig.savefig(out_fig, bbox_inches="tight", dpi=600)
print(f"Saved figure -> {out_fig}")


# %%
# -----------------------------
# Select top 6 RF features
# -----------------------------
import seaborn as sns
top_rf_features = df_rf_sorted["feature"].head(5).tolist()
fontsize = 20

# Subset the dataframe with top RF features
df_top_rf = df[top_rf_features]

# Compute correlation matrix for top RF features
corr_top_rf = df_top_rf.corr(method="pearson")

# Rename rows/columns using aliases
corr_top_rf_alias = corr_top_rf.rename(index=feature_to_alias, columns=feature_to_alias)

# -----------------------------
# Plot heatmap
# -----------------------------
fig, ax = plt.subplots(figsize=(5, 4), dpi=160)

sns.heatmap(
    corr_top_rf_alias,
    annot=True,
    cmap="vlag",
    fmt=".2f",
    annot_kws={"size": fontsize - 6},
    square=True,
    ax=ax,
    cbar_kws={"shrink": 0.8}
)

plt.xticks(rotation=0, fontsize=fontsize - 2)
plt.yticks(rotation=0, fontsize=fontsize - 2)
plt.tight_layout()

out_fig_top_rf = "../figures/top_rf_heatmap.pdf"
fig.savefig(out_fig_top_rf, bbox_inches="tight", dpi=600)
print(f"Saved figure -> {out_fig_top_rf}")


