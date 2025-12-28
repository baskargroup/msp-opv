# %%
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

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
X = df[input_features].to_numpy()
y = df["J"].to_numpy().flatten()

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "../data/data.parquet"    # <- change if needed
ENGINE = "pyarrow"
OUT_DIR_FIG = "../figures"
OUT_CSV = "../data/sfs_linear.csv"
OUT_FIG_CURVE = os.path.join(OUT_DIR_FIG, "sfs_linear_curve.pdf")
OUT_FIG_BARS  = os.path.join(OUT_DIR_FIG, "sfs_linear_gains.pdf")

os.makedirs(OUT_DIR_FIG, exist_ok=True)

np.random.seed(42)
CV_FOLDS = 5
TEST_SIZE = 0.2

# Choose your features/target

target_feature = "J"

# Estimator (linear pipeline)
estimator = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(alpha=4e-4, random_state=42, max_iter=5000))
])

# %%
# -----------------------------
# Helpers
# -----------------------------
def run_sfs(X, y, feat_names, estimator, k_max, cv_splits):
    """
    Greedy forward SFS for a single estimator.
    Returns dict with order, scores, gains, rows (tidy records).
    """
    n_features = X.shape[1]
    K = min(k_max, n_features)

    remaining = list(range(n_features))
    selected = []
    order_names, scores_at_k, gains = [], [], []

    prev_score = 0.0
    rows = []

    for k in tqdm(range(1, K + 1)):
        best_feat, best_score = None, -np.inf

        # Try adding each remaining feature and evaluate by CV R^2
        for j in remaining:
            cols = selected + [j]
            X_sub = X[:, cols]
            cv_scores = cross_val_score(
                estimator, X_sub, y, scoring="r2", cv=cv_splits, n_jobs=-1
            )
            mean_score = float(np.mean(cv_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_feat = j

        # Commit best feature
        selected.append(best_feat)
        remaining.remove(best_feat)

        gain_k = best_score - prev_score
        feat_name = feat_names[best_feat]

        order_names.append(feat_name)
        scores_at_k.append(best_score)
        gains.append(gain_k)

        rows.append({
            "k": k,
            "feature": feat_name,
            "cv_r2_at_k": best_score,
            "gain": gain_k
        })

        prev_score = best_score

    return {
        "order": order_names,
        "scores": scores_at_k,
        "gains": gains,
        "rows": rows
    }



# %%
# Optional holdout split (not used in SFS CV, but retained for future use)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42
)

# CV splitter
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

# Run SFS
print("Running SFS (linear model)...")
res = run_sfs(X_train, y_train, input_features, estimator,
                k_max=len(input_features), cv_splits=kf)

# Save CSV (tidy/long)
df_sfs = pd.DataFrame(res["rows"])
df_sfs.to_csv(OUT_CSV, index=False)
print(f"Saved CSV -> {OUT_CSV}")


# %%
# load the csv again
df_sfs = pd.read_csv(OUT_CSV)

# -------------------------
# Plot 1: k vs CV R^2 curve
# -------------------------
fontsize = 22
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': fontsize})

ks = np.arange(1, len(df_sfs["cv_r2_at_k"]) + 1)
fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
ax.plot(ks, df_sfs["cv_r2_at_k"], marker="o", linewidth=2, alpha=0.9,
        label="linear")
ax.set_xlabel("Number of selected features (k)")
ax.set_ylabel("Cross-validated $R^2$")



# ax.set_xticks(ks)
ax.legend()
plt.tight_layout()
fig.savefig(OUT_FIG_CURVE, bbox_inches="tight", dpi=600)
print(f"Saved figure -> {OUT_FIG_CURVE}")

# --------------------------------------------
# Plot 2: Bar plot of marginal gains by feature
# --------------------------------------------
plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(figsize=(8, 6), dpi=140)

# Ensure features appear in selection order
df_sfs_sorted = df_sfs.sort_values("k")
df_sfs_sorted["feature"] = df_sfs_sorted["feature"].map(feature_to_alias)
ax.barh(df_sfs_sorted["feature"][:10], df_sfs_sorted["gain"][:10], alpha=0.85)
ax.set_xlabel("Gain in CV $R^2$")
ax.set_ylabel("Features (selection order)")
ax.invert_yaxis()  # top to bottom = earliest to latest
ax.set_xlim(left=min(0, df_sfs["gain"].min()) * 1.1, right=max(0, df_sfs["gain"].max()) * 1.15)

# Annotate the gain beside each bar
for i, (gain, feature) in enumerate(zip(df_sfs_sorted["gain"][:10], df_sfs_sorted["feature"][:10])):
    ax.text(gain + 0.005, i, f"{gain:.3f}", va='center', ha='left', fontsize=fontsize - 4)
    
plt.tight_layout()
fig.savefig(OUT_FIG_BARS, bbox_inches="tight", dpi=600)
print(f"Saved figure -> {OUT_FIG_BARS}")


