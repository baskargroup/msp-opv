# %%
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score



warnings.filterwarnings("ignore", category=UserWarning)

import yaml
from tqdm import tqdm

# load the alias -> feature map
with open("../data/feature_map.yaml", "r") as f:
    alias_to_feature = yaml.safe_load(f)

# (optional) reverse map
feature_to_alias = {v: k for k, v in alias_to_feature.items()}

# pick aliases you want to use (start with d1, d2, d3, etc.)
aliases = ["c1", "c2", "c3", "d1", "d12"]

# resolve to real feature names
input_features = [alias_to_feature[a] for a in aliases]

# load data and select columns
df = pd.read_parquet("../data/data.parquet", engine="pyarrow")

target = 'J'

X = df[input_features]
y = df[target]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train Random Forest
# -----------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# %%
# changge seed
np.random.seed(45)
confidence_type = "ci"  # options: "ci" or "ensemble"
# -----------------------------
# Evaluate Model
# -----------------------------
y_test_vals = y_test.values
y_pred_vals = y_pred.flatten()
mse = mean_squared_error(y_test_vals, y_pred_vals)
r2 = r2_score(y_test_vals, y_pred_vals)
print(f"Random Forest MSE: {mse:.4f}, R^2: {r2:.4f}")

# -----------------------------
# Confidence Estimation
# -----------------------------
all_preds = np.array([tree.predict(X_test) for tree in rf.estimators_])

if confidence_type == "ensemble":
    std_rf = all_preds.std(axis=0)
    confidence = 1 - (std_rf / std_rf.max())
elif confidence_type == "ci":
    ci_lower = np.percentile(all_preds, 5, axis=0)
    ci_upper = np.percentile(all_preds, 95, axis=0)
    ci_width = ci_upper - ci_lower
    confidence = 1 - (ci_width / ci_width.max())
else:
    raise ValueError("Invalid confidence_type. Choose 'ensemble' or 'ci'.")

# -----------------------------
# Results
# -----------------------------
results_df = pd.DataFrame({
    'GT': y_test_vals,
    'Prediction': y_pred_vals,
    'Confidence': confidence,
    'Absolute_Error': np.abs(y_test_vals - y_pred_vals),
    'Percent_Error': np.abs((y_test_vals - y_pred_vals) / y_test_vals) * 100
})

if confidence_type == "ci":
    results_df['CI_Lower'] = ci_lower
    results_df['CI_Upper'] = ci_upper
    results_df['CI_Width'] = ci_width
    results_df['Within_CI'] = np.logical_and(y_test_vals >= ci_lower, y_test_vals <= ci_upper)

print(results_df.head(10))

# %%
results_df['Absolute_Error'].min(), results_df['Absolute_Error'].max(), results_df['Absolute_Error'].mean(), results_df['Absolute_Error'].std()

# %%
# # -----------------------------
# # Visualization
# # -----------------------------
# # plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.family"] = "Times New Roman"
# fontsize = 30
# plt.figure(figsize=(8, 8))
# scatter = plt.scatter(y_test_vals, y_pred_vals, c=confidence, cmap='viridis', s=50, alpha=0.7)
# plt.plot([min(y_test_vals), max(y_test_vals)],
#          [min(y_test_vals), max(y_test_vals)], 'r--', label='Ideal Prediction')
# plt.xlabel('True J', fontsize=fontsize)
# # reduce padding between x label with ticks
# # plt.gca().xaxis.set_label_coords(0.5, -0.09)
# plt.ylabel('Predicted J', fontsize=fontsize)
# cbar = plt.colorbar(scatter)
# cbar.set_label(f'Confidence ({confidence_type})', fontsize=fontsize)
# plt.xticks(fontsize=fontsize - 5)
# plt.yticks(fontsize=fontsize - 5)
# plt.legend(fontsize=fontsize - 5)
# plt.tight_layout()
# plt.savefig(f'./figures/rf_confidence_{confidence_type}_scatter.png', dpi=300)
# plt.show()

# %%
import seaborn as sns

# -----------------------------
# Confidence vs. Absolute Error: KDE Plot
# -----------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(8, 8), dpi=50)
fontsize = 35

# 2D KDE plot
sns.kdeplot(
    x=confidence,
    y=results_df['Absolute_Error'],
    fill=True,  # filled contours
    cmap="viridis",
    thresh=0.1,  # only plot above 5% of max density
    levels=100,
)



# Labels and aesthetics
plt.xlabel(f'Confidence ({confidence_type})', fontsize=fontsize)
plt.ylabel('Absolute Error', fontsize=fontsize)
plt.xticks(fontsize=fontsize - 5)
plt.yticks(fontsize=fontsize - 5)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
# plt.xlim(0.5, 1)
# plt.ylim(-0.2, 0.5)
# plt.legend(fontsize=fontsize - 10)
plt.tight_layout()
plt.grid(False)
plt.savefig(f'../figures/kde.png', dpi=300)
plt.show()


# %%
# -----------------------------
# Smoothed Prediction vs. Ground Truth with Shaded CI Band
# -----------------------------
from scipy.stats import binned_statistic
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = False
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['font.family'] ='Times New Roman'

plt.figure(figsize=(8.5, 8), dpi=50)
fontsize = 35

# Sort by true J for smooth plotting
sorted_idx = np.argsort(y_test_vals)
x_sorted = y_test_vals[sorted_idx]
y_sorted = y_pred_vals[sorted_idx]
ci_lower_sorted = ci_lower[sorted_idx]
ci_upper_sorted = ci_upper[sorted_idx]

# Optionally smooth with bins (optional: remove if unnecessary)
n_bins = 20
bin_means, bin_edges, _ = binned_statistic(x_sorted, y_sorted, statistic='mean', bins=n_bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


ci_lower_binned, _, _ = binned_statistic(x_sorted, ci_lower_sorted, statistic='mean', bins=n_bins)
ci_upper_binned, _, _ = binned_statistic(x_sorted, ci_upper_sorted, statistic='mean', bins=n_bins)

# Fill between CI bounds
plt.fill_between(
    bin_centers, 
    ci_lower_binned, 
    ci_upper_binned, 
    color='gray',
    alpha=0.6, 
    label='5-95% CI band'
)

# Plot mean predicted value line
plt.plot(bin_centers, bin_means, color='blue', label='Mean Prediction', linewidth=2)

# Ideal y=x line
plt.plot(bin_centers, bin_centers, 'r--', label='Ideal (y = x)', linewidth=2)

# Labels and formatting
# set xlabel(r'Partial Dependence of $J$ (mA/cm$^2$)', fontsize=tick_size-2)
plt.xlabel(r'True J (mA/cm$^2$)', fontsize=fontsize)
plt.gca().xaxis.set_label_coords(0.5, -0.07)
plt.ylabel(r'Predicted J (mA/cm$^2$)', fontsize=fontsize)


plt.xticks(fontsize=fontsize - 5)
plt.yticks(fontsize=fontsize - 5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=fontsize - 8)
plt.tight_layout()
# overlay the scatter plot
plt.scatter(
    y_test_vals, 
    y_pred_vals, 
    c=confidence, 
    cmap='viridis',
    # cmap='binary',
    s=15, 
    alpha=0.8, 
    # edgecolor='k', 
    linewidth=0.5,
)

# add r2 score to the plot
r2_text = r'R$^2$ = %.3f' % r2
plt.text(
    0.60, 0.15, r2_text,
    fontsize=fontsize - 5,
    ha='left', va='top',
    transform=plt.gca().transAxes,
    # bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
)

# confidence colorbar
cbar = plt.colorbar()
cbar.set_label(f'Confidence ({confidence_type})', fontsize=fontsize)
# cbar tick params
cbar.ax.tick_params(labelsize=fontsize-5)
# grid off
plt.grid(False)
plt.tight_layout()
plt.savefig(f'../figures/rf_prediction_ci_band_smooth.pdf', dpi=300)

plt.show()


# %%
from sklearn.linear_model import Lasso
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from tqdm import tqdm

# ============================== helper functions ==============================
import numpy as np

def _fractions(start_frac=0.01, step_frac=0.04, max_frac=0.99, 
               scale="linear", num=50, split=0.1):
    """
    Generate fractions for progressive sampling.

    Parameters
    ----------
    start_frac : float
        Minimum training fraction.
    step_frac : float
        Step size (only used if scale='linear').
    max_frac : float
        Maximum training fraction.
    scale : str, optional
        'linear', 'log', or 'semi-log'.
    num : int, optional
        Number of points (for log/semi-log spacing).
    split : float, optional
        Split point for semi-log mode. Fractions <= split use log spacing,
        and > split to max_frac use linear spacing.
    """
    if scale == "linear":
        fracs, f = [], start_frac
        while f < max_frac - 1e-9:  # stop before max_frac
            fracs.append(round(f, 4))
            f += step_frac
        if not np.isclose(fracs[-1], max_frac):
            fracs.append(max_frac)
        return fracs

    elif scale == "log":
        fracs = np.logspace(np.log10(start_frac), np.log10(max_frac), num)
        return np.round(fracs, 4).tolist()

    elif scale == "semi-log":
        # logspace from start_frac to split
        n1 = num // 2
        n2 = num - n1
        fracs_log = np.logspace(np.log10(start_frac), np.log10(split), n1, endpoint=False)
        fracs_lin = np.linspace(split, max_frac, n2)
        fracs = np.unique(np.concatenate([fracs_log, fracs_lin]))
        return np.round(fracs, 4).tolist()

    else:
        raise ValueError("scale must be 'linear', 'log', or 'semi-log'")


def _progressive_core(X, y, model_maker, fracs, n_repeats=10, seed=42):
    mean_train, std_train, mean_test, std_test = [], [], [], []
    if fracs is None:
        print("Using default linear fractions from 0.01 to 0.99 with step 0.04")
        fracs = _fractions(start_frac=0.01, step_frac=0.04, max_frac=0.99, scale="linear")

    for frac in tqdm(fracs):
        r2_tr_runs, r2_te_runs = [], []
        for r in range(n_repeats):
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, train_size=frac, random_state=seed + 100*r
            )
            model = model_maker(seed)
            model.fit(Xtr, ytr)
            yhat_tr = model.predict(Xtr)
            yhat_te = model.predict(Xte)
            r2_tr_runs.append(r2_score(ytr, yhat_tr))
            r2_te_runs.append(r2_score(yte, yhat_te))

        mean_train.append(np.mean(r2_tr_runs)); std_train.append(np.std(r2_tr_runs))
        mean_test.append(np.mean(r2_te_runs));  std_test.append(np.std(r2_te_runs))
        # if test score is above 0.99 break
        if mean_test[-1] > 0.999:
            print(f"Reached R2 > 0.99 at fraction {frac}, stopping early.")
            break

    return {
        "fractions": fracs,
        "train_mean": mean_train, "train_std": std_train,
        "test_mean":  mean_test,  "test_std":  std_test
    }

# -----------------------------
# 1) Random Forest curve
# -----------------------------
def progressive_rf(
    X, y, fracs=None,
    n_repeats=10, seed=42,
    rf_params=None
):
    if rf_params is None:
        rf_params = dict(n_estimators=100, max_depth=None, n_jobs=-1)

    def make_model(seed_):
        return RandomForestRegressor(random_state=seed_, **rf_params)

    return _progressive_core(X, y, make_model, fracs=fracs, n_repeats=n_repeats, seed=seed)


# ==============================
# Config / Inputs
# ==============================
K = 7
RANDOM_STATE = 42


# %%

start_frac = 0.001
step_frac = 0.001
max_frac = 0.95
num = 30
scale = "log"
n_repeats = 10

fracs = _fractions(start_frac, step_frac, max_frac, scale, num)
print(len(fracs), fracs)

# %%
# Progressive training for Random Forest
progressive_results = progressive_rf(
    X, y, fracs=fracs, n_repeats=n_repeats, seed=RANDOM_STATE
)

# Extract results
fractions = progressive_results["fractions"]
train_mean = progressive_results["train_mean"]
train_std = progressive_results["train_std"]
test_mean = progressive_results["test_mean"]
test_std = progressive_results["test_std"]

# %%
# Plot train and test R² scores
fontsize = 35
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = fontsize

plt.figure(figsize=(8, 8))
plt.plot(fractions[:len(train_mean)], train_mean, label="Train R²", color="blue", marker='o')
plt.fill_between(
    fractions[:len(train_mean)],
    np.array(train_mean) - np.array(train_std),
    np.array(train_mean) + np.array(train_std),
    color="blue",
    alpha=0.2,
)
plt.plot(fractions[:len(test_mean)], test_mean, label="Test R²", color="orange", marker='o')
plt.fill_between(
    fractions[:len(test_mean)],
    np.array(test_mean) - np.array(test_std),
    np.array(test_mean) + np.array(test_std),
    color="orange",
    alpha=0.2,
)
# draw a dottend line at y = 0.99
plt.axhline(y=0.99, color='k', linestyle='--', linewidth=1)
plt.text(0.2, 0.95, 'R² = 0.99', color='k', fontsize=fontsize-15)

# Add labels and legend
plt.xlabel("Training Fraction", fontsize=fontsize)
plt.ylabel("R² Score", fontsize=fontsize)
# plt.title("Progressive Training: Train vs Test R² Scores", fontsize=fontsize)
# three x ticks
plt.xscale("log")
plt.xticks([0.001, 0.01, 0.1, 1], labels=[r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'], fontsize=fontsize-5)
plt.yticks(fontsize=fontsize - 5)
plt.legend(fontsize=fontsize - 5)
# plt.grid(True, linestyle="--", alpha=0.6)
# log scale for x-axis
plt.tight_layout()
plt.savefig(f'../figures/progressive.pdf', dpi=300)
plt.show()


