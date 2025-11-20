# %%
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.ndimage import gaussian_filter
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
df['STAT_e'] = (df['STAT_e'] - df['STAT_e'].min()) / (df['STAT_e'].max() - df['STAT_e'].min())


# Compute Ld = sqrt(D * 10^c3)
D = 2.5e-7  # m^2/s
df['Ld'] = np.sqrt(D * 10**df[alias_to_feature['c3']])/1e-9 # Convert to nm
input_features.append('Ld')

df['min(c1, c2)'] = df[[alias_to_feature['c1'], alias_to_feature['c2']]].min(axis=1)
input_features.append('min(c1, c2)')

# input_features = ['log_n', 'log_p', 'STAT_e', 'CT_f_e_conn', 'Ld', 'min(c1, c2)']
input_features = ['Ld', 'min(c1, c2)', 'STAT_e', 'CT_f_e_conn']
# input_features = ['Ld', 'min(c1, c2)']

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

# check the r2 score on test set
r2 = r2_score(y_test, y_pred)
print(f"R^2 on test set: {r2:.4f}")

# %%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = False
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['font.family'] ='Times New Roman'

label_size = 20
tick_size = int(label_size * 0.9)
tick_style = dict(
    axis='both',
    which='both',
    direction='out',
    length=6,
    width=1.5,
    colors='black',
    bottom=True,
    left=True
)

grid_resolution = 50
from sklearn.inspection import partial_dependence

# Define features_all based on input_features
features_all = input_features

feature_labels = {
    'Ld': r'L$_d$ (nm)',
    'min(c1, c2)': r'$\min(c_1, c_2)$',
    'STAT_e': r'STAT$_e$',
    'CT_f_e_conn': r'CT$_{f,e,conn}$'
}


# pairs
pairs = [('Ld', 'min(c1, c2)'),
        #  ('CT_f_e_conn', 'min(c1, c2)'),
        #  ('Ld', 'STAT_e'),
        #  ('Ld', 'CT_f_e_conn'),
         ('STAT_e', 'min(c1, c2)'),
        #  ('STAT_e', 'CT_f_e_conn')
        ]

# First compute all PDPs and store results
pdp_results = {}

for f1, f2 in pairs:
    ix1 = features_all.index(f1)
    ix2 = features_all.index(f2)
    pdp = partial_dependence(
        rf,
        X=X_train,
        features=[(ix1, ix2)],
        grid_resolution=grid_resolution,
        kind='average'
    )
    pdp_results[(ix1, ix2)] = pdp

# Set fixed Jsc range and shared contour levels
vmin = 0.5
vmax = 6.5
levels = np.linspace(vmin, vmax, 13)


def plot_pdp_pair(ax, ix1, ix2, vmin, vmax):
    pdp = pdp_results[(ix1, ix2)]
    x_vals, y_vals = pdp['grid_values']
    xx, yy = np.meshgrid(x_vals, y_vals)
    zz = pdp['average'][0].T
    
    # Smooth the PDP values using Gaussian filter for smooth iso-contour lines
    zz_smooth = gaussian_filter(zz, sigma=1.0)
    
    # Use smoothed data for contour plots with consistent colorbar range
    cp = ax.contourf(xx, yy, zz_smooth, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.contour(xx, yy, zz_smooth, levels=levels, colors='black', linewidths=0.6)
    
    feature_x = features_all[ix1]
    feature_y = features_all[ix2]
    ax.set_xlabel(feature_labels[feature_x], fontsize=label_size)
    ax.set_ylabel(feature_labels[feature_y], fontsize=label_size)
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.tick_params(**tick_style)
    return cp

fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=150)
for ax, (f1, f2) in zip(axes.flat, pairs):
    ix1 = features_all.index(f1)
    ix2 = features_all.index(f2)
    cp = plot_pdp_pair(ax, ix1, ix2, vmin, vmax)
    cbar = fig.colorbar(cp, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
    cbar.set_ticks(levels)
    cbar.set_label(r'Partial Dependence of $J$ (mA/cm$^2$)', fontsize=label_size)
    cbar.ax.tick_params(labelsize=tick_size)

plt.tight_layout()
plt.savefig("../figures/pdp_selected.pdf", dpi=200, bbox_inches='tight')
plt.savefig("../figures/pdp_selected.png", dpi=200, bbox_inches='tight')
# plt.show()





