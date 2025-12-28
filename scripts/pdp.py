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

ix_ld = features_all.index('Ld')
ix_min = features_all.index('min(c1, c2)')  # Correcting 'min(c2,c3)' to 'min(c1, c2)'

fig, ax = plt.subplots(figsize=(8, 8), dpi=50)

# --- PDP: Ld vs min(c2,c3) ---
pdp_ld_min = partial_dependence(
    rf,
    X=X_train,
    features=[(ix_ld, ix_min)],
    grid_resolution=grid_resolution,
    kind='average'
)
x_vals, y_vals = pdp_ld_min['grid_values']
xx, yy = np.meshgrid(x_vals, y_vals)
zz = pdp_ld_min['average'][0].T

cp = ax.contourf(xx, yy, zz, levels=10, cmap='viridis')
contours = ax.contour(xx, yy, zz, levels=10, colors='black', linewidths=0.6)
# Removed contour labels

# ax.set_xlabel('Ld (nm)', fontsize=label_size)
# ax.set_ylabel('min(c2,c3)', fontsize=label_size)
ax.set_xlabel(r'L$_d$ (nm)', fontsize=label_size)
# unit for min(c2,c3) is m^2/Vs
ax.set_ylabel(r'$\min(c_2, c_3)$', fontsize=label_size)


ax.set_yticks(np.arange(-7, -2.9, 1))
ax.tick_params(axis='both', labelsize=tick_size)
ax.tick_params(**tick_style)

# Colorbar
cbar = fig.colorbar(cp, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
cbar.set_label(r'Partial Dependence of $J$ (mA/cm$^2$)', fontsize=label_size)
cbar.ax.tick_params(labelsize=tick_size)
# plt.title("L", fontsize=label_size)
plt.savefig("../figures/pdp_ld_min.png", dpi=600, bbox_inches='tight')
plt.savefig("../figures/pdp_ld_min.pdf", dpi=600, bbox_inches='tight')
# plt.show()



