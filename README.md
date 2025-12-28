# msp-opv

A data-driven study for Material–Structure–Property (MSP) mapping in organic photovoltaics. The repository includes a 25k-sample physics-informed dataset, a compact feature set that links material parameters with microstructure descriptors, and a random-forest surrogate model for predicting short-circuit current.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
msp-opv/
├── data/
│   ├── data.parquet          # Main dataset (25k samples)
│   ├── feature_map.yaml      # Feature alias mapping
│   ├── sfs_linear.csv        # Sequential feature selection results (linear)
│   └── sfs_rf.csv            # Sequential feature selection results (RF)
├── scripts/                  # Python scripts (executable)
│   ├── linear.py             # Linear model (Lasso) with SFS
│   ├── rf.py                 # Random Forest with SFS
│   ├── model_with_confidence.py  # RF model with confidence estimation
│   ├── pdp.py                # Partial dependence plots
│   └── sfs_and_correlation_plot.py  # SFS comparison and correlation plots
├── notebooks/                # Jupyter notebooks (same functionality as scripts)
│   ├── linear.ipynb
│   ├── rf.ipynb
│   ├── model_with_confidence.ipynb
│   ├── pdp.ipynb
│   └── sfs_and_correlation_plot.ipynb
├── figures/                  # Generated plots and visualizations
├── requirements.txt          # Python dependencies
└── README.md
```

**Note:** The notebooks in `notebooks/` provide the same functionality as the scripts in `scripts/` but are provided for convenience and interactive exploration.

## Scripts Overview

### 1. `linear.py` / `linear.ipynb`
- Implements a Lasso regression model with sequential feature selection (SFS)
- Performs cross-validated forward feature selection
- Generates:
  - `sfs_linear.csv`: Feature selection results
  - `sfs_linear_curve.pdf`: CV R² vs. number of features
  - `sfs_linear_gains.pdf`: Marginal gains by feature

### 2. `rf.py` / `rf.ipynb`
- Implements a Random Forest regressor with sequential feature selection
- Uses cross-validation to evaluate feature importance
- Generates:
  - `sfs_rf.csv`: Feature selection results
  - `sfs_rf_curve.pdf`: CV R² vs. number of features
  - `sfs_rf_gains.pdf`: Marginal gains by feature

### 3. `model_with_confidence.py` / `model_with_confidence.ipynb`
- Trains a Random Forest model with confidence estimation
- Implements two confidence metrics:
  - **Ensemble**: Based on standard deviation across trees
  - **CI**: Based on 5-95% confidence intervals
- Generates:
  - KDE plots of confidence vs. absolute error
  - Prediction plots with confidence bands
  - Progressive training curves (R² vs. training fraction)

### 4. `pdp.py` / `pdp.ipynb`
- Computes and visualizes partial dependence plots (PDPs)
- Analyzes feature interactions using 2D PDPs
- Focuses on key features: `Ld` (exciton diffusion length), `min(c1, c2)` (carrier mobilities), `STAT_e` (interfacial area), and `CT_f_e_conn`
- Generates contour plots showing partial dependence of short-circuit current (J) on feature pairs

### 5. `sfs_and_correlation_plot.py` / `sfs_and_correlation_plot.ipynb`
- Compares SFS results between linear and RF models
- Generates correlation heatmaps of input features
- Creates side-by-side visualizations of:
  - SFS performance curves
  - Feature importance rankings
  - Feature correlation matrices

## Usage

### Running Scripts

All scripts should be run from the project root directory. They expect data files in `../data/` relative to the script location:

```bash
# From project root
cd scripts
python linear.py
python rf.py
python model_with_confidence.py
python pdp.py
python sfs_and_correlation_plot.py
```

### Data Requirements

The scripts expect:
- `data/data.parquet`: Main dataset with features and target variable `J` (short-circuit current)
- `data/feature_map.yaml`: Mapping between feature aliases (c1, c2, c3, d1-d21) and full feature names

### Key Features

The analysis focuses on:
- **Material parameters**: Carrier mobilities (c1, c2), recombination (c3)
- **Microstructure descriptors**: Statistical descriptors (d1-d21)
- **Derived features**: 
  - `Ld`: Exciton diffusion length (nm)
  - `min(c1, c2)`: Minimum of carrier mobilities
  - `STAT_e`: Normalized interfacial area
  - `CT_f_e_conn`: Charge transfer connectivity

### Output

All figures are saved to the `figures/` directory in PDF and/or PNG format. CSV results are saved to `data/`.

## Dependencies

- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `matplotlib`: Plotting
- `scikit-learn`: Machine learning models
- `PyYAML`: YAML file parsing
- `tqdm`: Progress bars
- `pyarrow`: Parquet file support
- `seaborn`: Statistical visualizations
- `scipy`: Scientific computing

See `requirements.txt` for the complete list.

## License

See `LICENSE` file for details.
