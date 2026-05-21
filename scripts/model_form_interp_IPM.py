import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import mapie
print(mapie.__version__)

# from mapie.regression import MapieRegressor
from mapie.regression import SplitConformalRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# -------------------------
# Set random seed
# -------------------------

SEED = 42
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

input_file = Path.cwd() / "images_pointsensors_pulse25X_v4/pointsensors_mavm.csv"
output_path = Path.cwd() / "ipm_interp"
output_path.mkdir(parents=True, exist_ok=True)

d_values = pd.read_csv(input_file, index_col=0)
d_types = list(d_values.index)

coords_numpy = np.array([
    [0.0116, -0.0245,  0.0194],
    [0.0138, -0.0245,  0.0013],
    [0.0067, -0.0245,  0.0120],
    [0.0110,  0.0245,  0.0031],
    [-0.0105, 0.0245, -0.0050],
    [-0.0058, 0.0245,  0.0171],
    [-0.0180, -0.0006, 0.0164],
    [-0.0180, -0.0040,-0.0085],
    [-0.0180, -0.0047, 0.0073],
    [-0.0180,  0.0124,-0.0032]
])

coords = pd.DataFrame(coords_numpy,
                      columns=["x", "y", "z"],
                      index=["TC1", "TC2", "TC3", "TC4", "TC5",
                             "TC6", "TC7", "TC8", "TC9", "TC10"])

# Merge dataframes
merged_df = coords.join(d_values.T, how='inner')

# -----------------------------------------------------------------------------
# Loop through each d_type
# -----------------------------------------------------------------------------

for d_type in d_types:
    print(80 * "-")
    print(f"Training IPM model for {d_type} ...")

    # Training data
    X_train = merged_df[['x', 'y', 'z']].values
    y_train = merged_df[d_type].values

    # Normalize inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # -----------------------------------------------------------------------------
    # Define Interval Predictor Model (IPM)
    # -----------------------------------------------------------------------------
    base_model = RandomForestRegressor(n_estimators=500, random_state=SEED)
    # ipm = MapieRegressor(estimator=base_model, method="plus", cv="prefit", alpha=0.05)

    ipm = SplitConformalRegressor(estimator=base_model, confidence_level=0.95, prefit=True)

    # Fit base model first
    base_model.fit(X_train_scaled, y_train)
    # Fit IPM on top
    # ipm.fit(X_train_scaled, y_train)
    ipm.conformalize(X_train_scaled, y_train)

    # -----------------------------------------------------------------------------
    # Predict on training points
    # -----------------------------------------------------------------------------
    y_pred, y_pis = ipm.predict_interval(X_train_scaled, allow_infinite_bounds=True)  # 95% PI
    lower_train = y_pis[:, 0].squeeze()
    upper_train = y_pis[:, 1].squeeze()

    # y_pred = ipm.predict(X_train_scaled)  # 95% PI

    pred_df = pd.DataFrame({
        "Measured": y_train,
        "Predicted": y_pred,
        "Lower95": lower_train,
        "Upper95": upper_train
    })

    # pred_df = pd.DataFrame({
    #     "Measured": y_train,
    #     "Predicted": y_pred
    # })

    print(pred_df)

    # -----------------------------------------------------------------------------
    # Create interpolation grid
    # -----------------------------------------------------------------------------
    x_grid = np.linspace(X_train[:,0].min(), X_train[:,0].max(), 75)
    y_grid = np.linspace(X_train[:,1].min(), X_train[:,1].max(), 75)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    Z_fixed = (35 - 15/2 - 5) * 1e-3  # fixed Z plane

    X_query = np.column_stack((
        Xg.ravel(),
        Yg.ravel(),
        np.full_like(Xg.ravel(), Z_fixed)
    ))

    X_query_scaled = scaler.transform(X_query)
    y_grid_pred, y_grid_pi = ipm.predict_interval(X_query_scaled, allow_infinite_bounds=True)

    pred_mean = y_grid_pred.reshape(Xg.shape)
    pred_lower = y_grid_pi[:,0].reshape(Xg.shape)
    pred_upper = y_grid_pi[:,1].reshape(Xg.shape)

    # -----------------------------------------------------------------------------
    # Plot 3D surface
    # -----------------------------------------------------------------------------
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    interval_alpha = 0.3

    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(Xg, Yg, pred_mean, cmap=cmap, edgecolor='none', alpha=0.7)
    ax.plot_surface(Xg, Yg, pred_lower, color='grey', alpha=interval_alpha, edgecolor='none')
    ax.plot_surface(Xg, Yg, pred_upper, color='grey', alpha=interval_alpha, edgecolor='none')

    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')
    ax.set_zlabel(r'Predicted $d$ [$^\circ C$] with 95% PI')
    ax.set_title(f'IPM interpolation for "{d_type}" at z={Z_fixed:.4f} m')
    fig.colorbar(surf, shrink=0.5, aspect=10, label=r'Predicted $d$ [$^\circ C$]')
    fig.savefig(output_path / f"{d_type}_ipm_surface.png", dpi=300, bbox_inches="tight")

    # -----------------------------------------------------------------------------
    # Plot training points
    # -----------------------------------------------------------------------------
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    point_names = list(merged_df.index)

    ax.scatter(point_names, y_train, color='blue', marker='o', label='Measured')
    ax.scatter(point_names, y_pred, color='red', marker='x', label='Predicted')

    for i, name in enumerate(point_names):
        ax.plot([name, name], [lower_train[i], upper_train[i]], color='gray', alpha=0.6)

    ax.set_xlabel('Thermocouple')
    ax.set_ylabel(r'$d$ [$^\circ C$]')
    ax.set_title(f'Measured vs predicted for "{d_type}" with 95% PI')
    fig.legend()
    ax.grid(True)
    fig.savefig(output_path / f"{d_type}_training_points.png", dpi=300, bbox_inches="tight")