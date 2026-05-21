import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Set random seed
# -------------------------

SEED = 42
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

input_file = Path.cwd() / "images_pointsensors_pulse25X_v4/pointsensors_mavm.csv"
print(input_file)

output_path = Path.cwd() / "quadratic_interp"
output_path.mkdir(parents=True, exist_ok=True)

d_values = pd.read_csv(input_file, index_col=0)

d_types = list(d_values.index)
print(d_types)

print(80 * "-")
print("Loaded MAVM table")
print(d_values)

coords_numpy = np.array([[0.0116, -0.0245, 0.0194], 
                         [0.0138, -0.0245, 0.0013], 
                         [0.0067, -0.0245, 0.012], 
                         [0.0110, 0.0245, 0.0031],
                         [-0.0105, 0.0245, -0.005],
                         [-0.0058, 0.0245, 0.0171],
                         [-0.018, -0.0006, 0.0164],
                         [-0.018, -0.004, -0.0085],
                         [-0.018, -0.0047, 0.0073],
                         [-0.018, 0.0124, -0.0032]]
                         )

coords = pd.DataFrame(coords_numpy, 
                      columns=["x", "y", "z"], 
                      index=["TC1", "TC2", "TC3", "TC4", "TC5", 
                             "TC6", "TC7", "TC8", "TC9", "TC10"])

print(80 * "-")
print("Loaded TC coordinate table (m)")
print(coords)

# Merge two dataframes
merged_df = coords.join(d_values.T, how='inner')

print(80 * "-")
print("Merged dataframe")
print(merged_df)

# -----------------------------------------------------------------------------
# Loop through each d_type
# -----------------------------------------------------------------------------

for d_type in d_types:

    print(80 * "-")
    print(f"Predicting {d_type} ...")

    # -------------------------------------------------------------------------
    # Fit quadratic polynomial to training points
    # -------------------------------------------------------------------------
    
    # Extract arrays for regression
    x_vals = merged_df['x'].values
    y_vals = merged_df['y'].values
    z_vals = merged_df['z'].values
    d_vals = merged_df[d_type].values
    point_names = list(merged_df.index)
    print(point_names)

    # Build quadratic design matrix
    X = np.column_stack((
        x_vals, y_vals, z_vals,# linear terms
        x_vals**2 # quadratic term
    ))
    
    # Add constant term for intercept
    X = sm.add_constant(X)
    
    # Fit ordinary least squares model
    model = sm.OLS(d_vals, X)
    results = model.fit()
    
    print(results.summary())

    # -------------------------------------------------------------------------
    # Evaluate on training points
    # -------------------------------------------------------------------------

    X_new = np.column_stack((
        x_vals, y_vals, z_vals,
        x_vals**2
    ))
    X_new = sm.add_constant(X_new)
    predictions = results.get_prediction(X_new)
    pred_summary = predictions.summary_frame(alpha=0.05)  # 95% prediction interval
    
    print("\nPredictions with 95% prediction interval:")
    print(pred_summary)
    
    print("X shape:", X.shape)
    print("X_new shape:", X_new.shape)
    rank = np.linalg.matrix_rank(X)
    print("Design matrix rank:", rank, "out of", X.shape[1])


    pred_mean_train = pred_summary['mean'].values
    lower_train = pred_summary['obs_ci_lower'].values
    upper_train = pred_summary['obs_ci_upper'].values

    # -------------------------------------------------------------------------
    # Create interpolation grid
    # -------------------------------------------------------------------------

    x_grid = np.linspace(min(x_vals), max(x_vals), 75)
    y_grid = np.linspace(min(y_vals), max(y_vals), 75)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    Z_fixed = (35 - 15/2- 5) * 1e-3  # top surface of a sample
    
    X_query = np.column_stack((
        Xg.ravel(),
        Yg.ravel(),
        np.full_like(Xg.ravel(), Z_fixed),
        (Xg**2).ravel()
    ))
    
    X_query = sm.add_constant(X_query, has_constant='add')
    
    # -------------------------------------------------------------------------
    # Predict on grid
    # -------------------------------------------------------------------------

    pred_summary = results.get_prediction(X_query).summary_frame(alpha=0.05)
    pred_mean = pred_summary['mean'].values.reshape(Xg.shape)
    pred_lower = pred_summary['obs_ci_lower'].values.reshape(Xg.shape)
    pred_upper = pred_summary['obs_ci_upper'].values.reshape(Xg.shape)
    
    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    # Predicted surface plot
    
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    interval_alpha = 0.3  # transparency for PI surfaces
    fig = plt.figure(figsize=(20,6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xg, Yg, pred_mean, cmap=cmap, edgecolor='none', alpha=0.7)
    
    # PI surfaces
    ax.plot_surface(Xg, Yg, pred_lower, color='grey', alpha=interval_alpha, edgecolor='none', label='Obs CI Lower')
    ax.plot_surface(Xg, Yg, pred_upper, color='grey', alpha=interval_alpha, edgecolor='none', label='Obs CI Upper')
    
    # ax.scatter(x_vals, y_vals, d_vals, color='red')
    ax.set_xlabel(r'$x$ (m)', labelpad=10)
    ax.set_ylabel(r'$y$ (m)', labelpad=10)
    ax.set_zlabel(r'Predicted $d$ [$^{\circ C}$]', labelpad=15)
    
    ax.set_title(
        f'Quadratic fit surface for "{d_type}" at z={Z_fixed} m', pad=0
    )

    fig.colorbar(surf, shrink=0.5, aspect=10, label=r'Predicted $d$ [$^{\circ C}$]')

    fig.savefig(output_path/f'{d_type}_quad_surf.png',dpi=300,bbox_inches="tight")

    # Scatter plot for training points

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    
    ax.scatter(point_names, d_vals, color='blue', marker='o', label='Measured')
    ax.scatter(point_names, pred_mean_train, color='red', marker='x', label='Predicted')
    
    for i, name in enumerate(point_names):
        ax.plot([name, name], [lower_train[i], upper_train[i]], color='gray', alpha=0.6)
    
    ax.set_xlabel('Thermocouple')
    ax.set_ylabel(r'$d$ [$^\circ C$]')
    ax.set_title(r'Measured $d$ vs predicted $d$ for ' +  f'"{d_type}" with 95% PI')
    fig.legend()
    ax.grid(True)
    fig.savefig(output_path / f"{d_type}_training_points.png", dpi=300, bbox_inches="tight")
