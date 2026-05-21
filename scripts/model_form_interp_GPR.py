import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import gpytorch

# -------------------------
# Set random seed
# -------------------------

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

input_file = Path.cwd() / "images_pointsensors_pulse25X_v4/pointsensors_mavm.csv"
print(input_file)

output_path = Path.cwd() / "gpr_interp"
output_path.mkdir(parents=True, exist_ok=True)

d_values = pd.read_csv(input_file, index_col=0)

d_types = list(d_values.index)
print(d_types)

print(80 * "-")
print("Loaded MAVM table")
print(d_values)

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

print(80 * "-")
print("Loaded TC coordinate table (m)")
print(coords)

# Merge two dataframes
merged_df = coords.join(d_values.T, how='inner')

print(80 * "-")
print("Merged dataframe")
print(merged_df)

# -----------------------------------------------------------------------------
# Define GPR model
# -----------------------------------------------------------------------------

class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        # Mean function
        self.mean_module = gpytorch.means.ConstantMean()

        # Covariance / kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# -----------------------------------------------------------------------------
# Loop through each d_type
# -----------------------------------------------------------------------------

for d_type in d_types:

    print(80 * "-")
    print(f"Training GPR model for {d_type} ...")

    # -------------------------------------------------------------------------
    # Training data
    # -------------------------------------------------------------------------

    x_vals = merged_df['x'].values
    y_vals = merged_df['y'].values
    z_vals = merged_df['z'].values
    d_vals = merged_df[d_type].values

    X_train = np.column_stack((x_vals, y_vals, z_vals))

    # Convert to torch tensors
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(d_vals, dtype=torch.float32)

    # -------------------------------------------------------------------------
    # Model and likelihood
    # -------------------------------------------------------------------------

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # -------------------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------------------

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood,
        model
    )

    training_iter = 300

    for i in range(training_iter):

        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        if (i + 1) % 50 == 0:
            print(
                f"Iter {i+1}/{training_iter} - "
                f"Loss: {loss.item():.4f} "
                f"noise: {likelihood.noise.item():.6f}"
            )

        optimizer.step()

    # -------------------------------------------------------------------------
    # Evaluate on training points
    # -------------------------------------------------------------------------

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        observed_pred = likelihood(model(train_x))
        pred_mean = observed_pred.mean.numpy()
        lower, upper = observed_pred.confidence_region()

        lower = lower.numpy()
        upper = upper.numpy()

    print("\nTraining point predictions:")
    pred_df = pd.DataFrame({
        "Measured": d_vals,
        "Predicted": pred_mean,
        "Lower95": lower,
        "Upper95": upper
    })

    print(pred_df)

    # -------------------------------------------------------------------------
    # Create interpolation grid
    # -------------------------------------------------------------------------

    x_grid = np.linspace(min(x_vals), max(x_vals), 75)
    y_grid = np.linspace(min(y_vals), max(y_vals), 75)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    Z_fixed = (35 - 15/2 - 5) * 1e-3 # top surface of a sample

    X_query = np.column_stack((
        Xg.ravel(),
        Yg.ravel(),
        np.full_like(Xg.ravel(), Z_fixed)
    ))

    test_x = torch.tensor(X_query, dtype=torch.float32)

    # -------------------------------------------------------------------------
    # Predict on grid
    # -------------------------------------------------------------------------

    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        observed_pred = likelihood(model(test_x))
        pred_mean = observed_pred.mean.numpy()
        pred_lower, pred_upper = observed_pred.confidence_region()
        pred_lower = pred_lower.numpy()
        pred_upper = pred_upper.numpy()

    pred_mean = pred_mean.reshape(Xg.shape)
    pred_lower = pred_lower.reshape(Xg.shape)
    pred_upper = pred_upper.reshape(Xg.shape)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    cmap = sns.color_palette("coolwarm", as_cmap=True)
    interval_alpha = 0.3  # transparency for PI surfaces
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(Xg, Yg, pred_mean, cmap=cmap, edgecolor='none', alpha=0.7)

    # Confidence interval surfaces
    ax.plot_surface(Xg, Yg, pred_lower, color='grey', alpha=interval_alpha, edgecolor='none')
    ax.plot_surface(Xg, Yg, pred_upper, color='grey', alpha=interval_alpha, edgecolor='none')

    ax.set_xlabel(r'$x$ (m)', labelpad=10)
    ax.set_ylabel(r'$y$ (m)', labelpad=10)
    ax.set_zlabel(r'Predicted $d$ [$^\circ C$]', labelpad=15)

    ax.set_title(
        f'GPR interpolation for {d_type} at z={Z_fixed:.4f} m'
    )

    fig.colorbar(surf, shrink=0.5, aspect=10, label=r'Predicted $d$ [$^\circ C$]')

    fig.savefig(output_path / f"{d_type}_gpr_surface.png", dpi=300, bbox_inches="tight")
