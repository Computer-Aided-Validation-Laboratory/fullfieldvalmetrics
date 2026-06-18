import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
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
output_path = Path.cwd() / "ipm_interp_2"
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

# ----------------------------
# IPM based on simple NN
# ----------------------------
class IPM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Lower bound head
        self.lower_head = nn.Linear(hidden_dim, 1)

        # Upper bound head
        self.upper_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)

        lower = self.lower_head(h)
        upper = self.upper_head(h)

        # Enforce ordering
        lower, upper = torch.minimum(lower, upper), torch.maximum(lower, upper)

        return lower, upper


# ----------------------------
# Loss function
# ----------------------------
def ipm_loss(x, y, lower, upper, lam_width=1.0, lam_violation=100.0):

    # Interval width penalty to minimise width
    width = torch.mean(upper - lower)

    # Constraint violations (soft-constrained)
    violation_lower = torch.relu(lower - y)
    violation_upper = torch.relu(y - upper)

    violation = torch.mean(violation_lower + violation_upper)

    loss = lam_width * width + lam_violation * violation

    return loss, width, violation


# ----------------------------
# Training
# ----------------------------
def train_ipm(model, X, y, epochs=2000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):
        optimizer.zero_grad()

        lower, upper = model(X)

        loss, width, violation = ipm_loss(X, y, lower, upper)

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(
                f"Epoch {epoch} | "
                f"Loss: {loss.item():.4f} | "
                f"Width: {width.item():.4f} | "
                f"Violation: {violation.item():.6f}"
            )

    return model


# ----------------------------
# Prediction
# ----------------------------
def predict_ipm(model, X):
    X = torch.tensor(X, dtype=torch.float32)
    lower, upper = model(X)
    return lower.detach().numpy(), upper.detach().numpy()

# -----------------------------------------------------------------------------
# Loop through each d_type
# -----------------------------------------------------------------------------

for d_type in d_types:
    print(80 * "-")
    print(f"Training IPM model for {d_type} ...")

    # Training data
    X_train = merged_df[['x', 'y', 'z']].values
    y_train = merged_df[d_type].values

    # Normalise inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # -----------------------------------------------------------------------------
    # Define Interval Predictor Model (IPM)
    # -----------------------------------------------------------------------------

    model = IPM(input_dim=3, hidden_dim=64)
    model = train_ipm(model, X_train_scaled, y_train, epochs=2000, lr=1e-3)

    # -----------------------------------------------------------------------------
    # Predict on training points
    # -----------------------------------------------------------------------------

    lower_train, upper_train = predict_ipm(model, X_train_scaled)
    lower_train = lower_train.squeeze()
    upper_train = upper_train.squeeze()
    y_pred = (lower_train + upper_train) / 2

    print(lower_train.shape, upper_train.shape)

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

    pred_lower, pred_upper = predict_ipm(model, X_query_scaled)
    pred_mean = (pred_lower + pred_upper) / 2

    pred_mean = pred_mean.reshape(Xg.shape)
    pred_lower = pred_lower.reshape(Xg.shape)
    pred_upper = pred_upper.reshape(Xg.shape)


    print(Xg.shape, Yg.shape, pred_mean.shape)

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