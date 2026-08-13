import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ablation_funcs import mape_func, interval_score, regression_accuracy_metrics, load_data

# -------------------------
# Set random seed
# -------------------------

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------------------------------------------------------
# Folders and files
# -----------------------------------------------------------------------------

INPUT_FILE = (
    Path.cwd()
    / "images_pointsensors_pulse25X_v4"
    / "pointsensors_dextremes.csv"
)

EXP_DIR = Path.cwd() / "ipm_interp_2"

TOLERANCE=0.1
EPOCHS = 6000

# -----------------------------------------------------------------------------
# Define IPM model
# -----------------------------------------------------------------------------

class IPM(nn.Module):

    def __init__(self, input_dim, hidden_dim=64):

        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.lower_head = nn.Linear(hidden_dim, 1)
        self.upper_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        h = self.shared(x)

        lower = self.lower_head(h)
        upper = self.upper_head(h)

        lower = torch.minimum(lower, upper)
        upper = torch.maximum(lower, upper)

        return lower, upper

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def ipm_loss(x, y, lower, upper, lam_width=1.0, lam_violation=100.0):

    # Interval width penalty to minimise width
    width = torch.mean(upper - lower)

    # Constraint violations (soft-constrained)
    violation_lower = torch.relu(lower - y)
    violation_upper = torch.relu(y - upper)

    violation = torch.mean(violation_lower + violation_upper)

    loss = lam_width * width + lam_violation * violation

    return loss, width, violation


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


def fit_ipm_model(df,
                  d_type,
                  epochs=2000,
                  lr=1e-3):
    
    # -------------------------------------------------------------------------
    # Training data
    # -------------------------------------------------------------------------

    X_train = df[['x','y','z']].values
    y_train = df[d_type].values.reshape(-1,1)

    # -------------------------------------------------------------------------
    # Normalise inputs and outputs
    # -------------------------------------------------------------------------

    X_scaler = StandardScaler()

    X_scaled = X_scaler.fit_transform(X_train)

    y_mean = y_train.mean()
    y_std = y_train.std()

    if y_std < 1e-12:
        y_std = 1.0

    y_scaled = (y_train - y_mean) / y_std

    # -------------------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------------------

    model = IPM(
        input_dim=3,
        hidden_dim=64
    )

    model = train_ipm(
        model,
        X_scaled,
        y_scaled.flatten(),
        epochs=epochs,
        lr=lr
    )

    norm_params = {
        "X_scaler": X_scaler,
        "y_mean": y_mean,
        "y_std": y_std
    }

    return model, norm_params


def predict_ipm(model,
                norm_params,
                X_query):

    X_scaled = norm_params["X_scaler"].transform(X_query)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():
        lower_norm, upper_norm = model(X_tensor)

    lower_norm = lower_norm.numpy().flatten()
    upper_norm = upper_norm.numpy().flatten()

    pred_norm = 0.5 * (lower_norm + upper_norm)

    # Denormalise predictions

    y_mean = norm_params["y_mean"]
    y_std = norm_params["y_std"]

    pred_mean = pred_norm * y_std + y_mean
    lower = lower_norm * y_std + y_mean
    upper = upper_norm * y_std + y_mean

    return pred_mean, lower, upper


def evaluate_training_points(model,
                             norm_params,
                             df,
                             d_type):

    X_query = df[['x','y','z']].values

    pred_mean, lower, upper = predict_ipm(model, norm_params, X_query)

    out_df = pd.DataFrame({
        "TC": df.index,
        "measured": df[d_type].values,
        "predicted": pred_mean,
        "lower_95": lower,
        "upper_95": upper
    })

    return out_df


def leave_one_out_ablation(df,
                           d_type,
                           epochs=2000,
                           lr=1e-3):

    results = []

    tc_names = list(df.index)

    for excluded_tc in tc_names:

        print(f"Excluding {excluded_tc}")

        train_df = df.drop(
            index=excluded_tc
        )

        test_df = df.loc[
            [excluded_tc]
        ]

        model, norm_params = fit_ipm_model(
            train_df,
            d_type,
            epochs,
            lr
        )

        pred_mean, lower, upper = predict_ipm(
            model,
            norm_params,
            test_df[['x','y','z']].values
        )

        measured = test_df[d_type].values[0]

        predicted = pred_mean[0]

        lower_95 = lower[0]
        upper_95 = upper[0]

        error = predicted - measured

        abs_error = abs(error)

        rel_error = (
            abs_error / abs(measured)
            if abs(measured) > 1e-12
            else np.nan
        )

        within_pi = (
            lower_95 <= measured <= upper_95
        )

        if measured < lower_95:
            pi_error = lower_95 - measured

        elif measured > upper_95:
            pi_error = measured - upper_95

        else:
            pi_error = 0.0

        pi_width = upper_95 - lower_95

        iscore = interval_score(
            measured,
            lower_95,
            upper_95
        )

        results.append({
            "excluded_tc": excluded_tc,
            "measured": measured,
            "predicted": predicted,

            # mean-based errors
            "error": error,
            "abs_error": abs_error,
            "rel_error": rel_error,

            # confidence interval
            "lower_95": lower_95,
            "upper_95": upper_95,
            "pi_width": pi_width,

            # interval-based errors
            "within_pi": within_pi,
            "pi_error": pi_error,
            "interval_score": iscore
        })

    return pd.DataFrame(results)


def main():

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    train_dir = EXP_DIR / "training_predictions"
    ablation_dir = EXP_DIR / "ablation_results"

    train_dir.mkdir(exist_ok=True)
    ablation_dir.mkdir(exist_ok=True)

    merged_df, d_types = load_data(
        INPUT_FILE
    )

    summary_errors = []

    for d_type in d_types:

        print("="*80)
        print(f"Processing {d_type}")
                
        # ---------------------------------------------------------------------
        # Fit model to all TC data
        # ---------------------------------------------------------------------

        model, norm_params = fit_ipm_model(merged_df, d_type, epochs=EPOCHS, lr=1e-3)

        # -------------------------------------------------------------
        # Training predictions using the model fitted to all TC data
        # -------------------------------------------------------------

        train_pred_df = evaluate_training_points(
            model,
            norm_params,
            merged_df,
            d_type
        )

        train_pred_df.to_csv(train_dir / f"{d_type}_training_predictions.csv", index=False)

        # -------------------------------------------------------------
        # Ablation study
        # -------------------------------------------------------------

        ablation_df = leave_one_out_ablation(merged_df, d_type, epochs=EPOCHS, lr=1e-3)

        ablation_df.to_csv(ablation_dir / f"{d_type}_ablation.csv", index=False)

        mae = mean_absolute_error(ablation_df["measured"], ablation_df["predicted"])
        rmse = np.sqrt(mean_squared_error(ablation_df["measured"], ablation_df["predicted"]))


        mape = mape_func(ablation_df["measured"], ablation_df["predicted"])

        within_tol = (
            np.abs(
                ablation_df["predicted"] -
                ablation_df["measured"]
            )
            <=
            TOLERANCE*np.abs(ablation_df["measured"])
        )

        accuracy = within_tol.mean()

        metrics = regression_accuracy_metrics(
            ablation_df["measured"],
            ablation_df["predicted"],
            tolerance=TOLERANCE
        )
        
        summary_errors.append(
            {
            "d_type": d_type,

            # mean-based errors
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,

            "mean_abs_error":ablation_df["abs_error"].mean(),

            "mean_rel_error": ablation_df["rel_error"].mean(skipna=True),

            # prediction interval
            "mean_pi_width": ablation_df["pi_width"].mean(),

            # interval-based errors
            "mean_pi_error": ablation_df["pi_error"].mean(),
            "pi_coverage": ablation_df["within_pi"].mean(),
            "mean_interval_score": ablation_df["interval_score"].mean(),

            # prediction accuracy
            "accuracy_2pct": accuracy,
            "accuracy_2pct_matrix": metrics["accuracy"],
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "TN": metrics["TN"],
            "FN": metrics["FN"],
            }
        )

    summary_df = pd.DataFrame(summary_errors)

    summary_df.to_csv(EXP_DIR / "ablation_summary.csv", index=False)

    print(summary_df)

if __name__ == "__main__":
    main()
