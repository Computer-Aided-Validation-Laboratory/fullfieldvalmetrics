import numpy as np
import pandas as pd
from pathlib import Path
import torch
import gpytorch
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
# Folders and files
# -----------------------------------------------------------------------------

INPUT_FILE = (Path.cwd()
    / "images_pointsensors_pulse25X_v4"
    / "pointsensors_mavm.csv"
)

EXP_DIR = Path.cwd() / "gpr_interp"

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
# Functions
# -----------------------------------------------------------------------------

def load_data(input_file: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load data and merge it into one dataframe

    Parameters
    ----------
    input_file : Path
        Input file

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Merged dataframe and thermocouple name list
    """

    d_values = pd.read_csv(input_file, index_col=0)

    coords_numpy = np.array([
        [0.0116, -0.0245, 0.0194],
        [0.0138, -0.0245, 0.0013],
        [0.0067, -0.0245, 0.012],
        [0.0110,  0.0245, 0.0031],
        [-0.0105, 0.0245, -0.005],
        [-0.0058, 0.0245, 0.0171],
        [-0.018, -0.0006, 0.0164],
        [-0.018, -0.004, -0.0085],
        [-0.018, -0.0047, 0.0073],
        [-0.018,  0.0124, -0.0032]
    ])

    coords = pd.DataFrame(
        coords_numpy,
        columns=["x", "y", "z"],
        index=[
            "TC1", "TC2", "TC3", "TC4", "TC5",
            "TC6", "TC7", "TC8", "TC9", "TC10"
        ]
    )

    merged_df = coords.join(d_values.T, how='inner')

    return merged_df, list(d_values.index)


def fit_gpr_model(df: pd.DataFrame,
                  d_type: str,
                  training_iter: int = 100) -> tuple[ExactGPModel, GaussianLikelihood, dict]:
    """ Train GPR model

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    d_type : str
        Validation metric type
    training_iter : int, optional
        Number of training iterations

    Returns
    -------
    tuple[ExactGPModel, GaussianLikelihood, dict]
        Trained GPR model and normalisation data
    """

    # -------------------------------------------------------------------------
    # Training data
    # -------------------------------------------------------------------------

    X_train = df[['x', 'y', 'z']].values
    y_train = df[d_type].values.reshape(-1, 1)

    # -------------------------------------------------------------------------
    # Normalise inputs and outputs
    # -------------------------------------------------------------------------

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_norm = (X_train - X_mean) / X_std

    y_mean = y_train.mean()
    y_std = y_train.std()

    print(f"Training data std. is {np.round(y_std)}.")

    if y_std < 1e-12:
        y_std = 1.0
        print(f"Changing training data std. to 1.0.")

    y_norm = (y_train - y_mean) / y_std

    # Convert to torch tensors
    train_x = torch.tensor(X_norm, dtype=torch.float32)
    train_y = torch.tensor(y_norm.flatten(), dtype=torch.float32)

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

    # Store normalisation parameters
    norm_params = {
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std
    }

    return model, likelihood, norm_params

def predict_gpr(model_fitted: gpytorch.models.ExactGP,
                likelihood: GaussianLikelihood,
                norm_params: dict,
                X_query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Predict using trained GPR model.

    Parameters
    ----------
    model_fitted : gpytorch.models.ExactGP
        Trained GPR model
    likelihood : GaussianLikelihood
        Trained likelihood
    norm_params : dict
        Data normalisation parameters
    X_query : np.ndarray
        Points where the GPR model is to be evaluated

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Predictions
    """

    X_query_norm = ((X_query - norm_params["X_mean"]) / norm_params["X_std"])
    test_x = torch.tensor(X_query_norm, dtype=torch.float32)

    model_fitted.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        observed_pred = likelihood(model_fitted(test_x))
        pred_mean_norm = observed_pred.mean.numpy()
        lower_norm, upper_norm = (observed_pred.confidence_region())

    # Denormalise predictions

    y_mean = norm_params["y_mean"]
    y_std = norm_params["y_std"]

    pred_mean = pred_mean_norm * y_std + y_mean
    lower = lower_norm.numpy() * y_std + y_mean
    upper = upper_norm.numpy() * y_std + y_mean

    return pred_mean, lower, upper


def evaluate_training_points(model_fitted: gpytorch.models.ExactGP,
                             likelihood: GaussianLikelihood,
                             norm_params: dict,
                             df: pd.DataFrame,
                             d_type: str) -> pd.DataFrame:
    """ Evaluate trained GPR model at the training locations

    Parameters
    ----------
    model_fitted : gpytorch.models.ExactGP
        Trained GPR model
    likelihood : GaussianLikelihood
        Trained likelihood
    norm_params : dict
        Data normalisation parameters
    df : pd.DataFrame
        Training data
    d_type : str
        Validation metric type

    Returns
    -------
    pd.DataFrame
        Prediction results
    """

    X_query = df[['x', 'y', 'z']].values

    pred_mean, lower, upper = predict_gpr(model_fitted, likelihood, norm_params, X_query)

    out_df = pd.DataFrame({
        "TC": df.index,
        "measured": df[d_type].values,
        "predicted": pred_mean,
        "lower_95": lower,
        "upper_95": upper
    })

    return out_df


def leave_one_out_ablation(df: pd.DataFrame, 
                           d_type: str,
                           training_iter: int = 100) -> pd.DataFrame:
    """Perform ablation study by excluding one TC data 
    and fitting the model to the rest of the TC data

    Parameters
    ----------
    df : pd.DataFrame
        TC validation data
    d_type : str
        Validation metric type
    training_iter : int, optional
        Number of training iterations

    Returns
    -------
    pd.DataFrame
        Ablation study results
    """

    tc_names = list(df.index)

    results_list = []

    for excluded_tc in tc_names:

        print(f"Excluding {excluded_tc}")

        train_df = df.drop(index=excluded_tc)
        test_df = df.loc[[excluded_tc]]

        # Fit GPR model
        model, likelihood, norm_params = fit_gpr_model(train_df, d_type, training_iter)

        # Predict excluded point
        X_test = test_df[['x', 'y', 'z']].values

        pred_mean, lower, upper = predict_gpr(model, likelihood, norm_params, X_test)

        measured = test_df[d_type].values[0]

        predicted = pred_mean[0]
        lower_95 = lower[0]
        upper_95 = upper[0]

        error = predicted - measured
        abs_error = abs(error)

        within_pi = (
            lower_95 <= measured <= upper_95
        )

        if measured < lower_95:
            pi_error = lower_95 - measured
        elif measured > upper_95:
            pi_error = measured - upper_95
        else:
            pi_error = 0.0

        if np.abs(measured) > 1e-12:
            rel_error = abs(error) / abs(measured)
        else:
            rel_error = np.nan

        pi_width = upper_95 - lower_95

        results_list.append({
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
        })

    return pd.DataFrame(results_list)


def predict_surface(model_fitted: gpytorch.models.ExactGP,
                    likelihood: GaussianLikelihood,
                    norm_params: dict,
                    df: pd.DataFrame,
                    Z_fixed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict the validation metric on a surface with fixed Z.

    Parameters
    ----------
    model_fitted : gpytorch.models.ExactGP
        Trained GPR model
    likelihood : GaussianLikelihood
        Trained likelihood
    norm_params : dict
        Data normalisation parameters
    df : pd.DataFrame
        TC validation data
    Z_fixed : float
        z coordinate at which to calculate the predictions

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Prediction coordinated and prediction data
    """

    x_vals = df['x'].values
    y_vals = df['y'].values

    x_grid = np.linspace(min(x_vals), max(x_vals), 75)
    y_grid = np.linspace(min(y_vals), max(y_vals), 75)
    Xg, Yg = np.meshgrid(x_grid, y_grid)

    X_query = np.column_stack((
        Xg.ravel(),
        Yg.ravel(),
        np.full_like(Xg.ravel(), Z_fixed)
    ))

    pred_mean, pred_lower, pred_upper = predict_gpr(model_fitted, likelihood, norm_params, X_query)

    pred_mean = pred_mean.reshape(Xg.shape)
    pred_lower = pred_lower.reshape(Xg.shape)
    pred_upper = pred_upper.reshape(Xg.shape)

    return Xg, Yg, pred_mean, pred_lower, pred_upper


def plot_surface(Xg: np.ndarray, Yg: np.ndarray, mean: np.ndarray,
                 CI_lower: np.ndarray, CI_upper: np.ndarray, 
                 d_type: str, output_dir: Path) -> None:
    """ Plot prediction surface with CIs.

    Parameters
    ----------
    Xg : np.ndarray
        X coordinate grid
    Yg : np.ndarray
        Y coordinate grid
    mean : np.ndarray
        Mean values grid
    CI_lower : np.ndarray
        Lower CI grid
    CI_upper : np.ndarray
        Upper CI grid
    d_type : str
        Validation metric type
    output_dir : Path
        Output path
    """

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Mean surface
    surf = ax.plot_surface(Xg, Yg, mean, cmap="coolwarm", edgecolor='none')
    # PI surfaces
    interval_alpha = 0.3  # transparency for CI surfaces
    ax.plot_surface(Xg, Yg, CI_lower, color='grey', alpha=interval_alpha, edgecolor='none', label='Obs CI Lower')
    ax.plot_surface(Xg, Yg, CI_upper, color='grey', alpha=interval_alpha, edgecolor='none', label='Obs CI Upper')

    ax.set_xlabel(r'$x$ (m)', labelpad=10)
    ax.set_ylabel(r'$y$ (m)', labelpad=10)
    ax.set_zlabel(r'Predicted $d$ [$^{\circ C}$]', labelpad=15)

    ax.set_title(f'Quadratic fit surface for "{d_type}"', pad=0)

    fig.colorbar(surf, shrink=0.5, aspect=10, label=r'Predicted $d$ [$^{\circ C}$]')

    fig.savefig(output_dir / f"{d_type}_surface.png", dpi=300, bbox_inches="tight")

    plt.close(fig)


def main():

    EXP_DIR.mkdir(exist_ok=True)

    coeff_dir = EXP_DIR / "model_coefficients"
    train_dir = EXP_DIR / "training_predictions"
    ablation_dir = EXP_DIR / "ablation_results"
    surface_dir = EXP_DIR / "surfaces"

    for d in [coeff_dir, train_dir, ablation_dir, surface_dir]:
        d.mkdir(parents=True, exist_ok=True)

    merged_df, d_types = load_data(INPUT_FILE)

    summary_errors = []
    training_iter = 600

    for d_type in d_types:

        print("=" * 80)
        print(f"Processing {d_type}")

        # ---------------------------------------------------------------------
        # Fit model to all TC data
        # ---------------------------------------------------------------------

        model_fitted, likelihood, norm_params = fit_gpr_model(merged_df, d_type, training_iter)

        # ---------------------------------------------------------------------
        # Training predictions using the model fitted to all TC data
        # ---------------------------------------------------------------------

        train_pred_df = evaluate_training_points(
            model_fitted,
            likelihood,
            norm_params,
            merged_df,
            d_type
        )
        train_pred_df.to_csv(train_dir / f"{d_type}_training_predictions.csv", index=False)

        # ---------------------------------------------------------------------
        # Ablation study
        # ---------------------------------------------------------------------

        ablation_df = leave_one_out_ablation(merged_df, d_type, training_iter)
        ablation_df.to_csv(ablation_dir / f"{d_type}_ablation.csv", index=False)

        mae = mean_absolute_error(ablation_df["measured"], ablation_df["predicted"])
        rmse = np.sqrt(mean_squared_error(ablation_df["measured"], ablation_df["predicted"]))
        # summary_errors.append({
        #     "d_type": d_type,
        #     "MAE": mae,
        #     "RMSE": rmse,
        #     "mean_abs_error": ablation_df["abs_error"].mean(),
        #     "mean_rel_error": ablation_df["rel_error"].mean(skipna=True)
        # })


        summary_errors.append(
            {
                "d_type": d_type,
        
                # mean-based errors
                "MAE": mae,
                "RMSE": rmse,
                "mean_abs_error": ablation_df["abs_error"].mean(),
                # "median_abs_error": ablation_df["abs_error"].median(),
                "mean_rel_error": ablation_df["rel_error"].mean(skipna=True),
                # "median_rel_error": ablation_df["rel_error"].median(skipna=True),
        
                # prediction interval
                "mean_pi_width": ablation_df["pi_width"].mean(),
                # "median_pi_width": ablation_df["pi_width"].median(),
        
                # interval-based errors
                "mean_pi_error": ablation_df["pi_error"].mean(),
                # "median_pi_error": ablation_df["pi_error"].median(),
                "pi_coverage": ablation_df["within_pi"].mean(),
        
                # model fit

            }
        )

        # ---------------------------------------------------------------------
        # Surface prediction using the model fitted to all TC data
        # ---------------------------------------------------------------------

        Z_fixed = (35 - 15/2 - 5) * 1e-3

        Xg, Yg, pred_mean, pred_lower, pred_upper = predict_surface(
            model_fitted,
            likelihood,
            norm_params,
            merged_df,
            Z_fixed
        )
        
        plot_surface(Xg, Yg, 
                     pred_mean, pred_lower, pred_upper, 
                     d_type, surface_dir)

    # -------------------------------------------------------------------------
    # Save ablation study results
    # -------------------------------------------------------------------------

    summary_df = pd.DataFrame(summary_errors)

    summary_df.to_csv(EXP_DIR / "ablation_summary.csv", index=False)

    print(summary_df)

if __name__ == "__main__":
    main()
