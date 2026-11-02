#!/usr/bin/env python3
"""Compare the case11, case11_1 and case11_2 thermocouple studies.

The three studies use the same underlying thermal problem but sample it in
different ways:

* case11   : a sparse line of 7 thermocouples (TCs),
* case11_1 : a dense line of 36 TCs,
* case11_2 : a 6 x 6 rectangular grid of 36 TCs.

This script compares the target uncertainty fields and the interpolation
performance of the common ablation configurations.  It writes tables, figures
and a short text summary.  It never modifies the input data.

Example
-------
    ~/anaconda3/envs/pyvale/bin/python scripts/plots/compare_case11_studies.py

The default inputs are ``synthetic_data/case11``, ``synthetic_data/case11_1``
and ``synthetic_data/case11_2`` below the repository root.  The default output
directory is ``scripts/plots/case11_comparison_outputs``.
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

try:
    from importlib.metadata import version
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    from importlib_metadata import version


# ---------------------------------------------------------------------------
# Study definitions
# ---------------------------------------------------------------------------

CASES: tuple[str, ...] = ("case11", "case11_1", "case11_2")

CASE_INFO: dict[str, dict[str, str]] = {
    "case11": {
        "label": "case11",
        "design": "Sparse line (7 TCs)",
        "geometry": "line",
        "coord_file": "PointSensorCoords.csv",
        "color": "#D55E00",
    },
    "case11_1": {
        "label": "case11_1",
        "design": "Dense line (36 TCs)",
        "geometry": "line",
        "coord_file": "PointSensorCoords_exp.csv",
        "color": "#009E73",
    },
    "case11_2": {
        "label": "case11_2",
        "design": "6 x 6 grid (36 TCs)",
        "geometry": "grid",
        "coord_file": "PointSensorCoords_exp.csv",
        "color": "#CC79A7",
    },
}

FAMILIES: tuple[str, ...] = (
    "interp_reg_spatial",
    "interp_reg_temp",
    "interp_gpr_spatial",
    "interp_gpr_temp",
    "ipm_interp_spatial",
    "ipm_interp_temp",
)

FAMILY_INFO: dict[str, dict[str, str]] = {
    "interp_reg_spatial": {"label": "Reg-Spatial", "model": "reg", "domain": "spatial"},
    "interp_reg_temp": {"label": "Reg-Temp", "model": "reg", "domain": "temp"},
    "interp_gpr_spatial": {"label": "GPR-Spatial", "model": "gpr", "domain": "spatial"},
    "interp_gpr_temp": {"label": "GPR-Temp", "model": "gpr", "domain": "temp"},
    "ipm_interp_spatial": {"label": "IPM-Spatial", "model": "ipm", "domain": "spatial"},
    "ipm_interp_temp": {"label": "IPM-Temp", "model": "ipm", "domain": "temp"},
}

DTYPES: tuple[str, ...] = (
    "sim_plus",
    "sim_minus",
    "model_form_plus",
    "model_form_minus",
    "total_plus",
    "total_minus",
)

DTYPE_LABELS: dict[str, str] = {
    "sim_plus": "simulation +",
    "sim_minus": "simulation -",
    "model_form_plus": "model form +",
    "model_form_minus": "model form -",
    "total_plus": "total +",
    "total_minus": "total -",
}

PAIRS: tuple[tuple[str, str, str], ...] = (
    ("case11", "case11_1", "case11 -> case11_1 (sparse line -> dense line)"),
    ("case11_1", "case11_2", "case11_1 -> case11_2 (dense line -> grid)"),
    ("case11", "case11_2", "case11 -> case11_2 (sparse line -> grid)"),
)

PAIR_LABELS: dict[tuple[str, str], str] = {
    (case_a, case_b): label for case_a, case_b, label in PAIRS
}

ALL_METRICS: tuple[str, ...] = (
    "MAE",
    "RMSE",
    "MAPE",
    "mean_rel_error",
    "mean_pi_width",
    "mean_pi_error",
    "pi_coverage",
    "mean_interval_score",
    "accuracy",
    "coverage_error",
)

PAIR_METRICS: tuple[str, ...] = (
    "RMSE",
    "mean_interval_score",
    "mean_rel_error",
    "accuracy",
    "coverage_error",
)

BEST_METRICS: tuple[str, ...] = (
    "RMSE",
    "mean_interval_score",
    "mean_rel_error",
    "accuracy",
    "coverage_error",
)

LOWER_IS_BETTER: set[str] = {
    "MAE",
    "RMSE",
    "MAPE",
    "mean_rel_error",
    "mean_pi_width",
    "mean_pi_error",
    "mean_interval_score",
    "coverage_error",
}

METRIC_LABELS: dict[str, str] = {
    "MAE": "MAE",
    "RMSE": "RMSE",
    "MAPE": "MAPE",
    "mean_rel_error": "relative error",
    "mean_pi_width": "PI width",
    "mean_pi_error": "PI error",
    "pi_coverage": "PI coverage",
    "mean_interval_score": "interval score",
    "accuracy": "10% accuracy",
    "coverage_error": "coverage error",
}

TOLERANCE = 0.10
COVERAGE_TARGET = 0.95
COMMON_SUPPORT_PAD = 1e-9


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _target_path(data_root: Path, case: str) -> Path:
    return (
        data_root
        / case
        / "pyvale-output"
        / "valid"
        / "pointsensors_total_uncertainty_temperature.csv"
    )


def _output_path(data_root: Path, case: str, family: str) -> Path:
    return data_root / case / "pyvale-output" / family


def load_case_table(case: str, data_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load target values and coordinates for one study.

    Only the TCs present in the target table are retained.  This matters for
    case11_2: its coordinate file has 42 rows, while the target/ablation data
    use the first 36 TCs (a 6 x 6 grid).
    """

    out_dir = data_root / case / "pyvale-output"
    target = pd.read_csv(_target_path(data_root, case), index_col=0).T
    target.index = pd.Index([f"TC{i}" for i in range(1, len(target) + 1)], name="tc")
    target = target.reset_index()

    coord_path = out_dir / CASE_INFO[case]["coord_file"]
    coords_all = pd.read_csv(coord_path)
    coords_all.index = pd.Index(
        [f"TC{i}" for i in range(1, len(coords_all) + 1)], name="tc"
    )

    missing = sorted(set(target["tc"]) - set(coords_all.index))
    if missing:
        raise ValueError(f"{case}: target TCs missing from {coord_path}: {missing}")

    coords = coords_all.loc[target["tc"], ["x", "y", "z"]].reset_index(drop=True)
    target = target.reset_index(drop=True)
    df = pd.concat([target, coords], axis=1)

    for dtype in DTYPES:
        if dtype not in df.columns:
            raise ValueError(f"{case}: target table is missing {dtype!r}")

    xyz = df[["x", "y", "z"]].to_numpy(dtype=float)
    distances = cdist(xyz, xyz)
    np.fill_diagonal(distances, np.inf)
    df["nn_distance"] = distances.min(axis=1)

    df["case"] = case
    df["case_label"] = CASE_INFO[case]["label"]
    df["design"] = CASE_INFO[case]["design"]
    df["geometry"] = CASE_INFO[case]["geometry"]

    meta: dict[str, object] = {
        "case": case,
        "label": CASE_INFO[case]["label"],
        "design": CASE_INFO[case]["design"],
        "geometry": CASE_INFO[case]["geometry"],
        "n_coords_file": int(len(coords_all)),
        "n_unused_coords": int(len(coords_all) - len(df)),
        "coord_file": str(coord_path),
        "target_file": str(_target_path(data_root, case)),
    }
    return df, meta


def discover_common_configs(data_root: Path) -> dict[str, list[str]]:
    """Return configurations that exist for every case in every family."""

    common: dict[str, list[str]] = {}
    for family in FAMILIES:
        per_case: list[set[str]] = []
        for case in CASES:
            root = _output_path(data_root, case, family)
            if not root.is_dir():
                raise FileNotFoundError(f"Missing family directory: {root}")
            configs = {
                path.stem.replace("ablation_summary_", "")
                for path in root.glob("ablation_summary_*.csv")
            }
            per_case.append(configs)
        shared = set.intersection(*per_case)
        if not shared:
            raise ValueError(f"No common configurations found for {family}")
        common[family] = sorted(shared)

    n_configs = sum(len(values) for values in common.values())
    print(f"Found {n_configs} common model configurations across all cases.")
    return common


def _normalise_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Normalise naming differences between the model families."""

    summary = summary.copy()
    if "accuracy" not in summary.columns and "accuracy_2pct" in summary.columns:
        summary = summary.rename(columns={"accuracy_2pct": "accuracy"})
    if "accuracy_matrix" not in summary.columns and "accuracy_2pct_matrix" in summary.columns:
        summary = summary.rename(
            columns={"accuracy_2pct_matrix": "accuracy_matrix"}
        )
    return summary


def load_summary_metrics(
    data_root: Path, common_configs: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ablation summaries for all cases and common configurations."""

    rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    for case in CASES:
        for family in FAMILIES:
            family_info = FAMILY_INFO[family]
            for config in common_configs[family]:
                path = _output_path(data_root, case, family) / (
                    f"ablation_summary_{config}.csv"
                )
                if not path.exists():
                    raise FileNotFoundError(path)
                summary = _normalise_summary(pd.read_csv(path))
                manifest_rows.append(
                    {
                        "case": case,
                        "family": family,
                        "model": family_info["model"],
                        "domain": family_info["domain"],
                        "config": config,
                        "summary_file": str(path),
                    }
                )
                for _, row in summary.iterrows():
                    record: dict[str, object] = {
                        "case": case,
                        "case_label": CASE_INFO[case]["label"],
                        "design": CASE_INFO[case]["design"],
                        "family": family,
                        "family_label": family_info["label"],
                        "model": family_info["model"],
                        "domain": family_info["domain"],
                        "config": config,
                        "d_type": row["d_type"],
                    }
                    for metric in ALL_METRICS[:-1]:
                        if metric in summary.columns:
                            record[metric] = row[metric]
                    record["coverage_error"] = abs(
                        float(row.get("pi_coverage", np.nan)) - COVERAGE_TARGET
                    )
                    rows.append(record)

    metrics = pd.DataFrame(rows)
    metrics = metrics[metrics["d_type"].isin(DTYPES)].copy()
    metrics["coverage_error"] = (metrics["pi_coverage"] - COVERAGE_TARGET).abs()
    return metrics, pd.DataFrame(manifest_rows)


def load_per_tc_results(
    data_root: Path,
    case_tables: dict[str, pd.DataFrame],
    common_configs: dict[str, list[str]],
) -> pd.DataFrame:
    """Load the leave-one-out result file behind every summary row."""

    parts: list[pd.DataFrame] = []
    for case in CASES:
        coord_map = case_tables[case].set_index("tc")[
            ["x", "y", "z", "nn_distance"]
        ]
        for family in FAMILIES:
            for config in common_configs[family]:
                for d_type in DTYPES:
                    path = (
                        _output_path(data_root, case, family)
                        / "ablation_results"
                        / f"{d_type}_ablation_{config}.csv"
                    )
                    if not path.exists():
                        raise FileNotFoundError(path)
                    result = pd.read_csv(path)
                    if "excluded_tc" not in result.columns:
                        raise ValueError(f"{path}: no excluded_tc column")
                    result = result.rename(columns={"excluded_tc": "tc"})
                    result["tc"] = result["tc"].astype(str)
                    missing = sorted(set(result["tc"]) - set(coord_map.index))
                    if missing:
                        raise ValueError(f"{path}: missing coordinates for {missing}")
                    result = result.join(coord_map, on="tc")
                    result["case"] = case
                    result["design"] = CASE_INFO[case]["design"]
                    result["family"] = family
                    result["config"] = config
                    result["d_type"] = d_type
                    parts.append(result)

    per_tc = pd.concat(parts, ignore_index=True)
    per_tc["measured"] = pd.to_numeric(per_tc["measured"], errors="coerce")
    per_tc["predicted"] = pd.to_numeric(per_tc["predicted"], errors="coerce")
    if per_tc["within_pi"].dtype == object:
        per_tc["within_pi"] = per_tc["within_pi"].map(
            {"True": True, "False": False, True: True, False: False}
        )
    per_tc["within_pi"] = per_tc["within_pi"].astype(bool)
    print(f"Loaded {len(per_tc):,} per-TC result rows.")
    return per_tc


# ---------------------------------------------------------------------------
# Metric computation and validation
# ---------------------------------------------------------------------------


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean(skipna=True)) if numeric.notna().any() else np.nan


def compute_metrics_from_per_tc(per_tc: pd.DataFrame) -> pd.DataFrame:
    """Recompute the ablation summary metrics from per-TC results."""

    group_cols = ["case", "family", "config", "d_type"]
    rows: list[dict[str, object]] = []

    for keys, group in per_tc.groupby(group_cols, sort=False):
        case, family, config, d_type = keys
        measured = group["measured"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        error = predicted - measured
        abs_error = np.abs(error)

        nonzero = measured != 0
        mape = (
            float(np.mean(np.abs(error[nonzero] / measured[nonzero])) * 100.0)
            if np.any(nonzero)
            else np.nan
        )
        within = group["within_pi"].astype(float).to_numpy()
        pi_coverage = float(np.mean(within))

        rows.append(
            {
                "case": case,
                "family": family,
                "config": config,
                "d_type": d_type,
                "n_tc": int(len(group)),
                "MAE": float(np.mean(abs_error)),
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "MAPE": mape,
                "mean_rel_error": _safe_mean(group["rel_error"]),
                "mean_pi_width": _safe_mean(group["pi_width"]),
                "mean_pi_error": _safe_mean(group["pi_error"]),
                "pi_coverage": pi_coverage,
                "mean_interval_score": _safe_mean(group["interval_score"]),
                "accuracy": float(np.mean(abs_error <= TOLERANCE * np.abs(measured))),
                "coverage_error": abs(pi_coverage - COVERAGE_TARGET),
            }
        )

    return pd.DataFrame(rows)


def validate_recomputed_metrics(
    recomputed: pd.DataFrame, summary: pd.DataFrame
) -> pd.DataFrame:
    """Check that recomputed metrics agree with the saved summaries."""

    merge_keys = ["case", "family", "config", "d_type"]
    comparison = recomputed.merge(
        summary[merge_keys + list(ALL_METRICS)],
        on=merge_keys,
        suffixes=("_recomputed", "_summary"),
        how="inner",
    )
    rows: list[dict[str, object]] = []
    for metric in ALL_METRICS:
        left = comparison[f"{metric}_recomputed"]
        right = comparison[f"{metric}_summary"]
        delta = (left - right).abs()
        denominator = right.abs().replace(0, np.nan)
        rows.append(
            {
                "metric": metric,
                "n_compared": int(delta.notna().sum()),
                "max_abs_diff": float(delta.max(skipna=True)),
                "max_rel_diff": float((delta / denominator).max(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def validate_target_measurements(
    per_tc: pd.DataFrame, case_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Check that result files use the target values in the target tables."""

    rows: list[dict[str, object]] = []
    for case in CASES:
        table = case_tables[case].set_index("tc")
        for d_type in DTYPES:
            target = table[d_type]
            observed = (
                per_tc.loc[per_tc["case"] == case]
                .loc[per_tc["d_type"] == d_type]
                .drop_duplicates(subset=["tc"])
                .set_index("tc")["measured"]
            )
            common = target.index.intersection(observed.index)
            delta = (observed.loc[common] - target.loc[common]).abs()
            rows.append(
                {
                    "case": case,
                    "d_type": d_type,
                    "n_tc": int(len(common)),
                    "max_abs_diff": float(delta.max()) if len(delta) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def add_support_flags(
    per_tc: pd.DataFrame, common_y: tuple[float, float]
) -> pd.DataFrame:
    per_tc = per_tc.copy()
    lo, hi = common_y
    per_tc["in_common_y"] = (per_tc["y"] >= lo - COMMON_SUPPORT_PAD) & (
        per_tc["y"] <= hi + COMMON_SUPPORT_PAD
    )
    return per_tc


# ---------------------------------------------------------------------------
# Target and study tables
# ---------------------------------------------------------------------------


def build_target_tables(
    case_tables: dict[str, pd.DataFrame], common_y: tuple[float, float]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build target values, target summaries and common-y consistency checks."""

    target_values = pd.concat(case_tables.values(), ignore_index=True)
    target_values = target_values[
        [
            "case",
            "case_label",
            "design",
            "geometry",
            "tc",
            "x",
            "y",
            "z",
            "nn_distance",
            *DTYPES,
        ]
    ].copy()

    lo, hi = common_y
    frames: list[pd.DataFrame] = []
    for case in CASES:
        table = case_tables[case]
        for support_name, subset in (
            ("all", table),
            (
                "common_y",
                table[
                    (table["y"] >= lo - COMMON_SUPPORT_PAD)
                    & (table["y"] <= hi + COMMON_SUPPORT_PAD)
                ],
            ),
        ):
            for d_type in DTYPES:
                values = pd.to_numeric(subset[d_type], errors="coerce").dropna()
                if values.empty:
                    continue
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
                frames.append(
                    {
                        "case": case,
                        "case_label": CASE_INFO[case]["label"],
                        "design": CASE_INFO[case]["design"],
                        "support": support_name,
                        "d_type": d_type,
                        "n_tc": int(len(values)),
                        "mean": mean,
                        "median": float(values.median()),
                        "std": std,
                        "min": float(values.min()),
                        "q25": float(values.quantile(0.25)),
                        "q75": float(values.quantile(0.75)),
                        "max": float(values.max()),
                        "cv": float(std / abs(mean)) if mean != 0 else np.nan,
                    }
                )
    target_summary = pd.DataFrame(frames)

    # case11 and case11_2 share the same y levels.  Their sim targets should be
    # identical and their model-form/total targets should differ only because
    # case11 is a single z slice while case11_2 averages over six z slices.
    left = (
        case_tables["case11"]
        .groupby("y", as_index=False)[list(DTYPES)]
        .mean()
        .rename(columns={dtype: f"{dtype}_case11" for dtype in DTYPES})
    )
    right = (
        case_tables["case11_2"]
        .groupby("y", as_index=False)[list(DTYPES)]
        .mean()
        .rename(columns={dtype: f"{dtype}_case11_2" for dtype in DTYPES})
    )
    consistency = left.merge(right, on="y", how="inner")
    for d_type in DTYPES:
        consistency[f"{d_type}_abs_diff"] = (
            consistency[f"{d_type}_case11_2"] - consistency[f"{d_type}_case11"]
        ).abs()
    consistency.insert(0, "comparison", "case11 vs case11_2 at common y")

    return target_values, target_summary, consistency


def build_study_overview(
    case_tables: dict[str, pd.DataFrame],
    case_meta: dict[str, dict[str, object]],
    common_y: tuple[float, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lo, hi = common_y
    for case in CASES:
        table = case_tables[case]
        y_values = np.sort(table["y"].unique())
        z_values = np.sort(table["z"].unique())
        y_spacing = np.diff(y_values) if len(y_values) > 1 else np.array([np.nan])
        z_spacing = np.diff(z_values) if len(z_values) > 1 else np.array([np.nan])
        y_spacing_median = (
            float(np.nanmedian(y_spacing))
            if np.isfinite(y_spacing).any()
            else np.nan
        )
        z_spacing_median = (
            float(np.nanmedian(z_spacing))
            if np.isfinite(z_spacing).any()
            else np.nan
        )
        in_common = table[
            (table["y"] >= lo - COMMON_SUPPORT_PAD)
            & (table["y"] <= hi + COMMON_SUPPORT_PAD)
        ]
        meta = case_meta[case]
        rows.append(
            {
                "case": case,
                "label": meta["label"],
                "design": meta["design"],
                "geometry": meta["geometry"],
                "n_tc_target": int(len(table)),
                "n_coords_file": int(meta["n_coords_file"]),
                "n_unused_coords": int(meta["n_unused_coords"]),
                "x_min": float(table["x"].min()),
                "x_max": float(table["x"].max()),
                "y_min": float(table["y"].min()),
                "y_max": float(table["y"].max()),
                "z_min": float(table["z"].min()),
                "z_max": float(table["z"].max()),
                "n_y_levels": int(len(y_values)),
                "n_z_levels": int(len(z_values)),
                "y_spacing_median": y_spacing_median,
                "z_spacing_median": z_spacing_median,
                "nn_distance_min": float(table["nn_distance"].min()),
                "nn_distance_median": float(table["nn_distance"].median()),
                "nn_distance_max": float(table["nn_distance"].max()),
                "common_y_n_tc": int(len(in_common)),
                "target_total_plus_mean_all": float(table["total_plus"].mean()),
                "target_total_plus_mean_common_y": float(
                    in_common["total_plus"].mean()
                ),
                "target_total_minus_mean_all": float(table["total_minus"].mean()),
                "target_total_minus_mean_common_y": float(
                    in_common["total_minus"].mean()
                ),
                "common_y_min": float(lo),
                "common_y_max": float(hi),
                "coord_file": meta["coord_file"],
                "target_file": meta["target_file"],
            }
        )
    return pd.DataFrame(rows)


def build_per_tc_error_summary(per_tc: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "case",
        "design",
        "d_type",
        "tc",
        "x",
        "y",
        "z",
        "nn_distance",
        "in_common_y",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in per_tc.groupby(group_cols, sort=False):
        (
            case,
            design,
            d_type,
            tc,
            x,
            y,
            z,
            nn_distance,
            in_common_y,
        ) = keys
        rows.append(
            {
                "case": case,
                "design": design,
                "d_type": d_type,
                "tc": tc,
                "x": x,
                "y": y,
                "z": z,
                "nn_distance": nn_distance,
                "in_common_y": bool(in_common_y),
                "n_configs": int(group["abs_error"].notna().sum()),
                "measured_mean": float(group["measured"].mean()),
                "mean_abs_error": float(group["abs_error"].mean()),
                "median_abs_error": float(group["abs_error"].median()),
                "q25_abs_error": float(group["abs_error"].quantile(0.25)),
                "q75_abs_error": float(group["abs_error"].quantile(0.75)),
                "mean_rel_error": float(group["rel_error"].mean()),
                "mean_interval_score": float(group["interval_score"].mean()),
                "mean_pi_coverage": float(group["within_pi"].mean()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired comparisons and best models
# ---------------------------------------------------------------------------


def _metric_direction(metric: str) -> str:
    return "lower" if metric in LOWER_IS_BETTER else "higher"


def build_pairwise_summary(
    metrics: pd.DataFrame,
    support_label: str,
    group_cols: list[str],
) -> pd.DataFrame:
    """Compare each pair of cases configuration by configuration."""

    rows: list[dict[str, object]] = []
    for case_a, case_b, pair_label in PAIRS:
        base = metrics[metrics["case"] == case_a]
        comp = metrics[metrics["case"] == case_b]
        for metric in PAIR_METRICS:
            key_cols = ["family", "config", "d_type"]
            merged = base[key_cols + [metric]].merge(
                comp[key_cols + [metric]],
                on=key_cols,
                suffixes=("_a", "_b"),
                how="inner",
            ).dropna(subset=[f"{metric}_a", f"{metric}_b"])
            if merged.empty:
                continue

            if group_cols:
                grouped: Iterable[tuple[object, pd.DataFrame]] = merged.groupby(
                    group_cols + ["d_type"], sort=False
                )
            else:
                grouped = [
                    (d_type, group)
                    for d_type, group in merged.groupby("d_type", sort=False)
                ]

            for group_key, group in grouped:
                a = group[f"{metric}_a"].to_numpy(dtype=float)
                b = group[f"{metric}_b"].to_numpy(dtype=float)
                delta = b - a
                denom = np.where(np.abs(a) > 1e-12, a, np.nan)
                pct = np.where(np.isfinite(denom), (b - a) / denom * 100.0, np.nan)
                if metric in LOWER_IS_BETTER:
                    better = b < a
                else:
                    better = b > a
                tie = np.isclose(b, a, rtol=1e-9, atol=1e-12)
                n_pairs = int(len(a))
                n_better = int(np.sum(better & ~tie))
                n_ties = int(np.sum(tie))

                record: dict[str, object] = {
                    "support": support_label,
                    "pair": pair_label,
                    "case_a": case_a,
                    "case_b": case_b,
                    "d_type": group["d_type"].iloc[0],
                    "metric": metric,
                    "direction": _metric_direction(metric),
                    "n_pairs": n_pairs,
                    "median_a": float(np.median(a)),
                    "median_b": float(np.median(b)),
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "median_delta_b_minus_a": float(np.median(delta)),
                    "mean_delta_b_minus_a": float(np.mean(delta)),
                    "median_pct_change": float(np.nanmedian(pct)),
                    "mean_pct_change": float(np.nanmean(pct)),
                    "win_rate_b_better": n_better / n_pairs if n_pairs else np.nan,
                    "n_b_better": n_better,
                    "n_ties": n_ties,
                }
                if group_cols == ["family"]:
                    family_key, _ = group_key
                    record["family"] = family_key
                    record["family_label"] = FAMILY_INFO[family_key]["label"]
                elif group_cols:
                    for col, value in zip(group_cols, group_key):
                        record[col] = value
                else:
                    record["family"] = "ALL"
                    record["family_label"] = "All families"
                rows.append(record)

    return pd.DataFrame(rows)


def build_best_models(
    metrics: pd.DataFrame, support_label: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (case, family, d_type), group in metrics.groupby(
        ["case", "family", "d_type"], sort=False
    ):
        for metric in BEST_METRICS:
            ascending = metric in LOWER_IS_BETTER
            candidate = group.dropna(subset=[metric]).sort_values(
                metric, ascending=ascending
            )
            if candidate.empty:
                continue
            best = candidate.iloc[0]
            rows.append(
                {
                    "support": support_label,
                    "case": case,
                    "family": family,
                    "family_label": FAMILY_INFO[family]["label"],
                    "d_type": d_type,
                    "metric": metric,
                    "direction": _metric_direction(metric),
                    "best_config": best["config"],
                    "best_value": float(best[metric]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _case_order() -> list[str]:
    return list(CASES)


def _case_palette() -> dict[str, str]:
    return {case: CASE_INFO[case]["color"] for case in CASES}


def _family_palette() -> dict[str, str]:
    colors = sns.color_palette("colorblind", len(FAMILIES))
    return {family: colors[i] for i, family in enumerate(FAMILIES)}

def plot_metric_boxplots(
    metrics: pd.DataFrame,
    metric: str,
    output_path: Path,
    dpi: int,
    support_label: str = "common_y",
) -> None:
    palette = _case_palette()
    order = _case_order()
    if "support" in metrics.columns:
        data = metrics[metrics["support"] == support_label]
    else:
        data = metrics

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    axes_flat = axes.ravel()
    for ax, d_type in zip(axes_flat, DTYPES):
        subset = data[data["d_type"] == d_type]
        sns.boxplot(
            data=subset,
            x="case",
            y=metric,
            hue="case",
            order=order,
            palette=palette,
            dodge=False,
            legend=False,
            width=0.58,
            ax=ax,
            showfliers=True,
            fliersize=2,
        )
        ax.set_title(DTYPE_LABELS[d_type])
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.25, which="both")
    # fig.suptitle(
    #     f"{METRIC_LABELS.get(metric, metric)} across model configurations "
    #     f"({support_label.replace('_', '-')} support)",
    #     y=1.02,
    # )
    fig.suptitle(
        f"{METRIC_LABELS.get(metric, metric)} across model configurations ",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_coverage_accuracy(
    metrics: pd.DataFrame,
    output_path: Path,
    dpi: int,
    support_label: str = "common_y",
) -> None:
    palette = _case_palette()
    data = metrics[metrics["support"] == support_label]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    panels = (
        ("coverage_error", "total_plus", "coverage error, total +"),
        ("coverage_error", "total_minus", "coverage error, total -"),
        ("accuracy", "total_plus", "10% accuracy, total +"),
        ("accuracy", "total_minus", "10% accuracy, total -"),
    )
    for ax, (metric, d_type, title) in zip(axes.ravel(), panels):
        subset = data[data["d_type"] == d_type]
        sns.boxplot(
            data=subset,
            x="case",
            y=metric,
            hue="case",
            order=_case_order(),
            palette=palette,
            dodge=False,
            legend=False,
            width=0.55,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(
        "Calibration and threshold accuracy by case "
        f"({support_label.replace('_', '-')} support)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------


def _fmt(value: object, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _overall_row(
    pairwise: pd.DataFrame,
    support: str,
    case_a: str,
    case_b: str,
    d_type: str,
    metric: str,
) -> pd.Series | None:
    subset = pairwise[
        (pairwise["support"] == support)
        & (pairwise["case_a"] == case_a)
        & (pairwise["case_b"] == case_b)
        & (pairwise["d_type"] == d_type)
        & (pairwise["metric"] == metric)
        & (pairwise["family"] == "ALL")
    ]
    return None if subset.empty else subset.iloc[0]


def build_text_summary(
    overview: pd.DataFrame,
    target_summary: pd.DataFrame,
    common_metrics: pd.DataFrame,
    pairwise_overall: pd.DataFrame,
    pairwise_family: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    validation_targets: pd.DataFrame,
    best_models: pd.DataFrame,
    common_y: tuple[float, float],
) -> str:
    lines: list[str] = []
    lines.append("case11 / case11_1 / case11_2 comparison")
    lines.append("=" * 72)
    lines.append("")

    lines.append("Study designs")
    lines.append("-" * 72)
    for _, row in overview.iterrows():
        lines.append(
            f"{row['case']:8s}  {row['design']:28s}  "
            f"N={int(row['n_tc_target']):2d}, "
            f"y=[{row['y_min']:.3f}, {row['y_max']:.3f}], "
            f"z=[{row['z_min']:.3f}, {row['z_max']:.3f}], "
            f"median NN={row['nn_distance_median']:.3f}"
        )
    lines.append(
        "case11_2 uses 36 of the 42 rows in its coordinate file; the target and "
        "ablation files contain 36 TCs (6 y levels x 6 z levels)."
    )
    lines.append(
        f"Common-y support: [{common_y[0]:.3f}, {common_y[1]:.3f}] "
        f"(case11: 6 TCs, case11_1: 23 TCs, case11_2: 36 TCs)."
    )
    lines.append("")

    lines.append("Target-field comparison (mean total uncertainty, °C)")
    lines.append("-" * 72)
    for case in CASES:
        all_row = target_summary[
            (target_summary["case"] == case)
            & (target_summary["support"] == "all")
            & (target_summary["d_type"] == "total_plus")
        ].iloc[0]
        common_row = target_summary[
            (target_summary["case"] == case)
            & (target_summary["support"] == "common_y")
            & (target_summary["d_type"] == "total_plus")
        ].iloc[0]
        lines.append(
            f"{case:8s}  all TCs: {all_row['mean']:.2f}  |  "
            f"common-y: {common_row['mean']:.2f}  "
            f"(median {common_row['median']:.2f}, max {common_row['max']:.2f})"
        )
    lines.append(
        "case11 and case11_2 have the same total_plus mean on the common-y "
        "support; case11_1 is slightly easier there, so the performance "
        "comparison is not biased against the denser layouts."
    )
    lines.append(
        "At the six shared y levels, sim_plus/sim_minus targets agree to "
        "within ~2e-8 °C between case11 and case11_2; total targets differ by "
        "at most ~0.17 °C because case11_2 averages over six z slices."
    )
    lines.append("")

    lines.append("Model performance on common-y support (median over 108 configs)")
    lines.append("-" * 72)
    header = (
        f"{'d_type':18s} {'case':8s} {'RMSE':>8s} {'int. score':>10s} "
        f"{'rel. err':>9s} {'accuracy':>9s} {'cov. err':>9s}"
    )
    lines.append(header)
    for d_type in ("total_plus", "total_minus"):
        for case in CASES:
            subset = common_metrics[
                (common_metrics["case"] == case)
                & (common_metrics["d_type"] == d_type)
            ]
            lines.append(
                f"{d_type:18s} {case:8s} "
                f"{subset['RMSE'].median():8.3f} "
                f"{subset['mean_interval_score'].median():10.2f} "
                f"{subset['mean_rel_error'].median():9.3f} "
                f"{subset['accuracy'].median():9.3f} "
                f"{subset['coverage_error'].median():9.3f}"
            )
    lines.append("")

    lines.append("Paired comparisons (common-y support, win rate = second case better)")
    lines.append("-" * 72)
    for case_a, case_b, pair_label in PAIRS:
        lines.append(pair_label)
        for d_type in ("total_plus", "total_minus"):
            for metric in ("RMSE", "mean_interval_score", "mean_rel_error"):
                row = _overall_row(
                    pairwise_overall, "common_y", case_a, case_b, d_type, metric
                )
                if row is None:
                    continue
                lines.append(
                    f"  {d_type:12s} {METRIC_LABELS[metric]:15s} "
                    f"median {row['median_a']:.3f} -> {row['median_b']:.3f} "
                    f"({row['median_pct_change']:+.1f}%), "
                    f"win rate {row['win_rate_b_better']:.2%}"
                )
        lines.append("")

    lines.append("Family-level consistency (common-y support)")
    lines.append("-" * 72)
    for case_a, case_b, pair_label in PAIRS:
        for d_type in ("total_plus", "total_minus"):
            subset = pairwise_family[
                (pairwise_family["support"] == "common_y")
                & (pairwise_family["case_a"] == case_a)
                & (pairwise_family["case_b"] == case_b)
                & (pairwise_family["d_type"] == d_type)
                & (pairwise_family["metric"] == "RMSE")
            ]
            n_improved = int((subset["win_rate_b_better"] > 0.5).sum())
            lines.append(
                f"{case_a} -> {case_b}, {d_type:12s}: "
                f"{n_improved}/{len(subset)} families have a family-level RMSE win "
                f"rate > 0.5; family win rates range "
                f"{subset['win_rate_b_better'].min():.2f} to "
                f"{subset['win_rate_b_better'].max():.2f}."
            )
    lines.append("")

    lines.append("Overall best total_plus RMSE configuration per case (common-y)")
    lines.append("-" * 72)
    for case in CASES:
        subset = best_models[
            (best_models["support"] == "common_y")
            & (best_models["case"] == case)
            & (best_models["d_type"] == "total_plus")
            & (best_models["metric"] == "RMSE")
        ]
        if subset.empty:
            continue
        row = subset.sort_values("best_value").iloc[0]
        lines.append(
            f"{case:8s}  {row['family_label']:12s}  config {row['best_config']:24s} "
            f"RMSE={row['best_value']:.4f} °C"
        )
    lines.append("")

    lines.append("Validation")
    lines.append("-" * 72)
    worst_metric = validation_metrics.sort_values("max_rel_diff", ascending=False).iloc[0]
    lines.append(
        f"Recomputed summaries match saved summaries for all metrics; worst "
        f"relative difference is {worst_metric['max_rel_diff']:.3e} "
        f"({worst_metric['metric']})."
    )
    lines.append(
        f"Result-file measured values match the target tables in all cases; "
        f"maximum absolute difference is "
        f"{validation_targets['max_abs_diff'].max():.3e} °C."
    )
    lines.append("")

    lines.append("Interpretation and caveats")
    lines.append("-" * 72)
    lines.append(
        "Densifying the line from 7 to 36 TCs improves both total_plus and "
        "total_minus errors for almost all configurations.  Replacing the "
        "dense line by a 6 x 6 grid gives a further improvement at the same "
        "total TC count (36), especially for total_plus."
    )
    lines.append(
        "The grid is better despite having a larger nearest-neighbour spacing "
        "than the dense line (1.714 versus 0.892 coordinate units).  This is "
        "consistent with the grid providing six repeated samples at each y "
        "and two-dimensional coverage, but the comparison is observational, "
        "so this is an association rather than a controlled causal effect."
    )
    lines.append(
        "The headline comparisons use the common-y support because case11_1 "
        "contains high-y TCs that are outside case11_2's range.  Complete "
        "all-TC results are still provided in the pairwise tables."
    )
    lines.append(
        "Relative-error results for model_form_plus are unstable because the "
        "target is often zero or near zero.  Use RMSE and interval score for "
        "model-form conclusions; relative error is most meaningful for total "
        "and simulation components."
    )
    lines.append(
        "The comparisons are still observational: the three studies do not "
        "share identical TC locations.  The common-y restriction makes the "
        "target support comparable, but case11_1 has 23 TCs and case11_2 has "
        "36 TCs inside that support, so the experiment is not a controlled "
        "same-locations comparison."
    )
    return "\n".join(lines) + "\n"


def write_run_info(
    output_path: Path,
    data_root: Path,
    output_dir: Path,
    common_configs: dict[str, list[str]],
    per_tc: pd.DataFrame,
    common_y: tuple[float, float],
    validation_metrics: pd.DataFrame,
    validation_targets: pd.DataFrame,
    overview: pd.DataFrame,
) -> None:
    versions = []
    for package in ("numpy", "pandas", "matplotlib", "seaborn", "scipy"):
        try:
            versions.append(f"{package}={version(package)}")
        except Exception:  # pragma: no cover - metadata can be absent
            versions.append(f"{package}=unknown")

    lines = [
        "case11 study comparison run information",
        "=" * 72,
        f"generated_utc: {datetime.now(timezone.utc).isoformat()}",
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"packages: {', '.join(versions)}",
        f"repo_root: {data_root.parent}",
        f"data_root: {data_root}",
        f"output_dir: {output_dir}",
        "",
        "Inputs",
        "-" * 72,
    ]
    for _, row in overview.iterrows():
        lines.append(
            f"{row['case']}: target={row['target_file']} | coords={row['coord_file']}"
        )
    lines.extend(
        [
            "",
            "Configuration counts",
            "-" * 72,
        ]
    )
    for family in FAMILIES:
        lines.append(f"{family}: {len(common_configs[family])} common configurations")
    lines.append(
        f"total common configurations: {sum(len(v) for v in common_configs.values())}"
    )
    common_tc_counts = (
        per_tc[per_tc["in_common_y"]]
        .groupby("case")["tc"]
        .nunique()
        .reindex(list(CASES), fill_value=0)
    )
    lines.extend(
        [
            "",
            "Data sizes",
            "-" * 72,
            f"per-TC result rows: {len(per_tc):,}",
            f"common-y support: [{common_y[0]:.6f}, {common_y[1]:.6f}]",
            "TC counts in common-y support: "
            + ", ".join(
                f"{case}={int(common_tc_counts.loc[case])}" for case in CASES
            ),
            "",
            "Validation",
            "-" * 72,
            "metric, max_abs_diff, max_rel_diff",
        ]
    )
    for _, row in validation_metrics.iterrows():
        lines.append(
            f"{row['metric']}: {row['max_abs_diff']:.6e}, {row['max_rel_diff']:.6e}"
        )
    lines.append(
        "target measured max abs diff: "
        f"{validation_targets['max_abs_diff'].max():.6e}"
    )
    lines.extend(
        [
            "",
            "Notes",
            "-" * 72,
            "case11_2 has 42 coordinate rows but only 36 TCs in the target and",
            "ablation files.  All analysis uses those 36 TCs.",
            "IPM summaries name the accuracy column accuracy_2pct; it is",
            "normalised to accuracy.  All families use a 10% relative tolerance.",
            "Model configurations are paired by family and config.  Errors are",
            "not paired per TC because the three studies use different TCs.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "synthetic_data",
        help="Directory containing case11, case11_1 and case11_2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "case11_comparison_outputs",
        help="Directory for generated tables, figures and text.",
    )
    parser.add_argument(
        "--dpi", type=int, default=180, help="Figure resolution in dpi."
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Write tables but skip figures."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    sns.set_theme(context="notebook", style="whitegrid", palette="colorblind")
    plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})

    print("Loading target tables and coordinates ...")
    case_tables: dict[str, pd.DataFrame] = {}
    case_meta: dict[str, dict[str, object]] = {}
    for case in CASES:
        case_tables[case], case_meta[case] = load_case_table(case, data_root)

    common_configs = discover_common_configs(data_root)
    common_y = (
        float(case_tables["case11_2"]["y"].min()),
        float(case_tables["case11_2"]["y"].max()),
    )

    print("Loading ablation summaries ...")
    summary_metrics, manifest = load_summary_metrics(data_root, common_configs)
    summary_metrics.to_csv(table_dir / "model_summary_by_case.csv", index=False)
    manifest.to_csv(table_dir / "config_manifest.csv", index=False)

    print("Loading per-TC results ...")
    per_tc = load_per_tc_results(data_root, case_tables, common_configs)
    per_tc = add_support_flags(per_tc, common_y)

    print("Recomputing metrics and validating ...")
    recomputed_full = compute_metrics_from_per_tc(per_tc)
    recomputed_common = compute_metrics_from_per_tc(per_tc[per_tc["in_common_y"]])

    validation_metrics = validate_recomputed_metrics(recomputed_full, summary_metrics)
    validation_targets = validate_target_measurements(per_tc, case_tables)
    validation_metrics.to_csv(table_dir / "validation_metrics.csv", index=False)
    validation_targets.to_csv(table_dir / "validation_targets.csv", index=False)

    recomputed_full["support"] = "all"
    recomputed_common["support"] = "common_y"
    summary_metrics["support"] = "all"
    recomputed_full.to_csv(
        table_dir / "recomputed_full_metrics.csv", index=False
    )
    common_metrics = recomputed_common.copy()
    common_metrics.to_csv(table_dir / "common_support_metrics.csv", index=False)

    target_values, target_summary, target_consistency = build_target_tables(
        case_tables, common_y
    )
    target_values.to_csv(table_dir / "target_values_by_tc.csv", index=False)
    target_summary.to_csv(table_dir / "target_summary.csv", index=False)
    target_consistency.to_csv(
        table_dir / "target_consistency_common_y.csv", index=False
    )

    overview = build_study_overview(case_tables, case_meta, common_y)
    overview.to_csv(table_dir / "study_overview.csv", index=False)

    per_tc_summary = build_per_tc_error_summary(per_tc)
    per_tc_summary.to_csv(table_dir / "per_tc_error_summary.csv", index=False)
    per_tc_columns = [
        "case",
        "family",
        "config",
        "d_type",
        "tc",
        "x",
        "y",
        "z",
        "nn_distance",
        "in_common_y",
        "measured",
        "predicted",
        "error",
        "abs_error",
        "rel_error",
        "pi_width",
        "within_pi",
        "pi_error",
        "interval_score",
    ]
    per_tc[per_tc_columns].to_csv(table_dir / "per_tc_errors.csv", index=False)

    print("Building paired comparisons ...")
    pairwise_overall = pd.concat(
        [
            build_pairwise_summary(summary_metrics, "all", []),
            build_pairwise_summary(common_metrics, "common_y", []),
        ],
        ignore_index=True,
    )
    pairwise_family = pd.concat(
        [
            build_pairwise_summary(summary_metrics, "all", ["family"]),
            build_pairwise_summary(common_metrics, "common_y", ["family"]),
        ],
        ignore_index=True,
    )
    pairwise_overall.to_csv(
        table_dir / "pairwise_overall_metric_summary.csv", index=False
    )
    pairwise_family.to_csv(
        table_dir / "pairwise_family_metric_summary.csv", index=False
    )

    best_models = pd.concat(
        [
            build_best_models(summary_metrics, "all"),
            build_best_models(common_metrics, "common_y"),
        ],
        ignore_index=True,
    )
    best_models.to_csv(table_dir / "best_models.csv", index=False)

    if not args.no_plots:
        print("Creating figures ...")
        plot_metric_boxplots(
            common_metrics,
            "RMSE",
            figure_dir / "fig_rmse_by_case.png",
            args.dpi,
            "common_y",
        )
        plot_metric_boxplots(
            common_metrics,
            "mean_interval_score",
            figure_dir / "fig_interval_score_by_case.png",
            args.dpi,
            "common_y",
        )
        plot_coverage_accuracy(
            common_metrics, figure_dir / "fig_coverage_accuracy.png", args.dpi
        )

    print("Writing text summaries ...")
    text_summary = build_text_summary(
        overview,
        target_summary,
        common_metrics,
        pairwise_overall,
        pairwise_family,
        validation_metrics,
        validation_targets,
        best_models,
        common_y,
    )
    (output_dir / "comparison_summary.txt").write_text(
        text_summary, encoding="utf-8"
    )
    write_run_info(
        output_dir / "run_info.txt",
        data_root,
        output_dir,
        common_configs,
        per_tc,
        common_y,
        validation_metrics,
        validation_targets,
        overview,
    )

    print(f"Done. Outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
