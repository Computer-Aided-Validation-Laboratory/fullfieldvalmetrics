# case11 / case11_1 / case11_2 comparison

This folder contains the comparison code for the three case11 thermocouple
layouts:

| Study | Layout | TCs used | y range | z range |
| --- | --- | ---: | --- | --- |
| `case11` | sparse line | 7 | 4.125 - 28.875 | 6.0 only |
| `case11_1` | dense line | 36 | 0.892 - 32.108 | 6.0 only |
| `case11_2` | 6 x 6 rectangular grid | 36 | 4.125 - 24.750 | 1.714 - 10.286 |

`case11_2` has 42 rows in `PointSensorCoords_exp.csv`, but its target and
ablation files contain only 36 TCs. The comparison therefore uses the 36 TCs
that are present in both the target table and the ablation results.

## Run

```bash
~/anaconda3/envs/pyvale/bin/python scripts/plots/compare_case11_studies.py
```

By default the script reads:

- `synthetic_data/case11/pyvale-output`
- `synthetic_data/case11_1/pyvale-output`
- `synthetic_data/case11_2/pyvale-output`

and writes to `scripts/plots/case11_comparison_outputs`. Optional arguments:

```bash
--data-root PATH     # different synthetic_data directory
--output-dir PATH    # different output directory
--dpi 180            # figure resolution
--no-plots           # tables only
```

No input files are modified.

## What is compared

### Target fields

- Target uncertainty components: `sim_plus`, `sim_minus`,
  `model_form_plus`, `model_form_minus`, `total_plus`, `total_minus`.
- Full-sampling statistics, plus a **common-y support** from
  `4.125` to `24.750`. The common-y restriction avoids comparing case11_1's
  high-y TCs against a grid that does not extend that far.
- At the six shared y levels, `sim_plus` and `sim_minus` agree to within
  `~2e-8 °C` between case11 and case11_2. `total_plus`/`total_minus` differ
  by at most `~0.17 °C` because case11_2 averages over six z slices.

### Interpolation performance

- 108 model configurations are common to all three studies:
  - 16 regularised spatial, 4 regularised temperature,
  - 32 GPR spatial, 16 GPR temperature,
  - 20 IPM spatial, 20 IPM temperature.
- Metrics from the saved `ablation_summary_*.csv` files: MAE, RMSE,
  relative error, prediction-interval width/error, coverage, interval score
  and 10% threshold accuracy.
- Per-TC results are reloaded from `ablation_results/*.csv` to recompute
  metrics and to build spatial error profiles. Recomputed metrics are checked
  against the saved summaries; all maximum relative differences are below
  `2e-12`.
- Configurations are paired by `(family, config, d_type)`. Errors are **not**
  paired per TC across studies because the studies use different TC locations.
- For interval coverage, the comparison uses `abs(coverage - 0.95)`, where
  lower is better. `accuracy` is higher-is-better; all error and interval
  metrics are lower-is-better.

IPM summary files call the 10% accuracy column `accuracy_2pct`; it is
normalised to `accuracy` in the output tables.

## Output tables

All tables are in `case11_comparison_outputs/tables`:

| File | Contents |
| --- | --- |
| `study_overview.csv` | Layout, coordinate ranges, spacing, nearest-neighbour distances and target means per study. |
| `target_values_by_tc.csv` | Target values and coordinates for every used TC. |
| `target_summary.csv` | Target statistics by case, support (`all`, `common_y`) and component. |
| `target_consistency_common_y.csv` | case11 vs case11_2 target differences at shared y levels. |
| `model_summary_by_case.csv` | Saved ablation-summary metrics for all common configurations. |
| `recomputed_full_metrics.csv` | Metrics recomputed from per-TC result files on all TCs. |
| `common_support_metrics.csv` | Metrics recomputed on the common-y support. |
| `pairwise_overall_metric_summary.csv` | Paired comparison statistics for each case pair, `d_type` and metric. |
| `pairwise_family_metric_summary.csv` | Same statistics split by model family. |
| `best_models.csv` | Best configuration per case, family, `d_type`, metric and support. |
| `per_tc_error_summary.csv` | Median/mean absolute error and interval statistics per TC and `d_type`. |
| `per_tc_errors.csv` | Raw per-TC results for all common configurations. |
| `validation_metrics.csv` | Recomputed-vs-saved summary differences. |
| `validation_targets.csv` | Result-file measured values vs target-table values. |
| `config_manifest.csv` | Input summary-file paths for every case/family/config. |

## Output figures

All figures are in `case11_comparison_outputs/figures`:

| File | Contents |
| --- | --- |
| `fig_rmse_by_case.png` | RMSE distributions across configurations (common-y). |
| `fig_interval_score_by_case.png` | Interval-score distributions across configurations. |
| `fig_coverage_accuracy.png` | Coverage error and 10% accuracy. |

## Key results (common-y support)

Median values across the 108 common configurations:

| Component | Study | RMSE | Interval score | Relative error | 10% accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `total_plus` | case11 | 4.581 | 31.58 | 0.729 | 0.167 |
| `total_plus` | case11_1 | 2.937 | 10.88 | 0.460 | 0.217 |
| `total_plus` | case11_2 | 2.312 | 9.30 | 0.287 | 0.306 |
| `total_minus` | case11 | 5.062 | 21.11 | 0.505 | 0.167 |
| `total_minus` | case11_1 | 2.583 | 9.86 | 0.261 | 0.261 |
| `total_minus` | case11_2 | 1.842 | 8.60 | 0.188 | 0.361 |

Pairwise changes for `total_plus` (win rate = second study better):

- case11 -> case11_1: RMSE `-30.0%` (87%), interval score `-61.1%` (89%).
- case11_1 -> case11_2: RMSE `-31.6%` (81%), interval score `-38.4%` (80%).
- case11 -> case11_2: RMSE `-54.8%` (99%), interval score `-67.4%` (97%).

For `total_minus`, the corresponding RMSE changes are `-37.1%`, `-31.6%`
and `-53.0%`; the interval-score changes are `-44.2%`, `-43.2%` and
`-62.3%`.

All six model families show a family-level RMSE win rate above 0.5 for every
pair and both total components. The best common-y `total_plus` RMSE values are
`1.705 °C` (case11, GPR-Spatial), `0.107 °C` (case11_1, GPR-Spatial) and
`0.052 °C` (case11_2, GPR-Temp).

## Caveats

- The three studies do **not** share identical TC locations. The common-y
  restriction makes the target support comparable, but case11_1 has 23 TCs
  and case11_2 has 36 TCs inside that support. The comparison is observational,
  not a controlled same-location experiment.
- The grid is better than the dense line even though its nearest-neighbour
  spacing is larger (`1.714` vs `0.892`). This is consistent with six repeated
  samples per y level and two-dimensional coverage, but it is an association,
  not an isolated causal effect.
- Relative error for `model_form_plus` is unstable because its target is often
  zero or near zero. Use RMSE and interval score for model-form conclusions.
- The target tables for case11_2 contain 36 TCs, not the 42 rows present in
  the coordinate file. The unused final y level (`28.875`) is not part of the
  case11_2 ablation results.
