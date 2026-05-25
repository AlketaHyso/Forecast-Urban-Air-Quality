from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "outputs" / "forecast_benchmark" / "tables"
ENHANCED_DIR = PROJECT_ROOT / "outputs" / "enhanced_forecast_benchmark" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "main_benchmark_bootstrap"
TABLE_DIR = OUTPUT_DIR / "tables"

N_BOOT = 20000
RANDOM_STATE = 42


def bootstrap_mean_ci(diffs: np.ndarray, n_boot: int, random_state: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    samples = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(diffs.mean()), float(lo), float(hi)


def load_comparisons() -> pd.DataFrame:
    comparisons = pd.read_csv(ENHANCED_DIR / "comparison_vs_baseline.csv")
    return comparisons[
        [
            "target",
            "target_label_x",
            "horizon_days",
            "enhanced_best_model",
            "baseline_best_model",
            "enhanced_mean_mae",
            "baseline_mean_mae",
            "mae_improvement",
            "enhanced_mean_mape",
            "baseline_mean_mape",
            "mape_improvement",
        ]
    ].rename(columns={"target_label_x": "target_label"})


def paired_city_bootstrap(
    comparisons: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    enhanced_metrics: pd.DataFrame,
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict] = []

    for row in comparisons.itertuples(index=False):
        base_subset = baseline_metrics[
            (baseline_metrics["target"] == row.target)
            & (baseline_metrics["horizon_days"] == row.horizon_days)
            & (baseline_metrics["model"] == row.baseline_best_model)
        ][["city", "mae", "rmse", "mape", "smape"]].rename(
            columns={
                "mae": "baseline_mae",
                "rmse": "baseline_rmse",
                "mape": "baseline_mape",
                "smape": "baseline_smape",
            }
        )

        enhanced_subset = enhanced_metrics[
            (enhanced_metrics["target"] == row.target)
            & (enhanced_metrics["horizon_days"] == row.horizon_days)
            & (enhanced_metrics["model"] == row.enhanced_best_model)
        ][["city", "mae", "rmse", "mape", "smape"]].rename(
            columns={
                "mae": "enhanced_mae",
                "rmse": "enhanced_rmse",
                "mape": "enhanced_mape",
                "smape": "enhanced_smape",
            }
        )

        merged = enhanced_subset.merge(base_subset, on="city", how="inner").sort_values("city").reset_index(drop=True)
        mae_diffs = merged["enhanced_mae"].to_numpy(dtype=float) - merged["baseline_mae"].to_numpy(dtype=float)
        mape_diffs = merged["enhanced_mape"].to_numpy(dtype=float) - merged["baseline_mape"].to_numpy(dtype=float)

        point_mae, lo_mae, hi_mae = bootstrap_mean_ci(mae_diffs, n_boot, random_state)
        point_mape, lo_mape, hi_mape = bootstrap_mean_ci(mape_diffs, n_boot, random_state + 1)

        rows.append(
            {
                "target": row.target,
                "target_label": row.target_label,
                "horizon_days": int(row.horizon_days),
                "enhanced_model": row.enhanced_best_model,
                "baseline_model": row.baseline_best_model,
                "unit": "city_level_mae_difference",
                "n_units": int(len(merged)),
                "point_estimate_enhanced_minus_baseline_mae": point_mae,
                "ci_lower_95_enhanced_minus_baseline_mae": lo_mae,
                "ci_upper_95_enhanced_minus_baseline_mae": hi_mae,
                "point_estimate_baseline_minus_enhanced_mae": -point_mae,
                "ci_lower_95_baseline_minus_enhanced_mae": -hi_mae,
                "ci_upper_95_baseline_minus_enhanced_mae": -lo_mae,
                "point_estimate_enhanced_minus_baseline_mape": point_mape,
                "ci_lower_95_enhanced_minus_baseline_mape": lo_mape,
                "ci_upper_95_enhanced_minus_baseline_mape": hi_mape,
                "point_estimate_baseline_minus_enhanced_mape": -point_mape,
                "ci_lower_95_baseline_minus_enhanced_mape": -hi_mape,
                "ci_upper_95_baseline_minus_enhanced_mape": -lo_mape,
            }
        )

    return pd.DataFrame(rows)


def cluster_city_bootstrap(
    comparisons: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    enhanced_predictions: pd.DataFrame,
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict] = []

    baseline_predictions = baseline_predictions.rename(columns={"training_end_date": "feature_date"})

    for row in comparisons.itertuples(index=False):
        base_subset = baseline_predictions[
            (baseline_predictions["target"] == row.target)
            & (baseline_predictions["horizon_days"] == row.horizon_days)
            & (baseline_predictions["model"] == row.baseline_best_model)
        ][["city", "feature_date", "target_date", "abs_error", "actual"]].rename(
            columns={"abs_error": "baseline_abs_error"}
        )

        enhanced_subset = enhanced_predictions[
            (enhanced_predictions["target"] == row.target)
            & (enhanced_predictions["horizon_days"] == row.horizon_days)
            & (enhanced_predictions["model"] == row.enhanced_best_model)
        ][["city", "feature_date", "target_date", "abs_error", "actual"]].rename(
            columns={"abs_error": "enhanced_abs_error"}
        )

        merged = enhanced_subset.merge(
            base_subset,
            on=["city", "feature_date", "target_date", "actual"],
            how="inner",
        ).sort_values(["city", "feature_date", "target_date"]).reset_index(drop=True)
        merged["abs_error_diff"] = merged["enhanced_abs_error"] - merged["baseline_abs_error"]

        groups = {city_name: grp for city_name, grp in merged.groupby("city")}
        city_names = list(groups)
        rng = np.random.default_rng(random_state + int(row.horizon_days))
        samples: list[float] = []
        for _ in range(n_boot):
            chosen = rng.choice(city_names, size=len(city_names), replace=True)
            vals = np.concatenate([groups[city]["abs_error_diff"].to_numpy(dtype=float) for city in chosen])
            samples.append(float(vals.mean()))

        point = float(merged["abs_error_diff"].mean())
        lo, hi = np.quantile(np.asarray(samples, dtype=float), [0.025, 0.975])
        rows.append(
            {
                "target": row.target,
                "target_label": row.target_label,
                "horizon_days": int(row.horizon_days),
                "enhanced_model": row.enhanced_best_model,
                "baseline_model": row.baseline_best_model,
                "unit": "city_clustered_forecast_cases",
                "n_units": int(len(city_names)),
                "n_forecast_cases": int(len(merged)),
                "point_estimate_enhanced_minus_baseline_mae": point,
                "ci_lower_95_enhanced_minus_baseline_mae": float(lo),
                "ci_upper_95_enhanced_minus_baseline_mae": float(hi),
                "point_estimate_baseline_minus_enhanced_mae": -point,
                "ci_lower_95_baseline_minus_enhanced_mae": float(-hi),
                "ci_upper_95_baseline_minus_enhanced_mae": float(-lo),
            }
        )

    return pd.DataFrame(rows)


def build_summary_markdown(city_boot: pd.DataFrame) -> str:
    lines = [
        "# Main Benchmark Bootstrap Summary",
        "",
        "This file summarizes paired bootstrap comparisons for the main Table 4 benchmark using the globally selected best enhanced and best baseline model for each target-horizon combination.",
        "",
    ]

    for target_name, target_block in city_boot.groupby("target_label"):
        lines.append(f"## {target_name}")
        for row in target_block.sort_values("horizon_days").itertuples(index=False):
            lines.append(
                "- "
                f"{int(row.horizon_days)}-day: {row.enhanced_model} vs {row.baseline_model}; "
                f"baseline minus enhanced MAE = {row.point_estimate_baseline_minus_enhanced_mae:.4f} "
                f"(95% CI {row.ci_lower_95_baseline_minus_enhanced_mae:.4f} to {row.ci_upper_95_baseline_minus_enhanced_mae:.4f})."
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    comparisons = load_comparisons()
    baseline_metrics = pd.read_csv(BASELINE_DIR / "metrics_by_city.csv")
    enhanced_metrics = pd.read_csv(ENHANCED_DIR / "metrics_by_city.csv")
    baseline_predictions = pd.read_csv(BASELINE_DIR / "rolling_predictions.csv", parse_dates=["training_end_date", "target_date"])
    enhanced_predictions = pd.read_csv(ENHANCED_DIR / "rolling_predictions.csv", parse_dates=["feature_date", "target_date"])

    city_boot = paired_city_bootstrap(comparisons, baseline_metrics, enhanced_metrics, N_BOOT, RANDOM_STATE)
    cluster_boot = cluster_city_bootstrap(comparisons, baseline_predictions, enhanced_predictions, N_BOOT, RANDOM_STATE)
    combined = comparisons.merge(
        city_boot[
            [
                "target",
                "horizon_days",
                "point_estimate_enhanced_minus_baseline_mae",
                "ci_lower_95_enhanced_minus_baseline_mae",
                "ci_upper_95_enhanced_minus_baseline_mae",
                "point_estimate_baseline_minus_enhanced_mae",
                "ci_lower_95_baseline_minus_enhanced_mae",
                "ci_upper_95_baseline_minus_enhanced_mae",
                "point_estimate_enhanced_minus_baseline_mape",
                "ci_lower_95_enhanced_minus_baseline_mape",
                "ci_upper_95_enhanced_minus_baseline_mape",
                "point_estimate_baseline_minus_enhanced_mape",
                "ci_lower_95_baseline_minus_enhanced_mape",
                "ci_upper_95_baseline_minus_enhanced_mape",
            ]
        ],
        on=["target", "horizon_days"],
        how="left",
    )

    city_boot.to_csv(TABLE_DIR / "bootstrap_city_level_summary.csv", index=False)
    cluster_boot.to_csv(TABLE_DIR / "bootstrap_city_cluster_summary.csv", index=False)
    combined.to_csv(TABLE_DIR / "comparison_vs_baseline_with_bootstrap.csv", index=False)

    with (OUTPUT_DIR / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write(build_summary_markdown(city_boot))

    metadata = {
        "n_bootstrap_resamples": N_BOOT,
        "random_state": RANDOM_STATE,
        "source_comparison_table": str(ENHANCED_DIR / "comparison_vs_baseline.csv"),
    }
    with (OUTPUT_DIR / "bootstrap_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved main benchmark bootstrap outputs to:")
    for name in [
        "bootstrap_city_level_summary.csv",
        "bootstrap_city_cluster_summary.csv",
        "comparison_vs_baseline_with_bootstrap.csv",
    ]:
        print(f"  - {TABLE_DIR / name}")


if __name__ == "__main__":
    main()
