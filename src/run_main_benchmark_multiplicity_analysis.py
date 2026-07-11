from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "outputs" / "forecast_benchmark" / "tables"
ENHANCED_DIR = PROJECT_ROOT / "outputs" / "enhanced_forecast_benchmark" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "main_benchmark_multiplicity"
TABLE_DIR = OUTPUT_DIR / "tables"

N_BOOT = 20000
RANDOM_STATE = 42
FAMILY_ALPHA = 0.05
N_PRIMARY_COMPARISONS = 6
PER_COMPARISON_ALPHA = FAMILY_ALPHA / N_PRIMARY_COMPARISONS


def bootstrap_mean_samples(diffs: np.ndarray, n_boot: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)


def quantile_interval(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi


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
        ]
    ].rename(columns={"target_label_x": "target_label"})


def classify_direction(ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return "enhanced_supported"
    if ci_high < 0:
        return "baseline_supported"
    return "not_clearly_separated"


def paired_city_adjusted_summary(
    comparisons: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    enhanced_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for idx, row in enumerate(comparisons.itertuples(index=False), start=1):
        base_subset = baseline_metrics[
            (baseline_metrics["target"] == row.target)
            & (baseline_metrics["horizon_days"] == row.horizon_days)
            & (baseline_metrics["model"] == row.baseline_best_model)
        ][["city", "mae"]].rename(columns={"mae": "baseline_mae"})

        enhanced_subset = enhanced_metrics[
            (enhanced_metrics["target"] == row.target)
            & (enhanced_metrics["horizon_days"] == row.horizon_days)
            & (enhanced_metrics["model"] == row.enhanced_best_model)
        ][["city", "mae"]].rename(columns={"mae": "enhanced_mae"})

        merged = enhanced_subset.merge(base_subset, on="city", how="inner").sort_values("city").reset_index(drop=True)
        baseline_minus_enhanced = merged["baseline_mae"].to_numpy(dtype=float) - merged["enhanced_mae"].to_numpy(dtype=float)
        samples = bootstrap_mean_samples(baseline_minus_enhanced, N_BOOT, RANDOM_STATE + idx)

        ci95_low, ci95_high = quantile_interval(samples, 0.05)
        bonf_low, bonf_high = quantile_interval(samples, PER_COMPARISON_ALPHA)

        rows.append(
            {
                "target": row.target,
                "target_label": row.target_label,
                "horizon_days": int(row.horizon_days),
                "enhanced_model": row.enhanced_best_model,
                "baseline_model": row.baseline_best_model,
                "unit": "city_level_mae_difference",
                "n_units": int(len(merged)),
                "point_estimate_baseline_minus_enhanced_mae": float(baseline_minus_enhanced.mean()),
                "ci95_lower_baseline_minus_enhanced_mae": ci95_low,
                "ci95_upper_baseline_minus_enhanced_mae": ci95_high,
                "bonferroni_lower_baseline_minus_enhanced_mae": bonf_low,
                "bonferroni_upper_baseline_minus_enhanced_mae": bonf_high,
                "ci95_classification": classify_direction(ci95_low, ci95_high),
                "bonferroni_classification": classify_direction(bonf_low, bonf_high),
            }
        )

    return pd.DataFrame(rows)


def clustered_case_adjusted_summary(
    comparisons: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    enhanced_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    baseline_predictions = baseline_predictions.rename(columns={"training_end_date": "feature_date"})

    for idx, row in enumerate(comparisons.itertuples(index=False), start=1):
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
        merged["baseline_minus_enhanced_abs_error"] = (
            merged["baseline_abs_error"] - merged["enhanced_abs_error"]
        )

        groups = {city_name: grp for city_name, grp in merged.groupby("city")}
        city_names = list(groups)
        rng = np.random.default_rng(RANDOM_STATE + 100 + idx)
        samples: list[float] = []
        for _ in range(N_BOOT):
            chosen = rng.choice(city_names, size=len(city_names), replace=True)
            vals = np.concatenate(
                [groups[city]["baseline_minus_enhanced_abs_error"].to_numpy(dtype=float) for city in chosen]
            )
            samples.append(float(vals.mean()))

        sample_arr = np.asarray(samples, dtype=float)
        ci95_low, ci95_high = quantile_interval(sample_arr, 0.05)
        bonf_low, bonf_high = quantile_interval(sample_arr, PER_COMPARISON_ALPHA)

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
                "point_estimate_baseline_minus_enhanced_mae": float(merged["baseline_minus_enhanced_abs_error"].mean()),
                "ci95_lower_baseline_minus_enhanced_mae": ci95_low,
                "ci95_upper_baseline_minus_enhanced_mae": ci95_high,
                "bonferroni_lower_baseline_minus_enhanced_mae": bonf_low,
                "bonferroni_upper_baseline_minus_enhanced_mae": bonf_high,
                "ci95_classification": classify_direction(ci95_low, ci95_high),
                "bonferroni_classification": classify_direction(bonf_low, bonf_high),
            }
        )

    return pd.DataFrame(rows)


def build_summary_markdown(city_df: pd.DataFrame, cluster_df: pd.DataFrame) -> str:
    lines = [
        "# Main Benchmark Multiplicity Check",
        "",
        "This summary applies a conservative Bonferroni adjustment across the six pre-specified main target-horizon benchmark contrasts.",
        f"- Family-wise alpha: {FAMILY_ALPHA:.2f}",
        f"- Number of primary contrasts: {N_PRIMARY_COMPARISONS}",
        f"- Per-comparison alpha: {PER_COMPARISON_ALPHA:.6f}",
        "",
        "## Clustered bootstrap summary (primary uncertainty view)",
    ]

    for row in cluster_df.sort_values(["target", "horizon_days"]).itertuples(index=False):
        lines.append(
            "- "
            f"{row.target_label}, h={row.horizon_days}: baseline minus enhanced MAE = "
            f"{row.point_estimate_baseline_minus_enhanced_mae:.4f}; 95% CI "
            f"{row.ci95_lower_baseline_minus_enhanced_mae:.4f} to {row.ci95_upper_baseline_minus_enhanced_mae:.4f}; "
            f"Bonferroni-adjusted CI {row.bonferroni_lower_baseline_minus_enhanced_mae:.4f} to "
            f"{row.bonferroni_upper_baseline_minus_enhanced_mae:.4f}; "
            f"classification: {row.bonferroni_classification}."
        )

    lines.append("")
    lines.append("## City-level paired bootstrap summary (supporting uncertainty view)")
    for row in city_df.sort_values(["target", "horizon_days"]).itertuples(index=False):
        lines.append(
            "- "
            f"{row.target_label}, h={row.horizon_days}: baseline minus enhanced MAE = "
            f"{row.point_estimate_baseline_minus_enhanced_mae:.4f}; 95% CI "
            f"{row.ci95_lower_baseline_minus_enhanced_mae:.4f} to {row.ci95_upper_baseline_minus_enhanced_mae:.4f}; "
            f"Bonferroni-adjusted CI {row.bonferroni_lower_baseline_minus_enhanced_mae:.4f} to "
            f"{row.bonferroni_upper_baseline_minus_enhanced_mae:.4f}; "
            f"classification: {row.bonferroni_classification}."
        )

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    comparisons = load_comparisons()
    baseline_metrics = pd.read_csv(BASELINE_DIR / "metrics_by_city.csv")
    enhanced_metrics = pd.read_csv(ENHANCED_DIR / "metrics_by_city.csv")
    baseline_predictions = pd.read_csv(
        BASELINE_DIR / "rolling_predictions.csv",
        parse_dates=["training_end_date", "target_date"],
    )
    enhanced_predictions = pd.read_csv(
        ENHANCED_DIR / "rolling_predictions.csv",
        parse_dates=["feature_date", "target_date"],
    )

    city_df = paired_city_adjusted_summary(comparisons, baseline_metrics, enhanced_metrics)
    cluster_df = clustered_case_adjusted_summary(comparisons, baseline_predictions, enhanced_predictions)

    city_df.to_csv(TABLE_DIR / "city_level_bonferroni_summary.csv", index=False)
    cluster_df.to_csv(TABLE_DIR / "clustered_bonferroni_summary.csv", index=False)

    metadata = {
        "n_bootstrap_resamples": N_BOOT,
        "random_state": RANDOM_STATE,
        "family_alpha": FAMILY_ALPHA,
        "n_primary_comparisons": N_PRIMARY_COMPARISONS,
        "per_comparison_alpha": PER_COMPARISON_ALPHA,
        "interpretation_note": (
            "Positive baseline-minus-enhanced values favor the enhanced benchmark. "
            "Negative values favor the local baseline benchmark."
        ),
    }
    with (OUTPUT_DIR / "multiplicity_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    with (OUTPUT_DIR / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write(build_summary_markdown(city_df, cluster_df))

    print("Saved multiplicity-analysis outputs to:")
    print(f"  - {TABLE_DIR / 'city_level_bonferroni_summary.csv'}")
    print(f"  - {TABLE_DIR / 'clustered_bonferroni_summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'multiplicity_metadata.json'}")
    print(f"  - {OUTPUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
