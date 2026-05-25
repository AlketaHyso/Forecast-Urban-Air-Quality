from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR_NAME = "limited_history_robustness"
DEFAULT_OUTPUT_DIR_NAME = "limited_history_bootstrap"

N_BOOT = 20000
RANDOM_STATE = 42

COMPARISONS = [
    ("few_shot_cross_city_90d", "local_only", "Few-shot vs Local-only"),
    ("full_pooled_reference", "local_only", "Full pooled vs Local-only"),
    ("few_shot_cross_city_90d", "full_pooled_reference", "Few-shot vs Full pooled"),
]


def paired_city_bootstrap(metrics: pd.DataFrame, n_boot: int, random_state: int) -> pd.DataFrame:
    pivot = metrics.pivot_table(index="city", columns="regime", values="mae").reset_index()
    rng = np.random.default_rng(random_state)
    rows: list[dict] = []

    for regime_a, regime_b, label in COMPARISONS:
        diffs = pivot[regime_a].to_numpy(dtype=float) - pivot[regime_b].to_numpy(dtype=float)
        samples = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
        lo, hi = np.quantile(samples, [0.025, 0.975])
        rows.append(
            {
                "comparison": label,
                "regime_a": regime_a,
                "regime_b": regime_b,
                "unit": "city_level_mae_difference",
                "n_units": int(len(diffs)),
                "point_estimate_mae_diff": float(diffs.mean()),
                "ci_lower_95": float(lo),
                "ci_upper_95": float(hi),
            }
        )
    return pd.DataFrame(rows)


def cluster_city_bootstrap(predictions: pd.DataFrame, n_boot: int, random_state: int) -> pd.DataFrame:
    wide = predictions.pivot_table(
        index=["city", "feature_date", "target_date"],
        columns="regime",
        values="abs_error",
    ).reset_index()
    groups = {city_name: grp for city_name, grp in wide.groupby("city")}
    city_names = list(groups)
    rng = np.random.default_rng(random_state)
    rows: list[dict] = []

    for regime_a, regime_b, label in COMPARISONS:
        point = float((wide[regime_a] - wide[regime_b]).mean())
        samples: list[float] = []
        for _ in range(n_boot):
            chosen = rng.choice(city_names, size=len(city_names), replace=True)
            vals = np.concatenate([(groups[city][regime_a] - groups[city][regime_b]).to_numpy(dtype=float) for city in chosen])
            samples.append(float(vals.mean()))
        lo, hi = np.quantile(np.asarray(samples, dtype=float), [0.025, 0.975])
        rows.append(
            {
                "comparison": label,
                "regime_a": regime_a,
                "regime_b": regime_b,
                "unit": "city_clustered_forecast_cases",
                "n_units": int(len(city_names)),
                "point_estimate_mae_diff": point,
                "ci_lower_95": float(lo),
                "ci_upper_95": float(hi),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir-name", default=DEFAULT_SOURCE_DIR_NAME)
    parser.add_argument("--output-dir-name", default=DEFAULT_OUTPUT_DIR_NAME)
    args = parser.parse_args()

    source_dir = PROJECT_ROOT / "outputs" / args.source_dir_name / "tables"
    output_dir = PROJECT_ROOT / "outputs" / args.output_dir_name
    table_dir = output_dir / "tables"

    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(source_dir / "metrics_by_city.csv")
    predictions = pd.read_csv(source_dir / "rolling_predictions.csv", parse_dates=["feature_date", "target_date"])

    city_boot = paired_city_bootstrap(metrics, N_BOOT, RANDOM_STATE)
    cluster_boot = cluster_city_bootstrap(predictions, N_BOOT, RANDOM_STATE)
    summary = pd.concat([city_boot, cluster_boot], axis=0, ignore_index=True)

    city_boot.to_csv(table_dir / "bootstrap_city_level_summary.csv", index=False)
    cluster_boot.to_csv(table_dir / "bootstrap_city_cluster_summary.csv", index=False)
    summary.to_csv(table_dir / "bootstrap_summary_all.csv", index=False)

    metadata = {
        "n_bootstrap_resamples": N_BOOT,
        "random_state": RANDOM_STATE,
        "comparisons": [label for _, _, label in COMPARISONS],
    }
    with (output_dir / "bootstrap_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved limited-history bootstrap outputs to:")
    for name in [
        "bootstrap_city_level_summary.csv",
        "bootstrap_city_cluster_summary.csv",
        "bootstrap_summary_all.csv",
    ]:
        print(f"  - {table_dir / name}")


if __name__ == "__main__":
    main()
