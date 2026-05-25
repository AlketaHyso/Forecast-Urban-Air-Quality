from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from run_enhanced_forecasting_benchmark import (
    BASELINE_MODEL_PRIORITY,
    EVAL_STRIDE_DAYS,
    INPUT_PATH,
    MIN_TRAIN_ROWS,
    RANDOM_STATE,
    TARGETS,
    TARGET_WINDOWS,
    TRAIN_WINDOW_DAYS,
    build_base_features,
    build_models,
    mae,
    make_horizon_frame,
    mape,
    predict_model,
    rmse,
    smape,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "outputs" / "forecast_benchmark" / "tables"
ENHANCED_DIR = PROJECT_ROOT / "outputs" / "enhanced_forecast_benchmark" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "enhanced_feature_group_ablation"
TABLE_DIR = OUTPUT_DIR / "tables"

ABLATION_SPECS = [
    (
        "full_features",
        "Full Feature Set",
        {
            "include_weather": True,
            "include_target_context": True,
            "include_city_indicators": True,
        },
    ),
    (
        "no_weather",
        "No Weather Covariates",
        {
            "include_weather": False,
            "include_target_context": True,
            "include_city_indicators": True,
        },
    ),
    (
        "no_cross_pollutant_context",
        "No Cross-Pollutant Context",
        {
            "include_weather": True,
            "include_target_context": False,
            "include_city_indicators": True,
        },
    ),
    (
        "no_city_indicators",
        "No City Indicators",
        {
            "include_weather": True,
            "include_target_context": True,
            "include_city_indicators": False,
        },
    ),
]
ABLATION_PRIORITY = {name: idx for idx, (name, _, _) in enumerate(ABLATION_SPECS)}
DEFAULT_MODEL = "HistGBM Global Features"
VALID_MODELS = ["Ridge Global Features", "HistGBM Global Features"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run feature-group ablation experiments for the enhanced pooled benchmark."
    )
    parser.add_argument(
        "--model-name",
        choices=VALID_MODELS,
        default=DEFAULT_MODEL,
        help="Enhanced model to evaluate under the ablation regimes.",
    )
    return parser.parse_args()


def build_baseline_reference(baseline_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        baseline_summary.assign(model_priority=baseline_summary["model"].map(BASELINE_MODEL_PRIORITY))
        .sort_values(["target", "horizon_days", "mean_mae", "mean_rank", "model_priority"])
        .groupby(["target", "horizon_days"], as_index=False)
        .first()
        .drop(columns=["model_priority"])
        .rename(
            columns={
                "model": "baseline_best_model",
                "mean_mae": "baseline_mean_mae",
                "mean_rmse": "baseline_mean_rmse",
                "mean_mape": "baseline_mean_mape",
                "mean_smape": "baseline_mean_smape",
            }
        )
    )


def run_ablation(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    daily = daily.sort_values(["city", "date"]).reset_index(drop=True)
    unique_dates = sorted(daily["date"].unique())

    prediction_rows: list[dict] = []
    baseline_summary = pd.read_csv(BASELINE_DIR / "model_summary.csv")
    enhanced_summary = pd.read_csv(ENHANCED_DIR / "model_summary.csv")

    for target_col, target_label in TARGETS.items():
        for ablation_name, ablation_label, feature_flags in ABLATION_SPECS:
            base_features, feature_columns = build_base_features(daily, target_col, **feature_flags)
            for horizon in TARGET_WINDOWS:
                frame = make_horizon_frame(base_features, target_col, horizon, feature_columns)

                origin_dates = [
                    d
                    for d in unique_dates
                    if d >= unique_dates[TRAIN_WINDOW_DAYS - 1] and d <= unique_dates[-(horizon + 1)]
                ][::EVAL_STRIDE_DAYS]

                model_template = build_models()[model_name]

                for origin_date in origin_dates:
                    train_mask = (
                        (frame["target_date"] <= origin_date)
                        & (frame["target_date"] > origin_date - pd.Timedelta(days=TRAIN_WINDOW_DAYS))
                    )
                    test_mask = frame["date"] == origin_date

                    train_df = frame.loc[train_mask].copy()
                    test_df = frame.loc[test_mask].copy()
                    if len(train_df) < MIN_TRAIN_ROWS or test_df.empty:
                        continue

                    x_train = train_df[feature_columns]
                    y_train = train_df["target"]
                    x_test = test_df[feature_columns]
                    model_obj = clone(model_template)
                    preds = predict_model(model_name, model_obj, x_train, y_train, x_test)

                    for idx, (_, row) in enumerate(test_df.iterrows()):
                        prediction_rows.append(
                            {
                                "target": target_col,
                                "target_label": target_label,
                                "city": row["city"],
                                "model": model_name,
                                "feature_regime": ablation_name,
                                "feature_regime_label": ablation_label,
                                "horizon_days": horizon,
                                "feature_date": row["date"],
                                "target_date": row["target_date"],
                                "actual": float(row["target"]),
                                "predicted": float(preds[idx]),
                            }
                        )

    predictions = pd.DataFrame(prediction_rows)
    predictions["abs_error"] = np.abs(predictions["actual"] - predictions["predicted"])
    predictions["squared_error"] = (predictions["actual"] - predictions["predicted"]) ** 2

    metrics_by_city = (
        predictions.groupby(
            ["target", "target_label", "city", "model", "feature_regime", "feature_regime_label", "horizon_days"],
            as_index=False,
        )
        .apply(
            lambda frame: pd.Series(
                {
                    "n_predictions": len(frame),
                    "mae": mae(frame["actual"].to_numpy(), frame["predicted"].to_numpy()),
                    "rmse": rmse(frame["actual"].to_numpy(), frame["predicted"].to_numpy()),
                    "mape": mape(frame["actual"].to_numpy(), frame["predicted"].to_numpy()),
                    "smape": smape(frame["actual"].to_numpy(), frame["predicted"].to_numpy()),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    metrics_by_city["rank_mae_within_target_horizon"] = (
        metrics_by_city.groupby(["target", "city", "horizon_days"])["mae"].rank(method="average")
    )

    regime_summary = (
        metrics_by_city.groupby(
            ["target", "target_label", "model", "feature_regime", "feature_regime_label", "horizon_days"],
            as_index=False,
        )
        .agg(
            cities=("city", "count"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
            mean_smape=("smape", "mean"),
            mean_rank=("rank_mae_within_target_horizon", "mean"),
        )
    )
    regime_summary["feature_priority"] = regime_summary["feature_regime"].map(ABLATION_PRIORITY)
    regime_summary = (
        regime_summary.sort_values(["target", "horizon_days", "feature_priority"])
        .drop(columns=["feature_priority"])
        .reset_index(drop=True)
    )

    baseline_reference = build_baseline_reference(baseline_summary)
    reference_enhanced = enhanced_summary[
        (enhanced_summary["model"] == model_name)
    ][["target", "horizon_days", "mean_mae", "mean_rmse", "mean_mape", "mean_smape"]].rename(
        columns={
            "mean_mae": "reference_enhanced_mean_mae",
            "mean_rmse": "reference_enhanced_mean_rmse",
            "mean_mape": "reference_enhanced_mean_mape",
            "mean_smape": "reference_enhanced_mean_smape",
        }
    )

    contribution_summary = regime_summary.pivot_table(
        index=["target", "target_label", "model", "horizon_days"],
        columns="feature_regime",
        values=["mean_mae", "mean_mape"],
        aggfunc="first",
    )
    contribution_summary.columns = [
        f"{metric}_{feature_regime}" for metric, feature_regime in contribution_summary.columns.to_flat_index()
    ]
    contribution_summary = contribution_summary.reset_index()
    contribution_summary = contribution_summary.merge(
        baseline_reference[["target", "horizon_days", "baseline_best_model", "baseline_mean_mae", "baseline_mean_mape"]],
        on=["target", "horizon_days"],
        how="left",
    )
    contribution_summary = contribution_summary.merge(
        reference_enhanced,
        on=["target", "horizon_days"],
        how="left",
    )

    contribution_summary["delta_mae_no_weather_minus_full"] = (
        contribution_summary["mean_mae_no_weather"] - contribution_summary["mean_mae_full_features"]
    )
    contribution_summary["delta_mae_no_cross_pollutant_minus_full"] = (
        contribution_summary["mean_mae_no_cross_pollutant_context"] - contribution_summary["mean_mae_full_features"]
    )
    contribution_summary["delta_mae_no_city_indicators_minus_full"] = (
        contribution_summary["mean_mae_no_city_indicators"] - contribution_summary["mean_mae_full_features"]
    )
    contribution_summary["baseline_minus_full_mae"] = (
        contribution_summary["baseline_mean_mae"] - contribution_summary["mean_mae_full_features"]
    )
    contribution_summary["baseline_minus_no_weather_mae"] = (
        contribution_summary["baseline_mean_mae"] - contribution_summary["mean_mae_no_weather"]
    )
    contribution_summary["baseline_minus_no_cross_pollutant_mae"] = (
        contribution_summary["baseline_mean_mae"] - contribution_summary["mean_mae_no_cross_pollutant_context"]
    )
    contribution_summary["baseline_minus_no_city_indicators_mae"] = (
        contribution_summary["baseline_mean_mae"] - contribution_summary["mean_mae_no_city_indicators"]
    )

    return predictions, metrics_by_city, regime_summary, contribution_summary


def build_summary_markdown(contribution_summary: pd.DataFrame, model_name: str) -> str:
    lines = [
        "# Enhanced Feature-Group Ablation Summary",
        "",
        f"This file summarizes feature-group ablations for `{model_name}` within the pooled feature-based benchmark. "
        "Positive delta values mean that removing the feature group increased MAE and therefore hurt performance.",
        "",
    ]

    for target_name, block in contribution_summary.groupby("target_label"):
        lines.append(f"## {target_name}")
        for row in block.sort_values("horizon_days").itertuples(index=False):
            lines.append(
                "- "
                f"{int(row.horizon_days)}-day: full MAE = {row.mean_mae_full_features:.4f}; "
                f"no weather delta = {row.delta_mae_no_weather_minus_full:+.4f}; "
                f"no cross-pollutant delta = {row.delta_mae_no_cross_pollutant_minus_full:+.4f}; "
                f"no city-indicator delta = {row.delta_mae_no_city_indicators_minus_full:+.4f}; "
                f"best baseline ({row.baseline_best_model}) MAE = {row.baseline_mean_mae:.4f}."
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    predictions, metrics_by_city, regime_summary, contribution_summary = run_ablation(args.model_name)

    predictions.to_csv(TABLE_DIR / "rolling_predictions.csv", index=False)
    metrics_by_city.to_csv(TABLE_DIR / "metrics_by_city.csv", index=False)
    regime_summary.to_csv(TABLE_DIR / "regime_summary.csv", index=False)
    contribution_summary.to_csv(TABLE_DIR / "contribution_summary.csv", index=False)

    with (OUTPUT_DIR / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write(build_summary_markdown(contribution_summary, args.model_name))

    metadata = {
        "model_name": args.model_name,
        "ablation_specs": [
            {"name": name, "label": label, **flags}
            for name, label, flags in ABLATION_SPECS
        ],
        "train_window_days": TRAIN_WINDOW_DAYS,
        "eval_stride_days": EVAL_STRIDE_DAYS,
        "min_train_rows": MIN_TRAIN_ROWS,
        "random_state": RANDOM_STATE,
        "baseline_reference_path": str(BASELINE_DIR / "model_summary.csv"),
        "enhanced_reference_path": str(ENHANCED_DIR / "model_summary.csv"),
    }
    with (OUTPUT_DIR / "ablation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved enhanced feature-group ablation outputs to:")
    for name in [
        "rolling_predictions.csv",
        "metrics_by_city.csv",
        "regime_summary.csv",
        "contribution_summary.csv",
    ]:
        print(f"  - {TABLE_DIR / name}")


if __name__ == "__main__":
    main()
