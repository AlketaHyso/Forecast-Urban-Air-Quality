from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from run_enhanced_forecasting_benchmark import (
    EVAL_STRIDE_DAYS,
    INPUT_PATH,
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
ENHANCED_DIR = PROJECT_ROOT / "outputs" / "enhanced_forecast_benchmark" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "city_specific_feature_benchmark"
TABLE_DIR = OUTPUT_DIR / "tables"

VALID_MODELS = ["HistGBM Global Features", "Ridge Global Features"]
DEFAULT_MODEL = "HistGBM Global Features"
LOCAL_MIN_TRAIN_ROWS = 250


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a city-specific feature-based benchmark using the same advanced feature design as the pooled benchmark."
    )
    parser.add_argument(
        "--model-name",
        choices=VALID_MODELS,
        default=DEFAULT_MODEL,
        help="Enhanced model architecture to fit city by city.",
    )
    return parser.parse_args()


def run_city_specific_benchmark(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    daily = daily.sort_values(["city", "date"]).reset_index(drop=True)
    unique_dates = sorted(daily["date"].unique())

    model_template = build_models()[model_name]
    prediction_rows: list[dict] = []

    for target_col, target_label in TARGETS.items():
        base_features, feature_columns = build_base_features(
            daily,
            target_col,
            include_city_indicators=False,
        )
        frame = make_horizon_frame(base_features, target_col, 1, feature_columns)
        available_cities = sorted(frame["city"].unique())

        for horizon in TARGET_WINDOWS:
            frame = make_horizon_frame(base_features, target_col, horizon, feature_columns)
            origin_dates = [
                d
                for d in unique_dates
                if d >= unique_dates[TRAIN_WINDOW_DAYS - 1] and d <= unique_dates[-(horizon + 1)]
            ][::EVAL_STRIDE_DAYS]

            for city in available_cities:
                city_frame = frame.loc[frame["city"] == city].copy()
                if city_frame.empty:
                    continue

                for origin_date in origin_dates:
                    train_mask = (
                        (city_frame["target_date"] <= origin_date)
                        & (city_frame["target_date"] > origin_date - pd.Timedelta(days=TRAIN_WINDOW_DAYS))
                    )
                    test_mask = city_frame["date"] == origin_date

                    train_df = city_frame.loc[train_mask].copy()
                    test_df = city_frame.loc[test_mask].copy()
                    if len(train_df) < LOCAL_MIN_TRAIN_ROWS or test_df.empty:
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
                                "city": city,
                                "model": model_name,
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
        predictions.groupby(["target", "target_label", "city", "model", "horizon_days"], as_index=False)
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

    model_summary = (
        metrics_by_city.groupby(["target", "target_label", "model", "horizon_days"], as_index=False)
        .agg(
            cities=("city", "count"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
            mean_smape=("smape", "mean"),
        )
        .sort_values(["target", "horizon_days"])
        .reset_index(drop=True)
    )

    enhanced_metrics = pd.read_csv(ENHANCED_DIR / "metrics_by_city.csv")
    pooled_reference = enhanced_metrics[
        enhanced_metrics["model"] == model_name
    ][["target", "target_label", "city", "horizon_days", "mae", "rmse", "mape", "smape"]].rename(
        columns={
            "mae": "pooled_mae",
            "rmse": "pooled_rmse",
            "mape": "pooled_mape",
            "smape": "pooled_smape",
        }
    )
    comparison_by_city = metrics_by_city.rename(
        columns={
            "mae": "local_mae",
            "rmse": "local_rmse",
            "mape": "local_mape",
            "smape": "local_smape",
        }
    ).merge(
        pooled_reference,
        on=["target", "target_label", "city", "horizon_days"],
        how="inner",
    )
    comparison_by_city["local_minus_pooled_mae"] = (
        comparison_by_city["local_mae"] - comparison_by_city["pooled_mae"]
    )
    comparison_by_city["pooled_minus_local_mae"] = (
        comparison_by_city["pooled_mae"] - comparison_by_city["local_mae"]
    )
    comparison_by_city["winner"] = np.where(
        comparison_by_city["local_mae"] < comparison_by_city["pooled_mae"],
        "local_city_specific",
        "pooled_cross_city",
    )

    comparison_summary = (
        comparison_by_city.groupby(["target", "target_label", "horizon_days"], as_index=False)
        .agg(
            cities=("city", "count"),
            mean_local_mae=("local_mae", "mean"),
            mean_pooled_mae=("pooled_mae", "mean"),
            mean_local_minus_pooled_mae=("local_minus_pooled_mae", "mean"),
            local_wins=("winner", lambda s: int((s == "local_city_specific").sum())),
            pooled_wins=("winner", lambda s: int((s == "pooled_cross_city").sum())),
        )
        .sort_values(["target", "horizon_days"])
        .reset_index(drop=True)
    )

    return predictions, metrics_by_city, model_summary, comparison_by_city, comparison_summary


def build_summary_markdown(comparison_summary: pd.DataFrame, model_name: str) -> str:
    lines = [
        "# City-Specific Feature Benchmark Summary",
        "",
        f"This file compares a city-specific `{model_name}` model against the pooled cross-city `{model_name}` reference, using the same structured feature design and rolling-origin evaluation schedule.",
        "",
    ]

    for target_name, block in comparison_summary.groupby("target_label"):
        lines.append(f"## {target_name}")
        for row in block.sort_values("horizon_days").itertuples(index=False):
            lines.append(
                "- "
                f"{int(row.horizon_days)}-day: local mean MAE = {row.mean_local_mae:.4f}, "
                f"pooled mean MAE = {row.mean_pooled_mae:.4f}, "
                f"local minus pooled = {row.mean_local_minus_pooled_mae:+.4f}; "
                f"local wins = {int(row.local_wins)}, pooled wins = {int(row.pooled_wins)}."
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    predictions, metrics_by_city, model_summary, comparison_by_city, comparison_summary = run_city_specific_benchmark(
        args.model_name
    )

    predictions.to_csv(TABLE_DIR / "rolling_predictions.csv", index=False)
    metrics_by_city.to_csv(TABLE_DIR / "metrics_by_city.csv", index=False)
    model_summary.to_csv(TABLE_DIR / "model_summary.csv", index=False)
    comparison_by_city.to_csv(TABLE_DIR / "comparison_by_city.csv", index=False)
    comparison_summary.to_csv(TABLE_DIR / "comparison_summary.csv", index=False)

    with (OUTPUT_DIR / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write(build_summary_markdown(comparison_summary, args.model_name))

    metadata = {
        "model_name": args.model_name,
        "train_window_days": TRAIN_WINDOW_DAYS,
        "eval_stride_days": EVAL_STRIDE_DAYS,
        "local_min_train_rows": LOCAL_MIN_TRAIN_ROWS,
        "random_state": RANDOM_STATE,
        "uses_city_indicators": False,
        "pooled_reference_path": str(ENHANCED_DIR / "metrics_by_city.csv"),
    }
    with (OUTPUT_DIR / "city_specific_feature_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved city-specific feature benchmark outputs to:")
    for name in [
        "rolling_predictions.csv",
        "metrics_by_city.csv",
        "model_summary.csv",
        "comparison_by_city.csv",
        "comparison_summary.csv",
    ]:
        print(f"  - {TABLE_DIR / name}")


if __name__ == "__main__":
    main()
