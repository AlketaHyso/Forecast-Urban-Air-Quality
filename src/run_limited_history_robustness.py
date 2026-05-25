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
    MIN_TRAIN_ROWS,
    TARGETS,
    TRAIN_WINDOW_DAYS,
    build_base_features,
    build_models,
    mae,
    make_horizon_frame,
    mape,
    rmse,
    smape,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENHANCED_DIR = PROJECT_ROOT / "outputs" / "enhanced_forecast_benchmark" / "tables"
DEFAULT_TARGET_COL = "pm2_5_mean"
DEFAULT_HORIZON_DAYS = 1
LOCAL_ADAPTATION_DAYS = 90
LOCAL_MIN_ROWS = 90
ALIGNMENT_TOLERANCE = 1e-10

REGIME_SPECS = [
    ("local_only", "Local Only"),
    ("zero_shot_cross_city", "Zero-Shot Cross-City"),
    ("few_shot_cross_city_90d", "Few-Shot Cross-City (90d)"),
    ("full_pooled_reference", "Full Pooled Reference"),
]
REGIME_PRIORITY = {name: idx for idx, (name, _) in enumerate(REGIME_SPECS)}


def mase(actual: np.ndarray, predicted: np.ndarray, scale: np.ndarray) -> float:
    safe_scale = np.where(scale < 1e-8, np.nan, scale)
    return float(np.nanmean(np.abs(actual - predicted) / safe_scale))


def compute_local_scale(city_train_df: pd.DataFrame) -> float:
    values = city_train_df["target"].to_numpy(dtype=float)
    if len(values) < 2:
        return float("nan")
    diffs = np.abs(np.diff(values))
    scale = float(np.mean(diffs))
    return scale if scale > 0.0 else float("nan")


def fit_predict_histgbm(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    model = clone(build_models()["HistGBM Global Features"])
    fitted = model.fit(train_df[feature_columns], train_df["target"])
    preds = np.asarray(fitted.predict(test_df[feature_columns]), dtype=float)
    return np.maximum(preds, 0.0)


def default_output_dir_name(target_col: str, horizon_days: int) -> str:
    if target_col == DEFAULT_TARGET_COL and horizon_days == DEFAULT_HORIZON_DAYS:
        return "limited_history_robustness"
    return f"limited_history_robustness_{target_col}_h{horizon_days}"


def run_experiment() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    return run_experiment_for_target(DEFAULT_TARGET_COL, DEFAULT_HORIZON_DAYS)


def run_experiment_for_target(
    target_col: str,
    horizon_days: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    daily = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    daily = daily.sort_values(["city", "date"]).reset_index(drop=True)
    unique_dates = sorted(daily["date"].unique())

    target_label = TARGETS[target_col]
    base_features, feature_columns = build_base_features(daily, target_col)
    frame = make_horizon_frame(base_features, target_col, horizon_days, feature_columns)

    enhanced_metrics = pd.read_csv(ENHANCED_DIR / "metrics_by_city.csv")
    enhanced_summary = pd.read_csv(ENHANCED_DIR / "model_summary.csv")

    origin_dates = [
        d
        for d in unique_dates
        if d >= unique_dates[TRAIN_WINDOW_DAYS - 1] and d <= unique_dates[-(horizon_days + 1)]
    ][::EVAL_STRIDE_DAYS]

    prediction_rows: list[dict] = []

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

        # Reference regime: exactly the same pooled HistGBM configuration used in the main enhanced benchmark.
        full_reference_preds = fit_predict_histgbm(train_df, test_df, feature_columns)
        full_reference_lookup = (
            test_df[["city", "date", "target_date"]]
            .assign(predicted=full_reference_preds)
            .set_index(["city", "date", "target_date"])["predicted"]
        )

        for city, city_test_df in test_df.groupby("city", sort=False):
            city_train_df = train_df.loc[train_df["city"] == city].copy()
            other_city_train_df = train_df.loc[train_df["city"] != city].copy()
            local_adaptation_train_df = city_train_df.loc[
                city_train_df["target_date"] > origin_date - pd.Timedelta(days=LOCAL_ADAPTATION_DAYS)
            ].copy()
            local_scale = compute_local_scale(city_train_df)

            regime_trains = {
                "local_only": city_train_df,
                "zero_shot_cross_city": other_city_train_df,
                "few_shot_cross_city_90d": (
                    pd.concat([other_city_train_df, local_adaptation_train_df], axis=0)
                    .sort_values(["city", "date"])
                    .reset_index(drop=True)
                ),
                "full_pooled_reference": train_df,
            }

            for regime_name, regime_label in REGIME_SPECS:
                regime_train_df = regime_trains[regime_name]
                if regime_name == "local_only" and len(regime_train_df) < LOCAL_MIN_ROWS:
                    continue
                if regime_train_df.empty:
                    continue

                if regime_name == "full_pooled_reference":
                    preds = np.asarray(
                        [
                            float(full_reference_lookup.loc[(row["city"], row["date"], row["target_date"])])
                            for _, row in city_test_df.iterrows()
                        ],
                        dtype=float,
                    )
                else:
                    preds = fit_predict_histgbm(regime_train_df, city_test_df, feature_columns)

                for idx, (_, row) in enumerate(city_test_df.iterrows()):
                    prediction_rows.append(
                        {
                            "target": target_col,
                            "target_label": target_label,
                            "city": city,
                            "feature_date": row["date"],
                            "target_date": row["target_date"],
                            "horizon_days": horizon_days,
                            "regime": regime_name,
                            "regime_label": regime_label,
                            "actual": float(row["target"]),
                            "predicted": float(preds[idx]),
                            "mase_scale": local_scale,
                            "train_rows_regime": int(len(regime_train_df)),
                            "train_rows_local_full_window": int(len(city_train_df)),
                            "train_rows_local_adaptation_window": int(len(local_adaptation_train_df)),
                            "train_rows_other_cities": int(len(other_city_train_df)),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["city", "feature_date", "regime_label"]
    ).reset_index(drop=True)
    predictions["abs_error"] = np.abs(predictions["actual"] - predictions["predicted"])
    predictions["squared_error"] = (predictions["actual"] - predictions["predicted"]) ** 2
    predictions["ape"] = np.where(
        np.abs(predictions["actual"]) < 1e-8,
        np.nan,
        np.abs((predictions["actual"] - predictions["predicted"]) / predictions["actual"]) * 100.0,
    )
    predictions["smape_component"] = np.where(
        (np.abs(predictions["actual"]) + np.abs(predictions["predicted"])) < 1e-8,
        np.nan,
        200.0
        * np.abs(predictions["actual"] - predictions["predicted"])
        / (np.abs(predictions["actual"]) + np.abs(predictions["predicted"])),
    )
    predictions["ase"] = np.where(
        predictions["mase_scale"] < 1e-8,
        np.nan,
        predictions["abs_error"] / predictions["mase_scale"],
    )

    metrics_by_city = (
        predictions.groupby(
            ["target", "target_label", "city", "horizon_days", "regime", "regime_label"], as_index=False
        )
        .apply(
            lambda frame: pd.Series(
                {
                    "n_predictions": len(frame),
                    "mae": mae(frame["actual"].to_numpy(), frame["predicted"].to_numpy()),
                    "rmse": rmse(frame["actual"].to_numpy(), frame["predicted"].to_numpy()),
                    "mape": mape(frame["actual"].to_numpy(), frame["predicted"].to_numpy()),
                    "smape": smape(frame["actual"].to_numpy(), frame["predicted"].to_numpy()),
                    "mase": float(np.nanmean(frame["ase"].to_numpy(dtype=float))),
                    "avg_train_rows_regime": float(frame["train_rows_regime"].mean()),
                    "avg_local_rows_full_window": float(frame["train_rows_local_full_window"].mean()),
                    "avg_local_rows_adaptation_window": float(frame["train_rows_local_adaptation_window"].mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    metrics_by_city["rank_mae"] = (
        metrics_by_city.groupby(["target", "city", "horizon_days"])["mae"].rank(method="average")
    )

    regime_summary = (
        metrics_by_city.groupby(
            ["target", "target_label", "horizon_days", "regime", "regime_label"], as_index=False
        )
        .agg(
            cities=("city", "count"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
            mean_smape=("smape", "mean"),
            mean_mase=("mase", "mean"),
            mean_rank=("rank_mae", "mean"),
            avg_train_rows_regime=("avg_train_rows_regime", "mean"),
            avg_local_rows_full_window=("avg_local_rows_full_window", "mean"),
            avg_local_rows_adaptation_window=("avg_local_rows_adaptation_window", "mean"),
        )
    )
    regime_summary["regime_priority"] = regime_summary["regime"].map(REGIME_PRIORITY)
    regime_summary = (
        regime_summary.sort_values(["horizon_days", "mean_mae", "mean_rank", "regime_priority"])
        .drop(columns=["regime_priority"])
        .reset_index(drop=True)
    )

    city_regime_pivot = metrics_by_city.pivot_table(
        index=["target", "target_label", "city", "horizon_days"],
        columns="regime",
        values="mae",
    ).reset_index()
    city_regime_pivot.columns.name = None
    city_regime_pivot["zero_shot_vs_local_mae_gain"] = (
        city_regime_pivot["local_only"] - city_regime_pivot["zero_shot_cross_city"]
    )
    city_regime_pivot["few_shot_vs_local_mae_gain"] = (
        city_regime_pivot["local_only"] - city_regime_pivot["few_shot_cross_city_90d"]
    )
    city_regime_pivot["full_pooled_vs_local_mae_gain"] = (
        city_regime_pivot["local_only"] - city_regime_pivot["full_pooled_reference"]
    )
    city_regime_pivot["few_shot_vs_zero_shot_mae_gain"] = (
        city_regime_pivot["zero_shot_cross_city"] - city_regime_pivot["few_shot_cross_city_90d"]
    )
    city_regime_pivot["full_pooled_vs_few_shot_mae_gain"] = (
        city_regime_pivot["few_shot_cross_city_90d"] - city_regime_pivot["full_pooled_reference"]
    )

    best_regime_by_city = (
        metrics_by_city.assign(regime_priority=metrics_by_city["regime"].map(REGIME_PRIORITY))
        .sort_values(["target", "city", "horizon_days", "mae", "rank_mae", "regime_priority"])
        .groupby(["target", "city", "horizon_days"], as_index=False)
        .first()
        .rename(
            columns={
                "regime": "best_regime",
                "regime_label": "best_regime_label",
                "mae": "best_regime_mae",
                "rmse": "best_regime_rmse",
                "mase": "best_regime_mase",
            }
        )
        .drop(columns=["regime_priority"])
    )

    full_pooled_city = metrics_by_city.loc[metrics_by_city["regime"] == "full_pooled_reference"].copy()
    enhanced_histgbm_city = enhanced_metrics.loc[
        (enhanced_metrics["target"] == target_col)
        & (enhanced_metrics["horizon_days"] == horizon_days)
        & (enhanced_metrics["model"] == "HistGBM Global Features")
    ].copy()

    alignment_by_city = full_pooled_city.merge(
        enhanced_histgbm_city[["target", "city", "horizon_days", "mae", "rmse", "mape", "smape"]],
        on=["target", "city", "horizon_days"],
        how="inner",
        suffixes=("_limited_history", "_enhanced_reference"),
    )
    for metric in ["mae", "rmse", "mape", "smape"]:
        alignment_by_city[f"{metric}_delta"] = (
            alignment_by_city[f"{metric}_limited_history"]
            - alignment_by_city[f"{metric}_enhanced_reference"]
        )

    full_pooled_summary = regime_summary.loc[regime_summary["regime"] == "full_pooled_reference"].copy()
    enhanced_histgbm_summary = enhanced_summary.loc[
        (enhanced_summary["target"] == target_col)
        & (enhanced_summary["horizon_days"] == horizon_days)
        & (enhanced_summary["model"] == "HistGBM Global Features")
    ].copy()
    reference_alignment = full_pooled_summary.merge(
        enhanced_histgbm_summary[
            ["target", "target_label", "horizon_days", "mean_mae", "mean_rmse", "mean_mape", "mean_smape"]
        ],
        on=["target", "target_label", "horizon_days"],
        how="inner",
        suffixes=("_limited_history", "_enhanced_reference"),
    )
    for metric in ["mean_mae", "mean_rmse", "mean_mape", "mean_smape"]:
        reference_alignment[f"{metric}_delta"] = (
            reference_alignment[f"{metric}_limited_history"]
            - reference_alignment[f"{metric}_enhanced_reference"]
        )

    max_alignment_delta = max(
        float(reference_alignment[[c for c in reference_alignment.columns if c.endswith("_delta")]].abs().max().max()),
        float(alignment_by_city[[c for c in alignment_by_city.columns if c.endswith("_delta")]].abs().max().max()),
    )
    if max_alignment_delta > ALIGNMENT_TOLERANCE:
        raise RuntimeError(
            "Full pooled reference does not align with the existing HistGBM enhanced benchmark. "
            f"Maximum absolute metric delta was {max_alignment_delta:.12f}."
        )

    return (
        predictions,
        metrics_by_city,
        regime_summary,
        city_regime_pivot,
        best_regime_by_city,
        reference_alignment,
        alignment_by_city,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=sorted(TARGETS.keys()), default=DEFAULT_TARGET_COL)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--output-dir-name", default=None)
    args = parser.parse_args()

    target_col = args.target
    horizon_days = args.horizon
    output_dir_name = args.output_dir_name or default_output_dir_name(target_col, horizon_days)
    output_dir = PROJECT_ROOT / "outputs" / output_dir_name
    table_dir = output_dir / "tables"

    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    (
        predictions,
        metrics_by_city,
        regime_summary,
        city_regime_pivot,
        best_regime_by_city,
        reference_alignment,
        alignment_by_city,
    ) = run_experiment_for_target(target_col, horizon_days)

    predictions.to_csv(table_dir / "rolling_predictions.csv", index=False)
    metrics_by_city.to_csv(table_dir / "metrics_by_city.csv", index=False)
    regime_summary.to_csv(table_dir / "regime_summary.csv", index=False)
    city_regime_pivot.to_csv(table_dir / "city_regime_comparison.csv", index=False)
    best_regime_by_city.to_csv(table_dir / "best_regime_by_city.csv", index=False)
    reference_alignment.to_csv(table_dir / "reference_alignment_summary.csv", index=False)
    alignment_by_city.to_csv(table_dir / "reference_alignment_by_city.csv", index=False)

    metadata = {
        "target": target_col,
        "target_label": TARGETS[target_col],
        "horizon_days": horizon_days,
        "model_family": "HistGBM Global Features",
        "local_adaptation_days": LOCAL_ADAPTATION_DAYS,
        "train_window_days": TRAIN_WINDOW_DAYS,
        "eval_stride_days": EVAL_STRIDE_DAYS,
        "regimes": [{name: label} for name, label in REGIME_SPECS],
        "alignment_tolerance": ALIGNMENT_TOLERANCE,
    }
    with (output_dir / "limited_history_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved limited-history robustness outputs to:")
    for name in [
        "rolling_predictions.csv",
        "metrics_by_city.csv",
        "regime_summary.csv",
        "city_regime_comparison.csv",
        "best_regime_by_city.csv",
        "reference_alignment_summary.csv",
        "reference_alignment_by_city.csv",
    ]:
        print(f"  - {table_dir / name}")


if __name__ == "__main__":
    main()
