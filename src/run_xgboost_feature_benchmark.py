from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from xgboost import XGBRegressor

from run_enhanced_forecasting_benchmark import (
    EVAL_STRIDE_DAYS,
    INPUT_PATH,
    MIN_TRAIN_ROWS,
    RANDOM_STATE,
    TARGETS,
    TARGET_WINDOWS,
    TRAIN_WINDOW_DAYS,
    build_base_features,
    mae,
    make_horizon_frame,
    mape,
    rmse,
    smape,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENHANCED_TABLE_DIR = PROJECT_ROOT / "outputs" / "enhanced_forecast_benchmark" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xgboost_feature_benchmark"
TABLE_DIR = OUTPUT_DIR / "tables"

MODEL_NAME = "XGBoost Global Features"
HISTGBM_NAME = "HistGBM Global Features"


def build_xgboost_model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5.0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=4,
        verbosity=0,
    )


def predict_nonnegative(model: XGBRegressor, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    fitted = model.fit(x_train, y_train)
    preds = np.asarray(fitted.predict(x_test), dtype=float)
    return np.maximum(preds, 0.0)


def summarize_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return metrics_by_city, model_summary


def compare_against_reference(
    xgb_metrics_by_city: pd.DataFrame,
    reference_metrics_by_city: pd.DataFrame,
    *,
    reference_label: str,
    reference_mae_col: str,
    winner_positive_label: str,
    winner_negative_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_by_city = xgb_metrics_by_city.rename(
        columns={
            "mae": "xgb_mae",
            "rmse": "xgb_rmse",
            "mape": "xgb_mape",
            "smape": "xgb_smape",
        }
    ).merge(
        reference_metrics_by_city,
        on=["target", "target_label", "city", "horizon_days"],
        how="inner",
    )

    comparison_by_city["xgb_minus_reference_mae"] = (
        comparison_by_city["xgb_mae"] - comparison_by_city[reference_mae_col]
    )
    comparison_by_city["reference_minus_xgb_mae"] = (
        comparison_by_city[reference_mae_col] - comparison_by_city["xgb_mae"]
    )
    comparison_by_city["winner"] = np.where(
        comparison_by_city["xgb_mae"] < comparison_by_city[reference_mae_col],
        winner_positive_label,
        winner_negative_label,
    )
    comparison_by_city["reference_label"] = reference_label

    comparison_summary = (
        comparison_by_city.groupby(["target", "target_label", "horizon_days"], as_index=False)
        .agg(
            cities=("city", "count"),
            mean_xgb_mae=("xgb_mae", "mean"),
            mean_reference_mae=(reference_mae_col, "mean"),
            mean_xgb_minus_reference_mae=("xgb_minus_reference_mae", "mean"),
            xgb_wins=("winner", lambda s: int((s == winner_positive_label).sum())),
            reference_wins=("winner", lambda s: int((s == winner_negative_label).sum())),
        )
        .sort_values(["target", "horizon_days"])
        .reset_index(drop=True)
    )
    comparison_summary["reference_label"] = reference_label
    return comparison_by_city, comparison_summary


def run_xgboost_benchmark() -> tuple[
    pd.DataFrame,
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

    model_template = build_xgboost_model()
    prediction_rows: list[dict] = []
    latest_rows: list[dict] = []

    for target_col, target_label in TARGETS.items():
        base_features, feature_columns = build_base_features(daily, target_col)

        for horizon in TARGET_WINDOWS:
            frame = make_horizon_frame(base_features, target_col, horizon, feature_columns)
            origin_dates = [
                d for d in unique_dates
                if d >= unique_dates[TRAIN_WINDOW_DAYS - 1] and d <= unique_dates[-(horizon + 1)]
            ][::EVAL_STRIDE_DAYS]

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

                preds = predict_nonnegative(clone(model_template), x_train, y_train, x_test)
                for idx, (_, row) in enumerate(test_df.iterrows()):
                    prediction_rows.append(
                        {
                            "target": target_col,
                            "target_label": target_label,
                            "city": row["city"],
                            "model": MODEL_NAME,
                            "horizon_days": horizon,
                            "feature_date": row["date"],
                            "target_date": row["target_date"],
                            "actual": float(row["target"]),
                            "predicted": float(preds[idx]),
                        }
                    )

            latest_origin = unique_dates[-1]
            train_mask = (
                (frame["target_date"] <= latest_origin)
                & (frame["target_date"] > latest_origin - pd.Timedelta(days=TRAIN_WINDOW_DAYS))
            )
            latest_feature_rows = base_features.loc[base_features["date"] == latest_origin].copy()
            train_df = frame.loc[train_mask].copy()
            if len(train_df) < MIN_TRAIN_ROWS or latest_feature_rows.empty:
                continue

            x_train = train_df[feature_columns]
            y_train = train_df["target"]
            x_test = latest_feature_rows[feature_columns]
            preds = predict_nonnegative(clone(model_template), x_train, y_train, x_test)

            for idx, (_, row) in enumerate(latest_feature_rows.iterrows()):
                latest_rows.append(
                    {
                        "target": target_col,
                        "target_label": target_label,
                        "city": row["city"],
                        "model": MODEL_NAME,
                        "latest_observed_date": latest_origin,
                        "forecast_date": latest_origin + pd.Timedelta(days=horizon),
                        "horizon_days": horizon,
                        "predicted": float(preds[idx]),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    predictions["abs_error"] = np.abs(predictions["actual"] - predictions["predicted"])

    metrics_by_city, model_summary = summarize_metrics(predictions)

    enhanced_metrics = pd.read_csv(ENHANCED_TABLE_DIR / "metrics_by_city.csv")
    histgbm_reference = enhanced_metrics[
        enhanced_metrics["model"] == HISTGBM_NAME
    ][["target", "target_label", "city", "horizon_days", "mae", "rmse", "mape", "smape"]].rename(
        columns={
            "mae": "histgbm_mae",
            "rmse": "histgbm_rmse",
            "mape": "histgbm_mape",
            "smape": "histgbm_smape",
        }
    )
    histgbm_by_city, histgbm_summary = compare_against_reference(
        metrics_by_city,
        histgbm_reference,
        reference_label=HISTGBM_NAME,
        reference_mae_col="histgbm_mae",
        winner_positive_label="xgboost",
        winner_negative_label="histgbm",
    )

    best_enhanced_by_city = pd.read_csv(ENHANCED_TABLE_DIR / "best_models_by_city.csv")[
        ["target", "target_label", "city", "horizon_days", "model", "mae", "rmse", "mape", "smape"]
    ].rename(
        columns={
            "model": "reference_best_model",
            "mae": "best_enhanced_mae",
            "rmse": "best_enhanced_rmse",
            "mape": "best_enhanced_mape",
            "smape": "best_enhanced_smape",
        }
    )
    best_enhanced_by_city_comparison, best_enhanced_summary = compare_against_reference(
        metrics_by_city,
        best_enhanced_by_city,
        reference_label="Best Existing Enhanced Model",
        reference_mae_col="best_enhanced_mae",
        winner_positive_label="xgboost",
        winner_negative_label="best_existing_enhanced",
    )

    latest_forecasts = pd.DataFrame(latest_rows)

    return (
        predictions,
        metrics_by_city,
        model_summary,
        latest_forecasts,
        histgbm_by_city,
        histgbm_summary,
        best_enhanced_by_city_comparison,
        best_enhanced_summary,
    )


def build_summary_markdown(histgbm_summary: pd.DataFrame, best_enhanced_summary: pd.DataFrame) -> str:
    lines = [
        "# XGBoost Feature Benchmark Summary",
        "",
        "This experiment fits an `XGBoost Global Features` model using the same pooled feature design, rolling-origin schedule, training window, and forecast horizons as the existing enhanced benchmark.",
        "",
        "## Hyperparameters",
        "",
        "- objective = `reg:squarederror`",
        "- n_estimators = `300`",
        "- learning_rate = `0.05`",
        "- max_depth = `4`",
        "- min_child_weight = `5.0`",
        "- subsample = `0.9`",
        "- colsample_bytree = `0.9`",
        "- tree_method = `hist`",
        "",
        "## Comparison vs HistGBM",
        "",
    ]

    for target_name, block in histgbm_summary.groupby("target_label"):
        lines.append(f"### {target_name}")
        for row in block.sort_values("horizon_days").itertuples(index=False):
            lines.append(
                "- "
                f"{int(row.horizon_days)}-day: XGBoost mean MAE = {row.mean_xgb_mae:.4f}, "
                f"HistGBM mean MAE = {row.mean_reference_mae:.4f}, "
                f"XGBoost minus HistGBM = {row.mean_xgb_minus_reference_mae:+.4f}; "
                f"XGBoost wins = {int(row.xgb_wins)}, HistGBM wins = {int(row.reference_wins)}."
            )
        lines.append("")

    lines.extend(["## Comparison vs Best Existing Enhanced Model", ""])
    for target_name, block in best_enhanced_summary.groupby("target_label"):
        lines.append(f"### {target_name}")
        for row in block.sort_values("horizon_days").itertuples(index=False):
            lines.append(
                "- "
                f"{int(row.horizon_days)}-day: XGBoost mean MAE = {row.mean_xgb_mae:.4f}, "
                f"best existing enhanced mean MAE = {row.mean_reference_mae:.4f}, "
                f"XGBoost minus incumbent = {row.mean_xgb_minus_reference_mae:+.4f}; "
                f"XGBoost wins = {int(row.xgb_wins)}, incumbent wins = {int(row.reference_wins)}."
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    (
        predictions,
        metrics_by_city,
        model_summary,
        latest_forecasts,
        histgbm_by_city,
        histgbm_summary,
        best_enhanced_by_city_comparison,
        best_enhanced_summary,
    ) = run_xgboost_benchmark()

    predictions.to_csv(TABLE_DIR / "rolling_predictions.csv", index=False)
    metrics_by_city.to_csv(TABLE_DIR / "metrics_by_city.csv", index=False)
    model_summary.to_csv(TABLE_DIR / "model_summary.csv", index=False)
    latest_forecasts.to_csv(TABLE_DIR / "latest_forecasts.csv", index=False)
    histgbm_by_city.to_csv(TABLE_DIR / "comparison_vs_histgbm_by_city.csv", index=False)
    histgbm_summary.to_csv(TABLE_DIR / "comparison_vs_histgbm_summary.csv", index=False)
    best_enhanced_by_city_comparison.to_csv(TABLE_DIR / "comparison_vs_best_enhanced_by_city.csv", index=False)
    best_enhanced_summary.to_csv(TABLE_DIR / "comparison_vs_best_enhanced_summary.csv", index=False)

    metadata = {
        "model_name": MODEL_NAME,
        "train_window_days": TRAIN_WINDOW_DAYS,
        "eval_stride_days": EVAL_STRIDE_DAYS,
        "min_train_rows": MIN_TRAIN_ROWS,
        "random_state": RANDOM_STATE,
        "hyperparameters": {
            "objective": "reg:squarederror",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 5.0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "tree_method": "hist",
        },
        "reference_paths": {
            "enhanced_metrics_by_city": str(ENHANCED_TABLE_DIR / "metrics_by_city.csv"),
            "best_enhanced_by_city": str(ENHANCED_TABLE_DIR / "best_models_by_city.csv"),
        },
    }
    with (OUTPUT_DIR / "xgboost_feature_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    with (OUTPUT_DIR / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write(build_summary_markdown(histgbm_summary, best_enhanced_summary))

    print("Saved XGBoost feature benchmark outputs to:")
    for name in [
        "rolling_predictions.csv",
        "metrics_by_city.csv",
        "model_summary.csv",
        "latest_forecasts.csv",
        "comparison_vs_histgbm_by_city.csv",
        "comparison_vs_histgbm_summary.csv",
        "comparison_vs_best_enhanced_by_city.csv",
        "comparison_vs_best_enhanced_summary.csv",
    ]:
        print(f"  - {TABLE_DIR / name}")


if __name__ == "__main__":
    main()
