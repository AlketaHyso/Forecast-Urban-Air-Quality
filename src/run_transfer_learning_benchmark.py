from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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


warnings.filterwarnings("ignore", category=ConvergenceWarning)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "outputs" / "forecast_benchmark" / "tables"
ENHANCED_DIR = PROJECT_ROOT / "outputs" / "enhanced_forecast_benchmark" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "transfer_learning_benchmark"
TABLE_DIR = OUTPUT_DIR / "tables"

TRANSFER_MODEL_NAME = "MLP Transfer Learning"
TRANSFER_MODEL_ORDER = [TRANSFER_MODEL_NAME]
TRANSFER_MODEL_PRIORITY = {name: idx for idx, name in enumerate(TRANSFER_MODEL_ORDER)}
FAMILY_PRIORITY = {"baseline": 0, "enhanced": 1, "transfer_learning": 2}

HIDDEN_LAYER_SIZES = (48, 24)
PRETRAIN_MAX_ITER = 120
FINETUNE_MAX_ITER = 40
MLP_ALPHA = 0.001
LEARNING_RATE_INIT = 0.001
LOCAL_FINE_TUNE_MIN_ROWS = 120


def build_transfer_model(max_iter: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation="relu",
        solver="adam",
        alpha=MLP_ALPHA,
        batch_size=64,
        learning_rate_init=LEARNING_RATE_INIT,
        max_iter=max_iter,
        shuffle=True,
        warm_start=True,
        random_state=RANDOM_STATE,
    )


def pretrain_global_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[StandardScaler, MLPRegressor]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_columns])
    y_train = train_df["target"].to_numpy(dtype=float)

    model = build_transfer_model(PRETRAIN_MAX_ITER)
    model.fit(x_train, y_train)
    return scaler, model


def predict_transfer_for_city(
    pretrained_model: MLPRegressor,
    scaler: StandardScaler,
    city_train_df: pd.DataFrame,
    city_test_df: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    model = copy.deepcopy(pretrained_model)
    if len(city_train_df) >= LOCAL_FINE_TUNE_MIN_ROWS:
        x_local = scaler.transform(city_train_df[feature_columns])
        y_local = city_train_df["target"].to_numpy(dtype=float)
        model.max_iter = FINETUNE_MAX_ITER
        model.fit(x_local, y_local)

    x_test = scaler.transform(city_test_df[feature_columns])
    preds = np.asarray(model.predict(x_test), dtype=float)
    return np.maximum(preds, 0.0)


def run_transfer_learning_benchmark() -> tuple[
    pd.DataFrame,
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

    prediction_rows: list[dict] = []
    latest_rows: list[dict] = []

    baseline_summary = pd.read_csv(BASELINE_DIR / "model_summary.csv")
    baseline_best_by_city = pd.read_csv(BASELINE_DIR / "best_models_by_city.csv")
    enhanced_summary = pd.read_csv(ENHANCED_DIR / "model_summary.csv")
    enhanced_best_by_city = pd.read_csv(ENHANCED_DIR / "best_models_by_city.csv")

    for target_col, target_label in TARGETS.items():
        base_features, feature_columns = build_base_features(daily, target_col)
        for horizon in TARGET_WINDOWS:
            frame = make_horizon_frame(base_features, target_col, horizon, feature_columns)

            origin_dates = [
                d
                for d in unique_dates
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

                scaler, pretrained_model = pretrain_global_model(train_df, feature_columns)

                for city, city_test_df in test_df.groupby("city", sort=False):
                    city_train_df = train_df.loc[train_df["city"] == city].copy()
                    preds = predict_transfer_for_city(
                        pretrained_model=pretrained_model,
                        scaler=scaler,
                        city_train_df=city_train_df,
                        city_test_df=city_test_df,
                        feature_columns=feature_columns,
                    )
                    for idx, (_, row) in enumerate(city_test_df.iterrows()):
                        prediction_rows.append(
                            {
                                "target": target_col,
                                "target_label": target_label,
                                "city": row["city"],
                                "model": TRANSFER_MODEL_NAME,
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

            scaler, pretrained_model = pretrain_global_model(train_df, feature_columns)
            for city, city_rows in latest_feature_rows.groupby("city", sort=False):
                city_train_df = train_df.loc[train_df["city"] == city].copy()
                preds = predict_transfer_for_city(
                    pretrained_model=pretrained_model,
                    scaler=scaler,
                    city_train_df=city_train_df,
                    city_test_df=city_rows,
                    feature_columns=feature_columns,
                )
                for idx, (_, row) in enumerate(city_rows.iterrows()):
                    latest_rows.append(
                        {
                            "target": target_col,
                            "target_label": target_label,
                            "city": row["city"],
                            "model": TRANSFER_MODEL_NAME,
                            "latest_observed_date": latest_origin,
                            "forecast_date": latest_origin + pd.Timedelta(days=horizon),
                            "horizon_days": horizon,
                            "predicted": float(preds[idx]),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    predictions["abs_error"] = np.abs(predictions["actual"] - predictions["predicted"])

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

    metrics_by_city["rank_mae"] = (
        metrics_by_city.groupby(["target", "city", "horizon_days"])["mae"].rank(method="average")
    )

    model_summary = (
        metrics_by_city.groupby(["target", "target_label", "model", "horizon_days"], as_index=False)
        .agg(
            cities=("city", "count"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
            mean_smape=("smape", "mean"),
            mean_rank=("rank_mae", "mean"),
        )
    )
    model_summary["model_priority"] = model_summary["model"].map(TRANSFER_MODEL_PRIORITY)
    model_summary = (
        model_summary.sort_values(["target", "horizon_days", "mean_mae", "mean_rank", "model_priority"])
        .drop(columns=["model_priority"])
        .reset_index(drop=True)
    )

    best_transfer = (
        model_summary.rename(
            columns={
                "model": "transfer_model",
                "mean_mae": "transfer_mean_mae",
                "mean_mape": "transfer_mean_mape",
                "mean_rank": "transfer_mean_rank",
            }
        )
        .copy()
    )

    best_enhanced = (
        enhanced_summary.sort_values(["target", "horizon_days", "mean_mae", "mean_rank"])
        .groupby(["target", "horizon_days"], as_index=False)
        .first()
        .rename(
            columns={
                "model": "enhanced_best_model",
                "mean_mae": "enhanced_mean_mae",
                "mean_mape": "enhanced_mean_mape",
                "mean_rank": "enhanced_mean_rank",
            }
        )
    )
    best_baseline = (
        baseline_summary.sort_values(["target", "horizon_days", "mean_mae", "mean_rank"])
        .groupby(["target", "horizon_days"], as_index=False)
        .first()
        .rename(
            columns={
                "model": "baseline_best_model",
                "mean_mae": "baseline_mean_mae",
                "mean_mape": "baseline_mean_mape",
                "mean_rank": "baseline_mean_rank",
            }
        )
    )

    comparison = (
        best_transfer.merge(
            best_enhanced[
                [
                    "target",
                    "target_label",
                    "horizon_days",
                    "enhanced_best_model",
                    "enhanced_mean_mae",
                    "enhanced_mean_mape",
                    "enhanced_mean_rank",
                ]
            ],
            on=["target", "target_label", "horizon_days"],
            how="left",
        )
        .merge(
            best_baseline[
                [
                    "target",
                    "target_label",
                    "horizon_days",
                    "baseline_best_model",
                    "baseline_mean_mae",
                    "baseline_mean_mape",
                    "baseline_mean_rank",
                ]
            ],
            on=["target", "target_label", "horizon_days"],
            how="left",
        )
    )
    comparison["transfer_vs_enhanced_mae_improvement"] = (
        comparison["enhanced_mean_mae"] - comparison["transfer_mean_mae"]
    )
    comparison["transfer_vs_baseline_mae_improvement"] = (
        comparison["baseline_mean_mae"] - comparison["transfer_mean_mae"]
    )
    comparison["transfer_vs_enhanced_mape_improvement"] = (
        comparison["enhanced_mean_mape"] - comparison["transfer_mean_mape"]
    )
    comparison["transfer_vs_baseline_mape_improvement"] = (
        comparison["baseline_mean_mape"] - comparison["transfer_mean_mape"]
    )

    best_by_city = (
        metrics_by_city.sort_values(["target", "city", "horizon_days", "mae", "rmse"])
        .groupby(["target", "city", "horizon_days"], as_index=False)
        .first()
    )

    city_family_comparison = (
        baseline_best_by_city[["target", "city", "horizon_days", "model", "mae"]]
        .rename(columns={"model": "baseline_model", "mae": "baseline_mae"})
        .merge(
            enhanced_best_by_city[["target", "city", "horizon_days", "model", "mae"]]
            .rename(columns={"model": "enhanced_model", "mae": "enhanced_mae"}),
            on=["target", "city", "horizon_days"],
            how="inner",
        )
        .merge(
            best_by_city[["target", "city", "horizon_days", "model", "mae"]]
            .rename(columns={"model": "transfer_model", "mae": "transfer_mae"}),
            on=["target", "city", "horizon_days"],
            how="inner",
        )
    )
    city_family_comparison["transfer_vs_enhanced_mae_improvement"] = (
        city_family_comparison["enhanced_mae"] - city_family_comparison["transfer_mae"]
    )
    city_family_comparison["transfer_vs_baseline_mae_improvement"] = (
        city_family_comparison["baseline_mae"] - city_family_comparison["transfer_mae"]
    )

    winner_rows: list[dict] = []
    for _, row in city_family_comparison.iterrows():
        family_scores = [
            ("baseline", row["baseline_mae"], row["baseline_model"]),
            ("enhanced", row["enhanced_mae"], row["enhanced_model"]),
            ("transfer_learning", row["transfer_mae"], row["transfer_model"]),
        ]
        family_scores.sort(key=lambda item: (item[1], FAMILY_PRIORITY[item[0]]))
        winner_family, winner_mae, winner_model = family_scores[0]
        winner_rows.append(
            {
                "target": row["target"],
                "city": row["city"],
                "horizon_days": row["horizon_days"],
                "winning_family": winner_family,
                "winning_model": winner_model,
                "winning_mae": winner_mae,
                "baseline_mae": row["baseline_mae"],
                "enhanced_mae": row["enhanced_mae"],
                "transfer_mae": row["transfer_mae"],
            }
        )
    family_winners = pd.DataFrame(winner_rows)
    family_win_counts = (
        family_winners.groupby(["target", "horizon_days", "winning_family"], as_index=False)
        .size()
        .rename(columns={"size": "n_city_wins"})
        .sort_values(["target", "horizon_days", "winning_family"])
        .reset_index(drop=True)
    )

    latest_forecasts = pd.DataFrame(latest_rows)
    return (
        predictions,
        metrics_by_city,
        model_summary,
        best_by_city,
        latest_forecasts,
        comparison,
        city_family_comparison,
        family_winners,
        family_win_counts,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    (
        predictions,
        metrics_by_city,
        model_summary,
        best_by_city,
        latest_forecasts,
        comparison,
        city_family_comparison,
        family_winners,
        family_win_counts,
    ) = run_transfer_learning_benchmark()

    predictions.to_csv(TABLE_DIR / "rolling_predictions.csv", index=False)
    metrics_by_city.to_csv(TABLE_DIR / "metrics_by_city.csv", index=False)
    model_summary.to_csv(TABLE_DIR / "model_summary.csv", index=False)
    best_by_city.to_csv(TABLE_DIR / "best_models_by_city.csv", index=False)
    latest_forecasts.to_csv(TABLE_DIR / "latest_forecasts.csv", index=False)
    comparison.to_csv(TABLE_DIR / "comparison_vs_existing.csv", index=False)
    city_family_comparison.to_csv(TABLE_DIR / "city_family_comparison.csv", index=False)
    family_winners.to_csv(TABLE_DIR / "family_winners_by_city.csv", index=False)
    family_win_counts.to_csv(TABLE_DIR / "family_win_counts.csv", index=False)

    metadata = {
        "model": TRANSFER_MODEL_NAME,
        "architecture": {
            "hidden_layer_sizes": list(HIDDEN_LAYER_SIZES),
            "activation": "relu",
            "solver": "adam",
            "alpha": MLP_ALPHA,
            "learning_rate_init": LEARNING_RATE_INIT,
        },
        "training_scheme": {
            "pretrain_scope": "pooled rows from all cities within the rolling training window",
            "fine_tune_scope": "city-specific rows from the same rolling training window",
            "pretrain_max_iter": PRETRAIN_MAX_ITER,
            "fine_tune_max_iter": FINETUNE_MAX_ITER,
            "local_fine_tune_min_rows": LOCAL_FINE_TUNE_MIN_ROWS,
            "train_window_days": TRAIN_WINDOW_DAYS,
            "eval_stride_days": EVAL_STRIDE_DAYS,
        },
        "targets": TARGETS,
    }
    with (OUTPUT_DIR / "transfer_learning_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved transfer learning benchmark outputs to:")
    for name in [
        "rolling_predictions.csv",
        "metrics_by_city.csv",
        "model_summary.csv",
        "best_models_by_city.csv",
        "latest_forecasts.csv",
        "comparison_vs_existing.csv",
        "city_family_comparison.csv",
        "family_winners_by_city.csv",
        "family_win_counts.csv",
    ]:
        print(f"  - {TABLE_DIR / name}")


if __name__ == "__main__":
    main()
