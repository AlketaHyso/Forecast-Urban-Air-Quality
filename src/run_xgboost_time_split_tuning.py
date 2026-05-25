from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from xgboost import XGBRegressor

from run_enhanced_forecasting_benchmark import (
    ENHANCED_MODEL_PRIORITY,
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xgboost_time_split_tuning"
TABLE_DIR = OUTPUT_DIR / "tables"

MODEL_NAME = "XGBoost Global Features"
DEV_ORIGIN_COUNT = 12
DEFAULT_CONFIG_NAME = "xgb_default"

XGB_CANDIDATES = [
    {
        "config_name": "xgb_default",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 5.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    {
        "config_name": "xgb_shallow_regularized",
        "n_estimators": 220,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 8.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.5,
        "reg_alpha": 0.0,
    },
    {
        "config_name": "xgb_lower_lr_more_trees",
        "n_estimators": 420,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 5.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    {
        "config_name": "xgb_deeper",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    {
        "config_name": "xgb_subsampled_sparse",
        "n_estimators": 360,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
]


def build_xgboost_model(params: dict[str, float | int | str]) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbosity=0,
        **{k: v for k, v in params.items() if k != "config_name"},
    )


def predict_nonnegative(model: XGBRegressor, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    fitted = model.fit(x_train, y_train)
    preds = np.asarray(fitted.predict(x_test), dtype=float)
    return np.maximum(preds, 0.0)


def origin_schedule(unique_dates: list[pd.Timestamp], horizon: int) -> list[pd.Timestamp]:
    return [
        d
        for d in unique_dates
        if d >= unique_dates[TRAIN_WINDOW_DAYS - 1] and d <= unique_dates[-(horizon + 1)]
    ][::EVAL_STRIDE_DAYS]


def split_origin_dates(origins: list[pd.Timestamp]) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    return origins[:DEV_ORIGIN_COUNT], origins[DEV_ORIGIN_COUNT:]


def train_test_frames(frame: pd.DataFrame, origin_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = (
        (frame["target_date"] <= origin_date)
        & (frame["target_date"] > origin_date - pd.Timedelta(days=TRAIN_WINDOW_DAYS))
    )
    test_mask = frame["date"] == origin_date
    return frame.loc[train_mask].copy(), frame.loc[test_mask].copy()


def build_prediction_rows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    origins: list[pd.Timestamp],
    candidate_params: list[dict[str, float | int | str]],
    target_col: str,
    target_label: str,
    horizon: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for origin_date in origins:
        train_df, test_df = train_test_frames(frame, origin_date)
        if len(train_df) < MIN_TRAIN_ROWS or test_df.empty:
            continue

        x_train = train_df[feature_columns]
        y_train = train_df["target"]
        x_test = test_df[feature_columns]

        for params in candidate_params:
            preds = predict_nonnegative(clone(build_xgboost_model(params)), x_train, y_train, x_test)
            for idx, (_, row) in enumerate(test_df.iterrows()):
                rows.append(
                    {
                        "target": target_col,
                        "target_label": target_label,
                        "city": row["city"],
                        "config_name": str(params["config_name"]),
                        "horizon_days": horizon,
                        "feature_date": row["date"],
                        "target_date": row["target_date"],
                        "actual": float(row["target"]),
                        "predicted": float(preds[idx]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_candidate_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_by_city = (
        predictions.groupby(["target", "target_label", "city", "config_name", "horizon_days"], as_index=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "n_predictions": len(frame),
                    "mae": mae(frame["actual"].to_numpy(dtype=float), frame["predicted"].to_numpy(dtype=float)),
                    "rmse": rmse(frame["actual"].to_numpy(dtype=float), frame["predicted"].to_numpy(dtype=float)),
                    "mape": mape(frame["actual"].to_numpy(dtype=float), frame["predicted"].to_numpy(dtype=float)),
                    "smape": smape(frame["actual"].to_numpy(dtype=float), frame["predicted"].to_numpy(dtype=float)),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    summary = (
        metrics_by_city.groupby(["target", "target_label", "config_name", "horizon_days"], as_index=False)
        .agg(
            cities=("city", "count"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
            mean_smape=("smape", "mean"),
        )
        .sort_values(["target", "horizon_days", "mean_mae", "config_name"])
        .reset_index(drop=True)
    )
    return metrics_by_city, summary


def select_best_xgb_config(dev_summary: pd.DataFrame) -> pd.DataFrame:
    order = {config["config_name"]: idx for idx, config in enumerate(XGB_CANDIDATES)}
    selected = (
        dev_summary.assign(config_priority=dev_summary["config_name"].map(order))
        .sort_values(["target", "horizon_days", "mean_mae", "config_priority"])
        .groupby(["target", "target_label", "horizon_days"], as_index=False)
        .first()
        .drop(columns=["config_priority"])
        .rename(
            columns={
                "config_name": "selected_xgb_config",
                "mean_mae": "dev_selected_xgb_mean_mae",
                "mean_rmse": "dev_selected_xgb_mean_rmse",
                "mean_mape": "dev_selected_xgb_mean_mape",
                "mean_smape": "dev_selected_xgb_mean_smape",
            }
        )
    )
    return selected


def select_dev_incumbent_from_enhanced(dev_origins_by_horizon: dict[int, list[pd.Timestamp]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    enhanced_predictions = pd.read_csv(
        ENHANCED_TABLE_DIR / "rolling_predictions.csv",
        parse_dates=["feature_date", "target_date"],
    )

    dev_rows: list[pd.DataFrame] = []
    for horizon, dev_origins in dev_origins_by_horizon.items():
        block = enhanced_predictions[
            (enhanced_predictions["horizon_days"] == horizon)
            & (enhanced_predictions["feature_date"].isin(dev_origins))
        ].copy()
        if not block.empty:
            dev_rows.append(block)

    dev_predictions = pd.concat(dev_rows, ignore_index=True)
    dev_predictions["abs_error"] = np.abs(dev_predictions["actual"] - dev_predictions["predicted"])

    dev_metrics_by_city = (
        dev_predictions.groupby(["target", "target_label", "city", "model", "horizon_days"], as_index=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "n_predictions": len(frame),
                    "mae": mae(frame["actual"].to_numpy(dtype=float), frame["predicted"].to_numpy(dtype=float)),
                    "rmse": rmse(frame["actual"].to_numpy(dtype=float), frame["predicted"].to_numpy(dtype=float)),
                    "mape": mape(frame["actual"].to_numpy(dtype=float), frame["predicted"].to_numpy(dtype=float)),
                    "smape": smape(frame["actual"].to_numpy(dtype=float), frame["predicted"].to_numpy(dtype=float)),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    dev_summary = (
        dev_metrics_by_city.groupby(["target", "target_label", "model", "horizon_days"], as_index=False)
        .agg(
            cities=("city", "count"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
            mean_smape=("smape", "mean"),
        )
    )

    selected = (
        dev_summary.assign(model_priority=dev_summary["model"].map(ENHANCED_MODEL_PRIORITY))
        .sort_values(["target", "horizon_days", "mean_mae", "model_priority"])
        .groupby(["target", "target_label", "horizon_days"], as_index=False)
        .first()
        .drop(columns=["model_priority"])
        .rename(
            columns={
                "model": "selected_enhanced_model",
                "mean_mae": "dev_selected_enhanced_mean_mae",
                "mean_rmse": "dev_selected_enhanced_mean_rmse",
                "mean_mape": "dev_selected_enhanced_mean_mape",
                "mean_smape": "dev_selected_enhanced_mean_smape",
            }
        )
    )
    return selected, enhanced_predictions


def evaluate_selected_enhanced(
    enhanced_predictions: pd.DataFrame,
    selected_enhanced: pd.DataFrame,
    eval_origins_by_horizon: dict[int, list[pd.Timestamp]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_blocks: list[pd.DataFrame] = []
    for row in selected_enhanced.itertuples(index=False):
        horizon_origins = eval_origins_by_horizon[int(row.horizon_days)]
        block = enhanced_predictions[
            (enhanced_predictions["target"] == row.target)
            & (enhanced_predictions["horizon_days"] == int(row.horizon_days))
            & (enhanced_predictions["model"] == row.selected_enhanced_model)
            & (enhanced_predictions["feature_date"].isin(horizon_origins))
        ].copy()
        if not block.empty:
            block["selected_enhanced_model"] = row.selected_enhanced_model
            eval_blocks.append(block)

    eval_predictions = pd.concat(eval_blocks, ignore_index=True)

    metrics_by_city = (
        eval_predictions.groupby(
            ["target", "target_label", "city", "selected_enhanced_model", "horizon_days"],
            as_index=False,
        )
        .apply(
            lambda frame: pd.Series(
                {
                    "n_predictions": len(frame),
                    "mae": mae(frame["actual"].to_numpy(dtype=float), frame["predicted"].to_numpy(dtype=float)),
                    "rmse": rmse(frame["actual"].to_numpy(dtype=float), frame["predicted"].to_numpy(dtype=float)),
                    "mape": mape(frame["actual"].to_numpy(dtype=float), frame["predicted"].to_numpy(dtype=float)),
                    "smape": smape(frame["actual"].to_numpy(dtype=float), frame["predicted"].to_numpy(dtype=float)),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    summary = (
        metrics_by_city.groupby(["target", "target_label", "selected_enhanced_model", "horizon_days"], as_index=False)
        .agg(
            cities=("city", "count"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
            mean_smape=("smape", "mean"),
        )
        .rename(
            columns={
                "mean_mae": "eval_selected_enhanced_mean_mae",
                "mean_rmse": "eval_selected_enhanced_mean_rmse",
                "mean_mape": "eval_selected_enhanced_mean_mape",
                "mean_smape": "eval_selected_enhanced_mean_smape",
            }
        )
    )
    return metrics_by_city, summary


def build_eval_comparison(
    xgb_eval_metrics_by_city: pd.DataFrame,
    xgb_eval_summary: pd.DataFrame,
    selected_xgb: pd.DataFrame,
    selected_enhanced: pd.DataFrame,
    enhanced_eval_metrics_by_city: pd.DataFrame,
    enhanced_eval_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tuned_by_city = xgb_eval_metrics_by_city.merge(
        selected_xgb[["target", "horizon_days", "selected_xgb_config"]],
        on=["target", "horizon_days"],
        how="inner",
    )
    tuned_by_city = tuned_by_city[
        tuned_by_city["config_name"] == tuned_by_city["selected_xgb_config"]
    ].copy()

    default_by_city = xgb_eval_metrics_by_city[
        xgb_eval_metrics_by_city["config_name"] == DEFAULT_CONFIG_NAME
    ].copy()

    enhanced_by_city = enhanced_eval_metrics_by_city.copy()

    comparison_by_city = tuned_by_city.rename(
        columns={
            "config_name": "tuned_config_name",
            "mae": "tuned_mae",
            "rmse": "tuned_rmse",
            "mape": "tuned_mape",
            "smape": "tuned_smape",
        }
    ).merge(
        default_by_city.rename(
            columns={
                "mae": "default_mae",
                "rmse": "default_rmse",
                "mape": "default_mape",
                "smape": "default_smape",
            }
        )[
            ["target", "target_label", "city", "horizon_days", "default_mae", "default_rmse", "default_mape", "default_smape"]
        ],
        on=["target", "target_label", "city", "horizon_days"],
        how="left",
    ).merge(
        enhanced_by_city.rename(
            columns={
                "mae": "enhanced_mae",
                "rmse": "enhanced_rmse",
                "mape": "enhanced_mape",
                "smape": "enhanced_smape",
            }
        )[
            [
                "target",
                "target_label",
                "city",
                "horizon_days",
                "selected_enhanced_model",
                "enhanced_mae",
                "enhanced_rmse",
                "enhanced_mape",
                "enhanced_smape",
            ]
        ],
        on=["target", "target_label", "city", "horizon_days"],
        how="left",
    )

    comparison_by_city["tuned_minus_default_mae"] = comparison_by_city["tuned_mae"] - comparison_by_city["default_mae"]
    comparison_by_city["tuned_minus_enhanced_mae"] = comparison_by_city["tuned_mae"] - comparison_by_city["enhanced_mae"]
    comparison_by_city["winner_vs_default"] = np.where(
        comparison_by_city["tuned_mae"] < comparison_by_city["default_mae"],
        "tuned_xgboost",
        "default_xgboost",
    )
    comparison_by_city["winner_vs_enhanced"] = np.where(
        comparison_by_city["tuned_mae"] < comparison_by_city["enhanced_mae"],
        "tuned_xgboost",
        "selected_enhanced",
    )

    tuned_eval_summary = xgb_eval_summary.merge(
        selected_xgb[["target", "target_label", "horizon_days", "selected_xgb_config"]],
        on=["target", "target_label", "horizon_days"],
        how="inner",
    )
    tuned_eval_summary = tuned_eval_summary[
        tuned_eval_summary["config_name"] == tuned_eval_summary["selected_xgb_config"]
    ].rename(
        columns={
            "mean_mae": "eval_tuned_xgb_mean_mae",
            "mean_rmse": "eval_tuned_xgb_mean_rmse",
            "mean_mape": "eval_tuned_xgb_mean_mape",
            "mean_smape": "eval_tuned_xgb_mean_smape",
        }
    )

    default_eval_summary = xgb_eval_summary[
        xgb_eval_summary["config_name"] == DEFAULT_CONFIG_NAME
    ][
        ["target", "target_label", "horizon_days", "mean_mae", "mean_rmse", "mean_mape", "mean_smape"]
    ].rename(
        columns={
            "mean_mae": "eval_default_xgb_mean_mae",
            "mean_rmse": "eval_default_xgb_mean_rmse",
            "mean_mape": "eval_default_xgb_mean_mape",
            "mean_smape": "eval_default_xgb_mean_smape",
        }
    )

    summary = tuned_eval_summary.merge(
        selected_xgb[
            [
                "target",
                "target_label",
                "horizon_days",
                "selected_xgb_config",
                "dev_selected_xgb_mean_mae",
                "dev_selected_xgb_mean_rmse",
                "dev_selected_xgb_mean_mape",
                "dev_selected_xgb_mean_smape",
            ]
        ],
        on=["target", "target_label", "horizon_days", "selected_xgb_config"],
        how="left",
    ).merge(
        default_eval_summary,
        on=["target", "target_label", "horizon_days"],
        how="left",
    ).merge(
        selected_enhanced[
            [
                "target",
                "target_label",
                "horizon_days",
                "selected_enhanced_model",
                "dev_selected_enhanced_mean_mae",
                "dev_selected_enhanced_mean_rmse",
                "dev_selected_enhanced_mean_mape",
                "dev_selected_enhanced_mean_smape",
            ]
        ],
        on=["target", "target_label", "horizon_days"],
        how="left",
    ).merge(
        enhanced_eval_summary,
        on=["target", "target_label", "horizon_days", "selected_enhanced_model"],
        how="left",
    )

    wins = (
        comparison_by_city.groupby(["target", "target_label", "horizon_days"], as_index=False)
        .agg(
            tuned_wins_vs_default=("winner_vs_default", lambda s: int((s == "tuned_xgboost").sum())),
            default_wins=("winner_vs_default", lambda s: int((s == "default_xgboost").sum())),
            tuned_wins_vs_enhanced=("winner_vs_enhanced", lambda s: int((s == "tuned_xgboost").sum())),
            enhanced_wins=("winner_vs_enhanced", lambda s: int((s == "selected_enhanced").sum())),
        )
    )

    summary = summary.merge(wins, on=["target", "target_label", "horizon_days"], how="left")
    summary["eval_tuned_minus_default_mae"] = (
        summary["eval_tuned_xgb_mean_mae"] - summary["eval_default_xgb_mean_mae"]
    )
    summary["eval_tuned_minus_enhanced_mae"] = (
        summary["eval_tuned_xgb_mean_mae"] - summary["eval_selected_enhanced_mean_mae"]
    )
    return comparison_by_city, summary


def run_experiment() -> tuple[
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

    dev_prediction_rows: list[pd.DataFrame] = []
    eval_prediction_rows: list[pd.DataFrame] = []
    dev_origins_by_horizon: dict[int, list[pd.Timestamp]] = {}
    eval_origins_by_horizon: dict[int, list[pd.Timestamp]] = {}
    origin_records: list[dict] = []

    for target_col, target_label in TARGETS.items():
        base_features, feature_columns = build_base_features(daily, target_col)

        for horizon in TARGET_WINDOWS:
            frame = make_horizon_frame(base_features, target_col, horizon, feature_columns)
            origins = origin_schedule(unique_dates, horizon)
            dev_origins, eval_origins = split_origin_dates(origins)
            dev_origins_by_horizon[horizon] = dev_origins
            eval_origins_by_horizon[horizon] = eval_origins

            for origin_date in dev_origins:
                origin_records.append(
                    {
                        "target": target_col,
                        "target_label": target_label,
                        "horizon_days": horizon,
                        "split": "development",
                        "origin_date": origin_date,
                    }
                )
            for origin_date in eval_origins:
                origin_records.append(
                    {
                        "target": target_col,
                        "target_label": target_label,
                        "horizon_days": horizon,
                        "split": "evaluation",
                        "origin_date": origin_date,
                    }
                )

            dev_prediction_rows.append(
                build_prediction_rows(
                    frame,
                    feature_columns,
                    dev_origins,
                    XGB_CANDIDATES,
                    target_col,
                    target_label,
                    horizon,
                )
            )
            eval_prediction_rows.append(
                build_prediction_rows(
                    frame,
                    feature_columns,
                    eval_origins,
                    XGB_CANDIDATES,
                    target_col,
                    target_label,
                    horizon,
                )
            )

    dev_predictions = pd.concat(dev_prediction_rows, ignore_index=True)
    eval_predictions = pd.concat(eval_prediction_rows, ignore_index=True)

    dev_metrics_by_city, dev_summary = summarize_candidate_predictions(dev_predictions)
    eval_metrics_by_city, eval_summary = summarize_candidate_predictions(eval_predictions)

    selected_xgb = select_best_xgb_config(dev_summary)
    selected_enhanced, enhanced_predictions = select_dev_incumbent_from_enhanced(dev_origins_by_horizon)
    enhanced_eval_metrics_by_city, enhanced_eval_summary = evaluate_selected_enhanced(
        enhanced_predictions,
        selected_enhanced,
        eval_origins_by_horizon,
    )
    comparison_by_city, comparison_summary = build_eval_comparison(
        eval_metrics_by_city,
        eval_summary,
        selected_xgb,
        selected_enhanced,
        enhanced_eval_metrics_by_city,
        enhanced_eval_summary,
    )

    origin_table = pd.DataFrame(origin_records).sort_values(["target", "horizon_days", "split", "origin_date"]).reset_index(drop=True)

    return (
        origin_table,
        dev_summary,
        selected_xgb,
        selected_enhanced,
        eval_summary,
        enhanced_eval_summary,
        comparison_by_city,
        comparison_summary,
        dev_metrics_by_city,
    )


def build_summary_markdown(comparison_summary: pd.DataFrame) -> str:
    lines = [
        "# XGBoost Time-Split Tuning Summary",
        "",
        "This experiment selected XGBoost hyperparameters on early rolling origins and evaluated the chosen configuration only on later held-out origins.",
        "",
    ]

    for target_name, block in comparison_summary.groupby("target_label"):
        lines.append(f"## {target_name}")
        for row in block.sort_values("horizon_days").itertuples(index=False):
            lines.append(
                "- "
                f"{int(row.horizon_days)}-day: selected `{row.selected_xgb_config}` "
                f"(development mean MAE = {row.dev_selected_xgb_mean_mae:.4f}); "
                f"evaluation tuned XGBoost MAE = {row.eval_tuned_xgb_mean_mae:.4f}, "
                f"default XGBoost MAE = {row.eval_default_xgb_mean_mae:.4f}, "
                f"selected enhanced `{row.selected_enhanced_model}` MAE = {row.eval_selected_enhanced_mean_mae:.4f}; "
                f"tuned wins vs default = {int(row.tuned_wins_vs_default)}-{int(row.default_wins)}, "
                f"tuned wins vs enhanced = {int(row.tuned_wins_vs_enhanced)}-{int(row.enhanced_wins)}."
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    (
        origin_table,
        dev_summary,
        selected_xgb,
        selected_enhanced,
        eval_summary,
        enhanced_eval_summary,
        comparison_by_city,
        comparison_summary,
        dev_metrics_by_city,
    ) = run_experiment()

    origin_table.to_csv(TABLE_DIR / "origin_split_schedule.csv", index=False)
    dev_summary.to_csv(TABLE_DIR / "development_candidate_summary.csv", index=False)
    dev_metrics_by_city.to_csv(TABLE_DIR / "development_candidate_metrics_by_city.csv", index=False)
    selected_xgb.to_csv(TABLE_DIR / "selected_xgb_configs.csv", index=False)
    selected_enhanced.to_csv(TABLE_DIR / "selected_enhanced_incumbents.csv", index=False)
    eval_summary.to_csv(TABLE_DIR / "evaluation_candidate_summary.csv", index=False)
    enhanced_eval_summary.to_csv(TABLE_DIR / "evaluation_selected_enhanced_summary.csv", index=False)
    comparison_by_city.to_csv(TABLE_DIR / "evaluation_comparison_by_city.csv", index=False)
    comparison_summary.to_csv(TABLE_DIR / "evaluation_comparison_summary.csv", index=False)

    metadata = {
        "model_name": MODEL_NAME,
        "development_origin_count": DEV_ORIGIN_COUNT,
        "train_window_days": TRAIN_WINDOW_DAYS,
        "eval_stride_days": EVAL_STRIDE_DAYS,
        "min_train_rows": MIN_TRAIN_ROWS,
        "random_state": RANDOM_STATE,
        "candidate_grid": XGB_CANDIDATES,
        "selection_rule": "lowest development mean MAE across city-level MAEs, with candidate-order tie break",
        "enhanced_incumbent_rule": "lowest development mean MAE across city-level MAEs, with enhanced model-order tie break",
    }
    with (OUTPUT_DIR / "xgboost_time_split_tuning_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    with (OUTPUT_DIR / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write(build_summary_markdown(comparison_summary))

    print("Saved XGBoost time-split tuning outputs to:")
    for name in [
        "origin_split_schedule.csv",
        "development_candidate_summary.csv",
        "development_candidate_metrics_by_city.csv",
        "selected_xgb_configs.csv",
        "selected_enhanced_incumbents.csv",
        "evaluation_candidate_summary.csv",
        "evaluation_selected_enhanced_summary.csv",
        "evaluation_comparison_by_city.csv",
        "evaluation_comparison_summary.csv",
    ]:
        print(f"  - {TABLE_DIR / name}")


if __name__ == "__main__":
    main()
