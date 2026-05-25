from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from run_enhanced_forecasting_benchmark import (
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
    rmse,
    smape,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "histgbm_feature_pruning_experiment"
TABLE_DIR = OUTPUT_DIR / "tables"

MODEL_NAME = "HistGBM Global Features"
N_PERMUTATIONS = 5
DEV_ORIGIN_COUNT = 12
SELECTION_RULE = "remove features with mean permutation delta <= 0 and positive-share < 0.50 on development origins"


def fit_histgbm_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> object:
    model = clone(build_models()[MODEL_NAME])
    return model.fit(train_df[feature_columns], train_df["target"])


def predict_with_model(
    model: object,
    x_test: pd.DataFrame,
) -> np.ndarray:
    preds = np.asarray(model.predict(x_test), dtype=float)
    return np.maximum(preds, 0.0)


def origin_schedule(unique_dates: list[pd.Timestamp], horizon: int) -> list[pd.Timestamp]:
    return [
        d
        for d in unique_dates
        if d >= unique_dates[TRAIN_WINDOW_DAYS - 1] and d <= unique_dates[-(horizon + 1)]
    ][::EVAL_STRIDE_DAYS]


def split_origin_dates(origins: list[pd.Timestamp]) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    dev = origins[:DEV_ORIGIN_COUNT]
    eval_only = origins[DEV_ORIGIN_COUNT:]
    return dev, eval_only


def train_test_frames(
    frame: pd.DataFrame,
    origin_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = (
        (frame["target_date"] <= origin_date)
        & (frame["target_date"] > origin_date - pd.Timedelta(days=TRAIN_WINDOW_DAYS))
    )
    test_mask = frame["date"] == origin_date
    train_df = frame.loc[train_mask].copy()
    test_df = frame.loc[test_mask].copy()
    return train_df, test_df


def compute_dev_importance(
    frame: pd.DataFrame,
    feature_columns: list[str],
    dev_origins: list[pd.Timestamp],
    rng: np.random.Generator,
) -> pd.DataFrame:
    origin_blocks: list[dict] = []

    for origin_date in dev_origins:
        train_df, test_df = train_test_frames(frame, origin_date)
        if len(train_df) < MIN_TRAIN_ROWS or test_df.empty:
            continue

        model = fit_histgbm_model(train_df, feature_columns)
        x_test = test_df[feature_columns].copy()
        base_preds = predict_with_model(model, x_test)
        origin_blocks.append(
            {
                "origin_date": origin_date,
                "model": model,
                "x_test": x_test,
                "actual": test_df["target"].to_numpy(dtype=float),
                "base_pred": base_preds,
            }
        )

    if not origin_blocks:
        return pd.DataFrame()

    all_x = pd.concat(
        [block["x_test"].assign(origin_date=block["origin_date"]) for block in origin_blocks],
        ignore_index=True,
    )
    all_actual = np.concatenate([block["actual"] for block in origin_blocks])
    all_base_pred = np.concatenate([block["base_pred"] for block in origin_blocks])
    base_mae = mae(all_actual, all_base_pred)

    rows: list[dict] = []
    origin_masks = {
        block["origin_date"]: (all_x["origin_date"] == block["origin_date"]).to_numpy()
        for block in origin_blocks
    }

    for feature_name in feature_columns:
        deltas: list[float] = []
        for _ in range(N_PERMUTATIONS):
            permuted_all = all_x.copy()
            permuted_all[feature_name] = rng.permutation(permuted_all[feature_name].to_numpy())

            preds_parts: list[np.ndarray] = []
            for block in origin_blocks:
                mask = origin_masks[block["origin_date"]]
                x_block = permuted_all.loc[mask, feature_columns]
                preds_parts.append(predict_with_model(block["model"], x_block))

            perm_preds = np.concatenate(preds_parts)
            perm_mae = mae(all_actual, perm_preds)
            deltas.append(float(perm_mae - base_mae))

        rows.append(
            {
                "feature_name": feature_name,
                "mean_delta_mae": float(np.mean(deltas)),
                "median_delta_mae": float(np.median(deltas)),
                "positive_share": float(np.mean(np.asarray(deltas) > 0.0)),
                "n_repeats": N_PERMUTATIONS,
            }
        )

    return pd.DataFrame(rows)


def aggregate_importance(dev_importance: pd.DataFrame) -> pd.DataFrame:
    return dev_importance.sort_values(["mean_delta_mae", "positive_share"], ascending=[True, True]).reset_index(drop=True)


def select_features_to_drop(importance_summary: pd.DataFrame) -> list[str]:
    mask = (
        (importance_summary["mean_delta_mae"] <= 0.0)
        & (importance_summary["positive_share"] < 0.50)
    )
    return importance_summary.loc[mask, "feature_name"].tolist()


def evaluate_feature_sets(
    frame: pd.DataFrame,
    feature_columns: list[str],
    dropped_features: list[str],
    eval_origins: list[pd.Timestamp],
) -> pd.DataFrame:
    reduced_features = [f for f in feature_columns if f not in dropped_features]
    rows: list[dict] = []

    for origin_date in eval_origins:
        train_df, test_df = train_test_frames(frame, origin_date)
        if len(train_df) < MIN_TRAIN_ROWS or test_df.empty:
            continue

        configs = {
            "full_features": feature_columns,
            "reduced_features": reduced_features,
        }

        for config_name, active_features in configs.items():
            model = fit_histgbm_model(train_df, active_features)
            preds = predict_with_model(model, test_df[active_features])
            for idx, (_, row) in enumerate(test_df.iterrows()):
                rows.append(
                    {
                        "city": row["city"],
                        "feature_date": row["date"],
                        "target_date": row["target_date"],
                        "config": config_name,
                        "actual": float(row["target"]),
                        "predicted": float(preds[idx]),
                    }
                )

    predictions = pd.DataFrame(rows)
    predictions["abs_error"] = np.abs(predictions["actual"] - predictions["predicted"])
    predictions["squared_error"] = (predictions["actual"] - predictions["predicted"]) ** 2
    return predictions


def summarise_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_by_city = (
        predictions.groupby(["city", "config"], as_index=False)
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
        metrics_by_city.groupby("config", as_index=False)
        .agg(
            cities=("city", "count"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
            mean_smape=("smape", "mean"),
        )
    )
    return metrics_by_city, summary


def build_group_labels(feature_names: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for feature_name in feature_names:
        if feature_name.startswith("city_"):
            labels[feature_name] = "city_indicator"
        elif feature_name in {
            "temperature_2m_mean",
            "temperature_2m_max",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_mean",
            "wind_speed_10m_max",
            "pressure_msl_mean",
        }:
            labels[feature_name] = "weather"
        elif feature_name in {
            "pm2_5_mean",
            "pm10_mean",
            "nitrogen_dioxide_mean",
            "ozone_mean",
            "european_aqi_max",
        }:
            labels[feature_name] = "cross_pollutant_context"
        elif feature_name in {"day_of_week", "month", "day_of_year", "is_weekend", "doy_sin", "doy_cos"}:
            labels[feature_name] = "calendar"
        elif feature_name.startswith("lag_") or feature_name.startswith("roll_") or feature_name == "target_current":
            labels[feature_name] = "autoregressive_temporal"
        else:
            labels[feature_name] = "other"
    return labels


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    daily = daily.sort_values(["city", "date"]).reset_index(drop=True)
    unique_dates = sorted(daily["date"].unique())
    rng = np.random.default_rng(RANDOM_STATE)

    importance_rows: list[pd.DataFrame] = []
    selected_rows: list[dict] = []
    eval_metrics_rows: list[pd.DataFrame] = []
    eval_summary_rows: list[dict] = []
    group_importance_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []

    for target_col, target_label in TARGETS.items():
        base_features, feature_columns = build_base_features(daily, target_col)
        feature_groups = build_group_labels(feature_columns)

        for horizon in TARGET_WINDOWS:
            frame = make_horizon_frame(base_features, target_col, horizon, feature_columns)
            origins = origin_schedule(unique_dates, horizon)
            dev_origins, eval_origins = split_origin_dates(origins)

            dev_importance = compute_dev_importance(frame, feature_columns, dev_origins, rng)
            if dev_importance.empty:
                continue

            dev_importance["target"] = target_col
            dev_importance["target_label"] = target_label
            dev_importance["horizon_days"] = horizon
            dev_importance["feature_group"] = dev_importance["feature_name"].map(feature_groups)
            importance_rows.append(dev_importance)

            importance_summary = aggregate_importance(dev_importance)
            importance_summary["feature_group"] = importance_summary["feature_name"].map(feature_groups)
            dropped_features = select_features_to_drop(importance_summary)

            selected_rows.append(
                {
                    "target": target_col,
                    "target_label": target_label,
                    "horizon_days": horizon,
                    "n_total_features": len(feature_columns),
                    "n_dropped_features": len(dropped_features),
                    "dropped_features": "; ".join(dropped_features),
                }
            )

            group_summary = (
                importance_summary.groupby("feature_group", as_index=False)
                .agg(
                    features=("feature_name", "count"),
                    mean_delta_mae=("mean_delta_mae", "mean"),
                    mean_positive_share=("positive_share", "mean"),
                )
            )
            for row in group_summary.itertuples(index=False):
                group_importance_rows.append(
                    {
                        "target": target_col,
                        "target_label": target_label,
                        "horizon_days": horizon,
                        "feature_group": row.feature_group,
                        "features": int(row.features),
                        "mean_delta_mae": float(row.mean_delta_mae),
                        "mean_positive_share": float(row.mean_positive_share),
                    }
                )

            eval_predictions = evaluate_feature_sets(frame, feature_columns, dropped_features, eval_origins)
            eval_predictions["target"] = target_col
            eval_predictions["target_label"] = target_label
            eval_predictions["horizon_days"] = horizon
            prediction_rows.append(eval_predictions)

            metrics_by_city, summary = summarise_predictions(eval_predictions)
            metrics_by_city["target"] = target_col
            metrics_by_city["target_label"] = target_label
            metrics_by_city["horizon_days"] = horizon
            eval_metrics_rows.append(metrics_by_city)

            summary_lookup = summary.set_index("config")
            full_mae = float(summary_lookup.loc["full_features", "mean_mae"])
            reduced_mae = float(summary_lookup.loc["reduced_features", "mean_mae"])
            full_wins = int((metrics_by_city.pivot(index="city", columns="config", values="mae")["full_features"] < metrics_by_city.pivot(index="city", columns="config", values="mae")["reduced_features"]).sum())
            reduced_wins = int((metrics_by_city.pivot(index="city", columns="config", values="mae")["reduced_features"] < metrics_by_city.pivot(index="city", columns="config", values="mae")["full_features"]).sum())
            eval_summary_rows.append(
                {
                    "target": target_col,
                    "target_label": target_label,
                    "horizon_days": horizon,
                    "n_dev_origins": len(dev_origins),
                    "n_eval_origins": len(eval_origins),
                    "full_mean_mae": full_mae,
                    "reduced_mean_mae": reduced_mae,
                    "reduced_minus_full_mae": reduced_mae - full_mae,
                    "full_wins": full_wins,
                    "reduced_wins": reduced_wins,
                    "n_dropped_features": len(dropped_features),
                    "dropped_features": "; ".join(dropped_features),
                }
            )

    importance = pd.concat(importance_rows, ignore_index=True)
    selection = pd.DataFrame(selected_rows)
    group_importance = pd.DataFrame(group_importance_rows)
    eval_predictions = pd.concat(prediction_rows, ignore_index=True)
    eval_metrics = pd.concat(eval_metrics_rows, ignore_index=True)
    eval_summary = pd.DataFrame(eval_summary_rows)
    return importance, selection, group_importance, eval_predictions, eval_metrics, eval_summary


def build_summary_markdown(eval_summary: pd.DataFrame) -> str:
    lines = [
        "# HistGBM Feature-Pruning Experiment Summary",
        "",
        "Feature selection was performed on early development origins using permutation importance, and the reduced model was then evaluated on later held-out origins only.",
        "",
    ]
    for target_name, block in eval_summary.groupby("target_label"):
        lines.append(f"## {target_name}")
        for row in block.sort_values("horizon_days").itertuples(index=False):
            lines.append(
                "- "
                f"{int(row.horizon_days)}-day: full eval MAE = {row.full_mean_mae:.4f}, "
                f"reduced eval MAE = {row.reduced_mean_mae:.4f}, "
                f"reduced minus full = {row.reduced_minus_full_mae:+.4f}; "
                f"full wins = {int(row.full_wins)}, reduced wins = {int(row.reduced_wins)}; "
                f"dropped features = {row.dropped_features or 'none'}."
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    importance, selection, group_importance, eval_predictions, eval_metrics, eval_summary = run_experiment()

    importance.to_csv(TABLE_DIR / "development_permutation_importance.csv", index=False)
    selection.to_csv(TABLE_DIR / "selected_features.csv", index=False)
    group_importance.to_csv(TABLE_DIR / "group_importance_summary.csv", index=False)
    eval_predictions.to_csv(TABLE_DIR / "evaluation_predictions.csv", index=False)
    eval_metrics.to_csv(TABLE_DIR / "evaluation_metrics_by_city.csv", index=False)
    eval_summary.to_csv(TABLE_DIR / "evaluation_summary.csv", index=False)

    with (OUTPUT_DIR / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write(build_summary_markdown(eval_summary))

    metadata = {
        "model_name": MODEL_NAME,
        "n_permutations": N_PERMUTATIONS,
        "development_origin_count": DEV_ORIGIN_COUNT,
        "selection_rule": SELECTION_RULE,
        "train_window_days": TRAIN_WINDOW_DAYS,
        "eval_stride_days": EVAL_STRIDE_DAYS,
        "min_train_rows": MIN_TRAIN_ROWS,
        "random_state": RANDOM_STATE,
    }
    with (OUTPUT_DIR / "feature_pruning_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved HistGBM feature-pruning experiment outputs to:")
    for name in [
        "development_permutation_importance.csv",
        "selected_features.csv",
        "group_importance_summary.csv",
        "evaluation_predictions.csv",
        "evaluation_metrics_by_city.csv",
        "evaluation_summary.csv",
    ]:
        print(f"  - {TABLE_DIR / name}")


if __name__ == "__main__":
    main()
