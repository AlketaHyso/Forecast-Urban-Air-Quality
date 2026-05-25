from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BASELINE_DIR = PROJECT_ROOT / "outputs" / "forecast_benchmark" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "enhanced_forecast_benchmark"
TABLE_DIR = OUTPUT_DIR / "tables"

INPUT_PATH = PROCESSED_DIR / "albania_air_quality_daily.csv"

TARGETS = {
    "pm2_5_mean": "Daily Mean PM2.5",
    "european_aqi_max": "Daily Max European AQI",
}

TARGET_CONTEXT = {
    "pm2_5_mean": ["european_aqi_max", "pm10_mean", "nitrogen_dioxide_mean", "ozone_mean"],
    "european_aqi_max": ["pm2_5_mean", "pm10_mean", "nitrogen_dioxide_mean", "ozone_mean"],
}
ENHANCED_MODEL_ORDER = [
    "Persistence Current",
    "Ridge Global Features",
    "HistGBM Global Features",
]
ENHANCED_MODEL_PRIORITY = {name: idx for idx, name in enumerate(ENHANCED_MODEL_ORDER)}
BASELINE_MODEL_ORDER = [
    "Naive",
    "Seasonal Naive",
    "Drift",
    "Holt Damped",
    "ARIMA",
    "Random Forest Lag7",
]
BASELINE_MODEL_PRIORITY = {name: idx for idx, name in enumerate(BASELINE_MODEL_ORDER)}

WEATHER_FEATURES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "pressure_msl_mean",
]

LAGS = [1, 2, 3, 7, 14, 21, 28]
ROLL_WINDOWS = [3, 7, 14, 28]
TARGET_WINDOWS = [1, 2, 3]
TRAIN_WINDOW_DAYS = 365
EVAL_STRIDE_DAYS = 28
MIN_TRAIN_ROWS = 400
RANDOM_STATE = 42


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    safe_actual = np.where(np.abs(actual) < 1e-8, np.nan, actual)
    return float(np.nanmean(np.abs((actual - predicted) / safe_actual)) * 100.0)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(predicted)
    safe_denominator = np.where(denominator < 1e-8, np.nan, denominator)
    return float(np.nanmean(200.0 * np.abs(actual - predicted) / safe_denominator))


def build_base_features(
    daily: pd.DataFrame,
    target_col: str,
    *,
    include_weather: bool = True,
    include_target_context: bool = True,
    include_city_indicators: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    weather_features = WEATHER_FEATURES if include_weather else []
    target_context = TARGET_CONTEXT[target_col] if include_target_context else []
    columns = ["city", "date", target_col] + weather_features + target_context
    base = daily[columns].copy().sort_values(["city", "date"]).reset_index(drop=True)
    group = base.groupby("city", group_keys=False)

    base["target_current"] = base[target_col]
    for lag in LAGS:
        base[f"lag_{lag}"] = group[target_col].shift(lag)

    shifted = group[target_col].shift(1)
    for window in ROLL_WINDOWS:
        base[f"roll_mean_{window}"] = shifted.groupby(base["city"]).rolling(window).mean().reset_index(level=0, drop=True)
        base[f"roll_std_{window}"] = shifted.groupby(base["city"]).rolling(window).std().reset_index(level=0, drop=True)
        base[f"roll_max_{window}"] = shifted.groupby(base["city"]).rolling(window).max().reset_index(level=0, drop=True)

    base["day_of_week"] = base["date"].dt.dayofweek
    base["month"] = base["date"].dt.month
    base["day_of_year"] = base["date"].dt.dayofyear
    base["is_weekend"] = base["day_of_week"].isin([5, 6]).astype(int)
    base["doy_sin"] = np.sin(2 * np.pi * base["day_of_year"] / 365.25)
    base["doy_cos"] = np.cos(2 * np.pi * base["day_of_year"] / 365.25)

    city_dummy_columns: list[str] = []
    if include_city_indicators:
        city_dummies = pd.get_dummies(base["city"], prefix="city", dtype=int)
        base = pd.concat([base, city_dummies], axis=1)
        city_dummy_columns = city_dummies.columns.tolist()

    feature_columns = [
        "target_current",
        *[f"lag_{lag}" for lag in LAGS],
        *[f"roll_mean_{window}" for window in ROLL_WINDOWS],
        *[f"roll_std_{window}" for window in ROLL_WINDOWS],
        *[f"roll_max_{window}" for window in ROLL_WINDOWS],
        *weather_features,
        *target_context,
        "day_of_week",
        "month",
        "day_of_year",
        "is_weekend",
        "doy_sin",
        "doy_cos",
        *city_dummy_columns,
    ]

    return base, feature_columns


def make_horizon_frame(base_features: pd.DataFrame, target_col: str, horizon: int, feature_columns: list[str]) -> pd.DataFrame:
    frame = base_features.copy()
    frame["target"] = frame.groupby("city")[target_col].shift(-horizon)
    frame["target_date"] = frame["date"] + pd.to_timedelta(horizon, unit="D")
    frame = frame.dropna(subset=feature_columns + ["target"]).reset_index(drop=True)
    return frame


def build_models() -> dict[str, object]:
    return {
        "Persistence Current": None,
        "Ridge Global Features": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=2.0)),
            ]
        ),
        "HistGBM Global Features": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=4,
            max_iter=120,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
    }


def predict_model(model_name: str, model_obj: object, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    if model_name == "Persistence Current":
        return x_test["target_current"].to_numpy(dtype=float)

    fitted = model_obj.fit(x_train, y_train)
    preds = np.asarray(fitted.predict(x_test), dtype=float)
    return np.maximum(preds, 0.0)


def run_enhanced_benchmark() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    daily = daily.sort_values(["city", "date"]).reset_index(drop=True)
    unique_dates = sorted(daily["date"].unique())

    prediction_rows: list[dict] = []
    latest_rows: list[dict] = []
    comparison_rows: list[dict] = []

    baseline_summary = pd.read_csv(BASELINE_DIR / "model_summary.csv")

    for target_col, target_label in TARGETS.items():
        base_features, feature_columns = build_base_features(daily, target_col)
        for horizon in TARGET_WINDOWS:
            frame = make_horizon_frame(base_features, target_col, horizon, feature_columns)
            models = build_models()

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

                for model_name, model_obj in models.items():
                    preds = predict_model(model_name, model_obj, x_train, y_train, x_test)
                    for idx, (_, row) in enumerate(test_df.iterrows()):
                        prediction_rows.append(
                            {
                                "target": target_col,
                                "target_label": target_label,
                                "city": row["city"],
                                "model": model_name,
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

            for model_name, model_obj in models.items():
                preds = predict_model(model_name, model_obj, x_train, y_train, x_test)
                for idx, (_, row) in enumerate(latest_feature_rows.iterrows()):
                    latest_rows.append(
                        {
                            "target": target_col,
                            "target_label": target_label,
                            "city": row["city"],
                            "model": model_name,
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
    model_summary["model_priority"] = model_summary["model"].map(ENHANCED_MODEL_PRIORITY)
    model_summary = (
        model_summary
        .sort_values(["target", "horizon_days", "mean_mae", "mean_rank", "model_priority"])
        .drop(columns=["model_priority"])
        .reset_index(drop=True)
    )

    best_enhanced = (
        model_summary.assign(model_priority=model_summary["model"].map(ENHANCED_MODEL_PRIORITY))
        .sort_values(["target", "horizon_days", "mean_mae", "mean_rank", "model_priority"])
        .groupby(["target", "horizon_days"], as_index=False)
        .first()
        .drop(columns=["model_priority"])
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
        baseline_summary.assign(model_priority=baseline_summary["model"].map(BASELINE_MODEL_PRIORITY))
        .sort_values(["target", "horizon_days", "mean_mae", "mean_rank", "model_priority"])
        .groupby(["target", "horizon_days"], as_index=False)
        .first()
        .drop(columns=["model_priority"])
        .rename(
            columns={
                "model": "baseline_best_model",
                "mean_mae": "baseline_mean_mae",
                "mean_mape": "baseline_mean_mape",
                "mean_rank": "baseline_mean_rank",
            }
        )
    )

    comparison = best_enhanced.merge(best_baseline, on=["target", "horizon_days"], how="left")
    comparison["mae_improvement"] = comparison["baseline_mean_mae"] - comparison["enhanced_mean_mae"]
    comparison["mape_improvement"] = comparison["baseline_mean_mape"] - comparison["enhanced_mean_mape"]

    latest_forecasts = pd.DataFrame(latest_rows)
    best_by_city = (
        metrics_by_city.sort_values(["target", "city", "horizon_days", "mae", "rmse"])
        .groupby(["target", "city", "horizon_days"], as_index=False)
        .first()
    )

    return predictions, metrics_by_city, model_summary, best_by_city, latest_forecasts, comparison


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    predictions, metrics_by_city, model_summary, best_by_city, latest_forecasts, comparison = run_enhanced_benchmark()

    predictions.to_csv(TABLE_DIR / "rolling_predictions.csv", index=False)
    metrics_by_city.to_csv(TABLE_DIR / "metrics_by_city.csv", index=False)
    model_summary.to_csv(TABLE_DIR / "model_summary.csv", index=False)
    best_by_city.to_csv(TABLE_DIR / "best_models_by_city.csv", index=False)
    latest_forecasts.to_csv(TABLE_DIR / "latest_forecasts.csv", index=False)
    comparison.to_csv(TABLE_DIR / "comparison_vs_baseline.csv", index=False)

    metadata = {
        "targets": TARGETS,
        "lags": LAGS,
        "rolling_windows": ROLL_WINDOWS,
        "train_window_days": TRAIN_WINDOW_DAYS,
        "eval_stride_days": EVAL_STRIDE_DAYS,
        "models": [
            "Persistence Current",
            "Ridge Global Features",
            "HistGBM Global Features",
        ],
        "weather_features": WEATHER_FEATURES,
        "target_context": TARGET_CONTEXT,
    }
    with (OUTPUT_DIR / "enhanced_benchmark_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved enhanced forecasting benchmark outputs to:")
    for name in [
        "rolling_predictions.csv",
        "metrics_by_city.csv",
        "model_summary.csv",
        "best_models_by_city.csv",
        "latest_forecasts.csv",
        "comparison_vs_baseline.csv",
    ]:
        print(f"  - {TABLE_DIR / name}")


if __name__ == "__main__":
    main()
