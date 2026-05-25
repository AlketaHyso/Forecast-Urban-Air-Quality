from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecast_benchmark"
TABLE_DIR = OUTPUT_DIR / "tables"

INPUT_PATH = PROCESSED_DIR / "albania_air_quality_daily.csv"

TARGETS = {
    "pm2_5_mean": "Daily Mean PM2.5",
    "european_aqi_max": "Daily Max European AQI",
}

MODEL_NAMES = [
    "Naive",
    "Seasonal Naive",
    "Drift",
    "Holt Damped",
    "ARIMA",
    "Random Forest Lag7",
]
MODEL_PRIORITY = {name: idx for idx, name in enumerate(MODEL_NAMES)}

INITIAL_TRAIN_DAYS = 365
MAX_HORIZON = 3
EVAL_STRIDE = 28
SEASONAL_PERIOD = 7
RF_LAGS = 7
RANDOM_STATE = 42
ARIMA_ORDERS = [
    (1, 1, 0),
    (1, 1, 1),
    (0, 1, 1),
]

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class ForecastRun:
    model: str
    predictions: list[float]


def clip_nonnegative(values: list[float] | np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=float)
    return np.maximum(arr, 0.0).tolist()


def forecast_naive(train: np.ndarray, horizon: int) -> ForecastRun:
    return ForecastRun("Naive", clip_nonnegative([train[-1]] * horizon))


def forecast_seasonal_naive(
    train: np.ndarray,
    horizon: int,
    seasonal_period: int = SEASONAL_PERIOD,
) -> ForecastRun:
    if len(train) < seasonal_period:
        return ForecastRun("Seasonal Naive", forecast_naive(train, horizon).predictions)

    last_season = train[-seasonal_period:].astype(float).tolist()
    preds = [last_season[(step - 1) % seasonal_period] for step in range(1, horizon + 1)]
    return ForecastRun("Seasonal Naive", clip_nonnegative(preds))


def forecast_drift(train: np.ndarray, horizon: int) -> ForecastRun:
    if len(train) < 2:
        return forecast_naive(train, horizon)
    slope = (train[-1] - train[0]) / max(len(train) - 1, 1)
    preds = [train[-1] + slope * step for step in range(1, horizon + 1)]
    return ForecastRun("Drift", clip_nonnegative(preds))


def forecast_holt_damped(train: np.ndarray, horizon: int) -> ForecastRun:
    try:
        fit = ExponentialSmoothing(
            train,
            trend="add",
            damped_trend=True,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=False)
        preds = fit.forecast(horizon)
        return ForecastRun("Holt Damped", clip_nonnegative(preds))
    except Exception:
        return ForecastRun("Holt Damped", forecast_naive(train, horizon).predictions)


def forecast_arima(train: np.ndarray, horizon: int) -> ForecastRun:
    best_fit = None
    best_aic = np.inf
    for order in ARIMA_ORDERS:
        try:
            fit = ARIMA(
                train,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit
        except Exception:
            continue

    if best_fit is None:
        return ForecastRun("ARIMA", forecast_naive(train, horizon).predictions)

    try:
        preds = best_fit.forecast(horizon)
        return ForecastRun("ARIMA", clip_nonnegative(preds))
    except Exception:
        return ForecastRun("ARIMA", forecast_naive(train, horizon).predictions)


def forecast_random_forest(train: np.ndarray, horizon: int, lags: int = RF_LAGS) -> ForecastRun:
    if len(train) <= lags + 5:
        return ForecastRun("Random Forest Lag7", forecast_naive(train, horizon).predictions)

    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    for idx in range(lags, len(train)):
        lag_values = train[idx - lags : idx].tolist()
        x_rows.append(lag_values + [float(idx)])
        y_rows.append(float(train[idx]))

    model = RandomForestRegressor(
        n_estimators=120,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    try:
        model.fit(x_rows, y_rows)
    except Exception:
        return ForecastRun("Random Forest Lag7", forecast_naive(train, horizon).predictions)

    history = train.astype(float).tolist()
    preds: list[float] = []
    for _ in range(horizon):
        features = history[-lags:] + [float(len(history))]
        pred = float(model.predict([features])[0])
        pred = max(pred, 0.0)
        preds.append(pred)
        history.append(pred)

    return ForecastRun("Random Forest Lag7", preds)


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


def run_benchmark() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    daily = daily.sort_values(["city", "date"]).reset_index(drop=True)

    prediction_rows: list[dict] = []
    latest_rows: list[dict] = []

    for target_col, target_label in TARGETS.items():
        for city, city_df in daily.groupby("city"):
            series_df = city_df[["date", target_col]].dropna().copy()
            values = series_df[target_col].to_numpy(dtype=float)
            dates = series_df["date"].to_list()

            if len(values) < INITIAL_TRAIN_DAYS + MAX_HORIZON + 10:
                continue

            for train_end in range(INITIAL_TRAIN_DAYS, len(values) - MAX_HORIZON + 1, EVAL_STRIDE):
                train = values[train_end - INITIAL_TRAIN_DAYS : train_end]
                actual_window = values[train_end : train_end + MAX_HORIZON]
                actual_dates = dates[train_end : train_end + MAX_HORIZON]
                training_end_date = dates[train_end - 1]

                runs = [
                    forecast_naive(train, MAX_HORIZON),
                    forecast_seasonal_naive(train, MAX_HORIZON),
                    forecast_drift(train, MAX_HORIZON),
                    forecast_holt_damped(train, MAX_HORIZON),
                    forecast_arima(train, MAX_HORIZON),
                    forecast_random_forest(train, MAX_HORIZON),
                ]

                for run in runs:
                    for horizon_idx in range(MAX_HORIZON):
                        prediction_rows.append(
                            {
                                "target": target_col,
                                "target_label": target_label,
                                "city": city,
                                "model": run.model,
                                "horizon_days": horizon_idx + 1,
                                "training_end_date": training_end_date,
                                "target_date": actual_dates[horizon_idx],
                                "actual": float(actual_window[horizon_idx]),
                                "predicted": float(run.predictions[horizon_idx]),
                            }
                        )

            full_train = values.copy()
            latest_date = dates[-1]
            latest_runs = [
                forecast_naive(full_train, MAX_HORIZON),
                forecast_seasonal_naive(full_train, MAX_HORIZON),
                forecast_drift(full_train, MAX_HORIZON),
                forecast_holt_damped(full_train, MAX_HORIZON),
                forecast_arima(full_train, MAX_HORIZON),
                forecast_random_forest(full_train, MAX_HORIZON),
            ]
            future_dates = pd.date_range(latest_date + pd.Timedelta(days=1), periods=MAX_HORIZON, freq="D")
            for run in latest_runs:
                for horizon_idx in range(MAX_HORIZON):
                    latest_rows.append(
                        {
                            "target": target_col,
                            "target_label": target_label,
                            "city": city,
                            "model": run.model,
                            "latest_observed_date": latest_date,
                            "forecast_date": future_dates[horizon_idx],
                            "horizon_days": horizon_idx + 1,
                            "predicted": float(run.predictions[horizon_idx]),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    latest_forecasts = pd.DataFrame(latest_rows)
    predictions["abs_error"] = np.abs(predictions["actual"] - predictions["predicted"])
    predictions["squared_error"] = (predictions["actual"] - predictions["predicted"]) ** 2

    return predictions, latest_forecasts


def summarise(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    model_summary["model_priority"] = model_summary["model"].map(MODEL_PRIORITY)
    model_summary = (
        model_summary
        .sort_values(["target", "horizon_days", "mean_mae", "mean_rank", "model_priority"])
        .drop(columns=["model_priority"])
        .reset_index(drop=True)
    )

    return metrics_by_city, model_summary


def best_models(metrics_by_city: pd.DataFrame) -> pd.DataFrame:
    ordered = metrics_by_city.sort_values(["target", "city", "horizon_days", "mae", "rmse"])
    return ordered.groupby(["target", "city", "horizon_days"], as_index=False).first()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    predictions, latest_forecasts = run_benchmark()
    metrics_by_city, model_summary = summarise(predictions)
    best_by_city = best_models(metrics_by_city)

    predictions.to_csv(TABLE_DIR / "rolling_predictions.csv", index=False)
    metrics_by_city.to_csv(TABLE_DIR / "metrics_by_city.csv", index=False)
    model_summary.to_csv(TABLE_DIR / "model_summary.csv", index=False)
    best_by_city.to_csv(TABLE_DIR / "best_models_by_city.csv", index=False)
    latest_forecasts.to_csv(TABLE_DIR / "latest_forecasts.csv", index=False)

    metadata = {
        "targets": TARGETS,
        "models": MODEL_NAMES,
        "initial_train_days": INITIAL_TRAIN_DAYS,
        "max_horizon_days": MAX_HORIZON,
        "evaluation_stride_days": EVAL_STRIDE,
        "seasonal_naive_period_days": SEASONAL_PERIOD,
        "random_forest_lags": RF_LAGS,
        "arima_orders": ARIMA_ORDERS,
        "input_path": str(INPUT_PATH),
    }
    with (OUTPUT_DIR / "benchmark_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved forecasting benchmark outputs to:")
    print(f"  - {TABLE_DIR / 'rolling_predictions.csv'}")
    print(f"  - {TABLE_DIR / 'metrics_by_city.csv'}")
    print(f"  - {TABLE_DIR / 'model_summary.csv'}")
    print(f"  - {TABLE_DIR / 'best_models_by_city.csv'}")
    print(f"  - {TABLE_DIR / 'latest_forecasts.csv'}")


if __name__ == "__main__":
    main()
