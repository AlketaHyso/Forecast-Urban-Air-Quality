from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BENCHMARK_DIR = PROJECT_ROOT / "outputs" / "forecast_benchmark" / "tables"
ENHANCED_DIR = PROJECT_ROOT / "outputs" / "enhanced_forecast_benchmark" / "tables"
ALERT_DIR = PROJECT_ROOT / "outputs" / "operational_alerts"


def aqi_label(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score <= 20:
        return "Good"
    if score <= 40:
        return "Fair"
    if score <= 60:
        return "Moderate"
    if score <= 80:
        return "Poor"
    if score <= 100:
        return "Very Poor"
    return "Extremely Poor"


def aqi_level(score: float) -> int:
    levels = {
        "Unknown": 0,
        "Good": 1,
        "Fair": 2,
        "Moderate": 3,
        "Poor": 4,
        "Very Poor": 5,
        "Extremely Poor": 6,
    }
    return levels[aqi_label(score)]


def pm25_signal(value: float, p90: float, p95: float) -> str:
    if pd.isna(value):
        return "Unknown"
    if value >= p95:
        return "Extreme"
    if value >= p90:
        return "Elevated"
    return "Normal"


def final_alert_level(aqi_score: float, pm25_pred: float, p90: float, p95: float) -> str:
    level = aqi_level(aqi_score)
    signal = pm25_signal(pm25_pred, p90, p95)
    if level >= 6 or (level >= 5 and signal == "Extreme"):
        return "Critical"
    if level >= 5 or signal == "Extreme":
        return "Severe"
    if level >= 4 or signal == "Elevated":
        return "High"
    if level >= 3:
        return "Warning"
    if level >= 2:
        return "Watch"
    return "Stable"


def alert_score(alert: str) -> int:
    scores = {
        "Stable": 1,
        "Watch": 2,
        "Warning": 3,
        "High": 4,
        "Severe": 5,
        "Critical": 6,
    }
    return scores.get(alert, 0)


def recommended_action(alert: str) -> str:
    actions = {
        "Stable": "Routine monitoring only.",
        "Watch": "Monitor trends and review the next daily update.",
        "Warning": "Prepare local warning communication and monitor sensitive groups.",
        "High": "Issue caution for outdoor exposure and review city conditions closely.",
        "Severe": "Treat as a high-priority air-quality episode and intensify monitoring.",
        "Critical": "Escalate immediately and consider emergency public-health messaging.",
    }
    return actions.get(alert, "Review forecast outputs.")


def main() -> None:
    ALERT_DIR.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(PROCESSED_DIR / "albania_air_quality_daily.csv", parse_dates=["date"])
    baseline_best = pd.read_csv(BENCHMARK_DIR / "best_models_by_city.csv")
    baseline_latest = pd.read_csv(BENCHMARK_DIR / "latest_forecasts.csv", parse_dates=["latest_observed_date", "forecast_date"])
    enhanced_best = pd.read_csv(ENHANCED_DIR / "best_models_by_city.csv")
    enhanced_latest = pd.read_csv(ENHANCED_DIR / "latest_forecasts.csv", parse_dates=["latest_observed_date", "forecast_date"])

    latest_observed = (
        daily.sort_values(["city", "date"])
        .groupby("city", as_index=False)
        .tail(1)[["city", "date", "pm2_5_mean", "european_aqi_max", "aqi_label"]]
        .rename(
            columns={
                "date": "latest_observed_date",
                "pm2_5_mean": "latest_pm2_5_mean",
                "european_aqi_max": "latest_european_aqi_max",
                "aqi_label": "latest_aqi_label",
            }
        )
    )

    pm25_reference = (
        daily.groupby("city")["pm2_5_mean"]
        .agg(
            pm25_p90=lambda s: float(np.nanpercentile(s.dropna(), 90)),
            pm25_p95=lambda s: float(np.nanpercentile(s.dropna(), 95)),
        )
        .reset_index()
    )

    baseline_cmp = baseline_best.rename(
        columns={
            "model": "baseline_model",
            "mae": "baseline_mae",
            "rmse": "baseline_rmse",
            "mape": "baseline_mape",
            "rank_mae": "baseline_rank",
        }
    )
    enhanced_cmp = enhanced_best.rename(
        columns={
            "model": "enhanced_model",
            "mae": "enhanced_mae",
            "rmse": "enhanced_rmse",
            "mape": "enhanced_mape",
            "rank_mae": "enhanced_rank",
        }
    )
    champion = baseline_cmp.merge(
        enhanced_cmp,
        on=["target", "city", "horizon_days"],
        how="inner",
    )
    champion["selected_source"] = np.where(
        champion["enhanced_mae"] < champion["baseline_mae"],
        "enhanced",
        "baseline",
    )
    champion["selected_model"] = np.where(
        champion["selected_source"] == "enhanced",
        champion["enhanced_model"],
        champion["baseline_model"],
    )
    champion["selected_mae"] = np.where(
        champion["selected_source"] == "enhanced",
        champion["enhanced_mae"],
        champion["baseline_mae"],
    )

    selected_rows: list[dict] = []
    for _, row in champion.iterrows():
        latest_target = enhanced_latest if row["selected_source"] == "enhanced" else baseline_latest
        subset = latest_target[
            (latest_target["target"] == row["target"])
            & (latest_target["city"] == row["city"])
            & (latest_target["horizon_days"] == row["horizon_days"])
            & (latest_target["model"] == row["selected_model"])
        ]
        for _, forecast_row in subset.iterrows():
            selected_rows.append(
                {
                    "target": row["target"],
                    "city": row["city"],
                    "horizon_days": int(row["horizon_days"]),
                    "selected_source": row["selected_source"],
                    "selected_model": row["selected_model"],
                    "selected_mae": float(row["selected_mae"]),
                    "forecast_date": forecast_row["forecast_date"],
                    "predicted": float(forecast_row["predicted"]),
                }
            )

    selected = pd.DataFrame(selected_rows)
    selected_wide = (
        selected.pivot_table(
            index=["city", "forecast_date", "horizon_days"],
            columns="target",
            values="predicted",
            aggfunc="first",
        )
        .reset_index()
    )
    model_wide = (
        selected.pivot_table(
            index=["city", "forecast_date", "horizon_days"],
            columns="target",
            values="selected_model",
            aggfunc="first",
        )
        .reset_index()
    )
    source_wide = (
        selected.pivot_table(
            index=["city", "forecast_date", "horizon_days"],
            columns="target",
            values="selected_source",
            aggfunc="first",
        )
        .reset_index()
    )

    selected_wide = selected_wide.merge(
        model_wide,
        on=["city", "forecast_date", "horizon_days"],
        suffixes=("", "_model"),
    )
    selected_wide = selected_wide.merge(
        source_wide,
        on=["city", "forecast_date", "horizon_days"],
        suffixes=("", "_source"),
    )
    selected_wide = selected_wide.merge(latest_observed, on="city", how="left")
    selected_wide = selected_wide.merge(pm25_reference, on="city", how="left")

    selected_wide["predicted_aqi_label"] = selected_wide["european_aqi_max"].apply(aqi_label)
    selected_wide["predicted_aqi_level"] = selected_wide["european_aqi_max"].apply(aqi_level)
    selected_wide["pm25_signal"] = selected_wide.apply(
        lambda row: pm25_signal(row["pm2_5_mean"], row["pm25_p90"], row["pm25_p95"]), axis=1
    )
    selected_wide["final_alert_level"] = selected_wide.apply(
        lambda row: final_alert_level(
            row["european_aqi_max"], row["pm2_5_mean"], row["pm25_p90"], row["pm25_p95"]
        ),
        axis=1,
    )
    selected_wide["alert_score"] = selected_wide["final_alert_level"].apply(alert_score)
    selected_wide["recommended_action"] = selected_wide["final_alert_level"].apply(recommended_action)
    selected_wide["pm2_5_change"] = selected_wide["pm2_5_mean"] - selected_wide["latest_pm2_5_mean"]
    selected_wide["aqi_change"] = selected_wide["european_aqi_max"] - selected_wide["latest_european_aqi_max"]

    selected_wide = selected_wide.rename(
        columns={
            "pm2_5_mean": "predicted_pm2_5_mean",
            "european_aqi_max": "predicted_european_aqi_max",
            "pm2_5_mean_model": "selected_pm2_5_model",
            "european_aqi_max_model": "selected_aqi_model",
            "pm2_5_mean_source": "selected_pm2_5_source",
            "european_aqi_max_source": "selected_aqi_source",
        }
    )

    ranking = (
        selected_wide.sort_values(
            ["city", "alert_score", "predicted_european_aqi_max", "predicted_pm2_5_mean"],
            ascending=[True, False, False, False],
        )
        .groupby("city", as_index=False)
        .first()[
            [
                "city",
                "forecast_date",
                "horizon_days",
                "alert_score",
                "predicted_aqi_label",
                "predicted_european_aqi_max",
                "predicted_pm2_5_mean",
                "aqi_change",
                "pm2_5_change",
                "final_alert_level",
                "recommended_action",
                "selected_aqi_model",
                "selected_pm2_5_model",
                "selected_aqi_source",
                "selected_pm2_5_source",
            ]
        ]
        .rename(
            columns={
                "forecast_date": "peak_risk_date",
                "horizon_days": "peak_horizon_days",
                "alert_score": "max_alert_score",
                "predicted_european_aqi_max": "peak_predicted_aqi",
                "predicted_pm2_5_mean": "peak_predicted_pm25",
                "aqi_change": "max_aqi_change",
                "pm2_5_change": "max_pm25_change",
            }
        )
    )
    ranking = ranking.sort_values(
        ["max_alert_score", "peak_predicted_aqi", "peak_predicted_pm25"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    heatmap = selected_wide[
        ["city", "forecast_date", "horizon_days", "alert_score", "final_alert_level", "predicted_european_aqi_max"]
    ].copy()

    selected_wide.to_csv(ALERT_DIR / "forecast_alerts.csv", index=False)
    ranking.to_csv(ALERT_DIR / "city_attention_ranking.csv", index=False)
    heatmap.to_csv(ALERT_DIR / "alert_heatmap_table.csv", index=False)
    champion.to_csv(ALERT_DIR / "champion_model_selection.csv", index=False)

    issue_date = pd.to_datetime(selected_wide["latest_observed_date"]).max()
    forecast_start = pd.to_datetime(selected_wide["forecast_date"]).min()
    forecast_end = pd.to_datetime(selected_wide["forecast_date"]).max()

    metadata = {
        "primary_signal": "predicted_european_aqi_max",
        "support_signal": "predicted_pm2_5_mean compared to city historical p90/p95",
        "case_study_issue_date": issue_date.strftime("%Y-%m-%d"),
        "forecast_window_start": forecast_start.strftime("%Y-%m-%d"),
        "forecast_window_end": forecast_end.strftime("%Y-%m-%d"),
        "pm25_reference_window": f"Daily PM2.5 history available up to the issue date {issue_date.strftime('%Y-%m-%d')}",
        "selection_logic": "For each city, target, and horizon, choose the lower-MAE model family between baseline and enhanced benchmarks.",
        "alert_levels": ["Stable", "Watch", "Warning", "High", "Severe", "Critical"],
        "generated_files": [
            "forecast_alerts.csv",
            "city_attention_ranking.csv",
            "alert_heatmap_table.csv",
            "champion_model_selection.csv",
        ],
    }
    with (ALERT_DIR / "alert_logic_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved operational alert outputs to:")
    for name in metadata["generated_files"]:
        print(f"  - {ALERT_DIR / name}")


if __name__ == "__main__":
    main()
