from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED_DIR / "albania_air_quality_merged_hourly.csv"


NUMERIC_COLUMNS = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "carbon_monoxide",
    "european_aqi",
    "european_aqi_pm2_5",
    "european_aqi_pm10",
    "european_aqi_nitrogen_dioxide",
    "european_aqi_ozone",
    "european_aqi_sulphur_dioxide",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "pressure_msl",
]


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
    mapping = {
        "Unknown": 0,
        "Good": 1,
        "Fair": 2,
        "Moderate": 3,
        "Poor": 4,
        "Very Poor": 5,
        "Extremely Poor": 6,
    }
    return mapping[aqi_label(score)]


def dominant_pollutant(row: pd.Series) -> str:
    candidates = {
        "PM2.5": row.get("european_aqi_pm2_5_mean"),
        "PM10": row.get("european_aqi_pm10_mean"),
        "NO2": row.get("european_aqi_nitrogen_dioxide_mean"),
        "O3": row.get("european_aqi_ozone_mean"),
        "SO2": row.get("european_aqi_sulphur_dioxide_mean"),
    }
    clean = {name: value for name, value in candidates.items() if pd.notna(value)}
    if not clean:
        return "Unknown"
    return max(clean, key=clean.get)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    agg_spec: dict[str, list[str]] = {
        "pm2_5": ["mean", "max"],
        "pm10": ["mean", "max"],
        "nitrogen_dioxide": ["mean", "max"],
        "ozone": ["mean", "max"],
        "sulphur_dioxide": ["mean", "max"],
        "carbon_monoxide": ["mean", "max"],
        "european_aqi": ["mean", "max"],
        "european_aqi_pm2_5": ["mean", "max"],
        "european_aqi_pm10": ["mean", "max"],
        "european_aqi_nitrogen_dioxide": ["mean", "max"],
        "european_aqi_ozone": ["mean", "max"],
        "european_aqi_sulphur_dioxide": ["mean", "max"],
        "temperature_2m": ["mean", "min", "max"],
        "relative_humidity_2m": ["mean"],
        "precipitation": ["sum"],
        "wind_speed_10m": ["mean", "max"],
        "pressure_msl": ["mean"],
        "hour": ["count"],
    }

    daily = (
        df.groupby(["city", "latitude", "longitude", "date", "year", "month", "day"], as_index=False)
        .agg(agg_spec)
    )

    daily.columns = [
        "_".join([part for part in col if part]).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in daily.columns
    ]
    daily = daily.rename(columns={"hour_count": "hours_observed"})

    daily["aqi_label"] = daily["european_aqi_max"].apply(aqi_label)
    daily["aqi_level"] = daily["european_aqi_max"].apply(aqi_level)
    daily["pollution_episode"] = daily["aqi_level"] >= 4
    daily["high_risk_day"] = daily["aqi_level"] >= 5
    daily["dominant_pollutant"] = daily.apply(dominant_pollutant, axis=1)

    daily["pm2_5_exceedance"] = daily["pm2_5_max"] > 25
    daily["pm10_exceedance"] = daily["pm10_max"] > 50
    daily["no2_exceedance"] = daily["nitrogen_dioxide_max"] > 90
    daily["o3_exceedance"] = daily["ozone_max"] > 100

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["city", "date"]).reset_index(drop=True)

    summary = (
        daily.groupby("city", as_index=False)
        .agg(
            days_observed=("date", "count"),
            avg_pm2_5=("pm2_5_mean", "mean"),
            avg_pm10=("pm10_mean", "mean"),
            avg_no2=("nitrogen_dioxide_mean", "mean"),
            avg_aqi=("european_aqi_mean", "mean"),
            max_aqi=("european_aqi_max", "max"),
            episode_days=("pollution_episode", "sum"),
            high_risk_days=("high_risk_day", "sum"),
        )
    )
    summary["episode_share_pct"] = (summary["episode_days"] / summary["days_observed"] * 100).round(2)
    summary["high_risk_share_pct"] = (summary["high_risk_days"] / summary["days_observed"] * 100).round(2)
    summary = summary.sort_values(["high_risk_days", "episode_days", "avg_aqi"], ascending=[False, False, False])

    latest_date = daily["date"].max()
    latest_snapshot = daily[daily["date"] == latest_date].copy()
    latest_snapshot = latest_snapshot.sort_values("european_aqi_max", ascending=False)

    events = daily[daily["pollution_episode"]].copy()
    events = events.sort_values(["aqi_level", "european_aqi_max", "pm2_5_max"], ascending=[False, False, False])

    daily.to_csv(PROCESSED_DIR / "albania_air_quality_daily.csv", index=False)
    summary.to_csv(PROCESSED_DIR / "city_risk_summary.csv", index=False)
    latest_snapshot.to_csv(PROCESSED_DIR / "latest_city_snapshot.csv", index=False)
    events.to_csv(PROCESSED_DIR / "pollution_episode_days.csv", index=False)

    metadata = {
        "aqi_label_scheme": {
            "Good": "0-20",
            "Fair": "21-40",
            "Moderate": "41-60",
            "Poor": "61-80",
            "Very Poor": "81-100",
            "Extremely Poor": ">100",
        },
        "episode_rule": "pollution_episode = aqi_level >= 4",
        "high_risk_rule": "high_risk_day = aqi_level >= 5",
        "daily_outputs": [
            "albania_air_quality_daily.csv",
            "city_risk_summary.csv",
            "latest_city_snapshot.csv",
            "pollution_episode_days.csv",
        ],
    }
    with (PROCESSED_DIR / "daily_pipeline_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Created daily analytical datasets:")
    for name in metadata["daily_outputs"]:
        print(f"  - {PROCESSED_DIR / name}")


if __name__ == "__main__":
    main()
