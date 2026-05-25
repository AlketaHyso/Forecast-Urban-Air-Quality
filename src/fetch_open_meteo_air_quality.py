from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

START_DATE = "2024-01-01"
END_DATE = (date.today() - timedelta(days=1)).isoformat()
TIMEZONE = "Europe/Tirane"

AIR_VARS = [
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
]

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "pressure_msl",
]

CITIES = [
    {"city": "Tirane", "latitude": 41.3275, "longitude": 19.8187},
    {"city": "Durres", "latitude": 41.3231, "longitude": 19.4414},
    {"city": "Elbasan", "latitude": 41.1125, "longitude": 20.0822},
    {"city": "Shkoder", "latitude": 42.0683, "longitude": 19.5126},
    {"city": "Vlore", "latitude": 40.4661, "longitude": 19.4914},
    {"city": "Fier", "latitude": 40.7239, "longitude": 19.5560},
    {"city": "Korce", "latitude": 40.6186, "longitude": 20.7808},
    {"city": "Berat", "latitude": 40.7058, "longitude": 19.9522},
]


def fetch_json(base_url: str, params: dict[str, str]) -> dict:
    url = f"{base_url}?{urlencode(params)}"
    with urlopen(url) as response:
        return json.load(response)


def rows_from_hourly(city: dict[str, float], payload: dict, variables: list[str]) -> list[dict]:
    hourly = payload["hourly"]
    rows: list[dict] = []
    timestamps = hourly["time"]
    for idx, timestamp in enumerate(timestamps):
        row = {
            "city": city["city"],
            "latitude": city["latitude"],
            "longitude": city["longitude"],
            "timestamp": timestamp,
        }
        for variable in variables:
            values = hourly.get(variable, [])
            row[variable] = values[idx] if idx < len(values) else None
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def merge_rows(air_rows: list[dict], weather_rows: list[dict]) -> list[dict]:
    weather_lookup = {
        (row["city"], row["timestamp"]): row for row in weather_rows
    }
    merged: list[dict] = []
    for air_row in air_rows:
        key = (air_row["city"], air_row["timestamp"])
        combined = dict(air_row)
        weather_row = weather_lookup.get(key, {})
        for key_name, value in weather_row.items():
            if key_name not in combined:
                combined[key_name] = value
        merged.append(combined)
    return merged


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_air_rows: list[dict] = []
    all_weather_rows: list[dict] = []

    for city in CITIES:
        air_payload = fetch_json(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            {
                "latitude": str(city["latitude"]),
                "longitude": str(city["longitude"]),
                "hourly": ",".join(AIR_VARS),
                "timezone": TIMEZONE,
                "domains": "cams_europe",
                "start_date": START_DATE,
                "end_date": END_DATE,
            },
        )
        weather_payload = fetch_json(
            "https://archive-api.open-meteo.com/v1/archive",
            {
                "latitude": str(city["latitude"]),
                "longitude": str(city["longitude"]),
                "hourly": ",".join(WEATHER_VARS),
                "timezone": TIMEZONE,
                "start_date": START_DATE,
                "end_date": END_DATE,
            },
        )

        air_rows = rows_from_hourly(city, air_payload, AIR_VARS)
        weather_rows = rows_from_hourly(city, weather_payload, WEATHER_VARS)
        all_air_rows.extend(air_rows)
        all_weather_rows.extend(weather_rows)

    merged_rows = merge_rows(all_air_rows, all_weather_rows)

    write_csv(RAW_DIR / "albania_air_quality_hourly_raw.csv", all_air_rows)
    write_csv(RAW_DIR / "albania_weather_hourly_raw.csv", all_weather_rows)
    write_csv(PROCESSED_DIR / "albania_air_quality_merged_hourly.csv", merged_rows)
    write_csv(PROCESSED_DIR / "city_catalog.csv", CITIES)

    metadata = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": TIMEZONE,
        "cities": CITIES,
        "air_variables": AIR_VARS,
        "weather_variables": WEATHER_VARS,
        "sources": {
            "air_quality": "https://open-meteo.com/en/docs/air-quality-api",
            "weather": "https://open-meteo.com/en/docs/historical-weather-api",
        },
    }
    with (PROCESSED_DIR / "download_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved raw and processed files to:")
    print(f"  - {RAW_DIR}")
    print(f"  - {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
