## Supplementary Note S1. Reproducibility Details

This supplementary note summarizes the materials and execution logic needed to reproduce the data pipeline, benchmark experiments, and publication outputs reported in the study. The workflow was implemented as a script-based project that covers data acquisition, daily preprocessing, local and pooled benchmark execution, operational alert generation, and publication-asset export.

### S1. Project structure

The main project directories are:

- `src/` for data collection, preprocessing, benchmarking, and output-generation scripts
- `data/raw/` for downloaded hourly air-quality and weather files
- `data/processed/` for merged hourly data, daily analytical datasets, and pipeline metadata
- `outputs/forecast_benchmark/` for local baseline benchmark outputs
- `outputs/enhanced_forecast_benchmark/` for pooled feature-based benchmark outputs
- `outputs/operational_alerts/` for champion selection and alert-layer outputs
- `outputs/publication_assets/` for publication-ready figures and tables

### S2. Data acquisition

Hourly air-quality and meteorological data were collected with the script `src/fetch_open_meteo_air_quality.py`. The script queries the Open-Meteo Air Quality API and the Open-Meteo Historical Weather API separately for each city and then merges the hourly responses by city and timestamp.

The fixed project settings used for the reported dataset were:

- study cities: Tirane, Durres, Elbasan, Shkoder, Vlore, Fier, Korce, and Berat
- timezone: `Europe/Tirane`
- air-quality date range: `2024-01-01` to `2026-04-28`
- air-quality domain: `cams_europe`

The API query structure is fully specified in `src/fetch_open_meteo_air_quality.py` and archived in machine-readable form in `data/processed/download_metadata.json`.

The air-quality endpoint template was:

```text
https://air-quality-api.open-meteo.com/v1/air-quality
  ?latitude={latitude}
  &longitude={longitude}
  &hourly=pm2_5,pm10,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide,european_aqi,european_aqi_pm2_5,european_aqi_pm10,european_aqi_nitrogen_dioxide,european_aqi_ozone,european_aqi_sulphur_dioxide
  &timezone=Europe/Tirane
  &domains=cams_europe
  &start_date=2024-01-01
  &end_date=2026-04-28
```

The weather endpoint template was:

```text
https://archive-api.open-meteo.com/v1/archive
  ?latitude={latitude}
  &longitude={longitude}
  &hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,pressure_msl
  &timezone=Europe/Tirane
  &start_date=2024-01-01
  &end_date=2026-04-28
```

The script writes the following core files:

- `data/raw/albania_air_quality_hourly_raw.csv`
- `data/raw/albania_weather_hourly_raw.csv`
- `data/processed/albania_air_quality_merged_hourly.csv`
- `data/processed/city_catalog.csv`
- `data/processed/download_metadata.json`

The current project metadata record the source URLs, city coordinates, queried variables, and date range. Documentation access timestamps are not recorded automatically by the scripts, so web access dates should be inserted manually in the final manuscript reference list.

### S3. Daily preprocessing and labeling

Daily preprocessing was carried out with `src/prepare_daily_air_quality.py`. The script converts timestamps to datetime format, derives calendar fields, coerces environmental variables to numeric form, and aggregates hourly observations to city-day level.

The retained daily summaries include:

- pollutant means and maxima for PM2.5, PM10, NO2, O3, SO2, and carbon monoxide
- AQI means and maxima, together with pollutant-specific AQI sub-index means and maxima
- meteorological summaries including temperature mean/minimum/maximum, relative humidity mean, precipitation sum, wind speed mean/maximum, and mean sea-level pressure
- `hours_observed` for daily completeness checking

AQI categories were assigned from `european_aqi_max` using the following thresholds:

- Good: `0-20`
- Fair: `21-40`
- Moderate: `41-60`
- Poor: `61-80`
- Very Poor: `81-100`
- Extremely Poor: `>100`

The script also creates:

- `pollution_episode = aqi_level >= 4`
- `high_risk_day = aqi_level >= 5`
- daily dominant-pollutant labels based on pollutant-specific AQI sub-index means

The preprocessing script exports:

- `data/processed/albania_air_quality_daily.csv`
- `data/processed/city_risk_summary.csv`
- `data/processed/latest_city_snapshot.csv`
- `data/processed/pollution_episode_days.csv`
- `data/processed/daily_pipeline_metadata.json`

### S4. Local baseline benchmark

The local benchmark was executed with `src/run_forecasting_benchmark.py`. Each model was fitted independently for each city-target series under a rolling-origin backtesting design.

The baseline model family consisted of:

1. Naive
2. Seasonal Naive with a 7-day period
3. Drift
4. Holt Damped
5. ARIMA with restricted candidate orders `(1,1,0)`, `(1,1,1)`, and `(0,1,1)`
6. Random Forest Lag7

The benchmark configuration was:

- targets: daily mean `pm2_5_mean` and daily maximum `european_aqi_max`
- training window: 365 days
- forecast horizons: 1, 2, and 3 days
- rolling-origin stride: 28 days
- Random Forest lag count: 7
- Random Forest settings: `n_estimators = 120`, `min_samples_leaf = 2`, `random_state = 42`

The script writes benchmark outputs and metadata, including:

- `outputs/forecast_benchmark/tables/model_summary.csv`
- `outputs/forecast_benchmark/tables/best_models_by_city.csv`
- `outputs/forecast_benchmark/tables/latest_forecasts.csv`
- `outputs/forecast_benchmark/benchmark_metadata.json`

### S5. Enhanced pooled benchmark

The pooled feature-based benchmark was executed with `src/run_enhanced_forecasting_benchmark.py`. Forecasting was reformulated as a supervised learning problem across cities, with separate direct targets defined for the 1-day, 2-day, and 3-day horizons.

The enhanced model family consisted of:

1. Persistence Current
2. Ridge Global Features
3. HistGradientBoosting Global Features

The fixed pooled feature design included:

- current target value on issue date `t`
- lagged target values at `t-1`, `t-2`, `t-3`, `t-7`, `t-14`, `t-21`, and `t-28`
- rolling target means, standard deviations, and maxima over 3-, 7-, 14-, and 28-day windows ending at `t-1`
- calendar fields and sinusoidal seasonal encodings
- issue-day meteorological covariates
- issue-day cross-pollutant context variables
- one-hot encoded city indicators

The pooled benchmark settings were:

- training window: 365 days
- forecast horizons: 1, 2, and 3 days
- rolling-origin stride: 28 days
- minimum pooled training rows: 400
- Ridge setting: `alpha = 2.0`
- HistGradientBoosting settings: `learning_rate = 0.05`, `max_depth = 4`, `max_iter = 120`, `min_samples_leaf = 20`, `random_state = 42`

Temporal alignment follows an end-of-day forecasting design. All predictors are constructed using information available by the end of issue date `t`, while the target is evaluated on day `t + h`. Issue-day meteorological and cross-pollutant covariates are measured on day `t`, lagged target features are drawn from `t-1` backward, and rolling summaries use windows ending at `t-1`.

The script writes:

- `outputs/enhanced_forecast_benchmark/tables/model_summary.csv`
- `outputs/enhanced_forecast_benchmark/tables/best_models_by_city.csv`
- `outputs/enhanced_forecast_benchmark/tables/latest_forecasts.csv`
- `outputs/enhanced_forecast_benchmark/enhanced_benchmark_metadata.json`

### S6. Operational alert layer

The operational layer was produced with `src/generate_operational_alerts.py`. It combines the lower-error champion models from the local and enhanced benchmark families and translates their AQI and PM2.5 forecasts into alert levels.

The alert mapping uses:

- AQI-driven default levels: Stable, Watch, Warning, High, Severe, and Critical
- PM2.5 escalation relative to city-specific 90th and 95th percentiles
- a joint AQI-PM2.5 rule for the highest alert levels

The operational script writes:

- `outputs/operational_alerts/champion_model_selection.csv`
- `outputs/operational_alerts/selected_forecasts.csv`
- `outputs/operational_alerts/alert_ranking.csv`
- additional city-level alert-layer outputs used for figures and case-study reporting

### S7. Publication outputs

Publication-ready figures and tables were generated with `src/generate_publication_assets.py`. This script reads the benchmark and operational outputs and exports the summary materials used in the manuscript.

The exported materials are archived under:

- `outputs/publication_assets/tables/`
- `outputs/publication_assets/figures/`

### S8. Software environment

The analyses reported in the current project environment were executed with:

- Python `3.12.2`
- `pandas 2.2.1`
- `numpy 1.26.4`
- `statsmodels 0.14.6`
- `scikit-learn 1.4.1.post1`
- `plotly 6.6.0`
- `ipywidgets 8.1.7`
- `jupyterlab 4.1.2`
- `xgboost 3.0.3` for the supplementary XGBoost experiments

The project dependency manifest is stored in `requirements.txt`. The current manifest lists the required packages but does not pin exact versions, so the version list above should be treated as the executed environment used for the present results.

### S9. Recommended execution order

The core pipeline can be rerun from the project root with:

```powershell
python src\fetch_open_meteo_air_quality.py
python src\prepare_daily_air_quality.py
python src\run_forecasting_benchmark.py
python src\run_enhanced_forecasting_benchmark.py
python src\generate_operational_alerts.py
python src\generate_publication_assets.py
```

Supplementary analyses are produced by the corresponding scripts in `src/`, including:

- `src/run_enhanced_feature_group_ablation.py`
- `src/run_city_specific_feature_benchmark.py`
- `src/run_xgboost_feature_benchmark.py`
- `src/run_xgboost_time_split_tuning.py`
- `src/run_main_benchmark_bootstrap.py`

### S10. Practical reproducibility scope

The workflow is reproducible in the sense that the project archive includes the data-query logic, preprocessing rules, benchmark configuration, metadata files, and script sequence required to regenerate the reported outputs from the same Open-Meteo sources and date range. The main remaining reproducibility gap is that web-documentation access dates are not logged automatically and package versions were not originally pinned in `requirements.txt`; both should be finalized in the archived repository or publication supplement.
