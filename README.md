# Albania Air-Quality Forecasting Benchmark

Benchmark-driven short-horizon urban air-quality forecasting for Albanian cities using open gridded environmental data.

## Overview

This repository contains the data pipeline, forecasting benchmarks, operational alert logic, and publication assets used for a reproducible urban air-quality forecasting workflow for Albania. The study focuses on eight Albanian cities and evaluates short-horizon forecasts for:

- daily mean PM2.5
- daily maximum European Air Quality Index (AQI)

The main goal is not to identify one universally best forecasting model, but to compare local and pooled model families under a common benchmark design and use those results to guide operational model selection.

## Study scope

- Cities: Tirane, Durres, Elbasan, Shkoder, Vlore, Fier, Korce, Berat
- Data frequency: hourly inputs aggregated to daily resolution
- Study period: 2024-01-01 to 2026-04-28
- Forecast horizons: 1, 2, and 3 days ahead
- Core targets: `pm2_5_mean` and `european_aqi_max`

## Data sources

The workflow uses openly accessible environmental data from Open-Meteo:

- Air Quality API: CAMS Europe-based gridded air-quality products
- Historical Weather API: hourly meteorological covariates used as contextual predictors

The data-collection logic is implemented in `src/fetch_open_meteo_air_quality.py`.

## Repository structure

- `src/` data collection, preprocessing, forecasting, alerting, and publication scripts
- `docs/` manuscript notes, methodological notes, and supplementary materials
- `data/raw/` downloaded hourly source files
- `data/processed/` merged hourly data, daily analytical tables, and metadata
- `outputs/forecast_benchmark/` local baseline benchmark outputs
- `outputs/enhanced_forecast_benchmark/` pooled feature-based benchmark outputs
- `outputs/operational_alerts/` champion selection and alert-layer outputs
- `outputs/publication_assets/` publication-ready tables and figures

## Main workflow

The core pipeline can be reproduced from the project root with:

```powershell
python src\fetch_open_meteo_air_quality.py
python src\prepare_daily_air_quality.py
python src\run_forecasting_benchmark.py
python src\run_enhanced_forecasting_benchmark.py
python src\generate_operational_alerts.py
python src\generate_publication_assets.py
```

## Benchmark design

### Local baseline benchmark

The local benchmark evaluates the following model family independently for each city-target series:

1. Naive
2. Seasonal Naive (7-day period)
3. Drift
4. Holt Damped
5. ARIMA
6. Random Forest Lag7

### Enhanced pooled benchmark

The pooled benchmark evaluates:

1. Persistence Current
2. Ridge Global Features
3. HistGradientBoosting Global Features

The pooled feature design combines lagged target values, rolling summaries, calendar features, seasonal encodings, issue-day meteorological covariates, cross-pollutant context variables, and city indicators.

### Evaluation setup

- 365-day rolling training window
- 28-day rolling-origin stride
- strictly chronological train-test separation
- 1-, 2-, and 3-day direct forecast horizons

## Operational layer

The repository also includes an illustrative operational alerting layer that:

- selects champion models from the benchmark results
- converts AQI and PM2.5 forecasts into alert levels
- ranks cities by short-term forecasted risk

This operational layer should be interpreted as a prototype built on benchmarked forecasts rather than as a fully validated public warning service.

## Reproducibility materials

The project includes metadata and supplementary documentation intended to support reproducibility, including:

- API query logic and download metadata
- daily preprocessing and AQI-label rules
- benchmark metadata for local and enhanced experiments
- supplementary reproducibility notes in `docs/`

For a concise editorial supplement, see:

- `docs/supplementary_reproducibility_note_editorial.md`
- `docs/Supplementary_Note_S1_Reproducibility_Details.docx`

## Software environment

The reported experiments were executed in Python 3.12.2 using:

- `pandas 2.2.1`
- `numpy 1.26.4`
- `statsmodels 0.14.6`
- `scikit-learn 1.4.1.post1`
- `plotly 6.6.0`
- `ipywidgets 8.1.7`
- `jupyterlab 4.1.2`
- `xgboost 3.0.3` for supplementary XGBoost experiments

The dependency manifest is listed in `requirements.txt`.

## Main outputs

Examples of generated outputs include:

- benchmark summaries by model, city, target, and horizon
- latest forecast tables
- champion model selections
- city alert rankings
- publication-ready figures and tables

## Manuscript support

This repository accompanies a benchmark-driven forecasting study on urban air quality in Albania. The codebase is organized to support:

- transparent benchmark comparison
- reproducible script-based execution
- supplementary methodological documentation
- manuscript table and figure generation
