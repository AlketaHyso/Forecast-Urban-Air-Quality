# Forecasting Design

## Purpose

The forecasting layer extends the monitoring prototype into an early-warning prototype by evaluating short-horizon prediction methods for:

- daily mean PM2.5
- daily max European AQI

## Benchmark Setup

- geographic scope: 8 Albanian cities
- targets: 2
- models: 5
- horizons: 1, 2, and 3 days ahead
- evaluation method: rolling-origin backtesting

## Models

1. Naive
2. Drift
3. Holt Damped
4. ARIMA with restricted order search
5. Random Forest with 7 lagged values

## Design Choices

The benchmark uses a 365-day rolling training window and rolls the origin forward with a 14-day stride. This keeps the evaluation computationally reasonable while preserving repeated out-of-sample testing across seasons and pollution regimes.

## Main Outputs

- rolling predictions
- city-level performance metrics
- cross-city model summary
- best model by city and horizon
- latest 1-3 day forecasts
