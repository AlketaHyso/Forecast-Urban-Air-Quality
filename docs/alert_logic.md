# Alert Logic

## Purpose

The operational alert layer converts model outputs into city-level attention signals for the next 1-3 days.

## Inputs

- champion model selection by city, target, and horizon
- latest PM2.5 forecasts
- latest AQI forecasts
- latest observed daily conditions
- city-level historical PM2.5 percentiles

## Model Selection Rule

For each city, target, and horizon, the system compares:

- the best local model from the baseline benchmark
- the best city-level model from the enhanced benchmark

The operational pipeline selects the family with the lower MAE and uses that model's latest forecast in the alert layer.

## Primary and Supporting Signals

### Primary signal

- predicted daily max European AQI

### Supporting signal

- predicted daily mean PM2.5 relative to each city's historical p90 and p95

## Alert Levels

- Stable
- Watch
- Warning
- High
- Severe
- Critical

## Logic Summary

- AQI drives the base severity level
- PM2.5 can escalate the alert when the forecast is extreme relative to that city's own historical distribution
- the forecast source itself is chosen by benchmark performance rather than fixed in advance
- the final city ranking is based on the strongest alert across the 1-3 day forecast window

## Operational Outputs

- forecast alert table by city and horizon
- city attention ranking
- alert heatmap table for notebook visualization
- champion model selection table
