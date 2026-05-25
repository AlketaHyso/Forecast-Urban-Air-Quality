# Model Strengthening Design

## Why a Second Benchmark?

The first benchmark compared local univariate methods city by city. That was a good baseline, but it left two important modeling opportunities underused:

1. engineered lag and rolling features
2. pooled learning across cities

## Strengthening Strategy

The enhanced benchmark introduces global tabular models trained across all cities for each target and horizon.

## Main Changes

- direct forecasting by horizon instead of only recursive local forecasting
- richer autoregressive features
- rolling summary features
- calendar features
- weather context features
- shared learning across cities through city indicators

## Enhanced Model Set

1. Persistence Current
2. Ridge Global Features
3. HistGBM Global Features

## Fairness Principle

The enhanced models use only information available up to the forecast issue date. They do not use future weather values or future pollution observations.

## Comparison Goal

The purpose is not to force a more complex model to win, but to test whether stronger feature engineering and pooled learning can outperform the first benchmark's best local models.
