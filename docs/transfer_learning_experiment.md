# Transfer Learning Experiment Design

This note documents the standalone transfer-learning benchmark added after the baseline and enhanced forecasting experiments. It does not replace or overwrite the previous benchmarks; it adds a third modeling branch with separate outputs.

## Goal

The transfer-learning experiment was designed to stay methodologically aligned with the existing study:

- same daily analytical dataset
- same two targets:
  - `pm2_5_mean`
  - `european_aqi_max`
- same forecast horizons:
  - 1 day
  - 2 days
  - 3 days
- same rolling-origin evaluation schedule
- same 365-day rolling training window
- same 28-day evaluation stride
- same engineered feature space used by the enhanced pooled benchmark

## Transfer-Learning Logic

The experiment uses a two-stage neural workflow built on the enhanced tabular feature set:

1. A pooled multilayer perceptron (MLP) is pretrained on the full rolling training window across all cities.
2. A copy of the pretrained model is then fine-tuned on the target city's local rows from the same rolling training window.
3. The fine-tuned city-specific model generates the forecast for that city and horizon.

This creates a transfer path from pooled cross-city learning to local city adaptation while preserving the study's leakage-aware chronological design.

## Model Configuration

- model name: `MLP Transfer Learning`
- hidden layers: `(48, 24)`
- activation: `relu`
- solver: `adam`
- pretraining iterations: `120`
- local fine-tuning iterations: `40`
- local fine-tuning minimum rows: `120`
- feature scaling: `StandardScaler`, fit on the pooled rolling training data

## Output Locations

Benchmark outputs:

- `outputs/transfer_learning_benchmark/tables/`

Publication-ready assets:

- `outputs/transfer_learning_publication_assets/tables/`
- `outputs/transfer_learning_publication_assets/figures/single_column/`
- `outputs/transfer_learning_publication_assets/figures/double_column/`

## Main Comparison Files

- `comparison_vs_existing.csv`
  - compares transfer learning against the best enhanced and best baseline models at the family-summary level
- `city_family_comparison.csv`
  - city-level MAE comparison among baseline, enhanced, and transfer families
- `family_win_counts.csv`
  - number of city-level wins by family for each target and horizon

## Interpretation

This experiment should be interpreted as an additional benchmark branch rather than as a replacement for the previous experiments. The primary question it addresses is whether a pooled-to-local fine-tuning step can improve short-horizon forecasting relative to the existing baseline and enhanced model families.
