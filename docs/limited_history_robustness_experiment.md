# Limited-History Robustness Experiment

This note documents the constrained-history robustness experiment added as a separate methodological extension to the main forecasting study. It does not alter the baseline benchmark or the enhanced pooled benchmark. Instead, it tests whether pooled cross-city learning remains useful when local historical data are limited.

## Scope

- target: `pm2_5_mean`
- forecast horizon: `1` day ahead
- model family: `HistGBM Global Features`

The experiment focuses on the short-horizon PM2.5 setting because it is operationally relevant and because it provides a direct test of whether pooled learning can retain value when a city has limited local data.

## Shared Design Principles

The experiment follows the same core logic as the main enhanced benchmark:

- same daily analytical dataset
- same leakage-aware chronological ordering
- same rolling-origin evaluation schedule
- same 365-day rolling training window
- same 28-day evaluation stride
- same feature-construction pipeline used in the enhanced pooled model

To guarantee alignment with the main study, the `full_pooled_reference` regime is checked internally against the existing `HistGBM Global Features` benchmark for `PM2.5` at the `1`-day horizon. The saved alignment tables confirm exact agreement for the main performance metrics.

## Training Regimes

Four training regimes are compared:

1. `Local Only`
   - the model is trained only on rows from the target city inside the rolling 365-day training window
2. `Zero-Shot Cross-City`
   - the model is trained on rows from all other cities, without using target-city rows
3. `Few-Shot Cross-City (90d)`
   - the model is trained on all other-city rows plus only the most recent 90 days of target-city rows
4. `Full Pooled Reference`
   - the same unconstrained pooled multi-city HistGBM regime used in the main enhanced benchmark

## Metrics

The experiment stores the following evaluation metrics:

- mean absolute error (`MAE`)
- root mean squared error (`RMSE`)
- mean absolute percentage error (`MAPE`)
- symmetric mean absolute percentage error (`sMAPE`)
- mean absolute scaled error (`MASE`)

`MASE` is scaled using the target city's own local historical training sequence at each rolling origin so that regime comparisons remain target-city specific and directly comparable.

## Output Locations

Raw experiment outputs:

- `outputs/limited_history_robustness/tables/`

Publication-ready assets:

- `outputs/limited_history_publication_assets/tables/`
- `outputs/limited_history_publication_assets/figures/single_column/`
- `outputs/limited_history_publication_assets/figures/double_column/`

## Intended Interpretation

This experiment is meant as a robustness check for the main pooled-learning result, not as a replacement for the original benchmark. The key question is whether cross-city pooled learning remains practically useful when the target city has restricted local history, and whether a small local adaptation window can improve on both local-only training and the unconstrained pooled reference.
