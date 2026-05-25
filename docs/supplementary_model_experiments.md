# Supplementary Model Experiments for Article Integration

This note summarizes the supplementary experiments added after the main pooled-feature benchmark. The goal is to keep the experiments reproducible, separate from the core benchmark outputs, and easy to integrate into the manuscript if needed.

## 1. Reproducibility

All supplementary experiments were run without modifying the original benchmark outputs. Each experiment writes to its own output directory.

### 1.1. Feature-group ablation

Command:

```powershell
python "C:\Users\Alketa\Documents\albania-air-quality-jlab\src\run_enhanced_feature_group_ablation.py"
```

Main outputs:

- `outputs/enhanced_feature_group_ablation/summary.md`
- `outputs/enhanced_feature_group_ablation/tables/contribution_summary.csv`

Design:

- model: `HistGBM Global Features`
- same pooled feature design as the main enhanced benchmark
- same `365-day` rolling training window
- same `28-day` origin stride
- same `1-, 2-, and 3-day` horizons
- one feature group removed at a time:
  - weather covariates
  - cross-pollutant context
  - city indicators

### 1.2. City-specific advanced-model comparison

Command:

```powershell
python "C:\Users\Alketa\Documents\albania-air-quality-jlab\src\run_city_specific_feature_benchmark.py"
```

Main outputs:

- `outputs/city_specific_feature_benchmark/summary.md`
- `outputs/city_specific_feature_benchmark/tables/comparison_summary.csv`
- `outputs/city_specific_feature_benchmark/tables/comparison_by_city.csv`

Design:

- model: `HistGBM Global Features`
- same temporal, meteorological, and cross-pollutant features as the pooled enhanced benchmark
- trained separately for each city
- city indicators excluded because each model is single-city
- same `365-day` rolling training window
- same `28-day` origin stride
- same `1-, 2-, and 3-day` horizons

### 1.3. XGBoost benchmark comparison

Command:

```powershell
python "C:\Users\Alketa\Documents\albania-air-quality-jlab\src\run_xgboost_feature_benchmark.py"
```

Main outputs:

- `outputs/xgboost_feature_benchmark/summary.md`
- `outputs/xgboost_feature_benchmark/tables/model_summary.csv`
- `outputs/xgboost_feature_benchmark/tables/comparison_vs_histgbm_summary.csv`
- `outputs/xgboost_feature_benchmark/tables/comparison_vs_best_enhanced_summary.csv`

Design:

- model: `XGBoost Global Features`
- same pooled feature design as the enhanced benchmark
- same `365-day` rolling training window
- same `28-day` origin stride
- same `1-, 2-, and 3-day` horizons
- fixed hyperparameters:
  - `n_estimators = 300`
  - `learning_rate = 0.05`
  - `max_depth = 4`
  - `min_child_weight = 5.0`
  - `subsample = 0.9`
  - `colsample_bytree = 0.9`
  - `tree_method = hist`

### 1.4. Time-split XGBoost tuning

Command:

```powershell
python "C:\Users\Alketa\Documents\albania-air-quality-jlab\src\run_xgboost_time_split_tuning.py"
```

Main outputs:

- `outputs/xgboost_time_split_tuning/summary.md`
- `outputs/xgboost_time_split_tuning/tables/origin_split_schedule.csv`
- `outputs/xgboost_time_split_tuning/tables/selected_xgb_configs.csv`
- `outputs/xgboost_time_split_tuning/tables/selected_enhanced_incumbents.csv`
- `outputs/xgboost_time_split_tuning/tables/evaluation_comparison_summary.csv`

Design:

- candidate model family: `XGBoost Global Features`
- same pooled feature design as the enhanced benchmark
- same `365-day` rolling training window
- same `28-day` origin stride
- same `1-, 2-, and 3-day` horizons
- the `18` rolling origins were split chronologically into:
  - `12 development origins`: `30 December 2024` to `3 November 2025`
  - `6 evaluation origins`: `1 December 2025` to `20 April 2026`
- XGBoost hyperparameters were selected only on the development origins
- the enhanced incumbent model was also selected only on the development origins
- final comparison was performed only on the held-out evaluation origins

### 1.5. Exploratory feature-pruning experiment

Command:

```powershell
python "C:\Users\Alketa\Documents\albania-air-quality-jlab\src\run_histgbm_feature_pruning_experiment.py"
```

Main outputs:

- `outputs/histgbm_feature_pruning_experiment/summary.md`
- `outputs/histgbm_feature_pruning_experiment/tables/evaluation_summary.csv`
- `outputs/histgbm_feature_pruning_experiment/tables/selected_features.csv`

Design:

- model: `HistGBM Global Features`
- same development/evaluation origin split as the tuned XGBoost experiment
- permutation-based feature ranking on development origins
- reduced model evaluated only on held-out evaluation origins

## 2. Recommended Article Positioning

### 2.1. Keep in the main paper

These supplementary experiments can support the article without changing the main benchmark narrative:

- feature-group ablation
- city-specific advanced-model comparison
- XGBoost benchmark comparison

If space is limited, the XGBoost time-split tuning can be mentioned briefly in the main text and expanded in supplementary material.

### 2.2. Do not make the feature-pruning analysis a main standalone section

The exploratory feature-pruning analysis did not produce a uniformly favorable result:

- it improved `PM2.5 2-day` and `PM2.5 3-day`
- it was nearly neutral for `AQI 2-day`
- it worsened `AQI 3-day`, `AQI 1-day`, and `PM2.5 1-day`

Because the effect was mixed and did not strengthen the main message consistently, it is better treated as:

- a short exploratory note in `Discussion`, or
- a supplementary sensitivity analysis

It should not be presented as a central positive result.

## 3. Article-Ready Methodology Text

If these supplementary experiments are included formally, they fit best after the enhanced pooled benchmark and before champion selection.

```text
4.x. Supplementary comparative experiments
Three supplementary comparative experiments were conducted to better interpret the behavior of the enhanced pooled benchmark and to examine alternative modeling choices without altering the main benchmark design. First, a feature-group ablation analysis was performed for the HistGradientBoosting Global Features model by removing weather covariates, cross-pollutant context variables, and one-hot encoded city indicators one group at a time while keeping the rolling-origin evaluation design unchanged. Second, a city-specific advanced-model comparison was conducted to test whether the gains of the pooled enhanced benchmark could be reproduced by fitting the same HistGradientBoosting feature-based model separately for each city using only local observations. Third, an XGBoost Global Features model was evaluated using the same pooled feature design, training window, origin stride, and forecast horizons as the main enhanced benchmark.

To assess whether XGBoost performance could be improved without biasing the final comparison, a separate time-split tuning experiment was also conducted. For this experiment, the 18 rolling forecast origins were divided chronologically into 12 development origins and 6 later evaluation origins. XGBoost hyperparameters were selected only on the development origins, and the tuned configuration was then compared against the default XGBoost specification and the development-selected enhanced incumbent model exclusively on the held-out evaluation origins.
```

## 4. Article-Ready Results Text

### 4.1. Feature-group ablation

```text
The feature-group ablation analysis showed that meteorological covariates provided the most consistent contribution within the enhanced pooled benchmark. Removing weather variables increased cross-city mean MAE in all six target-horizon combinations. By contrast, cross-pollutant context was most helpful at shorter horizons and showed weaker or slightly inconsistent effects at longer horizons, while city indicators had a smaller and more mixed contribution.
```

### 4.2. City-specific advanced-model comparison

```text
A supplementary comparison was conducted to test whether the strongest enhanced model could achieve lower forecast error when trained separately for each city rather than across all cities jointly. The results did not support that expectation. For daily maximum AQI, the pooled cross-city HistGradientBoosting model outperformed the city-specific variant at all three horizons in the cross-city average summary, reducing mean MAE from 3.3289 to 2.8248 at 1 day, from 4.3398 to 3.8443 at 2 days, and from 4.9093 to 4.2889 at 3 days. For daily mean PM2.5, the pooled model also remained stronger overall, reducing mean MAE from 2.2297 to 1.9219 at 1 day, from 3.2258 to 3.1856 at 2 days, and from 2.9143 to 2.7939 at 3 days. Across all 48 city-target-horizon comparisons, the city-specific advanced model won 9 cases, compared with 39 for the pooled cross-city version.
```

### 4.3. XGBoost benchmark comparison

```text
A supplementary XGBoost benchmark was conducted using the same pooled feature design as the enhanced benchmark. XGBoost produced mixed results. Relative to HistGradientBoosting, it performed slightly better for AQI at the 2-day and 3-day horizons and clearly better for PM2.5 at the 1-day horizon, where mean MAE decreased from 1.9219 to 1.7767 and XGBoost won all 8 city-level comparisons. However, it did not outperform the strongest existing enhanced model across all settings, indicating that XGBoost should be interpreted as a competitive challenger rather than a uniformly superior replacement.
```

### 4.4. Time-split XGBoost tuning

```text
The time-split XGBoost tuning experiment showed that limited hyperparameter tuning could improve performance selectively, but not uniformly. On the held-out evaluation origins, the tuned XGBoost configuration improved AQI forecasting at the 2-day horizon, reducing mean MAE from 2.5316 under the default XGBoost specification to 2.4306 and outperforming the development-selected HistGradientBoosting incumbent (2.6756). For AQI at the 1-day horizon and PM2.5 at the 1-day and 3-day horizons, the selected configuration remained the default XGBoost specification. By contrast, tuning did not improve performance at all settings and worsened results for AQI at the 3-day horizon and PM2.5 at the 2-day horizon relative to default XGBoost. These results suggest that XGBoost can be competitive, but that modest hyperparameter tuning does not support replacing the existing enhanced benchmark uniformly.
```

## 5. Article-Ready Discussion Text

```text
The supplementary experiments provide additional context for interpreting the enhanced benchmark. The city-specific advanced-model comparison suggests that the gains of the pooled benchmark were not explained only by the use of a stronger model class and richer predictors, because the same feature-based HistGradientBoosting model generally performed worse when trained separately for each city. The XGBoost experiments likewise indicate that alternative tree-based global models can be competitive, especially for short-horizon PM2.5 forecasting and 2-day-ahead AQI forecasting, but they do not support a universal replacement of the existing enhanced benchmark. Finally, the exploratory feature-pruning analysis produced mixed effects across targets and horizons, suggesting that selective simplification may be promising for some PM2.5 settings but does not yet provide a stable basis for changing the core benchmark design.
```

## 6. Suggested Editorial Decision on Feature Importance / Pruning

Recommended choice:

- mention it briefly
- keep it out of the main results hierarchy
- if needed, place it in supplementary material

Recommended wording:

```text
An exploratory time-split feature-pruning analysis was also conducted for the HistGradientBoosting model. Although reduced feature sets improved PM2.5 forecasting at the 2-day and 3-day horizons, the results were mixed across targets and did not support a consistent improvement over the full feature design. For this reason, the analysis is treated as supplementary sensitivity evidence rather than as a main benchmark result.
```
