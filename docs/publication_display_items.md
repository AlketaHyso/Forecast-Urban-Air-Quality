# Publication Display Items

## Main Tables

### Table 1

**Title:** Study-Defined Operational Alert Mapping

**Caption:** Study-defined mapping from predicted AQI categories and city-relative PM2.5 escalation rules to the final operational alert levels used in the illustrative case study. The city-specific p90 and p95 thresholds refer to each city's historical daily PM2.5 distribution available up to the forecast issue date.

**Saved files:**

- `outputs/publication_assets/tables/table_1_operational_alert_mapping.csv`
- `outputs/publication_assets/tables/table_1_operational_alert_mapping.md`

### Table 2

**Title:** Baseline Benchmark Summary by Target and Forecast Horizon

**Caption:** Best-performing baseline forecasting model for each target and forecast horizon. The table reports the strongest local model from the baseline benchmark together with cross-city mean MAE and mean MAPE.

**Saved files:**

- `outputs/publication_assets/tables/table_2_baseline_best_models.csv`
- `outputs/publication_assets/tables/table_2_baseline_best_models.md`

### Table 3

**Title:** Enhanced-versus-Baseline Benchmark Comparison by Target and Forecast Horizon

**Caption:** Comparison between the best enhanced model and the best baseline model for each target and forecast horizon. Positive improvement values indicate lower forecast error for the enhanced benchmark.

**Saved files:**

- `outputs/publication_assets/tables/table_3_enhanced_vs_baseline.csv`
- `outputs/publication_assets/tables/table_3_enhanced_vs_baseline.md`

### Table 4

**Title:** Case-Study City Attention Ranking for the Forecast Issued on 28 April 2026

**Caption:** City attention ranking for the fixed case-study forecast issued on 28 April 2026 and covering 29 April to 1 May 2026. For each city, the table retains the single forecast row within the 1- to 3-day window that produced the highest final alert score, with ties broken by higher predicted AQI. It reports the retained date and horizon, the peak predicted AQI and PM2.5, and the selected AQI and PM2.5 champion models.

**Saved files:**

- `outputs/publication_assets/tables/table_4_city_attention_ranking.csv`
- `outputs/publication_assets/tables/table_4_city_attention_ranking.md`

## Main Figures

### Figure 1

**Title:** Mean MAE Comparison Between Baseline and Enhanced Benchmarks

**Caption:** Mean absolute error of the best baseline and best enhanced models for each target and forecast horizon. The enhanced benchmark improves AQI forecasting at all horizons and PM2.5 forecasting at the 1-day horizon, while the longer PM2.5 horizons remain mixed.

**Saved file stem:** `figure_1_best_model_mae_comparison`

### Figure 2

**Title:** Champion Model-Family Wins by Target and Forecast Horizon

**Caption:** Number of city-level champion selections won by the enhanced and baseline model families across targets and horizons. AQI selections lean toward the enhanced family overall, while PM2.5 results remain strongly horizon-dependent.

**Saved file stem:** `figure_2_champion_source_counts`

### Figure 3

**Title:** Case-Study City Attention Ranking for the Forecast Issued on 28 April 2026

**Caption:** City attention ranking for the fixed case-study forecast issued on 28 April 2026, based on the retained peak-risk row for each city within the 1- to 3-day window from 29 April to 1 May 2026. The horizontal bars show the predicted AQI of the retained row used in the alert layer.

**Saved file stem:** `figure_3_city_attention_ranking`

### Figure 4

**Title:** Alert Heatmap for the Forecast Issued on 28 April 2026

**Caption:** Heatmap of alert scores across cities and forecast dates for the fixed case-study forecast issued on 28 April 2026. Darker cells indicate stronger alert levels in the operational alert layer.

**Saved file stem:** `figure_4_alert_heatmap`

### Figure 5

**Title:** AQI Paths for the Highest-Priority Cities in the Forecast Issued on 28 April 2026

**Caption:** Observed and forecast daily AQI trajectories for the four highest-priority cities in the fixed case-study forecast issued on 28 April 2026. The figure illustrates how the selected champion forecasts extend recent city-level AQI dynamics.

**Saved file stem:** `figure_5_top_city_aqi_paths`

## Supplementary Figures

### Figure S1

**Title:** Black-and-White Methodology Workflow Schematic

**Caption:** Block schematic of the study workflow, showing open gridded inputs, hourly harmonization, daily aggregation, the local baseline and pooled enhanced benchmarks, rolling-origin evaluation, champion selection, and the illustrative operational case study.

**Saved file stem:** `figure_s1_methodology_workflow_bw`

### Figure S2

**Title:** Black-and-White Forecasting and Evaluation Design Schematic

**Caption:** Schematic view of the forecasting design, showing the two daily targets, the local and pooled benchmark families, the aligned rolling-origin backtesting setup, and the MAE-based champion-selection logic used for deployment decisions.

**Saved file stem:** `figure_s2_forecasting_evaluation_design_bw`

## Supplementary Data

### Supplementary Data S1

**Title:** Daily Aggregated Analytical Dataset

**Caption:** Daily aggregated analytical dataset used in the forecasting experiments.

**Saved file:**

- `data/processed/albania_air_quality_daily.csv`

## Supplementary Tables

### Table S1

**Title:** Champion Source Counts by Target and Forecast Horizon

**Caption:** Champion source counts by target and horizon, showing how often the enhanced model family outperformed the baseline family.

**Saved files:**

- `outputs/publication_assets/tables/table_s1_champion_source_counts.csv`
- `outputs/publication_assets/tables/table_s1_champion_source_counts.md`

### Table S2

**Title:** Detailed Champion Model Selection by City, Target, and Forecast Horizon

**Caption:** Detailed champion model selection by target, city, and forecast horizon, including the selected source family and MAE comparison.

**Saved files:**

- `outputs/publication_assets/tables/table_s2_champion_model_selection.csv`
- `outputs/publication_assets/tables/table_s2_champion_model_selection.md`

### Table S3

**Title:** Summary of the Aggregated Forecasting Dataset

**Caption:** Summary of Supplementary Data S1, including temporal coverage, record counts, and the main analytical structure of the daily layer.

**Saved files:**

- `outputs/publication_assets/tables/table_s3_aggregated_dataset_summary.csv`
- `outputs/publication_assets/tables/table_s3_aggregated_dataset_summary.md`

### Table S4

**Title:** Variable Dictionary for the Daily Analytical Dataset

**Caption:** Full list of variables included in Supplementary Data S1, grouped by role in the forecasting workflow.

**Saved files:**

- `outputs/publication_assets/tables/table_s4_daily_dataset_variables.csv`
- `outputs/publication_assets/tables/table_s4_daily_dataset_variables.md`

### Table S5

**Title:** City-Level AQI MAE Differences Between Enhanced and Baseline Champions

**Caption:** City-level AQI MAE comparison by forecast horizon, showing the best enhanced model, the corresponding best baseline comparator, and the MAE difference used to assess the geographic spread of AQI gains.

**Saved files:**

- `outputs/publication_assets/tables/table_s5_city_level_aqi_mae_deltas.csv`
- `outputs/publication_assets/tables/table_s5_city_level_aqi_mae_deltas.md`

## Supplementary File

- `outputs/publication_assets/supplementary_material.md`
