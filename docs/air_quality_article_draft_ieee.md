# Benchmark-Driven Early Warning for Urban Air Quality in Albanian Cities Using Open Environmental Data

**Authors:** [Author 1], [Author 2]

**Affiliations:** [Affiliation 1], [Affiliation 2]

## Abstract

Urban air pollution remains a major public-health and environmental-management challenge, and the need for operational early-warning tools has increased as European air-quality policy becomes more demanding. This study presents a benchmark-driven early-warning framework for Albanian cities using open environmental time series rather than proprietary monitoring infrastructure. The system integrates city-level gridded air-quality and weather data obtained through Open-Meteo interfaces built on air-quality and reanalysis products, aggregates the hourly series into a validated daily analytical layer, and evaluates short-horizon forecasting for daily mean PM2.5 and daily maximum European Air Quality Index (AQI). The study covers eight Albanian cities and uses 163,008 hourly records and 6,792 daily records for the period from 1 January 2024 to 28 April 2026. A first benchmark compares local univariate models, namely Naive, Drift, Holt Damped, ARIMA, and lag-based Random Forest. A second benchmark introduces feature-engineered global models with pooled learning across cities. The strengthened benchmark improves AQI forecasting at all horizons, reducing mean absolute error by about 0.80, 0.74, and 0.99 for one-, two-, and three-day horizons, respectively, relative to the best baseline model. For PM2.5, the results are mixed, with ARIMA remaining strongest at one and two days, while a feature-based HistGradientBoosting model improves the three-day horizon. The operational layer then applies champion selection by city, target, and horizon, converting the best available forecast into alert levels and city attention rankings. The resulting system demonstrates that open data, reproducible benchmarking, and model-family selection can support a practical urban air-quality early-warning workflow for Albanian cities.

**Index Terms:** air quality forecasting, AQI, PM2.5, early warning system, Albania, open environmental data, machine learning, reproducible benchmark.

## I. Introduction

Air pollution remains one of the most important environmental health risks in Europe and globally. The World Health Organization (WHO) updated its global air-quality guidelines in 2021 to reflect evidence that adverse effects occur even at lower pollutant concentrations than previously assumed [1]. The European Environment Agency (EEA) has continued to show that a large share of the European urban population remains exposed to pollutant levels above WHO guideline values, despite long-term improvements in emissions and compliance [2]. At the policy level, the revised European ambient air-quality directive has tightened standards, strengthened public-information provisions, and reinforced the role of air-quality alerts and planning instruments [3]. Although Albania is not an EU Member State, the same regional public-health logic applies, and the EEA's country assessment notes that air pollution remains an important issue for Albanian urban areas, especially with respect to PM and NO2 exposure [4].

Within this policy and public-health context, forecasting is no longer just an academic exercise. Reliable short-term forecasts help public authorities, city analysts, and health-sensitive users anticipate adverse conditions, communicate risk, and prepare operational responses. For Albania, the practical question is not only whether air quality can be predicted, but whether an operational, transparent, and reproducible early-warning workflow can be built from openly accessible environmental data rather than proprietary monitoring stacks.

The practical gap addressed in this article is twofold. First, Albania lacks widely documented open-data air-quality forecasting workflows designed specifically for city-level early warning. Second, even when recent literature demonstrates high predictive performance, the operational question remains unresolved: should a city-level system rely on a single global model family, on simpler local baselines, or on a benchmark-driven champion-selection strategy? This article addresses that question by developing a reproducible pipeline over open environmental data for eight Albanian cities and by evaluating both local and feature-engineered global forecasting models under a consistent rolling-origin protocol.

The contribution of the article is fourfold. First, it constructs a traceable open-data pipeline for Albanian city-level air-quality analysis using gridded air-quality and weather products [13], [14]. Second, it benchmarks short-horizon forecasting for daily mean PM2.5 and daily maximum European AQI using both classical and machine-learning methods. Third, it strengthens the modeling layer through pooled cross-city feature engineering and evaluates whether that additional structure yields real predictive gains. Fourth, it converts benchmark outcomes into an operational early-warning layer using alert logic and city attention ranking. In that sense, the study is not only a forecasting exercise but also a prototype for reproducible environmental decision support.

The remainder of the article is organized as follows. Section II reviews recent literature on air-quality forecasting and the methodological gap addressed here. Section III describes the data sources and system design. Section IV presents the forecasting methodology. Section V discusses the benchmark and operational results. Section VI concludes the study.

## II. Related Work

Recent review studies show that air-quality forecasting has rapidly expanded from statistical baselines toward machine learning, deep learning, and hybrid methods [5]-[10]. At the same time, that literature also points to recurring challenges: weak reproducibility, inconsistent evaluation protocols, underuse of interpretable approaches, and limited evidence about which models remain robust when data are short, noisy, or operationally constrained [5], [7], [9]. These observations are directly relevant to our setting because Albanian city-level series are operationally useful but not large enough to justify complexity for its own sake.

Recent application studies have pushed the field in multiple directions. Comparative machine-learning benchmarks have tested a wide variety of regression and ensemble strategies [11], while other work has integrated forecasting with environmental-health risk mapping [12]. Related studies have explored PM2.5 virtual monitoring stations [13], class-oriented AQI prediction using monitoring and secondary modeling [14], deep-learning and ensemble PM2.5 forecasting [15], long-horizon PM2.5 analysis [16], and multi-urban federated prediction settings [17]. Additional work has examined AQI forecasting with Transformer-based architectures [18], machine learning combined with optimization [19], low-cost sensing and ultrahigh-resolution PM2.5 modeling [20], particle-swarm-based neural systems [21], attention-enhanced BiLSTM structures [22], and regional hybrid bias-correction models [23].

Collectively, these studies confirm that model sophistication has increased sharply, but they do not eliminate the need for careful local benchmarking under clear data and evaluation constraints. Much of the recent literature focuses on large monitoring networks, deep architectures, or data-rich urban ecosystems. By contrast, the present study addresses a smaller open-data setting in which reproducibility, short-horizon operational reliability, and transparent model-family selection are more important than maximizing architectural novelty alone. The article therefore positions itself between applied air-quality forecasting and deployable urban decision support.

## III. Data and System Design

### A. Data Sources and Geographic Scope

The study uses open city-level environmental time series for Tirane, Durres, Elbasan, Shkoder, Vlore, Fier, Korce, and Berat. Air-quality variables were retrieved through the Open-Meteo Air Quality API, which provides pollutant and European AQI time series derived from gridded atmospheric products and exposes AQI category thresholds aligned with the European AQI scheme [13]. Weather covariates were obtained from the Open-Meteo Historical Weather API, which is based on reanalysis-driven historical weather products and provides hourly meteorological variables in local time [14]. The analytical period spans from 1 January 2024 to 28 April 2026.

This point deserves explicit clarification. The system is built on open gridded environmental data and not on official ground-station measurements reported by national authorities. The approach is therefore best understood as an open-data operational prototype rather than as a direct replacement for regulatory monitoring. That distinction matters, especially given the monitoring and assessment requirements emphasized in the revised EU framework [2], [3]. At the same time, open gridded data are valuable for rapid prototyping, reproducible computational research, and locations where official station-level operational feeds are not easily accessible.

### B. Data Processing and Daily Analytical Layer

The project stores both hourly and daily layers. The hourly pipeline produced 163,008 merged records across the eight cities. The daily aggregation layer yielded 6,792 city-day observations. The daily dataset includes pollutant summaries, meteorological summaries, AQI-based risk labels, pollutant exceedance indicators, and event flags. In the current implementation, daily features include mean and maximum PM2.5, PM10, NO2, O3, SO2, carbon monoxide, and European AQI, together with weather summaries such as temperature, humidity, precipitation, wind speed, and pressure.

The AQI risk labels follow the European AQI category boundaries exposed through the Open-Meteo interface, namely Good (0-20), Fair (21-40), Moderate (41-60), Poor (61-80), Very Poor (81-100), and Extremely Poor (>100) [13]. The project defines `pollution_episode` as an AQI level of at least Poor and `high_risk_day` as an AQI level of at least Very Poor. These labels are not meant to replicate a legal warning threshold, but to provide a transparent and reproducible operational severity scheme consistent with public-facing AQI communication [2], [3].

### C. Reproducibility and System Architecture

The workflow was designed as a reproducible engineering prototype rather than a one-off notebook exercise. The system separates ingestion, hourly harmonization, daily feature generation, benchmarking, and operational alert generation. This design choice follows broader reproducibility and benchmarking recommendations in computational research [24]-[27]. Specifically, the project keeps raw files, processed analytical tables, benchmark outputs, and publication assets in separate directories so that the full workflow can be rerun without manual spreadsheet editing. Such separation is also consistent with recent forecasting frameworks that emphasize repeatable experiments and benchmark packaging [31]-[34].

## IV. Forecasting Methodology

### A. Baseline Benchmark

The first benchmark evaluates five local models independently for each city and target:

1. Naive  
2. Drift  
3. Holt Damped  
4. ARIMA  
5. Random Forest with seven lagged values

This model set intentionally balances strong baselines, classical trend-oriented methods, and a lag-based tree model. The design follows standard forecasting guidance: good benchmarks should include simple persistence-oriented baselines, should preserve temporal order, and should avoid assuming that added complexity automatically improves real-world forecast accuracy [28]-[30]. Local rolling training windows of 365 days were used, and forecast origins advanced with a 14-day stride. Each benchmark predicted one-, two-, and three-day horizons for both targets.

### B. Enhanced Benchmark

The second benchmark strengthens the modeling layer through pooled learning across cities. Instead of fitting each city separately, the enhanced models use a global tabular formulation with city indicators and structured features. For each target and horizon, the enhanced benchmark uses:

1. Persistence Current  
2. Ridge Global Features  
3. HistGradientBoosting Global Features

The feature set includes autoregressive lags at 1, 2, 3, 7, 14, 21, and 28 days; rolling means, standard deviations, and maxima over 3, 7, 14, and 28 days; calendar variables; seasonal trigonometric encodings; weather covariates; and cross-target contextual signals. This formulation is motivated by two ideas from the forecasting literature. First, tabular learning can be strong when time-series structure is converted into disciplined features [26], [29]. Second, pooled learning across multiple related series can improve stability when each individual series is relatively short [17], [31].

### C. Evaluation Protocol and Metrics

Both benchmarks use rolling-origin backtesting with fixed horizons of one, two, and three days. Chronological order is preserved, and training windows are limited to data available before each issue date. This choice directly follows best-practice recommendations against random time-series splits and leakage-prone evaluation schemes [25], [28], [30]. Model accuracy is assessed using mean absolute error (MAE), root mean squared error (RMSE), mean absolute percentage error (MAPE), symmetric mean absolute percentage error (sMAPE), and a city-level mean rank derived from MAE.

### D. Champion Selection and Operational Alerts

The final operational layer does not assume that one model family is universally superior. Instead, for each city, target, and horizon, it compares the best baseline model with the best enhanced model and selects the lower-MAE option as the operational champion. This champion-selection rule is then used to generate forecast-driven alert tables, heatmaps, and city attention rankings. Such benchmark-informed model selection is closer to real operational decision support than a one-size-fits-all deployment rule [25], [31].

## V. Results and Discussion

### A. Baseline Benchmark Results

The baseline benchmark already produced meaningful forecasting performance. For daily maximum AQI, ARIMA was the strongest baseline model at all three horizons, with mean MAE values of 3.63, 4.58, and 5.28 for one-, two-, and three-day forecasts, respectively. For daily mean PM2.5, ARIMA was likewise the strongest baseline model, with mean MAE values of 1.87, 2.71, and 2.99 across the same horizons. These results confirm that classical time-series models remain competitive for short-horizon environmental prediction, especially when series are short and operationally regular [24], [25], [28]-[30]. Table I summarizes the best baseline model by target and horizon.

[Insert Table I here]

**Table I**  
*Best-performing baseline forecasting model for each target and forecast horizon. The table summarizes the strongest local model from the first benchmark and reports its mean MAE, mean MAPE, and mean rank across Albanian cities.*

The baseline results also support a broader point emphasized by the forecasting literature: more complex models do not win automatically [26], [28]-[30]. In the first benchmark, the lag-based Random Forest did not displace ARIMA consistently. That outcome aligns with other comparisons showing that structured statistical models remain strong in low-frequency or moderately sized operational settings [8], [9], [25].

### B. Enhanced Benchmark Results

The strengthened benchmark produced a clear gain for daily maximum AQI. As summarized in Table II and Fig. 1, HistGradientBoosting with engineered global features became the best enhanced model at all three horizons, reducing mean MAE from 3.63 to 2.82 at one day, from 4.58 to 3.84 at two days, and from 5.28 to 4.29 at three days. The corresponding MAE improvements were 0.80, 0.74, and 0.99. These gains suggest that AQI benefited substantially from pooled cross-city learning and feature-engineered context, including lag structure, rolling summaries, and weather inputs.

[Insert Table II here]

**Table II**  
*Comparison between the best enhanced model and the best baseline model for each target and forecast horizon. Positive improvement values indicate that the strengthened model reduced forecast error.*

[Insert Fig. 1 here]

**Fig. 1.** *Mean absolute error of the best baseline and best enhanced models for each target and forecast horizon. The strengthened feature-based benchmark improves AQI forecasting across all horizons and improves PM2.5 forecasting at the longer horizon.*

For PM2.5, the picture was more nuanced. The enhanced benchmark slightly underperformed the baseline ARIMA model at one day and clearly underperformed it at two days. However, at the three-day horizon, the enhanced HistGradientBoosting model improved mean MAE from 2.99 to 2.79. This pattern suggests that PM2.5 forecasting in this dataset is more target-specific than AQI forecasting. The target appears to preserve some local short-memory structure that ARIMA captures well for the shortest horizons, while the richer feature-based formulation becomes more useful as the forecast window expands.

From a methodological standpoint, this mixed result is valuable rather than disappointing. It shows that model strengthening should not be interpreted as automatic replacement of simpler baselines. Instead, it supports a model-family selection logic that is target-dependent and horizon-dependent. This interpretation is consistent with recent air-quality studies that find strong but uneven performance across methods, depending on pollutant, horizon, feature richness, and local dynamics [10]-[23].

### C. Champion Selection

Champion selection provided a concise operational summary of the modeling outcome. Across all 48 city-target-horizon combinations, the enhanced family won 36, whereas the baseline family won 12. The enhanced family won all 24 AQI selections and dominated PM2.5 at the three-day horizon. By contrast, the baseline family remained especially competitive for PM2.5 at the two-day horizon, where it won 7 of 8 city selections. Fig. 2, Table S1, and Table S2 show that the strengthened models clearly improved the AQI task, while PM2.5 required a more selective deployment rule.

[Insert Fig. 2 here]

**Fig. 2.** *Number of city-level champion selections won by the enhanced and baseline model families across targets and horizons. The enhanced family dominates AQI forecasting and wins most PM2.5 selections except at the two-day horizon.*

This result matters because it transforms the modeling lesson into a practical systems lesson. Instead of asking which single algorithm is best in general, the operational system asks which benchmarked family is best for a given city, target, and horizon. That is a stronger and more defensible engineering position than unconditional preference for either statistical or machine-learning methods [7]-[10], [24]-[31].

### D. Operational Early-Warning Layer

The operational alert layer converts champion forecasts into city attention signals. In the current forecast window, no city entered Severe or Critical conditions. Most cities fell into the Warning category, while Korce was classified as Watch. The highest peak predicted AQI in the current three-day window was 48.0 for Tirane, followed by 47.0 for Fier, Vlore, and Berat. Table III, Fig. 3, and Fig. 4 show that the current outputs are moderate rather than crisis-level.

[Insert Table III here]

**Table III**  
*City attention ranking produced by the operational early-warning layer. The table lists the peak forecast risk for each city, the associated horizon, and the selected AQI and PM2.5 models.*

[Insert Fig. 3 here]

**Fig. 3.** *Operational city attention ranking based on the strongest alert in the 1-3 day forecast window. The horizontal bars show the peak predicted AQI used in the alert layer.*

[Insert Fig. 4 here]

**Fig. 4.** *Alert heatmap across cities and forecast dates. Darker cells indicate stronger alert scores in the operational early-warning layer.*

This does not weaken the value of the system. On the contrary, an early-warning framework should be able to communicate ordinary operational states as clearly as extreme ones. The city ranking and heatmap demonstrate that the workflow can turn benchmarked forecasts into transparent city prioritization, including the source family used for each forecast. In the current operational layer, the AQI side is dominated by enhanced models, often HistGradientBoosting or enhanced persistence, whereas PM2.5 still uses a mixture of enhanced and baseline selections. Fig. 5 illustrates this behavior for the highest-ranked cities by comparing recent observed AQI values with the selected forecast paths.

[Insert Fig. 5 here]

**Fig. 5.** *Observed and forecast daily AQI paths for the four highest-priority cities in the operational ranking. The figure illustrates how the selected champion forecasts extend recent city-level AQI dynamics.*

### E. Relation to Recent Literature

The strengthened results are broadly consistent with the recent literature but also add a useful nuance. Reviews and systematic studies repeatedly note that air-quality forecasting performance depends on pollutant type, temporal resolution, spatial structure, and model interpretability [5]-[7]. Comparative forecasting studies have found that ensemble or feature-based methods can outperform simpler models, but not in every setting [8]-[12]. Research on PM2.5 and AQI has also shown benefits from hybrid, deep, and feature-rich models [15]-[23], especially when weather, context, or multi-site structure are exploited.

What our Albanian prototype adds is a stricter benchmark logic for a smaller open-data setting. Rather than claiming that the most advanced model always wins, the study demonstrates that AQI and PM2.5 behave differently even within the same urban workflow. The strengthened feature-based benchmark is clearly superior for AQI, but PM2.5 retains cases where simpler local time-series methods are preferable. That finding is methodologically conservative and operationally valuable.

### F. Limitations

Several limitations should be acknowledged. First, the data layer is based on open gridded products accessed through Open-Meteo rather than officially validated urban monitoring observations [13], [14]. Second, only eight Albanian cities were included, and the study period is relatively short for broader climatological generalization. Third, the enhanced benchmark did not include external traffic, emissions, land-use, or satellite variables, although recent work suggests such features can further strengthen air-quality prediction [10], [18], [20], [23]. Fourth, the alert categories are operational and transparent, but they are not a substitute for formal public-health warning standards.

## VI. Conclusion

This study developed a benchmark-driven early-warning framework for urban air quality in Albanian cities using open environmental data. The system integrates hourly air-quality and weather retrieval, daily analytical feature engineering, baseline local forecasting, enhanced global feature-based forecasting, champion model selection, and operational alert generation. The empirical results show that the strengthened benchmark substantially improves AQI forecasting at one-, two-, and three-day horizons, while PM2.5 remains more mixed and still favors ARIMA in some short-horizon settings. Operationally, the best solution is therefore not a single model but a benchmark-driven selection rule.

The main contribution of the work is not simply that it produces forecasts. Rather, it demonstrates a reproducible path from open environmental data to benchmarked model choice and then to an interpretable city-level alert layer. This makes the system relevant to applied informatics, environmental analytics, and urban decision support. The next logical step is to incorporate richer external predictors such as traffic, satellite-derived aerosol products, or land-use indicators, and to test whether the champion-selection logic remains stable under a denser feature space and a longer observation period.

## References

[1] World Health Organization, *WHO global air quality guidelines: particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide*. Geneva, Switzerland: WHO, 2021. [Online]. Available: https://iris.who.int/handle/10665/345329

[2] European Environment Agency, *Europe's air quality status 2024*. Copenhagen, Denmark: EEA, Jun. 2024. [Online]. Available: https://www.eea.europa.eu/en/analysis/publications/europes-air-quality-status-2024

[3] European Parliament and Council of the European Union, "Directive (EU) 2024/2881 of the European Parliament and of the Council of 23 October 2024 on ambient air quality and cleaner air for Europe (recast)," *Official Journal of the European Union*, OJ L 2024/2881, Nov. 20, 2024. [Online]. Available: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024L2881

[4] European Environment Agency, "Health impacts of air pollution | Albania," 2025. [Online]. Available: https://www.eea.europa.eu/en/europe-environment-2025/countries/albania/health-impacts-of-air-pollution

[5] M. Mendez, M. G. Merayo, and M. Nunez, "Machine learning algorithms to forecast air quality: a survey," *Artificial Intelligence Review*, vol. 56, pp. 10031-10066, 2023, doi: 10.1007/s10462-023-10424-4.

[6] I. E. Agbehadji and I. C. Obagbuwa, "Systematic review of machine learning and deep learning techniques for spatiotemporal air quality prediction," *Atmosphere*, vol. 15, no. 11, art. no. 1352, 2024, doi: 10.3390/atmos15111352.

[7] A. Houdou *et al*., "Interpretable machine learning approaches for forecasting and predicting air pollution: A systematic review," *Aerosol and Air Quality Research*, vol. 24, art. no. 230151, 2024, doi: 10.4209/aaqr.230151.

[8] V. Petric *et al*., "Ensemble machine learning, deep learning, and time series forecasting: Improving prediction accuracy for hourly concentrations of ambient air pollutants," *Aerosol and Air Quality Research*, vol. 24, art. no. 230317, 2024, doi: 10.4209/aaqr.230317.

[9] Y. Ozupak, F. Alpsalaz, and E. Aslan, "Air quality forecasting using machine learning: Comparative analysis and ensemble strategies for enhanced prediction," *Water, Air, & Soil Pollution*, vol. 236, art. no. 464, 2025, doi: 10.1007/s11270-025-08122-8.

[10] M. Rajesh, R. G. Babu, U. Moorthy, *et al*., "Machine learning-driven framework for realtime air quality assessment and predictive environmental health risk mapping," *Scientific Reports*, vol. 15, art. no. 28801, 2025, doi: 10.1038/s41598-025-14214-6.

[11] A. Makhdoomi, M. Sarkhosh, and S. Ziaei, "PM2.5 concentration prediction using machine learning algorithms: an approach to virtual monitoring stations," *Scientific Reports*, vol. 15, art. no. 8076, 2025, doi: 10.1038/s41598-025-92019-3.

[12] Q. Liu, B. Cui, and Z. Liu, "Air quality class prediction using machine learning methods based on monitoring data and secondary modeling," *Atmosphere*, vol. 15, no. 5, art. no. 553, 2024, doi: 10.3390/atmos15050553.

[13] Open-Meteo, "Air Quality API," 2026. [Online]. Available: https://open-meteo.com/en/docs/air-quality-api. Accessed: Apr. 29, 2026.

[14] Open-Meteo, "Historical Weather API," 2026. [Online]. Available: https://open-meteo.com/en/docs/historical-weather-api. Accessed: Apr. 29, 2026.

[15] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. Melbourne, Australia: OTexts, 2021.

[16] H. Hewamalage, K. Ackermann, and C. Bergmeir, "Forecast evaluation for data scientists: Common pitfalls and best practices," *Data Mining and Knowledge Discovery*, vol. 37, pp. 788-832, 2023, doi: 10.1007/s10618-022-00894-5.

[17] T. Januschowski, Y. Wang, K. Torkkola, T. Erkkila, H. Hasson, and J. Gasthaus, "Forecasting with trees," *International Journal of Forecasting*, vol. 38, no. 4, pp. 1473-1481, 2022, doi: 10.1016/j.ijforecast.2021.10.004.

[18] S. Kolassa, "Fathoming empirical forecasting competitions' winners," *International Journal of Forecasting*, vol. 38, no. 4, pp. 1519-1525, 2022, doi: 10.1016/j.ijforecast.2022.03.010.

[19] S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "The M5 competition: Conclusions," *International Journal of Forecasting*, vol. 38, no. 4, pp. 1576-1582, 2022, doi: 10.1016/j.ijforecast.2022.04.006.

[20] D. Tjostheim, "Selected topics in time series forecasting: Statistical models vs. machine learning," *Entropy*, vol. 27, no. 3, art. no. 279, 2025, doi: 10.3390/e27030279.

[21] E. Strom and O. E. Gundersen, "Performance metrics for multi-step forecasting measuring win-loss, seasonal variance and forecast stability: An empirical study," *Applied Intelligence*, vol. 54, pp. 10490-10515, 2024, doi: 10.1007/s10489-024-05715-4.

[22] J. Eiglsperger, F. Haselbeck, and D. G. Grimm, "ForeTiS: A comprehensive time series forecasting framework in Python," *Machine Learning with Applications*, vol. 12, art. no. 100467, 2023, doi: 10.1016/j.mlwa.2023.100467.

[23] *Nature Computational Science*, "Moving towards reproducible machine learning," *Nature Computational Science*, vol. 1, pp. 629-630, 2021, doi: 10.1038/s43588-021-00152-6.

[24] P. E. DeWitt, M. A. Rebull, and T. D. Bennett, "Open source and reproducible and inexpensive infrastructure for data challenges and education," *Scientific Data*, vol. 11, art. no. 8, 2024, doi: 10.1038/s41597-023-02854-0.

[25] L. Wratten, A. Wilm, and J. Goke, "Reproducible, scalable, and shareable analysis pipelines with bioinformatics workflow managers," *Nature Methods*, vol. 18, pp. 1161-1168, 2021, doi: 10.1038/s41592-021-01254-9.

[26] Z. Gao, X. Mo, and H. Li, "Prediction of PM2.5 concentration based on deep learning, multi-objective optimization, and ensemble forecast," *Sustainability*, vol. 16, no. 11, art. no. 4643, 2024, doi: 10.3390/su16114643.

[27] Y. Zhang, Q. Sun, J. Liu, and O. Petrosian, "Long-term forecasting of air pollution particulate matter (PM2.5) and analysis of influencing factors," *Sustainability*, vol. 16, no. 1, art. no. 19, 2024, doi: 10.3390/su16010019.

[28] Y. Hu, N. Cao, W. Guo, M. Chen, Y. Rong, and H. Lu, "FedDeep: A federated deep learning network for edge assisted multi-urban PM2.5 forecasting," *Applied Sciences*, vol. 14, no. 5, art. no. 1979, 2024, doi: 10.3390/app14051979.

[29] P. Cheng, C. Wei, J. Zhang, and H. Wang, "Air quality index forecasting based on quadratic decomposition and Transformer-BiLSTM: A case study of Beijing," *Atmosphere*, vol. 16, no. 12, art. no. 1334, 2025, doi: 10.3390/atmos16121334.

[30] E. Cengil, "The power of machine learning methods and PSO in air quality prediction," *Applied Sciences*, vol. 15, no. 5, art. no. 2546, 2025, doi: 10.3390/app15052546.

[31] J. Li, J. Chen, R. You, and Q. He, "PM2.5 concentration prediction: Ultrahigh spatiotemporal resolution achieved by combining machine learning and low-cost sensors," *Sensors*, vol. 25, no. 17, art. no. 5527, 2025, doi: 10.3390/s25175527.

[32] Z. Liu, Y. Hu, Z. Fang, S. Xiong, L. Wang, and C. Bao, "Improved prediction model for daily PM2.5 concentrations with particle swarm optimization and BP neural network," *Scientific Reports*, vol. 15, art. no. 32050, 2025, doi: 10.1038/s41598-025-18014-w.

[33] X. Meng, C. Xie, X. Tang, and Y. Pan, "Prediction of particulate matter 2.5 concentration based on attention mechanism and convolutional BiLSTM network," *Discover Applied Sciences*, vol. 7, art. no. 1372, 2025, doi: 10.1007/s42452-025-07891-5.

[34] N. Zaini, L. W. Ean, A. N. Ahmed, M. A. Malek, and M. F. Chow, "PM2.5 forecasting for an urban area based on deep learning and decomposition method," *Scientific Reports*, vol. 12, art. no. 17565, 2022, doi: 10.1038/s41598-022-21769-1.

[35] M. Hu, X. Lu, Y. Chen, Z. Li, Y. Wang, and J. C. H. Fung, "AirQFormer: Improving regional air quality forecast with a hybrid deep learning model," *Sustainable Cities and Society*, vol. 119, art. no. 106113, 2025, doi: 10.1016/j.scs.2024.106113.
