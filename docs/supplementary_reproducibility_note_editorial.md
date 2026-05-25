## Supplementary Note S1. Reproducibility Details

This supplementary note summarizes the main elements needed to reproduce the data preparation and forecasting workflow used in the study. The reported results were generated through a script-based pipeline covering data acquisition, daily preprocessing, benchmark execution, operational alert generation, and publication-output export.

### S1. Data sources and acquisition

Hourly air-quality data were obtained from the Open-Meteo Air Quality API and hourly meteorological data were obtained from the Open-Meteo Historical Weather API. Data were collected separately for eight Albanian cities: Tirane, Durres, Elbasan, Shkoder, Vlore, Fier, Korce, and Berat. All requests used city coordinates in the WGS84 system, the `Europe/Tirane` timezone, and the study period from 1 January 2024 to 28 April 2026. Air-quality requests were issued to the Open-Meteo air-quality endpoint with the `cams_europe` domain, while meteorological requests were issued to the historical weather archive endpoint.

The air-quality queries included hourly PM2.5, PM10, NO2, O3, SO2, carbon monoxide, the overall European AQI, and pollutant-specific AQI sub-indices. Meteorological queries included hourly temperature, relative humidity, precipitation, wind speed, wind direction, and mean sea-level pressure. The query structure is fully specified in the project’s data-collection script, and the downloaded dataset metadata are preserved in machine-readable form together with the processed data archive.

### S2. Daily preprocessing and labeling

Hourly air-quality and weather records were merged by city and timestamp and then aggregated to daily resolution. The daily dataset retained pollutant means and maxima, AQI means and maxima, pollutant-specific AQI sub-index summaries, and meteorological summaries including temperature, humidity, precipitation, wind speed, and mean sea-level pressure. The number of hourly observations contributing to each city-day was also recorded for completeness checking.

Daily AQI categories were assigned from the daily maximum AQI using the thresholds Good (0-20), Fair (21-40), Moderate (41-60), Poor (61-80), Very Poor (81-100), and Extremely Poor (>100). The daily analytical pipeline also generated binary episode labels for Poor-or-worse days and high-risk labels for Very Poor-or-worse days, together with dominant-pollutant indicators derived from pollutant-specific AQI sub-indices.

### S3. Benchmark configuration

Two benchmark families were evaluated. The local baseline family consisted of Naive, Seasonal Naive with a 7-day period, Drift, Holt Damped, ARIMA with restricted candidate orders, and Random Forest with seven lagged target values. The enhanced benchmark family consisted of Persistence Current, Ridge Global Features, and HistGradientBoosting Global Features.

Both benchmark families used a 365-day rolling training window, 1-, 2-, and 3-day forecast horizons, 28-day rolling forecast-origin steps, and strictly chronological train-test separation. In the pooled benchmark, feature construction included the current target value, lagged target values, rolling target summaries, calendar variables, sinusoidal seasonal encodings, issue-day meteorological covariates, issue-day cross-pollutant context variables, and one-hot encoded city indicators.

The pooled benchmark followed an end-of-day forecasting design. Let `t` denote the forecast issue date and `h` the forecast horizon. All predictors were constructed using information available by the end of day `t`, whereas the prediction target was evaluated on day `t + h`. Issue-day meteorological and cross-pollutant variables were measured on day `t`, lagged target features used only prior observations, and rolling target summaries were computed over windows ending at `t - 1`.

### S4. Operational alert layer

The operational layer combined the lower-error champion models from the local and enhanced benchmark families and translated their AQI and PM2.5 forecasts into alert levels. Predicted AQI served as the primary alert signal, while predicted PM2.5 acted as an escalation factor relative to city-specific historical 90th and 95th percentiles. This alerting layer should be interpreted as an illustrative operational prototype rather than as a retrospectively validated warning service.

### S5. Software environment

The reported experiments were executed in Python 3.12.2 using the following package versions: `pandas 2.2.1`, `numpy 1.26.4`, `statsmodels 0.14.6`, `scikit-learn 1.4.1.post1`, `plotly 6.6.0`, `ipywidgets 8.1.7`, and `jupyterlab 4.1.2`. The supplementary XGBoost experiments were run with `xgboost 3.0.3`. The project dependency manifest is archived with the codebase, and the benchmark scripts also write machine-readable metadata files that record the main benchmark settings and feature-group definitions.

### S6. Reproducibility scope

The workflow is reproducible in the practical sense that the project archive includes the data-query logic, preprocessing rules, benchmark configuration, and machine-readable metadata needed to regenerate the reported outputs from the same sources and study period. The main remaining requirements for archival release are to provide the final public repository link or DOI and to ensure that web-source access dates in the manuscript reference list match the archived version of the workflow.
