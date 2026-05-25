# Publication Results Summary

The strengthened modeling phase produced a clear methodological result for daily maximum European AQI. Under the aligned rolling-origin schedule, the enhanced feature-based benchmark outperformed the best local baseline model at all three forecast horizons. The strongest enhanced model was `HistGBM Global Features`, which reduced mean MAE by approximately `0.29` at one day, `0.30` at two days, and `0.20` at three days relative to the best baseline model.

For daily mean PM2.5, the picture was more mixed. The enhanced benchmark performed best at the one-day horizon, the baseline `ARIMA` model remained stronger at the two-day horizon, and the three-day comparison was nearly neutral and slightly favored `ARIMA`. This means the strengthened system should not replace the baseline family wholesale. Instead, it should operate through champion selection, choosing the lower-error model family for each city, target, and horizon.

The champion-selection layer confirmed this more nuanced pattern. Across all `48` city-target-horizon combinations, the enhanced family won `29`, while the baseline family won `19`. For AQI, the enhanced family won `14/24` selections overall, while for PM2.5 the balance shifted by horizon: enhanced led at one day and three days, whereas the baseline family remained stronger at two days.

Operationally, the fixed case-study forecast issued on `2026-04-28` for the `2026-04-29` to `2026-05-01` window does not indicate an extreme episode. Most cities fall in the `Warning` category, while `Korce` falls in `Watch`. The city attention ranking is therefore useful not because it signals a crisis, but because it demonstrates that the pipeline can translate benchmarked forecasts into transparent operational prioritization.

These findings support a strong narrative for the article: benchmark-driven model selection improves the reliability of a gridded-data urban forecasting workflow, but the best solution is target-dependent and horizon-dependent rather than universally tied to a single forecasting method.
