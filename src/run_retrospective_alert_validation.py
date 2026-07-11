from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from generate_operational_alerts import alert_score, aqi_label, final_alert_level, pm25_signal


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BASELINE_DIR = PROJECT_ROOT / "outputs" / "forecast_benchmark" / "tables"
ENHANCED_DIR = PROJECT_ROOT / "outputs" / "enhanced_forecast_benchmark" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "retrospective_alert_validation"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

ALERT_LEVELS = ["Stable", "Watch", "Warning", "High", "Severe", "Critical"]
ALERT_LEVEL_TO_SCORE = {level: alert_score(level) for level in ALERT_LEVELS}
EVENT_THRESHOLDS = [
    ("High-or-worse", "Poor-or-worse operational alert threshold", ALERT_LEVEL_TO_SCORE["High"]),
    ("Severe-or-worse", "Very Poor-or-worse operational alert threshold", ALERT_LEVEL_TO_SCORE["Severe"]),
]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def derive_champion_map() -> pd.DataFrame:
    baseline_best = pd.read_csv(BASELINE_DIR / "best_models_by_city.csv")
    enhanced_best = pd.read_csv(ENHANCED_DIR / "best_models_by_city.csv")

    baseline_cmp = baseline_best.rename(
        columns={
            "model": "baseline_model",
            "mae": "baseline_mae",
            "rmse": "baseline_rmse",
            "mape": "baseline_mape",
            "rank_mae": "baseline_rank",
        }
    )
    enhanced_cmp = enhanced_best.rename(
        columns={
            "model": "enhanced_model",
            "mae": "enhanced_mae",
            "rmse": "enhanced_rmse",
            "mape": "enhanced_mape",
            "rank_mae": "enhanced_rank",
        }
    )

    champion = baseline_cmp.merge(
        enhanced_cmp,
        on=["target", "city", "horizon_days"],
        how="inner",
    )
    champion["selected_source"] = np.where(
        champion["enhanced_mae"] < champion["baseline_mae"],
        "enhanced",
        "baseline",
    )
    champion["selected_model"] = np.where(
        champion["selected_source"] == "enhanced",
        champion["enhanced_model"],
        champion["baseline_model"],
    )
    champion["selected_mae"] = np.where(
        champion["selected_source"] == "enhanced",
        champion["enhanced_mae"],
        champion["baseline_mae"],
    )

    return champion[
        [
            "target",
            "city",
            "horizon_days",
            "baseline_model",
            "baseline_mae",
            "enhanced_model",
            "enhanced_mae",
            "selected_source",
            "selected_model",
            "selected_mae",
        ]
    ].copy()


def load_selected_rolling_predictions(champion: pd.DataFrame) -> pd.DataFrame:
    baseline_predictions = pd.read_csv(
        BASELINE_DIR / "rolling_predictions.csv",
        parse_dates=["training_end_date", "target_date"],
    ).rename(columns={"training_end_date": "origin_date"})
    enhanced_predictions = pd.read_csv(
        ENHANCED_DIR / "rolling_predictions.csv",
        parse_dates=["feature_date", "target_date"],
    ).rename(columns={"feature_date": "origin_date"})

    selected_frames: list[pd.DataFrame] = []
    for source_name, predictions in [
        ("baseline", baseline_predictions),
        ("enhanced", enhanced_predictions),
    ]:
        selected_map = champion.loc[champion["selected_source"] == source_name].copy()
        merged = selected_map.merge(
            predictions,
            left_on=["target", "city", "horizon_days", "selected_model"],
            right_on=["target", "city", "horizon_days", "model"],
            how="inner",
        )
        merged = merged[
            [
                "target",
                "city",
                "horizon_days",
                "origin_date",
                "target_date",
                "selected_source",
                "selected_model",
                "selected_mae",
                "actual",
                "predicted",
            ]
        ].copy()
        selected_frames.append(merged)

    selected = pd.concat(selected_frames, ignore_index=True)
    selected["origin_date"] = selected["target_date"] - pd.to_timedelta(selected["horizon_days"], unit="D")
    selected = selected.sort_values(["city", "target", "horizon_days", "origin_date"]).reset_index(drop=True)
    return selected


def build_case_level_frame(selected: pd.DataFrame) -> pd.DataFrame:
    case_index = ["city", "origin_date", "target_date", "horizon_days"]

    predicted_wide = (
        selected.pivot_table(index=case_index, columns="target", values="predicted", aggfunc="first")
        .reset_index()
    )
    actual_wide = (
        selected.pivot_table(index=case_index, columns="target", values="actual", aggfunc="first")
        .reset_index()
    )
    model_wide = (
        selected.pivot_table(index=case_index, columns="target", values="selected_model", aggfunc="first")
        .reset_index()
    )
    source_wide = (
        selected.pivot_table(index=case_index, columns="target", values="selected_source", aggfunc="first")
        .reset_index()
    )

    cases = predicted_wide.merge(actual_wide, on=case_index, suffixes=("_predicted", "_actual"))
    cases = cases.merge(model_wide, on=case_index, suffixes=("", "_model"))
    cases = cases.merge(source_wide, on=case_index, suffixes=("", "_source"))

    cases = cases.rename(
        columns={
            "pm2_5_mean_predicted": "predicted_pm2_5_mean",
            "european_aqi_max_predicted": "predicted_european_aqi_max",
            "pm2_5_mean_actual": "observed_pm2_5_mean",
            "european_aqi_max_actual": "observed_european_aqi_max",
            "pm2_5_mean": "selected_pm2_5_model",
            "european_aqi_max": "selected_aqi_model",
            "pm2_5_mean_source": "selected_pm2_5_source",
            "european_aqi_max_source": "selected_aqi_source",
        }
    )

    return cases.sort_values(["origin_date", "city", "horizon_days"]).reset_index(drop=True)


def add_alert_labels(cases: pd.DataFrame) -> pd.DataFrame:
    daily = pd.read_csv(PROCESSED_DIR / "albania_air_quality_daily.csv", parse_dates=["date"])
    pm25_reference = (
        daily.groupby("city")["pm2_5_mean"]
        .agg(
            pm25_p90=lambda series: float(np.nanpercentile(series.dropna(), 90)),
            pm25_p95=lambda series: float(np.nanpercentile(series.dropna(), 95)),
        )
        .reset_index()
    )

    enriched = cases.merge(pm25_reference, on="city", how="left")
    enriched["predicted_aqi_label"] = enriched["predicted_european_aqi_max"].apply(aqi_label)
    enriched["observed_aqi_label"] = enriched["observed_european_aqi_max"].apply(aqi_label)
    enriched["predicted_pm25_signal"] = enriched.apply(
        lambda row: pm25_signal(row["predicted_pm2_5_mean"], row["pm25_p90"], row["pm25_p95"]),
        axis=1,
    )
    enriched["observed_pm25_signal"] = enriched.apply(
        lambda row: pm25_signal(row["observed_pm2_5_mean"], row["pm25_p90"], row["pm25_p95"]),
        axis=1,
    )
    enriched["predicted_final_alert_level"] = enriched.apply(
        lambda row: final_alert_level(
            row["predicted_european_aqi_max"],
            row["predicted_pm2_5_mean"],
            row["pm25_p90"],
            row["pm25_p95"],
        ),
        axis=1,
    )
    enriched["observed_final_alert_level"] = enriched.apply(
        lambda row: final_alert_level(
            row["observed_european_aqi_max"],
            row["observed_pm2_5_mean"],
            row["pm25_p90"],
            row["pm25_p95"],
        ),
        axis=1,
    )
    enriched["predicted_alert_score"] = enriched["predicted_final_alert_level"].map(ALERT_LEVEL_TO_SCORE)
    enriched["observed_alert_score"] = enriched["observed_final_alert_level"].map(ALERT_LEVEL_TO_SCORE)
    enriched["exact_alert_match"] = (
        enriched["predicted_final_alert_level"] == enriched["observed_final_alert_level"]
    ).astype(int)
    enriched["within_one_alert_level"] = (
        (enriched["predicted_alert_score"] - enriched["observed_alert_score"]).abs() <= 1
    ).astype(int)
    enriched["alert_score_gap"] = enriched["predicted_alert_score"] - enriched["observed_alert_score"]
    return enriched


def build_confusion_tables(cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(
        cases["observed_final_alert_level"],
        cases["predicted_final_alert_level"],
        dropna=False,
    ).reindex(index=ALERT_LEVELS, columns=ALERT_LEVELS, fill_value=0)

    row_normalized = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    return counts, row_normalized


def summarise_overall(cases: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            {
                "n_cases": int(len(cases)),
                "n_cities": int(cases["city"].nunique()),
                "n_origins": int(cases["origin_date"].nunique()),
                "forecast_horizons": int(cases["horizon_days"].nunique()),
                "exact_match_rate": float(cases["exact_alert_match"].mean()),
                "within_one_level_rate": float(cases["within_one_alert_level"].mean()),
                "mean_absolute_alert_gap": float((cases["alert_score_gap"].abs()).mean()),
            }
        ]
    )
    return summary


def summarise_by_horizon(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for horizon, frame in cases.groupby("horizon_days"):
        rows.append(
            {
                "horizon_days": int(horizon),
                "n_cases": int(len(frame)),
                "n_origins": int(frame["origin_date"].nunique()),
                "exact_match_rate": float(frame["exact_alert_match"].mean()),
                "within_one_level_rate": float(frame["within_one_alert_level"].mean()),
                "mean_absolute_alert_gap": float((frame["alert_score_gap"].abs()).mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("horizon_days").reset_index(drop=True)


def binary_event_metrics(frame: pd.DataFrame, threshold_score: int) -> dict[str, float]:
    observed = frame["observed_alert_score"] >= threshold_score
    predicted = frame["predicted_alert_score"] >= threshold_score

    tp = int((predicted & observed).sum())
    fp = int((predicted & ~observed).sum())
    tn = int((~predicted & ~observed).sum())
    fn = int((~predicted & observed).sum())

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, len(frame))
    f1 = safe_divide(2 * precision * recall, precision + recall) if not np.isnan(precision) and not np.isnan(recall) else float("nan")

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "prevalence": safe_divide(tp + fn, len(frame)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
    }


def summarise_binary_events(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for horizon_key, frame in [("All", cases), *cases.groupby("horizon_days")]:
        horizon_label = "All" if horizon_key == "All" else str(int(horizon_key))
        for threshold_name, threshold_description, threshold_score in EVENT_THRESHOLDS:
            metrics = binary_event_metrics(frame, threshold_score)
            rows.append(
                {
                    "horizon_days": horizon_label,
                    "threshold": threshold_name,
                    "threshold_description": threshold_description,
                    "n_cases": int(len(frame)),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def summarise_alert_distribution(cases: pd.DataFrame) -> pd.DataFrame:
    observed_counts = (
        cases["observed_final_alert_level"]
        .value_counts()
        .reindex(ALERT_LEVELS, fill_value=0)
        .rename("observed_cases")
    )
    predicted_counts = (
        cases["predicted_final_alert_level"]
        .value_counts()
        .reindex(ALERT_LEVELS, fill_value=0)
        .rename("predicted_cases")
    )
    distribution = pd.concat([observed_counts, predicted_counts], axis=1).reset_index()
    return distribution.rename(columns={"index": "alert_level"})


def build_manuscript_summary_table(
    overall_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    binary_summary: pd.DataFrame,
) -> pd.DataFrame:
    base_rows = [
        {
            "evaluation_scope": "Overall",
            "n_cases": int(overall_summary.loc[0, "n_cases"]),
            "exact_match_rate": float(overall_summary.loc[0, "exact_match_rate"]),
            "within_one_level_rate": float(overall_summary.loc[0, "within_one_level_rate"]),
            "mean_absolute_alert_gap": float(overall_summary.loc[0, "mean_absolute_alert_gap"]),
        }
    ]

    for _, row in horizon_summary.iterrows():
        base_rows.append(
            {
                "evaluation_scope": f"{int(row['horizon_days'])}-day",
                "n_cases": int(row["n_cases"]),
                "exact_match_rate": float(row["exact_match_rate"]),
                "within_one_level_rate": float(row["within_one_level_rate"]),
                "mean_absolute_alert_gap": float(row["mean_absolute_alert_gap"]),
            }
        )

    manuscript = pd.DataFrame(base_rows)
    binary_pivot = (
        binary_summary[
            ["horizon_days", "threshold", "precision", "recall", "f1"]
        ]
        .copy()
        .assign(
            evaluation_scope=lambda frame: frame["horizon_days"].map(
                {
                    "All": "Overall",
                    "1": "1-day",
                    "2": "2-day",
                    "3": "3-day",
                }
            )
        )
        .drop(columns=["horizon_days"])
        .pivot(index="evaluation_scope", columns="threshold", values=["precision", "recall", "f1"])
    )
    binary_pivot.columns = [
        f"{metric}_{threshold.lower().replace('-', '_').replace(' ', '_')}"
        for metric, threshold in binary_pivot.columns
    ]
    binary_pivot = binary_pivot.reset_index()

    manuscript = manuscript.merge(binary_pivot, on="evaluation_scope", how="left")
    return manuscript


def render_confusion_figure(counts: pd.DataFrame, row_normalized: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    image = ax.imshow(row_normalized.fillna(0.0).to_numpy(), cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(ALERT_LEVELS)))
    ax.set_xticklabels(ALERT_LEVELS, rotation=35, ha="right")
    ax.set_yticks(range(len(ALERT_LEVELS)))
    ax.set_yticklabels(ALERT_LEVELS)
    ax.set_xlabel("Predicted final alert level")
    ax.set_ylabel("Observed final alert level")
    ax.set_title("Retrospective alert-level confusion matrix")

    for row_idx, observed_level in enumerate(ALERT_LEVELS):
        row_total = int(counts.loc[observed_level].sum())
        for col_idx, predicted_level in enumerate(ALERT_LEVELS):
            count_value = int(counts.loc[observed_level, predicted_level])
            share_value = row_normalized.loc[observed_level, predicted_level]
            if row_total == 0 or np.isnan(share_value):
                label = f"{count_value}"
            else:
                label = f"{count_value}\n{share_value:.0%}"
            text_color = "white" if not np.isnan(share_value) and share_value >= 0.55 else "black"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=9, color=text_color)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized share")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "alert_confusion_matrix.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "alert_confusion_matrix.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_metadata(cases: pd.DataFrame) -> None:
    metadata = {
        "selection_basis": (
            "Fixed benchmark-defined champion map by city, target, and horizon, derived from the "
            "main rolling-origin benchmark summaries and then reapplied to historical out-of-sample forecasts."
        ),
        "evaluation_scope": {
            "n_cases": int(len(cases)),
            "n_cities": int(cases["city"].nunique()),
            "n_origins": int(cases["origin_date"].nunique()),
            "horizons": sorted(cases["horizon_days"].unique().tolist()),
        },
        "alert_levels": ALERT_LEVELS,
        "binary_thresholds": [
            {
                "threshold": threshold_name,
                "description": threshold_description,
                "minimum_alert_score": threshold_score,
            }
            for threshold_name, threshold_description, threshold_score in EVENT_THRESHOLDS
        ],
        "pm25_threshold_basis": "City-specific p90 and p95 from the processed daily study dataset.",
        "outputs": {
            "case_level_predictions": str(TABLE_DIR / "alert_case_predictions.csv"),
            "overall_summary": str(TABLE_DIR / "overall_alert_performance.csv"),
            "horizon_summary": str(TABLE_DIR / "alert_performance_by_horizon.csv"),
            "binary_metrics": str(TABLE_DIR / "binary_event_metrics.csv"),
            "alert_distribution": str(TABLE_DIR / "alert_level_distribution.csv"),
            "confusion_counts": str(TABLE_DIR / "alert_confusion_counts.csv"),
            "confusion_row_normalized": str(TABLE_DIR / "alert_confusion_row_normalized.csv"),
            "confusion_figure_png": str(FIGURE_DIR / "alert_confusion_matrix.png"),
            "confusion_figure_tiff": str(FIGURE_DIR / "alert_confusion_matrix.tiff"),
        },
    }
    with (OUTPUT_DIR / "retrospective_alert_validation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    champion = derive_champion_map()
    selected = load_selected_rolling_predictions(champion)
    cases = add_alert_labels(build_case_level_frame(selected))

    overall_summary = summarise_overall(cases)
    horizon_summary = summarise_by_horizon(cases)
    binary_summary = summarise_binary_events(cases)
    alert_distribution = summarise_alert_distribution(cases)
    manuscript_summary = build_manuscript_summary_table(overall_summary, horizon_summary, binary_summary)
    confusion_counts, confusion_row_normalized = build_confusion_tables(cases)

    champion.to_csv(TABLE_DIR / "champion_map_used.csv", index=False)
    cases.to_csv(TABLE_DIR / "alert_case_predictions.csv", index=False)
    overall_summary.to_csv(TABLE_DIR / "overall_alert_performance.csv", index=False)
    horizon_summary.to_csv(TABLE_DIR / "alert_performance_by_horizon.csv", index=False)
    binary_summary.to_csv(TABLE_DIR / "binary_event_metrics.csv", index=False)
    alert_distribution.to_csv(TABLE_DIR / "alert_level_distribution.csv", index=False)
    manuscript_summary.to_csv(TABLE_DIR / "manuscript_alert_validation_summary.csv", index=False)
    confusion_counts.to_csv(TABLE_DIR / "alert_confusion_counts.csv")
    confusion_row_normalized.to_csv(TABLE_DIR / "alert_confusion_row_normalized.csv")

    render_confusion_figure(confusion_counts, confusion_row_normalized)
    write_metadata(cases)

    print("Saved retrospective alert validation outputs to:")
    print(f"  - {TABLE_DIR / 'champion_map_used.csv'}")
    print(f"  - {TABLE_DIR / 'alert_case_predictions.csv'}")
    print(f"  - {TABLE_DIR / 'overall_alert_performance.csv'}")
    print(f"  - {TABLE_DIR / 'alert_performance_by_horizon.csv'}")
    print(f"  - {TABLE_DIR / 'binary_event_metrics.csv'}")
    print(f"  - {TABLE_DIR / 'alert_level_distribution.csv'}")
    print(f"  - {TABLE_DIR / 'manuscript_alert_validation_summary.csv'}")
    print(f"  - {TABLE_DIR / 'alert_confusion_counts.csv'}")
    print(f"  - {TABLE_DIR / 'alert_confusion_row_normalized.csv'}")
    print(f"  - {FIGURE_DIR / 'alert_confusion_matrix.png'}")
    print(f"  - {FIGURE_DIR / 'alert_confusion_matrix.tiff'}")
    print(f"  - {OUTPUT_DIR / 'retrospective_alert_validation_metadata.json'}")


if __name__ == "__main__":
    main()
