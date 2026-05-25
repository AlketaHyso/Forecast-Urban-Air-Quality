from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BASELINE_DIR = PROJECT_ROOT / "outputs" / "forecast_benchmark" / "tables"
ENHANCED_DIR = PROJECT_ROOT / "outputs" / "enhanced_forecast_benchmark" / "tables"
ALERT_DIR = PROJECT_ROOT / "outputs" / "operational_alerts"
PUB_DIR = PROJECT_ROOT / "outputs" / "publication_assets"
TABLE_DIR = PUB_DIR / "tables"
SINGLE_DIR = PUB_DIR / "figures" / "single_column"
DOUBLE_DIR = PUB_DIR / "figures" / "double_column"

SINGLE_WIDTH_IN = 85.1 / 25.4
DOUBLE_WIDTH_IN = 177.8 / 25.4
DPI = 300

FIGURE_TITLE_SIZE = 9
PANEL_TITLE_SIZE = 8
AXIS_LABEL_SIZE = 8
TICK_LABEL_SIZE = 7
LEGEND_SIZE = 7
LEGEND_TITLE_SIZE = 7
ANNOTATION_SIZE = 6.5
SMALL_ANNOTATION_SIZE = 6
LINE_WIDTH = 0.9
BAR_EDGE_WIDTH = 0.8
GRID_WIDTH = 0.45
SPINE_WIDTH = 0.8
MARKER_SIZE = 4.2
PANEL_LABEL_SIZE = 7.2

MODEL_SHORT_NAMES = {
    "Naive": "Naive",
    "Seasonal Naive": "SNaive-7",
    "Drift": "Drift",
    "Holt Damped": "Holt",
    "ARIMA": "ARIMA",
    "Random Forest Lag7": "RF-Lag7",
    "Persistence Current": "Persist",
    "Ridge Global Features": "Ridge",
    "HistGBM Global Features": "HistGBM",
}
BASELINE_MODEL_ORDER = [
    "Naive",
    "Seasonal Naive",
    "Drift",
    "Holt Damped",
    "ARIMA",
    "Random Forest Lag7",
]
BASELINE_MODEL_PRIORITY = {name: idx for idx, name in enumerate(BASELINE_MODEL_ORDER)}

BASELINE_COLOR = "#A9B8CC"
ENHANCED_COLOR = "#1F77B4"
OBSERVED_COLOR = "#2F2F2F"
FORECAST_COLOR = "#D95F02"
ALERT_HEATMAP_COLORS = [
    "#E8F3EC",  # Stable
    "#FDE7A9",  # Watch
    "#F7B267",  # Warning
    "#E76F51",  # High
    "#C44536",  # Severe
    "#6A1B1A",  # Critical
]
ALERT_STYLES = {
    "Stable": {"facecolor": ALERT_HEATMAP_COLORS[0], "label": "Stable"},
    "Watch": {"facecolor": ALERT_HEATMAP_COLORS[1], "label": "Watch"},
    "Warning": {"facecolor": ALERT_HEATMAP_COLORS[2], "label": "Warning"},
    "High": {"facecolor": ALERT_HEATMAP_COLORS[3], "label": "High"},
    "Severe": {"facecolor": ALERT_HEATMAP_COLORS[4], "label": "Severe"},
    "Critical": {"facecolor": ALERT_HEATMAP_COLORS[5], "label": "Critical"},
}

PUBLICATION_RCPARAMS = {
    "font.size": 8,
    "axes.titlesize": PANEL_TITLE_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "legend.title_fontsize": LEGEND_TITLE_SIZE,
    "figure.titlesize": FIGURE_TITLE_SIZE,
    "axes.linewidth": SPINE_WIDTH,
    "grid.linewidth": GRID_WIDTH,
    "lines.linewidth": LINE_WIDTH,
    "hatch.linewidth": 0.7,
    "axes.titleweight": "regular",
    "font.weight": "regular",
}


def save_figure(
    fig: plt.Figure,
    stem: str,
    height_single: float,
    height_double: float,
    *,
    width_single: float | None = None,
    width_double: float | None = None,
) -> list[Path]:
    outputs: list[Path] = []
    for width, height, target_dir in [
        (width_single or SINGLE_WIDTH_IN, height_single, SINGLE_DIR),
        (width_double or DOUBLE_WIDTH_IN, height_double, DOUBLE_DIR),
    ]:
        fig.set_size_inches(width, height)
        target_dir.mkdir(parents=True, exist_ok=True)
        for ext in ["png", "pdf", "tiff"]:
            path = target_dir / f"{stem}.{ext}"
            fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white")
            outputs.append(path)
    return outputs


def short_model_name(name: str) -> str:
    return MODEL_SHORT_NAMES.get(name, name)


def apply_axes_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", width=SPINE_WIDTH)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)


def add_panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.5,
        1.03,
        text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        color="black",
        clip_on=False,
    )


def write_table(df: pd.DataFrame, stem: str) -> list[Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / f"{stem}.csv"
    md_path = TABLE_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return [csv_path, md_path]


def load_data() -> dict[str, pd.DataFrame]:
    baseline_summary = pd.read_csv(BASELINE_DIR / "model_summary.csv")
    enhanced_summary = pd.read_csv(ENHANCED_DIR / "model_summary.csv")
    comparison = pd.read_csv(ENHANCED_DIR / "comparison_vs_baseline.csv")
    champion = pd.read_csv(ALERT_DIR / "champion_model_selection.csv")
    ranking = pd.read_csv(ALERT_DIR / "city_attention_ranking.csv", parse_dates=["peak_risk_date"])
    alerts = pd.read_csv(ALERT_DIR / "forecast_alerts.csv", parse_dates=["forecast_date", "latest_observed_date"])
    heatmap = pd.read_csv(ALERT_DIR / "alert_heatmap_table.csv", parse_dates=["forecast_date"])
    daily = pd.read_csv(PROCESSED_DIR / "albania_air_quality_daily.csv", parse_dates=["date"])
    return {
        "baseline_summary": baseline_summary,
        "enhanced_summary": enhanced_summary,
        "comparison": comparison,
        "champion": champion,
        "ranking": ranking,
        "alerts": alerts,
        "heatmap": heatmap,
        "daily": daily,
    }


def generate_tables(data: dict[str, pd.DataFrame]) -> tuple[list[tuple[str, list[Path]]], pd.DataFrame]:
    baseline_best = (
        data["baseline_summary"]
        .assign(model_priority=lambda df: df["model"].map(BASELINE_MODEL_PRIORITY))
        .sort_values(["target", "horizon_days", "mean_mae", "mean_rank", "model_priority"])
        .groupby(["target", "horizon_days"], as_index=False)
        .first()[["target", "target_label", "horizon_days", "model", "mean_mae", "mean_mape"]]
        .rename(columns={"model": "best_model"})
    )
    baseline_best = baseline_best.round({"mean_mae": 4, "mean_mape": 4})
    comparison_table = data["comparison"][
        [
            "target",
            "horizon_days",
            "enhanced_best_model",
            "enhanced_mean_mae",
            "baseline_best_model",
            "baseline_mean_mae",
            "mae_improvement",
            "mape_improvement",
        ]
    ].copy()
    comparison_table = comparison_table.round(
        {
            "enhanced_mean_mae": 4,
            "baseline_mean_mae": 4,
            "mae_improvement": 4,
            "mape_improvement": 4,
        }
    )
    ranking_table = data["ranking"][
        [
            "city",
            "final_alert_level",
            "peak_risk_date",
            "peak_horizon_days",
            "peak_predicted_aqi",
            "peak_predicted_pm25",
            "selected_aqi_model",
            "selected_pm2_5_model",
        ]
    ].copy()
    ranking_table["peak_risk_date"] = ranking_table["peak_risk_date"].dt.strftime("%Y-%m-%d")
    ranking_table = ranking_table.round({"peak_predicted_aqi": 2, "peak_predicted_pm25": 2})

    source_counts = (
        data["champion"]
        .groupby(["target", "horizon_days", "selected_source"], as_index=False)
        .size()
        .rename(columns={"size": "n_city_selections"})
    )
    champion_detail = data["champion"][
        [
            "target",
            "city",
            "horizon_days",
            "selected_source",
            "selected_model",
            "selected_mae",
            "baseline_model",
            "baseline_mae",
            "enhanced_model",
            "enhanced_mae",
        ]
    ].copy()
    champion_detail = champion_detail.round({"selected_mae": 4, "baseline_mae": 4, "enhanced_mae": 4})

    manifests: list[tuple[str, list[Path]]] = []
    manifests.append(("Table 1", write_table(baseline_best, "table_1_baseline_best_models")))
    manifests.append(("Table 2", write_table(comparison_table, "table_2_enhanced_vs_baseline")))
    manifests.append(("Table 3", write_table(ranking_table, "table_3_city_attention_ranking")))
    manifests.append(("Table S1", write_table(source_counts, "table_s1_champion_source_counts")))
    manifests.append(("Table S2", write_table(champion_detail, "table_s2_champion_model_selection")))
    return manifests, baseline_best


def fig_best_model_mae_comparison(comparison: pd.DataFrame) -> tuple[plt.Figure, str]:
    fig, axes = plt.subplots(1, 2, sharey=False)
    targets = [
        ("european_aqi_max", "Daily Max AQI"),
        ("pm2_5_mean", "Daily Mean PM2.5"),
    ]
    for ax, (target, title) in zip(axes, targets):
        subset = comparison[comparison["target"] == target].sort_values("horizon_days")
        y = np.arange(len(subset))
        ax.barh(
            y + 0.16,
            subset["baseline_mean_mae"],
            height=0.28,
            color=BASELINE_COLOR,
            edgecolor="black",
            linewidth=BAR_EDGE_WIDTH,
            label="Baseline",
        )
        ax.barh(
            y - 0.16,
            subset["enhanced_mean_mae"],
            height=0.28,
            color=ENHANCED_COLOR,
            edgecolor="black",
            linewidth=BAR_EDGE_WIDTH,
            label="Enhanced",
        )
        max_val = max(subset["baseline_mean_mae"].max(), subset["enhanced_mean_mae"].max())
        label_pad = max_val * 0.03
        for ypos, value, model_name in zip(y + 0.16, subset["baseline_mean_mae"], subset["baseline_best_model"]):
            ax.text(value + label_pad, ypos, short_model_name(model_name), va="center", ha="left", fontsize=ANNOTATION_SIZE)
        for ypos, value, model_name in zip(y - 0.16, subset["enhanced_mean_mae"], subset["enhanced_best_model"]):
            ax.text(value + label_pad, ypos, short_model_name(model_name), va="center", ha="left", fontsize=ANNOTATION_SIZE)
        ax.set_yticks(y)
        ax.set_yticklabels([f"H{h}" for h in subset["horizon_days"]])
        ax.set_xlabel("Mean MAE")
        ax.grid(axis="x", linestyle=":", linewidth=GRID_WIDTH)
        ax.set_xlim(0, max_val * 1.42)
        add_panel_label(ax, title)
        apply_axes_style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, fontsize=LEGEND_SIZE)
    fig.tight_layout(rect=[0, 0.10, 1, 0.90])
    return fig, "figure_1_best_model_mae_comparison"


def fig_champion_source_counts(champion: pd.DataFrame) -> tuple[plt.Figure, str]:
    counts = (
        champion.groupby(["target", "horizon_days", "selected_source"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    fig, axes = plt.subplots(1, 2, sharey=True)
    targets = [
        ("european_aqi_max", "Daily Max AQI"),
        ("pm2_5_mean", "Daily Mean PM2.5"),
    ]
    for ax, (target, title) in zip(axes, targets):
        subset = counts[counts["target"] == target].pivot(index="horizon_days", columns="selected_source", values="n").fillna(0)
        subset = subset.reindex(columns=["baseline", "enhanced"], fill_value=0)
        x = np.arange(len(subset.index))
        width = 0.34
        baseline_bars = ax.bar(
            x - width / 2,
            subset["baseline"],
            width=width,
            color=BASELINE_COLOR,
            edgecolor="black",
            linewidth=BAR_EDGE_WIDTH,
            label="Baseline",
        )
        enhanced_bars = ax.bar(
            x + width / 2,
            subset["enhanced"],
            width=width,
            color=ENHANCED_COLOR,
            edgecolor="black",
            linewidth=BAR_EDGE_WIDTH,
            label="Enhanced",
        )
        for bars in [baseline_bars, enhanced_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.08,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=ANNOTATION_SIZE,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{h}" for h in subset.index])
        ax.set_xlabel("Forecast horizon")
        ax.grid(axis="y", linestyle=":", linewidth=GRID_WIDTH)
        ax.set_ylim(0, 8.8)
        ax.text(
            0.5,
            0.88,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=PANEL_LABEL_SIZE,
            color="black",
        )
        apply_axes_style(ax)
    axes[0].set_ylabel("Number of city selections")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, fontsize=LEGEND_SIZE)
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    return fig, "figure_2_champion_source_counts"


def fig_city_attention_ranking(ranking: pd.DataFrame) -> tuple[plt.Figure, str]:
    subset = ranking.sort_values(["max_alert_score", "peak_predicted_aqi", "peak_predicted_pm25"], ascending=[True, True, True])
    fig, ax = plt.subplots()
    y = np.arange(len(subset))
    legend_items: dict[str, object] = {}
    for ypos, (_, row) in zip(y, subset.iterrows()):
        style = ALERT_STYLES.get(row["final_alert_level"], ALERT_STYLES["Stable"])
        bar = ax.barh(
            ypos,
            row["peak_predicted_aqi"],
            height=0.81,
            color=style["facecolor"],
            edgecolor="black",
            linewidth=BAR_EDGE_WIDTH,
        )[0]
        ax.text(
            row["peak_predicted_aqi"] + 0.35,
            ypos,
            row["final_alert_level"],
            va="center",
            ha="left",
            fontsize=ANNOTATION_SIZE,
        )
        if row["final_alert_level"] not in legend_items:
            legend_items[row["final_alert_level"]] = bar
    ax.set_yticks(y)
    ax.set_yticklabels(subset["city"])
    ax.set_xlabel("Peak predicted AQI")
    ax.xaxis.set_label_coords(0.5, -0.18)
    ax.grid(axis="x", linestyle=":", linewidth=GRID_WIDTH)
    ax.set_xlim(0, subset["peak_predicted_aqi"].max() + 6.5)
    apply_axes_style(ax)
    legend_order = [level for level in ALERT_STYLES if level in legend_items]
    handles = [legend_items[level] for level in legend_order]
    fig.legend(handles, legend_order, frameon=False, loc="lower center", ncol=len(handles), fontsize=LEGEND_SIZE)
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    return fig, "figure_3_city_attention_ranking"


def fig_alert_heatmap(heatmap: pd.DataFrame, ranking: pd.DataFrame) -> tuple[plt.Figure, str]:
    order = ranking.sort_values(["max_alert_score", "peak_predicted_aqi", "peak_predicted_pm25"], ascending=[False, False, False])["city"].tolist()
    frame = heatmap.copy()
    frame["forecast_date_str"] = frame["forecast_date"].dt.strftime("%Y-%m-%d")
    pivot = frame.pivot(index="city", columns="forecast_date_str", values="alert_score").reindex(order)
    fig, ax = plt.subplots()
    cmap = mpl.colors.ListedColormap(ALERT_HEATMAP_COLORS)
    bounds = np.arange(0.5, 6.6, 1.0)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    image = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Alert score", fontsize=AXIS_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE, width=SPINE_WIDTH)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if not np.isnan(value):
                text_color = "white" if value >= 4 else "black"
                ax.text(j, i, int(value), ha="center", va="center", color=text_color, fontsize=SMALL_ANNOTATION_SIZE)
    apply_axes_style(ax)
    fig.tight_layout()
    return fig, "figure_4_alert_heatmap"


def fig_top_city_aqi_paths(daily: pd.DataFrame, alerts: pd.DataFrame, ranking: pd.DataFrame) -> tuple[plt.Figure, str]:
    top_cities = ranking.head(4)["city"].tolist()
    top_city_models = ranking.set_index("city")["selected_aqi_model"].to_dict()
    fig, axes = plt.subplots(2, 2, sharex=False, sharey=False)
    axes = axes.flatten()
    for ax, city in zip(axes, top_cities):
        history = daily[daily["city"] == city].sort_values("date").tail(21)
        forecast = alerts[alerts["city"] == city].sort_values("forecast_date")
        ax.plot(history["date"], history["european_aqi_max"], color=OBSERVED_COLOR, linewidth=1.1, label="Observed")
        ax.plot(
            forecast["forecast_date"],
            forecast["predicted_european_aqi_max"],
            color=FORECAST_COLOR,
            linewidth=1.0,
            marker="o",
            markersize=MARKER_SIZE,
            linestyle="--",
            label="Forecast",
        )
        ax.set_title(city, fontsize=7.8)
        ax.tick_params(axis="x", rotation=40, labelsize=TICK_LABEL_SIZE, width=SPINE_WIDTH)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, width=SPINE_WIDTH)
        ax.grid(axis="y", linestyle=":", linewidth=GRID_WIDTH)
        apply_axes_style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, fontsize=LEGEND_SIZE)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    return fig, "figure_5_top_city_aqi_paths"


def generate_figures(data: dict[str, pd.DataFrame]) -> list[tuple[str, list[Path]]]:
    manifests: list[tuple[str, list[Path]]] = []
    figure_builders = [
        ("Figure 1", fig_best_model_mae_comparison, data["comparison"], 2.8, 3.0, None, None),
        ("Figure 2", fig_champion_source_counts, data["champion"], 2.8, 3.0, None, None),
        ("Figure 3", fig_city_attention_ranking, data["ranking"], 1.71, 2.01, 2.08, 3.67),
        ("Figure 4", fig_alert_heatmap, (data["heatmap"], data["ranking"]), 1.53, 1.77, 1.68, 3.06),
        ("Figure 5", fig_top_city_aqi_paths, (data["daily"], data["alerts"], data["ranking"]), 4.4, 4.8, None, None),
    ]

    for label, builder, payload, h_single, h_double, w_single, w_double in figure_builders:
        if isinstance(payload, tuple):
            fig, stem = builder(*payload)
        else:
            fig, stem = builder(payload)
        paths = save_figure(fig, stem, h_single, h_double, width_single=w_single, width_double=w_double)
        manifests.append((label, paths))
        plt.close(fig)

    return manifests


def write_manifest(table_items: list[tuple[str, list[Path]]], figure_items: list[tuple[str, list[Path]]]) -> None:
    lines = ["# Publication Asset Manifest", ""]
    lines.append("## Tables")
    lines.append("")
    for label, paths in table_items:
        lines.append(f"### {label}")
        lines.append("")
        for path in paths:
            lines.append(f"- `{path.name}`")
        lines.append("")

    lines.append("## Figures")
    lines.append("")
    for label, paths in figure_items:
        lines.append(f"### {label}")
        lines.append("")
        for path in paths:
            lines.append(f"- `{path}`")
        lines.append("")

    (PUB_DIR / "manifest.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication tables and figures.")
    parser.add_argument("--figures-only", action="store_true", help="Generate figures only and skip tables.")
    parser.add_argument("--tables-only", action="store_true", help="Generate tables only and skip figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.figures_only and args.tables_only:
        raise SystemExit("Choose only one of --figures-only or --tables-only.")

    PUB_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PUBLICATION_RCPARAMS)
    data = load_data()
    table_items: list[tuple[str, list[Path]]] = []
    figure_items: list[tuple[str, list[Path]]] = []

    if not args.figures_only:
        table_items, _ = generate_tables(data)
    if not args.tables_only:
        figure_items = generate_figures(data)
    write_manifest(table_items, figure_items)
    print(f"Saved publication assets to {PUB_DIR}")


if __name__ == "__main__":
    main()
