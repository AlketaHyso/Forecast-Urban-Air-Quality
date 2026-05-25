from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR_NAME = "limited_history_robustness"
DEFAULT_OUTPUT_DIR_NAME = "limited_history_publication_assets"
SOURCE_DIR = PROJECT_ROOT / "outputs" / DEFAULT_SOURCE_DIR_NAME / "tables"
PUB_DIR = PROJECT_ROOT / "outputs" / DEFAULT_OUTPUT_DIR_NAME
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
ANNOTATION_SIZE = 6.2
LINE_WIDTH = 0.9
GRID_WIDTH = 0.45
SPINE_WIDTH = 0.8
BAR_EDGE_WIDTH = 0.8

REGIME_ORDER = [
    "Local Only",
    "Zero-Shot Cross-City",
    "Few-Shot Cross-City (90d)",
    "Full Pooled Reference",
]
REGIME_COLOR = {
    "Local Only": "#A9B8CC",
    "Zero-Shot Cross-City": "#6BAED6",
    "Few-Shot Cross-City (90d)": "#D95F02",
    "Full Pooled Reference": "#1F77B4",
}

PUBLICATION_RCPARAMS = {
    "font.size": 8,
    "axes.titlesize": PANEL_TITLE_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.titlesize": FIGURE_TITLE_SIZE,
    "axes.linewidth": SPINE_WIDTH,
    "grid.linewidth": GRID_WIDTH,
    "lines.linewidth": LINE_WIDTH,
}


def save_figure(fig: plt.Figure, stem: str, height_single: float, height_double: float) -> list[Path]:
    outputs: list[Path] = []
    for width, height, target_dir in [
        (SINGLE_WIDTH_IN, height_single, SINGLE_DIR),
        (DOUBLE_WIDTH_IN, height_double, DOUBLE_DIR),
    ]:
        fig.set_size_inches(width, height)
        target_dir.mkdir(parents=True, exist_ok=True)
        for ext in ["png", "pdf", "tiff"]:
            path = target_dir / f"{stem}.{ext}"
            fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white")
            outputs.append(path)
    return outputs


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
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return [csv_path, md_path]


def apply_axes_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", width=SPINE_WIDTH)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)


def regime_legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=REGIME_COLOR[label], edgecolor="black", label=label)
        for label in REGIME_ORDER
    ]


def load_data() -> dict[str, pd.DataFrame]:
    regime_summary = pd.read_csv(SOURCE_DIR / "regime_summary.csv")
    city_comparison = pd.read_csv(SOURCE_DIR / "city_regime_comparison.csv")
    best_regime = pd.read_csv(SOURCE_DIR / "best_regime_by_city.csv")
    alignment_summary = pd.read_csv(SOURCE_DIR / "reference_alignment_summary.csv")
    return {
        "regime_summary": regime_summary,
        "city_comparison": city_comparison,
        "best_regime": best_regime,
        "alignment_summary": alignment_summary,
    }


def generate_tables(data: dict[str, pd.DataFrame]) -> list[tuple[str, list[Path]]]:
    summary = data["regime_summary"].copy()
    summary["regime_label"] = pd.Categorical(summary["regime_label"], categories=REGIME_ORDER, ordered=True)
    summary = summary.sort_values("regime_label")
    summary_table = summary[
        [
            "regime_label",
            "mean_mae",
            "mean_rmse",
            "mean_mase",
            "mean_mape",
            "avg_train_rows_regime",
            "avg_local_rows_full_window",
            "avg_local_rows_adaptation_window",
        ]
    ].rename(
        columns={
            "regime_label": "Training regime",
            "mean_mae": "Mean MAE",
            "mean_rmse": "Mean RMSE",
            "mean_mase": "Mean MASE",
            "mean_mape": "Mean MAPE",
            "avg_train_rows_regime": "Avg. regime train rows",
            "avg_local_rows_full_window": "Avg. local rows in full window",
            "avg_local_rows_adaptation_window": "Avg. local rows in 90-day window",
        }
    ).round(4)

    city = data["city_comparison"].copy()
    city_table = city[
        [
            "city",
            "local_only",
            "zero_shot_cross_city",
            "few_shot_cross_city_90d",
            "full_pooled_reference",
            "zero_shot_vs_local_mae_gain",
            "few_shot_vs_local_mae_gain",
            "full_pooled_vs_local_mae_gain",
            "full_pooled_vs_few_shot_mae_gain",
        ]
    ].rename(
        columns={
            "city": "City",
            "local_only": "Local only MAE",
            "zero_shot_cross_city": "Zero-shot MAE",
            "few_shot_cross_city_90d": "Few-shot 90d MAE",
            "full_pooled_reference": "Full pooled MAE",
            "zero_shot_vs_local_mae_gain": "Zero-shot gain vs local",
            "few_shot_vs_local_mae_gain": "Few-shot gain vs local",
            "full_pooled_vs_local_mae_gain": "Full pooled gain vs local",
            "full_pooled_vs_few_shot_mae_gain": "Full pooled gain vs few-shot",
        }
    ).round(4)

    regime_counts = (
        data["best_regime"]["best_regime_label"]
        .value_counts()
        .reindex(REGIME_ORDER, fill_value=0)
        .rename_axis("Training regime")
        .reset_index(name="Number of city wins")
    )

    manifests: list[tuple[str, list[Path]]] = []
    manifests.append(("Table LH1", write_table(summary_table, "table_lh1_regime_summary")))
    manifests.append(("Table LHS1", write_table(city_table, "table_lhs1_city_level_mae")))
    manifests.append(("Table LHS2", write_table(regime_counts, "table_lhs2_best_regime_counts")))
    return manifests


def create_regime_summary_figure(summary: pd.DataFrame) -> list[Path]:
    summary = summary.copy()
    summary["regime_label"] = pd.Categorical(summary["regime_label"], categories=REGIME_ORDER, ordered=True)
    summary = summary.sort_values("regime_label")

    metrics = [
        ("mean_mae", "Mean MAE"),
        ("mean_rmse", "Mean RMSE"),
        ("mean_mase", "Mean MASE"),
    ]
    fig, axes = plt.subplots(1, 3, constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.30, left=0.08, right=0.99, wspace=0.35)
    x = np.arange(len(summary))
    panel_titles = [
        "(a) MAE",
        "(b) RMSE",
        "(c) MASE",
    ]

    for ax, (metric, ylabel), panel_title in zip(axes, metrics, panel_titles):
        values = summary[metric].to_numpy(dtype=float)
        colors = [REGIME_COLOR[label] for label in summary["regime_label"]]
        ax.bar(x, values, color=colors, edgecolor="black", linewidth=BAR_EDGE_WIDTH)
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Local", "Zero-shot", "Few-shot", "Full pooled"],
            rotation=18,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.set_ylim(0, float(np.max(values)) * 1.14)
        ax.grid(axis="y", alpha=0.25)
        apply_axes_style(ax)
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=ANNOTATION_SIZE)

    target_label = str(summary["target_label"].iloc[0])
    horizon_days = int(summary["horizon_days"].iloc[0])
    fig.legend(
        handles=regime_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.suptitle(f"{horizon_days}-day-ahead {target_label} constrained-history benchmark", y=0.975)
    return save_figure(fig, "figure_lh1_regime_summary", height_single=4.08, height_double=3.08)


def create_city_heatmap_figure(city: pd.DataFrame) -> list[Path]:
    city = city.copy().sort_values("city")
    matrix = city[
        ["local_only", "zero_shot_cross_city", "few_shot_cross_city_90d", "full_pooled_reference"]
    ].to_numpy(dtype=float)
    row_labels = city["city"].tolist()
    col_labels = ["Local", "Zero-shot", "Few-shot", "Full pooled"]

    fig, ax = plt.subplots(constrained_layout=True)
    image = ax.imshow(matrix, cmap=mpl.colormaps["YlOrRd_r"], aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Training regime")
    ax.set_ylabel("City")
    apply_axes_style(ax)

    for row_idx, city_name in enumerate(row_labels):
        for col_idx, _ in enumerate(col_labels):
            value = matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=ANNOTATION_SIZE,
                color="black",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("MAE")
    return save_figure(fig, "figure_lh2_city_mae_heatmap", height_single=3.5, height_double=2.8)


def create_regime_win_figure(best_regime: pd.DataFrame) -> list[Path]:
    counts = (
        best_regime["best_regime_label"]
        .value_counts()
        .reindex(REGIME_ORDER, fill_value=0)
        .reset_index()
    )
    counts.columns = ["regime_label", "wins"]

    fig, ax = plt.subplots(constrained_layout=True)
    colors = [REGIME_COLOR[label] for label in counts["regime_label"]]
    ax.bar(
        np.arange(len(counts)),
        counts["wins"],
        color=colors,
        edgecolor="black",
        linewidth=BAR_EDGE_WIDTH,
    )
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(["Local", "Zero-shot", "Few-shot", "Full pooled"], rotation=20, ha="right")
    ax.set_ylabel("Number of city wins")
    ax.set_ylim(0, max(1, counts["wins"].max()) + 1.2)
    ax.grid(axis="y", alpha=0.25)
    apply_axes_style(ax)
    for idx, value in enumerate(counts["wins"]):
        ax.text(idx, value, str(int(value)), ha="center", va="bottom", fontsize=ANNOTATION_SIZE)

    fig.legend(
        handles=regime_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    return save_figure(fig, "figure_lh3_best_regime_counts", height_single=3.02, height_double=2.32)


def write_manifest(table_entries: list[tuple[str, list[Path]]], figure_entries: list[tuple[str, list[Path]]]) -> Path:
    lines = [
        "# Limited-History Publication Assets",
        "",
        "## Tables",
    ]
    for label, paths in table_entries:
        lines.append(f"- **{label}**")
        for path in paths:
            lines.append(f"  - `{path}`")
    lines.extend(["", "## Figures"])
    for label, paths in figure_entries:
        lines.append(f"- **{label}**")
        for path in paths:
            lines.append(f"  - `{path}`")
    manifest_path = PUB_DIR / "manifest.md"
    manifest_path.write_text("\n".join(lines), encoding="utf-8")
    return manifest_path


def write_summary(data: dict[str, pd.DataFrame]) -> Path:
    summary = data["regime_summary"].copy()
    summary["regime_label"] = pd.Categorical(summary["regime_label"], categories=REGIME_ORDER, ordered=True)
    summary = summary.sort_values("regime_label")
    align = data["alignment_summary"].iloc[0]
    target_label = str(summary["target_label"].iloc[0])
    horizon_days = int(summary["horizon_days"].iloc[0])
    lines = [
        "# Limited-History Robustness Summary",
        "",
        (
            "This file summarizes the constrained-history "
            f"{target_label} {horizon_days}-day robustness experiment built around the existing HistGBM pooled benchmark."
        ),
        "",
        f"- Full pooled reference alignment with the existing enhanced benchmark: mean MAE delta = {align['mean_mae_delta']:.6f}, mean RMSE delta = {align['mean_rmse_delta']:.6f}.",
        "",
        "## Regime Summary",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "- "
            f"{row['regime_label']}: mean MAE = {row['mean_mae']:.4f}, "
            f"mean RMSE = {row['mean_rmse']:.4f}, "
            f"mean MASE = {row['mean_mase']:.4f}, "
            f"mean MAPE = {row['mean_mape']:.4f}."
        )

    summary_path = PUB_DIR / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--source-dir-name", default=DEFAULT_SOURCE_DIR_NAME)
    parser.add_argument("--output-dir-name", default=DEFAULT_OUTPUT_DIR_NAME)
    args = parser.parse_args()

    global SOURCE_DIR, PUB_DIR, TABLE_DIR, SINGLE_DIR, DOUBLE_DIR
    SOURCE_DIR = PROJECT_ROOT / "outputs" / args.source_dir_name / "tables"
    PUB_DIR = PROJECT_ROOT / "outputs" / args.output_dir_name
    TABLE_DIR = PUB_DIR / "tables"
    SINGLE_DIR = PUB_DIR / "figures" / "single_column"
    DOUBLE_DIR = PUB_DIR / "figures" / "double_column"

    PUB_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SINGLE_DIR.mkdir(parents=True, exist_ok=True)
    DOUBLE_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(PUBLICATION_RCPARAMS)

    data = load_data()
    table_entries: list[tuple[str, list[Path]]] = []
    if not args.figures_only:
        table_entries = generate_tables(data)
    figure_entries = [
        ("Figure LH1", create_regime_summary_figure(data["regime_summary"])),
        ("Figure LH2", create_city_heatmap_figure(data["city_comparison"])),
        ("Figure LH3", create_regime_win_figure(data["best_regime"])),
    ]
    manifest_path = write_manifest(table_entries, figure_entries)
    summary_path = write_summary(data)

    print("Saved limited-history publication assets to:")
    print(f"  - {manifest_path}")
    print(f"  - {summary_path}")
    for label, paths in table_entries + figure_entries:
        print(f"  - {label}")
        for path in paths:
            print(f"    - {path}")


if __name__ == "__main__":
    main()
