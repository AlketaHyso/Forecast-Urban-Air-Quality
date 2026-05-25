from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from run_enhanced_forecasting_benchmark import TARGETS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFER_DIR = PROJECT_ROOT / "outputs" / "transfer_learning_benchmark" / "tables"
PUB_DIR = PROJECT_ROOT / "outputs" / "transfer_learning_publication_assets"
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
ANNOTATION_SIZE = 6.4
LINE_WIDTH = 0.9
GRID_WIDTH = 0.45
SPINE_WIDTH = 0.8
BAR_EDGE_WIDTH = 0.8

BASELINE_COLOR = "#A9B8CC"
ENHANCED_COLOR = "#1F77B4"
TRANSFER_COLOR = "#D95F02"

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
        values = [str(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return [csv_path, md_path]


def apply_axes_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", width=SPINE_WIDTH)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)


def load_data() -> dict[str, pd.DataFrame]:
    comparison = pd.read_csv(TRANSFER_DIR / "comparison_vs_existing.csv")
    city_family = pd.read_csv(TRANSFER_DIR / "city_family_comparison.csv")
    family_winners = pd.read_csv(TRANSFER_DIR / "family_winners_by_city.csv")
    family_win_counts = pd.read_csv(TRANSFER_DIR / "family_win_counts.csv")
    return {
        "comparison": comparison,
        "city_family": city_family,
        "family_winners": family_winners,
        "family_win_counts": family_win_counts,
    }


def generate_tables(data: dict[str, pd.DataFrame]) -> list[tuple[str, list[Path]]]:
    target_label_map = TARGETS

    comparison = data["comparison"].copy()
    comparison["target_label"] = comparison["target"].map(target_label_map)
    overview = comparison[
        [
            "target_label",
            "horizon_days",
            "transfer_model",
            "transfer_mean_mae",
            "enhanced_best_model",
            "enhanced_mean_mae",
            "baseline_best_model",
            "baseline_mean_mae",
            "transfer_vs_enhanced_mae_improvement",
            "transfer_vs_baseline_mae_improvement",
        ]
    ].copy()
    overview = overview.rename(
        columns={
            "target_label": "Forecast target",
            "horizon_days": "Horizon (days)",
            "transfer_model": "Transfer model",
            "transfer_mean_mae": "Transfer mean MAE",
            "enhanced_best_model": "Best enhanced model",
            "enhanced_mean_mae": "Enhanced mean MAE",
            "baseline_best_model": "Best baseline model",
            "baseline_mean_mae": "Baseline mean MAE",
            "transfer_vs_enhanced_mae_improvement": "MAE gain vs enhanced",
            "transfer_vs_baseline_mae_improvement": "MAE gain vs baseline",
        }
    ).round(4)

    family_wins = data["family_win_counts"].copy()
    family_wins["target_label"] = family_wins["target"].map(target_label_map)
    family_wins = family_wins[
        ["target_label", "horizon_days", "winning_family", "n_city_wins"]
    ].rename(
        columns={
            "target_label": "Forecast target",
            "horizon_days": "Horizon (days)",
            "winning_family": "Winning family",
            "n_city_wins": "Number of city wins",
        }
    )

    city_delta = data["city_family"].copy()
    city_delta["target_label"] = city_delta["target"].map(target_label_map)
    city_delta = city_delta[
        [
            "target_label",
            "city",
            "horizon_days",
            "transfer_model",
            "transfer_mae",
            "enhanced_model",
            "enhanced_mae",
            "baseline_model",
            "baseline_mae",
            "transfer_vs_enhanced_mae_improvement",
            "transfer_vs_baseline_mae_improvement",
        ]
    ].rename(
        columns={
            "target_label": "Forecast target",
            "city": "City",
            "horizon_days": "Horizon (days)",
            "transfer_model": "Transfer model",
            "transfer_mae": "Transfer MAE",
            "enhanced_model": "Enhanced model",
            "enhanced_mae": "Enhanced MAE",
            "baseline_model": "Baseline model",
            "baseline_mae": "Baseline MAE",
            "transfer_vs_enhanced_mae_improvement": "MAE gain vs enhanced",
            "transfer_vs_baseline_mae_improvement": "MAE gain vs baseline",
        }
    ).round(4)

    manifests: list[tuple[str, list[Path]]] = []
    manifests.append(("Table TL1", write_table(overview, "table_tl1_transfer_vs_existing")))
    manifests.append(("Table TL2", write_table(family_wins, "table_tl2_family_win_counts")))
    manifests.append(("Table TLS1", write_table(city_delta, "table_tls1_city_level_deltas")))
    return manifests


def create_mae_comparison_figure(comparison: pd.DataFrame) -> list[Path]:
    comparison = comparison.copy()
    comparison["target_label"] = comparison["target"].map(TARGETS)
    comparison = comparison.sort_values(["target_label", "horizon_days"])

    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    width = 0.24

    for ax, (target, frame) in zip(axes, comparison.groupby("target", sort=False)):
        frame = frame.sort_values("horizon_days")
        x = np.arange(len(frame))
        ax.bar(
            x - width,
            frame["baseline_mean_mae"],
            width=width,
            color=BASELINE_COLOR,
            edgecolor="black",
            linewidth=BAR_EDGE_WIDTH,
            label="Baseline",
        )
        ax.bar(
            x,
            frame["enhanced_mean_mae"],
            width=width,
            color=ENHANCED_COLOR,
            edgecolor="black",
            linewidth=BAR_EDGE_WIDTH,
            label="Enhanced",
        )
        ax.bar(
            x + width,
            frame["transfer_mean_mae"],
            width=width,
            color=TRANSFER_COLOR,
            edgecolor="black",
            linewidth=BAR_EDGE_WIDTH,
            label="Transfer",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in frame["horizon_days"]])
        ax.set_xlabel("Forecast horizon (days)")
        ax.set_ylabel("Mean MAE")
        ax.set_title(TARGETS[target])
        ax.grid(axis="y", alpha=0.25)
        apply_axes_style(ax)

    axes[1].legend(loc="upper left", frameon=False)
    return save_figure(fig, "figure_tl1_family_mae_comparison", height_single=2.6, height_double=2.4)


def create_transfer_delta_heatmap(city_family: pd.DataFrame) -> list[Path]:
    city_family = city_family.copy()
    city_order = sorted(city_family["city"].unique())
    horizons = sorted(city_family["horizon_days"].unique())

    values = city_family["transfer_vs_enhanced_mae_improvement"].to_numpy(dtype=float)
    max_abs = max(float(np.nanmax(np.abs(values))), 1e-6)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    cmap = mpl.colormaps["RdBu_r"]

    for ax, target in zip(axes, TARGETS.keys()):
        panel = city_family.loc[city_family["target"] == target].copy()
        pivot = (
            panel.pivot(index="city", columns="horizon_days", values="transfer_vs_enhanced_mae_improvement")
            .reindex(index=city_order, columns=horizons)
        )
        image = ax.imshow(pivot.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(np.arange(len(horizons)))
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_yticks(np.arange(len(city_order)))
        ax.set_yticklabels(city_order)
        ax.set_xlabel("Forecast horizon (days)")
        ax.set_title(TARGETS[target])
        apply_axes_style(ax)
        for row_idx, city in enumerate(city_order):
            for col_idx, horizon in enumerate(horizons):
                value = pivot.loc[city, horizon]
                if pd.isna(value):
                    label = "NA"
                else:
                    label = f"{value:.2f}"
                ax.text(
                    col_idx,
                    row_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=ANNOTATION_SIZE,
                    color="black",
                )

    cbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label("MAE gain vs enhanced")
    return save_figure(fig, "figure_tl2_transfer_vs_enhanced_heatmap", height_single=3.2, height_double=2.9)


def create_family_win_figure(family_win_counts: pd.DataFrame) -> list[Path]:
    family_win_counts = family_win_counts.copy()
    family_win_counts["label"] = family_win_counts["winning_family"].map(
        {
            "baseline": "Baseline",
            "enhanced": "Enhanced",
            "transfer_learning": "Transfer",
        }
    )
    color_map = {
        "Baseline": BASELINE_COLOR,
        "Enhanced": ENHANCED_COLOR,
        "Transfer": TRANSFER_COLOR,
    }
    family_order = ["Baseline", "Enhanced", "Transfer"]

    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    for ax, target in zip(axes, TARGETS.keys()):
        panel = family_win_counts.loc[family_win_counts["target"] == target].copy()
        horizons = sorted(panel["horizon_days"].unique())
        x = np.arange(len(horizons))
        width = 0.24
        for offset, family in zip([-width, 0.0, width], family_order):
            family_panel = (
                panel.loc[panel["label"] == family]
                .set_index("horizon_days")
                .reindex(horizons, fill_value=0)
                .reset_index()
            )
            ax.bar(
                x + offset,
                family_panel["n_city_wins"],
                width=width,
                color=color_map[family],
                edgecolor="black",
                linewidth=BAR_EDGE_WIDTH,
                label=family,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_xlabel("Forecast horizon (days)")
        ax.set_ylabel("Number of city wins")
        ax.set_ylim(0, 8.5)
        ax.set_title(TARGETS[target])
        ax.grid(axis="y", alpha=0.25)
        apply_axes_style(ax)

    axes[1].legend(loc="upper left", frameon=False)
    return save_figure(fig, "figure_tl3_family_win_counts", height_single=2.6, height_double=2.4)


def write_manifest(entries: list[tuple[str, list[Path]]], figure_entries: list[tuple[str, list[Path]]]) -> Path:
    lines = [
        "# Transfer Learning Publication Assets",
        "",
        "## Tables",
    ]
    for label, paths in entries:
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
    comparison = data["comparison"].copy()
    comparison["target_label"] = comparison["target"].map(TARGETS)
    summary_lines = [
        "# Transfer Learning Benchmark Summary",
        "",
        "This file summarizes the standalone transfer-learning experiment added on top of the existing baseline and enhanced benchmarks.",
        "",
    ]
    for _, row in comparison.sort_values(["target_label", "horizon_days"]).iterrows():
        summary_lines.append(
            "- "
            f"{row['target_label']}, horizon {int(row['horizon_days'])}: "
            f"transfer MAE = {row['transfer_mean_mae']:.4f}, "
            f"best enhanced MAE = {row['enhanced_mean_mae']:.4f}, "
            f"best baseline MAE = {row['baseline_mean_mae']:.4f}, "
            f"gain vs enhanced = {row['transfer_vs_enhanced_mae_improvement']:.4f}, "
            f"gain vs baseline = {row['transfer_vs_baseline_mae_improvement']:.4f}."
        )
    summary_path = PUB_DIR / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return summary_path


def main() -> None:
    PUB_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SINGLE_DIR.mkdir(parents=True, exist_ok=True)
    DOUBLE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    mpl.rcParams.update(PUBLICATION_RCPARAMS)

    table_entries = generate_tables(data)
    figure_entries = [
        ("Figure TL1", create_mae_comparison_figure(data["comparison"])),
        ("Figure TL2", create_transfer_delta_heatmap(data["city_family"])),
        ("Figure TL3", create_family_win_figure(data["family_win_counts"])),
    ]
    manifest_path = write_manifest(table_entries, figure_entries)
    summary_path = write_summary(data)

    print("Saved transfer learning publication assets to:")
    print(f"  - {manifest_path}")
    print(f"  - {summary_path}")
    for label, paths in table_entries + figure_entries:
        print(f"  - {label}")
        for path in paths:
            print(f"    - {path}")


if __name__ == "__main__":
    main()
