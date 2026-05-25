from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "publication_assets" / "figures"
DOUBLE_DIR = FIGURES_DIR / "double_column"
SINGLE_DIR = FIGURES_DIR / "single_column"
STEM = "figure_s1_methodology_workflow_compact_color"


LINE_COLOR = "#2B2B2B"
BOX_EDGE = "#3A3A3A"
ARROW_STYLE = "-|>"

STAGE_COLORS = {
    "inputs": "#EAF3FF",
    "harmonization": "#EFF8F1",
    "daily": "#FFF6E8",
    "baseline": "#F6ECFF",
    "enhanced": "#EAF7F7",
    "evaluation": "#FFF0F2",
    "output": "#F2F2F2",
}


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    facecolor: str,
    fontsize: float = 10,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.25,
        edgecolor=BOX_EDGE,
        facecolor=facecolor,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=LINE_COLOR,
        family="DejaVu Sans",
        linespacing=1.22,
        zorder=2,
    )


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=ARROW_STYLE,
        mutation_scale=14,
        linewidth=1.05,
        color=LINE_COLOR,
        shrinkA=5,
        shrinkB=5,
        zorder=3,
    )
    ax.add_patch(arrow)


def draw_figure(figsize: tuple[float, float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Compact, evenly spaced vertical layout.
    main_w = 0.64
    main_x = 0.18
    branch_w = 0.39
    left_x = 0.04
    right_x = 0.57

    top_h = 0.095
    mid_h = 0.088
    branch_h = 0.115
    lower_h = 0.095
    gap = 0.038

    y_output = 0.055
    y_eval = y_output + lower_h + gap
    y_branch = y_eval + lower_h + gap
    y_daily = y_branch + branch_h + gap
    y_harmon = y_daily + lower_h + gap
    y_top = y_harmon + mid_h + gap

    add_box(
        ax,
        main_x,
        y_top,
        main_w,
        top_h,
        "Open gridded inputs\n"
        "Open-Meteo Air Quality API\n"
        "Open-Meteo Historical Weather API\n"
        "8 Albanian cities, hourly data",
        facecolor=STAGE_COLORS["inputs"],
    )
    add_arrow(ax, (0.50, y_top), (0.50, y_harmon + mid_h))

    add_box(
        ax,
        0.20,
        y_harmon,
        0.60,
        mid_h,
        "Hourly harmonization\n"
        "Merge air-quality and weather records by city and timestamp\n"
        "Parse timestamps and numeric fields",
        facecolor=STAGE_COLORS["harmonization"],
        fontsize=9.8,
    )
    add_arrow(ax, (0.50, y_harmon), (0.50, y_daily + lower_h))

    add_box(
        ax,
        main_x,
        y_daily,
        main_w,
        lower_h,
        "Daily analytical layer\n"
        "Aggregate to city-day summaries\n"
        "Pollutant, AQI, and meteorological features\n"
        "AQI labels and risk indicators",
        facecolor=STAGE_COLORS["daily"],
    )

    add_arrow(ax, (0.39, y_daily), (0.27, y_branch + branch_h))
    add_arrow(ax, (0.61, y_daily), (0.73, y_branch + branch_h))

    add_box(
        ax,
        left_x,
        y_branch,
        branch_w,
        branch_h,
        "Local baseline benchmark\n"
        "One city, one model\n"
        "Naive, Drift, Holt Damped,\n"
        "ARIMA, Random Forest Lag7",
        facecolor=STAGE_COLORS["baseline"],
        fontsize=9.5,
    )
    add_box(
        ax,
        right_x,
        y_branch,
        branch_w,
        branch_h,
        "Enhanced pooled benchmark\n"
        "One pooled model per target and horizon\n"
        "Persistence, Ridge, HistGBM\n"
        "Lag, rolling, calendar, weather, city indicators",
        facecolor=STAGE_COLORS["enhanced"],
        fontsize=9.1,
    )

    add_arrow(ax, (0.23, y_branch), (0.37, y_eval + lower_h))
    add_arrow(ax, (0.77, y_branch), (0.63, y_eval + lower_h))

    add_box(
        ax,
        main_x,
        y_eval,
        main_w,
        lower_h,
        "Rolling-origin evaluation and comparison\n"
        "365-day lookback window\n"
        "28-day step between origins\n"
        "1-, 2-, and 3-day forecast horizons\n"
        "18 backtest origins per city-model-horizon",
        facecolor=STAGE_COLORS["evaluation"],
        fontsize=9.5,
    )
    add_arrow(ax, (0.50, y_eval), (0.50, y_output + lower_h))

    add_box(
        ax,
        main_x,
        y_output,
        main_w,
        lower_h,
        "Champion selection and illustrative case study\n"
        "Choose lower-MAE family by city, target, and horizon\n"
        "City attention ranking, alert heatmap,\n"
        "AQI forecast paths",
        facecolor=STAGE_COLORS["output"],
        fontsize=9.4,
    )

    for ext, dpi in [("pdf", None), ("png", 300), ("tiff", 300)]:
        path = out_dir / f"{STEM}.{ext}"
        save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)

    plt.close(fig)


def main() -> None:
    draw_figure((10.5, 12.0), DOUBLE_DIR)
    draw_figure((7.0, 12.0), SINGLE_DIR)
    print(f"Saved compact methodology schematic to {DOUBLE_DIR} and {SINGLE_DIR}")


if __name__ == "__main__":
    main()
