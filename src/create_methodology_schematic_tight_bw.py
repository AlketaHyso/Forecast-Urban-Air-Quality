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
STEM = "figure_s1_methodology_workflow_tight_bw"


LINE_COLOR = "black"
BOX_FACE = "white"
BOX_EDGE = "black"
ARROW_STYLE = "-|>"


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fontsize: float = 10.0,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.003,rounding_size=0.02",
        linewidth=1.15,
        edgecolor=BOX_EDGE,
        facecolor=BOX_FACE,
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
        linespacing=1.12,
        zorder=2,
    )


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=ARROW_STYLE,
        mutation_scale=8,
        linewidth=0.85,
        color=LINE_COLOR,
        shrinkA=2,
        shrinkB=2,
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

    # Tight layout with short, symmetric arrows and reduced internal whitespace.
    main_x = 0.20
    main_w = 0.60
    side_w = 0.34
    left_x = 0.08
    right_x = 0.58

    gap = 0.012
    h_top = 0.070
    h_harmon = 0.060
    h_daily = 0.070
    h_side = 0.076
    h_eval = 0.086
    h_output = 0.070

    y_output = 0.095
    y_eval = y_output + h_output + gap
    y_side = y_eval + h_eval + gap
    y_daily = y_side + h_side + gap
    y_harmon = y_daily + h_daily + gap
    y_top = y_harmon + h_harmon + gap

    add_box(
        ax,
        main_x,
        y_top,
        main_w,
        h_top,
        "Open gridded inputs\n"
        "Open-Meteo Air Quality API\n"
        "Open-Meteo Historical Weather API\n"
        "8 Albanian cities, hourly data",
        fontsize=8.9,
    )
    add_arrow(ax, (0.50, y_top), (0.50, y_harmon + h_harmon))

    add_box(
        ax,
        0.22,
        y_harmon,
        0.56,
        h_harmon,
        "Hourly harmonization\n"
        "Merge air-quality and weather records by city and timestamp\n"
        "Parse timestamps and numeric fields",
        fontsize=8.4,
    )
    add_arrow(ax, (0.50, y_harmon), (0.50, y_daily + h_daily))

    add_box(
        ax,
        main_x,
        y_daily,
        main_w,
        h_daily,
        "Daily analytical layer\n"
        "Aggregate to city-day summaries\n"
        "Pollutant, AQI, and meteorological features\n"
        "AQI labels and risk indicators",
        fontsize=8.8,
    )

    # Short symmetric arrows from lower side of daily box to the inner top edges of side boxes.
    add_arrow(ax, (0.40, y_daily), (0.40, y_side + h_side))
    add_arrow(ax, (0.60, y_daily), (0.60, y_side + h_side))

    add_box(
        ax,
        left_x,
        y_side,
        side_w,
        h_side,
        "Local baseline benchmark\n"
        "One city, one model\n"
        "Naive, Drift, Holt Damped,\n"
        "ARIMA, Random Forest Lag7",
        fontsize=8.3,
    )
    add_box(
        ax,
        right_x,
        y_side,
        side_w,
        h_side,
        "Enhanced pooled benchmark\n"
        "One pooled model per target and horizon\n"
        "Persistence, Ridge, HistGBM\n"
        "Lag, rolling, calendar, weather, city indicators",
        fontsize=8.0,
    )

    # Short symmetric arrows from inner bottom edges of side boxes to the evaluation box.
    add_arrow(ax, (0.32, y_side), (0.42, y_eval + h_eval))
    add_arrow(ax, (0.68, y_side), (0.58, y_eval + h_eval))

    add_box(
        ax,
        main_x,
        y_eval,
        main_w,
        h_eval,
        "Rolling-origin evaluation and comparison\n"
        "365-day lookback window\n"
        "28-day step between origins\n"
        "1-, 2-, and 3-day forecast horizons\n"
        "18 backtest origins per city-model-horizon",
        fontsize=8.4,
    )
    add_arrow(ax, (0.50, y_eval), (0.50, y_output + h_output))

    add_box(
        ax,
        main_x,
        y_output,
        main_w,
        h_output,
        "Champion selection and illustrative case study\n"
        "Choose lower-MAE family by city, target, and horizon\n"
        "City attention ranking, alert heatmap,\n"
        "AQI forecast paths",
        fontsize=8.3,
    )

    for ext, dpi in [("pdf", None), ("png", 300), ("tiff", 300)]:
        path = out_dir / f"{STEM}.{ext}"
        save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)

    plt.close(fig)


def main() -> None:
    draw_figure((10.5, 10.0), DOUBLE_DIR)
    draw_figure((7.0, 10.0), SINGLE_DIR)
    print(f"Saved tight black-and-white methodology schematic to {DOUBLE_DIR} and {SINGLE_DIR}")


if __name__ == "__main__":
    main()
