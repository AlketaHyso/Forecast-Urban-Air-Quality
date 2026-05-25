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
STEM = "figure_s1_methodology_workflow_bw"


LINE_COLOR = "black"
BOX_FACE = "white"
BOX_EDGE = "black"
ARROW_STYLE = "-|>"


def add_box(ax, x: float, y: float, w: float, h: float, text: str, fontsize: int = 10) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.3,
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
        linespacing=1.25,
        zorder=2,
    )


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=ARROW_STYLE,
        mutation_scale=14,
        linewidth=1.1,
        color=LINE_COLOR,
        shrinkA=6,
        shrinkB=6,
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

    add_box(
        ax,
        0.18,
        0.86,
        0.64,
        0.10,
        "Open gridded inputs\n"
        "Open-Meteo Air Quality API\n"
        "Open-Meteo Historical Weather API\n"
        "8 Albanian cities, hourly data",
    )
    add_arrow(ax, (0.50, 0.85), (0.50, 0.795))

    add_box(
        ax,
        0.20,
        0.70,
        0.60,
        0.09,
        "Hourly harmonization\n"
        "Merge air-quality and weather records by city and timestamp\n"
        "Parse timestamps and numeric fields",
    )
    add_arrow(ax, (0.50, 0.695), (0.50, 0.645))

    add_box(
        ax,
        0.18,
        0.54,
        0.64,
        0.10,
        "Daily analytical layer\n"
        "Aggregate to city-day summaries\n"
        "Pollutant, AQI, and meteorological features\n"
        "AQI labels and risk indicators",
    )

    add_arrow(ax, (0.39, 0.535), (0.29, 0.478))
    add_arrow(ax, (0.61, 0.535), (0.71, 0.478))

    add_box(
        ax,
        0.05,
        0.35,
        0.38,
        0.12,
        "Local baseline benchmark\n"
        "One city, one model\n"
        "Naive, Drift, Holt Damped,\n"
        "ARIMA, Random Forest Lag7",
        fontsize=9.5,
    )
    add_box(
        ax,
        0.57,
        0.35,
        0.38,
        0.12,
        "Enhanced pooled benchmark\n"
        "One pooled model per target and horizon\n"
        "Persistence, Ridge, HistGBM\n"
        "Lag, rolling, calendar, weather, city indicators",
        fontsize=9.2,
    )

    add_arrow(ax, (0.24, 0.345), (0.37, 0.257))
    add_arrow(ax, (0.76, 0.345), (0.63, 0.257))

    add_box(
        ax,
        0.18,
        0.15,
        0.64,
        0.10,
        "Rolling-origin evaluation and comparison\n"
        "365-day lookback window\n"
        "28-day step between origins\n"
        "1-, 2-, and 3-day forecast horizons\n"
        "18 backtest origins per city-model-horizon",
    )
    add_arrow(ax, (0.50, 0.145), (0.50, 0.115))

    add_box(
        ax,
        0.18,
        0.01,
        0.64,
        0.10,
        "Champion selection and illustrative case study\n"
        "Choose lower-MAE family by city, target, and horizon\n"
        "City attention ranking, alert heatmap,\n"
        "AQI forecast paths",
    )

    for ext, dpi in [("pdf", None), ("png", 300), ("tiff", 300)]:
        path = out_dir / f"{STEM}.{ext}"
        save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)

    plt.close(fig)


def main() -> None:
    draw_figure((10.5, 14), DOUBLE_DIR)
    draw_figure((7.0, 14), SINGLE_DIR)
    print(f"Saved methodology schematic to {DOUBLE_DIR} and {SINGLE_DIR}")


if __name__ == "__main__":
    main()
