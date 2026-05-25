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
STEM = "figure_s2_forecasting_evaluation_design_bw"


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
        linespacing=1.22,
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
        0.14,
        0.86,
        0.72,
        0.09,
        "Daily analytical layer\n"
        "City-day pollutant, AQI,\n"
        "and meteorological summaries",
        fontsize=9.8,
    )
    add_arrow(ax, (0.50, 0.855), (0.50, 0.785))

    add_box(
        ax,
        0.15,
        0.70,
        0.70,
        0.10,
        "Forecast targets and issue logic\n"
        "Daily mean PM2.5 and daily maximum European AQI\n"
        "Forecasts issued after the issue-day\n"
        "daily summary is complete",
        fontsize=9.2,
    )

    add_arrow(ax, (0.38, 0.695), (0.27, 0.605))
    add_arrow(ax, (0.62, 0.695), (0.73, 0.605))

    add_box(
        ax,
        0.05,
        0.48,
        0.39,
        0.12,
        "Local baseline family\n"
        "One city, one model\n"
        "Naive, Drift, Holt Damped,\n"
        "ARIMA, Random Forest Lag7",
        fontsize=8.9,
    )
    add_box(
        ax,
        0.56,
        0.48,
        0.39,
        0.12,
        "Enhanced pooled family\n"
        "One pooled model per\n"
        "target and horizon\n"
        "Persistence, Ridge, HistGBM\n"
        "Lag, rolling, calendar,\n"
        "weather, city indicators",
        fontsize=8.8,
    )

    add_arrow(ax, (0.24, 0.475), (0.38, 0.365))
    add_arrow(ax, (0.76, 0.475), (0.62, 0.365))

    add_box(
        ax,
        0.18,
        0.25,
        0.64,
        0.11,
        "Aligned rolling-origin backtesting\n"
        "365-day lookback, 28-day step,\n"
        "1-, 2-, and 3-day horizons\n"
        "18 forecast origins per city-model-horizon",
        fontsize=9.3,
    )
    add_arrow(ax, (0.50, 0.245), (0.50, 0.175))

    add_box(
        ax,
        0.14,
        0.06,
        0.72,
        0.10,
        "Evaluation and deployment decision\n"
        "City-level MAE, RMSE, MAPE, sMAPE\n"
        "Cross-city comparison primarily\n"
        "by mean MAE\n"
        "Champion selection by city, target, and horizon",
        fontsize=9.0,
    )

    for ext, dpi in [("pdf", None), ("png", 300), ("tiff", 300)]:
        path = out_dir / f"{STEM}.{ext}"
        save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)

    plt.close(fig)


def main() -> None:
    draw_figure((10.5, 11.5), DOUBLE_DIR)
    draw_figure((7.2, 11.5), SINGLE_DIR)
    print(f"Saved forecasting schematic to {DOUBLE_DIR} and {SINGLE_DIR}")


if __name__ == "__main__":
    main()
