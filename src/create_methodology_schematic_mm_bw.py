from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "publication_assets" / "figures"
DOUBLE_DIR = FIGURES_DIR / "double_column"
SINGLE_DIR = FIGURES_DIR / "single_column"
STEM = "figure_s1_methodology_workflow_mm_bw"


MM_PER_INCH = 25.4
PADDING_MM = 2.0
ARROW_MM = 3.0
SIDE_GAP_MM = 8.0
MARGIN_X_MM = 5.0
MARGIN_Y_MM = 4.0

BOX_EDGE = "black"
BOX_FACE = "white"
LINE_COLOR = "black"
FONT_FAMILY = "DejaVu Sans"


BOX_SPECS = {
    "top": {
        "text": "Open gridded inputs\n"
        "Open-Meteo Air Quality API\n"
        "Open-Meteo Historical Weather API\n"
        "8 Albanian cities, hourly data",
        "fontsize": 8.7,
    },
    "harmon": {
        "text": "Hourly harmonization\n"
        "Merge air-quality and weather records by city and timestamp\n"
        "Parse timestamps and numeric fields",
        "fontsize": 8.5,
    },
    "daily": {
        "text": "Daily analytical layer\n"
        "Aggregate to city-day summaries\n"
        "Pollutant, AQI, and meteorological features\n"
        "AQI labels and risk indicators",
        "fontsize": 8.6,
    },
    "left": {
        "text": "Local baseline benchmark\n"
        "One city, one model\n"
        "Naive, Drift, Holt Damped,\n"
        "ARIMA, Random Forest Lag7",
        "fontsize": 8.4,
    },
    "right": {
        "text": "Enhanced pooled benchmark\n"
        "One pooled model per target and horizon\n"
        "Persistence, Ridge, HistGBM\n"
        "Lag, rolling, calendar, weather, city indicators",
        "fontsize": 8.1,
    },
    "eval": {
        "text": "Rolling-origin evaluation and comparison\n"
        "365-day lookback window\n"
        "28-day step between origins\n"
        "1-, 2-, and 3-day forecast horizons\n"
        "18 backtest origins per city-model-horizon",
        "fontsize": 8.3,
    },
    "output": {
        "text": "Champion selection and illustrative case study\n"
        "Choose lower-MAE family by city, target, and horizon\n"
        "City attention ranking, alert heatmap,\n"
        "AQI forecast paths",
        "fontsize": 8.3,
    },
}


def mm_to_inches(value_mm: float) -> float:
    return value_mm / MM_PER_INCH


def measure_text_block(text: str, fontsize: float, width_mm: float) -> tuple[float, float]:
    fig = plt.figure(figsize=(mm_to_inches(width_mm), mm_to_inches(60.0)), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, 60.0)
    ax.axis("off")

    txt = ax.text(
        width_mm / 2,
        30.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        family=FONT_FAMILY,
        color=LINE_COLOR,
        linespacing=1.05,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = txt.get_window_extent(renderer=renderer)
    width = bbox.width / fig.dpi * MM_PER_INCH
    height = bbox.height / fig.dpi * MM_PER_INCH
    plt.close(fig)
    return width, height


def compute_layout(width_mm: float) -> tuple[float, dict[str, dict[str, float | str]]]:
    boxes: dict[str, dict[str, float | str]] = {}
    for name, spec in BOX_SPECS.items():
        text_w, text_h = measure_text_block(spec["text"], spec["fontsize"], width_mm)
        boxes[name] = {
            "text": spec["text"],
            "fontsize": spec["fontsize"],
            "w": text_w + 2 * PADDING_MM,
            "h": text_h + 2 * PADDING_MM,
        }

    left_group = boxes["left"]["w"] + SIDE_GAP_MM + boxes["right"]["w"]
    inner_margin = max((width_mm - left_group) / 2, MARGIN_X_MM)
    left_x = inner_margin
    right_x = width_mm - inner_margin - boxes["right"]["w"]

    center_x = width_mm / 2
    boxes["left"]["x"] = left_x
    boxes["right"]["x"] = right_x
    for name in ("top", "harmon", "daily", "eval", "output"):
        boxes[name]["x"] = center_x - boxes[name]["w"] / 2

    boxes["output"]["y"] = MARGIN_Y_MM
    boxes["eval"]["y"] = boxes["output"]["y"] + boxes["output"]["h"] + ARROW_MM
    boxes["left"]["y"] = boxes["eval"]["y"] + boxes["eval"]["h"] + 2 * ARROW_MM
    boxes["right"]["y"] = boxes["left"]["y"]
    boxes["daily"]["y"] = boxes["left"]["y"] + boxes["left"]["h"] + 2 * ARROW_MM
    boxes["harmon"]["y"] = boxes["daily"]["y"] + boxes["daily"]["h"] + ARROW_MM
    boxes["top"]["y"] = boxes["harmon"]["y"] + boxes["harmon"]["h"] + ARROW_MM

    total_height = boxes["top"]["y"] + boxes["top"]["h"] + MARGIN_Y_MM
    return total_height, boxes


def add_box(ax, box: dict[str, float | str]) -> None:
    patch = FancyBboxPatch(
        (box["x"], box["y"]),
        box["w"],
        box["h"],
        boxstyle="round,pad=0,rounding_size=2.2",
        linewidth=1.15,
        edgecolor=BOX_EDGE,
        facecolor=BOX_FACE,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        box["x"] + box["w"] / 2,
        box["y"] + box["h"] / 2,
        box["text"],
        ha="center",
        va="center",
        fontsize=box["fontsize"],
        family=FONT_FAMILY,
        color=LINE_COLOR,
        linespacing=1.05,
        zorder=2,
    )


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=0.95,
        color=LINE_COLOR,
        shrinkA=0,
        shrinkB=0,
        zorder=3,
    )
    ax.add_patch(arrow)


def add_line(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_line(Line2D([start[0], end[0]], [start[1], end[1]], color=LINE_COLOR, linewidth=0.9, zorder=2))


def draw_figure(width_mm: float, out_dir: Path) -> None:
    total_height_mm, boxes = compute_layout(width_mm)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(mm_to_inches(width_mm), mm_to_inches(total_height_mm)), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, total_height_mm)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    for key in ("top", "harmon", "daily", "left", "right", "eval", "output"):
        add_box(ax, boxes[key])

    center_x = width_mm / 2

    top = boxes["top"]
    harmon = boxes["harmon"]
    daily = boxes["daily"]
    left = boxes["left"]
    right = boxes["right"]
    eval_box = boxes["eval"]
    output = boxes["output"]

    add_arrow(ax, (center_x, top["y"]), (center_x, harmon["y"] + harmon["h"]))
    add_arrow(ax, (center_x, harmon["y"]), (center_x, daily["y"] + daily["h"]))

    branch_y = daily["y"] - ARROW_MM
    left_center = left["x"] + left["w"] / 2
    right_center = right["x"] + right["w"] / 2
    add_arrow(ax, (center_x, daily["y"]), (center_x, branch_y))
    add_line(ax, (left_center, branch_y), (right_center, branch_y))
    add_arrow(ax, (left_center, branch_y), (left_center, left["y"] + left["h"]))
    add_arrow(ax, (right_center, branch_y), (right_center, right["y"] + right["h"]))

    merge_y = eval_box["y"] + eval_box["h"] + ARROW_MM
    add_arrow(ax, (left_center, left["y"]), (left_center, merge_y))
    add_arrow(ax, (right_center, right["y"]), (right_center, merge_y))
    add_line(ax, (left_center, merge_y), (right_center, merge_y))
    add_arrow(ax, (center_x, merge_y), (center_x, eval_box["y"] + eval_box["h"]))

    add_arrow(ax, (center_x, eval_box["y"]), (center_x, output["y"] + output["h"]))

    for ext, dpi in (("pdf", None), ("png", 300), ("tiff", 300)):
        path = out_dir / f"{STEM}.{ext}"
        save_kwargs = {"facecolor": "white"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)

    plt.close(fig)


def main() -> None:
    draw_figure(267.0, DOUBLE_DIR)
    draw_figure(178.0, SINGLE_DIR)
    print(f"Saved millimeter-layout methodology schematic to {DOUBLE_DIR} and {SINGLE_DIR}")


if __name__ == "__main__":
    main()
