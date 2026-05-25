from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import create_methodology_schematic_mm_bw as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "publication_assets" / "figures"
DOUBLE_DIR = FIGURES_DIR / "double_column"
SINGLE_DIR = FIGURES_DIR / "single_column"
STEM = "figure_s1_methodology_workflow_mm_bw_tightx_textrelaxed"
SIDE_MARGIN_MM = 2.0
FONT_DELTA_PT = 0.35


def linespacing_with_extra_mm(fontsize_pt: float, extra_mm: float = 1.0) -> float:
    base_spacing = 1.05
    mm_per_line_unit = fontsize_pt * 0.3527777778
    return base_spacing + (extra_mm / mm_per_line_unit)


def add_box_relaxed(ax, box: dict[str, float | str]) -> None:
    patch = FancyBboxPatch(
        (box["x"], box["y"]),
        box["w"],
        box["h"],
        boxstyle="round,pad=0,rounding_size=2.2",
        linewidth=1.15,
        edgecolor=base.BOX_EDGE,
        facecolor=base.BOX_FACE,
        zorder=1,
    )
    ax.add_patch(patch)

    fontsize = float(box["fontsize"]) - FONT_DELTA_PT
    ax.text(
        float(box["x"]) + float(box["w"]) / 2,
        float(box["y"]) + float(box["h"]) / 2,
        box["text"],
        ha="center",
        va="center",
        fontsize=fontsize,
        family=base.FONT_FAMILY,
        color=base.LINE_COLOR,
        linespacing=linespacing_with_extra_mm(fontsize),
        zorder=2,
    )


def draw_figure(width_mm: float, out_dir: Path) -> None:
    total_height_mm, boxes = base.compute_layout(width_mm)
    out_dir.mkdir(parents=True, exist_ok=True)

    min_x = min(float(box["x"]) for box in boxes.values()) - SIDE_MARGIN_MM
    max_x = max(float(box["x"]) + float(box["w"]) for box in boxes.values()) + SIDE_MARGIN_MM
    cropped_width_mm = max_x - min_x
    shift_x = -min_x

    shifted_boxes: dict[str, dict[str, float | str]] = {}
    for name, box in boxes.items():
        shifted = dict(box)
        shifted["x"] = float(box["x"]) + shift_x
        shifted_boxes[name] = shifted

    fig = plt.figure(
        figsize=(base.mm_to_inches(cropped_width_mm), base.mm_to_inches(total_height_mm)),
        dpi=300,
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, cropped_width_mm)
    ax.set_ylim(0, total_height_mm)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    for key in ("top", "harmon", "daily", "left", "right", "eval", "output"):
        add_box_relaxed(ax, shifted_boxes[key])

    top = shifted_boxes["top"]
    harmon = shifted_boxes["harmon"]
    daily = shifted_boxes["daily"]
    left = shifted_boxes["left"]
    right = shifted_boxes["right"]
    eval_box = shifted_boxes["eval"]
    output = shifted_boxes["output"]

    center_x = float(daily["x"]) + float(daily["w"]) / 2
    left_center = float(left["x"]) + float(left["w"]) / 2
    right_center = float(right["x"]) + float(right["w"]) / 2

    base.add_arrow(ax, (center_x, float(top["y"])), (center_x, float(harmon["y"]) + float(harmon["h"])))
    base.add_arrow(ax, (center_x, float(harmon["y"])), (center_x, float(daily["y"]) + float(daily["h"])))

    branch_y = float(daily["y"]) - base.ARROW_MM
    base.add_arrow(ax, (center_x, float(daily["y"])), (center_x, branch_y))
    base.add_line(ax, (left_center, branch_y), (right_center, branch_y))
    base.add_arrow(ax, (left_center, branch_y), (left_center, float(left["y"]) + float(left["h"])))
    base.add_arrow(ax, (right_center, branch_y), (right_center, float(right["y"]) + float(right["h"])))

    merge_y = float(eval_box["y"]) + float(eval_box["h"]) + base.ARROW_MM
    base.add_arrow(ax, (left_center, float(left["y"])), (left_center, merge_y))
    base.add_arrow(ax, (right_center, float(right["y"])), (right_center, merge_y))
    base.add_line(ax, (left_center, merge_y), (right_center, merge_y))
    base.add_arrow(ax, (center_x, merge_y), (center_x, float(eval_box["y"]) + float(eval_box["h"])))

    base.add_arrow(ax, (center_x, float(eval_box["y"])), (center_x, float(output["y"]) + float(output["h"])))

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
    print(f"Saved text-relaxed methodology schematic to {DOUBLE_DIR} and {SINGLE_DIR}")


if __name__ == "__main__":
    main()
