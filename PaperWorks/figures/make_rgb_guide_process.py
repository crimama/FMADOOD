"""Render the paper figure illustrating the implemented RGB-guide pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from PIL import Image

from src.flow_tte_phase2_refinement import (
    _normalized_half_score,
    _resize_to_native,
    fast_guided_filter,
    load_half_guidance,
)


ROOT = Path(__file__).resolve().parents[2]
OBJECT = "wallplugs"
SAMPLE_ID = "010_underexposed"
RGB_PATH = Path(
    f"/home/hun/Volume/DATA/mvtec_ad_2/{OBJECT}/test_public/bad"
) / f"{SAMPLE_ID}.png"
GT_PATH = Path(
    f"/home/hun/Volume/DATA/mvtec_ad_2/{OBJECT}/test_public/ground_truth/bad"
) / f"{SAMPLE_ID}_mask.png"
SCORE_PATH = (
    ROOT
    / "results/remote_runs/dsba3/flowtte_gapdecomp_anchor_20260712_v1"
    / f"chunks/gpu2_vial_wallplugs/anomaly_maps/{OBJECT}/test/bad"
    / f"{SAMPLE_ID}.tiff"
)
OUTPUT_DIR = ROOT / "PaperWorks/figures"


def add_arrow(
    fig: plt.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str = "",
    label_xy: tuple[float, float] | None = None,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.6,
        color="#475569",
        connectionstyle="arc3,rad=0.0",
        zorder=10,
    )
    fig.add_artist(arrow)
    if label:
        x, y = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        fig.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="semibold",
            color="#334155",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none"},
            zorder=11,
        )


def show_panel(
    ax: plt.Axes,
    image: np.ndarray,
    title: str,
    subtitle: str,
    *,
    cmap: str | None = None,
) -> None:
    ax.imshow(image, cmap=cmap, vmin=0.0 if cmap else None, vmax=1.0 if cmap else None)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.9)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color="#0f172a", pad=5)
    ax.text(
        0.5,
        -0.09,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        color="#475569",
    )


def show_overlay_panel(
    ax: plt.Axes,
    rgb: np.ndarray,
    score: np.ndarray,
    ground_truth: np.ndarray,
    title: str,
    subtitle: str,
) -> None:
    ax.imshow(rgb)
    ax.imshow(score, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.52)
    ax.contour(
        ground_truth.astype(np.float32),
        levels=[0.5],
        colors=["#ef4444"],
        linewidths=1.4,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.9)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color="#0f172a", pad=5)
    ax.text(
        0.5,
        -0.09,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        color="#475569",
    )
    ax.legend(
        handles=[Line2D([0], [0], color="#ef4444", lw=1.8, label="GT boundary")],
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#cbd5e1",
        fontsize=7.5,
    )


def main() -> None:
    rgb = np.asarray(Image.open(RGB_PATH).convert("RGB"))
    ground_truth = np.asarray(Image.open(GT_PATH).convert("L")) > 0
    coarse = np.asarray(Image.open(SCORE_PATH), dtype=np.float32)
    coarse_half, score_min, score_span = _normalized_half_score(coarse)
    guidance_half = load_half_guidance(RGB_PATH, coarse.shape)
    guided_half = fast_guided_filter(guidance_half, coarse_half, radius=8, eps=0.01)
    refined = _resize_to_native(
        guided_half * np.float32(score_span) + np.float32(score_min), coarse.shape
    )
    refined_normalized = (refined - score_min) / max(score_span, 1e-12)

    fig = plt.figure(figsize=(17.2, 5.25), facecolor="white")
    grid = fig.add_gridspec(
        2,
        5,
        left=0.025,
        right=0.975,
        bottom=0.11,
        top=0.88,
        wspace=0.46,
        hspace=0.55,
        width_ratios=(1.0, 1.0, 0.82, 1.0, 1.0),
    )
    ax_rgb = fig.add_subplot(grid[0, 0])
    ax_gray = fig.add_subplot(grid[0, 1])
    ax_coarse = fig.add_subplot(grid[1, 0])
    ax_half = fig.add_subplot(grid[1, 1])
    ax_mechanism = fig.add_subplot(grid[:, 2])
    ax_guided = fig.add_subplot(grid[:, 3])
    ax_final = fig.add_subplot(grid[:, 4])

    show_panel(
        ax_rgb,
        rgb,
        "(a) Query RGB",
        f"MVTec AD 2: {OBJECT} / {SAMPLE_ID}",
    )
    show_panel(
        ax_gray,
        guidance_half,
        "(b) Grayscale guidance",
        "[0,1], half resolution",
        cmap="gray",
    )
    coarse_normalized = (coarse - score_min) / max(score_span, 1e-12)
    show_panel(
        ax_coarse,
        coarse_normalized,
        "(c) Coarse anomaly map",
        "latent 1-NN + weak NLL",
        cmap="viridis",
    )
    show_panel(
        ax_half,
        coarse_half,
        "(d) Normalized coarse map",
        "min–max normalized, half resolution",
        cmap="viridis",
    )
    show_panel(
        ax_guided,
        np.clip(guided_half, 0.0, 1.0),
        "(f) Guided response",
        "local linear filtering, r=8, ε=0.01",
        cmap="viridis",
    )
    show_overlay_panel(
        ax_final,
        rgb,
        np.clip(refined_normalized, 0.0, 1.0),
        ground_truth,
        "(g) Refined map on query",
        "refined score overlay; red contour denotes GT only here",
    )

    fig.suptitle(
        "Query-native RGB-guided spatial refinement",
        x=0.5,
        y=0.97,
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
    )

    # Make the local guided-filter mechanism explicit.
    ax_mechanism.set_axis_off()
    ax_mechanism.set_title(
        "(e) Local linear filtering",
        fontsize=10.5,
        fontweight="bold",
        color="#0f172a",
        pad=5,
    )
    mechanism_boxes = (
        (0.73, "Local statistics", "$\\mu_I,\\;\\mu_P,\\;\\sigma_I^2,\\;\\mathrm{cov}(I,P)$"),
        (0.46, "Linear coefficients", "$a=\\frac{\\mathrm{cov}(I,P)}{\\sigma_I^2+\\epsilon}$\n$b=\\mu_P-a\\mu_I$"),
        (0.19, "Window aggregation", "$Q_i=\\bar a_i I_i+\\bar b_i$"),
    )
    for y, heading, formula in mechanism_boxes:
        ax_mechanism.text(
            0.5,
            y,
            f"{heading}\n{formula}",
            transform=ax_mechanism.transAxes,
            ha="center",
            va="center",
            fontsize=8.6,
            linespacing=1.35,
            color="#1e3a8a",
            bbox={
                "boxstyle": "round,pad=0.55",
                "fc": "#eff6ff",
                "ec": "#60a5fa",
                "lw": 1.2,
            },
        )
    ax_mechanism.annotate(
        "",
        xy=(0.5, 0.56),
        xytext=(0.5, 0.65),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": "#475569", "lw": 1.4},
    )
    ax_mechanism.annotate(
        "",
        xy=(0.5, 0.29),
        xytext=(0.5, 0.38),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": "#475569", "lw": 1.4},
    )

    # The guidance and score branches merge in the local linear model.
    add_arrow(fig, (0.205, 0.687), (0.238, 0.687), "grayscale\n+ downsample")
    add_arrow(fig, (0.205, 0.300), (0.238, 0.300), "normalize\n+ downsample")
    add_arrow(fig, (0.405, 0.687), (0.432, 0.595))
    add_arrow(fig, (0.405, 0.300), (0.432, 0.405))
    add_arrow(fig, (0.572, 0.500), (0.620, 0.500))
    add_arrow(fig, (0.782, 0.500), (0.817, 0.500), "restore scale\n+ upsample")

    fig.text(
        0.5,
        0.025,
        "The RGB branch supplies query-native structure; the score branch supplies anomaly evidence. "
        "The filter aligns existing evidence without creating a new anomaly score.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#334155",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            OUTPUT_DIR / f"rgb_guide_process.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
