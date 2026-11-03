"""ScopeAD overview at the manuscript's two-column width.

Run from any directory to regenerate the manuscript overview.
All labels, rules and operations are vector objects; only the observed images
and score maps are raster data. Cached maps illustrate the mechanism, not a new
evaluation. The displayed output precedes CQ99; no ground truth is rendered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from figure_style import use_times_new_roman

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
CACHE = PAPER.parent / "figures/cache_rsg4_mechanism"
OUT = PAPER / "figures/fig_scopead_overview"
SCALES = (448, 672, 896, 1120)
LAYERS = (5, 11, 23)
RANK_WEIGHTS = np.array([1 / 16, 7 / 48, 13 / 48, 25 / 48])
WIDTH, HEIGHT = 7.0, 3.21
INK = "#22282D"
GRAY = "#60676D"
RULE = "#CED3D6"
BLUE = "#386A96"
ORANGE = "#B66A32"
TEAL = "#407C70"
PALE = "#F3F5F6"
BLUE_PALE = "#DEE9F2"
ZMAX = 2.0
PIXEL = (380, 133)  # (row, col) of the defect pixel followed in panel (d)

plt.rcParams.update({
    "font.size": 7.2,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": .6,
})
use_times_new_roman()


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT))
        self.ax.axis("off")
        self.labels = []

    def text(self, x, y, label, size=7.2, color=INK, weight="normal",
             ha="center", va="center", **kwargs):
        artist = self.ax.text(x, y, label, fontsize=size, color=color,
                              fontweight=weight, ha=ha, va=va, **kwargs)
        self.labels.append(artist)
        return artist

    def line(self, points, color=RULE, width=.7, style="-", zorder=1):
        x, y = zip(*points)
        self.ax.plot(x, y, color=color, linewidth=width, linestyle=style,
                     solid_capstyle="butt", zorder=zorder)

    def arrow(self, start, end, color=GRAY, width=.8, style="-", rad=0):
        self.ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=7,
            linewidth=width, color=color, linestyle=style,
            shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}",
            zorder=3))

    def box(self, x, y, width, height, fill="white", edge=RULE, lw=.7):
        self.ax.add_patch(Rectangle((x, y), width, height, facecolor=fill,
                                    edgecolor=edge, linewidth=lw, zorder=2))

    def image(self, x, y, width, array, color=RULE, score=False):
        height = width * array.shape[0] / array.shape[1]
        ax = self.fig.add_axes([x / WIDTH, y / HEIGHT,
                               width / WIDTH, height / HEIGHT])
        ax.set_xticks([])
        ax.set_yticks([])
        if score:
            ax.imshow(array, cmap="inferno", vmin=0, vmax=ZMAX,
                      interpolation="bilinear")
        else:
            ax.imshow(array, interpolation="bilinear")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(.65)
        return ax

    def validate_text_bounds(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        bounds = self.fig.bbox
        outside = []
        for artist in self.labels:
            box = artist.get_window_extent(renderer)
            if (box.x0 < bounds.x0 or box.y0 < bounds.y0 or
                    box.x1 > bounds.x1 or box.y1 > bounds.y1):
                outside.append(artist.get_text())
        if outside:
            raise RuntimeError(f"Labels outside the exported page: {outside}")

    def add_outer_frame(self, padding=.07, inset=.015):
        # Preserve the content's physical layout and leave space around the
        # footer and the fitted-state wire before drawing the enclosing rule.
        positions = [(ax, ax.get_position(original=True).frozen())
                     for ax in self.fig.axes]
        width, height = WIDTH + 2 * padding, HEIGHT + 2 * padding
        self.fig.set_size_inches(width, height)
        for ax, pos in positions:
            ax.set_position([(padding + pos.x0 * WIDTH) / width,
                             (padding + pos.y0 * HEIGHT) / height,
                             pos.width * WIDTH / width,
                             pos.height * HEIGHT / height])
        self.fig.add_artist(Rectangle(
            (inset / width, inset / height),
            1 - 2 * inset / width, 1 - 2 * inset / height,
            transform=self.fig.transFigure, fill=False,
            edgecolor="#9AA1A7", linewidth=.7, clip_on=False, zorder=10))


def load_maps():
    with np.load(CACHE / "overview_walnuts__defect.npz") as data:
        rgb = data["rgb"].copy()
        layers = {l: np.asarray(data[f"r672_normalized_l{l}"], np.float32)
                  for l in LAYERS}
        scale_maps = {}
        for r in SCALES:
            z5, z11, z23 = (np.asarray(data[f"r{r}_normalized_l{l}"], np.float32)
                            for l in LAYERS)
            scale_maps[r] = .5 * z11 + .5 * z23 + .125 * z5
    # Cached layer maps are bag-averaged and resized. These two operations
    # commute with linear layer fusion; rank sorting is applied afterwards.
    sorted_maps = np.sort(np.stack(list(scale_maps.values())), axis=0)
    combined = np.einsum("k,khw->hw", RANK_WEIGHTS, sorted_maps)
    with np.load(CACHE / "overview_walnuts__support.npz") as data:
        thumbs = data["thumbs"].copy()
    return rgb, layers, scale_maps, combined, thumbs


def support(c, thumbs, dy=-.08):
    """Panel (a). dy shifts the whole row down to tighten the vertical layout."""
    c.text(.08, 3.18 + dy, "(a) Normal-reference fitting", 8.2,
           weight="bold", ha="left")
    for i in reversed(range(3)):
        c.image(.14 + .055 * i, 2.61 + .035 * i + dy, .36,
                thumbs[i], color="white")
    c.text(.39, 2.46 + dy, "Normal supports", 7.2)
    c.arrow((.68, 2.79 + dy), (.87, 2.79 + dy))
    c.box(.90, 2.60 + dy, 1.10, .38, fill=PALE)
    c.text(1.45, 2.79 + dy, "Frozen DINOv2", 7.5)
    c.text(1.45, 2.46 + dy, "Layers × scales", 7.2, GRAY)
    c.arrow((2.03, 2.79 + dy), (2.23, 2.79 + dy))
    for j, color in enumerate((ORANGE, BLUE, BLUE)):
        for i in range(4):
            c.box(2.33 + i * .095 + j * .018, 2.65 + j * .08 + dy,
                  .065, .057, fill=color, edge="white", lw=.25)
    c.text(2.53, 3.06 + dy, "Independent banks", 7.2)
    c.text(2.53, 2.48 + dy, r"$B_m$", 8.0)
    c.arrow((2.86, 2.79 + dy), (3.10, 2.79 + dy))
    c.box(3.13, 2.60 + dy, 1.24, .38, fill=PALE)
    c.text(3.75, 2.79 + dy, "Cross-fit 1-NN", 7.5)
    c.arrow((4.40, 2.79 + dy), (4.76, 2.79 + dy))
    c.text(5.76, 3.06 + dy, "Normal reference", 7.2)
    c.box(4.79, 2.60 + dy, 1.94, .38, fill="#EDF3F1", edge=TEAL, lw=.65)
    c.text(5.76, 2.79 + dy, r"$B_m,\ c_m$", 9.0, TEAL)
    c.text(5.76, 2.46 + dy, "Fixed during inference", 7.2, GRAY)
    c.line([(5.76, 2.36 + dy), (5.76, 2.25), (.035, 2.25), (.035, .93)],
           TEAL, .65, (0, (3, 2)))
    c.arrow((.035, .93), (.13, .93), TEAL, .65)


def matching(c, rgb):
    c.text(.08, 2.13, "(b) Multi-scale matching", 8.2, weight="bold", ha="left")
    c.text(.43, 1.97, r"Query $x^q$", 7.2)
    c.image(.17, 1.42, .51, rgb)
    c.arrow((.72, 1.63), (.94, 1.63))
    c.box(.97, 1.44, .51, .38, PALE)
    c.text(1.225, 1.63, "ViT", 7.5)
    c.arrow((1.225, 1.41), (1.225, 1.13))
    c.box(.16, .77, 1.32, .33, fill=PALE)
    c.text(.82, .935, "Independent 1-NN", 7.5)
    c.arrow((.82, .74), (.82, .61))
    c.text(.82, .48, r"$z_m=d_m/c_m$", 9.0, TEAL)
    c.line([(1.40, .48), (1.64, .48), (1.64, 1.73)], TEAL, .65)


def layer_fusion(c, layers, scale_maps):
    c.text(1.82, 2.13, "(c) Asymmetric layer fusion", 8.2,
           weight="bold", ha="left")
    c.text(2.195, 1.99, "Deep", 7.2, BLUE)
    c.text(2.195, .36, "Shallow", 7.2, ORANGE)
    size = .43
    aspect = layers[5].shape[0] / layers[5].shape[1]
    for l, y, color in [(23, 1.50, BLUE), (11, .99, BLUE), (5, .48, ORANGE)]:
        center = y + size * aspect / 2
        c.image(1.98, y, size, layers[l], color, score=True)
        c.arrow((1.64, center), (1.94, center), TEAL, .65)
        c.text(1.83, center + .115, rf"$z_{{{l}}}$", 7.5, color)
        target = (2.96, 1.25) if l == 23 else (2.90, 1.18) if l == 11 else (2.96, 1.10)
        c.arrow((2.45, center), target, color, 1.5 if l in (23, 11) else .55)
    c.ax.add_patch(Circle((3.02, 1.18), .115, facecolor="white",
                          edgecolor=GRAY, linewidth=.65, zorder=4))
    c.text(3.02, 1.18, "+", 11, zorder=5)
    c.arrow((3.16, 1.18), (3.41, 1.18), INK, .75)
    c.image(3.45, .995, .42, scale_maps[672], INK, score=True)
    c.text(3.66, .87, r"$S_{r,b}$", 8.2)
    c.line([(3.91, 1.18), (4.06, 1.18), (4.06, 1.96)], INK, .75)
    c.arrow((4.06, 1.96), (4.49, 1.96), INK, .75)


def rank_fusion(c, scale_maps, combined):
    c.text(4.21, 2.13, "(d) Rényi rank aggregation", 8.2,
           weight="bold", ha="left")
    c.text(5.69, 1.97, "Bag mean + resize", 7.2, GRAY)
    py, px = PIXEL
    scores = np.array([float(scale_maps[r][py, px]) for r in SCALES])
    ordered = np.sort(scores)
    centers = [4.50 + i * .65 for i in range(4)]
    size = .39
    for i, r in enumerate(SCALES):
        ax = c.image(centers[i] - size / 2, 1.57, size, scale_maps[r], score=True)
        ax.plot(px, py, "o", ms=4.5, mfc="none", mec="white", mew=.9)
        c.text(centers[i], 1.47, str(r), 7.0, GRAY)
    for x in centers:
        c.line([(x, 1.41), (x, 1.36)], GRAY, .55)
    c.line([(centers[0], 1.36), (centers[-1], 1.36)], GRAY, .55)
    c.arrow((5.475, 1.36), (5.475, 1.28), GRAY, .65)
    c.text(5.475, 1.20, "Sort at each pixel", 7.5)
    # Paired bars occupy score-rank positions, not resolution positions.
    for k, x in enumerate(centers):
        c.box(x - .12, .84, .24, .23 * ordered[k] / ordered[-1],
              fill="#ACB6C2", edge="#ACB6C2", lw=.4)
        c.text(x, .735, r"$\times$", 7.5, GRAY)
        c.box(x - .12, .42, .24, .22 * RANK_WEIGHTS[k] / RANK_WEIGHTS[-1],
              fill=BLUE, edge=BLUE, lw=.4)
    c.line([(4.32, .84), (6.63, .84)], RULE, .5)
    c.line([(4.32, .42), (6.63, .42)], RULE, .5)
    c.text(4.32, .61, "Rank weights", 7.0, BLUE, ha="left")
    c.text(5.19, .19, r"$\widetilde A(p)=\sum_k w_k S_{(k)}(p)$", 8.2)
    c.arrow((6.03, .19), (6.37, .19), INK, .75)
    ax = c.image(6.41, .055, .40, combined, INK, score=True)
    ax.plot(px, py, "o", ms=4.5, mfc="none", mec="white", mew=.9)


def footer(c):
    c.text(.08, .075, "Shared map scale", 7.0, GRAY, ha="left")
    ax = c.fig.add_axes([1.16 / WIDTH, .045 / HEIGHT, .55 / WIDTH, .055 / HEIGHT])
    ax.imshow(np.linspace(0, ZMAX, 256)[None, :], cmap="inferno", aspect="auto",
              vmin=0, vmax=ZMAX)
    ax.set_axis_off()
    c.text(1.09, .075, "0", 7.0)
    c.text(1.77, .075, "2+", 7.0, ha="left")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    rgb, layers, scale_maps, combined, thumbs = load_maps()
    c = Canvas()
    support(c, thumbs)
    matching(c, rgb)
    layer_fusion(c, layers, scale_maps)
    rank_fusion(c, scale_maps, combined)
    footer(c)
    c.add_outer_frame()
    c.validate_text_bounds()
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "svg", "png"):
        c.fig.savefig(out.with_suffix("." + fmt),
                       dpi=450 if fmt == "png" else 300,
                       facecolor="white", pad_inches=0)
    metadata = {
        "method": "ScopeAD",
        "method_source": "sec/3_method.tex",
        "generator": "figures/source/" + Path(__file__).name,
        "width_inches": float(c.fig.get_figwidth()),
        "height_inches": float(c.fig.get_figheight()),
        "minimum_text_size_pt": min(t.get_fontsize() for t in c.labels),
        "design": "compact two-stage overview; aligned modules; weighted sum node; paired pixelwise score and rank-weight bars; one aggregation equation",
        "outer_frame": {"color": "#9AA1A7", "linewidth_pt": .7,
                        "content_padding_inches": .07, "inset_inches": .015},
        "layers": list(LAYERS), "scales": list(SCALES), "bags": 4,
        "tokens_per_image_per_bag": 256, "bank_count": 48,
        "calibration": "c_m=max(Q_0.95(D_m),epsilon); LSO for N>1, LTO for N=1",
        "layer_weights": [.125, .5, .5],
        "aggregation_order": "linear layer fusion -> bag mean -> bilinear resize -> pixelwise ascending sort -> rank-weighted sum",
        "scale_rank_weights": RANK_WEIGHTS.tolist(),
        "rank_assignment": "score rank at each pixel, not fixed resolution",
        "rank_example": "measured scores at defect pixel (row 380, col 133) of the cached display maps",
        "rank_example_scores_by_scale": [float(scale_maps[r][PIXEL]) for r in SCALES],
        "rank_score_bar_heights": "proportional to ascending scores, normalized to the largest score",
        "rank_weight_bar_heights": "proportional to Renyi weights, normalized to the largest weight",
        "output": "pre-postprocessing map A_tilde; CQ99 and image scoring in method text",
        "cq_rendered": False,
        "ground_truth_use": "none",
        "display_range": [0, ZMAX],
        "layer_map_display": "resolution 672; bag-averaged",
        "display_source": "cache_rsg4_mechanism/overview_walnuts__defect.npz",
        "display_source_sha256": hashlib.sha256((CACHE / "overview_walnuts__defect.npz").read_bytes()).hexdigest(),
        "display_provenance": "linear fusion and Renyi aggregation reconstructed from cached FP16, resized, bag-averaged normalized layer maps; no new inference or evaluation",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plt.close(c.fig)
    print(out)


if __name__ == "__main__":
    main()
