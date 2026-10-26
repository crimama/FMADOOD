"""ScopeAD overview at the manuscript's two-column width.

Run with --sync-paper to update the figure included by main.tex.
All labels, rules and operations are vector objects; only the observed images
and score maps are raster data. Cached maps illustrate the mechanism, not a new
evaluation. The displayed output precedes CQ99; no ground truth is rendered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache_rsg4_mechanism"
PAPER = HERE.parent / "cvpr-2026-author-kit-template"
OUT = HERE / "fig_scopead_overview_submission"
SCALES = (448, 672, 896, 1120)
LAYERS = (5, 11, 23)
RANK_WEIGHTS = np.array([1 / 16, 7 / 48, 13 / 48, 25 / 48])
WIDTH, HEIGHT = 7.0, 3.80
INK = "#22282D"
GRAY = "#60676D"
RULE = "#CED3D6"
BLUE = "#386A96"
ORANGE = "#B66A32"
TEAL = "#407C70"
PALE = "#F3F5F6"
BLUE_PALE = "#DEE9F2"
ZMAX = 2.0

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 7.2,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": .6,
})


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


def support(c, thumbs):
    c.text(.08, 3.65, "(a) Normal-reference fitting", 8.5, weight="bold", ha="left")
    c.text(6.92, 3.65, "Normal images only; frozen backbone", 7.1,
           GRAY, ha="right")

    for i in reversed(range(3)):
        c.image(.11 + .075 * i, 3.00 + .045 * i, .43,
                thumbs[i], color="white")
    c.text(.40, 2.87, r"$N$ supports", 7.1)
    c.arrow((.73, 3.21), (.91, 3.21))

    c.box(.95, 3.00, 1.22, .47, fill=PALE)
    c.text(1.56, 3.35, "Frozen DINOv2", 7.4, weight="bold")
    c.text(1.56, 3.18, "ViT-L/14", 7.4)
    c.text(1.56, 2.87, "3 layers × 4 scales", 7.0)
    c.arrow((2.20, 3.21), (2.38, 3.21))

    for j, color in enumerate((ORANGE, BLUE, BLUE)):
        for i in range(4):
            c.box(2.49 + i * .105 + j * .022, 3.08 + j * .09,
                  .072, .065, fill=color, edge="white", lw=.3)
    c.text(2.79, 3.48, r"48 banks $B_m$", 7.4, weight="bold")
    c.text(2.79, 2.98, r"$m=(l,r,b)$", 7.2)
    c.text(2.79, 2.83, "4 bags; 256 tokens/image", 6.8)
    c.arrow((3.11, 3.21), (3.34, 3.21))

    c.text(4.08, 3.48, "Leave-source-out 1-NN", 7.4, weight="bold")
    for i in range(4):
        x = 3.53 + i * .27
        c.box(x, 3.06, .19, .19,
              fill="white" if i == 0 else "#DFECE8",
              edge=ORANGE if i == 0 else TEAL)
    c.arrow((3.63, 3.31), (4.02, 3.31), ORANGE, rad=-.30)
    c.text(3.63, 2.98, "hold out", 6.8, ORANGE)
    c.text(4.31, 2.98, r"match to rest: $D_m$", 6.8, TEAL)
    c.text(4.08, 2.82, "LTO for one shot", 6.8, GRAY)
    c.arrow((4.65, 3.21), (4.96, 3.21))

    c.text(5.98, 3.48, "Normal reference", 7.5, weight="bold")
    c.text(5.98, 3.23, r"$c_m=\max\{Q_{0.95}(D_m),\epsilon\}$", 8.2)
    c.text(5.98, 2.97, r"Fixed state: $\{B_m,c_m\}_{m=1}^{48}$", 7.4, TEAL)

    # One fitted-state wire, outside the query content, feeds matching and
    # normalization together. No query arrow enters the support-fitting path.
    c.line([(5.98, 2.85), (5.98, 2.68), (.035, 2.68), (.035, 1.06)],
           TEAL, .75, (0, (3, 2)))
    c.arrow((.035, 1.06), (.13, 1.06), TEAL, .75)


def matching(c, rgb):
    c.text(.08, 2.47, "(b) Multi-scale 1-NN", 8.1, weight="bold", ha="left")
    c.text(.45, 2.24, r"Query $x^q$", 7.3)
    c.image(.12, 1.61, .65, rgb)
    c.arrow((.80, 1.87), (.95, 1.87))
    c.box(.99, 1.63, .49, .49, PALE)
    c.text(1.235, 1.94, "same", 7.0, GRAY)
    c.text(1.235, 1.77, "ViT", 7.8, weight="bold")
    c.text(.62, 1.47, "Per bank", 7.0, GRAY)
    c.arrow((1.235, 1.60), (1.235, 1.32))

    c.box(.14, .83, 1.35, .49, fill="white", edge=RULE)
    c.text(.815, 1.15, r"$d_m=\min_{u\in B_m}(1-f_m^\top u)$", 7.6)
    c.text(.815, .95, "cosine distance", 6.8, GRAY)
    c.arrow((.815, .80), (.815, .68))
    c.text(.815, .51, r"$z_m=d_m/c_m$", 9.4, TEAL)
    c.text(.815, .31, "Normal-reference units", 6.8, GRAY)
    c.line([(1.47, .51), (1.66, .51), (1.66, 1.92)], TEAL, .75)


def layer_fusion(c, layers, scale_maps):
    c.text(1.79, 2.47, "(c) Asymmetric layer fusion", 8.1,
           weight="bold", ha="left")
    c.text(2.99, 2.25, "Deep reference + shallow detail", 7.0, GRAY)
    size = .50
    aspect = layers[5].shape[0] / layers[5].shape[1]
    for l, y, color, coeff in [(23, 1.71, BLUE, "0.5"),
                              (11, 1.13, BLUE, "0.5"),
                              (5, .55, ORANGE, r"\lambda")]:
        center = y + size * aspect / 2
        c.image(1.95, y, size, layers[l], color, score=True)
        c.arrow((1.66, center), (1.91, center), TEAL, .75)
        c.text(1.79, center + .14, rf"$z_{{{l}}}$", 7.8, color)
        c.arrow((2.49, center), (3.14, 1.38 + (.045 if l == 23 else -.045)),
                color, .9)
        c.text(2.70, center + .11, rf"$\times\,{coeff}$", 7.6, color,
               bbox={"facecolor": "white", "edgecolor": "none", "pad": .3})
    c.ax.add_patch(Circle((3.23, 1.38), .075, facecolor="white",
                          edgecolor=INK, linewidth=.8, zorder=4))
    c.text(3.23, 1.38, "+", 10.5, zorder=5)
    c.arrow((3.32, 1.38), (3.51, 1.38), INK)
    c.image(3.55, 1.145, .56, scale_maps[672], INK, score=True)
    c.text(3.83, 1.00, r"$S_{r,b}$", 8.6)
    c.text(2.96, .29, r"$S_{r,b}=\frac{z_{11}+z_{23}}{2}+\lambda\,z_5$", 8.6)
    c.line([(4.14, 1.38), (4.28, 1.38), (4.28, 2.25)], INK, .8)
    c.arrow((4.28, 2.25), (4.80, 2.25), INK)


def rank_fusion(c, scale_maps, combined):
    c.text(4.40, 2.47, "(d) Rényi rank aggregation", 8.1,
           weight="bold", ha="left")
    c.text(5.65, 2.25, "Bag mean → bilinear resize", 7.0, GRAY)
    for i, r in enumerate(SCALES):
        x = 4.48 + i * .65
        c.image(x, 1.76, .43, scale_maps[r], score=True)
        c.text(x + .215, 1.64, str(r), 6.8, GRAY)
    c.line([(4.70, 1.54), (6.645, 1.54)], width=.7)
    for i in range(4):
        c.line([(4.695 + i * .65, 1.58), (4.695 + i * .65, 1.54)], width=.7)
    c.arrow((5.35, 1.54), (5.35, 1.37))
    c.text(5.35, 1.23, "Pixelwise ascending sort", 7.4, weight="bold")

    # The bars depict the fixed rank weights, not data values or resolutions.
    c.text(4.50, .71, r"$w_k$", 7.6, BLUE, ha="right")
    for i, (weight, label) in enumerate(zip(RANK_WEIGHTS,
                                          ("1/16", "7/48", "13/48", "25/48"))):
        x = 4.68 + i * .34
        c.text(x, 1.04, rf"$S_{{({i + 1})}}$", 7.5)
        c.box(x - .115, .48, .23, .43 * weight / RANK_WEIGHTS[-1],
              BLUE_PALE, BLUE, .75)
        c.text(x, .36, label, 6.8, BLUE)
    c.arrow((5.88, .72), (6.08, .72), INK)
    c.image(6.14, .43, .75, combined, INK, score=True)
    c.text(6.515, .27, r"Map $\widetilde A$", 7.6)
    c.text(5.20, .13, r"$\widetilde A(p)=\sum_k w_k S_{(k)}(p)$", 7.9)


def footer(c):
    c.text(.08, .065, "Maps: common color scale", 6.8, GRAY, ha="left")
    ax = c.fig.add_axes([1.97 / WIDTH, .035 / HEIGHT, .65 / WIDTH, .055 / HEIGHT])
    ax.imshow(np.linspace(0, ZMAX, 256)[None, :], cmap="inferno", aspect="auto",
              vmin=0, vmax=ZMAX)
    ax.set_axis_off()
    c.text(1.89, .065, "0", 6.8)
    c.text(2.68, .065, "2+", 6.8, ha="left")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--sync-paper", action="store_true")
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
        "method_source": "../cvpr-2026-author-kit-template/sec/3_method.tex",
        "generator": Path(__file__).name,
        "width_inches": float(c.fig.get_figwidth()),
        "height_inches": float(c.fig.get_figheight()),
        "minimum_text_size_pt": min(t.get_fontsize() for t in c.labels),
        "design": "two-stage overview; open white layout; deep/shallow color coding; explicit rank aggregation",
        "outer_frame": {"color": "#9AA1A7", "linewidth_pt": .7,
                        "content_padding_inches": .07, "inset_inches": .015},
        "layers": list(LAYERS), "scales": list(SCALES), "bags": 4,
        "tokens_per_image_per_bag": 256, "bank_count": 48,
        "calibration": "c_m=max(Q_0.95(D_m),epsilon); LSO for N>1, LTO for N=1",
        "layer_weights": [.125, .5, .5],
        "aggregation_order": "linear layer fusion -> bag mean -> bilinear resize -> pixelwise ascending sort -> rank-weighted sum",
        "scale_rank_weights": RANK_WEIGHTS.tolist(),
        "rank_assignment": "score rank at each pixel, not fixed resolution",
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
    if args.sync_paper:
        for ext in ("pdf", "svg", "png", "meta.json"):
            shutil.copyfile(out.with_suffix("." + ext),
                            HERE / ("fig_scopead_overview." + ext))
        for ext in ("pdf", "meta.json"):
            shutil.copyfile(out.with_suffix("." + ext),
                            PAPER / "figures" / ("fig_scopead_overview." + ext))
    plt.close(c.fig)
    print(out)


if __name__ == "__main__":
    main()
