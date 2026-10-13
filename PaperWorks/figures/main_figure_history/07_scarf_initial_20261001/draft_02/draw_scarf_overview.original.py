"""Draw the SCARF (RSG4) overview figure from real cached intermediate maps.

Inputs come from extract_scarf_overview_maps.py (walnuts, MVTec AD 2
test_public, fixed 16-shot manifest, seed 0). Ground truth is used only for
the display contour on the final map.

Usage:
    .venv/bin/python PaperWorks/figures/draw_scarf_overview.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache_rsg4_mechanism"
OUT = HERE / "fig_scarf_overview"
NAME = "overview_walnuts"
RES = 672  # resolution whose per-component maps are shown

W, H = 7.0, 3.15  # inches; CVPR text width is 6.875 in
Z_MAX = 2.0       # shared colour limit for every z-unit map
EXCESS_GAIN = 4.0  # excess map uses Z_MAX / EXCESS_GAIN (stated in the panel)

NAVY = "#16324F"
MUTED = "#5A6B7D"
DEEP = "#2F6DA3"
DEEP_BG = "#E3EEF7"
SHALLOW = "#D07A1F"
SHALLOW_BG = "#FDF0DF"
SUPPORT = "#3E8E7E"
SUPPORT_BG = "#E6F3EF"
PANEL_BG = "#F7F9FB"
GT = "#38E1FF"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "font.size": 5.6, "svg.fonttype": "none", "pdf.fonttype": 42,
})


def load():
    defect = np.load(CACHE / f"{NAME}__defect.npz")
    support = np.load(CACHE / f"{NAME}__support.npz")
    meta = json.loads((CACHE / f"{NAME}__meta.json").read_text())
    return defect, support, meta


class Canvas:
    """Figure-inch coordinates for boxes, arrows, text and image insets."""

    def __init__(self):
        self.fig = plt.figure(figsize=(W, H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")

    def box(self, x, y, w, h, fill, edge, lw=0.7, radius=0.05, ls="-", z=0):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=z))

    def text(self, x, y, s, size=5.6, color=NAVY, weight="normal", ha="center",
             va="center", **kw):
        self.ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                     ha=ha, va=va, **kw)

    def arrow(self, p, q, color=MUTED, lw=0.8, ls="-", rad=0.0, head=6):
        self.ax.add_patch(FancyArrowPatch(
            p, q, arrowstyle="-|>", mutation_scale=head, linewidth=lw, color=color,
            linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
            zorder=3))

    def inset(self, x, y, w, h):
        ax = self.fig.add_axes([x / W, y / H, w / W, h / H])
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def image(self, x, y, w, h, array, edge, cmap=None, vmax=Z_MAX, lw=1.0):
        ax = self.inset(x, y, w, h)
        if cmap is None:
            ax.imshow(array, interpolation="bilinear")
        else:
            ax.imshow(np.asarray(array, dtype=np.float32), cmap=cmap, vmin=0.0,
                      vmax=vmax, interpolation="bilinear")
        for spine in ax.spines.values():
            spine.set_edgecolor(edge)
            spine.set_linewidth(lw)
        return ax


def draw_support_band(c, support, meta):
    c.box(0.05, 2.06, 6.90, 1.04, PANEL_BG, "#D5DEE7", lw=0.6, radius=0.08, z=-2)
    c.text(0.13, 2.99, "A   Support fitting", size=6.8, weight="bold", ha="left",
           color=SUPPORT)
    c.text(1.17, 2.99, "once, normal images only, no labels, no training",
           size=5.4, ha="left", color=MUTED)

    # Fanned support cards (real normal images).
    for i, thumb in enumerate(support["thumbs"][::-1]):
        k = 3 - i
        c.image(0.16 + 0.07 * k, 2.17 + 0.045 * k, 0.52, 0.52, thumb, "white", lw=0.8)
    c.text(0.55, 2.13, f"N = {len(meta['support'])} normal supports", size=5.2,
           color=MUTED)

    # Frozen backbone and multi-component banks.
    c.arrow((0.99, 2.48), (1.12, 2.48))
    c.box(1.13, 2.18, 1.20, 0.68, DEEP_BG, DEEP)
    c.text(1.73, 2.75, "Frozen DINOv2 ViT-L/14", size=5.8, weight="bold")
    c.text(1.73, 2.59, r"layers $l\in\{5, 11, 23\}$")
    c.text(1.73, 2.46, r"scales $r\in\{448,\dots,1120\}$")
    c.text(1.73, 2.33, "4 bags × 256 tokens / image")
    c.text(1.73, 2.22, r"$\Rightarrow$ 48 banks $B_m$, $m=(l,r,b)$", size=5.2,
           color=MUTED)

    # Leave-source-out matching sketch.
    c.arrow((2.34, 2.52), (2.47, 2.52))
    c.box(2.48, 2.18, 1.22, 0.68, SUPPORT_BG, SUPPORT)
    c.text(3.09, 2.75, "Leave-source-out NN", size=5.8, weight="bold")
    for j in range(4):
        x = 2.66 + 0.2 * j
        held = j == 0
        c.ax.add_patch(Rectangle((x, 2.42 + (0.09 if held else 0)), 0.15, 0.15,
                                 facecolor="white" if held else "#CFE4DD",
                                 edgecolor=SHALLOW if held else SUPPORT,
                                 linewidth=0.7, linestyle="--" if held else "-",
                                 zorder=2))
    c.arrow((2.82, 2.6), (2.99, 2.53), color=SHALLOW, lw=0.6, rad=-0.3, head=4)
    c.text(2.73, 2.36, r"hold $x_i$", size=5.0, color=SHALLOW)
    c.text(3.27, 2.36, "match to rest", size=5.0, color=SUPPORT)
    c.text(3.09, 2.24, r"normal distances $D_m^{\mathrm{LSO}}$", size=5.2)

    # Real LSO distributions in z units.
    c.arrow((3.71, 2.52), (3.86, 2.52))
    draw_lso_hist(c, support)

    # Noise reference.
    c.box(5.68, 2.18, 1.20, 0.68, SUPPORT_BG, SUPPORT)
    c.text(6.28, 2.75, "Noise reference", size=5.8, weight="bold")
    c.text(6.28, 2.57, r"$\sigma_S=\mathrm{median}_i\,\sigma(x_i^s)$", size=6.0)
    c.text(6.28, 2.41, "Immerkær 3×3 estimate,", size=5.2)
    c.text(6.28, 2.30, "structure-insensitive", size=5.2)
    # Raw support images also feed the noise reference (routed above the boxes).
    c.ax.plot([0.74, 0.74, 6.28], [2.84, 2.915, 2.915], color=SUPPORT, lw=0.5)
    c.arrow((6.28, 2.915), (6.28, 2.87), color=SUPPORT, lw=0.5, head=4)


def draw_lso_hist(c, support):
    ax = c.inset(3.97, 2.31, 1.55, 0.50)
    bins = np.linspace(0, 2.0, 61)
    for layer, color, label in ((23, DEEP, "layer 23"), (5, SHALLOW, "layer 5")):
        z = support[f"lso_l{layer}"] / support[f"lso_u_l{layer}"]
        ax.hist(z, bins=bins, density=True, histtype="stepfilled", alpha=0.28,
                color=color, linewidth=0)
        ax.hist(z, bins=bins, density=True, histtype="step", color=color,
                linewidth=0.7, label=label)
    tau = float(support["lso_tau_l5"])
    ax.axvline(1.0, color=NAVY, lw=0.8)
    ax.axvline(tau, color=SHALLOW, lw=0.8, ls=(0, (2, 1.2)))
    ax.set_xlim(0, 2.0)
    ax.set_yticks([])
    ax.set_xticks([0, 1, 2])
    ax.tick_params(axis="x", labelsize=4.8, length=1.5, pad=1, colors=NAVY)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.patch.set_alpha(0)
    ax.legend(fontsize=4.6, frameon=False, loc="center right", handlelength=1.0,
              borderaxespad=0.1, labelspacing=0.2)
    c.text(4.745, 2.85, r"real $D^{\mathrm{LSO}}_m / c_m$ ($r$=672)", size=5.4,
           weight="bold")
    ax.text(0.97, 0.98, r"$c_m{=}Q_{0.95}\!\rightarrow\!1$", transform=ax.get_xaxis_transform(),
            fontsize=4.8, color=NAVY, ha="right", va="top")
    ax.text(tau + 0.03, 0.62, rf"$\tau_5{{=}}\frac{{Q_{{0.99}}}}{{Q_{{0.95}}}}{{=}}{tau:.2f}$",
            transform=ax.get_xaxis_transform(), fontsize=4.8, color=SHALLOW,
            ha="left", va="top")


def draw_query_band(c, d, meta):
    c.box(0.05, 0.05, 6.90, 1.95, PANEL_BG, "#D5DEE7", lw=0.6, radius=0.08, z=-2)
    c.text(0.13, 1.89, "B   Query inference", size=6.8, weight="bold", ha="left",
           color=DEEP)
    c.text(1.30, 1.89, "per image, fixed support statistics, no test-time update",
           size=5.4, ha="left", color=MUTED)

    # Query and noise route.
    aspect = d["rgb"].shape[0] / d["rgb"].shape[1]
    qw = 0.88
    c.image(0.14, 0.98, qw, qw * aspect, d["rgb"], NAVY, lw=0.6)
    c.text(0.58, 1.78, r"query $x^q$ (shift 2)", size=5.2, color=MUTED)
    c.arrow((0.58, 0.97), (0.58, 0.85), color=NAVY)
    c.arrow((1.03, 0.49), (1.12, 0.49), color=NAVY)
    rho = meta["queries"]["defect"]["noise_ratio"]
    c.box(0.14, 0.14, 0.88, 0.70, "white", SUPPORT)
    c.text(0.58, 0.75, "Resolution route", size=5.8, weight="bold")
    c.ax.text(0.58, 0.61, r"$\rho=\sigma(x^q)/$", fontsize=5.8, color=NAVY, ha="right",
              va="center")
    c.ax.text(0.58, 0.61, r"$\sigma_S$", fontsize=5.8, color=SUPPORT, ha="left",
              va="center", fontweight="bold")
    c.text(0.58, 0.48, rf"$\rho$={rho:.2f} $\leq\kappa$=9", size=5.4, color=SUPPORT,
           weight="bold")
    c.text(0.58, 0.35, r"$\rightarrow$ all 4 scales", size=5.2, color=SUPPORT)
    c.text(0.58, 0.22, r"($\rho>\kappa$: 448, 672 only)", size=4.9, color=MUTED)

    draw_fusion_module(c, d)
    draw_aggregation(c, d)

    # One hand-off between the bands: fixed support statistics.
    c.arrow((3.6, 2.07), (3.6, 1.79), color=SUPPORT, ls=(0, (3, 2)), lw=0.8, head=6)
    c.text(3.66, 1.93, r"fixed: $B_m,\ c_m,\ \tau_5,\ \sigma_S$", size=5.2,
           color=SUPPORT, ha="left", weight="bold")


def draw_fusion_module(c, d):
    x0, x1 = 1.13, 4.62
    c.box(x0, 0.12, x1 - x0, 1.66, "white", SHALLOW, lw=0.9, radius=0.06, z=-1)
    c.text((x0 + x1) / 2, 1.70, "Support-calibrated asymmetric residual fusion",
           size=6.0, weight="bold", color=SHALLOW)
    c.text((x0 + x1) / 2, 1.585, rf"per component $m=(l,r,b)$:  $z_m=d_m/c_m$  "
           rf"(1-NN cosine distance in z units; shown at $r$={RES}, bag mean)",
           size=4.9, color=MUTED)

    mw = 0.52
    mh = mw * d["rgb"].shape[0] / d["rgb"].shape[1]
    key = lambda s: d[f"r{RES}_{s}"]

    # Column 1: normalised per-layer distances.
    cx = 1.30
    rows = {"l23": 1.06, "l11": 0.60, "l5": 0.16}
    for layer, y in rows.items():
        edge = SHALLOW if layer == "l5" else DEEP
        c.image(cx, y, mw, mh, key(f"normalized_{layer}"), edge, cmap="inferno")
        c.text(cx - 0.085, y + mh / 2, rf"$z_{{{layer[1:]}}}$", size=6.2,
               color=edge, weight="bold")

    # Deep base.
    dx = 2.10
    dy = 0.83
    c.arrow((cx + mw + 0.02, rows["l23"] + mh / 2), (dx - 0.02, dy + mh * 0.75),
            color=DEEP)
    c.arrow((cx + mw + 0.02, rows["l11"] + mh / 2), (dx - 0.02, dy + mh * 0.25),
            color=DEEP)
    c.image(dx, dy, mw, mh, key("deep"), DEEP, cmap="inferno", lw=1.2)
    c.text(dx + mw / 2, dy + mh + 0.07, r"deep base $z_d=\frac{z_{11}+z_{23}}{2}$",
           size=5.4, color=DEEP)
    c.text(dx + mw / 2, dy - 0.065, "always kept", size=5.0, color=DEEP)

    # Shallow gate: transfer function with this image's z5 histogram.
    gx, gy, gw, gh = 2.06, 0.20, 0.62, 0.36
    c.arrow((cx + mw + 0.02, rows["l5"] + mh / 2), (gx - 0.03, gy + gh / 2),
            color=SHALLOW)
    draw_gate(c, key("normalized_l5"), gx, gy, gw, gh)

    ex = 2.95
    c.arrow((gx + gw + 0.03, gy + gh / 2), (ex - 0.02, rows["l5"] + mh / 2),
            color=SHALLOW)
    c.image(ex, rows["l5"], mw, mh, key("shallow_residual"), SHALLOW, cmap="inferno",
            vmax=Z_MAX / EXCESS_GAIN, lw=1.2)
    gate = np.asarray(key("gate"), dtype=np.float32) > 0.5
    c.text(ex + mw / 2, rows["l5"] + mh + 0.15, r"excess $e_5=[z_5-\tau_5]_+$",
           size=5.4, color=SHALLOW)
    c.text(ex + mw / 2, rows["l5"] + mh + 0.06,
           f"open on {100 * gate.mean():.1f}% px  (colour ×{EXCESS_GAIN:g})",
           size=4.6, color=MUTED)

    # Sum.
    px, py = 3.72, 0.98
    c.arrow((dx + mw + 0.02, dy + mh / 2), (px - 0.06, py + 0.03), color=DEEP)
    c.arrow((ex + mw + 0.02, rows["l5"] + mh / 2), (px - 0.05, py - 0.04), color=SHALLOW)
    c.ax.add_patch(plt.Circle((px, py), 0.055, facecolor="white", edgecolor=NAVY,
                              linewidth=0.8, zorder=4))
    c.text(px, py, "+", size=8, weight="bold", zorder=5)
    sx = 3.88
    c.arrow((px + 0.06, py), (sx - 0.02, py), color=NAVY)
    sy = py - mh / 2
    fused = np.asarray(key("deep"), np.float32) + np.asarray(key("shallow_residual"),
                                                            np.float32)
    c.image(sx, sy, mw, mh, fused, NAVY, cmap="inferno", lw=1.2)
    c.text(sx + mw / 2, sy + mh + 0.07, r"$S=z_d+e_5$", size=5.8, weight="bold")

    # Contrast: naive full addition lights every textured normal walnut.
    naive = np.asarray(key("deep"), np.float32) + np.asarray(key("normalized_l5"),
                                                            np.float32)
    ny = 0.16
    c.image(sx + 0.06, ny, mw * 0.78, mh * 0.78, naive, "#A0A8B0",
            cmap="inferno", vmax=Z_MAX * 1.5, lw=0.6)
    c.text(sx + mw + 0.03, ny + mh * 0.39, "normal\ntexture\nlit", size=4.5,
           color=MUTED, ha="left", linespacing=1.1)
    c.text(sx + mw / 2, ny + mh * 0.78 + 0.055, r"cf. add all $z_d+z_5$:", size=4.7,
           color=MUTED)
    c.text(sx + mw / 2, ny - 0.0, "", size=1)


def draw_gate(c, z5, gx, gy, gw, gh):
    ax = c.inset(gx, gy, gw, gh)
    tau = 1.19
    values = np.asarray(z5, np.float32).ravel()
    bins = np.linspace(0, 2.0, 41)
    hist, edges = np.histogram(values, bins=bins, density=True)
    hist = hist / hist.max()
    centres = (edges[:-1] + edges[1:]) / 2
    ax.fill_between(centres, 0, hist * 0.55, step="mid", color="#B8C2CC", lw=0)
    above = centres > tau
    ax.fill_between(centres[above], 0, np.maximum(hist[above], 0.04) * 0.55,
                    step="mid", color=SHALLOW, lw=0)
    xs = np.linspace(0, 2.0, 200)
    ax.plot(xs, np.maximum(0, xs - tau), color=NAVY, lw=0.9)
    ax.axvline(tau, color=SHALLOW, lw=0.7, ls=(0, (2, 1.2)))
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 0.85)
    ax.set_xticks([0, tau, 2])
    ax.set_xticklabels(["0", r"$\tau_5$", "2"])
    ax.tick_params(axis="x", labelsize=4.8, length=1.5, pad=1, colors=NAVY)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.patch.set_alpha(0)
    c.text(gx + gw / 2, gy + gh + 0.07, r"shallow gate", size=5.4,
           color=SHALLOW, weight="bold")


def draw_aggregation(c, d):
    mw = 0.80
    mh = mw * d["rgb"].shape[0] / d["rgb"].shape[1]
    ax0, ay = 4.92, 0.62
    c.arrow((4.43, 0.98), (ax0 - 0.02, ay + mh / 2), color=NAVY)
    c.image(ax0, ay, mw, mh, d["raw_map"], NAVY, cmap="inferno")
    c.text(ax0 + mw / 2, ay + mh + 0.17, "Map aggregation", size=5.8, weight="bold")
    c.text(ax0 + mw / 2, ay + mh + 0.06, r"bags $\to$ bilinear $\uparrow$ $\to$ scales",
           size=4.9, color=MUTED)
    c.text(ax0 + mw / 2, ay - 0.08, r"base map $A_0$", size=5.4)

    fx = 5.98
    c.arrow((ax0 + mw + 0.02, ay + mh / 2), (fx - 0.02, ay + mh / 2), color=NAVY)
    c.text((ax0 + mw + fx) / 2, ay + mh / 2 + 0.09, "CQ", size=5.6, weight="bold")
    ax = c.image(fx, ay, mw, mh, d["final_map"], NAVY, cmap="inferno")
    ax.contour(d["gt"], levels=[0.5], colors=[GT], linewidths=0.6)
    c.text(fx + mw / 2, ay + mh + 0.06, r"final map $A(x^q)$", size=5.8, weight="bold")
    c.text(fx + mw / 2, ay - 0.08, "cyan: GT (display only)", size=4.8,
           color=MUTED)
    c.text((ax0 + fx + mw) / 2, 0.43,
           "CQ: fill each top-1% connected component of $A_0$ with its maximum",
           size=4.7, color=MUTED)

    # Shared colour bar for z-unit maps.
    cax = c.inset(4.95, 0.24, 1.82, 0.04)
    cax.imshow(np.linspace(0, 1, 256)[None], cmap="inferno", aspect="auto")
    for spine in cax.spines.values():
        spine.set_linewidth(0.3)
    c.text(4.95, 0.17, "0", size=4.6, ha="left")
    c.text(6.77, 0.17, f"{Z_MAX:g}", size=4.6, ha="right")
    c.text(5.86, 0.17, "z units (1 = normal q95 of that component)", size=4.6,
           color=MUTED)


def main():
    defect, support, meta = load()
    canvas = Canvas()
    draw_support_band(canvas, support, meta)
    draw_query_band(canvas, defect, meta)
    for ext in ("pdf", "svg", "png"):
        canvas.fig.savefig(OUT.with_suffix(f".{ext}"), dpi=400 if ext == "png" else None)
    print(OUT)


if __name__ == "__main__":
    main()
