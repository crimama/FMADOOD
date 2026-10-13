"""Draw the editable main-method diagram for Outline_v10.md."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent / "fig_rsg4_method_main"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(16.8, 6.45))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 16.8)
ax.set_ylim(0, 6.45)
ax.axis("off")

navy = "#16324F"
blue = "#DCEAF5"
blue_edge = "#5D8DB3"
teal = "#E0F1EC"
teal_edge = "#479C88"
orange = "#FFF0D9"
orange_edge = "#D18B2F"
muted = "#536579"


def rect(x, y, w, h, fill, edge, radius=0.16, lw=1.35):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw,
    ))


def box(x, y, w, h, title, lines, fill, edge, formula=None):
    rect(x, y, w, h, fill, edge)
    ax.text(x + w / 2, y + h - 0.34, title, ha="center", va="center",
            fontsize=13.2, fontweight="bold", color=navy)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.77 - i * 0.32, line,
                ha="center", va="center", fontsize=10.6, color=navy)
    if formula:
        ax.text(x + w / 2, y + 0.27, formula, ha="center", va="center",
                fontsize=11.0, color=navy)


def arrow(xy1, xy2, color=muted, lw=1.6, style="-"):
    ax.add_patch(FancyArrowPatch(
        xy1, xy2, arrowstyle="-|>", mutation_scale=13,
        linewidth=lw, color=color, linestyle=style,
        connectionstyle="arc3,rad=0", shrinkA=0, shrinkB=0,
    ))


def path_arrow(points, color=muted, lw=1.55, style="-"):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color,
                linewidth=lw, linestyle=style)
    arrow(points[-2], points[-1], color=color, lw=lw, style=style)


# Pale bands make the fitting/inference boundary clear at a glance.
rect(0.20, 3.54, 16.38, 2.59, "#F7FAFD", "#DBE5EE", radius=0.24, lw=0.8)
rect(0.20, 0.19, 16.38, 3.06, "#F7FBF9", "#DDEBE6", radius=0.24, lw=0.8)
ax.text(0.46, 5.91, "A  NORMAL SUPPORT · fit once", fontsize=14.5,
        fontweight="bold", color=blue_edge, va="center")
ax.text(0.46, 3.04, "B  QUERY · infer each image", fontsize=14.5,
        fontweight="bold", color=teal_edge, va="center")

x = [0.46, 3.18, 5.90, 8.62, 11.34, 14.06]
w = 2.28
top_y, top_h = 4.01, 1.63
bot_y, bot_h = 0.71, 1.91

box(x[0], top_y, w, top_h, "Normal support", ["N normal images", "no anomaly labels"], blue, blue_edge)
box(x[1], top_y, w, top_h, "Noise reference", ["Original-image noise", "median support level σS"], blue, blue_edge)
box(x[2], top_y, w, top_h, "Feature banks", ["Frozen DINOv2 ViT-L/14", "L={5,11,23}; 4 scales", "4 bags per scale"], blue, blue_edge)
box(x[3], top_y, w, top_h, "Calibration", ["Support-only NN distances", "q95: component scale c", "q99: shallow threshold τ"], blue, blue_edge)

# Support splits into noise estimation and feature-bank construction.
arrow((x[0] + w, 4.78), (x[1] - 0.10, 4.78), blue_edge)
path_arrow([(x[0] + w / 2, top_y + top_h),
            (x[0] + w / 2, 5.76),
            (x[2] + w / 2, 5.76),
            (x[2] + w / 2, top_y + top_h + 0.02)], blue_edge, 1.35)
arrow((x[2] + w, 4.78), (x[3] - 0.10, 4.78), blue_edge)

box(x[0], bot_y, w, bot_h, "Query image", ["Unlabeled test image", "original H × W"], teal, teal_edge)
box(x[1], bot_y, w, bot_h, "Resolution route", ["Noise ratio ρ = σq / σS", "ρ > 9: 448, 672", "else: all 4 scales"], teal, teal_edge)
box(x[2], bot_y, w, bot_h, "1-NN matching", ["Frozen DINOv2 features", "cosine distance to bank", "per layer × scale × bag"], teal, teal_edge)
box(x[3], bot_y, w, bot_h, "Asymmetric fusion", ["q95-normalize distances", "deep = (z11 + z23) / 2", "shallow excess only"], orange, orange_edge,
    r"$S=\mathrm{deep}+[z_5-\tau_5]_+$")
box(x[4], bot_y, w, bot_h, "Spatial average", ["Bilinear upsample", "mean over 4 bags", "mean selected scales"], teal, teal_edge,
    r"$A_0\in\mathbb{R}^{H\times W}$")
box(x[5], bot_y, w, bot_h, "Final map", ["CQ: top 1% mask", "8-connected components", "fill with component max"], teal, teal_edge,
    r"$A(x^q)$")

for i in range(5):
    arrow((x[i] + w + 0.08, 1.67), (x[i + 1] - 0.09, 1.67),
          orange_edge if i == 3 else teal_edge)

# Dashed links denote support-derived quantities reused without query updating.
for i, label in [(1, "fixed σS"), (2, "fixed banks"), (3, "fixed c, τ")]:
    arrow((x[i] + w / 2, top_y - 0.06),
          (x[i] + w / 2, bot_y + bot_h + 0.08), blue_edge, 1.4, "--")
    ax.text(x[i] + w / 2 + 0.11, 3.35, label, fontsize=9.2,
            color=blue_edge, va="center")

ax.text(8.45, 0.35,
        "m = (layer, resolution, bag)     ·     Calibration is fitted on normal support only     ·     CQ is applied to the query map",
        ha="center", va="center", color=muted, fontsize=9.9)

fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.99)
for ext in ("svg", "pdf", "png"):
    fig.savefig(OUT.with_suffix(f".{ext}"), dpi=220, facecolor="white")
plt.close(fig)
