"""Reproduce the conceptual introduction and measured rank diagnostic.

The introduction is schematic. Rank shares come from the copied diagnostic
summary (AD2 public, seed 0, 8-class macro), not newly executed experiments.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
from figure_style import use_times_new_roman

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
plt.rcParams.update({
    "font.size": 7,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})
use_times_new_roman()
# Shared palette for comparison figures: baseline/before = BASE, ours/after = BLUE.
# Orange is reserved for the shallow layer in the overview figure.
BLUE, ORANGE, GRAY = "#28638A", "#BA5C32", "#737C87"
BASE = "#ACB6C2"
RED = "#B03A2E"
NORMAL_SHIFT = "#5F6670"


def save(fig, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def introduction():
    fig = plt.figure(figsize=(3.45, 2.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.18, .98, "Normal supports", ha="center", va="top",
            weight="bold", fontsize=8)
    ax.text(.73, .98, "Queries: changed lighting", ha="center", va="top",
            weight="bold", fontsize=8)

    # Real samples illustrate acquisition conditions, not measured scores.
    # The two curves below describe patch distances schematically.
    data_root = Path("/home/hun/Volume/DATA/mvtec_ad_2/walnuts")
    from PIL import Image

    def tile(x, y, rel, color=GRAY, mask=None, w=.15):
        im = Image.open(data_root / rel).convert("RGB")
        img = np.asarray(im.resize(
            (320, round(320 * im.height / im.width)), Image.LANCZOS))
        h = w * fig.get_figwidth() * img.shape[0] / img.shape[1] / fig.get_figheight()
        tax = fig.add_axes([x, y, w, h])
        tax.imshow(img, interpolation="lanczos")
        if mask is not None:
            m = Image.open(data_root / mask).convert("L").resize(img.shape[1::-1], Image.NEAREST)
            tax.contour(np.asarray(m) > 0, [.5], colors=[RED], linewidths=.7)
        tax.set_xticks([])
        tax.set_yticks([])
        for spine in tax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(.8)

    tile(.01, .73, "train/good/129_regular.png")
    tile(.19, .73, "train/good/266_regular.png")
    ax.text(.18, .68, "Regular lighting", ha="center", fontsize=7)
    tile(.53, .73, "test_public/good/000_overexposed.png", color=BLUE)
    tile(.80, .73, "test_public/bad/004_overexposed.png", color=RED,
         mask="test_public/ground_truth/bad/004_overexposed_mask.png")
    ax.text(.61, .68, "Normal", color=BLUE, ha="center", fontsize=7.5)
    ax.text(.88, .68, "Defect", color=RED, ha="center", fontsize=7.5)
    ax.annotate("", xy=(.505, .83), xytext=(.375, .83),
                arrowprops=dict(arrowstyle="->", lw=.8, color=GRAY))

    plot = fig.add_axes([.08, .20, .87, .40])
    x = np.linspace(0, 3.3, 400)
    # No fitted distributions or numerical threshold are implied.
    for mean, color in [(1.65, BLUE), (2.12, RED)]:
        y = np.exp(-.5 * ((x - mean) / .46) ** 2)
        plot.plot(x, y, color=color, lw=1.3)
        plot.fill_between(x, 0, y, color=color, alpha=.10, lw=0)
    plot.text(1.43, 1.08, "Shifted normal", color=BLUE,
              ha="right", fontsize=7.2)
    plot.annotate("", xy=(1.62, 1.0), xytext=(1.40, 1.09),
                  arrowprops=dict(arrowstyle="-", lw=.6, color=BLUE))
    plot.text(2.34, 1.08, "Defect", color=RED,
              ha="left", fontsize=7.2)
    plot.annotate("", xy=(2.15, 1.0), xytext=(2.36, 1.09),
                  arrowprops=dict(arrowstyle="-", lw=.6, color=RED))
    plot.set(xlim=(0, 3.3), ylim=(0, 1.32), yticks=[], xticks=[])
    plot.spines["left"].set_visible(False)
    plot.spines["bottom"].set_linewidth(.7)
    plot.set_xlabel("Patch matching distance →", fontsize=7.2, labelpad=2)
    ax.text(.5, .032, "High distance alone does not identify a defect.",
            ha="center", fontsize=8, weight="bold")
    save(fig, "fig_intro_ambiguity")


def rank_diagnostic():
    data = json.loads((HERE / "order_rank_diagnostic.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.80), sharey=True)
    fig.subplots_adjust(left=.15, right=.985, bottom=.27, top=.72, wspace=.22)
    for ax in axes:
        ax.set_ylim(0, 62)
        ax.set_yticks([0, 20, 40, 60])
        ax.tick_params(axis="both", labelsize=6)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E4E7EA", lw=.5)
    groups = [
        [(2, "normal_top1", 0), (16, "normal_top1", 0)],
        [(2, "normal_top1", 3), (2, "size_top1", 3),
         (16, "normal_top1", 3), (16, "size_top1", 3)],
    ]
    for ax, specs in zip(axes, groups):
        for j, (shot, field, index) in enumerate(specs):
            for offset, norm, color in [(-.18, "none", BASE), (.18, "q95", BLUE)]:
                v = data[f"{norm}_s{shot:02d}"][field]
                value = (v[0][index] if field == "size_top1" else v[index])*100
                ax.bar(j+offset, value, width=.34, color=color, zorder=3)
                ax.text(j+offset, value+1.1, f"{value:.1f}", ha="center",
                        va="bottom", fontsize=5.5, rotation=90)
        ax.set_xticks(range(len(specs)))
        ax.axhline(25, color="#22282D", lw=.6, ls=(0, (3, 2)), zorder=2.5)
    axes[0].set_xticklabels(["2-shot", "16-shot"])
    axes[0].set_title("(a) 448 px\nNormal pixels", fontsize=7.1, pad=5)
    axes[0].set_ylabel("Top-1 share (%)", fontsize=7, labelpad=3)
    axes[1].set_title("(b) 1120 px\nNormal vs. small defects", fontsize=7.1, pad=5)
    axes[1].set_xticklabels(["N", "S", "N", "S"])
    axes[1].text(.5, -.34, "2-shot", transform=axes[1].get_xaxis_transform(),
                 ha="center", fontsize=6)
    axes[1].text(2.5, -.34, "16-shot", transform=axes[1].get_xaxis_transform(),
                 ha="center", fontsize=6)
    axes[1].axvline(1.5, color="#D8DCE0", lw=.6)
    fig.legend(handles=[Rectangle((0,0),1,1,fc=BASE), Rectangle((0,0),1,1,fc=BLUE),
                        plt.Line2D([], [], color="#22282D", lw=.6, ls=(0, (3, 2)))],
               labels=["No calibration", "Q99", "Uniform (25%)"], loc="upper center",
               bbox_to_anchor=(.55, 1.01), ncol=3, frameon=False, fontsize=6.6)
    fig.text(.60, .025, "N: normal   S: small defects (<0.1% area)",
             ha="center", fontsize=5.9, color=GRAY)
    save(fig, "fig_scopead_order_rank_diagnostic")


def calibration():
    data = json.loads((HERE / "calibration_quantiles.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.05))
    fig.subplots_adjust(left=.13, right=.98, bottom=.25, top=.77, wspace=.64)
    markers = {"5": "^", "11": "o", "23": "s"}

    raw_rows = [r for r in data["example"] if r["stage"] == "raw"]
    raw_rows.sort(key=lambda r: (int(r["resolution"]), int(r["layer"])))
    raw_q = np.array([float(r["q95"]) for r in raw_rows])
    raw_vals = raw_q / raw_q.min()

    cal_rows = [r for r in data["example"] if r["stage"] == "calibrated"]
    cal_rows.sort(key=lambda r: (int(r["resolution"]), int(r["layer"])))
    cal_q = np.array([float(r["q95"]) for r in cal_rows])
    cal_vals = cal_q / cal_q.min()

    for i in range(len(raw_rows)):
        axes[0].plot([i, i], [cal_vals[i], raw_vals[i]], color="#D8DCE0", lw=0.7, zorder=1)
        axes[0].plot(i, raw_vals[i], markers[raw_rows[i]["layer"]], color=BASE, ms=3.5,
                     mec="white", mew=.3, zorder=2)
        axes[0].plot(i, cal_vals[i], markers[cal_rows[i]["layer"]], color=BLUE, ms=3.5,
                     mec=BLUE, mew=.3, zorder=3)

    axes[0].set(ylim=(.94, 1.88), yticks=[1.0, 1.25, 1.5, 1.75], xlim=(-.7, 11.7),
                xticks=[1, 4, 7, 10], xticklabels=["448", "672", "896", "1120"])
    axes[0].set_title("(a) Vial, 2-shot", fontsize=7.5, pad=6)
    axes[0].set_ylabel("Relative normal scale", fontsize=7)
    axes[0].set_xlabel("Input resolution (px)", fontsize=7, labelpad=3)
    axes[0].text(.5, .96, f"{raw_vals.max():.2f}× → {cal_vals.max():.2f}×",
                 transform=axes[0].transAxes, ha="center", va="top",
                 fontsize=7, color=BLUE)
    axes[0].text(.5, -.40, "▲ L5   ● L11   ■ L23",
                 transform=axes[0].transAxes, ha="center", fontsize=6.4)
    for x in (2.5, 5.5, 8.5):
        axes[0].axvline(x, color="#E2E5E8", lw=.5, zorder=0)

    # One line per class/shot setting. Endpoints use exactly the same ratios
    # as the former before-versus-after scatter; no aggregation or jitter.
    rows = data["spread"]
    before = np.array([float(r["raw_normal_q95_ratio"]) for r in rows])
    after = np.array([float(r["calibrated_normal_q95_ratio"]) for r in rows])
    for v0, v1 in zip(before, after):
        axes[1].plot([0, 1], [v0, v1], color="#B8C0C8", lw=.65, zorder=1)
    axes[1].scatter(np.zeros(len(rows)), before, color=BASE, s=14,
                    edgecolors="white", linewidths=.3, zorder=3)
    axes[1].scatter(np.ones(len(rows)), after, color=BLUE, s=14,
                    edgecolors="white", linewidths=.3, zorder=3)
    axes[1].set(xlim=(-.22, 1.22), ylim=(.94, 3.03),
                xticks=[0, 1], xticklabels=["Before", "After Q99"],
                yticks=[1, 1.5, 2, 2.5])
    axes[1].set_title("(b) Across 16 settings", fontsize=7.5, pad=6)
    axes[1].set_ylabel("Normal scale spread (max/min)", fontsize=7, labelpad=3)
    n_reduced = int(np.count_nonzero(after < before))
    axes[1].text(.5, .96, f"{n_reduced}/{len(rows)} decrease",
                 transform=axes[1].transAxes, ha="center", va="top",
                 fontsize=7, color=BLUE)
    axes[1].text(.5, -.40, "8 classes × 2/16-shot",
                 transform=axes[1].transAxes, ha="center", fontsize=6.4)
    axes[1].axhline(1, color="#919BA5", lw=.6, ls=(0, (2, 2)), zorder=0)
    fig.legend(handles=[
        plt.Line2D([], [], marker="o", color=BASE, ls="none", ms=4),
        plt.Line2D([], [], marker="o", color=BLUE, ls="none", ms=4)],
        labels=["Before", "After Q99"], loc="upper center",
        bbox_to_anchor=(.55, 1.01), ncol=2, frameon=False,
        fontsize=7, columnspacing=1.2, handletextpad=.3)
    for ax in axes:
        ax.tick_params(labelsize=6.4, pad=2)
        ax.grid(axis="y", color="#E5E8EA", lw=.5)
        ax.set_axisbelow(True)
    save(fig, "fig_scopead_calibration_paper")

# AD2 test_public, seed 0, CQ99, AUPRO@0.05 (%). Single resolutions apply the
# full method (separate banks, lambda=0.125, Q95) to one input resolution.
# Sources: results/remote_runs/dsba3/scarf_single_resolution_fp16_20261012/comparison.md
# (448/672/896 and 2-shot 1120), scopead_final_cost_20261019 lane_2 (16-shot 1120,
# same configuration; identical to Table 4) and scopead_renyi_cq_20261018 (4-res Renyi).
RESOLUTION_AUPRO = {
    "2-shot": dict(single=[51.14, 54.38, 52.65, 54.27], ours=59.92),
    "16-shot": dict(single=[52.01, 55.01, 55.12, 57.48], ours=60.19),
}


def resolution():
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.55), sharey=True)
    fig.subplots_adjust(left=.13, right=.985, bottom=.24, top=.86, wspace=.10)
    labels = ["448", "672", "896", "1120", "Ours\n(4-res)"]
    for k, (ax, (shot, d)) in enumerate(zip(axes, RESOLUTION_AUPRO.items())):
        values = d["single"] + [d["ours"]]
        x = [0, 1, 2, 3, 4.35]
        colors = [BASE] * 4 + [BLUE]
        ax.bar(x, [v - 48 for v in values], bottom=48, width=.62, color=colors, zorder=3)
        for xi, v, c in zip(x, values, colors):
            ax.text(xi, v + .25, f"{v:.2f}", ha="center", va="bottom", fontsize=5.4,
                    weight="bold" if c == BLUE else "normal", zorder=4)
        best = max(d["single"])
        ax.hlines(best, -.45, 4.7, color=GRAY, lw=.6, ls=(0, (3, 2)), zorder=2)
        ax.text(4.35, d["ours"] + 1.35, f"+{d['ours'] - best:.2f}", ha="center",
                va="bottom", fontsize=5.6, color=BLUE, zorder=4)
        ax.set_title(f"({'ab'[k]}) {shot}", fontsize=7.2, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5.8)
        ax.set_xlim(-.6, 4.95)
        ax.tick_params(axis="y", labelsize=5.8)
        ax.grid(axis="y", color="#E5E8EA", lw=.5)
        ax.set_axisbelow(True)
    axes[0].set(ylim=(48, 63.5), yticks=[48, 52, 56, 60])
    axes[0].set_ylabel("AUPRO$_{0.05}$ (%)", fontsize=6.6)
    save(fig, "fig_scopead_resolution_comparison")


STRATA_ARMS = [("none_renyi", "None + Rényi", "#C9D1D9"),
               ("q95_mean", "Q95 + Mean", "#7FA3BF"),
               ("q95_renyi", "Q95 + Rényi (Ours)", BLUE)]


def lighting_figure():
    """Official per-condition AUPRO@0.05 (full resolution, CQ99, seed 0)."""
    data = json.loads((HERE / "lighting_strata.json").read_text())["aupro"]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.34), sharey=True)
    fig.subplots_adjust(left=.13, right=.985, bottom=.17, top=.75, wspace=.10)
    macro = {}
    for ax, shot in zip(axes, ("02", "16")):
        for k, (arm, label, color) in enumerate(STRATA_ARMS):
            per = data[f"{shot}_{arm}"]
            reg = 100 * np.mean([per[o]["regular"] for o in per])
            chg = 100 * np.mean([np.mean([v for c, v in per[o].items() if c != "regular"])
                                 for o in per])
            macro[f"{shot}_{arm}"] = (reg, chg)
            dx = (k - 1) * .09
            ax.plot([dx, 1 + dx], [reg, chg], color=color, lw=.9, zorder=2)
            ax.plot([dx, 1 + dx], [reg, chg], "o", color=color, ms=4.0, zorder=3,
                    mec="#22282D" if arm == "q95_renyi" else color, mew=.4,
                    label=label if shot == "02" else None)
            if arm == "q95_renyi":
                for x, v, ha in ((dx - .07, reg, "right"), (1 + dx + .07, chg, "left")):
                    ax.text(x, v, f"{v:.2f}", ha=ha, va="center", fontsize=5.4, weight="bold")
        ax.set_title(f"({'ab'[shot == '16']}) {int(shot)}-shot", fontsize=7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Regular", "Changed lighting"], fontsize=5.9)
        ax.set_xlim(-.45, 1.55)
        ax.tick_params(axis="y", labelsize=5.8, pad=1.5)
        ax.grid(axis="y", color="#E5E8EA", lw=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("AUPRO@0.05 (%)", fontsize=6.4)
    axes[0].set_ylim(58.2, 62.6)
    fig.legend(loc="upper center", bbox_to_anchor=(.56, 1.0), ncol=3, fontsize=5.5,
               frameon=False, handlelength=1.2, columnspacing=.8, handletextpad=.3)
    save(fig, "fig_scopead_lighting_strata")
    return macro


if __name__ == "__main__":
    introduction()
    rank_diagnostic()
    calibration()
    resolution()
    print(json.dumps(lighting_figure(), indent=1))
