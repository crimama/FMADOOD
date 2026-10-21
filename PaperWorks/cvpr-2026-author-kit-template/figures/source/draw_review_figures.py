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

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})
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
    ax.text(.02, .97, "Few normal supports", va="top", weight="bold", fontsize=8)
    ax.text(.60, .97, "Queries", va="top", weight="bold", fontsize=8)

    # Real MVTec AD 2 walnuts images: two supports from the fixed 16-shot
    # manifest (regular lighting) and overexposed test queries. GT is drawn
    # only on the defect tile for display.
    data_root = Path("/home/hun/Volume/DATA/mvtec_ad_2/walnuts")
    from PIL import Image

    def load(rel, width=320):
        im = Image.open(data_root / rel).convert("RGB")
        return np.asarray(im.resize((width, round(width * im.height / im.width)), Image.LANCZOS))

    def tile(x, y, rel, mask=None, w=.14):
        img = load(rel)
        h = w * 3.45 * img.shape[0] / img.shape[1] / fig.get_figheight()
        tax = fig.add_axes([x, y, w, h])
        tax.imshow(img, interpolation="lanczos")
        if mask is not None:
            m = Image.open(data_root / mask).convert("L").resize(img.shape[1::-1], Image.NEAREST)
            tax.contour(np.asarray(m) > 0, [.5], colors=[RED], linewidths=.7)
        tax.set_xticks([])
        tax.set_yticks([])
        for spine in tax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(GRAY)
            spine.set_linewidth(.5)
    tile(.03, .725, "train/good/129_regular.png")
    tile(.19, .725, "train/good/266_regular.png")
    ax.text(.19, .675, "Limited coverage", ha="center", fontsize=6.7)
    tile(.55, .725, "test_public/good/000_overexposed.png")
    tile(.80, .725, "test_public/bad/004_overexposed.png",
         mask="test_public/ground_truth/bad/004_overexposed_mask.png")
    ax.text(.62, .675, "Shifted normal", ha="center", fontsize=6.7)
    ax.text(.87, .675, "Defect", ha="center", fontsize=6.7)
    ax.annotate("", xy=(.525, .845), xytext=(.355, .845),
                arrowprops=dict(arrowstyle="->", lw=.8, color=GRAY))
    ax.text(.44, .735, "Lighting change\n(overexposed)", ha="center", fontsize=5.6, color=GRAY)

    plot = fig.add_axes([.10, .15, .86, .40])
    x = np.linspace(0, 3.3, 300)
    curves = [
        (.55, .27, "#B5BBC3", "Support-like normal", .30),
        (1.70, .43, NORMAL_SHIFT, "Shifted normal", .14),
        (2.10, .43, RED, "Defect", .08),
    ]
    for mean, sd, color, label, alpha in curves:
        y = np.exp(-.5*((x-mean)/sd)**2)
        plot.plot(x, y, color=color, lw=1.1, label=label)
        plot.fill_between(x, 0, y, color=color, alpha=alpha)
    # A low-FPR threshold must sit above most shifted-normal scores, which
    # leaves only the right tail of the defect distribution detected.
    t = 2.45
    xd = x[x >= t]
    plot.fill_between(xd, 0, np.exp(-.5*((xd-2.10)/.43)**2), color=RED, alpha=.35, lw=0)
    plot.axvline(t, color="#22282D", lw=.8, ls=(0, (3, 2)))
    plot.text(t+.05, 1.12, "Low-FPR\nthreshold", ha="left", va="top", fontsize=6.2)
    plot.text(1.45, 1.30, "Overlapping high scores", ha="center", va="top", fontsize=6.2, color=GRAY)
    plot.annotate("detected\ndefect", xy=(2.62, .18), xytext=(3.02, .42), fontsize=5.8,
                  color=RED, ha="center", arrowprops=dict(arrowstyle="-", lw=.5, color=RED))
    plot.set(xlim=(0, 3.3), ylim=(0, 1.35), yticks=[], xticks=[])
    plot.spines["left"].set_visible(False)
    plot.set_xlabel("Matching distance →", fontsize=7, labelpad=1)
    plot.legend(loc="lower center", bbox_to_anchor=(.5, 1.03),
                ncol=3, fontsize=5.7, frameon=False, handlelength=1,
                handletextpad=.3, columnspacing=.7)
    ax.text(.5, .015, "Conceptual distributions; not measured", ha="center",
            fontsize=6.3, color=GRAY)
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
               labels=["No calibration", "Q95", "Uniform (25%)"], loc="upper center",
               bbox_to_anchor=(.55, 1.01), ncol=3, frameon=False, fontsize=6.6)
    fig.text(.60, .025, "N: normal   S: small defects (<0.1% area)",
             ha="center", fontsize=5.9, color=GRAY)
    save(fig, "fig_scopead_order_rank_diagnostic")


def calibration():
    data = json.loads((HERE / "calibration_quantiles.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75))
    fig.subplots_adjust(left=.13, right=.98, bottom=.25, top=.70, wspace=.58)
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

    axes[0].plot([], [], "s", color=BASE, ms=3.5, label="Before")
    axes[0].plot([], [], "s", color=BLUE, ms=3.5, label="After Q95")
    axes[0].set(ylim=(.92, 1.88), yticks=[1.0, 1.25, 1.5, 1.75], xlim=(-.7, 11.7),
                xticks=[1, 4, 7, 10], xticklabels=["448", "672", "896", "1120"])
    axes[0].set_title("(a) Vial, 2-shot", fontsize=7.3)
    axes[0].set_ylabel("Relative normal scale", fontsize=6.5)
    axes[0].set_xlabel("Resolution  (▲ L5  ● L11  ■ L23)", fontsize=5.8)
    axes[0].text(.5, .92, "Spread: 1.60x → 1.24x", transform=axes[0].transAxes,
                 ha="center", fontsize=5.8, color="#1F4E6B", weight="bold")
    for x in (2.5, 5.5, 8.5):
        axes[0].axvline(x, color="#E2E5E8", lw=.5, zorder=0)
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, 1.42), ncol=2,
                   fontsize=5.9, frameon=False, columnspacing=.6, handletextpad=.2)

    axes[1].fill_between([1, 2.9], [1, 1], [1, 2.9], color="#F3F6F9", zorder=0)
    axes[1].plot([1, 2.9], [1, 2.9], "--", color="#9DA5AE", lw=.7)
    axes[1].axhline(1.0, color="#CCD1D9", lw=.6, ls=":", zorder=1)
    for shot, marker, face in [(2, "o", BLUE), (16, "^", "white")]:
        rows = [r for r in data["spread"] if int(r["shot"]) == shot]
        axes[1].scatter([float(r["raw_normal_q95_ratio"]) for r in rows],
                        [float(r["calibrated_normal_q95_ratio"]) for r in rows],
                        facecolors=face, edgecolors=BLUE, marker=marker, s=13,
                        linewidths=.7, label=f"{shot}-shot", zorder=3)
        for r in rows:
            if r["ratio_reduced"] == "False":
                x, y = float(r["raw_normal_q95_ratio"]), float(r["calibrated_normal_q95_ratio"])
                axes[1].annotate(f"{r['class'].replace('_', ' ')} ({shot})", (x, y),
                                 xytext=(4, 3) if shot == 2 else (5, -1), textcoords="offset points",
                                 fontsize=5.4, color=GRAY, ha="left", va="bottom" if shot == 2 else "center")
    axes[1].set(xlim=(.95, 2.9), ylim=(.95, 2.9), xticks=[1, 2], yticks=[1, 2])
    axes[1].set_title("(b) 8 classes × 2 shots", fontsize=7.3)
    axes[1].set_xlabel("Discrepancy before (max/min)", fontsize=6.0)
    axes[1].set_ylabel("Discrepancy after (max/min)", fontsize=6.0)
    axes[1].text(.97, .94, "14/16 reduced\n(below y = x)", ha="right", va="top",
                 transform=axes[1].transAxes, fontsize=5.8)
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, 1.42), ncol=2,
                   fontsize=5.9, frameon=False, columnspacing=.6, handletextpad=.1)
    for ax in axes:
        ax.tick_params(labelsize=5.8, pad=2)
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
    "16-shot": dict(single=[52.01, 55.01, 55.12, 57.48], ours=60.16),
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
