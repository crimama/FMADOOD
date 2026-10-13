"""Draw the RSG4 mechanism-intuition figures from cached real intermediate maps.

Run extract_rsg4_mechanism_maps.py first (GPU), then:
    .venv/bin/python PaperWorks/figures/plot_rsg4_mechanism.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache_rsg4_mechanism"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.titlesize": 10, "axes.titleweight": "bold",
})
CMAP = "inferno"
NAVY, TEAL, ORANGE, GREY = "#16324F", "#2A9D8F", "#E76F51", "#8A9BA8"


def load(name):
    data = np.load(CACHE / f"{name}.npz")
    return {k: data[k].astype(np.float32) if data[k].dtype == np.float16 else data[k]
            for k in data.files}


def meta(case):
    return json.loads((CACHE / f"{case}__meta.json").read_text())


def show(ax, image, title, *, vmin=None, vmax=None, gt=None, cmap=CMAP, label=None):
    handle = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    if gt is not None and gt.any():
        ax.contour(gt, levels=[0.5], colors="#3CF0FF", linewidths=0.9)
    ax.set_title(title, pad=4)
    ax.set_xticks([]), ax.set_yticks([])
    if label:
        ax.text(0.03, 0.04, label, transform=ax.transAxes, color="white", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.55, lw=0))
    return handle


def tpr_at_normal_q(normal, defect, q=0.99):
    """Share of defect pixels above the in-image normal q-quantile (GT used post hoc)."""
    threshold = np.quantile(normal, q)
    return float((defect > threshold).mean()), float(threshold)


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# --------------------------------------------------------------------------
# Fig. M1  Scale calibration: same query, raw distance vs. support-normalised
# --------------------------------------------------------------------------
def fig_scale(case="ad_hazelnut", tag="clean", res=1120):
    d = load(f"{case}__{tag}")
    m = meta(case)
    gt = d["gt"]
    raw5, raw23 = d[f"r{res}_raw_l5"], d[f"r{res}_raw_l23"]
    z5, z23 = d[f"r{res}_normalized_l5"], d[f"r{res}_normalized_l23"]
    c5 = np.mean([s[0] for s in m["scales"][str(res)]])
    c23 = np.mean([s[2] for s in m["scales"][str(res)]])
    # An illustrative raw score between the two support q95 thresholds makes
    # the ambiguity concrete without implying that both layers observed it
    # at the same query pixel.
    d_star = 0.5 * (c5 + c23)
    z5_star, z23_star = d_star / c5, d_star / c23

    fig = plt.figure(figsize=(13.2, 5.9))
    grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.09, 1.65], wspace=0.30, hspace=0.34)
    show(fig.add_subplot(grid[:, 0]), d["rgb"], "Query + GT", gt=gt)
    raw_max = max(float(np.quantile(np.concatenate([raw5.ravel(), raw23.ravel()]), 0.999)),
                  c5, c23, d_star)
    show(fig.add_subplot(grid[0, 1]), raw5, "Raw distance  $d_5$", vmin=0, vmax=raw_max, gt=gt,
         label=f"$c_5$ = {c5:.3f}")
    h = show(fig.add_subplot(grid[0, 2]), raw23, "Raw distance  $d_{23}$", vmin=0, vmax=raw_max,
             gt=gt, label=f"$c_{{23}}$ = {c23:.3f}")
    cb = fig.colorbar(h, cax=fig.add_subplot(grid[0, 3])); cb.ax.tick_params(labelsize=7)
    z_max = max(float(np.quantile(np.concatenate([z5.ravel(), z23.ravel()]), 0.999)),
                z5_star, z23_star, 1.0)
    show(fig.add_subplot(grid[1, 1]), z5, "Normalised  $z_5 = d_5 / c_5$", vmin=0, vmax=z_max, gt=gt)
    h = show(fig.add_subplot(grid[1, 2]), z23, "Normalised  $z_{23} = d_{23} / c_{23}$",
             vmin=0, vmax=z_max, gt=gt)
    cb = fig.colorbar(h, cax=fig.add_subplot(grid[1, 3])); cb.ax.tick_params(labelsize=7)

    ax = fig.add_subplot(grid[0, 4])
    bins = np.linspace(0, raw_max, 70)
    for values, colour, name in ((raw5[~gt], TEAL, "$d_5$"), (raw23[~gt], ORANGE, "$d_{23}$")):
        ax.hist(values, bins=bins, color=colour, alpha=0.55, density=True, label=f"{name} normal px")
    ax.axvline(c5, color=TEAL, ls="--", lw=1.2, ymin=0, ymax=0.78)
    ax.axvline(c23, color=ORANGE, ls="--", lw=1.2, ymin=0, ymax=0.78)
    ax.axvline(d_star, color=NAVY, ls=":", lw=1.8, ymin=0, ymax=0.72)
    ax.text(0.025, 0.96,
            f"Example: $c_5={c5:.3f} < d^*={d_star:.3f} < c_{{23}}={c23:.3f}$\n"
            f"$z_5={z5_star:.2f}>1$, but $z_{{23}}={z23_star:.2f}<1$",
            transform=ax.transAxes, va="top", fontsize=8, color=NAVY,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CBD5E1", alpha=1.0, lw=0.6))
    ax.set_title("Unscaled: one score, conflicting readings"), ax.set_yticks([])
    ax.set_xlabel("Raw cosine distance $d$", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(grid[1, 4])
    bins = np.linspace(0, z_max, 70)
    for values, colour, name in ((z5[~gt], TEAL, "$z_5$"), (z23[~gt], ORANGE, "$z_{23}$")):
        ax.hist(values, bins=bins, color=colour, alpha=0.55, density=True, label=f"{name} normal px")
    ax.axvline(z5_star, color=TEAL, ls=":", lw=1.5, ymin=0, ymax=0.72)
    ax.axvline(z23_star, color=ORANGE, ls=":", lw=1.5, ymin=0, ymax=0.72)
    ax.axvline(1.0, color=NAVY, ls="--", lw=1.2)
    ax.text(1.02, 0.92, "z = 1  (support q95)", transform=ax.get_xaxis_transform(), fontsize=8, color=NAVY)
    ax.text(0.025, 0.96,
            f"Same $d^*$: $z_5={z5_star:.2f}$ (>1), $z_{{23}}={z23_star:.2f}$ (<1)",
            transform=ax.transAxes, va="top", fontsize=7.5, color=NAVY,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#CBD5E1", alpha=1.0, lw=0.6))
    ax.set_title("Scaled: both scores use the same q95 unit"), ax.set_yticks([])
    ax.set_xlabel("Distance in support-q95 units ($z$)", fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Scale calibration on a real query ({case.split('_')[1]}, {res} px, bag mean). "
                 "The marked $d^*$ is illustrative, not a paired query pixel.",
                 x=0.5, y=0.995, fontsize=10)
    save(fig, "fig_rsg4_mech_scale")


# --------------------------------------------------------------------------
# Fig. M2  Asymmetric residual fusion: deep base, shallow, gate, fused maps
# --------------------------------------------------------------------------
def fig_fusion(case="ad_hazelnut", tags=("clean",), res=1120):
    d = load(f"{case}__{tags[0]}")
    m = meta(case)
    gt = d["gt"]
    tau = float(np.mean(m["taus"][str(res)]))
    deep, z5 = d[f"r{res}_deep"], d[f"r{res}_normalized_l5"]
    residual = d[f"r{res}_shallow_residual"]
    all_add, gated = deep + z5, deep + residual

    fig = plt.figure(figsize=(13.2, 3.3))
    grid = fig.add_gridspec(1, 5, wspace=0.1)
    show(fig.add_subplot(grid[0, 0]), d["rgb"], "Query + GT", gt=gt)
    vmax = float(np.quantile(all_add, 0.999))
    show(fig.add_subplot(grid[0, 1]), deep, "Deep base  $z_d$", vmin=0, vmax=vmax, gt=gt)
    show(fig.add_subplot(grid[0, 2]), z5, "Shallow  $z_5$  (all of it)", vmin=0, vmax=vmax, gt=gt,
         label=f"$z_5$ > 1 on {100 * (z5 > 1).mean():.1f}% px")
    show(fig.add_subplot(grid[0, 3]), residual, f"Excess  $[z_5-\\tau_5]_+$,  $\\tau_5$={tau:.2f}",
         vmin=0, vmax=vmax, gt=gt, label=f"> 0.05 on {100 * (residual > 0.05).mean():.1f}% px")
    show(fig.add_subplot(grid[0, 4]), gated, "SCARF  $z_d + [z_5-\\tau_5]_+$", vmin=0, vmax=vmax, gt=gt)
    fig.suptitle(f"Cached component-level fusion example ({case.split('_')[1]}, {res} px, bag mean, "
                 "same colour scale). Ground truth is overlaid for reference; this single example is illustrative.",
                 y=1.04, fontsize=10)
    save(fig, "fig_rsg4_mech_fusion")


# --------------------------------------------------------------------------
# Fig. M3  Resolution reliability: per-resolution maps and the noise guard
# --------------------------------------------------------------------------
def fig_resolution(case="ad_hazelnut"):
    clean, noisy, off = (load(f"{case}__{t}") for t in ("clean", "noisy", "noisy_guardoff"))
    # 1%-FPR threshold from a defect-free noisy query scored the same way (GT-free).
    good = {"off": load(f"{case}__good_noisy_guardoff")["final_map"],
            "on": load(f"{case}__good_noisy")["final_map"]}
    m = meta(case)
    gt = clean["gt"]
    fig = plt.figure(figsize=(13.2, 6.0))
    grid = fig.add_gridspec(2, 5, wspace=0.08, hspace=0.3)
    vmax = float(np.quantile(clean["raw_map"], 0.999))
    show(fig.add_subplot(grid[0, 0]), clean["rgb"], "Clean query", gt=gt,
         label=f"ρ = {m['clean']['noise_ratio']:.2f}")
    for column, res in enumerate((448, 672, 896, 1120)):
        fused = clean[f"r{res}_deep"] + clean[f"r{res}_shallow_residual"]
        show(fig.add_subplot(grid[0, column + 1]), fused, f"clean · {res} px", vmin=0, vmax=vmax,
             gt=gt, label=f"{res // 14}×{res // 14} tokens")
    show(fig.add_subplot(grid[1, 0]), noisy["rgb"], "Noisy query (same image)", gt=gt,
         label=f"ρ = {m['noisy']['noise_ratio']:.1f} > κ = 9")
    for column, res in enumerate((448, 1120)):
        fused = off[f"r{res}_deep"] + off[f"r{res}_shallow_residual"]
        show(fig.add_subplot(grid[1, column + 1]), fused, f"noisy · {res} px", vmin=0, vmax=vmax, gt=gt)
    panels = ((off["final_map"], "Guard OFF (4 res.)", "off"), (noisy["final_map"], "Guard ON (448+672)", "on"))
    for column, (value, title, key) in enumerate(panels):
        ax = fig.add_subplot(grid[1, column + 3])
        threshold = float(np.quantile(good[key], 0.99))
        show(ax, value, title, vmin=0, vmax=vmax, gt=gt,
             label=f"defect hit @1% FPR: {100 * (value[gt] > threshold).mean():.0f}%")
    fig.suptitle("Resolution reliability. Top: high resolution adds small-detail resolution on a clean query. "
                 "Bottom: with sensor noise, 896/1120 tokens light up the whole image;\nthe support-referenced "
                 "noise ratio ρ = σ(query)/median σ(support) detects this and keeps only 448/672.\n"
                 "1% FPR threshold = q99 of a defect-free noisy query scored the same way.",
                 y=1.02, fontsize=10)
    save(fig, "fig_rsg4_mech_resolution")


# --------------------------------------------------------------------------
# Bagging-only view used by the current M1 discussion.
# Keep fig_bag_cq below for outline drafts that cite the original combined plate.
# --------------------------------------------------------------------------
def fig_bagging(case="ad_hazelnut", tag="clean", res=1120):
    d = load(f"{case}__{tag}")
    gt = d["gt"]
    bags = [d[f"r{res}_bag{b}"] for b in range(4)]
    mean = np.mean(bags, axis=0)
    spread = np.std(bags, axis=0)

    fig = plt.figure(figsize=(13.2, 3.2))
    grid = fig.add_gridspec(1, 6, wspace=0.08)
    vmax = float(np.quantile(mean, 0.999))
    for b in range(4):
        show(fig.add_subplot(grid[0, b]), bags[b], f"bag {b + 1}  (256 tok/img)",
             vmin=0, vmax=vmax, gt=gt)
    show(fig.add_subplot(grid[0, 4]), mean, "mean of 4 bags", vmin=0, vmax=vmax, gt=gt)
    show(fig.add_subplot(grid[0, 5]), spread, "std across bags", vmin=0,
         vmax=float(np.quantile(spread, 0.999)), gt=gt, cmap="viridis",
         label=f"mean std {spread.mean():.3f}")
    fig.suptitle(f"Four-bag score maps on a real query ({case.split('_')[1]}, {res} px). "
                 "The right panel shows per-pixel standard deviation across bags.",
                 y=0.99, fontsize=10)
    save(fig, "fig_rsg4_mech_bagging")


# --------------------------------------------------------------------------
# Fig. M1 (legacy combined plate)  Bagging and Component Quantile on a real query
# --------------------------------------------------------------------------
def fig_bag_cq(case="ad_hazelnut", tag="clean", res=1120):
    d = load(f"{case}__{tag}")
    gt = d["gt"]
    bags = [d[f"r{res}_bag{b}"] for b in range(4)]
    mean = np.mean(bags, axis=0)
    spread = np.std(bags, axis=0)
    fig = plt.figure(figsize=(13.2, 6.0))
    grid = fig.add_gridspec(2, 6, wspace=0.08, hspace=0.3)
    vmax = float(np.quantile(mean, 0.999))
    for b in range(4):
        show(fig.add_subplot(grid[0, b]), bags[b], f"bag {b + 1}  (256 tok/img)", vmin=0, vmax=vmax, gt=gt)
    show(fig.add_subplot(grid[0, 4]), mean, "mean of 4 bags", vmin=0, vmax=vmax, gt=gt)
    show(fig.add_subplot(grid[0, 5]), spread, "std across bags", vmin=0,
         vmax=float(np.quantile(spread, 0.999)), gt=gt, cmap="viridis",
         label=f"mean std {spread.mean():.3f}")

    raw, final = d["raw_map"], d["final_map"]
    threshold = float(np.quantile(raw, 0.99))
    vmax = float(np.quantile(final, 0.999))
    show(fig.add_subplot(grid[1, 0]), d["rgb"], "Query + GT", gt=gt)
    show(fig.add_subplot(grid[1, 1]), raw, "$A_0$  before CQ", vmin=0, vmax=vmax, gt=gt)
    show(fig.add_subplot(grid[1, 2]), (raw >= threshold).astype(float), "top-1% mask of $A_0$",
         vmin=0, vmax=1, gt=gt, cmap="gray")
    show(fig.add_subplot(grid[1, 3]), final, "$A$  after CQ", vmin=0, vmax=vmax, gt=gt)
    ax = fig.add_subplot(grid[1, 4:])
    rows = np.where(gt.any(axis=1))[0]
    row = int(rows[len(rows) // 2]) if len(rows) else raw.shape[0] // 2
    ax.plot(raw[row], color=GREY, lw=1.2, label="$A_0$ (before)")
    ax.plot(final[row], color=ORANGE, lw=1.6, label="$A$ (after CQ)")
    ax.axhline(threshold, color=NAVY, ls="--", lw=1, label="top-1% threshold $t$")
    if gt[row].any():
        cols = np.where(gt[row])[0]
        ax.axvspan(cols.min(), cols.max(), color="#3CF0FF", alpha=0.18, label="GT span")
    ax.set_title(f"Score profile along row {row}"), ax.set_xlabel("x (px)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Bagging (top, {res} px) and Component Quantile (bottom) on a real query. "
                 "CQ fills each top-1% component with its own maximum.", y=0.995, fontsize=10)
    save(fig, "fig_rsg4_mech_bag_cq")


# --------------------------------------------------------------------------
# Fig. M2b  Why gate the shallow layer: one normal + one defect query of an
# AD2 class, scored at the class-level low-FPR threshold.
# --------------------------------------------------------------------------
def fig_fusion_ad2(name="fruit_jelly", good="good_002_regular", bad="bad_003_overexposed"):
    from PIL import Image
    d = np.load(CACHE / f"scan_{name}.npz")
    tags = sorted({k.split("__")[0] for k in d.files})
    goods = [t for t in tags if t.startswith("good")]
    bads = [t for t in tags if t.startswith("bad")]

    def maps(tag):
        deep = d[f"{tag}__deep"].astype(np.float32)
        z5 = d[f"{tag}__normalized_l5"].astype(np.float32)
        res = d[f"{tag}__shallow_residual"].astype(np.float32)
        return {"deep": deep, "all": deep + z5, "gated": deep + res, "z5": z5, "res": res}

    # Class-level normal pool: 99th percentile of normal-image pixels = 1% FPR.
    thr = {m: float(np.quantile(np.concatenate([maps(t)[m].ravel() for t in goods]), 0.99))
           for m in ("deep", "all", "gated")}
    recall = {m: [] for m in thr}
    for t in bads:
        gt = d[f"{t}__gt"]
        if gt.sum() >= 30:
            for m in thr:
                recall[m].append(float((maps(t)[m][gt] > thr[m]).mean()))

    root = Path("/home/hun/Volume/DATA/mvtec_ad_2") / name / "test_public"
    def rgb(tag):
        kind, stem = tag.split("_", 1)
        image = Image.open(root / kind / f"{stem}.png").convert("RGB")
        h, w = d[f"{tag}__deep"].shape
        return np.asarray(image.resize((w, h), Image.BICUBIC))

    fig = plt.figure(figsize=(14.0, 5.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[5, 1.15], wspace=0.12)
    grid = outer[0].subgridspec(2, 5, wspace=0.05, hspace=0.18)
    for row, tag in enumerate((good, bad)):
        m = maps(tag)
        gt = d[f"{tag}__gt"]
        label = "normal query" if row == 0 else "defect query"
        show(fig.add_subplot(grid[row, 0]), rgb(tag), f"{label}", gt=gt)
        vmax = max(thr["all"], float(np.quantile(m["all"], 0.999)))
        for col, (key, title, t) in enumerate((
                ("deep", "Deep only  $z_d$", thr["deep"]),
                ("z5", "Shallow  $z_5$", None),
                ("all", "Add all  $z_d + z_5$", thr["all"]),
                ("gated", "SCARF  $z_d + [z_5-\\tau_5]_+$", thr["gated"]))):
            ax = fig.add_subplot(grid[row, col + 1])
            show(ax, m[key], title if row == 0 else "", vmin=0, vmax=vmax, gt=gt)
            if t is not None:
                alarm = m[key] > t
                if alarm.any():
                    ax.contourf(alarm, levels=[0.5, 1], colors=["#00FF7F"], alpha=0.35)
                if row == 0:
                    ax.text(0.03, 0.04, f"alarm px {100 * alarm.mean():.1f}%", transform=ax.transAxes,
                            color="white", fontsize=8.5,
                            bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.55, lw=0))
                elif gt.any():
                    ax.text(0.03, 0.04, f"defect hit {100 * alarm[gt].mean():.0f}%", transform=ax.transAxes,
                            color="white", fontsize=8.5,
                            bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.55, lw=0))
    ax = fig.add_subplot(outer[1])
    names = ("Deep\nonly", "Add\nall", "SCARF")
    values = [100 * np.mean(recall[m]) for m in ("deep", "all", "gated")]
    bars = ax.bar(names, values, color=[GREY, ORANGE, TEAL], width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", fontsize=9)
    ax.set_ylim(0, 100), ax.set_ylabel("defect px above class normal q99 (%)", fontsize=8.5)
    ax.set_title(f"{name}: {len(recall['deep'])} defect queries")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Why the shallow layer is gated (MVTec AD 2 {name}, 16-shot, bag/resolution mean). "
                 "Green = pixels above the 1%-FPR threshold set on the class's normal test images.\n"
                 "Adding all of $z_5$ shifts every normal pixel up, so the threshold rises and weak defects "
                 "fall below it; the gate adds $z_5$ only where it exceeds its own support q99.",
                 y=1.0, fontsize=10)
    save(fig, "fig_rsg4_mech_fusion_ad2")


if __name__ == "__main__":
    fig_scale()
    fig_fusion()
    fig_resolution()
    fig_bagging()
    fig_bag_cq()
    fig_fusion_ad2()
