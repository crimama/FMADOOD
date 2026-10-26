"""Rank diagnostic of order aggregation: calibration none vs Q95 (AD2 public, seed 0)."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/remote_runs/dsba3/scopead_order_rank_diagnostic_20261018"
OUT = ROOT / "PaperWorks/figures"
RES = [448, 672, 896, 1120]
COLORS = ["#C9DCEE", "#7FA8D1", "#3D6FA8", "#183B66"]


def load():
    cells = {}
    for path in RUN.rglob("order_rank_diagnostic.json"):
        record = json.loads(path.read_text())
        key = (record["statistical_normalization"], int(record["shots"]))
        metrics = json.loads((path.parent / "metrics.json").read_text())[record["object"]]
        cells.setdefault(key, {})[record["object"]] = (record, metrics)
    return cells


def size_macro(rows):
    """Macro over classes that contain defect regions of the given size bin."""
    out = []
    for b in range(3):
        present = [np.asarray(s["size_top1_share"][b], float) for s in rows if sum(s["size_top1_share"][b]) > 0]
        out.append(np.mean(present, axis=0) if present else np.zeros(len(RES)))
    return np.asarray(out)


def macro(cells):
    table = {}
    for key, per_class in sorted(cells.items()):
        rows = [r["summary"] for r, _ in per_class.values()]
        mets = [m for _, m in per_class.values()]
        mean = lambda name: np.mean([np.asarray(s[name], dtype=float) for s in rows], axis=0)
        agree = [s["evidence_best_agreement"] for s in rows if s["evidence_best_agreement"] is not None]
        table[key] = dict(
            classes=len(rows),
            aupro=100 * np.mean([m["seg_AUPRO_0.05"] for m in mets]),
            pauroc=100 * np.mean([m["seg_AUROC_0.05"] for m in mets]),
            normal_top1=mean("normal_top1_share"), defect_top1=mean("defect_top1_share"),
            normal_weight=mean("normal_effective_renyi_weight"), defect_weight=mean("defect_effective_renyi_weight"),
            dominance=float(np.mean([s["normal_top1_dominance"] for s in rows])),
            tv=float(np.mean([s["top1_total_variation_defect_vs_normal"] for s in rows])),
            agreement=float(np.mean(agree)) if agree else float("nan"),
            size_top1=size_macro(rows),
            size_classes=[int(sum(1 for s in rows if sum(s["size_top1_share"][b]) > 0)) for b in range(3)],
            routed=int(sum(r["counts"]["routed_images"] for r, _ in per_class.values())),
            images=int(sum(r["counts"]["images"] for r, _ in per_class.values())),
            per_class={c: dict(dominance=r["summary"]["normal_top1_dominance"],
                               tv=r["summary"]["top1_total_variation_defect_vs_normal"],
                               top1_res=RES[int(np.argmax(r["summary"]["normal_top1_share"]))],
                               aupro=100 * m["seg_AUPRO_0.05"])
                       for c, (r, m) in sorted(per_class.items())},
        )
    return table


def draw(table):
    shots = sorted({s for _, s in table})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, len(shots), figsize=(3.7 * len(shots), 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    groups = (("normal", "Normal\npixels"), ("small", "Small\ndefects"), ("large", "Large\ndefects"))
    for ax, shot in zip(axes, shots):
        bars, ticks, centers = [], [], []
        x0 = 0.0
        for norm, name in (("none", "No calibration"), ("q95", "Q95 calibration")):
            if (norm, shot) not in table:
                continue
            t = table[(norm, shot)]
            values = {"normal": t["normal_top1"], "small": t["size_top1"][0], "large": t["size_top1"][2]}
            xs = [x0 + 0.85 * i for i in range(len(groups))]
            for x, (key, label) in zip(xs, groups):
                bars.append((x, values[key]))
                ticks.append((x, label))
            centers.append((np.mean(xs), name))
            x0 = xs[-1] + 1.35
        for r in range(len(RES)):
            xs = [b[0] for b in bars]
            heights = np.array([b[1][r] for b in bars])
            bottoms = np.array([sum(b[1][:r]) for b in bars])
            ax.bar(xs, heights, 0.7, bottom=bottoms, color=COLORS[r], edgecolor="white", lw=0.6,
                   label=f"{RES[r]} px" if shot == shots[0] else None)
        ax.set_xticks([t[0] for t in ticks])
        ax.set_xticklabels([t[1] for t in ticks], fontsize=7)
        for c, name in centers:
            ax.text(c, 1.04, name, ha="center", fontsize=8.5, color="#222222")
        ax.set_ylim(0, 1.13)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(f"({'ab'[shots.index(shot)]}) {shot}-shot", fontsize=10, pad=14)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Share of pixels where the\nresolution is ranked first")
    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=8)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.76, bottom=0.17, wspace=0.08)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"fig_scopead_order_rank_diagnostic.{ext}", dpi=240, facecolor="white")
    plt.close(fig)



def main():
    table = macro(load())
    serial = {f"{n}_s{s:02d}": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in t.items()}
              for (n, s), t in table.items()}
    (OUT / "fig_scopead_order_rank_diagnostic.json").write_text(json.dumps(serial, indent=2) + "\n")
    for (n, s), t in table.items():
        print(f"{n:4s} s{s:02d} classes={t['classes']} AUPRO={t['aupro']:.2f} pAUROC={t['pauroc']:.2f} "
              f"dom={t['dominance']:.3f} TV={t['tv']:.3f} agree={t['agreement']:.3f} routed={t['routed']}/{t['images'] + t['routed']}")
        print("   normal_top1", np.round(t["normal_top1"], 3), "defect_top1", np.round(t["defect_top1"], 3))
        print("   normal_w", np.round(t["normal_weight"], 3), "defect_w", np.round(t["defect_weight"], 3))
        print("   size_top1", np.round(t["size_top1"], 3).tolist(), "classes", t["size_classes"])
        print("   per_class", {c: (v["top1_res"], round(v["dominance"], 2), round(v["tv"], 2)) for c, v in t["per_class"].items()})
    if "--no-figure" not in sys.argv:
        draw(table)


if __name__ == "__main__":
    main()

