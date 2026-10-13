"""Draw the Paper v10 acquisition-shift figure from recorded run manifests.

Run from the repository root with ``.venv/bin/python
PaperWorks/figures/plot_acquisition_shift_by_shot.py``. The figure uses only
seed-0 development results; no method fitting or evaluation is performed.
"""

from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SHOTS = (1, 2, 4, 8, 16)
CONDITIONS = ("regular", "shift_1", "shift_2", "shift_3")
METHODS = ("RSG4", "SuperAD", "PatchCore")
STYLES = {
    "RSG4": ("Ours (RSG4)", "#0072B2", "o"),
    "SuperAD": ("SuperAD", "#D55E00", "s"),
    "PatchCore": ("PatchCore", "#4D4D4D", "^"),
}


def add_manifest(data, path, method, shot):
    manifest = json.loads(path.read_text())
    for obj, metadata in manifest["objects"].items():
        key = (method, shot, obj)
        if key in data:
            raise ValueError(f"duplicate stratum result: {key}")
        data[key] = metadata["condition_strata"]


def load_results():
    data = {}
    rsg4 = ROOT / "results/rsg4_fewshot_seed0_20260926"
    for path in rsg4.rglob("run_manifest.json"):
        label = next(part for part in path.parts if part.startswith("RSG4_seed0_s"))
        add_manifest(data, path, "RSG4", int(label.rsplit("s", 1)[1]))

    comparator_roots = (
        ROOT / "results/local_runs/paper_v10_20260928",
        ROOT / "results/remote_runs/dsba3/paper_v10_20260928",
    )
    for root in comparator_roots:
        for stage in root.glob("stage1_*"):
            for path in stage.rglob("run_manifest.json"):
                label = path.relative_to(stage).parts[0]
                method, shot = label.rsplit("_s", 1)
                add_manifest(data, path, {"superad": "SuperAD", "patchcore": "PatchCore"}[method],
                             int(shot))
    return data


def summarize(data):
    absolute = defaultdict(list)
    for method in METHODS:
        for shot in SHOTS:
            entries = {obj: strata for (name, count, obj), strata in data.items()
                       if name == method and count == shot}
            if len(entries) != 8:
                raise ValueError(f"expected eight classes for {method}, {shot}-shot")
            for condition in CONDITIONS:
                values = [strata[condition]["AUPRO_0_05"] for strata in entries.values()
                          if condition in strata]
                expected = 8 if condition in {"regular", "shift_1"} else 7
                if len(values) != expected:
                    raise ValueError(f"expected {expected} classes for {method}/{shot}/{condition}")
                absolute[method, condition].append(100 * np.mean(values))
    return absolute


def draw(absolute):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.0), sharex=True,
                             sharey=True, constrained_layout=True)
    x = np.arange(len(SHOTS))
    for col, condition in enumerate(CONDITIONS):
        ax = axes[col]
        ax.set_title("Regular" if condition == "regular" else f"Shift {col}")
        for method in METHODS:
            label, color, marker = STYLES[method]
            ax.plot(x, absolute[method, condition], label=label, color=color,
                    marker=marker, markersize=4.5, linewidth=1.7)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(x, SHOTS)
        ax.set_xlabel("Support shot")
    axes[0].set_ylabel("AUPRO@0.05 (%)")
    axes[0].set_ylim(30, 65)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3,
               frameon=False, columnspacing=2.2)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"fig_acquisition_shift_by_shot.{suffix}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw(summarize(load_results()))
