#!/usr/bin/env python3
"""Aggregate per-object metrics.json files under <root>/**/<label>/ into class macros per label.

Output: JSON {label: {n, objects, metrics:{name: macro*100}}}. Read-only helper for manual result tables.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

KEYS = ["seg_AUPRO_0.05", "seg_AUROC_0.05", "image_AUROC", "seg_AUROC_1.00", "seg_F1_oracle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    runs = defaultdict(dict)  # label -> object -> metrics
    for root in args.roots:
        root = Path(root)
        for mpath in root.rglob("metrics.json"):
            rel = mpath.relative_to(root).parts
            label = None
            for i, part in enumerate(rel):
                if part.startswith(("multiscale_aggregation", "reference_matching")) and i > 0:
                    label = rel[i - 1]
                    break
            if label is None:
                continue
            data = json.loads(mpath.read_text())
            for obj, m in data.items():
                if isinstance(m, dict) and ("seg_AUROC_0.05" in m or "image_AUROC" in m):
                    runs[label][obj] = m
    out = {}
    for label, objs in sorted(runs.items()):
        metrics = {}
        for k in KEYS:
            vals = [m[k] for m in objs.values() if k in m]
            if len(vals) == len(objs) and vals:
                metrics[k] = 100.0 * sum(vals) / len(vals)
        out[label] = {"n": len(objs), "objects": sorted(objs), "metrics": metrics}
    for label, v in out.items():
        m = v["metrics"]
        print(f"{label}\tn={v['n']}\t" + "\t".join(f"{k}={m[k]:.2f}" for k in m))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
