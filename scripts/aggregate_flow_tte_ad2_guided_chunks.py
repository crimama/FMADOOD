"""Aggregate matched raw/morph and guided-r8/morph AD2 chunk metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

_OBJECTS = ("can", "fabric", "fruit_jelly", "rice", "vial", "wallplugs", "walnuts", "sheet_metal")


def collect(root: Path, arm: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "chunks").glob(f"*/{arm}/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for object_name in _OBJECTS:
            if object_name not in payload:
                continue
            if object_name in rows:
                raise RuntimeError(f"Duplicate {arm} result for {object_name}")
            rows[object_name] = payload[object_name]
    missing = sorted(set(_OBJECTS) - set(rows))
    if missing:
        raise RuntimeError(f"Missing {arm} results: {missing}")
    return rows


def mean(rows: dict[str, dict[str, Any]], metric: str) -> float:
    return sum(metric_value(rows[obj], metric) for obj in _OBJECTS) / len(_OBJECTS)


def metric_value(row: dict[str, Any], metric: str) -> float:
    if metric == "seg_F1_raw" and metric not in row:
        return float(row["seg_F1"])
    return float(row[metric])


def aggregate(root: Path) -> dict[str, Any]:
    raw = collect(root, "raw")
    guided = collect(root, "guided_r8_morph")
    metrics = ("seg_AUROC", "seg_F1_raw", "seg_F1")
    raw_mean = {metric: mean(raw, metric) for metric in metrics}
    guided_mean = {metric: mean(guided, metric) for metric in metrics}
    return {
        "objects": list(_OBJECTS),
        "control": "raw_continuous_plus_closefill_erode",
        "candidate": "guided_r8_eps1e-2_plus_closefill_erode",
        "raw_mean": raw_mean,
        "guided_mean": guided_mean,
        "guided_minus_raw": {
            metric: guided_mean[metric] - raw_mean[metric] for metric in metrics
        },
        "per_object": {
            obj: {
                "raw": raw[obj],
                "guided": guided[obj],
                "delta_seg_AUROC": float(guided[obj]["seg_AUROC"]) - float(raw[obj]["seg_AUROC"]),
                "delta_seg_F1": float(guided[obj]["seg_F1"]) - float(raw[obj]["seg_F1"]),
            }
            for obj in _OBJECTS
        },
    }


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args(list(argv))
    root = Path(args.root)
    payload = aggregate(root)
    (root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["object\traw_AUROC\traw_F1\tguided_AUROC\tguided_F1\tdelta_AUROC\tdelta_F1"]
    for obj, row in payload["per_object"].items():
        lines.append(
            f"{obj}\t{row['raw']['seg_AUROC']:.9f}\t{row['raw']['seg_F1']:.9f}\t"
            f"{row['guided']['seg_AUROC']:.9f}\t{row['guided']['seg_F1']:.9f}\t"
            f"{row['delta_seg_AUROC']:+.9f}\t{row['delta_seg_F1']:+.9f}"
        )
    (root / "per_object_metrics.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
