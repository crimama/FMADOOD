#!/usr/bin/env python3
"""Summarize class-macro image AUROC and oracle max-F1 from saved maps."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import tifffile

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from flow_tte.metrics import f1_score_max


def image_score(path: Path, top_fraction: float) -> float:
    values = np.asarray(tifffile.imread(path), dtype=np.float32).reshape(-1)
    count = max(1, math.ceil(values.size * top_fraction))
    return float(np.partition(values, values.size - count)[-count:].mean())


def condition_metrics(root: Path, top_fraction: float) -> dict[str, object]:
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for object_name, record in metrics["per_object"].items():
        map_root = root / "anomaly_maps" / object_name / "test"
        labels: list[bool] = []
        scores: list[float] = []
        for anomaly_dir in sorted(path for path in map_root.iterdir() if path.is_dir()):
            for map_path in sorted(anomaly_dir.glob("*.tiff")):
                labels.append(anomaly_dir.name != "good")
                scores.append(image_score(map_path, top_fraction))
        f1 = f1_score_max(np.asarray(labels, dtype=np.bool_), np.asarray(scores, dtype=np.float32))
        f1_values.append(f1)
        per_class[object_name] = {"i_AUROC": float(record["i_AUROC"]), "Img_F1": f1}
    return {
        "i_AUROC": float(metrics["i_AUROC"]),
        "Img_F1": float(np.mean(f1_values)),
        "per_class": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    args = parser.parse_args()
    conditions = ("id", "brightness", "contrast", "defocus_blur", "gaussian_noise")
    result = {
        "protocol": {
            "dataset": "MVTec AD / MVTec-OOD",
            "corruptions": ["brightness", "contrast", "defocus_blur", "gaussian_noise"],
            "severity": 3,
            "Img_F1": "oracle max-F1 per class, then unweighted 15-class macro",
            "image_score": "mean top 1% of raw anomaly map",
        },
        "conditions": {
            name: condition_metrics(args.run_root / name, args.top_fraction) for name in conditions
        },
    }
    ood = [result["conditions"][name] for name in conditions[1:]]
    result["OOD_Avg"] = {
        key: float(np.mean([item[key] for item in ood])) for key in ("i_AUROC", "Img_F1")
    }
    total = [result["conditions"][name] for name in conditions]
    result["Total_Avg"] = {
        key: float(np.mean([item[key] for item in total])) for key in ("i_AUROC", "Img_F1")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
