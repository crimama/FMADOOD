#!/usr/bin/env python3
"""Resolution-major controller helpers and stage summarizer.

The remote shell owns GPU child processes. This module validates each finished
stage, writes its leaderboard/combined summary atomically, prints KEEP gates,
and returns 42 for an invalid 672 parity stage before 896 can start.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from flow_tte_resolution_ladder import RESOLUTIONS, keep_gate_status  # noqa: E402


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_stage_objects(stage_root: Path) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for path in sorted(stage_root.glob("chunks/*/objects/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = str(payload["object"])
        if name in rows:
            raise RuntimeError(f"duplicate stage object: {name}")
        rows[name] = payload
    if len(rows) != 8:
        raise RuntimeError(f"expected 8 stage objects, found {len(rows)}")
    return rows


def summarize_stage(run_root: Path, resolution: int) -> tuple[dict[str, Any], bool]:
    stage_root = run_root / "stages" / str(resolution)
    objects = load_stage_objects(stage_root)
    invalid = resolution == 672 and any(
        not bool(row.get("anchor_parity", {}).get("pass")) for row in objects.values()
    )
    keys = {
        "mean_f1": "pooled_oracle_f1_float16",
        "mean_pixel_ap": "pooled_pixel_ap_float16",
        "mean_pauroc_0.05": "pooled_pauroc_0.05_float16",
        "mean_component_recall": "gt_component_recall_at_oracle",
        "mean_small_component_recall": "small_defect_component_recall_at_oracle",
        "mean_normal_fpr": "normal_image_mean_fpr_at_oracle",
    }
    means = {out: float(np.mean([float(r["metrics"][key]) for r in objects.values()])) for out, key in keys.items()}
    means["gpu_hours"] = sum(float(r["diagnostics"]["elapsed_seconds"]) for r in objects.values()) / 3600.0
    summary = {
        "resolution": resolution, "valid": not invalid, "object_count": 8,
        "means": means, "full_frame": True, "tile_rule": "disabled",
        "query_chunk_size": next(iter(objects.values()))["diagnostics"]["query_chunk_size"],
        "refit": "DVT and flow independently refit from fixed support at this resolution",
    }
    # Produce one canonical, object-complete leaderboard independent of shard order.
    columns = ["object", "f1", "pixel_ap", "pauroc_0.05", "component_recall", "small_component_recall", "runtime_seconds", "peak_gpu_allocated_bytes"]
    lines = ["\t".join(columns)]
    for name, row in sorted(objects.items()):
        m, d = row["metrics"], row["diagnostics"]
        lines.append("\t".join(map(str, (name, m["pooled_oracle_f1_float16"], m["pooled_pixel_ap_float16"], m["pooled_pauroc_0.05_float16"], m["gt_component_recall_at_oracle"], m["small_defect_component_recall_at_oracle"], d["elapsed_seconds"], d["peak_gpu_allocated_bytes"]))))
    leaderboard = run_root / f"stage_{resolution}_leaderboard.tsv"
    write_atomic(leaderboard, "\n".join(lines) + "\n")
    write_atomic(stage_root / "stage_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if invalid:
        write_atomic(stage_root / "INVALID", "anchor parity tolerance=0 failed\n")
        write_atomic(run_root / "stage_672_INVALID.json", json.dumps(summary, indent=2) + "\n")
    else:
        write_atomic(stage_root / "COMPLETE", "valid=true\n")
    return summary, not invalid


def update_combined(run_root: Path) -> dict[str, Any]:
    stages: dict[int, dict[str, Any]] = {}
    for resolution in RESOLUTIONS:
        path = run_root / "stages" / str(resolution) / "stage_summary.json"
        if path.is_file():
            stages[resolution] = json.loads(path.read_text())
    means = {r: s["means"] for r, s in stages.items() if s["valid"]}
    payload = {
        "schema": "flowtte-resolution-ladder-summary-v1",
        "stage_order": list(RESOLUTIONS), "completed_stages": list(stages),
        "stages": {str(k): v for k, v in stages.items()},
        "keep_gate_section_10_4": keep_gate_status(means),
        "decision_owner": "orchestrator",
    }
    write_atomic(run_root / "ladder_summary.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=RESOLUTIONS, required=True)
    args = parser.parse_args(argv)
    summary, valid = summarize_stage(args.run_root, args.resolution)
    combined = update_combined(args.run_root)
    print(json.dumps({"stage": summary, "keep_gate_section_10_4": combined["keep_gate_section_10_4"]}, sort_keys=True))
    return 0 if valid else 42


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
