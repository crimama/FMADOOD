# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
# ─── How to run ───
# python3 scripts/run_darc_common_eval.py --data-root /data/mvtec_ad_2 \
#   --map-root /results/chunk0 --map-root /results/chunk1 --output-root /results/eval \
#   --method-label DARC-G0 --resource-label P16-random --objects can fabric
"""Evaluate raw anomaly maps on one common native-grid, image-disjoint protocol."""

from __future__ import annotations

# pyright: reportMissingImports=false
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from flow_tte.darc_common_eval import evaluate_object  # noqa: E402
from flow_tte.darc_common_report import (  # noqa: E402
    EvaluationRun,
    ObjectResult,
    RunMetadata,
    write_evaluation_outputs,
)
from flow_tte.darc_map_io import audit_only, load_object_maps  # noqa: E402


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class RunConfig:
    data_root: Path
    map_roots: Tuple[Path, ...]
    output_root: Path
    method_label: str
    resource_label: str
    comparable: bool
    objects: Tuple[str, ...]


class _ParsedArgs(argparse.Namespace):
    data_root: Path
    map_root: List[Path]
    output_root: Path
    method_label: str
    resource_label: str
    comparable: bool
    objects: List[str]

    def __init__(self) -> None:
        super().__init__()
        self.data_root = Path()
        self.map_root = []
        self.output_root = Path()
        self.method_label = ""
        self.resource_label = ""
        self.comparable = False
        self.objects = []


def parse_args(argv: Sequence[str]) -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--resource-label", required=True)
    parser.add_argument("--comparable", action="store_true")
    parser.add_argument("--objects", nargs="+", required=True)
    namespace = parser.parse_args(list(argv), namespace=_ParsedArgs())
    objects = tuple(namespace.objects)
    if len(set(objects)) != len(objects):
        parser.error("--objects must not contain duplicates")
    return RunConfig(
        data_root=namespace.data_root,
        map_roots=tuple(namespace.map_root),
        output_root=namespace.output_root,
        method_label=namespace.method_label,
        resource_label=namespace.resource_label,
        comparable=namespace.comparable,
        objects=objects,
    )


def run(config: RunConfig) -> None:
    results: List[ObjectResult] = []
    for object_name in config.objects:
        print(f"loading {object_name}", flush=True)
        maps = load_object_maps(config.data_root, config.map_roots, object_name)
        metrics = evaluate_object(maps)
        results.append(ObjectResult(audits=audit_only(maps), metrics=metrics))
        message = " ".join(
            (
                f"evaluated {object_name}: pAUROC@.05={metrics.all_test.p_auroc_005:.6f}",
                f"AP={metrics.all_test.p_ap:.6f} F1={metrics.all_test.oracle_f1:.6f}",
            ),
        )
        print(message, flush=True)
        del maps
    write_evaluation_outputs(
        EvaluationRun(
            metadata=RunMetadata(
                data_root=config.data_root,
                map_roots=config.map_roots,
                output_root=config.output_root,
                method_label=config.method_label,
                resource_label=config.resource_label,
                comparable=config.comparable,
            ),
            objects=tuple(results),
        ),
    )


def main() -> None:
    run(parse_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
