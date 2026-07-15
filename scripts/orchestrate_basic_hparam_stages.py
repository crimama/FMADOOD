"""Persistently advance Basic hyperparameter stages across dsba3/dsba5."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_flow_tte_basic_hparam_parallel_remote.sh"
STATE = ROOT / "results/basic_hparam_loooff_master"
STAGES = {
    1: ("flowtte_basic_hparam_loooff_stage1_capacity_20260713_v1", "hparam_stage1"),
    2: ("flowtte_basic_hparam_loooff_stage2_optimization_20260713_v1", "hparam_stage2"),
    3: ("flowtte_basic_hparam_loooff_stage3_regularization_20260713_v1", "hparam_stage3"),
    4: ("flowtte_basic_hparam_loooff_stage4_layers_20260713_v1", "hparam_stage4"),
}


def call(mode: str, stage: int, selected: dict[str, str]) -> str:
    run_name, kind = STAGES[stage]
    env = dict(os.environ)
    env.update(selected)
    env.update({"RUN_NAME": run_name, "EXPERIMENT_KIND": kind})
    result = subprocess.run(
        ["bash", str(LAUNCHER), mode], cwd=ROOT, env=env,
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return result.stdout


def wait_complete(stage: int, selected: dict[str, str]) -> None:
    while True:
        output = call("status", stage, selected)
        (STATE / f"stage{stage}_status.txt").write_text(output)
        if output.count("COMPLETE=yes") == 2:
            return
        if "ERRORS=0" not in output:
            raise RuntimeError(output)
        time.sleep(120)


def select(stage: int) -> tuple[str, dict]:
    run_name, _ = STAGES[stage]
    candidates = []
    for host in ("dsba3", "dsba5"):
        root = ROOT / f"results/remote_runs/{host}/{run_name}/configs"
        for path in root.glob("*/summary.json"):
            payload = json.loads(path.read_text())
            metric = payload["guided_mean"]
            candidates.append((metric["seg_AUROC"], metric["seg_F1"], path.parent.name))
    if not candidates:
        raise RuntimeError(f"no summaries for stage {stage}")
    auroc, f1, name = max(candidates)
    return name, {"seg_AUROC": auroc, "seg_F1": f1}


def main() -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    selected: dict[str, str] = {}
    for stage in range(1, 5):
        if stage > 1:
            call("start", stage, selected)
        wait_complete(stage, selected)
        call("pull", stage, selected)
        winner, metrics = select(stage)
        record = {"stage": stage, "winner": winner, "metrics": metrics}
        (STATE / f"stage{stage}_selection.json").write_text(json.dumps(record, indent=2) + "\n")
        if stage == 1:
            _, depth, width = winner.split("_")
            selected.update({"SELECTED_DEPTH": depth[1:], "SELECTED_WIDTH": width[1:]})
        elif stage == 2:
            parts = winner.split("_")
            selected.update({
                "SELECTED_EPOCHS": parts[1][1:],
                "SELECTED_LR": {"lr1": "1e-4", "lr2": "2e-4", "lr5": "5e-4"}[parts[2]],
            })
        elif stage == 3:
            selected.update({
                "SELECTED_LOGDET": "2e-2" if "ld2" in winner else "1e-3",
                "SELECTED_BRIGHTNESS": "0.8,1.2" if "b08" in winner else "1.0,1.0",
            })
        (STATE / "selected_env.json").write_text(json.dumps(selected, indent=2) + "\n")
    (STATE / "complete.txt").write_text("all_stages_complete=true\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
