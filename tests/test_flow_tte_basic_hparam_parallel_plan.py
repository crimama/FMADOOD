from __future__ import annotations

import subprocess
from pathlib import Path


def test_parallel_plan_assigns_each_unique_config_once() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["bash", str(root / "scripts/run_flow_tte_basic_hparam_parallel_remote.sh"), "plan"],
        check=True,
        capture_output=True,
        text=True,
    )
    sections = completed.stdout.strip().split("\n\n")
    assignment_rows = sections[0].splitlines()[1:]
    config_rows = sections[1].splitlines()[1:]

    assigned = [
        config
        for row in assignment_rows
        for config in row.split("\t")[2].split(",")
    ]
    declared = [row.split("\t")[0] for row in config_rows]

    assert len(assignment_rows) == 6
    assert len(declared) == 9
    assert len(assigned) == len(set(assigned)) == 9
    assert set(assigned) == set(declared)


def test_parallel_plan_has_complete_depth_width_factorial() -> None:
    root = Path(__file__).resolve().parents[1]
    output = subprocess.check_output(
        ["bash", str(root / "scripts/run_flow_tte_basic_hparam_parallel_remote.sh"), "plan"],
        text=True,
    )
    config_rows = output.strip().split("\n\n")[1].splitlines()[1:]
    capacity = {
        (int(parts[1]), int(parts[2]))
        for row in config_rows
        if (parts := row.split("\t"))[0].startswith("cap_")
    }
    assert capacity == {(depth, width) for depth in (1, 2, 4) for width in (1, 2)}
