from __future__ import annotations

import subprocess
from pathlib import Path

BASH = Path("/usr/bin/bash")
LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_flow_tte_basic_hparam_parallel_remote.sh"
)


def run_launcher(mode: str) -> str:
    return subprocess.check_output(  # noqa: S603
        [str(BASH), str(LAUNCHER), mode],
        text=True,
    )


def test_current_baseline_contract_matches_selected_mvtecad2_setting() -> None:
    output = run_launcher("current")

    assert output.splitlines() == [
        "setting\tvalue",
        "objects\tcan,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal",
        "backbone\tdinov2_vitl14",
        "shots\t16",
        "seed\t0",
        "flow_depth\t2",
        "hidden_width\t1",
        "epochs\t3",
        "learning_rate\t2e-4",
        "lambda_logdet\t2e-2",
        "brightness\t0.8,1.2",
        "feature_layers\t11,15,19,23",
        "feature_fusion\tlayer_norm_mean",
        "expansion_budget\t1.0",
        "loo_standardization\toff",
        "dvt\tposition_mean:1.0",
        "context\tnone",
        "raw_rgb_guide\tnone",
        "refinement\tguided_r8",
        "binary_postprocess\tclosefill_erode:17:16",
    ]


def test_parallel_plan_assigns_each_unique_config_once() -> None:
    sections = run_launcher("plan").strip().split("\n\n")
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
    output = run_launcher("plan")
    config_rows = output.strip().split("\n\n")[1].splitlines()[1:]
    capacity = {
        (int(parts[1]), int(parts[2]))
        for row in config_rows
        if (parts := row.split("\t"))[0].startswith("cap_")
    }
    assert capacity == {(depth, width) for depth in (1, 2, 4) for width in (1, 2)}
