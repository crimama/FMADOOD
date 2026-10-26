#!/usr/bin/env python3
"""Final-ScopeAD cost arms (AD2 16-shot seed 0, cost_profile on): 4-res / single 1120, cap 256 / 1024."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SRC = ROOT / "configs/benchmarks/scopead_renyi_cq_20261018/ad2_scale_order_renyi_cq_s16.yaml"
labels = []
for res_name, resolutions in (("mr", [448, 672, 896, 1120]), ("r1120", [1120])):
    for cap in (256, 1024):
        cfg = yaml.safe_load(SRC.read_text())
        mc = cfg["method_config"]
        mc["resolutions"] = resolutions
        mc["bank_tokens_per_source"] = cap
        if res_name == "r1120":
            mc["noise_guard_resolutions"] = [1120]
            mc["multiscale_aggregation"] = "mean"
        cfg["evaluation"]["cost_profile"] = True
        label = f"ad2_{res_name}_cap{cap}_s16_seed0_cost"
        (OUT / f"{label}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
        labels.append(label)
(OUT / "cost.txt").write_text("\n".join(labels) + "\n")
print(labels)
