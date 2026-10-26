#!/usr/bin/env python3
"""Final-ScopeAD validation suites: calibration x aggregation multi-seed, lambda and q sweeps, corruption."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
B = ROOT / "configs/benchmarks"
AD2 = str(B / "scopead_renyi_cq_20261018/ad2_scale_order_renyi_cq_s{n:02d}.yaml")
MVT = str(B / "scopead_mvtec_renyi_cq_20261018/mvtec_scale_order_renyi_cq_s16.yaml")
PUBLIC = "configs/benchmarks/public_v13_20260929/seed{s}_support_{n}.json"
E1 = "configs/benchmarks/v7_evidence/E1_seeds/manifests/seed{s}_support_{n}.json"


def load(path):
    return yaml.safe_load(Path(path).read_text())


def save(label, cfg):
    (OUT / f"{label}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    return label


suites = {"calib_seeds": [], "lambda_sweep": [], "q_sweep": [], "corruption_clean": []}
# 1. calibration x aggregation, seeds 1/2, 2/16-shot (Q95+Renyi comes from the multi-seed main run)
arms = {
    "none_mean": ("none", "mean"),
    "q95_mean": ("q95", "mean"),
    "none_renyi": ("none", "scale_order_renyi"),
}
for seed in (1, 2):
    for n in (2, 16):
        for arm, (norm, agg) in arms.items():
            cfg = load(AD2.format(n=n))
            cfg["seed"] = seed
            cfg["protocol"]["fixed_reference_manifest"] = E1.format(s=seed, n=n)
            cfg["method_config"]["statistical_normalization"] = norm
            cfg["method_config"]["multiscale_aggregation"] = agg
            suites["calib_seeds"].append(save(f"ad2_{arm}_s{n:02d}_seed{seed}", cfg))
# 2. shallow weight sweep, seed 0 (lambda=0.125 is the seed-0 final run)
for n in (2, 16):
    for lam in (0.0, 0.0625, 0.25, 0.5):
        cfg = load(AD2.format(n=n))
        cfg["method_config"]["layer_fusion_gamma"] = lam
        tag = str(lam).replace(".", "p")
        suites["lambda_sweep"].append(save(f"ad2_lambda{tag}_s{n:02d}", cfg))
# 3. reference quantile sweep, seed 0 (q=0.95 is the seed-0 final run)
for n in (2, 16):
    for q in (0.5, 0.8, 0.9, 0.99, 0.999):
        cfg = load(AD2.format(n=n))
        cfg["method_config"]["tail_quantile"] = q
        tag = str(q).replace(".", "p")
        suites["q_sweep"].append(save(f"ad2_q{tag}_s{n:02d}", cfg))
# 4. corruption (MVTec AD 16-shot, seed 0, same manifest as the comparators)
cfg = load(MVT)
cfg["protocol"]["fixed_reference_manifest"] = "configs/benchmarks/table5_rsg4_20261001/manifests/mvtec_seed0_s16_generated.json"
suites["corruption_clean"].append(save("mvtec_renyi_cq_s16_seed0_corruption", cfg))
for name, labels in suites.items():
    (OUT / f"{name}.txt").write_text("\n".join(labels) + "\n")
print({k: len(v) for k, v in suites.items()})
