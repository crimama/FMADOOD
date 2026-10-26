#!/usr/bin/env python3
"""Generate final-ScopeAD multi-seed configs (seeds 1/2) from the seed-0 Renyi configs."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SRC = {
    "ad2": ROOT / "configs/benchmarks/scopead_renyi_cq_20261018/ad2_scale_order_renyi_cq_s{n:02d}.yaml",
    "mvtec": ROOT / "configs/benchmarks/scopead_mvtec_renyi_cq_20261018/mvtec_scale_order_renyi_cq_s{n:02d}.yaml",
    "visa": ROOT / "configs/benchmarks/scopead_visa_renyi_cq_20261018/visa_scale_order_renyi_cq_s{n:02d}.yaml",
}
PUBLIC = "configs/benchmarks/public_v13_20260929/seed{s}_support_{n}.json"
E1 = "configs/benchmarks/v7_evidence/E1_seeds/manifests/seed{s}_support_{n}.json"
AD2_MANIFEST = {1: PUBLIC, 4: PUBLIC, 8: PUBLIC, 2: E1, 16: E1}
labels = {"ad2": [], "mvtec": [], "visa": []}
for ds, tmpl in SRC.items():
    for seed in (1, 2):
        for n in (1, 2, 4, 8, 16):
            cfg = yaml.safe_load(Path(str(tmpl).format(n=n)).read_text())
            cfg["seed"] = seed
            cfg["protocol"]["fixed_reference_manifest"] = (
                AD2_MANIFEST[n].format(s=seed, n=n) if ds == "ad2" else None
            )
            label = f"{ds}_renyi_cq_s{n:02d}_seed{seed}"
            (OUT / f"{label}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
            labels[ds].append(label)
for ds, ls in labels.items():
    (OUT / f"{ds}.txt").write_text("\n".join(ls) + "\n")
print({k: len(v) for k, v in labels.items()})
