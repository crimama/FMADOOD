# Fixed linear shallow fusion: seed 0 development comparison

Hypothesis: Deep + 0.125 × Shallow may improve generalization over gated SCARF. Only shallow mode and gamma change. Lambda .125 was selected using prior AD2 public 2-shot seed-0 AUPRO evidence and is frozen; no retuning.

Matched conditions: frozen DINOv2 ViT-L/14 layers 5/11/23, FP16, resolutions 448/672/896/1120, cap256/source, four bags, q95 tail scaling, q99 residual threshold, CQ99, fixed k9 lowres guard, final-map top1% mean image score. No trainable backbone parameters. Maps/weights off. Labels and GT excluded from fitting/prediction.

Scope: seed 0, shots 1/2/4/8/16, AD2 public full8, MVTec AD full15, VisA full12. All are seen development populations, not untouched final holdouts. Normal-only existing support manifests; generated matched MVTec 16-shot manifest. Exact hashes and allocation: plan.json.

Metrics: matched linear-minus-gated macro deltas per dataset/shot. AD2 primary AUPRO .05, secondary partial pixel AUROC .05; higher is better. MVTec/VisA paper metrics image AUROC and full-FPR pixel AUROC; higher is better. Retain all evaluator metrics and class results; distinguish partial/full AUROC.

Budget: 350 matched class runs, 318 new and 32 reused AD2 2/16-shot runs with exact config/package-code parity. Four existing dsba3 GPUs 0–3, one class worker/GPU, sequential existing runner, timeout10800 seconds/class. Stop affected shard on runtime/manifest/metric/provenance validity failure and retain logs. Verify support/query/GT and bank/calibration provenance before conclusions. Single-seed results are development evidence, not multi-seed confirmation.
