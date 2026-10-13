# SCARF linear0125 multi-seed (seeds 1,2) and AD2 calibration factorial continuation

Question: does the fixed shallow addition (linear0125, z_d + 0.125 z_5) keep or improve performance relative to deep-only when the support draw changes? Post-hoc development evidence, not a final claim. Lambda 0.125 was selected on AD2 public 2-shot seed 0 and is frozen.

Arms (paired within dataset, seed, shot, support): linear0125 (ablation_shallow_mode all, gamma 0.125) and deep-only (ablation_shallow_mode off). Fixed: frozen DINOv2 ViT-L/14 layers 5/11/23, FP16 autocast, resolutions 448/672/896/1120, cap256/source, four bags, q95 tail scaling, CQ99, top-1% mean image score from the final map, k9 low-resolution noise guard. No training, maps or weights saved. Ground truth is evaluator-only.

Scope: seeds 1 and 2; shots 2 and 16; AD2 public full8, MVTec AD full15, VisA full12 (seen development populations). AD2 uses the existing E1 seed manifests (seed1/2 support_2/16), shared with the three-seed baselines in Table 4. MVTec AD and VisA use no fixed manifest: the pipeline draws the support deterministically from sorted train/good with the config seed, so both arms receive the identical support; these draws are not paired with baseline methods. Budget: 8 configs per dataset, 280 class runs.

Metrics: AD2 AUPRO@0.05 (primary), pAUROC@0.05, image AUROC; MVTec/VisA image AUROC and full-FPR pixel AUROC. Report per-seed macro, mean and sample SD over seeds 0-2 (seed 0 from the existing linear0125 and deep-only runs), paired class-level win/loss, and no subset selection.

Continuation: after each GPU's VisA calibration-factorial worker ends, the same GPU runs the AD2 calibration factorial (50 configs x 8 classes = 400 logical runs, plan in scarf_calibration_factorial_seed0_fp16_20261009/plan.json) under the same immutable factorial protocol. Runtime or provenance failures invalidate the cell; retain failed logs and resume with the same configuration. Timeout 10,800 s per class. Disjoint class shards per GPU, one worker per GPU, existing container hun_fsad_tta_012.

