# SCARF bounded negative shallow residual - 2026-10-06

Hypothesis: letting the shallow residual contribute a bounded negative term
below the support threshold tau lowers deep-layer false positives where the
shallow layer is confidently normal, improving AUPRO@0.05 on AD2 public.

Method change (only): residual_spike fusion
deep + gamma * [relu(s - tau) - alpha * (1 - exp(min(s - tau, 0)))].
alpha = 0 reproduces current SCARF exactly; the negative term is bounded in
[-gamma*alpha, 0] and monotone. Prior evidence: unbounded linear (all_ungated,
equivalent to linear alpha=1) lost 5.7 pp (MVTec 1-shot) and 10.8 pp (VisA
1-shot) AUPRO, so only bounded alpha is tested.

Fixed: SCARF FP16 autocast, cap256, scales 448/672/896/1120, DINOv2 ViT-L/14
layers 5/11/23, tau support quantile 0.99, gamma 1, same VOID support
manifests and evaluator as the completed FP16 three-seed suite.

Scope: AD2 test_public full8, seed 0, shots 1/4/16, alpha in {0, 0.1, 0.25,
0.5}; 12 configs x 8 classes = 96 class runs. alpha=0 is the in-suite control
and must match the existing FP16 seed0 values.
Split: seen development; GT/labels used by the evaluator only.
Metrics: primary AUPRO@0.05, secondary pAUROC@0.05, I-AUROC; higher is better.
Decision: extend to seeds 1/2 only when an alpha beats alpha=0 by more than the
existing FP16 seed SD at the same shot. Maps/weights off.
