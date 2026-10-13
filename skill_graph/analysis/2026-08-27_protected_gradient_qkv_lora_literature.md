# Protected-gradient QKV-LoRA literature and method decision

Date: 2026-08-27  
Scope: support-only DINOv2 QKV-LoRA under shifted-assumed and synthetic-anomaly
augmentation

## Decision

The next bounded experiment uses asymmetric protected PCGrad.  The existing
synthetic-anomaly preservation gradient is treated as the protected reference;
only a conflicting nuisance-removal gradient component is projected away.
The backbone, LoRA placement/rank, loss terms and weights, augmentation stream,
optimizer, scoring path, and inference architecture remain unchanged.

This is a mechanism diagnostic, not a claim that PCGrad is novel or sufficient
for anomaly detection.  The project-specific question is whether the negative
transfer observed in a real QKV-LoRA adapter is an optimization conflict that
can be removed without family routing or a score bypass.

## Evidence mapping

### Gradient conflict and asymmetric protection

- [PCGrad, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)
  defines negative task-gradient alignment as conflict and removes the
  interfering component by projection.  It directly supports changing the
  update rule without changing the model or inference path.
- [A-GEM, ICLR 2019](https://arxiv.org/abs/1812.00420) uses a protected
  reference gradient and projects a conflicting current-task update.  Its
  asymmetry motivates leaving the anomaly-preservation gradient untouched and
  projecting only nuisance correction.
- [CAGrad, NeurIPS 2021](https://arxiv.org/abs/2110.14048) and
  [Gradient Vaccine, ICLR 2021](https://openreview.net/pdf?id=F1vEjWK-lH_)
  provide broader multi-objective alternatives.  They are not selected for
  this first experiment because they introduce a symmetric compromise or an
  additional target-similarity/calibration choice.

### Synthetic anomaly evidence and anomaly leakage

- [SimpleNet, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.html)
  supports feature-space synthetic anomalies as useful supervision in a
  normal-data setting.
- [DecAD, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_DecAD_Decoupling_Anomalies_in_Latent_Space_for_Multi-Class_Unsupervised_Anomaly_ICCV_2025_paper.html)
  explicitly motivates separating normalizing/reconstruction behavior from
  anomaly evidence to avoid anomaly leakage or erasure.
- [Learning without Forgetting, ECCV 2016](https://arxiv.org/abs/1606.09282)
  is relevant to frozen-reference preservation, but adding a second teacher
  distillation target would be a larger method change than needed for the
  first conflict test.

### Adapter/subspace alternatives

- [O-LoRA](https://arxiv.org/abs/2310.14152) protects previous-task adapter
  subspaces.  It is not the first choice here because the protected object is
  a synthetic anomaly response, not an already identified previous-task
  parameter subspace; defining that subspace would introduce a second
  scientific change.
- [DINOv2 upstream](https://github.com/facebookresearch/dinov2) motivates a
  strong frozen visual reference, but does not itself solve nuisance/anomaly
  identifiability.

## Why the current evidence selects gradient protection

The Oracle-Family-Expert run established all of the following simultaneously:

1. LoRA gradients and trained adapters are nonzero.
2. Removing cross-family averaging improves hard-null margin from `-0.1439931`
   to `-0.0832106`.
3. Stronger correction direction can coincide with anomaly collapse, notably
   for `can/J1` and `vial/J1`.
4. Strict synthetic contribution transfers to held G2 in only `1/12` cells.
5. A score-only firewall and oracle family routing are closed branches.

This pattern is more consistent with negative transfer inside a shared update
than with an inactive adapter or a threshold calibration defect.

## Frozen one-change equation

Let `L_N` be clean nuisance correction plus clean identity, and `L_P` be
synthetic anomaly correction, recomposition, anomaly identity, and RN guard
with all predecessor weights unchanged.  For all LoRA parameters concatenated
as a single logical gradient vector:

`g_N' = g_N - min(dot(g_N,g_P),0)/(||g_P||^2 + 1e-12) * g_P`

and the optimizer receives `g_N' + g_P`.

The exact preregistration, controls, traces, and kill rules are in
`experiments/2026-08-27_protected_pcgrad_qkv_lora_stage_minus1/preregistration.md`.

## Critical limitation

Raw gradient projection does not guarantee that AdamW's preconditioned,
momentum-bearing parameter delta remains aligned with the protected objective.
The experiment therefore records both raw post-projection alignment and the
actual parameter-delta alignment.  If the latter remains conflicting, the
result is a mechanism failure rather than permission to tune the loss or hide
the violation.
