# Local relation development protocol — 2026-09-19

Authorized scope: local RTX4090 and existing dsba3 container, method implementation,
controlled experiments, conditional replication and generalization. No deployment,
container replacement, map/checkpoint outputs, or unrelated worktree changes.

Hypothesis: appearance-nearest normal patches can fail to explain the query's
local feature relationships. A normal-calibrated relational penalty can recover
such anomalies without rigid image registration. Prior QR/RC/VR/CG branches
modified translations/reference choices, not the local star edge distribution.
UniVAD (CVPR2025) already models components/graphs; DCP-SFR (CVPR2026) preserves
defect cues and refines structure. This is a tested candidate, not a novelty claim.

## Fixed method boundary

RSG3 frozen DINOv2 ViT-L/14; layers 5,11,23; scales 448,672,896,1120;
16 normal supports, seed 0; random 256 tokens/source, 4 bags, kNN=1;
tailscale, shallow residual tau q=.99, CQ percentile99/q1/no dilation.
Identical shared pipeline, support/query hashes and streaming metric contract.
Trainable parameters: zero. New branch retains eight scalar local edges per
sampled token, not dense reference feature maps. No fitted backbone changes.

LocalRelation: sorted 8-neighbor cosine star edges at layer11. Top5 appearance
candidates; each receives appearance opportunity cost plus positive relation
discrepancy above normal nearest-match q95. Take minimum and add with weight .25
to unchanged RSG score before CQ. Normal q95 is fitted per scale/bag with whole
source exclusion; one-shot uses token exclusion (not independent image evidence).
This is local feature geometry, NOT joint correspondence optimization or SE(3)
invariance. Sorting removes neighbor direction, so it cannot detect every spatial
rearrangement. Null permutes spatial features before edge construction and restores
token centers; independent normal recalibration, unchanged appearance banks/RNG.

## Initial budget and gates

A=reference RSG3; B=LocalRelation weight0 exact control; C=actual local relations;
D=shuffled relation null. 4 x full8 x seed0 = 32 class evaluations. Local can/fabric,
dsba3 remaining six classes distributed across four already-exposed GPU UUIDs.
Stages sequential per lane. One class timeout 3600 seconds. Safe runtime retries
only after diagnosis; failed execution is not negative method evidence.

All data are seen development test_public, not held-out final evaluation.
Primary AUPRO@.05 increases; secondary pAUROC@.05; oracle F1 diagnostic only.
No GT enters fit/predict. Record costs, activation rates and class-level changes.
B must reproduce A (numerical tolerance 1e-6 maps in synthetic/real smoke; metrics
1e-5 absolute on matched full evaluation). C must beat B and D on primary metric
to justify expansion; first single seed alone cannot establish significance.
If C has no positive macro effect or the null equals/exceeds it, stop this candidate
after diagnosis and retain negative evidence. Do not tune against all eight labels
until a success appears. At most one justified correction after a diagnosed defect.

Conditional replication: freeze config, paired A/C/D seeds1-4 (96 additional class
runs), together with seed0. Practical adoption proposal: mean paired AUPRO >=+1pp,
pAUROC >=-0.5pp; report all paired differences/uncertainty, not a range-as-p-value.
Only if passed, separate CQ/smoothing and calibration controls and expand shots
1/2/4/8 and datasets under a recorded bounded follow-up budget. Unpassed gates
make those runs unnecessary, not unfinished. No paper-ready final claim in this task.

## Initial execution provenance

Existing containers: local `fsad-tta-local-bcma`, dsba3 `hun_fsad_tta_012`.
Every lane runs A -> B -> C -> D on the same assigned GPU. Local GPU access was
restored by restarting the idle existing container; no container was replaced.

| Host / GPU UUID | Classes |
|---|---|
| local / GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b | can, fabric |
| dsba3 / GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f | fruit_jelly, wallplugs |
| dsba3 / GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae | rice, walnuts |
| dsba3 / GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 | sheet_metal |
| dsba3 / GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62 | vial |

Initial immutable remote snapshot:
`075cb6a0935be9b47171fbfc9eca894657975871333aec0041deb2f74abc6b1b`.
Shared package SHA256:
`47b5f00ebd8639e481bc2d345923c96224957955a4948a6530cc22211a6045c2`.
Local torch is 2.6.0+cu124; remote is 2.1.2+cu118. Comparisons are paired within
each class's execution environment, not between unpaired machines. Actual
backbone state/repository hashes and package versions are in every manifest.
Remote logs remain under `/workspace/results_remote/local_relation_20260919`;
only compact quantitative/config/manifest/status evidence is pulled locally.

## Conditional replication execution — declared before launch

The completed seed0 full8 gate passed: C exceeds B and D on AUPRO. The
[screening report](../../results/comparisons/local_relation_20260919/comparison.md)
records the small effect and class concentration; neither changes the gate.
Run A/C/D seeds1–4 without changing any setting except seed (96 class runs).
Every local seed runs A -> C -> D on can/fabric using the same local GPU above.
Remote lane0/1/2/3 use the corresponding GPU UUIDs above and seed1/2/3/4,
respectively; each lane runs A -> C -> D on all six remaining classes.
This preserves within-seed/class/GPU pairing and permits disjoint seed lanes.
Local seeds run sequentially; remote seed lanes run concurrently, one job/GPU.

Roots: local `/results/local_relation_replicates_20260919/seed_N/lane_0`;
remote `/workspace/results_remote/local_relation_replicates_20260919/lane_N`.
Use the same existing containers and a new immutable local-mode remote snapshot;
no git push, environment upgrade or backbone/method change. Each completed seed
will be combined as one full8 A/C/D comparison through the generic collector.
The five-seed adoption gate above determines whether further controls/generalization
are necessary; do not stop replication early after seeing an unfavorable seed.

## Evidence corrections established before this run

Historical RSG3 seed0 full8=59.14576/77.10613; five-seed AUPRO=59.47789±0.60012.
DINOv3 66.62217 is partial4; full8=55.86512/82.35507. R0 used two scales, not one.
CB5 and TC1 are different branches, not successive ablations. TB1 57.74275 ->
TB1_CQ 58.67094 is the direct CQ contrast; RSG3 adds +.47482 AUPRO with pAUROC loss.
Main fewshot table includes inconsistent values: v2 4shot raw AUPRO=57.3349
(rounded; final exact raw value should be used by collector). Existing paper
tables are not used as quantitative inputs. Full artifact audit is already in
the conversation; do not overwrite unrelated historical reports.

## Additional related-work check during execution

This is a post-launch interpretation check, not a change to the method or gates.
The existing project bibliography already includes AnomalyDINO (WACV2025).

- [Matching Local Self-Similarities across Images and Videos](https://www.wisdom.weizmann.ac.il/~vision/SelfSimilarities.html),
  Shechtman and Irani, CVPR2007: matches compact local self-similarity descriptors
  across images/videos. The problem is correspondence/detection, not normal-only
  few-shot industrial anomaly detection; it establishes the descriptor lineage,
  not an AD2 performance baseline or an identical implementation.
- [MIND: Modality independent neighbourhood descriptor for multi-modal deformable registration](https://doi.org/10.1016/j.media.2012.05.008),
  Heinrich et al., Medical Image Analysis 2012: constructs local structural
  descriptors from within-image patch similarities and compares descriptors for
  registration. Different data, features and objective; not evidence that this
  sorted DINO-feature residual is equivalent or that it improves anomaly detection.
- [UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection](https://www.openaccess.thecvf.com/content/CVPR2025/html/Gu_UniVAD_A_Training-free_Unified_Model_for_Few-shot_Visual_Anomaly_Detection_CVPR_2025_paper.html),
  Gu et al., CVPR2025: uses normal references without domain-specific training,
  component-aware patch matching and graph-enhanced component modeling. A generic
  graph/structure/few-shot claim is therefore not enough to establish novelty.

Primary author/publisher descriptions and indexed abstracts/section snippets
were checked. Direct PDF/full-page fetches were blocked or timed out; this is
a bounded conceptual check, not an exhaustive formula-level novelty audit.
Inference: any contribution claim must distinguish the support-conditioned
calibration and residual matching rule, demonstrate a relevant failure mode,
and pass controls. The backbone is frozen: these runs do not measure or repair
its intrinsic geometry, and do not establish global registration or SE(3)
invariance. Real/null penalties can have different magnitudes despite the same
q95 calibration; causal locality claims need further calibration controls if
the candidate passes the performance/adoption gates.

## Execution closure — 2026-09-19

All triggered evaluations completed: 32 screening + 96 paired replications,
without failures. The original practical adoption gate failed; conditional
CQ/calibration, shot and dataset follow-ups were therefore not launched.
No method defaults changed. The [final report](../../results/comparisons/local_relation_20260919/comparison.md)
contains five-seed effects, class concentration, matched-condition/transport
audits and preserved canonical-ledger provenance. This closure records the
decision; it does not revise the predeclared hypothesis, budget or thresholds.
