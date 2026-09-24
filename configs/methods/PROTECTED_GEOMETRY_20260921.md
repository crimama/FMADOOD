# Protected normal-contrast geometry — 2026-09-21

User authorized implementing the proposed four-arm controlled experiment.
This is a development candidate, NOT a novelty or defect-preservation claim.
Related-work sanity check: NFAD (arXiv2608.29112) already uses perturbation
nuisance subspaces; SPARC (2608.18585) uses training-free spatial calibration.
The repository's GuardedMetric already explores trained bounded metrics with
synthetic-defect protection. This candidate instead tests a closed-form,
normal-local-contrast protection constraint without synthetic defects/gradients.
These distinctions are hypotheses to evaluate, not sufficient novelty by themselves.

## Hypothesis and operator

Normal-only nuisance contraction may damage local distinctions needed by AD.
Protecting normal adjacent-patch contrasts stable across photometric views may
improve shift AUPRO relative to unprotected contraction and existing QR1.
Normal contrast is a proxy, not a known defect direction or preservation proof.

For each layer/resolution, extract unit features from the selected support and
two deterministic views: RGB gain(0.7,0.7,0.7), RGB gain(0.8,1.0,1.2), uint8
round/clip. No geometric warp. These finite perturbations do not represent all
deployment shifts. Use at most256 evenly spaced native-grid patches per image
for uncentered displacement second moments, equal image/view weighting.
U: top16 nonzero eigenvectors of nuisance moment. Numerical rank tolerance
max(1e-10,largest_eigenvalue*1e-6); no energy/rank sweep.

V: top32 eigenvectors of the moment of unit-normalized clean horizontal/vertical
adjacent differences that have cosine>=0.9 AND norm ratio in[0.5,2] under BOTH
views. Difference norms must exceed1e-5. Cap256 evenly spaced eligible edges per
image; equal contributing-image weighting. Never treat rejected edges as defects.

Unprotected: A=U. Protected: A=(I−VV^T)U. Apply
T=I−0.5AA^T to BOTH support/query unit features, THEN unit-normalize for the
existing cosine scorer. Before final normalization eigenvalues(T)∈[0.5,1] and
TV=V (up to numerical error). Do NOT re-orthonormalize A: its singular values
retain suppression of overlap with V. No nuisance basis→identity; protected
mode with no reliable V→identity (explicit fallback, not a successful protection).
All original bank sample identities/seed/counts are retained; support MAD scales
are re-estimated in the transformed space. This is a declared method condition,
not an unchanged-feature comparison. Query adaptation is disabled in C/D.

Final normalization does NOT preserve exact protected contrasts. Record overlap,
effective contraction, linear protected-subspace error, transformed nuisance
energy, and local-pair cosine-distance ratio AFTER normalization on support
evidence. These fitting diagnostics are not independent generalization evidence.
Persistent state: small fitted factors/statistics only; no gradients, optimizers,
backbone weights, query-conditioned state, dataset/GT inputs to the method.

## Fixed comparison and budget

PG_A: original QR0, geometry off. PG_B: original QR1, geometry off.
PG_C: geometry on, protection off, QR adaptation off.
PG_D: geometry on, protection on, QR adaptation off.
64 evaluations:4arms×nested1/16shot×full8, seed0, SD_support manifests.
Frozen DINOv2-L layers11/23,448/672,256 bank tokens per source, original R1 MAD
mean fusion. All arms use the same class/method entry point, common pipeline,
source support, preprocessing and public query data. A/B must match original
implementations in unit tests; C/D differ only in protected versus unprotected A.
No synthetic anomaly images/labels, no extra normal calibration used by fitting.

Existing real_acquisition evaluation: all public conditions; primary shift_1,
regular safety, others secondary.16 extra normal calibration images excluded
from the entire support16 pool are evaluator-only (not a strict one-image
deployment-budget claim). Fixed operating points, LOO threshold sensitivity,
condition AUPRO@.05/pAUROC@.05 and per-image margins remain diagnostic.
Data are repeatedly seen development, NOT a sealed final test. No maps/weights.

## Prospective adoption rule

At BOTH shots require D−B and D−C shift_1 macro AUPRO>=+0.5pp, each positive in
>=5/8 classes, and positive after excluding the largest-gain class separately.
Require D regular AUPRO no worse than A or B by more than0.5pp and shift_1
pAUROC no worse than B or C by more than0.5pp. Inspect all class outcomes.
Protection-specific support requires lower absolute post-normalization local
contrast distortion for D than C in>=5/8 classes at both shots, alongside the
performance gate; fitting evidence alone is insufficient. Report raw contraction
and fallbacks so an identity-like operator is not mistaken for a new mechanism.
Mean defect/normal margin is NOT an optimization or adoption target.
Passing prompts a separate replication decision, not automatic seeds/scorers.
Failure stops this candidate configuration; do not silently tune on outcomes.

## Execution boundary

User re-enabled local experiments before launch. Use existing local
fsad-tta-local-bcma container/GPU UUID GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b
for can/fabric, and existing dsba3 hun_fsad_tta_012 physicalGPU2 UUID
GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 for the other6 classes. All four arms
and both shots of a class stay on the same device. Existing Python3.11/Torch2.6
runtimes; one worker per device, sequential generic experiments.py run.
Validate remotely before full run. Never change snapshot during a running suite.
No code push, provisioning, container replacement or artifact cleanup.
Remote root /workspace/results_remote/protected_geometry_20260921.
Use generic collect for canonical Markdown/CSV; preserve original artifacts.
