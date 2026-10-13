# Oracle-Family-Expert LoRA Stage -1 preregistration

Date: 2026-08-27  
Protocol: `oracle-family-expert-lora-stage-minus1-20260827`  
Status: `PREREGISTERED / NO NEW-PROTOCOL RESULTS INSPECTED`  
Seed: exactly one, `20260826`

## Negative evidence intake

TARCD-LoRA S1--S3 showed that a single query-backbone LoRA can optimize its
support objective but does not generalize a correction better than the
shift-family static mean.  S3 fixed the zero-start gradient pathology, yet its
held correction cosine was `0.4186983`, its hard-null margin was `-0.1439931`,
and strict synthetic contribution passed `0/12` cells.

SCOPE-LoRA Cycle 1/2 then isolated a real rank-4 normal-correction signal, but
the score bypass had zero global margin over PT and did not beat the static
hard null.  Its preregistered outer loop is closed.  This experiment is a new,
separately named upper-bound diagnostic and does not alter CEB.

## Sole question and single architectural change

Can removal of cross-family averaging make actual backbone LoRA training beat
the static shift-family correction while retaining held synthetic anomaly
evidence?

The sole mechanism change from TARCD S3 is **oracle family factorization**:
instead of one adapter trained across J0/J1/J2, train three independent
adapters, each on exactly one known training shift family.  The oracle family
label is privileged and evaluation-only routing is not deployable.  All other
choices remain fixed:

- DINOv2 ViT-L/14 frozen except fused-QKV LoRA in zero-based blocks 8--11;
- rank `4`, alpha `8`, zero-B initialization, no dropout;
- S3 target-normalized vector-Huber correction loss;
- S2 paired clean/anomaly correction, cross-recomposition, identity, and RN
  guard coefficients;
- S3 randomization namespace
  `tarcd-lora-s2-dcr-stage-minus1-support-adapted-20260826`, so shift,
  anomaly, mask, control, and schedule realizations remain paired;
- AdamW `1e-4`, zero weight decay, bf16 forward, fp32 loss, clip `1.0`;
- exact support, held bases, generators, masks, scoring, and controls.

For one object/family expert, the exhaustive schedule is
`16 support bases x 3 fit synthetic generators = 48` updates.  Across three
experts, each object still receives 144 candidate updates, but parameters are
not shared between families.  There is no early stopping or checkpoint
selection.  Each family schedule preserves the relative order obtained by
filtering that family's 48 rows from S3's frozen shuffled 144-row schedule.

## Custody

- Dataset: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Objects: `can`, `fabric`, `vial`, `walnuts`.
- Exact SuperAD-16 support manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Fit: only the exact 16 `train/good` support paths per object and their
  deterministic J0/J1/J2 plus G0/G1/G4 descendants.
- Evaluation: the same four support-disjoint `train/good` bases and held
  `G2_structural_scratch` descendants used by TARCD and SCOPE.
- Oracle shift labels may select the family expert but may not use anomaly
  presence, mask, generator, object shortcut, filename condition, benchmark
  split, or score.
- `TESTpublic`, private test, GT, query updates, query-memory updates, and
  benchmark thresholds remain sealed.

## Arms and measurements

Each object/family task retains TARCD-S3's candidate and controls:

- `D2_DCR_RN_GUARD`: oracle-family expert candidate;
- `D0_DIRECT_CLEAN` and `D1_DIRECT_RECOMPOSITION`: synthetic/RN ablations;
- `M0_ENDPOINT_KD`, zero-init identity, shuffled correction, random
  norm-matched LoRA, privileged `N3_STATIC_MEAN_ORACLE`, frozen teacher, and
  paired-teacher oracle.

Record correction cosine/norm, shifted-normal q99 reduction, anomaly
cosine/norm, RN topology, hard-null margin, synthetic and RN contribution,
adapter hash/norm, first/last/mean losses and gradients, and compact fixed-step
training traces.  Traces are written at steps `1,12,24,36,48`; they monitor
optimization trend only and cannot select a checkpoint.

## Preregistered gates

Aggregation is cell -> three-family object median -> four-object median.

`PROMISING_ORACLE_FAMILY_LORA_SIGNAL` requires:

1. all 12 cells structurally valid, finite/nonconstant, exact zero-init and
   duplicate-RN errors within prior tolerances, exact custody hashes, 48-step
   traces at the five frozen checkpoints, and no forbidden access;
2. global candidate-minus-best-hard-null correction-cosine margin `>=0.05`
   and at least three of four object medians strictly `>0`;
3. global shifted-normal q99 reduction `>=0`, with every object `>=-0.05`;
4. every object median anomaly cosine `>=0.85` and anomaly norm ratio in
   `[0.70,1.40]`;
5. strict synthetic-contribution gate passes in at least `6/12` cells.

`PROGRESSIVE_ORACLE_FAMILY_SIGNAL` is a non-claim continuation signal only. It
requires all structural and anomaly guards, global q99 reduction `>=0`, no
object below `-0.05`, at least `3/12` strict synthetic-contribution cells, and
a global hard-null margin `>=-0.094`.  The last threshold is a frozen improvement
of at least `+0.05` over S3's `-0.1439931`, not success over the hard null.

Otherwise the verdict is `INSUFFICIENT_ORACLE_FAMILY_SIGNAL`, which kills
family-routed/hypernetwork LoRA as the next mechanism.  A progressive result
licenses exactly one subsequent architectural repair; it does not license a
rank, loss-weight, learning-rate, threshold, seed, or generator-strength sweep.

## Remote execution and other-user safety

- Use only dsba3 host GPUs `0,1,2,3` and the project-owned fixed container
  `hun_fsad_tta_012`.
- Mapping is fixed as `can/fabric/vial/walnuts -> 0/1/2/3`; each GPU executes
  J0, J1, and J2 sequentially.
- Before snapshot creation or launch, abort if any selected host GPU has a
  compute process, more than 1 GiB allocated, or utilization above 10%.
- Do not inspect, enter, stop, restart, rename, mount, or modify any other
  user's container, process, file, volume, or job.  Do not use dsba2 until a
  project-owned isolated container and project-owned data path exist there.
- Run from one immutable snapshot with exact GPU UUID mapping, offline DINO
  cache, remote focused tests, actual CUDA model smoke, and compact artifacts
  only.  Dense feature/score/anomaly maps are forbidden.

The run stops after the support-only aggregate.  No public branch is opened.
