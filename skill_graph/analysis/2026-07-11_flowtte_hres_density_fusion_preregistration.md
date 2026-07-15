# High-Resolution Density + Reliability-Gated Fusion Preregistration

Date sealed: 2026-07-11
Status: frozen before any high-resolution density head is trained or any AD2
public map is generated
Supersedes for this branch: raw DARC ladder (`R1`) is killed; this document
defines the learned/calibrated replacement recommended by the DARC continuation
assessment.

## 0. Why This Branch Exists

The retained method (`H+DVT` MLP flow, fixed 16-shot, 672/patch-16) is a strong
continuous ranker whose AUROC is near reported SuperADD context but whose F1 is
far lower. Every prior attempt to close that gap by changing the flow
architecture, adding CLS/register/coordinate conditioning, hard
foreground/background suppression, local contrast, morphology, or flow
hyperparameters failed or was too small. The single reproducible positive lever
in the whole record is **resolution**: `can` median defect bbox is native
~`15x13` px, which is sub-token at 672/patch-16, and grid-compatible oracle F1
climbs monotonically `0.355 -> 0.465 -> 0.511 -> 0.593 -> 0.707` from
`672 -> 896 -> 1120 -> 1344 -> 2048`.

The DARC-v1 attempt to exploit this lever failed, but its failure was localized:
the **hard coordinate-local window plus registration plus robust reconstruction**
destroyed ranking (`L0` added ~1.74M false-positive pixels over `G0`), while
native-resolution **global** `G0` remained the strongest of the four raw arms.
Two things were never fairly tested and are the subject of this preregistration:

1. a **learned high-resolution normal-density score head** instead of a single
   raw cosine residual;
2. the **reliability-gated coarse-to-fine fusion** whose confidence field was
   defined in the DARC design (`max(e_coarse, confidence * e_micro)`) but never
   implemented or frozen, and was therefore excluded from every executed pilot.

## 1. Claim, Evidence, Boundary, Positioning

- **Claim**: The `H+DVT` coarse branch's F1 deficit is caused by sub-token cue
  dilution at 672 and by unreliable high-resolution residuals, not by the flow
  projection or by missing global context. A high-resolution **global** normal
  density head, fused with the unchanged coarse branch through a normal-only
  calibrated reliability gate, improves operating-point precision and small-defect
  localization class-agnostically, without hard spatial exclusion.
- **Evidence**: the resolution oracle-F1 gradient above; no-NF identity distance
  (`0.832461`) nearly ties NF (`0.836739`) on the 672 grid, so the flow adds
  almost nothing where cues are already token-resolved; DARC `G0` beat every
  hard-local arm; DARC explicitly recommends retaining unrestricted high-res `G0`
  and training a normal-only high-res density/score head.
- **Boundary**: MVTec AD2 public GT is shadow/development data and has already
  influenced this design; it cannot open a final claim gate. All density-head
  training and all calibration use only normal images (AD1 `train/good` for
  mechanism gates; support/held-out normals for AD2 shadow cells). No AD2-public
  label is used to tune resolution, gate, head, threshold, or morphology. A final
  claim requires untouched/private labels.
- **Positioning**: the novelty candidate is the combination of a learned
  high-resolution normal-density branch with the unchanged coarse H+DVT semantic
  branch under a frozen reliability gate. It is **not** high resolution alone,
  **not** registration, **not** reconstruction, and **not** a new flow topology
  on the 672 grid.

## 2. Frozen Method

### 2.1 Coarse branch (unchanged anchor)

Exactly the retained reference, used as a no-harm anchor and never retuned:
`dinov3_vith16plus`, layers `[7,15,23,31]`, `layer_norm_mean`, DVT
`position_mean alpha=1.0`, patch-wise MLP flow, fixed 16-shot support JSON,
`density_weight=0.25`, latent NN distance. Its raw continuous map is `e_coarse`.

### 2.2 Fine branch (new, learned, normal-only)

1. **Extraction**: native high-resolution tiled H+ layer-7 features. Tiling,
   stride, and paired-resize coordinate convention are inherited verbatim from the
   frozen DARC coordinate contract (native `512x512` crops, half-pixel affine
   maps, patch center `16*(index+0.5)-0.5`, right/bottom remainder crop only).
   Matching remains **position-free global** over the support normal token pool;
   there is no coordinate-local window, no registration, and no reconstruction.
2. **Density head**: a normal-only score head over layer-7 tokens producing a
   calibrated residual `e_micro`. The head is trained on normal tokens only; its
   exact form (small MLP flow / normal-density regressor) and training budget are
   frozen in the run's method hash before any AD2 map is generated. Training data
   for a cell is that cell's memory-fold normal tokens only; held-out normals of
   the fold are never in the head's training set.
3. **Micro resolution set**: `{896, 1120, 1344}` as the primary ladder; `2048` is
   a capacity probe reported separately for cost. The resolution set is fixed
   before AD2 execution and not selected on AD2 F1.

### 2.3 Reliability gate and fusion (the previously-undefined piece)

The gate is defined and frozen on **normal-only** statistics before any AD2 label
is seen:

```text
confidence[token] = g( support_NN_dispersion[token],
                       fine_head_calibration[token] )
```

- `support_NN_dispersion` is the per-token dispersion of the token's distances to
  its K nearest support normals (multimodal / high-variance normal appearance =>
  low confidence). This is a reliability weight, **not** a hard
  foreground/background mask and **not** a subtractive position correction, both
  of which are already-killed directions.
- `fine_head_calibration` is the leave-one-image-out empirical upper-tail
  position of `e_micro` under normals of the fold, so that a high `e_micro` on a
  token whose normals are themselves noisy is discounted.
- `g` maps to `[0,1]` with monotone, frozen coefficients fit only on normal
  tokens. No AD2 label, no anomaly pixel, and no oracle threshold participates in
  fitting `g`.

Fusion is the frozen DARC rule at native grid:

```text
e_fused = max( e_coarse, confidence * e_micro )
```

The fused threshold, when a fixed-F1 diagnostic is reported, uses
image-disjoint normals only.

## 3. Evaluation and Comparability

- Target: MVTec AD2 single-image, all 8 public objects, full
  `test_public/good,bad`; AD2 public = shadow.
- Resource protocols reported separately and never mixed: `P16-random`,
  `P16-superad`, `Pfull`. The genuine few-shot claim path is `P16-random`.
- Primary metrics: native-grid raw continuous `seg_AUROC_0.05`, AP, oracle
  max-F1, component recall; fixed normal-threshold F1 is a transductive
  diagnostic only; one identical fixed `shared-superadd-v1` morphology profile is
  a controlled comparator, never the claim.
- SuperADD Table 1 (`0.8393/0.6261`) stays `comparable=false` external context:
  P-full, good-excluded population, train-derived threshold/morphology on
  `H/4 x W/4`. No `delta_vs_superadd` superiority number is computed.
- The coarse anchor must be re-emitted in the same run so all deltas are paired
  and same-code.

## 4. Gates

- **Gate A (mechanism, AD1 synthetic, normal-only development)**: on the frozen
  `darc-line-cue-v1` thin cues, the fine density head's held-out-normal p99.9
  residual is at least 20% below the raw cosine residual it replaces, while
  retaining at least 90% of thin-cue AP. Fails => the learned head does not beat
  raw residual and the branch stops.
- **Gate B (resolution, global, AD1 synthetic)**: paired multi-resolution global
  `G0`-style density map beats 672-equivalent on AP with a positive paired
  image-bootstrap lower bound and control pAUROC loss `<= 0.005`. Confirms
  resolution is exploitable **without** hard-local constraint. This is the pure
  check the raw ladder never isolated.
- **Gate C (fusion no-harm, AD2 shadow)**: on the frozen shadow objects
  (`can, fabric, vial, wallplugs, rice`), `e_fused` mean oracle F1 `>= +0.03`
  over the coarse anchor, mean pAUROC loss `<= 0.002`, no class F1 loss over
  `0.02`, and gains on `can` plus at least two other gap classes. The confidence
  gate and resolution set are frozen from Gates A/B, not re-fit on these objects.
- **Gate D (claim, untouched/private only)**: matched FlowTTE and SuperAD, three
  `P16` seeds and `Pfull`, stable fixed-threshold F1. No AD2-public retune.

## 5. Stop Rules

- Gate A failure kills the learned-density-head novelty; resolution utilities and
  the coarse anchor remain.
- Gate B failure kills resolution-as-lever and, with the raw-ladder kill already
  recorded, closes the high-resolution family; only the coarse branch survives.
- Any of: AD2-public retuning, unequal normal access across arms, bad-only-only
  gains, oracle-post-processing dependence, or a confidence gate fit on AD2
  labels invalidates the claim regardless of metric gain.
- `can` must not be tuned in isolation; the fine branch is justified only as a
  class-agnostic small-defect-resolution fix that also helps other gap classes.

## 6. What This Preregistration Deliberately Excludes

- Hard coordinate-local windows, registration/RANSAC alignment, and robust
  reconstruction (killed by the DARC full-`can` pilot).
- CLS/register/coordinate conditioning inside any flow (repeatedly baseline or
  below).
- Transformer/Conv2D flow topologies on the 672 grid (smooth the field, weaken
  local contrast).
- Hard foreground/background suppression, local contrast, and subtractive
  position calibration on the final score field (harmful).
- Morphology or threshold policy as the method claim (controlled comparator only).

## 7. Open Risks

- High-resolution tiled extraction is expensive; the `2048` probe is cost-only.
- The confidence gate `g` must be fully frozen on normal-only statistics before
  Gate C; any leakage of AD2 labels into `g` voids the fusion claim.
- The SuperADD row remains external context, so even a strong Gate C result is
  shadow evidence, not a matched-baseline superiority claim.
