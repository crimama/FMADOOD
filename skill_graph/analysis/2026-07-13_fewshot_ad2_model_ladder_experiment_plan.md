# Few-Shot AD under Distribution Shift: AD2 Model-Ladder Plan

Date: 2026-07-13  
Status: design frozen; execution not started  
Protocol: `fmad-experiment-protocol`

## 1. Claim, boundary, and positioning

### Claim

The primary task is few-shot anomaly detection under distribution shift, with
single-image MVTec AD2 as the main benchmark. The Basic model must beat
SuperAD under a paper-aligned DINOv2 and 16-reference contract. Stronger
backbones are reported as scaling variants: Ours+ uses DINOv2-R and Ours++
uses DINOv3-H+.

### Evidence target

- Basic: same-condition superiority over SuperAD-16 on pixel
  `AUROC_0.05` and segmentation F1.
- Ours+: improved few-shot performance from a stronger DINOv2-family
  representation with the detector and data budget frozen.
- Ours++: with only 16 normal references, approach the reported full-normal
  SuperADD pixel `AUROC_0.05`.

### Boundary

- There is no `Ours++ Full` model and no new full-normal candidate run.
- SuperADD remains an external full-normal reference, not a same-condition
  few-shot baseline.
- Until the F1 gap is closed, Ours++ may be described as near SuperADD only
  for continuous pixel AUROC, not for complete segmentation performance.
- `full supervision` is not used as terminology; the correct contrast is
  `few-shot normal references` versus `full-normal training data`.

### Positioning

1. fair few-shot method comparison through Basic versus SuperAD;
2. backbone scaling through Basic -> Ours+ -> Ours++;
3. data efficiency through Ours++ 16-shot versus reported full-normal
   SuperADD context.

## 2. Negative-evidence intake

This plan does not authorize an unconstrained repeat of the previous H+
hyperparameter sweep. Existing all-eight H+ evidence shows:

- coupling layers 1 and 4 both lost to the 2-layer anchor; 4 layers reduced
  F1 from `0.527427` to `0.511609`;
- 5 Flow epochs reduced F1 to `0.495342`;
- learning rates `1e-4` and `5e-4` were worse than `2e-4`;
- DVT alpha above 1, high clamp, and large density weights were harmful;
- the only repeatable narrow signals were log-det regularization near
  `1e-2..2e-2` and support brightness augmentation up to `[0.8,1.2]`.

These H+ results do not prove the same optimum for DINOv2 Basic, but they
lower the priority of deep/wide sweeps. Basic receives one bounded capacity
check; a failed 4-layer result hard-stops deeper 6/8-layer runs.

## 3. Model ladder

| Model | Backbone | Resolution | Few-shot support | Primary comparator |
|---|---|---|---|---|
| Ours Basic | DINOv2 ViT-L/14, code layers `[5,11,17,23]` | 672; sheet_metal 448 | exact SuperAD-selected 16 for the claim; random 1/2/4/8/16 for robustness | same-run SuperAD-16 |
| Ours+ | DINOv2-R | Basic resolution and frozen detector | fixed 16 and random-16 seeds | Basic; SuperAD context only when backbone differs |
| Ours++ | DINOv3-H+, layers `[7,15,23,31]` | 672; sheet_metal 448 | fixed 16 and random-16 seeds | reported SuperADD context |

The detector is Flow-LatentBank with static memory, latent 1-NN, DVT,
density, RGB guided-r8, and fixed binary morphology. LOO standardization is
disabled following the matched Basic component ablation.
After Basic tuning, detector parameters are frozen for Ours+ and Ours++.

## 4. Evaluation alignment

### Paper-aligned SuperAD claim arm

- `target_dataset=MVTec AD2 single-image`
- `data_root=/home/hunim/Volume/DATA/mvtec_ad_2`
- all eight public objects and full `test_public/good,bad`
- DINOv2 ViT-L/14, layers `[5,11,17,23]`
- resolution 672 except `sheet_metal=448`
- full `train/good` is only the selection pool
- exact same 16 DINO CLS greedy-coreset paths for SuperAD and Basic
- identical `AUROC_0.05` and best-threshold F1 evaluator
- same-run SuperAD and RN-FMLK/full-memory DINO kNN artifacts

Different support policies or backbones are separate ablations and cannot
support the strict Basic-versus-SuperAD verdict.

### Random few-shot arm

- shots `1,2,4,8,16`;
- selection seeds `0,1,2,3,4` without replacement;
- identical per-seed support paths across applicable methods;
- report mean, standard deviation, and per-object results;
- do not select hyperparameters independently for each shot or seed.

## 5. Basic hyperparameter protocol

### Stage B0: frozen anchor

The current Basic anchor is 2 coupling layers, hidden multiplier 1, 3
epochs, LR `2e-4`, clamp `1.9`, `lambda_logdet=1e-3`, tail weight `0.3`,
tail ratio `0.05`, density weight `0.25`, DVT alpha 1, static memory, and
guided-r8 plus line17/angles16 morphology.

### Stage B1: joint depth-width capacity factorial

Depth and width are scientifically important and remain explicit factors.
Evaluate the complete `3 x 2` block below instead of selecting one axis
greedily before observing the other:

| Run | Coupling layers | Hidden multiplier |
|---:|---:|---:|
| C1 | 1 | 1 |
| C2 | 1 | 2 |
| C3 | 2 | 1 |
| C4 | 2 | 2 |
| C5 | 4 | 1 |
| C6 | 4 | 2 |

Everything else remains at the B0 anchor. This estimates the paired
per-object main effects of depth and width and exposes their interaction in
six runs. Six and eight layers are excluded because the prior H+ 4-layer
result was already harmful; 1 layer is more informative for testing whether
the few-shot Flow is over-capacity.

All six runs first emit unguided continuous p-AUROC and raw F1. Guided-r8 and
morphology are evaluated only for the anchor and at most the top two capacity
rows because those fixed postprocessors do not help estimate Flow capacity.
Capacity tuning stops after this block; it is not expanded to intermediate
depths or widths.

### Stage B2: compact epoch-learning-rate cross

Freeze the selected capacity and evaluate the five rows below:

| Run | Flow epochs | Learning rate |
|---:|---:|---:|
| O1 | 1 | `2e-4` |
| O2 | 3 | `1e-4` |
| O3 | 3 | `2e-4` |
| O4 | 3 | `5e-4` |
| O5 | 5 | `2e-4` |

O3 is the selected-capacity anchor, so this stage adds four new runs. This
cross checks under/over-training and LR sensitivity without paying for a
full 3-by-3 factorial. If neither axis shows a positive gradient, retain
three epochs and `2e-4`.

### Stage B3: compact regularization factorial

Freeze the selected capacity and evaluate one `2 x 2` block:

| Run | `lambda_logdet` | Support brightness |
|---:|---:|---|
| R1 | `1e-3` | `[1.0,1.0]` |
| R2 | `2e-2` | `[1.0,1.0]` |
| R3 | `1e-3` | `[0.8,1.2]` |
| R4 | `2e-2` | `[0.8,1.2]` |

R1 is the selected-capacity anchor, so only three new runs are required.
The block directly measures whether the two previously positive H+ signals
transfer to DINOv2 and whether they interact. Clamp remains fixed at 1.9. A
single `clamp=1.5` confirmation is allowed only if the winning row improves
F1 while losing p-AUROC, because the prior evidence suggests lower clamp may
trade ranking for mask quality. Rotation-8 remains closed.

### Stage B4: backbone feature-layer block

Freeze the detector and optimization settings before changing feature
extraction. All rows use exactly four ViT-L block outputs:

| Run | Label | DINOv2 ViT-L blocks | Purpose |
|---:|---|---|---|
| L1 | Early | `[2,5,8,11]` | local texture and low/mid-level structure |
| L2 | Current | `[5,11,17,23]` | evenly spaced full-depth anchor |
| L3 | Mid-late | `[8,13,18,23]` | reduce early texture dominance |
| L4 | Late | `[11,15,19,23]` | emphasize semantic representations |

L2 is the anchor, so this stage adds three runs. Indices follow the existing
zero-based code contract for the 24-block DINOv2 ViT-L encoder. Layer
selection is reported separately from detector tuning because it changes
the representation rather than the anomaly-scoring module.

### Stage B5: score and memory robustness

Test only after Stages B1--B4 freeze the Basic model:

- density weight `{0,0.1,0.25}`;
- latent-neighbor count `k={1,3,5}` with a consistent aggregation rule;
- latent bank budget `{4096,16384,full static}`;
- raw latent-distance scale stability diagnostics (LOO stays disabled);
- layer fusion: current normalized mean versus one predeclared weighted mean.

These axes determine whether the gain is a robust support-density estimate or
merely a larger kNN memory. RN-FMLK remains the hard null.

### Stage B6: low-priority confirmation only

- DVT alpha `{0.75,1.0}`;
- tail ratio `{0.05,0.10}` at tail weight 0.3;
- residual/distance mixture with at most one alternative to the anchor.

Higher DVT alpha, longer training, broad LR search, large density weights,
transformer/Conv2D Flow, TTE expansion, rotation-8, category-specific guide
settings, and class-specific morphology are not reopened.

### Postprocessing policy

RGB guide and morphology are ablation switches, not tunable test-set knobs.
Radius, epsilon, line length, angle count, and erosion remain frozen. The
final report separates continuous p-AUROC, guided raw F1, and morphology F1.

## 6. Selection discipline

No hyperparameter may be selected from a single class, single support seed,
or F1 alone. Candidate ranking is lexicographic:

1. macro pixel `AUROC_0.05`;
2. F1 retention/improvement;
3. no class worse than `-0.03` AUROC or `-0.05` F1;
4. support-seed stability;
5. compute/memory cost.

Only one setting survives each stage. All subsequent shot and backbone runs
use that setting unchanged. Public TEST results used during development must
be labeled development evidence; the paper should use a preregistered
selection/confirmation split or untouched evaluation population where
available.

## 7. Execution gates

1. Reproduce Basic anchor and same-run SuperAD/RN-FMLK with exact reference
   hashes.
2. Run the complete six-row Stage B1 depth-width factorial and select at most
   two rows for guided/morphology confirmation.
3. Freeze capacity and run the five-row B2 epoch/LR cross (four new runs).
4. Run the four-row B3 regularization factorial (three new runs).
5. Freeze the detector and run the four-row B4 layer block (three new runs).
6. Run B5 only if the preceding stages produce a positive all-eight gradient.
7. Freeze Basic and evaluate the exact-reference SuperAD claim arm.
8. Evaluate Basic random shots `1/2/4/8/16 x 5 seeds`.
9. Transfer the frozen detector to Ours+ and Ours++; start with fixed-16 and
   random-16, then expand the shot curve only if the 16-shot gate passes.

## 8. Claim gates

### Basic claim gate

- Basic beats same-run SuperAD-16 on both macro p-AUROC and F1;
- Basic beats RN-FMLK on both metrics;
- at least 5/8 object wins and no catastrophic class harm;
- exact reference, split, preprocessing, and evaluator parity;
- finite maps and non-collapsed positive coverage.

### Ours+ continuation gate

Ours+ continues beyond 16-shot only if it improves Basic by at least 0.3
p-AUROC point or 0.5 F1 point without catastrophic class harm.

### Ours++ reporting gate

Ours++ is compared with reported SuperADD as external context. Near-parity may
be claimed separately for p-AUROC when the gap is at most 0.5 point. F1 is
reported without parity language until that gap is also at most 0.5 point.

## 9. Final paper outputs

1. fair SuperAD table: SuperAD, RN-FMLK, Basic on the exact 16 references;
2. random shot curve: Basic at `1/2/4/8/16`, five seeds;
3. backbone scaling: Basic, Ours+, Ours++, and reported SuperADD context;
4. component ablation: Flow-LB, LOO, DVT, density, RGB guide, morphology;
5. capacity/regularization table: depth, width, log-det, clamp, brightness;
6. class-wise delta and compute/memory analysis.

All completed runs preserve compact manifests, reference hashes, metrics,
logs, and summaries, and remove regenerable dense anomaly maps after the
verdict is recorded.
