# PSAR-D1 detector checkpoint preregistration

Date: 2026-08-28  
Protocol: `psar-d1-checkpoint-20260828-v1`  
Status: `PREREGISTERED / RESULT-FREE / LAUNCH-BLOCKED-UNTIL-SEAL`  
Claim tier: three-object detector diagnostic; not a paper-level claim and not
LoRA authorization.

## Question, panel, and terminal branches

PSAR-D1 asks one question: does a same-base, same-shift synthetic
anomaly-over-clean ranking term improve a real query anomaly map beyond the
same absolute head, two donor-disjoint unpaired ranking nulls, exact
SuperAD-16, and exact RN-FMLK?

The result-blind checkpoint panel is fixed to:

- strong object: `fabric`;
- weak/stress object: `vial`;
- control object: `walnuts`;
- exact deterministic repeat: `fabric_repeat`.

`revision_budget=0` after the first readable TESTpublic map or metric. A valid
negative closes PSAR-D1. Only `PROMISING_DIAGNOSTIC` opens a separately sealed,
unchanged all-eight confirmation with seeds `20260828`, `20260829`, and
`20260830`. LoRA remains closed until that confirmation is positive and a
separate capacity-gap experiment is preregistered.

## Dataset, support, and preprocessing

- Dataset is MVTec AD 2 at resolution 672, preprocessing
  `no_mask_no_rotation`, full `test_public/good` and `test_public/bad`, no
  query adaptation, no query-memory insertion, and no private split.
- Support is the exact SuperAD-selected P16 manifest
  `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`,
  SHA-256 `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Ordered support paths, resolved relative paths, file SHA-256 values, source
  roles, and memory cardinalities are recorded for every object. Only
  `train/good` support bytes may be opened before the public map barrier.
- P12/P15 are role-controlled, unlabeled transductive memories, not independent
  validation sets. Indices 12--15 never become fit examples or loss targets.

## Backbone and residual input

The backbone is frozen DINOv2 ViT-L/14 (`dinov2_vitl14`), shorter edge 672,
zero-based layers `[5,11,17,23]`. Its expected parameter-state SHA-256 is
`59a755e385188971c57f52a38773028b508722c804ce6c9279d1dc2ac14ccbe8`.
The seal also binds the resolved local hub repository tree, loader source,
model class, patch size, depth, torch/torchvision/CUDA versions, and model
state. A mutable download, state-hash mismatch, or missing offline artifact is
invalid.

For each layer, query patch tokens and memory patch tokens remain raw: no
centering, standardization, L2 normalization, coreset, PCA, or query-dependent
normalization. The per-patch input is exactly
`src.flow_tte.superadd_parity.nearest_distance_map`, i.e. the unsquared
Euclidean 1-NN distance divided by channel count. Four layer maps are stacked
in layer order as four float32 channels.

## Learned head and arms

Each object has four independent learned heads with byte-identical initial
state:

`Conv1x1(4,16,bias=True) -> GELU -> depthwise Conv3x3(16,pad=1,bias=False)
-> GELU -> Conv1x1(16,1,bias=True)`.

There is no normalization, sigmoid, residual scaling, smoothing, morphology,
rank transform, min-max transform, per-query transform, or postprocessing.
The six frozen output arms are:

- `B0_SuperAD16`: exact `configs/baselines/same16_superad_ad2.yaml` identity;
- `B1_RN_FMLK`: exact `configs/baselines/same16_rn_fmlk_ad2.yaml` identity;
- `I0_ABSOLUTE_HEAD`: absolute loss only;
- `N0_PLUS_UNPAIRED_BASE`: absolute loss plus the `k+` clean-reference pair;
- `N0_MINUS_UNPAIRED_BASE`: absolute loss plus the `k-` clean-reference pair;
- `M0_PSAR`: absolute loss plus the correct same-base clean-reference pair.

For fit base `i in 0..11`, set `d=(i+1) mod 12`, `k+=(i+2) mod 12`, and
`k-=(i-2) mod 12`; all four identities must differ. The common physical memory
is P16 with `i,d,k+,k-` removed and therefore has exactly 12 images. The
synthetic anomaly for every learned arm uses base `i` and donor `d`; only the
pair reference differs: correct shifted clean `i` for M0, shifted clean `k+`
or `k-` for the nulls, and no pair for I0.

## Synthetic cells and deterministic schedule

Fit generators are the exact image-space implementations from
`scripts/run_cider_cf_stage_minus1.py`: `G0_poisson_patch`,
`G1_material_blotch`, and `G4_frequency_local`, including `make_mask` and
`synthesize` without enlargement or mask repair.

The four fit shifts, applied with the same realized parameters to anomaly,
correct clean, and unpaired clean images, are:

1. exposure EV `-0.1333`;
2. exposure EV `+0.1333`;
3. Gaussian pixel noise with sigma equal to the uint8 pixel RMS between the
   base and its `+0.1333` exposure, clipped to `[1,32]` and rendered by
   `gaussian_noise_views(..., count=1)`;
4. spatial multiplicative `cos_1_1` illumination, using the exact coordinate
   field from `scripts/spatial_illumination_bank.py` but solving target
   coefficient of variation `0.15`.

The schedule is exactly `4 epochs x 12 bases x 4 shifts x 3 generators = 576`
updates per object and arm, in nested lexicographic order with no shuffle,
early stopping, validation selection, scheduler, accumulation, or replay
selection. Each epoch regenerates its cell. The cell seed is the first eight
big-endian bytes of SHA-256 over UTF-8 fields joined by byte `0x1f`:
`PSAR-D1-v1, object, 20260828, epoch, index, shift, generator, role`, reduced
modulo `2^32`. Every generated mask/image/shift is hashed in the ledger.

Optimizer is AdamW with learning rate `3e-4`, weight decay `1e-4`, betas
`(0.9,0.999)`, epsilon `1e-8`, batch size 1, global gradient-norm clip `1.0`,
and `foreach=False`, `fused=False`. Frozen-backbone forward uses CUDA bf16
autocast; raw distance, head logits, interpolation, and loss are float32.
Deterministic algorithms are enabled; TF32 and cuDNN benchmark are disabled.

## Exact loss

For logit `l` and target `y`, `bce(l,y)=softplus(l)-y*l`. With binary anomaly
mask `m`, anomaly logits `s+`, correct clean logits `s0`, and pair-reference
logits `sref`:

- `Lpos=sum(m*bce(s+,1))/sum(m)`;
- `Lneg=sum((1-m)*bce(s+,0))/sum(1-m)`;
- `Lclean=mean(bce(s0,0))`;
- `Labs=0.5*(Lpos+Lneg)+Lclean`;
- `Lpair=sum(m*softplus(0.5-(s+-sref)))/sum(m)`.

I0 uses `Labs`; M0 and both N0 arms use `Labs+Lpair` with pair weight 1.0.
The pixel mask is resized once to the native token grid with area
interpolation and thresholded `>0`; empty positive or background regions are
invalid. Logits are never passed through sigmoid for training.

## Normal-only calibration

For calibration base `c in 12..15`, the fixed donor is
`12+((c-12+1) mod 4)`. Each learned arm scores the four clean fit-shift views
of each calibration base against P15 memory `P16 minus {c}`: exactly 16 maps
per arm and object. Query images, query scores, GT, synthetic anomalies, masks,
and identity maps are forbidden in calibration.

Each image contributes equal mass. `tau` is the smallest observed float32
score whose mean per-image empirical CDF is at least 0.95. Activity is
strictly `map > tau`. Threshold value, source rows, observed tie count/mass,
and calibration-map hashes are recorded.

## Held synthetic shortcut audit

The fixed audit universe is `4 calibration bases x 4 generators x 4 held
nuisances = 64` cells per object. Generators are G0/G1/G4 plus held
`G2_structural_scratch`. Held nuisances are EV `-0.20`, EV `+0.20`, Gaussian
sigma `1.5` times the fitted RMS-matched sigma before clipping to `[1,32]`, and
`cos_1_1` spatial illumination at CV `0.225`. Donor is the calibration cycle;
memory excludes both base and donor, hence P14. No audit result changes the
head, threshold, or public map.

The M0 guard passes only if all maps are finite/nonconstant, G2 pixel
`seg_AUROC_0.05 >= 0.65`, and its G2 value exceeds each N0 arm by at least
`0.03`. For each learned arm, form a raw row-major anomaly-logit vector for
each of the 64 cells. A deterministic leave-one-cell-out Euclidean 1-NN probe
with lexical tie-breaking predicts generator, base, and nuisance identities;
each multiclass balanced accuracy must be at most `0.35`. The scalar anomaly
contrast is mean logit inside the mask minus mean logit outside. The maximum
absolute average-tie Spearman correlation of this contrast with normalized
mask area, perimeter, center-x, or center-y must be at most `0.35`.

## Public map barrier and evaluator

Map generation lists query image paths but cannot open GT. For all four jobs
and all six arms it first commits the ordered query inventory, raw float32
map files, byte hashes, shape, finite/nonconstant counts, head/state or method
identity, support/source rows, calibration threshold, and canonical package
hash. Only after the controller verifies a complete 24-package inventory may
a distinct evaluation phase open `test_public/ground_truth`.

Learned token-grid logits are resized directly to native image HxW using
float32 bilinear interpolation with `align_corners=False`. Evaluation calls
`src.post_eval.eval_segmentation_metrics(..., pro_integration_limit=.05,
delete_tiff_files=False, compute_aupro=True)` directly. Default 0.30 PRO,
smoothing, morphology, binary repair, map rescaling, and query normalization
are forbidden. `seg_F1_raw` and `seg_AUPRO_0.05`, if emitted, must equal
`seg_F1` and `AU-PRO@0.05` exactly.

The literal checkpoint output fields are `seg_AUROC_0.05`, `seg_F1`,
`AU-PRO@0.05`, `map_coverage`, `normal_FPR`, `runtime`, `memory`,
`object_results`, `delta_vs_superad`, `delta_vs_rn_fmlk`, and
`delta_vs_hard_null`. A fixed 2,000-replicate paired, split-stratified image
bootstrap uses seed `20260828` and the same resample indices for every arm;
intervals are descriptive only.

## Runtime, memory, repeat, and GPUs

Inference time starts with decoded RGB available and ends with a CPU float32
map, including preprocessing, backbone, resident-memory scoring, head,
interpolation, and CUDA synchronization; it excludes disk I/O, bank build,
training, calibration, and evaluation. After three warmups, every remaining
query is timed. Report separate training/calibration/bank wall times.
Inference peak allocated CUDA memory is reset immediately before the same
window with banks resident; training peak is separate.

All resource comparisons run in the same dsba3 container and GPU model/window.
Fixed allocation is GPU0 `fabric` then `fabric_repeat`, GPU1 `vial`, and GPU2
`walnuts`. GPU3 and dsba2 GPU6 are unused because the current sealed remote
preset does not bind them. Only aggregate telemetry for these assigned GPUs
may be queried; no process/container/file belonging to another user may be
listed, opened, altered, or stopped. A busy assigned GPU defers launch.

`fabric_repeat` must reproduce the canonical scientific package bytes exactly
after excluding timestamps, wall-time/resource fields, run-root prefixes, and
compression metadata; every metric must additionally differ by at most
`1e-6`. Both are required.

## Ordered verdict gates

1. Any seal/schema/count/GT-barrier/source/model/schedule/repeat/finite/shape/
   nonconstant/map-coverage error is `INVALID_FAIL_CLOSED`.
2. Held pAUROC/null-margin/identity-probe/Spearman failure is
   `VALID_SHORTCUT_FAILURE / CLOSE_PSAR_D1`.
3. Harm/resource failure is
   `VALID_HARM_OR_RESOURCE_FAILURE / CLOSE_PSAR_D1`: macro M0 F1 harm below
   `-0.05`, macro M0 AU-PRO harm below `-0.03`, fabric or walnuts M0 pAUROC/
   F1/AU-PRO harm below `-0.03/-0.05/-0.05`, macro or walnuts normal FPR
   above the worse baseline by `0.01`, macro or fabric/walnuts defect-active
   coverage below `0.01` or below half the better baseline, M0 inference time
   above `2*max(B0,B1)`, or M0 peak memory above B0 by 2 GiB.
4. If M0 macro pAUROC is not above `max(B0,B1)+0.005`, not above
   `max(N0+,N0-)+0.005`, or not above I0 by `0.002`, the verdict is
   `VALID_BUT_NO_GAIN / CLOSE_PSAR_D1`.
5. Only a run passing every earlier gate is `PROMISING_DIAGNOSTIC`; it is not
   KEEP and authorizes only the unchanged all-eight confirmation.

Gate comparisons use the better baseline for higher-is-better metric harm and
the worse baseline for FPR. `delta_vs_hard_null` subtracts the larger null
value separately for every higher-is-better metric.

## Seal, execution, and cleanup

The non-self-referential source manifest binds this preregistration, static
contract, output schema, support manifest, DINO loader/backbone/state,
distance implementation, generators/shifts, evaluator, dataset/method/
pipeline/result sources, exact B0/B1 configs, new core/runner/aggregator/
remote wrapper/verifier, and all tests. The remote verifies the archive before
extraction and every entry after extraction. Runtime receipts include host,
assigned GPU UUID, fixed container identity, software versions, model hash,
data/query pathset hashes, and source rows.

Dense `anomaly_maps/`, optimizer states, and isolated source snapshots are
temporary. After the final compact aggregate/report is pulled and verified,
cleanup removes only the exact experiment-owned dense-map and snapshot paths;
compact JSON ledgers, metrics, hashes, logs, receipts, and cleanup evidence are
retained.
