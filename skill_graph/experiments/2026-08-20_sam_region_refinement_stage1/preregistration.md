# SAM Region-Refinement Stage-1 Preregistration

Date: 2026-08-20
Protocol: `sam-region-refinement-stage1-20260820`
Status: **FROZEN 2026-08-22** under `docs/EXPERIMENT_GUIDELINES.md` before any GPU execution of this stage.
Claim class: mechanism/diagnostic only; NOT benchmark-comparable until this gate passes.
Target venue: CVPR 2027.

## 1. Research question and hypothesis

Baseline1-core's largest deficit is region coherence (private AU-PRO `0.5440`, `-0.1285` vs RoBiS;
`gt_component_recall_at_oracle` `0.484`). RoBiS closes this with SAM refinement + adaptive
binarization at Pfull. 16 normal images cannot supply a learned region prior; we test whether a
**frozen, class-agnostic segmentation foundation model (SAM)** can supply that region prior for free.

Hypothesis: prompting SAM from the continuous anomaly map and snapping anomaly evidence to
SAM's object/part region boundaries improves AU-PRO and component recall **without** absorbing or
inflating defect evidence, under the P16 / static-inference / no-test-tuning contract.

This is a **map-level, post-scoring** operator. It is not a normal-side representation change
(so it is outside the closed "identify-nuisance-from-normals-then-suppress" family), and it is not a
feature-space fusion (so it is outside the closed two-band `ANOMALY_ABSORPTION` branch).

## 2. External-prior feasibility contract (SAM)

Per the ledger's external-prior rule, frozen before use:
- Prior: Segment Anything Model (SAM), ViT-H, checkpoint `sam_vit_h_4b8939.pth` (or HF
  `facebook/sam-vit-huge`); SAM2 only as a documented alternative. Exact version/commit recorded in
  `RESEARCH_NOTES/phase0_sam_env_recon.md`.
- Weights source + license: Meta AI, **Apache-2.0** (verify in recon). Frozen; no fine-tuning.
- I/O schema: input = query RGB + prompts derived only from our anomaly map; output = binary
  region masks + IoU/stability scores. SAM sees **no labels, no filename/split tokens, no support**.
- Independence check (hard stop): SAM output must not correlate with the benchmark filename/split
  condition tokens (`regular/overexposed/underexposed/shift_*`, private `regular/mixed`). If a
  filename-shuffle changes SAM's masks materially, abort.
- No test tuning: all SAM hyperparameters (points-per-side, prompt rule, IoU/stability thresholds)
  are frozen on support/held-normal or set to SAM defaults; never selected on any test metric.
- Static inference (R7): no weight/statistic update at query time.

## 3. Data custody and split

- Dataset: MVTec AD2 single-image, few-shot P16 (exact fixed SuperAD-selected 16-support manifest,
  SHA `9f223f7c...`).
- Phase-1 objects (reduced, deliberately the hard/diverse cases): `sheet_metal` (worst AU-PRO
  regression), `rice` (breaks region-morphology on no-harm), `can` (low-FPR operating-point
  collapse), `fabric` (diffuse texture defects, renormalization-fragile).
- Evaluation split: `test_public/good,bad` for map generation; **`test_private` and mixed variants
  forbidden**. Ground truth opened only AFTER all arm maps are written and hashed.
- Base anomaly maps: reuse Baseline1-core (`density_weight=0`) continuous maps; the Flow is frozen
  and unchanged across arms (parameter checksum recorded before/after, identical).
- Every shard emits: support manifest SHA, base-map source SHA, SAM checkpoint SHA, code/config SHA,
  path-access ledger, and `query_or_test_private_access=false`.

## 4. Method (candidate M1_SAM_refine)

Frozen operator, per query image, applied to the Baseline1-core continuous map `M`:
1. Derive prompts from `M` only: connected-components of `M` above a support-fit percentile
   `q_prompt` (fixed, calibrated on held-normal, NOT on test); for each component take its argmax as
   a point prompt and its bounding box as a box prompt. Cap at top-`P` components (frozen `P`).
2. Query SAM with those prompts; keep masks passing SAM's own IoU/stability thresholds (frozen).
3. Refined map `M'` = a **defect-preserving** combination that can only *redistribute within
   evidence*, never invent it: `M'(x) = M(x)` outside all masks; inside a mask `R`,
   `M'(x) = max(M(x), agg_R)` where `agg_R` is a bounded region statistic of `M` over `R`
   (e.g. high percentile), so a region is lifted toward coherence only where `M` already carries
   evidence. Exact `agg` (percentile value) frozen in the sealed config.
   Rationale: forbids the two failure modes seen before — per-image renormalization (invalidated the
   map-level rank protocol) and unconditional broad lifting (gamed component recall). No pixel gets
   evidence that `M` did not already place somewhere in its region.

Frozen operator defaults (set 2026-08-20; may be revisited ONLY on support/held-normal, NEVER on
any test metric or on the Phase-1 objects post-hoc):
- `q_prompt` = support-fit global threshold = 99.5th percentile of Baseline1-core map scores over the
  16 support images' leave-one-out self-scores (one threshold per object; label-free, static).
- prompt candidates = 4-connected components of `{M > q_prompt}`; keep the top-`P` by area.
- `P` = 8 (defects are few regions; caps SAM calls / avoids texture flooding).
- prompt geometry = per component: point = argmax of `M` in the component; box = component bbox
  (no dilation). Feed point+box to SAM predictor with `multimask_output=True`; keep the mask with the
  highest SAM predicted-IoU (SAM defaults; no test-tuned IoU/stability threshold).
- region aggregate `agg_R` = 90th percentile of `M` over mask `R` (bounded, robust to single-pixel
  outliers); `M'(x) = max(M(x), agg_R)` inside `R`, `M'(x)=M(x)` elsewhere. Overlapping masks: apply
  in descending SAM-IoU order; a pixel takes the max over all its covering regions.

Deployed score/threshold: for the primary AU-PRO/component-recall endpoints, evaluation is
threshold-free / pooled (§6). A separate label-free adaptive binarization (C3) is a Phase-2
question and is NOT part of this gate.

## 5. Arms and controls

- `B0_base`: Baseline1-core map, no refinement (hard floor).
- `M1_SAM_refine`: the candidate above.
- `N0_random_box`: identical operator but prompts from RANDOM boxes (same count/size distribution),
  not from `M`. Tests whether the gain needs correct anomaly-conditioned prompting or any SAM region.
- `N1_generic_morph`: existing close→fill→erode morphology in place of SAM (the current operational
  default). Tests whether SAM beats generic region smoothing at all.
- `N2_map_agnostic_SAM`: SAM automatic full masks, lift every region by its `agg` regardless of `M`
  peaks. Tests whether the benefit is real evidence-conditioned refinement vs. global region lifting.

## 6. Endpoints and gates (frozen)

Primary (pooled, threshold-free, all Phase-1 objects):
- AU-PRO@0.05 (`src/post_eval.py` / `src/flow_tte/aupro.py`, integration limit 0.05).
- `gt_component_recall_at_global_oracle`.

Secondary: pixel-AP, full pixel-AUROC, Seg-F1 (raw oracle, reported not gated).

Safety (all must hold):
- **Anomaly-preservation assay**: defect-pixel evidence retained inside SAM masks; mean
  defect-pixel score under M1 ≥ 0.97 × under B0 (no absorption). Broad + local synthetic defect
  contrast retention ≥ 0.95.
- **Per-object no-harm**: no Phase-1 object loses > 0.01 AU-PRO vs B0 (guards the guided-r8 `can`
  collapse precedent).
- Reproducibility: local host vs docker-container parity on a duplicated object, max abs delta
  ≤ 1e-3 (matches prior parity tolerance).

Continuation gate (ALL required to promote to Phase-2 all-8 TESTpublic shadow):
1. M1 AU-PRO >= B0 by a grounded material margin on >= 3/4 objects AND macro-positive.
   Grounding (G-2, frozen before results): margin threshold = `+0.01` macro AU-PRO,
   symmetric with the per-object no-harm rule below; it is the smallest delta this
   pipeline's historical AU-PRO attributions treated as material (e.g. DVT factorial
   AU-PRO effects) and exceeds the `1e-3` parity tolerance by 10x.
2. M1 component recall >= B0, macro-positive.
3. M1 beats `N0_random_box` AND `N1_generic_morph` AND `N2_map_agnostic_SAM` on AU-PRO (macro).
4. All safety gates pass.
If any fails -> `KILL_FOR_CLAIM`, redesign the refinement operator; do not tune `q_prompt`/`P`/`agg`
post-hoc on these objects (that is post-hoc tuning of a failed operator).
G-2 two-lane: if gate 1 misses only on magnitude while gates 2-4 pass and the direction is
positive, record `NEAR_MISS`; one bounded iteration is allowed only if the iteration lane
(operator redesign spec, not parameter tuning) was preregistered before unblinding. None is
declared now, so a magnitude miss ends this operator as frozen.

## 6b. Canonical metric contract (G-4)

All arms additionally emit the existing paper metric suite used across prior tables —
no new claim metrics: pAUROC@0.05 (`seg_AUROC_0.05`, reported as `AUROC@0.05`),
Seg-F1 (raw oracle, provenance-labeled), AU-PRO@0.05, pixel-AP. Secondary diagnostics
(component recall, tail statistics) stay diagnostic-only. GT opens only after all arm
maps are written and hashed.

## 9. Guideline compliance notes (2026-08-22)

- Amendment (pre-execution, no arm results exist): `N1_generic_morph` is implemented
  (`scripts/sam_region_refine_stage1/apply_arms.py:n1_fast`) as a fixed 3x3 grayscale
  closing (`scipy.ndimage.grey_closing`, nearest mode), NOT the historical close->fill->erode
  chain. Rationale: that chain is a threshold-domain operator tied to the retired operational
  default and cannot run threshold-free on continuous maps; N1's role here is only a generic
  region-smoothing comparator for gate 3, which 3x3 closing satisfies. This amendment is
  recorded before unblinding and is not a post-hoc adjustment.
- G-5: not applicable (this is not a synthetic-shift-proxy experiment); the SAM
  independence check in §2 stands.
- Infra amendment (2026-08-22, pre-execution): the pinned container was recreated under the
  same name/mounts from a committed image (`hun_fsad_tta_image:stage1a0-local-hf`) with one
  additional read-only mount `/home/hun/.cache/huggingface -> /workspace/hf_cache`; the driver
  sets `HF_HOME=/workspace/hf_cache` and `FMAD_DINOV3_OFFLINE=1` so Baseline1-core loads the
  cached gated DINOv3-H+ checkpoint offline (verified: offline AutoModel load OK, patch_size 16).
  No code path or arm definition changed; v1 root discarded after all four generations failed
  on the missing cache before any map was produced.
- G-6 outlier stratification: `fabric` is a known outlier object (repeated
  single-object-driven failures in TTE, positional-key, DVT factorial). It is retained
  deliberately as the hardest case, but per-object reporting must tag any verdict that
  hinges solely on fabric as `OBJECT_DRIVEN_FAILURE`.
- G-8 runway estimate: base maps ~1-2 GPU-hour/object on the local RTX 4090 (sheet_metal
  cheaper at 448), SAM arms ~1 s/image x ~200 query images/object/arm plus N2 automatic
  masks (~10 s/image), evaluation CPU-bound. Total ~6-10 h single-GPU sequential.
  Failure still yields the region-coherence mechanism insight, satisfying the
  paper-content-on-failure check. Local GPU only; no dsba3/dsba5 work in this stage.

## 7. Boundaries

No TTA, no query-time parameter/statistic update, no memory update, no test_private/mixed/GT access
before maps, no filename/split-token use, no config selection on any test metric. SAM is frozen and
used only as a query-image region prior. `benchmark_comparable=false` for this stage.

## 8. Infra (from Phase-0 recon — `RESEARCH_NOTES/phase0_sam_env_recon.md`)

- Container: `hun_fsad_tta_local_stage1a0` (image `hun_fsad_tta_image:stage1a0-local`); `docker start`; WorkingDir `/workspace`; GPU device 0 (RTX 4090). Python `/usr/bin/python3` 3.10.12, torch 2.1.2+cu118.
- Mounts: repo→`/workspace` (rw); AD2→`/home/hunim/Volume/DATA/mvtec_ad_2` (**ro**).
- SAM: `segment_anything` 1.0 (Apache-2.0), checkpoint `/workspace/models/sam/sam_vit_h_4b8939.pth` (2,564,550,879 B). Use Meta `segment_anything` directly (transformers SAM2 broken by torch/pytree mismatch).
- Smoke validated: ViT-H on RTX 4090, ~1.0 s/image, ~5.7 GiB peak (`scripts/sam_env_smoke.py`).
- TODO (next delegation): isolated runner + aggregator + focused tests (operator determinism, no-absorption invariant, custody/access ledger), NOT touching existing experiment code.
