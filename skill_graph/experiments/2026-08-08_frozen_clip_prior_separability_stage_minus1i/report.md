# Stage -1i Frozen CLIP Prior Separability Report

## 1. Motivation

The metadata auxiliary-variable direction was discarded because it is too image-dependent and can become an answer-bearing side channel. Stage -1i tests a narrower, pixel-only question: whether a frozen external visual prior, CLIP ViT-L/14@336, provides independent evidence that separates normal photometric shift from defect-induced local/broad shift better than the DINOv3-H+ layer-7 reference.

This is a support-only mechanism diagnostic. It is not a descriptor-swap method claim, not a SuperAD comparison, and not a public benchmark result.

## 2. Implementable Design

- Dataset scope: MVTec AD2 single-image support-only diagnostic.
- Objects: `fruit_jelly`, `walnuts`, `vial`, `fabric`.
- Sources: exact fixed 16 `train/good` support images per object from the frozen support manifest.
- Candidate prior: frozen OpenAI CLIP ViT-L/14@336 cached JIT at `/workspace/.cache/clip/ViT-L-14-336px.pt`.
- Reference prior: frozen DINOv3-H+ layer 7.
- Conditions: `clean`, deterministic unseen photometric `U`, `U + D_local`, `U + D_broad`.
- Scoring: leave-one-source-out clean memory from the other 15 support sources, fixed `k=5` mean nearest-neighbor distance.
- Forbidden: metadata, filename-derived condition, split signal, `test_public`, `test_private`, `ground_truth`, query adaptation, parameter updates, descriptor/fusion tuning, and sweeps.

CLIP patch tokens are extracted manually from the JIT visual module and L2-normalized. `torchvision` is not required for CLIP preprocessing; resize/crop/normalization are implemented directly with PIL, numpy, and torch.

## 3. Evaluation Alignment

The design tests only whether a frozen pixel-derived prior changes the scale-normalized relation between normal photometric shift and synthetic defect evidence. It does not test segmentation quality, SuperAD dominance, map coverage, or deployability.

The result is compared to DINO only through within-prior scale-normalized ratios:

- `shift`: CLIP absolute `U-clean` tail inflation ratio versus DINO. Lower is better; strict gate <= `0.80`.
- `local core`: CLIP local defect-core gain ratio versus DINO. Higher is better; strict gate >= `0.80`.
- `broad core`: CLIP broad defect-core gain ratio versus DINO. Higher is better; strict gate >= `0.80`.
- `local sel`: CLIP local selectivity ratio versus DINO. Higher is better; strict gate >= `1.25`.
- `broad sel`: CLIP broad selectivity ratio versus DINO. Higher is better; strict gate >= `1.25`.

## 4. Code Modification / Creation

- `scripts/run_frozen_clip_prior_separability_stage_minus1i.py`: support-only runner, CLIP JIT patch extraction, DINO layer-7 comparator, fixed `k=5` scoring, metrics, access ledger.
- `scripts/aggregate_frozen_clip_prior_separability_stage_minus1i.py`: validity checks, strict/continuation gate aggregation, summary TSV/JSON.
- `scripts/run_frozen_clip_prior_separability_stage_minus1i_remote.sh`: dsba5 fixed-container controller.
- `tests/test_frozen_clip_prior_separability_stage_minus1i.py`: kNN, metric, gate, access, controller, and cached-JIT patch extraction regression tests.

## 5. Added Code Evaluation

Local implementation verification:

- `py_compile`: passed.
- Focused pytest: `7 passed`.
- Remote controller syntax: `bash -n` passed.
- Git whitespace check: passed.

Important runtime note: CPU loading of this old CLIP JIT is limited because the cached model was baked through a CUDA-oriented TorchScript export path. Actual cached-JIT extraction was validated on CUDA after casting the image tensor to `visual.conv1.weight.dtype`. The concrete extraction check produced `(576, 1024)` patch tokens, grid `(24, 24)`, finite values, and unit-norm max error `1.19e-07` for both black and random 336 images.

## 6. Remote Experiment Execution

Run root:

`results/remote_runs/dsba5/frozen_clip_prior_stage_m1i_20260808_v3`

Remote preflight:

- `status=passed`
- `runner_sha256=3bdb0d7c597f3337c4dbae6dfee9603c2dc95bdb6ad41c2a131e4d4b8c20095f`
- `aggregate_sha256=0d182da4685d20dc7894be7433acd2ccbece8b3a5197f136bb7eb0af30c0f709`
- `support_manifest_sha256=9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`
- host GPU `3`, container CUDA slot `0`
- CLIP JIT: `/workspace/.cache/clip/ViT-L-14-336px.pt`

Run status: `complete`.

Custody:

- `manifest_ok=true`
- all object `support_ok=true`
- all object `access_ok=true`
- all object `coverage_ok=true`
- all object `geometry_pass=true`
- `metadata_used=false`
- `filename_condition_used=false`
- no `test_public`, `test_private`, or `ground_truth` access.

Snapshot cleanup: completed after pull. No `anomaly_maps/` directory exists under the pulled result root.

## 7. Evaluation Results and Analysis

| object | shift ratio | local core ratio | broad core ratio | local selectivity ratio | broad selectivity ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| fruit_jelly | 3.43598 | 0.67925 | 0.24213 | 0.25023 | 0.12048 |
| walnuts | 0.24476 | 1.75838 | 0.66450 | 0.49467 | 0.20622 |
| vial | 13.47296 | 1.18431 | 0.11233 | 0.09466 | 0.01286 |
| fabric | 0.52876 | 2.74793 | 1.34388 | 1.25653 | 0.96616 |

Gate result:

- Strict useful prior: `0/4`.
- Continuation: `0/4`.
- Validity: manifest/access/coverage/geometry all valid.

Object-level interpretation:

- `walnuts` and `fabric` reduce normal-shift sensitivity relative to DINO. They also show local defect gain above DINO, and `fabric` shows broad defect gain above DINO.
- However, selectivity is not robust. `fabric` is the only object with local selectivity above the strict threshold, and its broad selectivity remains below `1.25`.
- `fruit_jelly` and `vial` fail decisively on shift inflation: CLIP is much more shift-sensitive than DINO, especially `vial`.
- Broad defect retention is weak on `fruit_jelly`, `walnuts`, and `vial`.

The result therefore does not support CLIP as a cross-object independent evidence source for few-shot AD under distribution shift. The positive pieces are object- and texture-dependent, not a stable mechanism. They do not authorize CLIP/DINO fusion, descriptor swap, prompt prior, or per-object prior selection follow-ups.

## 8. Conclusion

Verdict: `KILL_FROZEN_CLIP_PRIOR / NO_CONTINUE`.

The asset retained from this negative result is a clean diagnostic harness for frozen visual-prior separability under the support-only no-peek contract. The branch itself is closed: metadata/filename signals are explicitly forbidden, and CLIP fusion or descriptor-swap follow-up is not justified by this evidence.

## 9. Post-Conclusion Storage Cleanup

No dense anomaly maps were generated by this diagnostic. Verification:

- `find results/remote_runs/dsba5/frozen_clip_prior_stage_m1i_20260808_v3 -type d -name anomaly_maps -print` returned no paths.
- Compact artifacts retained: `summary.json`, `object_summary.tsv`, object summaries, run manifest, access ledger, checksums, preflight logs, controller log.
