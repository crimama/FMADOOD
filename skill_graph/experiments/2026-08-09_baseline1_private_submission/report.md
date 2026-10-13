# Baseline1-core MVTec AD2 Private Submission

Date: 2026-08-09  
Verdict: `OFFICIAL_ANCHOR_OBTAINED / NOT_A_SUPERIORITY_CLAIM`

## Negative Evidence Intake

The recent two-band protected residual DVT branch is `KILL_FOR_CLAIM / NO_CONTINUE`.
This submission instead freezes the earlier Baseline1-core reference system to
obtain an official-server anchor; it is not a promotion of a new method.

## Motivation

Create an official private/private-mixed result for the DINOv3-H+/16 P16 row
of the backbone-wise Main Table.

## Implementable Design

- Dataset: MVTec AD2 single-image, all eight objects.
- Candidate: Baseline1-core: DINOv3 ViT-H+/16 `[7,15,23,31]`, fixed
  SuperAD-selected 16 supports, DVT position-mean, 3-epoch Flow, raw latent
  1-NN.
- Query population: `test_private` and `test_private_mixed` only.
- Continuous map: native-size finite float16 TIFF.
- Binary map: fixed TESTpublic F1 threshold per object; threshold source is
  `configs/baselines/baseline1_core_public_thresholds.json`.
- Forbidden: private GT access, private metrics, private result-driven tuning,
  query adaptation, and test-time memory updates.

## Evaluation Alignment

This is an official-score anchor, but only a backbone-aligned comparison with
SuperADD. The algorithms and output postprocessing are not paired.

## Code Modification / Creation

- `scripts/run_baseline1_private_submission.py`
- `configs/baselines/baseline1_core_public_thresholds.json`
- `tests/test_baseline1_private_submission.py`

## Added Code Evaluation

- Focused test: `4 passed`.
- `py_compile` and `git diff --check`: passed.
- Local direct execution was blocked before inference because the host Python
  lacks `transformers`; inference therefore uses the existing local GPU
  container with the cached DINOv3 runtime.

## Remote Execution

Not used. This is an official-submission artifact generation run without a
local benchmark metric claim. It runs in the existing local project container
that mounts this workspace and `/home/hun/Volume/DATA/mvtec_ad_2` read-only.

## Results and Analysis

Map generation, official checker validation, and server scoring all completed.
The checker reported `All checks successful` over 8,180 files.

Server-returned metrics:

| Split | AU-PRO@0.05 | Class F1 | Seg F1 |
| --- | ---: | ---: | ---: |
| private | 0.5440 | 0.7350 | 0.4594 |
| private_mixed | 0.5314 | 0.7419 | 0.4405 |

Equal-budget comparison. Against SuperAD-16, the only other P16 row, Ours is
`-0.0124` on private Seg-F1 and `+0.0154` on private-mixed Seg-F1. Ours uses
the stronger DINOv3-H+ backbone and still trails on private, so this run does
not establish a method-level superiority claim at equal normal-data budget.

Distribution-shift degradation. The private to private-mixed Seg-F1 drop is
`-0.0189` (`-4.11%`) for Ours, versus `-0.0307` (`-5.35%`) SuperADD,
`-0.0448` (`-8.78%`) RoBiS, and `-0.0467` (`-9.90%`) SuperAD-16. AU-PRO drops
`-0.0126` (`-2.32%`) for Ours versus `-0.0760` (`-11.30%`) for RoBiS, and
Class F1 does not drop at all (`+0.0069`). Ours is the least shift-sensitive
row in the table despite using 16 normal references and static inference.

Threshold transfer. Public raw-oracle F1 `0.530177` transferred to private
Seg-F1 `0.4594` (`-0.0708`, `-13.4%`). The public SuperADD F1 gap `-0.0959`
and the private gap `-0.1148` are close, so the residual gap is attributable
to segmentation quality rather than threshold transfer. The `-0.318`
normal-only calibration gap seen in the Pfull diagnostic does not appear on
this public-label threshold path.

Largest remaining absolute gap. Private AU-PRO@0.05 `0.5440` versus RoBiS
`0.6725` is `-0.1285`, now larger than the Seg-F1 gap against any row. Region
coherence, not thresholding, is the leading deficit.

## Continuation Assessment

- No configuration was selected from private or private-mixed feedback, and
  none will be.
- The shift-degradation result is a two-point observation on a
  submission-limited server. It must be reproduced through a public-split
  shift proxy before it is claimed as a contribution.
- The AU-PRO deficit is the highest-value target. The validated all-eight
  low-pass DVT operator improves AU-PRO over raw by `+.010513` versus
  `+.005437` for current DVT, and repairs `sheet_metal` AU-PRO from `-.06559`
  to `-.00274`. Any combination with current DVT must occur at map level; the
  feature-space two-band route already failed by anomaly absorption.

## Conclusion

`OFFICIAL_ANCHOR_OBTAINED / NOT_A_SUPERIORITY_CLAIM`. The DINOv3-H+/16 P16 row
of the Main Table now has official private and private-mixed values. Ours does
not beat SuperAD-16 on private Seg-F1 at equal budget, but it is the least
shift-sensitive method in the table on all three returned metrics. The next
targets are region coherence (AU-PRO) and a repeatable public shift proxy.

## Post-Conclusion Storage Cleanup

The checker-required submission TIFF/PNG files are non-regenerable until the
official evaluation completes and are retained. No auxiliary `anomaly_maps/`
tree is produced.
