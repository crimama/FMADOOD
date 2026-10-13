# Defect-Side Cross-Image-Calibrated Region Completion Stage -1 Preregistration

Date: 2026-08-11

Status: `PREREGISTERED / PENDING`

## Resource Boundary

`dsba3` is excluded.

Frozen allocations:

- local GPU 0: `can,fabric,vial`
- dsba5 host GPU 3 / container CUDA slot 0: parity shard `rice` plus duplicate
  `can`

## Frozen Inputs

- dataset: MVTec AD2 single-image
- objects: `can,fabric,vial` locally, `rice` plus duplicate `can` on dsba5
- support: fixed SuperAD-16 manifest
- deterministic public hash selection: `16 good + 32 bad` per object
- backbone: frozen DINOv3 H+/16
- coarse branch: unchanged current 672 full-image H+DVT observation with fused
  layers `[7,15,23,31]`
- fine branch: frozen layer 7 on a 1344 shorter-edge image observed through 672 overlapping crops
  with two deterministic phases
- private split: forbidden
- dense maps: never persisted
- GT rule: public GT masks are opened only after every implemented arm map has
  been staged in memory
- required anchor: `P16_H+DVT_anchor`

## Implemented Arms

`P16_H+DVT_anchor`, `raw_g0_layer7`, `raw_tiled_fine_knn`, `lift_only`,
`calibrated_coarse_fine_max`, `calibrated_coarse_fine_mean`,
`matched_coarse_to_fine_predictor`, `one_phase_excess`, `two_phase_tccr`,
`geometry_shuffle`, `phase_shuffle`, `rgb_edge`,
`equal_compute_duplicate_tiles`.

`raw_tiled_fine_knn` uses both crop phases and the same two-phase minimum, so it
is compute-matched to the TCCR observation path. `equal_compute_duplicate_tiles`
repeats phase 0 with an independent forward pass and never compares a query to
itself.

## Explicit Omission

None. No arm, gate, threshold, or continuation rule may be retuned after
preregistration.

## Gates

Continue only if two-phase TCCR improves seed recoverability and AU-PRO /
component recall over raw G0, raw tiled fine kNN, matched coarse-to-fine
predictor, geometry shuffle, phase shuffle, and RGB edge controls without
invalid support LOO tail calibration. Kill for claim if gains are absorbed by
hard nulls, predictor residuals, raw tiled kNN, or if support LOO tail coverage
is invalid.
