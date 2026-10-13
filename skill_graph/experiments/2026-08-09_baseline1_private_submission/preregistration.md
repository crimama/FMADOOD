# Baseline1-core MVTec AD2 Private Submission

Date: 2026-08-09

## Motivation

Establish the first official-server result for the backbone-wise Main Table
without presenting the recently rejected two-band DVT branch as a method
candidate.

## Implementable Design

- Target: MVTec AD2 single-image; all eight objects.
- Candidate: frozen Baseline1-core, DINOv3 ViT-H+/16, fixed P16 support,
  DVT position-mean plus Flow latent raw 1-NN.
- Private inputs: `test_private` and `test_private_mixed` RGB images only.
- No private ground truth, result metrics, filenames beyond canonical output
  naming, query adaptation, or private-driven tuning.
- Raw map output: float16 TIFF. Binary output: fixed public-shadow per-object
  pixel-F1 thresholds from `baseline1_knn5_default_ab_20260807_v5`.
- Execution: object-disjoint chunks may run concurrently, but every chunk uses
  the identical frozen configuration and writes only its own object paths.

## Alignment and Boundary

This is a backbone-aligned DINOv3-H+/16 comparison against SuperADD, not a
strict paired method comparison. It is valid for adding an official private
result row, but cannot support a superiority claim because preprocessing,
postprocessing, and method implementation differ.

## Gate

Proceed only when all 16 object/split inventories match the official checker,
all raw maps are finite 2-D float16 TIFFs, all binary maps are 0/255 PNGs, and
`check_and_prepare_data_for_upload.py` succeeds. Upload only after this gate.
