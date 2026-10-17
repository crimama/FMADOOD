# Table 6 comparator private submissions (1120, 16-shot, seed 0)

- Purpose: fill the five comparator rows of Outline Table 6 (MVTec AD 2 test_private, test_private_mixed).
- Methods and configs (the public 1120 comparator configs; the private submission script reads both private splits itself):
  - PatchCore: public_1120_20261003/patchcore_s16_seed0.yaml (DINOv2 ViT-L/14, layers 5/11/17/23, FP32)
  - AnomalyDINO: public_1120_20261003/anomalydino_s16_seed0.yaml (layer 23, FP32)
  - SuperAD: public_1120_20261003/superad_s16_seed0.yaml (layers 5/11/17/23, FP32)
  - RAD: public_1120_remaining_20261005/rad_s16_seed0_1120.yaml (DINOv3 ViT-B/16, layers 3/6/9/11)
  - SubspaceAD: public_1120_remaining_20261005/subspacead_s16_seed0_1120.yaml (DINOv2-G with registers, layers 22-28)
- Support: configs/methods/VOID_support_16.json, seed 0. Calibration: 16-shot support excluded, every eighth remaining train/good image.
- Threshold: class-wise normal-pixel p95 x 1.421 (same rule as ScopeAD). No private labels for fitting or thresholding.
- Execution: scripts/prepare_rsg4_private_submission.py through launch_private_sequential.sh, one method at a time on the four dsba3 GPU UUIDs.
- Outputs: /var/tmp/comparators_private_1120_20261014/<method>/ in container hun_fsad_tta_012 (not pulled by default).
- Expected counts per method: 4090 TIFF and 4090 thresholded PNG.
