# Single-resolution diagnostic (AD2 development, seed 0)

Purpose: measure how much each input resolution contributes alone under the Full method (linear0125, q95), to judge whether bank-level reliability weights (a structural extension of per-bank calibration) have headroom. Configs equal ad2_A_linear0125_tailscale_s02/s16 except resolutions = [r] and the declared noise-guard resolutions = [r], r in 448, 672, 896 (1120-only exists in the calibration factorial Stage B). AD2 public full8, shots 2 and 16, seed 0, FP16, cap256, same support manifests; 6 configs x 8 classes = 48 class runs. Diagnostic only, not a candidate method; seen development population.

