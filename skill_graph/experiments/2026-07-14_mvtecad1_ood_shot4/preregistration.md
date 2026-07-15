# MVTec-OOD severity-3 robustness preregistration

- Method: current AD1 default, 4 first-support shots.
- Frozen encoder: DINOv2-R ViT-B/14, layers `[2,5,8,11]`, 448 input.
- Detector: Flow + static latent bank, density weight 0, no DVT/TTE/context/morphology.
- Image score: mean top 1% of the raw anomaly map. RGB guided-r8 remains the
  localization branch and therefore does not alter this image-only benchmark.
- OOD construction: `imagecorruptions==1.1.2`, severity 3, applied to every test
  image only: brightness, contrast, defocus blur, Gaussian noise. Normal support
  images and ground-truth masks remain unchanged.
- Metrics: per-class image AUROC and oracle max-F1, followed by an unweighted
  15-class macro. OOD Avg averages the four corruptions; Total Avg averages ID
  and the four OOD conditions.
- Comparison limitation: reported Table-2 baselines use the full normal training
  set. This 4-shot result must be labeled as such and is not a strict training-data
  parity claim.
