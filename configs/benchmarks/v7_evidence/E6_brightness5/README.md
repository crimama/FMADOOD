# E6: Brightness Corrupted Test Set

This experiment uses MVTec AD 2 data with brightness corruption (severity 5).

## Configuration

- Dataset: MVTec AD 2 with brightness_s5 corruption
- Method: reference_matching R1
- Backbone: DINOv2 ViT-L14
- Support: 16-shot

## Files

- `NA_gated.yaml`: Noise-attributed gated shallow mode
- `NR_lowres.yaml`: Noise-remedied low-resolution fallback mode

## Metrics

Evaluated with MVTec AD 2 metrics (pixel-level AUROC, PRO@0.05).
