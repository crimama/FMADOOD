# Priority Context

- MVTec AD2 is the primary dataset and official main localization metric is AU-PRO@0.05; current reported 75--79 values are pixel partial-AUROC@0.05, not AU-PRO.

# Working Memory

## 2026-07-15 — Reviewer questions to retain

- PPD may be too naive: investigate frequency-aware alternatives and whether positional subtraction is valid when objects occupy diverse positions.
- Test contour-aware PPD or contour-conditioned application strength.
- Include a controlled full-shot experiment and explain why additional normal data provides little or no gain.
- Input/RGB guidance may be poorly aligned with distribution shift; separate optional boundary refinement from shift adaptation.
- Provide a principled reason and direct ablation for latent distance versus base density, full likelihood/NLL, and log-determinant scoring.

# MANUAL

