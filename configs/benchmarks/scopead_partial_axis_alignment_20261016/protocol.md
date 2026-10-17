# Partial axis calibration: fixed development experiment

Hypothesis: partial factor removal retains discriminative scale while reducing nuisance. Method only: d = raw/(a_layer**alphaL * c_resolution**alphaR), existing support-only log-centered Q95 factors. No trainable parameters. New arms (0.5,0), (0,0.5), (0.5,0.5); controls N/Q95/L/R/LR reused only after provenance audit. Defaults preserve old normalizers.

Primary AUPRO@0.05 higher is better; report pAUROC@0.05 higher is better and every class effect. AD2 test_public is seen development, seed0, fixed VOID normal support manifest, 2/16-shot full8, CQ off. Exactly prior frozen ViT-L/14 FP16 layers5/11/23, four resolutions448/672/896/1120, four bags, 256 tokens/source, support/bank, noise guard, fusion and evaluator. Labels/GT enter only evaluation/diagnostics.

Budget exactly48 new class cells (3x2x8). Stop after these: no adaptive grid, crossfit, extra seeds or final holdout. Maps/weights off, logs retained, compact artifacts pulled. Stage immutable source without checkout activation or git push; reuse existing dsba3 container and previously assigned four exposed GPU UUIDs in lane.sh.
