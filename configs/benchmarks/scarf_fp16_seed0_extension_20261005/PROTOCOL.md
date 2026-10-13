# SCARF FP16 seed0 extension

Authorized 2026-10-05 to produce precision-matched three-seed results.
AD2 test_public is seen development data; labels/GT evaluation-only.
No method change: frozen DINOv2-L/14 FP16 autocast, reference_matching_fast, cap256, scales448/672/896/1120. Reuse prior FP16 seeds1/2 at shots1/4/8.
New budget: shots1/4/8 seed0 x full8 =24 class runs; four existing dsba3 GPUs, two disjoint classes each, sequential configs through scripts/experiments.py run. Stop failed shards and preserve logs. Metrics/maps/count/fingerprint and execution condition checks before aggregation.
Primary macro AUPRO@.05; secondary pAUROC@.05/I-AUROC, higher better. Compute full8 macros per seed then mean/sample SD. Maps/weights off.
Execute prior immutable source release; stage configs without activation. Keep prior comparator FP32 single1120 condition explicit.
