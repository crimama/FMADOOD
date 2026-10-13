# Component ablation (local RTX 4090), seed 0

Questions (post-hoc development ablations, not preregistered final claims):
- A: does layerwise independent matching beat joint matching on concatenated features? Control: layermean (equal-weight mean of three per-layer score maps, tailscale-normalised). Candidate: concat (L2-normalised layers 5/11/23 concatenated, one bank per bag, cosine distance equals the mean per-layer cosine distance; same stored float count; tailscale-normalised single score).
- B: is support-based score normalisation needed? nonorm = linear0125 with score_normalization none.
- C: does the shallow layer help? deeponly = linear0125 with ablation_shallow_mode off.

Reference arms (not rerun): linear0125 and gated from scarf_linear0125_seed0_fp16_20261008 (remote dsba3). Local reproduction check, AD2 fabric 2-shot linear0125: AUPRO.05 .6049 (remote .6049), image AUROC .6460 (remote .6461). Code gained only the opt-in ablation_feature_matching option (default layerwise unchanged); package code hash differs from the remote runs for that reason.

Fixed: DINOv2 ViT-L/14 layers 5/11/23 frozen, FP16 autocast, resolutions 448/672/896/1120, cap256/source, 4 bags, q95 tailscale, CQ99, k9 lowres guard, top1% final-map image score, same support manifests, seed 0, shots 1/2/4/8/16, AD2 public full8, MVTec AD full15, VisA full12 (seen development populations, not holdouts). lambda .125 frozen; concat has no shallow term.

Metrics (higher is better): AD2 AUPRO@.05 primary, pAUROC@.05; MVTec/VisA image AUROC and full-FPR pixel AUROC. Compare each arm to linear0125 per dataset/shot, macro over classes with win/loss counts.
Budget: 4 arms x 5 shots x 35 classes = 700 class runs, sequential on one local GPU, timeout 10800 s per class; order AD2, MVTec, VisA. Stop on runtime/provenance failure. Single seed is development evidence.
