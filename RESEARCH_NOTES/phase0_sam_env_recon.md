# Phase 0 — SAM Environment Recon (2026-08-20)

Verified by orchestrator after the Codex(gpt-5.6-luna) env-prep run was cut off by a
session boundary; artifacts persisted and were re-validated end-to-end.

## Container / mounts / runtime
- Container: `hun_fsad_tta_local_stage1a0` (image `hun_fsad_tta_image:stage1a0-local`).
- Start: `docker start hun_fsad_tta_local_stage1a0` (restart-safe; state persists).
- WorkingDir: `/workspace`. GPU: device `0` exposed (RTX 4090, 24 GB). Runtime `runc` with GPU device request.
- Mounts:
  - repo `/home/hun/Volume/RESEARCH/FSAD-TTA` → `/workspace` (rw)
  - dataset `/home/hun/Volume/DATA/mvtec_ad_2` → `/home/hunim/Volume/DATA/mvtec_ad_2` (**ro**)
- Python inside container: `/usr/bin/python3` (3.10.12), torch `2.1.2+cu118`, CUDA available on RTX 4090.

## SAM install
- `segment-anything` **1.0** (Meta original), installed inside the container. **License: Apache-2.0.**
- Checkpoint: `/workspace/models/sam/sam_vit_h_4b8939.pth` (ViT-H, 2,564,550,879 bytes),
  persistent via the rw repo mount (host `models/sam/`).
- NOTE: `transformers` (4.56.2) SAM2 path is currently broken in this image — importing transformers
  raises `torch.utils._pytree.register_pytree_node` AttributeError against torch 2.1.2. **Use the
  Meta `segment_anything` package directly; do not route SAM through transformers.**

## Smoke test (`scripts/sam_env_smoke.py`, new, no experiment logic)
- Image: `can/train/good/000_regular.png` (1024×2232). Synthetic Gaussian heatmap → point/box prompt.
- Result: `mask_shape=(1,1024,2232)`, `mask_area=123090`, `score=0.8145`, `runtime≈1.0 s`,
  `peak_gpu_mem≈5.7 GiB`. torch 2.1.2+cu118, RTX 4090.
- Guards: refuses any `test_private` path.

## Blockers / notes
- None for `segment_anything`. transformers-SAM2 blocked (pytree) — irrelevant since we use Meta SAM.
- 5.7 GiB peak per SAM call at ViT-H leaves ample headroom on the 24 GB card for batched runs.
