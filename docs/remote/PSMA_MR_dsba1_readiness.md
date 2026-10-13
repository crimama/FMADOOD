# PSMA_MR dsba1 remote readiness

Status: readiness inspection recorded on 2026-09-09 (Asia/Seoul). This is an
execution-readiness record, not a benchmark result or a final-evaluation claim.

## Scope and safety boundary

- Existing containers only: `hun_fsad_tta_sra_20260908` and
  `hun_fsad_tta_c1_dsba1`.
- No container was created, recreated, deleted, stopped, or re-exposed.
- No jobs were killed and no remote data/cache artifacts were deleted.
- The ignored `configs/remote/dsba1.env` remains the only local SSH/credential
  source. The tracked presets below contain no credentials.
- GPU selection must use the assigned UUID in the preset. Do not substitute an
  unassigned host index or use the default `hun_superad_run` container.

## Verified remote inventory

| Container | Exposed host GPU IDs | Preset UUID | Workdir/check-out target | Data mount | DINOv2 cache |
| --- | --- | --- | --- | --- | --- |
| `hun_fsad_tta_sra_20260908` | `0,2,4` | `GPU-40a9f4d3-f585-9544-8209-359b2ec00252` (host 0) | `/workspace/fmad-ood-dsba1-sra` | `/home/hunim/Volume/DATA` read-only | `/workspace/.cache/torch` read-only |
| `hun_fsad_tta_c1_dsba1` | `6` | `GPU-59115c6a-8ccb-5474-72e3-67cdf2d363fd` (host 6) | `/workspace/fmad-ood-dsba1-c1` | `/home/hunim/Volume/DATA` read-only | `/workspace/.cache/torch` |

Both containers were running and idle at inspection. The remote MVTec AD 2
dataset exists at `/home/hunim/Volume/DATA/mvtec_ad_2` (about 31 GB) with
`train`, `validation`, `test_public`, `test_private`, and `test_private_mixed`
trees. The cached `dinov2_vitl14_pretrain.pth` and
`dinov2_vitl14_reg4_pretrain.pth` weights, plus the local DINOv2 hub repository,
are present. The runtime reported Python 3.10.12, PyTorch 2.1.2+cu118, CUDA
11.8, and one process-visible CUDA device when each assigned UUID was selected.

The SRA container currently has no source checkout in `/workspace`. The C1
container has a legacy non-git source copy in `/workspace`. Both presets use a
new, narrowly named checkout path so the legacy copy and existing results are
not overwritten. A checkout may be created only after the selected local code
has been reviewed, committed, pushed without force, and verified from `origin`.

## Tracked presets

- [`configs/remote/dsba1_sra_20260908.env.example`](../../configs/remote/dsba1_sra_20260908.env.example)
- [`configs/remote/dsba1_c1.env.example`](../../configs/remote/dsba1_c1.env.example)

Use one preset per command session:

```bash
set +x
source configs/remote/dsba1.env
source configs/remote/dsba1_sra_20260908.env.example
```

The presets set `TORCH_HOME`, `DINOV2_LOCAL_REPO`, and
`FMAD_DINOV2_OFFLINE=1` so the run cannot silently download a model. The
ignored base must be sourced before the preset; do not print the resulting
environment and do not copy its values into tracked files.

## Publication and checkout sequence

1. Review the exact selected source scope with the coordinator. The remote
   checkout must contain the committed `run.py`, `src/fmad/**`, the selected
   PSMA_MR/default configs, and any dependency metadata needed by the existing
   image. Unrelated worktree deletions or historical scripts are out of scope.
2. Verify the pushed commit from `origin` and create the new checkout in the
   existing container workspace only; do not overwrite `/workspace` or the
   legacy C1 copy.
3. Set the preset's `REMOTE_WORKDIR` to the new checkout and use the helper's
   serialized update path:

   ```bash
   scripts/run_remote.sh update-code
   ```

   `update-code` must report a clean checkout and `git pull --ff-only` commit
   before any method run. Existing containers are retained throughout.

## Bounded smoke command

After the checkout and commit are verified, run only a CUDA/runtime smoke on
each assigned UUID (host 0, 2, 4 in SRA and host 6 in C1), not a full benchmark:

```bash
scripts/run_remote.sh run -- python3 -c \
  'import torch; x=torch.randn(1024,1024,device="cuda"); y=x@x; torch.cuda.synchronize(); print(torch.cuda.get_device_name(), float(y.mean()))'
```

Keep each smoke to a few minutes at most, retain the helper's remote log, and
record the selected UUID, commit, CUDA/PyTorch versions, and exit code. This
smoke proves runtime readiness only; it does not produce PSMA_MR metric
evidence.

## First PSMA_MR run boundary

The first method invocation should remain a small development smoke (for
example one explicitly selected object and the canonical 16-shot protocol),
with `save_anomaly_maps=false` and `save_weights=false`. Preserve
`config.yaml`, `run_manifest.json`, `training.jsonl`, `summary.tsv`, and
`metrics.json` if the run completes. The default query population is
`test_public`; it is seen/development evidence unless a separately verified
sealed split is used. Do not report an interrupted or partial run as a method
failure.
