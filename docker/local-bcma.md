# Dedicated local FSAD runtime

Image recipe: `docker/Dockerfile.local-bcma`.
Image tag: `fsad-tta-local:bcma-cu124-20260911`.
Container: `fsad-tta-local-bcma` (persistent, never auto-removed).
Created container ID: `e2ba5f9f660ff72bb97ef2c6d28fa71d6d7fb9d989102fc052dae822ef6e06c2`.
Built image ID: `sha256:036828932804955f17a296cec6b00ec8678f923ce7754b5fe67e9bd58f1d3daa`.
Verified inside container: Torch 2.6.0+cu124, RTX 4090 CUDA tensor arithmetic,
all 8 data classes accessible, and 30 focused tests passed (2026-09-11).

One RTX 4090 is selected by UUID
`GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b` and appears as `cuda:0`.
Runtime has no network, no published ports and no Docker socket. Code/configs,
test files, AD2 data and the existing Torch cache are mounted read-only.
Only `/results` and temporary runtime storage are writable by UID/GID 1000.
Do not modify the mounted source/configs while a queue is using them.

Host output root: `results/local_runs/bcma_docker_20260911/`.
Queue root inside the container: `/results/ablation`.
The pre-container A/B output directories remain historical evidence and are
not merged into this container-only B0–B3 comparison.

Validation: container CUDA tensor operation, offline DINO cache loading, then
the BCMA, contextual score, CLI configuration, and output-policy tests. Run
all 8 classes for B0, B1, B2, B3 sequentially under seed 0 / random normal16.
Primary metric is AUPRO@0.05; secondary is pAUROC@0.05. TESTpublic is a seen
development set, not a sealed final evaluation. Dense maps and weights remain
off. Per-object timeout is 3600 seconds and the queue stops on errors.

Inspect with `docker exec fsad-tta-local-bcma cat /results/ablation/status.json`.
Restart the same container if stopped; do not recreate it. A container restart
does not itself resume a terminated queue: inspect completed metrics first.

## TailContrast D0–D3 continuation (2026-09-11)

The same container/GPU is reused. BCMA's previous queue is complete. The new
queue uses `/results/tail_contrast_20260911` (host path under the same output
root), with `/results/tail_contrast_20260911.launch.log` for START/DONE events.
Training and evaluation still run through `run.py -> fmad.cli.main() ->
run_pipeline()`. The existing ablation script only sequences configurations;
it does not replace CLI parsing, support selection, training, or evaluation.

Method: `src/fmad/methods/tail_contrast.py`. Configurations:
`configs/methods/tail_contrast_D0.yaml` through `tail_contrast_D3.yaml`.
41 focused container tests passed before launch, including exact score-gradient
replay, D0/B2 equality, switch dependencies, and ledger idempotency. Ruff was
unavailable; diff whitespace checks passed. GPU fitting started on D0/can.

A host-side watcher runs `scripts/experiments.py collect --watch`
against that output directory, updating `results/comparisons.md` and the same
canonical CSV only when queue status changes. Its log is
`results/local_runs/bcma_docker_20260911/tail_contrast_20260911.ledger.log`.
The watcher does not launch/stop evaluation and is not automatically restarted
after host reboot. It exits on completion/failure or after a 33-hour deadline.

The method-specific queue/collector scripts have been consolidated into
`scripts/experiments.py`. The active September 11 queue continues from its
already-loaded launcher (PID 218436); it was not restarted and its child
`run.py` calls are unchanged. Only the retired host collector was replaced.
Source methods/configs were not changed during this cleanup.

Future queues use explicit config paths, not a method-specific family switch:

```bash
python scripts/experiments.py run \
  --configs configs/methods/tail_contrast_D0.yaml configs/methods/tail_contrast_D1.yaml configs/methods/tail_contrast_D2.yaml configs/methods/tail_contrast_D3.yaml \
  --labels D0 D1 D2 D3 \
  --objects can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts \
  --data-root /data/mvtec_ad_2 --output-dir /results/NEW_RUN \
  --device cuda:0 --split test_public_seen_development
```

New statuses store their plan. For the current legacy status only, the host
collector receives the labels/objects explicitly:

```bash
.venv/bin/python scripts/experiments.py collect \
  --output-dir results/local_runs/bcma_docker_20260911/tail_contrast_20260911 \
  --labels D0 D1 D2 D3 \
  --objects can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts \
  --reference-method BCMA_B2 --watch
```

Do not launch a second collector for the same ledger while one is running.

## GuardedMetric E1–E3 continuation (2026-09-11)

D0–D3 is complete. The authorized new sequence uses the same container and GPU,
with `/results/guarded_metric_20260911` as a fresh output root. E1/E2/E3 configs
are `configs/methods/guarded_metric_E1.yaml` through `guarded_metric_E3.yaml`.
The design/budget is locked in `configs/methods/GUARDED_METRIC.md`.

```bash
docker exec -d -e CUDA_VISIBLE_DEVICES=GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b fsad-tta-local-bcma sh -c '
  exec python scripts/experiments.py run \
    --configs configs/methods/guarded_metric_E1.yaml configs/methods/guarded_metric_E2.yaml configs/methods/guarded_metric_E3.yaml \
    --labels E1 E2 E3 \
    --objects can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts \
    --data-root /data/mvtec_ad_2 --output-dir /results/guarded_metric_20260911 \
    --device cuda:0 --split test_public_seen_development --timeout 3600 \
    > /results/guarded_metric_20260911.launch.log 2>&1'
```

This command is a one-time launch, not a resume command. Check for the existing
status/output before running; never overwrite an existing queue. One detached
host collector uses `.venv/bin/python scripts/experiments.py collect --watch
--max-hours 25 --output-dir results/local_runs/bcma_docker_20260911/guarded_metric_20260911`.
Its output is `guarded_metric_20260911.ledger.log` beside the launch log.

Monitor START/DONE events with
`tail -f results/local_runs/bcma_docker_20260911/guarded_metric_20260911.launch.log`.
Per-class logs are `guarded_metric_20260911/E1_can.log` etc.; training step losses
are in each run's `training.jsonl`. Do not change mounted code/configs while
the queue is active. Completion/failure does not stop the container or remove
artifacts. The host watcher exits at terminal queue state or its deadline.
