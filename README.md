# Controlled FMAD experiments

The common runner selects normal support images, orders queries, evaluates predictions,
and saves final results. Methods implement fitting and prediction; they do not receive
the dataset object or ground truth.

## Layout

```text
run.py                  source-checkout launcher
src/fmad/
  cli.py                configuration and component registration
  protocol.py           fixed-shot selection, seed, immutable method context
  pipeline.py           shared fit/predict/evaluate loop
  datasets/             MVTec AD 2 public evaluation data
  backbones/            DINOv2 loading, preprocessing, provenance
  methods/              SuperAD, final-layer RN-FMLK baseline, and PSMA_MR
  evaluation/           float16 streaming segmentation metrics
  results.py            isolated runs, final metrics and provenance
configs/                default and fixed-reference baseline configurations
scripts/                remote execution, cache preflight, resource monitoring
tests/                  common pipeline and retained infrastructure regressions
skill_graph/            historical findings and cleanup records
```

## Project guidance

The [current research state](skill_graph/current.md) records live methods,
protocol status, pending decisions, and planning dates. Use the
[active checklist](skill_graph/todo.md) for current tasks and the
[skill-graph index](skill_graph/index.md) for historical evidence. Archived
documents are retained for traceability; their old experiment decisions are
not current execution permissions.

## Remote experiments: reuse one existing container

The normal cycle is local edit/test → commit and push the selected change →
container `update-code` → experiment on assigned GPU(s) → compact result pull.
Commit/push and initial provisioning are not performed automatically by the helper.

Prerequisites: configure the designated server and existing container; its
`REMOTE_WORKDIR` must contain a clean Git checkout with an upstream branch and
Git access. Bash and `flock` must be available in the container. Verify the
existing data/cache mounts, GPU exposure and environment; this workflow does
not rebuild images or change mounts/devices. The container's output directory
must be bind-mounted to the host `REMOTE_RESULTS_DIR` used by `pull`.

```bash
# Source your private preset (never commit credentials).
source configs/remote/147.47.134.98-via-147.47.39.144.env
scripts/run_remote.sh start
scripts/run_remote.sh update-code
export REMOTE_GPU_DEVICES=GPU-assigned-uuid
scripts/run_remote.sh run -- python3 run.py --config configs/default.yaml \
  --data-root /your/container/data/mvtec_ad_2 --device cuda:0 \
  --output-dir /workspace/results_remote
scripts/run_remote.sh pull
```

`update-code` runs `git pull --ff-only`, refuses a dirty checkout, and prints
the resulting commit. It does not copy uncommitted local edits. With a single
selected GPU, `cuda:0` refers to that selected GPU; prefer UUIDs to ambiguous
host/container indices. Selection does not reserve an idle GPU: confirm assigned
GPU availability before launching. Incompatible exposure needs separate action,
not automatic container recreation.

`start` keeps a running container or starts the same stopped container. A missing
container is an error. Explicit `stop` stops but never removes it; completion and
failure do neither. `setup` is disabled. Legacy `sync`, `sync-data`, and `build`
are separate maintenance operations, not the experiment cycle; never `sync`
over a live checkout. An image build does not update an existing container.

`run`/`exec` and `update-code` share a nonblocking container lock. Experiment
stdout/stderr, commit and exit status remain under `/tmp/fmad-execution-logs/`
inside that retained container; `logs` reads those experiment logs. Execution
is foreground, not a detached scheduler: SSH disconnect or forced termination
does not guarantee a complete terminal record or automatic resume. Inspect
the process and recorded state before retrying; do not infer method failure
from an interrupted run.

`pull` transfers only `metrics.json`, `summary.tsv`, `config.yaml`, and
`run_manifest.json`, preserving their relative run directories. Training logs,
anomaly maps and weights stay remote. No remote output is deleted. Finish and
verify the run before pulling; a partial directory is not a completed result.

For metrics-only retrieval, use `scripts/run_remote.sh pull --metrics-only`.
This copies only `metrics.json` and `summary.tsv`; configs, manifests, logs,
maps, and weights are not transferred. Existing local files are not deleted.
Keep provenance remotely and verify completion there before recording values
in the cumulative comparison ledger. The local `experiments.py collect`
workflow needs additional provenance/status files, so it cannot directly
collect a metrics-only download as though it were a complete local run.

### dsba3 policy — 2026-09-12

Future experiments target all four user-assigned dsba3 GPUs. Match the existing
local `fsad-tta-local-bcma` environment described by
`docker/Dockerfile.local-bcma`, including package versions, offline DINOv2 cache,
data/support identities, configs, and metric contract. Inspect actual container
mounts and GPU UUIDs before selecting devices; GPU availability alone does not
prove container exposure or environment parity. No map/checkpoint export is
enabled by this change.

The user explicitly selected local working-tree code as authoritative for this
server. Its private `configs/remote/dsba3.env` sets `REMOTE_CODE_MODE=local`:
`update-code` makes a deterministic SHA-256 snapshot of `run.py`,
`pyproject.toml`, `src`, `configs`, `scripts`, `tests`, and `docker`. Private
remote presets, credentials, caches, and weights are excluded. An unchanged
snapshot skips transfer; a changed snapshot is hash-checked and activated under
the same experiment lock. Releases are separate, read-only trees under
`/workspace/.local-sync/releases`; `/workspace/.local-sync/current` selects the
active tree. Removed local files cannot survive in the new release. Old code,
releases, and experiment results are preserved. No Git commit/push is needed
for this explicitly requested local-snapshot mode; other servers keep the Git
default. `REMOTE_REPO_DIR/.local-sync` on the host must map to
`REMOTE_LOCAL_ROOT` inside the container.

The existing `hun_fsad_tta_012` container was restarted, not recreated. Its
original Python 3.10/Torch 2.1.2+cu118 is retained but is not used by the helper's
`run`/`exec` commands. Those use the copied local Python 3.11.16 runtime at
`/workspace/.local-runtime/python311-cu124-036828932804`, including Torch
2.6.0+cu124 and the same ten pinned packages. Package imports, a CUDA tensor
smoke check, CLI help, and 56 targeted tests passed remotely. The OS and GPU are
still different (Ubuntu 22.04/RTX A6000 versus local Debian/RTX 4090); this is
validated application compatibility, not a bitwise-identical machine/image.
Raw `docker exec ... python` and the legacy `shell` mode do not select this
isolated runtime; use `run`/`exec` as below.

`/data/mvtec_ad_2`, `/model-cache`, and `/results` resolve to retained data/cache
mounts and the new `results_local_compatible` output directory. Dataset relative
file names/sizes match locally (not a full dataset byte-content audit); the
DINOv2 repository and ViT-L/14 checkpoint SHA-256 values match exactly. No data
or model download was needed. Logs/provenance remain remotely, and map/weight
exports remain off.

```bash
source configs/remote/dsba3.env
scripts/run_remote.sh update-code       # local changed snapshot only
scripts/run_remote.sh exec -- python3 run.py --help  # no experiment
scripts/run_remote.sh pull --metrics-only           # after completed runs
```

Reuse this environment and data on subsequent code updates. A dependency
change requires an explicit environment update and validation; syncing code
does not silently reinstall packages. Initial setup launched no experiment.

Four GPUs need explicit per-process device assignment. The current
`experiments.py run --workers 4` option creates four workers on the same
`--device`; it does **not** distribute work across four GPUs. Verify a
multi-device launch arrangement before remote execution. All four assigned
UUIDs are configured and visible; this does not reserve GPUs or stop other
processes. Scheduling is separate from this environment/code-sync setup.

## Install and run

Python 3.10 or newer is required. For CPU model execution:

```bash
python -m pip install -e '.[cpu,dev]'
python run.py --config configs/default.yaml --data-root /data/mvtec_ad_2 --device cpu
# Installed entry point, also usable outside the checkout:
fmad-run --config configs/default.yaml --data-root /data/mvtec_ad_2 --device cpu
```

GPU environments install `.[model,dev]` and a FAISS GPU build compatible with their
CUDA runtime; `docker/Dockerfile.superad` describes the CUDA 11.8 environment.
Set `method_config.faiss_on_cpu: true` to use CPU search with GPU feature extraction.
`cuda:N` denotes the process-visible device and respects an existing
`CUDA_VISIBLE_DEVICES` setting.

DINOv2 uses a local Torch Hub repository/cache first. Set `DINOV2_LOCAL_REPO` and
`TORCH_HOME` to existing caches, and `FMAD_DINOV2_OFFLINE=1` to require cached model
assets. Otherwise initial execution may download the upstream repository/weights.
Backbone repository and state hashes are recorded in the run manifest. Pin expected
hashes through `backbone_config.expected_repo_tree_sha256` and
`backbone_config.expected_state_sha256` when reproducing a specific backbone.

## Controlled comparisons

- Keep `protocol`, `dataset_config`, `backbone_config`, `evaluation`, and seed fixed
  when comparing methods. Backbone/preprocessing ablations must be explicit settings.
- `protocol.shots` selects exactly that many normal images, without replacement,
  from the sorted pool using the seed. Insufficient data fails rather than reducing shots.
- A `protocol.fixed_reference_manifest` overrides random selection. Existing
  object-to-path-list manifests are supported; entries must belong to the active
  object's `train/good` pool, with no duplicates and the exact requested count.
- The common runner records ordered support/query identities, input-content hashes,
  ground-truth hash, effective object preprocessing, and backbone provenance.
- Seeds are set before CLI model construction and each object's fit. Identical seeds
  do not guarantee bitwise equality across GPU kernels, hardware, or library versions.

The earlier default selected a 10% coreset inside SuperAD. The new exact-shot seeded
selection changes that default experiment protocol. Use the same fixed manifest for
historical comparisons. CPU normalized kNN now uses the same cosine-distance definition
as GPU search; old CPU results using the inverted L2 score are not equivalent.

## Add a method

Add a module under `src/fmad/methods/`, decorate its class with
`@METHOD_REGISTRY.register("my_method")`, and import it in `register_methods()` in
`methods/__init__.py`. Derive from `BaseMethod` and implement:

```python
def fit(self, support_paths, context):
    # Build state using only the selected normal references.
    ...

def predict_batch(self, image_paths):
    # Return one Prediction(score_map, image_score, inference_seconds) per path,
    # in the same order. Maps must be finite 2D arrays at native image resolution.
    ...
```

Alternatively implement `predict(path) -> (score_map, image_score)` and use the base
class's timed batching. `object_metadata()` may return compact method-specific insights.
Select the method with `--method my_method`; runner, evaluation, and results code stay common.
For direct Python use, seed before constructing the method with `seed_generators(seed)`.

## First runnable PSMA_MR

PSMA_MR is the first runnable candidate for the paired normal/synthetic residual
adapter idea. It is a method-only change behind the common runner: keep the
dataset, seeded support selection, backbone identity, preprocessing, evaluator,
and metric contract fixed when comparing it with a baseline. This is a
directional development method, not a SOTA claim or a completed benchmark
result.

The first configuration contract is intentionally small and is flattened under
`method_config` (there is no nested `method.params` block):

```yaml
method: PSMA_MR
method_config:
  adapter_enabled: true
  adapter_rank: 4
  resolutions: [448, 672]
  steps: 24
  learning_rate: 0.001
```

`adapter_enabled` selects the trainable adapter (`true`, the default) or the
frozen-backbone control (`false`). `adapter_rank` defaults to `4` and is kept
fixed for the first comparison. `resolutions` defaults to `[448, 672]`; a
one-element list (for example `[672]`) makes the matched single-resolution
control, while `[448, 672]` is the multi-resolution setting. Keep the same
one-element resolution in both single-resolution arms.

The optional training controls are also flat `method_config` keys: `steps`
(default `24`, with `0` disabling adapter updates), `learning_rate` (default
`0.001`), `weight_decay` (`0.0`), `margin` (`0.10`), and the objective weights
`normal_consistency_weight` (`1.0`), `synthetic_margin_weight` (`1.0`), and
`feature_anchor_weight` (`0.10`). Memory controls are
`max_patches_per_source` (`256`), `max_bank_patches` (same as the source cap),
`max_train_patches` (`96`), and `nn_chunk_size` (`1024`); view controls are
`photometric_strength` (`1.0`), `synthetic_anomaly_types`, and
`gradient_clip` (`5.0`). These names are the implementation contract; aliases
such as `train_steps` and `lr` are not supported.

For each object, the common runner chooses the normal support images and passes
only those paths plus immutable object context to PSMA_MR. Fit creates paired
normal-preserving nuisance views and paired synthetic-local-anomaly views from
that support pool. The trainable part is one shared rank-4 final-layer residual
adapter; the common pretrained backbone remains frozen. The fit objective
combines normal score consistency, a paired synthetic anomaly margin, and a
feature anchor. Reference construction is leave-source-out: the source image
and all of its descendants are excluded from the references used for that
source's training signal, preventing self-match leakage.

`MethodContext.rotation` is honored on support data with the same eight fixed
rotations used by SuperAD; queries are scored without rotation augmentation.
`MethodContext.masking` is passed to the shared backbone foreground-mask
operation for support and query scoring. If masking is requested but the
backbone has no mask operation, fitting fails explicitly; use the dataset's
`no_mask_no_rotation` preprocessing mode only when that neutral behavior is
intended. A tiny synthetic view whose anomaly mask has no visible foreground
token may be skipped for that training signal and is counted in training
diagnostics; the method does not require every tiny input to expose every
synthetic family.

The frozen backbone model is loaded once and reused for every configured
resolution through the non-mutating resolution-aware preprocessing API. DINOv2
feature extraction follows `--device` (for example `cuda:0`), while DINOv2
returns CPU feature arrays and PSMA_MR keeps its adapter parameters, support
banks, and nearest-neighbour scoring in CPU PyTorch. Consequently, `--device
cpu` is fully CPU execution; a CUDA device accelerates only the frozen
backbone extraction for this method.

At prediction time, each configured resolution uses the same support identities
and the cosine nearest-neighbour score in the adapted final-layer feature
space. Each map is returned at the query's native image size, then the
per-resolution maps are averaged; the image score is derived from the fused map
using the common top-1% statistic. Queries never update the adapter, reference
banks, or calibration.

The four first controls are:

| Adapter | Resolutions | Purpose |
| --- | --- | --- |
| off | one fixed value, e.g. `[672]` | frozen single-resolution control |
| on | the same one fixed value | adapter contribution at one resolution |
| off | `[448, 672]` | multi-resolution fusion control without adaptation |
| on | `[448, 672]` | PSMA_MR candidate |

Only `adapter_enabled` and `resolutions` change across these four arms; keep
`adapter_rank: 4`, `steps`, `learning_rate`, support selection, and all common
pipeline settings fixed.

### Running through the common CLI

`--method` is the existing registry selector; `PSMA_MR` is its exact value.
The first default run therefore uses the normal source-checkout workflow:

```bash
python run.py --config configs/default.yaml --method PSMA_MR \
  --data-root /data/mvtec_ad_2 --device cuda:0 \
  --output-dir ./results
```

The integration exposes `--resolutions 448,672` and `--adapter-rank 4` as
CLI overrides for the corresponding `method_config` values. Set
`method_config.adapter_enabled` in YAML for the adapter on/off arm; put all
PSMA_MR-specific settings under `method_config`. The generic parser also
supports the common `--seed`, `--shots`, `--objects`,
`--fixed-reference-manifest`, `--debug-output`, and output flags documented
below. There is no generic `--method-config` or `--multi-resolution` flag.

The runner seeds before model construction, selects the support set, fits each
object, predicts its ordered queries, and streams evaluation through the common
metric contract. Keep the seed, support manifest or shot count, dataset,
backbone, preprocessing, query population, and evaluator unchanged across the
four controls. The current dataset runner evaluates its default `test_public`
population; until an explicitly verified sealed split exists, treat results as
development evidence and record actual support/query and label exposure.

See the [concise PSMA_MR method guide](docs/PSMA_MR.md) for the method boundary,
control matrix, and output policy.

## Results

Every execution creates a unique directory below
`results/<method>/<dataset>/<backbone>/seed=<seed>/run-<config hash>-<run ID>/`:

- `config.yaml`: resolved configuration.
- `metrics.json`: per-object and macro-average final metrics.
- `run_manifest.json`: configuration/code hashes, environment, input and model provenance.
- `training.jsonl`: fit start/end/failure, elapsed time, and method-emitted step metrics.

`results/summary.tsv` accumulates full-precision object metrics and timing with run IDs.
Missing metrics are blank, not zero. An incompatible existing TSV header is rejected.
Anomaly maps and fitted weights/state are independent opt-in outputs, both off by default:

```yaml
output:
  save_anomaly_maps: false
  save_weights: false
```

Use `--save-anomaly-maps` and/or `--save-weights` to enable them.
`--no-save-anomaly-maps` and `--no-save-weights` override enabled YAML settings.
Maps are streamed to `anomaly_maps/<object>/<anomaly type>/<image filename>.npz`
with the `score_map` array: native-resolution float32 scores before evaluation's
float16 quantization, with no heatmap normalization. They are not accumulated in RAM.
Fitted state is saved immediately after each object's fit under `weights/<object>/`.
SuperAD/RN-FMLK save FAISS memory-bank indices and metadata; they have no gradient-trained
parameters, and the frozen pretrained backbone is not duplicated.

Trainable methods can call `self.log_training(step=epoch, loss=float(loss))` from `fit()`
to record loss/other JSON-compatible metrics in the common log. SuperAD records reference
feature progress and memory-bank size, with no invented epoch or loss values.
To support weight saving, implement `save_checkpoint(directory)` in the method, writing
for example `torch.save(self.model.state_dict(), directory / "model.pt")`. The pipeline
supplies an empty temporary directory and publishes it after successful serialization;
requesting weights from an unsupported method fails explicitly. Checkpoint contents are
method-defined; saving weights alone does not imply resumable optimizer/training state.

`--debug-output` independently enables timing CSV and per-object diagnostic metrics.

The retained metric contract is `mvtec_ad2_float16_streaming_v1`:
`seg_AUROC_0.05` is standardized partial pixel AUROC at FPR 0.05,
`seg_AUPRO_0.05` is normalized region overlap up to FPR 0.05, and
`seg_F1_oracle` selects the best threshold on evaluation ground truth. The latter is
an oracle diagnostic, not a deployable threshold selected without test labels.
Scores are quantized to float16 for bounded-memory aggregation. Classification and
legacy dense-map evaluation are not supported by this runner.

## Checks

```bash
python -m pytest -q
ruff check --select E4,E7,E9,F src/fmad tests run.py scripts
uv build --wheel
```

Tests use synthetic data and mocked DINOv2 weights, plus real CPU FAISS scoring.
They do not establish full-dataset numerical parity or GPU runtime performance.
Historical experiment launchers and their dedicated tests were retired; see
`skill_graph/analysis/2026-09-08_common_pipeline_refactor/` for the removal inventory.
