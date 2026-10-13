# Common controlled-experiment pipeline — 2026-09-08

User-authorized implementation following the final repository review. Three Luna/max
implementation agents handled pipeline/method contracts, backbone dependencies, and
results persistence; the main agent integrated packaging, cleanup, CLI, and verification.

## Implemented decisions

- Active Python runtime is the installable `src/fmad` package. `run.py` is a thin
  checkout launcher; `fmad-run` is the installed entry point.
- Pipeline owns exact-shot support selection, seeded ordering, backbone resolution
  and warmup, query iteration, GT access, metric calculation, and persistence.
- Method boundary is `fit(support_paths, MethodContext)` plus `predict_batch(paths)`;
  single-image methods may use the default timed batch wrapper. No dataset or GT API
  is passed to methods. New method registration stays within `methods/`.
- SuperAD and RN-FMLK are the retained baselines. Historical Flow/TTE, C1,
  Shift-Residual, LoRA, PSAR, SAM-refinement, and other standalone method families
  and their dedicated tests/configs/runners were retired. The exact 213-file list is
  [retired_files.json](retired_files.json). Common evaluator/backbone dependencies
  were moved before retiring their old source locations.
- `scripts/` retains only package marker, DINOv2 cache preflight, resource monitor,
  remote infrastructure runner, and its askpass helper.
- Default artifacts (updated output policy): `config.yaml`, `metrics.json`,
  `run_manifest.json`, `training.jsonl` per unique run, and accumulated `summary.tsv`.
  `output.save_anomaly_maps` and `output.save_weights` are independent opt-in flags,
  both false by default. Maps are streamed as native float32 compressed NPZ arrays;
  fitted method state is saved after each object's fit. SuperAD exports FAISS memory
  and metadata, not duplicate frozen backbone weights. Step/loss logging uses
  `BaseMethod.log_training`; training failures are recorded. Debug detail remains opt-in.
- Result identity includes full-config hash and unique run ID; provenance includes
  package source hash, environment versions, input identity/content hashes, GT hash,
  selected support, effective preprocessing, and backbone state/repository hashes.

## Scientific compatibility

The numeric streaming implementation was moved without changing the float16 metric
algorithm. `seg_AUROC_0.05` means standardized partial pixel AUROC, not full AUROC;
`seg_F1_oracle` uses a test-GT-selected threshold; `seg_AUPRO_0.05` is normalized at
FPR 0.05. No new benchmark values were computed.

Default support selection intentionally changes from method-owned 10% coreset to
seeded exact-shot sampling. Existing fixed-reference manifests remain supported for
matched comparisons. Normalized CPU FAISS scoring was corrected to cosine distance,
matching the intended GPU scoring definition. Old CPU inverted-L2 results and old
default-support results must not be presented as the same experiment protocol.

Strict offline backbone loading requires repository and checkpoint caches and loads
the state locally with hub pretrained download disabled. Online use still permits
the mutable upstream DINOv2 main ref; expected hashes can pin a reproduction.

## Validation and limits

The whole retained test suite passed: **92 tests**. Core Ruff error/import checks,
wheel construction, standalone installed-package CLI/import smoke, and scoped
`git diff --check` passed. Tests include
actual CPU FAISS numeric scoring, synthetic two-method end-to-end evaluation,
collision-safe result persistence, backbone mocks, and retained remote-run security
tests. Independent Luna/max review found no high-severity blocker. Its two medium
findings (missing GPU/preprocessing environment versions and late malformed-protocol
validation) were repaired and covered by the final tests.

Subsequent output-policy change: 44 targeted tests passed for the default policy,
all four map/weight flag combinations, quantitative invariance, loss/failure logging,
unsupported checkpoint rejection, CLI on/off overrides, and FAISS export round-trip.

The configured `basedpyright` **all** mode was also run: it did not pass (687 errors
in that check snapshot), largely missing/unknown annotations across the previously
excluded `fmad` runtime and native-library types. Strict typing was not relaxed to
hide these findings. The targeted Ruff gate is E4/E7/E9/F, not every configured
ALL-style rule. A full CUDA/DINOv2 dataset run and Docker build were not performed;
these tests do not establish benchmark parity or GPU performance.

## Existing quantitative archive custody

The previously consolidated result archive was not modified by this refactor.
Verified SHA-256 values:

- `../2026-09-08_results_consolidation/quantitative_ledger.json`:
  `830f5fb4db76d9b4290d3f13c4bab6c70a51e7874471df80f98bc8e751ce0cd2`
- `../2026-09-08_results_consolidation/source_index.tsv`:
  `574e5b861d627ac4f16ea42aea14d3674e31885b163b641f1228934cad1b8ae2`

This verifies preservation of the existing consolidation, not lossless coverage of
all original raw result files. Historical research documents may reference retired
source paths and should be read as historical evidence, not current launch instructions.
The current runtime guide is the repository [README](../../../README.md).
