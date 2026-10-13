# CUG-RN-S1 Stage -1 Report

Date: 2026-08-25  
Protocol: `cug-rn-s1-stage-minus1-support-only-20260825`  
Status: `COMPLETED / STOPPED_AFTER_PANEL`  
Verdict: `NEAR_MISS / ONE_BOUNDED_ITERATION`

## Negative Evidence Intake

CUG-RN-S1 was deliberately narrower than prior synthetic-anomaly branches.
Synthetic anomalies were source-only counterfactual instruments for learning a
PSD map of a normal paired-shift gradient. They were not target labels, a
deployed anomaly head, or a query-time selector. The score stayed exact hard
nearest-normal distance after the same adapter was applied to support and
evaluated descendants.

No TESTpublic, TESTprivate, public image, query image, or ground truth was
opened. This report therefore makes no benchmark-performance claim.

## Implemented Design

- DINOv2 ViT-L/14 layer-11 features on the frozen 672-pixel canvas, pooled to
  `16 x 16` and normalized by the preregistered channel/L2 transform.
- Deterministic capped source-shift PCA, rank 8.
- One target update of norm `0.10` through a source-learned spectrally capped
  PSD operator; the target gradient used paired normal shifts only.
- Four leave-generator-out folds x three seeds, with `B0`, identity,
  normal-only meta, CUG, static projection, pair-shuffle, and 16
  spectrum-matched random controls.
- Exact global hard-RN deployment scoring and exact support-16 LOO scale.
- Fail-closed support custody, code/config/snapshot hashes, access ledgers, and
  sequential `can -> panel4 -> conditional all8` authorization markers.

The frozen Adam update is evaluated from its standard equations with
`beta=(0.9,0.999)`, `eps=1e-8`, `lr=0.03`, no weight decay, and 60 steps. A
local six-step float64 test matched `torch.optim.Adam` within `1e-15`; this
avoids an incompatible dsba3 `torch._dynamo -> transformers` import path
without changing the algorithm.

## Code Modification / Creation

- `src/flow_tte/cug_rn.py`
- `scripts/run_cug_rn_stage_minus1.py`
- `scripts/aggregate_cug_rn_stage_minus1.py`
- `scripts/run_cug_rn_stage_minus1_remote.sh`
- `tests/test_cug_rn.py`
- `tests/test_cug_rn_stage_minus1_runner.py`
- `tests/test_aggregate_cug_rn_stage_minus1.py`
- `tests/test_run_cug_rn_stage_minus1_remote.py`

ATAC-v1 and ATAC-SB-v2 code and evidence were not reused as this protocol's
identity; CUG-RN-S1 has separate protocol and schema names.

## Local Verification

- Python compilation: passed for the core, runner, aggregator, and focused
  tests.
- Focused pytest: `15 passed in 2.83s`.
- Remote wrapper: `bash -n` passed and `selftest=passed`.
- Critical Ruff rules `E9,F63,F7,F82`: passed.
- Synthetic dry run: finite, nonconstant, operator-valid, changed score,
  adapter identity error `5.960464477539063e-08`, step norm exactly `0.1`.
- Post-pull local reaggregation reproduced the remote sentinel and panel JSON
  byte-for-byte. Summary SHA256 values are
  `b1915f7d113b7d58a2e5b4c4f5b4890c42a60842a71abec6963a732aca65bd07`
  and `6a080face1cca37174a6f60770ef753b95b709ee458f4de82972ccf3eb301411`.

## Remote Execution

The fixed dsba3 container was `hun_fsad_tta_012`; actual host/container UUID
parity showed GPUs 0--3 visible and idle at launch. The exact support manifest
SHA256 was
`7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.

| Immutable run | Status | Scientific access |
|---|---|---|
| `cug_rn_s1_stage_minus1_dsba3_seq_20260825_v1` | preflight smoke failed on the container PyTorch/Transformers compatibility path | no gate entered |
| `cug_rn_s1_stage_minus1_dsba3_seq_20260825_v2` | controller wrapper path failed immediately | no gate entered |
| `cug_rn_s1_stage_minus1_dsba3_seq_20260825_v3` | `complete_support_only`, stopped after panel | sentinel plus authorized panel only |

The successful run started at `2026-08-25T00:39:58Z` and completed at
`2026-08-25T01:35:08Z`. Its immutable snapshot SHA256 was
`8fed85b25c14c01dd1911ecc209e26e61d23c8926f3e4449b0effc727dc98fd3`,
which matched the local launch manifest after pull.

The panel resource marker had correct GPU 0--3 indices/UUIDs and only 1 MiB
used on every GPU. It also captured a transient 29% utilization sample on GPU0
immediately after the sentinel. A shell `errexit` propagation defect allowed
that extra, non-preregistered `<=10%` wrapper check to continue. No active GPU
context or external job was present, GPU allocation/sequentiality was intact,
and an independent audit accepted the scientific result. The future wrapper
now retries six times and returns explicitly on failure.

## Results

The GPU0 `can` sentinel was valid and authorized the panel. The panel then ran
`can/fabric/vial/walnuts` on GPUs `0/1/2/3` respectively.

| Object | Valid rows | Safe-positive rows | Full-causal rows | Rows passing all topology nulls |
|---|---:|---:|---:|---:|
| can | 12/12 | 12/12 | 0/12 | 0/12 |
| fabric | 12/12 | 12/12 | 0/12 | 10/12 |
| vial | 12/12 | 12/12 | 0/12 | 8/12 |
| walnuts | 12/12 | 12/12 | 0/12 | 0/12 |

Panel counts were valid `4/4`, safe-positive `4/4`, and full-causal `0/4`.
The all-eight gate required `4/4`, `>=3/4`, and `>=2/4`; its causal gap was
therefore `-2`. No all-eight shard was created.

Median M0 normal-tail reductions were `0.001276`, `0.027053`, `0.045332`, and
`0.017185` for can/fabric/vial/walnuts. These are real support-only effects, but
they are not identified as synthetic-guard effects:

- M0's normal-tail gain failed to exceed the strongest matched null in
  `12/12`, `10/12`, `12/12`, and `7/12` rows respectively.
- M0's held-family contrast delta failed to exceed the strongest matched null
  in `12/12`, `12/12`, `12/12`, and `11/12` rows.
- Can failed the neighbor-retention null in every row; walnuts failed the
  scale-ratio null in every row. Fabric and vial were closer on topology but
  still lacked comparator separation.

Thus CUG-RN-S1 often produced a numerically safe positive update, but source
synthetic guarding was not causally distinguishable from ordinary meta,
projection, shuffle, or random-spectrum operators.

## SuperAD Baseline and Unified Metrics

Same-condition SuperAD evaluation is intentionally deferred. Values remain
`superad_seg_AUROC_0.05=N/A`, `method_seg_AUROC_0.05=N/A`,
`delta_vs_superad=N/A`, and `benchmark_comparable=false` because no public
image was authorized.

## Continuation Assessment

The exact result is `NEAR_MISS / ONE_BOUNDED_ITERATION`, not a method claim and
not authorization for all-eight or public evaluation. One separately named,
bounded support-only iteration is allowed only if it directly targets the two
observed failures: matched-null superiority and topology preservation. Merely
tuning step size, seeds, or thresholds on these panel results is not licensed.

## Post-Conclusion Storage Cleanup

The pulled artifact is
`results/remote_runs/dsba3/cug_rn_s1_stage_minus1_dsba3_seq_20260825_v3`
(`2.5M`, 33 files). Compact SHA verification passed. Local and remote
`anomaly_maps` counts are both zero, no all-eight directory exists, and all
four GPUs returned to 1 MiB / 0% utilization.
