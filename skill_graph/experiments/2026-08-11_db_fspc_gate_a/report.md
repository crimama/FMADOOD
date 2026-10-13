# DB-FSPC Gate A Support-Only VLM Audit

Date: 2026-08-11  
Verdict: `GATE_A_FAIL / NO_GATE_B`  
Runtime sub-verdict: `vLLM FEASIBLE ON FROZEN GPU BUDGET`

## Scope and Claim Boundary

Gate A tested whether an open-source VLM runtime can produce stable,
support-only part/material graphs before any anomaly-scoring method is built.
It is label-free and reads only the 16 frozen `train/good` sentinels. It does
not enumerate or open `test_public`, `test_private`, or ground-truth data and
cannot make an anomaly-performance claim.

The result separates two questions cleanly:

- The frozen vLLM runtime is operational and fits the 24 GB GPU budget.
- The Qwen3-VL-2B graph outputs are not parse-stable, geometrically equivariant,
  or null-safe enough to support DB-FSPC.

## Frozen Runtime and Inputs

- Host: dsba5 host GPU 3, exposed as container CUDA slot 0, TITAN RTX.
- Model: `Qwen3-VL-2B-Instruct`, revision `8964489`.
- Runtime: vLLM `0.11.0`, transformers `4.57.1`, qwen-vl-utils `0.0.14`,
  NumPy `1.26.4`, FP16.
- Context/output: max model length 4096, max output tokens 1024, greedy
  temperature-0 decoding, 448-pixel long side.
- Objects: `can,vial,fabric,rice`; support indices `[0,5,10,15]`.
- Calls per support image: nine baseline calls across three prompts/repeats,
  eight perturbations, and four nulls; `21 x 16 = 336` total calls.
- Support manifest SHA-256:
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.

## Execution

Run `db_fspc_gate_a_exact_20260811_r1` started immediately after the dsba5
TCCR process exited. It completed from approximately `00:11:42` to
`01:17:55 UTC` (`65 m 44 s`). All `336/336` raw responses and `336/336`
per-call rows were retained.

Preflight hashes:

- runner:
  `c16e665a3e9d50c07e492cbb715aa189636dcb5eeaa4c364cf76837772e5f555`
- parser/core:
  `7b2ad51b80be8761ec51c70d39431c8603c78341673e01052bb9d8526b312fbc`

`dsba3` was excluded from preflight, synchronization, and execution.

## Gate Results

| Gate | Observed | Frozen requirement | Result |
|---|---:|---:|---|
| worst required parse rate | 0.000 | >= 0.99 | fail |
| worst schema-exact agreement | 0.000 | >= 0.95 | fail |
| worst vocabulary Jaccard | 0.000 | >= 0.80 | fail |
| worst count agreement | 0.000 | >= 0.90 | fail |
| worst adjacency agreement | 0.000 | >= 0.90 | fail |
| inverse-hflip box IoU | 0.536361 | >= 0.60, or point error <= 0.05 | fail |
| inverse-hflip point error | 0.149062 | <= 0.05, or box IoU >= 0.60 | fail |
| macro whole-object fallback | 0.727273 | <= 0.40 | fail |
| blank/text-only hallucinations | 16 | 0 | fail |
| peak GPU memory | 13,428 MiB | <= 23,040 MiB | pass |
| p95 latency | 27.731 s | <= 45 s | pass |
| OOM retries | 0 | 0 | pass |
| shuffled-vocabulary null | false | false | pass |

The runner therefore recorded eight exact failure keys:
`required_perturbation_parse_rate`,
`required_perturbation_schema_exact`,
`required_perturbation_vocab_jaccard`,
`required_perturbation_count_agreement`,
`required_perturbation_adjacency_agreement`,
`hflip_inverse_geometry`, `whole_object_fallback`, and
`blank_text_null_hallucination`.

## Failure Analysis

Only `57/336` responses (`16.96%`) passed the fail-closed parser overall. Of
those, 28 were null responses; only `29/272` non-null support/perturbation
responses (`10.66%`) parsed successfully.

| Object | Successful rows / 84 | Overall row parse rate | Baseline-repeat parse rate |
|---|---:|---:|---:|
| can | 8 / 84 | 0.0952 | 0.0000 |
| vial | 19 / 84 | 0.2262 | 0.1667 |
| fabric | 24 / 84 | 0.2857 | 0.1667 |
| rice | 6 / 84 | 0.0714 | 0.0000 |

The most common fail-closed causes were count mismatch (`105` responses),
invalid box (`31`), invalid adjacency edge (`30`), and empty nodes (`16`),
followed by truncated or extra JSON. This is not a vLLM crash: the model often
emitted a readable structure but did not honor the requested count, coordinate,
adjacency, or single-object JSON contract.

Support-fold vocabulary Jaccard was `0` for `can/rice`, `0.3333` for `vial`,
and `1.0` only for `fabric`. Among successfully parsed non-null graphs,
whole-object fallback was `1.0` for `fabric/rice` and `0.9091` for `vial`;
`can` had no valid non-null graph from which to estimate fallback.

Null behavior also fails the intended structural prior:

- blank image: `0/16` hallucinations;
- text-only prompt: `16/16` hallucinations;
- unrelated montage: `12/16` hallucinations;
- wrong-object image: `0/16` parser-counted hallucinations, largely because
  those outputs failed parsing rather than demonstrating target rejection.

## Custody and Runtime Validity

The access ledger passed the support-only boundary:

- `support_train_good_opened=true`;
- `test_public_enumerated=false` and `test_public_opened=false`;
- `test_private_opened=false`;
- `ground_truth_opened=false`;
- `anomaly_metrics_computed=false` and `fusion_computed=false`.

Thus Gate A is a valid negative structural/runtime audit. It says nothing
about anomaly AUROC, F1, or localization quality.

## Conclusion

`GATE_A_FAIL / NO_GATE_B`.

Open-source vLLM is usable as the inference runtime on dsba5 GPU 3: the exact
Qwen3-VL-2B revision loaded in FP16, completed all calls without OOM, stayed
within 13.43 GiB, and met the latency gate. The blocked part is the method, not
the serving engine. A small instruction VLM prompted for free-form part graphs
does not yield a stable nuisance-invariant structural coordinate system under
this frozen support-only protocol.

DB-FSPC Gate B is not licensed for two independent reasons: Gate A failed,
and the prerequisite TCCR experiment did not earn `PROMISING_DIAGNOSTIC`.
No anomaly-scoring fusion or test-time-training experiment should be built on
these graph outputs.

## Artifacts and Cleanup

- Summary SHA-256:
  `deb292ef2dc353c1bb6b453cb99a5322af4dd61fde370285e2c70d15a511256a`.
- Manifest SHA-256:
  `5fc7508738e91d2753b0b8c4f1277f3a809e08dbc714b46826d02d4932398e85`.
- Per-call TSV SHA-256:
  `2a2b4c81fc75c68e01bc510eebfa9005c24aec689dfa95b25076e8d2fef1ddc1`.

No dense anomaly maps were produced. The pulled evidence consists of compact
JSON/TSV plus 336 raw text responses totaling 337,899 bytes. The immutable
dsba5 source snapshot was removed after verified pull; compact results remain.
