# TARCD-LoRA-S3-VR Stage -1 Report

Date: 2026-08-26  
Protocol: `tarcd-lora-s3-vr-stage-minus1-support-adapted-20260826`  
Status: `COMPLETED / PREREGISTERED_STOP + OBSERVATIONAL_NONCLAIM_PANEL`  
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## Outcome

S3-VR removed the numerical defect that motivated this one-change successor:
the first D2 correction-gradient norm fell by more than six orders of
magnitude relative to S2 and passed the frozen conditioning gate on every
object. It did not solve the scientific failure. The four-object observational
panel improved median correction cosine by `+0.0936927` versus S2, short of the
required `+0.10`, and improved the median hard-null margin by only `+0.0065332`,
short of the required `+0.05`. No cell beat the hard-null margin gate, no
synthetic-contribution ablation passed, and object mechanism passes were
`0/4`.

The preregistered run stopped correctly before panel expansion because two
sequential GPU0 `can` runs did not produce bitwise-identical trained adapters.
At the user's explicit request to use the available GPUs and collect actual
results actively, a separately marked `observational_nonclaim` run then used
dsba3 GPUs 0--3 to execute the fixed four-object panel from the same immutable
snapshot. That panel is supplementary negative evidence only; it does not
override the preregistered repeatability stop or authorize a scientific claim.

## Frozen Design

- Backbone: frozen DINOv2 ViT-L/14 with fused-qkv rank-4, alpha-8 LoRA in
  zero-based blocks 8--11; all non-LoRA parameters remained frozen.
- Sole change from S2: target-norm-normalized vector Huber replaced the
  singular zero-start direction-plus-log-scale correction loss.
- Seed: exactly `20260826`; no sweep.
- Fit support: exact target SuperAD-16 `train/good` manifest, SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Held evaluation: four support-disjoint `train/good` bases per object,
  three held shifts, and held generator `G2_structural_scratch`.
- Training: `144` deterministic AdamW steps per trained arm, learning rate
  `1e-4`, bf16 forward, fp32 loss, clip `1.0`, and no early stopping.
- Objects/GPU map: GPU0 `can`, GPU1 `fabric`, GPU2 `vial`, GPU3 `walnuts`.
- S2 randomization, arms, RN guard, synthetic families, schedule, custody,
  thresholds, and controls were retained.

This remains a support-only mechanism diagnostic, not a segmentation
benchmark. `superad_seg_AUROC_0.05=N/A`, `method_seg_AUROC_0.05=N/A`,
`delta_vs_superad=N/A`, and `benchmark_comparable=false`.

## Implementation and Local Validation

- `src/flow_tte/tarcd_lora_s3_vr.py`: normalized vector-Huber correction
  objective and analytic zero-start probe.
- `scripts/run_tarcd_lora_s3_vr_stage_minus1.py`: separately named matched S3
  runner and conditioning diagnostics.
- `scripts/aggregate_tarcd_lora_s3_vr_stage_minus1.py`: fail-closed S3
  aggregation, three-run repeatability recomputation, and S2-paired
  continuation gate.
- `scripts/check_tarcd_lora_s3_vr_repeatability.py`: adapter and normalized
  object-payload auditor.
- `scripts/run_tarcd_lora_s3_vr_stage_minus1_remote.sh`: sequential GPU0
  sentinel/repeat controller and fixed four-GPU panel.

The S3 wrapper initially inherited S2 through process substitution, causing
the sourced script to resolve `BASH_SOURCE` under `/dev/fd` and contaminate the
local pull root as `/dev/results/...`. The CUDA DINOv2/LoRA model smoke itself
had already passed; failure occurred afterward while creating the local launch
manifest. The wrapper now captures its own repository root and any explicit
pull-root override before sourcing S2, then restores the S3-local paths. A
regression test and self-test guard cover this failure.

Fresh validation after the fix passed:

- focused pytest: `37 passed`;
- Python compile and selected critical Ruff checks: passed;
- shell syntax and wrapper self-test: passed;
- deterministic synthetic vector-gradient smoke: passed;
- actual remote CUDA/bf16 DINOv2-L LoRA model-path preflight: passed.

Frozen-config SHA-256:
`e08609511005d7a1b1d9d5c1c2994545c80a4385f0f99334c128133e0ab1bbc7`.

## Preregistered Execution and Stop

Primary run:
`tarcd_lora_s3_vr_stage_minus1_dsba3_20260826T101215Z_resume_v2`.

- Start/completion: `2026-08-26T09:42:31Z` to
  `2026-08-26T10:42:56Z`.
- Status: `complete_support_only`; stopped after `sentinel_invalid` with
  `stop_reason=bitwise_repeatability_failed`.
- Both sequential GPU0 `can` runs were valid and produced identical aggregate
  scientific metrics: correction cosine `0.2891698`, hard-null margin
  `-0.1476580`, and q99 reduction `3.1199%`.
- Adapter bytes differed for D0, D2, and M0. B1, D1, shuffled correction, and
  random-matched states repeated. The audit therefore recorded
  `adapter_state_bitwise_equal=false` and
  `bitwise_repeatability_pass=false`.
- Compact ledger SHA-256:
  `e2d2738caeb08378684a3822b56994ea3ada7c3417147bf17dd69cc809fcda97`;
  all 17 entries passed after pull.

The controller followed the preregistration and did not open GPUs 1--3 for a
claim-bearing panel.

## Explicit Observational Four-GPU Panel

Supplementary run:
`tarcd_lora_s3_vr_stage_minus1_dsba3_20260826T104400Z_observational_panel_v1`.

- Start/completion: `2026-08-26T10:46:26Z` to
  `2026-08-26T11:54:28Z`.
- Status: `complete_support_only`, `stopped_after=panel_support_only`.
- The run records `claim_mode=observational_nonclaim`,
  `repeatability=panel_failed`, and
  `scientific_claim_authorized=false`.
- Immutable snapshot archive SHA-256:
  `955de1f9928d0faebf7ddfdb8851ffa5859a496f181b34423c85bbe62d2bd898`.
- Compact ledger SHA-256:
  `5651129a2f726971bfa13fbd3e39d064c454e61cf0af34faccef538e5b090849`;
  all 29 entries passed after pull.
- The ten embedded code hashes exactly match the current frozen S3 source.
- Independent local reaggregation was byte-identical to the remote aggregate;
  both have SHA-256
  `7ef6dd97df2b2dc20eb4b5c15e8f16dabb05037b9df1008380111f8768964465`.

## Actual Results

All four objects and all twelve held cells were valid and finite/nonconstant,
but no cell and no object passed the complete mechanism gate.

| object | GPU | valid/pass cells | D2 correction cosine | D2 q99 reduction | hard-null margin | generator-ID BA | synthetic / RN pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| can | 0 | 3/3 / 0/3 | 0.2893 | 3.12% | -0.1476 | 0.2708 | 0/3 / 2/3 |
| fabric | 1 | 3/3 / 0/3 | 0.5009 | 0.58% | -0.1404 | 0.3333 | 0/3 / 3/3 |
| vial | 2 | 3/3 / 0/3 | 0.4435 | 12.24% | -0.2052 | 0.2500 | 0/3 / 2/3 |
| walnuts | 3 | 3/3 / 0/3 | 0.2514 | 0.24% | -0.0916 | 0.2917 | 0/3 / 3/3 |

The aggregate median of object q99 medians was `1.8471%`. The all-cell D2
median correction cosine was `0.4186983`; the median object hard-null margin
was `-0.1439931`.

### Conditioning and continuation

| diagnostic | observed | frozen requirement | result |
|---|---:|---:|---:|
| maximum first-gradient ratio versus S2 | `7.59e-7` | `<=0.01` | pass |
| all-cell correction cosine | `0.4186983` | `>=0.4250057` | fail by `0.0063073` |
| improvement versus S2 correction cosine | `+0.0936927` | `>=+0.10` | fail |
| median object hard-null margin | `-0.1439931` | `>=-0.1005264` | fail by `0.0434668` |
| improvement versus S2 hard-null margin | `+0.0065332` | `>=+0.05` | fail |
| catastrophic q99 objects | `0` | `<=1` | pass |

The first D2 pre-clip gradient norms were `5.77`, `13.08`, `1.15`, and `9.56`
for can, fabric, vial, and walnuts, versus S2's `1.09e7--2.24e7`. Vector Huber
therefore isolated and fixed the zero-start conditioning pathology. Since both
held improvement floors failed, `continuation_gate_pass=false`.

### Cell gate decomposition

| D2 component | passing held cells |
|---|---:|
| correction cosine `>=0.70` | 1/12 |
| correction norm ratio in `[0.50,1.50]` | 4/12 |
| anomaly cosine `>=0.90` | 6/12 |
| anomaly norm ratio in `[0.80,1.25]` | 11/12 |
| complete clean+anomaly RN topology | 6/12 |
| hard-null correction margin `>=0.05` | 0/12 |
| strict synthetic contribution | 0/12 |
| RN-guard ablation contribution | 10/12 |
| q99 no-harm `>=-5%` | 12/12 |
| complete cell gate | 0/12 |

The static-mean correction was the strongest hard null in `10/12` cells and
endpoint KD in the other `2/12`. Thus the candidate never demonstrated that
its instance-conditioned shifted-assumed correction generalized better than
a support-derived static template.

### Arm medians across twelve held cells

| arm | q99 reduction | correction cosine | correction norm |
|---|---:|---:|---:|
| M0 endpoint KD | 2.94% | 0.2731 | 0.5612 |
| D0 direct clean | 1.43% | 0.4170 | 0.3596 |
| D1 + recomposition | 0.96% | 0.4184 | 0.3420 |
| D2 + RN guard | 0.95% | 0.4187 | 0.3417 |
| shuffled correction | -26.55% | 0.0869 | 3.9983 |
| random matched LoRA | -0.16% | -0.0035 | 0.1255 |
| static mean correction | 2.21% | 0.5333 | 0.6998 |
| paired-teacher oracle | 6.45% | 1.0000 | 1.0000 |

S3 raised D2 correction cosine from S2's `0.3250` to `0.4187`, but correction
norm fell from `0.7886` to `0.3417`, and D2 remained nearly indistinguishable
from D1. The RN term passed its local ablation in `10/12`, while the synthetic
cross-recomposition contribution fell from S2's partial `4/12` to `0/12`.
The changed loss therefore improved direction but did not establish robust
shift/anomaly factorization or instance-specific correction magnitude.

## Repeatability and Custody

The panel `can` run also differed from the two prerequisites. Three-way
adapter hashes failed, and normalized object payloads were not equal because
the observational panel's metrics changed slightly. This strengthens the
repeatability blocker; it is not a positive extra seed.

All object ledgers report an empty `forbidden_kinds_opened` list.
`TESTpublic`, private test, ground truth, query images, query-memory updates,
and public authorization remained false. No public metric was computed. Both
pulled run roots contain zero `anomaly_maps/` directories, and dsba3 GPUs 0--3
returned idle after execution.

## Conclusion

TARCD-LoRA-S3-VR answers the intended causal question. S2's huge zero-start
gradient was a real loss-conditioning defect, and normalized vector Huber
repairs it. That repair is insufficient: held correction remains weaker than
the preregistered improvement floors and every hard-null comparison, while
synthetic contribution disappears under the strict ablation.

Final verdict: `KILL_FOR_CLAIM / NO_CONTINUE`. No second seed, beta/loss/rank,
learning-rate, threshold, or public/test/private/GT branch is authorized. The
broad shifted-assumed-plus-synthetic research thesis remains a design-level
question, but this single-head backbone-LoRA line is closed. A future method
would need a separately preregistered architectural change that explicitly
factorizes nuisance shift correction from anomaly residuals and demonstrates
held instance-conditioned superiority over the static-mean template.
