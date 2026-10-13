# TARCD-LoRA-S2-DCR Stage -1 Report

Date: 2026-08-26  
Protocol: `tarcd-lora-s2-dcr-stage-minus1-support-adapted-20260826`  
Status: `COMPLETED / SINGLE_SEED_SUPPORT_ADAPTED_PANEL`  
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## Negative Evidence Intake

TARCD-LoRA-S1 was valid on four objects and twelve held cells, but retained
the paired teacher correction in `0/12`. Its candidate median correction
cosine and norm ratio were `0.0667` and `0.3435`, and its median q99 reduction
was `0.32%` across held cells. S1 is frozen: this experiment did not reopen its
seed, loss-weight, rank, threshold, or generator-strength choices.

S2-DCR was preregistered as a separately named mechanism. It directly
supervises paired clean and anomaly correction, adds anomaly
cross-recomposition, and constrains exact-RN geometry only outside a frozen
slack region. The experiment tests whether these changes solve S1's
wrong-direction and under-correction failure without absorbing anomaly signal.

## Motivation

The retained research hypothesis is that shifted-assumed augmentation and
synthetic anomalies must both participate in learning because normal support
alone cannot identify which representation residual is nuisance and which is
anomaly evidence. The bounded question here is narrower: can direct paired
correction plus synthetic cross-recomposition teach a target-support LoRA
whose held correction generalizes beyond both the fitted support bases and the
fitted anomaly generators?

## Implementable Design

- Backbone: frozen DINOv2 ViT-L/14 with rank-4, alpha-8 fused-qkv LoRA in
  blocks 8--11; every non-LoRA parameter remains frozen.
- Seed: exactly one, `20260826`.
- Fit set: exact target SuperAD-16 `train/good` paths per object, manifest
  SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Held set: four deterministic support-disjoint `train/good` bases per object;
  all descendants remained sealed from fitting and model selection.
- Fit interventions: three shifts and three synthetic families; evaluation
  holds out `G2_structural_scratch` over all three shifts.
- Schedule: `16 x 3 x 3 = 144` deterministic updates per trained arm, AdamW
  at `1e-4`, bf16 forward, fp32 loss, gradient clip `1.0`, and no early stop.
- Activity weighting: object-specific q25 of nonzero teacher correction norms,
  with floor `1e-4`.
- Candidate `D2_DCR_RN_GUARD`: direct clean correction, direct anomaly
  correction, anomaly recomposition, clean/anomaly identity, and slack-only RN
  topology penalties.
- Ablations: `D0_DIRECT_CLEAN`, `D1_DIRECT_RECOMPOSITION`, and frozen S1
  `M0_ENDPOINT_KD`.
- Hard controls: shuffled correction, random norm-matched LoRA, static mean
  correction, zero-init identity, frozen teacher, and paired-teacher oracle.

The complete equations, thresholds, custody boundary, and sequential verdict
rule were frozen in `preregistration.md` before implementation and execution.

## Evaluation Alignment

This is a support-only mechanism diagnostic on MVTec AD2 single-image data,
not a segmentation benchmark. The panel is `can/fabric/vial/walnuts`, with
four held bases, three held shift cells, and one held synthetic generator per
object. A cell requires correction direction/scale, anomaly
direction/scale, clean and anomaly RN topology, hard-null superiority,
synthetic ablation contribution, RN-guard contribution, and q99 no-harm.

`TESTpublic`, private test, ground truth, query images, query-memory updates,
and public authorization remained forbidden. The implementation has no input
path for those sources.

## Code Modification / Creation

- `src/flow_tte/tarcd_lora_s2_dcr.py`: separately named DCR loss, activity
  weighting, quiet-target control, and slack RN guard.
- `scripts/run_tarcd_lora_s2_dcr_stage_minus1.py`: custody-checked single-seed
  runner, all arms, held-cell evaluation, and compact evidence emission.
- `scripts/aggregate_tarcd_lora_s2_dcr_stage_minus1.py`: fail-closed sentinel
  and panel aggregation with frozen gates and verdict logic.
- `scripts/run_tarcd_lora_s2_dcr_stage_minus1_remote.sh`: parameterized dsba3
  GPU controller, GPU0 sentinel, four-GPU panel, compact pull, and dense-map
  cleanup.
- Four focused test modules cover the core, runner, aggregator, and shell
  controller contracts.

Frozen S1 source files and prior killed protocols were not modified.

## Added Code Evaluation

- `py_compile` passed for the S2 core, runner, and aggregator.
- Focused pytest passed: `15 passed` for the four new test modules.
- Critical Python static checks passed for syntax/runtime-name errors.
- `bash -n scripts/run_tarcd_lora_s2_dcr_stage_minus1_remote.sh` passed.
- Remote-wrapper self-test returned `selftest=passed`.
- Deterministic synthetic support-only smoke passed with finite nonconstant
  maps, exact no-op controls, finite loss/gradients, and all forbidden-access
  flags false.
- Actual dsba3 CUDA/bf16 DINOv2 model-path preflight passed before the
  scientific run.
- Frozen-config SHA-256:
  `7d40f5722a044391ee3349f5c80a784352f3511a86ce20c896dced039bca0034`.

## Remote Execution

Successful immutable run:
`tarcd_lora_s2_dcr_stage_minus1_dsba3_20260826T013826Z_single_seed`.

- Fixed container: `hun_fsad_tta_012`.
- GPU0 `can` sentinel passed custody, schema, finite/nonconstant, and no-op
  validity; expansion used validity only, not scientific performance.
- Panel allocation: GPU0 `can`, GPU1 `fabric`, GPU2 `vial`, GPU3 `walnuts`.
- Start/completion: `2026-08-26T01:39:01Z` to
  `2026-08-26T03:18:02Z`; status `complete_support_only`, stopped after
  `panel_support_only`.
- Snapshot archive SHA-256:
  `fd761c8305d1e571b60ee8ae151e19a7253b19e9f954e818d854e0fbd09b056b`.
- Compact hash-ledger SHA-256:
  `b7c75da14d109654da1249e9d0a3defe502dac3df13ad31f4d32cdc23c354f47`;
  all 25 ledger entries passed after pull.
- Local S2 core, runner, and aggregator hashes exactly match the hashes
  embedded in the remote aggregate.
- Pulled compact root:
  `results/remote_runs/dsba3/tarcd_lora_s2_dcr_stage_minus1_dsba3_20260826T013826Z_single_seed`.
- Independent local reaggregation was byte-identical to the remote aggregate;
  both have SHA-256
  `88a02e28fde0c57bbccf96824d2a0fecee1345740010d5d43f9604cbb7b3f962`.

## SuperAD Baseline and Unified Metrics

`superad_seg_AUROC_0.05=N/A`, `method_seg_AUROC_0.05=N/A`,
`delta_vs_superad=N/A`, and `benchmark_comparable=false`. No public query was
authorized because this support-only mechanism gate failed.

## Results and Analysis

All four objects and all twelve held cells were valid and finite/nonconstant,
but no cell and no object passed the complete mechanism gate.

| object | GPU | valid/pass cells | D2 q99 reduction | hard-null correction margin | generator-ID BA | synthetic / RN pass cells |
|---|---:|---:|---:|---:|---:|---:|
| can | 0 | 3/3 / 0/3 | 11.39% | -0.2013 | 0.2708 | 0/3 / 2/3 |
| fabric | 1 | 3/3 / 0/3 | 2.21% | -0.0902 | 0.3750 | 2/3 / 2/3 |
| vial | 2 | 3/3 / 0/3 | 17.94% | -0.2670 | 0.2708 | 0/3 / 0/3 |
| walnuts | 3 | 3/3 / 0/3 | 0.97% | -0.0997 | 0.3333 | 2/3 / 3/3 |

The aggregate median of object medians was `6.80%` q99 reduction, but the
aggregate hard-null correction margin was `-0.1505`. Fabric additionally
failed the frozen generator-ID ceiling (`0.375 > 0.35`). Support LOO ranges
were finite (`0.1033--0.3710`), identity no-op and duplicate-RN maximum errors
were exactly zero, and no catastrophic q99 object occurred.

### Cell gate decomposition

| D2 component | passing held cells |
|---|---:|
| correction cosine `>=0.70` | 0/12 |
| correction norm ratio in `[0.50,1.50]` | 6/12 |
| anomaly cosine `>=0.90` | 6/12 |
| anomaly norm ratio in `[0.80,1.25]` | 11/12 |
| complete clean+anomaly RN topology | 6/12 |
| hard-null correction margin `>=0.05` | 0/12 |
| strict synthetic contribution | 4/12 |
| RN-guard ablation contribution | 7/12 |
| q99 no-harm `>=-5%` | 12/12 |
| complete cell gate | 0/12 |

Correction direction remains the universal fatal failure. D2 median cosine
was `0.3250` with range `0.0156--0.6454`, never reaching `0.70`. It beat M0
in only `5/12` cells and clean-only D0 in only `4/12`; the static-mean control
was the best hard correction null in `10/12` cells. This means the fitted
instance-conditioned adapter did not generalize better than a simple
fit-support shift template.

### Arm medians across twelve held cells

| arm | q99 reduction | correction cosine | correction norm | anomaly cosine |
|---|---:|---:|---:|---:|
| M0 endpoint KD | 2.94% | 0.2731 | 0.5612 | 0.9053 |
| D0 direct clean | 3.36% | 0.3795 | 0.8310 | 0.8883 |
| D1 + recomposition | 2.19% | 0.3241 | 0.7875 | 0.8880 |
| D2 + RN guard | 2.22% | 0.3250 | 0.7886 | 0.8882 |
| shuffled correction | -19.34% | 0.1272 | 3.8332 | 0.7862 |
| random matched LoRA | -0.16% | -0.0040 | 0.1765 | 0.9107 |
| static mean correction | 2.21% | 0.5333 | 0.6998 | 0.9102 |
| paired-teacher oracle | 6.45% | 1.0000 | 1.0000 | 1.0000 |

Direct supervision is a real improvement over S1: S1 candidate medians were
`0.0667` correction cosine, `0.3435` norm ratio, and `0.32%` q99 reduction,
whereas S2 D2 reached `0.3250`, `0.7886`, and `2.22%`. The improvement is not
enough for the absolute causal gate, and anomaly cosine fell from `0.9119` to
`0.8882`.

Synthetic cross-recomposition reduced raw recomposition error relative to D0
in `10/12` cells, so it is not wholly inactive. Only `4/12` met the stricter
`<=0.90` ratio plus correction no-harm condition, however. The evidence
therefore supports keeping synthetic anomalies as a candidate source of
supervision, but does not show robust shift/anomaly separation in this loss.

The RN slack term was small or inactive and did not materially separate D2
from D1: median correction cosine differed by less than `0.001`, median q99
reduction by less than `0.04` percentage points, and the preregistered RN
ablation gate passed `7/12`. Its current weight/form is not a demonstrated
mechanism contribution.

Training losses fell strongly over 144 steps, so this is not simply a failure
to optimize the support objective. A plausible but unproven explanation is a
generalization/identifiability failure: direct loss fits support descendants
but does not infer an instance-conditioned held correction. The first-step
correction cosine gradients were also extremely large and clipped (millions
to tens of millions), which may put zero-init LoRA in a poorly conditioned
regime; this is an inference from diagnostics, not a separately isolated
causal result.

### Repeatability audit

The repeated `can` sentinel and panel produced exactly identical candidate
held-cell metrics and training scalar summaries, but D2 adapter bytes were not
identical (`5ba44e87...` versus `98ccb99b...`). D0, M0, and shuffled states
also differed; D1, zero-init, and random-matched state hashes repeated. Thus
metric-level repeatability was sufficient to leave the negative verdict
unchanged, but bitwise training determinism remains unresolved and would be a
hard prerequisite for any separately named successor.

## Continuation Assessment

The preregistered continuation conditions were not met: object mechanism pass
was `0/4`, cell pass was `0/12`, and median hard-null correction margin was
negative. No second seed, rank sweep, loss-weight sweep, threshold sweep, or
public branch is authorized for this protocol.

The broad idea that shifted-assumed augmentation and synthetic anomalies
should be used together is not refuted. What is rejected is this particular
single-head LoRA/direct-correction realization. If a future separately named
method is designed, it should first address target identifiability and
conditioning rather than tune coefficients: for example, an explicit
factorized shift head with a separately protected anomaly residual, a
held-generator conditional correction target that must beat the static-mean
template, and a well-conditioned non-cosine warm start. Those changes require
a new preregistration and another support-only gate.

## Conclusion

TARCD-LoRA-S2-DCR is experimentally valid but fails its claim. Direct paired
supervision materially improves S1's correction direction and scale, while
synthetic recomposition shows partial activity, yet the learned correction
passes neither the absolute direction gate nor the hard-null comparison in
any held cell. Final verdict: `KILL_FOR_CLAIM / NO_CONTINUE`.

## Post-Conclusion Storage Cleanup

Only compact JSON/log/hash evidence was pulled. The result tree contains zero
`anomaly_maps/` directories, and dsba3 GPUs 0--3 returned to approximately
1 MiB allocation and 0% utilization after completion. Public/test/private/GT
and query access flags are false for every panel shard.
