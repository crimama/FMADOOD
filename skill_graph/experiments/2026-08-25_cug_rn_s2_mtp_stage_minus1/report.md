# CUG-RN-S2-MTP Stage -1 Report

Date: 2026-08-25  
Protocol: `cug-rn-s2-mtp-stage-minus1-support-only-20260825`  
Status: `COMPLETED / SUPPORT_ONLY_PANEL_STOP`  
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## Negative Evidence Intake

This separately named last bounded continuation directly targets CUG-RN-S1's
two observed failures: matched controls absorbed its endpoint gains, and exact
RN neighbor/distance-scale topology failed on `can`/`walnuts`. It does not tune
S1 seeds, update norm, step count, learning rate, or target thresholds.

## Motivation

S2 moves both S1 post-fit failure modes into the source objective: superiority
over update-budget-matched nulls is a hinge-style margin, while raw global-RN
neighbor identity and multiplicative tail-width scale are hard loss terms.

## Implementable Design

See `preregistration.md`. The frozen change is a worst-constraint source
objective over matched-null endpoint margins, raw RN neighbor identity, and RN
distance-scale preservation.

## Evaluation Alignment

- Dataset: MVTec AD2 single-image.
- Population: exact support-16 plus disjoint `train/good` held normals only.
- Benchmark comparable: false.
- SuperAD segmentation metrics: N/A; no public/test/GT access is authorized.
- Stop point: `can` validity sentinel followed conditionally by the four-object
  support-only panel; never all-eight in this protocol.

## Code Modification / Creation

- Added isolated S2 core, runner, fail-closed aggregator, and dsba3 wrapper.
  The primary files are `src/flow_tte/cug_rn_s2_mtp.py`,
  `scripts/run_cug_rn_s2_mtp_stage_minus1.py`,
  `scripts/aggregate_cug_rn_s2_mtp_stage_minus1.py`, and
  `scripts/run_cug_rn_s2_mtp_stage_minus1_remote.sh`.
- Re-fit `N_META`, `S_PROJ`, and `N_PAIR_SHUFFLE` on the same 16-anchor
  global-RN source path as the candidate; `I_N` is a fourth direct comparator.
- Added an independent `1e-7` duplicate-determinism validity bound, exact
  distributed panel-shard membership, full frozen-config/access validation,
  aggregator hashing, and a pinned dsba3 ED25519 host key.
- Added focused core, runner, aggregator, and wrapper regression tests. Frozen
  CUG-RN-S1 source/protocol files were not modified.

## Added Code Evaluation

- Python compilation and shell syntax: passed.
- New S2 focused tests: `16 passed`.
- S1+S2 adjacent regression: `31 passed`.
- Ruff critical `E,F` checks: passed.
- Remote-wrapper selftest: passed.
- Independent pre-launch implementation/custody review: `READY`.
- Runner/aggregator frozen-config equality: true, SHA-256
  `cdf990b5943c69bb61fe0730114548153effb1c70be9ac021cccd9be0201f746`.
- The aggregator rejects missing or extra shard directories instead of
  silently ignoring non-panel objects.
- Synthetic support-only smoke is finite, nonconstant, changes the score,
  preserves the no-op within `5.97e-8`, produces a valid operator and exact
  `0.10` update norm, and records four unique training rotations.

## Remote Execution

- Immutable run:
  `cug_rn_s2_mtp_stage_minus1_dsba3_seq_20260825_v1`.
- Fixed container `hun_fsad_tta_012`; GPU0 `can` sentinel followed, only after
  `valid 12/12`, by panel shards on host GPUs `0/1/2/3` for
  `can/fabric/vial/walnuts`.
- Run completed as `complete_support_only` and stopped after `panel`.
- Support manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Frozen-config SHA-256:
  `cdf990b5943c69bb61fe0730114548153effb1c70be9ac021cccd9be0201f746`.
- Snapshot SHA-256:
  `9e93427db87f5a2b563f838aa090e3febbfcb0c5450e2308ed80c8562e67b754`.
- Pulled compact evidence:
  `results/remote_runs/dsba3/cug_rn_s2_mtp_stage_minus1_dsba3_seq_20260825_v1`.

## SuperAD Baseline and Unified Metrics

Same-condition SuperAD evaluation remains unopened. The exact support manifest
is frozen for custody only; `superad_seg_AUROC_0.05=N/A`,
`method_seg_AUROC_0.05=N/A`, `delta_vs_superad=N/A`, and
`benchmark_comparable=false`.

## Results and Analysis

The remote and independently repeated local aggregates were byte-identical.
All 48 panel rows were valid with zero duplicate error, but no row passed
either direct matched-null margin and no object passed topology or full
activity.

| object | valid | safe | normal margin | anomaly margin | rank | neighbor | scale | topology | full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| can | 12 | 10 | 0 | 0 | 12 | 11 | 0 | 0 | 0 |
| fabric | 12 | 11 | 0 | 0 | 12 | 1 | 3 | 0 | 0 |
| vial | 12 | 12 | 0 | 0 | 12 | 0 | 0 | 0 | 0 |
| walnuts | 12 | 1 | 0 | 0 | 12 | 0 | 0 | 0 | 0 |
| total | 48 | 34 | 0 | 0 | 48 | 12 | 3 | 0 | 0 |

This is not merely a held-target generalization failure. On the source fits,
normal and anomaly margins remained below the frozen floor in `48/48` rows,
neighbor and scale violations remained nonzero in `48/48`, and no source fit
satisfied all four constraints. Final source loss ranged from `0.006086` to
`0.010955` (median `0.008567`). Thus the max-constraint optimizer did not find
a feasible operator even on its training episodes.
This is operational infeasibility under the frozen rank-8, norm-0.10,
60-step protocol, not a mathematical proof that no global solution exists.

Target-side candidate normal-tail reduction was positive in `37/48` rows and
safe-positive held in `34/48`, so the update was not numerically inert.
However, it beat `I_N` on normal reduction in `0/48`, evaluation-random Q95 on
normal reduction in `0/48`, and evaluation-random Q95 on held-family contrast
in `0/48`. Median matched normal/anomaly margins were negative for every
object. The added objective therefore did not create uniquely learned value;
matched orientations and simple controls absorbed all gains.

More specifically, the candidate failed to beat `N_META` on normal reduction
in `45/48` rows and failed to beat `S_PROJ` on held-family contrast in `42/48`.
It was trained against four spectrum rotations but lost to the disjoint
16-rotation evaluation Q95 on both endpoints in every row. This exposes
operator-space null generalization as a separate failure from target-object
generalization.

RN rank was nearly unchanged and passed `48/48`. Absolute candidate neighbor
retention remained high (object medians `0.9872`--`0.9941`) and median absolute
log scale error was only `0.00176`--`0.00398`, but the controls preserved those
quantities better. The comparative neighbor gate passed only `12/48` and scale
only `3/48`. Relative to S1, `can` neighbor rows improved from `0/12` to
`11/12`, but total neighbor rows fell `24 -> 12`, scale rows fell `35 -> 3`,
safe-positive rows fell `48 -> 34`, and both margin counts fell to zero.

The sentinel and panel `can` row payloads and operator hashes were identical.
All six forbidden-access flags were false in every object and manifest;
forbidden access ledgers were empty, and code/snapshot hashes were identical
across shards.

## Continuation Assessment

The panel gate required valid `4/4`, topology `4/4`, and full activity on at
least `3/4`. The observed counts were valid `4/4`, topology `0/4`, full `0/4`.
Accordingly this exact CUG-RN-S2-MTP branch is closed. No threshold, weight,
seed, step-count, or operator retry is licensed, and no all-eight/public run is
authorized.

## Conclusion

Directly optimizing margin plus RN neighbor/scale preservation did not rescue
the CUG meta-update. The decisive evidence is upstream of target transfer:
the same constraints were unsatisfied on every source fit. The appropriate
next research step, if pursued under a new preregistration, is a source-only
feasibility assay of the RN-isometric update set before training another
operator. A subsequent separately named safety-certificate/abstention assay
could test whether support-LOO RN gaps and tail-width stability can identify
the rare feasible fixed update while defaulting to raw B0, but it must not
reuse the CUG operator or synthetic objective. Otherwise retain exact RN/no
update and close the broader RN-update/TTA lane.

## Post-Conclusion Storage Cleanup

Compact artifacts occupy `2,695,099` bytes. Local and remote result roots each
contain zero `anomaly_maps/` directories. After completion, dsba3 GPUs 0--3
were all idle at `1 MiB`, `0%` utilization.
