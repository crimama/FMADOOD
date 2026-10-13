# C1-DIAG-AT-P16-can-vial-s3-v1: Execution Progress Run 1

Latest timestamped process observation: `2026-09-06 14:13:35 UTC`; progress `24/24`  
Record type: point-in-time operations progress report  
Scientific status: `NO_PRIMARY_METRICS / NO_CONCLUSION`

The latest recorded observation has six sealed groups and all 24 fits
complete. CALIBRATE v8 produced postseal sources before its process was
absent; a later inventory attempt failed before global inventory creation, and
no public or primary metric result is available. This file is not an
automatically updated live status.

The calibration attempts are recorded below, beginning with session `87432`
and including the later v7 termination and v8 postseal phase; no public
evaluation or source-held performance conclusion is implied by the fit
snapshot above.

Current handoff state is evaluation-only repair after the user's explicit
request to resume evaluation. Credential rotation was not independently
verified; only the safe `SSH_ASKPASS` path is authorized. The preserved
session receipts record prior failures; they do not imply a scientific result.

This file records execution progress only. It does not modify the frozen
protocol, grant execution authority, or create a new workflow state.

## Earlier snapshot: 2026-09-06 05:00:13 UTC

At the recorded timestamp:

- `can/20260906`: four fits, checkpoint seal, and cache cleanup completed;
  `0` cache-cell payload files remain after cleanup; evaluation has not run.
- `can/20260907`: four fits, checkpoint seal, and cache cleanup completed;
  `0` cache-cell payload files remain after cleanup; evaluation has not run.
- `can/20260908`: preflight in progress.
- Progress: `8/24` head fits completed at this snapshot.
- The global 24-head seal, calibration, held source diagnostic, public
  inventory, and public evaluation have not occurred.
- Remote run root: `/workspace/results_remote/c1_full_dsba1_20260906_run1`.
- The environment agent is continuing the remote run.

This is a timestamped snapshot; it must not be read as a claim that the same
count remains current after this record.

## Independent audit of the first group

The verifier's independent audit of the first completed group passed. The
audited fits each had:

- exactly `576` updates;
- exactly `241` head parameters;
- paired identical initialization, hash prefix
  `358768c2...`;
- backbone state hash prefix `59a755...`;
- support manifest hash prefix `7e93e...`;
- expected dsba1 host GPU 6 UUID;
- checkpoint SHA-256 matching the sealed checkpoint record.

The first group's actual cache total was `86,894,864` stored bytes. Its
theoretical raw size was `99,919,872` bytes, against a cap of `114,907,853`
bytes. After cleanup, only the manifest and runtime evidence remain; dense
maps count is `0`.

## Frozen custody and execution accounting

The full run is organized as six groups of four arms, for 24 planned fits.
The source-frozen runner hash prefix is `fa950549...`; the cache implementation
hash prefix is `c1_cachecd5d25...`. These prefixes are provenance labels from
the run evidence, not replacement hashes.

The earlier 44-test receipt and the actual setup smoke pass are setup and
implementation evidence only. They are not performance evidence and do not
substitute for the pending full fits, calibration, held diagnostic, public
inventory, or GT stream metrics.

## Pending evidence

No scientific metric, baseline comparison, held-transfer result, public
identity result, promotion/kill gate, or method conclusion exists at this
snapshot. The frozen protocol and its predeclared gates remain authoritative;
this progress record only reports the operational state above.

## Later snapshot: 2026-09-06 05:14:40 UTC

At this later observation:

- All three `can` scientific seeds (`20260906`, `20260907`, `20260908`) have
  four-fit receipts, checkpoint seals, and cache cleanup complete.
- Progress is `12/24` head fits complete.
- For each completed can seed, `0` cache-cell payload files remain after
  cleanup; evaluation has not run.
- `vial/20260906` preflight is in progress; its four fits are not complete.
- The first-group independent audit evidence above is unchanged. The two
  follow-up can groups are recorded here only as observed fit/seal/cleanup
  progress, without a new independent audit claim.
- Global 24-head seal, calibration, held source diagnostic, public inventory,
  and public metrics remain unexecuted.

This section is another point-in-time operations snapshot. It does not imply
that the listed state remains current after `2026-09-06 05:14:40 UTC`.

## Later snapshot: 2026-09-06 05:33:53 UTC

At this later observation:

- `vial/20260906`: four fit receipts, checkpoint seal, and cache cleanup are
  complete.
- The completed group retains `0` cache payload files and `0` payload bytes.
  Its manifest records `actual_grid=[65,48]`,
  `actual_total_bytes=54,635,390`, and `cap_bytes=71,817,408`.
- Four groups are sealed; progress is `16/24` head fits complete.
- `vial/20260907` arm `00` is currently fitting.
- This observation records training and storage management only. It is not a
  performance evaluation or scientific result.

The earlier snapshots and first-group independent audit remain unchanged. The
global seal, calibration, held diagnostic, public inventory, and public metrics
remain unreported in this point-in-time record.

## Latest snapshot: 2026-09-06 05:44:11 UTC

At this latest confirmed observation:

- Five groups are sealed; progress is `20/24` head fits complete.
- Each completed group has `0` remaining cache payload files after cleanup.
- The verifier independently passed all 20 completed fits: the original
  `can/20260906` evidence plus the 16 checkpoint/head-state hashes from the
  remaining four completed groups.
- Each audited fit has exactly `576` updates and `241` head parameters, with
  matching group initialization, object/seed/arm identity, support manifest,
  backbone state, and dsba1 GPU 6 binding.
- Observed actual cache totals are approximately 86.9 MB for can against a
  114.9 MB cap, and 54.6 MB for vial against a 71.8 MB cap.
- The final `vial/20260908` group and the global 24-head seal remain
  incomplete.
- No calibration, held diagnostic, public inventory, public evaluation, or
  performance metric result exists.

This is a confirmed point-in-time operations snapshot, not a live status.

## Latest snapshot: 2026-09-06 05:56:50 UTC

At this latest confirmed observation:

- Six groups are sealed and all `24/24` head fits are complete.
- All completed groups retain `0` cache payload files after cleanup.
- The verifier recorded `FIT_24_GATE_PASS`.
- The global seal SHA-256 is
  `34acdb4aff3629d8588f305d90e3d23f04b1a14fe25ad13b2cb010c538da25cc`.
- Identity custody contains exactly 24 records, with 24 unique records and
  no duplicates, covering exactly 2 objects x 3 seeds x 4 arms.
- All 24 checkpoints match their group-seal hashes. Every fit has exactly 576
  updates and 241 head parameters; paired initialization and source,
  support, backbone, and dsba1 GPU 6 provenance checks passed.
- Compact fit artifacts are preserved at
  `results/remote_runs/dsba1/full_fit_receipts_20260906` (approximately
  2.1 MiB); checkpoint and cache-payload copies are not retained there.
- Root approval was given to begin calibration and the source-held stage.
  No execution start or completion receipt exists for either stage at this
  snapshot.
- No held/public evaluation, GT stream metric, or scientific conclusion
  exists.

This is a confirmed point-in-time operations snapshot, not a live status.

## Calibration attempt: session 87432

The first calibration attempt ran for 43 seconds and failed at the operational
input contract because 12 canonical C1 checkpoint paths were required. Its
preserved receipt is
`results/remote_runs/dsba1/full_fit_receipts_20260906/calibrate_failure_87432.json`
with SHA-256
`857b5e8a1ddb4ffc60e3e4fbdbefbd37b5d8e8447bc73c8a70c4a3a1cf0e3e1e`.
The receipt records `postseal_sources_present=false`,
`ground_truth_accessed=false`, `metrics_generated=0`, `residual_process=false`,
and the unchanged global seal SHA
`34acdb4aff3629d8588f305d90e3d23f04b1a14fe25ad13b2cb010c538da25cc`.

The 13-file fit-source snapshot and its `source_sha256.txt` were preserved.
The current operational state is evaluation-only repair, awaiting review and
CPU reconciliation of the 24 completed fits before resuming. Reviewer work on
the full-SHA exact check and actual-failure binding remains in progress.
This is an operational execution failure, not a scientific failure or a
performance result. Cache payloads remain at zero and dense prediction maps
are not accumulated.

## Latest CPU evidence: evaluation-only repair

The latest CPU evidence records the following acceptance state for the
already-sealed fit artifacts:

- The evaluation/runner suites report `28 passed`; the factorial/cache and
  streaming base suites report `20 passed` with `11 warnings`; the acceptance
  manifest records `48` tests passed in total.
- The actual-24 CPU validation v2 receipt reports `PASS`, with 24 checkpoint
  records and 24 file-hash matches. It also records backbone provenance,
  `cuda_used=false`, `ground_truth_accessed=false`, and
  `dense_maps_persisted=false`.
- The original fit source, 24-fit result, and global seal remain immutable.
  The evaluation repair is a naming/validation-lineage correction and is not
  a new fit or a scientific result. A preceding six-failure CPU trace was
  classified as five fixture issues and one dependency-visibility issue; the
  later acceptance suites include the test-only fixes and synchronized
  existing dependency visibility.

The evidence is bound to these local artifacts and full SHA-256 values:

- `results/remote_runs/dsba1/full_fit_receipts_20260906/actual24_cpu_validation_v2_receipt.json` —
  `71df95be3807a6967a414ccb2314db0c3fac98b11bbb3b3d17e4762de783107d`
- `results/remote_runs/dsba1/full_fit_receipts_20260906/cpu_acceptance_manifest_v2.json` —
  `fb98ec13c3c5357b73742b0fcdfad249060cd965df837a5473b64854fe9d45bd`
- `results/remote_runs/dsba1/full_fit_receipts_20260906/evaluation_wiring_repair_receipt.json` —
  `a6d90a6ff576619aa98a3b7b18f8ec912b710bcfc21272b0ebed41fb8f606ce3`
- `results/remote_runs/dsba1/full_fit_receipts_20260906/fit_source_snapshot/source_sha256.txt` —
  `6a57a3471528523f7ddacab2aafb209aa313a34122f038fea9775c49c8debfde`

CALIBRATE has not been relaunched. The final independent CPU audit remains
pending; no postseal sources, held/public evaluation, GT access, or scientific
metric exists. The dsba1 host-GPU-6-only custody and zero retained cache
payload/dense-map policy remain in force.

## Latest operational repair update: three CALIBRATE attempts

The 24/24 fit seal and `FIT_24_GATE_PASS` remain unchanged. Three subsequent
CALIBRATE attempts have preserved failure receipts in the local full-fit
artifact directory. All three receipts record no postseal sources, no public
GT access, and zero metrics; these are operational failures, not scientific
failures or performance results:

- Session `87432`: adapter/input naming contract failure (`12 canonical C1
  checkpoint paths are required`), receipt
  `calibrate_failure_87432.json`, SHA-256
  `857b5e8a1ddb4ffc60e3e4fbdbefbd37b5d8e8447bc73c8a70c4a3a1cf0e3e1e`.
- Session `44682`: revision import-shadow/closure failure
  (`ModuleNotFoundError: No module named src.backbones`), receipt
  `calibrate_failure_44682.json`, SHA-256
  `a7a682559798d43c53c4f2520f7412feaaccee052c8f2f342fcf161f68bee8c2`.
- Session `65001`: direct-fit rotation initialization failure
  (`SuperAD object has no attribute _rotation`), receipt
  `calibrate_failure_65001.json`, SHA-256
  `ebdf4e31be061d41accdca52c6cf3e77002330383eb3f2e3590148014e1df76b`.

The operations-history manifest binds these attempts to the unchanged global
seal and records no persistent calibration maps. The source-corrected repair
lineage is retained at
`results/remote_runs/dsba1/full_fit_receipts_20260906/oldrepair_revision_source_snapshot/evaluation_revision_v2_source_manifest.txt`
with SHA-256
`dca67ba9794a7a69c1a95e8c4c68f2aed23f888ac96c330a34fd41d65345a0a8`;
the import-closure diagnostics are retained in the corresponding
`calibrate_import_closure_manifest_v1.txt` and
`calibrate_runtime_import_diagnostic_v1.txt`. The repair receipt is
`evaluation_wiring_repair_receipt.json`; this lineage does not replace the
original 13-file fit-source snapshot.

## CPU deployment acceptance and current handoff

The deployment-v3 acceptance artifact records an active `/workspace` bootstrap
with the unified runner, CPU-only `CUDA_VISIBLE_DEVICES`, `48 passed, 11
warnings in 15.37s`, actual-24 validation `PASS`, and an idempotent repair.
It also records no fit or CALIBRATE execution, no GT access, no dense-map
persistence, and unchanged source fit artifacts. Its full SHA-256 is
`1a29facdb1ca5b2d387a6d7c52bffeaeab7f23020d9bafa07a0a17f79d92ca41` at
`results/remote_runs/dsba1/full_fit_receipts_20260906/cpu_deployment_acceptance_v3.json`.
The operations-history manifest SHA-256 is
`c22adba17b644739b414485bae99d96b68a78a4ca1d4f00be0d31e76d2b15fe0`.

After architect clearance, the next evaluation-only repair requires explicit
`rotation=False` and `mask=False` assertions for each SuperAD/RN direct-fit
path, together with an append-only runtime-v2 lineage correction. Remote CPU
revalidation is in progress; no new PASS receipt or CALIBRATE relaunch receipt
exists yet. The active execution root is `/workspace`; the evaluation
snapshot remains audit-only. dsba1 host GPU 6 custody, zero retained fit/cache
payloads, and zero dense-map accumulation remain in force. The final
independent CPU audit and all scientific evaluation metrics are still pending.

## Latest CPU runtime gate and CALIBRATE v7 observation

The latest CPU runtime evidence is independently recorded as
`CPU_RUNTIME_V7_PASS`. The v7 receipt reports `65 passed, 11 warnings in
13.62s`, actual-24 validation `PASS` with `rows=24`, and a new write-once
runtime repair receipt with `rows=24` and second-invocation
`idempotent=true`. The validation is CPU-only and records no model inference,
CALIBRATE execution, public query, GT access, or dense-map persistence. Its
source review is ready for the next evaluation-only gate, with runner SHA
`a9181c1bb00b832dc417f98e21fd1e9e0031c99a151c89f5ae13c4f3dbd301c2` and
evaluation SHA
`58d6b02048e489550cc8a76dd165888ca1b912e1db92db6f002a8c554ec94e37`.

The evidence is bound to:

- `results/remote_runs/dsba1/full_fit_receipts_20260906/cpu_runtime_acceptance_v7.json` —
  SHA-256 `06913fa9832f3b2e7f6683afc89e7b600a9151adbd08e47b22fca6633c467e6a`;
- `results/remote_runs/dsba1/full_fit_receipts_20260906/cpu_runtime_acceptance_v7_trace.txt` —
  SHA-256 `ac833dd6de014a527f990e4c2890e23673c77a2724929bfda5d2a9cb997ea380`;
- `results/remote_runs/dsba1/full_fit_receipts_20260906/evaluation_runtime_repair_receipt_v3.json` —
  SHA-256 `472fbd2ef608c74ec205fda6caf321e42975ca8e8170eae14b792eaf3ac73070`;
- `results/remote_runs/dsba1/full_fit_receipts_20260906/source_snapshot_revision_v6_manifest.txt` —
  SHA-256 `60ee8edbcb0c0f1ebbd57eb81f882fc2e1aa15049aa344e4354e6472b6b4c299`.

Earlier v5/v6 CPU attempts are retained as operational history: the new
validator rejected an actual legacy field and payload-schema mismatch. These
were repaired without changing the model or the original fit artifacts.

At the observed timestamp `2026-09-06 07:48:58 UTC` (elapsed approximately
310 seconds), CALIBRATE v7 was running under the existing detached controller
(controller PID `13385`, worker PID `13394`) from the unified active root
`/workspace`. The environment observation reported two reference-fit log
groups at `16/16`; these logs correspond to one adapter/object's
`SuperAD.fit` and `RNFMLK.fit`, so object-level progress is not identifiable.
Host GPU 6 was mapped to container CUDA slot 0, with approximately
`13481 MiB` and `35%` utilization. This is an in-progress observation, not a
completion receipt: postseal sources were not yet present, and public GT or
new metrics had not been accessed or generated. The controller run log is
`calibrate_controller_v7.log` under the active run root
`/workspace/results_remote/c1_full_dsba1_20260906_run1/`.

Architect clearance currently covers the correction and held stage only.
Before direct-fit evaluation, each SuperAD/RN path must assert
`rotation=False` and `mask=False`, followed by the append-only runtime-v2
lineage correction. Remote CPU revalidation is ongoing; no new PASS or
CALIBRATE completion receipt is claimed. The original 24-fit/global-seal
lineage, fit-cache payload count (`0`), dense-map prohibition, dsba1 host-GPU-6
custody, audit-only evaluation snapshot, and absence of scientific metrics
remain unchanged.

The latest runtime observation supersedes the resource snapshot above for
current-process accounting: controller PID `13385`, worker/controller PIDs
`13393` and `13394`, elapsed `30:01`, CPU approximately `121%`, and GPU 6
approximately `13481 MiB` / `37%`. The CALIBRATE v7 process is still running;
no postseal source, completion receipt, or failure receipt was observed at
that point.

Future optimization note (not an execution change): bounded
`_raw_layer_memories` already contains the P16 features, while the calibration
P15 `memory_provider` re-extracts them. That re-extraction is not required by
the dense-map-to-disk-0 policy. This note does not authorize changing the
current run, source, or protocol.

## CALIBRATE v7 termination and custody

CALIBRATE v7 terminated at the observed timestamp `2026-09-06 08:23:28 UTC`
after approximately `2380` seconds. The held-source score/mask validation
failed before postseal completion. The preserved failure receipt records
`postseal_sources_present=false`, no completion status, no residual process,
and no retry; this is an operational failure, not a scientific failure or a
performance result.

The failure evidence is bound to:

- `results/remote_runs/dsba1/full_fit_receipts_20260906/calibrate_failure_v7.json` —
  SHA-256 `4641461ba687bf2acc3e951574db71ac7a7ee0df016adb03107d2885938c96e5`;
- `results/remote_runs/dsba1/full_fit_receipts_20260906/calibrate_failure_v7.log` —
  SHA-256 `dbc22a6277aad31b2d0f3f15787df10218620ec4cbaf0e2351eac4c29ff82e6f`;
- `results/remote_runs/dsba1/full_fit_receipts_20260906/calibrate_v7_custody_receipt.json` —
  SHA-256 `430cc77b7eed51ae5ee70e833f81da3eead55bd425da1a612860bb417bc61492`.

The custody receipt confirms 24 fit weights with zero checkpoint mismatches,
six group seals, and unchanged global-seal and runtime-v3 lineage. It records
zero dense/map-named payload files, zero calibration-created fit-cache
payloads, zero residual processes, and no established public GT, inventory, or
aggregate evaluation. The verifier's public-GT non-access conclusion is based
on the CALIBRATE code path and run-root evidence; no separate OS-open audit was
performed. Consequently, no valid scientific calibration or held metric exists.

The debugger diagnosis is that raw RN features require native 2-D validation
before head fitting. Architect guidance requires one shared out-of-arm
`cell_provider` invocation. Shape/provider/cache validation and the runner's
append-v4 repair remain in progress and are not yet verified. The original
24-fit artifacts, global seal, frozen source/protocol, and parameters remain
unchanged; no new execution is authorized by this record.

## Security hold

An SSH credential was exposed in tool stdout through direct askpass-helper
execution. No credential value or verbatim command is recorded here. The
scope of local tool output and external logs is not yet known. Remote
reconnect, synchronization, and execution are therefore paused pending user
password rotation and environment/secret update. This document contains no
secret material.

## Current local repair state

This section records an earlier local snapshot and is superseded by the final
local handoff state below.

The evaluator source review is `PASS` for evaluator SHA
`e5582f4ec042b20af5f86aa081924a5d9493b945d682b7c268b77c3d31f897d6` and
evaluation-test SHA
`4617cfde9ca6d557e48fbe19f38251ab3caf55d8a2e66e6185fe42b7b3ce5e47`.
The preceding shape/provider/cache repairs and the three review findings were
covered by that source review; dynamic tests requiring Torch were skipped.

A standalone runner check reported `39 passed`, but the integrated review
keeps the evaluator `NOT_READY` because it found these unresolved contract
items:

- P0: restoration from JSON inventory dictionaries to `MapInventoryEntry` is
  missing;
- P1: actual-inventory hash and coverage revalidation is missing;
- P1: the postseal top-level `24/96` inventory requirement is not enforced.

The debugger is applying the minimal runner/test-only repair. No v4 receipt
has been created, and there has been no remote synchronization, CPU retry, or
CALIBRATE retry. The security hold remains active. This is not a final handoff
or scientific readiness decision; no GT/public metric or performance result
exists.

## Final local handoff state before security release

This section is a pre-resume snapshot; its security-hold and remote-idle
statements are superseded by the v8 update below.

Within the reviewed local scope, the source gate is
`SOURCE_REVIEW_READY_FOR_REMOTE_CPU`. The independent check reports `40
passed`, `1 skipped`, and `py_compile PASS`; the four reviewed files have
identical pre/post hashes. The reviewed source identities are:

- runner: `ead2113b5a0cba797865117b634bcec0c328a86922e8a386f68963415a22bb30`;
- runner test: `c39db616464a07d6386f2139cc04fc148c5289195fb04a80d39b9cd93a8faa22`;
- evaluator: `e5582f4ec042b20af5f86aa081924a5d9493b945d682b7c268b77c3d31f897d6`;
- evaluator test: `4617cfde9ca6d557e48fbe19f38251ab3caf55d8a2e66e6185fe42b7b3ce5e47`.

The bounded review closed the schema/phase-tampering assertions and includes
JSON inventory restoration, actual-inventory hash and coverage checks, and
the held `24/96` contract. No P0/P1 remains within this reviewed scope.
Previous report identifier `9bd` and measured identifier `6cf` differ for a
retrospectively unresolved reason; both remain unconfirmed history and were
not used for remote execution or a v4 receipt.

The remote state remains idle on the old v7 source and immutable 24-fit
weights after the CALIBRATE v7 failure. No new source sync, CPU gate, v4
repair receipt, recalibration, GT access, or metric generation has occurred.
The security hold remains active until the user confirms SSH password rotation
and environment update. No secret value is read or stored.

The authorized resume order is: confirm credential rotation; take and verify
the exact source snapshot and synchronize it; run CPU actual-24 plus v4
write-once/idempotency and the complete CPU suites; pass the independent CPU
gate; retry CALIBRATE on the same 24 weights only; audit postseal sources;
perform baseline parity and GT-free global inventory; then run GT pass 2 and
aggregate. This is a local handoff plan, not evidence that any later step has
started or completed.

## CPU runtime v8 and CALIBRATE v8 current observation

Following the user's explicit request to resume evaluation, the security hold
was released for this scoped operation. Password rotation was not independently
verified; only the safe `SSH_ASKPASS` path is authorized, and no credential
file or secret value was read or stored.

The independent CPU runtime v8 gate passed a fresh compact recheck with `81
passed`, `11 warnings`, and `13.24s`. It also passed actual-24 validation,
runtime-v4 write-once, and idempotency checks. The original raw stdout was not
persisted; the compact recheck trace is the evidence for this claim. The
acceptance artifact records the same 24 weights and unchanged global seal,
with no scientific training/evaluation, GPU inference, public GT, or dense-map
generation.

Evidence paths and SHA-256 values:

- `results/remote_runs/dsba1/full_fit_receipts_20260906/cpu_runtime_acceptance_v8.json` —
  `1a2338103de6735905586f1b40fc0ed23a56c4a108254ac7e534fa5c5c62f235`;
- `results/remote_runs/dsba1/full_fit_receipts_20260906/cpu_runtime_acceptance_v8_recheck_trace.txt` —
  `75e766a7596308edec21cf54eb14e58ed59ac5f6dbdfda09eb3503e62ceaff7f`;
- `results/remote_runs/dsba1/full_fit_receipts_20260906/evaluation_runtime_repair_receipt_v4.json` —
  `9fb33550376f27a3366d8097724ccc9f0bff396a6cbf2064031a40b74fa56620`;
- `results/remote_runs/dsba1/full_fit_receipts_20260906/evaluation_revision_v8_source_manifest.txt` —
  `6dbdd295aa77b0e0b2d0c80369f4d603f1d15b75caa0f98ca4cae3d57c6e8657`.

At the read-only observation `2026-09-06 13:54:36 UTC`, CALIBRATE v8 was
still running as a single task under PID `14669`, elapsed `731s`, with CPU
approximately `97.2%`. Host GPU 6 retained the known UUID and the observation
reported `13481 MiB` memory with `0%` utilization. No controller start receipt
was available, so no start time is inferred from elapsed time. GPU 0 is not
interpreted as stalled or finished. The controller log is
`calibrate_controller_v8.log`; postseal sources and a completion status were
not present, and no error pattern was observed at that observation.

The active remote source remains the immutable v8-audited snapshot and the
original 24 fit weights. No completion, metric, or scientific success claim
is made. The next stop condition is a receipt-backed CALIBRATE result followed
by postseal audit; until then, do not proceed to public GT or aggregate
metrics.

## CALIBRATE v8 postseal and inventory outcome

CALIBRATE v8 completed its postseal-source phase; PID `14669` was absent at
the observed `2026-09-06 14:13:35 UTC` check. The detached completion receipt
does not preserve an explicit process exit code. Independent
`POSTSEAL_V8_PASS` evidence confirms 24 calibration packages with 16 maps
each (`384` maps total) and `24` held cells with `96` held maps. The held
scores, approximately pAUROC `0.48718` and AU-PRO `0`, are secondary
synthetic diagnostics only; they are not public identity, baseline efficacy,
or real-shift robustness results.

Postseal evidence is bound to:

- `results/remote_runs/dsba1/full_fit_receipts_20260906/postseal_sources_v8.json` —
  SHA-256 `2f1ae1d8fbd967d4e5db8ea9798af16438429591e4b584b2b15c5dfc860916db`;
- `results/remote_runs/dsba1/full_fit_receipts_20260906/calibrate_controller_v8.log` —
  SHA-256 `58e31633c9eef7e44037f1a48b1c2aa0b48ea06528f2dd66e0191ac94427d6db`;
- `results/remote_runs/dsba1/full_fit_receipts_20260906/calibrate_v8_completion_receipt.json` —
  SHA-256 `c90151c47a6a87b7fdb05f2ba8adf165774a234ff63daa425bb7723b1d57959b`;
- `results/remote_runs/dsba1/full_fit_receipts_20260906/postseal_ready_v8_trace.txt` —
  SHA-256 `f451b3c3021307217f7a80d87b0cb91bc944c611037180b575ee0c8013517ca2`.

The subsequent INVENTORY v8 task exited on duplicate `object_name` handling
(`Namespace` construction at runner line 1136, with the same issue observed
at lines 1157 and 1241). The global inventory was not created, and public GT,
evaluate, and aggregate phases remain at zero. The malformed status artifact
and log are preserved, but no reliable process exit code is inferred:

- `results/remote_runs/dsba1/full_fit_receipts_20260906/inventory_status_v8.json` —
  SHA-256 `e0bbb1ef2af3363766882ae55def60b8457abb3999fc4faceb88c1e1fc74c93e`;
- `results/remote_runs/dsba1/full_fit_receipts_20260906/inventory_controller_v8.log` —
  SHA-256 `61e0d8f2fce481e8b610d1f9ed9d83a0f1eeca0b20e48e420cc9a839610b8234`.

The original 24 weights, global seal, source snapshot, and CALIBRATE v8
postseal artifacts remain immutable; dense payload persistence is zero. The
narrow repair and provenance review remains open. No retraining or CALIBRATE
rerun is authorized, and no public performance conclusion may be drawn from
the held synthetic diagnostic.

## CPU runtime v9 and public-inventory gate status

The independent V9 CPU verifier reports `81 passed`, `11 warnings`, and exit
code `0`. It confirms the real v5 runtime repair and postseal reuse, with
current runner SHA
`172c86a90a1fe77c51b63fbccb57df125a17f3e2e100c85ca874af7e6f816670` and
unchanged evaluator SHA
`e5582f4ec042b20af5f86aa081924a5d9493b945d682b7c268b77c3d31f897d6`.
The minimal namespace and schema-envelope corrections are covered by this
CPU-only verification; old v4 and postseal artifacts were reused, with no
recalibration. The V9 acceptance manifest is
`results/remote_runs/dsba1/full_fit_receipts_20260906/cpu_runtime_acceptance_v9/manifest.txt`
(SHA-256 `626b802e8382d7d166da4dcd96877a2e93075281f1b436bc9af347e2588b2dff`),
and the v5 repair receipt is
`results/remote_runs/dsba1/full_fit_receipts_20260906/cpu_runtime_acceptance_v9/receipt/evaluation_runtime_repair_receipt_v5.json`
(SHA-256 `536c236167ee3d5ddfc093e9d27cba9be50ae04287b473ef9068e6f195f88122`).

The separate public-inventory route has not restarted. Its first tiny CPU
fixture route failed on legacy arm names, and `route_retry1` failed on the GT
callback count; both are unit-fixture failures, not scientific results. Both
manifests explicitly record no inventory launch and no scientific phase:

- `.../cpu_runtime_acceptance_v9/route/manifest.txt` — launcher exit `1`,
  invalid legacy arm identity;
- `.../cpu_runtime_acceptance_v9/route_retry1/manifest.txt` — launcher exit
  `1`, `evaluated gt_open_count=4` versus `gt_calls length=2`.

No public inventory, public metric, or effect conclusion exists. The V9 audit
staging is not a deployed remote source; its source manifest records the
staged-only state and no remote synchronization.

## Held synthetic diagnostic evidence

The held results are a secondary synthetic diagnostic only. They are pooled
over `12` maps per arm (`3` seeds × `4` held bases), not seed mean ± SD, and
`public_identity_comparable=false`. All held pAUROC values are the
standardized raw partial-AUC lower bound
`0.48717948717948717`, which is below random `0.5`; all held AU-PRO values are
`0`. Shape, finite-value, and nonconstant checks passed. A sign-flip cause is
not established by this evidence.

| object | C00 | C01_T | C10_A | C11_AT |
|---|---:|---:|---:|---:|
| can held F1 | 0.092243623808584 | 0.092243619406290 | 0.092464851189867 | 0.093160122263085 |
| vial held F1 | 0.102396573623208 | 0.102392298921878 | 0.114914903307724 | 0.114765672072231 |

These held diagnostics do not support a public identity, baseline, or
effectiveness claim. No public metric or primary effect conclusion is
available; further route repair remains operationally gated.
