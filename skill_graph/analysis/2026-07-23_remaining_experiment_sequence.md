# Remaining AD2 Experiment Sequence

Date: 2026-07-23  
State: `SEQUENCE_FROZEN / REMOTE_EXECUTION_BLOCKED`

## Purpose

This note replaces the executable order in the 2026-07-13 model-ladder plan
where later evidence has changed the detector contract. It does not alter the
historical record. In particular, the 2026-07-19 mechanism audit found DVT
mixed/ranking-only, density immaterial, Flow without DVT positive on the
support-only assay, and layer 17 dominant over the fixed four-layer envelope.

The next jobs therefore cannot inherit the older DVT-on, density-0.25,
guided/morphology detector as though it were already frozen.

## Sequential gates

1. **Comparable anchor and mechanism confirmation**
   - Run the joint Same16 Basic/SuperAD/RN-FMLK and
     A0/A6/A7/A6-L17/L17 public-shadow controller.
   - Require all eight objects, exact support identities and manifest hash,
     original-resolution pAUROC/F1/AU-PRO, complete map coverage, and both
     independent aggregations before cleanup.
   - This is the only runnable experiment at the current gate.

2. **Freeze one detector branch**
   - A6 promotion requires both the simplification gate and A6-over-A7 Flow
     hard-null gate. If either fails, retain Basic/A0 for reporting and stop
     A6-derived promotion.
   - If A6 passes and the A6-L17 repair gate fails, freeze A6.
   - If both pass, freeze A6-L17.
   - A7 and L17 remain hard controls; they cannot be promoted independently.
   - The independent post-run adjudicator writes `post_a6_decision.json`.
     Shot scaling is authorized only when the selected detector beats both
     same-run SuperAD and RN-FMLK on all three macro metrics, wins at least
     five of eight objects per metric, and has no object collapse.
   - A successful `pull` means evidence retrieval only. Before creating or
     launching a shot run, require the controller's
     `authorize-shot-scaling` mode to pass against the local authorization
     marker.

3. **Output-chain decision**
   - The historical Basic P1 guided-map row is not globally safe because
     `can` crosses the object-collapse boundary.
   - Do not spend a fresh Basic run solely on support-fit P2: thresholding and
     morphology cannot repair the failed continuous-map ranking gate.
   - A new P1/P2 run is allowed only for the newly frozen detector and must
     preregister support-only threshold fitting before execution.

4. **Scorer and PPD decision**
   - Do not rerun the broad Phase-3 scorer family. Its completed DINOv3-H+
     suite left raw 1-NN best on F1 (`0.530635`), and every alternative failed
     its keep gate.
   - Do not reopen density sweeps or DVT-dependent positional scoring before
     A6 confirmation; these branches are already negative or mechanism-
     inconsistent.
   - After A6 confirmation, allow at most one detector-matched scorer/null
     confirmation if the frozen contrast leaves a specific unresolved causal
     question. No broad search is authorized.

5. **Shot curve**
   - The 2026-07-14 nested-shot preregistration is superseded before execution
     because it freezes DVT, density, guidance, and morphology settings that
     are no longer the candidate contract.
   - Freeze one replacement protocol after step 2: shots `1/2/4/8/16`, seeds
     `0..4`, identical nested references per seed, and no per-shot tuning.
   - Start with random-16. Expand to the full curve only if the frozen
     16-shot comparison gate passes.

6. **Backbone scaling**
   - Transfer the exact frozen detector first to Ours+ and then Ours++.
   - Run fixed-16 and random-16 before any full shot expansion.
   - Preserve the original continuation thresholds and do not reuse
     incompatible historical DINOv3 rows as same-condition evidence.

7. **Final evidence package**
   - Produce the fair-comparison, shot-curve, scaling, output-chain, and
     mechanism tables only from contract-valid rows.
   - Retain compact manifests, hashes, logs, metrics, bootstrap summaries,
     and cleanup evidence; label all `test_public` results public-shadow.

## Current blocker

The local execution image has no `ssh` command and no usable Python
interpreter. The latest fail-closed v5 controller attempt stopped
at its local `ssh is required` guard before snapshot creation, upload, or
remote mutation. The fixed dsba3 container remains the required execution
environment; no generic-container fallback is authorized.
