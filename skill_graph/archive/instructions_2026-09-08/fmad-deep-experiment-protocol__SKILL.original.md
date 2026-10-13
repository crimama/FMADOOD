---
name: fmad-deep-experiment-protocol
description: Repeatable FMAD-OOD research-to-experiment workflow that runs deep-research-autoresearch novelty discovery, promotes the best candidate into fmad-experiment-protocol SuperAD-centered experiment gates, then feeds the result back into the next research cycle. Use when the user asks to combine or repeatedly loop FMAD experiment protocol with deep research/autoresearch, or needs literature-grounded method discovery followed by implementation, remote execution, and SuperAD-aligned quantitative comparison in the FMAD-OOD repository.
---

# FMAD Deep Experiment Protocol

Use this skill as a wrapper, not a replacement, for:

- `/home/hun/.codex/skills/deep-research-autoresearch/SKILL.md`
- `/home/hun/.codex/skills/fmad-experiment-protocol/SKILL.md`

The combined workflow turns research discovery into a gated FMAD experiment, then uses the experiment verdict as evidence for the next cycle. `deep-research-autoresearch` owns novelty discovery and adversarial design iteration. `fmad-experiment-protocol` owns implementation readiness, remote execution, SuperAD-centered metric comparison, verdicts, and storage cleanup.

## Core Rule

Do not let a design-stage research score become an experiment verdict. A deep-research candidate can only be promoted into a strict claim verdict (`KEEP`, `CONDITIONAL KEEP`, `KILL_FOR_CLAIM`, `KILL`, or `BLOCKED_BASELINE`) or continuation verdict (`PROMISING_DIAGNOSTIC`, `CONTINUE_DIAGNOSTIC`) through the FMAD experiment protocol and the same-condition SuperAD metric contract.

Separate method-claim success from research continuation. The strict SuperAD/RN-FMLK/hard-null/F1/coverage bar decides whether a result is claim-worthy; it must not automatically stop early exploration when the result contains a plausible weak improvement signal. A failed strict claim may still enter `CONTINUE_DIAGNOSTIC` for one small, pre-registered follow-up.

For the current FMAD-OOD project, the default outer-loop target is **single-image MVTec AD2** on `/home/hunim/Volume/DATA/mvtec_ad_2`, not M2AD or multi-view/controlled observation. Deep research must not promote M2AD, SAGE-Lite, SCMV, MAVERICK, KITE, THOR, view-triplet, illumination-triplet, or controlled-observation candidates unless the user explicitly requests that branch. GPU availability is only a scheduling detail and must not be treated as evidence for changing the research task.

When the user asks to "keep going", "repeat", "continue the process", "loop", or gives a long-running `/goal`, run this as a bounded outer loop of complete research-to-experiment cycles. Never run an unbounded loop silently; set an explicit cycle cap, stop condition, or next audit point from the user request or the defaults below.

Post-50 evidence rule: the current FMAD-OOD state is **50+ experiments, 0 deployable positives**. The loop must start from failure-boundary evidence, not from optimism about small compact gains. A candidate cannot be promoted from research to implementation unless it explicitly addresses why prior branches failed by easy proxy, near-identity, collapse/absorption, RN-FMLK/kNN duplication, hard-null absorption, active-policy collapse, or oracle-only deployability failure.

## Continuous Outer Loop Mode

Use this mode when the goal is not one method but an iterative research engine.

Default outer-loop budget:

- `cycle_cap`: `3` complete research-to-experiment cycles unless the user gives another number.
- `research_stage_cap_per_cycle`: use the `deep-research-autoresearch` stage budget parser; default `5`.
- `remote_metric_runs_per_cycle`: at most one promoted candidate smoke/benchmark plus its same-condition SuperAD baseline unless the user authorizes broader sweeps.

Outer-loop stop conditions:

- Stop with `SUCCESS` when a candidate reaches FMAD `KEEP`.
- Stop with `PROMISING_STOP` when a candidate reaches `CONDITIONAL KEEP` and the next step is a materially larger benchmark, paper table, or user-level scope decision.
- Continue with `CONTINUE_DIAGNOSTIC` when strict claim gates fail but the FMAD report identifies a bounded positive signal and a single next diagnostic. Count this as a cycle but do not treat it as success.
- Stop with `BASELINE_BLOCKED` when the only blocker is missing or non-comparable SuperAD; run/repair that baseline before more method ideation.
- Stop with `NO_PROMOTABLE_CANDIDATE` when a full deep-research cycle produces no implementable, non-duplicate candidate and no bounded diagnostic continuation signal.
- Stop with `DESIGN_KILL_CLOSED_BRANCH` when the best candidate is a retread of a closed family: passive single-image feature surgery, SuperAD score/fusion/reference/radius tweak, RN-FMLK/kNN novelty, prompt prior, shallow descriptor swap, oracle paired/test-good/GT-region dependence, easy synthetic proxy, or killed support-admission/adaptation branch without a new mechanism.
- Stop with `CYCLE_CAP_REACHED` when the outer-loop budget is consumed.
- Stop immediately for destructive cleanup, forbidden direction conflict, missing credentials, or remote resource blocker that cannot be safely recovered.

Each cycle has this shape:

1. **Cycle Intake**
   - Read the previous cycle verdict, failure mode, surviving evidence, and forbidden/rejected directions.
   - Update the joint evidence ledger before proposing new methods.
   - Convert the last experiment into a compact lesson: what failed, why it failed, what mechanism is ruled out, and what constraint remains open.
   - Add a negative-evidence intake row: deployable positives remain zero, which closed branch is implicated, which oracle upper bound remains, and what exact minimum bar the next candidate must pass.

2. **Novelty Search**
   - Run the bounded `deep-research-autoresearch` loop for the cycle.
   - Penalize candidates that merely tune the last failure, repackage SuperAD, reuse a forbidden direction, or lack a same-condition SuperAD comparison path.
   - Prefer a new mechanism that directly addresses the previous cycle's strongest failure mode.
   - Require each stage to name one open family it belongs to: deployable support/evidence identification, real active observation utility, anti-absorption normal-only adaptation, or genuinely non-kNN evidence source.
   - Reject methods that only solve a proxy task or only create independence from SuperAD without an anomaly-alignment argument.

3. **Promotion Audit**
   - Promote at most one candidate into implementation.
   - Require concrete method boundary, target benchmark, implementation slice, SuperAD baseline source, unified primary metric, null/control matrix, and keep/kill gate.
   - Require same-run SuperAD-16, RN-FMLK/full-memory DINO kNN, hard-null envelope, weak+strong+control no-harm, and AUROC/F1/coverage gates in the FMAD plan when the claim is a segmentation method claim.
   - Treat compact positives as `PROMISING_DIAGNOSTIC` only. They do not satisfy the outer-loop success condition.
   - Also pre-register a soft continuation gate: what small gain, partial null margin, nonzero coverage, or mechanism diagnostic is enough to justify one next run if the strict claim fails.
   - If no candidate passes, finalize the cycle as design-stage only and count it against the outer-loop budget.

4. **FMAD Experiment Gate**
   - Execute the nine-stage FMAD protocol for the promoted candidate.
   - If implementation is needed, keep it minimal and config-driven.
   - If metric claims are needed, run remote SuperAD-aligned evaluation.

5. **Cycle Postmortem**
   - Record `verdict`, `superad_delta`, `winning_or_failing_gate`, `root_failure_mode`, `next_hypothesis_seed`, and `do_not_repeat`.
   - Record `claim_verdict` separately from `continuation_verdict`. A method can be `KILL_FOR_CLAIM` while still `CONTINUE_DIAGNOSTIC`.
   - Update `tasks/lessons.md` when the cycle reveals a recurring mistake or user correction.
   - Feed the postmortem into the next cycle's evidence ledger.

6. **Continue Or Stop Audit**
   - Check stop conditions before launching the next cycle.
   - Continue only if a remaining open mechanism exists, the cycle budget remains, and the next cycle does not require an unapproved broader benchmark or destructive action.

Maintain a loop ledger at `.omx/specs/<outer-loop-slug>/outer-loop.md` with one row per cycle:

```markdown
| cycle | research_artifact | promoted_method | superad_source | primary_metric | fmad_report | verdict | root_failure_mode | next_seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
```

## Required Order

1. **Load Both Source Skills**
   - Read both source `SKILL.md` files when this wrapper is invoked.
   - Use their detailed instructions for artifacts, scripts, evaluator requirements, remote execution, and SuperAD comparison.
   - If either source skill changes, its source text overrides duplicated assumptions in this wrapper.

2. **Parse Research Budget**
   - Use the `deep-research-autoresearch` stage budget parser.
   - Default to `5` stages unless the user supplies a number such as `10 stages`, `stage=8`, or `10단계`.
   - Treat the budget as a hard cap for the novelty loop, not for implementation or experiment runtime.

3. **Build A Joint Evidence Ledger**
   - Initialize or update the deep-research loop artifact under `.omx/specs/<loop-dir>/`.
   - Record prior failures, partial positives, forbidden directions, resource limits, external references, and deployment assumptions.
   - Record the post-50 failure constraints: no deployable positive yet; RN-FMLK mandatory; compact sampled positives are not paper-level evidence; weak-category rescue must not break strong/control categories; F1/coverage collapse is a kill condition.
   - Record the early-stage continuation constraints: compact/preflight gains can justify one more diagnostic when they show a plausible upward gradient, but must be labeled non-claim evidence.
   - Add FMAD-specific fields to the ledger: target dataset, category/object set, split source, SuperAD baseline availability, primary metric, metric granularity, remote target/GPU policy, and known blocker status.
   - For the current default branch, set `target_dataset=MVTec AD2 single-image` and `data_root=/home/hunim/Volume/DATA/mvtec_ad_2`; any M2AD/multi-view/controlled-observation switch requires explicit user authorization and a new alignment statement.

4. **Run The Deep Research Autoresearch Loop**
   - For each stage, perform evidence update, external research or no-search justification, one candidate design or revision, separate adversarial evaluator review, and stage recording.
   - Each candidate method must include `Claim`, `Evidence`, `Boundary`, and `Positioning`.
   - Each evaluator output should also attack FMAD readiness: SuperAD comparability, metric leakage, split leakage, deployability, hard nulls, and whether the method is only a SuperAD add-on or score trick.

5. **Promotion Gate Into FMAD Protocol**
   - Promote only the best surviving design, not every stage idea.
   - Promotion requires a concrete method boundary, implementation scope, target benchmark, same-condition SuperAD baseline source or `BLOCKED_BASELINE`, unified primary metric, controls/nulls, and pre-registered keep/kill criteria.
   - Promotion also requires a closed-branch risk audit, anomaly-alignment argument, identity-nontriviality argument, retention/no-harm plan, null envelope, and map/claim coverage plan.
   - Promotion should include both `strict_claim_gate` and `soft_continuation_gate`. The soft gate exists to prevent premature cutoff of weak but plausible signals; it must not be reused as a success claim.
   - If the best candidate is still vague, redundant, forbidden, or lacks a comparable SuperAD plan, finalize the deep-research artifact as design-stage only and stop before code or GPU work.

6. **Convert The Candidate To FMAD Stages**
   - Rewrite the promoted design into the FMAD protocol order:
     1. Motivation
     2. Implementable Design
     3. Evaluation Alignment
     4. Code Modification / Creation
     5. Added Code Evaluation
     6. Remote Experiment Execution
     7. Evaluation Results and Analysis
     8. Conclusion
     9. Post-Conclusion Storage Cleanup
   - Keep the SuperAD baseline and unified metric schema explicit before implementation.

7. **Implement And Verify Before Remote Metrics**
   - Implement only the smallest runnable diagnostic, smoke, or method variant.
   - Keep new behavior config-driven where applicable.
   - Run local code checks required by `fmad-experiment-protocol`: compile, focused tests, import/registry sanity, config sanity, CLI help, dry-run diagnostics, or equivalent.
   - Verify result writers emit `superad_<metric>`, `method_<metric>`, `delta_vs_superad`, `comparable`, and path metadata where metric claims are expected.

8. **Run SuperAD-Aligned Remote Experiments**
   - Use the current FMAD remote policy from `tasks/lessons.md`, `configs/remote/`, and `fmad-experiment-protocol`.
   - Run or pull the same-condition SuperAD baseline before or alongside the candidate method.
   - Do not use historical or cross-dataset SuperAD values as a gate comparator.
   - Pull both SuperAD and candidate artifacts into `results/remote_runs/...`.

9. **Close The Loop With Results**
   - Compare against same-condition SuperAD first, then secondary nulls, controls, ablations, and oracle diagnostics.
   - Compare against RN-FMLK and the hard-null envelope before calling any compact or full result positive.
   - If AUROC improves while F1, coverage, strong-category retention, or mechanism-specific nulls fail, record the strict claim as failed, not `CONDITIONAL KEEP`.
   - Then decide whether the result is `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC` or `KILL_FOR_CLAIM / NO_CONTINUE` based on whether there is a bounded next experiment with a plausible improvement gradient.
   - Feed the final experiment verdict back into the evidence ledger or `skill_graph/` so future deep-research loops inherit the failure mode or surviving constraint.
   - If the result is `KILL`, restart another novelty loop only when Continuous Outer Loop Mode is active and the stop audit leaves cycle budget plus a non-duplicate next hypothesis seed.
   - If the result is `BLOCKED_BASELINE`, run/repair the SuperAD baseline path before making any method claim.

## Artifact Contract

For a complete combined cycle, produce or update:

- `.omx/specs/<loop-dir>/`: deep-research loop artifacts, evidence ledger, stage records, score trace, finalization record.
- `.omx/specs/<outer-loop-slug>/outer-loop.md`: cycle ledger when Continuous Outer Loop Mode is active.
- `skill_graph/ideas/` or `skill_graph/analysis/`: promoted method or design-stage synthesis.
- `skill_graph/experiments/YYYY-MM-DD_experiment_name/report.md`: FMAD nine-stage report with SuperAD baseline source and unified metrics.
- `skill_graph/log.md` and `skill_graph/index.md`: lifecycle and discoverability links.
- `tasks/todo.md`: current work, plan, progress, result.
- `results/remote_runs/...`: pulled SuperAD and candidate result artifacts when remote metrics are run.

Keep artifacts proportionate for quick diagnostics, but never omit the SuperAD comparability judgment for metric claims.

## Evaluator Addendum

When using the deep-research evaluator inside this combined workflow, require these extra labels or JSON fields in addition to the normal evaluator schema:

- `fmad_experiment_ready`: `true` or `false`
- `superad_baseline_source`: artifact path, rerun plan, or `BLOCKED_BASELINE`
- `primary_metric`: exact metric and granularity
- `metric_comparable`: `true` or `false`
- `baseline_blockers`: list of missing baseline, split, metric, or protocol blockers
- `next_fmad_gate`: design-stage kill, implementable diagnostic, SuperAD baseline run, remote smoke, or full benchmark

## Final Output

The final response must include:

- best method name and deep-research score trace
- outer-loop cycle number, stop condition, and whether another cycle is allowed
- negative-evidence/admissibility status: closed-branch risk, open-family fit, and whether compact evidence is only diagnostic
- whether the output is design-stage only or experiment-validated
- SuperAD baseline source and unified primary metric
- FMAD verdict: `KEEP`, `CONDITIONAL KEEP`, `PROMISING_DIAGNOSTIC`, `KILL`, `BLOCKED_BASELINE`, `BLOCKED_DATA`, or `DESIGN_KILL_CLOSED_BRANCH`
- continuation verdict when different from the strict claim verdict: `CONTINUE_DIAGNOSTIC`, `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`, or `KILL_FOR_CLAIM / NO_CONTINUE`
- key evidence, controls, hard nulls, and remaining risks
- artifact paths and verification evidence
- anomaly map cleanup evidence when a completed experiment produced or could have produced dense maps
