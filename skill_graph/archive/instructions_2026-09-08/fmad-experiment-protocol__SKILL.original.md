---
name: fmad-experiment-protocol
description: Stage-gated FMAD-OOD research experiment workflow with SuperAD-centered baseline and unified metric comparison. Use when the user asks to run, design, evaluate, implement, or analyze an FMAD experiment or method protocol; when they mention 실험 프로토콜, protocol대로, 설계평가, 구현평가, keep/kill gate, SuperAD comparison, metric alignment, remote GPU execution, benchmark/smoke run, or result analysis in the FMAD-OOD repository.
---

# FMAD Experiment Protocol

Use this skill to execute FMAD-OOD experiment or method work through the repository's required stage gates. Do not jump from idea to code or from code to benchmark numbers.

When the user also invokes `$deep-research-autoresearch` or asks to combine discovery with experiment execution, use `$fmad-deep-experiment-protocol` as the wrapper. This skill remains the authority for FMAD gates, SuperAD comparison, remote execution, and experiment verdicts.

## Current Target Scope Guard

For this FMAD-OOD project, the default target is **single-image MVTec AD2** on `/home/hunim/Volume/DATA/mvtec_ad_2`, compared against same-condition SuperAD with segmentation metrics. Treat GPU IDs as execution resources only; GPU availability or a request to "continue" does not imply a dataset, task, or method-family switch.

Unless the user explicitly asks for M2AD, multi-view, controlled observation, view/illumination triplets, SAGE-Lite, SCMV, MAVERICK, KITE, THOR, or another named non-MVTec branch, do not route to `/home/hunim/Volume/DATA/M2AD_hf` or any multi-view/controlled-observation experiment. Internal documents may be used as failure-boundary context, but they must not override the current user goal.

Before any remote run or method promotion, state and verify:

- `target_dataset=MVTec AD2 single-image`
- `data_root=/home/hunim/Volume/DATA/mvtec_ad_2`
- object/category set and split population
- method family and why it is not a forbidden closed-branch retread
- SuperAD baseline source or rerun plan
- primary metrics: `seg_AUROC_0.05`, `seg_F1`, and map/claim coverage where relevant

If a proposed next step would switch the task to M2AD/multi-view/controlled-observation, stop and realign unless the user explicitly requested that switch.

## SuperAD Baseline And Metric Contract

SuperAD is the primary quantitative baseline for FMAD-OOD method claims. Internal nulls, fixed policies, random policies, oracle variants, and ablations remain necessary controls, but they are secondary to the same-condition SuperAD comparison.

- Define the SuperAD baseline source before metric claims: existing artifact path, rerun plan, or `BLOCKED_BASELINE`.
- Use only same-condition SuperAD numbers for gate decisions: same dataset, category/object set, split manifest, train/test protocol, data root, backbone/preprocessing, and metric granularity.
- Historical or cross-dataset SuperAD values are context only. They cannot support superiority claims or `KEEP` verdicts.
- Declare one primary metric in the design, with identical directionality and aggregation for SuperAD and the candidate method. Higher must mean better unless the report explicitly converts the metric.
- Do not compare image AUROC to pixel/segmentation AUROC, PRO, or F1. If the method is image-only, produce image-level SuperAD on the same samples or mark the result diagnostic-only.
- For MVTec AD2, reuse recorded SuperAD results only when the exact split, preprocessing, backbone, category set, and metric match. Otherwise rerun SuperAD.
- For M2AD, VisA, or other benchmarks, produce a same-condition SuperAD branch before a method verdict.

## Post-50 Negative Evidence Contract

The current FMAD-OOD evidence state is: **50+ experiments, 0 deployable positive methods**. Treat apparent gains as diagnostic until they survive the full gate below. The accumulated result is a failure-boundary map, not permission to tune killed branches.

Important distinction: the full gate below is a **method-claim promotion bar**, not an automatic early-stage stopping bar. Early research should not be cut off merely because a compact/preflight result is not yet paper-level. If a result has a plausible upward gradient, keep it in a diagnostic continuation lane while clearly denying deployable or paper-level claims.

Before design or implementation, run a negative-evidence intake:

- state whether the candidate is a retread of a closed branch;
- identify the likely failure basin: easy proxy, near-identity no-op, anomaly absorption/collapse, RN-FMLK/kNN duplicate, hard-null absorbed, active-policy collapse, or oracle-only;
- record the same-run SuperAD-16/RN-FMLK/hard-null evidence that motivates the run;
- stop with `DESIGN_KILL_CLOSED_BRANCH` if the candidate is only a threshold, radius, score-fusion, descriptor, prompt, passive feature surgery, oracle-paired, or support-admission tweak of a killed family.

Minimum promotion bar for any method claim:

- same-run SuperAD-16 dominance where the benchmark is MVTec AD2;
- RN-FMLK/full-memory DINO kNN dominance as a mandatory hard baseline/null;
- hard-null envelope dominance for the proposed mechanism;
- weak + strong + control category no-harm, not just weak-object rescue;
- `seg_AUROC_0.05`, `seg_F1`, and map/claim coverage preserved together for segmentation claims;
- compact-vs-full boundary explicitly labeled. Compact positives are `PROMISING_DIAGNOSTIC` only, not paper-level or deployable positives.

Allowed starting families are limited to: deployable support/evidence identification, real active-observation utility, normal-only adaptation with anti-absorption, or a genuinely non-kNN evidence source. Everything else needs an explicit design-stage justification before code.

### Early-Stage Continuation Lane

Use this lane for Stage -1, Stage -1a0, compact, reduced-object, mechanism-preflight, and other cheap exploratory runs.

Do **not** require a candidate to satisfy every SuperAD/RN-FMLK/hard-null/F1/coverage gate before one more scoped diagnostic. Instead classify the result into one of three tiers:

- **Claim tier:** `KEEP` or `CONDITIONAL KEEP`. Requires the full promotion bar.
- **Continuation tier:** `CONTINUE_DIAGNOSTIC` or `PROMISING_DIAGNOSTIC`. Allows the next small experiment, revision, ablation, or scale-up because there is a plausible improvement gradient. It is not a method success.
- **Hard-stop tier:** `KILL`, `BLOCKED_BASELINE`, `BLOCKED_DATA`, or `DESIGN_KILL_CLOSED_BRANCH`. Stops this exact branch because the failure is fatal or uninformative.

A result may enter `CONTINUE_DIAGNOSTIC` when most of these are true:

- at least one primary or secondary metric moves in the intended direction against the immediate baseline, fixed policy, or previous branch;
- the improvement is not explained entirely by an implementation bug, data leakage, no-peek violation, metric mismatch, or label/GT access;
- at least one weak category, object subset, ablation, or mechanism diagnostic shows nontrivial positive signal;
- strong/control categories are not catastrophically broken, or the breakage is bounded enough to test a targeted repair;
- hard-null dominance is partial or marginal rather than complete absorption across all cases;
- F1/coverage is not zero everywhere when the claim requires a map;
- the next step is small, pre-registered, and directly tests whether the signal can survive the specific failed gate.

Use hard-stop `KILL` early only when one of these is true:

- data leakage, no-peek violation, invalid split, wrong metric, or code-path bug invalidates the result;
- the method is a forbidden closed-branch retune with no new mechanism;
- all relevant outputs collapse to near-constant, zero coverage, zero admission, or no evaluated samples;
- every meaningful branch is matched or beaten by hard nulls with no surviving positive subset or gradient;
- strong/control categories catastrophically collapse with no targeted repair hypothesis;
- the result is exactly RN-FMLK/kNN, oracle-only, or easy-proxy separability repackaged as novelty.

For `CONTINUE_DIAGNOSTIC`, record the next diagnostic as a **single-step continuation contract**: what failed, what small change may plausibly improve it, what metric/gate must improve, and what condition will hard-stop the branch. Do not use continuation tier as permission for broad sweeps or post-hoc tuning.

Recommended language:

- `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`: fails paper-level claim now, but one constrained follow-up is justified.
- `KILL_FOR_CLAIM / NO_CONTINUE`: fails paper-level claim and has no plausible continuation signal.
- `PROMISING_DIAGNOSTIC`: compact or mechanism-positive, not comparable enough for claim.

### MVTec AD2 SuperAD Paper-Aligned Contract

For claims against the SuperAD paper/Table 1 protocol on MVTec AD2, SuperAD is **not full-shot**. The paper-aligned baseline is fixed to 16 normal reference images per category.

- Reference construction: use all category `train/good` images only as the selection pool; extract DINOv2 ViT-L/14 CLS token representations from that full pool; then select exactly 16 reference images with PatchCore-style greedy coreset selection.
- Candidate reference budget: any candidate method that uses normal train/reference images must use the same 16 selected reference image paths as the same-run SuperAD branch, unless the run is explicitly labeled as a different-budget ablation. Different-budget ablations cannot support a `KEEP`/`CONDITIONAL KEEP` claim against the paper-aligned SuperAD baseline.
- Full-shot handling: `shots <= 0`, all-train memory banks, fixed64, or larger memory banks are ablations/diagnostics only. Do not call them the SuperAD paper baseline.
- Dataset population: Table 1-style comparison requires all eight public objects (`can`, `fabric`, `fruit_jelly`, `rice`, `vial`, `wallplugs`, `walnuts`, `sheet_metal`) and full `test_public/good,bad`. If an object such as `walnuts` is missing or empty, the all-object Table 1 comparison is `BLOCKED_DATA`; a reduced-object run is only an internally comparable diagnostic.
- Backbone and feature contract: use DINOv2 ViT-L/14, multi-level layers 6/12/18/24 (0-indexed `[5, 11, 17, 23]` in code), shorter-side resolution 672 for all categories except `sheet_metal` at 448, and the same preprocessing/masking path for SuperAD and candidate where applicable.
- Foreground/background masking: for `vial` and `wallplugs`, preserve the paper's PCA-based foreground mask settings when using the SuperAD path: one PCA component, threshold `1.0`, variance-based inversion, and 3x3 morphology.
- Fabric/walnuts binary postprocessing: the paper describes closed-region filling for `fabric` and `walnuts` after initial detection. Treat this as a binary-submission/post-threshold step. For continuous TESTpublic evaluator comparisons, either do not apply extra category-specific binary postprocessing, or apply and document an identical postprocess policy for SuperAD and candidate.
- Metric contract: Table 1 uses pixel-level `AUROC_0.05` and best-threshold segmentation F1 on binarized/thresholded TESTpublic predictions, reported in percent. In this repository, `src.post_eval.eval_segmentation` returns `seg_AUROC` as pixel AUROC with `max_fpr=0.05` and `seg_F1` as best-threshold pixel F1. Do not compare compact-subset values, image AUROC, full-FPR pixel AUROC, or unrelated PRO to the Table 1 baseline.

## Required Order

Follow these stages unless the user explicitly narrows or changes the workflow:

1. **Motivation**
   - State the concrete problem and why this experiment is worth running now.
   - Tie the proposal to a specific failure pattern, open question, or user-selected direction.
   - Express the research claim in terms of Claim, Evidence, Boundary, and Positioning when the work is method-oriented.
   - Include the negative-evidence intake: why the proposal is not a repeat of a closed branch and which known failure basin it tries to break.

2. **Implementable Design**
   - Reduce the idea to the smallest runnable diagnostic, smoke, or method variant.
   - Define method boundary, config switches, target categories, SuperAD baseline source, controls, and pre-registered keep/kill gates.
   - Declare both strict claim gates and early continuation gates. The continuation gate should say what weak positive signal is enough to justify exactly one next diagnostic, even if the strict claim gate fails.
   - Declare the unified primary metric and output schema before implementation: `superad_<metric>`, `method_<metric>`, `delta_vs_superad`, `comparable`, and optional confidence interval fields.
   - For segmentation claims, pre-register `seg_AUROC_0.05`, `seg_F1`, positive map coverage or hard-null-subtracted claim coverage, RN-FMLK margin, and hard-null margin.
   - For active-observation claims, pre-register fixed/random/nearest-response/sequence-replay/light-label nulls, sequence diversity, top-sequence fraction, and acquisition-cost evidence.
   - For adaptation/support claims, pre-register anti-absorption, retention/no-harm, source/random/score/admission-shuffle nulls, and admission/coverage diagnostics.
   - For MVTec AD2 paper-aligned SuperAD comparisons, declare `reference_selection_pool=full train/good`, `expected_superad_shots=16`, `candidate_reference_policy=same_superad_reference_set`, full `TESTpublic`, and whether fabric/walnuts binary postprocessing is in or out of scope.
   - If a same-condition SuperAD result is missing, include a SuperAD baseline run in the plan or stop with `BLOCKED_BASELINE`; do not substitute historical values as the gate comparator.
   - Prefer one claim per test. Split exploratory diagnostics from benchmark claims.

3. **Evaluation Alignment**
   - Check explicitly whether the design tests the motivating claim.
   - Check explicitly whether SuperAD and the candidate method are comparable on dataset, split, category/object set, metric granularity, and aggregation.
   - For MVTec AD2 paper-aligned SuperAD comparisons, check that SuperAD and the candidate share the exact 16 selected reference images, full `test_public/good,bad`, DINOv2/resolution/preprocessing contract, and `AUROC_0.05`/F1 evaluator.
   - Reject direct comparison when dataset or metric differs, for example MVTec AD2 pixel AUROC vs M2AD Motor image AUROC.
   - Look for precedent conflicts in `tasks/lessons.md` and recent `skill_graph/` records.
   - Check baseline-strength conservation when category count permits: if gains concentrate on weak categories while strong/control categories collapse, treat the result as a no-harm failure even if the mean improves.
   - Check whether the proposed evidence source is anomaly-aligned rather than merely independent, synthetic-proxy separable, or low-score/nearest-support selected.
   - If the design has structural doubt that cannot be fixed by spec edits, stop at design-stage kill and document why.
   - Avoid cutting early-stage experiments solely because they are not full-claim comparable. If the design can produce bounded diagnostic evidence, keep the result label diagnostic and specify the next continuation gate.

4. **Code Modification / Creation**
   - Implement only the code needed for the current stage.
   - Keep changes config-driven where applicable with explicit `enable: true/false` or equivalent controls.
   - Record exact files created or modified.

5. **Added Code Evaluation**
   - Verify the implementation matches the design boundary before any benchmark run.
   - Run the strongest practical local checks: `py_compile`, focused tests, import/registry sanity, config sanity, CLI help, or dry-run diagnostics.
   - Verify that result writers emit the unified SuperAD/method metric fields and preserve the SuperAD artifact path or rerun command.
   - For SuperAD-16 comparisons, verify that split manifests or metrics include the full train selection-pool count, `selected_reference_count=16`, selected reference basenames/paths, candidate reference count, and `reference_budget_matched=true`.
   - Treat code-complete as insufficient until this evidence exists.

6. **Remote Experiment Execution**
   - Run benchmark or smoke metrics only through the established remote workflow when metric claims are needed.
   - Use the current project remote policy from `tasks/lessons.md` and `configs/remote/`. The current default is canonical target `147.47.39.144`, fixed container `hun_superad_run`; host GPUs `0,1,2,3` are allowed, but choose the actually idle subset after remote `nvidia-smi` and record host GPU IDs separately from in-container CUDA slots.
   - If the existing fixed container was started with stale GPU exposure that cannot run the selected idle subset, recreate only that fixed project container before running the next experiment. Do not recreate it while a project experiment is actively running.
   - Run or pull the same-condition SuperAD baseline before or alongside the candidate method whenever the baseline artifact is absent.
   - Use the same data root, split manifest, category/object list, preprocessing, backbone where applicable, and metric writer for SuperAD and the candidate method.
   - For MVTec AD2 paper-aligned SuperAD comparisons, run a strict remote preflight before GPU benchmark execution: every configured Table 1 object must have non-empty `train/good`, `test_public/good`, and `test_public/bad`; full train/test caps must be disabled; SuperAD effective shots must be 16; and candidate/SuperAD reference budget must match.
   - Pull results from `/workspace/results_remote/...` to local `results/remote_runs/...`.
   - Pull both candidate-method and SuperAD artifacts; record exact remote and local paths.
   - Use local execution only for syntax, unit, registry, or non-metric sanity checks.

7. **Evaluation Results and Analysis**
   - Compare results against the same-condition SuperAD baseline first, then report nulls, fixed/random policies, ablations, and oracle diagnostics.
   - Report unified metrics: `superad_<metric>`, `method_<metric>`, `delta_vs_superad`, `comparable=true/false`, winner, and confidence/bootstrap evidence when available.
   - Report RN-FMLK/full-memory DINO kNN as mandatory baseline/null for MVTec AD2-style passive feature or segmentation claims.
   - Report `seg_AUROC_0.05`, `seg_F1`, and map/claim coverage together. Do not let AUROC movement hide F1 or coverage collapse.
   - Report weak-object pass count, strong-category no-harm, control no-harm, hard-null margin, and baseline-strength-vs-delta correlation when enough categories exist.
   - For Table 1-style reports, state whether numbers are decimals or percent and include both `AUROC_0.05` and segmentation F1. If a reduced object set was used, label it reduced-object diagnostic and do not compare its mean to the eight-object paper mean.
   - Report gains, losses, win rate, strong-category no-harm, category asymmetry, or diagnostic values as applicable, but do not let secondary null wins replace the SuperAD gate.
   - Explain whether the result supports the claim or exposes a different failure mode.
   - Separately answer two questions:
     1. Does this support a strict method claim now?
     2. Does this justify a small next diagnostic because there is a plausible improvement gradient?
   - When the answer is no to the first but yes to the second, use `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC` or `PROMISING_DIAGNOSTIC`, not plain `KILL`.

8. **Conclusion**
   - End with a verdict: `KEEP`, `CONDITIONAL KEEP`, `PROMISING_DIAGNOSTIC`, `CONTINUE_DIAGNOSTIC`, `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`, `KILL_FOR_CLAIM / NO_CONTINUE`, `KILL`, `BLOCKED_BASELINE`, `BLOCKED_DATA`, or `DESIGN_KILL_CLOSED_BRANCH`.
   - Use `KEEP` only when same-condition SuperAD, RN-FMLK, hard-null, weak/strong/control no-harm, and AUROC/F1/coverage gates all pass.
   - Use `CONDITIONAL KEEP` only when SuperAD is comparable and the result is positive but limited by scope or variance while still passing RN-FMLK, hard-null, no-harm, and coverage gates.
   - Use `PROMISING_DIAGNOSTIC` for compact, reduced-object, or mechanism-preflight positives that are not full comparable. Do not present this as a deployable positive.
   - Use `CONTINUE_DIAGNOSTIC` when the result is not claim-worthy but contains a plausible, bounded positive signal that should be tested once more.
   - Use `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC` when strict gates fail but the branch should not be closed yet.
   - Use `KILL_FOR_CLAIM / NO_CONTINUE` when strict gates fail and the result gives no credible next diagnostic.
   - Use `BLOCKED_BASELINE` when the same-condition SuperAD baseline is missing or metric comparability fails; do not convert this into a positive verdict.
   - Use `DESIGN_KILL_CLOSED_BRANCH` before code when a candidate re-enters a closed branch without a new anomaly-aligned mechanism.
   - Use `BLOCKED_DATA` or `BLOCKED_BASELINE` for MVTec AD2 Table 1-style runs when any required object is missing, the SuperAD-16 reference budget is not met, the candidate uses a different reference set without ablation labeling, or the evaluator is not `AUROC_0.05`/best-threshold F1.
   - State what asset remains even for negative results: code utility, diagnostic evidence, failure pattern, or next constrained direction.
   - If continuation is allowed, state the single next experiment, its budget/scope, the expected weak signal, and the hard-stop condition.

9. **Post-Conclusion Storage Cleanup**
   - After the result analysis and binary conclusion are recorded, delete regenerable dense output trees named `anomaly_maps/` from the completed remote result root and the local pullback root.
   - Preserve compact evidence artifacts: summary TSV/JSON, metrics, configs, run logs, diagnostics, reports, and any explicitly selected figure assets.
   - If full `anomaly_maps/` must be kept for a specific follow-up or figure audit, document the path, reason, and expected cleanup point in `tasks/todo.md` and the experiment report.
   - Verify cleanup by checking that no `anomaly_maps/` directory remains under the completed experiment result roots, and record the evidence.

## Mandatory Artifacts

For substantial experiment cycles, update or create:

- `tasks/todo.md`: current work, plan checklist, progress, result.
- `skill_graph/ideas/` or `skill_graph/analysis/`: protocol-aligned design or synthesis note when the work creates a new direction.
- `skill_graph/experiments/YYYY-MM-DD_experiment_name/report.md`: experiment report with the nine protocol stages, SuperAD baseline source, unified metric schema, and comparability judgment.
- `skill_graph/log.md` and `skill_graph/index.md`: lifecycle entry and discoverability link.
- SuperAD baseline artifact or rerun command, including remote/local paths, exact split/metric metadata, and for SuperAD-16 runs the selected reference list plus candidate reference-match evidence.
- Post-conclusion storage cleanup evidence for `anomaly_maps/`.

When the task is a quick diagnostic or explicitly scoped sub-step, keep artifacts proportionate but still record the gate, result path, and verdict.

## Enforcement Rules

- Do not run benchmark experiments locally when remote protocol is required.
- Do not report success without showing the result against the pre-registered gate.
- Do not declare `KEEP` or `CONDITIONAL KEEP` unless SuperAD and the candidate method share dataset, split, category/object set, and primary metric.
- For MVTec AD2 paper-aligned claims, do not declare `KEEP` or `CONDITIONAL KEEP` unless SuperAD and the candidate also share the same 16 SuperAD-selected reference images and full `TESTpublic` object population.
- Do not compare image-level metrics to pixel/segmentation metrics, or cross-dataset SuperAD values, as a gate.
- Do not treat full-shot/all-train SuperAD, fixed64 SuperAD, compact sampled SuperAD, or reduced-object SuperAD as the SuperAD paper/Table 1 baseline.
- If the same-condition SuperAD baseline is missing, run it first or return `BLOCKED_BASELINE` before making a method metric claim.
- Do not scale a single-category positive directly to a general claim; add weak and strong category checks.
- Do not treat compact sampled positives, sampled SuperAD wins, or weak-baseline wins as deployable evidence.
- Do not discard compact sampled positives, weak-baseline wins, or partial null margins automatically; classify them as diagnostic and decide whether a bounded continuation is justified.
- Do not promote a method that loses to RN-FMLK/full-memory DINO kNN unless the method's target was explicitly different and pre-registered.
- Do not claim mechanism credit when a random/source/coordinate/score/light-label/nearest-response null matches or beats the method.
- Do not ignore F1 or map/claim coverage collapse when AUROC moves.
- Do not tune killed support-admission, passive feature-surgery, prompt-prior, descriptor-swap, SuperAD-fusion, or oracle-paired branches without a new admissibility audit.
- Do not revive user-forbidden directions unless explicitly reauthorized.
- Do not switch from the current MVTec AD2 single-image SuperAD-aligned objective to M2AD, multi-view, controlled-observation, or SAGE-Lite-style branches unless explicitly reauthorized.
- Treat internal documents as insight sources, not authority over the current user instruction.
- Prefer design-stage kill over GPU runs when literature, lessons, or adversarial review reveal structural misalignment.
- Prefer `CONTINUE_DIAGNOSTIC` over early hard `KILL` when there is a measured weak positive signal and a clear one-step follow-up that can test it cheaply.
- Do not leave full `anomaly_maps/` trees after an experiment has a recorded result and conclusion, unless a documented exception explicitly preserves them.

## Minimal Report Skeleton

Use this shape for experiment reports:

```markdown
# <Experiment Name>

Date: <YYYY-MM-DD>
Verdict: KEEP | CONDITIONAL KEEP | PROMISING_DIAGNOSTIC | CONTINUE_DIAGNOSTIC | KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC | KILL_FOR_CLAIM / NO_CONTINUE | KILL | BLOCKED_BASELINE | BLOCKED_DATA | DESIGN_KILL_CLOSED_BRANCH

## Negative Evidence Intake
## Motivation
## Implementable Design
## Evaluation Alignment
## Code Modification / Creation
## Added Code Evaluation
## Remote Execution
## SuperAD Baseline and Unified Metrics
## Results and Analysis
## Continuation Assessment
## Conclusion
## Post-Conclusion Storage Cleanup
```
