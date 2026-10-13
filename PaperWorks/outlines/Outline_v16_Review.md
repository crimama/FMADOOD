# Outline v16 — Independent skeptical CVPR review

Review date: 2026-10-03, Asia/Seoul. Reviewed snapshot: `Outline_v16.md`, 400 lines, SHA256 `cf2177691423380c1dfc66f2e83960f92fc84b9780d1e2111b70845c36377de1`. Line locations below refer to this snapshot; section names remain the primary reference after revisions.

Scope: independent manuscript/evidence review. This reviewer writes only this new review file. The outline, manuscript, methods, configurations, ledger, and remote experiments are not modified. `docs/method.md` was not read. The review follows `ai-journal-reviewer/SKILL.md` and its full-review checklist. It evaluates local recorded evidence; it is not a comprehensive literature novelty certification or a review of the compiled TeX manuscript.

## Decision and main assessment

**Outline status: revisions required, suitable for continued drafting. Submission-ready methodological case: not yet established.** The outline is substantially more defensible than a narrative claiming rank preservation, FPR control, or broad superiority. The remaining central weakness is scientific: the current evidence establishes recovery from an overweighted shallow-addition baseline much more strongly than a reason to retain shallow fusion over deep-only or a well-chosen global weight. Running experiments target the right uncertainty, but their pending outcome cannot count as a contribution already demonstrated. This assessment does not estimate acceptance probability.

The defensible paper is a focused empirical study of a support-derived, one-sided shallow shrinkage rule within frozen reference matching. Its central question is whether that rule provides reproducible region-localization value beyond strong simple alternatives. An additional benchmark table, a high complete-system score, or a longer account of failed mechanisms cannot independently settle that question.

| ID | Severity | Resolution type | Acceptance status at this review |
|---|---|---|---|
| C1 | Critical for the methodological claim | New matched evidence + conditional claim revision | OPEN: weighted/hard results and repeated deep comparison missing |
| M1 | Major | Writing/protocol synchronization | OPEN: exact weight run and selection rule already fixed elsewhere |
| M2 | Major | Provenance audit; bridge only if needed | OPEN: cross-suite pairing not yet certified in the outline |
| M3 | Major | Writing/metric definition | OPEN: standardized pAUROC and aggregation need explicit definition |
| M4 | Major | Existing evidence integration | OPEN: strict-low-FPR counterexample and pre-CQ results omitted |
| M5 | Major | Writing/data exposure and inference boundary | PARTLY RESOLVED: disclosure is good; held-out-generalization gap remains |
| M6 | Major | Related-work/bibliography and claim scope | OPEN: stale bibliography link; nearest fusion distinction unfinished |
| m1 | Minor | Wording and main-paper prioritization | OPEN: abstract and main figure/table hierarchy need tightening |
| m2 | Minor | Measurement definition | OPEN: gate-activation aggregation should be specified |
| m3 | Minor | Presentation/rounding | OPEN: five-decimal manuscript precision is unnecessary |

“OPEN” means a concrete author action remains, not that the existing measurements are invalid. Completing writing corrections does not close C1 or establish acceptance.

## Accepted aspects and numerical checks

1. **The mathematical correction is sound.** §3.3, lines 133–139 restrict the pointwise bound to nonnegative matched component scores before CQ and explicitly reject ranking/FPR/recall preservation. The counterexample is valid. Do not replace it with an operational guarantee during TeX compression. The bound says nothing about the necessity of the method and should not be promoted to a principal theorem.
2. **Implementation descriptions checked here are consistent.** q95 scale, support q99 threshold, LSO/LTO after sampling, cosine distance, full/weighted/hard/gated equations, and per-bag fusion agree with `reference_matching.py` and `fusion_factorial.py`. CQ is correctly identified as predictor-side component max fill. The threshold activation condition is invariant to the positive scale divisor when the corresponding threshold is divided by the same scale.
3. **The headline factorial arithmetic is correct.** D−B = +5.13007 AUPRO pp; D−C = +1.66385; D−E = +0.47482 AUPRO / −0.24459 pAUROC. The normalization×fusion interaction is approximately −1.193pp, not positive synergy.
4. **Table 2 means and sample SDs were independently recomputed.** Canonical seed macros reproduce all seven rows: SCARF, SuperAD, PatchCore, AnomalyDINO, SubspaceAD, RAD, and FastRecon, including both metrics. For SCARF these are 59.62938±0.79638 and 77.23270±0.29345; for SuperAD 55.15555±0.46035 and 76.53584±0.60479. The three-draw AUPRO difference is +4.47383pp. This is arithmetic validation, not a new certification of baseline implementations.
5. **E1's SD convention is correct.** Recomputing from the five gated macro values, including VOID seed0, gives 56.06787385±2.38627209 at 2 shots and 59.47789394±0.60011306 at 16 shots with sample SD. The outline correctly distinguishes this from its three-draw whole-pipeline summary and from residual-versus-deep repeatability.
6. **Important negative findings are already retained.** §4.2 reports all-five-shot pAUROC losses against deep-only; §4.4 keeps MVTec/VisA low-shot failures; §4.5 accurately reports the source-fixed composite-hypothesis failure. The documented negative results do not require re-execution simply to confirm them.

Evidence: [factorial report](../../results/comparisons/fusion_factorial_20260922/comparison.md), [canonical CSV](../../results/comparisons/psma_mr_vs_history_20260909/psma_mr_vs_history_20260909.csv), [E1 report](../../results/v7_evidence/E1_v2/comparison.md), [source-fixed report](../../results/comparisons/source_mechanism_20260922/comparison.md), [fusion implementation](../../src/fmad/methods/fusion_factorial.py), [reference matching](../../src/fmad/methods/reference_matching.py), [CQ implementation](../../src/fmad/postprocess.py).

## C1 — Central methodological value over simple alternatives is still unestablished

**Locations:** Abstract line 13; Introduction P3–P5 and contributions, lines 33–59; Tables 1a/1b and interpretation, lines 180–216; §8.2 N1/N2, lines 348–355.

**Evidence and consequence:** The +5.13pp headline is against `deep + shallow`, which gives shallow twice the weight of each deep layer. At 16 shots the margin is +0.93pp over q95 mean3 and +0.47pp over deep-only, with −0.24pp standardized pAUROC. At 2 shots it is only +0.03875pp AUPRO over deep-only, with −0.24007pp pAUROC; at 1 shot both metrics lose. E1's five draws compare residual to full addition, so they cannot establish repeatability of the much smaller residual–deep difference. The 0.80 SD of SCARF alone also cannot test a paired difference.

The rule can be written, for `z5 > 0`, as `e5 = w(z5; tau) z5`, with `w = max(1 − tau/z5, 0)` and `w(0; tau) = 0`. Thus the distinguishing choice is support-derived, score-dependent weighting and threshold subtraction. Describing it as calibration plus a ReLU does not by itself establish methodological novelty, and renaming it “excess” does not address that objection. This algebra is not a claim that a particular prior paper implements the same rule.

**Exact replacement for contribution 2, usable before results:**

> We evaluate support-derived excess fusion against deep-only scoring, equal layer averaging, and unrestricted shallow addition under a fixed reference-matching pipeline. Existing comparisons show recovery from unrestricted addition, while repeated and weighted-baseline evaluations are needed to establish the incremental value of the excess rule.

The second sentence is appropriate for the working draft; a submission must replace it with actual completed findings or omit the unfinished claim. Do not state “we separate relative weighting and excess fusion” as an accomplished empirical result until the new weighted/hard controls are complete.

**What closes the scientific issue:** Finish the registered global-weight/hard screen with valid reuse, then complete the missing paired deep/locked-weight/excess repetitions if that screen warrants them. Report draw-wise paired macro differences and the class-level pattern, not overlapping unpaired error bars as a significance test. Preserve a concrete criterion for “useful” rather than changing the desired metric after viewing results. Frozen cross-dataset comparisons can test transfer, with exposure disclosed. If deep-only or a fixed weight matches the residual rule, reduce the methodological claim; honest wording alone does not create a positive novelty result. No request for an indiscriminate large grid is made here.

## M1 — The pending slots contradict the already fixed running protocol

**Locations:** §3.3 line 129; §8.1 lines 325–338; Table 1a global-weight row, line 195.

**Concrete mismatch:** The outline says exact λ=.5 is a “PENDING protocol decision” and lists only {.125,.25,2} as missing positive weights. [PROTOCOL.md](../../configs/benchmarks/scarf_v16_fusion_20261003/PROTOCOL.md), lines 15–21 and 54–80, already fixes exact λ=.5, all six candidate values, the 2-shot selection population, the tie rule, and 80 new class-runs. Both `w05_s02.yaml` and `w05_s16.yaml` exist and are listed in [launch.json](../../results/remote_runs/dsba3/scarf_v16_fusion_20261003/launch.json). Local copied shard status files indicate running queues with partial class completion; they do not contain a complete new full8 macro at this review's inspection. Runtime status can change independently of this report.

**Exact replacement for §8.1's candidate and selection paragraphs:**

> The recorded development protocol evaluates λ∈{0,.125,.25,.5,1,2} in `d + λ z5`. Deep-only (λ=0), full addition (λ=1), and residual controls are reused subject to the per-class provenance audit. Four positive-weight settings {.125,.25,.5,2} and hard gating are newly executed at 2 and 16 shots: ten configurations and 80 class-runs. Exact λ=.5 is fixed in the protocol to check finite-precision differences from the proportional equal-mean score; its result, rather than the decision to execute it, is pending.
> Select one global λ using only 2-shot seed0 full8 macro AUPRO@.05, including λ=0. For a tie within 1e−10 in the unscaled metric, choose the smaller λ. Lock this choice before interpreting 16-shot results; the 16-shot sweep is sensitivity and shot-transfer analysis, not a second tuning objective. All runs use the already seen AD2 public population, original FP32 feature execution, cap256, four scales/bags, fixed CQ, and no noise routing.

**Table replacement:** `Global λ* (selected on AD2 2-shot seed0; fixed at 16 shots)`; use `PENDING RESULT` for the new rows. In line 129 replace “새 protocol이 ... 채택하면” with “새 protocol은 정확한 λ=.5 실행을 이미 채택했다.” Ensure the author's selection record exists before interpreting the already computed or still running 16-shot outputs.

**Closure:** wording/config/launch agreement; no method edits and no additional experiment are needed to correct this issue.

## M2 — Cross-suite arithmetic is not yet a certified matched comparison

**Locations:** Abstract line 13; §4.1 lines 176–178; Table 1a caption/source note, lines 184–198; §8.1 reuse row, line 331.

**Evidence and consequence:** The outline appropriately requests an audit in line 198, but the abstract already groups the factorial and mean3 differences under a “controlled” comparison. Some reused arms come from VOID, others from the figure-pilot/factorial suites, and new weighted arms run in the new immutable release. A sampled legacy figure-pilot manifest records Python 3.11.16/Torch 2.6.0+cu124; a sampled imported VOID manifest records Python 3.11.15/Torch 2.6.0+cu124. The new launch specifies `/usr/bin/python3`, but the locally available launch/status records alone do not establish its full runtime. These observations require explicit environment accounting, not a conclusion that the results are necessarily incompatible. Neither the same nominal seed nor the label “FP32” certifies identical feature banks.

The original factorial D/B/E pairing has direct evidence; the source-fixed report also records exact metric/bank agreement for its corresponding factorial controls. Do not downgrade that established within-suite evidence merely because a newer cross-suite audit is unfinished.

**Exact replacement for Table 1a caption:**

> Completed seed0 development results, grouped by originating suite. The A–E factorial rows are matched within the recorded protocol. Mean3 and newly executed weighted/hard rows are combined for paired interpretation only after verifying ordered relative support identities, support/query/GT contents, backbone weights, feature-bank hashes, preprocessing, evaluator contract, and runtime differences. Nominally shared settings alone are not treated as proof of identical execution.

**Exact interim abstract replacement:**

> In a matched 16-shot development factorial on MVTec AD 2, excess fusion improves AUPRO@.05 by 5.13 points over full shallow addition and by 0.47 points over deep-only, while standardized partial pixel AUROC is 0.24 points lower. A separate equal-averaging experiment records a 0.93-point AUPRO gap; its cross-suite pairing is being audited.

The last clause is for the draft only; finish the audit before submission. If the audit passes, describe its established scope concisely. If bank/content/evaluator identity fails or is absent, use an appropriate matched bridge/control run before treating sub-point differences as method effects. A matching complete-system FP32 result on one seed is not automatically a bridge for every 2-shot arm.

## M3 — Metric definitions are insufficiently precise for a ranking-centered paper

**Locations:** §4.1 P2 line 174; Abstract line 13; all pAUROC columns.

**Evidence:** [streaming.py](../../src/fmad/evaluation/streaming.py), lines 133–188, casts score maps into the float16 ordering contract, weights each connected defect region equally for PRO, divides integrated PRO by .05, and applies a standardized partial-AUC formula. The reported pAUROC values around 70–77 are not a raw integral over [0,.05] and are not simply that integral divided by .05. With `a=.05` and `I=∫₀ᵃ TPR(f)df`, the implementation uses `0.5 [1 + (I − a²/2)/(a − a²/2)]`.

**Exact replacement for §4.1 P2's first sentences:**

> We report class-macro AUPRO integrated over FPR∈[0,.05] and normalized by .05, and standardized partial pixel AUROC over the same interval. The latter equals 0.5[1+(I−a²/2)/(a−a²/2)] for a=.05 and I=∫₀ᵃ TPR(f)df. Metrics pool native-image pixels within each class; PRO weights connected ground-truth regions equally, whereas pixel AUROC weights foreground pixels. Background pixels in anomalous images contribute to FPR alongside pixels from normal images. Scores follow the shared float16 evaluation contract. We average class metrics equally and report sample SD across draw-wise macros; all displayed values are multiplied by 100.

**Interpretation addition:**

> The opposite AUPRO and pAUROC directions indicate metric-dependent tradeoffs. They do not by themselves prove recovery of small defects; that claim requires the separate native-region analysis.

**Closure:** documentation of the existing evaluator; no new metric definition or rerun is requested. Keep whole-map float16 evaluation distinct from optional FP16 feature extraction.

## M4 — Existing strict-low-FPR and pre-CQ evidence should replace avoidable uncertainty

**Locations:** Introduction P1 line 23; §3.4 line 152; §4.5 P1 line 261; §5 line 291; §8.2 N4 line 353.

**Evidence:** The existing [factorial report](../../results/comparisons/fusion_factorial_20260922/comparison.md), lines 197–224, already contains raw/CQ results and matched-FPR curve points. At all-test FPR=.1%, PRO is **21.380 full addition, 20.153 deep-only, and 19.733 residual**. At shift1 FPR=.1%, the corresponding values are **17.775, 20.700, and 17.622**. Thus “improved low-FPR ranking” must refer to the integrated [.0,.05] metric in specified comparisons; improvement does not extend uniformly toward the strictest operating region. This is distinct from the already included source-fixed FPR failure.

The same report has raw AUPRO 53.485 full, 57.743 deep, 58.321 residual; raw residual−full≈+4.837pp and residual−deep≈+.579pp. CQ is therefore not the sole explanation for the recorded effect, although residual's final score remains conditional on its pipeline. No new no-CQ inference is necessary to establish this existing observation.

**Exact addition after §4.5 P1:**

> Existing curve summaries already show a boundary at strict FPR: at FPR=.1%, residual PRO is 19.733 versus 20.153 for deep-only and 21.380 for full addition on the all-test population. These points are linearly interpolated from the recorded float16 ROC, not source-calibrated deployment thresholds. The integrated AUPRO@.05 gain should not be read as improvement at every operating point. Raw maps also show a residual–deep AUPRO gap of approximately .579pp before CQ in the same 16-shot factorial.

**Exact N4 clarification:**

> Existing raw/CQ macro scores and aggregate PRO-at-matched-FPR summaries are complete. Pending work is identification of individual recovered/lost native GT regions and any size-stratified claim, particularly against the newly selected weight.

**Closure:** reuse/report existing evidence. New maps are needed only for the genuinely missing region-level question and under the appropriate output scope.

## M5 — Disclosure is strong, but it does not convert development results into generalization evidence

**Locations:** §3.2 line 106; §4.1 P4 line 178; §4.3 line 236; §4.4 line 240; §8.1/8.2 lines 338–355.

**Assessment:** The outline correctly states exposure of AD2 public, repeated supports, MVTec/VisA diagnostics, and private submissions. Preserve this through TeX. The 2→16-shot experiment shares classes/query population, and selecting the weight using 2-shot labels creates a declared development-selected comparator. It tests shot-transfer of that weight; it is not an untouched dataset transfer experiment. The residual q95/q99 and other design choices also have development history, so a frozen support-fitted threshold cannot be sold as a baseline with zero research tuning.

**Exact addition to the setting paragraph:**

> All reported AD2 public fusion comparisons are development evaluations. The global shallow weight is selected with 2-shot public labels and then frozen for the 16-shot comparison on the same query population. Additional support draws characterize sensitivity to normal-reference selection on that population; they do not provide independent test-population replication. The residual rule's design constants also reflect prior benchmark development.

**Scientific gap:** New frozen comparisons on already exposed MVTec/VisA conditions can strengthen scope evidence, but should remain labeled as development transfer. A claim of independently validated generalization requires documented fresh evaluation access/splits and a frozen decision procedure; neither new seeds nor an unlabeled private GT interface supplies this automatically. The present paper can be honest without claiming such validation, but the scope of its empirical conclusion must then remain restricted. Do not invent an untouched holdout in this revision.

## M6 — Related-work distinction and bibliography remain incomplete

**Locations:** §2 lines 63–69; contributions lines 55–59; §4.3 external-method table, lines 220–234.

**Concrete correction:** Line 69 links the legacy WACV `main.bib` as the current bibliography. That file's `superad2024` entry at lines 204–208 has a different title, author list, and arXiv identifier from the newly prepared `cvpr2027_v16/references.bib` entry. The new file explicitly records SuperAD as arXiv **2505.19750** and identifies the legacy **2407.01234** entry as unrelated. Do not propagate the legacy entry into the new TeX draft. This review checked the local entries; it does not independently certify every bibliography field. A public search attempted during this review returned no usable source content.

**Exact replacement for line 69:**

> References for the new draft are maintained in `cvpr2027_v16/references.bib` with their source-verification record. Legacy WACV bibliography entries are historical inputs rather than citation authority; in particular the old SuperAD entry must not be reused. Technical positioning should cite original papers and distinguish the actual reproduced configuration from the original method.

**Needed novelty paragraph:** State what is shared with the closest reference-matching methods, then identify the exact delta: per-component support-distance scaling plus support-derived shallow excess, under fixed feature banks. Explain why a support-based adaptive weight is being investigated despite a strong deep-only base. The q95 operation, quantile threshold, positive-part function, frozen backbone, and bagging are ingredients, not separate novelty claims. A focused check of the closest selective-fusion/calibration work remains necessary; this review makes no “first method” assertion.

**Baseline presentation:** Table 2 honestly identifies several backbone/reimplementation differences. Retain these labels visibly, and keep a compact budget statement for the four-scale pipeline versus 672-only SuperAD. A low reimplementation number is not evidence that the original method is intrinsically weak. For the main story, deep-only and the locked weight have higher evidentiary priority than adding more heterogeneous, low-performing external rows. Verification of each reproduced baseline's own intended protocol is distinct from the arithmetic checks above.

## Minor corrections and exact wording

**m1 — Abstract and presentation hierarchy.** “Combining distances from several layers provides complementary observations” (line 13) presupposes useful complementarity. Replace it with: **“Distances from different layers respond differently to reference mismatches, and their contribution to localization depends on their scale and relative weighting.”** Lead the core result with residual versus deep/locked weight, then give recovery versus full addition. Do not remove the deep-only pAUROC loss merely to shorten the abstract.

For the main paper, use one core table with deep, equal mean, full, locked weight, hard, and residual; one compact complete-system table; and one panel combining paired uncertainty/transfer boundaries. Move MAD factorial details, full intermediate-shot tables, private results, noise, and execution optimizations to the supplement. Keep the source-fixed failure and the strict-FPR counterexample in the main text even if their full table moves. The current four-table/three-figure plan is a budget to test in typesetting, not proof that the central argument fits eight pages. This report does not assess the compiled layout.

**m2 — Gate activation is an averaged diagnostic.** At line 263 add: **“These statistics average per-bag binary gates, project token cells to native image coordinates, average over resolutions, compute pixel-weighted normal/defect means within each class, and then average classes equally.”** The original factorial report, lines 168–193, explicitly distinguishes this quantity from final pixel detection and shows that pixel-weighted and image-mean defect activation can move in opposite directions. Preserve that measurement definition in supplementary captions.

**m3 — Round manuscript results consistently.** Five decimals belong in the evidence ledger; use two decimals in the TeX tables and consistent pp precision in prose. For very small effects, `.039pp` or `less than .04pp` is more honest than rounding it to `.0` or describing it as substantial. Preserve the source-suite last-digit discrepancy already documented for 16-shot VOID/factorial values; do not silently reclassify it as a new effect.

## Concrete revision loop and acceptance checks

1. **Immediate author edits:** synchronize §8 with the actual protocol (M1); define standardized metrics (M3); include existing strict-FPR/raw evidence (M4); replace the bibliography authority and tighten the abstract/contribution wording (M6, m1). These are reviewable corrections using available evidence.
2. **Provenance decision:** finish the separate reuse audit and mark exactly which cross-suite comparisons passed (M2). Do not label it passed merely because this review requested it. Any needed execution bridge is a separate recorded validity action.
3. **Scientific decision after the registered screen:** state the actual 2-shot-selected λ, its locked 16-shot performance, hard-gate outcome, per-class regressions, and execution completeness. Only then decide the bounded paired/transfer follow-up described in the protocol (C1). Do not optimize away an unfavorable deep-only or hard-gate result.
4. **Second consistency pass:** check Abstract → Contributions → Table 1 → Discussion → Conclusion against the same completed evidence. Update the outline's §10 with issue IDs, actual locations changed, and unresolved scientific gaps. A written response saying “will fix” does not count as an applied revision.
5. **Submission gate:** an accurate draft can proceed while results are pending; a submitted paper cannot present PENDING experiments as established contributions. A defensible methodological case still needs the evidence identified in C1. If it is absent or negative, narrow the paper's claim and reassess its substantive contribution rather than predicting acceptance.

**Recommended current status for the outline's §10:** “Independent review completed; writing/protocol corrections required; mathematical overclaims resolved in this snapshot; weighted-baseline value, paired residual–deep uncertainty, and cross-suite reuse remain open.”

## Second pass — verification of applied revisions, 2026-10-03

**Current assessment: the observed writing and protocol corrections are accepted within the limits below; C1 scientific evidence and M2 reuse certification remain OPEN.** This section supersedes the first-pass issue statuses only where explicitly stated. The original review above is preserved as the historical assessment of the earlier snapshot. This is acceptance of particular revisions, not a recommendation to accept the paper or an estimate of acceptance probability.

Reviewed at approximately 03:25 Asia/Seoul: [Outline_v16.md](Outline_v16.md), **408 lines**, SHA256 `fbeb3924e08428a0ca22c59ec85f3885ff01bb7629a9ea90c9469f6e667966b3`. All locations in this addendum refer to that snapshot. The comparison contract was reread in [PROTOCOL.md](../../configs/benchmarks/scarf_v16_fusion_20261003/PROTOCOL.md), 101 lines, SHA256 `246e4c846cf92c5c0a840c511a3511a15290d93a58f0d0e88ef814bc29cdac68`. This pass verifies the applied wording against the previously checked evidence and protocol; it does not certify live job status or newly completed results. Only this review file was edited. TeX review remains assigned separately; `docs/method.md` was not read.

### Issue-by-issue acceptance record

| ID / original severity | Observed revision and location in the new snapshot | Second-pass status and remaining boundary |
|---|---|---|
| C1 / Critical | Abstract line 13 leads with +0.47 AUPRO versus deep-only and −0.24 standardized pAUROC, before +5.13 versus full addition. Contribution 2, line 56, says repeated and weighted comparisons are needed to establish incremental value. Lines 297–305 and Decision C at line 350 preserve the conditional conclusion. | **Wording correction CLOSED; scientific evidence OPEN.** No completed locked-weight/hard result or paired residual–deep uncertainty has been supplied by this revision. N1 remains PENDING at line 356. |
| M1 / Major | Lines 131, 197 and 338–344 now distinguish the already adopted exact λ=.5 run from its pending result. Line 344 specifies all six weights, 2-shot seed0 full8 macro AUPRO selection, unscaled tolerance 1e−10, smaller-weight ties, the lock before interpreting 16 shots, and 10 configs × 8 classes. | **CLOSED for protocol synchronization.** These agree with protocol lines 15–21, 54–64 and 75–80. Actual selected λ and completed performance remain pending, not a writing omission. Minor status-language cleanup is listed below. |
| M2 / Major | Abstract line 13 explicitly calls the mean3 comparison a separate experiment with pairing under audit. Table 1a line 186 requires ordered support/content, backbone/bank hashes, preprocessing, evaluator and runtime comparison; line 200 requires final provenance checks. Line 405 explicitly keeps the audit open. | **Disclosure correction accepted; reuse audit OPEN.** The revised prose is not an audit artifact and does not certify cross-suite matching. Preserve the original within-factorial comparisons while the responsible agent establishes the admissible reuse scope. |
| M3 / Major | Line 176 now gives `0.5[1+(I−a²/2)/(a−a²/2)]`, defines a=.05 and I, normalizes AUPRO by .05, distinguishes region and pixel weighting, includes normal pixels in anomalous images in FPR, and states equal-class macros, draw-wise sample SD and ×100 reporting. It also separates evaluator float16 ordering from feature FP16. | **CLOSED.** The stated metric formula and aggregation match the evaluator already checked in the first pass. The paragraph does not infer tiny-defect recovery or target-FPR control from these metrics. |
| M4 / Major | Line 265 adds all-test .1%-FPR PRO 19.733/20.153/21.380 for excess/deep/full, corresponding shift1 counterexamples, interpolation/source-threshold qualifications, and pre-CQ AUPRO evidence. N4 at line 359 now distinguishes completed raw/CQ and aggregate PRO from missing individual-region analysis. | **CLOSED.** The negative operating-point evidence is now visible, and completed evidence is no longer scheduled as missing. Native-region recovery/loss and size claims still require their own analysis; this does not reopen the corrected omission. |
| M5 / Major | Lines 108 and 180 distinguish support-only fitting from benchmark-driven design, acknowledge weight selection using public labels, and describe 2→16-shot evaluation on the same query population as development shot-transfer. Lines 299 and 361 retain limits on independence and uncertainty. | **CLOSED for exposure and inference-scope disclosure.** No independent-test-population validation is established or claimed. Additional support draws remain sensitivity evidence. This review does not add an unrequested fresh-holdout experiment. |
| M6 / Major | Line 69 points to the new bibliography and its provenance record and rejects the legacy SuperAD entry. Both linked files exist. Line 71 supplies the score-dependent-weight algebra and states that it is not a novelty proof; lines 63–67 distinguish shared ingredients and strong simple comparators. | **Bibliography-authority and positioning wording CLOSED.** This pass does not independently certify every new citation. The closest selective-fusion literature check remains explicitly unfinished at line 67; comprehensive novelty is not marked established. |
| m1 / Minor | Abstract line 13 replaces presumed complementarity with scale/weight-dependent responses and places the deep comparator first. §9, lines 365–383, prioritizes core fusion evidence and retains negative findings. | **Abstract wording and outline-level prioritization CLOSED.** Whether the four-table/three-figure plan fits the actual main paper is unverified and belongs to the separate TeX review. |
| m2 / Minor | Line 269 now states bag-gate averaging → native-coordinate projection → resolution averaging → class-wise pixel-weighted normal/defect means → equal-class averaging. It warns that image-mean and pixel-weighted activation can move oppositely. | **CLOSED.** Line 267 still correctly distinguishes activation from final FPR, recall and causal normal-error removal. |
| m3 / Minor | The applied-revision table at line 406 explicitly adopts two decimals in TeX main tables, .039pp for the tiny difference, and full precision in supporting records. | **Outline formatting policy accepted; TeX application UNVERIFIED.** The outline retains precise evidence values. No claim is made that manuscript tables were inspected or corrected by this reviewer. |

### Remaining nonblocking wording cleanup

These are exact proposed replacements for the parent author to apply if appropriate; this reviewer has not edited the outline. They do not reopen the substantive M1 correction.

1. **§8.1 Reuse row, line 337:** replace `새 run 전 provenance 재사용 판정` with **`paired Δ 해석·기존 cell 재사용 전 provenance 판정 — 감사 미완료`**. The protocol requires the audit before paired interpretation; this status should not imply that already launched new runs were waiting for permission to start.
2. **§8.1 Hard gate status, line 340:** replace bare `PENDING` with **`실행 대상으로 확정; 결과 PENDING, 실제 진행 상태는 실행 기록 참조`**. This distinguishes an agreed experiment from an unmade design decision without guessing whether the job is queued or running.
3. **§7 equal-mean row, line 313:** replace claim `Equal mean보다 우수` with **`별도 suite의 equal mean 대비 관측 격차`**, and status `단일 draw 근거` with **`단일 draw; cross-suite provenance 감사 미완료`**. Table 1a and the abstract already carry this caveat; repeating it in the standalone claim matrix prevents that row from being quoted as a certified paired effect while M2 remains open.

### Scientific closure requirements retained after the writing loop

**C1 remains a results question.** Report the actual 2-shot-selected global λ and the locked 16-shot outcome, hard-gate comparison, full-class completion and regressions before deciding the bounded repeat/transfer stage. For repeatability, report draw-wise paired excess-minus-deep and excess-minus-locked-weight differences; E1's excess-versus-full repeats and the SD of SCARF alone cannot replace them. If the strong comparators match or outperform excess, revise the central contribution accordingly. The revised outline already provides this decision path and does not pre-claim a favorable outcome.

**M2 remains a provenance question.** The responsible audit must state which historical/new cells can legitimately be paired, the evidence establishing their content and execution identities, and any unresolved differences. A partial audit must retain a partial certification scope. If necessary identities cannot be established, choose a documented matched bridge before assigning a cross-suite difference to fusion. This pass has not run or authorized a bridge itself.

**Main-paper story now has an internally consistent boundary.** The accepted narrative is empirical support-based shallow shrinkage under a fixed pipeline, with a small, metric-dependent deep-only difference and explicit negative conditions. Adaptive-weight algebra, pointwise score bounds, complete-system benchmark performance and recovery versus full addition do not independently establish necessity over the locked weight. Retain the strict-FPR counterexample and development exposure when compressing the paper. The literature check and typeset layout/rounding checks remain separate verification work.

**Suggested §10 status after this second pass:** “Applied writing/protocol revisions independently checked against the 408-line snapshot; M1/M3/M4/M5/m2 closed, C1 wording and M6 bibliography/positioning corrected, abstract prioritization accepted. C1 scientific evidence and M2 reuse audit remain OPEN. Closest-literature verification and actual TeX layout/rounding are not certified by this pass.”

## Final second-pass addendum — corrected source mapping, 2026-10-03

**Correction of reviewer error:** the previous version of this addendum improperly extended the negative historical VOID/mean3 reuse audit to the separate FF_A–E factorial. That inference and its blanket withdrawal of the factorial's matched status are retracted. This corrected addendum replaces only that last addendum; earlier review snapshots are preserved. **The separate FF_A–E study is a matched 2×2 normalization–residual factorial plus a deep-only control. Audited VOID/mean3 reuse remains INVALID for the intended execution-matched primary matrix; 48 replacement controls remain pending.** C1 scientific evidence remains OPEN.

Source of this correction: the parent relayed Huygens’ completed, exact source-mapping follow-up. This reviewer is recording that confirmed finding and has not repeated the broader audit. The parent will record the provenance footnote in the suite audit addendum and paper.

### Correct evidence identities and permitted inference

| Evidence family | Established source scope | Review conclusion |
|---|---|---|
| Separate FF_A–E factorial | The [factorial report](../../results/comparisons/fusion_factorial_20260922/comparison.md) maps to 40 manifests from the `fusion_factorial_20260922` local lane0 / dsba3 lanes0–3. Huygens found **zero shared run IDs with the audited VOID records**. Within each of eight classes, all five arms match bank, runtime, package code, support, query, GT, backbone and evaluator. | **VALID within-class controlled comparison.** FF_A–D form the 2×2 normalization (MAD/q95) × fusion (full/excess) factorial; FF_E is the q95 deep-only control. Suite identity and nearly equal numerical values must not be used to substitute VOID artifacts for these runs. |
| Audited historical VOID off/all/gated and figure-pilot mean3 | [REUSE_AUDIT.md](../../results/remote_runs/dsba3/scarf_v16_fusion_20261003/REUSE_AUDIT.md), especially its historical-pair table, establishes the VOID/mean3 bank/runtime mismatches. Historical VOID off/all match one another; that pair does not establish matching to VOID gated or mean3. | **INVALID assembly for the intended execution-matched primary comparison.** Keep their recorded metrics as historical development observations with the actual run identities. The negative decision has been reached; it is not merely awaiting an audit. |
| New weighted/hard suite and replacement off/all/gated | The parent retains 80 weighted/hard class-runs and adds 48 off/all/gated class-runs at 2/16 shots, full8. | **128 class-runs planned; complete matched controls and result interpretation PENDING.** Internal matching of the separate FF study does not certify its reuse with this new suite. |

Huygens reports Python **3.11.16**, Torch **2.6 + cu124**, and package hash prefix **`5acd7aca48edd1941…`** for all 40 FF manifests. Full environment metadata has two variants across classes, while the five arms within each class match. The correct statement is therefore within-class execution matching across arms. No claim that every class used one globally identical environment is needed for the matched within-class contrasts and their class-macro aggregation.

**Consequences for the paper's numbers:** the FF-derived 16-shot D−E difference of **+0.47482 AUPRO / −0.24459 standardized pAUROC**, and D−B **+5.13007 AUPRO**, retain their interpretation as matched development contrasts in that factorial. They remain single-draw, metric-dependent observations and do not establish repeated superiority over deep-only or a tuned weight. The historical **2-shot +0.03875pp** VOID difference does not inherit the FF study's matching certification. The **+0.93319pp** comparison to separate mean3 is likewise a cross-suite difference, with no certification of a matched fusion-only effect from the FF audit.

The existing strict-FPR and pre-CQ evidence originating from the separate FF report also retains its FF provenance; the negative VOID audit is not grounds to invalidate those comparisons. Preserve those negative findings and raw/CQ evidence under their actual source identities. Numerical coincidence across the reports cannot transfer either a PASS or a MISMATCH between run families.

### Bounded replacement wording for the parent

The previous version’s blanket abstract/table replacements and request to qualify every factorial gap as bank/runtime-confounded are withdrawn. Use the following scoped wording instead.

**FF table caption / provenance paragraph:**

> FF_A–D constitute a matched 2×2 normalization–residual factorial, with FF_E as a deep-only control. All five arms share bank, runtime, code, support/query data, backbone and evaluator within each class. These 40 runs are separate from the historical VOID controls audited for reuse; their run IDs do not overlap. The FF-derived contrasts are valid within this development study. Cross-suite mean3 and new weighted comparisons require their own provenance qualification.

**Reuse / execution status paragraph:**

> Historical VOID/mean3 assembly is invalid for the new execution-matched primary matrix. We retain those results as separately labeled development context and add off/all/gated at both shots, full8: 48 controls in addition to 80 weighted/hard class-runs, for 128 class-runs in total. Primary comparisons and global-weight selection will use the completed current-suite controls after their matching checks. The separate FF factorial remains valid within its own matched setting.

The abstract may retain the **matched 16-shot factorial** wording and the FF-derived +0.47/−0.24 and +5.13 figures with the correct source mapping. It should not describe their bank/runtime as mismatched on the basis of VOID. The equal-mean cross-suite qualification remains necessary. The parent's forthcoming provenance footnote should make these source identities explicit; this reviewer has not independently inspected that future edit.

### Final status after the source correction

| Item | Corrected review status |
|---|---|
| Separate FF_A–E | **Matching confirmed by the relayed Huygens mapping.** The previous addendum's invalidation is withdrawn. The first/second-pass acceptance of its within-class controlled contrasts is restored within this scope. |
| M2: historical VOID/mean3 reuse | **Negative reuse decision confirmed for the new primary matrix.** Scientific remediation remains **OPEN: 48 replacement controls and their matching checks PENDING**. |
| C1 | **OPEN.** Valid FF contrasts support bounded development observations, but locked-weight/hard comparison and paired repeatability remain missing. Correcting the provenance conflation does not supply those results. |
| M1 | The parent reports that outline/TeX/protocol now reflect the **128-class-run** plan. This source correction does not independently verify those edits or live execution status. |
| Other writing findings | The prior limited closures stand. M4's FF strict-FPR/pre-CQ evidence remains accepted with its correct source identity; literature certification and actual TeX layout/rounding remain outside this pass. |

The new primary selection still uses full8 current-suite 2-shot off/all endpoints and weight candidates, the declared 1e−10 unscaled tie rule favoring smaller λ, and a lock before interpreting the 16-shot sweep. No additional experiment beyond the adopted 48-control extension is requested by this correction.

**This bounded correction completes the reviewer update.** Only the last addendum of this review file was revised. The accepted interpretation is: **separate FF factorial internally matched; historical VOID/mean3 assembly invalid for primary reuse; 128-run execution plan retained with 48 controls pending; C1 methodological evidence still open**. No acceptance-likelihood claim is made.


## 2026-10-03 — Resolution 서사 반영 후 검토·수정 루프

사용자가 승인한 수정 방향에 대한 작성 에이전트의 집중 재검토다. 앞선 독립 리뷰 이력은 그대로 보존한다. 대상은 Outline v16 및 CVPR TeX의 초록·서론·방법·실험·한계·결론이다.

| 검토 항목 | 발견 / 피드백 | 반영 및 확인 |
|---|---|---|
| Resolution이 초록과 문제 정의에서 누락되는가? | 기존 중심 서사는 layer 단위와 shallow 기여만 설명했다. | 연구 질문을 다층·다해상도 관측의 결합으로 확장. §1 P3–P4, §3.1–3.6, §4.3, §5–6에 같은 구분을 적용했다. |
| Multiscale 구성과 routing 선택이 혼동되는가? | 최고 단일 해상도 비교와 noise routing 비교는 서로 다른 질문이다. | §4.3 Q1에 E12 구성/용량 통제, Q2에 clean/noise 정책 비교를 배치했다. |
| Fixed-low control이 정말 미완료인가? | 원자료 재확인에서 NA_lowres clean/noise full15 결과가 발견됐다. | 완료값 84.60962/69.00893을 복원하고 fixed4 85.36138/64.67413 및 routed 85.36138/69.00893과 비교했다. 추가 실행 슬롯으로 만들지 않았다. |
| Routing의 근거가 과장되는가? | Gaussian-s3의 모든 1,725 query에 발동하고 clean에는 발동하지 않았다. | 관측된 두 population의 branch 선택 결과로 제한. 추가 cutoff 개발 데이터, AD2 미발동, 일반 shift 탐지 및 미래 clean 보존 미보증을 명시했다. |
| 세 선택의 결합을 이론·synergy로 포장하는가? | 독립 ablation으로 fusion×routing 상호작용을 추정할 수 없다. | 근거 매트릭스에 미확립 상태 추가. q95는 confidence가 아니며 선택 해상도는 동일 가중 평균임을 수식·그림에 명시했다. |
| 본문 배치와 일관성이 맞는가? | 확장 수식의 첫 렌더에서 오른쪽 여백이 좁았고 참고문헌이 9쪽으로 넘어갔다. | 수식 줄바꿈 수정, 중복 실행 설명 압축. 최종 본문 8쪽(참고문헌 포함), 보충 4쪽. |

검증: 기존 main table 4개는 편집 전과 동일함을 비교했다. Fixed-low는 clean/noise 각각 15개 class cell을 확인했고 noise 값이 remedy 결과와 정확히 일치했다. Clean 비용 −0.751753pp와 noise 이득 +4.334792pp를 원자료로 재계산했다. Outline/README의 local link 누락 없음. 최종 LaTeX 로그에 warning, undefined reference/citation, overfull/underfull box 없음. 본문 1·4·6·8쪽 및 보충 2쪽을 렌더링해 초록·수식·방법 개요·해상도 근거·표·결론 배치를 확인했다.

남은 과학적 공백: matched global-weight/hard-gate 결과와 선택, deep/selected-weight/excess의 paired 반복, 고정 설정 transfer. 이 수정은 그 비교의 성공이나 CVPR 게재 가능성을 확정하지 않는다. 원래 v15, 방법 코드, 실험 config, 누적 ledger 및 원자료는 이 편집에서 변경하지 않았다.
