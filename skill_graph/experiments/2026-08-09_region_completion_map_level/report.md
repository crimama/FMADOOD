# Region-Completion Map-Level Combination Report

Date: 2026-08-10

Protocol: `region-completion-map-level-all8-20260809`

Verdict: `INVALID` — the protocol's own operator was unsound. The hypothesis
was never tested.

Scope: `superad_seg_AUROC_0.05=N/A`, `superad_seg_F1=N/A`,
`delta_vs_superad=N/A`, `comparable=false`.

## Outcome

Validity gates 1, 2, 5, 6 passed. Gates 3 and 4 failed. Per the preregistration
the verdict short-circuits to `INVALID`, and the hard-null, strict, and
continuation gates were **not evaluated** (`evaluated: false` in all three).
No endpoint of this experiment may be used as evidence for or against the
hypothesis.

## Root Cause — A Preregistration Error, Not an Implementation Defect

The protocol defined `R(M)` as per-image rank normalization and justified
validity gate 3 with the claim that `R` is "strictly monotone per image" and
would therefore leave `seg_AUROC_0.05` and `seg_AUPRO_0.05` unchanged.

That claim is false for this evaluator. All reported metrics are **pooled over
every image of an object** via accumulated float16 histograms
(`src/flow_tte_phase3_scorer_suite.py:503-563`). A per-image monotone map is
not monotone on the pooled pixel set: it rescales each image independently and
therefore changes the cross-image ordering that pooled metrics depend on.

Gate 3 measured exactly this and failed:

| object | metric | `position_current` | `D0_rank_current` | abs delta |
| --- | --- | ---: | ---: | ---: |
| can | seg_AUROC_0.05 | 0.573555 | 0.616240 | 0.042685 |
| can | seg_AUPRO_0.05 | 0.116656 | 0.202696 | 0.086040 |
| fabric | seg_AUROC_0.05 | 0.961722 | 0.548660 | 0.413062 |
| fabric | seg_AUPRO_0.05 | 0.482645 | 0.508653 | 0.026008 |

`R` itself is implemented correctly; its unit tests, including the mean-rank
tie rule, pass. The defect is in the operator this protocol chose.

The mechanism of the damage is concrete. Per-image rank normalization forces
**every** image to span the full `[0,1]` range, including good images that
contain no defect at all. The top-scoring pixels of a defect-free normal image
are thereby promoted to the same value as genuine defect pixels. `fabric`,
whose per-image score scales were already well aligned, is destroyed:
`seg_F1` falls `0.668776 -> 0.057339`. `can`, whose problem *is* cross-image
scale, improves. The macro average hides both.

Gate 4 failed as a consequence of the same design: the `D3` shuffled marginal
check could not hold under an operator that redefines each image's scale.

## What This Does and Does Not Mean

- The preregistered closure clause — "if the hard null absorbs the effect or
  the continuation gate fails, the current-plus-low-pass combination family is
  closed" — **does not fire**. That clause is conditioned on a valid experiment
  reaching its endpoints. This experiment never tested the hypothesis, so
  closing the family on it would be an unjustified kill.
- The complementarity evidence that motivated the protocol is untouched. The
  all-eight low-pass DVT result stands: AU-PRO `+.010513` versus raw against
  current DVT's `+.005437`, with `sheet_metal` AU-PRO repaired from `-.06559`
  to `-.00274`, and a shuffled-position null beaten by `+.056823/+.063081`.
- No gate was rewritten after seeing results, and no endpoint was read. The
  short-circuit behaved as designed.

## Hypothesis-Generating Observation (Not Evidence)

`D0_rank_current` is not the no-op the protocol assumed; it is a real
per-image normalization arm. Under it, `can` gains AU-PRO `+0.0860` and
pAUROC@0.05 `+0.0427`, while `fabric` collapses. Macro
`gt_component_recall_at_oracle` rises `0.4850 -> 0.6550`, but that rise is an
artifact of the same mechanism — every image is given high-scoring pixels, so
more components clear the oracle threshold — and must not be read as region
recovery.

The `can` direction is consistent with the independent finding of
`can-collapse-attribution-all8-20260809` that `can` fails at the operating
point rather than in its representation. It was observed after the fact under
an invalid protocol and therefore carries no confirmatory weight. It may only
be re-tested under a fresh preregistration.

## Corrected Design Requirement

Any successor must combine maps on a scale that remains comparable **across**
images. A normalization fitted on support or held-out normal data, applied
identically to every image, satisfies this; a per-image transform does not.
Because that requirement is the same "cross-image score scale" axis being
measured by `normal-tail-structure-predictability-all8-20260809`, the successor
should be designed after that protocol reports, not before.

## Execution Record

- Anchors `raw`, `position_current`, `position_lowpass` and arms
  `D0`/`D1`/`D2`: local RTX 4090 container, all eight objects.
- `D3_rank_max_shuffled` hard null: dsba5 container `hun_fsad_tta_012`, host
  GPU 3, immutable snapshot.
- Support manifest custody verified against SHA-256
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.
- No `anomaly_maps/` retained locally or in the pulled remote run.

## Conclusion

`INVALID`. The experiment failed on its own construction, not on its
hypothesis. Per-image rank normalization is unsound for a pooled-histogram
evaluator and is withdrawn as a building block. The region-completion question
remains open and untested.

Primary artifact: `results/region_completion_map_level_20260809/summary.json`.
