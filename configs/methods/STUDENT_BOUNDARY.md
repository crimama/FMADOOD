# K2 — student-aligned synthetic boundary (2026-09-12)

## Hypothesis and single changed component

K1's frozen-clean teacher boundary differs from the adapted student's actual
clean/shifted-clean/synthetic-background ranking pool. Existing full8 K1 logs
show active synthesis changes (345/512 steps), increased ranking pressure and
increased preservation error; vial loses 1.89786pp AUPRO. These are observations,
not proof that boundary mismatch causes the regression.

K2 replaces only the criterion used to choose among K1's strengths .25/.5/1.
For each candidate, score clean, shifted clean, synthetic and shifted synthetic
with the current student and the same source-excluded bank as fitting. Preview
the next paired transform and Torch sampler state without consuming either.
Use the exact H1 tail/random negative and random-positive sampling helper.

    gap(a, v) = Q25(sampled student positives(a, v))
                - Q95(sampled student negatives(a, v))
    g(a) = min_v gap(a, v), v in {unshifted, shifted}
    a* = argmin_a (g(a) - margin), subject to g(a) >= margin

Margin remains .05. Quantiles are empirical inverse CDFs. Ties select the
first candidate; no feasible candidate falls back to original strength 1.
There is no assumption of monotone score versus blend strength. The gap is a
selection proxy, not a guarantee for every hinge pair or unseen query FPR.
Log each candidate's gap/rank loss, selected strength/feasibility, and the
absolute difference between previewed and actual training rank loss.

The existing loss, loss weights, normal preservation/controller, feature bank,
adapter/metric, optimizer, RNG streams, family schedule, common pipeline and
inference are unchanged. Shared tail pair sampling is factored into a helper
with exact legacy value/gradient/RNG parity tests. `student_boundary: false`
reproduces K1; also disabling `boundary_synthetic` reproduces H1.

This is a bounded engineering ablation of K1, not a novelty claim. It differs
from F2's student reliability reweighting and J1's hinge reduction: no sample
weight or reduction changes. It does not address synthetic morphology, expand
shift families, or hard-enforce preservation. Student feedback can reinforce
bad geometry or increase preservation error; fallback may make K2 inactive.

## Locked development comparison

- H1 -> K1 -> K2; seed0, normal16, 448+672, layer23, 64 steps; common CLI.
- Same support-selection policy/content identities and query/GT per class.
  Compare manifests before accepting any metric delta as matched evidence.
- Seen TESTpublic development, not sealed final. Primary full8 macro
  AUPRO@.05; pixel pAUROC@.05 secondary; oracle F1 diagnostic only.
- Primary comparison K2 minus K1; H1 is the unchanged base control.
  Gain <= 0 means no primary improvement. Report vial and all other classes;
  macro gain alone does not establish that the vial regression is solved.
- Mechanism check: near-zero preview/replay error validates alignment of the
  sampled objective; strength/feasibility indicate intervention activity.
  Neither alone proves generalization or causality. Preserve P/b reporting.
- Local existing container GPU0: can/fabric/fruit_jelly/rice/sheet_metal/walnuts.
  Existing dsba3 container physical GPU2 only: vial/wallplugs. Two workers/GPU;
  each class stays on one device for every method. No other GPU allocation.
- At most 24 class runs, 3600s/class; no sweep, multi-seed expansion, automatic
  revision or extra benchmark. Stop on execution/finite/output integrity errors.

## Compute, storage and provenance

No additional learned parameters or inference branch (100384 parameters).
K2 selection uses two clean and six candidate-view feature extractions per
resolution/step, in addition to the unchanged four-view training loop. K1 uses
three candidate teacher views and an amortized clean teacher view. Report
fitting overhead; do not describe the comparison as compute-matched.

Keep only current clean/candidate score maps temporarily; discard every
candidate before the next. No cross-step dense maps or feature cache; only
scalar diagnostics and a shared RNG handle survive. Maps/checkpoints remain
off, normal support only, no query data or GT in fitting/selection.

Reuse scripts/experiments.py run/collect. Fix its existing parallel active-status
rendering before launch (no pipeline change). Use the existing remote local-code
snapshot update path, no Git push/container replacement. Retain remote logs;
retrieve compact metrics/config/manifests and update the canonical ledger/CSV.
