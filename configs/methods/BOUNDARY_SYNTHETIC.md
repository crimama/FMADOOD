# K1 — normal-boundary-conditioned synthetic construction (2026-09-12)

## One problem, one changed component

Target: synthetic discrimination can be too easy to teach useful separation
from hard normal patterns. I2 had zero ranking loss on 306/512 steps but lost
0.45986pp AUPRO; this supports insufficiency of the proxy, not a proven cause.
J1's loss is about four times its sampled mean on 456/512 observed steps;
hard mining is confounded with loss reweighting. Evidence: results/comparisons.md
and the preceding read-only whole-log analysis.

K1 changes only synthetic strength selection in H1. No J1 tail reduction,
I2 local-only loss, new score head, backbone update, or inference change.

For source x and its source-excluded frozen normal bank B_-x:

    t_x = empirical_quantile_0.95(s_frozen(x; B_-x)) + 0.05
    x_a = round(x + a * (A(x) - x)), a in {0.25, 0.5, 1}
    p_a = empirical_quantile_0.25(s_frozen(x_a; B_-x)[synthetic mask])
    a* = argmin_a (p_a - t_x), subject to p_a >= t_x

If no candidate satisfies the floor, use original strength 1 and log infeasible.
Do not label low teacher confidence as correctness. Scores need not vary
monotonically with strength. Ties use the first candidate. Quantiles are
empirical inverse CDFs (ceil(q*n)-th value), not interpolated quantiles.
H1's normal tail fraction and margin define the normal threshold and offset.
The lower positive quartile avoids choosing by a few high-score pixels.
The teacher is frozen baseline retrieval, bypassing adapter and metric.

Preserve the original synthetic family/mask/source schedule and RNG draws.
Both families use fresh draws; no additional RNG is consumed by selection.
Apply the unchanged paired photometric transform after selecting a strength.
Keep cross-location normal competition and every loss coefficient unchanged.
`boundary_synthetic: false` reproduces H1 exactly.

## Difference from existing attempts and limitations

E3/F2 already used weaker synthetic blends and student-score reliability gates.
K1 instead conditions construction on a frozen, source-excluded hard-normal
boundary; it does not gate/reweight positives or change hard-positive sampling.
This is a bounded engineering hypothesis, not a novelty or realism claim.
Normal-only teacher geometry may itself be wrong; weak artifacts may remain
unrealistic, masks can have patch-boundary spillover, and clean-normal q95 does
not guarantee separation from shifted normals. A high fallback rate means the
intervention rarely changes training. AUPRO superiority is not guaranteed.

Extra fitting work: three synthetic teacher forwards per step per resolution,
plus one clean teacher forward per source/resolution cached as a scalar only.
No extra trainable parameters (100,384 total), no inference overhead, no query
data/GT in selection. No dense teacher/feature cache across steps, only at most
16 scalar normal thresholds and scalar training diagnostics.

## Locked development comparison and stop

- H1 then K1; seed0, normal16, 448+672, 64 steps; same CLI/data/support policy.
- Seen TESTpublic development; full8 macro AUPRO@0.05 primary; pAUROC secondary;
  oracle F1 diagnostic. No sealed/generalization/significance claim.
- Local GPU0: can, fabric, fruit_jelly, rice, sheet_metal, walnuts.
  dsba3 physical GPU2: vial, wallplugs. Two workers per GPU, same device per pair.
- Budget: 16 class runs, at most 3600s/class, no sweep or automatic method revision.
- Stop on errors; a valid primary-metric loss means no performance improvement.
  Check support/query/GT/config/code/runtime parity before any method claim.
- Reuse existing containers and generic scripts/experiments.py run/collect.
  Update the existing comparison Markdown/CSV together when results complete.
- Maps and weights off. Keep remote logs; retrieve only compact quantitative
  results and minimal provenance. Do not change active checkouts or GPU exposure.
