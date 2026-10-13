# MR-JW Stage -1 report

Status: `MECHANISM_REFUTED` (and therefore `KILL_FOR_CLAIM`; no public
continuation).

## Execution and custody

The immutable GPU run was
`mr_jw_stage_minus1_dsba3_all4_20260824_v1` in fixed container
`hun_fsad_tta_012`, with `can:0`, `fabric:1`, `vial:2`, and `walnuts:3` run in
parallel. All four shards completed. The exact support-16 manifest SHA-256 was
`7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`;
all shards used the same DINOv2 state SHA-256
`43adbf57036cbb15f2dc957a3a3e8278fc6b31fefd1a75f487c3d7228f324bb3`.
The launched core and runner hashes match the retained local files exactly.

Only manifest-listed `train/good` images were opened. All shards record false
for TESTpublic, TESTprivate, ground-truth, directory-enumeration,
synthetic-fitting, and query-memory-update access. Dense maps were never
written; both remote and pulled run roots contain zero `anomaly_maps/`
directories. Compact evidence is retained at
`results/remote_runs/dsba3/mr_jw_stage_minus1_dsba3_all4_20260824_v1/`.

Local verification before launch comprised Python compilation, shell syntax,
the synthetic dry-run, and focused unit/runner/aggregate/remote tests. The
final focused suite is `12 passed`. Remote preflight verified GPUs 0--3 idle,
the fixed container, support hash, snapshot compilation/smoke, and offline
DINOv2 cache. Local aggregate-v2 replay reproduced every metric from the
remote rows.

## Primary result

Synthetic values are evaluation-only paired-change mask pAUROC@0.05; they are
not benchmark metrics.

| object | M0 MR-JW | independent scale NN | raw 672 RN | spatial shuffle | identity shuffle | M0-independent | M0-best rowwise shuffle | nuisance margin | clean LOO safe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| can | 0.796842 | 0.812797 | 0.800920 | 0.776626 | 0.725277 | -0.015955 | +0.003411 | -0.003097 | no |
| fabric | 0.897802 | 0.918319 | 0.891000 | 0.894530 | 0.900355 | -0.020517 | -0.012504 | -0.018442 | yes |
| vial | 0.713259 | 0.726552 | 0.725111 | 0.701442 | 0.695451 | -0.013294 | -0.007682 | -0.004368 | no |
| walnuts | 0.883050 | 0.895181 | 0.889739 | 0.872279 | 0.869641 | -0.012131 | -0.001854 | -0.006598 | no |
| macro | 0.822738 | 0.838212 | 0.826693 | 0.811219 | 0.797681 | **-0.015474** | **-0.004657** | **-0.008126** | 1/4 |

The immediate comparator is decisive: independent per-scale nearest-neighbour
matching beats MR-JW on all four objects and by `0.015474` macro. The
pre-registered rowwise best-shuffle contrast is also negative overall, and no
object has both primary directions positive. Only scratch has positive raw-672
retention; donor-patch and material-blotch retention are negative, though all
three remain inside their non-inferiority null envelopes.

The duplicate-scale fp32 control passes on every object: maximum observed
error is `8.64e-7` against the analytic `3.81e-6` bound. Every map is finite
and nonconstant. These checks show that the negative result is not a collapsed
implementation or scale-count artifact.

## Distance to the frozen gates

| gate | observed | required | distance | result |
|---|---:|---:|---:|---|
| M0 - independent | -0.015474 | > +0.012222 null q95 | -0.027696 | fail |
| M0 - rowwise best shuffle | -0.004657 | > +0.012222 null q95 | -0.016880 | fail |
| M0 - raw 672 retention | -0.003954 | >= -0.012222 | +0.008268 | pass |
| nuisance non-inferiority | -0.008126 | >= -0.013344 | +0.005217 | aggregate pass |
| positive object directions | 0 | >= 3 | -3 | fail |
| positive family retention | 1 | >= 2 | -1 | fail |

The aggregate nuisance non-inferiority check passes, but fabric fails its
object-level nuisance envelope (`-0.018442` versus allowed `-0.005266`), so the
apparent aggregate safety margin is not uniform.

Clean q95 itself is below the shuffled controls, but the image-level LOO tail
ratio exceeds the worse shuffle envelope on can, vial, and walnuts. This is a
pre-registered fatal stability failure. MR-JW also concentrates as much as
`84.8%` and `78.1%` of clean cells on one support image for can and walnuts,
respectively, consistent with the anticipated support/template identity
shortcut rather than a transferable anomaly witness.

## Aggregate label correction

The immutable remote aggregate-v1 retained under `aggregate/` labeled the run
`KILL_FOR_CLAIM` because it applied the clean-stability fatal before the direct
mechanism test. The preregistration says a non-positive M0-independent or
M0-shuffle margin is `MECHANISM_REFUTED`, with other failures labeled
`KILL_FOR_CLAIM`. Aggregate schema v2 corrects only that precedence and is
retained under `aggregate_v2/`; all rows, metrics, null gates, and pass/fail
facts are identical. Hard custody/finite/dtype validity remains clean, while
the separate clean-stability fatal remains recorded.

## Verdict and next direction

The shared multi-resolution witness hypothesis is directly refuted. Do not
tune the three resolutions, pooling grid, witness weights, synthetic families,
or null gates, and do not open TESTpublic for MR-JW. The independent
per-resolution NN mean is a stronger diagnostic baseline but is not a new
contribution and should not be promoted from synthetic support evidence alone.

Synthetic anomalies remain useful only as a sealed challenge battery: here
they successfully exposed that a plausible joint-consistency mechanism loses
to independent matching and support/template shuffles. A successor must add a
new information source or causal constraint rather than tightening witness
identity; it requires a new failure-basin-divergence preregistration.
