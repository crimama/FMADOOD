# WAFF-H+ Stage -1 Report

Date: 2026-07-19  
Verdict: `REDESIGN_INFEASIBLE_CONSTRAINT_SURFACE / NO_PUBLIC_ACCESS`

WAFF-H+ tested whether the default H+ flow could be rescued under distribution
shift by changing only the positive fixed-sum weighting of latent hard-1NN
distance and frozen-standardized NLL. The valid support-only v6 run used exact
SuperAD-16 train/good paths, DINOv3-H+/16 layers `[7,15,23,31]`, static memory,
and four can/rice object-seed cases. It did not enumerate TESTpublic/private,
perform TTA, update/query-transform memory, or compute public metrics.

All four shards passed runtime validity: the score remained finite and
nonconstant, weights summed to `1.25`, static memory was unchanged, and the
integrated scorer matched direct component scoring within `9.54e-7`. WAFF did
reduce the in-cell worst-axis margin-drop relative to the default, by
`-0.024348` on can seed 0 and only `-0.004101` to `-0.004453` on the other
three cases.

The decisive result was the fixed-sum 101-point simplex audit. No case had a
weight satisfying shifted-normal tail and shifted-defect retention jointly.
Minimum joint violations were `0.030534` (can/0), `0.033264` (can/1),
`0.308887` (rice/0), and `0.063389` (rice/1). Thus the failure is not primarily
an optimizer or step-count failure: the two-component surface produced by the
clean-only flow has no sampled feasible point under the frozen constraints.

The strongest supported conclusion is limited to this mechanism: post-flow
component reweighting can slightly rotate the score direction, but it cannot
repair the distribution-shift tradeoff because nuisance tail inflation and
defect-margin degradation are already entangled in the latent geometry. L2 was
not launched and public access remains forbidden.

The next candidate is therefore representation-level rather than another
weight search: Geometry-Bounded Counterfactual Margin Flow (GB-CMF) retains the
exact default static inference score while applying C/S/D/SD factorial losses
and augmented constraints to the flow coupling parameters. Its Stage -1A is a
separate four-case, five-arm support-only falsification experiment.

Compact evidence:
`results/remote_runs/dsba3/waff_hplus_stage_minus1_20260719_l1_v6/summary.json`.

