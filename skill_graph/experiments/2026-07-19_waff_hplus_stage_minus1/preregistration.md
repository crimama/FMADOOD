# WAFF-H+ Cycle 3 Preregistration

Date: 2026-07-19
Status: `DESIGN_FROZEN / NOT_IMPLEMENTED / NO_PUBLIC_ACCESS`

## Method boundary

WAFF-H+ changes the default H+ FlowTTE training/scoring objective, not its
backbone, DVT, memory, query path, or anomaly-map postprocessing.  DINOv3-H+/16
layers `[7,15,23,31]`, `layer_norm_mean`, DVT position mean, exact SuperAD-16,
the MLP flow, and the single static inference pass remain fixed.  TTA, query
update/transform, routing, map switching, and score-map fusion are forbidden.

After the clean-support flow fit and before any test enumeration, aligned
clean/shift/defect/shifted-defect cells are scored by the exact H+ components:
LOO latent 1-NN distance and NLL penalty.  Clean cells must exclude their own
memory source (k=2 or source-excluded bank); all other cells use k=1.  A single
factorial scorer replaces the current fixed component weighting at inference.

Only low-frequency cut-paste, chromatic blotch, and local Gaussian blur are
allowed.  High-frequency ribbon/scratch constructs are forbidden.

## Objective

The base clean-support H+ NLL remains.  The added terms are a defect-interior
versus top-5%-background ranking loss, the paired interaction
`((D-C)-(SD-S))^2`, and two fixed worst-case constraints:

- the maximum held-axis shifted-defect ranking drop may not exceed zero;
- the maximum held-axis shifted-normal q=0.995 tail excess may not exceed zero.

The constraints use a preregistered augmented-Lagrangian update; public metrics
cannot select weights, seeds, epochs, or generators.

## Matched controls

- byte-identical disabled default H+;
- same-cell nonfactorial ERM;
- mean-factorial without the worst-axis constraint;
- WAFF-H+;
- mask shuffle, label shuffle, and low-frequency RGB-energy hard nulls.

## Sequential gate

L0 verifies disabled byte parity, cross-source clean LOO, aligned shapes,
nonzero constraint gradients, static memory, and zero test access.  L1 is a
`can/rice` runtime/non-collapse smoke without selection credit.  L2 is an
all-eight, two-seed, nested LODF×LOSF support-only assay.

Public access requires both seeds to pass every condition: WAFF-H+ over
nonfactorial `+0.005` pAUROC and `+0.010` fixed F1; worst-axis shifted-defect
drop no worse than nonfactorial with 97.5% UCB at most `0.001`; shifted-normal
tail reduction at least 30%; hard-null margins `+0.005/+0.010`; every family
raw-noninferior; object harm at most `0.005/0.010`; and identical directional
decisions across seeds.  Failure means `KILL_FOR_CLAIM / NO_PUBLIC_ACCESS`
with no reweighting, seed swap, generator swap, or repeat.

Only after L2 and an all-16 seed-0 replay pass may a one-shot all-eight public
comparison run against same-run default H+, SuperAD-16, and RN-FMLK.  Until
then, no default-outperformance claim is permitted.
