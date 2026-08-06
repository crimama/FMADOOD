# DSCR High-Resolution Density Shadow Diagnostic

Date sealed: 2026-07-16, before full-object metrics were produced
Status: resolved; `KILL_FOR_CLAIM / NO_CONTINUE` (see `report.md`)
Protocol: `fmad-experiment-protocol`, early-stage continuation lane

## Negative Evidence Intake

The killed DARC-v1 hard-local branch is not reopened. In the prior `can` pilot,
local matching, detached registration, and reconstruction all lost heavily to
the unrestricted high-resolution global arm G0. This experiment therefore
excludes coordinate-local windows, registration, reconstruction, score fusion,
RGB guidance, morphology, and threshold tuning.

The only tested question is whether a normal-only density objective scored by
the same negative log-likelihood can improve the unrestricted high-resolution
global feature evidence. Raw global cosine 1-NN remains the direct hard null.

## Motivation

The current method has a training/scoring mismatch, while the prior resolution
gate found a positive high-resolution signal. A learned density head was
specified but never executed. This cell tests that missing mechanism without
re-entering the failed spatial-counterfactual family.

## Implementable Design

- Target: MVTec AD2 single-image public shadow diagnostic.
- Frozen reduced objects: `can`, `fabric`, `rice`, `vial`, `wallplugs`.
- Resource: P16-random, PCG64 seed 0, fold 0 only; 12 memory normals and four
  held-out normal calibration images per object.
- Fine features: frozen DINOv3 H+ layer 7, native 512 crops and unrestricted
  global support token pool inherited from the sealed high-resolution design.
- Arms on identical support, fold, query, and feature geometry:
  - `G0`: raw global cosine 1-NN residual.
  - `D0`: learned normal-density negative log-likelihood.
  - `D0c`: normal-only empirical upper-tail calibration of `D0`.
- Frozen density budget: two coupling layers, three epochs, learning rate
  `2e-4`, deterministic seed 0, maximum 32,768 memory tokens.
- Compact population: the first four deterministic good and first four
  deterministic bad queries per object, evaluated after a two-query execution
  smoke. A full-population launch was superseded before any object completed or
  any metric was computed because unrestricted G0 scoring made it a multi-hour
  claim-scale run; the compact cell is explicitly mechanism-preflight only.
- Servers: dsba3 GPUs 0--3 for `can/fabric/rice/wallplugs`; dsba5 GPU 0 for
  `vial`. Method, data, support, and scoring settings are identical across hosts.

## Evaluation Alignment

Primary diagnostic metrics are pixel `seg_AUROC_0.05`, AP, and oracle F1 on the
same native grid. Normal-only fixed F1 is reported only when the common
cross-fitted evaluator has enough good images. Coverage is complete only when
every logical query has one finite map for each of G0, D0, and D0c.

The five-object, eight-query-per-object, one-seed, one-fold public result is not
comparable to an all-eight SuperAD-16 or RN-FMLK method-claim row. Existing
baseline numbers are context only. Promotion beyond a diagnostic requires the
same 16 reference paths, full populations, all eight objects, the same
evaluator, and an untouched confirmation set.

## Pre-registered Decision Gate

The compact cell cannot earn `KEEP` or `CONDITIONAL KEEP`.
`PROMISING_DIAGNOSTIC` requires all of:

1. D0 or D0c beats G0 on macro pAUROC@0.05 and oracle F1 point estimates.
2. The winning density arm improves at least three of five objects on both
   metrics.
3. No object loses more than 0.03 pAUROC or 0.05 oracle F1 versus G0.
4. Maps are finite, nonconstant, and complete for every query and arm.

`CONTINUE_DIAGNOSTIC` is allowed only if the density arm improves at least one
metric on at least three objects without a catastrophic no-harm violation, and
the next step is exactly one matched whitening/Mahalanobis hard-null cell.

Otherwise the verdict is `KILL_FOR_CLAIM / NO_CONTINUE`; the learned density
head is removed from the proposed core and the failed spatial branch remains
closed.

## Execution Provenance

Both remote servers passed CLI/data/model-cache preflight and contained the
same retained execution snapshot:

```text
run_flow_tte_darc_density_cell.py  6d650a37d95e43bc152c091f10b48fda1e3f1bbd2a30101e9eacf0f2962df0cc
hres_density.py                    f141c5de18d254a11585068e11374eb81070d4fb9862a469451e69c39ffa573b
darc_ad2_density_runtime.py        9db576210531dda16b6c28a4292e2e396463e91aac0c83c20c06edac2f8aa5ba
evaluate_dscr_density_compact.py   be42b07c880ee688048a0ec2f07f1a8cb1c68417c104c34f904ff00dd0c46360
```

Those files are deleted in the current local worktree but retained in the
remote experiment snapshot and in Git `HEAD`; the evaluator is a new local
file copied byte-for-byte to both servers. The retained method snapshot blocks
a method claim and must be disclosed in the final report; the deleted local
stack is not restored for this diagnostic.
