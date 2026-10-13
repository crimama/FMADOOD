# Simple-RN Representative + DVT TESTpublic Report

Date: 2026-08-15
Verdict: `NO_SIGNAL_DIAGNOSTIC`

## Motivation

Test the simplest viable direction after RN-based methods outperformed more
complex gated or tangent variants: keep exact RN fixed, then vary only the
normal support bank and support-only DVT.

## Implementable Design

The canonical runner is `scripts/run_simple_rn_testpublic.py` with protocol
`simple-rn-representative-dvt-testpublic-20260815` and schema
`simple-rn-testpublic-object-v1`.  The aggregate expects nine arms:
`B0_full_RN`, `R16_repr_RN`, `R32_repr_RN`, `R64_repr_RN`, `R128_repr_RN`,
`N64_random_RN`, `DVT_full_RN`, `DVT_R64_repr_RN`, and
`N_DVT_R64_position_shuffle_RN`.

## Evaluation Alignment

All arms share the pinned full support manifest and full TESTpublic object
population.  Full support arms are diagnostic-only relative to historical
SuperAD16.  Same-run comparisons among RN arms, representative subsets, random
control, DVT, and position-shuffle null are valid.

## Code Modification / Creation

- Extended `scripts/aggregate_simple_rn_testpublic.py` to validate the final
  nine-arm schema, pinned full manifest, nested representative support sets,
  no dense maps, and DVT position-shuffle contrast.
- Added `scripts/run_simple_rn_representative_dvt_testpublic_remote.sh` as a
  two-host remote controller using the canonical runner and aggregate.
- Added focused tests for aggregate custody/deltas and wrapper mode/path
  contracts.

## Added Code Evaluation

The integrated contract passed 33 focused and adjacent tests, scoped ruff,
compile, shell syntax, and diff checks.  The immutable remote benchmark is
running as `simple_rn_representative_dvt_testpublic_20260815_rn_dvt`.

## Remote Execution

Complete.  Canonical shard ownership was dsba3 `can,fabric,walnuts` and dsba5
`fruit_jelly,rice,sheet_metal,vial,wallplugs`, with host GPU3 mapped to
container `cuda:0`.

## Results and Analysis

All-eight macro results rank `B0_full_RN` first at pAUROC `0.749400`, F1
`0.336069`, and AUPRO `0.435772`.  `R128_repr_RN` is the closest compressed
arm at `0.748028 / 0.334363 / 0.436637`, only `-0.001372` pAUROC from the full
bank.  At 64 sources, deterministic image-level k-center (`0.745542`) does
not beat the random control (`0.745809`).

DVT is consistently harmful at the macro level.  `DVT_full_RN` loses
`0.016819` pAUROC and `0.068303` AUPRO versus full RN.  `DVT_R64_repr_RN`
loses `0.012599` pAUROC versus the matching raw R64 bank and also loses
`0.006572` versus its position-shuffle null.  Therefore neither learned DVT
content nor its spatial alignment is supported by this diagnostic.

## Conclusion

Keep exact final-layer RN as the main method.  A 128-image representative bank
is a near-lossless efficiency option, but image-level k-center has no mechanism
advantage over random selection at equal size.  The next experiment therefore
moves selection down to spatial patch cells, where coverage can be optimized
without changing the successful RN comparison rule.  DVT is not promoted.
