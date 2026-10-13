# Patch-Coverage RN TESTpublic Report

Date: 2026-08-15
Verdict: `NO_SIGNAL_DIAGNOSTIC`

## Motivation

Keep exact RN fixed and test whether spatially stratified source selection
preserve normal coverage better than image-level representative selection.

## Implementable Design

The object schema is `patch-coverage-rn-object-v1` and the aggregate schema is
`patch-coverage-rn-aggregate-v1`. The aggregate requires five arms:
`B0_full_RN`, `P25_cell_kcenter_RN`, `P50_cell_kcenter_RN`,
`P75_cell_kcenter_RN`, and `N50_cell_random_RN`.

Each shard must expose a 16x16 cell-count selection panel, selection custody
hashes, complete TESTpublic coverage, evaluator-native metric keys, and
explicit no-TTA/no-update flags.

## Evaluation Alignment

The run is a fullshot public-shadow diagnostic over all eight MVTec AD2 public
objects. Same-run arm comparisons are valid. Historical SuperAD16 remains
context only because the reference budget differs.

## Code Modification / Creation

- Added `scripts/aggregate_patch_coverage_rn_testpublic.py`.
- Added `scripts/run_patch_coverage_rn_testpublic_remote.sh`.
- Added focused aggregate and remote-wrapper tests.
- Added this experiment record.

## Added Code Evaluation

Local contract verification passed:

- focused and adjacent pytest: 37 tests passed;
- compile: aggregate and runner passed `py_compile`;
- shell syntax: remote wrapper passed `bash -n`;
- ruff: aggregate, runner, and focused tests passed.

## Remote Execution

Immutable run `patch_coverage_rn_testpublic_20260815_v4_detached` completed on dsba3
GPU3 (`can -> fabric -> walnuts`) and dsba5 GPU3 (`fruit_jelly -> rice ->
sheet_metal -> vial -> wallplugs`). All eight compact shards have complete
TESTpublic coverage and no failure marker. The controller was detached with
`setsid` (`PPID=1`, independent SID/PGID), so closing the client session did
not terminate the replacement run.

## Results and Analysis

| Arm | pAUROC | F1 | AU-PRO | Pixel AP | Memory reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B0_full_RN` | **0.749400** | 0.336069 | 0.435772 | **0.279366** | 1.00x |
| `N50_cell_random_RN` | 0.749247 | 0.334359 | **0.438802** | 0.276926 | 1.99x |
| `P50_cell_kcenter_RN` | 0.749121 | **0.336345** | 0.432074 | 0.278353 | 1.99x |
| `P75_cell_kcenter_RN` | 0.749053 | 0.335862 | 0.434989 | 0.278825 | 1.33x |
| `P25_cell_kcenter_RN` | 0.746212 | 0.334652 | 0.430656 | 0.274882 | 3.97x |

Against full RN, P50 changes pAUROC/F1/AU-PRO by
`-0.000279/+0.000276/-0.003698`. Against the equal-budget random N50 null,
P50 changes those metrics by `-0.000126/+0.001986/-0.006728`. Therefore the
spatial k-center selector does not provide a consistent performance advantage;
the retained performance is explained by memory budget rather than learned
representativeness. Random N50 is the best efficiency arm by pAUROC and AU-PRO,
but full RN remains the absolute pAUROC and Pixel-AP winner.

## Conclusion

Do not promote spatial k-center support selection as a new method. For maximum
absolute performance, retain the simplest full-support exact RN. If memory or
latency becomes binding, a deterministic 50% random support subset is a nearly
lossless engineering option, not a method contribution. The result further
supports keeping future work focused on improving the comparison/readout rule
without discarding normal memory coverage.
