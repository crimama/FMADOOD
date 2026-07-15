# FlowTTE Image-Disjoint Support-Consensus S0

Date: 2026-07-11  
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`  
Disposition: result retained; implementation, tests, launchers, preregistration,
and local/remote run artifacts discarded.

## Scope

- MVTec AD2 `test_public`, reduced gate:
  `can,fabric,fruit_jelly,rice`.
- Exact Phase-3 H+ DVT MLP anchor with fixed 16 supports.
- Candidate: image-disjoint per-support q25/IQR residual.
- Controls: Phase-3 base, q25-only, shuffled support-image IDs.
- No SuperADD rerun and no learned decoder.

## Results

Values are `seg_AUROC_0.05 / seg_F1`.

| Object | Base | Residual | Residual delta | q25 | Shuffled-ID |
|---|---:|---:|---:|---:|---:|
| can | 0.569858 / 0.000710 | 0.569833 / 0.000709 | -0.000025 / -0.000001 | 0.568172 / 0.000664 | 0.569912 / 0.000709 |
| fabric | 0.968368 / 0.697949 | 0.968440 / 0.698101 | +0.000072 / +0.000152 | 0.965078 / 0.690147 | 0.968411 / 0.698043 |
| fruit_jelly | 0.779994 / 0.476761 | 0.779994 / 0.476760 | -0.000000 / -0.000001 | 0.765285 / 0.450023 | 0.779995 / 0.476760 |
| rice | 0.944240 / 0.712533 | 0.944269 / 0.712517 | +0.000029 / -0.000016 | 0.944816 / 0.710569 | 0.944272 / 0.712518 |
| **Macro** | **0.815615 / 0.471988** | **0.815634 / 0.472022** | **+0.000019 / +0.000033** | **0.810838 / 0.462850** | **0.815648 / 0.472008** |

## Decision Evidence

- Macro F1 gain `+0.000033` was far below the frozen `+0.005` gate.
- `can` regressed by `-0.000025` AUROC and approximately `-0.000001` F1.
- Shuffled IDs matched the candidate at macro level and exceeded it on `can`
  AUROC by `0.000079`.
- q25-only reduced macro AUROC by `0.004777` and F1 by `0.009138`.
- Phase-3 base reproduction error was `7.58e-7`; grouped-min versus flat-bank
  distance error was `0.0`; finite/nonconstant map coverage was `1.0`.

## Conclusion

The tested support-image consensus statistic did not provide an independent
anomaly signal for the current fused latent representation. It does not justify
a learned decoder, an all-eight expansion, or further tuning. The result only
supports redirecting future work toward upstream layer/spatial correspondence;
that causal explanation was not directly proven by this experiment.

The original run had a qualified audit trail because dense maps were removed
before root aggregation/pull and the method hash was recovered post-run. This
does not change the conservative hard-kill decision, but the run is not claimed
as fully preregistration-compliant.
