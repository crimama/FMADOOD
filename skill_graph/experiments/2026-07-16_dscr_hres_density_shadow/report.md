# DSCR High-Resolution Density Shadow Diagnostic

Date: 2026-07-16

Verdict: **KILL_FOR_CLAIM / NO_CONTINUE**

## Negative Evidence Intake

The killed hard-local correspondence branch was not reopened. The earlier
`can` DARC-v1 cell showed that local matching, registration, and reconstruction
all lost heavily to unrestricted high-resolution G0. This cell tested only the
remaining non-retread question: whether a normal-only learned density score can
improve the same unrestricted high-resolution global evidence.

## Motivation

The current structural diagnosis identified a training/scoring mismatch under
distribution shift. A learned normal density head scored by its own negative
log-likelihood was the last unexecuted metric-alignment candidate. Raw global
cosine 1-NN G0 was retained as the direct hard null.

## Implementable Design

- MVTec AD2 single-image public shadow diagnostic.
- Objects: `can`, `fabric`, `rice`, `vial`, `wallplugs`.
- Seed 0, fold 0, P16-random support: 12 memory normals and four disjoint held-out
  normal calibration images per object.
- First four deterministic good and first four deterministic bad queries.
- Frozen DINOv3 H+ layer-7 unrestricted high-resolution global token pool.
- Identical-support arms: G0 raw cosine residual, D0 learned density NLL, and
  D0c normal-only empirical upper-tail calibration of D0.
- Density budget: two coupling layers, three epochs, LR `2e-4`, and a 32,768
  token training cap.

## Evaluation Alignment

All arms were evaluated on identical native-grid query IDs. The evaluator
checked finite, nonconstant maps and exact G0/D0/D0c ID parity, then reported
pooled pAUROC@0.05, AP, oracle F1, component recall, and a four-way
image-disjoint normal-only fixed threshold at quantile `0.9999`.

This five-object, eight-query, one-seed, one-fold public result is mechanism
evidence only. It is not comparable to an all-eight SuperAD-16 or RN-FMLK claim
row.

## Code Modification / Creation

No method implementation was changed. Execution used the same retained remote
snapshot on dsba3 and dsba5:

```text
run_flow_tte_darc_density_cell.py  6d650a37d95e43bc152c091f10b48fda1e3f1bbd2a30101e9eacf0f2962df0cc
hres_density.py                    f141c5de18d254a11585068e11374eb81070d4fb9862a469451e69c39ffa573b
darc_ad2_density_runtime.py        9db576210531dda16b6c28a4292e2e396463e91aac0c83c20c06edac2f8aa5ba
```

The compact evaluator was added locally, unit-tested, and copied byte-for-byte
to both servers (`be42b07c...c46360`). The method snapshot is deleted in the
current local worktree but retained remotely and in Git `HEAD`; this provenance
limitation independently blocks a method claim.

## Added Code Evaluation

- Local compilation and focused test:
  `python3 -m py_compile ... && python3 -m pytest -q tests/test_dscr_density_compact_eval.py`
  -> `1 passed`.
- Remote evaluator `--help` preflight passed on dsba3 and dsba5.
- A paired two-query execution smoke passed on dsba3 `can` and dsba5 `vial`;
  each produced one good and one bad map for all three arms.
- `ruff` was unavailable both as a command and Python module, so lint was not
  run.

## Remote Execution

dsba3 GPUs 0--3 ran `can`, `fabric`, `rice`, and `wallplugs`; dsba5 GPU 0 ran
`vial`. Every compact cell completed with eight queries, 16 support paths, one
fold, an operational shard count of one, and the same runner hash and density
settings. Completion manifests and controller logs are preserved under
`remote/`.

The initially attempted full-population jobs were stopped and removed before
any object completed or any metric was produced because unrestricted G0 made
them claim-scale multi-hour runs. The compact 4+4 population and gate were
sealed before its metrics were computed.

## SuperAD Baseline and Unified Metrics

A historical separate-repository same-evaluator context row reports RN-FMLK
`0.747608/0.331328` and SuperAD-16 `0.765802/0.385534` pAUROC/F1. Those runs use
different reference sets and all eight objects, so they are context only and
are not subtracted from this diagnostic. The only valid paired comparison here
is D0/D0c versus G0 within each row.

## Results and Analysis

| object | G0 pAUROC/F1 | D0 pAUROC/F1 | D0c pAUROC/F1 | D0 Δp/ΔF1 |
|---|---:|---:|---:|---:|
| can | .521668/.000142 | .488418/.000106 | .487710/.000095 | -.033250/-.000036 |
| fabric | .527093/.216515 | .514495/.242294 | .514226/.235145 | -.012597/+.025779 |
| rice | .600696/.020434 | .650914/.042351 | .648759/.041008 | +.050217/+.021917 |
| vial | .673175/.156019 | .551273/.052433 | .554489/.053888 | -.121901/-.103586 |
| wallplugs | .506876/.007731 | .494154/.004443 | .494109/.004391 | -.012722/-.003287 |
| **macro** | **.565902/.080168** | **.539851/.068325** | **.539858/.066906** | **-.026051/-.011843** |

D0 improved pAUROC on only `rice` (1/5) and oracle F1 on `fabric` and `rice`
(2/5). D0c had the same win counts. Both density arms lowered macro pAUROC,
oracle F1, AP, and fixed F1. D0c was essentially tied with D0 on macro pAUROC
and worse on macro oracle F1, so tail calibration did not repair the ranking.

The macro fixed normal FPR decreased from G0 `2.1246e-4` to D0 `1.5373e-4`,
but fixed F1 also decreased from `.009808` to `.008520`. On `rice`, where
density improved anomaly ranking, normal FPR rose from `9.98e-5` to `2.49e-4`.
The apparent FPR gain is therefore not a stable shift solution; it mostly
reflects more conservative thresholds on objects whose recall collapsed.

## Continuation Assessment

The pre-registered `PROMISING_DIAGNOSTIC` gate fails every substantive
condition:

1. Neither density arm beats G0 on macro pAUROC and oracle F1.
2. Win counts are below 3/5 on both metrics.
3. `can` loses more than 0.03 pAUROC, while `vial` loses more than 0.03
   pAUROC and more than 0.05 oracle F1.
4. Map validity and coverage pass, so the negative result cannot be dismissed
   as missing or corrupt output.

`CONTINUE_DIAGNOSTIC` also fails because the density arms do not improve either
primary metric on at least three objects and violate the no-harm condition.
The optional whitening/Mahalanobis follow-up is therefore not launched.

## Conclusion

The high-resolution signal is real only as an unrestricted raw-distance
candidate; replacing its score with a learned normal density does not provide
distribution-shift robustness. Density NLL absorbs object- and acquisition-
specific normal variation, giving one positive object (`rice`) but severe
negative transfer on `vial` and a class-harm violation on `can`. Normal-only
tail calibration changes score scale, not the damaging order statistics.

The learned global density head is removed from the proposed core. The earlier
hard-local branch remains closed. A future solution must explicitly separate
nuisance shift from defect evidence and demonstrate cross-object no-harm;
another density-capacity or calibration sweep is not justified by this cell.

## Post-Conclusion Storage Cleanup

After all metrics and manifests were written, every compact and smoke
`anomaly_maps/` directory was deleted on dsba3 and dsba5. Verification returned
zero remaining map directories. Compact metrics, manifests, completion files,
and controller logs were pulled locally under `remote/`; no dense maps are
retained.
