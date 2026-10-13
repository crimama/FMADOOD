# PSAR-D1 detector checkpoint report

Status: `INVALID_FAIL_CLOSED / CLOSE PSAR-D1`

The frozen contract is in `preregistration.md`, `static_contract.json`, and
`output_schema.json`. The final sealed generation crossed the first readable
TESTpublic-map boundary but did not complete the deterministic repeat or the
24-package map barrier. No GT-backed scientific metric was evaluated.

## Execution ledger

- Preflight attempt 1 stopped before model/data execution because the isolated
  snapshot target was created at the run root instead of `<run>/repo`.  The
  exact empty attempt paths were removed; no TESTpublic map or metric existed.
- Preflight attempt 2 reached remote preflight `PASS`, then the local controller
  hit an unbound `RETURN`-trap variable.  The exact sealed pre-smoke attempt was
  removed; no TESTpublic map or metric existed.
- Smoke attempt 1 passed the bootstrap smoke but failed closed in the CUDA
  smoke because flat DINO patch tokens were passed to a grid-map helper.  The
  cell smoke and TESTpublic phases did not run.  The exact failed attempt was
  removed after preserving the failure facts.
- Source generation
  `182af88b3248359ca1b8b77f29782dfecba00a0fe1166393406322082a9624b4`
  passed static preflight, remote preflight, bootstrap smoke, CUDA smoke, and
  the deterministic one-cell smoke.  Its sealed snapshot archive was
  `ddc745b8c1a2e1463a780d5ba62e73f7a47be3220be6101ed8eadf3f107e18a2`.
- That generation completed all four learned-head checkpoints for `fabric`,
  `vial`, and `walnuts` (12/12 total), but produced exactly zero `.npy` maps,
  zero evaluator `.tiff` maps, and zero `map_inventory.json` files.  GT and the
  metric evaluator therefore remained closed.
- Root cause: `equal_image_q95()` evaluated the equal-image ECDF by rescanning
  every calibration pixel for every unique float32 score.  With 16 native
  maps this is `O(U*N)` and approaches quadratic work because `U` is close to
  `N`; the same implementation would recur in canonical-baseline calibration.
  This is an implementation-complexity defect, not a method or threshold
  change.
- The pre-map run was terminated fail-closed on 2026-08-28 after the stall was
  independently reproduced and reviewed.  Final marker: `launch_failed_rc=143`.
  Only the three exact run-name/source-hash processes were signalled.  GPU
  0/1/2 subsequently read `1 MiB, 0%`, and the experiment lock was absent.
- Because no readable TESTpublic map or metric existed, the preregistered
  post-map `revision_budget=0` boundary was not crossed.  The next generation
  may contain only an exact-semantics q95 complexity repair plus its regression
  tests and source reseal; the preregistration, static contract, output schema,
  arms, data, thresholds, seeds, and evaluator remain unchanged.
- The authorized repair replaced the `O(U*N)` equal-image q95 rescan with an
  exact-semantics implementation. The only resealed source entries were
  `src/flow_tte/psar_d1.py`
  (`126595a72678805c4c42cdc4f13d74767d6f2a22c06ce64b8552ef36cbe6c025`)
  and `tests/test_psar_d1.py`
  (`c12e7cb906b9c3b9392c633f63787abc1b8b8166d3984c65aa6b409deec443f1`).
  Focused verification passed `101` tests; the 16-map 224x224 q95 benchmark
  averaged `3.479 ms` over 20 calls. This is operational evidence only.
- Final source generation
  `1c9314da3e13412613f2c9c56f880e0d5c0d6257f9fee2632506fd9956a703e0`
  preserved preregistration
  `d146fc2cf6584c828673c89208015c2ffa49efdda877d5339c021b8f968ff0fa`,
  static contract
  `9e5e8108b0a864ba2fb10e0d842a4923fba2a71e20c74675598402cacf0d84a4`,
  and output schema
  `9503c7c33b89c611d7bd92d8f7b4885b430560546dcbef256fbd80138b347740`.
  Remote preflight sealed all `81` source files, and bootstrap, assigned-GPU
  CUDA, and deterministic cell smokes passed.
- The sole post-map run was
  `psar_d1_dsba3_q95_reseal_20260829_v1` in the pinned dsba3 container image
  `sha256:8dbd129dc64fbe1bbce27212902b27e2ecd3fea01794565c80f5ef415369b2f1`
  on host GPUs 0--2. Canonical `fabric`, `vial`, and `walnuts` packages each
  completed all four learned heads and their committed map inventories:
  respectively `624/936`, `560/840`, and `600/900` raw `.npy`/evaluator
  `.tiff` maps. This proves the q95 repair removed the prior operational stall;
  these intermediate packages are not a valid scientific result.
- `fabric_repeat` completed all four heads and its held calibration audit, then
  stopped after `495` raw maps. `np.save` could not allocate another
  `20,054,016` bytes because `/workspace` was full; the subsequent
  `checkpoint_summary.json` write also failed with `Errno 28`. The repeat
  summary is therefore zero bytes, no repeat map inventory exists, no evaluator
  phase ran, and the terminal marker is `launch_failed_rc=74`.
- Compact evidence was pulled to
  `results/remote_runs/dsba3/psar_d1_dsba3_q95_reseal_20260829_v1/`.
  The final 26-file local/remote manifest is
  `c3eb233287e3d24d5c97246c3a22f112f6bbdbf02ba0d4a8314ffb0400484764`.
  After hash verification, only the exact run's `4,971` dense map/head files
  and isolated snapshot were removed. The retained cleanup receipt has SHA-256
  `870d42c7814133fda5a0f8d3875ecdf97dd42369df0992a36ff13829d1a4ed49`;
  both local and remote trees contain zero dense files and zero map directories.

## Results

No valid scientific result. The complete 24-package barrier failed before GT
could be opened, so there is no aggregate `seg_AUROC_0.05`, `seg_F1`,
`AU-PRO@0.05`, delta, bootstrap, or method-comparison claim to report. Canonical
object outputs are retained only as execution/custody evidence and must not be
mined as a partial PSAR-D1 result.

## Verdict and next branch

The missing deterministic-repeat inventory is a repeat/count/map-coverage
error, so ordered gate 1 applies before every shortcut, harm, resource, and gain
gate: `INVALID_FAIL_CLOSED`.

Because readable TESTpublic maps had already been written,
`revision_budget=0` forbids a storage-capacity repair and rerun. Close PSAR-D1;
do not open the all-eight confirmation or LoRA branch. The q95 change is retained
as a verified operational repair, but PSAR itself remains scientifically
unevaluated rather than positive or negative.
