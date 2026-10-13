# SO-CFM Synthetic-Guard Stage -1 Log

## 2026-08-11

- Added an ops contract for local/dsba3/dsba5 execution using only fixed containers and explicit single-GPU visibility.
- Aggregation freezes support-only gates; any positive result is capped at `PROMISING_DIAGNOSTIC`.
- `test_public_accessed=false`, `test_private_accessed=false`, dense map persistence disabled by contract.
- Implemented the conditional velocity field, deterministic synthetic defects, leave-one-support-fold-out
  runner, discrete-orbit and hard-null arms, compact progress logging, and frozen aggregator.
- Focused verification passed: 15 tests; Python compilation, shell syntax, fatal Ruff selectors, and
  `git diff --check` were clean at launch.
- Remote v1 failed before opening metrics because PyTorch 2.1 lacked the newer PyTree registration name.
  Added a compatibility alias and removed the failed isolated snapshots.
- Remote v2 failed before opening metrics because the local torchvision NMS schema shim duplicated the
  native remote operator. Disabled that shim only in the remote launch environment and removed the
  failed isolated snapshots.
- Final immutable v3 allocation: local GPU0 ran `can,fabric`; dsba3 host GPU3, exposed to the process
  as `cuda:0`, ran `vial`; dsba5 host GPU3, fixed-container device 0, ran `walnuts`.
- All final shards used runner SHA-256 `42a76d876bd4f58c9959455ddcab9db807452c112bb527248c582d32cd999af3`,
  core SHA-256 `18b2814c5c395b0066a233bfe642c61e6f5ee96a0d4226f5386b8ff2b951f471`, and the frozen
  support manifest SHA-256 `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Final verdict: `KILL_FOR_CLAIM_CONTINUE_DIAGNOSTIC`. A3 macro coverage/tail reduction were
  `0.520364/-0.003347`; A1 was `0.732499/0.012623`. A3 beat A1 tail reduction on `0/4`, passed
  the `0.20` tail gate on `0/4`, and passed mask-shuffle retention separation on `0/4`.
- Synthetic anomaly guard did not establish benefit: A3 retention was not worse than A2 on only `1/4`,
  while its terminal guard loss was zero in all 16 object-fold fits. The current hinge therefore
  often never activated; this is a formulation failure, not evidence that every anomaly-aware objective fails.
- All final result bundles were pulled locally. The dsba3/dsba5 v3 isolated code snapshots were removed;
  no feature caches, transformed images, masks, anomaly maps, score maps, or dense maps were persisted.
