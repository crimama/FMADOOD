# ECO-O1 complete attempt ledger

## Attempt 001 — pre-result controller failure

- Run: `eco_o1_lifted_j0_g2_dsba3_20260827T230208Z_o1v1`
- Host/container/GPU scope: frozen dsba3 controller, container
  `16c4c25074fff40040aed82b552a8cd77fbdfcf52bb553b0a318f7faeef5def6`,
  GPUs `0,1,2,3` only.
- Source manifest SHA256:
  `9cb65af091e13959fa15bf50f1ea1dab1164e954a3ef8a20918257fad3b87431`.
- Implementation amendment SHA256:
  `3e6178f2fae5111801cff6d64b73f3be4768ec01b5220dff365d1e180194f8b5`.
- Remote controller SHA256:
  `b8992ec78c2fa4db19d3835abcb1bfb40e87eb7a77d3f0821e82399a5e9833d8`.
- Snapshot archive SHA256:
  `1f849f263d47a179e53e824fcd27c3543d0c512a1caf02ce46bf79b246fdba61`.
- P0: remote frozen-snapshot tests `57/57` passed; evidence explicitly records
  `result_bearing=false`, `data_opened=0`, and `model_opened=0`.
- P1: two synthetic CPU dry runs were byte-identical. No data/model access and
  no scientific metric exposure occurred.
- Failure: `support_replay` pre-created its leaf output directory, while the
  runner's fail-closed `_output_root` requires that leaf not to exist. The
  runner therefore stopped before loading data/model or emitting
  `support/support_seal.json`.
- Result-bearing artifacts: no support seal, held shard, repeat shard, operator
  record, feature cell, or detector/benchmark output exists.
- Mechanical terminal:
  `INVALID_FAIL_CLOSED/END_ECO_FAMILY`; retained without reinterpretation or
  overwrite. The invalid snapshot is intentionally retained remotely.
- Pulled invalid artifact SHA256:
  `a2efb320ef4c9d9a982cfba9bd509699b93a7c8138048074efe9bd42275b543c`.
- Local immutable pull:
  `results/remote_runs/dsba3/eco_o1_lifted_j0_g2_dsba3_20260827T230208Z_o1v1`.
- Prospective classification: the preregistered
  `pre-result instrumentation/runtime failure` exception applies because no
  data/model-derived scientific metric was emitted. A controller-only repair
  may use a fresh immutable run name and new source/amendment hashes; the
  preregistration, static contract, operator, arms, thresholds, population,
  and decision rule remain byte-identical.

## Attempt 002 — pre-result runner argument failure

- Run: `eco_o1_lifted_j0_g2_dsba3_20260827T231246Z_o1v2`.
- Source manifest SHA256:
  `8ee98bac4e535a489b1020a307f40baa0b5f8e6ed81e2b7902a5d442f045e9e5`.
- Implementation amendment SHA256:
  `55deae5eccf411ac05b5575a98259b17bf58c1ec3adc5a3352a41b98e2f0b545`.
- Remote controller SHA256:
  `6290150aff4d2123ca7c0146764549f6237491f6f6c91a3e73aa52dd280f3210`.
- Snapshot archive SHA256:
  `860971a3081da7e1b314fd8b98d676d95caee99fd980f713a135ac867ae3edd6`.
- P0: remote frozen-snapshot tests `57/57` passed; evidence records
  `result_bearing=false`, `data_opened=0`, and `model_opened=0`.
- P1: the two CPU-only dry runs remained byte-identical and result-free.
- Failure: `_support_replay` attempted
  `argparse.Namespace(**vars(args), output_root=old)`. Because `vars(args)`
  already contains `output_root`, Python raised `TypeError` before calling the
  frozen G2 support calibration function.
- Result-bearing artifacts: no support seal, held shard, repeat shard, operator
  record, feature cell, or detector/benchmark output exists.
- Mechanical terminal:
  `INVALID_FAIL_CLOSED/END_ECO_FAMILY`; retained without overwrite. The
  invalid snapshot remains remote and is not eligible for cleanup.
- Pulled invalid artifact SHA256:
  `c4d57d4832b027e9452d6f80dfb76e5cab85b9498b1dff6221bcba700d70c6ea`.
- Local immutable pull:
  `results/remote_runs/dsba3/eco_o1_lifted_j0_g2_dsba3_20260827T231246Z_o1v2`.
- Independent contract decision: a final separately sealed repair attempt is
  permitted because both failed invocations stopped before any scientific
  exposure. This does not authorize a second sentinel. Attempt 003 is the final
  process attempt; any further pre-result failure, result-bearing invalidity,
  or scientific fail closes ECO without another repair.

## Attempt 003 — final pre-result parent-custody incompatibility

- Run: `eco_o1_lifted_j0_g2_dsba3_20260827T232207Z_o1v3`.
- Source manifest SHA256:
  `258322d1c42afbf4c7bbebf7c41e6e10025d0c02a8774297a65879e20a950f09`.
- Implementation amendment SHA256:
  `2c3c8db2d8a11927276d0335ed25c2b5286df11f49f01131e894240e68f597eb`.
- Remote controller SHA256:
  `90e07a9cb1a3efbee592e9ca54b6a6dccdaac5bee4e16a92f6bffacc332a04a3`.
- Snapshot archive SHA256:
  `eedd3839924339ecf453e2ac4a29637c615be737e7cc19924b36208aa00eeb73`.
- P0: remote frozen-snapshot tests `58/58` passed; the compact evidence is
  result-free and records `data_opened=0` and `model_opened=0`.
- P1: two CPU-only synthetic dry runs were byte-identical.
- C0 imported the exact passed parent N0 bytes, SHA256
  `df8404a51aef2bb6148bb75d95b548b6debbf23c35d517a8da91016774b0932e`,
  then stopped in the frozen G2 `_load_passed_n0` check with
  `CustodyError: valid passed N0 artifact required` before support image
  decoding, model opening, or support-seal output.
- Direct cause: the parent N0 custody embeds the valid G2 snapshot SHA256
  `65b887c016edcb37769a871c0244f5ddab2f12f987e3a81b4971ea1a834d7241`,
  while the frozen G2 `_common_custody()` derives `snapshot_sha256` from the
  current O1 snapshot marker, here
  `eedd3839924339ecf453e2ac4a29637c615be737e7cc19924b36208aa00eeb73`.
  `_load_passed_n0` requires full custody-dictionary equality, so a valid
  parent artifact cannot satisfy that check inside a separately sealed child
  snapshot even though its artifact hash is exact.
- Result-bearing artifacts: no support seal, held shard, repeat shard,
  operator record, feature cell, detector map, or benchmark metric exists.
- Mechanical terminal:
  `INVALID_FAIL_CLOSED/END_ECO_FAMILY`; detector bridge and metric checkpoint
  are not authorized. This is the predeclared final process attempt, so the
  custody contract is not repaired and no fourth invocation is allowed.
- Pulled invalid artifact SHA256:
  `c332e2717731db11a028ae1bf2ce58e9c92ac05ea18775f8bdef9cc94ca4c8e5`.
- Local immutable pull:
  `results/remote_runs/dsba3/eco_o1_lifted_j0_g2_dsba3_20260827T232207Z_o1v3`.
- The invalid snapshot is intentionally retained remotely and is not eligible
  for cleanup.

## Final family decision

All three invocations failed before an O1 scientific result was produced.
Attempts 001 and 002 exposed bounded execution defects; attempt 003 exposed a
parent/child custody model that cannot admit the exact parent N0 artifact in a
newly sealed snapshot without changing implementation semantics. Under the
predeclared final-attempt stop and the post-G2 no-more-support-only policy, the
terminal research decision is `INVALID_FAIL_CLOSED / END_ECO_FAMILY`.

This is not `VALID_BUT_NO_GAIN`, not a negative detector result, and not a
benchmark comparison. The detector bridge, three-object checkpoint, all-eight
evaluation, and LoRA remain unopened. Any future work must be a separately
named non-ECO generation; it cannot be presented as an ECO rescue or a retry
of this single sentinel.
