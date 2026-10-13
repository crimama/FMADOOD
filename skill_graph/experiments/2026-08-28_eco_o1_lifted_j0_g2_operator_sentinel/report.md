# ECO-O1 lifted-J0/G2 operator sentinel report

Date: 2026-08-28

Protocol: `eco-o1-lifted-j0-g2-operator-sentinel-20260828-v1`

Parent: `eco_g2_nativecore_dsba3_20260827T212434Z_g2v5`

Final attempt: `eco_o1_lifted_j0_g2_dsba3_20260827T232207Z_o1v3`

Terminal: `INVALID_FAIL_CLOSED / END_ECO_FAMILY`

## Outcome

The single ECO-G2-authorized operator-sentinel generation is closed without a
scientific operator result. Three separately named invocations all stopped
before any O1 support seal, held shard, repeat shard, operator record, feature
cell, anomaly map, or detector metric was emitted. The final invocation
reached the frozen parent-artifact replay and exposed a structural custody
incompatibility: the exact valid G2 N0 artifact carries its parent snapshot
identity, while the reused frozen G2 validator requires that identity to equal
the current O1 child snapshot.

Attempt 003 was prospectively declared the final process attempt. Therefore no
fourth repair/run is allowed. The sentinel did not pass, so the detector bridge
and fixed three-object metric checkpoint are not authorized. No further ECO
geometry, activity, operator, feature-only, or support-only rescue is allowed.
LoRA remains closed.

This terminal must not be interpreted as `VALID_BUT_NO_GAIN` or as evidence
that the lifted operator lacks anomaly-detection utility. No scientific O1 or
query-level result exists. It is an invalid-execution family closure under the
frozen fail-closed and portfolio stop rules.

## Final-attempt evidence

- Remote scope: the fixed dsba3 project container
  `16c4c25074fff40040aed82b552a8cd77fbdfcf52bb553b0a318f7faeef5def6`
  and authorized GPUs 0--3 only.
- Frozen preregistration SHA256:
  `303137f0ea381bc0673f8e481d758c20f8fbb849e63fb45c29acb106e63cfd0c`.
- Frozen static-contract SHA256:
  `e587ee880deb930a7eaf6962b5a78b5d41a33bd21cee2de78b637863c8612474`.
- Frozen post-G2 policy SHA256:
  `95c0a83b4aeee1a7701661ab9c78c52f91ae4fd85ddf4e20ef9aaad6f6a32239`.
- Attempt-003 source-manifest SHA256:
  `258322d1c42afbf4c7bbebf7c41e6e10025d0c02a8774297a65879e20a950f09`.
- Attempt-003 implementation-amendment SHA256:
  `2c3c8db2d8a11927276d0335ed25c2b5286df11f49f01131e894240e68f597eb`.
- Remote-controller SHA256:
  `90e07a9cb1a3efbee592e9ca54b6a6dccdaac5bee4e16a92f6bffacc332a04a3`.
- Attempt-003 snapshot archive SHA256:
  `eedd3839924339ecf453e2ac4a29637c615be737e7cc19924b36208aa00eeb73`.
- Remote P0: `58/58` tests passed. Its evidence is explicitly result-free with
  `data_opened=0` and `model_opened=0`; SHA256
  `7f566daab53015ee3a12f93f1a8372a9a6dcc44d84b47cdc88f846fc88b08af5`.
- P1: two CPU synthetic dry-run artifacts were byte-identical.
- Imported parent N0 SHA256:
  `df8404a51aef2bb6148bb75d95b548b6debbf23c35d517a8da91016774b0932e`,
  exactly matching the completed G2 artifact.
- Pulled invalid artifact SHA256:
  `c332e2717731db11a028ae1bf2ce58e9c92ac05ea18775f8bdef9cc94ca4c8e5`.
- Local compact result:
  `results/remote_runs/dsba3/eco_o1_lifted_j0_g2_dsba3_20260827T232207Z_o1v3`.

The terminal artifact records
`INVALID_FAIL_CLOSED/END_ECO_FAMILY`, `structural_valid=false`,
`benchmark_comparable=false`, `lora_authorized=false`, and
`next_authorized_stage=NONE_INVALID_FAIL_CLOSED`.

## Direct root cause

The final stack stopped in the frozen G2 support-calibration path:

`O1 _support_replay -> G2 _support_calibration -> G2 _load_passed_n0`.

The parent N0 artifact is valid and retains G2 custody
`snapshot_sha256=65b887c016edcb37769a871c0244f5ddab2f12f987e3a81b4971ea1a834d7241`.
However, G2 `_load_passed_n0` requires
`value["custody"] == _common_custody(inputs)`, and `_common_custody()` obtains
`snapshot_sha256` from the current snapshot's `snapshot_hash.txt`. Inside the
final O1 snapshot that value is
`eedd3839924339ecf453e2ac4a29637c615be737e7cc19924b36208aa00eeb73`.
The two snapshot identities are intentionally different, so the full-dict
equality rejects the exact parent artifact with
`CustodyError: valid passed N0 artifact required`.

The failure occurs before support image decoding, frozen-model opening, or
support-seal emission. Continuing would require changing the parent/child
custody semantics and resealing another implementation. That path is excluded
by the independently fixed final-attempt stop and by the user's prohibition on
another support-only continuation after this sentinel.

## Complete attempt ledger

| Attempt | Last valid gate | Failure | Result/disposition |
| --- | --- | --- | --- |
| `...T230208Z_o1v1` | P1 byte identity | Controller pre-created the fail-closed support output leaf | Pre-result invalid; artifact SHA256 `a2efb320ef4c9d9a982cfba9bd509699b93a7c8138048074efe9bd42275b543c`; snapshot retained |
| `...T231246Z_o1v2` | P1 byte identity | Duplicate `output_root` keyword in `argparse.Namespace` | Pre-result invalid; artifact SHA256 `c4d57d4832b027e9452d6f80dfb76e5cab85b9498b1dff6221bcba700d70c6ea`; snapshot retained |
| `...T232207Z_o1v3` | Exact parent N0 hash present | Parent G2 snapshot custody cannot equal current O1 snapshot custody | Final pre-result invalid; artifact SHA256 `c332e2717731db11a028ae1bf2ce58e9c92ac05ea18775f8bdef9cc94ca4c8e5`; snapshot retained; ECO closed |

The immutable detailed ledger is in `attempts.md`. Failed snapshots remain
remote because the contract permits cleanup only for a valid completed
snapshot after verified pull.

## Scientific and access boundary

- No new data/model-derived O1 metric was exposed in any attempt.
- No O1 support calibration, held observation, repeat, or 16-cell decision was
  completed.
- No TESTpublic/private/ground-truth/query item was opened.
- No anomaly score or pixel map was defined or persisted.
- No `seg_AUROC_0.05`, `seg_F1`, `AU-PRO@0.05`, map coverage, normal FPR,
  runtime/memory checkpoint, object delta, repeat, or confidence interval was
  computed.
- No SuperAD-16, RN-FMLK, identity, or mechanism-matched hard-null detector
  comparison occurred.
- No training, optimizer, checkpoint, adapter, or LoRA path was opened.

The copied `c0/n0.json` retains the historical access ledger of its parent G2
run; it is not evidence that attempt 003 reopened those 20 images. The final
attempt failed while validating the copied artifact, before the replay decoded
support images or opened the model.

## Final branch decision

1. Close the existing native ECO line: `END_ECO_FAMILY`.
2. Do not implement or run the ECO detector bridge.
3. Do not run the weak/strong/control metric checkpoint or all-eight ECO
   evaluation.
4. Do not add another support-only diagnostic, operator repair, custody repair,
   threshold change, or LoRA rescue to this family.
5. Preserve all three success/failure attempts in the ledger; do not delete
   their invalid snapshots.

Any later anomaly-detector or backbone-adaptation study must start as a new,
separately named method generation with its detector and metric path defined
prospectively. It cannot inherit an ECO pass claim or be represented as a
continuation of this exhausted sentinel.
