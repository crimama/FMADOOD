# BASTA-v2-NC Stage 02 local numerical-contract report

Status: `LOCAL_NUMERICAL_CONTRACT_PASS / NO_REMOTE`  
Claim ceiling: numerical custody only  
Decision-bearing artifact: `attempt-02/result.json`

## Sequential history

The first same-directory result was provisional and was rejected by the
independent verifier. It sampled rather than exhaustively checked the oracle,
did not implement all decision gates, used an incomplete fixture 11, and did
not establish pre-execution custody or independent-process replay. That result
is superseded and contributes no PASS evidence.

Attempt 02 repaired only the Stage-02 numerical-control implementation. The
BASTA path, `rho=0.8`, `delta=2^-12`, scientific nulls, encoder, support
protocol, and all remote/data boundaries were unchanged.

## Frozen custody

- Manifest SHA-256:
  `5af510aa22ab574308ec5811e9f382079059e2e542fb3f8313d28411ec5fd67c`
- Fixture-file SHA-256:
  `cb3b9e902652dc5b3b0f8c53b3f49f2af34348151d7c197c0493936f5e6f56ca`
- Fixture-bank SHA-256:
  `e5fa3f54d17e9eeac6a9912fdfad0ce291ffdc8b3d6af7b084b376be0309e726`
- Intermediate SHA-256, both replays:
  `0c6de8bfd57a9644c31d745b6f739f959a3f9367dffb1c8ac983f7404a95b5dd`
- Compact replay SHA-256, both replays:
  `85e4fa9ee1bb3ab57f8e0dc62e3f7bb62f7df99fa74481baa653ae529be453a2`
- `manifest.sha256` and `result.sha256` independently returned `OK`.

The manifest was written before the evidence-bearing run and freezes the
numerical module, runner, tests, full ATAC source, isolated
`_patch_tokens_linear` source, Stage-02 method/evaluator, fixture file,
environment, constants, and operation order.

## Result

| Gate | Evidence | Verdict |
| --- | ---: | --- |
| Exact fixture coverage | 48 tensors, 112,896 coordinates | PASS |
| Path RN32 parity | 903,168 intermediate checks | PASS |
| Subtraction RN32 parity | 338,688 operation checks | PASS |
| Literal branch box | every stored branch in `[0,1]` | PASS |
| Branch half-ULP | max error / half-ULP = `1.0` | PASS |
| Stored-path midpoint | `2.9802322387695312e-08 <= 5.960464477539063e-08` | PASS |
| Existing patch identity | bitwise equal to isolated existing function | PASS |
| Subtraction budget | max error = max `B_num` = `2.9802322387695312e-08` | PASS |
| Fixed mutation | first difference `q, fixture 3, H0 W0 C0` | PASS |
| Gradual underflow | signed min-subnormal multiply/divide | PASS |
| Independent replay | two fresh Python subprocesses | PASS |
| Replay identity | intermediate and compact hashes identical | PASS |

The exact stored-input residual reached
`2.9802322387695312e-08`, while the executed fp32 residual was `0`. This is a
valid boundary case: the stored-branch residual was cancelled by the specified
fp32 sum/subtraction, and its error exactly equals `B_num`. It must not be
interpreted as proof that the exact residual or model curvature is zero.

## Verification

- Focused test suite: `14 passed`.
- Targeted Ruff check: passed.
- Objective result evaluator: `0.91`, `PASS / conditional-keep`.
- Independent numerical verifier: `PASS`; all nine prior rejection defects
  repaired, a separate temporary-directory prepare/run reproduced the result,
  and 23,400 signed RN32 stress cases produced no mismatch.
- Remaining nonfatal hardening: manifest semantic strings are protected by
  local hashes rather than an external trusted signature; therefore no claim
  may exceed local numerical custody.

## Boundary and next step

No dataset, DINO, support, SuperAD, RN-FMLK, public/private/ground truth,
PyTorch, CUDA, GPU, or dsba3 access occurred. This result makes no anomaly,
performance, generalization, or novelty claim.

The only authorized continuation is a new Stage-03 design that changes one
thing: complete the scientific-null specification. It does not authorize
Stage-03 implementation or remote execution by itself. dsba3 remains locked
until that design receives independent objective and corresponding-author
approval.
