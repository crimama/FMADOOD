# FACT-RN Stage -1 Report

Status: stopped at the preregistered `can` sentinel gate on 2026-08-24.  
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`.

## 1. Motivation

FACT-RN tested whether offline intervention learning could construct a
support-conditioned canonical feature space that removes normal nuisance while
preserving defect evidence. It was intentionally evaluated before any
TESTpublic experiment because prior support-only projection and RN-selection
branches repeatedly produced near-identity or null-absorbed behavior.

## 2. Implementable design

The frozen candidate uses DINOv2 ViT-L/14 descriptors, a permutation-invariant
support encoder, probabilistic part frames, two conditional coupling blocks,
an explicit content/nuisance split, and one exact nearest-neighbor distance in
canonical content space. The tested arms and training recipe are fixed in the
[preregistration](preregistration.md).

## 3. Evaluation alignment

- Dataset: MVTec AD2 single-image.
- Sentinel object: `can`.
- Target support: exact fixed SuperAD-16 manifest, SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Offline learning: 56 `train/good` images from the other seven objects.
- Evaluation: 16 held target normal images and sealed local/broad synthetic
  defects.
- Custody: zero TESTpublic, TESTprivate, label, or ground-truth entries; no
  query update, memory update, score-time parameter update, or dense maps.
- Comparability: mechanism diagnostic only; no SuperAD performance claim.

## 4. Code and immutable evidence

The valid sentinel summary was produced by the current FACT-RN core and runner:

- core SHA-256: `4324a8e3c5bb840402a94f375e35f0e38f55e9796ae718b3b067141e2dfa4296`
- runner SHA-256: `01be318f3af65c5c3a0d170bd1445b095bd7a88146675d9b41cb6ff8d84b5249`
- checkpoint SHA-256: `88cb6b5acfa5d31686cc1d7a6937e5d8557c45aea5196c0def4c797c24be0d56`
- summary SHA-256: `31263d2c4d1c623b3377e130abd3d74afac2972c713bf66a0a922c97c8ace5ca`

Evidence path:
`results/fact_rn_stage_minus1_20260817_v2/local_can/eval/can/summary.json`.

## 5. Added-code verification

- `python3 -m py_compile` passed for the FACT-RN core, runner, and aggregator.
- `bash -n scripts/run_fact_rn_stage_minus1_remote.sh` passed.
- `.venv/bin/python -m pytest -q` over the six focused FACT-RN test modules:
  `37 passed in 2.23s`.
- The fail-closed normalizer accepted the sentinel shard, including exact arm
  coverage, finite/nonconstant maps, path custody, and immutable provenance.

## 6. Remote execution decision

No remote shard was launched. The preregistration says to stop if a shuffled
null matches `M0`; the valid local sentinel already triggered that fatal gate.
Launching the remaining objects would spend compute after the branch was
closed and would not authorize public evaluation.

## 7. Results and analysis

Object-level means on `can`:

| arm | normal q99 | nuisance correction | neighbor retention | defect retention |
| --- | ---: | ---: | ---: | ---: |
| `B0_raw_RN` | 0.609397 | 0.000000 | 1.000000 | 1.000000 |
| `M0_FACT` | 0.605645 | 0.017426 | 0.993248 | 0.984061 |
| `N0_intervention_shuffle` | 0.608519 | 0.010909 | 0.993868 | 0.978383 |
| `N1_part_frame_shuffle` | 0.605646 | 0.017426 | 0.993248 | 0.984053 |
| `N2_no_noncontraction` | 0.610170 | 0.011222 | 0.994714 | 0.985609 |
| `N3_support_PCA_projection` | 0.613533 | -0.060660 | 0.910533 | 0.912542 |

The q95/q99 reductions relative to raw RN were only `0.4611%/0.6156%`, far
below the strict `10%` and soft `5%` bars. Nuisance correction was `1.7426%`,
far below `30%`. Most importantly, `M0-N1` nuisance-correction margin was only
`3.13e-7`; their largest per-row q99 difference was `8.05e-6`, and all other
reported behavior was likewise effectively identical. This is not an
aggregation artifact: the implementation changes the support part order for
`N1`, while exact global 1NN scoring leaves the learned result insensitive to
that intervention. The learned part/frame mechanism therefore did not provide
the claimed causal contribution.

## 8. Conclusion

Strict claim: `KILL_FOR_CLAIM`.  
Continuation: `NO_CONTINUE`.

The sentinel retained defects and raw-neighbor structure, but did so while
remaining nearly identity-like and being matched by the preregistered
part/frame shuffle. The four-object aggregate is intentionally absent because
the earlier fatal gate closed this exact FACT-RN formulation. TESTpublic,
TESTprivate, and GT remain unauthorized.

The next research cycle should not tune frame count, coupling depth, or loss
weights on this formulation. A successor would need a separately named
mechanism whose conditioning changes a pre-registered causal intervention
metric before any anomaly benchmark is opened.

## 9. Storage cleanup

`find results/fact_rn_stage_minus1_20260817_v2 -type d -name anomaly_maps`
returned no paths. Compact checkpoint, hashes, logs, and JSON evidence were
preserved; no remote cleanup was needed because no remote run was launched.
