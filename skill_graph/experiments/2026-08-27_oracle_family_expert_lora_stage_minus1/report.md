# Oracle-Family-Expert LoRA Stage -1 report

Date: 2026-08-27  
Protocol: `oracle-family-expert-lora-stage-minus1-20260827`  
Status: `COMPLETED / SUPPORT-ONLY FOUR-GPU PANEL`  
Verdict: `INSUFFICIENT_ORACLE_FAMILY_SIGNAL / END_FAMILY_ROUTING_BRANCH`

## Outcome

Three independent oracle shift-family experts made the strongest LoRA signal
in this sequence less negative, but did not establish a protected mechanism.
The global candidate-minus-best-hard-null correction margin improved from
TARCD-S3's `-0.1439931` to `-0.0832106`, global shifted-normal q99 reduction
was `+0.0102168`, and strict synthetic contribution appeared for the first
time in `1/12` cells.  These are useful directional signals, not a passed
mechanism gate.

The anomaly-retention gate failed because the object medians for `can` and
`vial` were `0.7718805` and `0.7677122`, below the frozen cosine floor `0.85`.
Strict synthetic contribution remained sparse (`1/12`, below the progressive
minimum `3/12`).  The preregistered progressive and strong gates therefore
both failed.  Family-routed or hypernetwork LoRA is closed as the next
mechanism; no family-label, rank, learning-rate, threshold, loss-weight, seed,
or generator-strength sweep is licensed.

## Remote execution and custody

- Immutable run:
  `oracle_family_expert_lora_stage_minus1_dsba3_20260826T1617Z_v2`.
- Host/container: dsba3, fixed project container `hun_fsad_tta_012`.
- GPU mapping: `can/fabric/vial/walnuts -> 0/1/2/3`; J0/J1/J2 ran
  sequentially on each object's GPU.
- Remote preflight: `12 passed in 1.87s`; offline DINOv2 ViT-L/14 loading and
  an actual CUDA fused-QKV LoRA gradient/model smoke passed.
- Support manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Immutable snapshot SHA-256:
  `471c66ae69f92cc8738265db3f5f6d17655cd228e9982db1a9011552d6017828`.
- Aggregate SHA-256:
  `958b7f8c419056dd678ef013d5057c4e616a4d440148a93472df0c226e9d6fbd`.
- All `12/12` task statuses were complete, the summary index contained 12
  unique entries, `failed.txt` was absent, and no result process remained.
- Forbidden kinds and public/private/test/GT/query-memory authorization flags
  were empty.  No dense feature, score, or anomaly-map artifacts were kept.
- Only the fixed project container and this exact run path were inspected.
  No other user's container, process, file, volume, or job was queried or
  modified.

The first result-free `v1` preflight stopped before GPU training because the
container default Python did not expose pytest.  The wrapper was repaired to
use the project test-tools path, and only that failed run's own immutable
snapshot was removed before launching `v2`.

## Frozen design

- DINOv2 ViT-L/14 was frozen except rank-4, alpha-8 fused-QKV LoRA in
  zero-based blocks 8--11.
- One adapter was trained for each oracle family J0/J1/J2, using exactly
  `16 support bases x G0/G1/G4 = 48` updates per family and 144 candidate
  updates per object.
- S3 target-normalized vector Huber and S2 direct correction,
  cross-recomposition, identity, and RN terms were retained.
- S3's randomization namespace and filtered schedule order were reused.
- Traces at steps `1/12/24/36/48` were observational only; no checkpoint or
  early-stop selection occurred.
- Evaluation used four support-disjoint normal bases and held
  `G2_structural_scratch`; benchmark/public/private/test/GT access remained
  sealed.

This is an oracle upper-bound mechanism diagnostic, not a deployable method or
segmentation benchmark.  `benchmark_comparable=false`.

## Aggregate result

| measure | observed | progressive requirement | result |
|---|---:|---:|---|
| structural/custody | true | true | pass |
| global hard-null margin | `-0.0832106` | `>=-0.094` | pass |
| global q99 reduction | `+0.0102168` | `>=0` | pass |
| catastrophic objects | `0` | `0` | pass |
| anomaly-retention objects | `2/4` | `4/4` | fail |
| strict synthetic contribution | `1/12` | `>=3/12` | fail |
| RN contribution | `8/12` | diagnostic | positive partial |

Global medians were correction cosine `0.4341165`, correction norm ratio
`0.5475463`, anomaly cosine `0.8596385`, and anomaly norm ratio `0.9816918`.
The candidate had positive hard-null margin in `3/12` cells, passed the
per-cell anomaly guard in `7/12`, and had nonnegative q99 reduction in
`11/12`.

### Object medians

| object | correction cosine | correction norm | hard-null margin | q99 reduction | anomaly cosine | synthetic / RN |
|---|---:|---:|---:|---:|---:|---:|
| can | `0.3789534` | `0.5595827` | `-0.1263802` | `+0.0105300` | `0.7718805` | `0/3` / `2/3` |
| fabric | `0.4892796` | `0.5782015` | `-0.0473219` | `+0.0099036` | `0.9473966` | `0/3` / `2/3` |
| vial | `0.5214068` | `0.5355100` | `-0.1190993` | `+0.1729842` | `0.7677122` | `0/3` / `1/3` |
| walnuts | `0.2623255` | `0.3629060` | `+0.0105193` | `+0.0025205` | `0.9507888` | `1/3` / `3/3` |

### Family medians

| family | correction cosine | correction norm | hard-null margin | q99 reduction | anomaly cosine | synthetic / RN |
|---|---:|---:|---:|---:|---:|---:|
| J0 | `0.2972479` | `0.5475463` | `-0.0879558` | `+0.0018693` | `0.9160266` | `0/4` / `3/4` |
| J1 | `0.6615335` | `0.6353979` | `-0.0607713` | `+0.0660619` | `0.8420303` | `0/4` / `3/4` |
| J2 | `0.4341165` | `0.3090356` | `-0.0354785` | `+0.0210140` | `0.8632836` | `1/4` / `2/4` |

The strongest joint non-synthetic cell was `fabric/J1`: correction cosine
`0.6635219`, hard-null margin `+0.0048377`, q99 `+0.0099036`, and anomaly
cosine `0.9473966`.  The only strict synthetic-contribution cell was
`walnuts/J2`; it retained anomaly evidence (`0.9447995`) and improved q99
(`+0.0129435`), but its hard-null margin was still slightly negative
(`-0.0039985`) and correction norm was only `0.3629060`.

## Problem analysis

The adapter is not inactive.  Every candidate trace had a nonzero first
gradient and a nonzero trained adapter, and family factorization substantially
raised J1 correction direction.  The failure is instead an objective/geometry
conflict:

1. J1 learned the clearest shift correction, but its stronger correction was
   coupled to anomaly deletion on `can` and especially `vial`.
2. J2 most often approached the hard null and produced the sole synthetic
   contribution, but its correction magnitude collapsed to a family median
   norm ratio of `0.3090`.
3. The family-specific upper bound still beat the strongest static correction
   in only three cells.  Cross-family averaging was therefore a contributor,
   not the root cause.
4. Synthetic recomposition and RN regularization provide partial local
   gradients, but they do not guarantee that nuisance-removal updates leave
   the frozen anomaly-bearing feature component unchanged.

The next step must address gradient/representation interference inside the
backbone adaptation itself.  A score-only bypass would repeat the closed CEB
branch, and oracle family routing would violate this experiment's kill rule.

## Continuation boundary

One separately named, deployable support-only protocol may be considered only
if its sole change protects frozen anomaly evidence during actual QKV-LoRA
updates without oracle family labels or a score-only firewall.  It must retain
the same support custody, held G2 assay, hard nulls, one seed, and no-sweep
discipline.  Until that protocol is separately preregistered, no further GPU
run is authorized by this result.

No benchmark or public performance claim is licensed.
