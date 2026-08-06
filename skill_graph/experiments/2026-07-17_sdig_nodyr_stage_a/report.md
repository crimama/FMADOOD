# SDIG-NODyR Stage A: MVTec AD2 TESTpublic Shadow Report

Date: 2026-07-18  
Status: `COMPLETE / TESTPUBLIC_SHADOW_ONLY / NEGATIVE_OR_MIXED_CAUSAL_SIGNAL`

## 1. Outcome

The raw NODyR probe was executed on all eight MVTec AD2 objects with the exact
recorded SuperAD-selected 16 normal references, frozen DINOv2 ViT-L/14 layers
15 and 23, ridge `0.01`, and no calibration, fusion, query update, density,
foreground prior, RGB guide, or morphology.

NODyR reached macro pixel pAUROC@0.05/F1 `0.729019/0.326280`. The best single
intrinsic null at macro level was `state_angle` at
`0.728246/0.325678`, leaving only `+0.000773/+0.000602`. This is far below the
pre-registered `+0.005/+0.005` shadow-signal threshold. More importantly, the
per-object independent-null envelope beat NODyR on both metrics for every one
of the eight objects.

The result therefore does not support transition direction as a distinct
anomaly mechanism. The maximum defensible verdict is a negative/mixed public
shadow diagnostic; it is not a claim-tier keep/kill decision because no
independent development/private population, source-orbit bootstrap, or exact
same-run baseline suite was available.

## 2. Negative Evidence Intake

This experiment was motivated by repeated regressions from correcting the same
score field with position centering, hard-local matching, density, reference
consensus, and condition-aware offsets. The testable alternative was that
normal acquisition changes might move the representation state while
preserving its cross-layer transition direction, whereas defects might bend
that direction locally.

The experiment was deliberately limited to raw relation evidence. A failed raw
probe could not be rescued by calibration or fusion, and discarded prior
branches were not reintroduced.

## 3. Frozen Design and Controls

For each patch,

```text
x = LN(F15)
d = unit(LN(F23) - x)
d_hat = unit(Wx + b)
NODyR = 1 - cosine(d_hat, d)
```

The per-object dense ridge operator was fitted only on all patches from the
fixed 16 normal support images. The independent controls were:

- `state_angle`: predict the normalized layer-23 state, then compare derived
  predicted and observed transition directions;
- `state_l2`: normalized state prediction residual;
- `raw_delta_l2`: unnormalized depth-delta prediction residual;
- `shuffle_nodyr`: identical direction model after cyclic cross-image pairing
  and a SHA256-seeded toroidal spatial offset;
- `unit_delta_l2_equiv`: algebraic audit only, excluded from the envelope.

## 4. Implementation and Execution

The standalone core and runner are in `src/flow_tte/nodyr.py` and
`scripts/run_sdig_nodyr_ad2.py`. The four-GPU controller is
`scripts/run_sdig_nodyr_stage_a_remote.sh`; aggregation is performed by
`scripts/aggregate_sdig_nodyr_stage_a.py`.

The first full attempt exposed an execution-only storage problem: six
full-resolution float32 TIFF variants were being retained concurrently and
grew to 35 GB before an object completed. The partial attempt was terminated
before aggregation and removed. The scorer, support set, layers, ridge, and
metrics were unchanged. The final run cached compact patch scores, materialized
one float32 TIFF variant at a time, evaluated it, and immediately removed its
dense maps. A second execution refinement removed an unregistered duplicate
condition-wise pixel evaluation; the pre-registered condition upper-tail
statistics remained intact.

The valid run used dsba3 host GPUs `0,1,2,3` with allocation:

- GPU 0: `can`, `fruit_jelly`
- GPU 1: `fabric`, `sheet_metal`
- GPU 2: `rice`, `wallplugs`
- GPU 3: `vial`, `walnuts`

## 5. Unified Results

| variant | pAUROC@0.05 | F1 | NODyR delta pAUROC | NODyR delta F1 |
|---|---:|---:|---:|---:|
| NODyR | 0.729019 | 0.326280 | - | - |
| state_angle | 0.728246 | 0.325678 | +0.000773 | +0.000602 |
| state_l2 | 0.726109 | 0.313992 | +0.002910 | +0.012287 |
| raw_delta_l2 | 0.726168 | 0.313545 | +0.002851 | +0.012735 |
| shuffle_nodyr | 0.572248 | 0.078820 | +0.156770 | +0.247460 |

The shuffled control confirms that correct cross-layer pairing carries useful
signal. It does not identify direction scoring as the source of that signal,
because the matched `state_angle` control is essentially tied.

| object | NODyR pAUROC | envelope pAUROC | margin | NODyR F1 | envelope F1 | margin |
|---|---:|---:|---:|---:|---:|---:|
| can | 0.533995 | 0.535126 | -0.001130 | 0.000756 | 0.000997 | -0.000241 |
| fabric | 0.700953 | 0.703178 | -0.002224 | 0.270401 | 0.272036 | -0.001634 |
| fruit_jelly | 0.783811 | 0.784249 | -0.000438 | 0.427675 | 0.436833 | -0.009158 |
| rice | 0.903176 | 0.903741 | -0.000566 | 0.601197 | 0.605234 | -0.004037 |
| sheet_metal | 0.809758 | 0.809955 | -0.000197 | 0.343642 | 0.344588 | -0.000946 |
| vial | 0.639432 | 0.645386 | -0.005954 | 0.278314 | 0.278400 | -0.000085 |
| wallplugs | 0.575652 | 0.588785 | -0.013133 | 0.023604 | 0.025881 | -0.002277 |
| walnuts | 0.885374 | 0.886163 | -0.000789 | 0.664650 | 0.669832 | -0.005182 |

The apparently positive macro margin is an aggregation artifact: it compares
NODyR with the best single null after macro averaging, while each object can
have a different strongest null. Under the object-wise envelope required for a
mechanism claim, NODyR loses on all eight objects.

## 6. Distribution-Shift Analysis

The image-level p95 score tails show that direction normalization did not make
normal samples acquisition-invariant.

| condition | objects | good p95 | bad p95 | bad-good separation |
|---|---:|---:|---:|---:|
| regular | 8 | 0.133087 | 0.147345 | 0.014258 |
| overexposed | 8 | 0.139673 | 0.152909 | 0.013236 |
| underexposed | 8 | 0.141571 | 0.155162 | 0.013591 |
| shift_1 | 8 | 0.162501 | 0.175191 | 0.012690 |
| shift_2 | 7 | 0.165181 | 0.185086 | 0.019905 |
| shift_3 | 7 | 0.168161 | 0.184019 | 0.015858 |

Relative to regular samples, shifted-good p95 increased by `+0.029414` for
`shift_1`, `+0.032094` for `shift_2`, and `+0.035074` for `shift_3`. Both good
and bad distributions move upward together. Thus the acquisition nuisance is
not removed; it acts on the transition-angle law itself rather than only on
the starting state or transition magnitude.

This is consistent with two additional diagnostics:

- NODyR's per-image mean/p95 statistics correlate with `state_angle` at
  `0.999887/0.999636`, showing that the candidate is practically a
  reparameterization of derived state prediction.
- NODyR's patch score has only `-0.057594` mean correlation with raw transition
  magnitude, so magnitude normalization worked as intended. The remaining
  shift sensitivity therefore cannot be explained simply by transition norm.

The localization collapse is object-specific. For `can`, bad-good p95
separation is negative for regular, overexposed, and all three shift cells
except underexposure. For `wallplugs`, it is negative for regular,
overexposed, underexposed, `shift_1`, and `shift_3`. These reversed score
orderings explain their near-zero F1 (`0.000756` and `0.023604`) and show why a
single monotone calibration could not repair the raw evidence.

## 7. Context Baselines and Claim Boundary

Recorded context rows are shown only to locate the result; they are not strict
same-run comparisons.

| context | pAUROC@0.05 | F1 | NODyR delta pAUROC | NODyR delta F1 |
|---|---:|---:|---:|---:|
| SuperAD-16 | 0.765802 | 0.385534 | -0.036783 | -0.059254 |
| RN-FMLK | 0.747608 | 0.331328 | -0.018589 | -0.005048 |
| H+ reference | 0.836739 | 0.527427 | -0.107720 | -0.201147 |

Strict promotion remains unavailable because same-run SuperAD/RN-FMLK/H+
artifacts, source-orbit bootstrap, sealed independent development data, and a
private confirmation evaluator were absent. TESTpublic results must not be
used to tune the layer pair, ridge, calibration, or fusion.

## 8. Verification and Artifacts

- population: 379 good + 705 bad = 1,084 images;
- six validated variants per image = 6,504 finite nonconstant maps;
- exact support manifest SHA256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`;
- valid runner SHA256:
  `fae139234016cea7a5d46dbba10957663cfb9c855ac2e6fa9cf532101498190b`;
- maximum `unit_delta_l2_equiv - 2*NODyR` absolute error:
  `5.364418e-07`;
- focused local verification: `20 passed`, Python compilation, shell syntax,
  and `git diff --check` passed;
- eight object summaries and eight run manifests present;
- no traceback, runtime error, CUDA OOM, or disk-full log entry;
- retained `anomaly_maps` and `patch_score_cache` directories: zero.

Compact valid artifacts are stored at
`results/remote_runs/dsba3/sdig_nodyr_stage_a_all8_20260717_v2`.

## 9. Conclusion

The retry successfully answers the intended methodological question. Correct
cross-layer pairing matters, but transition direction is not independently
better than state prediction. Acquisition shift changes the angular transition
statistics of normal samples, and the candidate inherits nearly the same score
field as `state_angle`. The method therefore does not solve the original
identifiability problem; it relocates it into a different parameterization.

No calibration or fusion continuation is justified from this shadow result.
Any new attempt must introduce evidence that is not a deterministic
reparameterization of the same two-layer state relation and must freeze its
choices on an independent normal-shift development population before returning
to TESTpublic/private evaluation.
