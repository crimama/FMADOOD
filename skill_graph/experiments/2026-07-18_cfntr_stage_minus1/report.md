# CF-NTR Stage -1 support-only identifiability report

## Verdict

`KILL_FOR_CLAIM / NO_CONTINUE`

The physical tangent is strongly more specific than generic low rank, and it
preserves most synthetic defect evidence, but it is not stable enough in the
tail and removes far too little held-out normal drift. No TESTpublic/private
image or label was read. The result does not authorize the planned
full-population shadow.

## Valid run

- Run: `cfntr_stage_minus1_20260718_supportonly_v3`
- dsba3 allocation: GPUs 0–3, two frozen objects per GPU
- Snapshot SHA-256:
  `e368cc1cc2945c54f6dc39a3699c6d500a1eee3b4612f8cef8b02ec47f40d7f0`
- Exact support manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`
- Compact results:
  `results/remote_runs/dsba3/cfntr_stage_minus1_20260718_supportonly_v3`
- Invalid v1/v2 attempts were not pooled. Their failure and amendment trail is
  recorded in `attempts.md`.

## Gate results

| Gate | Result | Main evidence | Interpretation |
|---|---:|---|---|
| G1 physical specificity | PASS | physical capture 0.711801; clean-PCA 0.202415; hard-null margin 0.509387; LCB 0.501043 | the sealed photometric directions are identifiable beyond generic rank-4 structure |
| G2 basis stability | FAIL | global median 0.984643, but minimum object p05 0.732572 < 0.75 | the bulk is stable, but bootstrap tail instability remains |
| G3 held-out drift | FAIL | macro reduction 0.262992 < 0.65; LCB 0.256818 < 0.50; every object < 0.40 | the identified direction does not remove enough spatial drift |
| G4 defect retention | PASS | energy retention 0.930158; pAUROC/F1 deltas -0.000170/-0.000534 | synthetic anomaly evidence is mostly preserved, with a small consistent loss |
| G5 audit | PASS | all 136,192 maps finite and nonconstant; exact row/arm parity; no public access | the v3 measurement is valid |

The G3 margin over the stronger clean-PCA/random null is positive (0.151677),
so the physical basis is not useless. It fails because absolute removal is
weak, not because it is indistinguishable from a generic low-rank basis.

## Failure localization

### Depth is the main stability fault line

The G2 object failures are `fabric` (0.732572), `wallplugs` (0.739426), and
`walnuts` (0.733074). Instability concentrates in layer 23: its layer-specific
p05 is 0.675690/0.717081/0.699578 for these objects. `walnuts` layer 17 is also
weak (0.739979). Earlier layers are generally stable.

Across the 6,400 stability comparisons, the layer-specific p05 drops from
0.9306/0.8989/0.8003 at layers 5/11/17 to 0.7162 at layer 23. Of the 260
comparisons below 0.75, 223 (85.8%) occur at layer 23. This is not mainly a
four-fold bootstrap omission artifact: 243/260 failures still contain all
four folds. The central median therefore hides a late-layer, object-dependent
tail rather than random failure of the whole estimator.

### Global capture does not translate into patch-tail suppression

| Layer | Physical energy capture | Held-out p95 drift reduction |
|---:|---:|---:|
| 5 | 0.874256 | 0.485229 |
| 11 | 0.807510 | 0.338455 |
| 17 | 0.672776 | 0.129384 |
| 23 | 0.492662 | 0.098899 |

The basis is fitted from image-level mean central differences, while G3 is a
patchwise 95th-percentile residual criterion. This creates a target mismatch:
the rank-4 tangent explains the global mean shift direction but misses
spatially heterogeneous and late-layer residual tails. All transform families
remain weak (mean reductions 0.211–0.317), so the failure is not attributable
to only one photometric axis.

The G3 bootstrap interval is only [0.256818, 0.269166], far from the 0.65
threshold. Only 6.6% of the 3,072 shift rows reach 0.65. Even among the 1,498
rows whose mean-direction capture exceeds 0.8, 56.9% remove less than 0.4 of
the patch-tail drift and 29.9% remove less than 0.2. This directly separates
"the nuisance direction was found" from "the distribution shift was
corrected": the former is true at the image mean, while the latter is false
at the spatial tail.

The failure is also not reducible to G2 instability. `rice` has stability p05
0.957450 and energy capture 0.835805, yet its drift reduction is only 0.333528;
`vial` is similarly stable at 0.917224 but removes only 0.250643. Thus a more
stable estimate of the same global subspace cannot by itself close G3.

A post-hoc early-layer diagnostic does not rescue the claim. Layer 5 alone is
best, but its macro reduction is only 0.485229 and its minimum object reduction
is 0.360534, still below the frozen 0.65/0.40 requirements.

### Safety is acceptable but not free

The bootstrap 95% intervals are:

- pAUROC delta: [-0.000297, -0.000065]
- F1 delta: [-0.000891, -0.000233]
- anomaly-energy retention: [0.928188, 0.932238]

The loss is small enough for G4, but consistently negative. CutPaste is the
most affected defect (pAUROC/F1 -0.000637/-0.001633; retention 0.915912).
Layer 5, where drift removal is strongest, retains only 0.813469 anomaly
energy; layer 23 retains 0.970995. Increasing projection strength at early
layers therefore risks trading nuisance removal for anomaly absorption.

G4 must not be interpreted as proof that the correction is safe at useful
strength. The current projection leaves 73.7% of normal patch-tail drift in
place, so its small metric damage is partly the consequence of a weak
intervention. The safety margin is especially narrow for `can`, whose total
anomaly-energy retention is already 0.901549 against the 0.90 gate.

## Structural conclusion

The experiment rejects the current formulation: a single global, hard rank-4
physical projection shared over every patch and used equally across layers.
Its image-level nuisance direction is real, but it is neither sufficiently
local nor sufficiently depth-stable to provide the required distribution-shift
correction.

Any future work must be a new method and protocol, not tuning of this run. The
most defensible redesign is a patch-conditioned or mixture tangent with
layer-specific soft shrinkage, calibrated only on support-side held-out normal
shifts and guarded by anomaly-energy retention. It must first repeat Stage -1;
no real TESTpublic shadow is justified by this result.

## Audit and reproducibility

- Exact totals: 3,072 shift rows, 4,096 defect rows, 512 retained masks, and
  136,192 finite/nonconstant maps.
- Stimulus amendment used 10 deterministic pixel-location retries and zero
  feature-level retries; corrected scores and anomaly metrics never selected a
  retry.
- Pulled SHA manifest passed. Independent local reaggregation reproduced
  `per_object.tsv`, `bootstrap.tsv`, and `verdict.txt` byte-for-byte.
- Logs contain no traceback/error; no dense feature, score-map, transformed
  image, or patch-cache directory exists.
- Remote cleanup retained the compact result root and removed the isolated v3
  source snapshot. The cleanup guard was repaired to pass stdin into
  `docker exec -i` and to distinguish a completed zombie controller from a
  live process; its focused regression suite passes.
