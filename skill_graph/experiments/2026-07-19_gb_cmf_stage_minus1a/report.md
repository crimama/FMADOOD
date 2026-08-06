# GB-CMF Stage -1A Report

Date: 2026-07-19  
Verdict: `KILL_FOR_CLAIM / NO_PUBLIC_ACCESS / NO_STAGE_MINUS1B`

## Outcome

Geometry-Bounded Counterfactual Margin Flow (GB-CMF) tested the remaining
representation-level hypothesis after WAFF-H+ proved that fixed-sum score
reweighting had no feasible support-only point. GB-CMF kept the deployed score
exactly equal to hard latent 1NN distance plus `0.25` frozen-standardized NLL,
but refined only the default H+ coupling subnets with clean, shifted-clean,
defect, and shifted-defect factorial cells and augmented tail/drop constraints.

The valid v3 run completed the preregistered four cases and five arms on dsba3
GPUs 0--3: exactly 20 fits. It failed all scientific continuation gates except
map/access coverage. No public, private, test, or ground-truth path was
enumerated, Stage -1B was not authorized, and there is no scientific retry or
post-result tuning.

## Case Results

Values are held-source macro pAUROC@0.05 / fixed F1. Raw metrics omit the fixed
threshold transform but preserve the same underlying score ordering.

| case | default | nonfactorial | GB-CMF | affine null | nonlinear null |
| --- | ---: | ---: | ---: | ---: | ---: |
| rice/0, cutpaste x exposure | .854963/.548727 | .912846/.536336 | .911363/.536781 | .854971/.548727 | .854953/.548727 |
| can/1, chromatic x gamma | .702534/.358108 | .760698/.452404 | .735204/.379290 | .702575/.358887 | .702542/.358108 |
| rice/1, blur x contrast | .970764/.627212 | .981815/.585089 | .964979/.651040 | .970790/.627212 | .970790/.627212 |
| can/0, cutpaste x white balance | .921124/.642900 | .935650/.665648 | .884443/.653699 | .921152/.643360 | .921140/.643360 |

The descriptive source-level bootstrap does not support broad improvement.
GB-CMF minus default had median `+0.011818` pAUROC and `+0.011919` F1, but the
95% intervals crossed zero (`[-0.011172,+0.032390]` and
`[-0.033065,+0.049265]`). More importantly, GB-CMF was consistently worse than
the matched nonfactorial refinement in pAUROC: median `-0.023865`, interval
`[-0.033540,-0.014036]`. Can and rice both disagreed in direction across
seeds.

The apparent macro averages also hide large source-specific harm. Can/0 source
`105` lost `0.1688` pAUROC and `0.2213` F1 versus default, including raw losses
of `0.1716/0.2389`. In the same case source `289` lost `0.0514` pAUROC while
gaining `0.1336` F1. Rice/0 source `205` gained `0.0514` pAUROC but lost
`0.1077` F1. This ranking/threshold disagreement is incompatible with a stable
default replacement.

## Constraint Failure

Only can/0 was jointly feasible on all held sources. The maximum held
`g_tail/g_drop` for GB-CMF was:

| case | max g_tail | max g_drop | joint feasible |
| --- | ---: | ---: | --- |
| rice/0 | +.044851 | +.000019 | no |
| can/1 | +.063141 | +.051972 | no |
| rice/1 | +.126822 | ~0 | no |
| can/0 | -.010227 | ~0 | yes |

The duals grew as high as `7.5051`, but the final train-cell tail constraints
still reached `+.1325`, `+.1189`, `+.2364`, and `+.1711` across the four cases.
The optimizer therefore did not solve the intended worst-axis feasible set;
it only reacted to the most recently visited stochastic cell for each key.

There is also an objective-level asymmetry. The implemented inequality term is
`lambda*g + rho/2*relu(g)^2`. Once a cell has `g<0`, its positive dual rewards
making that already-satisfied constraint more negative, while another
axis/family may remain violated. Rice/1 exposed this most clearly: its GB-CMF
loss reached `-5.2582` and ended at `-0.9471` despite a `+.2364` train tail
violation. This is valid evidence about the frozen candidate, not a reason to
rewrite or rerun it.

## Geometry Failure

The decisive structural failure is that GB-CMF was geometry-audited after
training, not geometry-bounded during training. A mean-squared parameter
proximal term with weight `1e-3` allowed only `0.0031--0.0051` maximum parameter
movement, yet the invertible flow score geometry changed drastically:

- all 16 GB-CMF held sources failed the geometry gate;
- pair-distance Spearman fell to `.910--.924` on rice/0, `.905--.925` on
  can/1, `.654--.720` on rice/1, and `.631--.765` on can/0, below `.95`;
- negative-logdet share ratios were about `2.1--4.5`, and coupling-logdet q99
  ratios about `2.1--4.3`, both above the `1.25` bound;
- total-score standard-deviation ratios expanded to `2.49--3.73` on rice/1 and
  `1.98--5.20` on can/0.

The soft/hard proxy was not the main cause. Fifteen of 16 sources passed it
with almost perfect rank and top-tail agreement. Rice/1 source `036` alone
missed only the q995 tolerance (`.0139 > .01`). Thus the training surrogate
was usually faithful to the deployed hard score; the flow itself distorted
latent/logdet geometry.

The nonfactorial arm often outperformed GB-CMF despite moving farther in
parameter space (`0.0066--0.0090`). This shows that C/D/SD ranking can contain
useful distribution-shift signal, but the current augmented constraints do not
convert it into stable joint feasibility. Parameter distance is also an
inadequate trust region for coupling flows because tiny subnet changes can
produce large accumulated log-determinant and score-scale changes.

## Structural Conclusions and Next Design

This experiment closes the current GB-CMF candidate. A future method must not
be a hyperparameter retry. It must change three structural properties before a
new support-only preregistration:

1. Replace the signed reward term by a projected inequality penalty such as
   `([relu(lambda+rho*g)]^2-lambda^2)/(2*rho)`, so satisfied cells cannot earn
   an additional negative reward.
2. Optimize an actual grouped worst-axis/worst-family risk over replayed cells
   (max or CVaR), rather than updating duals from the latest cyclic mini-cell.
3. Put geometry inside the optimization trust region: constrain held-train
   pair ranks, score/component scale, and especially coupling/logdet drift, or
   use function-space rollback. A parameter-space proximal term alone is not a
   geometry bound.

Any successor must first show joint feasibility and geometry preservation
against default and nonfactorial on a new frozen support-only assay. It cannot
inherit GB-CMF's method name, claim credit, Stage -1B authorization, or public
access.

## Validity and Custody

The first two launches were infrastructure-invalid and produced no scientific
result: v1 attempted to deepcopy a CUDA generator, and v2 lacked the cuBLAS
workspace configuration required by deterministic CUDA `cdist`. The valid v3
used an explicit offline clone and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

All 67 compact artifact checksums passed. Local reaggregation matched the
remote summary and verdict exactly except for the expected `created_utc`
field. Each case hashed all 128 manifest entries, opened exactly the 16
whitelisted support images, achieved 100% whitelist coverage, and recorded
zero public/private/test/ground-truth attempts. The isolated remote snapshot
was removed, the completed compact result was retained, and dsba3 GPUs 0--3
returned to 1 MiB memory and 0% utilization.

Focused local verification passed Ruff, shell syntax, compileall, diff checks,
and 45 GB-CMF/WAFF tests. The repository-wide suite could not collect because
the minimal local environment omits optional research dependencies including
OpenCV, Pillow, tifffile, scikit-learn, and SciPy. Strict basedpyright still
reports the established dynamic NumPy/JSON typing debt in experiment modules;
the modified core pipeline itself type-checks cleanly.

Compact evidence:
`results/remote_runs/dsba3/gb_cmf_stage_minus1a_20260719_v3/summary.json`.
