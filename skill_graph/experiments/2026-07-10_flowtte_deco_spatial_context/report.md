# FlowTTE DeCoFlow Spatial Context Diagnostic

## Question

Can DeCoFlow-style local spatial context rescue the collapsed `can` class
without materially hurting stable classes?

## Method and controls

- Backbone: `dinov3_vith16plus`, layers `7,15,23,31`
- Fusion: `layer_norm_mean`
- DVT: `position_mean`, alpha `1.0`
- Support: 16 images selected by `dinov3_cls_greedy_coreset`, seed `0`
- Support augmentation: brightness `0.8,1.2`
- Flow: 2 coupling layers, 3 epochs, LR `2e-4`, clamp `1.9`
- Loss: tail weight `0.3`, top ratio `0.05`, logdet weight `0.02`
- Scoring: latent distance, density weight `0.25`
- Morphology: excluded

Three matched branches run on `can,fabric,rice,wallplugs`:

1. `fused`: current MLP reference
2. `conv2d_flow`: architectural null for the spatial flow
3. `spatial_context_flow`: asymmetric coupling where a gated depthwise 3x3
   context is concatenated only into the scale subnet; the shift subnet sees
   the identity split only

The causal SCM comparison is (3) versus (2). Comparison with (1) measures
practical competitiveness but does not isolate context because the coupling
subnet architecture also differs.

## Pre-registered reduced gate

Promote to all eight objects only if:

- `can` improves by at least `+0.02` F1 or `+0.01` AUROC versus matched
  `conv2d_flow`;
- at least two of `fabric,rice,wallplugs` avoid an F1 decrease greater than
  `0.01`;
- mean control F1 decreases by no more than `0.005`; and
- mean control AUROC decreases by no more than `0.01`.

## Runtime

Remote fixed container: `hun_fsad_tta_012`, GPUs 0, 1, and 2.

Run roots:

- `/workspace/results_remote/flowtte_deco_scm_reduced_20260710_v1_fused`
- `/workspace/results_remote/flowtte_deco_scm_reduced_20260710_v1_conv2d_flow`
- `/workspace/results_remote/flowtte_deco_scm_reduced_20260710_v1_spatial_context_flow`

Status at launch verification: all three processes were active at 100% GPU
utilization. `can` completed scoring in each branch and all branches advanced
to `fabric`. Final metrics are written after the four-object loop completes.

## Final reduced results

| branch | can AUROC | can F1 | control mean AUROC | control mean F1 | all-4 mean F1 |
|---|---:|---:|---:|---:|---:|
| fused MLP | 0.527792 | 0.000366 | 0.941270 | 0.707184 | 0.530480 |
| Conv2D | 0.518246 | 0.000331 | 0.936859 | 0.695026 | 0.521352 |
| Conv2D + SCM | 0.527250 | 0.000352 | 0.938580 | 0.699727 | 0.524883 |

SCM versus the matched Conv2D null gives `can` AUROC `+0.009004` and F1
`+0.000020`, while the three-class control means improve by AUROC `+0.001721`
and F1 `+0.004701`. Per-class control F1 changes are fabric `+0.010425`, rice
`-0.000267`, and wallplugs `+0.003946`.

## Decision

The reduced gate **fails** because the can rescue threshold required either
AUROC `+0.010` or F1 `+0.020`; the observed AUROC gain is just below the former
and F1 is effectively unchanged. Therefore the all-eight expansion is not
promoted. SCM is a no-harm architectural diagnostic with a positive control
mean effect, but it does not solve the can collapse under this setting.

## MLP Flow + SCM Follow-up

To isolate SCM without replacing the original MLP flow with Conv2D coupling,
the original affine MLP flow was retained. A gated depthwise 3x3 context from
the identity split was provided only to the scale MLP; the shift MLP received
the identity split alone. The same reduced protocol was distributed across
GPUs 0, 1, and 2.

| class | MLP AUROC | MLP+SCM AUROC | delta | MLP F1 | MLP+SCM F1 | delta |
|---|---:|---:|---:|---:|---:|---:|
| can | 0.527792 | 0.522422 | -0.005369 | 0.000366 | 0.000378 | +0.000012 |
| fabric | 0.977473 | 0.973827 | -0.003646 | 0.753199 | 0.724703 | -0.028496 |
| rice | 0.944247 | 0.937653 | -0.006594 | 0.706361 | 0.702124 | -0.004238 |
| wallplugs | 0.902096 | 0.901451 | -0.000645 | 0.661992 | 0.686551 | +0.024559 |
| mean | 0.837902 | 0.833838 | -0.004064 | 0.530480 | 0.528439 | -0.002041 |

The MLP+SCM follow-up also fails the promotion gate. It does not rescue can,
reduces mean AUROC and F1, and produces a material fabric F1 regression. The
wallplugs gain is real under this seed but is insufficient to justify an
all-eight run or adoption as the default method.
