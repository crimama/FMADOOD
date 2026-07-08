# FlowTTE DVT Structural Analysis

## Scope

This analysis diagnoses the DVT-style `position_mean` denoising probe before further hyperparameter tuning.

Setting:

- Dataset: MVTec AD2 TESTpublic, all 8 public objects.
- Backbone: `dinov3_vitl16`, `layer_norm_mean` over layers `[5, 11, 17, 23]`.
- Support: fixed SuperAD-16 reference JSON used by the current Flow-LatentBank no-TTE runs.
- Flow: 2 coupling layers, hidden multiplier 1, 3 epochs, lr `2e-4`, clamp `1.9`.
- Score: latent nearest distance + `0.25 * density_penalty`.
- TTE expansion: off, `expansion_budget=1.0`.
- DVT probe: support-fitted position mean artifact field, alpha `1.0`.
- Diagnostic sample: first 12 good and 12 bad public test images per object for score decomposition.

Raw artifacts:

- `results/remote_runs/dsba3/flowtte_dvt_structural_analysis_20260708_v1/`
- Key summaries copied here:
  - `structural_object_summary.tsv`
  - `structural_correlation_summary.tsv`
  - `side_ablation_summary.tsv`
  - `summary.json`

## Prior Result Being Explained

Alpha `1.0` DVT position denoise improved the all-object mean over no-DVT:

| metric | no-DVT | DVT alpha 1.0 | delta |
|---|---:|---:|---:|
| `seg_AUROC_0.05` | 0.797743 | 0.825207 | +0.027464 |
| `seg_F1` | 0.437800 | 0.468348 | +0.030548 |

Object-level behavior is mixed:

- Large gain: `fabric`, `sheet_metal`.
- Small/partial gain: `rice`, `vial`, `wallplugs`.
- Harm: `can`, `fruit_jelly`, `walnuts`.

## Main Structural Findings

### 1. Raw feature support becomes compact, but NF latent memory does not.

Mean over 8 objects:

| quantity | denoised/raw ratio |
|---|---:|
| support feature variance | 0.451323 |
| support feature leave-one-out NN distance | 0.882623 |
| support latent leave-one-out NN distance | 1.188963 |

So DVT alpha `1.0` does compress raw patch feature variation, but after NF training and support standardization the latent nearest-neighbor spacing expands on average. Therefore the effect is not simply "memory bank volume compression" in the final scoring space.

### 2. The improvement tracks score separation, not artifact magnitude.

Correlation across 8 objects:

| predictor | corr with AUROC delta | corr with F1 delta |
|---|---:|---:|
| final score separation change | 0.685442 | 0.653105 |
| latent distance separation change | 0.501601 | 0.482790 |
| density penalty separation change | 0.476038 | 0.499938 |
| artifact/foreground proxy corr | 0.342482 | 0.085934 |

This suggests the useful signal is whether denoising improves bad-vs-good ranking after FlowTTE scoring. Artifact overlap or size alone is not a reliable selector.

### 3. Query-side correction is the dominant effect.

Mean bad-good final score separation:

| scenario | mean separation |
|---|---:|
| raw support, raw query | 0.633344 |
| denoised support, raw query | 0.270657 |
| raw support, denoised query | 0.668832 |
| denoised support, denoised query | 0.681876 |

Support-only denoise creates a strong mismatch and hurts separation. Query-only denoise is already close to both-denoise. This indicates the useful part of the current probe is mostly test/query feature correction, not support memory compaction.

### 4. The learned "artifact" field is not purely a simple global positional bias.

The artifact field has moderate effective rank, not rank-1 behavior:

- effective rank range: about `8.78` to `16.32`.
- top-3 energy share range: about `0.55` to `0.76`.

The RGB foreground proxy also shows class-dependent overlap:

- `sheet_metal`, `vial`: artifact field strongly overlaps the proxy foreground/structure.
- `can`, `fabric`, `rice`, `wallplugs`, `walnuts`: high artifact regions mostly avoid the proxy foreground.

Because improved and harmed objects appear on both sides of this proxy, a full position-wise vector subtraction is too coarse as a structural rule.

## Object-Level Readout

| object | AUROC delta | F1 delta | final sep change | feature var ratio | latent LOO ratio |
|---|---:|---:|---:|---:|---:|
| can | -0.026367 | -0.000383 | -0.064836 | 0.389314 | 1.261886 |
| fabric | +0.143312 | +0.196011 | +0.305526 | 0.610734 | 1.051333 |
| fruit_jelly | -0.014053 | -0.023382 | -0.017817 | 0.388848 | 1.273079 |
| rice | +0.003646 | -0.002501 | +0.292502 | 0.559051 | 1.041612 |
| sheet_metal | +0.122860 | +0.089939 | +0.221194 | 0.420118 | 1.091555 |
| vial | +0.008998 | -0.004449 | +0.014413 | 0.165308 | 1.626521 |
| wallplugs | -0.001308 | +0.013455 | -0.018711 | 0.631901 | 1.084368 |
| walnuts | -0.017378 | -0.024304 | -0.344023 | 0.445307 | 1.081352 |

## Conclusion

The current DVT position-mean probe should not be framed as directly compressing the NF latent memory volume. It compresses raw patch feature variation, but the NF stage re-standardizes and often expands latent nearest-neighbor spacing. The performance gain appears when denoising improves bad-vs-good final score separation, mainly through query-side correction and then FlowTTE latent/density scoring.

The next structural branch should therefore avoid broad alpha tuning and test more constrained denoising operators:

1. Query-side or symmetric-but-query-driven denoising diagnostics, because support-only denoise is clearly mismatched.
2. Low-rank/truncated artifact subtraction instead of full position-vector subtraction.
3. Structure-aware artifact fitting, e.g. background/low-variance support positions only, to avoid subtracting object-specific normal structure.
4. A support-only selector based on structural diagnostics, not labels, only after the operator is constrained.

Working claim:

> DVT-style position denoising helps FlowTTE when it removes query-side position/context artifacts that otherwise dominate NF latent ranking. The useful mechanism is ranking-separation stabilization, not simple memory-bank volume compression.
