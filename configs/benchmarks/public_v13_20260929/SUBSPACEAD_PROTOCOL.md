# SubspaceAD v13 adapter protocol (2026-09-29)

## Scope and evidence

- Method: Lendering et al., “SubspaceAD: Training-Free Few-Shot Anomaly
  Detection via Subspace Modeling,” CVPR 2026, pp. 28557–28566.
- Official implementation: `CLendering/SubspaceAD`, commit
  `ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7`, Apache-2.0.
- The adapter runs the official few-shot method on the v13 MVTec AD 2
  `test_public` inventory. It is not a use of the upstream batched zero-shot
  mode, which fits on unlabeled test images.

## Backbone artifact and runtime

- Backbone: frozen DINOv2-with-registers-Giant / ViT-G/14,
  `facebook/dinov2-with-registers-giant`.
- Pinned Hub model revision for `model.safetensors`:
  `d45b6a06115df245b3b06b5104f84d5174fbda7b`.
- Public, unauthenticated artifact URL:
  <https://huggingface.co/facebook/dinov2-with-registers-giant/resolve/d45b6a06115df245b3b06b5104f84d5174fbda7b/model.safetensors>
- Artifact size: **4,546,030,112 bytes** (4.23 GiB); expected SHA-256:
  `c03832d44691e99b62ae28c4dfa2f134853a3614b3756c94f525109cce5a5051`.
- The adapter loads only from the explicit local `backbone_config.model_dir`
  (`/model-cache/hub/hf_dinov2_with_registers_giant_d45b6a06`), verifies the
  weight hash once per process, uses `local_files_only=True`, and requires
  `config.json` plus `model.safetensors` in that directory. It does not download
  weights at runtime.
- Compatible runtime packages: `transformers>=4.48.0,<5.0` and
  `safetensors>=0.4.3`. The model config records Transformers 4.48.0.dev0 and
  declares model type `dinov2_with_registers`. The local Docker runtime did not
  have Transformers installed at the initial audit; no package was installed.
  The `dsba3` package check is deferred until its active run finishes.
- Official preprocessing is square resize to the object resolution, bicubic
  interpolation, RGB mean `(0.485, 0.456, 0.406)` and standard deviation
  `(0.229, 0.224, 0.225)`, with no center crop. The v13 dataset contract sets
  `sheet_metal` to 448 pixels and the other seven objects to 672.
- Layers 22–28 are averaged as raw patch hidden states. Forward calls request
  `output_hidden_states=True` and explicitly set `output_attentions=False`.
  The official runner always materializes all 40 attention maps even though
  this protocol disables its saliency mask. Omitting those unused tensors
  preserves the selected hidden states and avoids about 19.1 GiB of float32
  attention maps at 672 pixels (24 heads, 2,309 tokens, 40 layers). The 41
  returned hidden-state tensors are about 555 MiB. These are source-based
  estimates, not measured high-water marks.

## Fit and scoring

- The shared pipeline selects the exact support paths from each configured
  manifest and passes only those paths to `fit()`. Fit does not enumerate the
  dataset, read query images, or access masks or acquisition labels.
- Per support image, extract its original plus 30 `RandomRotation` views with
  angle range `[0, 345]`. The shared dataset preprocessing remains
  `no_mask_no_rotation`; these are the method’s own support-only augmentations.
- Fit standard PCA over the averaged layer-22-to-28 patch features with 99%
  explained variance, using the official two-pass float64 mean/covariance and
  eigendecomposition. Standard PCA streams feature batches; it does not
  materialize all 1,142,784 token rows for a 16-shot, 672-pixel fit. At the
  default one-image batch, a batch is 13.5 MiB in float32 and 27 MiB after
  float64 conversion. The 1,536-square float64 covariance is 18 MiB; the
  explicit covariance-pass tensor set is about 90 MiB, plus solver workspace.
- Anomaly score is squared PCA reconstruction residual. Resize each map
  bilinearly to the query’s native image dimensions and apply the official
  3×3 Gaussian blur with sigma 4. The shared pipeline requires native-size
  maps. Image AUROC uses the maximum of the final map per v13; SubspaceAD's
  paper default is the top-1%-mean and is not used here.
- Use the shared streaming evaluator for full-eight-class macro AU-PRO@0.05,
  pixel partial AUROC, image AUROC, and acquisition strata. Do not fill these
  cells with paper-reported MVTec AD / VisA results. Maps and fitted weights
  remain disabled in all configs.

## Configurations and support manifests

Fifteen configs cover 3 seeds × 1/2/4/8/16 shots:
`subspacead_seed{0,1,2}_s{01,02,04,08,16}.yaml`.

| Seed | Manifest source |
| :-- | :-- |
| 0 | `configs/methods/VOID_support_{1,2,4,8,16}.json` |
| 1, 2; 1/4/8 shots | `configs/benchmarks/public_v13_20260929/seed{1,2}_support_{1,4,8}.json` |
| 1, 2; 2/16 shots | `configs/benchmarks/v7_evidence/E1_seeds/manifests/seed{1,2}_support_{2,16}.json` |

The 8- and 16-shot settings extend beyond the paper’s reported 1/2/4-shot
few-shot table. Metrics must come from the configured manifest and v13
evaluator, separately for every seed and shot count.

## Readiness boundary

The initial local host and `dsba3` cache check found no DINOv2-G/14 checkpoint.
The checkpoint was subsequently placed in the local mounted model cache and
its SHA-256 was verified by the coordinator; transfer to `dsba3` is pending.
No experiment or GPU run was launched as part of adapter preparation.
