# FastRecon and RegAD fidelity audit

**Audit date:** 2026-09-29  
**Scope:** current registered implementations versus primary papers and their
official repositories, with a v13 matched-support recommendation. No test suite
or GPU run was performed for this audit or the new adapter.

## Decision

FastRecon is the first of these two methods that can use the v13 support
manifests at 1/2/4/8/16 shots without method training. The new
`fastrecon_paper` adapter follows the ICCV paper's ResNet-50, 368-pixel input,
0.15 coreset ratio, distribution-regularized regression, squared pixel
residual, and sigma-4 smoothing. It has 15 configurations matching the current
AnomalyDINO seed/shot support manifest matrix.

Two material implementation differences are declared. The original FastRecon
repository has no detected license and its checked-in script does not match
several paper settings, so the adapter is an independent implementation of the
paper method and does not copy that repository's code. Also, the paper's
unregularized inverse becomes singular at 8 and 16 shots under its stated
368-pixel/15%-coreset settings. The adapter evaluates the same least-squares
objective using its Moore-Penrose row-space solution. That is a defined
extension of the objective for rank-deficient coresets; it is not an exact
replication of the repository's matrix-inverse path. The method should be
called a paper-faithful implementation with a declared solver extension, not an
official-code reproduction.

RegAD cannot run first in the current pipeline. Its actual paper method needs a
category-agnostic ResNet-18/STN/Siamese model trained on source categories,
followed by a support-only positional Gaussian fit. The registered project
module has neither the learned model nor that fit. The official training script
also selects checkpoints using the target test labels, which violates the v13
evaluation boundary. RegAD needs an isolated training and checkpoint-selection
path before it is suitable for this comparison.

## Primary sources and licenses

| Method | Primary paper | Official repository and reviewed revision | License status |
| --- | --- | --- | --- |
| FastRecon | Zheng Fang et al., “FastRecon: Few-shot Industrial Anomaly Detection via Fast Feature Reconstruction,” ICCV 2023, pp. 17481–17490. [CVF paper](https://openaccess.thecvf.com/content/ICCV2023/html/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.html), [PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.pdf), DOI [10.1109/ICCV51070.2023.01603](https://doi.org/10.1109/ICCV51070.2023.01603). | [FzJun26th/FastRecon](https://github.com/FzJun26th/FastRecon), default-branch commit `fcffadb6b824f08c30358521612de67858fd0f51` (2023-10-25). | GitHub reports no license; the repository tree has no `LICENSE`. The adapter does not copy the repository implementation. |
| RegAD | Chaoqin Huang et al., “Registration Based Few-Shot Anomaly Detection,” ECCV 2022 (Oral), LNCS 13691, pp. 303–319. [Springer chapter](https://doi.org/10.1007/978-3-031-20053-3_18), [arXiv 2207.07361](https://arxiv.org/abs/2207.07361), [PDF](https://arxiv.org/pdf/2207.07361). | [MediaBrain-SJTU/RegAD](https://github.com/MediaBrain-SJTU/RegAD), default-branch commit `5e2c1f8c18d302b0354471567846fee3ed2ff063` (2022-09-07). | MIT, present in repository `LICENSE`. |

The Springer chapter's registered page span is 303–319 (ECCV 2022 proceedings,
Part V), confirmed by the chapter DOI metadata. The FastRecon ICCV paper is the
published version; the RegAD arXiv link is the authors' preprint copy of the
ECCV paper. Repository details above were read from the pinned commits and
license files, not inferred from search snippets.

## Current v13 baseline gaps

| Method | Current implementation | Paper method / material gap |
| --- | --- | --- |
| FastRecon | `src/fmad/methods/fastrecon.py` extracts the configured DINOv2 features, retrieves each query patch's top 10 normalized neighbors, solves a per-patch ridge fit with `lambda=0.01`, and reports an L2 residual with a top-1%-patch image statistic. | FastRecon uses a pooled support coreset and estimates one support-feature mean map. It solves a query-to-support linear reconstruction with a distribution-centering penalty (`lambda=0.3`), scores squared L2 reconstruction error, and reports maximum patch error. There is no nearest-neighbor query or local K=10 regression in the paper method. Current DINOv2 features, resolution, solver, regularizer, and image score are material departures. |
| RegAD | `src/fmad/methods/regad.py` normalizes DINOv2 maps, chooses one support image by global cosine similarity, performs softmax dense matching at temperature 0.1, and scores cosine discrepancy. | RegAD trains a shared registration model: ImageNet ResNet-18 through three convolution blocks, three spatial transformers (paper default `rotation_scale`), and a Siamese encoder/predictor with a symmetrized negative-cosine feature loss. Inference augments normal supports, estimates a Gaussian per spatial position, scores registered features with Mahalanobis distance, and maps through inverse affine transforms. The current implementation has no STN, learned encoder/predictor, category-agnostic training, support augmentation, positional covariance, or Mahalanobis scoring. |

Existing FastRecon and RegAD run results remain valid measurements of the
registered project adaptations. They are not paper replications and must not be
expanded as such.

## FastRecon paper method and implementation

The paper's method is training-free for the target category. It uses an
ImageNet-pretrained ResNet-50, takes layer2/layer3 features, smooths each with
3x3 average pooling, combines them at the layer2 46x46 grid, and makes a
K-center greedy coreset from the normal support feature pool. For a support
feature map mean `mu`, query patches `Q`, coreset rows `S`, and `lambda=0.3`, the
paper's Eq. (6) objective is:

```text
min_W ||Q - W S||_F^2 + lambda ||mu - W S||_F^2
```

`S` has shape `C x D`, while `Q` and `mu` have shape `R x D`. The minimizer's
reconstruction can be written as:

```text
Qbar = ((Q + lambda * mu) / (1 + lambda)) @ P_row(S)
P_row(S) = V_r.T @ V_r, where V_r are the retained right singular vectors of S
```

This is the unique fitted `W S` even when `W` itself is not unique. It is the
Moore-Penrose least-squares solution of the stated objective. The module uses a
numerical-rank SVD tolerance of `eps * max(C,D) * largest_singular_value`.
FastRecon's paper settings are 368x368 input, `lambda=0.3`, coreset sampling
ratio `s=0.15`, and ResNet-50 layer2/layer3 features. The repository's code
uses a sparse random projection (`eps=0.9`) before K-center selection; the new
adapter keeps that acceleration and seeds both the projection and first center
from the v13 seed for reproducibility. It aggregates only the exact normal
support paths selected by the common pipeline.

The paper defines pixel scores as squared L2 residuals, bilinearly upsamples
and smooths the map with a Gaussian width of 4, and defines image score as the
maximum patch residual. The adapter keeps those pixel-map operations and
returns a native-size map for the streaming evaluator. For the v13 image-level
metric, it sets image score to the maximum value of the final native map, as
required by the v13 protocol. This differs from the paper's maximum before map
resize/smoothing and is recorded in method metadata.

### Official FastRecon code audit

The paper links the repository above, but the checked-in README still says
“code will be available soon” and there is no repository license. The single
main script is not a safe drop-in replication:

- It instantiates `wide_resnet50_2`, whereas the paper specifies ResNet-50.
- Its parser defaults are 7200x2400 input, `coreset_sampling_ratio=0.001`, and
  `lambda=2`; its README example uses 0.01, while the paper states 368x368,
  0.15, and 0.3.
- Its training loader is the whole category's `train` folder rather than the
  selected v13 supports. Each `training_step` overwrites the feature pool, and
  `training_epoch_end` builds the coreset from only the last batch held in that
  field (the parser default batch size is 1).
- The script uses an explicit inverse of `(1+lambda) S S^T` for every query.
  That agrees with the full-rank least-squares normal equation but fails when
  `S S^T` is singular. It also uses an unsquared L2 residual, while the paper's
  pixel equation specifies squared L2.

The source locations are the repository's `main.py` feature extractor,
transform, coreset fit, solver, and score code; the paper hyperparameters are in
§4.1 “Implementation Details.” An independent implementation avoids silently
copying the unlicensed repository source.

### Shot scaling and solve requirements

At 368 pixels the combined map has `46 x 46 = 2,116` patches and
`D = 512 + 1,024 = 1,536` channels. The official paper ratio gives
`C = floor(0.15 * shots * 2,116)` centers. The table estimates float32 support
and matrix storage. “Inverse work” is the nominal `2 C^3 / 3` FLOP count if the
official Gram inverse were evaluated directly for a query; it is shown to make
the rank issue and scaling visible, not as a prediction of the new SVD path.

| Shots | Support patches | Coreset centers C | Full support features | Coreset | CxC float32 Gram | Nominal Gram inverse work | Rank condition |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2,116 | 317 | 12.4 MiB | 1.9 MiB | 0.38 MiB | 0.02 GFLOP | C < D; full row rank possible |
| 2 | 4,232 | 634 | 24.8 MiB | 3.7 MiB | 1.53 MiB | 0.17 GFLOP | C < D; full row rank possible |
| 4 | 8,464 | 1,269 | 49.6 MiB | 7.4 MiB | 6.14 MiB | 1.36 GFLOP | C < D; full row rank possible |
| 8 | 16,928 | 2,539 | 99.2 MiB | 14.9 MiB | 24.59 MiB | 10.91 GFLOP | C > D; Gram necessarily singular |
| 16 | 33,856 | 5,078 | 198.4 MiB | 29.8 MiB | 98.37 MiB | 87.29 GFLOP | C > D; Gram necessarily singular |

The paper reports 0.52s fit at 2-shot, 1.95s at 8-shot, and 0.04s/query on
one GTX 3090. Those measurements are useful reference points only: the paper
does not report 16-shot timing, and its official script's defaults do not
reproduce all paper settings. K-center fit, a thin SVD of `C x D`, ResNet-50
features, and one map pass are appropriate workloads for the existing local
4090 / assigned A6000 machines, but local timing remains unmeasured. At 8 and
16 shots the adapter's SVD projection is mandatory for a defined regression;
the project's paper-replication gate should verify this implementation on a
single-class pilot before allocating the full matrix.

## RegAD paper method and implementation requirements

RegAD is not a direct support-only nearest-neighbor method. The paper's main
setting trains one category-agnostic model on source categories and tests on an
unseen target category. Its official defaults are a pretrained ResNet-18, 224
pixels, `rotation_scale` STNs, 50 epochs, batch size 32, SGD learning rate
`1e-4`, momentum `0.9`, and cosine learning-rate decay. At support fitting and
inference, it creates 22 support variants per image (eight small rotations,
eight translations, horizontal flip, grayscale, and three right-angle
rotations), concatenates registered layer features, fits a Gaussian at each
spatial position with `0.01 I`, scores Mahalanobis distance, applies inverse
affine alignment, and smooths with sigma 4.

The paper's main comparison uses 2/4/8 shots and reports results averaged over
10 support draws. It also reports 16-shot in a separate full-data comparison,
so 16-shot is a valid new support count but not part of the main FSAD table. The
repository ships category/shot-specific checkpoints and fixed support rounds;
those do not constitute checkpoints for the eight MVTec AD 2 objects or the
v13 manifests.

Before a v13 RegAD run, the implementation would need a declared source split
for each held-out MVTec AD 2 class, training for each intended shot setting,
support-manifest injection instead of the repository's fixed `.pt` samples,
deterministic support augmentation, and target-label-free checkpoint
selection. The checked-in `train.py` evaluates target test labels and masks at
every epoch and saves the checkpoint maximizing image-AUROC plus pixel-AUROC.
That checkpoint rule cannot be used for this protocol. The paper's reported
4.47s is support adaptation time after training, not a measured budget for the
category-agnostic neural training stage. Accordingly RegAD would require
substantially more setup and compute than FastRecon; its cost is 50 training
epochs per target/shot checkpoint if reproducing the repository training
protocol, in addition to support fitting and per-query Mahalanobis scoring.

## v13 configurations and launch order

The new configs are `fastrecon_paper_seed{0,1,2}_s{01,02,04,08,16}.yaml`.
Seed 0 uses the existing VOID shot manifests; seeds 1 and 2 use the v13 nested
1/4/8 prefixes and E1 2/16 manifests. A direct manifest comparison verified
that all five shot lists are exact prefixes of their 16-shot list for all eight
classes and all three seeds. This mirrors the AnomalyDINO matrix and keeps the
support paths matched, as required by `PROTOCOL.md`.

Run FastRecon before RegAD among these two methods. A reasonable shot order is
1-shot local single-class validity, then 2/4-shot, followed by 8/16-shot with
the rank-deficient SVD extension visible in metadata. The project-level v13
order still controls GPU execution: the registered AnomalyDINO work and its
collection precede new published-method expansions. `fastrecon_paper` and
`fastrecon_resnet50` are registered by import decorators in
`src/fmad/methods/fastrecon_paper.py`; the shared method importer still needs a
`fastrecon_paper` import branch before these configs are runnable. No results
or timing claims are made by this audit.

## References

1. Zheng Fang, Xiaoyang Wang, Haocheng Li, Jiejie Liu, Qiugui Hu, and Jimin
   Xiao. FastRecon: Few-shot Industrial Anomaly Detection via Fast Feature
   Reconstruction. ICCV 2023, pp. 17481–17490.
2. Chaoqin Huang, Haoyan Guan, Aofan Jiang, Ya Zhang, Michael Spratlin, and
   Yanfeng Wang. Registration Based Few-Shot Anomaly Detection. ECCV 2022,
   LNCS 13691, pp. 303–319. (Oral.)
