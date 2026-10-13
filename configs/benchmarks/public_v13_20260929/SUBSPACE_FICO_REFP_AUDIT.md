# Public v13 method gap audit (2026-09-29)

Scope: official paper, released code/license, fitting data access, and whether
published numbers satisfy the v13 MVTec AD 2 `test_public` contracts. No
experiment was run and no metric was inferred from another benchmark.

## Decision

| Method | Legitimate use in v13 | Evidence status |
| :-- | :-- | :-- |
| SubspaceAD | Eligible for the few-shot shot table and its acquisition-stratum table, after running the official method on the exact v13 support manifests and scoring with the shared evaluator. Its paper values cannot fill v13 cells. | CVPR 2026 paper and Apache-2.0 official code; official loader includes AD 2. |
| FiCo | Do not count as a few-shot comparator or fill the current matched-budget strata row. It could only appear in a separately labeled full-train, test-time-reference/adaptation shift block after an AD 2 run. | AAAI 2025 paper and GPL-3.0 official code; the published experiments use full normal MVTec training data and synthetic corruptions, with sample-level AUROC only. |
| ReFP-AD | Can fill only the separate full-data reference table with the reported AD 2 value, attributed as a published result. Do not rank it with the few-shot methods or infer acquisition-stratum values. | ECCV 2026 accepted paper, Appendix D, reports `test_public` AU-PRO0.05; released repo is MIT but its runnable dataset list does not include AD 2. |

## SubspaceAD — few-shot eligible, but rerun required

- **Publication / implementation:** Lendering et al., “SubspaceAD: Training-Free
  Few-Shot Anomaly Detection via Subspace Modeling,” CVPR 2026, pp. 28557–28566.
  The official CVF paper is the published source. Its arXiv v3 notes that the
  AU-PRO evaluation was corrected and metrics recomputed. Official repository
  `CLendering/SubspaceAD`, inspected at `ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7`,
  is Apache-2.0.
- **Fit boundary / shots:** the method fits PCA using patch tokens from only
  `k` normal training images for few-shot inference; no query features, query
  labels, or ground truth enter that fit. The paper reports 1/2/4 shots (five
  seeds) on MVTec-AD and VisA, not AD 2. Its separate batched zero-shot mode
  fits on the entire unlabeled test set and is not eligible here. The released
  code has an AD 2 loader whose fit source is `train/good` and whose query path
  is `test_public`; it has a `k_shot` option but samples randomly rather than
  accepting v13 manifests directly. Use a manifest-bound adapter/subset so the
  exact shared support images are used at every shot and seed, and record them.
- **Method settings to preserve:** frozen DINOv2-G/14; mean of layers 22–28;
  PCA retaining 99% explained variance; 30 random rotation augmentations per
  support image in the released few-shot recipe (the MVTec-AD transistor
  exception is dataset-specific and does not name an AD 2 class); reconstruction
  residual maps, bilinear upsampling and Gaussian sigma 4. The rotations are
  derived only from support images, so they do not spend additional source
  images or breach support-only fitting. Disable masks and do not select settings
  using acquisition labels. Record any class-specific augmentation decision.
- **Metric boundary:** SubspaceAD reports PRO, pixel AUROC and image AUROC on
  MVTec-AD/VisA. Those published PRO values are not AD 2 AU-PRO@FPR 0.05 and
  must not be copied into either v13 table. The paper uses top-1%-mean map score
  for image AUROC; v13 specifies map-maximum I-AUROC. Run its score maps through
  the v13 evaluator for the table, with the primary AU-PRO@0.05 calculated on
  the same full-eight `test_public` inventory as other methods.

**Decision:** SubspaceAD is the only one of these three that can enter the
matched-support few-shot tables. The table remains blank until its support
manifest, inference maps, and v13 metrics are actually run and collected.

## FiCo — different training and shift protocol

- **Publication / implementation:** Chen et al., “Filter or Compensate: Towards
  Invariant Representation from Distribution Shift for Anomaly Detection,”
  *Proceedings of the AAAI Conference on Artificial Intelligence* 39(3),
  2420–2428 (2025), DOI `10.1609/aaai.v39i3.32243`. Official repository
  `znchen666/FiCo`, inspected at
  `69c5e9dec2c0f733decd946fdb455454d881438f`, is GPL-3.0.
- **Fit and inference boundary:** the official MVTec recipe trains the
  teacher/student-derived FiCo model per original MVTec category using the
  category's normal `train` images, rather than a selected 1/2/4/8/16 support
  manifest. It uses WideResNet50 and 256×256 input. At inference the official
  script also supplies a fixed normal training image to its EFDM test-time
  feature compensation. The inference code is no-grad and does not read target
  labels, but its full-train fit plus normal reference differs materially from
  the v13 support-only budget.
- **Shift and metric boundary:** the paper studies original MVTec plus
  synthetically generated brightness, contrast, defocus-blur and Gaussian-noise
  conditions. These are not the AD 2 real-acquisition strata. It reports
  sample/image AUROC, not AD 2 AU-PRO@0.05 or per-stratum results. Although its
  released evaluator constructs anomaly maps, the pixel-metric code is commented
  out and no published localization values establish v13 metric compatibility.

**Decision:** Do not use the paper's FiCo values to fill the v13 AD 2 strata.
Do not present a 16-shot FiCo row: restricting FiCo's training to a shared
few-shot manifest would change its published method setting. If FiCo is retained
as a shift-method reference, it needs a separate full-train AD 2 run and an
explicitly labeled full-data/TTA stratum block. That run would have to train on
normal AD 2 training data, fix and log the normal EFDM reference policy, predict
without query labels, and calculate the shared AU-PRO@0.05 from its maps.

## ReFP-AD — published full-data `test_public` reference

- **Publication / implementation:** Lendering et al., “ReFP-AD: Rectified Flow
  Preconditioning for Energy-Based Anomaly Detection,” accepted at ECCV 2026;
  arXiv v1 `2608.01793` (2026). Official repository
  `CLendering/ReFP-AD`, inspected at
  `0118c9899926e1394bd411aba23ee45f0b07407e`, is MIT.
- **Published AD 2 result:** Appendix D, Table D.3 reports **38.6% AU-PRO0.05**
  and 72.4% image AUROC for ReFP-AD on MVTec AD 2. Its caption defines the
  reported metric as Image-AUROC / AU-PRO0.05 (%) for AD 2. Appendix D says the
  AD 2 results use the official public test split and the same unified training
  regime: a single shared model trained jointly across all categories. For this
  AD 2 experiment, it raises SGLD steps from 60 to 150 and hidden dimensions of
  the flow and EBM MLPs to 1536; remaining hyperparameters are said to match the
  main setup. The main setup uses frozen DINOv2 ViT-G/14, input resized to 700
  then center-cropped to 672, middle layers 22–28, and a gradient-norm score map
  bilinearly upsampled and smoothed at sigma 4.
- **Scope caveat:** MVTec AD 2 defines eight scenarios and its public test split
  includes labeled images from all lighting conditions. Appendix D calls its
  ReFP result the official public split, but gives no per-class table, named
  class list, or explicit aggregation implementation for that AD 2 result. Thus
  38.6 is a valid *published public-split full-data reference*, not a verified
  replication of the v13 per-class macro/streaming contract. Preserve that
  qualification in Table 1b's note; do not present the figure as a new local run
  or claim project-evaluator parity. Table D.3 has no acquisition-condition
  breakdown, so it cannot fill the regular/shift_1/shift_2/shift_3 cells.
- **Code boundary:** the public ReFP README and `config.py` describe runnable
  MVTec-AD and VisA modes, not MVTec AD 2. Its released evaluator produces
  per-category I-AUROC/P-AUROC for those datasets, while Appendix D says its AD
  2 comparisons used original code where available or Anomalib otherwise.
  Therefore the 38.6 value is sourced from the paper supplement, not reproduced
  from the current released AD 2 runner.

**Decision:** Table 1b may cite 38.6% as the published full-data reference,
clearly identified as the Appendix D `test_public` result with the scope caveat
above. Keep it outside the few-shot table and acquisition-strata ranking.

## Primary sources checked

- SubspaceAD published CVF page and PDF:
  <https://openaccess.thecvf.com/content/CVPR2026/html/Lendering_SubspaceAD_Training-Free_Few-Shot_Anomaly_Detection_via_Subspace_Modeling_CVPR_2026_paper.html>
- SubspaceAD arXiv version history and official code/license:
  <https://arxiv.org/abs/2602.23013> ·
  <https://github.com/CLendering/SubspaceAD/tree/ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7>
- FiCo AAAI version, arXiv and official code/license:
  <https://ojs.aaai.org/index.php/AAAI/article/view/32243> ·
  <https://arxiv.org/abs/2412.10115> ·
  <https://github.com/znchen666/FiCo/tree/69c5e9dec2c0f733decd946fdb455454d881438f>
- ReFP-AD paper/supplement and official code/license:
  <https://arxiv.org/abs/2608.01793> ·
  <https://github.com/CLendering/ReFP-AD/tree/0118c9899926e1394bd411aba23ee45f0b07407e>
- MVTec AD 2 dataset paper (eight scenarios, public-test description, and
  AU-PRO@0.05 definition): <https://arxiv.org/abs/2503.21622>
- Current project comparison contract: `configs/benchmarks/public_v13_20260929/PROTOCOL.md`.
