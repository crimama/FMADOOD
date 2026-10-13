# MVTec AD2 Main Table: Backbone-wise Baseline Comparison

## Table contract

- Dataset: MVTec AD2, eight objects. Metrics are class macro averages unless noted.
- `P16`: 16 normal references per object. `Pfull`: all `train/good` normal images;
  a held-out normal subset used only for threshold calibration is stated in the note.
- `Pub. F1` is a segmentation F1. Rows marked `raw oracle` use the best threshold
  on `test_public`; rows marked `calibrated` derive their threshold from normal
  training data. They must not be treated as interchangeable.
- A dash means the source did not provide a value under this metric contract.
- The completed primary paired comparison is the DINOv2-L/14 P16 block: the
  same 16 images, public population, evaluator, and metrics were used for
  SuperAD and RSG4. All other filled rows are backbone-aligned or external
  reported context until their baseline is rerun under the paired contract.
- Rows are grouped by backbone. Bold public values identify the stronger method
  only inside a matched-backbone and matched-budget block, not across blocks.

| Method | Backbone | Normal-data budget | Pub. pAUROC@0.05 | Pub. AU-PRO@0.05 | Pub. Seg-F1 | Private Seg-F1 | Private-mixed Seg-F1 |
|---|---|---|---:|---:|---:|---:|---:|
| SuperAD-16 (paired public; reported private) | DINOv2 ViT-L/14 | P16 | 0.761923 | 0.548353 | **0.380446 (raw oracle)** | 0.4718 | 0.4251 |
| **RSG4 (ours; paired public)** | **DINOv2 ViT-L/14** | **P16, same references** | **0.771061** | **0.591458** | 0.358678 (raw oracle) | — | — |
| Ours | DINOv2 ViT-L/14 | Pfull | — | — | — | — | — |
| RoBiS (reported) | DINOv2 ViT-B/14-reg | Pfull | — | — | — | 0.5100 | 0.4652 |
| Ours | DINOv2 ViT-B/14-reg | P16 | — | — | — | — | — |
| Ours | DINOv2 ViT-B/14-reg | Pfull | — | — | — | — | — |
| SuperADD (reported) | DINOv3 ViT-H+/16 | Pfull | 0.839300 | — | 0.626113 | 0.5742 | 0.5435 |
| **Ours Baseline1-core (K1)** | **DINOv3 ViT-H+/16** | **P16** | **0.839561** | **0.511573** | **0.530177 (raw oracle)** | **0.4594** | **0.4405** |
| Ours (SuperADD-budget diagnostic) | DINOv3 ViT-H+/16 | Pfull: 7/8 fit + 1/8 calibration | 0.833807 | — | 0.230264 (calibrated) | — | — |

## Official private table (server-returned, 2026-08-09)

The Baseline1-core row is now an official-server result, not a public shadow.
Class F1 is the image-level metric returned by the server; it has no public
shadow counterpart in this table.

| Method | Backbone | Budget | priv AU-PRO@0.05 | priv Seg-F1 | priv Class-F1 | mixed AU-PRO@0.05 | mixed Seg-F1 | mixed Class-F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SuperAD-16 (reported) | DINOv2 ViT-L/14 | P16 | — | 0.4718 | — | — | 0.4251 | — |
| RoBiS (reported) | DINOv2 ViT-B/14-reg | Pfull | 0.6725 | 0.5100 | — | 0.5965 | 0.4652 | — |
| SuperADD (reported) | DINOv3 ViT-H+/16 | Pfull | — | 0.5742 | — | — | 0.5435 | — |
| **Ours Baseline1-core** | **DINOv3 ViT-H+/16** | **P16** | **0.5440** | **0.4594** | **0.7350** | **0.5314** | **0.4405** | **0.7419** |

### Equal-budget reading

Against SuperAD-16, the only P16 row, Ours is `-0.0124` on private Seg-F1 and
`+0.0154` on private-mixed Seg-F1. Ours uses the stronger DINOv3-H+ backbone
and still trails on private. This is not a superiority claim at equal budget.

### Distribution-shift degradation

Ours degrades least from private to private-mixed among all four rows.

| Method | Budget | Seg-F1 Δ | relative |
|---|---|---:|---:|
| **Ours Baseline1-core** | **P16** | **-0.0189** | **-4.11%** |
| SuperADD | Pfull | -0.0307 | -5.35% |
| RoBiS | Pfull | -0.0448 | -8.78% |
| SuperAD-16 | P16 | -0.0467 | -9.90% |

AU-PRO@0.05 degrades `-0.0126` (`-2.32%`) for Ours versus `-0.0760`
(`-11.30%`) for RoBiS. Class F1 does not degrade at all (`+0.0069`).
This is currently a two-point observation on a submission-limited server and
is not yet a repeatable claim; a public-split shift proxy is required before
it enters the paper as a contribution.

### Threshold-transfer note

The binary maps used per-object `test_public` pixel-F1 oracle thresholds fixed
before private scoring (`configs/baselines/baseline1_core_public_thresholds.json`).
Public raw-oracle F1 `0.530177` transferred to private Seg-F1 `0.4594`, a loss
of `-0.0708` (`-13.4%`). The public-to-SuperADD F1 gap (`-0.0959`) and the
private gap (`-0.1148`) are close, so the residual gap is segmentation quality,
not threshold transfer. The separate `-0.318` normal-only calibration gap in
the Pfull diagnostic does not appear on this public-label threshold path.

Source: `skill_graph/experiments/2026-08-09_baseline1_private_submission/report.md`.

## Row provenance and interpretation

### DINOv2 ViT-L/14 — paired P16 block

The first two rows are the Main Table's primary few-shot comparison. Both use
the exact fixed support-16 manifest, `test_public/good,bad`, the all-eight
object population, DINOv2 ViT-L/14, and the same streaming metric contract.
RSG4 improves pAUROC@0.05 by `+0.009138` and AU-PRO@0.05 by `+0.043105`, while
trailing SuperAD by `-0.021768` oracle F1. The RSG4 noise guard fires on none of
the 1,084 public queries, so this clean-data row is exactly the RSG3 core with a
non-activating robustness safeguard; the clean advantage is not attributed to
the guard.

Source: `results/v7_evidence/E10_dsba3/comparison.md` and
`results/rsg4_fewshot_seed0_20260926/comparison.md`.
SuperAD's private/private-mixed F1 values (`0.4718/0.4251`) are copied from
the paper's official-server results. They are method-level reported context,
not a second run of the local public-shadow artifact.

Source: https://arxiv.org/abs/2505.19750

### DINOv2 ViT-B/14-reg — RoBiS block

RoBiS reports only official private metrics in the accessible release: private
AU-PRO@0.05 `0.6725` / Seg-F1 `0.5100`, and private-mixed AU-PRO@0.05 `0.5965` /
Seg-F1 `0.4652`. Its method uses category-wise learned models plus adaptive
binarization and SAM refinement, so it is not yet a paired few-shot comparison.
The two Ours rows are the outstanding experiments required for a backbone-wise
comparison.

Source: https://github.com/xrli-U/RoBiS

### DINOv3 ViT-H+/16 — SuperADD block

SuperADD's reported public/private/private-mixed Seg-F1 values are
`0.626113/0.5742/0.5435`; public pAUROC@0.05 is `0.839300`. The P16 Baseline1
row is a raw-1NN development/public-shadow result and retains the fixed
support-16 policy. The Pfull diagnostic aligns the normal-data budget and
normal-only threshold split, but it does not reproduce SuperADD's patching,
memory, or post-processing pipeline. Its public oracle F1 is `0.548522`, but
that value is deliberately excluded from the table because it uses public labels.

Sources:

- `skill_graph/methods/baseline1.md`
- `results/baseline1_knn5_default_ab_20260807_v5/summary.json`
- `skill_graph/experiments/2026-07-13_flowtte_ad2_fullnormal_superadd_thresholds/report.md`
- https://arxiv.org/abs/2605.14808

## Completion order

1. Run Ours with DINOv2 ViT-B/14-reg under P16, then Pfull, using the common
   public evaluator.
2. Reproduce RoBiS's public result or generate an explicitly matched Ours
   public protocol before claiming a public backbone-wise comparison.
3. Rerun the DINOv3-H+/16 P16 and Pfull Ours rows with the SuperADD-aligned
   continuous-map, threshold, and post-processing contract.
4. Submit frozen final rows to the MVTec evaluation server for private and
   private-mixed results; do not select configurations on those splits.
