# RegAD on MVTec AD2 — fixed evaluation protocol

This is a development comparison of the released RegAD reference on the
MVTec AD2 public test split, not a paper-ready final evaluation. It contains
15 configurations: seeds 0–2, 1/2/4/8/16 fixed normal supports per class, and
the same eight classes (`can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`,
`vial`, `wallplugs`, `walnuts`). Each configuration evaluates all eight
classes (120 class/config cells total). Seed/shot support identities are frozen
in the manifest named by that configuration; manifests are shared with the
corresponding existing AD2 support protocols.

## Reference model and selection caveat

The adapter uses the official RegAD source at commit
`5e2c1f8c18d302b0354471567846fee3ed2ff063` and the released 2-shot MVTec
bottle checkpoint. The checkpoint file SHA256 is
`ccc0a792083edc1506d53b3cdbdb9a1ede95437d80c30e4204ca9e181fedb152`; the
archive SHA256 is
`082900cd7ce4fe2d4413a64dc1b985d79a6e4aa7fca8fdc4d86cdbfd781ef9e7`.
The same source checkpoint is strictly loaded for every AD2 target class,
seed, and support shot count. Its original released-checkpoint selection used
MVTec source test image- and pixel-AUROC. This source-label selection is a
provenance caveat; it did not use AD2 evaluation labels. AD2 test labels and
masks are consumed only by the shared evaluator. Results from this exposed
public test split remain development evidence.

## Method and transfer conditions

The inference bridge keeps the released RegAD registration network and test
path: 22 augmented views per normal support image, positional Gaussian fitting
with unbiased covariance and a fixed `0.01 I` ridge, Mahalanobis anomaly
scoring, and Gaussian smoothing with sigma 4. Support feature extraction and
covariance computation use chunks. The shared evaluator receives maps resized
to each original image's spatial size. No anomaly maps or fitted weights are
saved.

The upstream MVTec transform resizes square images to 224 pixels, without a
crop or ImageNet normalization. AD2 includes non-square images, while the
released spatial transformer requires fixed 224×224 geometry. This transfer
therefore stretches each AD2 image to 224×224 before inference; it is a
documented dataset-geometry adaptation. The common evaluator and other shared
pipeline conditions remain fixed.

## Metrics and run lanes

The primary metric is AUPRO integrated to FPR 0.05; the secondary metric is
partial AUROC to FPR 0.05. Full AUROC and image AUROC are retained as supporting
metrics. Binary postprocessing is disabled. Use the split key
`mvtec_ad2_test_public_dev_analysis_no_label_selection`.

The 120 object/configuration cells run as four disjoint object shards. Every
lane executes the same 15 configurations with one worker, one assigned GPU
UUID, and its own output root:

| Lane | Assigned GPU | Objects | Output |
|---|---|---|---|
| 0 | `GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f` | `can`, `fruit_jelly` | `/workspace/results_remote/regad_ad2_table4_20260930/full_matrix/lane0` |
| 1 | `GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae` | `fabric`, `sheet_metal` | `/workspace/results_remote/regad_ad2_table4_20260930/full_matrix/lane1` |
| 2 | `GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639` | `rice`, `walnuts` | `/workspace/results_remote/regad_ad2_table4_20260930/full_matrix/lane2` |
| 3 | `GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62` | `vial`, `wallplugs` | `/workspace/results_remote/regad_ad2_table4_20260930/full_matrix/lane3` |

The shards have 242–290 public-test images each per configuration, based on
the frozen class image counts. This layout gives all four queues identical
configuration order and supports lossless shard combination after completion.

Each lane has its own output root. The pilot is a separate seed0/2-shot,
`can`-only validity run; it is not substituted for the full eight-class
seed0/2-shot result.
