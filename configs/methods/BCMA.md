# BCMA local development ablation

B0 is exactly the existing contextual A0 cosine nearest-neighbor scorer. B1
learns a global diagonal metric; B2 adds center-excluded ring context; B3 adds
normal self-correspondence regularization. No backbone parameters are trained.
Every stage retains the same support-token bank and full-bank minimum.

Weights are `1 + 0.2*tanh(global + context)`, initialized to one. The pointwise
distance bound is not a guarantee of metric non-regression. Selection uses
detached exact full-bank argmin followed by a differentiable selected distance.

Training uses 64 joint-resolution steps, cycling sources within each epoch and
alternating synthetic families between epochs. Both resolutions see every
family/source. Losses act on the native-resolution mean score, not branch
logits. Clean scores are anchored to detached baseline scores; shifted-normal
scores target clean baseline scores. Synthetic positives are ranked above hard
and random normal pixels, including detached hard scores from clean views.
The detached normal terms avoid retaining multiple full-map autograd graphs;
negative pixels in synthetic images receive ranking gradients. B3 supervision
is same-source transformed correspondence, not cross-instance ground truth.

Protocol: seed 0, random 16 normal supports, DINOv2 ViT-L/14 layer 23,
448+672, no masking/rotation, test_public (seen development), common float16
streaming evaluator. Primary AUPRO@0.05; secondary partial AUROC@0.05. Oracle
F1 is diagnostic. Per-object timeout 3600 seconds; stop queue on execution error.
No hyperparameter sweep, dense map persistence, checkpoints, or remote execution.

Run through `scripts/experiments.py run --configs` with the four
`configs/methods/bcma_B0.yaml` through `bcma_B3.yaml` paths, `--labels B0 B1 B2 B3`,
the explicit `--objects`, `--data-root`, `--split test_public_seen_development`,
and `--output-dir <new-directory>`. See `docker/local-bcma.md` for the common CLI.
The output directory must be new to preserve previous runs. Each class emits
metrics, config, manifest and training logs through the common pipeline.
