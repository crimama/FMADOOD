# Public v13 support-only threshold protocol

Status: fixed design for the `test_public` 16-shot SegF1/ClassF1 extension.

## Threshold rule

For each method, seed, and object, select the same 16 normal images from the
declared support manifest used by its 16-shot run. For each support image
`x_i`, instantiate a fresh copy of the configured backbone and method, fit on
the other 15 support images, and predict the map for `x_i`. No labels or masks
are opened in this calibration stage. Pool every finite float32 score pixel
from the 16 leave-one-out maps and calculate NumPy's linear 95th percentile
`q95`. Freeze the object threshold as:

```text
T = 1.421 * q95
```

At evaluation time, fit a fresh model on all 16 support images and predict
`test_public`. Binarize each native score map with the strict rule
`score > T`; apply no morphology. SegF1 is the pooled pixel F1 for each object.
ClassF1 is image-level F1 for each object, where an image is predicted
anomalous if its thresholded map contains any positive pixel. Macro values are
means over the declared object scope. Per-object sidecars store F1 in `[0, 1]`;
paper tables can report these values multiplied by 100.

The evaluator reads test masks only after `T` is frozen and the query map has
been predicted. Masks and image labels never reach method fitting, support
calibration, or prediction. The threshold is per object, method, and seed;
there is no cross-method score normalization or label-selected threshold.

## Precedent and tradeoffs

The multiplier and strict comparison preserve the project's documented
SuperADD rule: pooled normal-pixel p95 times 1.421, followed by `score > T`
with no SuperADD morphology. The precedent used additional held-out
`train/good` images. This extension replaces those images with leave-one-out
predictions over only the registered 16-shot support set.

Leave-one-out prediction prevents each calibration image from scoring against
itself in its fitted memory and reduces in-sample threshold bias. Each
calibration fit has 15 images while the final evaluation model has 16; this is
the cost of obtaining an out-of-fit normal score using only 16 support images.
The fixed multiplier is score-scale specific and its F1 performance under
16-shot leave-one-out calibration is unvalidated. SegF1 and ClassF1 are
therefore reported as support-threshold development metrics, separately from
the existing query-label oracle SegF1 diagnostic. No threshold is selected by
test-public labels.

The runner retains neither support nor query score-map files. It temporarily
holds flattened support scores in process memory to compute an exact pooled
quantile, then streams query maps into confusion counts. Its JSON sidecar
records support identities and hashes, each excluded support image, the
threshold inputs and value, query/mask coverage, and the F1 counts.
