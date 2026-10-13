# Source/Part-Conditioned Local Transport Full-Shot Stage -1 Preregistration

Date: 2026-08-11  
Protocol ID: `spct-fullshot-stage-minus1-20260811`  
Status: `PRE-REGISTERED / NOT RUN`  
Maximum verdict: `PROMISING_DIAGNOSTIC`

## 1. Boundary and custody

Target data are MVTec AD2 single-image `train/good` images only, rooted at
`/home/hunim/Volume/DATA/mvtec_ad_2`. The frozen objects are `can`, `fabric`,
`vial`, and `walnuts`. No `test_public`, `test_private`, ground truth,
filename condition tokens, real anomaly metric, query-conditioned fit,
test-time gradient, feature update, memory update, or TTA is allowed.

All candidate, calibration, and synthetic-fit rows are support-only. The
result must record `query_adaptation=false`, `parameter_update=false`,
`test_access=false`, `ground_truth_access=false`, and
`benchmark_comparable=false`; SuperAD and RN-FMLK fields remain `N/A`.

The historical Probe16 manifest is
`../2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`
with SHA-256 `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
The full population manifest is `full_train_good_paths.json`, with frozen
SHA-256
`58f6b249d82ac713c9a0051675244db5216ebbe995d18ecb123058c9499a592d`.
Both manifest hashes and the selected relative-path set hashes are written to
every run artifact before feature extraction.

Audited full `train/good` counts and SHA-path-mod-4 inventory are:

| object | total full support | SHA mod-4 inventory |
| --- | ---: | --- |
| can | 412 | 103 / 103 / 103 / 103 |
| fabric | 387 | 97 / 97 / 97 / 96 |
| vial | 291 | 73 / 73 / 73 / 72 |
| walnuts | 432 | 108 / 108 / 108 / 108 |

## 2. Paired Probe16 outer evaluation and budgets

The exact 16 historical support images per object form four SHA-ordered outer
Probe16 folds of four held sources. Both budgets score precisely the same held
source, signed nuisance code, anomaly family, mask, and patch grid in each
fold.

- `support16`: the other 12 Probe16 images are the fit support.
- `full_train_good`: every full-manifest `train/good` image except the four held Probe16
  sources is fit support: `408` can, `383` fabric, `287` vial, and `428`
  walnuts images per outer fold.

Thus full support is available for graph, retrieval, and tangent fitting but
never contains the held query source. Full-population SHA-mod-4 counts are
audited for custody, while Probe16 defines the paired external comparison.

For bounded but representative threshold fitting, at most 64 fit sources per
fold are selected by `(SHA256(relative_path), relative_path)` and used as
source-excluded normal calibration queries. For synthetic-margin fitting, at
most 32 fit sources are selected by the same rule. These caps affect only the
number of fit sources used as calibration or synthetic queries: every
fold-eligible support source remains in the graph, retrieval memory, and
tangent displacement bank. The cap, actual selected count, and selection rule
are recorded per fold.

## 3. Frozen representation and normal nuisance bank

The backbone is frozen DINOv2 ViT-L/14 at resolution 672. Every image is
adaptively pooled to a `24 x 24` grid before graphing/scoring. Layer 5 rows
are L2-normalized retrieval keys `k`; layer 11 rows are L2-normalized values
`z`. No layer, pooling size, key/value assignment, or score fusion is tuned.

For each fit support image, all eight frozen signed nuisance endpoints are
generated: `exposure±`, `gamma±`, `contrast±`, and reciprocal
red/blue-white-balance±. The displacement for source `i`, patch `p`, and code
`c` is `d(i,p,c)=z_end(i,p,c)-z_clean(i,p)`. These are normal-only data.

## 4. Correspondence graph, retrieval, and score

For a fit node `a=(i,p)`, build a same-position cross-source graph: its
`k_graph=8` neighbours are the nearest layer-5 keys from other fit sources at
the same pooled patch `p`; exact ties use source path SHA order. Graph roots
with fewer than four distinct available sources use all available sources and
are marked; their rate is a validity diagnostic.

Let `G(a)` contain root `a` and its graph neighbours. First average each signed
normal displacement across `G(a)`, then take the rank-4 uncentered SVD of the
eight resulting mean displacement vectors. Its orthonormal columns are
`U_a`; available rank is used only when source cardinality makes rank four
impossible and is recorded. This is the local normal transport family

```text
T_a(alpha) = z_clean(a) + U_a alpha.
```

For query patch `q_p`, frozen layer-5 retrieval chooses the `k_retrieval=4`
nearest fit roots at the same `p`. For candidate root `a`, define

```text
e_a = z_query(p) - z_clean(a)
alpha_a = U_a^T e_a
r_perp,a = (I - U_a U_a^T) e_a
s_a = ||r_perp,a||^2 / (sigma_a^2 + 1e-6)
S_SPCT(q,p) = min_{a in retrieved(q,p)} s_a.
```

`sigma_a^2` is the fit-only median energy of graph-neighbour clean
differences plus the eight normal endpoint displacements. `alpha_a` and
retrieval identity are compact diagnostics only. Neither transport direction
nor magnitude is an anomaly score, and no query feature is transported.

## 5. Pseudo-query split

For every held Probe16 source and signed nuisance endpoint, construct a held
normal, a thin `scratch` anomaly, and a broad `blur` anomaly exactly once with
SHA-derived seeds. Scratch and blur masks are evaluation-only and never enter
fit data.

`R2_local_tangent_margin` uses only the deterministically selected, at most 32
fit-source CutPaste and chromatic-blotch synthetic pairs from a disjoint seed
domain. Synthetic key/value features are extracted once per support image and
reused across folds; a held source is still excluded from the fold that scores
it. A deterministic rank-32 randomized SVD (`n_iter=5`) is fitted to the
selected fit-normal orthogonal residuals. The margin coordinates are
`u=(V r_perp)^2`, and the same frozen `V` and rows fit the ordinary and
label-swapped nonnegative heads. The basis rank, seed, and hash are recorded.
It cannot open or tune on held scratch/blur masks. Graph, retrieval, tangent,
query, and backbone are frozen. All ten arms run for both budgets even if an
early diagnostic gate is weak.

## 6. Arms

All arms have identical held rows, calibration splits, and budget coverage.

| arm | definition |
| --- | --- |
| `R0_retrieval_endpoint` | frozen layer-5 retrieved candidates with the normalized layer-11 endpoint residual `||e||^2` |
| `R1_local_tangent_orthogonal` | primary graph-conditioned local tangent orthogonal residual `S_SPCT` |
| `R2_local_tangent_margin` | R1 plus the fit-only CutPaste/chromatic residual-margin metric |
| `N0_global_tangent` | same retrieval with one fit-only object/fold rank-4 tangent from the eight global mean code displacements |
| `N1_source_shuffle` | retained pooled part but deterministic derangement of retrieved source identity and its tangent state |
| `N2_part_shuffle` | retained source but deterministic pooled-position permutation of retrieval/tangent state |
| `N3_tangent_shuffle` | same graph/retrieval, rank-matched local tangent bases assigned by one fixed fold-level root permutation |
| `N4_margin_label_swap` | R2 margin fit with clean/defect pair ranks reversed |
| `O0_clean_key_retrieval` | forbidden oracle: unavailable clean held layer-5 key retrieves fit roots, shifted value is scored |
| `O1_same_source_part` | forbidden oracle: unavailable held clean source/pooled part and its normal tangent |

`O0` and `O1` are strictly excluded from threshold fitting, arm choice,
continuation decisions, and positive claims. They may only diagnose why a
negative candidate failed.

## 7. Metrics and gates

Primary metric: held synthetic patch `diagnostic_seg_pAUROC_0.05`, separately
for scratch/local and blur/broad, then macro-averaged across the four objects.
Also report AUPRC, fit-q95 normal FPR and defect TPR, graph
valid-root/source coverage, tangent configuration and displacement hashes, and
compact per-arm anchor/source/part selections. Retrieval Jaccard, top-k recall,
parallel-energy capture, and source entropy are exploratory diagnostics only if
available; their absence cannot invalidate this bounded run.

Report an oracle-gap diagnostic only:

```text
recovery = (pAUROC(R1)-pAUROC(R0)) / (pAUROC(O1)-pAUROC(R0))
```

when its denominator exceeds `.01`; otherwise it is `N/A`. It never promotes a
claim by itself. Use 10,000 paired hierarchical bootstrap draws (objects then
held sources).

Validity requires exact manifests/counts/folds, held-source exclusion,
normal-only tangent fit, fit-only calibration and margin pairs, equal arm
coverage, finite/nonconstant scores, no forbidden access, and no persisted
dense maps.

Mechanism evidence is deliberately permissive but falsifiable: R1 must exceed
R0 by `.005` macro pAUROC with at least 2/4 object wins, not lose to N0 by more
than `.005`, beat at least one of N1/N2/N3 by `.01`, retain local and broad pAUROC
at least `.55`, increase FPR by at most `.03`, and recover at least 10% of the
defined O1 gap. Strong evidence is `.02` macro gain, positive bootstrap lower
bound, 3/4 wins, and 25% recovery. Failure does not cancel execution of the
already preregistered remaining arms/budget; it changes only interpretation.

The margin gate requires a finite, nonconstant fitted PCA basis and ordinary
and label-swapped objectives. R2 is positive evidence only if it improves R1
and N4 by `.01` macro (or has a positive paired lower bound), worsens normal
FPR by at most `.03`, and is nonnegative on both held families. A failed R2
does not erase frozen R1 evidence.

## 8. Execution queues

After code, manifest, and local contract verification, queues are strictly
sequential within each GPU and all arms are included in each object/budget job:

| queue | machine/GPU mapping | jobs in order |
| --- | --- | --- |
| A | dsba3 host GPU 3, container `cuda:0` | `vial/support16`, `vial/full_train_good`, `walnuts/support16`, `walnuts/full_train_good` |
| B | dsba5 host GPU 3 (container-visible physical 0), container `cuda:0` | `fabric/support16`, `fabric/full_train_good`, `can/support16`, `can/full_train_good` |

Each queue requires idle/preflight, immutable snapshot hash, full-manifest
filename-list SHA match, and compact shard pull before aggregation. A failed
object is retried only with the same frozen inputs; no parameter sweep or
selective rerun is allowed.

The execution backend may use a numerically parity-checked CUDA
implementation of the frozen graph, retrieval, tangent, and score operators.
This does not change candidate sets, tie ordering, ranks, equations, or gates.
`score_batch_rows=262144` is an execution-memory bound, and the tangent
displacement content hash is computed once per fold before CUDA upload.
