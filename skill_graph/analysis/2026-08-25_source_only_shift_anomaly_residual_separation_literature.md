# Source-only shift–anomaly residual separation: literature and experiment design

Date: 2026-08-25  
Scope: MVTec AD2 single-image, source `train/good` plus deterministic synthetic
descendants only. No target query/update/scoring, TESTpublic/private, or GT was
accessed.

## Executive conclusion

There are many published ways to *encourage* separation, but they separate
different objects:

- shared versus domain-private information;
- group-shared versus instance-varying latent factors;
- linearly nuisance-predictive versus guarded coordinates;
- invariant versus transformation-equivariant representations;
- corrupted input versus paired clean-normal target;
- normal-product appearance versus synthetic defect/location tokens.

None directly proves the required statement: “using only source normals,
assumed-shift augmentation, and synthetic anomalies, the learned residual split
will preserve unseen real anomalies while removing an unseen acquisition
shift.” The missing ingredient is identifiability. Synthetic shift/anomaly
labels provide weak intervention supervision, but their validity is only as
strong as held-family transfer and shortcut controls.

The immediate recommendation is therefore not full meta-learning. Run one
bounded source-only separability-plus-predictive-actionability gate. The design
loop's best candidate was CRLP at `0.64` (`revise`); protected projection and
commutator variants scored worse and are not promoted.

## How prior work performs separation

| Family | Representative primary work | Separation device | What it can and cannot establish here |
|---|---|---|---|
| Shared/private domain factorization | [Domain Separation Networks, NeurIPS 2016](https://papers.nips.cc/paper_files/paper/2016/hash/45fbc6d3e05ebd93369ce542e8f2322d-Abstract.html), [DIVA, MIDL 2020](https://proceedings.mlr.press/v121/ilse20a.html), [mDSDI, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/b0f2ad44d26e1a6f244201fe0fd864d1-Abstract.html) | Separate shared/domain/class/residual encoders or latents; use domain confusion, domain prediction, covariance/orthogonality, reconstruction, and task losses. | Gives explicit compartments, but their semantics are induced by labels and reconstruction. A style-correlated defect may enter the private/domain component and be erased. DSN also uses unlabeled target data, so it is not our source-only setting. |
| Weakly supervised pair/group disentanglement | [ML-VAE, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11867), [Adaptive Group-VAE, ICML 2020](https://proceedings.mlr.press/v119/locatello20a.html), [Pairwise Similarities, AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/5754) | Group examples known to share some factors; aggregate shared posterior coordinates and leave changing factors in instance-specific coordinates. | Best conceptual fit for same-base interventions. Guarantees depend on trustworthy grouping, sparse/independent factor changes, and sufficient data; synthetic anomaly and shift generators can violate those assumptions. |
| Invariant/equivariant split | [Class–Pose Decomposition, AISTATS 2023](https://proceedings.mlr.press/v206/marchetti23b.html), [AugSelf, NeurIPS 2021](https://papers.nips.cc/paper/2021/hash/94130ea17023c4837f0dcdda95034b65-Abstract.html) | Make one channel invariant while a second channel predicts or transforms with the known augmentation/group action; add contrastive anti-collapse. | Better than globally erasing shift information because the equivariant branch remains auditable. Formal group guarantees do not cover arbitrary lighting, clipping, resampling, or defects whose visibility changes under the transform. |
| Linear concept erasure | [INLP, ACL 2020](https://aclanthology.org/2020.acl-main.647/), [LEACE, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract.html) | Repeatedly project into nuisance-classifier nullspaces, or solve a closed-form minimum-distortion affine erasure with zero concept cross-covariance. | Clean source-only baselines when shift labels are known. Guarantees are linear guardedness, not causal removal or anomaly preservation. Local global/local projection failures make this a control, not the new main mechanism. |
| Contrastive covariance/subspace decomposition | [cPCA, Nature Communications 2018](https://www.nature.com/articles/s41467-018-04608-8), [generalized-eigen BSS, JMLR 2003](https://www.jmlr.org/papers/v4/parra03a.html) | Find directions enriched in foreground residual covariance relative to matched background/control covariance. | Useful when controls are genuinely matched. Small control eigenvalues, content mismatch, and generator provenance can create spurious directions; this is close to already closed local subspace branches. |
| Cross-environment objectives | [MLDG, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11596), [IRM](https://arxiv.org/abs/1907.02893), [REx, ICML 2021](https://proceedings.mlr.press/v139/krueger21a.html) | Simulate meta-train/meta-test domain splits, require one optimal predictor across environments, or equalize/extrapolate environment risks. | These constrain optimization or predictors, not a sample-level shift/anomaly factor. They cannot by themselves say whether a residual is benign shift or anomaly. |
| Gradient conflict management | [PCGrad, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html), [CAGrad, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html) | Project conflicting task gradients or optimize a conflict-averse aggregate. | Can stabilize a later shift-removal/anomaly-preservation learner, but is not evidence that the factors are identifiable. A conflict may itself be the warning that they are entangled. |
| Paired clean restoration | [DRAEM, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_Detection_ICCV_2021_paper.html), [DSR, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910526.pdf), [DeSTSeg, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_DeSTSeg_Segmentation_Guided_Denoising_Student-Teacher_for_Anomaly_Detection_CVPR_2023_paper.html), [RealNet, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_RealNet_A_Feature_Selection_Network_with_Realistic_Synthetic_Anomaly_for_CVPR_2024_paper.html) | Corrupt a normal image/feature and predict or reconstruct its paired clean target; score the residual or a learned segmentation head. | Stronger shortcut defense than a direct synthetic classifier because the target is clean-normal restoration. It still does not prove nuisance/anomaly factorization or real-defect transfer. |
| Direct synthetic negatives | [SimpleNet, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.html), [GLASS, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8382_ECCV_2024_paper.php) | Produce calibrated feature/image-space synthetic negatives and learn a discriminative normal/anomaly boundary. | Efficient baseline, but the easiest solution may be generator, mask, boundary, distance, or texture recognition. Local CIDER-CF empirically exhibited this failure. |
| Synthetic generator factorization | [AnomalyDiffusion, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28696), [AnoGen, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11002.pdf), [SeaS, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Dai_SeaS_Few-shot_Industrial_Anomaly_Image_Generation_with_Separation_and_Sharing_ICCV_2025_paper.html) | Separate defect appearance from location and/or normal-product tokens, often with mask-guided attention or frozen diffusion backbones. | Addresses generator-side shortcuts, but generally assumes real anomaly supports and masks/boxes. It is a different supervision regime and does not itself supply a shift-invariant detector. |

## The most transferable lessons

1. **Use interventions as weak supervision.** Same-base `x`, `T(x)`, `A(x)`, and
   `T(A(x))` are more informative than independently sampled shifted and
   synthetic images because content is paired.
2. **Do not erase all shift-predictive information.** If a later learner is
   justified, keep an explicit shift-aware/equivariant audit channel while the
   scoring channel is made invariant.
3. **Anchor synthetic learning to the clean pair.** Predicting/restoring the
   paired clean feature is safer than a direct synthetic-vs-normal classifier.
4. **Held generator is necessary but insufficient.** Mask geometry, residual
   norm, spectrum, support, base identity, resampling, and texture provenance
   require their own matched probes/nulls.
5. **Decodability and actionability are separate.** A probe can decode shift
   versus anomaly while every correction damages exact RN geometry.
6. **Invariance is conditional.** When lighting, crop, or interpolation changes
   defect visibility, shift and anomaly are physically entangled; that cell
   should be stratified as nonseparable rather than optimized away.

## Local evidence that constrains the design

- Normal-only target structure predicted the damaging nuisance on only `1/8`
  objects, so clean target data alone does not identify the shift.
- FACT-RN was nearly identity and indistinguishable from a shuffle control.
- CIDER-CF separated synthetic anomalies but also decoded generator identity
  (`0.5749` balanced accuracy versus `0.20` chance) and beat no hard null.
- SO-CFM's synthetic hinge was inactive.
- GB-CMF already optimized a four-cell clean/shift/anomaly/shifted-anomaly
  refinement and destroyed geometry.
- CUG-RN-S2 failed all `48/48` margin and topology rows despite direct source
  objectives.

This closes direct four-cell refinement, passive/global subspace removal, and
another CUG optimizer retry. It does not close a frozen upstream feasibility
assay with a genuinely independent prediction target.

## Candidate audit and convergence

| Stage | Candidate | Evaluator | Main lesson |
|---|---|---:|---|
| 1 | CRLP | 0.64, revise | Decodability is measurable, but needs exact protocol and an independent actionability gate. |
| 2 | PCRE | 0.59, revise | Protected LEACE is still a projection retread and can protect generator provenance. |
| 3 | CRCG | 0.46, reject | The crossed interaction is one telescoping mixed finite difference; paired-oracle RN correction is not independent evidence. |

The loop stopped after two non-improvements. No representation-learning method
was promoted.

## Recommended first experiment: CRLP-Predict Stage -1

The experiment should answer two distinct questions and require both to pass.

### Data and split

- Source `train/good` images and deterministic descendants only.
- Shift families: existing exposure, calibrated Gaussian-noise control, and
  spatial-illumination bank, with any geometric family in a separate aligned
  stratum.
- Image-space anomaly families: existing Poisson patch, material blotch,
  structural scratch, and local-frequency synthesis. Feature-ascent anomalies
  remain a separate non-image control because composition order is undefined.
- Outer folds jointly exclude an object, a complete shift family, a complete
  anomaly family, and all descendants of each held base. Every normalizer,
  matcher, threshold, rank, and regression coefficient is fit inside the
  remaining training folds.

### Gate A: confound-matched decodability

Use frozen normalized residual direction as the treatment and retain residual
norm separately. Compare a regularized linear shift-versus-anomaly probe to:

- magnitude-only, mask-only, spectrum-only, support-only, interpolation-only,
  and generator-ID probes;
- family-label and base-pair shuffles;
- norm/support/spectrum-matched random residuals.

Required result: every held stratum exceeds its best hard null by a fixed
preregistered margin, while generator/confound probes remain near chance and
feature maps remain finite, nonconstant, and identity-preserving.

### Gate B: independent predictive actionability

Train a fixed-capacity ridge map on shifted-clean pairs to predict
`r_T=f(Tx)-f(x)` from the shifted frozen feature. Apply the frozen map to held
`f(T(A(x)))`; it never sees a synthetic anomaly during fitting. Compare:

- exact RN/no correction;
- shift-label shuffle and base-pair shuffle;
- magnitude-only and same-complexity random maps;
- shift-only LEACE/GED controls;
- paired clean-shift oracle as an upper bound, never as treatment evidence.

Require held-shift normal-tail reduction, anomaly residual cosine/norm
retention, exact-RN neighbor identity and distance-scale preservation, and
superiority over every hard null. Separately preregister that Gate-A held-cell
margin must positively predict Gate-B held-cell RN improvement; otherwise the
probe is an easy proxy and the branch stops.

### Stop rule

Any failed held object x shift x generator conjunction, hard-null tie,
generator/confound leakage, topology violation, or nonpositive prospective
Gate-A/Gate-B association closes this branch. Do not respond with a nonlinear
learner or hyperparameter sweep. A full pass licenses only a separately named
source-trained invariant/equivariant meta-learner design.

## Status

- Research artifact:
  `.omx/specs/deep-research-autoresearch-source-shift-anomaly-residual-separation/`
- Design verdict: `DESIGN_STAGE / REVISE`
- FMAD experiment verdict: not evaluated
- Benchmark comparable: `false`
- SuperAD metrics: `N/A`
- Public/test/private/GT access: none
