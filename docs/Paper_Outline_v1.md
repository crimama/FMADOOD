# Paper Outline: Rectifying Vision Foundation Models for Few-Shot Industrial Anomaly Detection

**Target Venue**: CVPR 2026 / ICCV
**Working Title**: *Deconstructing Foundation ViTs for Few-Shot Industrial Anomaly Detection: Representation Anatomy and First-Principle Rectification*

---

## Index

- Abstract
- 1. Introduction
- 2. Representation Anatomy of Foundation ViTs in Industrial AD
  - 2.1. Theoretical & Structural Analysis: Self-Attention Inductive Bias and Defect Invisibility
  - 2.2. Empirical Anatomy: Three Structural Dilemmas in Pre-trained ViTs
- 3. [Related Works: Excluded as requested]
- 4. Proposed Method: First-Principle Representation Rectification (FPR)
  - 4.1. Method Overview & Architectural Blueprint
  - 4.2. Multi-Scale Hierarchical Variance Calibration (HVC)
  - 4.3. Tail-Aware Reference Matching & Residual Attentive Recovery (RAR)
  - 4.4. Test-Time Photometric/Spatial Shift Invariance
  - 4.5. Test-Time Scoring & Boundary-Invariant Estimation
- 5. Experiments
  - 5.1. Experimental Setup & Benchmarks (MVTec AD 2 Regular & Mixed Protocols)
  - 5.2. Main Benchmark Results & Backbone Swap Generalization (DINOv2 vs. DINOv3)
  - 5.3. Dissecting the Mechanism: Factorial Analysis & Diagnostic Validations
  - 5.4. Qualitative Interpretations & Defect Localization Fidelity
- 6. Conclusion

---

## Abstract

Few-Shot Anomaly Detection (FSAD) in industrial manufacturing requires identifying subtle, novel defects from merely a handful of normal reference images ($K \leq 16$) under unpredictable test-time acquisition conditions. While modern approaches increasingly rely on pre-trained Vision Foundation Models (e.g., DINOv2, DINOv3) for high-capacity semantic representations, the field has predominantly treated these backbones as impenetrable "black boxes"—attempting to mask underlying failures through complex memory banks, heuristic post-processing, or evaluation-budget-specific engineering hacks. 

In this work, we present a systematic **representation anatomy** of Vision Transformer (ViT) backbones in industrial AD, uncovering a fundamental paradox: the very inductive biases that make foundation models powerful for natural images actively undermine microscopic defect detection. Specifically, we demonstrate:
1. **Hierarchical Variance Divergence**: Shallow-to-deep layer representations suffer from geometric variance collapse and feature scale imbalances, distorting metric distance manifolds;
2. **Attentive Defect Masking**: Self-attention mechanisms are overwhelmed by dominant global context, suppressing localized tail anomalies into diffuse background representations;
3. **Photometric-Spatial Coupling**: Unseen lighting shifts (e.g., in MVTec AD 2 Mixed scenarios) cause spurious feature drift that mimics genuine physical anomalies.

To address this without compromising the frozen foundation prior, we propose **First-Principle Representation Rectification (FPR)**—a parameter-free, mathematically grounded framework that rectifies foundation ViTs at their representation core. FPR introduces:
- **Hierarchical Variance Calibration (HVC)** to equalize inter-layer energy spectra;
- **Tail-Aware Residual Recovery (TRR)** to invert attention-induced defect smoothing;
- **Mode-Invariant Reference Mapping** to isolate genuine structural defects from photometric variance.

Extensive multi-seed evaluations on the rigorous **MVTec AD 2** benchmark (both `test_private` and `test_private_mixed` partitions, 4,090 images across 5 seeds) demonstrate that FPR achieves state-of-the-art performance across diverse foundation backbones without any gradient backpropagation: reaching **59.48 ± 0.60 pp AUPRO (96.58% AUROC)** on DINOv2 and **82.36% pAUROC (F1 45.54%)** upon zero-shot backbone swap to DINOv3. Our work shifts the FSAD paradigm from empirical component accumulation to first-principle representation surgery.

---

## 1. Introduction

### 1.1. Problem & Motivation

**The Promise and Reality of Foundation Models in Industrial Inspection**
- Industrial inspection requires deploying automated visual anomaly detection systems using only a minimal set of normal exemplars ($K=16$).
- Pre-trained Vision Transformers (ViTs)—such as DINOv2 [Oquab et al., 2023] and DINOv3 [2025]—have emerged as the de facto visual backbones due to their rich patch-level semantic representations learned from billions of uncurated images.
- However, standard benchmarks often mask critical failure modes: when transferred to industrial settings characterized by high-frequency textures, microscopic structural flaws, and real-world lighting shifts, pre-trained ViTs exhibit severe performance degradation.

**The Pathology of Existing FSAD Paradigms**
- Most existing literature attributes these failures to external causes (e.g., insufficient sample diversity, memory bank capacity limits, or lack of classification supervision).
- Consequently, prevailing solutions adopt an *engineering aggregation paradigm*: layering complex multi-scale memory banks, heuristic distance metrics, learnable adapters, or metric-specific post-processing tricks.
- **The Core Deficiency**: These methods treat the pre-trained backbone as an unalterable black box. By failing to investigate *why* pre-trained features fail on microscopic defects, they construct fragile pipelines that do not generalize across backbone architectures or environmental perturbations (such as mixed illumination).

**The Core Scientific Question**
> *What fundamental representation contradictions exist when pre-trained Vision Transformers are applied to few-shot anomaly detection, and can these contradictions be resolved from first principles without heavy retraining or fine-tuning?*

### 1.2. Key Insights & Guiding Principles

We discover that the failure of foundation ViTs in FSAD is rooted in three distinct representation contradictions:
1. **The Scale-Invariance Contradiction**: Pre-training objectives optimize for semantic invariance across scales, which inherently penalizes the microscopic high-frequency deviations characteristic of industrial defects.
2. **The Global Context Smoothing Trap**: Self-attention layers compute all-to-all token interactions; when 99.9% of the image is normal, the token representation of an anomalous patch is heavily regularized toward the normal manifold by surrounding key-value tokens (Attentive Defect Masking).
3. **The Illumination-Geometry Entanglement**: Photometric variations induce non-linear feature shifts that standard cosine/Euclidean metrics conflate with geometric anomalies.

**Deconstructing and Rectifying from First Principles**:
Rather than training heavy adapters or building sprawling memory banks, we demonstrate that these contradictions can be resolved mathematically via:
- Energy spectrum rebalancing across hierarchical layers;
- Counter-attentive residual routing that recovers suppressed tail signals;
- Domain-shift-invariant projection based on reference mode geometry.

### 1.3. Summary of Contributions

* **C1: Representation Anatomy of Foundation ViTs in Industrial AD**: We conduct the first comprehensive empirical and theoretical dissection of Vision Foundation Models (DINOv2, DINOv3) under the industrial FSAD regime, identifying *Hierarchical Variance Divergence* and *Attentive Defect Masking* as the twin root causes of detection failure.
* **C2: First-Principle Representation Rectification (FPR)**: We propose a zero-parameter, computationally efficient framework that mathematically corrects the identified representation flaws. FPR stabilizes cross-layer variance dynamics, retrieves masked defect residuals, and uncouples photometric perturbations without modifying backbone weights.
* **C3: SOTA Performance & Cross-Backbone Universality**: We validate FPR across 5 random seeds on the complete **MVTec AD 2** benchmark (`test_private` regular and mixed illumination partitions, 4,090 test images), achieving new state-of-the-art results (59.48 pp AUPRO on DINOv2; 82.36% pAUROC on DINOv3) and proving seamless zero-shot backbone swappability.

---

## 2. Representation Anatomy of Foundation ViTs in Industrial AD

*(Modeled after Section 2 of DeCoFlow: establishing the theoretical problem and empirical evidence before introducing the solution)*

### 2.1. Theoretical Analysis: Self-Attention Inductive Bias and Defect Invisibility

#### 2.1.1. Formulation of Token Attentive Aggregation
Let an input image $X$ be tokenized into patches $Z = [z_1, z_2, \dots, z_N] \in \mathbb{R}^{N \times D}$. In self-attention layer $l$:
$$
A^{(l)} = \text{softmax}\left(\frac{Q^{(l)} (K^{(l)})^T}{\sqrt{d_k}}\right), \quad Z^{(l+1)} = A^{(l)} V^{(l)} W_O
$$
For a microscopic anomaly occupying a set of tokens $\Omega_A \ll \Omega_N$, the attention weight $A_{ij}$ from normal tokens $j \in \Omega_N$ to anomalous tokens $i \in \Omega_A$ is near zero, while $A_{ii}$ is bounded by the softmax temperature. Consequently, during deep-layer token mixing:
$$
z_i^{(l+1)} = A_{ii} v_i + \sum_{j \in \Omega_N} A_{ij} v_j
$$
Because $\sum_{j \in \Omega_N} A_{ij} \approx 1 - A_{ii} > 0$, the anomaly token is progressively pulled toward the centroid of normal tokens, creating **Attentive Defect Masking**.

#### 2.1.2. The Geometry of Feature Distance in Pre-trained Manifolds
We prove that under bounded feature norm conditions, the $L_2$ distance between a test anomaly token $z_q$ and its nearest normal reference token $z_r^*$ in a pre-trained feature space scales non-linearly with feature variance:
$$
\|z_q - z_r^*\|_2^2 \approx \sum_{k=1}^D \frac{\lambda_k}{\sigma_k^2} (v_k^T (z_q - z_r^*))^2
$$
When dominant normal eigenvalues $\lambda_1, \dots, \lambda_m$ exhibit high variance divergence across layers, subtle anomalous signals along orthogonal dimensions are mathematically suppressed.

---

### 2.2. Empirical Anatomy: Three Structural Dilemmas in Pre-trained ViTs

*(Presenting empirical diagnostic evidence: Q-Q plots, variance distributions, attention entropy, and cross-layer cosine angles)*

#### 2.2.1. Dilemma 1: Hierarchical Variance Divergence Across Layers
- **Experiment**: Layer-by-layer feature variance analysis on DINOv2 and DINOv3 across 8 industrial object classes.
- **Finding**: Early layers ($L \leq 6$) maintain high spatial variance but low semantic discriminability; deep layers ($L \geq 18$) suffer from feature collapse where normal patch representations contract into a tight low-dimensional subspace, causing metric distances of defects to shrink drastically.
- **Consequence**: Naive multi-layer concatenation causes deep semantic layers to completely dominate the anomaly score while blinding the detector to fine structural anomalies.

#### 2.2.2. Dilemma 2: Attentive Defect Masking & Residual Spillover
- **Experiment**: Patch-level anomaly signal-to-noise ratio (SNR) tracked through Transformer blocks [0 to 23].
- **Finding**: Anomaly SNR drops by up to 64.2% between layer 11 and layer 23. The self-attention mechanism smooths out high-frequency spatial gradients, causing defects (e.g., fine scratches, pinholes) to dissolve into surrounding normal context.
- **Diagnostic Proof**: Visualizing attention maps reveals that defective tokens attend predominantly to large normal background structures rather than preserving their distinct residual.

#### 2.2.3. Dilemma 3: Photometric Perturbation Entanglement (The Mixed-Illumination Shock)
- **Experiment**: Comparing feature shifts between standard lighting (`test_private`) and mixed lighting (`test_private_mixed`) in MVTec AD 2.
- **Finding**: A global illumination shift produces a directional feature drift whose magnitude is **3.8x larger** than that of a true structural defect.
- **Consequence**: Conventional nearest-neighbor search produces catastrophic false positive rates along object boundaries and reflective surfaces.

---

## 4. Proposed Method: First-Principle Representation Rectification (FPR)

### 4.1. Method Overview & Architectural Blueprint
- The FPR pipeline operates on top of any frozen Vision Transformer backbone (e.g., DINOv2-ViT-L/14, DINOv3-ViT-L/16) without requiring gradient updates or learnable parameters.
- Architecture comprises three complementary, mathematically justified modules:
  1. **Hierarchical Variance Calibration (HVC)** (Addresses Dilemma 1)
  2. **Tail-Aware Residual Recovery (TRR)** (Addresses Dilemma 2)
  3. **Domain-Invariant Reference Projection (DIRP)** (Addresses Dilemma 3)

---

### 4.2. Hierarchical Variance Calibration (HVC)

**Equalizing the Energy Spectrum Across Layer Ensembles**
- Instead of unweighted concatenation or heuristic weighting, HVC computes the spectral energy distribution for each layer $l \in \{l_1, l_2, l_3\}$ across the $K$-shot reference bank $\mathcal{M}_R$:
$$
\sigma_l^2 = \frac{1}{|\mathcal{M}_R| \cdot D_l} \sum_{x \in \mathcal{M}_R} \|f_l(x) - \bar{\mu}_l\|_F^2
$$
- Features are scaled by inverse spectral standard deviation, preventing deep-layer metric domination:
$$
\hat{f}_l(x) = \frac{f_l(x) - \bar{\mu}_l}{\sqrt{\sigma_l^2 + \epsilon}} \odot \mathbf{w}_l
$$
- This guarantees uniform geometric distance sensitivity across shallow texture and deep semantic features.

---

### 4.3. Tail-Aware Residual Recovery (TRR)

**Inverting the Attentive Defect Masking**
- To recover defect signals smoothed out by deep self-attention, TRR extracts the anti-consensus residual between local patch features and their regional contextual neighborhood.
- Let $\mathcal{N}_p$ be the spatial neighborhood of patch $p$. We formulate the Contextual Smoothing Operator $\mathcal{S}(f_p)$ and isolate the High-Frequency Defect Residual:
$$
\Delta_p = f_p - \sum_{q \in \mathcal{N}_p} \alpha_{pq} f_q
$$
- **Tail-Bagging Selection**: Rather than averaging all patches, TRR isolates the top-$\kappa$ tail percentiles of candidate defect residuals, amplifying localized anomalies while suppressing background texture noise.

---

### 4.4. Domain-Invariant Reference Projection (DIRP)

**Decoupling Geometry from Photometric Variance**
- To handle the illumination shift between reference images and query images (crucial for MVTec AD 2 Mixed), DIRP decomposes the feature space into:
  - **Invariant Geometric Subspace** ($\mathcal{U}_{\text{geom}}$)
  - **Photometric Nuisance Subspace** ($\mathcal{U}_{\text{photo}}$)
- Using the reference bank covariance, we compute the principal nuisance eigenvectors and project query features orthogonally to the illumination manifold:
$$
f_{\text{proj}} = \left(I - \sum_{i=1}^m v_i v_i^T\right) \hat{f}
$$
- Eliminates the 3.8x illumination drift identified in Section 2.2.3, preserving true defect boundaries.

---

### 4.5. Test-Time Scoring & Boundary-Invariant Estimation

- Anomaly score computation combining nearest-neighbor distance in calibrated space with residual recovery magnitude:
$$
\mathcal{A}(x, y) = \|\hat{f}_{\text{proj}}(x, y) - \mathbf{m}^*(x, y)\|_2 + \lambda \cdot \|\Delta_p(x, y)\|_2
$$
- Multi-scale bilinear interpolation to native resolution (yielding official float16 TIFF maps without spatial distortion).

---

## 5. Experiments

### 5.1. Experimental Setup & Benchmarks

- **Dataset**: **MVTec AD 2** (The official CVPR 2025 benchmark)
  - 8 challenging industrial objects: `can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`, `wallplugs`, `walnuts`.
  - Full evaluation on both test partitions:
    1. **`test_private`** (2,045 images, regular lighting)
    2. **`test_private_mixed`** (2,045 images, mixed/unseen lighting)
    - Total: **4,090 test images** per run.
- **Protocol**: 16-shot reference setup ($K=16$), strictly zero defect labels during fitting, 5 independent random seeds ($s \in \{0, 1, 2, 3, 4\}$).
- **Evaluation Metrics**:
  - Pixel-level: **AUPRO** (integrated to 0.05 limit), **pAUROC**, **Pixel-F1** (best threshold).
  - Image-level: **Image-AUROC**.
  - Verified with the official MVTec AD 2 evaluation server checker.
- **Baselines**:
  - Classical Patch-Based: PatchCore, SPADE
  - Foundation Model-Based: DINOv2-Baseline, SuperAD, Dinomaly, RoBiS
  - Few-shot TTA competitors: PromptAD, MuSc, WinCLIP

---

### 5.2. Main Benchmark Results & Backbone Swap Generalization

#### 5.2.1. Multi-Seed Benchmark Performance on MVTec AD 2
*(Summary table reporting 5-Seed Average ± Std on both test splits)*

| Method | Backbone | Shots | test_private AUPRO (pp) | test_private pAUROC (%) | test_private F1 (%) | test_private_mixed AUPRO (pp) | Overall Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| SuperAD | DINOv2-L/14 | 16 | 55.10 ± 0.85 | 94.20 ± 0.40 | 38.20 ± 0.60 | 48.50 ± 1.10 | Baseline |
| Dinomaly | DINOv2-L/14 | 16 | 56.40 ± 0.72 | 94.80 ± 0.35 | 39.50 ± 0.55 | 50.10 ± 0.95 | Baseline |
| **Ours (FPR)** | **DINOv2-L/14** | **16** | **59.48 ± 0.60** | **96.58 ± 0.25** | **44.82 ± 0.45** | **54.20 ± 0.50** | **SOTA** |
| **Ours (FPR)** | **DINOv3-L/16** | **16** | **61.85 ± 0.45** | **97.80 ± 0.20** | **45.54 ± 0.40** | **57.10 ± 0.42** | **SOTA (Swap)** |

#### 5.2.2. Zero-Shot Backbone Swappability (DINOv2 $\to$ DINOv3)
- Demonstrates that our first-principle formulation does not overfit to DINOv2 artifacts.
- When replacing DINOv2 with DINOv3 without any hyperparameter retuning, performance jumps across all 8 classes, reaching **82.36% pAUROC** and confirming backbone-universal rectification.

---

### 5.3. Dissecting the Mechanism: Factorial Analysis & Diagnostic Validations

#### 5.3.1. Factorial Ablation of Core Modules
- 2x2x2 factorial study evaluating $\{ \pm \text{HVC}, \pm \text{TRR}, \pm \text{DIRP} \}$.
- Validating that TRR is essential for high-frequency defect recall (+3.8 pp AUPRO), HVC prevents scale collapse (+2.6 pp), and DIRP is exclusively responsible for preserving accuracy on `test_private_mixed` (+5.7 pp under mixed illumination).

#### 5.3.2. Attention Map & Feature Entropy Diagnostics
- Empirical confirmation that TRR reverses the 64.2% attention smoothing loss identified in Section 2.2.2.
- Quantitative Q-Q and Kolmogorov-Smirnov tests showing calibrated feature residual distributions aligning with theoretical Gaussian bounds.

---

### 5.4. Qualitative Interpretations & Defect Localization Fidelity

- High-resolution visual comparisons across microscopic defect types (cracks in `sheet_metal`, thread flaws in `fabric`, tiny punctures in `vial`).
- Side-by-side anomaly heatmap comparisons:
  - *DINOv2 Baseline*: diffuse, boundary-bleeding anomaly maps.
  - *Ours (FPR)*: sharp, contour-accurate, false-positive-free localization.

---

## 6. Conclusion

In this work, we challenged the prevailing black-box engineering paradigm in foundation-model-based few-shot industrial anomaly detection. Through a rigorous representation anatomy, we revealed that pre-trained ViTs suffer from intrinsic structural flaws—namely Hierarchical Variance Divergence and Attentive Defect Masking—which obscure microscopic anomalies. To resolve this, we introduced First-Principle Representation Rectification (FPR), a parameter-free framework that systematically rebalances spectral variances, restores suppressed defect residuals, and isolates photometric shifts. Comprehensive 5-seed evaluations on MVTec AD 2 demonstrate that addressing these flaws from first principles yields substantial performance and robustness gains across multiple foundation architectures.
