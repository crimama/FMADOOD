# FlowTTE Current Structural Problem Diagnosis

Date: 2026-07-08

Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

## Scope

This diagnosis summarizes the current structural problem of FlowTTE on MVTec
AD2 single-image all-eight public objects. It is based on the existing
component, register, DVT, H+ backbone, SuperADD-aligned setting, and H+
priority diagnostic artifacts. No new experiment was run.

The current strongest branch is:

```text
DINOv3-H+/16 patch features, layers [7,15,23,31]
-> DVT-style position-mean denoise, alpha=1.0
-> NF latent projection
-> fixed 16-shot support latent bank, no TTE
-> patch nearest-neighbor distance + weak density
-> anomaly map
-> optional class-agnostic morphology
```

## Main Diagnosis

The current bottleneck is not a single weak class or a missing per-class
hyperparameter. The bottleneck is the map-level conversion from a reasonably
good continuous patch ranking into a stable, foreground-localized binary
segmentation map.

Evidence:

- H+ backbone alignment nearly closes the reported SuperADD AUROC gap:
  `0.836739` vs `0.839300`.
- The same run still trails reported SuperADD F1 by `-0.098686`:
  `0.527427` vs `0.626113`.
- Class-agnostic morphology improves F1 only from `0.527427` to `0.542316`,
  explaining about `0.014888` of the `0.098686` F1 gap.
- Removing NF with an H+ identity feature-distance control does not improve
  the all-object mean: `0.832461/0.524804`, slightly below H+ NF
  `0.836739/0.527427`.

Interpretation:

```text
continuous ranking quality: mostly acceptable
binary/spatial localization quality: still insufficient
```

So the structural problem is not best described as "NF is bad" or "a few
classes need tuning." It is better described as:

> FlowTTE has a strong patch-ranking signal after H+ and DVT, but the score map
> is not sufficiently calibrated, foreground-aware, or spatially coherent to
> produce SuperADD-level F1.

## Component-Level Failure Map

### 1. TTE Expansion Is Not The Current Positive Mechanism

Earlier all-eight diagnostics showed static no-TTE memory is stronger than TTE
expansion. Expansion reintroduces the original risk: test-time memory updates
can absorb anomaly evidence and weaken ranking separation. This remains a
closed direction unless revisited through one class-agnostic anti-absorption
gate.

### 2. NF Projection Is Not The Main Mean-Metric Bottleneck

NF is not the source of a clear positive claim, but it is also not the dominant
current failure after H+ alignment.

Evidence:

- H+ NF latent: `0.836739/0.527427`
- H+ identity feature NN: `0.832461/0.524804`
- NF NLL-only in component ablation collapsed to `0.682066/0.225544`
- flow distance-only also underperformed the base

Interpretation:

```text
NF as standalone likelihood scorer: weak
NF as latent transform with distance+density: usable but not novel enough
NF removal: does not solve the F1 gap
```

The more precise issue is score geometry alignment: the latent space supports
ranking, but it does not enforce foreground-localized and spatially coherent
maps.

### 3. DVT Helps, But Not By Simply Compressing The Memory Bank

DVT-style position-mean denoise is a real positive diagnostic:

- no-DVT DINOv3-L base: `0.797743/0.437800`
- DVT alpha `1.0`: `0.825207/0.468348`
- H+ plus DVT alpha `1.0`: `0.836739/0.527427`

However, structural analysis showed:

- raw support feature variance ratio after denoise: `0.451323`
- feature leave-one-out ratio: `0.882623`
- latent leave-one-out ratio: `1.188963`

Interpretation:

```text
DVT compacts raw feature variation
but NF latent retrieval can re-expand nearest-neighbor geometry
```

So DVT should not be framed only as memory-bank volume compression. Its useful
mechanism is better framed as score-separation stabilization, especially on
query-side anomaly evidence.

### 4. Register Is Not A Strong Structural Routing Key Yet

Register-based structural variants did not improve the all-object mean:

- no-context DINOv3: `0.797743/0.437800`
- CLS soft w10: `0.805427/0.447118`
- register top-M4: `0.798846/0.434411`
- register-conditioned NF: `0.796554/0.436152`

Failure analysis showed CLS context separability is stronger than register on
6/8 objects:

- CLS mean bad-good min-distance delta: `+0.008771`
- register mean bad-good min-distance delta: `+0.001372`

Register-conditioned NF does improve density separation on many objects, but
that signal is not currently scoring-aligned enough to improve segmentation.

Interpretation:

```text
CLS: useful weak global context for soft calibration
register: not a direct memory routing key
register-conditioned NF: diagnostic signal, not current main mechanism
```

### 5. SuperADD-Like Settings Cannot Be Copied Wholesale

The SuperADD-style preprocessing fallback with ViT-L, tiling, resize factor,
brightness augmentation, and DVT alpha `1.0` degraded performance:

- previous DVT alpha `1.0`: `0.825207/0.468348`
- SuperADD-style ViT-L fallback: `0.762906/0.384584`

This means the current FlowTTE latent scoring is sensitive to feature-map
distribution changes. Matching settings blindly is not a structural solution.

The useful positive from SuperADD alignment was specifically the H+ backbone,
not the entire preprocessing bundle.

## Current Structural Problem Statement

The most defensible problem statement is:

> FlowTTE currently behaves as a strong DINOv3-H+ patch-distance anomaly ranker,
> improved by DVT-style denoising, but it lacks a class-agnostic mechanism that
> separates true local anomaly evidence from foreground/background nuisance and
> converts the continuous score field into spatially coherent segmentation.

This explains the observed pattern:

```text
AUROC close to SuperADD
F1 still far below SuperADD
morphology helps only marginally
NF removal does not fix the issue
register routing does not fix the issue
SuperADD-like preprocessing fallback can hurt
```

## What Not To Do Next

Do not continue with:

- per-class thresholds;
- per-class morphology;
- object-specific score weights;
- broad alpha sweeps;
- register-only top-M sweeps;
- NF NLL-only scoring;
- TTE expansion without a pre-registered class-agnostic anti-absorption gate.

Weak objects should remain analysis buckets and no-harm checks, not tuning
targets.

## Next Class-Agnostic Diagnostic

The next useful diagnostic should test one shared structural mechanism across
all eight objects:

```text
H+ DVT FlowTTE score map
-> class-agnostic foreground/background or score-field calibration
-> same evaluator on all eight objects
```

Recommended first diagnostic:

- keep H+ backbone, fixed support, DVT alpha `1.0`, NF latent scoring;
- add one unsupervised foreground/background suppression or score calibration
  rule learned only from support/test image statistics without labels;
- apply the exact same rule and parameters to all objects;
- evaluate all-object mean, weak-bucket gains, and strong-object no-harm;
- hard-stop if mean F1 does not improve beyond morphology-only `0.542316` or
  if AUROC drops materially.

Candidate measurements before running another benchmark:

- foreground/background score ratio;
- positive score coverage;
- connected-component count and fragmentation;
- overlap between high-score regions and patch norm/CLS similarity/background
  shortcut maps;
- whether errors are diffuse background false positives or broken foreground
  fragments.

## Claim Status

No strict method claim is supported now. The current result is a diagnostic
asset:

- H+ and DVT provide a real positive ranking signal.
- NF latent projection is not the main all-object bottleneck.
- The remaining problem is a class-agnostic score-field/localization structure,
  not class-specific hyperparameter tuning.
