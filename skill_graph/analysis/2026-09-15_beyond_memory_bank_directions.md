# Beyond the memory bank — direction search (2026-09-15)

Status: DESIGN. Written while CA1–CA4 (coreset / kNN-mean on the 4-scale R1 bank) run.

## What the last 24 h established about the *problem*, independent of method

1. Sub-patch defects dominate three of the eight classes: GT area medians can 0.006 %,
   fabric 0.012 %, wallplugs 0.038 % (can ≈ 12×12 px in 1024×2232). A 14-px ViT patch at
   448 input covers more than the whole defect. Raising input scale gave fabric +25 pp AUPRO
   (HR2) and the 4-scale mean MS1 56.75/76.43 — the first effect in weeks that exceeded the
   sampling-noise floor (0.86 pp). Localization granularity, not reference choice, was the
   bottleneck.
2. Nearest-neighbour scoring has an intrinsic coverage–absorption conflict: a denser normal
   bank lowers the normal tail (pAUROC +1.3) but lets sub-patch defects find near-duplicate
   normals (fabric AUPRO 62.8 → 46.0). Any min-distance estimator inherits this.
3. can (specular metal, over/under-exposed and shifted test variants) is ≤ 18.5 AUPRO for every
   frozen-feature method at every scale/bank; the only method above 20 is the trained
   low-rank adapter D3 (24.3). A learned component is required for can.
4. Everything correction/transport/gating-shaped on top of retrieval is closed (5 independent
   negatives, ECO/NCFC/CG). Masked feature prediction (P0–P2, 14–16 AUPRO) and contextual
   scoring (A1–A3, 5–15) failed catastrophically; flows on latent banks (73–76 pAUROC) trailed
   kNN. These are not to be re-run.

## Candidate families that do not score by distance to stored normal tokens

| # | Family | Why it attacks the observed failure modes | Main risk in the 16-shot shifted setting | Priority |
|---|---|---|---|---|
| A | **Teacher–student / reverse distillation** (STFPM, RD4AD, EfficientAD): student trained on the 16 normals to reproduce frozen teacher features; anomaly = discrepancy | Supervision is *dense* (every pixel of every crop is a target), so 16 images ≈ 10^7 targets — few-shot in images but not in samples. A CNN teacher (WRN50 layers 1–3, strides 4/8/16) localizes below the 14-px ViT patch. No bank ⇒ no absorption. | Student under-generalises to the lighting/shift test variants ⇒ normal false positives. Needs photometric + geometric augmentation; EfficientAD's ImageNet penalty would need external images (policy question). | **1** |
| B | **Region-coherent scoring with a segmentation prior** (SAM/SAM2 or DINOv2 feature clustering → region proposals; pool pixel evidence per region) | AU-PRO rewards region coverage; RoBiS (SAM + adaptive binarisation) leads MVTec AD 2 AU-PRO by ~13 pp over kNN-only. Orthogonal to the detector; applies to MS1 maps. | Earlier region-completion attempt was invalid; SAM not installed and the container is offline (weights must be brought in manually). Sub-patch defects can be swallowed by large regions unless pooling is max-like. | 2 |
| C | **Feature-space synthetic discriminator** (SimpleNet-style: Gaussian-perturbed normal features vs normal, small per-class head, no bank at test time) | Learned discrimination is the only thing that moved can (D3 24.3). Cheap; per-pixel head at any scale. | Trivial solutions on 16 images; K1/K3-style heads on kNN maps gave only +1 pp before. Must be tested as a *bank-free* head, otherwise it is another add-on. | 3 |
| D | Normal density models on frozen tokens (PaDiM/flows) | — | Position-conditioned models break under the shift_1..3 variants; flows already trailed kNN here. | closed |
| E | Feature-space masked prediction / inpainting | — | P0–P2 catastrophic. | closed |
| F | VLM zero-shot (WinCLIP/AnomalyCLIP-like) | — | Language carries no information about 12-px scratches; MVTec AD 2 zero-shot baselines are weak. Useful only as an image-level prior. | closed |
| G | Diffusion reconstruction on 16 shots | — | Data-hungry; inference cost. | closed |

## Pilot RD-P1 (family A) — preregistered contract

* Objects: fabric (sub-patch, strong), can (hard), rice (fine-texture control), walnuts (control).
  Same 16 seed-0 supports, seen TEST public development split, same metric contract.
* Teacher: frozen WideResNet50-2 (`/model-cache/hub/checkpoints/wide_resnet50_2-95faca4d.pth`),
  layers 1–3 (strides 4/8/16). Student: RD4AD-style bottleneck + reverse decoder trained from
  scratch; loss = 1 − cos per layer. Input: smaller edge 1024, random 512×512 crops, photometric
  augmentation (existing `_apply_photometric`) + flips; 1 500 Adam steps, batch 8, lr 5e-3, no
  early stopping, no sweep. Score = sum over layers of (1 − cos) upsampled to image size.
* Arms: RD-CNN (above); RD-ViT (same student head on frozen DINOv2-L layers 11/23 at 1120) to
  separate "training" from "stride". Comparator: MS1 per class.
* Pass: fabric or can AUPRO above MS1 by more than the per-class sampling sd (2.1 / 1.0 pp)
  with pAUROC not below MS1 − 0.5; rice/walnuts within −1 pp. Report normal FPR at the LOO q99
  threshold as the over-fitting guardrail. Result policy: no post-hoc revision; failure enters
  the ledger and closes RD-P1.
* Explicit non-goals: no bank at test time, no kNN map input, no test-label calibration.

## Order

CA1–CA4 (running) → RD-P1 (GPU when free; local GPU0 can host it alongside CA at ~8 GB) →
if RD-P1 passes, combine with the MS1 detector only through a support-only fusion rule; if it
fails, family B (region prior) on MS1 maps is next, requiring SAM weights to be copied in.
