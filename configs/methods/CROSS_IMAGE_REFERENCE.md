# CRR implementation and development protocol

Implemented2026-09-14 following the authorized
[CRR design](../../skill_graph/analysis/2026-09-14_cross_image_reference_design.md).
The earlier DESIGN ONLY status records the previous turn, not the current
implementation/experiment authorization. No novelty or new performance claim.

## Locked comparison

CR0 exact R1 -> CR1 residual/local tail -> CR2 global class tail -> CR3 reference
geometry. Each adjacent YAML changes one Boolean only. CR1 versus historical
TF3 is not a single-component comparison. All trained CR stages omit L_pair.
The method module is src/fmad/methods/cross_image_reference.py, registered via
the existing CLI. No common feature, data, support-selection or evaluator edit.

Full8 seed0, normal16, seen TESTpublic development; frozen DINOv2 ViT-L14
layers11/23,448+672,256 bank tokens/source/resolution, fixed R1 MAD scales.
Primary AUPRO@.05, targeted pAUROC@.05, oracle F1 diagnostic. Joint success
requires AUPRO >= fresh CR0 and pAUROC > fresh CR0; inspect all per-class deltas.
No significance or final-holdout claim. Maximum32 class runs,64 updates per
trained class, no sweep/multi-seed,14400s per-class safety timeout. A runtime
failure is an invalid run, not evidence against the method family.

## Implementation details

- The129-parameter6->16->1 residual head starts with zero output, bounded at.5.
  CR1/CR2 mask the four geometry inputs; all trained stages extract the same
  six descriptors. Backbone, reference contents and scales remain unchanged.
- Reference radii exclude both the episode source and reference owner's source.
  Three distinct eligible sources require at least five supports for training;
  CR0 retains R1's two-support minimum. Fit uses15 sources; inference uses16.
- TBF's exact source/family/view/coordinate deck is reused. Reconstructed
  actual-change masks must hash-match that deck before pixel-mass weighting.
- Each update rescans current normal scores, uses stable fractional weighted
  top5% tail selection, samples128 positives and128 negatives per positive,
  and applies hinge margin.5; AdamW lr.01, weight decay0, gradient clip5.
- Cached descriptors and source/episode/mass arrays have an explicit128MiB
  CPU cap. The normal16/full64 upper-bound allocation is33MiB, including
  scalar descriptors and per-row metadata (not backbone banks or temporaries).
  Only compact pool hashes/counts persist after fit. Native predictions use
 4096-pixel chunks; do not retain six native descriptor maps or query maps.
- training.jsonl contains descriptor_cache progress with step=null, then64
  numeric optimization steps with loss/gradient and cross-source pair fraction.
  Distinguish those phases when counting optimizer updates. Maps/weights off.

## Verification before execution

Local existing fsad-tta-local-bcma: Torch2.6.0+cu124 / CUDA12.4.
Focused method/common-pipeline/collector/launcher/remote checks:152 passed,
5 skipped. Separately, real cached DINOv2 full-width/full16-bank CUDA smoke
passed CR0--CR3 at448/672 (two updates for trained stages; no benchmark metrics).
CR0/R1 exact parity, finite outputs, matching trained-stage descriptor/episode
hashes, support exclusions and frozen backbone were checked. Bash syntax passes.
Ruff is not installed in the container, so no Ruff result is claimed.

## Execution layout

User corrected the remote host to dsba3 physical GPUs0/1/2/3; local GPU0 stays
unchanged. Partitions:
local can/fabric; remote0 fruit_jelly/wallplugs, remote1 rice/walnuts,
remote2 sheet_metal, remote3 vial. Each lane runs CR0->CR1->CR2->CR3, one
worker/device. This is8 local plus24 remote class runs, not five macro votes.

The initial dsba2 connection blocker was resolved by the user's explicit dsba3
correction. Reuse the existing hun_fsad_tta_012 container and verify its data,
runtime and all four physical GPU UUIDs; do not create or replace a container.
Use scripts/run_remote.sh update-code for a validated immutable local snapshot;
do not update a checkout while it is in use. Recover compact results only.

The existing method-agnostic Bash is the launcher; no CRR-specific queue script:

```bash
# After the designated preset and all four physical GPU UUIDs are verified:
source configs/remote/dsba3.env
read -r -a crr_remote_devices <<< "${REMOTE_GPU_DEVICES//,/ }"
test "${#crr_remote_devices[@]}" -eq 4
bash scripts/run_experiments.sh --backend both \
  --container "$CONTAINER_NAME" \
  --configs configs/methods/cross_image_reference_CR0.yaml \
            configs/methods/cross_image_reference_CR1.yaml \
            configs/methods/cross_image_reference_CR2.yaml \
            configs/methods/cross_image_reference_CR3.yaml \
  --labels CR0 CR1 CR2 CR3 \
  --local-device GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b \
  --local-objects can fabric \
  --remote-devices "${crr_remote_devices[@]}" \
  --remote-objects fruit_jelly rice sheet_metal vial wallplugs walnuts \
  --data-root /data/mvtec_ad_2 --remote-data-root /data/mvtec_ad_2 \
  --output-root /results/cross_image_reference_20260914 \
  --log-dir results/experiment_launch/cross_image_reference_20260914 \
  --split test_public_seen_development --timeout 14400 --workers 1
```

That combined command is a fresh-launch template, not an instruction to rerun
already active/completed local lanes. If local starts while remote connection
details are pending, use the same generic launcher with --backend remote for
the remaining24 runs, preserving the same remote objects/config order/root.
Actual current launch status and provenance belong in skill_graph/current.md.
Use existing experiments.py collect and the canonical comparison Markdown/CSV
for completed evidence; do not append unscheduled runs as measured results.
