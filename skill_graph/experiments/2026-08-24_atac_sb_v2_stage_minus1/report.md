# ATAC-SB-v2 Stage -1 support-only report

Date: 2026-08-24  
Protocol: `atac-sb-v2-stage-minus1-20260824`  
Run: `atac_sb_v2_stage_minus1_dsba3_support_20260824_v3`  
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## Scope and custody

This was a support-only validity diagnostic on dsba3 host GPU 0, container CUDA
slot 0, fixed container `hun_fsad_tta_012`. It read the exact 16 `can/train/good`
paths from the frozen same16 manifest
(`7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`) under
`/home/hunim/Volume/DATA/mvtec_ad_2`. No `test_public`, `test_private`, ground
truth, labels, query memory, or dense anomaly maps were accessed or retained.

## Frozen mechanism

ATAC-SB-v2 uses the existing exact two-level Haar high-pass `H2` on the resized
fp32 RGB canvas. `Hhat=H2/max(1,||H2||_inf)` is normalized per image over RGB
and spatial axes to preserve a shared phase direction while bounding the
elementwise magnitude. With `b=4*x*(1-x)` and `d=b*Hhat`, the branches are
`x+ = x+0.05d`, `x0=x`, and `x-=x-0.05d`; locality repeats the same frozen
direction at epsilon `0.10`. No clipping or epsilon selection was performed.

## Gate result

| Gate | Result | Decision |
| --- | ---: | --- |
| Branch finite and inside `[0,1]` | pass on 16/16 | pass |
| RGB midpoint identity `<=1e-6` | `2.98e-8` per support | pass |
| Finite nonconstant candidate maps | pass for all support rows | pass |
| Epsilon locality Spearman | minimum `0.9962776918` | pass (`>=0.90`) |
| Epsilon locality scale ratio | `0.9437916937--0.9763883688` | pass (`[0.5,2]`) |
| Support LOO scale ratio | `1.4624960413` | pass (`<=2`) |
| Intervention confound Spearman | barrier/activity up to `0.3797837472`; 4 rows exceed `0.30` | fatal |
| Unscaled linear dtype control | failed on all 16 rows under frozen conservative bound | fatal |

The aggregate is `KILL / FATAL_SUPPORT_GATE`. The confound failures show that
the candidate response is not separated from the frozen barrier/headroom and
high-pass activity controls on this support. The dtype-control failure is also
retained as observed evidence; no tolerance, epsilon, normalization, or
control rule was widened after results.

## Promotion decision

The first can shard failed, so the preregistered all-eight support-only
expansion was not launched. No SuperAD/RN metrics, public metrics, F1, AUROC,
or method claim is valid. The exact SB-v2 candidate is therefore
`KILL_FOR_CLAIM / NO_CONTINUE`; a future bounded-domain candidate would need a
new name and preregistration rather than a post-result repair of SB-v2.

## Artifacts and cleanup

- Local pull: `results/remote_runs/dsba3/atac_sb_v2_stage_minus1_dsba3_support_20260824_v3/`
- Compact aggregate: `.../aggregate/summary.json`, `.../aggregate/verdict.txt`
- Object gate: `.../gate/can/gate_summary.json`
- No `anomaly_maps/` directory exists under the local pull or remote result root.

