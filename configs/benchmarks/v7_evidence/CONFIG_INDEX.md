# v7_evidence Config Index

Complete index of experiment configs for FSAD-TTA v7_evidence benchmark suite.

## Summary

- **Total configs**: 107
- **E1 (seeds)**: 16 configs + 8 manifests  
- **E2 (ungated xdata)**: 10 configs
- **E4 (VisA noise)**: 2 configs
- **E5 (noise mad)**: 1 config
- **E6 (brightness5)**: 2 configs + README
- **E7 (token loo)**: 2 configs
- **E8 (heldout comparators)**: README only
- **E9 (xdata baselines)**: 61 configs
- **E10 (ad2 strata)**: 8 configs
- **I1 (feature pooling)**: 4 configs

## Validation Status

- YAML syntax: 94/94 ✓
- Protocol validation: 94/94 ✓  
- E1 Manifests: All verified ✓

---

## E1_seeds/ - Fixed Reference Manifests

**Manifests**: 8 JSON files with support path selections
- Generated via `select_support_paths(sorted(pool), shots, seed)`
- Objects: can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts
- 2-shot nested: first 2 entries of 16-shot list

**Configs**: 16 YAML files
- Base: void_s02_{gated,all}.yaml, void_s16_{gated,all}.yaml
- Change: seed and fixed_reference_manifest
- Pattern: e1_seed{1,2,3,4}_s{02,16}_{gated,all}.yaml

---

## E2_ungated_xdata/ - Ungated Shallow Mode

**Configs**: 10 YAML files
- Base: fewshot MVTec/VisA dinov2 configs
- Change: method: fusion_factorial, method_config.shallow_mode: all  
- Pattern: e2_{mvtec,visa}_s{01,02,04,08,16}_all.yaml

---

## E4_visa_noise/ - VisA Noise Remediation

**Configs**: 2 YAML files
- e4_visa_rsg3.yaml: Unchanged reference_matching
- e4_visa_guard.yaml: + noise_guard config

---

## E5_noise_mad/ - MAD Score Normalization

**Configs**: 1 YAML file
- NA_mad.yaml: score_normalization: mad

---

## E6_brightness5/ - Brightness Corruption Robustness

**Configs**: 2 YAML + README
- e6_NA_gated.yaml, e6_NR_lowres.yaml: Unchanged
- README.md: Data root notes

---

## E7_token_loo/ - Per-Record Leave-Token-Out Calibration

**Configs**: 2 YAML files
- e7_s02_{gated,off}_token.yaml: + loo_estimator: token

---

## E8_heldout_comparators/ - External Baseline Comparison

**Contents**: README only
- References: patchcore_mvtec16.yaml, superad_mvtec16.yaml

---

## E9_xdata_baselines/ - Few-Shot Baselines

**Configs**: 61 YAML files
- Methods: regad, rd4ad, patchcore, fastrecon, rn_fmlk, superad (DINOv2/v3)
- Datasets: mvtec, visa
- Shots: 1,2,4,8 (all); +16 (superad_dinov3, visa)
- Pattern: e9_{dataset}_{method}_s{kk}.yaml
- Changes: dataset adaptation, protocol.shots, dataset_config cleanup for VisA

---

## E10_ad2_strata/ - Per-Stratum Evaluation on External Methods

**Configs**: 8 YAML files
- Methods: regad, rd4ad, patchcore, fastrecon, rn_fmlk, superad (DINOv2/v3), rsg3
- Base: Each method's source 16-shot config  
- Changes: protocol.shots=16, protocol.fixed_reference_manifest=configs/methods/VOID_support_16.json, evaluation.condition_strata=true
- Pattern: e10_{method}.yaml
- Validation: All configs pass YAML and protocol validation

---

## I1_pool/ - Local Feature Pooling Variants

**Configs**: 4 YAML files
- NA_pool.yaml: feature_pooling=avg3 on NA_gated baseline
- NA_lowres_pool.yaml: feature_pooling=avg3 on NA_lowres baseline
- NR_lowres_pool.yaml: noise_guard_action=lowres_pool on NR_lowres baseline
- e4_visa_guard_pool.yaml: noise_guard_action=lowres_pool on E4 VisA guard
- Changes: method_config additions for pooling and guard action
- Validation: All configs pass YAML and protocol validation

---

## Test Results

- test_loo_estimator.py: 15/15 PASSED ✓
- test_prepare_corrupted_seeded.py: 18/18 PASSED ✓
- test_noise_guard.py: All PASSED ✓
- test_reference_matching.py: All PASSED ✓
- test_fusion_factorial.py: All PASSED ✓
- **Total**: 121/121 PASSED ✓

---

All configs validated for YAML syntax, protocol compliance, and manifest correctness.

---

## E9 Update Notes

**16-shot configs for VisA**:
- `e9_visa_rd4ad_s16.yaml`: Derived from ad2_8shot (no ad2_16shot baseline exists)
- `e9_visa_rn_fmlk_s16.yaml`: Derived from ad2_8shot (no ad2_16shot baseline exists)
- `e9_visa_superad_s16.yaml`: Derived from ad2_8shot (no ad2_16shot baseline exists)

All MVTec configs cleaned: objects list removed from dataset_config.
