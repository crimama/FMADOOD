# ATAC-SB-v2 dsba3 support-only remote preregistration

- Fixed container: `hun_fsad_tta_012`; host `147.47.39.144:2222`.
- First run: `atac_sb_v2_stage_minus1_dsba3_support_20260824_v1`, host GPU 0,
  container CUDA slot 0, object `can` only.
- Support manifest and SHA-256 are frozen in the local preregistration.
- Remote runner reads only exact `train/good` support paths. It must not open
  `test_public`, `test_private`, or ground truth.
- If the can support gate passes, run the separately named all-eight support-only
  validity shards on host GPUs 0--3, two objects per GPU, with run IDs derived
  from `atac_sb_v2_stage_minus1_dsba3_support_20260824` and parameterized
  `HOST_GPU_ID`/`OBJECTS`.
- If can fails, stop without expansion. No epsilon or threshold rescue is
  permitted. Pull compact JSON/log/hash artifacts only and remove `anomaly_maps`.
- Remote outcome ceiling is `PROMISING_SUPPORT_VALIDITY`; no public metrics or
  SuperAD/RN comparison is authorized by this preregistration.

