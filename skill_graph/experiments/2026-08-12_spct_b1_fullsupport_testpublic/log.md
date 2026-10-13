# SPCT B1 Full-Support TESTpublic Stage -1 Log

## 2026-08-12 — preregistered

- Frozen the full-support TESTpublic scope, the canonical remote-prefix
  manifest rewrite, the object counts, the resolution exception for
  `sheet_metal`, and the B0/B1/N0/RN_FMLK_full arm set.
- Sealed `full_train_good_paths.json` at SHA-256
  `ac084290bd40d171f12e876b45c3baf9b2269010cc2e62a0e91d7c4b80badd53`.
- Locked the gate hierarchy: B1 must beat RN_FMLK_full on macro pAUROC,
  remain within `0.01` macro F1/AUPRO loss, beat B0 and N0 by `0.005` macro
  pAUROC, and avoid any object-level pAUROC loss worse than `0.03`.
- Status remains `PRE-REGISTERED / PENDING` until launch verification.

## 2026-08-12 — verified and launched

- Runner, aggregator, remote queue, custody, and map-evaluation contracts
  passed 28 focused tests plus Ruff, Python compile, Bash syntax, and a local
  exact-manifest smoke.
- Both GPU3 preflights passed. dsba3 exposed 48,675 MiB free and dsba5 exposed
  24,021 MiB free; host and container GPU UUIDs matched.
- The first v1 submission was stopped before any result after dsba5 exposed a
  missing execution-only `FMAD_TORCHVISION_NMS_PATCH=0` environment flag.
  The method, data, arms, and metrics were unchanged.
- Reverified the corrected wrapper with 28 focused tests and launched immutable
  v2 runs `spct_b1_testpublic_dsba3_v2` and
  `spct_b1_testpublic_dsba5_v2`.
- Initial live audit: both runner processes are alive; dsba5 is executing DINO
  feature extraction on GPU3 and dsba3 is in the CPU-side manifest/data startup
  phase. No shard result exists yet.

## 2026-08-12 20:46 KST — 7/8 shards complete

- dsba5 completed its entire five-object queue; dsba3 completed `can` and
  `fabric`. All seven summaries are valid with complete coverage and no dense
  map persistence or failure marker.
- The final `walnuts` shard is running on dsba3 GPU3. Its elapsed time is about
  69 minutes; the comparable completed `can`/`fabric` shards took about
  95/104 minutes.
- Seven-object interim macro pAUROC/F1/AU-PRO is
  `0.565170/0.095602/0.072508` for B1 and
  `0.729584/0.283812/0.423006` for full-memory RN-FMLK. B1 is ahead of matched
  B0 by `+0.006422/+0.011923/+0.008587`, but trails RN-FMLK by
  `-0.164414/-0.188209/-0.350499`.
