# Simple-RN Representative + DVT TESTpublic Log

- 2026-08-15: Pre-registered the TTA-free Simple-RN representative support +
  DVT all-eight TESTpublic diagnostic.
- 2026-08-15: Updated canonical aggregate contract to nine arms, including
  `N_DVT_R64_position_shuffle_RN`, and added the remote wrapper around
  `scripts/run_simple_rn_testpublic.py`.
- 2026-08-15: Verified the integrated runner, aggregate, wrapper, DVT controls,
  and adjacent metric paths (`33 passed`; scoped ruff, compile, shell syntax,
  and diff check clean).
- 2026-08-15: Launched immutable run
  `simple_rn_representative_dvt_testpublic_20260815_rn_dvt` on dsba3 GPU3
  (`can,fabric,walnuts`) and dsba5 GPU3
  (`fruit_jelly,rice,sheet_metal,vial,wallplugs`). Both first shards entered
  DINO feature extraction with live GPU utilization and no failure marker.
- 2026-08-15: `fruit_jelly` completed with `validity=true`; dsba5 advanced
  automatically to `rice`.  The first real shard exposed missing aggregate
  aliases for evaluator-native scalar names (`pooled_pauroc_0.05_float16`,
  `pooled_aupro_0.05`, `pooled_pixel_ap_float16`); the local final aggregator
  and a native-name regression test were corrected (`14 passed`).
- 2026-08-15: Completed and merged all eight shards.  Final macro pAUROC ranks
  full RN `0.749400`, R128 `0.748028`, random64 `0.745809`, and k-center R64
  `0.745542`.  Both DVT arms regress (`0.732581` full, `0.732942` R64), and
  DVT R64 is worse than its shuffled null (`0.739514`).  Verdict:
  `NO_SIGNAL_DIAGNOSTIC`.
- 2026-08-15: Fixed the remote wrapper so `TARGET=dsba3|dsba5` selects the
  matching preset in every remote mode, then added a regression test and
  re-pulled the missing dsba3 shards before aggregation.
