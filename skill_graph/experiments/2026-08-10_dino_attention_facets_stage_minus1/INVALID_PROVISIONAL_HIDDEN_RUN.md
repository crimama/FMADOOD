# Invalid provisional runs

The `hidden/` directory contains a partial, pre-verification run started on
2026-08-10. It is excluded from every Stage A aggregate because it used the
wrong object set (`fruit_jelly,walnuts,vial,fabric`), evaluated only the
`hidden` arm, ran from the mutable workspace, and started before the
real-HuggingFace extractor and validity gates were verified. The process was
stopped without deleting its partial evidence. Valid Stage A shards use
separate immutable-snapshot result roots and the frozen
`can,fabric,rice,walnuts` allocation.

The partial `full_stage_a/`, `stage_a_full/`, `full_stage_a_shards/`, and
`full_stage_a_combined/` directories are excluded for
the same reason. They were launched by stale pre-verification processes from
the mutable workspace, two of them contended for local GPU 0 concurrently,
and they began before the fatal duplicate-`torchvision::nms` runner defect was
removed. Their exact process trees were stopped; the partial files remain only
as an audit trail and are never accepted as orchestrated shard inputs.

The capped `debug_can_cap20/` directory is likewise a non-full debug run and
is excluded. A separate successful two-image smoke lives only under the local
container's `/tmp` and is used solely as end-to-end runtime evidence.
