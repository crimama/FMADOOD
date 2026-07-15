from __future__ import annotations

from pathlib import Path

import pytest

from flow_tte.darc_gate1_aggregate import (
    DarcAggregateError,
    _matching_provenance,
    _validate_bootstrap,
)
from flow_tte.darc_gate1_artifacts import write_json
from flow_tte.darc_gate1_provenance import SeedProvenance


def _provenance() -> SeedProvenance:
    return SeedProvenance(
        method_sha256="method",
        code_config_sha256="code",
        dataset_inventory_sha256="dataset",
        split_inventory_sha256="split",
        selection_sha256="selection",
    )


def test_bootstrap_accepts_digest_without_redundant_method_hash(tmp_path: Path) -> None:
    # Given
    provenance = _provenance()
    sources = {f"train/good/{index:03d}.png" for index in range(16)}
    path = tmp_path / "bootstrap_inputs.json"
    write_json(
        path,
        {
            "schema": "darc-stratified-source-bootstrap-v1",
            "seed": 20260710,
            "replicates": 10000,
            "provenance_sha256": provenance.digest(),
            "rows": [{"source": source} for source in sorted(sources)],
        },
    )

    # When / Then
    _validate_bootstrap(path, provenance, sources)


def test_nonbootstrap_artifact_still_requires_method_hash() -> None:
    # Given
    provenance = _provenance()
    payload = {"provenance_sha256": provenance.digest()}

    # When / Then
    with pytest.raises(DarcAggregateError, match="artifact provenance mismatch"):
        _matching_provenance(payload, provenance, Path("source_metrics.json"))
