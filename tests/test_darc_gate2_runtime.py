from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false
from typing import TYPE_CHECKING, Dict, List, cast

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

import flow_tte.darc_gate2_runtime as runtime
from flow_tte.darc_feature_stream import DarcFeatureStream, ImageFeatures
from flow_tte.darc_gate2_pipeline_types import (
    CropLadderResult,
    QueryLadderAudit,
    QueryLadderResult,
)
from flow_tte.darc_gate2_provenance import JsonValue
from flow_tte.darc_gate2_runtime_fold import FoldCompactResult
from flow_tte.darc_gate2_runtime_support import (
    GroupResidualPopulation,
    build_group_residual,
    build_normal_references,
    build_source_audit,
)
from flow_tte.darc_gate2_runtime_types import Gate2RuntimeConfig
from flow_tte.darc_gate2_scoring_types import RungScores, SupportValidityAudit
from flow_tte.darc_geometry import ImageSize
from flow_tte.darc_tiling import NativeCrop

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


def _config(tmp_path: Path, smoke: bool = False) -> Gate2RuntimeConfig:
    data_root = tmp_path / "data"
    normal_root = data_root / "bottle" / "train" / "good"
    normal_root.mkdir(parents=True)
    pixels = np.full((128, 128, 3), 127, dtype=np.uint8)
    for index in range(16):
        Image.fromarray(pixels).save(normal_root / f"{index:03d}.png")
    return Gate2RuntimeConfig(
        data_root=data_root,
        output_root=tmp_path / "output",
        object_name="bottle",
        device="cpu",
        seeds=(0,),
        code_config_sha256="c" * 64,
        smoke=smoke,
    )


def _ladder(
    query_id: str,
    offset: float = 0.0,
    all_fallback: bool = False,
) -> QueryLadderResult:
    values: FloatArray = np.asarray(
        np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32) + np.float32(offset),
        dtype=np.float32,
    )
    fallback: BoolArray = np.asarray(
        np.ones(4, dtype=np.bool_)
        if all_fallback
        else [False, False, False, True],
        dtype=np.bool_,
    )
    g0_values: FloatArray = np.asarray(values + np.float32(1.0), dtype=np.float32)
    l0_values: FloatArray = np.asarray(g0_values if all_fallback else values, dtype=np.float32)
    l1_values: FloatArray = np.asarray(
        g0_values if all_fallback else values / np.float32(2.0),
        dtype=np.float32,
    )
    r1_values: FloatArray = np.asarray(
        g0_values if all_fallback else values / np.float32(3.0),
        dtype=np.float32,
    )
    support = np.ones((4, 3), dtype=np.bool_)
    scores = RungScores(
        g0=g0_values,
        g0_valid=np.ones(4, dtype=np.bool_),
        l0=l0_values,
        l1=l1_values,
        r1=r1_values,
        common_fallback=fallback,
        support_validity=SupportValidityAudit(support, support, support, support),
    )
    crop = CropLadderResult(
        crop_index=0,
        crop=NativeCrop(0, 0, 4, 4),
        token_shape=ImageSize(2, 2),
        scores=scores,
    )
    digest = (query_id.encode().hex() + "0" * 64)[:64]
    return QueryLadderResult(
        query_id=query_id,
        native_size=ImageSize(4, 4),
        selected_support_ids=("a", "b", "c", "d", "e"),
        crops=(crop,),
        registration_audit=(),
        audit=QueryLadderAudit(digest, "1" * 64, "2" * 64),
    )


def test_normal_references_use_complete_g0_and_nonfallback_local_tokens() -> None:
    references = build_normal_references((_ladder("one"), _ladder("two", 0.1)))

    assert references.g0.shape == (8,)
    assert references.l0.shape == references.l1.shape == references.r1.shape == (6,)
    assert np.all(np.diff(references.g0) >= 0)
    assert np.max(references.l1) < np.max(references.l0)


def test_normal_references_allow_empty_local_domain_when_all_tokens_fallback() -> None:
    # Given: finite normal LOO residuals whose every local token uses common G0 fallback.
    results = (_ladder("one", all_fallback=True), _ladder("two", 0.1, all_fallback=True))

    # When: the frozen full-G0/nonfallback-local reference domains are compacted.
    references = build_normal_references(results)

    # Then: G0 remains complete while the three unused local empirical domains stay empty.
    assert references.g0.shape == (8,)
    assert references.l0.size == references.l1.size == references.r1.size == 0


def test_source_and_group_audits_seal_equal_paired_populations() -> None:
    conditions = tuple(_ladder(name) for name in ("clean", "thin1", "thin2", "broad"))
    masks: List[BoolArray] = []
    for index in range(3):
        mask = np.zeros((4, 4), dtype=np.bool_)
        mask[index, index] = True
        masks.append(mask)

    audit = build_source_audit(conditions, tuple(masks))
    group = build_group_residual(
        GroupResidualPopulation(
            object_name="bottle",
            seed=0,
            source_ids=("source",),
            fold_indices=(0,),
            population_rows=({"source": "source", "fold": 0, "population_sha256": "a" * 64},),
        ),
        (conditions[0].concatenate_l0_residuals(),),
        (conditions[0].concatenate_l1_residuals(),),
    )

    assert all(len(value) == 64 for value in audit)
    assert group.l0_population_sha256 == group.l1_population_sha256
    assert group.l1_p999 < group.l0_p999


def test_smoke_seed_writes_one_compact_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, smoke=True)
    prepared = runtime.prepare_gate2_run(config).pending[0]

    class _FakeStream:
        def extract(self, _image: np.ndarray) -> ImageFeatures:
            empty = np.ones((1, 1, 1), dtype=np.float16)
            return ImageFeatures(ImageSize(1, 1), (), empty, (), ())

    def fake_fold(*_args: object, **_kwargs: object) -> FoldCompactResult:
        population: Dict[str, JsonValue] = {
            "source": "bottle/train/good/000.png",
            "fold": 0,
            "population_sha256": "a" * 64,
        }
        return FoldCompactResult(
            source_rows=({},),
            source_ids=("bottle/train/good/000.png",),
            fold_indices=(0,),
            population_rows=(population,),
            l0_residuals=(np.asarray([1.0, 2.0], dtype=np.float32),),
            l1_residuals=(np.asarray([0.5, 1.0], dtype=np.float32),),
        )

    monkeypatch.setattr(runtime, "run_gate2_fold", fake_fold)
    report = runtime.run_gate2_seed(
        config,
        prepared,
        cast("DarcFeatureStream", cast("object", _FakeStream())),
    )

    assert report.source_count == 1
    complete = prepared.expectation.seed_root / "complete.json"
    assert complete.is_file()
    assert '"smoke": true' in complete.read_text(encoding="utf-8")
