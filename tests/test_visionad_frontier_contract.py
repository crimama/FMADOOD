from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from scripts.flow_tte_map_metrics import MapMetricSet, histogram_binary_metrics, rank_histogram
from scripts.run_flow_tte_mvtec_ad1 import parse_args
from scripts.visionad_aligned_backbone import VisionADAlignedBackbone
from src import dinov2_loader
from src.flow_tte.metrics import average_precision

if TYPE_CHECKING:
    from pathlib import Path


class DummyDINO(torch.nn.Module):
    patch_size = 14
    embed_dim = 768

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(12))
        self.weight = torch.nn.Parameter(torch.ones(1))

    def get_intermediate_layers(
        self,
        batch: torch.Tensor,
        layers: list[int],
    ) -> list[torch.Tensor]:
        _ = batch
        return [torch.zeros((1, 784, 768)) for _ in layers]


def test_visionad_vitb_contract_freezes_encoder_and_yields_28_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dinov2_loader, "load_dinov2_model", lambda _: DummyDINO())
    backbone = VisionADAlignedBackbone(
        "dinov2_vitb14_reg",
        "cpu",
        448,
        392,
        (2, 5, 8, 11),
    )

    tensor, grid = backbone.prepare_image(np.zeros((300, 500, 3), dtype=np.uint8))
    features = backbone.extract_features(tensor)

    assert tensor.shape == (3, 392, 392)
    assert grid == (28, 28)
    assert backbone.feature_grid == (28, 28)
    assert backbone.embedding_dim == 768
    assert not backbone.model.training
    assert all(not parameter.requires_grad for parameter in backbone.model.parameters())
    assert [feature.shape for feature in features] == [(784, 768)] * 4


def test_visionad_backbone_rejects_invalid_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dinov2_loader, "load_dinov2_model", lambda _: DummyDINO())
    with pytest.raises(ValueError, match="divisible"):
        VisionADAlignedBackbone("dinov2_vitb14_reg", "cpu", 448, 391, (2, 5, 8, 11))


def test_average_precision_groups_equal_scores() -> None:
    labels = np.array([True, False], dtype=np.bool_)
    scores = np.array([0.5, 0.5], dtype=np.float32)
    assert average_precision(labels, scores) == 0.5
    assert average_precision(labels[::-1], scores) == 0.5


def test_metric_payload_exposes_requested_names() -> None:
    payload = MapMetricSet(0.1, 0.2, 0.3, 0.4, 0.5, "top", "fp16", 0.3).as_dict()
    keys = ("i_AUROC", "i_AUPRC", "p_AUROC", "p_AUPRC", "p_AUPRO")
    assert [payload[key] for key in keys] == [0.1, 0.3, 0.2, 0.4, 0.5]


def test_rank_histogram_preserves_order_for_scores_above_float16_range() -> None:
    predictions = [np.array([[1.0e15, 2.0e15, 3.0e15, 4.0e15]], dtype=np.float32)]
    masks = [np.array([[False, False, True, True]], dtype=np.bool_)]

    total, positive = rank_histogram(predictions, masks)
    metrics = histogram_binary_metrics(total, positive, ordered_codes=True)

    assert total.sum() == 4
    assert metrics.pixel_auroc == 1.0
    assert metrics.pixel_ap == 1.0


def test_ad1_cli_parses_visionad_frontier_contract(tmp_path: Path) -> None:
    config = parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "out"),
            "--backbone-model",
            "dinov2_vitb14_reg",
            "--preprocess-recipe",
            "visionad_official",
            "--image-size",
            "448",
            "--crop-size",
            "392",
            "--feature-layers",
            "2,5,8,11",
            "--calibration-sample-size",
            "4096",
            "--support-brightness-range",
            "0.8,1.2",
            "--dvt-denoise-mode",
            "position_mean",
            "--expansion-budget",
            "1.0",
        ],
    )
    assert config.backbone_model == "dinov2_vitb14_reg"
    assert config.feature_layers == (2, 5, 8, 11)
    assert config.calibration_sample_size == 4096
    assert config.support_brightness_range.min_factor == 0.8
    assert config.dvt_denoise_mode == "position_mean"
    assert config.expansion_budget == 1.0


def test_ad1_cli_context_defaults_preserve_no_context(tmp_path: Path) -> None:
    config = parse_args(
        ["--data-root", str(tmp_path / "data"), "--output-root", str(tmp_path / "out")],
    )
    assert config.context_source == "none"
    assert config.memory_context_source == "auto"
    assert config.context_mode == "auto"
    assert config.context_weight == 0.0


def test_ad1_cli_parses_cls_soft_memory_context(tmp_path: Path) -> None:
    config = parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "out"),
            "--context-source",
            "cls",
            "--flow-context-source",
            "none",
            "--memory-context-source",
            "cls",
            "--context-mode",
            "soft_penalty",
            "--context-weight",
            "10",
        ],
    )
    assert config.context_source == "cls"
    assert config.flow_context_source == "none"
    assert config.memory_context_source == "cls"
    assert config.context_mode == "soft_penalty"
    assert config.context_weight == 10.0
