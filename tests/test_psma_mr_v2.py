"""Focused CPU contract tests for the approved PSMA_MR v2 revision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fmad.protocol import MethodContext


class TinyPSMAV2Backbone:
    feature_layers = (5, 11, 17, 23)
    model_name = "tiny_psma_v2"
    device = "cpu"

    def __init__(self, feature_dim: int = 8) -> None:
        self.smaller_edge_size = 8
        self.feature_dim = feature_dim
        self.model = torch.nn.Linear(3, feature_dim, bias=False)
        with torch.no_grad():
            self.model.weight.copy_(
                torch.arange(1, feature_dim * 3 + 1, dtype=torch.float32).reshape(feature_dim, 3)
                / 10.0
            )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.extract_calls = 0

    def prepare_image(self, image, *, resolution: int | None = None):
        selected = self.smaller_edge_size if resolution is None else int(resolution)
        if isinstance(image, (str, bytes, Path)):
            with Image.open(image) as opened:
                array = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
        else:
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(selected, selected), mode="bilinear", align_corners=False
        ).squeeze(0)
        return tensor, (selected // 2, selected // 2)

    def extract_features(self, image_tensor):
        self.extract_calls += 1
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)
        _, height, width = image_tensor.shape
        patch = image_tensor[:, : height - height % 2, : width - width % 2]
        patch = patch.reshape(3, height // 2, 2, width // 2, 2).mean(dim=(2, 4))
        tokens = patch.permute(1, 2, 0).reshape(-1, 3)
        base = self.model(tokens)
        return [base + (index * 0.01) for index in range(4)]

    def compute_background_mask(self, features, grid_size, **_kwargs):
        return np.ones(int(np.prod(grid_size)), dtype=bool)


def _write_image(path: Path, offset: int = 0) -> str:
    row = np.arange(12, dtype=np.int32)[:, None]
    column = np.arange(16, dtype=np.int32)[None, :]
    image = np.stack(
        [
            (row * 11 + column * 3 + offset) % 255,
            (row * 5 + column * 17 + offset * 2) % 255,
            (row * 19 + column * 7 + offset * 3) % 255,
        ],
        axis=-1,
    ).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return str(path)


def _method(backbone, **overrides):
    from fmad.methods.PSMA_MR import PSMA_MR

    config = {
        "revision": "v2",
        "resolutions": [8, 12],
        "adapter_enabled": True,
        "adapter_rank": 4,
        "steps": 4,
        "learning_rate": 0.001,
        "final_bank_cap": 8,
        "intermediate_bank_cap": 4,
        "train_tokens": 4,
        "nn_chunk_size": 4,
    }
    config.update(overrides)
    return PSMA_MR(backbone, config)


def test_v2_residual_clip_is_zero_safe_and_keeps_initial_u_gradient():
    from fmad.methods.PSMA_MR import _Rank4Residual

    adapter = _Rank4Residual(8, rank=4, seed=3, residual_epsilon=0.1)
    features = torch.randn(5, 8)
    output = adapter(features)
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(adapter.U.grad).all()
    assert torch.any(adapter.U.grad != 0)

    with torch.no_grad():
        adapter.U.fill_(2.0)
    before = torch.randn(5, 8)
    normalized = torch.nn.functional.layer_norm(before, (8,), eps=1e-5)
    latent = torch.tanh(torch.nn.functional.linear(normalized, adapter.V))
    residual = latent @ adapter.U.T
    residual_norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
    feature_norm = torch.linalg.vector_norm(before, dim=-1, keepdim=True)
    clipped = residual * torch.where(
        residual_norm > 0,
        (0.1 * feature_norm / residual_norm.clamp_min(torch.finfo(residual.dtype).tiny)).clamp_max(1.0),
        torch.ones_like(residual_norm),
    )
    expected = torch.nn.functional.normalize(before + clipped, dim=-1, eps=1e-8)
    assert torch.allclose(adapter(before), expected)
    zero = adapter(torch.zeros(2, 8))
    assert torch.equal(zero, torch.zeros_like(zero))


def test_v2_spatial_selection_is_deterministic_and_covers_coordinates():
    from fmad.methods.PSMA_MR import _spatial_coordinates, _spatial_stratified_indices

    coordinates = _spatial_coordinates((8, 8), 0)
    first = _spatial_stratified_indices(coordinates, 8, seed=17)
    second = _spatial_stratified_indices(coordinates, 8, seed=17)
    assert torch.equal(first, second)
    selected = coordinates[first]
    assert selected[:, 1].min() == 0
    assert selected[:, 1].max() >= 6
    assert selected[:, 2].min() == 0
    assert selected[:, 2].max() >= 6


def test_v2_uses_one_four_output_pass_and_fuses_eight_branches(tmp_path):
    support = [_write_image(tmp_path / "support" / f"{index}.png", index) for index in range(4)]
    query = _write_image(tmp_path / "query.png", 99)
    backbone = TinyPSMAV2Backbone()
    method = _method(backbone)
    method.fit(support, MethodContext("tiny", 0, 8, False, False))
    before_query = backbone.extract_calls
    score_map, image_score = method.predict(query)
    assert backbone.extract_calls - before_query == 2
    assert score_map.shape == (12, 16)
    assert np.isfinite(score_map).all()
    assert np.isfinite(image_score)
    metadata = method.object_metadata()
    assert metadata["revision"] == "v2"
    assert metadata["feature_layers"] == [5, 11, 17, 23]
    assert metadata["fusion"]["branch_count"] == 8
    assert all(branch["score"].startswith("raw_cosine") for branch in metadata["banks"].values())


def test_v2_calibration_is_clean_loo_only_and_schedule_is_balanced(tmp_path):
    support = [_write_image(tmp_path / "support" / f"{index}.png", index) for index in range(4)]
    method = _method(TinyPSMAV2Backbone(), steps=4)
    method.fit(support, MethodContext("tiny", 0, 8, False, False))
    metadata = method.object_metadata()
    calibration = metadata["calibration"]
    assert calibration["fitted_once_after_adapter"]
    assert calibration["clean_normal_only"]
    assert calibration["query_used"] is False
    assert calibration["ground_truth_used"] is False
    assert len(metadata["training"]["schedule"]) == 4
    assert [entry["source_index"] for entry in metadata["training"]["schedule"]] == [0, 0, 1, 1]
    assert [entry["resolution"] for entry in metadata["training"]["schedule"]] == [8, 12, 8, 12]
    assert metadata["training"]["synthetic_cache"] == "final_only"
    assert metadata["training"]["preserves_grid_coordinates"]
    assert all("query" not in json.dumps(item).lower() for item in calibration["branches"].values())


def test_v2_adapter_off_rebuilds_frozen_banks_and_calibration(tmp_path):
    support = [_write_image(tmp_path / "support" / f"{index}.png", index) for index in range(3)]
    context = MethodContext("tiny", 0, 8, False, False)
    method = _method(TinyPSMAV2Backbone(), adapter_enabled=False, steps=32)
    method.fit(support, context)
    metadata = method.object_metadata()
    assert metadata["adapter_enabled"] is False
    assert metadata["trainable_parameters"] == []
    assert metadata["calibration"]["fitted_once_after_adapter"]
    assert len(metadata["banks"]) == 8
