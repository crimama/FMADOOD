"""CPU-only contract tests for the PSMA_MR method.

These tests deliberately use a tiny deterministic backbone.  They exercise the
method boundary (support-only fitting, frozen feature extraction, multi-scale
preprocessing, and native-size predictions) without downloading DINO weights or
requiring a GPU.
"""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fmad.methods.base import Prediction
from fmad.pipeline import run_pipeline
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY


class TinyPSMABackbone:
    """Small frozen patch encoder with the shared backbone surface."""

    feature_layers = (0,)
    model_name = "tiny_psma"
    device = "cpu"

    def __init__(
        self,
        default_resolution: int = 8,
        feature_dim: int = 8,
        *,
        patterned_mask: bool = False,
    ) -> None:
        self.smaller_edge_size = default_resolution
        self.feature_dim = feature_dim
        self.patterned_mask = patterned_mask
        self.model = torch.nn.Linear(3, feature_dim, bias=False)
        with torch.no_grad():
            self.model.weight.copy_(
                torch.arange(1, feature_dim * 3 + 1, dtype=torch.float32).reshape(feature_dim, 3)
                / 10.0
            )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.prepare_calls: list[tuple[object, int | None]] = []
        self.extract_calls = 0
        self.mask_calls: list[dict] = []
        self.mask_outputs: list[np.ndarray] = []

    def state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.model.state_dict().items()}

    def set_resolution(self, smaller_edge_size: int) -> None:
        self.smaller_edge_size = int(smaller_edge_size)

    def prepare_image(self, img, *, resolution: int | None = None):
        """Return a fresh tensor; a resolution override must not mutate state."""
        self.prepare_calls.append((img, resolution))
        selected = self.smaller_edge_size if resolution is None else int(resolution)
        if isinstance(img, (str, bytes, Path)):
            with Image.open(img) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        else:
            array = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(selected, selected),
            mode="bilinear",
            align_corners=False,
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
        return [self.model(tokens)]

    def compute_background_mask(self, features, grid_size, **kwargs):
        self.mask_calls.append(dict(kwargs))
        count = int(np.prod(grid_size))
        if self.patterned_mask and kwargs.get("masking_type"):
            mask = np.ones(count, dtype=bool)
            mask[::2] = False
            self.mask_outputs.append(mask.copy())
            return mask
        mask = np.ones(count, dtype=bool)
        self.mask_outputs.append(mask.copy())
        return mask


def _write_image(path: Path, *, offset: int = 0, shape: tuple[int, int] = (10, 14)) -> str:
    height, width = shape
    row = np.arange(height, dtype=np.int32)[:, None]
    col = np.arange(width, dtype=np.int32)[None, :]
    array = np.stack(
        [
            (row * 11 + col * 3 + offset) % 255,
            (row * 5 + col * 17 + 2 * offset) % 255,
            (row * 19 + col * 7 + 3 * offset) % 255,
        ],
        axis=-1,
    ).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return str(path)


def _support(tmp_path: Path, count: int = 4) -> list[str]:
    return [
        _write_image(tmp_path / "support" / f"source-{idx}.png", offset=idx * 19)
        for idx in range(count)
    ]


def _context(seed: int = 7, resolution: int = 8) -> MethodContext:
    return MethodContext("tiny", seed, resolution, False, False)


def _config(**overrides) -> dict:
    config = {
        "resolutions": [8, 12],
        "adapter_rank": 4,
        "adapter_enabled": True,
        "steps": 2,
        "learning_rate": 0.01,
        "max_patches_per_source": 32,
        "max_bank_patches": 32,
        "max_train_patches": 32,
        "nn_chunk_size": 32,
    }
    config.update(overrides)
    return config


def _method(backbone: TinyPSMABackbone, **config):
    from fmad.methods.PSMA_MR import PSMA_MR

    return PSMA_MR(backbone, _config(**config))


def test_prepare_image_resolution_override_is_non_mutating():
    backbone = TinyPSMABackbone(default_resolution=8)
    image = np.full((5, 7, 3), 127, dtype=np.uint8)
    original_image = image.copy()
    original_default = backbone.smaller_edge_size

    first, first_grid = backbone.prepare_image(image, resolution=12)
    second, second_grid = backbone.prepare_image(image)

    assert first.shape[-2:] == (12, 12)
    assert first_grid == (6, 6)
    assert second.shape[-2:] == (8, 8)
    assert second_grid == (4, 4)
    assert backbone.smaller_edge_size == original_default
    np.testing.assert_array_equal(image, original_image)
    assert first.data_ptr() != second.data_ptr()


def test_fit_predict_is_finite_and_returns_original_size_maps(tmp_path):
    from fmad.methods.PSMA_MR import PSMA_MR

    support = _support(tmp_path)
    query = _write_image(tmp_path / "query" / "heldout.png", offset=91)
    backbone = TinyPSMABackbone()
    method = _method(backbone, steps=2)
    method.fit(support, _context())

    prediction = method.predict(query)
    if isinstance(prediction, Prediction):
        score_map, image_score = prediction.score_map, prediction.image_score
    else:
        score_map, image_score = prediction

    assert np.asarray(score_map).shape == (10, 14)
    assert np.isfinite(score_map).all()
    assert np.isfinite(image_score)
    assert np.asarray(score_map).dtype.kind == "f"
    assert isinstance(method, PSMA_MR)


def test_fitting_keeps_backbone_frozen_and_trains_rank_four_adapter(tmp_path):
    support = _support(tmp_path)
    backbone = TinyPSMABackbone()
    before = backbone.state()
    method = _method(backbone, steps=3)
    events: list[dict] = []
    method._training_log_sink = lambda _step, metrics: events.append(dict(metrics))

    method.fit(support, _context())

    assert all(torch.equal(before[name], value) for name, value in backbone.model.state_dict().items())
    adapter = getattr(method, "adapter", None)
    trainable = list(adapter.parameters()) if adapter is not None else []
    assert trainable, "PSMA_MR should expose its trainable adapter parameters"
    assert tuple(adapter.U.shape) == (8, 4)
    assert tuple(adapter.V.shape) == (4, 8)
    assert all(parameter.requires_grad for parameter in trainable)
    assert any(torch.isfinite(parameter).all() for parameter in trainable)
    assert any(torch.any(parameter.detach() != 0) for parameter in trainable)
    loss_events = [event for event in events if "normal_consistency" in event]
    assert loss_events
    assert {"normal_consistency", "synthetic_margin", "feature_anchor", "loss"} <= set(
        loss_events[-1]
    )
    for name in ("normal_consistency", "synthetic_margin", "feature_anchor", "loss"):
        assert np.isfinite(float(loss_events[-1][name]))

    metadata = method.object_metadata()
    assert metadata.get("adapter_rank", 4) == 4
    assert metadata.get("adapter_enabled", True) is True
    assert metadata["resolutions"] == [8, 12]
    assert metadata["feature_layers"] == [0]


def test_training_step_rebuilds_loo_bank_in_current_adapter_space(tmp_path, monkeypatch):
    psma_module = importlib.import_module("fmad.methods.PSMA_MR")
    support = _support(tmp_path, count=3)
    method = _method(
        TinyPSMABackbone(),
        resolutions=[8],
        steps=0,
        max_patches_per_source=8,
        max_bank_patches=8,
        max_train_patches=8,
    )
    method.fit(support, _context(seed=13))

    base_adapter = method.adapter
    assert base_adapter is not None

    class _SpyAdapter(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
            self.inputs: list[torch.Tensor] = []

        def forward(self, features):
            self.inputs.append(torch.as_tensor(features).detach().clone())
            return self.wrapped(features)

    spy_adapter = _SpyAdapter(base_adapter)
    method.adapter = spy_adapter
    original_distance = psma_module._chunked_cosine_distance
    scored_banks: list[torch.Tensor] = []

    def _capture_distance(queries, bank, chunk_size):
        scored_banks.append(torch.as_tensor(bank).detach().clone())
        return original_distance(queries, bank, chunk_size)

    monkeypatch.setattr(psma_module, "_chunked_cosine_distance", _capture_distance)

    first_input_index = len(spy_adapter.inputs)
    first_bank_index = len(scored_banks)
    first_loss, _ = method._training_step(0, 8, step=1)
    first_raw_bank = spy_adapter.inputs[first_input_index]
    first_scored_bank = scored_banks[first_bank_index]
    assert torch.isfinite(first_loss)
    assert torch.allclose(first_scored_bank, base_adapter(first_raw_bank))
    assert not torch.allclose(first_scored_bank, first_raw_bank)

    with torch.no_grad():
        base_adapter.U.fill_(0.35)

    second_input_index = len(spy_adapter.inputs)
    second_bank_index = len(scored_banks)
    second_loss, _ = method._training_step(0, 8, step=1)
    second_raw_bank = spy_adapter.inputs[second_input_index]
    second_scored_bank = scored_banks[second_bank_index]
    assert torch.isfinite(second_loss)
    assert torch.allclose(first_raw_bank, second_raw_bank)
    assert torch.allclose(second_scored_bank, base_adapter(second_raw_bank))
    assert not torch.allclose(first_scored_bank, second_scored_bank)


def test_paired_nuisance_reuses_transform_and_keeps_identity_views(tmp_path, monkeypatch):
    psma_module = importlib.import_module("fmad.methods.PSMA_MR")
    support = _support(tmp_path, count=1)
    method = _method(TinyPSMABackbone(), resolutions=[8], steps=0)
    transform = (1.11, 7.0, (0.91, 1.04, 1.08))
    sampled_transforms: list[object] = []
    applied_transforms: list[object] = []
    original_apply = psma_module._apply_photometric

    def _sample_transform(_rng, _strength):
        sampled_transforms.append(transform)
        return transform

    def _apply_transform(image, actual_transform):
        applied_transforms.append(actual_transform)
        return original_apply(image, actual_transform)

    monkeypatch.setattr(psma_module, "_sample_photometric_transform", _sample_transform)
    monkeypatch.setattr(psma_module, "_apply_photometric", _apply_transform)

    method.fit(support, _context(seed=17))

    assert sampled_transforms == [transform]
    assert len(applied_transforms) >= 3  # x plus both default synthetic families
    assert all(actual is transform for actual in applied_transforms)
    views = method._records[0][8]
    assert len(views.anomalies) == 2
    assert all(identity.shape == nuisance.shape for identity, nuisance, _, _ in views.anomalies)
    assert any(
        not torch.allclose(identity, nuisance)
        for identity, nuisance, _, _ in views.anomalies
    )
    pairing = method.object_metadata()["training"]["synthetic_pairing"]
    assert pairing["identity_included"] is True
    assert pairing["shared_photometric_transform"] is True


def test_fit_resets_adapter_and_banks_between_support_sets(tmp_path):
    support_a = _support(tmp_path / "a")
    support_b = _support(tmp_path / "b", count=3)
    backbone = TinyPSMABackbone()
    method = _method(backbone, steps=1)

    method.fit(support_a, _context(seed=3))
    first_metadata = copy.deepcopy(method.object_metadata())
    first_adapter = method.adapter
    method.fit(support_b, _context(seed=3))
    second_metadata = method.object_metadata()

    assert first_metadata != second_metadata
    assert method.adapter is not first_adapter
    assert len(support_b) == second_metadata.get("selected_reference_count", len(support_b))
    encoded = json.dumps(second_metadata)
    assert all(path not in encoded for path in support_a)


def test_same_seed_and_support_are_deterministic(tmp_path):
    support = _support(tmp_path)
    query = _write_image(tmp_path / "query.png", offset=37)

    first = _method(TinyPSMABackbone(), steps=2)
    second = _method(TinyPSMABackbone(), steps=2)
    context = _context(seed=11)
    first.fit(support, context)
    second.fit(support, context)

    first_pred = first.predict(query)
    second_pred = second.predict(query)
    first_map = first_pred.score_map if isinstance(first_pred, Prediction) else first_pred[0]
    second_map = second_pred.score_map if isinstance(second_pred, Prediction) else second_pred[0]
    np.testing.assert_allclose(first_map, second_map, rtol=0, atol=1e-6)


def test_multi_resolution_calls_are_explicit_and_do_not_mutate_backbone(tmp_path):
    support = _support(tmp_path)
    query = _write_image(tmp_path / "query.png", offset=73)
    backbone = TinyPSMABackbone(default_resolution=8)
    before = backbone.state()
    method = _method(backbone, resolutions=[8, 12], steps=1)

    method.fit(support, _context())
    method.predict(query)

    requested = [resolution for _, resolution in backbone.prepare_calls]
    assert 8 in requested and 12 in requested
    assert backbone.smaller_edge_size == 8
    assert all(torch.equal(before[name], value) for name, value in backbone.model.state_dict().items())


def test_context_rotation_and_masking_flags_preserve_source_grouping(tmp_path):
    support = _support(tmp_path, count=3)
    context = MethodContext("tiny", 23, 8, True, True)
    backbone = TinyPSMABackbone(patterned_mask=True)
    method = _method(
        backbone,
        resolutions=[8],
        steps=0,
        max_patches_per_source=16,
        max_bank_patches=16,
    )
    method.fit(support, context)

    metadata = method.object_metadata()
    assert metadata["context"]["masking"] is True
    assert metadata["context"]["rotation"] is True
    assert metadata["training"]["masking"] is True
    assert metadata["training"]["rotation"] is True
    assert metadata["training"]["rotation_descendants_per_source"] == 8
    assert backbone.mask_calls
    assert all(call["masking_type"] is True for call in backbone.mask_calls)
    assert any(not output.all() for output in backbone.mask_outputs)
    assert all(
        record[8].clean.shape[0] <= method.max_patches_per_source
        for record in method._records
    )
    for key, bank_sources in metadata["training"]["loo_bank_sources"].items():
        held_out = int(key.split(":", 1)[0].removeprefix("source"))
        assert held_out not in bank_sources

    plain_backbone = TinyPSMABackbone()
    plain_method = _method(plain_backbone, resolutions=[8], steps=0)
    plain_method.fit(support, MethodContext("tiny", 23, 8, False, False))
    plain_metadata = plain_method.object_metadata()
    assert plain_metadata["training"]["rotation_descendants_per_source"] == 1
    assert plain_backbone.mask_calls
    assert all(call["masking_type"] is False for call in plain_backbone.mask_calls)


def test_adapter_off_and_single_resolution_controls_are_valid(tmp_path):
    support = _support(tmp_path)
    query = _write_image(tmp_path / "query.png", offset=53)
    backbone = TinyPSMABackbone()
    method = _method(backbone, resolutions=[8], adapter_enabled=False, steps=2)

    method.fit(support, _context())
    result = method.predict(query)
    score_map = result.score_map if isinstance(result, Prediction) else result[0]

    assert np.asarray(score_map).shape == (10, 14)
    assert np.isfinite(score_map).all()
    metadata = method.object_metadata()
    assert metadata.get("adapter_enabled") is False
    assert metadata.get("resolutions") == [8]


def test_flat_adapter_rank_config_controls_residual_rank():
    method = _method(TinyPSMABackbone(), adapter_rank=3)
    assert method.rank == 3


def test_leave_source_out_bank_excludes_source_descendants_and_query(tmp_path):
    support = _support(tmp_path, count=3)
    query = _write_image(tmp_path / "query" / "query.png", offset=181)
    method = _method(TinyPSMABackbone(), steps=1)
    method.fit(support, _context())
    metadata = method.object_metadata()
    serialized = json.dumps(metadata)

    assert "leave" in serialized.lower() or "source" in serialized.lower()
    assert str(query) not in serialized
    synthetic_types = metadata["training"]["synthetic_anomaly_types"]
    assert len(synthetic_types) == 2
    for key, bank_sources in metadata["training"]["loo_bank_sources"].items():
        held_out = int(key.split(":", 1)[0].removeprefix("source"))
        assert held_out not in bank_sources
    for source in support:
        assert Path(source).name in serialized


def test_tiny_budget_does_not_promote_missing_anomaly_mask_to_all_patches(
    tmp_path, monkeypatch
):
    psma_module = importlib.import_module("fmad.methods.PSMA_MR")
    support = _support(tmp_path, count=2)
    token_calls = 0

    def _alternating_token_mask(_mask, _grid_size, patch_count):
        nonlocal token_calls
        output = torch.zeros(patch_count, dtype=torch.bool)
        output[token_calls % 2] = True
        token_calls += 1
        return output

    monkeypatch.setattr(psma_module, "_token_mask", _alternating_token_mask)
    method = _method(
        TinyPSMABackbone(),
        resolutions=[8],
        steps=0,
        max_patches_per_source=1,
        max_bank_patches=2,
        max_train_patches=1,
    )
    try:
        method.fit(support, _context(seed=29))
    except (AssertionError, RuntimeError, ValueError) as error:
        assert {"mask", "synthetic", "positive"} & set(str(error).lower().split())
        return
    views = method._records[0][8]
    assert len(views.anomalies) == 2
    assert bool(views.anomalies[0][2].any())
    assert not bool(views.anomalies[1][2].any())

    captured_distances: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_distance = psma_module._chunked_cosine_distance

    def _capture_distance(queries, bank, chunk_size):
        captured_distances.append(
            (torch.as_tensor(queries).detach().clone(), torch.as_tensor(bank).detach().clone())
        )
        return original_distance(queries, bank, chunk_size)

    monkeypatch.setattr(psma_module, "_chunked_cosine_distance", _capture_distance)
    try:
        result = method._training_step(0, 8, step=1)
    except (AssertionError, RuntimeError, ValueError) as error:
        assert {"mask", "synthetic", "positive"} & set(str(error).lower().split())
        return

    assert result is not None
    loss, _metrics = result
    assert torch.isfinite(loss)
    # Two normal distances plus identity/T distances for only the available
    # synthetic positive; a missing mask must be skipped/diagnosed, not turned
    # into an all-patch positive by a blanket fallback.
    assert len(captured_distances) <= 4


def test_psma_mr_registry_uses_exact_public_spelling():
    import fmad.methods

    fmad.methods.register_methods("PSMA_MR")
    assert METHOD_REGISTRY.get("PSMA_MR").__name__ == "PSMA_MR"
    assert "PSMA_MR" in METHOD_REGISTRY.list()


def test_cli_selects_psma_mr_and_flattened_controls():
    from fmad.cli import load_config, parse_args

    config = load_config(
        parse_args(
            [
                "--method",
                "PSMA_MR",
                "--resolutions",
                "8,12",
                "--adapter-rank",
                "4",
            ]
        )
    )

    assert config["method"] == "PSMA_MR"
    assert config["method_config"]["resolutions"] == [8, 12]
    assert config["method_config"]["adapter_rank"] == 4


class _TinyDataset:
    def __init__(self, root: Path):
        self.root = root
        self.data_root = str(root)

    def get_objects(self):
        from fmad.datasets.base import ObjectInfo

        return [ObjectInfo("tiny", ["bad"], 8, False, False)]

    def get_object_info(self, _name):
        return self.get_objects()[0]

    def get_train_images(self, _name):
        return _support(self.root, count=3)

    def get_test_images(self, _name):
        return {"good": [self.good], "bad": [self.bad]}

    def get_ground_truth_dir(self, _name):
        return str(self.root / "ground_truth")


def test_output_maps_and_checkpoints_are_opt_in_for_psma_mr(tmp_path):
    dataset = _TinyDataset(tmp_path / "dataset")
    dataset.good = _write_image(tmp_path / "dataset" / "good.png", offset=1)
    dataset.bad = _write_image(tmp_path / "dataset" / "bad.png", offset=2)
    ground_truth = tmp_path / "dataset" / "ground_truth" / "bad"
    ground_truth.mkdir(parents=True)
    Image.fromarray(np.ones((10, 14), dtype=np.uint8) * 255).save(ground_truth / "bad_mask.png")

    common = {
        "method": "PSMA_MR",
        "dataset": "fixture",
        "backbone": "fixture",
        "seed": 4,
        "protocol": {"shots": 2, "warmup_iters": 0},
        "method_config": _config(steps=1),
        "evaluation": {"inference_batch_size": 2},
    }

    # Exercise the output policy without constructing a real DINO backbone.
    class _FactoryBackbone(TinyPSMABackbone):
        pass

    baseline = run_pipeline(
        _method(_FactoryBackbone(), steps=1),
        dataset,
        _FactoryBackbone(),
        {**common, "output": {}},
        str(tmp_path / "baseline"),
    )
    run = run_pipeline(
        _method(_FactoryBackbone(), steps=1),
        dataset,
        _FactoryBackbone(),
        {**common, "output": {"save_anomaly_maps": True, "save_weights": True}},
        str(tmp_path / "saved"),
    )
    assert baseline.keys() == run.keys()
    assert not list((tmp_path / "baseline").rglob("anomaly_maps/*.npz"))
    assert list((tmp_path / "saved").rglob("anomaly_maps/**/*.npz"))
    assert list((tmp_path / "saved").rglob("weights/**"))


def test_checkpoint_contains_json_metadata(tmp_path):
    support = _support(tmp_path)
    method = _method(TinyPSMABackbone(), steps=1)
    method.fit(support, _context())
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    method.save_checkpoint(checkpoint)
    metadata_files = list(checkpoint.rglob("*.json"))
    assert metadata_files
    for path in metadata_files:
        payload = json.loads(path.read_text())
        assert isinstance(payload, dict)
