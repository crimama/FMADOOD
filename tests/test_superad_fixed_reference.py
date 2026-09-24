from pathlib import Path
import cv2
import numpy as np
import pytest
from fmad.methods.superad import SuperAD
from fmad.protocol import MethodContext
from fmad.protocol import load_fixed_reference_paths


def test_load_fixed_reference_paths_preserves_manifest_order(tmp_path):
    train = tmp_path / "can" / "train" / "good"
    train.mkdir(parents=True)
    paths = [str(train / f"{i}.png") for i in range(3)]
    for path in paths:
        Path(path).touch()
    manifest = tmp_path / "support.json"
    manifest.write_text('{"can": ["/other/can/train/good/2.png", "/other/can/train/good/0.png"]}')
    selected, digest = load_fixed_reference_paths(manifest, "can", paths, 2)
    assert [Path(p).name for p in selected] == ["2.png", "0.png"] and len(digest) == 64


@pytest.mark.parametrize(
    "requested, count, message",
    [
        (["/other/can/train/good/0.png"], 2, "expected"),
        (["/other/can/train/good/0.png", "/other/can/train/good/0.png"], 2, "duplicates"),
        (["/other/can/test_public/good/0.png"], 1, "outside"),
        (["/other/can/train/good/missing.png"], 1, "not present"),
    ],
)
def test_fixed_manifest_rejects_invalid_members(tmp_path, requested, count, message):
    train = tmp_path / "can" / "train" / "good"
    train.mkdir(parents=True)
    path = train / "0.png"
    path.touch()
    manifest = tmp_path / "support.json"
    import json

    manifest.write_text(json.dumps({"can": requested}))
    with pytest.raises(ValueError, match=message):
        load_fixed_reference_paths(manifest, "can", [str(path)], count)


class TinyBackbone:
    feature_layers = (0,)
    device = "cpu"
    model_name = "tiny"

    def prepare_image(self, image):
        return image, (1, 2)

    def extract_features(self, image):
        return [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)]

    def compute_background_mask(self, features, grid_size, **kwargs):
        return np.array([True, True])


def test_superad_cpu_faiss_scores_with_supplied_support(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.png"
        cv2.imwrite(str(path), np.zeros((2, 2, 3), dtype=np.uint8))
        paths.append(str(path))
    method = SuperAD(TinyBackbone(), {"faiss_on_cpu": True, "knn_metric": "L2_normalized"})
    method.fit(paths, MethodContext("can", 0, 2, False, False))
    score_map, score = method.predict(paths[0])
    assert score_map.shape == (2, 2) and np.isfinite(score) and np.all(score_map >= 0)
    assert np.allclose(score_map, 0.0)


def test_superad_checkpoint_contains_reusable_faiss_memory(tmp_path):
    import json
    import faiss

    image = tmp_path / "support.png"
    cv2.imwrite(str(image), np.zeros((2, 2, 3), dtype=np.uint8))
    method = SuperAD(TinyBackbone(), {"faiss_on_cpu": True})
    method.fit([str(image)], MethodContext("can", 0, 2, False, False))
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    method.save_checkpoint(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["kind"] == "faiss_memory_bank"
    restored = faiss.read_index(str(directory / metadata["indices"]["layer0"]))
    query = np.array([[0.6, 0.8]], dtype=np.float32)
    expected, expected_ids = method._knn_indices["layer0"].search(query, 1)
    observed, observed_ids = restored.search(query, 1)
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(observed_ids, expected_ids)


@pytest.mark.parametrize("metric, expected", [("L2_normalized", 1.0), ("L2", 2.0)])
def test_cpu_distance_definition_for_orthogonal_query(tmp_path, metric, expected):
    class Orthogonal(TinyBackbone):
        def extract_features(self, image):
            feature = [1.0, 0.0] if image.max() == 0 else [0.0, 1.0]
            return [np.tile(np.asarray(feature, dtype=np.float32), (2, 1))]

    support, query = tmp_path / "support.png", tmp_path / "query.png"
    cv2.imwrite(str(support), np.zeros((2, 2, 3), dtype=np.uint8))
    cv2.imwrite(str(query), np.full((2, 2, 3), 255, dtype=np.uint8))
    method = SuperAD(Orthogonal(), {"knn_metric": metric})
    method.fit([str(support)], MethodContext("can", 0, 2, False, False))
    score_map, score = method.predict(str(query))
    assert score == pytest.approx(expected)
    assert np.allclose(score_map, expected)


def test_gpu_search_configuration_uses_backbone_device(monkeypatch, tmp_path):
    import faiss
    import torch
    from types import SimpleNamespace

    backbone = TinyBackbone()
    backbone.device = "cuda:2"
    requested = []

    def gpu_index(resource, dimension, config):
        requested.append(config.device)
        return faiss.IndexFlatIP(dimension)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(faiss, "StandardGpuResources", object, raising=False)
    monkeypatch.setattr(faiss, "GpuIndexFlatConfig", SimpleNamespace, raising=False)
    monkeypatch.setattr(faiss, "GpuIndexFlatIP", gpu_index, raising=False)
    support = tmp_path / "support.png"
    cv2.imwrite(str(support), np.zeros((2, 2, 3), dtype=np.uint8))
    method = SuperAD(backbone, {})
    context = MethodContext("can", 0, 2, False, False)
    method.fit([str(support)], context)
    method.fit([str(support)], context)
    assert requested == [2, 2]
    assert len(method._faiss_resources) == 1
