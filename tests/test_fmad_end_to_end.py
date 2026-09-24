"""Exercise method replacement through the public runner and saved artifacts."""

import json

import numpy as np
import pytest
from PIL import Image

from fmad.methods.base import BaseMethod, Prediction
from fmad.pipeline import run_pipeline, _run_streaming_object
from tests.test_fmad_swappable_pipeline import Adapter, make_dataset


class BatchOnlyMethod(BaseMethod):
    def fit(self, support_paths, context):
        self.support = tuple(support_paths)
        self.context = context

    def predict_batch(self, paths):
        return [Prediction(np.array([[0, 1], [2, 3]], dtype=np.float32), 3, 0.01) for _ in paths]


def test_public_runner_saves_only_final_artifacts_for_swapped_methods(tmp_path):
    dataset = make_dataset(tmp_path)
    dataset.get_objects = lambda: [dataset.get_object_info("thing")]
    output = tmp_path / "results"
    metrics = []
    for name in ("first", "second"):
        config = {
            "method": name,
            "dataset": "fixture",
            "backbone": "fixture",
            "seed": 17,
            "protocol": {"shots": 2, "warmup_iters": 0},
        }
        method = BatchOnlyMethod(object(), {})
        metrics.append(run_pipeline(method, dataset, method.backbone, config, str(output)))
    assert metrics[0] == metrics[1]
    manifests = sorted(output.rglob("run_manifest.json"))
    assert len(manifests) == 2
    for manifest in manifests:
        assert {path.name for path in manifest.parent.iterdir()} == {
            "config.yaml",
            "metrics.json",
            "run_manifest.json",
            "training.jsonl",
        }
    assert (output / "summary.tsv").is_file()
    payloads = [json.loads(path.read_text()) for path in manifests]
    assert (
        payloads[0]["objects"]["thing"]["protocol"] == payloads[1]["objects"]["thing"]["protocol"]
    )
    assert payloads[0]["run_id"] != payloads[1]["run_id"]


def test_dataset_content_change_is_visible_in_provenance(tmp_path):
    dataset = make_dataset(tmp_path)
    kwargs = {"config": {"protocol": {"shots": 1, "warmup_iters": 0}}}
    first = _run_streaming_object(Adapter(), dataset, "thing", 0, str(tmp_path), 2, **kwargs)
    query = dataset.root / "thing/test_public/good/0.png"
    Image.fromarray(np.full((2, 2, 3), 255, dtype=np.uint8)).save(query)
    second = _run_streaming_object(Adapter(), dataset, "thing", 0, str(tmp_path), 2, **kwargs)
    before, after = first.metadata["protocol"], second.metadata["protocol"]
    assert before["query_identity_sha256"] == after["query_identity_sha256"]
    assert before["query_content_sha256"] != after["query_content_sha256"]


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size_fails_before_output_creation(tmp_path, batch_size):
    dataset = make_dataset(tmp_path)
    dataset.get_objects = lambda: [dataset.get_object_info("thing")]
    output = tmp_path / "results"
    config = {
        "method": "fake",
        "dataset": "fixture",
        "backbone": "fixture",
        "evaluation": {"inference_batch_size": batch_size},
    }
    with pytest.raises(ValueError, match="inference_batch_size"):
        run_pipeline(Adapter(), dataset, None, config, str(output))
    assert not output.exists()


def test_normal_query_cannot_reduce_evaluation_pixels_with_wrong_map_size(tmp_path):
    dataset = make_dataset(tmp_path)

    class WrongSize(Adapter):
        def predict_batch(self, paths):
            return [Prediction(np.zeros((1, 1)), 0.0, 0.01) for _ in paths]

    with pytest.raises(ValueError, match="native image"):
        _run_streaming_object(
            WrongSize(), dataset, "thing", 0, str(tmp_path), 2, config={"protocol": {"shots": 1}}
        )


@pytest.mark.parametrize(
    "protocol",
    [{"shots": True}, {"shots": 1.5}, {"shots": 0}, {"warmup_iters": False}, {"warmup_iters": 0.5}],
)
def test_malformed_protocol_does_not_create_results(tmp_path, protocol):
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="protocol"):
        run_pipeline(None, None, None, {"protocol": protocol}, str(output))
    assert not output.exists()
