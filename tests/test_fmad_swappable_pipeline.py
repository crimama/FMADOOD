import numpy as np
from PIL import Image
from fmad.datasets.base import ObjectInfo
from fmad.methods.base import Prediction
from fmad.pipeline import _run_streaming_object


class Dataset:
    def __init__(self, root):
        self.root = root
        self.data_root = str(root)

    def get_object_info(self, name):
        return ObjectInfo(name, ["bad"], 32, False, False)

    def get_train_images(self, name):
        return [str(self.root / name / "train" / "good" / f"{i}.png") for i in range(4)]

    def get_test_images(self, name):
        return {
            "good": [str(self.root / name / "test_public" / "good" / "0.png")],
            "bad": [str(self.root / name / "test_public" / "bad" / "1.png")],
        }

    def get_ground_truth_dir(self, name):
        return str(self.root / name / "test_public" / "ground_truth")


class Adapter:
    def __init__(self, offset=0):
        self.offset = offset
        self.support = None
        self.context = None
        self.backbone = object()

    def fit(self, support_paths, context):
        self.support = list(support_paths)
        self.context = context

    def predict_batch(self, paths):
        return [
            Prediction(
                np.array([[0, 1], [2, 3]], dtype=np.float32) + self.offset, 3 + self.offset, 0.01
            )
            for _ in paths
        ]

    def object_metadata(self):
        return {"adapter": "test"}


def make_dataset(tmp_path):
    root = tmp_path / "data"
    for path in [
        *(root / "thing" / "train" / "good" / f"{i}.png" for i in range(4)),
        root / "thing" / "test_public" / "good" / "0.png",
        root / "thing" / "test_public" / "bad" / "1.png",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(path)
    gt = root / "thing" / "test_public" / "ground_truth" / "bad"
    gt.mkdir(parents=True)
    Image.fromarray(np.array([[0, 0], [0, 255]], dtype=np.uint8)).save(gt / "1_mask.png")
    return Dataset(root)


def test_two_adapters_share_protocol_paths_and_metrics(tmp_path):
    dataset = make_dataset(tmp_path)
    config = {"protocol": {"shots": 2}, "evaluation": {"inference_batch_size": 2}}
    first, second = Adapter(), Adapter()
    r1 = _run_streaming_object(first, dataset, "thing", 17, str(tmp_path / "a"), 2, config=config)
    r2 = _run_streaming_object(second, dataset, "thing", 17, str(tmp_path / "b"), 2, config=config)
    assert first.support == second.support
    assert (
        r1.metadata["protocol"]["query_identity_sha256"]
        == r2.metadata["protocol"]["query_identity_sha256"]
    )
    assert r1.metadata["streaming_metrics"] == r2.metadata["streaming_metrics"]


def test_runner_rejects_batch_count_mismatch(tmp_path):
    dataset = make_dataset(tmp_path)

    class Broken(Adapter):
        def predict_batch(self, paths):
            return []

    try:
        _run_streaming_object(
            Broken(), dataset, "thing", 1, str(tmp_path), 2, config={"protocol": {"shots": 1}}
        )
    except ValueError as exc:
        assert "returned 0 outputs for 2 queries" in str(exc)
    else:
        raise AssertionError("batch mismatch must fail")


def test_default_run_writes_no_debug_files(tmp_path):
    _run_streaming_object(
        Adapter(),
        make_dataset(tmp_path),
        "thing",
        1,
        str(tmp_path / "run"),
        2,
        config={"protocol": {"shots": 1}},
    )
    assert not (tmp_path / "run" / "streaming_metrics").exists()
