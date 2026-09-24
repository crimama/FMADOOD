"""Output switches must not alter quantitative evaluation or retain dense maps."""

import json

import numpy as np
import pytest

from fmad.cli import load_config, parse_args
from fmad.pipeline import run_pipeline
from fmad.results import ResultsManager
from tests.test_fmad_end_to_end import BatchOnlyMethod
from tests.test_fmad_swappable_pipeline import make_dataset


class TrainableFixture(BatchOnlyMethod):
    def fit(self, support_paths, context):
        super().fit(support_paths, context)
        self.log_training(step=1, loss=0.25)

    def save_checkpoint(self, directory):
        np.save(directory / "weights.npy", np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "save_maps,save_weights", [(False, False), (True, False), (False, True), (True, True)]
)
def test_output_switches_save_requested_artifacts_only(tmp_path, save_maps, save_weights):
    dataset = make_dataset(tmp_path)
    dataset.get_objects = lambda: [dataset.get_object_info("thing")]
    method = TrainableFixture(object(), {})
    output = tmp_path / "results"
    config = {
        "method": "fixture",
        "dataset": "fixture",
        "backbone": "fixture",
        "protocol": {"shots": 1, "warmup_iters": 0},
        "output": {"save_anomaly_maps": save_maps, "save_weights": save_weights},
    }
    observed = run_pipeline(method, dataset, method.backbone, config, str(output))
    manifest = next(output.rglob("run_manifest.json"))
    run_dir = manifest.parent
    logs = [json.loads(line) for line in (run_dir / "training.jsonl").read_text().splitlines()]
    assert [row["event"] for row in logs][:3] == ["fit_started", "step", "fit_finished"]
    assert logs[1]["step"] == 1 and logs[1]["metrics"]["loss"] == 0.25
    assert method._training_log_sink is None
    assert (run_dir / "anomaly_maps").exists() is save_maps
    assert (run_dir / "weights").exists() is save_weights
    if save_maps:
        maps = list((run_dir / "anomaly_maps").rglob("*.npz"))
        assert len(maps) == 2
        for path in maps:
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["score_map"], [[0, 1], [2, 3]])
                assert archive["score_map"].dtype == np.float32
    if save_weights:
        np.testing.assert_array_equal(np.load(run_dir / "weights/thing/weights.npy"), [1, 2])
    baseline = run_pipeline(
        TrainableFixture(object(), {}),
        dataset,
        object(),
        {**config, "output": {}},
        str(tmp_path / "baseline"),
    )
    assert observed == baseline


def test_training_failure_is_logged_and_sink_restored(tmp_path):
    class Broken(TrainableFixture):
        def fit(self, support_paths, context):
            self.log_training(step=1, loss=0.5)
            raise RuntimeError("training failed")

    dataset = make_dataset(tmp_path)
    dataset.get_objects = lambda: [dataset.get_object_info("thing")]
    method = Broken(object(), {})
    output = tmp_path / "results"
    with pytest.raises(RuntimeError, match="training failed"):
        run_pipeline(
            method,
            dataset,
            method.backbone,
            {
                "method": "broken",
                "dataset": "fixture",
                "backbone": "fixture",
                "protocol": {"shots": 1},
            },
            str(output),
        )
    log = next(output.rglob("training.jsonl"))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[-1]["event"] == "fit_failed"
    assert rows[-1]["error_type"] == "RuntimeError"
    assert method._training_log_sink is None
    assert not list(output.rglob("metrics.json"))


def test_unsupported_checkpoint_option_fails_before_artifacts(tmp_path):
    with pytest.raises(NotImplementedError, match="checkpoint"):
        run_pipeline(
            BatchOnlyMethod(object(), {}),
            None,
            None,
            {"output": {"save_weights": True}},
            str(tmp_path / "results"),
        )
    assert not (tmp_path / "results").exists()


def test_cli_defaults_and_explicit_disable_override_yaml(tmp_path):
    assert load_config(parse_args([]))["output"] == {
        "save_anomaly_maps": False,
        "save_weights": False,
    }
    path = tmp_path / "config.yaml"
    path.write_text("output:\n  save_anomaly_maps: true\n  save_weights: true\n")
    config = load_config(
        parse_args(["--config", str(path), "--no-save-anomaly-maps", "--no-save-weights"])
    )
    assert not any(config["output"].values())
    assert all(
        load_config(parse_args(["--save-anomaly-maps", "--save-weights"]))["output"].values()
    )


@pytest.mark.parametrize("value", ["false", 1, None])
def test_output_options_require_booleans(tmp_path, value):
    with pytest.raises(ValueError, match="must be a boolean"):
        run_pipeline(
            None, None, None, {"output": {"save_weights": value}}, str(tmp_path / "results")
        )
    assert not (tmp_path / "results").exists()


def test_map_paths_cannot_escape_run(tmp_path):
    manager = ResultsManager(
        str(tmp_path), "m", "d", "b", 0, config={"output": {"save_anomaly_maps": True}}
    )
    with pytest.raises(ValueError, match="path component"):
        manager.save_anomaly_map("..", "bad/query.png", np.zeros((2, 2)))
