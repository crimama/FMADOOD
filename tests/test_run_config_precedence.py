import os
import pytest
import yaml

from fmad import cli


def _load_run_module(monkeypatch):
    return cli


def _config_args(monkeypatch, module, config_path, *cli_args):
    return module.load_config(module.parse_args(["--config", str(config_path), *cli_args]))


def test_yaml_runtime_values_are_preserved_without_flags(tmp_path, monkeypatch):
    module = _load_run_module(monkeypatch)
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(yaml.safe_dump({"device": "cpu", "output_dir": "custom-results"}))

    config = _config_args(monkeypatch, module, config_path)

    assert config["device"] == "cpu"
    assert config["output_dir"] == "custom-results"


def test_explicit_runtime_flags_override_yaml(tmp_path, monkeypatch):
    module = _load_run_module(monkeypatch)
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(yaml.safe_dump({"device": "cpu", "output_dir": "custom-results"}))

    config = _config_args(
        monkeypatch,
        module,
        config_path,
        "--device",
        "cuda:2",
        "--output-dir",
        "cli-results",
        "--method",
        "cli-method",
        "--dataset",
        "cli-dataset",
        "--backbone",
        "cli-backbone",
        "--seed",
        "7",
    )

    assert config["device"] == "cuda:2"
    assert config["output_dir"] == "cli-results"
    assert config["method"] == "cli-method"
    assert config["dataset"] == "cli-dataset"
    assert config["backbone"] == "cli-backbone"
    assert config["seed"] == 7


def test_existing_selection_defaults_remain_and_runtime_fallbacks_apply(monkeypatch):
    module = _load_run_module(monkeypatch)
    config = module.load_config(module.parse_args([]))

    assert config["method"] == "superad"
    assert config["dataset"] == "mvtec_ad2"
    assert config["backbone"] == "dinov2_vitl14"
    assert config["seed"] == 0
    assert config["device"] == "cuda:0"
    assert config["output_dir"] == "./results"


def test_runtime_device_respects_cpu_and_cuda_slot(monkeypatch):
    module = _load_run_module(monkeypatch)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert module.resolve_runtime_device("cpu") == "cpu"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert module.resolve_runtime_device("cuda:2") == "cuda:2"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert module.resolve_runtime_device("cuda:1") == "cuda:1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,5"


def test_protocol_defaults_are_recorded_and_legacy_keys_migrate(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"method_config": {"shots": 3, "warmup_iters": 0}}))
    config = cli.load_config(cli.parse_args(["--config", str(path)]))
    assert config["protocol"] == {"shots": 3, "warmup_iters": 0, "fixed_reference_manifest": None}
    assert "shots" not in config["method_config"]
    assert config["evaluation"]["metric_contract"] == "mvtec_ad2_float16_streaming_v1"
    assert config["debug_output"] is False


def test_conflicting_protocol_locations_fail(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"method_config": {"shots": 3}, "protocol": {"shots": 5}}))
    with pytest.raises(ValueError, match="conflicting"):
        cli.load_config(cli.parse_args(["--config", str(path)]))
