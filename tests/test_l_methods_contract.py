"""Shared pipeline/configuration controls for the three L directions."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from fmad.methods import register_methods
from fmad.methods.boundary_synthetic import BoundarySynthetic
from fmad.methods.operating_region import OperatingRegion
from fmad.methods.shift_restore import ShiftRestore
from fmad.methods.transfer_update import TransferUpdate
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image
from tests.test_transfer_update import config


ROOT = Path(__file__).parents[1]
VARIANTS = [("operating_region", "L1", {"operating_region", "operating_fprs", "operating_temperature"}),
            ("shift_restore", "L2C", {"shift_restore", "restoration_target", "restoration_weight"}),
            ("shift_restore", "L2", {"shift_restore", "restoration_target", "restoration_weight"}),
            ("transfer_update", "L3C", {"transfer_update", "cross_environment_meta", "meta_inner_lr", "meta_outer_weight"}),
            ("transfer_update", "L3", {"transfer_update", "cross_environment_meta", "meta_inner_lr", "meta_outer_weight"})]


@pytest.mark.parametrize("method,label,allowed", VARIANTS)
def test_config_changes_are_explicit_and_shared_pipeline_fixed(method, label, allowed):
    baseline = yaml.safe_load((ROOT / "configs/methods/boundary_synthetic_K1.yaml").read_text())
    candidate = yaml.safe_load((ROOT / f"configs/methods/{method}_{label}.yaml").read_text())
    register_methods(method)
    assert METHOD_REGISTRY.get(method) is not None
    assert candidate["method"] == method
    assert {k: v for k, v in candidate.items() if k not in {"method", "method_config"}} == {
        k: v for k, v in baseline.items() if k not in {"method", "method_config"}}
    assert {k: v for k, v in candidate["method_config"].items() if k not in allowed} == baseline["method_config"]
    assert candidate["output"] == {"save_anomaly_maps": False, "save_weights": False}


@pytest.mark.parametrize("cls,switch", [(OperatingRegion, "operating_region"),
                                      (ShiftRestore, "shift_restore"),
                                      (TransferUpdate, "transfer_update")])
def test_off_is_k1_even_when_weak_synthesis_is_selected(cls, switch, tmp_path, monkeypatch):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    reference = BoundarySynthetic(TinyPSMAV2Backbone(), config())
    disabled = cls(TinyPSMAV2Backbone(), config(**{switch: False}))
    for model in (reference, disabled):
        monkeypatch.setattr(model, "_teacher_quantile",
                            lambda image, banks, q, mask=None: .1 if mask is None else .2)
        model.fit(paths, MethodContext("can", 0, 8, False, False))
        assert model.anomaly_strength == .25
        assert all(row["boundary_strength"] == .25 for row in model.history)
    assert disabled.history == reference.history
    np.testing.assert_array_equal(disabled.predict(paths[0])[0], reference.predict(paths[0])[0])


@pytest.mark.parametrize("family,control,candidate,changed", [
    ("shift_restore", "L2C", "L2", "restoration_target"),
    ("transfer_update", "L3C", "L3", "cross_environment_meta")])
def test_control_changes_only_one_declared_component(family, control, candidate, changed):
    left, right = [yaml.safe_load((ROOT / f"configs/methods/{family}_{label}.yaml").read_text())
                   for label in (control, candidate)]
    assert left["method_config"].pop(changed) != right["method_config"].pop(changed)
    assert left == right
