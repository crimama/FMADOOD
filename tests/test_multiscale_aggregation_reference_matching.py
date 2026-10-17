import numpy as np
import pytest
import torch

from fmad.methods.multiscale_aggregation_reference_matching import (
    MultiscaleAggregationReferenceMatching,
    combine_resolution_maps,
)
from fmad.methods.statistical_reference_matching import StatisticalReferenceMatching
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import _write_image
from tests.test_reference_matching import TinyTripleBackbone
from tests.test_statistical_reference_matching import config


def test_combine_resolution_maps_exact_formulas():
    m0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    m1 = torch.tensor([[2.0, 4.0], [1.0, 8.0]])
    m2 = torch.tensor([[3.0, 6.0], [2.0, 2.0]])
    m3 = torch.tensor([[4.0, 8.0], [5.0, 6.0]])
    maps = [m0, m1, m2, m3]

    torch.testing.assert_close(combine_resolution_maps(maps, "mean"), torch.stack(maps).mean(0))
    expected_top3 = torch.tensor([[(2+3+4)/3, (4+6+8)/3], [(2+3+5)/3, (4+6+8)/3]])
    torch.testing.assert_close(combine_resolution_maps(maps, "scale_top3"), expected_top3)

    lse = combine_resolution_maps(maps, "scale_lse", lse_beta=2.0)
    expected_lse = (torch.logsumexp(2.0 * torch.stack(maps), dim=0) - np.log(4.0)) / 2.0
    torch.testing.assert_close(lse, expected_lse)

    # Shifting one resolution by a constant offset must be removed by spatial_contrast_top3
    # up to the preserved mean-median shift.
    shifted = [m0 + 10.0, m1, m2, m3]
    c_base = combine_resolution_maps(maps, "spatial_contrast_top3")
    c_shift = combine_resolution_maps(shifted, "spatial_contrast_top3")
    torch.testing.assert_close(c_shift - c_base, torch.full_like(c_base, 2.5))


@pytest.mark.parametrize("mode", [
    "mean", "scale_lse", "scale_top3", "spatial_contrast_top3",
    "scale_order_linear", "scale_order_renyi",
    "spatial_contrast_order_linear", "spatial_contrast_order_renyi"
])
def test_method_preserves_bank_and_matches_baseline_on_mean(tmp_path, mode):
    paths = [_write_image(tmp_path / f"{i}.png", i * 29) for i in range(2)]
    cfg = config(statistical_normalization="q95")
    cfg["resolutions"] = [8, 12]
    cfg["calibration_audit"] = False
    context = MethodContext("can", 0, 8, False, False)

    baseline = StatisticalReferenceMatching(TinyTripleBackbone(), cfg)
    baseline.fit(paths, context)

    candidate = MultiscaleAggregationReferenceMatching(
        TinyTripleBackbone(), dict(cfg, multiscale_aggregation=mode)
    )
    candidate.fit(paths, context)

    assert (
        candidate.object_metadata()["ablation_bank_feature_sha256"]
        == baseline.object_metadata()["ablation_bank_feature_sha256"]
    )
    amap, score = candidate.predict(paths[0])
    assert np.isfinite(amap).all() and np.isfinite(score)
    if mode == "mean":
        bmap, bscore = baseline.predict(paths[0])
        np.testing.assert_allclose(amap, bmap, rtol=1e-6, atol=1e-6)
        assert score == pytest.approx(bscore, abs=1e-6)
