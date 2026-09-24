import numpy as np
import pytest
import torch
import yaml
from pathlib import Path

from fmad.methods.region_recovery import RegionRecovery, bounded_reach, feature_links, recover_regions
from fmad.methods.reference_matching import ReferenceMatching
from fmad.postprocess import component_quantile
from fmad.protocol import MethodContext
from tests.test_reference_matching import TinyTripleBackbone
from tests.test_psma_mr_v2 import _write_image


def test_growth_is_bounded_and_barriers_block():
    seeds = np.zeros((1, 9), bool)
    seeds[0, 0] = True
    eligible = np.ones_like(seeds)
    h, v = np.ones((1, 8), bool), np.ones((0, 9), bool)
    assert bounded_reach(seeds, eligible, h, v, 4).sum() == 5
    h[0, 2] = False
    assert bounded_reach(seeds, eligible, h, v, 8).sum() == 3
    eligible[0, 1] = False
    assert bounded_reach(seeds, eligible, h, v, 8).sum() == 1


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA'))])
def test_feature_barriers(device):
    guide = torch.tensor([[1., 0.], [0., 1.]], device=device)
    h, v = feature_links(guide, (1, 2), (2, 8), .9)
    assert not h[:, 3].any()
    assert h[:, :3].all() and h[:, 4:].all() and v.all()
    h, v = feature_links(guide, (1, 2), (1, 1), .9)
    assert h.size == v.size == 0


def fixture():
    raw = np.arange(10000, dtype=np.float32).reshape(100, 100)
    baseline = component_quantile(raw, percentile=99., q=1., dilate_size=0)
    image = np.zeros((100, 100, 3), np.uint8)
    return raw, baseline, image, [raw, raw], torch.ones((1, 2)), (1, 1)


def test_recovery_bounds_repeatability_and_no_mutation():
    args = fixture()
    saved = [x.copy() for x in args[:3]]
    out, stats = recover_regions(*args)
    assert stats['raised_native_pixels'] > 0
    assert np.isfinite(out).all() and (out >= args[1]).all()
    assert out.max() == args[1].max()
    low, high = np.percentile(args[0], [95, 99])
    changed = out != args[1]
    assert ((args[0][changed] >= low) & (args[0][changed] < high)).all()
    assert (out[changed] == np.float32(high)).all()
    np.testing.assert_array_equal(out, recover_regions(*args)[0])
    for original, before in zip(args[:3], saved):
        np.testing.assert_array_equal(original, before)


def test_colour_votes_flat_and_coarse_native_guard():
    args = list(fixture())
    args[2][::2] = 255
    np.testing.assert_array_equal(recover_regions(*args)[0], args[1])
    args = list(fixture())
    args[3] = [args[0]]
    np.testing.assert_array_equal(recover_regions(*args)[0], args[1])
    args = list(fixture())
    np.testing.assert_array_equal(recover_regions(*args, steps=0)[0], args[1])
    out, _ = recover_regions(*args, max_edge=10)
    low = np.percentile(args[0], 95)
    np.testing.assert_array_equal(out[args[0] < low], args[1][args[0] < low])
    args[0] = args[1] = np.ones((100, 100), np.float32)
    out, stats = recover_regions(*args)
    assert stats['constant_fallback_queries'] == 1
    np.testing.assert_array_equal(out, args[1])


def config(enabled=False):
    path = Path(__file__).resolve().parents[1] / 'configs/methods/RR_A.yaml'
    cfg = yaml.safe_load(path.read_text())['method_config']
    return dict(cfg, resolutions=[8, 12], bank_tokens_per_source=4,
                query_chunk_size=4, bank_bags=2, recovery_enabled=enabled)


def test_disabled_exact_rsg3_parity_and_enabled_cleanup(tmp_path):
    paths = [_write_image(tmp_path / f'{i}.png', i * 17) for i in range(3)]
    context = MethodContext('can', 0, 8, False, False)
    ref = ReferenceMatching(TinyTripleBackbone(), config())
    method = RegionRecovery(TinyTripleBackbone(), config())
    ref.fit(paths, context)
    method.fit(paths, context)
    expected, score = ref.predict(paths[0])
    actual, actual_score = method.predict(paths[0])
    np.testing.assert_array_equal(actual, expected)
    assert actual_score == score
    method.recovery_enabled = True
    result, _ = method.predict(paths[0])
    assert np.isfinite(result).all() and (result >= expected).all()
    np.testing.assert_array_equal(result, method.predict(paths[0])[0])
    assert method._guide is None and method._scale_maps == []
    with pytest.raises(Exception):
        method.predict(str(tmp_path / 'missing.png'))
    assert method._guide is None and method._scale_maps == [] and not method._collect_recovery


def test_configs_differ_only_switch_and_boolean_validation():
    root = Path(__file__).resolve().parents[1] / 'configs/methods'
    a, b = [yaml.safe_load((root / f'RR_{x}.yaml').read_text()) for x in 'AB']
    assert not a['method_config'].pop('recovery_enabled')
    assert b['method_config'].pop('recovery_enabled')
    assert a == b
    with pytest.raises(ValueError, match='boolean'):
        RegionRecovery(TinyTripleBackbone(), config('false'))
