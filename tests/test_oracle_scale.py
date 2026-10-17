import numpy as np
import pytest
import torch

from fmad.evaluation.oracle_scale import (
    sample_coordinates, observation_samples, reconstruct_map,
    weighted_partial_pro, search_scales,
    expand_oracle_parameters, oracle_parameter_count, ORACLE_AXES,
)
from fmad.evaluation.streaming import StreamingSegmentationAccumulator


def test_sampled_pro_matches_common_full_pixel_evaluator_with_ties():
    mask = np.zeros((10, 12), bool)
    mask[1:3, 1:4] = True
    mask[6:9, 8:11] = True
    scores = np.random.default_rng(4).integers(0, 6, size=mask.shape).astype(np.float32)
    indices, nw, pw = sample_coordinates(mask, np.random.default_rng(1), 1000, 1000)
    got = weighted_partial_pro(scores.ravel()[indices], nw, pw)
    accumulator = StreamingSegmentationAccumulator()
    accumulator.update([scores], [mask])
    assert got == pytest.approx(accumulator.compute().seg_aupro_0_05, abs=1e-12)


def test_component_interpolation_matches_reconstruction_at_every_pixel():
    rng = np.random.default_rng(2)
    bags = [rng.uniform(0, 2, (4, 3, 3, 4)).astype(np.float32),
            rng.uniform(0, 2, (4, 3, 5, 7)).astype(np.float32)]
    shape = (12, 15)
    features = observation_samples(bags, shape, np.arange(np.prod(shape)))
    multipliers = rng.uniform(.1, 3, 24).astype(np.float32)
    result = reconstruct_map(bags, shape, multipliers)
    np.testing.assert_allclose(features @ multipliers, result.ravel(), atol=2e-6, rtol=2e-6)


def test_oracle_search_retains_baseline_and_can_find_a_better_scale():
    # First component separates anomalies, second overwhelms it with noise.
    rng = np.random.default_rng(14)
    labels = np.r_[np.zeros(120), np.ones(120)]
    x = np.stack([labels, rng.uniform(0, 6, len(labels))], 1).astype(np.float32)
    candidates, audit = search_scales(x, 1-labels, labels, iterations=3, seed=0)
    assert candidates[0][0] == 'q95'
    assert audit['sampled_best_aupro'] > audit['sampled_q95_aupro']
    assert all(np.all(w > 0) for _, w in candidates)
    assert audit['dimensions'] == 2


@pytest.mark.parametrize('mode', ORACLE_AXES)
def test_grouped_oracle_respects_sharing_bounds_and_fixed_layer_coefficients(mode):
    shape = (4, 4, 3)
    rng = np.random.default_rng(5)
    params = rng.uniform(-np.log(8), np.log(8), (7, oracle_parameter_count(mode, shape)))
    weights = expand_oracle_parameters(params, mode, shape).reshape(7, *shape)
    assert weights.min() >= 1/8 - 1e-12
    assert weights.max() <= 8 + 1e-12
    if mode in ('layer', 'resolution_layer', 'resolution_layer_fixed_layer'):
        np.testing.assert_allclose(weights, np.broadcast_to(weights[:,:,0:1,:], weights.shape))
    if mode == 'layer':
        np.testing.assert_allclose(weights, np.broadcast_to(weights[:,0:1,0:1,:], weights.shape))
    if mode == 'resolution':
        np.testing.assert_allclose(weights, np.broadcast_to(weights[:,:,0:1,0:1], weights.shape))
    if mode == 'bag':
        np.testing.assert_allclose(weights, np.broadcast_to(weights[:,0:1,:,0:1], weights.shape))
    if mode == 'resolution_layer_fixed_layer':
        np.testing.assert_allclose(weights.mean((1,2)), np.ones((7,3)), atol=1e-12)
    identity = expand_oracle_parameters(np.zeros_like(params), mode, shape)
    np.testing.assert_allclose(identity, 1., atol=1e-12)


def test_oracle_entrypoint_keeps_q95_provenance_and_records_gt_use(tmp_path, monkeypatch):
    import json
    import scripts.oracle_scale_calibration as entry
    from tests.test_fmad_swappable_pipeline import make_dataset
    from tests.test_reference_matching import TinyTripleBackbone
    from tests.test_statistical_reference_matching import config as method_config
    dataset = make_dataset(tmp_path)
    dataset.get_objects = lambda: [dataset.get_object_info('thing')]
    config = dict(seed=0, method='statistical_reference_matching', dataset='tiny',
        backbone='tiny', device='cpu', data_root=str(dataset.root),
        output_dir=str(tmp_path/'output'), protocol=dict(shots=2, warmup_iters=0),
        method_config=method_config(statistical_normalization='q95'),
        output=dict(save_anomaly_maps=False, save_weights=False),
        evaluation=dict(streaming=True,inference_batch_size=1,
            oracle_scale_calibration=dict(use_evaluation_ground_truth=True, iterations=0,
                axis_modes=list(ORACLE_AXES), population=12)))
    monkeypatch.setattr(entry, 'parse_args', lambda: None)
    monkeypatch.setattr(entry, 'load_config', lambda _: config)
    monkeypatch.setattr(entry.DATASET_REGISTRY, 'get', lambda _: lambda **kw: dataset)
    monkeypatch.setattr(entry.BACKBONE_REGISTRY, 'get', lambda _: lambda **kw: TinyTripleBackbone())
    entry.main()
    diagnostic = json.loads(next((tmp_path/'output').rglob('oracle_diagnostic.json')).read_text())
    assert diagnostic['selected_aupro'] >= diagnostic['baseline_aupro']
    assert diagnostic['deployable_calibration'] is False
    assert diagnostic['global_optimum_claimed'] is False
    assert diagnostic['query_count'] == 2
    assert len(diagnostic['selected_divisor_scales']) == 6
    assert set(diagnostic['axis_ablation']) == set(ORACLE_AXES)
    for value in diagnostic['axis_ablation'].values():
        assert value['delta_aupro'] >= 0
    assert next((tmp_path/'output').rglob('baseline_q95_metrics.json')).is_file()
