import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from fmad.methods.protected_geometry import (ProtectedGeometry, contraction_factor,
    edge_indices, moment_basis, photometric_views, stable_edges, transform_features)
from fmad.methods.query_reference import QueryReference
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import _write_image
from tests.test_query_reference import config
from tests.test_reference_matching import TinyPairBackbone


def test_trace_matching_retains_directions_and_matches_contraction():
    from fmad.methods.protected_geometry import trace_matched_factor
    torch.manual_seed(71)
    u = torch.linalg.qr(torch.randn(12, 4)).Q
    v = torch.linalg.qr(torch.randn(12, 3)).Q
    protected = contraction_factor(u, v)
    matched = trace_matched_factor(u, v)
    torch.testing.assert_close(matched.square().sum(), protected.square().sum())
    torch.testing.assert_close(matched @ matched.T,
                               (protected.square().sum() / 4) * (u @ u.T))
    assert torch.count_nonzero(trace_matched_factor(u, v[:, :0])) == 0
    assert trace_matched_factor(u, v[:, :0]).shape == (12, 0)
    q = F.normalize(torch.randn(5, 12), dim=-1)
    assert transform_features(q, trace_matched_factor(u, v[:, :0]), .5) is q
    assert trace_matched_factor(u[:, :0], v).shape == (12, 0)


def test_protected_linear_operator_is_bounded_and_preserves_v():
    torch.manual_seed(2)
    u = torch.linalg.qr(torch.randn(12, 4)).Q
    v = torch.linalg.qr(torch.randn(12, 3)).Q
    a = contraction_factor(u, v)
    t = torch.eye(12) - .5 * a @ a.T
    eig = torch.linalg.eigvalsh(t)
    assert float(eig.min()) >= .5 - 1e-6 and float(eig.max()) <= 1 + 1e-6
    torch.testing.assert_close(t @ v, v, atol=1e-6, rtol=1e-6)
    q = torch.randn(7, 12)
    torch.testing.assert_close(transform_features(q, a, .5, normalize=False), q @ t)
    torch.testing.assert_close(transform_features(q, a, .5), F.normalize(q @ t, dim=-1))
    torch.testing.assert_close(contraction_factor(u), u, rtol=0, atol=0)


def test_zero_evidence_and_absent_protection_are_identity():
    basis = moment_basis(torch.zeros(8, 8), 16)
    assert basis.shape == (8, 0)
    q = F.normalize(torch.randn(5, 8), dim=-1)
    a = contraction_factor(torch.eye(8)[:, :2], basis)
    assert a.shape == (8, 0)
    assert transform_features(q, a, .5) is q
    assert transform_features(q, torch.eye(8), 0.) is q


def test_stable_edges_need_both_views_and_do_not_wrap_rows():
    q = torch.randn(6, 8)
    left, right = edge_indices((2, 3), 'cpu')
    assert len(left) == 7 and (2, 3) not in set(zip(left.tolist(), right.tolist()))
    l, r = stable_edges(q, [q, q], (2, 3), 256, .9)
    assert len(l) == 7
    l, r = stable_edges(q, [q, -q], (2, 3), 256, .9)
    assert len(l) == 0


def test_photometric_views_do_not_mutate_input():
    image = np.full((4, 5, 3), 200, dtype=np.uint8)
    original = image.copy()
    views = list(photometric_views(image))
    np.testing.assert_array_equal(image, original)
    assert len(views) == 2 and all(v.dtype == np.uint8 and v.shape == image.shape for v in views)
    np.testing.assert_array_equal(views[0], np.full_like(image, 140))


@pytest.mark.parametrize('qr', [False, True])
def test_disabled_geometry_exactly_matches_qr(tmp_path, qr):
    paths = [_write_image(tmp_path / f'{i}.png', 17 * i) for i in range(3)]
    cfg = config(int(qr))
    reference = QueryReference(TinyPairBackbone(), cfg)
    method = ProtectedGeometry(TinyPairBackbone(), cfg)
    context = MethodContext('can', 0, 8, False, False)
    reference.fit(paths[:2], context); method.fit(paths[:2], context)
    np.testing.assert_array_equal(reference.predict(paths[2])[0], method.predict(paths[2])[0])
    for resolution in method.resolutions:
        torch.testing.assert_close(reference.scales[resolution], method.scales[resolution], rtol=0, atol=0)
        for a, b in zip(reference.banks[resolution]['layers'], method.banks[resolution]['layers']):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('protected', [False, True])
def test_geometry_fit_predict_is_finite_repeatable_and_backbone_frozen(tmp_path, protected):
    paths = [_write_image(tmp_path / f'{i}.png', 37 * i) for i in range(3)]
    cfg = dict(config(0), geometry_enabled=True, protect_contrasts=protected,
               nuisance_rank=2, protected_rank=2, geometry_evidence_tokens=8)
    backbone = TinyPairBackbone()
    original = copy.deepcopy(backbone.model.state_dict())
    method = ProtectedGeometry(backbone, cfg)
    context = MethodContext('can', 0, 8, False, False)
    method.fit(paths[:2], context)
    before = copy.deepcopy(method.geometry_factors)
    a = method.predict(paths[2])[0]
    b = method.predict(paths[2])[0]
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all() and a.shape == (12, 16)
    for r, factors in before.items():
        for x, y in zip(factors, method.geometry_factors[r]):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    for key, value in original.items():
        torch.testing.assert_close(value, backbone.model.state_dict()[key], rtol=0, atol=0)
    method.fit(paths[:2], context)
    np.testing.assert_allclose(a, method.predict(paths[2])[0], rtol=1e-5, atol=1e-5)
    assert method.object_metadata()['geometry_trainable_parameters'] == 0


@pytest.mark.parametrize('overrides', [dict(geometry_strength=1.), dict(protect_contrasts=True),
    dict(geometry_enabled=True,query_adaptation=True),dict(geometry_enabled='yes')])
def test_invalid_combinations_fail(overrides):
    with pytest.raises(ValueError):
        ProtectedGeometry(TinyPairBackbone(), dict(config(0), **overrides))
