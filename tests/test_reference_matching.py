from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from fmad.methods.bcma import BCMA
from fmad.methods.reference_matching import (ReferenceMatching, _distance_chunks, coreset_indices,
                                              local_scale, rank_normalize, trim_outlier_tokens)
from fmad.postprocess import component_quantile
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image


class TinyLateBackbone(TinyPSMAV2Backbone):
    feature_layers = (23,)

    def extract_features(self, image_tensor):
        return [super().extract_features(image_tensor)[-1]]


class TinyPairBackbone(TinyPSMAV2Backbone):
    feature_layers = (11, 23)

    def extract_features(self, image_tensor):
        outputs = super().extract_features(image_tensor)
        return [outputs[1], outputs[-1]]


def test_r0_matches_raw_bcma_control(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    config = {"resolutions": [8, 12], "bank_tokens_per_source": 4, "query_chunk_size": 4}
    reference = ReferenceMatching(TinyLateBackbone(), {**config, "variant": "R0"})
    baseline = BCMA(TinyLateBackbone(), {**config, "metric_adaptation": False})
    reference.fit(paths, context)
    baseline.fit(paths, context)
    expected, expected_score = baseline.predict(paths[0])
    actual, actual_score = reference.predict(paths[0])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual_score == expected_score
    assert reference.object_metadata()["normalization"] == "none"


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
@pytest.mark.parametrize("variant", ["R1", "R2"])
def test_r1_r2_fit_predict_are_finite_and_do_not_keep_records(tmp_path, device, variant):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    backbone = TinyPairBackbone()
    # Like the real wrapper, this extractor returns CPU features even when
    # method-side matching runs on CUDA. Fit also stages capped records on CPU.
    backbone.device = device
    method = ReferenceMatching(
        backbone,
        {"variant": variant, "feature_layers": [11, 23],
         "resolutions": [8, 12], "bank_tokens_per_source": 4,
         "query_chunk_size": 4, "context_lambda": 0.25},
    )
    method.fit(paths, context)
    anomaly_map, score = method.predict(paths[0])
    assert anomaly_map.shape == (12, 16)
    assert np.isfinite(anomaly_map).all() and np.isfinite(score)
    assert method.records == {}
    assert method.object_metadata()["trainable_parameters"] == 0
    for resolution, bank in method.banks.items():
        assert all(layer.device == torch.device(device) for layer in bank["layers"])
        assert method.scales[resolution].device.type == "cpu"
        assert torch.isfinite(method.scales[resolution]).all()
        if variant == "R2":
            assert bank["contexts"].device == torch.device(device)


def test_r2_reuses_one_reference_identity_across_layers():
    method = ReferenceMatching.__new__(ReferenceMatching)
    method.variant = "R2"
    method.layers = (11, 23)
    method.chunk = 2
    method.context_lambda = 0.0
    method._current_scales = torch.ones(2)
    query = [torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([[1., 0.], [0., 1.]])]
    bank = {"layers": [torch.tensor([[1., 0.], [0., 1.]]),
                        torch.tensor([[0., 1.], [1., 0.]])],
            "contexts": torch.tensor([[1., 0.], [0., 1.]])}
    result = method._score_tokens(query, (1, 2), bank)
    # The late layer chooses the opposite-index references; the first layer
    # must be scored against those same references, not its own nearest ones.
    torch.testing.assert_close(result, torch.full((2,), 0.5))


def test_constructor_rejects_invalid_resolution_lambda_and_layers():
    with pytest.raises(ValueError, match="resolutions"):
        ReferenceMatching(TinyLateBackbone(), {"variant": "R0", "resolutions": [0]})
    with pytest.raises(ValueError, match="context_lambda"):
        ReferenceMatching(TinyPairBackbone(), {"variant": "R2", "context_lambda": float("nan")})
    # R0 stays pinned to the single late layer.
    with pytest.raises(ValueError, match="feature layer"):
        ReferenceMatching(TinyLateBackbone(), {"variant": "R0", "feature_layers": [5, 23]})
    # The backbone must actually return the layers the method declares.
    with pytest.raises(ValueError, match="backbone feature_layers"):
        ReferenceMatching(TinyPairBackbone(), {"variant": "R1", "feature_layers": [5, 11]})
    with pytest.raises(ValueError, match="feature_layers"):
        ReferenceMatching(TinyPairBackbone(), {"variant": "R1", "feature_layers": []})


class TinyShallowPairBackbone(TinyPSMAV2Backbone):
    feature_layers = (5, 11)

    def extract_features(self, image_tensor):
        outputs = super().extract_features(image_tensor)
        return [outputs[0], outputs[1]]


class TinyTripleBackbone(TinyPSMAV2Backbone):
    feature_layers = (5, 11, 23)

    def extract_features(self, image_tensor):
        outputs = super().extract_features(image_tensor)
        return [outputs[0], outputs[1], outputs[-1]]


def test_r1_accepts_shallow_and_three_layer_stacks(tmp_path):
    """Layer choice is an ablation axis for R1; only the default is pinned."""
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}

    shallow = ReferenceMatching(TinyShallowPairBackbone(),
                                {**common, "feature_layers": [5, 11]})
    assert shallow.layers == (5, 11)
    shallow.fit(paths, context)
    smap, _ = shallow.predict(paths[0])
    assert np.isfinite(smap).all()

    triple = ReferenceMatching(TinyTripleBackbone(),
                               {**common, "feature_layers": [5, 11, 23]})
    assert triple.layers == (5, 11, 23)
    triple.fit(paths, context)
    for resolution in triple.resolutions:
        assert len(triple.banks[resolution]["layers"]) == 3
        assert triple.scales[resolution].shape[0] == 3
    tmap, _ = triple.predict(paths[0])
    assert np.isfinite(tmap).all()


def test_r1_default_layers_are_unchanged(tmp_path):
    """Omitting feature_layers must stay byte-identical to the [11, 23] default."""
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    default = ReferenceMatching(TinyPairBackbone(), dict(common))
    explicit = ReferenceMatching(TinyPairBackbone(), {**common, "feature_layers": [11, 23]})
    assert default.layers == (11, 23) == explicit.layers
    default.fit(paths, context)
    explicit.fit(paths, context)
    expected, expected_score = explicit.predict(paths[0])
    actual, actual_score = default.predict(paths[0])
    np.testing.assert_array_equal(actual, expected)
    assert actual_score == expected_score


def test_prediction_component_quantile_matches_shared_offline_transform(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    base = ReferenceMatching(TinyPairBackbone(), common)
    online = ReferenceMatching(
        TinyPairBackbone(),
        {**common, "prediction_postprocess": "component_quantile",
         "component_percentile": 99.0, "component_q": 1.0},
    )
    base.fit(paths, context)
    online.fit(paths, context)

    base_map, _ = base.predict(paths[0])
    online_map, online_score = online.predict(paths[0])
    expected = component_quantile(base_map, percentile=99.0, q=1.0)

    np.testing.assert_array_equal(online_map, expected)
    assert online_score == float(np.max(expected))
    metadata = online.object_metadata()
    assert metadata["prediction_postprocess"] == "component_quantile"
    assert metadata["score_definition"].endswith("_postprocess_component_quantile")


@pytest.mark.parametrize(
    ("key", "value"),
    [("prediction_postprocess", "bogus"), ("component_percentile", 101),
     ("component_q", -0.1), ("component_dilate_size", True)],
)
def test_prediction_postprocess_rejects_invalid_config(key, value):
    with pytest.raises(ValueError):
        ReferenceMatching(
            TinyPairBackbone(),
            {"variant": "R1", "feature_layers": [11, 23], key: value},
        )


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
def test_r1_r2_share_sampling_identities_and_fixed_scales(tmp_path, device):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 7, 8, False, False)
    common = {"feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    backbone = TinyPairBackbone()
    backbone.device = device
    r1 = ReferenceMatching(backbone, {**common, "variant": "R1"})
    r2 = ReferenceMatching(backbone, {**common, "variant": "R2"})
    r1.fit(paths, context)
    r2.fit(paths, context)
    for resolution in common["resolutions"]:
        assert r1.banks[resolution]["owners"] == r2.banks[resolution]["owners"]
        torch.testing.assert_close(r1.scales[resolution], r2.scales[resolution], rtol=0, atol=0)
    assert (r1.object_metadata()["normalization_reference"] ==
            "independent_layer_nn_leave_source_out_fixed_across_R1_R2")
    assert (r2.object_metadata()["normalization_reference"] ==
            "independent_layer_nn_leave_source_out_fixed_across_R1_R2")


def test_bank_seed_changes_only_the_token_subsample(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    default = ReferenceMatching(TinyPairBackbone(), common)
    explicit = ReferenceMatching(TinyPairBackbone(), {**common, "bank_seed": 0})
    other = ReferenceMatching(TinyPairBackbone(), {**common, "bank_seed": 5})
    for method in (default, explicit, other):
        method.fit(paths, context)
    for resolution in common["resolutions"]:
        # bank_seed equal to context.seed reproduces the default sampling exactly.
        torch.testing.assert_close(default.banks[resolution]["layers"][0],
                                   explicit.banks[resolution]["layers"][0], rtol=0, atol=0)
        # Same supports (owners), different token subsample.
        assert other.banks[resolution]["owners"] == default.banks[resolution]["owners"]
    assert any(not torch.equal(other.banks[r]["layers"][0], default.banks[r]["layers"][0])
               for r in common["resolutions"])
    assert default.object_metadata()["bank_seed"] is None
    assert other.object_metadata()["bank_sampling_seed_source"] == "bank_seed_override_same_supports"
    with pytest.raises(ValueError):
        ReferenceMatching(TinyPairBackbone(), {**common, "bank_seed": True})
    with pytest.raises(ValueError):
        ReferenceMatching(TinyPairBackbone(), {**common, "bank_seed": -1})


def test_default_config_matches_previous_behaviour():
    method = ReferenceMatching(TinyPairBackbone(), {"variant": "R1", "feature_layers": [11, 23]})
    assert method.object_metadata()["bank_selection"] == "random"
    assert method.object_metadata()["knn_k"] == 1
    assert method.bank_selection == "random"
    assert method.knn_k == 1
    torch.manual_seed(0)
    q = F.normalize(torch.randn(6, 5), dim=-1)
    b = F.normalize(torch.randn(9, 5), dim=-1)
    expected = (1.0 - q @ b.T).amin(1)
    torch.testing.assert_close(_distance_chunks(q, b, 4), expected)
    torch.testing.assert_close(_distance_chunks(q, b, 4, k=1), expected)


def test_coreset_indices_are_deterministic_distinct_and_cover():
    distinct = F.normalize(torch.tensor([[1., 0., 0.],
                                          [0., 1., 0.],
                                          [0., 0., 1.]]), dim=-1)
    features = torch.cat([distinct, distinct], dim=0)
    result_a = coreset_indices(features, 5)
    result_b = coreset_indices(features, 5)
    assert torch.equal(result_a, result_b)
    assert len(set(result_a.tolist())) == len(result_a)
    assert len(result_a) == 3
    mean = F.normalize(features.mean(dim=0, keepdim=True), dim=-1)
    expected_start = torch.argmax((features @ mean.T).squeeze(1)).item()
    assert result_a[0].item() == expected_start


def test_coreset_bank_uses_one_identity_across_layers(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    config = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4, "bank_selection": "coreset"}
    method_a = ReferenceMatching(TinyPairBackbone(), config)
    method_b = ReferenceMatching(TinyPairBackbone(), config)
    method_a.fit(paths, context)
    method_b.fit(paths, context)
    metadata = method_a.object_metadata()
    assert metadata["bank_selection"] == "coreset"
    assert metadata["bank_sampling_seed_source"] == "deterministic_coreset_no_seed"
    for resolution in config["resolutions"]:
        bank_a = method_a.banks[resolution]["layers"]
        bank_b = method_b.banks[resolution]["layers"]
        assert len(bank_a[0]) == len(bank_a[1])
        for layer_a, layer_b in zip(bank_a, bank_b):
            torch.testing.assert_close(layer_a, layer_b, rtol=0, atol=0)


def test_knn_mean_distance(tmp_path):
    torch.manual_seed(1)
    q = F.normalize(torch.randn(5, 4), dim=-1)
    b = F.normalize(torch.randn(6, 4), dim=-1)
    distances = 1.0 - q @ b.T
    expected = distances.topk(3, dim=1, largest=False).values.mean(dim=1)
    torch.testing.assert_close(_distance_chunks(q, b, 2, k=3), expected)
    expected_all = distances.mean(dim=1)
    torch.testing.assert_close(_distance_chunks(q, b, 2, k=100), expected_all)

    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    method = ReferenceMatching(
        TinyPairBackbone(),
        {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
         "bank_tokens_per_source": 4, "query_chunk_size": 4, "knn_k": 2},
    )
    method.fit(paths, context)
    anomaly_map, score = method.predict(paths[0])
    assert np.isfinite(anomaly_map).all() and np.isfinite(score)
    assert method.object_metadata()["knn_k"] == 2


def test_constructor_rejects_invalid_bank_selection_and_knn_k():
    common = {"variant": "R1", "feature_layers": [11, 23]}
    with pytest.raises(ValueError, match="bank_selection"):
        ReferenceMatching(TinyPairBackbone(), {**common, "bank_selection": "foo"})
    with pytest.raises(ValueError, match="knn_k"):
        ReferenceMatching(TinyPairBackbone(), {**common, "knn_k": 0})
    with pytest.raises(ValueError, match="knn_k"):
        ReferenceMatching(TinyPairBackbone(), {**common, "knn_k": True})
    with pytest.raises(ValueError, match="knn_k"):
        ReferenceMatching(TinyPairBackbone(), {**common, "knn_k": 1.5})


def test_bank_outlier_trim_and_bags_default_to_disabled(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    config = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    method_a = ReferenceMatching(TinyPairBackbone(), config)
    method_b = ReferenceMatching(TinyPairBackbone(), config)
    method_a.fit(paths, context)
    method_b.fit(paths, context)
    metadata = method_a.object_metadata()
    assert metadata["bank_outlier_trim"] == 0.0
    assert metadata["bank_bags"] == 1
    for resolution in config["resolutions"]:
        for layer_a, layer_b in zip(method_a.banks[resolution]["layers"],
                                    method_b.banks[resolution]["layers"]):
            torch.testing.assert_close(layer_a, layer_b, rtol=0, atol=0)


def test_trim_outlier_tokens_returns_new_records_without_mutating_input():
    torch.manual_seed(0)
    records = []
    for source in range(3):
        layer0 = F.normalize(torch.randn(4, 3), dim=-1)
        layer1 = F.normalize(torch.randn(4, 3), dim=-1)
        records.append({"source": source, "layers": [layer0, layer1],
                        "indices": torch.arange(source * 4, source * 4 + 4)})
    original_indices = [record["indices"].clone() for record in records]
    trimmed = trim_outlier_tokens(records, chunk=4, fraction=0.5)

    assert trimmed is not records
    for original, result in zip(records, trimmed):
        assert result is not original

    total_before = sum(len(record["indices"]) for record in records)
    total_after = sum(len(record["indices"]) for record in trimmed)
    assert total_after == 6  # round(0.5 * 12)
    assert total_after < total_before
    for result in trimmed:
        assert len(result["indices"]) >= 1

    # Input records (and their tensors) must be untouched.
    for record, before in zip(records, original_indices):
        torch.testing.assert_close(record["indices"], before)
        assert len(record["layers"][0]) == 4


def test_bank_outlier_trim_in_fit_shrinks_bank_and_records_removed_counts(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    config = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4, "bank_outlier_trim": 0.5}
    method = ReferenceMatching(TinyPairBackbone(), config)
    method.fit(paths, context)
    metadata = method.object_metadata()
    assert metadata["bank_outlier_trim"] == 0.5
    for resolution in config["resolutions"]:
        bank = method.banks[resolution]
        bank_size = len(bank["layers"][0])
        assert abs(bank_size - 6) <= 1
        for source in range(3):
            assert bank["owners"].count(source) >= 1
        removed = metadata["bank_trim_removed"][str(resolution)]
        assert removed == 12 - bank_size
    anomaly_map, score = method.predict(paths[0])
    assert np.isfinite(anomaly_map).all() and np.isfinite(score)


def test_bank_bags_builds_independent_banks_matching_bag_zero_to_default(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    default = ReferenceMatching(TinyPairBackbone(), common)
    bagged = ReferenceMatching(TinyPairBackbone(), {**common, "bank_bags": 3})
    default.fit(paths, context)
    bagged.fit(paths, context)
    for resolution in common["resolutions"]:
        bags = bagged.bag_banks[resolution]
        assert len(bags) == 3
        for layer_default, layer_bag0 in zip(default.banks[resolution]["layers"], bags[0]["layers"]):
            torch.testing.assert_close(layer_default, layer_bag0, rtol=0, atol=0)
        distinct = False
        for i in range(len(bags)):
            for j in range(i + 1, len(bags)):
                if any(not torch.equal(a, b) for a, b in zip(bags[i]["layers"], bags[j]["layers"])):
                    distinct = True
        assert distinct
    anomaly_map, score = bagged.predict(paths[0])
    assert np.isfinite(anomaly_map).all() and np.isfinite(score)
    metadata = bagged.object_metadata()
    assert metadata["bank_bags"] == 3
    assert metadata["bank_sampling_seed_source"] == "bagged_random_subsamples"
    assert metadata["score_definition"] == "bag_mean_layer_distance_with_support_loo_mad"
    assert set(metadata["bank_trim_removed"].keys()) == {str(r) for r in common["resolutions"]}


def test_bank_bags_and_trim_validation_errors():
    common = {"variant": "R1", "feature_layers": [11, 23]}
    with pytest.raises(ValueError):
        ReferenceMatching(TinyPairBackbone(), {**common, "bank_bags": 2, "bank_selection": "coreset"})
    with pytest.raises(ValueError):
        ReferenceMatching(TinyPairBackbone(), {"variant": "R2", "feature_layers": [11, 23], "bank_bags": 2})
    with pytest.raises(ValueError, match="bank_outlier_trim"):
        ReferenceMatching(TinyPairBackbone(), {**common, "bank_outlier_trim": 1.0})
    with pytest.raises(ValueError, match="bank_outlier_trim"):
        ReferenceMatching(TinyPairBackbone(), {**common, "bank_outlier_trim": -0.1})
    with pytest.raises(ValueError, match="bank_outlier_trim"):
        ReferenceMatching(TinyPairBackbone(), {**common, "bank_outlier_trim": True})
    with pytest.raises(ValueError, match="bank_bags"):
        ReferenceMatching(TinyPairBackbone(), {**common, "bank_bags": 0})


def test_bank_bags_score_matches_single_bank_when_bags_sample_all_tokens(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    # A huge cap makes every bag's subsample equal to the full (capped-only-
    # by-availability) token set, just reordered; nearest-neighbour distances
    # and the leave-source-out MAD scale are invariant to that reordering, so
    # bag-averaging must reproduce the single-bank prediction exactly.
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8],
              "bank_tokens_per_source": 10_000, "query_chunk_size": 4}
    single = ReferenceMatching(TinyPairBackbone(), common)
    bagged = ReferenceMatching(TinyPairBackbone(), {**common, "bank_bags": 2})
    single.fit(paths, context)
    bagged.fit(paths, context)
    expected, expected_score = single.predict(paths[0])
    actual, actual_score = bagged.predict(paths[0])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    assert abs(actual_score - expected_score) < 1e-6


def test_scale_fusion_defaults_to_mean_and_is_byte_identical(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    default = ReferenceMatching(TinyPairBackbone(), common)
    explicit = ReferenceMatching(TinyPairBackbone(), {**common, "scale_fusion": "mean"})
    assert default.object_metadata()["scale_fusion"] == "mean"
    default.fit(paths, context)
    explicit.fit(paths, context)
    expected, expected_score = default.predict(paths[0])
    actual, actual_score = explicit.predict(paths[0])
    assert np.array_equal(actual, expected)
    assert actual_score == expected_score
    assert default.object_metadata()["score_definition"] == explicit.object_metadata()["score_definition"]


def test_rank_normalize_known_values_and_edge_length():
    result = rank_normalize(torch.tensor([3.0, 1.0, 2.0]))
    torch.testing.assert_close(result, torch.tensor([1.0, 0.0, 0.5]))

    single = rank_normalize(torch.tensor([5.0]))
    torch.testing.assert_close(single, torch.tensor([0.0]))

    torch.manual_seed(3)
    random_scores = torch.randn(20)
    normalized = rank_normalize(random_scores)
    assert torch.all(normalized >= 0.0) and torch.all(normalized <= 1.0)
    order = torch.argsort(random_scores)
    ordered_norm = normalized[order]
    assert torch.all(ordered_norm[1:] >= ordered_norm[:-1])


def test_scale_fusion_max_is_finite_and_at_least_mean(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    mean_method = ReferenceMatching(TinyPairBackbone(), common)
    max_method = ReferenceMatching(TinyPairBackbone(), {**common, "scale_fusion": "max"})
    mean_method.fit(paths, context)
    max_method.fit(paths, context)
    mean_map, _ = mean_method.predict(paths[0])
    max_map, max_score = max_method.predict(paths[0])
    assert np.isfinite(max_map).all() and np.isfinite(max_score)
    assert max_map.shape == mean_map.shape
    assert np.all(max_map >= mean_map - 1e-6)
    metadata = max_method.object_metadata()
    assert metadata["scale_fusion"] == "max"
    assert metadata["score_definition"].endswith("_scale_fusion_max")


def test_scale_fusion_rank_is_finite_and_bounded(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    config = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4, "scale_fusion": "rank"}
    method = ReferenceMatching(TinyPairBackbone(), config)
    method.fit(paths, context)
    image = _write_image(tmp_path / "query.png", 99)
    anomaly_map, score = method.predict(image)
    assert np.isfinite(anomaly_map).all() and np.isfinite(score)
    assert np.all(anomaly_map >= 0.0) and np.all(anomaly_map <= 1.0)
    metadata = method.object_metadata()
    assert metadata["scale_fusion"] == "rank"
    assert metadata["score_definition"].endswith("_scale_fusion_rank")


def test_scale_fusion_rejects_bogus_value():
    with pytest.raises(ValueError, match="scale_fusion"):
        ReferenceMatching(TinyPairBackbone(), {"variant": "R1", "feature_layers": [11, 23],
                                                "scale_fusion": "bogus"})


def test_tail_normalisation_defaults_off_and_is_byte_identical(tmp_path):
    """score_normalization defaults to 'mad'; the existing path must not move."""
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    default = ReferenceMatching(TinyPairBackbone(), dict(common))
    explicit = ReferenceMatching(TinyPairBackbone(), {**common, "score_normalization": "mad"})
    assert default.score_normalization == "mad"
    default.fit(paths, context)
    explicit.fit(paths, context)
    expected, expected_score = explicit.predict(paths[0])
    actual, actual_score = default.predict(paths[0])
    np.testing.assert_array_equal(actual, expected)
    assert actual_score == expected_score
    assert default.object_metadata()["score_normalization"] == "mad"


def test_tail_normalisation_is_selectable_and_changes_the_score(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    mad = ReferenceMatching(TinyPairBackbone(), dict(common))
    tail = ReferenceMatching(TinyPairBackbone(), {**common, "score_normalization": "tail"})
    assert tail.score_normalization == "tail"
    mad.fit(paths, context)
    tail.fit(paths, context)
    for resolution in tail.resolutions:
        assert len(tail.tails[resolution]) == len(tail.layers)
        for params in tail.tails[resolution]:
            assert params["beta"] > 0 and 0.0 < params["p_u"] <= 1.0
            assert params["sample"].numel() > 0
    tail_map, _ = tail.predict(paths[0])
    mad_map, _ = mad.predict(paths[0])
    assert np.isfinite(tail_map).all()
    assert not np.allclose(tail_map, mad_map)
    assert tail.object_metadata()["score_normalization"] == "tail"


def test_tail_transform_is_monotone_and_calibrated():
    """Larger distances must never score lower, and the transform must map the
    support's own normal quantiles to the declared FPR levels."""
    method = ReferenceMatching(TinyPairBackbone(),
                               {"variant": "R1", "feature_layers": [11, 23],
                                "score_normalization": "tail"})
    sample = torch.linspace(0.1, 1.0, 400)
    params = method._tail_params(sample)
    probe = torch.tensor([0.05, 0.2, 0.5, 0.9, 1.5, 3.0])
    scored = method._tail_transform(params, probe)
    assert torch.all(scored[1:] >= scored[:-1])
    assert torch.isfinite(scored).all()
    # the 95th percentile of the support sample maps to about -log(0.05)
    at_q95 = method._tail_transform(params, torch.quantile(sample, 0.95).reshape(1))
    assert abs(float(at_q95) - float(-torch.log(torch.tensor(0.05)))) < 0.25


def test_tail_normalisation_rejects_unknown_mode():
    with pytest.raises(ValueError, match="score_normalization"):
        ReferenceMatching(TinyPairBackbone(), {"variant": "R1", "feature_layers": [11, 23],
                                               "score_normalization": "zscore"})


def test_tailscale_divides_by_the_normal_tail_not_the_centre(tmp_path):
    """tailscale keeps the linear divide-by-constant form (magnitude preserved)
    but normalises by the component's own LOO normal q95."""
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    mad = ReferenceMatching(TinyPairBackbone(), dict(common))
    ts = ReferenceMatching(TinyPairBackbone(), {**common, "score_normalization": "tailscale"})
    mad.fit(paths, context)
    ts.fit(paths, context)
    for resolution in ts.resolutions:
        for index in range(len(ts.layers)):
            # the tailscale divisor is the fitted q95, not 1.4826 * MAD
            torch.testing.assert_close(ts.scales[resolution][index],
                                       ts.tails[resolution][index]["u"].clamp_min(1e-4),
                                       rtol=0, atol=0)
    ts_map, _ = ts.predict(paths[0])
    mad_map, _ = mad.predict(paths[0])
    assert np.isfinite(ts_map).all()
    # linear rescale per component: ordering within a single-component score is kept
    assert ts.object_metadata()["score_normalization"] == "tailscale"
    assert ts.object_metadata()["tail_quantile"] == 0.95


def test_tail_quantile_is_validated():
    common = {"variant": "R1", "feature_layers": [11, 23]}
    for bad in (0.4, 1.0, 1.5, True, "0.9"):
        with pytest.raises(ValueError, match="tail_quantile"):
            ReferenceMatching(TinyPairBackbone(), {**common, "tail_quantile": bad})


def test_layer_fusion_defaults_to_mean_and_is_byte_identical(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    default = ReferenceMatching(TinyPairBackbone(), common)
    explicit = ReferenceMatching(TinyPairBackbone(), {**common, "layer_fusion": "mean"})
    assert default.object_metadata()["layer_fusion"] == "mean"
    assert default.object_metadata()["layer_fusion_q"] == 0.9
    default.fit(paths, context)
    explicit.fit(paths, context)
    expected, expected_score = default.predict(paths[0])
    actual, actual_score = explicit.predict(paths[0])
    assert np.array_equal(actual, expected)
    assert actual_score == expected_score
    assert default.object_metadata()["score_definition"] == explicit.object_metadata()["score_definition"]


def test_layer_fusion_max_is_finite_and_at_least_mean(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    mean_method = ReferenceMatching(TinyPairBackbone(), common)
    max_method = ReferenceMatching(TinyPairBackbone(), {**common, "layer_fusion": "max"})
    mean_method.fit(paths, context)
    max_method.fit(paths, context)
    mean_map, _ = mean_method.predict(paths[0])
    max_map, max_score = max_method.predict(paths[0])
    assert np.isfinite(max_map).all() and np.isfinite(max_score)
    assert max_map.shape == mean_map.shape
    assert np.all(max_map >= mean_map - 1e-6)


def test_layer_fusion_quantile_one_equals_max_and_half_is_between_extremes(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    max_method = ReferenceMatching(TinyPairBackbone(), {**common, "layer_fusion": "max"})
    q1_method = ReferenceMatching(
        TinyPairBackbone(), {**common, "layer_fusion": "quantile", "layer_fusion_q": 1.0})
    q5_method = ReferenceMatching(
        TinyPairBackbone(), {**common, "layer_fusion": "quantile", "layer_fusion_q": 0.5})
    max_method.fit(paths, context)
    q1_method.fit(paths, context)
    q5_method.fit(paths, context)
    max_map, _ = max_method.predict(paths[0])
    q1_map, _ = q1_method.predict(paths[0])
    q5_map, _ = q5_method.predict(paths[0])
    np.testing.assert_allclose(q1_map, max_map, rtol=0, atol=1e-6)

    # Per-layer scores, two layers: the elementwise median (q=0.5) must lie
    # between the elementwise min and max across the two layers.
    method = ReferenceMatching.__new__(ReferenceMatching)
    method.variant = "R1"
    method.layer_fusion = "quantile"
    method.layer_fusion_q = 0.5
    scores = [torch.tensor([1.0, 5.0, 2.0]), torch.tensor([3.0, 1.0, 8.0])]
    fused = method._fuse_layers(scores)
    lo = torch.minimum(scores[0], scores[1])
    hi = torch.maximum(scores[0], scores[1])
    assert torch.all(fused >= lo - 1e-6) and torch.all(fused <= hi + 1e-6)


def test_layer_fusion_max_composes_with_tailscale_normalization(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    config = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4,
              "layer_fusion": "max", "score_normalization": "tailscale"}
    method = ReferenceMatching(TinyPairBackbone(), config)
    method.fit(paths, context)
    anomaly_map, score = method.predict(paths[0])
    assert np.isfinite(anomaly_map).all() and np.isfinite(score)
    metadata = method.object_metadata()
    assert metadata["score_definition"].endswith("_layer_fusion_max")


def test_both_scale_and_layer_fusion_suffixes_appear_in_order(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    config = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4,
              "scale_fusion": "max", "layer_fusion": "max"}
    method = ReferenceMatching(TinyPairBackbone(), config)
    method.fit(paths, context)
    method.predict(paths[0])
    score_definition = method.object_metadata()["score_definition"]
    scale_pos = score_definition.index("_scale_fusion_max")
    layer_pos = score_definition.index("_layer_fusion_max")
    assert scale_pos < layer_pos


def test_layer_fusion_rejects_invalid_values_and_q_and_non_r1_variant():
    common = {"variant": "R1", "feature_layers": [11, 23]}
    with pytest.raises(ValueError, match="layer_fusion"):
        ReferenceMatching(TinyPairBackbone(), {**common, "layer_fusion": "bogus"})
    for bad_q in (0.0, 1.5, True):
        with pytest.raises(ValueError, match="layer_fusion_q"):
            ReferenceMatching(TinyPairBackbone(), {**common, "layer_fusion": "quantile",
                                                    "layer_fusion_q": bad_q})
    with pytest.raises(ValueError, match="layer_fusion"):
        ReferenceMatching(TinyLateBackbone(), {"variant": "R0", "feature_layers": [23],
                                                "layer_fusion": "max"})


class TinyNoisyPairBackbone(TinyPairBackbone):
    """TinyPairBackbone with a deterministic per-token perturbation.

    The plain tiny backbone's features are a rank-3 linear map of a smooth
    pixel gradient, so leave-source-out nearest-neighbour distances are all
    near-zero and every scale (global or local) collapses onto the same
    1e-4 floor -- useless for telling "off" and "scale" apart. A per-token,
    per-channel additive offset (NOT a per-row scalar factor, which cosine
    similarity is invariant to) breaks that collinearity deterministically
    (no RNG) while keeping fit/predict reproducible across separately-
    constructed instances.
    """

    def extract_features(self, image_tensor):
        outputs = super().extract_features(image_tensor)
        perturbed = []
        for layer in outputs:
            n, dim = layer.shape
            rows = torch.arange(n, dtype=torch.float32).unsqueeze(1)
            cols = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
            offset = 0.2 * torch.sin(rows * 0.9 + cols * 1.7)
            perturbed.append(layer + offset)
        return perturbed


class TinyDuplicatePairBackbone(TinyPSMAV2Backbone):
    """Both declared layers return the identical feature tensor.

    Used to force equal per-layer local scales without monkeypatching, so
    "weight" mode's precision-weighted fusion is exercisable against a plain
    mean.
    """
    feature_layers = (11, 23)

    def extract_features(self, image_tensor):
        outputs = super().extract_features(image_tensor)
        return [outputs[-1], outputs[-1]]


def test_local_reliability_defaults_off_and_is_byte_identical(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    default = ReferenceMatching(TinyPairBackbone(), dict(common))
    explicit = ReferenceMatching(TinyPairBackbone(), {**common, "local_reliability": "off"})
    assert default.local_reliability == "off"
    default.fit(paths, context)
    explicit.fit(paths, context)
    expected, expected_score = explicit.predict(paths[0])
    actual, actual_score = default.predict(paths[0])
    assert np.array_equal(actual, expected)
    assert actual_score == expected_score
    assert default.object_metadata()["local_reliability"] == "off"


def test_distance_chunks_return_indices_matches_gathered_distances():
    torch.manual_seed(0)
    query = F.normalize(torch.randn(11, 5), dim=-1)
    bank = F.normalize(torch.randn(9, 5), dim=-1)

    # Baseline (no indices) call is unchanged.
    baseline = _distance_chunks(query, bank, chunk=4, k=3)
    values, indices = _distance_chunks(query, bank, chunk=4, k=3, return_indices=True)
    torch.testing.assert_close(values, baseline)
    assert indices.shape == (11, 3)
    distances = 1.0 - query @ bank.T
    gathered = distances.gather(1, indices)
    torch.testing.assert_close(gathered.mean(dim=1), values)
    # Every row's gathered distances must equal its k smallest distances.
    expected_sorted = distances.topk(3, dim=1, largest=False).values
    torch.testing.assert_close(torch.sort(gathered, dim=1).values, expected_sorted)

    # k == 1 path (argmin companion).
    baseline_k1 = _distance_chunks(query, bank, chunk=4, k=1)
    values_k1, indices_k1 = _distance_chunks(query, bank, chunk=4, k=1, return_indices=True)
    torch.testing.assert_close(values_k1, baseline_k1)
    assert indices_k1.shape == (11, 1)
    torch.testing.assert_close(distances.gather(1, indices_k1).squeeze(1), values_k1)


def test_local_scale_is_pure_and_matches_shrinkage_limits():
    torch.manual_seed(1)
    neighbour_index = torch.randint(0, 6, (5, 4))
    neighbour_index_copy = neighbour_index.clone()
    loo = torch.rand(6) + 0.1
    loo_copy = loo.clone()
    global_scale = torch.tensor(2.0)

    zero = local_scale(neighbour_index, loo, global_scale, k=4, shrinkage=0.0)
    expected_local_mean = loo[neighbour_index].mean(dim=1).clamp_min(1e-4)
    torch.testing.assert_close(zero, expected_local_mean)

    huge = local_scale(neighbour_index, loo, global_scale, k=4, shrinkage=1e8)
    torch.testing.assert_close(huge, global_scale.expand_as(huge), atol=1e-3, rtol=1e-3)

    result = local_scale(neighbour_index, loo, global_scale, k=4, shrinkage=8.0)
    assert result.shape == (5,)
    assert (result >= 1e-4).all()

    # Pure: neither input tensor was mutated.
    torch.testing.assert_close(neighbour_index, neighbour_index_copy)
    torch.testing.assert_close(loo, loo_copy)


def test_local_reliability_scale_mode_is_finite_and_changes_prediction(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    off = ReferenceMatching(TinyNoisyPairBackbone(), dict(common))
    scaled = ReferenceMatching(TinyNoisyPairBackbone(), {**common, "local_reliability": "scale"})
    off.fit(paths, context)
    scaled.fit(paths, context)
    off_map, _ = off.predict(paths[0])
    scaled_map, _ = scaled.predict(paths[0])
    assert np.isfinite(scaled_map).all()
    assert scaled_map.shape == off_map.shape
    assert not np.array_equal(scaled_map, off_map)
    assert "_local_scale" in scaled.object_metadata()["score_definition"]


def test_local_reliability_weight_mode_is_finite(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    weighted = ReferenceMatching(TinyPairBackbone(), {**common, "local_reliability": "weight"})
    weighted.fit(paths, context)
    weighted_map, weighted_score = weighted.predict(paths[0])
    assert np.isfinite(weighted_map).all() and np.isfinite(weighted_score)
    assert "_local_weight" in weighted.object_metadata()["score_definition"]


def test_local_reliability_weight_mode_matches_mean_when_layers_are_identical(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}
    mean_method = ReferenceMatching(TinyDuplicatePairBackbone(), dict(common))
    weighted_method = ReferenceMatching(TinyDuplicatePairBackbone(),
                                        {**common, "local_reliability": "weight"})
    mean_method.fit(paths, context)
    weighted_method.fit(paths, context)
    expected, expected_score = mean_method.predict(paths[0])
    actual, actual_score = weighted_method.predict(paths[0])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert abs(actual_score - expected_score) < 1e-5


def test_local_reliability_composes_with_tailscale_and_bank_bags(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    method = ReferenceMatching(TinyPairBackbone(), {
        "variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
        "bank_tokens_per_source": 4, "query_chunk_size": 4,
        "local_reliability": "scale", "score_normalization": "tailscale", "bank_bags": 2,
    })
    method.fit(paths, context)
    result, score = method.predict(paths[0])
    assert np.isfinite(result).all() and np.isfinite(score)
    assert result.shape == (12, 16)


def test_local_reliability_constructor_rejects_invalid_configuration():
    common = {"variant": "R1", "feature_layers": [11, 23]}
    with pytest.raises(ValueError, match="local_reliability"):
        ReferenceMatching(TinyPairBackbone(), {**common, "local_reliability": "bogus"})
    with pytest.raises(ValueError, match="local_reliability_shrinkage"):
        ReferenceMatching(TinyPairBackbone(), {**common, "local_reliability": "scale",
                                                "local_reliability_shrinkage": -1.0})
    with pytest.raises(ValueError, match="local_reliability_k"):
        ReferenceMatching(TinyPairBackbone(), {**common, "local_reliability": "scale",
                                                "local_reliability_k": 0})
    with pytest.raises(ValueError, match="local_reliability"):
        ReferenceMatching(TinyLateBackbone(), {"variant": "R0", "feature_layers": [23],
                                                "local_reliability": "scale"})


def test_residual_spike_layer_fusion_behavior_and_validation():
    common = {"variant": "R1", "feature_layers": [11, 23]}
    # Validation errors
    with pytest.raises(ValueError, match="layer_fusion_tau"):
        ReferenceMatching(TinyPairBackbone(), {**common, "layer_fusion": "residual_spike", "layer_fusion_tau": -0.1})
    with pytest.raises(ValueError, match="layer_fusion_gamma"):
        ReferenceMatching(TinyPairBackbone(), {**common, "layer_fusion": "residual_spike", "layer_fusion_gamma": -1.0})
    with pytest.raises(ValueError, match="layer_fusion_shallow_idx"):
        ReferenceMatching(TinyPairBackbone(), {**common, "layer_fusion": "residual_spike", "layer_fusion_shallow_idx": -1})

    # Test fusion logic directly
    method = ReferenceMatching(TinyPairBackbone(), {
        **common,
        "layer_fusion": "residual_spike",
        "layer_fusion_tau": 1.0,
        "layer_fusion_gamma": 2.0,
        "layer_fusion_shallow_idx": 0,
    })
    # Case 1: shallow score <= tau (spike is 0, equals mean of deep layers)
    s0 = torch.tensor([0.5, 0.9, 1.0])
    s1 = torch.tensor([2.0, 2.0, 2.0])
    s2 = torch.tensor([4.0, 4.0, 4.0])
    fused = method._fuse_layers([s0, s1, s2])
    expected_base = (s1 + s2) / 2.0
    assert torch.allclose(fused, expected_base)

    # Case 2: shallow score > tau (spike added with gamma)
    s0 = torch.tensor([1.5, 2.0, 3.0])
    fused = method._fuse_layers([s0, s1, s2])
    spike = torch.tensor([0.5, 1.0, 2.0]) * 2.0
    expected = expected_base + spike
    assert torch.allclose(fused, expected)


def test_residual_spike_support_quantile_tau_mode(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(4)]
    context = MethodContext("can", 0, 8, False, False)
    config = {
        "variant": "R1",
        "feature_layers": [11, 23],
        "resolutions": [8],
        "bank_tokens_per_source": 4,
        "query_chunk_size": 4,
        "layer_fusion": "residual_spike",
        "layer_fusion_tau_mode": "support_quantile",
        "layer_fusion_tau_q": 0.99,
        "score_normalization": "tailscale",
    }
    method = ReferenceMatching(TinyPairBackbone(), config)
    method.fit(paths, context)
    tail_dict = method.tails[8][0]
    assert "spike_tau" in tail_dict
    assert tail_dict["spike_tau"] > 0

    # Test predict works with support_quantile tau
    pred, max_val = method.predict(paths[0])
    assert np.isfinite(pred).all() and np.isfinite(max_val)
