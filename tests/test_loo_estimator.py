"""Tests for loo_estimator configuration option in reference_matching."""

import numpy as np
import pytest
import torch
from PIL import Image

from fmad.protocol import MethodContext

from fmad.methods.reference_matching import ReferenceMatching
from tests.test_reference_matching import TinyPairBackbone, _write_image


class TestLooEstimatorOption:
    """Test loo_estimator config option for leave-one-out distance estimation."""

    @pytest.fixture
    def backbone(self):
        return TinyPairBackbone()

    @pytest.fixture
    def base_config(self):
        """Minimal R1 config for testing."""
        return {
            "variant": "R1",
            "feature_layers": [11, 23],
            "resolutions": [8, 12],
            "bank_tokens_per_source": 4,
            "query_chunk_size": 4,
        }

    def test_loo_estimator_default_is_source(self, backbone, base_config):
        """Default loo_estimator should be 'source'."""
        method = ReferenceMatching(backbone, base_config)
        assert method.loo_estimator == "source"

    def test_loo_estimator_explicit_source(self, backbone, base_config):
        """Explicit loo_estimator='source' should be accepted."""
        config = dict(base_config)
        config["loo_estimator"] = "source"
        method = ReferenceMatching(backbone, config)
        assert method.loo_estimator == "source"

    def test_loo_estimator_token(self, backbone, base_config):
        """loo_estimator='token' should be accepted."""
        config = dict(base_config)
        config["loo_estimator"] = "token"
        method = ReferenceMatching(backbone, config)
        assert method.loo_estimator == "token"

    def test_loo_estimator_invalid_value(self, backbone, base_config):
        """Invalid loo_estimator value should raise ValueError."""
        config = dict(base_config)
        config["loo_estimator"] = "invalid"
        with pytest.raises(ValueError, match="loo_estimator must be 'source' or 'token'"):
            ReferenceMatching(backbone, config)

    def test_loo_estimator_token_in_metadata_only_when_token(self, backbone, base_config):
        """metadata['loo_estimator'] should only be present when loo_estimator=='token'."""
        # Test source mode (default)
        config_source = dict(base_config)
        method_source = ReferenceMatching(backbone, config_source)
        metadata_source = method_source.object_metadata()
        assert "loo_estimator" not in metadata_source

        # Test token mode
        config_token = dict(base_config)
        config_token["loo_estimator"] = "token"
        method_token = ReferenceMatching(backbone, config_token)
        metadata_token = method_token.object_metadata()
        assert "loo_estimator" in metadata_token
        assert metadata_token["loo_estimator"] == "token"

    def test_loo_token_suffix_in_score_definition(self, backbone, base_config):
        """Score definition should end with '_loo_token' only when loo_estimator=='token'."""
        # Test source mode
        config_source = dict(base_config)
        method_source = ReferenceMatching(backbone, config_source)
        metadata_source = method_source.object_metadata()
        assert not metadata_source["score_definition"].endswith("_loo_token")

        # Test token mode
        config_token = dict(base_config)
        config_token["loo_estimator"] = "token"
        method_token = ReferenceMatching(backbone, config_token)
        metadata_token = method_token.object_metadata()
        assert metadata_token["score_definition"].endswith("_loo_token")

    def test_loo_token_suffix_after_all_other_suffixes(self, backbone, base_config):
        """_loo_token suffix should come after noise_guard and all other suffixes."""
        config = dict(base_config)
        config["loo_estimator"] = "token"
        config["noise_guard"] = True
        config["noise_guard_resolutions"] = [8, 12]  # Must be subset of resolutions

        method = ReferenceMatching(backbone, config)
        metadata = method.object_metadata()
        score_def = metadata["score_definition"]

        # Should end with _loo_token, even after _noise_guard_*
        assert score_def.endswith("_loo_token")
        # noise_guard suffix should come before _loo_token
        assert "_noise_guard_" in score_def
        idx_noise = score_def.find("_noise_guard_")
        idx_loo = score_def.find("_loo_token")
        assert idx_noise < idx_loo

    def test_token_mode_with_multishot(self, tmp_path, backbone, base_config):
        """Token mode should work with multiple support images."""
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]

        config = dict(base_config)
        config["loo_estimator"] = "token"

        method = ReferenceMatching(backbone, config)
        from fmad.protocol import MethodContext
        context = MethodContext("can", 0, 8, False, False)
        method.fit(paths, context)

        # Should fit successfully
        assert method.support_count == 3

    def test_token_mode_with_oneshot(self, tmp_path, backbone, base_config):
        """Token mode with 1 support should be numerically identical to source fallback."""
        path = _write_image(tmp_path / "0.png", 0)

        config = dict(base_config)
        config["loo_estimator"] = "token"

        method_token = ReferenceMatching(backbone, config)

        # Also create source mode method for comparison
        config_source = dict(base_config)
        config_source["loo_estimator"] = "source"
        method_source = ReferenceMatching(backbone, config_source)

        from fmad.protocol import MethodContext
        context = MethodContext("can", 0, 8, False, False)

        method_token.fit([path], context)
        method_source.fit([path], context)

        # With 1 shot, token mode uses the same leave-token-out computation as source fallback
        # Scales should be identical
        for res in method_token.scales.keys():
            np.testing.assert_allclose(
                method_token.scales[res].numpy(),
                method_source.scales[res].numpy(),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Scales differ at resolution {res}"
            )

    def test_token_mode_per_record_computation(self, tmp_path, backbone, base_config):
        """Token mode should compute distances per-record with diagonal excluded.

        This test validates that token mode:
        1. Computes each record's distances against only its own tokens
        2. Excludes the diagonal (self-matches)
        3. Produces different scales than source mode when >1 support
        4. Has no near-zero distances (which would indicate self-matches weren't masked)
        """
        import torch
        from torch.nn import functional as F

        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(2)]

        # Fit with token mode
        config_token = dict(base_config)
        config_token["loo_estimator"] = "token"
        method_token = ReferenceMatching(backbone, config_token)

        # Fit with source mode for comparison
        config_source = dict(base_config)
        config_source["loo_estimator"] = "source"
        method_source = ReferenceMatching(backbone, config_source)

        from fmad.protocol import MethodContext
        context = MethodContext("can", 0, 8, False, False)

        method_token.fit(paths, context)
        method_source.fit(paths, context)

        # Verify token mode produced finite, valid scales
        for res in method_token.scales.keys():
            scales = method_token.scales[res]
            assert torch.isfinite(scales).all(), f"Token mode produced non-finite scales at resolution {res}"
            assert (scales > 0).all(), f"Token mode scales are not positive at resolution {res}"

        # Verify 1-shot equivalence: with 1 support, token mode must match source mode exactly
        path_single = _write_image(tmp_path / "single.png", 42)
        config_token_1shot = dict(base_config)
        config_token_1shot["loo_estimator"] = "token"
        method_token_1shot = ReferenceMatching(backbone, config_token_1shot)

        config_source_1shot = dict(base_config)
        config_source_1shot["loo_estimator"] = "source"
        method_source_1shot = ReferenceMatching(backbone, config_source_1shot)

        from fmad.protocol import MethodContext as MC
        method_token_1shot.fit([path_single], MC("can", 0, 8, False, False))
        method_source_1shot.fit([path_single], MC("can", 0, 8, False, False))

        # With 1 shot, token and source must be numerically identical
        for res in method_token_1shot.scales.keys():
            np.testing.assert_allclose(
                method_token_1shot.scales[res].numpy(),
                method_source_1shot.scales[res].numpy(),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"1-shot token and source scales differ at resolution {res}"
            )

    def test_source_mode_multishot_uses_leave_source_out(self, tmp_path, backbone, base_config):
        """Source mode with >1 support should use leave-source-out."""
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]

        config = dict(base_config)
        config["loo_estimator"] = "source"  # explicit

        method = ReferenceMatching(backbone, config)
        from fmad.protocol import MethodContext
        context = MethodContext("can", 0, 8, False, False)

        method.fit(paths, context)
        assert method.support_count == 3

    def test_source_mode_oneshot_uses_leave_token_out(self, tmp_path, backbone, base_config):
        """Source mode with 1 support should fall back to leave-token-out."""
        path = _write_image(tmp_path / "0.png", 0)

        config = dict(base_config)
        config["loo_estimator"] = "source"

        method = ReferenceMatching(backbone, config)
        from fmad.protocol import MethodContext
        context = MethodContext("can", 0, 8, False, False)

        method.fit([path], context)
        assert method.support_count == 1




def test_token_mode_masks_self_for_every_record():
    """Regression: every record must exclude its OWN token, not only the first.

    An earlier implementation compared each record against the full multi-record
    bank and masked the matrix diagonal; the diagonal is the self-match only for
    the first record, so later records matched themselves at distance 0. The tiny
    test backbones emit near-duplicate tokens, which hides this, so the records
    here are crafted distinct unit vectors and the expected distances are
    computed independently.
    """
    from tests.test_reference_matching import TinyPairBackbone
    model = ReferenceMatching(TinyPairBackbone(), {
        "variant": "R1", "resolutions": [8], "bank_tokens_per_source": 4,
        "query_chunk_size": 4, "feature_layers": [11, 23], "loo_estimator": "token"})
    generator = torch.Generator().manual_seed(0)
    records = []
    for source in range(3):
        layers = [torch.nn.functional.normalize(torch.randn(5, 8, generator=generator), dim=-1)
                  for _ in model.layers]
        records.append({"source": source, "layers": layers, "grid": (1, 5),
                        "indices": torch.arange(5)})
    model.records = {8: records}
    model._fit_scales_for_resolution(8)
    captured = model._test_token_mode_distances[8]
    for layer_index in range(len(model.layers)):
        assert len(captured[layer_index]) == 3
        for record in records:
            x = record["layers"][layer_index]
            expected = (1.0 - x @ x.T).fill_diagonal_(10.0).amin(dim=1)
            actual = captured[layer_index][record["source"]]
            torch.testing.assert_close(actual, expected.cpu())
            assert float(actual.min()) > 1e-3, "a record matched itself"
