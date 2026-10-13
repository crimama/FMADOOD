"""Matched normalization x shallow-residual ablation of RSG3.

Auxiliary maps are transient prediction outputs, never fitting inputs.
"""
import hashlib
import time

import numpy as np
import torch

from fmad.methods.base import Prediction
from fmad.methods.reference_matching import ReferenceMatching
from fmad.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register('fusion_factorial')
class FusionFactorial(ReferenceMatching):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.shallow_mode = config.get('shallow_mode', 'gated')
        # mean3 is the fixed-rule ablation (equal 1/3 layer mean); it keeps the same
        # residual_spike config so tau and gate audits stay comparable across arms.
        allowed_norm = {'mad', 'tailscale', 'none'} if self.shallow_mode == 'mean3' else {'mad', 'tailscale'}
        if (self.shallow_mode not in {'all', 'hard', 'gated', 'off', 'mean3'} or
                self.variant != 'R1' or self.layer_fusion != 'residual_spike' or
                self.layer_fusion_tau_mode != 'support_quantile' or
                self.score_normalization not in allowed_norm or
                self.local_reliability != 'off' or self.scale_fusion != 'mean' or
                len(self.layers) != 3 or self.layer_fusion_shallow_idx != 0):
            raise ValueError('fusion_factorial requires matched three-layer R1 residual fusion')
        self._audit_active = False
        self._gate_grids = []
        self._bag_gates = []
        self._raw_map = None
        self.mechanism_audit = config.get('mechanism_audit', False)
        if type(self.mechanism_audit) is not bool:
            raise ValueError('mechanism_audit must be boolean')
        self._component_bags = []
        self._component_grids = []

    def _observe_layer_scores(self, distances, scores, *, tau):
        if not (self.mechanism_audit and self._audit_active):
            return
        if tau is None:
            raise ValueError('mechanism audit requires support threshold')
        deep = torch.stack(scores[1:]).mean(0)
        residual = self.layer_fusion_gamma * torch.relu(scores[0] - tau)
        full = self.layer_fusion_gamma * scores[0]
        hard = full * (scores[0] > tau)
        # mean3 fuses (s0 + s1 + s2) / 3, so the shallow share of the fused score is s0 / 3.
        used = {'all': full, 'hard': hard, 'gated': residual, 'mean3': scores[0] / 3,
                'off': torch.zeros_like(full)}[self.shallow_mode]
        components = {f'raw_l{layer}': value for layer, value in zip(self.layers, distances)}
        components.update({f'normalized_l{layer}': value for layer, value in zip(self.layers, scores)})
        components.update(deep=deep, shallow_full=full, shallow_residual=residual, shallow_used=used)
        self._component_bags.append({k: v.detach() for k, v in components.items()})

    def _fuse_layers(self, scores, tau=None):
        if tau is None:
            raise ValueError('support-calibrated shallow threshold is required')
        if self._audit_active:
            self._bag_gates.append((scores[0] > tau).float())
        if self.shallow_mode == 'mean3':
            return torch.stack(scores).mean(0)
        if self.shallow_mode == 'gated':
            return super()._fuse_layers(scores, tau=tau)
        deep = torch.stack(scores[1:]).mean(0)
        if self.shallow_mode == 'all':
            return deep + self.layer_fusion_gamma * scores[0]
        if self.shallow_mode == 'hard':
            return deep + self.layer_fusion_gamma * scores[0] * (scores[0] > tau)
        return deep

    def _score_tokens(self, layers, grid, bank, *, normalize=True):
        self._bag_gates = []
        self._component_bags = []
        score = super()._score_tokens(layers, grid, bank, normalize=normalize)
        if self._audit_active:
            self._gate_grids.append(torch.stack(self._bag_gates).mean(0).reshape(*grid).cpu().numpy())
            if self.mechanism_audit:
                self._component_grids.append({k: torch.stack([b[k] for b in self._component_bags])
                    .mean(0).reshape(*grid).cpu().numpy() for k in self._component_bags[0]})
        self._bag_gates = []
        self._component_bags = []
        return score

    def _postprocess_prediction(self, result, image):
        if self._audit_active:
            self._raw_map = result.copy()
        return super()._postprocess_prediction(result, image)

    def predict_batch(self, test_image_paths):
        predictions = []
        for path in test_image_paths:
            self._audit_active = True
            self._gate_grids = []
            self._raw_map = None
            self._component_grids = []
            started = time.perf_counter()
            try:
                score, image_score = self.predict(path)
                # One gate grid per resolution actually scored (a fired lowres guard scores a subset).
                if self._raw_map is None or len(self._gate_grids) != len(self.last_resolutions):
                    raise RuntimeError('incomplete fusion audit')
                auxiliary = {'raw_map': self._raw_map, 'gate_grids': self._gate_grids}
                if self.mechanism_audit:
                    auxiliary['component_grids'] = self._component_grids
                    auxiliary['component_device'] = str(self.device)
                diagnostics = {'fusion_audit': auxiliary}
                if self.image_score_candidates:
                    diagnostics['image_score_candidates'] = dict(self._last_image_candidates)
                predictions.append(Prediction(score, image_score, time.perf_counter() - started,
                    diagnostics))
            finally:
                self._audit_active = False
                self._gate_grids = []
                self._bag_gates = []
                self._raw_map = None
                self._component_bags = []
                self._component_grids = []
        return predictions

    def object_metadata(self):
        metadata = super().object_metadata()
        digest = hashlib.sha256()
        for resolution in self.resolutions:
            bags = self.bag_banks[resolution] if self.bank_bags > 1 else [self.banks[resolution]]
            for bank in bags:
                for layer in bank['layers']:
                    value = layer.detach().cpu().contiguous().numpy()
                    digest.update(str((resolution, value.shape, str(value.dtype))).encode())
                    digest.update(value.tobytes())
        # For mean3, use a different score definition that reflects equal weighting
        if self.shallow_mode == 'mean3':
            score_def = f'bag_mean(equal_mean3)_{self.score_normalization}_CQ'
        else:
            score_def = f'bag_mean(deep_mean+{self.shallow_mode}_shallow)_{self.score_normalization}_CQ'
        metadata.update(shallow_mode=self.shallow_mode, bank_feature_sha256=digest.hexdigest(),
            score_definition=score_def,
            query_labels_used=False, diagnostic_extra_backbone_calls=0)
        metadata['mechanism_audit'] = self.mechanism_audit
        return metadata
