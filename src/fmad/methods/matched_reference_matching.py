"""Normal-support cross-fitting with the same reference bank at fit and query.

Development candidate, not a conformal-coverage or novelty claim. Each fold holds
out one support source, calibrates against the remaining source-token identities,
and scores queries against that exact bank. Predictions average all folds/bags.
"""
from __future__ import annotations

import copy

import torch

from fmad.methods.statistical_reference_matching import (
    StatisticalReferenceMatching, fit_normalizer,
)
from fmad.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register('matched_reference_matching')
class MatchedReferenceMatching(StatisticalReferenceMatching):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.reference_calibration = config.get('reference_calibration', 'matched')
        if self.reference_calibration not in {'matched', 'pooled_control'}:
            raise ValueError('reference_calibration must be matched or pooled_control')
        if (self.statistical_normalization != 'q95' or self.knn_k != 1
                or not self.fast_reuse_calibration_bank
                or self.local_reliability != 'off'
                or self.calibration_audit
                or self.noise_guard_action == 'lowres_pool'
                or self.image_residual_tau_q is not None):
            raise ValueError('matched reference requires Q95 cosine1NN, source-LOO reuse, no calibration-map/image/pooled audit')
        self._original_bag_count = self.bank_bags
        self._fold_fit = {}

    def _fit_scales_for_resolution(self, resolution):
        result = super()._fit_scales_for_resolution(resolution)
        groups = getattr(self, '_last_source_loo_samples', None)
        records = self.records[resolution]
        if groups is None or any(len(g) != len(records) for g in groups):
            raise RuntimeError('missing source-LOO groups')
        self._fold_fit.setdefault(resolution, []).append(dict(
            sources=[int(r['source']) for r in records],
            groups=groups,
        ))
        return result

    def fit(self, support_paths, context):
        if len(support_paths) < 2:
            raise ValueError('matched reference requires at least two support images')
        self.bank_bags = self._original_bag_count
        self._fold_fit = {}
        super().fit(support_paths, context)
        original = super().object_metadata()
        self.full_bank_feature_sha256 = original.get('ablation_bank_feature_sha256')
        self.fold_provenance = {}
        for resolution, original_banks in list(self.bag_banks.items()):
            banks, tails, scales, provenance = [], [], [], []
            if len(self._fold_fit[resolution]) != self._original_bag_count:
                raise RuntimeError('incomplete original bag fits')
            for bi, bank in enumerate(original_banks):
                source_fit = self._fold_fit[resolution][bi]
                owners = torch.as_tensor(bank['owners'], device=self.device)
                for si, source in enumerate(source_fit['sources']):
                    keep = owners != source
                    if not bool(keep.any()) or bool(keep.all()):
                        raise RuntimeError('invalid source-excluded bank')
                    fold_bank = dict(
                        layers=[x[keep].contiguous() for x in bank['layers']],
                        owners=[v for v in bank['owners'] if v != source],
                        token_keys=[k for k in bank['token_keys'] if k[0] != source],
                    )
                    fold_tails = []
                    for li in range(len(self.layers)):
                        sample = source_fit['groups'][li][si]
                        if self.reference_calibration == 'matched':
                            tail = self._tail_params(sample)
                            self._set_tail_center(tail)
                            tail['statistical'] = fit_normalizer(sample, 'q95', tail)
                        else:
                            tail = copy.deepcopy(self.bag_tails[resolution][bi][li])
                        tail['statistical'].update(
                            reference_calibration=self.reference_calibration,
                            calibration_source_ids=([source] if self.reference_calibration == 'matched'
                                                    else source_fit['sources']),
                            query_excluded_source=source,
                        )
                        fold_tails.append(tail)
                    banks.append(fold_bank)
                    tails.append(fold_tails)
                    scales.append(torch.tensor([t['statistical']['scale'] for t in fold_tails]))
                    provenance.append(dict(original_bag=bi, calibration_source=source,
                        reference_sources=sorted(set(fold_bank['owners'])),
                        reference_token_count=len(fold_bank['owners']),
                        calibration_tokens_per_layer=[len(g[si]) for g in source_fit['groups']]))
            self.bag_banks[resolution] = banks
            self.banks[resolution] = banks[0]
            self.bag_tails[resolution] = tails
            self.tails[resolution] = tails[0]
            self.bag_scales[resolution] = scales
            self.scales[resolution] = scales[0]
            self.fold_provenance[str(resolution)] = provenance
        self.bank_bags = self._original_bag_count * len(support_paths)
        self._shared_union_cache = {}
        self._fast_full_bank_cache = None
        self._fold_fit = {}

    def object_metadata(self):
        m = super().object_metadata()
        m.update(
            normalization='source_specific_Q95_matched_reference' if self.reference_calibration == 'matched'
                          else 'pooled_Q95_excluded_reference_control',
            score_definition='mean_source_excluded_fold_scores_with_'+self.reference_calibration,
            reference_calibration=self.reference_calibration,
            original_random_bag_count=self._original_bag_count,
            source_fold_count=self.support_count,
            effective_fold_bag_count=self.bank_bags,
            full_bank_feature_sha256=self.full_bank_feature_sha256,
            reference_folds=self.fold_provenance,
            query_used_for_fit=False,
            reference_change='each query fold uses S-1 sources; all S folds averaged',
            calibration_label_use='normal support only; no query labels or features in fit',
            conformal_coverage_claimed=False,
        )
        return m
