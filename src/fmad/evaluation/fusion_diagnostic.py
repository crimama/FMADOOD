"""Evaluation-only raw/CQ and acquisition-stratified residual-gate evidence.

Also records coverage-void statistics. The shallow gate sits at the support q99,
so support-like normal pixels are expected to activate it at rate 0.01, and
deviation from that null is the void measurement. The deep exceedance rate has no
such null: the deep score averages two layer scores, which suppresses joint
exceedance, so it is read only relative to the matched in-distribution cell.
Masks stratify normal and defect pixels only after prediction.
"""
import cv2
import numpy as np

from fmad.evaluation.shift_diagnostic import acquisition_identity
from fmad.evaluation.streaming import StreamingSegmentationAccumulator


GATE_NULL_RATE = 0.01      # shallow residual gate is the support q99
DEEP_NULL_RATE = None      # no a-priori null: deep averages two layer scores
VOID_FIELDS = ('gate', 'residual', 'deep_q95', 'z5')


def _native(grids, shape):
    """Equal-weight resolution mean of token grids, nearest-expanded to native."""
    out = np.zeros(shape, np.float32)
    for grid in grids:
        if not np.isfinite(grid).all():
            raise ValueError('invalid component grid')
        out += cv2.resize(np.asarray(grid, np.float32), (shape[1], shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return out / len(grids)


def _void_fields(components, mask):
    """Per-query void statistics, stratified into normal and defect pixels."""
    shape = mask.shape
    deep = _native([c['deep'] for c in components], shape)
    residual = _native([c['shallow_residual'] for c in components], shape)
    z5 = _native([c['normalized_l5'] for c in components], shape)
    fields = {}
    for kind, sel in (('normal', ~mask), ('defect', mask)):
        if not sel.any():
            fields.update({f'{kind}_{name}': None for name in VOID_FIELDS[1:]})
            continue
        fields[f'{kind}_residual'] = float(residual[sel].mean())
        fields[f'{kind}_deep_q95'] = float((deep[sel] > 1.0).mean())
        fields[f'{kind}_z5'] = float(z5[sel].mean())
    return fields


def summarize(acc):
    coverage = dict(queries=acc.prediction_map_count, regions=acc.region_count,
                    normal_pixels=int(acc.negative.sum()), defect_pixels=int(acc.positive.sum()))
    if not acc.region_count or not acc.positive.sum() or not acc.negative.sum():
        return dict(coverage=coverage, metrics=None, reason='insufficient_positive_or_negative_mass')
    result = acc.compute()
    active = np.flatnonzero(acc.positive + acc.negative)[::-1]
    fpr = np.r_[0., np.cumsum(acc.negative[active]) / acc.negative.sum()]
    pro = np.r_[0., np.cumsum(acc.pro_weight[active]) / acc.region_count]
    # Interpolation follows the evaluator's tied-score ROC convention.
    return dict(coverage=coverage, metrics=dict(
        AUPRO_0_05=result.seg_aupro_0_05, pAUROC_0_05=result.seg_auroc_0_05,
        PRO_at_FPR={str(q): float(np.interp(q, fpr, pro)) for q in (.001, .005, .01, .05)}))


class FusionDiagnostic:
    def __init__(self):
        self.accumulators = {}
        self.samples = []

    def update(self, path, kind, score, mask, auxiliary):
        scene, condition = acquisition_identity(path)
        raw = np.asarray(auxiliary['raw_map'], dtype=np.float32)
        grids = auxiliary['gate_grids']
        if raw.shape != mask.shape or not np.isfinite(raw).all() or not grids:
            raise ValueError('invalid raw/gate diagnostic')
        for stage, values in (('raw', raw), ('CQ', score)):
            acc = self.accumulators.setdefault((condition, stage), StreamingSegmentationAccumulator())
            acc.update([values], [mask])
        # Mean binary gate across bags, then equal-weight resolution mean.
        # Nearest-cell expansion defines an activation statistic, not a score map.
        gate = np.zeros(mask.shape, np.float32)
        for grid in grids:
            if not np.isfinite(grid).all() or np.any(grid < 0) or np.any(grid > 1):
                raise ValueError('invalid gate activation')
            gate += cv2.resize(grid, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        gate /= len(grids)
        sample = dict(scene=scene, condition=condition, kind=kind,
            normal_pixels=int((~mask).sum()), defect_pixels=int(mask.sum()),
            normal_gate=float(gate[~mask].mean()) if (~mask).any() else None,
            defect_gate=float(gate[mask].mean()) if mask.any() else None)
        components = auxiliary.get('component_grids')
        if components:
            if len(components) != len(grids):
                raise ValueError('component grids must match resolution count')
            sample.update(_void_fields(components, mask))
        self.samples.append(sample)

    def compute(self):
        metrics = {}
        for (condition, stage), acc in self.accumulators.items():
            metrics.setdefault(condition, {})[stage] = summarize(acc)
        for stage in ('raw', 'CQ'):
            total = StreamingSegmentationAccumulator()
            for (condition, name), acc in self.accumulators.items():
                if name != stage:
                    continue
                for field in ('positive', 'negative', 'pro_weight', 'region_count', 'prediction_map_count',
                              'shape_match_count', 'nonconstant_prediction_map_count', 'non_finite_prediction_count'):
                    setattr(total, field, getattr(total, field) + getattr(acc, field))
            metrics.setdefault('all', {})[stage] = summarize(total)
        activation, void = {}, {}
        for condition in metrics:
            samples = [s for s in self.samples if condition == 'all' or s['condition'] == condition]
            row, stats = {}, {}
            for kind in ('normal', 'defect'):
                valid = [s for s in samples if s[f'{kind}_pixels']]
                mass = sum(s[f'{kind}_pixels'] for s in valid)
                row[f'{kind}_pixel_weighted'] = (sum(s[f'{kind}_gate'] * s[f'{kind}_pixels']
                    for s in valid) / mass) if mass else None
                row[f'{kind}_image_mean'] = float(np.mean([s[f'{kind}_gate'] for s in valid])) if valid else None
                for name in VOID_FIELDS:
                    present = [s for s in valid if s.get(f'{kind}_{name}') is not None]
                    pixels = sum(s[f'{kind}_pixels'] for s in present)
                    stats[f'{kind}_{name}'] = (sum(s[f'{kind}_{name}'] * s[f'{kind}_pixels']
                        for s in present) / pixels) if pixels else None
            normal_gate, defect_gate = stats.get('normal_gate'), stats.get('defect_gate')
            stats['void_inflation'] = normal_gate / GATE_NULL_RATE if normal_gate is not None else None
            stats['gate_selectivity'] = (defect_gate / normal_gate
                if normal_gate and defect_gate is not None else None)
            activation[condition], void[condition] = row, stats
        return dict(metrics=metrics, gate_activation=activation, void=void, samples=self.samples,
            void_null=dict(gate_rate=GATE_NULL_RATE, deep_q95_rate=DEEP_NULL_RATE,
                definition='void inflation = observed normal gate rate / calibrated null rate; '
                           'normal_deep_q95 has no null and is read relative to the matched cell'),
            gate_definition='hypothetical_support_q99_gate_all_arms; bag_mean_then_nearest_native_then_resolution_mean',
            threshold_interpretation='condition_specific_ROC_not_fixed_source_threshold_transfer',
            labels_used_only_after_prediction=True, dense_maps_saved=False)
