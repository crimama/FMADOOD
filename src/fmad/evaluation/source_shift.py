"""Source-fixed operating points and paired scene summaries, evaluation only.

No target labels enter calibration, fitting, or prediction. Dense evidence is
released per query; the method receives only paths for normal calibration.
"""
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from fmad.evaluation.shift_diagnostic import acquisition_identity
from fmad.evaluation.streaming import _ordered_float16_codes, _code_to_float16, _label8


TARGET_FPRS = (.001, .01, .05)


def threshold_code(histogram, target_fpr):
    if not 0 < target_fpr < 1 or histogram.sum() <= 0:
        raise ValueError('invalid source calibration histogram or target FPR')
    rank = int(np.ceil((1. - target_fpr) * int(histogram.sum())))
    return int(np.searchsorted(np.cumsum(histogram), rank, side='left'))


def select_calibration(train_paths, support_paths, object_name, count=16):
    support = {Path(p).resolve() for p in support_paths}
    candidates = []
    for value in sorted(train_paths):
        path = Path(value)
        if tuple(path.parts[-4:-1]) != (object_name, 'train', 'good'):
            raise ValueError('calibration requires explicit train/good paths')
        if path.resolve() not in support and acquisition_identity(path)[1] == 'regular':
            candidates.append(str(path))
    selected = candidates[:count]
    if len(selected) != count or len({Path(p).resolve() for p in selected}) != count:
        raise ValueError('insufficient distinct support-disjoint normal calibration images')
    content = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    support_hashes = {content(p) for p in support_paths}
    if support_hashes.intersection(content(p) for p in selected):
        raise ValueError('calibration contains support duplicate content')
    return selected


def operating_stats(values, mask, labels, region_count, thresholds):
    codes = _ordered_float16_codes(values)
    positive = np.bincount(codes[mask], minlength=65536)
    negative = np.bincount(codes[~mask], minlength=65536)
    sizes = np.bincount(labels.reshape(-1))
    weights = 1. / sizes[labels[mask]] if region_count else np.empty(0)
    pro = np.bincount(codes[mask], weights=weights, minlength=65536)
    normal_n, defect_n = int(negative.sum()), int(positive.sum())
    output = {}
    for name, code in thresholds.items():
        fp, tp = int(negative[code + 1:].sum()), int(positive[code + 1:].sum())
        region_sum = float(pro[code + 1:].sum())
        output[name] = dict(false_positive_pixels=fp, true_positive_pixels=tp,
            normal_pixels=normal_n, defect_pixels=defect_n, regions=region_count,
            region_recall_sum=region_sum, FPR=fp / normal_n if normal_n else None,
            TPR=tp / defect_n if defect_n else None,
            PRO=region_sum / region_count if region_count else None)
    return output


def region_moments(values, mask):
    if values.shape != mask.shape or not np.isfinite(values).all():
        raise ValueError('invalid component map')
    result = {}
    for name, selector in (('normal', ~mask), ('defect', mask)):
        samples = values[selector]
        result[name] = dict(count=int(samples.size),
            mean=float(samples.mean(dtype=np.float64)) if samples.size else None,
            std=float(samples.std(dtype=np.float64)) if samples.size else None)
    n, d = result['normal'], result['defect']
    result['defect_minus_normal'] = d['mean'] - n['mean'] if n['count'] and d['count'] else None
    result['gap_over_normal_std'] = (result['defect_minus_normal'] / n['std']
        if result['defect_minus_normal'] is not None and n['std'] > 1e-12 else None)
    return result


class SourceShiftDiagnostic:
    @classmethod
    def calibrate(cls, method, train_paths, support_paths, object_name):
        started = time.perf_counter()
        if not getattr(method, 'mechanism_audit', False):
            raise ValueError('source_shift_diagnostic requires mechanism_audit')
        paths = select_calibration(train_paths, support_paths, object_name)
        histograms = {'raw': [], 'CQ': []}
        for index, path in enumerate(paths):
            prediction = method.predict_batch([path])[0]
            raw = prediction.diagnostics['fusion_audit']['raw_map']
            for stage, values in (('raw', raw), ('CQ', prediction.score_map)):
                histograms[stage].append(np.bincount(_ordered_float16_codes(values).reshape(-1), minlength=65536))
            print(f'    source calibration {index + 1}/{len(paths)}', flush=True)
        info = dict(paths=paths, count=len(paths), support_disjoint=True,
            exposure='train_normal_previously_available_development_not_untouched_holdout',
            content_sha256=hashlib.sha256(b''.join(Path(p).read_bytes() for p in paths)).hexdigest(),
            selection='first16_sorted_train_good_regular_excluding_actual_support_paths_and_duplicate_content',
            quantization='float16_order_histogram_all_native_pixels',
            comparison='quantized_score_strictly_greater_than_source_threshold',
            primary_target_FPR=.01, thresholds={}, seconds=time.perf_counter() - started,
            extra_calibration_queries=len(paths), calibration_used_for_method_fit=False)
        thresholds = {}
        for stage, hist in histograms.items():
            total = np.sum(hist, axis=0, dtype=np.int64)
            thresholds[stage], info['thresholds'][stage] = {}, {}
            for fpr in TARGET_FPRS:
                key = str(fpr)
                code = threshold_code(total, fpr)
                loo = [threshold_code(total - h, fpr) for h in hist]
                thresholds[stage][key] = code
                info['thresholds'][stage][key] = dict(code=code, value=_code_to_float16(code),
                    calibration_FPR=float(total[code + 1:].sum() / total.sum()),
                    calibration_pixels=int(total.sum()),
                    leave_one_image_out_values=[_code_to_float16(x) for x in loo],
                    uncertainty='LOO_sensitivity_not_confidence_interval')
                if fpr == .01:
                    thresholds[stage]['0.01_loo_min'] = min(loo)
                    thresholds[stage]['0.01_loo_max'] = max(loo)
        return cls(thresholds, info)

    def __init__(self, thresholds, calibration):
        self.thresholds, self.calibration = thresholds, calibration
        self.records = []
        self.max_reconstruction_error = 0.

    def update(self, path, kind, score, mask, auxiliary):
        scene, condition = acquisition_identity(path)
        raw = auxiliary['raw_map']
        grids = auxiliary.get('component_grids')
        if not grids or not all(set(g) == set(grids[0]) for g in grids):
            raise ValueError('missing or inconsistent component grids')
        labels, region_count = _label8(mask)
        record = dict(scene=scene, condition=condition, kind=kind,
            operating={}, components={}, normal_pixels=int((~mask).sum()), defect_pixels=int(mask.sum()))
        for stage, values in (('raw', raw), ('CQ', score)):
            record['operating'][stage] = operating_stats(values, mask, labels, region_count, self.thresholds[stage])
            record['components'][stage] = region_moments(values, mask)
        reconstructed = np.zeros(mask.shape, np.float32)
        for name in grids[0]:
            # Match prediction's interpolation backend and device. OpenCV (and
            # CPU vs CUDA) coordinate rounding differs at large native sizes.
            native = torch.zeros(mask.shape, device=auxiliary.get('component_device', 'cpu'))
            for grid in grids:
                component = torch.as_tensor(grid[name], device=native.device)
                native += F.interpolate(component[None, None], size=mask.shape,
                    mode='bilinear', align_corners=False)[0, 0]
            native = (native / len(grids)).cpu().numpy()
            record['components'][name] = region_moments(native, mask)
            if name in ('deep', 'shallow_used'):
                reconstructed += native
        error = float(np.max(np.abs(reconstructed - raw)))
        self.max_reconstruction_error = max(self.max_reconstruction_error, error)
        if not np.allclose(reconstructed, raw, rtol=3e-5, atol=3e-6):
            raise ValueError(f'component projection does not reproduce raw score: {path}, max_abs={error}')
        self.records.append(record)

    def compute(self):
        indices = {(r['kind'], r['scene'], r['condition']): r for r in self.records}
        if len(indices) != len(self.records):
            raise ValueError('duplicate source-shift query identity')
        regular = {(k, s) for k, s, c in indices if c == 'regular'}
        shifted = {(k, s) for k, s, c in indices if c == 'shift_1'}
        if not regular or regular != shifted:
            raise ValueError('regular/shift_1 scene identities are not paired')
        summaries = {}
        for condition in sorted({r['condition'] for r in self.records}):
            summaries[condition] = {}
            for subset in ('all', 'good_only'):
                selected = [r for r in self.records if r['condition'] == condition and
                            (subset == 'all' or r['kind'] == 'good')]
                summaries[condition][subset] = {}
                for stage, thresholds in self.thresholds.items():
                    summary = summaries[condition][subset][stage] = {}
                    for threshold in thresholds:
                        points = [r['operating'][stage][threshold] for r in selected]
                        fields = ('false_positive_pixels', 'true_positive_pixels', 'normal_pixels',
                                  'defect_pixels', 'regions', 'region_recall_sum')
                        row = {field: sum(p[field] for p in points) for field in fields}
                        row.update(queries=len(selected),
                            FPR=row['false_positive_pixels'] / row['normal_pixels'] if row['normal_pixels'] else None,
                            TPR=row['true_positive_pixels'] / row['defect_pixels'] if row['defect_pixels'] else None,
                            PRO=row['region_recall_sum'] / row['regions'] if row['regions'] else None)
                        summary[threshold] = row
        paired = []
        for kind, scene in sorted(regular):
            a, b = indices[kind, scene, 'regular'], indices[kind, scene, 'shift_1']
            delta = {name: {tissue: (b['components'][name][tissue]['mean'] -
                        a['components'][name][tissue]['mean'])
                       if a['components'][name][tissue]['count'] and b['components'][name][tissue]['count'] else None
                       for tissue in ('normal', 'defect')} for name in a['components']}
            paired.append(dict(kind=kind, scene=scene, component_mean_change=delta,
                operating_change={stage: {key: {metric: b['operating'][stage][key][metric] -
                    a['operating'][stage][key][metric] if a['operating'][stage][key][metric] is not None and
                    b['operating'][stage][key][metric] is not None else None for metric in ('FPR', 'TPR', 'PRO')}
                    for key in self.thresholds[stage]} for stage in self.thresholds}))
        return dict(calibration=self.calibration, condition_operating_points=summaries,
            records=self.records, paired_regular_shift1=paired,
            component_projection='bag_mean_then_bilinear_native_then_resolution_mean',
            component_reconstruction_max_abs_error=self.max_reconstruction_error,
            pairing='scene_identity_region_summaries_not_registered_pixelwise_differences',
            labels_used_only_after_prediction=True, method_or_threshold_target_adaptation=False,
            maps_or_weights_saved=False)
