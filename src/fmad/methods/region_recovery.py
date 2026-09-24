"""Bounded, feature/colour-guided expansion of existing high anomaly scores.

Only current-query intermediate features/scores are used. This is a heuristic
region recovery rule, not a semantic mask or defect-preservation guarantee.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from fmad.methods.reference_matching import ReferenceMatching
from fmad.registry import METHOD_REGISTRY


def bounded_reach(seeds, eligible, horizontal, vertical, steps):
    """Synchronous four-neighbour growth; each iteration traverses one edge."""
    active = seeds.copy()
    for _ in range(steps):
        neighbours = np.zeros_like(active)
        neighbours[:, 1:] |= active[:, :-1] & horizontal
        neighbours[:, :-1] |= active[:, 1:] & horizontal
        neighbours[1:] |= active[:-1] & vertical
        neighbours[:-1] |= active[1:] & vertical
        grown = active | (neighbours & eligible)
        if np.array_equal(grown, active):
            break
        active = grown
    return active


@torch.no_grad()
def feature_links(guide, grid, shape, minimum_cosine):
    """Nearest-cell guide on the work grid; barriers occur at patch boundaries.

Pool only if an extremely elongated image makes the work grid smaller than
the feature grid. This never allocates pixel-by-channel feature tensors.
"""
    height, width = shape
    gh, gw = min(grid[0], height), min(grid[1], width)
    values = guide.T.reshape(1, guide.shape[1], *grid)
    if (gh, gw) != tuple(grid):
        values = F.adaptive_avg_pool2d(values, (gh, gw))
    values = F.normalize(values[0].permute(1, 2, 0), dim=-1)
    row = np.minimum(((np.arange(height) + .5) * gh / height).astype(int), gh - 1)
    col = np.minimum(((np.arange(width) + .5) * gw / width).astype(int), gw - 1)
    horizontal = np.ones((height, max(0, width - 1)), dtype=bool)
    vertical = np.ones((max(0, height - 1), width), dtype=bool)
    if gw > 1:
        cosine = (values[:, :-1] * values[:, 1:]).sum(-1).cpu().numpy()
        indices = np.where(col[:-1] != col[1:])[0]
        horizontal[:, indices] = cosine[row[:, None], col[indices][None, :]] >= minimum_cosine
    if gh > 1:
        cosine = (values[:-1] * values[1:]).sum(-1).cpu().numpy()
        indices = np.where(row[:-1] != row[1:])[0]
        vertical[indices] = cosine[row[indices][:, None], col[None, :]] >= minimum_cosine
    return horizontal, vertical


def recover_regions(raw, baseline, image, scale_maps, guide, grid, *,
                    max_edge=1024, steps=4, low_percentile=95., high_percentile=99.,
                    minimum_cosine=.9, maximum_colour_distance=.1, min_votes=2):
    """Lift reachable, moderately high native pixels only to the seed threshold.

CQ's baseline is preserved. Work-grid growth approximates native boundaries;
RGB and feature edges are barriers, not learned/calibrated probabilities.
"""
    if raw.shape != baseline.shape or image.shape[:2] != raw.shape:
        raise ValueError("region recovery inputs must be spatially aligned")
    if not np.isfinite(raw).all() or not np.isfinite(baseline).all():
        raise ValueError("nonfinite region recovery scores")
    low, high = np.percentile(raw, [low_percentile, high_percentile])
    stats = dict(queries=1, native_pixels=raw.size, work_pixels=0,
                 seed_work_pixels=0, eligible_work_pixels=0, reached_work_pixels=0,
                 raised_native_pixels=0, constant_fallback_queries=0)
    if high <= low or steps == 0:
        stats['constant_fallback_queries'] = int(high <= low)
        return baseline, stats
    ratio = min(1., max_edge / max(raw.shape))
    shape = tuple(max(1, round(size * ratio)) for size in raw.shape)
    height, width = shape
    pooled = F.adaptive_max_pool2d(torch.from_numpy(np.ascontiguousarray(raw))[None, None], shape)[0, 0].numpy()
    votes = np.zeros(shape, dtype=np.int16)
    for score in scale_maps:
        resized = cv2.resize(score, (width, height), interpolation=cv2.INTER_LINEAR)
        votes += resized >= np.percentile(score, low_percentile)
    seeds = pooled >= high
    eligible = (pooled >= low) & (votes >= min_votes)
    horizontal, vertical = feature_links(guide, grid, shape, minimum_cosine)
    rgb = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
    horizontal &= np.linalg.norm(rgb[:, 1:] - rgb[:, :-1], axis=-1) <= maximum_colour_distance
    vertical &= np.linalg.norm(rgb[1:] - rgb[:-1], axis=-1) <= maximum_colour_distance
    reached = bounded_reach(seeds, eligible, horizontal, vertical, steps)
    # Recheck eligibility at native resolution: coarse max pooling must never
    # elevate low-score native pixels merely because they share a work cell.
    mask = cv2.resize((reached & eligible).astype(np.uint8),
                      (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    mask &= (raw >= low) & (raw < high) & (baseline < high)
    result = baseline.copy()
    result[mask] = high
    stats.update(work_pixels=height * width, seed_work_pixels=int(seeds.sum()),
                 eligible_work_pixels=int(eligible.sum()), reached_work_pixels=int(reached.sum()),
                 raised_native_pixels=int(mask.sum()))
    return result, stats


@METHOD_REGISTRY.register("region_recovery")
class RegionRecovery(ReferenceMatching):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.recovery_enabled = config.get('recovery_enabled', False)
        if type(self.recovery_enabled) is not bool:
            raise ValueError('recovery_enabled must be boolean')
        if (self.variant != 'R1' or self.prediction_postprocess != 'component_quantile'
                or self.component_percentile != 99. or self.component_q != 1.
                or self.component_dilate_size != 0 or 5 not in self.layers
                or len(self.resolutions) < 2 or self.scale_fusion != 'mean'):
            raise ValueError('region_recovery requires R1/CQ99-max, layer5, multiscale mean')
        self._collect_recovery = False
        self._scale_maps = []
        self._guide = None
        self.recovery_stats = {}

    def fit(self, support_paths, context):
        super().fit(support_paths, context)
        self.recovery_stats = {}
        self.log_training(event='region_recovery_ready', recovery_enabled=self.recovery_enabled,
                          trainable_parameters=0, extra_backbone_calls=0)

    def predict(self, test_image_path):
        self._collect_recovery = self.recovery_enabled
        self._scale_maps = []
        self._guide = None
        try:
            return super().predict(test_image_path)
        finally:
            self._collect_recovery = False
            self._scale_maps = []
            self._guide = None

    def _score_tokens(self, layers, grid, bank, *, normalize=True):
        scores = super()._score_tokens(layers, grid, bank, normalize=normalize)
        if self._collect_recovery:
            self._scale_maps.append(scores.detach().reshape(*grid).cpu().numpy())
            if self._guide is None or grid[0] * grid[1] > self._guide[1][0] * self._guide[1][1]:
                self._guide = (layers[self.layers.index(5)].detach(), grid)
        return scores

    def _postprocess_prediction(self, result, image):
        baseline = super()._postprocess_prediction(result, image)
        if not self.recovery_enabled:
            return baseline
        if len(self._scale_maps) != len(self.resolutions) or self._guide is None:
            raise RuntimeError('missing single-query recovery evidence')
        output, stats = recover_regions(result, baseline, image, self._scale_maps, *self._guide)
        for key, value in stats.items():
            self.recovery_stats[key] = self.recovery_stats.get(key, 0) + value
        return output

    def object_metadata(self):
        metadata = super().object_metadata()
        metadata.update(recovery_enabled=self.recovery_enabled,
            recovery_rule='CQ_then_bounded_RGB_feature_multiscale_seed_expansion',
            recovery_parameters=dict(max_work_edge=1024, steps=4, low_percentile=95.,
                high_percentile=99., minimum_feature_cosine=.9, maximum_RGB_L2=.1, min_scale_votes=2),
            recovery_guide='layer5_largest_token_grid_nearest_cell_edges',
            recovery_scope='single_unlabelled_query_no_extra_support_or_backbone_calls',
            recovery_lift='native_scores_in_q95_q99_lifted_only_to_q99',
            recovery_diagnostics_including_warmup=dict(self.recovery_stats),
            recovery_is_semantic_mask=False, query_labels_used=False)
        if self.recovery_enabled:
            metadata['score_definition'] += '_bounded_guided_region_recovery'
        return metadata
