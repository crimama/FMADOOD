"""Post-prediction distribution samples; labels never enter method scoring."""
import hashlib

import cv2
import numpy as np


class CalibrationDiagnostic:
    """Bounded, paired native-coordinate samples of every observation and fusion.

    At most 256 pixels per image/stratum, then a 2048-point priority reservoir
    per class/stratum. This is an image-capped diagnostic sample, not the full
    pixel distribution. Primary metrics still use every native-resolution pixel.
    """
    def __init__(self):
        self.columns = None
        self.pools = {}
        self.queries = []
        self.native_pools = {}
        self.native_counts = {}

    def update(self, path, score, mask, auxiliary):
        grids = auxiliary['grids']
        resolutions, layers = auxiliary['resolutions'], auxiliary['layers']
        columns = [f'{stage}/r{r}/l{layer}' for r in resolutions
                   for stage in ('raw', 'normalized') for layer in layers] + ['fused']
        if self.columns is not None and columns != self.columns:
            raise ValueError('resolution routing changed during calibration diagnostic')
        self.columns = columns
        mask = np.asarray(mask, bool)
        if score.shape != mask.shape or len(grids) != len(resolutions):
            raise ValueError('calibration diagnostic shape mismatch')
        # Filename-based identity makes samples portable across dataset root paths.
        from pathlib import Path
        identity = '/'.join(Path(path).parts[-2:])
        if not mask.any() and 'native_bags' in auxiliary:
            self._update_native(identity, auxiliary)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little'))
        row = {'query': identity, 'strata': {}}
        for kind, selection in (('normal', ~mask), ('defect', mask)):
            eligible = np.flatnonzero(selection)
            if not len(eligible):
                continue
            indices = rng.choice(eligible, min(256, len(eligible)), replace=False)
            y, x = np.unravel_index(indices, mask.shape)
            values = []
            for grid in grids:
                if grid.shape[:2] != (2, len(layers)):
                    raise ValueError('invalid calibration grid')
                h, w = grid.shape[-2:]
                mx = ((x + .5) * w / mask.shape[1] - .5).astype(np.float32)[:, None]
                my = ((y + .5) * h / mask.shape[0] - .5).astype(np.float32)[:, None]
                for plane in grid.reshape(-1, h, w):
                    values.append(cv2.remap(plane, mx, my, cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REPLICATE).ravel())
            values.append(score.ravel()[indices])
            matrix = np.stack(values, axis=1).astype(np.float32)
            if not np.isfinite(matrix).all():
                raise ValueError('nonfinite calibration diagnostic')
            row['strata'][kind] = {'pixels': int(len(eligible)), 'sampled': len(indices),
                                   'sample_mean': matrix.mean(0).tolist()}
            priorities = rng.random(len(matrix))
            if kind in self.pools:
                old_keys, old_values = self.pools[kind]
                priorities = np.concatenate([old_keys, priorities])
                matrix = np.concatenate([old_values, matrix])
            keep = np.argsort(priorities)[:2048]
            self.pools[kind] = priorities[keep], matrix[keep]
        self.queries.append(row)

    def _update_native(self, identity, auxiliary):
        # Whole mask-empty images only: avoid assigning labels to mixed tokens.
        # These are native feature grids, before interpolation and bag mean.
        seed = int.from_bytes(hashlib.sha256(('native/' + identity).encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        for resolution, bags, thresholds in zip(auxiliary['resolutions'],
                auxiliary['native_bags'], auxiliary['raw_tail_thresholds']):
            if not np.isfinite(bags).all():
                raise ValueError('nonfinite native bag diagnostic')
            tokens = bags.shape[-2] * bags.shape[-1]
            indices = rng.choice(tokens, min(tokens, 256), replace=False)
            priorities = rng.random(len(indices))
            for bag_index, bag in enumerate(bags):
                for layer_index, layer in enumerate(auxiliary['layers']):
                    key = f'r{resolution}/b{bag_index}/l{layer}'
                    raw, normalized = bag[:, layer_index].reshape(2, -1)
                    counts = self.native_counts.setdefault(key, dict(tokens=0, images=0,
                        raw_above_support_q95=0, normalized_above_one=0, normalized_negative=0))
                    counts['tokens'] += tokens
                    counts['images'] += 1
                    counts['raw_above_support_q95'] += int((raw > thresholds[bag_index][layer_index]).sum())
                    counts['normalized_above_one'] += int((normalized > 1).sum())
                    counts['normalized_negative'] += int((normalized < 0).sum())
                    matrix = np.stack([raw[indices], normalized[indices]], axis=1)
                    keys = priorities
                    if key in self.native_pools:
                        old_keys, old_values = self.native_pools[key]
                        keys = np.concatenate([old_keys, keys])
                        matrix = np.concatenate([old_values, matrix])
                    keep = np.argsort(keys)[:1024]
                    self.native_pools[key] = keys[keep], matrix[keep]

    def compute(self):
        return {'columns': self.columns, 'queries': self.queries,
                'native_bags': {
                    'scope': 'mask-empty query images; native tokens before interpolation and bag mean',
                    'sampling': '256 tokens/image then priority reservoir 1024/observation/bag; image-capped',
                    'columns': ['raw', 'normalized'],
                    'counts': self.native_counts,
                    'samples': {k: v.tolist() for k, (_, v) in self.native_pools.items()},
                    'quantiles': {k: np.quantile(v, [.5, .95], axis=0).tolist()
                                  for k, (_, v) in self.native_pools.items()},
                    'threshold_note': 'normalized > 1 is a reference-tail check only for tailscale; raw uses support LOO q95'},
                'samples': {k: v.tolist() for k, (_, v) in self.pools.items()},
                'labels_used_only_after_prediction': True, 'dense_maps_saved': False,
                'sampling': 'up to 256 native pixels/image/stratum; priority reservoir 2048/class/stratum; same coordinates across observations',
                'interpolation': 'OpenCV bilinear diagnostic sampling; primary map uses unchanged torch bilinear',
                'aggregation': 'bag mean after each bag normalization; fused column is actual prediction',
                'interpretation': 'image-capped samples, not full-population pixel quantiles'}
