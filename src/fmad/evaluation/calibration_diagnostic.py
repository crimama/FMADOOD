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

    def compute(self):
        return {'columns': self.columns, 'queries': self.queries,
                'samples': {k: v.tolist() for k, (_, v) in self.pools.items()},
                'labels_used_only_after_prediction': True, 'dense_maps_saved': False,
                'sampling': 'up to 256 native pixels/image/stratum; priority reservoir 2048/class/stratum; same coordinates across observations',
                'interpolation': 'OpenCV bilinear diagnostic sampling; primary map uses unchanged torch bilinear',
                'aggregation': 'bag mean after each bag normalization; fused column is actual prediction',
                'interpretation': 'image-capped samples, not full-population pixel quantiles'}
