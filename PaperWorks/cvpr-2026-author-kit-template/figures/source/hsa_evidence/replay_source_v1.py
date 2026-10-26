#!/usr/bin/env python3
"""Post-hoc, CPU-only replay of matched 2-shot/seed-0 public score caches.

GT is used only for diagnostics after replay. Thresholds are frozen from the
existing Table 6 analysis, using every 100th normal pixel per native image.
No CQ, noise guard, fitting, or per-image score rescaling is applied.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOTS = [Path('/workspace/results_local_compatible/scopead_mechanism_20261020'),
         Path('/var/tmp/fsad-analysis/scopead_mechanism_20261020')]
ARMS = ['Raw_L23_1120', 'ScopeAD_Full']
EDGES = np.r_[0., np.geomspace(.001, 8., 241), np.inf]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(path):
    with np.load(path) as z:
        shape = tuple(int(x) for x in z['shape'])
        raw = torch.from_numpy(z['raw_1120'][:, 2].copy()).mean(0)
        raw = F.interpolate(raw[None, None], size=shape, mode='bilinear',
                            align_corners=False)[0, 0].numpy()
        maps = []
        for r in z['resolutions']:
            v = torch.from_numpy(z[f'q95_{int(r)}'].copy())
            score = ((v[:, 1] + v[:, 2]) / 2 + .125 * v[:, 0]).mean(0)
            maps.append(F.interpolate(score[None, None], size=shape,
                        mode='bilinear', align_corners=False)[0, 0])
        ordered = torch.sort(torch.stack(maps), dim=0).values
        weights = torch.tensor([1/16, 7/48, 13/48, 25/48])[:, None, None]
        full = (ordered * weights).sum(0).numpy()
    return [raw, full]


def mask_for(path, shape):
    p = Path(path)
    if p.parent.name != 'bad':
        return np.zeros(shape, dtype=bool)
    root = p.parents[1] / 'ground_truth' / 'bad'
    candidates = [root / (p.stem + '_mask.png'), root / (p.stem + '.png')]
    mp = next((x for x in candidates if x.exists()), None)
    if mp is None:
        raise FileNotFoundError(str(candidates))
    mask = np.asarray(Image.open(mp).convert('L')) > 0
    assert mask.shape == shape, (mp, mask.shape, shape)
    return mask


def groups(path):
    stem = Path(path).stem.lower()
    if 'regular' in stem and 'irregular' not in stem:
        return ['all', 'regular']
    return ['all', 'nonregular']


def empty_counts():
    return dict(normal=0, defect=0, raw_tp=0, full_tp=0, raw_fp=0, full_fp=0,
                recovered=0, lost=0, corrected_fp=0, introduced_fp=0)


def analyze(cls, output, threshold_path):
    start = time.time()
    indexes = []
    for root in ROOTS:
        for p in root.rglob('mechanism_cache_index.json'):
            info = json.loads(p.read_text())
            cfg = info['config']
            if (info['object'] == cls and cfg['protocol']['shots'] == 2
                    and cfg['seed'] == 0):
                indexes.append((p, info))
    assert len(indexes) == 1, [str(p) for p, _ in indexes]
    idx, info = indexes[0]
    thresholds = json.loads(threshold_path.read_text())[cls]
    th = [thresholds[a]['th_01'] for a in ARMS]
    hist = {g: np.zeros((2, 2, len(EDGES)-1), dtype=np.int64)
            for g in ['all', 'regular', 'nonregular']}
    totals = {g: empty_counts() for g in hist}
    samples = [[], []]
    images = []
    for n, (path, rec) in enumerate(sorted(info['records'].items())):
        assert rec['replay_max_abs_error'] <= 2e-5, rec
        maps = replay(idx.parent / 'mechanism_cache' / rec['file'])
        mask = mask_for(path, maps[0].shape)
        preds = [m > t for m, t in zip(maps, th)]
        raw, full = preds
        counts = dict(normal=int((~mask).sum()), defect=int(mask.sum()),
                      raw_tp=int((raw & mask).sum()), full_tp=int((full & mask).sum()),
                      raw_fp=int((raw & ~mask).sum()), full_fp=int((full & ~mask).sum()),
                      recovered=int((~raw & full & mask).sum()), lost=int((raw & ~full & mask).sum()),
                      corrected_fp=int((raw & ~full & ~mask).sum()),
                      introduced_fp=int((~raw & full & ~mask).sum()))
        assert counts['full_tp']-counts['raw_tp'] == counts['recovered']-counts['lost']
        assert counts['full_fp']-counts['raw_fp'] == counts['introduced_fp']-counts['corrected_fp']
        gs = groups(path)
        for a, m in enumerate(maps):
            samples[a].append(m[~mask][::100])
            for label, region in enumerate([~mask, mask]):
                h = np.histogram(m[region] / th[a], bins=EDGES)[0]
                assert h.sum() == region.sum()
                for g in gs:
                    hist[g][a, label] += h
        for g in gs:
            for key, value in counts.items():
                totals[g][key] += value
        images.append(dict(path=path, group=gs[-1], **counts))
        if (n+1) % 20 == 0:
            print(cls, n+1, '/', len(info['records']), flush=True)
    audit = {}
    for a, arm in enumerate(ARMS):
        estimated = float(np.percentile(np.concatenate(samples[a]), 99))
        recall = totals['all'][['raw_tp', 'full_tp'][a]] / totals['all']['defect']
        audit[arm] = dict(threshold=th[a], recomputed_threshold=estimated,
                         table_recall=thresholds[arm]['tpr_01'], replay_recall=recall,
                         exact_normal_fpr=totals['all'][['raw_fp', 'full_fp'][a]] / totals['all']['normal'])
        assert abs(estimated-th[a]) <= 2e-5, audit[arm]
        assert abs(recall-thresholds[arm]['tpr_01']) <= 1e-4, audit[arm]
    c = totals['nonregular']
    delta = (c['full_tp']-c['raw_tp']) / c['defect']
    baseline = c['raw_tp'] / c['defect']
    eligible = [v for v in images if v['group'] == 'nonregular' and v['defect'] >= 500]
    chosen = min(eligible, key=lambda v: (
        abs((v['full_tp']-v['raw_tp'])/v['defect']-delta)
        + .25*abs(v['raw_tp']/v['defect']-baseline), v['path']))
    selected = chosen['path']
    maps = replay(idx.parent / 'mechanism_cache' / info['records'][selected]['file'])
    mask = mask_for(selected, maps[0].shape)
    rgb = Image.open(selected).convert('RGB')
    rgb.thumbnail((640, 640), Image.Resampling.LANCZOS)
    size = rgb.size
    arrays = dict(rgb=np.asarray(rgb))
    for key, arr in [('mask', mask), ('raw', maps[0] > th[0]), ('full', maps[1] > th[1])]:
        arrays[key] = np.asarray(Image.fromarray(arr).resize(size, Image.Resampling.NEAREST))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f'{cls}_example.npz', **arrays)
    report = dict(object=cls, shots=2, seed=0, split='test_public (seen development)',
                  label_use='GT only after prediction; post-hoc diagnostic, not held-out evidence',
                  protocol='Class/arm pooled Q99 of every 100th normal pixel per native image; thresholds frozen from Table 6. Metrics on all native pixels. No CQ/noise guard.',
                  grouping='Filename containing regular -> regular; all others -> nonregular; audited against filenames.',
                  index=str(idx), index_sha256=sha(idx), config=info['config'],
                  script_sha256=sha(__file__), threshold_source_sha256=sha(threshold_path),
                  totals=totals, audit=audit,
                  histogram_edges=[float(x) if np.isfinite(x) else None for x in EDGES],
                  histograms={g: h.tolist() for g, h in hist.items()},
                  images=images, selected_example=chosen,
                  selection='Nonregular bad image >=500 defect pixels; minimize |image delta-class delta| + .25 |image raw recall-class raw recall|; path tie-break. Post-hoc representative, not maximum gain.',
                  elapsed_seconds=time.time()-start)
    (output / f'{cls}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(object=cls, audit=audit, totals=totals, selected=chosen,
                          elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class-name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold-source', type=Path, default=Path('/tmp/hsa_direct_validation.json'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    analyze(args.class_name, args.output, args.threshold_source)
