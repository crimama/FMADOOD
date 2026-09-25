#!/usr/bin/env python3
"""Plot matched MVTec AD 2 test_public acquisition strata by support shot.

Examples:
  python3 scripts/plot_shift_shots.py --void-summary RSG3=results/void_instrumentation/void_summary.json --output-dir results/v7_evidence/E11_shift_shots/figures
  python3 scripts/plot_shift_shots.py --series 'RSG3=results/v7_evidence/E11_shift_shots::e11_rsg3_s*' --series RSG3=results/v7_evidence/E10_dsba3/e10_rsg3 --output-dir results/v7_evidence/E11_shift_shots/figures

Repeated labels combine roots. A point is emitted only with the exact declared
class set; no available-case averaging or zero fill is performed.
"""

import argparse
import csv
import fnmatch
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape


SHOTS = (1, 2, 4, 8, 16)
FULL8 = frozenset(('can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts'))
SHIFT7 = FULL8 - {'fruit_jelly'}
CONDITIONS = ('shift_1', 'shift_2', 'shift_3')
COLORS = ('#1764a3', '#d07820', '#308753', '#a34370', '#764cac', '#726c5d', '#17a4a1', '#b3483d')
QUERY_FINGERPRINT_KEYS = ('query_identity_sha256', 'query_content_sha256',
                          'ground_truth_sha256', 'query_count')


def parse_named(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('Expected LABEL=PATH')
    label, path = value.split('=', 1)
    if not label or not path:
        raise argparse.ArgumentTypeError('Expected nonempty LABEL=PATH')
    return label, Path(path)


def parse_series(value):
    label, value_path = parse_named(value)
    raw = str(value_path)
    if '::' in raw:
        root, stage = raw.rsplit('::', 1)
        if not root or not stage or '/' in stage or stage in ('.', '..'):
            raise argparse.ArgumentTypeError('Expected LABEL=ROOT::STAGE_GLOB with one directory pattern')
        return label, Path(root), stage
    return label, value_path, None


def add(store, label, shots, obj, condition, value, source, support_hash, contract,
        query_fingerprint=None):
    if shots not in SHOTS or condition not in ('regular',) + CONDITIONS:
        return
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'Invalid AUPRO in {source}: {value}')
    key = (label, shots, obj, condition)
    entry = dict(value=float(value), source=str(source), support_hash=support_hash,
                 contract=contract, query_fingerprint=query_fingerprint)
    old = store.get(key)
    if old is not None:
        if abs(old['value'] - entry['value']) > 1e-9 or old['support_hash'] != support_hash:
            raise ValueError(f'Conflicting duplicate cell {key}: {old["source"]} vs {source}')
        return
    store[key] = entry


def load_runs(store, label, root, stage):
    paths = sorted(p for p in root.rglob('run_manifest.json')
                   if stage is None or any(fnmatch.fnmatchcase(part, stage) for part in p.relative_to(root).parts))
    if not paths:
        raise ValueError(f'No run manifests under {root} for stage {stage!r}')
    if stage is None:
        # A combined pull directory contains several experiment stages. An
        # unlabeled broad root would silently assign all of them one method.
        stages = {part for p in paths for part in p.relative_to(root).parts if part.startswith(('e10_', 'e11_'))}
        if len(stages) > 1:
            raise ValueError(f'Mixed stage root {root}: choose LABEL=ROOT::STAGE ({sorted(stages)})')
    for path in paths:
        # The run manifest already records support count and seed; parsing YAML
        # is unnecessary and would add a plotting dependency.
        manifest = json.loads(path.read_text())
        for obj, info in manifest['objects'].items():
            strata = info.get('condition_strata')
            if not strata:
                continue
            protocol = info.get('protocol', {})
            if protocol.get('context', {}).get('seed') != 0:
                raise ValueError(f'Expected seed 0: {path}')
            shots = info.get('support_count') or protocol.get('support_count')
            support_hash = protocol.get('support_identity_sha256')
            contract = info.get('streaming_metrics', {}).get('metric_contract')
            if contract != 'mvtec_ad2_float16_streaming_v1' or not support_hash:
                raise ValueError(f'Missing comparable metric/support provenance: {path}')
            fingerprint = {key: protocol.get(key) for key in QUERY_FINGERPRINT_KEYS}
            if any(value is None for value in fingerprint.values()):
                raise ValueError(f'Missing query/GT fingerprint: {path}')
            for condition, metrics in strata.items():
                value = metrics.get('AUPRO_0_05')
                if value is not None:
                    add(store, label, shots, obj, condition, value, path,
                        support_hash, contract, fingerprint)


def load_void(store, label, path, arm):
    data = json.loads(path.read_text())
    for cell in data['cells']:
        if cell['arm'] != arm or cell['condition'] == 'all':
            continue
        add(store, label, cell['shots'], cell['object'], cell['condition'],
            cell['aupro'], path, cell['support_manifest'],
            'mvtec_ad2_float16_streaming_v1')


def make_rows(store):
    # Run manifests carry per-class support identity hashes. Validate shared
    # support across plotted methods whenever that provenance is available.
    # VOID's legacy summary carries a whole-manifest hash, so it cannot join
    # this per-class check and is identified separately in provenance.json.
    by_cell = {}
    for (label, shot, obj, condition), entry in store.items():
        if entry['source'].endswith('run_manifest.json'):
            by_cell.setdefault((shot, obj, condition), []).append((label, entry))
    for key, entries in by_cell.items():
        if len({entry['support_hash'] for _, entry in entries}) > 1:
            raise ValueError(f'Cross-method support mismatch at {key}: '
                             + ', '.join(f'{label}={entry["support_hash"]}' for label, entry in entries))
    # These four fingerprints describe the whole class query population, not
    # an individual stratum. Every manifest-based method/shot for a class must
    # therefore agree before any point enters the plot.
    by_object = {}
    for (label, shot, obj, condition), entry in store.items():
        if entry['query_fingerprint'] is not None:
            by_object.setdefault(obj, []).append((label, shot, condition, entry))
    for obj, entries in by_object.items():
        fingerprints = {tuple(entry['query_fingerprint'][key] for key in QUERY_FINGERPRINT_KEYS)
                        for _, _, _, entry in entries}
        if len(fingerprints) > 1:
            detail = ', '.join(f'{label}/{shot}/{condition}={entry["source"]}'
                               for label, shot, condition, entry in entries)
            raise ValueError(f'Cross-method/shot query or GT fingerprint mismatch for {obj}: {detail}')
    labels = sorted({key[0] for key in store})
    rows, omitted = [], []
    for label in labels:
        for shot in SHOTS:
            for condition in CONDITIONS:
                classes = FULL8 if condition == 'shift_1' else SHIFT7
                shift_keys = [(label, shot, obj, condition) for obj in classes]
                regular_keys = [(label, shot, obj, 'regular') for obj in classes]
                missing = [key for key in shift_keys + regular_keys if key not in store]
                if missing:
                    omitted.append(dict(method=label, shots=shot, condition=condition,
                                        reason='missing matched class cells', missing=[f'{k[2]}:{k[3]}' for k in missing]))
                    continue
                shift = [store[key] for key in shift_keys]
                regular = [store[key] for key in regular_keys]
                if {x['contract'] for x in shift + regular} != {'mvtec_ad2_float16_streaming_v1'}:
                    raise ValueError(f'Mixed metric contract: {label} {shot} {condition}')
                for obj in classes:
                    a, b = store[(label, shot, obj, condition)], store[(label, shot, obj, 'regular')]
                    if a['support_hash'] != b['support_hash']:
                        raise ValueError(f'Mismatched support: {label} {shot} {obj} {condition}')
                shifted = 100 * sum(x['value'] for x in shift) / len(classes)
                base = 100 * sum(x['value'] for x in regular) / len(classes)
                rows.append(dict(method=label, shots=shot, condition=condition, class_count=len(classes),
                                 classes=','.join(sorted(classes)), shift_aupro_pp=shifted,
                                 matched_regular_aupro_pp=base, delta_pp=shifted-base,
                                 sources=sorted({x['source'] for x in shift + regular}),
                                 support_hashes={obj: store[(label, shot, obj, condition)]['support_hash'] for obj in sorted(classes)},
                                 query_fingerprints={obj: store[(label, shot, obj, condition)]['query_fingerprint']
                                                     for obj in sorted(classes)}))
    return rows, omitted


def svg_plot(rows, condition, metric, title, output):
    selected = [r for r in rows if r['condition'] == condition]
    if not selected:
        return
    labels = sorted({r['method'] for r in selected})
    vals = [r[metric] for r in selected]
    lo, hi = min(vals), max(vals)
    pad = max(1.0, (hi-lo)*0.12)
    if metric == 'delta_pp':
        lo, hi = min(lo-pad, 0), max(hi+pad, 0)
    else:
        lo, hi = max(0, lo-pad), min(100, hi+pad)
    width, height = 840, 500
    left, top, plotw, ploth = 80, 65, 570, 350
    def x(shot): return left + SHOTS.index(shot)*plotw/4
    def y(value): return top + ploth*(hi-value)/(hi-lo)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="30" font-family="sans-serif" font-size="19" font-weight="bold">{escape(title)}</text>']
    for i in range(6):
        v = lo + (hi-lo)*i/5
        yy = y(v)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left+plotw}" y2="{yy:.2f}" stroke="#e2e5e8"/>')
        parts.append(f'<text x="{left-12}" y="{yy+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{v:.1f}</text>')
    for shot in SHOTS:
        xx=x(shot)
        parts.append(f'<text x="{xx:.2f}" y="{top+ploth+26}" text-anchor="middle" font-family="sans-serif" font-size="13">{shot}</text>')
    parts.append(f'<text x="{left+plotw/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">Normal support images (shot)</text>')
    for idx, label in enumerate(labels):
        color=COLORS[idx%len(COLORS)]
        points=sorted((r for r in selected if r['method']==label),key=lambda r:r['shots'])
        # Draw only adjacent observed shot positions; missing points are not interpolated.
        for a,b in zip(points,points[1:]):
            if SHOTS.index(b['shots'])-SHOTS.index(a['shots'])==1:
                parts.append(f'<line x1="{x(a["shots"]):.2f}" y1="{y(a[metric]):.2f}" x2="{x(b["shots"]):.2f}" y2="{y(b[metric]):.2f}" stroke="{color}" stroke-width="2.5"/>')
        for p in points:
            parts.append(f'<circle cx="{x(p["shots"]):.2f}" cy="{y(p[metric]):.2f}" r="4.5" fill="{color}"/>')
        parts.append(f'<circle cx="685" cy="{90+idx*27}" r="4.5" fill="{color}"/>')
        parts.append(f'<text x="700" y="{95+idx*27}" font-family="sans-serif" font-size="13">{escape(label)}</text>')
    parts.append('</svg>')
    output.write_text('\n'.join(parts)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--series', type=parse_series, action='append', default=[], metavar='LABEL=RUN_ROOT[::STAGE]')
    parser.add_argument('--void-summary', type=parse_named, action='append', default=[], metavar='LABEL=JSON')
    parser.add_argument('--void-arm', default='gated')
    parser.add_argument('--output-dir', required=True, type=Path)
    args=parser.parse_args()
    if not args.series and not args.void_summary:
        parser.error('Provide at least one --series or --void-summary')
    store={}
    for label, root, stage in args.series: load_runs(store,label,root,stage)
    for label, path in args.void_summary: load_void(store,label,path,args.void_arm)
    rows,omitted=make_rows(store)
    if not rows:
        raise ValueError('No complete matched condition points')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'shift_shots.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=('method','shots','condition','class_count','classes','shift_aupro_pp','matched_regular_aupro_pp','delta_pp'))
        writer.writeheader()
        writer.writerows({key:r[key] for key in writer.fieldnames} for r in rows)
    (args.output_dir/'provenance.json').write_text(json.dumps(dict(
        rows=rows, omitted=omitted,
        query_fingerprints={obj: entry['query_fingerprint']
                            for (_, _, obj, _), entry in store.items()
                            if entry['query_fingerprint'] is not None},
        legacy_void_limitation=('VOID summary cells have query_identity only; '
                                'query content, ground truth, and query count cannot be '
                                'cross-validated against run manifests.') if args.void_summary else None,
    ),indent=2)+'\n')
    for condition in CONDITIONS:
        svg_plot(rows,condition,'shift_aupro_pp',f'{condition}: AUPRO@0.05 (%)',args.output_dir/f'{condition}_absolute.svg')
        svg_plot(rows,condition,'delta_pp',f'{condition} minus matched regular (pp)',args.output_dir/f'{condition}_delta.svg')
    print(f'{len(rows)} complete points; {len(omitted)} incomplete points omitted. Output: {args.output_dir}')


if __name__=='__main__': main()
