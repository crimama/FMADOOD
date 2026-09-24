#!/usr/bin/env python3
"""G1 guardrail: count noise-guard firings on every guardrail query, offline.

Uses the method's own statistic (fmad.methods.reference_matching.immerkaer_sigma)
against the median of the exact support images each run used, so the firing
decision is identical to what the guarded model computes. No model or label.
Protocol: configs/benchmarks/noise_remedy/PROTOCOL_20260923.md
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fmad.methods.PSMA_MR import _read_rgb  # noqa: E402
from fmad.methods.reference_matching import immerkaer_sigma  # noqa: E402


def sigma_of(path):
    return immerkaer_sigma(_read_rgb(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supports', required=True, help='JSON {class: [support paths]}')
    parser.add_argument('--root', required=True, help='dataset root with <class>/<split>/*/*.png')
    parser.add_argument('--split', default='test')
    parser.add_argument('--label', required=True)
    parser.add_argument('--k', type=float, default=9.0)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    supports = json.loads(Path(args.supports).read_text())
    with ProcessPoolExecutor(args.workers) as pool:
        out = {}
        for cls, paths in sorted(supports.items()):
            queries = sorted(str(p) for p in Path(args.root, cls, args.split).glob('*/*.png'))
            reference = float(np.median(list(pool.map(sigma_of, paths))))
            ratios = list(pool.map(sigma_of, queries, chunksize=8))
            ratios = [r / reference for r in ratios]
            fired = [q for q, r in zip(queries, ratios) if r > args.k]
            out[cls] = dict(queries=len(queries), fired=len(fired), ratio_max=max(ratios),
                            ratio_min=min(ratios), fired_examples=fired[:5])
    total_q = sum(v['queries'] for v in out.values()); total_f = sum(v['fired'] for v in out.values())
    worst = max(out.items(), key=lambda kv: kv[1]['ratio_max'])
    print(f"{args.label:22s} queries={total_q:5d} fired={total_f:4d}  "
          f"max ratio={worst[1]['ratio_max']:.2f} ({worst[0]})  min ratio={min(v['ratio_min'] for v in out.values()):.2f}")
    print(json.dumps(dict(label=args.label, k=args.k, classes=out)), file=sys.stderr)


if __name__ == '__main__':
    raise SystemExit(main())
