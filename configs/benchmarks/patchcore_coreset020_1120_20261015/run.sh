#!/usr/bin/env bash
# Run the existing public runner and private submission preparation unchanged.
set -euo pipefail
cd "$(dirname "$0")/../../.."
ROOT=${PATCHCORE_CORESET_ROOT:-/var/tmp/patchcore_coreset020_1120_20261015}
DATA=/home/hunim/Volume/DATA/mvtec_ad_2
CFG=configs/benchmarks/patchcore_coreset020_1120_20261015/patchcore_coreset020_s16_seed0.yaml
export PYTHONPATH=$PWD/src PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_HOME=/model-cache FMAD_DINOV2_OFFLINE=1
GPUS=(GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62)
OBJECTS=("can fabric" "fruit_jelly rice" "sheet_metal vial" "wallplugs walnuts")
mkdir -p "$ROOT/logs" "$ROOT/private/logs"
# Retain the shared lock for the full background execution.
exec 9>/tmp/fmad-remote-execution.lock
flock -sn 9 || { echo 'An exclusive code update is active'; exit 1; }
printf '%s\n' "$PWD" > "$ROOT/source_release.txt"
sha256sum "$CFG" configs/methods/VOID_support_16.json src/fmad/methods/patchcore.py > "$ROOT/source_sha256.txt"
cp "$CFG" "$ROOT/config.yaml"
date -u +%FT%TZ > "$ROOT/started_at.txt"

lane() {
  local i=$1 obj
  export CUDA_VISIBLE_DEVICES=${GPUS[$i]}
  printf 'GPU=%s objects=%s\n' "$CUDA_VISIBLE_DEVICES" "${OBJECTS[$i]}"
  python3 scripts/experiments.py run --configs "$CFG" --labels patchcore_coreset020_s16_seed0 \
    --data-root "$DATA" --device cuda:0 --objects ${OBJECTS[$i]} \
    --split mvtec_ad2_test_public_seen_development_coreset020_s16_seed0 \
    --timeout 86400 --output-dir "$ROOT/public/gpu$i"
  for obj in ${OBJECTS[$i]}; do
    printf 'private start %s %s\n' "$obj" "$(date -u +%FT%TZ)"
    python3 scripts/prepare_rsg4_private_submission.py --config "$CFG" \
      --data-root "$DATA" --output-root "$ROOT/private" --object "$obj" \
      --device cuda:0 > "$ROOT/private/logs/$obj.log" 2>&1
    printf 'private done %s %s\n' "$obj" "$(date -u +%FT%TZ)"
  done
}
pids=()
for i in 0 1 2 3; do
  lane "$i" > "$ROOT/logs/gpu$i.log" 2>&1 & pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  date -u +%FT%TZ > "$ROOT/FAILED"
  echo 'At least one lane failed; completed artifacts retained; no archive built.'
  exit 1
fi
python3 - "$ROOT/private" "$DATA" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image
root, data = map(Path, sys.argv[1:])
objects = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
counts = {}
for obj in objects:
    provenance = json.loads((root / 'provenance' / (obj + '.json')).read_text())
    assert len(provenance['support_images']) == 16
    assert not set(provenance['support_images']) & set(provenance['calibration_images'])
    for split in ['test_private', 'test_private_mixed']:
        expected = {p.stem for p in (data / obj / split).glob('*.png')}
        assert expected, (obj, split, 'no input images')
        tiffs = {p.stem: p for p in (root / 'anomaly_images' / obj / split).glob('*.tiff')}
        masks = {p.stem: p for p in (root / 'anomaly_images_thresholded' / obj / split).glob('*.png')}
        assert set(tiffs) == set(masks) == expected, (obj, split, 'coverage')
        for stem in sorted(expected):
            score = tifffile.imread(tiffs[stem])
            assert score.ndim == 2 and score.dtype == np.float16 and np.isfinite(score).all()
            with Image.open(masks[stem]) as image:
                mask = np.asarray(image)
                assert mask.shape == score.shape and set(np.unique(mask)) <= {0, 255}
        counts[obj + '/' + split] = len(expected)
assert sum(counts.values()) == 4090, counts
(root / 'validation.json').write_text(json.dumps({'state': 'valid', 'counts': counts, 'total_per_output_type': 4090}, indent=2) + '\n')
print('Private coverage, finite score maps and binary masks verified: 4090 + 4090')
PY
tar -czf "$ROOT/submission.tar.gz" -C "$ROOT/private" anomaly_images anomaly_images_thresholded
sha256sum "$ROOT/submission.tar.gz" > "$ROOT/submission.sha256"
date -u +%FT%TZ > "$ROOT/DONE"
echo 'Public complete; private archive ready for official checker and upload.'
