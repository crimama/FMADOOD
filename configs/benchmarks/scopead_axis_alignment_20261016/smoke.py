"""Real-backbone control parity and axis smoke; no evaluator labels."""
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from fmad.backbones.dinov2 import DINOv2Backbone
from fmad.datasets.mvtec_ad2 import MVTecAD2Dataset
from fmad.methods.reference_matching_fast import ReferenceMatchingFast
from fmad.methods.statistical_reference_matching import StatisticalReferenceMatching
from fmad.protocol import MethodContext, seed_generators, select_support_paths

root = Path('/workspace/results_local_compatible/scopead_axis_alignment_20261016')
root.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load(Path('configs/benchmarks/scopead_axis_alignment_20261016/ad2_axis_layer_s02.yaml').read_text())
data = MVTecAD2Dataset('/home/hunim/Volume/DATA/mvtec_ad_2', cfg['dataset_config'])
info = data.get_object_info('can')
support, manifest_sha = select_support_paths(data.get_train_images('can'), 2, 0, manifest_path=Path(cfg['protocol']['fixed_reference_manifest']), object_name='can')
# Deterministic filename selection only; no masks or labels enter any method.
queries = sorted(p for group in data.get_test_images('can').values() for p in group)[:2]
context = MethodContext('can', 0, info.resolution, info.masking, info.rotation)
seed_generators(0)
backbone = DINOv2Backbone('dinov2_vitl14', 'cuda:0', **cfg['backbone_config'])
backbone.set_resolution(context.resolution)
backbone.warmup(support[0], iters=1)
results, controls = {}, {}
for name in ('parent_none', 'none', 'parent_q95', 'q95', 'axis_layer', 'axis_resolution', 'axis_layer_resolution'):
 seed_generators(0)
 mc = dict(cfg['method_config'])
 if name.startswith('parent_'):
  mc['calibration_audit'] = False
  mc['score_normalization'] = 'none' if name == 'parent_none' else 'tailscale'
  method = ReferenceMatchingFast(backbone, mc)
 else:
  mc['statistical_normalization'] = name
  method = StatisticalReferenceMatching(backbone, mc)
 method.fit(support, context)
 metadata = method.object_metadata()
 predictions, routing = [], []
 for query in queries:
  prediction = method.predict_batch([query])[0]
  predictions.append(prediction.score_map.copy())
  routing.append(list(method.last_resolutions))
 assert all(np.isfinite(x).all() for x in predictions)
 results[name] = dict(bank=metadata['ablation_bank_feature_sha256'], routing=routing, metadata=metadata, prediction_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in predictions])
 if name.startswith('parent_'):
  controls[name[7:]] = predictions
 elif name in controls:
  errors = [float(np.max(np.abs(x-y))) for x,y in zip(predictions,controls[name])]
  assert max(errors) <= 1e-6, errors
  assert results[name]['bank'] == results['parent_'+name]['bank']
  assert routing == results['parent_'+name]['routing']
  results[name]['parent_max_abs_error'] = errors
 del method
 gc.collect()
 torch.cuda.empty_cache()
(root/'smoke.json').write_text(json.dumps(dict(source=Path('.fmad-source-sha256').read_text().strip(), support_manifest_sha256=manifest_sha, support=support, queries=queries, results=results), indent=2))
print('SMOKE_PASS', {k: v.get('parent_max_abs_error') for k,v in results.items()}, flush=True)
