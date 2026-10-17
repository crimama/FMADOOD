import gc,json,hashlib
from pathlib import Path
import torch,yaml
from fmad.backbones.dinov2 import DINOv2Backbone
from fmad.datasets.mvtec_ad2 import MVTecAD2Dataset
from fmad.methods.reference_matching_fast import ReferenceMatchingFast
from fmad.protocol import MethodContext,seed_generators,select_support_paths
cfg=yaml.safe_load(Path('/workspace/.local-sync/releases/2e433664bc283db78db3e8ddb69128dcc12a121e77ca02a13a7a490907d97a3e/configs/benchmarks/scopead_axis_alignment_20261016/ad2_axis_layer_s02.yaml').read_text())
data=MVTecAD2Dataset('/home/hunim/Volume/DATA/mvtec_ad_2',cfg['dataset_config'])
info=data.get_object_info('can'); context=MethodContext('can',0,info.resolution,info.masking,info.rotation)
support,_=select_support_paths(data.get_train_images('can'),2,0,manifest_path=Path(cfg['protocol']['fixed_reference_manifest']),object_name='can')
queries=sorted(p for group in data.get_test_images('can').values() for p in group)[:2]
seed_generators(0); backbone=DINOv2Backbone('dinov2_vitl14','cuda:0',**cfg['backbone_config']);backbone.set_resolution(info.resolution);backbone.warmup(support[0],iters=1)
results={}
for name in ('none','tailscale'):
 seed_generators(0);mc=dict(cfg['method_config']);mc.update(score_normalization=name,calibration_audit=False)
 method=ReferenceMatchingFast(backbone,mc);method.fit(support,context)
 hashes=[]; routing=[]
 for q in queries:
  prediction=method.predict_batch([q])[0]
  hashes.append(hashlib.sha256(prediction.score_map.tobytes()).hexdigest());routing.append(list(method.last_resolutions))
 results[name]=dict(bank=method.object_metadata()['ablation_bank_feature_sha256'],prediction_sha256=hashes,routing=routing)
 del method;gc.collect();torch.cuda.empty_cache()
print('HISTORICAL_PARITY_JSON',json.dumps(results),flush=True)
