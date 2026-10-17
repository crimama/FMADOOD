import json

import numpy as np
import pytest
import torch

from fmad.evaluation.reference_bank_diagnostic import (
    paired_reference_distances, summarize_reference_shift,
)


@pytest.mark.parametrize('sources',[2,3,16])
def test_paired_distances_equal_physical_source_removal(sources):
    rng=torch.Generator().manual_seed(9)
    bank=torch.nn.functional.normalize(torch.randn(sources*7,13,generator=rng),dim=1)
    query=torch.nn.functional.normalize(torch.randn(19,13,generator=rng),dim=1)
    owners=torch.arange(sources).repeat_interleave(7)
    full,excluded=paired_reference_distances(query,bank,owners,sources)
    torch.testing.assert_close(full,(1-query@bank.T).amin(1))
    for source in range(sources):
        expected=(1-query@bank[owners!=source].T).amin(1)
        torch.testing.assert_close(excluded[:,source],expected)
    assert torch.all(excluded>=full[:,None])


def test_tied_sources_and_missing_source():
    query=torch.tensor([[1.,0.]])
    bank=torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
    full,excluded=paired_reference_distances(query,bank,[0,1,2],3)
    assert full.item()==0 and torch.equal(excluded,torch.zeros(1,3))
    with pytest.raises(ValueError,match='every declared source'):
        paired_reference_distances(query,bank,[0,0,2],3)


def test_summary_retains_image_units_and_separates_two_shifts():
    full=np.arange(1,21,dtype=float).reshape(2,10)/20
    excluded=np.stack([full*1.5,full*2],axis=-1)
    record=summarize_reference_shift(full,excluded,2.)
    assert record['normal_images']==2
    assert record['query_full_q95']==pytest.approx(np.quantile(full,.95))
    assert record['excluded_to_full_q95']>1
    assert record['excluded_to_support_q95']<1
    assert len(record['per_image_excluded_q95'])==2
    json.dumps(record,allow_nan=False)


def test_entrypoint_preserves_q95_metrics_and_bank_state(tmp_path,monkeypatch):
    import scripts.reference_bank_diagnostic as entry
    from fmad.pipeline import run_pipeline
    from fmad.methods.statistical_reference_matching import StatisticalReferenceMatching
    from tests.test_fmad_swappable_pipeline import make_dataset
    from tests.test_reference_matching import TinyTripleBackbone
    from tests.test_statistical_reference_matching import config as method_config
    dataset=make_dataset(tmp_path)
    dataset.get_objects=lambda:[dataset.get_object_info('thing')]
    config=dict(seed=0,method='statistical_reference_matching',dataset='tiny',backbone='tiny',
        device='cpu',data_root=str(dataset.root),output_dir=str(tmp_path/'diagnostic'),
        protocol=dict(shots=2,warmup_iters=0),
        method_config=method_config(statistical_normalization='q95'),
        output=dict(save_anomaly_maps=False,save_weights=False),
        evaluation=dict(streaming=True,inference_batch_size=1,reference_bank_diagnostic=True))
    monkeypatch.setattr(entry,'parse_args',lambda:None)
    monkeypatch.setattr(entry,'load_config',lambda _:config)
    monkeypatch.setattr(entry.DATASET_REGISTRY,'get',lambda _:lambda **kw:dataset)
    monkeypatch.setattr(entry.BACKBONE_REGISTRY,'get',lambda _:lambda **kw:TinyTripleBackbone())
    entry.main()
    manifest=json.loads(next((tmp_path/'diagnostic').rglob('run_manifest.json')).read_text())
    diagnostic=manifest['objects']['thing']['reference_bank_diagnostic']
    assert diagnostic['normal_query_count']==1
    assert len(diagnostic['observations'])==6
    assert diagnostic['full_distance_max_parity_error']<2e-5
    assert diagnostic['method_fit_or_prediction_changed'] is False
    baseline=StatisticalReferenceMatching(TinyTripleBackbone(),config['method_config'])
    run_pipeline(baseline,dataset,baseline.backbone,config,str(tmp_path/'baseline'))
    got=json.loads(next((tmp_path/'diagnostic').rglob('metrics.json')).read_text())
    expected=json.loads(next((tmp_path/'baseline').rglob('metrics.json')).read_text())
    assert got==expected
    assert manifest['objects']['thing']['ablation_bank_feature_sha256']==baseline.object_metadata()['ablation_bank_feature_sha256']
