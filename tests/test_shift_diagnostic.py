import numpy as np
import pytest
from PIL import Image
from fmad.evaluation.shift_diagnostic import summarize_map

def test_separate_normal_fpr_and_defect_tpr():
    result=summarize_map(np.array([[0.,2.],[3.,4.]]),np.array([[0,0],[1,1]]),2.5)
    assert result['normal_fpr']==0 and result['defect_tpr']==1
    assert result['normal_mean']==1 and result['defect_mean']==3.5

def test_empty_defect_is_missing_not_zero():
    result=summarize_map(np.zeros((2,2)),np.zeros((2,2)),1.)
    assert result['defect_tpr'] is None
    with pytest.raises(ValueError):
        summarize_map(np.full((2,2),np.nan),np.zeros((2,2)),1.)

def test_acquisition_parser_rejects_unknown_conditions():
    from fmad.evaluation.shift_diagnostic import acquisition_identity
    assert acquisition_identity('001_shift_4.png')==('001','shift_4')
    assert acquisition_identity('010_regular.png')==('010','regular')
    with pytest.raises(ValueError):acquisition_identity('001_mixed.png')

def test_paired_evaluator_normal_only_calibration_and_no_input_mutation(tmp_path):
    from fmad.evaluation.shift_diagnostic import run_shift_diagnostic
    from fmad.protocol import select_support_paths
    paths=[]
    for i in range(40):
        p=tmp_path/f'{i:02}.png';Image.fromarray(np.full((4,4,3),100+i,dtype=np.uint8)).save(p);paths.append(str(p))
    class Dataset:
        def get_train_images(self,obj):return paths
    class Method:
        def predict(self,path):
            with Image.open(path) as image: a=np.asarray(image).mean(-1)
            return a,float(a.max())
    originals=[__import__('pathlib').Path(p).read_bytes() for p in paths]
    result=run_shift_diagnostic(Method(),Dataset(),'can',0,[('good',paths[0]),('bad',paths[1])],
        lambda d,o,k,p,s: np.ones(s,dtype=bool) if k=='bad' else np.zeros(s,dtype=bool))
    reserved,_=select_support_paths(paths,16,0)
    assert not set(reserved)&set(result['calibration_paths'])
    assert len(result['records'])==4
    assert [__import__('pathlib').Path(p).read_bytes() for p in paths]==originals

def test_real_acquisition_full_records_and_condition_metrics(tmp_path):
    from fmad.evaluation.shift_diagnostic import run_shift_diagnostic
    train=[]
    for i in range(40):
        p=tmp_path/f'train{i}.png';Image.fromarray(np.full((4,4,3),100+i,dtype=np.uint8)).save(p);train.append(str(p))
    class Dataset:
        def get_train_images(self,obj):return train
    class Method:
        def predict(self,path):
            with Image.open(path) as image:a=np.asarray(image).mean(-1)
            return a,float(a.max())
    queries=[]
    for kind in ['good','bad']:
        (tmp_path/kind).mkdir()
        for condition in ['regular','shift_1']:
            p=tmp_path/kind/f'000_{condition}.png'
            a=np.full((4,4,3),100,dtype=np.uint8)
            if kind=='bad':a[1:3,1:3]=250
            Image.fromarray(a).save(p);queries.append((kind,str(p)))
    def mask(d,o,k,p,s):
        m=np.zeros(s,dtype=bool)
        if k=='bad':m[1:3,1:3]=1
        return m
    r=run_shift_diagnostic(Method(),Dataset(),'can',0,queries,mask,mode='real_acquisition')
    assert len(r['records'])==4 and len(r['calibration_leave_one_image_out_thresholds'])==16
    assert set(r['condition_metrics'])=={'regular','shift_1'}
    assert all(v['prediction_map_count']==2 for v in r['condition_metrics'].values())
    with pytest.raises(ValueError,match='scene sets differ'):
        run_shift_diagnostic(Method(),Dataset(),'can',0,queries[:-1],mask,mode='real_acquisition')
