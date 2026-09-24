"""Opt-in paired synthetic-shift diagnosis; labels stay in evaluation only."""
from pathlib import Path
import tempfile
import hashlib

import numpy as np
from PIL import Image

from fmad.protocol import select_support_paths


def summarize_map(score, mask, threshold):
    score = np.asarray(score, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if score.shape != mask.shape or not np.isfinite(score).all():
        raise ValueError("invalid diagnostic map")
    normal, defect = score[~mask], score[mask]
    return {"normal_fpr": float((normal > threshold).mean()) if normal.size else None,
            "defect_tpr": float((defect > threshold).mean()) if defect.size else None,
            "normal_mean": float(normal.mean()) if normal.size else None,
            "defect_mean": float(defect.mean()) if defect.size else None}


def acquisition_identity(path):
    scene, condition = Path(path).stem.split("_", 1)
    if not scene.isdigit() or condition not in {"regular", "underexposed", "overexposed", "shift_1", "shift_2", "shift_3", "shift_4"}:
        raise ValueError(f"unknown acquisition identity: {Path(path).name}")
    return scene, condition


def run_shift_diagnostic(method, dataset, object_name, seed, queries, mask_loader, mode=True):
    """Fixed 16 normal calibration images; eight good/eight bad paired queries.

Calibration excludes the full seeded16 support pool even for one-shot runs.
Each method calibrates its own clean-normal pixel95 threshold. Query labels
select diagnostic strata and masks ONLY; they never enter fit or prediction.
"""
    if mode is not True and mode != "real_acquisition":
        raise ValueError("unsupported shift diagnostic mode")
    train = sorted(dataset.get_train_images(object_name))
    reserved, _ = select_support_paths(train,16,seed,object_name=object_name)
    calibration = [p for p in train if p not in set(reserved)][:16]
    if len(calibration) != 16:
        raise ValueError("diagnostic requires16 additional normal calibration images")
    samples=[]
    for path in calibration:
        score=np.asarray(method.predict(path)[0],dtype=np.float32).flatten()
        if not np.isfinite(score).all():
            raise ValueError("nonfinite calibration")
        samples.append(score[np.linspace(0,len(score)-1,min(4096,len(score)),dtype=int)])
    threshold=float(np.quantile(np.concatenate(samples),.95))
    if mode == "real_acquisition":
        from fmad.evaluation.streaming import StreamingSegmentationAccumulator
        loo=[float(np.quantile(np.concatenate(samples[:i]+samples[i+1:]),.95)) for i in range(len(samples))]
        records=[]
        accumulators={}
        identities={}
        for kind,path in queries:
            scene,condition=acquisition_identity(path)
            identity=(kind,scene)
            if identity in identities.setdefault(condition,set()):
                raise ValueError("duplicate acquisition scene")
            identities[condition].add(identity)
            score=np.asarray(method.predict(path)[0],dtype=np.float32)
            mask=mask_loader(dataset,object_name,kind,path,score.shape)
            if condition not in accumulators:
                accumulators[condition]=StreamingSegmentationAccumulator(max_fpr=.05)
            accumulator=accumulators[condition]
            accumulator.update([score],[mask])
            records.append({"query":f"{kind}/{Path(path).name}","scene_id":scene,
                            "condition":condition,"normal_image":kind=="good",
                            **summarize_map(score,mask,threshold),
                            "threshold_loo_min":summarize_map(score,mask,min(loo)),
                            "threshold_loo_max":summarize_map(score,mask,max(loo))})
        if "regular" not in identities or any(s!=identities["regular"] for s in identities.values()):
            raise ValueError("acquisition scene sets differ; paired protocol invalid")
        metrics={}
        for condition,accumulator in accumulators.items():
            final=accumulator.compute()
            metrics[condition]={"seg_AUPRO_0.05":final.seg_aupro_0_05,
                                "seg_AUROC_0.05":final.seg_auroc_0_05,
                                "prediction_map_count":final.prediction_map_count}
        return {"scope":"real_acquisition_public_seen_development_not_private_server",
                "threshold":threshold,"threshold_source":"clean_train_normal_pixel_q95_16_images_max4096_pixels_each",
                "calibration_paths":calibration,
                "calibration_content_sha256":hashlib.sha256(b"".join(Path(p).read_bytes() for p in calibration)).hexdigest(),
                "calibration_leave_one_image_out_thresholds":loo,
                "uncertainty":"LOO_sensitivity_range_not_confidence_interval",
                "support_exclusion":"seeded16_pool_excluded_at_every_shot",
                "query_selection":"all_public_images_all_available_conditions",
                "condition_metrics":metrics,"records":records}
    selected=[]
    for normal in (True,False):
        selected.extend([(kind,path) for kind,path in queries if (kind=="good")==normal][:8])
    records=[]
    with tempfile.TemporaryDirectory(prefix="fmad-shift-diagnostic-") as directory:
        target=Path(directory)/"query.png"
        for kind,path in selected:
            with Image.open(path) as image:
                rgb=np.asarray(image.convert("RGB"))
            for condition,gain in (("clean",1.),("brightness_0.7",.7)):
                if gain==1.:
                    score=method.predict(path)[0]
                else:
                    Image.fromarray(np.rint(rgb.astype(np.float32)*gain).clip(0,255).astype(np.uint8)).save(target)
                    score=method.predict(str(target))[0]
                mask=mask_loader(dataset,object_name,kind,path,np.asarray(score).shape)
                records.append({"query":f"{kind}/{Path(path).name}","condition":condition,
                                "normal_image":kind=="good",**summarize_map(score,mask,threshold)})
    return {"scope":"post_evaluation_seen_development_synthetic_brightness_not_real_acquisition_shift",
            "threshold":threshold,"threshold_source":"clean_train_normal_pixel_q95_16_images_max4096_pixels_each",
            "calibration_paths":calibration,
            "calibration_content_sha256":hashlib.sha256(b"".join(Path(p).read_bytes() for p in calibration)).hexdigest(),
            "support_exclusion":"seeded16_pool_excluded_at_every_shot",
            "query_selection":"first8_sorted_good_and_first8_sorted_bad",
            "records":records}
