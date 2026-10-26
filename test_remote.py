
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/.local-sync/current/src")
from fmad.registry import DATASET_REGISTRY
import fmad.datasets
from fmad.pipeline import _ground_truth_mask

SOURCE_ROOTS = (
    Path("/workspace/results_local_compatible/scopead_mechanism_20261020"),
    Path("/var/tmp/fsad-analysis/scopead_mechanism_20261020"),
)

target_indexes = []
for r in SOURCE_ROOTS:
    if r.is_dir():
        for p in r.rglob("mechanism_cache_index.json"):
            d = json.loads(p.read_text())
            cfg = d.get("config", {})
            if cfg.get("protocol", {}).get("shots") == 2 and cfg.get("seed") == 0:
                target_indexes.append(p)

print(f"Found {len(target_indexes)} class indexes for shot 2, seed 0")

weights = torch.tensor([1/16, 7/48, 13/48, 25/48])[:, None, None]

def get_maps(item, shape):
    raw_1120 = item["raw_1120"].mean(axis=0)
    q95_1120 = item["q95_1120"].mean(axis=0)
    m1 = F.interpolate(torch.from_numpy(raw_1120[2])[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0].numpy()
    rf = (raw_1120[1] + raw_1120[2])/2.0 + 0.125 * raw_1120[0]
    m2 = F.interpolate(torch.from_numpy(rf)[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0].numpy()
    qf = (q95_1120[1] + q95_1120[2])/2.0 + 0.125 * q95_1120[0]
    m3 = F.interpolate(torch.from_numpy(qf)[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0].numpy()
    maps = []
    for r in [448, 672, 896, 1120]:
        qr = item[f"q95_{r}"].mean(axis=0)
        sc = (qr[1] + qr[2])/2.0 + 0.125 * qr[0]
        maps.append(F.interpolate(torch.from_numpy(sc)[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0])
    stacked = torch.sort(torch.stack(maps, dim=0), dim=0, descending=False).values
    m4 = (stacked * weights).sum(dim=0).numpy()
    return {"Raw_L23_1120": m1, "Raw_Fusion_1120": m2, "Q95_Fusion_1120": m3, "ScopeAD_Full": m4}

arms = ["Raw_L23_1120", "Raw_Fusion_1120", "Q95_Fusion_1120", "ScopeAD_Full"]
class_results = {}

for idx_p in target_indexes:
    info = json.loads(idx_p.read_text())
    obj = info["object"]
    cfg = info["config"]
    dataset = DATASET_REGISTRY.get(cfg["dataset"])(data_root=cfg["data_root"], config=cfg.get("dataset_config", {}))
    cache_dir = idx_p.parent / "mechanism_cache"
    
    bg_scores = {a: [] for a in arms}
    fg_scores = {a: [] for a in arms}
    
    for p, rec in info["records"].items():
        is_bad = Path(p).parent.name == "bad"
        with np.load(cache_dir / rec["file"]) as arch:
            item = {k: arch[k] for k in arch.files}
        shape = tuple(int(x) for x in item["shape"])
        maps_dict = get_maps(item, shape)
        if is_bad:
            mask = _ground_truth_mask(dataset, obj, "bad", p, shape)
            for a in arms:
                bg_scores[a].append(maps_dict[a][~mask][::50])
                fg_scores[a].append(maps_dict[a][mask])
        else:
            for a in arms:
                bg_scores[a].append(maps_dict[a].reshape(-1)[::50])
                
    res_obj = {}
    for a in arms:
        all_bg = np.concatenate(bg_scores[a])
        all_fg = np.concatenate(fg_scores[a]) if len(fg_scores[a]) and len(np.concatenate(fg_scores[a])) else np.array([])
        th_01 = np.percentile(all_bg, 99.0)
        th_05 = np.percentile(all_bg, 95.0)
        tpr_01 = (all_fg > th_01).mean() if len(all_fg) else 0.0
        tpr_05 = (all_fg > th_05).mean() if len(all_fg) else 0.0
        res_obj[a] = {"tpr_01": float(tpr_01), "tpr_05": float(tpr_05), "th_01": float(th_01)}
    class_results[obj] = res_obj
    print(f"Done {obj:12s}: Raw_L23={res_obj['Raw_L23_1120']['tpr_01']*100:4.1f}% -> ScopeAD={res_obj['ScopeAD_Full']['tpr_01']*100:4.1f}%")

print("=== Macro Average over 8 classes (Shot 2, Seed 0) ===")
for a in arms:
    m_01 = np.mean([class_results[obj][a]["tpr_01"] for obj in class_results]) * 100
    m_05 = np.mean([class_results[obj][a]["tpr_05"] for obj in class_results]) * 100
    print(f"{a:18s} | Recall @ FPR=0.01: {m_01:5.2f}% | Recall @ FPR=0.05: {m_05:5.2f}%")
