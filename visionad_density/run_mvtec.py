from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (ROOT, SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flow_tte_mvtec_classic import (  # noqa: E402
    ClassicEvaluationConfig,
    ClassicMVTecDataset,
    evaluate_classic_mvtec,
)
from visionad_aligned_backbone import VisionADAlignedBackbone  # noqa: E402
from visionad_density.core import (  # noqa: E402
    DensityStats,
    combine_score,
    fuse_layers,
    restore_query_map,
    upsample_map,
)

OBJECTS = ("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper")


def extract(backbone: VisionADAlignedBackbone, tensor: torch.Tensor) -> torch.Tensor:
    layers = backbone.extract_features(tensor)
    return fuse_layers([torch.from_numpy(layer).to(backbone.device) for layer in layers])


def prepare_support(backbone: VisionADAlignedBackbone, paths: list[Path], mode: str) -> torch.Tensor:
    rows = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        tensor, _ = backbone.prepare_image(np.asarray(image))
        views = (
            tensor,
            torch.rot90(tensor, 1, (-2, -1)),
            torch.rot90(tensor, 2, (-2, -1)),
            torch.rot90(tensor, 3, (-2, -1)),
            torch.flip(tensor, (-2,)),
            torch.flip(tensor, (-1,)),
        )
        for tensor in views:
            if mode == "clamp":
                tensor = tensor.clamp_min(0)
            elif mode == "vflip":
                tensor = torch.flip(tensor, dims=(-2,))
            rows.append(extract(backbone, tensor))
    return torch.cat(rows, dim=0)


def query_features(backbone: VisionADAlignedBackbone, image: Image.Image, mode: str) -> torch.Tensor:
    tensor, _ = backbone.prepare_image(np.asarray(image))
    if mode == "clamp":
        tensor = tensor.clamp_min(0)
    elif mode == "vflip":
        tensor = torch.flip(tensor, dims=(-2,))
    return extract(backbone, tensor)


def write_map(root: Path, obj: str, anomaly: str, stem: str, score: np.ndarray) -> None:
    out = root / "anomaly_maps" / obj / "test" / anomaly
    out.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(out / f"{stem}.tiff"), score.astype(np.float32))


def align_maps_to_original_geometry(root: Path, dataset: ClassicMVTecDataset, objects: tuple[str, ...]) -> None:
    for obj in objects:
        for anomaly, paths in dataset.get_test_images(obj).items():
            for raw_path in paths:
                path = Path(raw_path)
                with Image.open(path) as image:
                    width, height = image.size
                map_path = root / "anomaly_maps" / obj / "test" / anomaly / f"{path.stem}.tiff"
                score = tiff.imread(str(map_path)).astype(np.float32)
                if score.shape != (height, width):
                    tensor = torch.from_numpy(score)[None, None]
                    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
                    tiff.imwrite(str(map_path), resized[0, 0].numpy().astype(np.float32))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--objects", default=",")
    p.add_argument("--shots", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--density-weight", type=float, default=0.25)
    p.add_argument("--evaluate-existing", action="store_true")
    args = p.parse_args()
    objects = tuple(x for x in args.objects.split(",") if x) if args.objects != "," else OBJECTS
    dataset = ClassicMVTecDataset(args.data_root, objects)
    roots = {"baseline": Path(args.output_root) / "baseline", "density": Path(args.output_root) / "pseudo_density"}
    support_manifest = {}
    for obj in objects:
        train = [Path(x) for x in dataset.get_train_images(obj)]
        generator = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(train), generator=generator)[: args.shots].tolist()
        support_manifest[obj] = [str(train[i]) for i in idx]
    if not args.evaluate_existing:
        backbone = VisionADAlignedBackbone("dinov2_vitb14_reg", args.device, 448, 392, tuple(range(2, 10)))
        for obj in objects:
            selected = [Path(x) for x in support_manifest[obj]]
            states = {}
            for mode in ("original", "vflip", "clamp"):
                support = prepare_support(backbone, selected, mode)
                states[mode] = (support, DensityStats.fit(support))
            for anomaly, paths in dataset.get_test_images(obj).items():
                for raw_path in paths:
                    path = Path(raw_path); image = Image.open(path).convert("RGB")
                    maps = {name: torch.zeros((1, 1, 28, 28), device=args.device) for name in roots}
                    for mode in ("original", "vflip", "clamp"):
                        query = query_features(backbone, image, mode)
                        support, density = states[mode]
                        for name, weight in (("baseline", 0.0), ("density", args.density_weight)):
                            patch = combine_score(query, support, density, weight)
                            maps[name] += restore_query_map(patch, (28, 28), mode == "vflip")
                    for name, root in roots.items():
                        write_map(root, obj, anomaly, path.stem, upsample_map(maps[name], 256))
    for name, root in roots.items():
        align_maps_to_original_geometry(root, dataset, objects)
        metrics = evaluate_classic_mvtec(ClassicEvaluationConfig(dataset, root, objects, 0.05, args.seed, 0.01, False))
        manifest = {"method": "VisionAD DADE official-mechanism port", "variant": name, "upstream_commit": "2d5e36f357409125ccd9646ba016824005599df4", "backbone": "dinov2_vitb14_reg", "feature_layers": list(range(2, 10)), "resize": 448, "crop": 392, "support_augmentation": ["rot0", "rot90", "rot180", "rot270", "vflip", "hflip"], "query_augmentation": ["original", "vflip", "clamp_min_0"], "score": "sum of three cosine patch 1-NN maps", "density_weight": 0.0 if name == "baseline" else args.density_weight, "shots": args.shots, "seed": args.seed, "support_paths": support_manifest, "metrics": metrics}
        (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
