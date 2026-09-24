#!/usr/bin/env python3
"""Seeded variant of prepare_corrupted_mvtec.py for held-out corruption sets.

The original generator draws noise from the unseeded global NumPy RNG, so its
roots cannot be regenerated. Here every image is corrupted under a seed derived
from (class/split/file, corruption, severity), so the same inputs always produce
pixel-identical outputs. Train and ground_truth are symlinked to the clean
source, exactly as in the original.

`impulse_noise` is reimplemented here. imagecorruptions calls
skimage.util.random_noise, which in recent skimage draws from an unseeded
default_rng and ignores the global NumPy seed, so the library version is not
reproducible even run-to-run. The replacement applies the same salt-and-pepper
rule (each element flipped with probability `amount`, then salt or pepper with
probability 0.5 each) from a seeded RandomState. Its distribution matches the
library; its individual bit pattern does not.
"""
import argparse
import hashlib
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
from imagecorruptions import corrupt  # noqa: E402


IMPULSE_AMOUNT = [.03, .06, .09, 0.17, 0.27]


def seeded_impulse_noise(image: np.ndarray, severity: int, rng: np.random.RandomState) -> np.ndarray:
    x = np.array(image) / 255.
    flipped = rng.random_sample(x.shape) < IMPULSE_AMOUNT[severity - 1]
    salted = rng.random_sample(x.shape) < 0.5
    x = x.copy()
    x[flipped & salted] = 1.0
    x[flipped & ~salted] = 0.0
    return np.clip(x, 0, 1) * 255


def image_seed(relative: str, corruption: str, severity: int) -> int:
    key = f"{relative}|{corruption}|{severity}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def process(task):
    src, dst, relative, corruption, severity = task
    # Always write lossless PNG. PIL picks the encoder from the suffix, so a .JPG
    # destination would silently add JPEG compression on top of the corruption.
    # The seed key stays the source-relative path; masks pair by filename stem.
    dst = str(Path(dst).with_suffix(".png"))
    if Path(dst).exists():
        return relative, pixel_digest(dst)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    seed = image_seed(relative, corruption, severity)
    image = np.array(Image.open(src).convert("RGB"))
    if corruption == "impulse_noise":
        corrupted = seeded_impulse_noise(image, severity, np.random.RandomState(seed))
    else:
        np.random.seed(seed)
        corrupted = corrupt(image, corruption_name=corruption, severity=severity)
    Image.fromarray(np.uint8(corrupted)).save(dst)
    return relative, pixel_digest(dst)


def pixel_digest(path) -> str:
    # Hash decoded pixels, not file bytes: PNG encoders differ across PIL versions.
    return hashlib.sha256(np.asarray(Image.open(path).convert("RGB")).tobytes()).hexdigest()


def generate(src_root: Path, dst_root: Path, corruption: str, severity: int, workers: int,
             classes: list[str] | None = None, dataset_format: str = "mvtec"):
    """Generate corrupted images from source dataset.

    Args:
        src_root: Source dataset root
        dst_root: Output root
        corruption: Corruption type name
        severity: Severity level (1-5)
        workers: Number of worker processes
        classes: Optional list of class names to restrict processing
        dataset_format: "mvtec" or "visa" dataset layout
    """
    target = dst_root / f"{corruption}_s{severity}"
    tasks = []

    if dataset_format == "visa":
        # VisA format: src_root/<class>/{train/good, test/{good,bad}, ground_truth/bad}
        for cls in sorted(p for p in src_root.iterdir() if p.is_dir()):
            if classes and cls.name not in classes:
                continue
            out = target / cls.name
            out.mkdir(parents=True, exist_ok=True)
            # Symlink train and ground_truth
            for name in ("train", "ground_truth"):
                src_link = cls / name
                if src_link.is_dir():
                    link = out / name
                    if not link.exists():
                        link.symlink_to(src_link)
            # Process test images from all subdirectories (good, bad, etc.)
            test_dir = cls / "test"
            if test_dir.is_dir():
                for subdir in sorted(test_dir.iterdir()):
                    if not subdir.is_dir():
                        continue
                    for image_file in sorted(subdir.iterdir()):
                        # Accept .png, .jpg, .jpeg, .bmp (case-insensitive)
                        if not image_file.is_file() or image_file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                            continue
                        relative = f"{cls.name}/test/{subdir.name}/{image_file.name}"
                        dst_path = out / "test" / subdir.name / image_file.name
                        tasks.append((str(image_file), str(dst_path), relative, corruption, severity))
    else:
        # MVTec format: src_root/<class>/{train, test/anomaly_type, ground_truth}
        for cls in sorted(p for p in src_root.iterdir() if (p / "test").is_dir()):
            if classes and cls.name not in classes:
                continue
            out = target / cls.name
            out.mkdir(parents=True, exist_ok=True)
            for name in ("train", "ground_truth"):
                link = out / name
                if not link.exists():
                    link.symlink_to(cls / name)
            # Accept .png, .jpg, .jpeg, .bmp (case-insensitive)
            for image in sorted(cls.glob("test/*/*")):
                if not image.is_file() or image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                    continue
                relative = f"{cls.name}/test/{image.parent.name}/{image.name}"
                tasks.append((str(image), str(out / "test" / image.parent.name / image.name),
                              relative, corruption, severity))

    with ProcessPoolExecutor(workers) as pool:
        digests = dict(pool.map(process, tasks, chunksize=8))
    manifest = hashlib.sha256("".join(f"{k}:{digests[k]}\n" for k in sorted(digests)).encode()).hexdigest()
    (target / "SEEDED_MANIFEST.json").write_text(json.dumps(
        dict(corruption=corruption, severity=severity, images=len(digests),
             content_sha256=manifest, seed_rule="sha256(relative|corruption|severity)[:4] LE",
             digest="sha256 of decoded RGB pixels per image, then of sorted list"), indent=2))
    print(f"{corruption}_s{severity}: {len(digests)} images, content sha256 {manifest[:16]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--conditions", nargs="+", required=True, help="corruption:severity")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Optional list of class names to restrict processing")
    parser.add_argument("--dataset-format", choices=["mvtec", "visa"], default="mvtec",
                        help="Dataset layout format: 'mvtec' or 'visa'")
    args = parser.parse_args()
    for condition in args.conditions:
        corruption, severity = condition.split(":")
        generate(Path(args.src_root).resolve(), Path(args.dst_root).resolve(),
                 corruption, int(severity), args.workers, classes=args.classes or None,
                 dataset_format=args.dataset_format)


if __name__ == "__main__":
    main()
