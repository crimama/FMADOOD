#!/usr/bin/env python3
"""Offline post-processing search over SAVED anomaly maps.

Re-evaluates anomaly maps a run already wrote to
``<run_dir>/anomaly_maps/<object>/<anomaly_type>/<filename>.npz`` (array key
``score_map``, see ``ResultsManager.save_anomaly_map``) under a candidate
label-free post-processing function, without a GPU and without re-running
the detector.  With ``--transform identity`` this reproduces the pipeline's
own streaming segmentation metrics exactly -- that is the harness's primary
correctness property, and is covered by
``tests/test_map_postprocess_eval.py``.

Query enumeration mirrors ``fmad.pipeline._run_streaming_object`` (same sort
key, same sample-key format) and ground-truth masks are loaded through the
pipeline's own ``_ground_truth_mask`` helper -- this script never
reimplements mask reading.  Note that helper always resolves the
ground-truth directory under the dataset's default ``"test_public"`` split
regardless of which ``--split`` produced the test images, exactly mirroring
a quirk in the online pipeline; pass a non-default ``--split`` only when its
ground_truth layout also lives under ``test_public``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from fmad.evaluation.streaming import StreamingSegmentationAccumulator
from fmad.postprocess import component_quantile

MaskLoader = Callable[[str, str, str, tuple[int, int]], np.ndarray]
Transform = Callable[..., np.ndarray]


def _has_scipy() -> bool:
    try:
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


def _numpy_box_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Uniform-filter mean fallback for local_contrast when scipy is absent."""
    pad = window // 2
    padded = np.pad(values, pad, mode="reflect").astype(np.float64)
    height, width = values.shape
    cumsum = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)))
    total = (cumsum[window:window + height, window:window + width]
             - cumsum[:height, window:window + width]
             - cumsum[window:window + height, :width] + cumsum[:height, :width])
    return (total / (window * window)).astype(np.float32)


def identity(score_map: np.ndarray, **_: Any) -> np.ndarray:
    return score_map.astype(np.float32, copy=True)


def local_contrast(score_map: np.ndarray, window: int = 65, **_: Any) -> np.ndarray:
    """Subtract a local baseline (scipy median filter, else a box-mean)."""
    if _has_scipy():
        from scipy.ndimage import median_filter
        baseline = median_filter(score_map, size=window, mode="reflect")
    else:
        baseline = _numpy_box_mean(score_map, window)
    return (score_map - baseline).astype(np.float32)


def highpass(score_map: np.ndarray, sigma: float = 16.0, **_: Any) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return (score_map - gaussian_filter(score_map, sigma)).astype(np.float32)


def blur(score_map: np.ndarray, sigma: float = 2.0, **_: Any) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(score_map, sigma).astype(np.float32)


def open_close(score_map: np.ndarray, size: int = 3, **_: Any) -> np.ndarray:
    from scipy.ndimage import grey_closing, grey_opening
    footprint = np.ones((int(size), int(size)), dtype=bool)
    opened = grey_opening(score_map, footprint=footprint)
    return grey_closing(opened, footprint=footprint).astype(np.float32)


def region_quantile(score_map: np.ndarray, regions: Any = None, q: float = 0.9, **_: Any) -> np.ndarray:
    """Stub: needs region proposals supplied by the caller.

    This harness deliberately does not invent a segmentation to derive
    regions from -- pass pre-computed region labels via ``regions`` once a
    caller can supply them.
    """
    raise NotImplementedError("region_quantile requires region proposals to be passed in")



def close_only(score_map: np.ndarray, size: int = 9) -> np.ndarray:
    """Grey-close only: fill score holes inside a detected blob without eroding it.

    AUPRO@0.05 normalises per connected component, so completing a partially
    covered component is worth more than suppressing isolated noise.
    """
    import cv2
    size = int(size)
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.morphologyEx(np.ascontiguousarray(score_map, dtype=np.float32),
                            cv2.MORPH_CLOSE, kernel).astype(np.float32)


def dilate(score_map: np.ndarray, size: int = 9) -> np.ndarray:
    """Grey dilation: grow every local maximum into its neighbourhood."""
    import cv2
    size = int(size)
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.dilate(np.ascontiguousarray(score_map, dtype=np.float32), kernel).astype(np.float32)


TRANSFORMS: dict[str, Transform] = {
    "identity": identity,
    "local_contrast": local_contrast,
    "highpass": highpass,
    "blur": blur,
    "open_close": open_close,
    "region_quantile": region_quantile,
    "close_only": close_only,
    "dilate": dilate,
    "component_quantile": component_quantile,
}


def load_score_map(path: Path) -> np.ndarray:
    """Load and validate a saved ``score_map`` array; fail loudly if bad."""
    if not path.is_file():
        raise FileNotFoundError(f"missing anomaly map: {path}")
    with np.load(path) as data:
        if "score_map" not in data:
            raise ValueError(f"{path}: npz is missing the 'score_map' array")
        array = np.asarray(data["score_map"], dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{path}: score map must be 2D, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: score map contains non-finite values")
    return array


def map_path_for(run_dir: Path, object_name: str, anomaly_type: str, image_path: str) -> Path:
    name = Path(image_path).name
    return Path(run_dir) / "anomaly_maps" / object_name / anomaly_type / f"{name}.npz"


def enumerate_queries(dataset, object_name: str, split: str) -> list[tuple[str, str]]:
    """Same ordering as ``fmad.pipeline._run_streaming_object``."""
    grouped = dataset.get_test_images(object_name, split)
    return sorted(
        [(anomaly_type, image_path) for anomaly_type, image_paths in grouped.items()
         for image_path in image_paths],
        key=lambda item: (Path(item[1]).name, item[0]),
    )


def evaluate_object(
    run_dir: Path,
    object_name: str,
    queries: Sequence[tuple[str, str]],
    mask_loader: MaskLoader,
    transform: Transform,
    transform_kwargs: dict,
    accumulator: StreamingSegmentationAccumulator,
) -> None:
    """Feed every query's (transformed map, mask) pair into ``accumulator``.

    Raises immediately -- never skips a sample -- on a missing map file, a
    map/mask shape mismatch, or a non-finite value anywhere in the pipeline.
    """
    for anomaly_type, image_path in queries:
        map_path = map_path_for(run_dir, object_name, anomaly_type, image_path)
        score_map = load_score_map(map_path)
        mask = np.asarray(mask_loader(object_name, anomaly_type, image_path, score_map.shape), dtype=bool)
        if mask.shape != score_map.shape:
            raise ValueError(f"{map_path}: mask shape {mask.shape} != map shape {score_map.shape}")
        transformed = np.asarray(transform(score_map, **transform_kwargs))
        if transformed.shape != score_map.shape:
            raise ValueError(f"{map_path}: transform changed shape to {transformed.shape}")
        if transformed.dtype != np.float32:
            raise ValueError(f"{map_path}: transform must return float32, got {transformed.dtype}")
        if not np.isfinite(transformed).all():
            raise ValueError(f"{map_path}: transform produced non-finite values")
        accumulator.update([transformed], [mask])


def _metrics_from(result) -> dict[str, float]:
    return {
        "seg_AUROC_0.05": result.seg_auroc_0_05,
        "seg_AUPRO_0.05": result.seg_aupro_0_05,
        "seg_F1_oracle": result.seg_f1,
    }


def _parse_transform_arg(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"--transform-arg must be key=value, got {raw!r}")
    key, value = raw.split("=", 1)
    for caster in (int, float):
        try:
            return key, caster(value)
        except ValueError:
            continue
    return key, value


def _print_table(objects: Sequence[str], results: dict[str, dict], macro: dict[str, float]) -> None:
    columns = ["seg_AUPRO_0.05", "seg_AUROC_0.05", "seg_F1_oracle"]
    print(f"  {'object':<15}" + "".join(f"{c:>16}" for c in columns))
    print(f"  {'-' * (15 + 16 * len(columns))}")
    for name in objects:
        row = results[name]
        print(f"  {name:<15}" + "".join(f"{row[c]:>16.4f}" for c in columns))
    print(f"  {'-' * (15 + 16 * len(columns))}")
    print(f"  {'MACRO':<15}" + "".join(f"{macro[c]:>16.4f}" for c in columns))


def run_harness(
    run_dir: Path,
    dataset,
    objects: Sequence[str],
    split: str,
    mask_loader: MaskLoader,
    transform: Transform,
    transform_kwargs: dict,
) -> dict[str, Any]:
    """Evaluate every object; never returns a partial macro number."""
    results: dict[str, dict[str, float]] = {}
    for object_name in objects:
        queries = enumerate_queries(dataset, object_name, split)
        if not queries:
            raise ValueError(f"{object_name}: no test images found for split {split!r}")
        accumulator = StreamingSegmentationAccumulator(max_fpr=0.05)
        evaluate_object(run_dir, object_name, queries, mask_loader, transform, transform_kwargs, accumulator)
        results[object_name] = _metrics_from(accumulator.compute())
    macro = {
        column: float(np.mean([results[name][column] for name in objects]))
        for column in ("seg_AUROC_0.05", "seg_AUPRO_0.05", "seg_F1_oracle")
    }
    return {"objects": results, "macro": macro}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="run-... directory containing anomaly_maps/")
    parser.add_argument("--data-root", required=True, help="MVTec AD 2 dataset root")
    parser.add_argument("--objects", nargs="+", required=True)
    parser.add_argument("--split", default="test_public")
    parser.add_argument("--transform", default="identity", choices=sorted(TRANSFORMS))
    parser.add_argument("--transform-arg", action="append", default=[], metavar="k=v")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    transform_kwargs = dict(_parse_transform_arg(raw) for raw in args.transform_arg)
    transform = TRANSFORMS[args.transform]

    from fmad.datasets.mvtec_ad2 import MVTecAD2Dataset
    from fmad.pipeline import _ground_truth_mask

    dataset = MVTecAD2Dataset(args.data_root, {"objects": args.objects})

    def mask_loader(object_name: str, anomaly_type: str, image_path: str, shape) -> np.ndarray:
        return _ground_truth_mask(dataset, object_name, anomaly_type, image_path, shape)

    payload = run_harness(Path(args.run_dir), dataset, args.objects, args.split,
                          mask_loader, transform, transform_kwargs)
    _print_table(args.objects, payload["objects"], payload["macro"])

    if args.json_out:
        out = {"transform": args.transform, "transform_args": transform_kwargs, **payload}
        Path(args.json_out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - CLI boundary: fail loudly, clear message
        print(f"map_postprocess_eval: error: {error}", file=sys.stderr)
        sys.exit(1)
