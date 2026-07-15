# ─── How to run ───
# python3 scripts/run_superadd_parity.py --data-root /data/mvtec_ad_2 \
#   --category can --device cuda:0 --output-root /workspace/results/superadd \
#   --resource-protocol Pfull
"""Run the audited HF adaptation of official SuperADD on MVTec AD2 public data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TypedDict

import cv2
import numpy as np
import PIL
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from flow_tte.darc_backbone import (  # noqa: E402
    DINOv3EarlyExitAdapter,
    patch_torch_pytree_compat,
)
from flow_tte.superadd_bank import fit_disk_backed_banks  # noqa: E402
from flow_tte.superadd_inference import (  # noqa: E402
    FROZEN_CONFIG_SHA256,
    FROZEN_MODEL_ID,
    FROZEN_MODEL_REVISION,
    FROZEN_RESOLVED_CONFIG_SHA256,
    FROZEN_TRANSFORMERS_VERSION,
    FROZEN_WEIGHT_SHA256,
    ImageGridExtractor,
    audit_frozen_model,
    resolve_frozen_model_files,
    score_image,
    verify_frozen_model_files,
    verify_frozen_runtime,
)
from flow_tte.superadd_morphology import (  # noqa: E402
    MorphologyConfig,
    postprocess_binary,
)
from flow_tte.superadd_outputs import (  # noqa: E402
    CanonicalMapPaths,
    CategoryRunPlan,
    MapIdentity,
    canonical_map_paths,
    category_run,
    files_sha256,
    text_sha256,
    write_category_manifest,
    write_map_artifacts,
)
from flow_tte.superadd_parity import (  # noqa: E402
    CoresetConfig,
    ManifestContext,
    ModelProvenance,
    TrainPartition,
    build_parity_manifest,
    fixed_threshold,
    partition_training_paths,
)
from flow_tte.superadd_patching import (  # noqa: E402
    PatchConfig,
    PreprocessConfig,
)


@dataclass(frozen=True)
class RunConfig:
    data_root: Path
    category: str
    device: str
    output_root: Path
    resource_protocol: str
    support_manifest: Optional[Path]


@dataclass(frozen=True)
class PublicItem:
    image_path: Path
    split: str
    map_paths: CanonicalMapPaths


class MapRow(TypedDict):
    image_path: str
    raw_map: str
    raw_array_sha256: str
    binary_map: str
    binary_array_sha256: str


class CalibrationRow(TypedDict):
    image_path: str
    shape: Tuple[int, int]
    dtype: str
    sha256: str


class SuperADDRunnerError(RuntimeError):
    """Raised when CLI resources do not satisfy the selected protocol."""


def parse_args(argv: Sequence[str]) -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--resource-protocol",
        choices=("Pfull", "P16-official-native-split"),
        required=True,
    )
    parser.add_argument("--support-manifest")
    args = parser.parse_args(list(argv))
    manifest = Path(args.support_manifest) if args.support_manifest else None
    if args.resource_protocol == "P16-official-native-split" and manifest is None:
        parser.error("--support-manifest is required for P16-official-native-split")
    return RunConfig(
        data_root=Path(args.data_root),
        category=args.category,
        device=args.device,
        output_root=Path(args.output_root),
        resource_protocol=args.resource_protocol,
        support_manifest=manifest,
    )


def load_support_paths(config: RunConfig) -> Tuple[Path, ...]:
    train_root = config.data_root / config.category / "train" / "good"
    train_paths = tuple(sorted(train_root.glob("*.png")))
    if config.resource_protocol == "Pfull":
        if not train_paths:
            raise FileNotFoundError("Pfull found no train/good PNG images")
        return _validated_support_paths(train_paths, train_root, expected_count=None)
    manifest_path = config.support_manifest
    if manifest_path is None:
        raise SuperADDRunnerError("P16-official-native-split support manifest was not parsed")
    raw: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SuperADDRunnerError("support manifest must be a JSON object")
    selected = raw.get(config.category, raw.get("support_paths"))
    if isinstance(selected, dict):
        selected = selected.get("support_paths")
    if not isinstance(selected, list) or any(not isinstance(item, str) for item in selected):
        raise SuperADDRunnerError(
            "P16-official-native-split manifest must expose a category string-list",
        )
    paths = tuple(
        Path(item) if Path(item).is_absolute() else manifest_path.parent / item
        for item in selected
    )
    return _validated_support_paths(paths, train_root, expected_count=16)


def _validated_support_paths(
    paths: Sequence[Path],
    train_root: Path,
    expected_count: Optional[int],
) -> Tuple[Path, ...]:
    try:
        resolved_root = train_root.resolve(strict=True)
        resolved = tuple(path.resolve(strict=True) for path in paths)
    except OSError as error:
        raise FileNotFoundError("support path or train/good root is inaccessible") from error
    if expected_count is not None and len(resolved) != expected_count:
        message = f"P16-official-native-split requires exactly {expected_count} support paths"
        raise SuperADDRunnerError(message)
    identities = set()
    for path in resolved:
        if path.suffix.lower() != ".png" or not path.is_file():
            raise SuperADDRunnerError("support paths must be PNG files")
        try:
            path.relative_to(resolved_root)
        except ValueError as error:
            raise SuperADDRunnerError(
                "support paths must resolve beneath the selected category train/good",
            ) from error
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in identities:
            raise SuperADDRunnerError("support paths must identify unique physical files")
        identities.add(identity)
    return resolved


def discover_public_items(config: RunConfig) -> Tuple[PublicItem, ...]:
    entries = []
    identities = set()
    for split in ("good", "bad"):
        split_root = config.data_root / config.category / "test_public" / split
        for unresolved in sorted(split_root.glob("*.png")):
            path = unresolved.resolve(strict=True)
            try:
                path.relative_to(split_root.resolve(strict=True))
            except ValueError as error:
                raise SuperADDRunnerError("public images must remain inside their split") from error
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in identities:
                raise SuperADDRunnerError("public images must identify unique physical files")
            identities.add(identity)
            entries.append((split, path))
    items = tuple(
        PublicItem(
            path,
            split,
            canonical_map_paths(
                MapIdentity(config.output_root, config.category, split, path.stem),
            ),
        )
        for split, path in entries
    )
    if not items:
        raise FileNotFoundError("test_public/good and test_public/bad contain no PNG images")
    return items


def save_public_maps(
    items: Sequence[PublicItem],
    extractor: ImageGridExtractor,
    banks: Sequence[torch.Tensor],
    threshold: float,
) -> Tuple[List[MapRow], bool]:
    rows: List[MapRow] = []
    early_exit_flags = []
    counts = {split: sum(item.split == split for item in items) for split in ("good", "bad")}
    seen = {"good": 0, "bad": 0}
    for item in items:
        seen[item.split] += 1
        raw, used_early_exit = score_image(item.image_path, extractor, banks)
        binary = postprocess_binary(raw, threshold, MorphologyConfig())
        write_map_artifacts(item.map_paths, raw, binary)
        rows.append(
            {
                "image_path": str(item.image_path),
                "raw_map": str(item.map_paths.raw),
                "raw_array_sha256": _array_sha256(raw),
                "binary_map": str(item.map_paths.binary),
                "binary_array_sha256": _array_sha256(binary),
            },
        )
        early_exit_flags.append(used_early_exit)
        print(
            f"{item.split} {seen[item.split]}/{counts[item.split]} {item.image_path.name}",
            flush=True,
        )
    return rows, all(early_exit_flags)


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(contiguous.shape).encode("ascii") + b"\0")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def implementation_sha256() -> str:
    paths = (
        _REPO_ROOT / "src/flow_tte/superadd_bank.py",
        _REPO_ROOT / "src/flow_tte/superadd_inference.py",
        _REPO_ROOT / "src/flow_tte/superadd_morphology.py",
        _REPO_ROOT / "src/flow_tte/superadd_outputs.py",
        _REPO_ROOT / "src/flow_tte/superadd_parity.py",
        _REPO_ROOT / "src/flow_tte/superadd_patching.py",
        _REPO_ROOT / "src/flow_tte/darc_backbone.py",
        Path(__file__).resolve(),
    )
    return files_sha256(paths)


def _require_unchanged_implementation(expected_sha256: str) -> str:
    actual_sha256 = implementation_sha256()
    if actual_sha256 != expected_sha256:
        raise SuperADDRunnerError("producer implementation changed during execution")
    return actual_sha256


def prepare_run_plan(
    config: RunConfig,
    supports: Sequence[Path],
    partition: TrainPartition,
    items: Sequence[PublicItem],
    implementation_sha256: str,
) -> CategoryRunPlan:
    preprocess = PreprocessConfig()
    patching = PatchConfig()
    coreset = CoresetConfig()
    morphology = MorphologyConfig()
    spec = {
        "category": config.category,
        "code_sha256": implementation_sha256,
        "coreset": vars(coreset),
        "feature_layers": (7, 15, 23, 31),
        "model": {
            "config_sha256": FROZEN_CONFIG_SHA256,
            "model_id": FROZEN_MODEL_ID,
            "revision": FROZEN_MODEL_REVISION,
            "resolved_config_sha256": FROZEN_RESOLVED_CONFIG_SHA256,
            "transformers_version": FROZEN_TRANSFORMERS_VERSION,
            "weight_sha256": FROZEN_WEIGHT_SHA256,
        },
        "morphology": vars(morphology),
        "patching": vars(patching),
        "preprocess": vars(preprocess),
        "public_inputs": _input_inventory(
            tuple(item.image_path for item in items),
            config.data_root,
        ),
        "prototype_paths": tuple(str(path) for path in partition.prototypes),
        "resource_protocol": config.resource_protocol,
        "seed": 42,
        "support_paths": tuple(str(path) for path in supports),
        "support_inputs": _input_inventory(supports, config.data_root),
        "support_manifest": _optional_input(config.support_manifest, config.data_root),
        "threshold": {"factor": 1.421, "percentile": 95.0},
        "threshold_paths": tuple(str(path) for path in partition.threshold),
        "runtime": _runtime_identity(config.device),
    }
    serialized = json.dumps(spec, separators=(",", ":"), sort_keys=True)
    return CategoryRunPlan(
        config.output_root,
        config.category,
        tuple(item.map_paths for item in items),
        text_sha256(serialized),
    )


def _input_inventory(paths: Sequence[Path], data_root: Path) -> Tuple[object, ...]:
    root = data_root.resolve()
    inventory = []
    for path in paths:
        resolved = path.resolve(strict=True)
        try:
            label = str(resolved.relative_to(root))
        except ValueError:
            label = str(resolved)
        inventory.append(
            {
                "label": label,
                "sha256": _file_sha256(resolved),
                "size": resolved.stat().st_size,
            },
        )
    return tuple(inventory)


def _optional_input(path: Optional[Path], data_root: Path) -> Optional[object]:
    if path is None:
        return None
    return _input_inventory((path,), data_root)[0]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_identity(device: str) -> object:
    return {
        "cublas_workspace_config": ":4096:8",
        "cuda_runtime": torch.version.cuda,
        "deterministic_algorithms": True,
        "device": str(torch.device(device)),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": str(torch.__version__),
    }


def _rediscover_run_plan(config: RunConfig, implementation_sha: str) -> CategoryRunPlan:
    supports = load_support_paths(config)
    partition = partition_training_paths(supports, threshold_fraction=8)
    return prepare_run_plan(
        config,
        supports,
        partition,
        discover_public_items(config),
        implementation_sha,
    )


def _require_cuda_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda":
        raise SuperADDRunnerError("first-run SuperADD execution requires a CUDA device")
    return device


def configure_determinism() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(mode=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def main(argv: Sequence[str]) -> int:
    config = parse_args(argv)
    configure_determinism()
    supports = load_support_paths(config)
    partition = partition_training_paths(supports, threshold_fraction=8)
    if not partition.prototypes or not partition.threshold:
        raise SuperADDRunnerError("official modulo-eight split requires both partitions")
    items = discover_public_items(config)
    implementation_sha = implementation_sha256()
    plan = prepare_run_plan(config, supports, partition, items, implementation_sha)
    with category_run(plan) as run:
        if run.resumed:
            print(json.dumps({"category": config.category, "status": "resumed_complete"}))
            return 0
        device = _require_cuda_device(config.device)
        rng = np.random.RandomState(42)
        patch_torch_pytree_compat()
        verify_frozen_runtime()
        frozen_files = resolve_frozen_model_files()
        from transformers import AutoConfig, AutoModel  # noqa: PLC0415

        with verify_frozen_model_files(frozen_files) as frozen_verification:
            model_config = AutoConfig.from_pretrained(
                frozen_verification.load_directory,
                _commit_hash=FROZEN_MODEL_REVISION,
                local_files_only=True,
                trust_remote_code=False,
            )
            model_config.name_or_path = FROZEN_MODEL_ID
            model = AutoModel.from_pretrained(
                frozen_verification.load_directory,
                config=model_config,
                use_safetensors=True,
                trust_remote_code=False,
                local_files_only=True,
            )
            model.config.name_or_path = FROZEN_MODEL_ID
            model = model.eval().to(device)
            audit = audit_frozen_model(model, frozen_files, frozen_verification)
        preprocess = PreprocessConfig()
        patching = PatchConfig()
        fit_extractors = tuple(
            ImageGridExtractor(
                DINOv3EarlyExitAdapter(
                    model,
                    layers=(layer,),
                    normalize_final=False,
                    normalize_features=False,
                ),
                device,
                preprocess,
                patching,
            )
            for layer in (7, 15, 23, 31)
        )
        banks, fit_early_exit = fit_disk_backed_banks(
            partition.prototypes,
            fit_extractors,
            device,
            rng,
            config.output_root / "categories" / config.category,
        )
        adapter = DINOv3EarlyExitAdapter(
            model,
            layers=(7, 15, 23, 31),
            normalize_final=False,
            normalize_features=False,
        )
        extractor = ImageGridExtractor(adapter, device, preprocess, patching)
        calibration_maps = []
        calibration_rows: List[CalibrationRow] = []
        calibration_flags = []
        for path in partition.threshold:
            score, used_early_exit = score_image(path, extractor, banks)
            calibration_maps.append(score)
            calibration_flags.append(used_early_exit)
            calibration_rows.append(
                {
                    "image_path": str(path),
                    "shape": (int(score.shape[0]), int(score.shape[1])),
                    "dtype": str(score.dtype),
                    "sha256": _array_sha256(score),
                },
            )
        threshold = fixed_threshold(calibration_maps)
        rows, public_early_exit = save_public_maps(items, extractor, banks, threshold)
        used_early_exit = fit_early_exit and all(calibration_flags) and public_early_exit
        final_implementation_sha = _require_unchanged_implementation(implementation_sha)
        final_plan = _rediscover_run_plan(config, final_implementation_sha)
        if final_plan != plan:
            raise SuperADDRunnerError("run inputs or runtime changed during execution")
        provenance = ModelProvenance(
            audit.model_id,
            audit.revision,
            audit.model_class,
            audit.patch_size,
            audit.depth,
            audit.register_count,
            audit.config_sha256,
            audit.resolved_config_sha256,
            audit.weight_sha256,
            audit.transformers_version,
        )
        manifest = build_parity_manifest(
            ManifestContext(
                config.category,
                config.resource_protocol,
                tuple(supports),
                partition,
                provenance,
                implementation_sha,
                used_early_exit,
            ),
        )
        payload = {
            **manifest,
            "run_spec_sha256": plan.spec_sha256,
            "threshold": {
                "value": threshold,
                "percentile": 95.0,
                "factor": 1.421,
                "calibration_maps": calibration_rows,
            },
            "map_count": len(rows),
            "map_index": rows,
            "distance": "euclidean_unsquared_divided_by_channel_count",
            "raw_map_dtype": "float32",
            "raw_map_layout": "anomaly_maps/<category>/test/<good|bad>/<stem>.tiff",
        }
        write_category_manifest(run, json.dumps(payload, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "category": config.category,
                "completion": str(
                    config.output_root / "categories" / config.category / "completion.json",
                ),
                "manifest": str(
                    config.output_root / "categories" / config.category / "manifest.json",
                ),
                "status": "complete",
                "threshold": threshold,
            },
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
