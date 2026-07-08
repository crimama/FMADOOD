# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
# pyright: reportMissingImports=false
from __future__ import annotations

import importlib
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from dinov3_backbone import (
    DINOv3Backbone,
    dinov3_hidden_state_index,
    dinov3_patch_token_start,
)
from flow_tte_mvtec_ad2_core import DatasetLike
from flow_tte_register_analysis_types import (
    ContextSet,
    FloatArray,
    ImageBundle,
    SupportBundle,
    TestItem,
)
from flow_tte_support import merge_layer_features, read_rgb


def build_dataset(data_root: Path, objects: Sequence[str]) -> DatasetLike:
    dataset_module = importlib.import_module("fmad.datasets.mvtec_ad2")
    dataset_cls = dataset_module.MVTecAD2Dataset
    return dataset_cls(
        data_root=str(data_root),
        config={"objects": list(objects), "preprocess": "no_mask_no_rotation"},
    )


def extract_bundle(backbone: DINOv3Backbone, image_path: Path) -> ImageBundle:
    image_tensor, grid_size = backbone.prepare_image(read_rgb(image_path))
    hidden_states = backbone._hidden_states(image_tensor)  # noqa: SLF001
    start = dinov3_patch_token_start(backbone._num_register_tokens)  # noqa: SLF001
    if start <= 1:
        message = "DINOv3 register analysis requires register tokens"
        raise RuntimeError(message)
    layer_features = [
        hidden_states[dinov3_hidden_state_index(layer)].squeeze(0)[start:].detach().cpu().numpy()
        for layer in backbone.feature_layers
    ]
    merged = merge_layer_features(layer_features, "layer_norm_mean")
    tokens = hidden_states[-1].squeeze(0).detach().cpu()
    cls = tokens[0].numpy().astype(np.float32, copy=False)
    register = tokens[1:start].mean(dim=0).numpy().astype(np.float32, copy=False)
    height, width = grid_size
    return ImageBundle(
        features=merged.reshape(height * width, merged.shape[-1]).astype(np.float32, copy=False),
        contexts=ContextSet(
            cls=cls,
            register=register,
            cls_register=np.concatenate([cls, register]).astype(np.float32, copy=False),
        ),
    )


def build_support(backbone: DINOv3Backbone, paths: Sequence[Path]) -> SupportBundle:
    features: List[FloatArray] = []
    register_patch_contexts: List[FloatArray] = []
    group_ids: List[np.ndarray] = []
    cls_contexts: List[FloatArray] = []
    register_contexts: List[FloatArray] = []
    cls_register_contexts: List[FloatArray] = []
    for group_id, path in enumerate(paths):
        bundle = extract_bundle(backbone, path)
        patch_count = int(bundle.features.shape[0])
        features.append(bundle.features)
        register_patch_contexts.append(
            np.broadcast_to(bundle.contexts.register, (patch_count, bundle.contexts.register.size))
            .copy()
            .astype(np.float32, copy=False),
        )
        group_ids.append(np.full(patch_count, group_id, dtype=np.int64))
        cls_contexts.append(bundle.contexts.cls)
        register_contexts.append(bundle.contexts.register)
        cls_register_contexts.append(bundle.contexts.cls_register)
    return SupportBundle(
        features=np.concatenate(features, axis=0).astype(np.float32, copy=False),
        register_patch_contexts=np.concatenate(register_patch_contexts, axis=0).astype(
            np.float32,
            copy=False,
        ),
        group_ids=np.concatenate(group_ids, axis=0).astype(np.int64, copy=False),
        contexts=ContextSet(
            cls=np.stack(cls_contexts, axis=0).astype(np.float32, copy=False),
            register=np.stack(register_contexts, axis=0).astype(np.float32, copy=False),
            cls_register=np.stack(cls_register_contexts, axis=0).astype(np.float32, copy=False),
        ),
    )


def stream_test_images(dataset: DatasetLike, object_name: str) -> Tuple[TestItem, ...]:
    test_images = dataset.get_test_images(object_name, split="test_public")
    items = [
        TestItem(anomaly_type=anomaly_type, path=Path(path))
        for anomaly_type, paths in test_images.items()
        for path in paths
    ]
    return tuple(sorted(items, key=lambda item: (item.path.name, item.anomaly_type)))
