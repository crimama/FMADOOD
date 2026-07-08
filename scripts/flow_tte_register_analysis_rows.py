# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
# pyright: reportMissingImports=false
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from flow_tte_register_analysis_types import (
    CONTEXT_SOURCES,
    ContextSource,
    FloatArray,
    SplitName,
    latent_volume_summary,
    summarize_good_bad_distances,
    summarize_list,
)


def build_context_rows(
    object_name: str,
    context_min: Dict[Tuple[ContextSource, SplitName], List[float]],
    context_mean: Dict[Tuple[ContextSource, SplitName], List[float]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for source in CONTEXT_SOURCES:
        min_summary = summarize_good_bad_distances(
            np.asarray(context_min.get((source, "good"), []), dtype=np.float32),
            np.asarray(context_min.get((source, "bad"), []), dtype=np.float32),
        )
        mean_summary = summarize_good_bad_distances(
            np.asarray(context_mean.get((source, "good"), []), dtype=np.float32),
            np.asarray(context_mean.get((source, "bad"), []), dtype=np.float32),
        )
        rows.append(
            {
                "object": object_name,
                "source": source,
                "good_count": str(min_summary.good_count),
                "bad_count": str(min_summary.bad_count),
                "min_good_mean": f"{min_summary.good_mean:.9f}",
                "min_bad_mean": f"{min_summary.bad_mean:.9f}",
                "min_delta_bad_good": f"{min_summary.delta_bad_good:.9f}",
                "mean_good_mean": f"{mean_summary.good_mean:.9f}",
                "mean_bad_mean": f"{mean_summary.bad_mean:.9f}",
                "mean_delta_bad_good": f"{mean_summary.delta_bad_good:.9f}",
            },
        )
    return rows


def build_retrieval_rows(
    object_name: str,
    top_m: int,
    retrieval: Dict[Tuple[ContextSource, SplitName], Dict[str, List[float]]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for source in CONTEXT_SOURCES:
        rows.extend(
            build_retrieval_row(object_name, source, split, top_m, retrieval)
            for split in ("good", "bad")
        )
    return rows


def build_retrieval_row(
    object_name: str,
    source: ContextSource,
    split: SplitName,
    top_m: int,
    retrieval: Dict[Tuple[ContextSource, SplitName], Dict[str, List[float]]],
) -> Dict[str, str]:
    bucket = retrieval.get((source, split), {})
    count, all_mean, _all_std, _all_p95 = summarize_list(bucket.get("all", []))
    _, routed_mean, _routed_std, _routed_p95 = summarize_list(bucket.get("routed", []))
    _, inflation_mean, _inflation_std, inflation_p95 = summarize_list(
        bucket.get("inflation", []),
    )
    _, retained_mean, _retained_std, _retained_p95 = summarize_list(bucket.get("retained", []))
    return {
        "object": object_name,
        "source": source,
        "split": split,
        "context_top_m": str(top_m),
        "sampled_patches": str(count),
        "nearest_retained_rate": f"{retained_mean:.9f}",
        "nearest_changed_rate": f"{1.0 - retained_mean:.9f}",
        "all_distance_mean": f"{all_mean:.9f}",
        "routed_distance_mean": f"{routed_mean:.9f}",
        "inflation_mean": f"{inflation_mean:.9f}",
        "inflation_p95": f"{inflation_p95:.9f}",
    }


def build_nf_rows(
    object_name: str,
    support_z: FloatArray,
    support_z_cond: FloatArray,
    nll_values: Dict[Tuple[str, SplitName], List[float]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for model_name, z_values in (
        ("unconditional", support_z),
        ("register_conditional", support_z_cond),
    ):
        rows.append(build_support_nf_row(object_name, model_name, z_values))
        rows.extend(
            build_test_nf_row(object_name, model_name, split, nll_values)
            for split in ("good", "bad")
        )
    return rows


def build_support_nf_row(object_name: str, model_name: str, z_values: FloatArray) -> Dict[str, str]:
    volume = latent_volume_summary(z_values)
    return {
        "object": object_name,
        "model": model_name,
        "split": "support",
        "sampled_patches": str(int(z_values.shape[0])),
        "nll_mean": "nan",
        "nll_std": "nan",
        "z_mean_variance": f"{volume.mean_variance:.9f}",
        "z_mean_log_variance": f"{volume.mean_log_variance:.9f}",
        "z_effective_rank": f"{volume.effective_rank:.9f}",
    }


def build_test_nf_row(
    object_name: str,
    model_name: str,
    split: SplitName,
    nll_values: Dict[Tuple[str, SplitName], List[float]],
) -> Dict[str, str]:
    count, mean, std, _p95 = summarize_list(nll_values.get((model_name, split), []))
    return {
        "object": object_name,
        "model": model_name,
        "split": split,
        "sampled_patches": str(count),
        "nll_mean": f"{mean:.9f}",
        "nll_std": f"{std:.9f}",
        "z_mean_variance": "nan",
        "z_mean_log_variance": "nan",
        "z_effective_rank": "nan",
    }
