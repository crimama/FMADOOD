"""Main pipeline — orchestrates method, dataset, backbone, evaluation, results."""

import os
import time
from typing import Dict, List

from fmad.methods.base import ObjectResult
from fmad.evaluation.metrics import Evaluator
from fmad.results import ResultsManager


# Dataset name mapping for evaluation (src/post_eval expects these names)
_EVAL_DATASET_NAMES = {
    "mvtec_ad2": "MVTec_AD_2",
    "mvtec": "MVTec",
    "visa": "VisA",
}


def run_pipeline(method, dataset, backbone, config: dict,
                 output_dir: str) -> Dict:
    """Run the full anomaly detection pipeline.

    Args:
        method: Initialized BaseMethod instance.
        dataset: Initialized BaseDataset instance.
        backbone: Initialized BaseBackbone instance.
        config: Full configuration dict.
        output_dir: Base output directory.

    Returns:
        Dict with evaluation metrics.
    """
    seed = config.get("seed", 0)
    method_name = config["method"]
    dataset_name = config["dataset"]
    backbone_name = config["backbone"]

    # Results manager
    results_mgr = ResultsManager(
        base_dir=output_dir,
        method=method_name,
        dataset=dataset_name,
        backbone=backbone_name,
        seed=seed,
    )
    results_mgr.save_config(config)
    run_dir = results_mgr.run_dir

    # Run per-object
    objects = dataset.get_objects()
    object_results: List[ObjectResult] = []

    print(f"\n{'='*60}")
    print(f"  Method: {method_name} | Dataset: {dataset_name} | Backbone: {backbone_name}")
    print(f"  Seed: {seed} | Objects: {len(objects)}")
    print(f"  Output: {run_dir}")
    print(f"{'='*60}\n")

    for obj_info in objects:
        print(f"--- {obj_info.name} (resolution={obj_info.resolution}, "
              f"masking={obj_info.masking}, rotation={obj_info.rotation}) ---")

        t0 = time.time()
        result = method.run_object(
            dataset=dataset,
            object_name=obj_info.name,
            seed=seed,
            output_dir=run_dir,
            save_maps=True,
        )
        elapsed = time.time() - t0
        object_results.append(result)

        n_samples = len(result.inference_times)
        mean_time = sum(result.inference_times.values()) / max(n_samples, 1)
        print(f"    {n_samples} samples, mean inference: {mean_time:.3f} s/sample, "
              f"total: {elapsed:.1f}s")

    # Save timing CSV
    results_mgr.save_object_times(object_results)

    # Evaluate
    print(f"\n{'='*60}")
    print("  Evaluating...")
    print(f"{'='*60}")

    eval_dataset_name = _EVAL_DATASET_NAMES.get(dataset_name, dataset_name)
    evaluated_objects = [obj.name for obj in objects]
    evaluator = Evaluator(config.get("evaluation", {}))
    metrics = evaluator.evaluate_run(
        dataset_name=eval_dataset_name,
        data_root=dataset.data_root,
        anomaly_maps_dir=os.path.join(run_dir, "anomaly_maps"),
        output_dir=run_dir,
        seed=seed,
        objects=evaluated_objects,
    )

    # Save & accumulate
    results_mgr.save_metrics(metrics)
    results_mgr.append_summary(metrics, object_results)

    # Print summary
    print(f"\n{'='*60}")
    print("  Results")
    print(f"{'='*60}")
    print(f"  {'Object':<15} {'seg_AUROC':>10} {'seg_F1':>10} {'best_thre':>10}")
    print(f"  {'-'*45}")
    for obj_name, obj_m in metrics.items():
        if not isinstance(obj_m, dict) or "seg_AUROC" not in obj_m:
            continue
        print(f"  {obj_name:<15} {obj_m['seg_AUROC']:>10.4f} {obj_m['seg_F1']:>10.4f} "
              f"{obj_m.get('best_thre', 0):>10.4f}")

    mean_auroc = metrics.get("mean_segmentation_au_roc", 0)
    mean_f1 = metrics.get("mean_segmentation_f1", 0)
    print(f"  {'-'*45}")
    print(f"  {'MEAN':<15} {mean_auroc:>10.4f} {mean_f1:>10.4f}")
    print(f"\n  Summary appended to: {results_mgr.summary_path}")

    return metrics
