from __future__ import annotations

from scripts.combine_mvtecad1_hybrid_metrics import combine_metrics


def _metrics(image_offset: float, pixel_offset: float) -> dict:
    row = {
        "i_AUROC": 0.9 + image_offset,
        "i_AUPRC": 0.8 + image_offset,
        "image_AUROC": 0.9 + image_offset,
        "image_AP": 0.8 + image_offset,
        "p_AUROC": 0.7 + pixel_offset,
        "p_AUPRC": 0.6 + pixel_offset,
        "p_AUPRO": 0.5 + pixel_offset,
        "pixel_AUROC": 0.7 + pixel_offset,
        "pixel_AP": 0.6 + pixel_offset,
        "pixel_PRO": 0.5 + pixel_offset,
    }
    return {
        "dataset": "MVTec AD1 classic",
        "objects": ["bottle"],
        "per_object": {"bottle": dict(row)},
        **row,
    }


def test_hybrid_uses_raw_image_and_refined_pixel_metrics() -> None:
    raw = _metrics(0.01, 0.02)
    refined = _metrics(-0.03, 0.04)
    hybrid = combine_metrics(raw, refined)

    assert hybrid["i_AUROC"] == raw["i_AUROC"]
    assert hybrid["i_AUPRC"] == raw["i_AUPRC"]
    assert hybrid["p_AUROC"] == refined["p_AUROC"]
    assert hybrid["p_AUPRC"] == refined["p_AUPRC"]
    assert hybrid["p_AUPRO"] == refined["p_AUPRO"]
    assert hybrid["per_object"]["bottle"]["image_AUROC"] == raw["image_AUROC"]
    assert hybrid["per_object"]["bottle"]["pixel_PRO"] == refined["pixel_PRO"]
    assert hybrid["hybrid_contract"]["gt_used_by_refinement"] is False
