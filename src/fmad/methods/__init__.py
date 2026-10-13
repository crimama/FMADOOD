"""Method implementations.

Concrete methods may depend on optional native packages.  Keep the package
import lightweight so shared contracts such as ``Prediction`` remain usable in
CPU-only tooling and tests.
"""

from __future__ import annotations


def register_methods(method_name: str | None = None) -> None:
    """Import method modules so their decorators populate the registry.

    ``register_methods()`` retains the historical all-method behavior used by
    existing callers and tests.  Runtime callers can pass the selected public
    method name to avoid importing optional dependencies for unrelated methods.
    """

    if method_name is None:
        from fmad.methods import PSMA_MR  # noqa: F401
        from fmad.methods import superad  # noqa: F401
    elif method_name == "PSMA_MR":
        from fmad.methods import PSMA_MR  # noqa: F401
    elif method_name == "contextual_score":
        from fmad.methods import contextual_score  # noqa: F401
    elif method_name == "bcma":
        from fmad.methods import bcma  # noqa: F401
    elif method_name == "tail_contrast":
        from fmad.methods import tail_contrast  # noqa: F401
    elif method_name == "guarded_metric":
        from fmad.methods import guarded_metric  # noqa: F401
    elif method_name == "relational_tail":
        from fmad.methods import relational_tail  # noqa: F401
    elif method_name == "selective_preservation":
        from fmad.methods import selective_preservation  # noqa: F401
    elif method_name == "paired_margin":
        from fmad.methods import paired_margin  # noqa: F401
    elif method_name == "anchored_retrieval":
        from fmad.methods import anchored_retrieval  # noqa: F401
    elif method_name == "tail_risk":
        from fmad.methods import tail_risk  # noqa: F401
    elif method_name == "boundary_synthetic":
        from fmad.methods import boundary_synthetic  # noqa: F401
    elif method_name == "student_boundary":
        from fmad.methods import student_boundary  # noqa: F401
    elif method_name == "projected_update":
        from fmad.methods import projected_update  # noqa: F401
    elif method_name == "operating_region":
        from fmad.methods import operating_region  # noqa: F401
    elif method_name == "shift_restore":
        from fmad.methods import shift_restore  # noqa: F401
    elif method_name == "transfer_update":
        from fmad.methods import transfer_update  # noqa: F401
    elif method_name == "family_balanced_paired":
        from fmad.methods import family_balanced_paired  # noqa: F401
    elif method_name == "reference_matching":
        from fmad.methods import reference_matching  # noqa: F401
    elif method_name == "reference_matching_fast":
        from fmad.methods import reference_matching_fast  # noqa: F401
    elif method_name == "region_recovery":
        from fmad.methods import region_recovery  # noqa: F401
    elif method_name == "fusion_factorial":
        from fmad.methods import fusion_factorial  # noqa: F401
    elif method_name == "local_relation":
        from fmad.methods import local_relation  # noqa: F401
    elif method_name == "coherent_matching":
        from fmad.methods import coherent_matching  # noqa: F401
    elif method_name == "tail_aware_fusion":
        from fmad.methods import tail_aware_fusion  # noqa: F401
    elif method_name == "cross_image_reference":
        from fmad.methods import cross_image_reference  # noqa: F401
    elif method_name == "query_reference":
        from fmad.methods import query_reference  # noqa: F401
    elif method_name == "spatial_confidence":
        from fmad.methods import spatial_confidence  # noqa: F401
    elif method_name == "protected_geometry":
        from fmad.methods import protected_geometry  # noqa: F401
    elif method_name == "reference_consensus":
        from fmad.methods import reference_consensus  # noqa: F401
    elif method_name == "mode_reference":
        from fmad.methods import mode_reference  # noqa: F401
    elif method_name == "validated_reference":
        from fmad.methods import validated_reference  # noqa: F401
    elif method_name == "correspondence_guard":
        from fmad.methods import correspondence_guard  # noqa: F401
    elif method_name == "masked_prediction":
        from fmad.methods import masked_prediction  # noqa: F401
    elif method_name == "reverse_distillation":
        from fmad.methods import reverse_distillation  # noqa: F401
    elif method_name in {"superad", "rn_fmlk"}:
        from fmad.methods import superad  # noqa: F401
    elif method_name == "patchcore":
        from fmad.methods import patchcore  # noqa: F401
    elif method_name == "fastrecon":
        from fmad.methods import fastrecon  # noqa: F401
    elif method_name == "fastrecon_paper":
        from fmad.methods import fastrecon_paper  # noqa: F401
    elif method_name == "regad":
        from fmad.methods import regad  # noqa: F401
    elif method_name == "regad_official":
        from fmad.methods import regad_official  # noqa: F401
    elif method_name == "anomalydino":
        from fmad.methods import anomalydino  # noqa: F401
    elif method_name == "subspacead":
        from fmad.methods import subspacead  # noqa: F401
    elif method_name == "rad":
        from fmad.methods import rad  # noqa: F401
