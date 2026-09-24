"""I1: preserve normal bulk while releasing the teacher's high-score tail.

The selection uses only frozen scores of normal support, never query labels.
This is a hypothesis about conflicting objectives, not a correctness gate.
"""
import torch

from fmad.methods.relational_tail import RelationalTail, relational_score_loss
from fmad.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register("selective_preservation")
class SelectivePreservation(RelationalTail):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.enabled = config.get("selective_preservation", True)
        if type(self.enabled) is not bool or not self.immediate_preservation:
            raise ValueError("selective_preservation requires a boolean switch and H1 controller")
        self.preserved_fraction = 1.

    def preservation_terms(self, clean, teacher, keep):
        if not self.enabled:
            return super().preservation_terms(clean, teacher, keep)
        teacher = teacher.detach()
        # Include ties: a flat teacher must not arbitrarily release pixels.
        threshold = torch.quantile(teacher.flatten(), 1. - self.tail_fraction)
        bulk = teacher <= threshold
        self.preserved_fraction = float(bulk.float().mean())
        selected_clean, selected_teacher = clean[bulk], teacher[bulk]
        return ((selected_clean - selected_teacher).square().mean(),
                relational_score_loss(selected_clean, selected_teacher))

    def training_diagnostics(self):
        return {**super().training_diagnostics(), "preserved_fraction": self.preserved_fraction}

    def object_metadata(self):
        return {**super().object_metadata(), "selective_preservation": self.enabled,
                "preservation_selection": "teacher_normal_bulk" if self.enabled else "full_map"}
