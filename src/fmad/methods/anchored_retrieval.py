"""I3: choose reference identities in the frozen space, adapt their distance.

Search is chunked; no query-by-bank-by-channel tensor or persistent index/map
cache. Query and selected normal reference still receive adapter gradients.
"""
import torch

from fmad.methods.relational_tail import RelationalTail
from fmad.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register("anchored_retrieval")
class AnchoredRetrieval(RelationalTail):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.enabled = config.get("anchored_retrieval", True)
        if type(self.enabled) is not bool or not self.immediate_preservation:
            raise ValueError("anchored_retrieval requires a boolean switch and H1 controller")

    def score(self, features, grid, bank, indices=None, baseline=False):
        if not self.enabled or baseline or self.adapter is None:
            return super().score(features, grid, bank, indices, baseline)
        original = features if indices is None else features[indices]
        adapted = self.adapter(features)
        correction = self.metric(adapted, grid)
        if indices is not None:
            adapted, correction = adapted[indices], correction[indices]
        output = []
        for start in range(0, len(original), self.chunk):
            with torch.no_grad():
                selected = (original[start:start + self.chunk] @ bank[0].T).argmax(1)
            reference = self.adapter(bank[0][selected])
            query, change = adapted[start:start + self.chunk], correction[start:start + self.chunk]
            distance = 1 - (query * reference).sum(1)
            output.append(distance + .5 * (change * (query - reference).square()).sum(1))
        return torch.cat(output)

    def object_metadata(self):
        return {**super().object_metadata(), "anchored_retrieval": self.enabled,
                "reference_selection": "frozen_cosine" if self.enabled else "adapted_metric"}
