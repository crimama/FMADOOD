from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from typing_extensions import final

from flow_tte.config import FlowTTEConfig
from flow_tte.memory import ReservoirMemory
from flow_tte.scoring import ScoreCalibration, ScoreInputs, score_flow_memory
from flow_tte.tensors import (
    FeatureArray,
    as_2d_float_tensor,
    as_patch_batch,
    resolve_device,
    to_numpy,
)
from flow_tte.trainer import FlowDensityEstimator, FlowTrainingStats


@dataclass(frozen=True)
class BatchResult:
    patch_scores: npt.NDArray[np.float32]
    nll: npt.NDArray[np.float32]
    selected_mask: npt.NDArray[np.bool_]
    image_scores: npt.NDArray[np.float32]
    image_score: float
    selected_count: int
    memory_size_before: int
    memory_size_after: int


@final
class FlowTTE:
    def __init__(self, config: Optional[FlowTTEConfig] = None) -> None:
        self.config: FlowTTEConfig = config or FlowTTEConfig()
        self.estimator: Optional[FlowDensityEstimator] = None
        self.memory: Optional[ReservoirMemory] = None
        self.training_stats: Optional[FlowTrainingStats] = None
        self.score_calibration: Optional[ScoreCalibration] = None

    def fit(
        self,
        support_features: FeatureArray,
        support_contexts: Optional[FeatureArray] = None,
        memory_contexts: Optional[FeatureArray] = None,
    ) -> FlowTrainingStats:
        device = resolve_device(self.config.device)
        support = as_patch_batch(support_features, device)
        shared_contexts = _optional_contexts(
            support_contexts,
            device,
            n_rows=int(support.flat_features.shape[0]),
            name="support_contexts",
        )
        m0_contexts = shared_contexts
        if memory_contexts is not None:
            m0_contexts = _optional_contexts(
                memory_contexts,
                device,
                n_rows=int(support.flat_features.shape[0]),
                name="memory_contexts",
            )
        condition_contexts = (
            shared_contexts if self.config.flow.condition_mode == "context" else None
        )
        if self.config.flow.condition_mode == "context" and condition_contexts is None:
            raise RuntimeError("FlowTTE fit requires support condition contexts")
        condition_dim = 0 if condition_contexts is None else int(condition_contexts.shape[1])
        estimator = FlowDensityEstimator(
            dim=int(support.flat_features.shape[1]),
            config=self.config.flow,
            device=self.config.device,
            condition_dim=condition_dim,
        )
        stats = estimator.fit(
            support_features,
            self.config.expansion.density_quantile,
            contexts=condition_contexts,
        )
        m0_z = estimator.transform(support_features, contexts=condition_contexts)
        self.memory = ReservoirMemory(
            m0_features=m0_z,
            budget=self.config.expansion.budget,
            random_seed=self.config.expansion.random_seed,
            m0_contexts=m0_contexts,
        )
        self.score_calibration = ScoreCalibration.fit(m0_z, self.config.score, m0_contexts)
        self.estimator = estimator
        self.training_stats = stats
        return stats

    def transform_features(
        self,
        features: FeatureArray,
        contexts: Optional[FeatureArray] = None,
    ) -> np.ndarray:
        estimator = self._require_estimator()
        evaluation = estimator.evaluate(features, contexts=contexts)
        return to_numpy(evaluation.batch.restore_features(evaluation.z))

    def score_then_expand(
        self,
        batch_features: FeatureArray,
        batch_contexts: Optional[FeatureArray] = None,
        memory_contexts: Optional[FeatureArray] = None,
    ) -> BatchResult:
        return self._score(
            batch_features,
            expand=True,
            batch_contexts=batch_contexts,
            memory_contexts=memory_contexts,
        )

    def score_static(
        self,
        batch_features: FeatureArray,
        batch_contexts: Optional[FeatureArray] = None,
        memory_contexts: Optional[FeatureArray] = None,
    ) -> BatchResult:
        return self._score(
            batch_features,
            expand=False,
            batch_contexts=batch_contexts,
            memory_contexts=memory_contexts,
        )

    def _score(
        self,
        batch_features: FeatureArray,
        expand: bool,
        batch_contexts: Optional[FeatureArray] = None,
        memory_contexts: Optional[FeatureArray] = None,
    ) -> BatchResult:
        estimator = self._require_estimator()
        memory = self._require_memory()
        calibration = self._require_score_calibration()
        memory_size_before = memory.bank.size()

        device = resolve_device(self.config.device)
        batch = as_patch_batch(batch_features, device)
        shared_contexts = _optional_contexts(
            batch_contexts,
            device,
            n_rows=int(batch.flat_features.shape[0]),
            name="batch_contexts",
        )
        query_contexts = shared_contexts
        if memory_contexts is not None:
            query_contexts = _optional_contexts(
                memory_contexts,
                device,
                n_rows=int(batch.flat_features.shape[0]),
                name="memory_contexts",
            )
        condition_contexts = (
            shared_contexts if self.config.flow.condition_mode == "context" else None
        )
        if self.config.flow.condition_mode == "context" and condition_contexts is None:
            raise RuntimeError("FlowTTE score requires batch condition contexts")
        evaluation = estimator.evaluate(batch_features, contexts=condition_contexts)
        density_penalty = estimator.density_penalty(evaluation.nll)
        score = score_flow_memory(
            inputs=ScoreInputs(
                query_z=evaluation.z,
                nll=evaluation.nll,
                nll_penalty=density_penalty,
                image_indices=evaluation.batch.image_indices,
                n_images=evaluation.batch.n_images,
                query_contexts=query_contexts,
            ),
            bank=memory.bank,
            config=self.config.score,
            calibration=calibration,
        )

        selected = evaluation.nll <= estimator.density_threshold
        if expand:
            selected_contexts = None if query_contexts is None else query_contexts[selected]
            memory.absorb(evaluation.z[selected], candidate_contexts=selected_contexts)
        return BatchResult(
            patch_scores=to_numpy(evaluation.batch.restore(score.patch_scores)),
            nll=to_numpy(evaluation.batch.restore(evaluation.nll)),
            selected_mask=to_numpy(evaluation.batch.restore(selected)).astype(bool),
            image_scores=to_numpy(score.image_scores),
            image_score=score.image_score,
            selected_count=int(selected.sum().detach().cpu()),
            memory_size_before=memory_size_before,
            memory_size_after=memory.bank.size(),
        )

    def _require_estimator(self) -> FlowDensityEstimator:
        if self.estimator is None:
            raise RuntimeError("FlowTTE is not fitted")
        return self.estimator

    def _require_memory(self) -> ReservoirMemory:
        if self.memory is None:
            raise RuntimeError("FlowTTE is not fitted")
        return self.memory

    def _require_score_calibration(self) -> ScoreCalibration:
        if self.score_calibration is None:
            raise RuntimeError("FlowTTE is not fitted")
        return self.score_calibration


def _optional_contexts(
    contexts: Optional[FeatureArray],
    device: torch.device,
    n_rows: int,
    name: str,
) -> Optional[torch.Tensor]:
    if contexts is None:
        return None
    tensor = as_2d_float_tensor(contexts, device)
    if tensor.shape[0] != n_rows:
        message = f"{name} row count must match patch features"
        raise RuntimeError(message)
    return tensor
