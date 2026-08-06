from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import torch
from typing_extensions import final

from flow_tte.config import FlowTTEConfig
from flow_tte.memory import ReservoirMemory
from flow_tte.scoring import (
    ScoreCalibration,
    ScoreInputs,
    ScoreResult,
    image_top_percent_mean,
    score_flow_memory,
)
from flow_tte.tensors import (
    FeatureArray,
    as_2d_float_tensor,
    as_patch_batch,
    resolve_device,
    to_numpy,
)
from flow_tte.trainer import FlowDensityEstimator, FlowTrainingStats
from flow_tte.waff import (
    WaffScorer,
    score_waff_components_tensor,
    source_excluded_nearest_distance,
)


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
        self.waff_scorer: Optional[WaffScorer] = None

    def fit(
        self,
        support_features: FeatureArray,
        support_contexts: Optional[FeatureArray] = None,
        memory_contexts: Optional[FeatureArray] = None,
        memory_selector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> FlowTrainingStats:
        self.waff_scorer = None
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
        if memory_selector is not None:
            if m0_contexts is not None:
                raise RuntimeError("FlowTTE memory selection does not support memory contexts")
            m0_z = memory_selector(m0_z)
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

    def set_waff_scorer(self, scorer: WaffScorer) -> None:
        """Attach a support-fitted static scorer without changing flow or memory state."""
        weight = np.asarray(scorer.weight)
        if weight.shape != (2,) or not bool(np.all(np.isfinite(weight))):
            raise ValueError("WAFF scorer must contain two finite weights")
        _ = self._require_estimator()
        _ = self._require_memory()
        self.waff_scorer = scorer

    def refresh_static_memory(self, support_features: FeatureArray) -> None:
        """Rebuild the static bank after an offline flow refinement.

        Distance and NLL calibration intentionally remain frozen at the clean-only
        fit so an offline representation treatment cannot hide score rescaling by
        recalibrating itself.
        """
        estimator = self._require_estimator()
        _ = self._require_score_calibration()
        if self.config.flow.condition_mode != "none":
            raise RuntimeError("offline static-memory refresh supports unconditioned flow only")
        latents = estimator.transform(support_features)
        self.memory = ReservoirMemory(
            m0_features=latents,
            budget=self.config.expansion.budget,
            random_seed=self.config.expansion.random_seed,
        )

    def clone_offline(self) -> "FlowTTE":
        """Clone fitted offline state without pickling the memory RNG generator."""
        estimator = self._require_estimator()
        memory = self._require_memory()
        calibration = self._require_score_calibration()
        if memory.buffer.shape[0] != 0:
            raise RuntimeError("offline clone requires an unexpanded static memory")
        candidate = FlowTTE(self.config)
        candidate.estimator = copy.deepcopy(estimator)
        candidate.memory = ReservoirMemory(
            m0_features=memory.m0_features,
            budget=self.config.expansion.budget,
            random_seed=self.config.expansion.random_seed,
            m0_contexts=memory.m0_contexts,
        )
        candidate.training_stats = self.training_stats
        candidate.score_calibration = calibration
        candidate.waff_scorer = self.waff_scorer
        return candidate

    def factorial_score_components(
        self,
        features: FeatureArray,
        contexts: Optional[FeatureArray] = None,
        *,
        query_source_ids: Optional[npt.ArrayLike] = None,
        bank_source_ids: Optional[npt.ArrayLike] = None,
    ) -> np.ndarray:
        """Compute the exact H+ distance/NLL components without mutating memory."""
        estimator = self._require_estimator()
        memory = self._require_memory()
        calibration = self._require_score_calibration()
        device = resolve_device(self.config.device)
        batch = as_patch_batch(features, device)
        condition_contexts = _optional_contexts(
            contexts,
            device,
            n_rows=int(batch.flat_features.shape[0]),
            name="contexts",
        )
        if self.config.flow.condition_mode == "context" and condition_contexts is None:
            raise RuntimeError("FlowTTE component scoring requires condition contexts")
        evaluation = estimator.evaluate(features, contexts=condition_contexts)
        if (query_source_ids is None) != (bank_source_ids is None):
            raise ValueError("query_source_ids and bank_source_ids must be provided together")
        if query_source_ids is None:
            query = memory.bank.query(
                evaluation.z,
                k=1,
                chunk_size=self.config.score.query_chunk_size,
                squared=self.config.score.use_squared_distance,
            )
            distances = query.distances[:, 0]
        else:
            if memory.bank.contexts is not None:
                raise RuntimeError("source-excluded WAFF scoring does not support memory contexts")
            query_ids = torch.as_tensor(query_source_ids, device=device).reshape(-1)
            bank_ids = torch.as_tensor(bank_source_ids, device=device).reshape(-1)
            distances = source_excluded_nearest_distance(
                evaluation.z,
                memory.bank.features,
                query_ids,
                bank_ids,
                chunk_size=self.config.score.query_chunk_size,
                squared=self.config.score.use_squared_distance,
            )
        components = torch.stack(
            (
                calibration.normalize_distance(distances),
                estimator.density_penalty(evaluation.nll),
            ),
            dim=1,
        )
        return to_numpy(components)

    def transform_features(
        self,
        features: FeatureArray,
        contexts: Optional[FeatureArray] = None,
    ) -> np.ndarray:
        estimator = self._require_estimator()
        evaluation = estimator.evaluate(features, contexts=contexts)
        return to_numpy(evaluation.batch.restore_features(evaluation.z))

    def latent_residuals(
        self,
        features: FeatureArray,
        contexts: Optional[FeatureArray] = None,
    ) -> np.ndarray:
        estimator = self._require_estimator()
        memory = self._require_memory()
        device = resolve_device(self.config.device)
        batch = as_patch_batch(features, device)
        condition_contexts = _optional_contexts(
            contexts,
            device,
            n_rows=int(batch.flat_features.shape[0]),
            name="contexts",
        )
        evaluation = estimator.evaluate(features, contexts=condition_contexts)
        nearest = memory.bank.query(
            evaluation.z,
            k=1,
            chunk_size=self.config.score.query_chunk_size,
        ).indices[:, 0]
        residuals = evaluation.z - memory.bank.features[nearest]
        return to_numpy(residuals)

    def score_static_shift_projected(
        self,
        batch_features: FeatureArray,
        shift_basis: np.ndarray,
        strength: float,
        batch_contexts: Optional[FeatureArray] = None,
        memory_contexts: Optional[FeatureArray] = None,
    ) -> BatchResult:
        estimator = self._require_estimator()
        memory = self._require_memory()
        calibration = self._require_score_calibration()
        device = resolve_device(self.config.device)
        batch = as_patch_batch(batch_features, device)
        shared_contexts = _optional_contexts(
            batch_contexts,
            device,
            n_rows=int(batch.flat_features.shape[0]),
            name="batch_contexts",
        )
        if self.config.flow.condition_mode == "context" and shared_contexts is None:
            raise RuntimeError("FlowTTE score requires batch condition contexts")
        evaluation = estimator.evaluate(batch_features, contexts=shared_contexts)
        nearest = memory.bank.query(
            evaluation.z,
            k=1,
            chunk_size=self.config.score.query_chunk_size,
        ).indices[:, 0]
        residuals = evaluation.z - memory.bank.features[nearest]
        basis = torch.as_tensor(shift_basis, device=device, dtype=evaluation.z.dtype)
        aligned_z = evaluation.z - strength * (residuals @ basis) @ basis.T
        density_penalty = estimator.density_penalty(evaluation.nll)
        query_contexts = shared_contexts if memory_contexts is None else _optional_contexts(
            memory_contexts,
            device,
            n_rows=int(batch.flat_features.shape[0]),
            name="memory_contexts",
        )
        score = score_flow_memory(
            inputs=ScoreInputs(
                query_z=aligned_z,
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
        score = self._rescore_with_waff(
            score,
            evaluation.batch.image_indices,
            evaluation.batch.n_images,
        )
        selected = evaluation.nll <= estimator.density_threshold
        memory_size = memory.bank.size()
        return BatchResult(
            patch_scores=to_numpy(evaluation.batch.restore(score.patch_scores)),
            nll=to_numpy(evaluation.batch.restore(evaluation.nll)),
            selected_mask=to_numpy(evaluation.batch.restore(selected)).astype(bool),
            image_scores=to_numpy(score.image_scores),
            image_score=score.image_score,
            selected_count=int(selected.sum().detach().cpu()),
            memory_size_before=memory_size,
            memory_size_after=memory_size,
        )

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
        score = self._rescore_with_waff(
            score,
            evaluation.batch.image_indices,
            evaluation.batch.n_images,
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

    def _rescore_with_waff(
        self,
        score: ScoreResult,
        image_indices: torch.Tensor,
        n_images: int,
    ) -> ScoreResult:
        if self.waff_scorer is None:
            return score
        components = torch.stack((score.distance_scores, score.density_penalty), dim=1)
        patch_scores = score_waff_components_tensor(self.waff_scorer, components)
        image_scores = image_top_percent_mean(
            patch_scores,
            image_indices,
            n_images,
            self.config.score.top_percent,
        )
        return ScoreResult(
            patch_scores=patch_scores,
            image_scores=image_scores,
            image_score=float(image_scores.max().detach().cpu()),
            distances=score.distances,
            distance_scores=score.distance_scores,
            density_penalty=score.density_penalty,
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
