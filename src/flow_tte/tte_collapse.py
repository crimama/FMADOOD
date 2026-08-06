"""Mechanism decomposition for the 2026-07-07 `flow_tte_budget125` collapse.

The historical component ablation reported `-0.027955` pAUROC and `-0.017479`
F1 for reservoir memory expansion at `expansion_budget=1.25` and closed the
test-time branch with "static no-TTE memory is essential".  That verdict pooled
four distinct mechanisms into one number:

`M1` defect self-confirmation -- absorbed defect patches become bank rows and
suppress every later occurrence of the same defect.
`M2` calibration invalidation -- the bank grows while the support-LOO distance
mean/std stays frozen at the clean support fit.
`M3` stream non-stationarity -- early images are scored against a small bank and
late images against a contaminated one, which breaks the single pooled
threshold that oracle F1 depends on.
`M4` shift adaptation -- the intended benefit of absorbing genuine normal
patches from unseen acquisition conditions.

This module holds the pure arm logic so the runner stays I/O only.  Arms reuse
the production `ReservoirMemory`/`TorchMemoryBank`/`ScoreCalibration` objects so
the replication arm is bit-comparable with the historical implementation rather
than a re-derivation of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
import torch

from flow_tte.config import ScoreConfig
from flow_tte.memory import ReservoirMemory
from flow_tte.scoring import ScoreCalibration

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

CANDIDATE_POLICIES = ("all", "gt_normal", "good_images")
PROTOCOLS = ("static", "online", "two_pass")


class TteCollapseError(ValueError):
    """Raised when an arm specification or its inputs are inconsistent."""


@dataclass(frozen=True)
class ArmSpec:
    """One decomposition arm.

    `candidate_policy` controls which patches may enter memory:
    `all` replicates the historical rule (NLL gate only), `gt_normal` removes
    only ground-truth defect patches, and `good_images` removes whole defective
    images.  The latter two use labels and are therefore diagnostic upper
    bounds, never deployable policies.
    """

    arm_id: str
    budget: float
    candidate_policy: str = "all"
    protocol: str = "online"
    recalibrate: bool = False
    order_seed: int = 0
    note: str = ""

    def __post_init__(self) -> None:
        if self.budget < 1.0:
            message = f"{self.arm_id}: budget must be at least 1.0"
            raise TteCollapseError(message)
        if self.candidate_policy not in CANDIDATE_POLICIES:
            message = f"{self.arm_id}: unknown candidate policy"
            raise TteCollapseError(message)
        if self.protocol not in PROTOCOLS:
            message = f"{self.arm_id}: unknown protocol"
            raise TteCollapseError(message)
        if self.protocol == "static" and self.budget != 1.0:
            message = f"{self.arm_id}: static protocol requires budget 1.0"
            raise TteCollapseError(message)
        if self.order_seed < 0:
            message = f"{self.arm_id}: order seed must be non-negative"
            raise TteCollapseError(message)

    @property
    def uses_labels(self) -> bool:
        return self.candidate_policy != "all"


@dataclass(frozen=True)
class ImageRecord:
    """Cached per-image state; latents are already flow-transformed."""

    relative_path: str
    condition: str
    is_good: bool
    latents: torch.Tensor
    nll: torch.Tensor
    defect: BoolArray
    density_penalty: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.latents.ndim != 2:
            message = f"{self.relative_path}: latents must be 2D"
            raise TteCollapseError(message)
        rows = int(self.latents.shape[0])
        if int(self.nll.shape[0]) != rows or self.defect.shape[0] != rows:
            message = f"{self.relative_path}: patch count mismatch"
            raise TteCollapseError(message)
        if self.density_penalty is not None and int(self.density_penalty.shape[0]) != rows:
            message = f"{self.relative_path}: density penalty count mismatch"
            raise TteCollapseError(message)
        if self.is_good and bool(self.defect.any()):
            message = f"{self.relative_path}: good image carries defect labels"
            raise TteCollapseError(message)


@dataclass(frozen=True)
class ArmResult:
    arm_id: str
    scores: tuple[FloatArray, ...]
    absorbed_rows: int
    contaminated_rows: int
    self_hit_rate: float
    calibration_mean: float
    calibration_std: float


def stream_order(n_images: int, order_seed: int) -> tuple[int, ...]:
    """Frozen filename order for seed 0, deterministic permutation otherwise."""
    if order_seed == 0:
        return tuple(range(n_images))
    rng = np.random.default_rng(order_seed)
    return tuple(int(index) for index in rng.permutation(n_images))


def candidate_mask(
    record: ImageRecord,
    density_threshold: float,
    policy: str,
) -> torch.Tensor:
    """Reproduce the production selection gate, then apply the arm policy."""
    selected = record.nll <= density_threshold
    if policy == "all":
        return selected
    if policy == "good_images":
        if record.is_good:
            return selected
        return torch.zeros_like(selected)
    defect = torch.as_tensor(record.defect, device=selected.device)
    return selected & ~defect


def _origin_labels(record: ImageRecord, mask: torch.Tensor) -> torch.Tensor:
    """Defect indicator column travelling through an identically seeded reservoir."""
    defect = torch.as_tensor(
        record.defect.astype(np.float32),
        device=record.latents.device,
        dtype=record.latents.dtype,
    )
    return defect[mask].reshape(-1, 1)


def _fit_bank_calibration(
    features: torch.Tensor,
    config: ScoreConfig,
) -> ScoreCalibration:
    return ScoreCalibration.fit(features, config, None)


def _nearest(
    memory: ReservoirMemory,
    query: torch.Tensor,
    config: ScoreConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = memory.bank.query(
        query,
        k=1,
        chunk_size=config.query_chunk_size,
        squared=config.use_squared_distance,
    )
    return result.distances[:, 0], result.indices[:, 0]


def run_arm(  # noqa: PLR0913
    spec: ArmSpec,
    support_latents: torch.Tensor,
    records: Sequence[ImageRecord],
    density_threshold: float,
    calibration: ScoreCalibration,
    config: ScoreConfig,
    absorb_seed: int = 0,
    density_weight: float = 0.0,
) -> ArmResult:
    """Score every image under one arm and report its memory-provenance stats.

    Online arms score an image against the bank state that precedes it, exactly
    as `FlowTTE.score_then_expand` does.  Two-pass arms absorb the whole stream
    first and then rescore everything against the final bank, which removes the
    non-stationarity term `M3` while keeping contamination `M1` intact.
    """
    if not records:
        message = f"{spec.arm_id}: no image records"
        raise TteCollapseError(message)
    support_rows = int(support_latents.shape[0])
    memory = ReservoirMemory(
        m0_features=support_latents,
        budget=spec.budget,
        random_seed=absorb_seed,
    )
    origin = ReservoirMemory(
        m0_features=torch.zeros(
            support_rows,
            1,
            device=support_latents.device,
            dtype=support_latents.dtype,
        ),
        budget=spec.budget,
        random_seed=absorb_seed,
    )
    order = stream_order(len(records), spec.order_seed)
    scores: list[Optional[FloatArray]] = [None] * len(records)

    for position in order:
        record = records[position]
        if spec.protocol == "online":
            scores[position] = _score_image(record, memory, calibration, config, density_weight)
        if spec.protocol == "static":
            continue
        mask = candidate_mask(record, density_threshold, spec.candidate_policy)
        if not bool(mask.any()):
            continue
        memory.absorb(record.latents[mask])
        origin.absorb(_origin_labels(record, mask))

    active = calibration
    if spec.recalibrate:
        active = _fit_bank_calibration(memory.bank.features, config)
    if spec.protocol in ("static", "two_pass"):
        for position in range(len(records)):
            scores[position] = _score_image(
                records[position], memory, active, config, density_weight
            )

    resolved = tuple(_require_scores(spec, value) for value in scores)
    contaminated = int(origin.buffer.sum().detach().cpu()) if origin.buffer.numel() else 0
    return ArmResult(
        arm_id=spec.arm_id,
        scores=resolved,
        absorbed_rows=int(memory.buffer.shape[0]),
        contaminated_rows=contaminated,
        self_hit_rate=_self_hit_rate(records, memory, config, support_rows),
        calibration_mean=float(active.distance_mean),
        calibration_std=float(active.distance_std),
    )


def _require_scores(spec: ArmSpec, value: Optional[FloatArray]) -> FloatArray:
    if value is None:
        message = f"{spec.arm_id}: an image was never scored"
        raise TteCollapseError(message)
    return value


def _score_image(
    record: ImageRecord,
    memory: ReservoirMemory,
    calibration: ScoreCalibration,
    config: ScoreConfig,
    density_weight: float = 0.0,
) -> FloatArray:
    distances, _ = _nearest(memory, record.latents, config)
    score = calibration.normalize_distance(distances)
    if density_weight > 0.0:
        if record.density_penalty is None:
            message = f"{record.relative_path}: density weighting needs a density penalty"
            raise TteCollapseError(message)
        score = score + density_weight * record.density_penalty
    return score.detach().cpu().numpy().astype(np.float32, copy=False)


def _self_hit_rate(
    records: Sequence[ImageRecord],
    memory: ReservoirMemory,
    config: ScoreConfig,
    support_rows: int,
) -> float:
    """Share of defect patches whose final 1-NN is an absorbed test row."""
    if memory.buffer.shape[0] == 0:
        return 0.0
    hits = 0
    total = 0
    for record in records:
        if not bool(record.defect.any()):
            continue
        defect = torch.as_tensor(record.defect, device=record.latents.device)
        _, indices = _nearest(memory, record.latents[defect], config)
        hits += int((indices >= support_rows).sum().detach().cpu())
        total += int(defect.sum().detach().cpu())
    if total == 0:
        return 0.0
    return float(hits) / float(total)
