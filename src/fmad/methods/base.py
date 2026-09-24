"""Base class for anomaly detection methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Dict, Sequence
from pathlib import Path

import numpy as np
from fmad.protocol import MethodContext


@dataclass(frozen=True)
class ObjectResult:
    """Immutable result for a single object's anomaly detection run."""

    object_name: str
    anomaly_maps: Dict[str, np.ndarray]  # {sample_key: anomaly_map}
    anomaly_scores: Dict[str, float]  # {sample_key: score}
    time_memorybank: float  # seconds
    inference_times: Dict[str, float]  # {sample_key: seconds}
    metadata: Dict = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    """Model-owned output for one query; evaluation remains pipeline-owned."""

    score_map: np.ndarray
    image_score: float
    inference_seconds: float
    diagnostics: Dict = field(default_factory=dict)


class BaseMethod(ABC):
    """Abstract base for anomaly detection methods.

    Subclasses implement fit() and predict_batch(); orchestration is pipeline-owned.
    """

    def __init__(self, backbone, config: dict):
        self.backbone = backbone
        self.config = config
        self._training_log_sink = None

    def log_training(self, step: int | None = None, **metrics) -> None:
        """Emit JSON-compatible loss/progress values to the common training log."""
        if self._training_log_sink is not None:
            self._training_log_sink(step, metrics)

    def save_checkpoint(self, directory: Path) -> None:
        """Write fitted weights/state to the supplied empty checkpoint directory."""
        raise NotImplementedError(f"{type(self).__name__} does not support checkpoint saving")

    @abstractmethod
    def fit(self, support_paths: list[str], context: MethodContext) -> None:
        """Fit using only the selected normal images and immutable object context."""

    def predict(self, test_image_path: str) -> tuple:
        """Predict anomaly map for a single test image.

        Args:
            test_image_path: Path to the test image.

        Returns:
            (anomaly_map, anomaly_score): numpy array and float score.
        """
        raise NotImplementedError("implement predict() or predict_batch()")

    def predict_batch(self, test_image_paths: Sequence[str]) -> list[Prediction]:
        """Default batching for methods that implement single-image prediction."""
        predictions: list[Prediction] = []
        for image_path in test_image_paths:
            started = time.perf_counter()
            score_map, image_score = self.predict(image_path)
            predictions.append(
                Prediction(
                    score_map=np.asarray(score_map),
                    image_score=float(image_score),
                    inference_seconds=time.perf_counter() - started,
                )
            )
        return predictions

    def object_metadata(self) -> Dict:
        """Return compact model-specific provenance for the active object."""
        return {}
