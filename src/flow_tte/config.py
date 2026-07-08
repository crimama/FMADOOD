from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import override


@dataclass(frozen=True)
class ConfigError(ValueError):
    field: str
    reason: str

    @override
    def __str__(self) -> str:
        return f"Invalid {self.field}: {self.reason}"


@dataclass(frozen=True)
class FlowConfig:
    n_coupling_layers: int = 4
    hidden_multiplier: int = 2
    clamp: float = 1.9
    transform_mode: str = "flow"
    condition_mode: str = "none"
    n_epochs: int = 30
    lr: float = 2e-4
    batch_size: int = 512
    seed: int = 0
    tail_weight: float = 0.3
    tail_top_k_ratio: float = 0.05
    lambda_logdet: float = 1e-3
    standardize: bool = True

    def __post_init__(self) -> None:
        if self.transform_mode not in ("flow", "identity"):
            raise ConfigError("transform_mode", "must be 'flow' or 'identity'")
        if self.condition_mode not in ("none", "context"):
            raise ConfigError("condition_mode", "must be 'none' or 'context'")
        if self.n_coupling_layers <= 0:
            raise ConfigError("n_coupling_layers", "must be positive")
        if self.hidden_multiplier <= 0:
            raise ConfigError("hidden_multiplier", "must be positive")
        if self.n_epochs <= 0:
            raise ConfigError("n_epochs", "must be positive")
        if self.batch_size <= 0:
            raise ConfigError("batch_size", "must be positive")
        if not 0.0 <= self.tail_weight <= 1.0:
            raise ConfigError("tail_weight", "must be in [0, 1]")
        if not 0.0 < self.tail_top_k_ratio <= 1.0:
            raise ConfigError("tail_top_k_ratio", "must be in (0, 1]")


@dataclass(frozen=True)
class ExpansionConfig:
    budget: float = 1.5
    density_quantile: float = 0.90
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.budget < 1.0:
            raise ConfigError("budget", "must be at least 1.0")
        if not 0.0 < self.density_quantile < 1.0:
            raise ConfigError("density_quantile", "must be in (0, 1)")


@dataclass(frozen=True)
class ScoreConfig:
    score_mode: str = "latent_distance"
    distance_weight: float = 1.0
    density_weight: float = 0.25
    context_mode: str = "none"
    context_weight: float = 0.0
    context_top_m: int = 1
    top_percent: float = 0.01
    query_chunk_size: int = 8192
    use_squared_distance: bool = False

    def __post_init__(self) -> None:
        if self.score_mode not in ("latent_distance", "nf_nll"):
            raise ConfigError("score_mode", "must be 'latent_distance' or 'nf_nll'")
        if self.context_mode not in ("none", "soft_penalty", "top_m"):
            raise ConfigError("context_mode", "must be 'none', 'soft_penalty', or 'top_m'")
        if self.distance_weight < 0.0:
            raise ConfigError("distance_weight", "must be non-negative")
        if self.density_weight < 0.0:
            raise ConfigError("density_weight", "must be non-negative")
        if self.context_weight < 0.0:
            raise ConfigError("context_weight", "must be non-negative")
        if self.context_mode == "none" and self.context_weight != 0.0:
            raise ConfigError("context_weight", "must be 0.0 when context_mode is 'none'")
        if self.context_top_m <= 0:
            raise ConfigError("context_top_m", "must be positive")
        if not 0.0 < self.top_percent <= 1.0:
            raise ConfigError("top_percent", "must be in (0, 1]")
        if self.query_chunk_size <= 0:
            raise ConfigError("query_chunk_size", "must be positive")


@dataclass(frozen=True)
class FlowTTEConfig:
    flow: FlowConfig = FlowConfig()
    expansion: ExpansionConfig = ExpansionConfig()
    score: ScoreConfig = ScoreConfig()
    device: str = "cuda"

    @classmethod
    def for_quick_probe(cls) -> "FlowTTEConfig":
        return cls(
            flow=FlowConfig(n_coupling_layers=2, n_epochs=3, batch_size=64, seed=0),
            expansion=ExpansionConfig(budget=1.25, density_quantile=0.90),
            score=ScoreConfig(density_weight=0.25, top_percent=0.05, query_chunk_size=1024),
            device="cpu",
        )
