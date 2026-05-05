from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str
    hf_id: str
    size_label: str = ""
    memory_estimate_gb: float = 0.0
    device: str = "cpu"
    quantization: str | None = None


@dataclass
class MethodConfig:
    name: str
    strategy_class: str
    params: dict = field(default_factory=dict)
    temperature: float | None = None
    group: str = "B"  # "A" for framework defaults, "B" for recommended configs


@dataclass
class DatasetConfig:
    name: str
    handler_class: str
    num_samples: int = 0
    metric: str = "accuracy"


@dataclass
class ExperimentConfig:
    experiment_id: str
    environment: str = "cpu"  # "cpu" or "gpu"
    models: list[ModelConfig] = field(default_factory=list)
    methods: list[MethodConfig] = field(default_factory=list)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    p_less_temperatures: list[float] = field(default_factory=lambda: [0.3, 0.7, 1.0, 1.5])
    seed: int = 42
    max_tokens: int = 512


@dataclass
class GridSearchConfig:
    method_name: str
    strategy_class: str
    param_grid: dict = field(default_factory=dict)
    subset_fraction: float = 0.2
    temperatures: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 1.0])