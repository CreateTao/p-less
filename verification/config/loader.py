from __future__ import annotations

import yaml

from verification.config.schemas import (
    ModelConfig, MethodConfig, DatasetConfig, ExperimentConfig,
)


def load_config(path: str) -> ExperimentConfig:
    """Load experiment config from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    models = [ModelConfig(**m) for m in data.get("models", [])]
    methods = [MethodConfig(**m) for m in data.get("methods", [])]
    dataset = DatasetConfig(**data.get("dataset", {}))

    return ExperimentConfig(
        experiment_id=data.get("experiment_id", ""),
        environment=data.get("environment", "cpu"),
        models=models,
        methods=methods,
        dataset=dataset,
        p_less_temperatures=data.get("p_less_temperatures", [0.3, 0.7, 1.0, 1.5]),
        seed=data.get("seed", 42),
        max_tokens=data.get("max_tokens", 512),
    )