from __future__ import annotations

import yaml

from verification.config.schemas import (
    ModelConfig, MethodConfig, DatasetConfig, ExperimentConfig,
)


def load_config(path: str) -> ExperimentConfig:
    """Load experiment config from YAML file.

    Supports both singular and plural keys:
    - `model` (single dict) or `models` (list of dicts)
    - `dataset` (single dict) or `datasets` (list of dicts, uses first)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Support both `model` (single) and `models` (list)
    raw_models = data.get("models", [])
    if not raw_models and "model" in data:
        raw_models = [data["model"]]
    models = [ModelConfig(**m) for m in raw_models]

    # Support both `dataset` (single) and `datasets` (list, uses first)
    raw_dataset = data.get("dataset", None)
    if raw_dataset is None:
        raw_datasets = data.get("datasets", [])
        if raw_datasets:
            # datasets can be list of dicts or list of bare strings
            first = raw_datasets[0]
            if isinstance(first, str):
                raw_dataset = {"name": first, "handler_class": first}
            else:
                raw_dataset = first
    if raw_dataset is None:
        raw_dataset = {}
    dataset = DatasetConfig(**raw_dataset)

    # Parse methods (list of dicts or list of bare strings)
    raw_methods = data.get("methods", [])
    parsed_methods = []
    for m in raw_methods:
        if isinstance(m, str):
            parsed_methods.append(MethodConfig(name=m, strategy_class=m))
        else:
            parsed_methods.append(MethodConfig(**m))
    methods = parsed_methods

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
