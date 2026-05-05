from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass
class SampleResult:
    """Per-sample evaluation result."""
    schema_version: str = SCHEMA_VERSION
    experiment_id: str = ""
    model_name: str = ""
    method_name: str = ""
    temperature: float = 1.0
    method_params: dict = field(default_factory=dict)
    dataset_name: str = ""
    question_id: int = 0
    prompt: str = ""
    generated_text: str = ""
    generated_tokens: int = 0
    correct: bool = False
    extracted_answer: str = ""
    ground_truth_answer: str = ""
    avg_candidate_set_size: float = 0.0
    avg_threshold_value: float = 0.0
    avg_sampling_time_ms: float = 0.0
    generation_time_s: float = 0.0
    seed: int = 42


@dataclass
class AggregatedResult:
    """Aggregated result for a (model, method, temperature, dataset) group."""
    schema_version: str = SCHEMA_VERSION
    experiment_id: str = ""
    model_name: str = ""
    method_name: str = ""
    temperature: float = 1.0
    method_params: dict = field(default_factory=dict)
    dataset_name: str = ""
    num_samples: int = 0
    accuracy: float = 0.0
    bootstrap_ci_lower: float = 0.0
    bootstrap_ci_upper: float = 0.0
    avg_generated_tokens: float = 0.0
    avg_candidate_set_size: float = 0.0
    avg_threshold_value: float = 0.0
    avg_sampling_time_ms: float = 0.0
    total_generation_time_s: float = 0.0
    seed: int = 42


@dataclass
class ComparisonResult:
    """Comparison between two methods on the same dataset/model."""
    schema_version: str = SCHEMA_VERSION
    method_a: str = ""
    method_b: str = ""
    dataset: str = ""
    model: str = ""
    temperature_a: float = 1.0
    temperature_b: float = 1.0
    paired_t_statistic: float = 0.0
    paired_t_pvalue: float = 0.0
    bootstrap_diff_ci_lower: float = 0.0
    bootstrap_diff_ci_upper: float = 0.0
    outcome: str = ""  # "Win", "Tie", "Lose"


@dataclass
class TOSTResult:
    """Two One-Sided Tests result for equivalence / non-degradation."""
    schema_version: str = SCHEMA_VERSION
    method_a: str = ""
    method_b: str = ""
    dataset: str = ""
    model: str = ""
    delta: float = 0.02
    lower_test_pvalue: float = 0.0
    upper_test_pvalue: float = 0.0
    is_equivalent: bool = False
    mean_diff: float = 0.0
    diff_ci_lower: float = 0.0
    diff_ci_upper: float = 0.0


@dataclass
class GridSearchResult:
    """Result of hyperparameter grid search for a baseline method."""
    schema_version: str = SCHEMA_VERSION
    method: str = ""
    dataset: str = ""
    model: str = ""
    best_params: dict = field(default_factory=dict)
    best_accuracy: float = 0.0
    all_configs_evaluated: list[dict] = field(default_factory=list)
    search_subset_fraction: float = 0.2