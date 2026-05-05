from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from verification.storage.schema import (
    SampleResult, AggregatedResult,
    TOSTResult, GridSearchResult,
)


def save_results(result, output_dir: str, filename: str = "") -> str:
    """Save a result dataclass as JSON file.

    Args:
        result: Any result dataclass (SampleResult, AggregatedResult, etc.)
        output_dir: Directory path for the output file
        filename: Optional filename; auto-generated if empty
    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    data = asdict(result)

    if not filename:
        filename = f"{result.schema_version}_{result.experiment_id}.json"

    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    return filepath


def load_results(filepath: str) -> dict:
    """Load a JSON result file as dict."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_sample_result(result: SampleResult, base_dir: str) -> str:
    """Save a per-sample result to organized subdirectory."""
    subdir = os.path.join(
        base_dir,
        result.dataset_name,
        result.model_name,
        result.method_name,
        f"t{result.temperature}",
    )
    filename = f"q_{result.question_id:04d}.json"
    return save_results(result, subdir, filename)


def save_aggregated_result(result: AggregatedResult, base_dir: str) -> str:
    """Save an aggregated result."""
    subdir = os.path.join(
        base_dir,
        result.dataset_name,
        result.model_name,
    )
    filename = f"aggregated_{result.method_name}_t{result.temperature}.json"
    return save_results(result, subdir, filename)


def load_sample_results(base_dir: str, dataset: str, model: str, method: str, temperature: float) -> list[dict]:
    """Load all per-sample results for a given config."""
    pattern_dir = os.path.join(
        base_dir, dataset, model, method, f"t{temperature}"
    )
    results = []
    if not os.path.exists(pattern_dir):
        return results

    for filename in sorted(os.listdir(pattern_dir)):
        if filename.startswith("q_") and filename.endswith(".json"):
            filepath = os.path.join(pattern_dir, filename)
            results.append(load_results(filepath))

    return results