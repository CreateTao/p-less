from __future__ import annotations

from dataclasses import dataclass

from verification.samplers.base import SamplingMetrics


@dataclass
class GenerationResult:
    """Output of a single generation run."""
    generated_text: str
    generated_token_ids: list[int]
    num_tokens: int
    per_step_metrics: list[SamplingMetrics] | None
    generation_time_s: float
    seed: int