from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class SamplingMetrics:
    """Per-step metrics recorded by sampling strategies."""
    candidate_set_size: int = 0
    threshold_value: float = 0.0
    sampling_time_ms: float = 0.0
    top_token_prob: float = 0.0


class SamplingStrategy(ABC):
    """Abstract base class for all sampling strategies.

    Every strategy implements sample(logits) -> next_token_id.
    Temperature scaling and truncation are handled internally by each strategy.
    """

    def __init__(self, temperature: float = 1.0):
        self._temperature = temperature
        self._last_metrics = SamplingMetrics()
        self._accumulated_metrics: list[SamplingMetrics] = []

    @abstractmethod
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next token given logits.

        Args:
            logits: Shape (vocab_size,) or (batch_size, vocab_size)
        Returns:
            next_token_id: Shape (1,) or (batch_size, 1)
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @property
    @abstractmethod
    def params(self) -> dict:
        """Strategy parameters as dict, for JSON storage."""
        ...

    @property
    def temperature(self) -> float:
        return self._temperature

    def get_last_metrics(self) -> SamplingMetrics:
        return self._last_metrics

    def reset_metrics(self) -> None:
        self._accumulated_metrics = []

    def accumulate_metrics(self) -> None:
        self._accumulated_metrics.append(SamplingMetrics(
            candidate_set_size=self._last_metrics.candidate_set_size,
            threshold_value=self._last_metrics.threshold_value,
            sampling_time_ms=self._last_metrics.sampling_time_ms,
            top_token_prob=self._last_metrics.top_token_prob,
        ))

    def get_accumulated_metrics(self) -> list[SamplingMetrics]:
        return self._accumulated_metrics

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        if self._temperature == 1.0:
            return logits
        return logits / self._temperature

    def _record_metrics(
        self,
        threshold_value: float,
        candidate_set_size: int,
        top_token_prob: float,
        start_time: float,
    ) -> None:
        elapsed = (time.perf_counter() - start_time) * 1000
        self._last_metrics = SamplingMetrics(
            candidate_set_size=candidate_set_size,
            threshold_value=threshold_value,
            sampling_time_ms=elapsed,
            top_token_prob=top_token_prob,
        )