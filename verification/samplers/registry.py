from __future__ import annotations

from verification.samplers.p_less import PLessStrategy, PLessNormStrategy
from verification.samplers.greedy import GreedyStrategy
from verification.samplers.top_p import TopPStrategy
from verification.samplers.top_k import TopKStrategy
from verification.samplers.min_p import MinPStrategy
from verification.samplers.epsilon import EpsilonSamplingStrategy
from verification.samplers.eta import EtaSamplingStrategy
from verification.samplers.temperature_only import TemperatureOnlyStrategy
from verification.samplers.defaults import VLLMDefaultStrategy, HFDefaultStrategy
from verification.samplers.base import SamplingStrategy


class StrategyRegistry:
    """Factory mapping strategy names to SamplingStrategy instances."""

    _registry: dict[str, type[SamplingStrategy]] = {
        "p_less": PLessStrategy,
        "p_less_norm": PLessNormStrategy,
        "greedy": GreedyStrategy,
        "top_p": TopPStrategy,
        "top_k": TopKStrategy,
        "min_p": MinPStrategy,
        "epsilon": EpsilonSamplingStrategy,
        "eta": EtaSamplingStrategy,
        "temperature_only": TemperatureOnlyStrategy,
        "vllm_default": VLLMDefaultStrategy,
        "hf_default": HFDefaultStrategy,
    }

    @classmethod
    def create(cls, name: str, params: dict | None = None, temperature: float = 1.0) -> SamplingStrategy:
        """Create a strategy from config parameters.

        Args:
            name: Strategy name key (e.g. 'top_p', 'p_less')
            params: Strategy-specific params (e.g. {'top_p': 0.9})
            temperature: Temperature for sampling
        Returns:
            Initialized SamplingStrategy instance
        """
        params = params or {}
        strategy_cls = cls._registry[name]

        # Greedy/VLLM/HF defaults don't accept temperature or params
        if name in ("greedy", "vllm_default", "hf_default"):
            return strategy_cls()

        # p-less/p-less-norm don't have method-specific params, only temperature
        if name in ("p_less", "p_less_norm"):
            return strategy_cls(temperature=temperature)

        # Other strategies accept both params and temperature
        return strategy_cls(**params, temperature=temperature)

    @classmethod
    def available_strategies(cls) -> list[str]:
        return list(cls._registry.keys())