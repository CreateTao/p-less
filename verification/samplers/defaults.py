from __future__ import annotations

from verification.samplers.temperature_only import TemperatureOnlyStrategy
from verification.samplers.top_k import TopKStrategy


class VLLMDefaultStrategy(TemperatureOnlyStrategy):
    """vLLM/SGLang default: top_p=1.0, top_k=-1, temperature=1.0.

    Equivalent to pure temperature sampling with no truncation.
    """

    def __init__(self):
        super().__init__(temperature=1.0)

    @property
    def name(self) -> str:
        return "vllm_default"

    @property
    def params(self) -> dict:
        return {"top_p": 1.0, "top_k": -1, "temperature": 1.0}


class HFDefaultStrategy(TopKStrategy):
    """HuggingFace generate() default: top_p=1.0, top_k=50, temperature=1.0.

    Only top_k=50 truncation is active (top_p=1.0 means no top-p truncation).
    """

    def __init__(self):
        super().__init__(top_k=50, temperature=1.0)

    @property
    def name(self) -> str:
        return "hf_default"

    @property
    def params(self) -> dict:
        return {"top_p": 1.0, "top_k": 50, "temperature": 1.0}