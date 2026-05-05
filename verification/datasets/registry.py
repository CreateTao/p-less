from __future__ import annotations

from verification.datasets.gsm8k import GSM8KHandler
from verification.datasets.csqa import CSQAHandler
from verification.datasets.qasc import QASCHandler
from verification.datasets.humaneval import HumanEvalHandler
from verification.datasets.bfcl import BFCLHandler
from verification.datasets.ifeval import IFEvalHandler
from verification.datasets.gpqa import GPQAHandler
from verification.datasets.writing_prompts import WritingPromptsHandler
from verification.datasets.mbpp import MBPPHandler
from verification.datasets.base import DatasetHandler


class DatasetRegistry:
    """Registry mapping dataset names to handler classes."""

    _registry: dict[str, type[DatasetHandler]] = {
        "gsm8k": GSM8KHandler,
        "csqa": CSQAHandler,
        "qasc": QASCHandler,
        "humaneval": HumanEvalHandler,
        "bfcl": BFCLHandler,
        "ifeval": IFEvalHandler,
        "gpqa": GPQAHandler,
        "writing_prompts": WritingPromptsHandler,
        "mbpp": MBPPHandler,
    }

    @classmethod
    def create(cls, name: str) -> DatasetHandler:
        """Create a dataset handler by name."""
        handler_cls = cls._registry[name]
        return handler_cls()

    @classmethod
    def available_datasets(cls) -> list[str]:
        return list(cls._registry.keys())