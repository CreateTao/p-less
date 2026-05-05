from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DatasetItem:
    """Single item from a dataset."""
    question_id: int
    prompt_input: str
    ground_truth: str
    metadata: dict = field(default_factory=dict)


class DatasetHandler(ABC):
    """Abstract handler for a specific dataset.

    Each dataset requires custom logic for:
    - Loading data
    - Formatting prompts
    - Extracting answers from generated text
    - Evaluating correctness
    """

    @abstractmethod
    def load(self, subset_fraction: float = 1.0) -> list[DatasetItem]:
        """Load dataset items. subset_fraction < 1.0 for GPU search phase."""
        ...

    @abstractmethod
    def format_prompt(self, item: DatasetItem, model_name: str) -> str:
        """Format a dataset item as a prompt string for the model."""
        ...

    @abstractmethod
    def extract_answer(self, generated_text: str, item: DatasetItem) -> str:
        """Extract the predicted answer from generated text."""
        ...

    @abstractmethod
    def evaluate(self, extracted_answer: str, ground_truth: str, item: DatasetItem) -> bool:
        """Return True if extracted answer matches ground truth."""
        ...

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Dataset identifier."""
        ...

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Primary metric name."""
        ...