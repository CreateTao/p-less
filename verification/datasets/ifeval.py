from __future__ import annotations

from verification.datasets.base import DatasetHandler, DatasetItem


class IFEvalHandler(DatasetHandler):
    """IFEval instruction following evaluation dataset handler.

    Uses the official IFEval evaluation framework for strict/prompt accuracy.
    """

    def load(self, subset_fraction: float = 1.0) -> list[DatasetItem]:
        from datasets import load_dataset
        ds = load_dataset("ifeval", split="train")
        if subset_fraction < 1.0:
            ds = ds.shuffle(seed=42).select(range(int(len(ds) * subset_fraction)))
        return [
            DatasetItem(
                question_id=i,
                prompt_input=item["prompt"],
                ground_truth="",  # No single ground truth; multiple constraints
                metadata={
                    "instruction_id_list": item.get("instruction_id_list", []),
                    "kwargs": item.get("kwargs", []),
                },
            )
            for i, item in enumerate(ds)
        ]

    def format_prompt(self, item: DatasetItem, model_name: str) -> str:
        """IFEval prompts are direct instructions."""
        return item.prompt_input

    def extract_answer(self, generated_text: str, item: DatasetItem) -> str:
        """For IFEval, the full generated text is the answer."""
        return generated_text

    def evaluate(self, extracted_answer: str, ground_truth: str, item: DatasetItem) -> bool:
        """Evaluate using the official IFEval metric.

        Uses instruction_id_list and kwargs from metadata to check
        whether the generated text satisfies each constraint.
        """
        try:
            from ifeval import eval_instruction_following
            result = eval_instruction_following(
                prompt=item.prompt_input,
                response=extracted_answer,
                instruction_id_list=item.metadata.get("instruction_id_list", []),
                kwargs=item.metadata.get("kwargs", []),
            )
            # Strict accuracy: all constraints must be satisfied
            return result.get("strict_follow", False)
        except ImportError:
            # Fallback: basic constraint checking without official framework
            return _basic_constraint_check(extracted_answer, item.metadata)

    @property
    def dataset_name(self) -> str:
        return "ifeval"

    @property
    def metric_name(self) -> str:
        return "strict_accuracy"


def _basic_constraint_check(text: str, metadata: dict) -> bool:
    """Basic constraint checking when official IFEval framework is unavailable."""
    instruction_ids = metadata.get("instruction_id_list", [])
    kwargs_list = metadata.get("kwargs", [])

    for inst_id, kwargs in zip(instruction_ids, kwargs_list):
        if "length" in inst_id:
            # Check length constraint
            max_length = kwargs.get("length", 0)
            if len(text.split()) > max_length:
                return False
        elif "punctuation" in inst_id:
            # Check no punctuation
            import re
            if re.search(r"[.,!?;:]", text):
                return False

    return True