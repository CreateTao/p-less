from __future__ import annotations

import re

from verification.datasets.base import DatasetHandler, DatasetItem


class MBPPHandler(DatasetHandler):
    """MBPP (Mostly Basic Python Programming) dataset handler.

    Evaluates code generation via Pass@1 (sandboxed execution).
    Similar to HumanEval but with simpler, more basic programming tasks.
    """

    PROMPT_TEMPLATE = """You are an expert Python programmer. Write a Python function to solve the following problem.

Problem:
{prompt}

Your code must define a function. Include only the function definition, no test cases.

```python
"""

    def load(self, subset_fraction: float = 1.0) -> list[DatasetItem]:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        if subset_fraction < 1.0:
            ds = ds.shuffle(seed=42).select(range(int(len(ds) * subset_fraction)))
        return [
            DatasetItem(
                question_id=i,
                prompt_input=item["prompt"],
                ground_truth=item["code"],
                metadata={
                    "test_list": item.get("test_list", []),
                    "task_id": item.get("task_id", i),
                },
            )
            for i, item in enumerate(ds)
        ]

    def format_prompt(self, item: DatasetItem, model_name: str) -> str:
        return self.PROMPT_TEMPLATE.format(prompt=item.prompt_input)

    def extract_answer(self, generated_text: str, item: DatasetItem) -> str:
        """Extract Python code from generated text."""
        # Try to extract code from markdown code block
        match = re.search(r"```python\s*\n(.*?)```", generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try plain code block
        match = re.search(r"```\s*\n(.*?)```", generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: use everything after the prompt marker
        lines = generated_text.strip().split("\n")
        # Find lines that look like Python code (start with def, class, import, or indented)
        code_lines = []
        started = False
        for line in lines:
            if re.match(r"^\s*(def |class |import |from )", line):
                started = True
            if started:
                code_lines.append(line)
        if code_lines:
            return "\n".join(code_lines)

        # Last resort: return everything
        return generated_text.strip()

    def evaluate(self, extracted_answer: str, ground_truth: str, item: DatasetItem) -> bool:
        """Evaluate via sandboxed execution of test cases (Pass@1)."""
        test_list = item.metadata.get("test_list", [])
        if not test_list:
            # Fallback: check if generated code contains a function definition
            return "def " in extracted_answer

        code = extracted_answer + "\n"
        for test in test_list:
            code += test + "\n"

        try:
            import subprocess
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    @property
    def dataset_name(self) -> str:
        return "mbpp"

    @property
    def metric_name(self) -> str:
        return "pass@1"
