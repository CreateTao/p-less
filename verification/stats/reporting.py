from __future__ import annotations

import os

import numpy as np


def generate_accuracy_table(
    results: dict,
    group: str = "A",
) -> str:
    """Generate markdown accuracy table.

    Args:
        results: Dict of {dataset: {model: {method: {temperature: accuracy}}}}
        group: "A" for framework defaults, "B" for recommended configs
    Returns:
        Markdown table string
    """
    lines = []

    if group == "A":
        header = "| Dataset | Model | Greedy | vLLM默认(T=1) | HF默认(T=1) | p-less(T=0.7) | p-less(T=1) | p-less提升 |"
        sep = "|---------|-------|--------|---------------|-------------|----------------|-------------|------------|"
    else:
        header = "| Dataset | Model | top-p(0.9) | top-p(0.95) | top-k(40) | top-k(10) | min-p(0.05) | p-less | p-less-norm | W/T/L总计 |"
        sep = "|---------|-------|------------|-------------|-----------|-----------|-------------|--------|-------------|------------|"

    lines.append(header)
    lines.append(sep)

    for dataset, models in results.items():
        for model, methods in models.items():
            row_values = []
            for method_name, temps in methods.items():
                # Use first temperature value
                for temp, acc in temps.items():
                    row_values.append(f"{acc:.1f}%")
            row = f"| {dataset} | {model} | " + " | ".join(row_values) + " |"
            lines.append(row)

    return "\n".join(lines)


def generate_wtl_table(summary) -> str:
    """Generate Win/Tie/Lose summary table."""
    lines = [
        "| Method | Win | Tie | Lose | Total | Win+Tie Rate |",
        "|--------|-----|-----|------|-------|-------------|",
    ]

    # Group by baseline method
    methods = {}
    for comp in summary.comparisons:
        key = comp.method_b
        if key not in methods:
            methods[key] = {"win": 0, "tie": 0, "lose": 0}
        methods[key][comp.outcome.lower()] += 1

    for method, counts in sorted(methods.items()):
        total = counts["win"] + counts["tie"] + counts["lose"]
        rate = (counts["win"] + counts["tie"]) / total if total > 0 else 0
        lines.append(
            f"| {method} | {counts['win']} | {counts['tie']} | {counts['lose']} | {total} | {rate:.1%} |"
        )

    # Aggregate row
    lines.append(
        f"| **Total** | {summary.total_win} | {summary.total_tie} | {summary.total_loss} "
        f"| {summary.total_comparisons} | {summary.win_tie_rate:.1%} |"
    )

    return "\n".join(lines)


def plot_temperature_robustness(
    results: dict,
    output_dir: str,
) -> str:
    """Generate temperature robustness plot.

    Args:
        results: Dict of {method: {temperature: accuracy}}
        output_dir: Directory to save plot
    Returns:
        Path to saved plot
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, temps in results.items():
        temperatures = sorted(temps.keys())
        accuracies = [temps[t] for t in temperatures]
        ax.plot(temperatures, accuracies, marker="o", label=method)

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Temperature across Methods")
    ax.legend()
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(output_dir, "temperature_robustness.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filepath