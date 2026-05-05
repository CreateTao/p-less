from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ComparisonResult:
    """Single comparison between p-less and a baseline."""
    method_a: str
    method_b: str
    dataset: str
    model: str
    temperature_a: float
    temperature_b: float
    t_statistic: float = 0.0
    p_value: float = 0.0
    outcome: str = ""  # "Win", "Tie", "Lose"


@dataclass
class WinTieLossSummary:
    """Aggregated Win/Tie/Lose across all comparisons."""
    p_less_method: str
    comparisons: list[ComparisonResult] = field(default_factory=list)
    total_win: int = 0
    total_tie: int = 0
    total_loss: int = 0
    total_comparisons: int = 0
    win_tie_rate: float = 0.0


def compute_win_tie_loss(
    comparisons: list[ComparisonResult],
    p_less_method: str = "p_less",
) -> WinTieLossSummary:
    """Aggregate Win/Tie/Lose across all (dataset x model x baseline) combinations."""
    total_win = sum(1 for c in comparisons if c.outcome == "Win")
    total_tie = sum(1 for c in comparisons if c.outcome == "Tie")
    total_loss = sum(1 for c in comparisons if c.outcome == "Lose")
    total = len(comparisons)
    win_tie_rate = (total_win + total_tie) / total if total > 0 else 0.0

    return WinTieLossSummary(
        p_less_method=p_less_method,
        comparisons=comparisons,
        total_win=total_win,
        total_tie=total_tie,
        total_loss=total_loss,
        total_comparisons=total,
        win_tie_rate=win_tie_rate,
    )