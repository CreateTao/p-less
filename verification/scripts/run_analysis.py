"""Statistical analysis and report generation.

Reads raw results from CPU/GPU benchmarks, computes:
- Accuracy comparison tables (group A: framework defaults, group B: recommended configs)
- Bootstrap confidence intervals
- Paired t-test / Wilcoxon comparisons
- TOST equivalence test summaries
- Win/Tie/Lose tables
- Temperature robustness plots

Outputs markdown tables, LaTeX tables, and PNG figures.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, ".")

from verification.storage.io import load_results
from verification.stats.bootstrap import bootstrap_ci, bootstrap_diff_ci
from verification.stats.paired_tests import paired_t_test, wilcoxon_test
from verification.stats.tost import tost_equivalence_test
from verification.stats.win_tie_loss import compute_win_tie_loss
from verification.stats.reporting import (
    generate_accuracy_table, generate_wtl_table, plot_temperature_robustness,
)


def load_aggregated_results(results_dir: str) -> list[dict]:
    """Recursively load all aggregated result JSON files."""
    results = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.startswith("aggregated_") and f.endswith(".json"):
                data = load_results(os.path.join(root, f))
                results.append(data)
    return results


def load_all_sample_results(results_dir: str) -> dict:
    """Load all per-sample results, organized by (dataset, model, method, temp)."""
    organized = defaultdict(list)
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.startswith("q_") and f.endswith(".json"):
                data = load_results(os.path.join(root, f))
                key = (
                    data.get("dataset_name", ""),
                    data.get("model_name", ""),
                    data.get("method_name", ""),
                    data.get("temperature", 1.0),
                )
                organized[key].append(data)
    return dict(organized)


def load_tost_results(results_dir: str) -> list[dict]:
    """Load all TOST result JSON files."""
    results = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.startswith("tost_") and f.endswith(".json"):
                data = load_results(os.path.join(root, f))
                results.append(data)
    return results


def load_grid_search_results(results_dir: str) -> list[dict]:
    """Load all grid search result JSON files."""
    results = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.startswith("grid_search_") and f.endswith(".json"):
                data = load_results(os.path.join(root, f))
                results.append(data)
    return results


def build_accuracy_report(
    aggregated: list[dict],
    output_dir: str,
) -> str:
    """Build accuracy comparison tables for group A and group B."""
    # Organize: {dataset: {model: {method: {temperature: accuracy}}}}
    data_a = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    data_b = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for agg in aggregated:
        dataset = agg.get("dataset_name", "")
        model = agg.get("model_name", "")
        method = agg.get("method_name", "")
        temp = agg.get("temperature", 1.0)
        accuracy = agg.get("accuracy", 0.0) * 100  # Convert to percentage
        group = agg.get("method_params", {}).get("group", "B")

        if group == "A":
            data_a[dataset][model][method][temp] = accuracy
        else:
            data_b[dataset][model][method][temp] = accuracy

    report_lines = ["# P-Less Verification Results\n"]

    # Group A table
    if data_a:
        report_lines.append("## Group A: Framework Defaults Comparison\n")
        report_lines.append(generate_accuracy_table(dict(data_a), group="A"))
        report_lines.append("")

    # Group B table
    if data_b:
        report_lines.append("## Group B: Recommended Configs Comparison\n")
        report_lines.append(generate_accuracy_table(dict(data_b), group="B"))
        report_lines.append("")

    return "\n".join(report_lines)


def build_statistical_report(
    samples: dict,
    output_dir: str,
) -> str:
    """Run pairwise statistical tests and build report."""
    from verification.stats.win_tie_loss import ComparisonResult as WTLComparisonResult

    report_lines = ["## Statistical Tests\n"]

    # Get all p-less keys
    pless_keys = [k for k in samples if k[2] in ("p_less", "p_less_norm")]
    baseline_keys = [k for k in samples if k[2] not in ("p_less", "p_less_norm")]

    comparisons = []  # list of WTLComparisonResult for compute_win_tie_loss
    comparison_details = []  # list of dicts for table rendering

    for pk in pless_keys:
        dataset, model, pless_method, pless_temp = pk
        pless_corrects = [s["correct"] for s in samples[pk]]

        for bk in baseline_keys:
            if bk[0] != dataset or bk[1] != model:
                continue

            baseline_corrects = [s["correct"] for s in samples[bk]]

            # Paired t-test (returns tuple: t_stat, p_value, outcome)
            t_stat, p_value, outcome = paired_t_test(pless_corrects, baseline_corrects)

            # Compute accuracy difference
            import numpy as np
            pless_acc = np.mean(pless_corrects)
            baseline_acc = np.mean(baseline_corrects)
            diff = pless_acc - baseline_acc

            # Bootstrap diff CI
            mean_diff, ci_lo, ci_hi = bootstrap_diff_ci(pless_corrects, baseline_corrects)

            # Build WTL ComparisonResult
            comp = WTLComparisonResult(
                method_a=pless_method,
                method_b=bk[2],
                dataset=dataset,
                model=model,
                temperature_a=pless_temp,
                temperature_b=bk[3],
                t_statistic=t_stat,
                p_value=p_value,
                outcome=outcome,
            )
            comparisons.append(comp)

            comparison_details.append({
                "pless_method": pless_method,
                "pless_temp": pless_temp,
                "baseline_method": bk[2],
                "baseline_temp": bk[3],
                "dataset": dataset,
                "model": model,
                "diff": diff,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "outcome": outcome,
                "bootstrap_diff_ci": (ci_lo, ci_hi),
            })

    if not comparisons:
        report_lines.append("No comparisons available.\n")
        return "\n".join(report_lines)

    # Paired t-test table
    report_lines.append("### Paired t-test: p-less vs Baselines\n")
    report_lines.append(
        "| p-less | T | Baseline | T | Dataset | Δ Acc | p-value | Result |"
    )
    report_lines.append(
        "|--------|---|----------|---|---------|-------|---------|--------|"
    )
    for c in comparison_details:
        sig = "*" if c["significant"] else ""
        report_lines.append(
            f"| {c['pless_method']} | {c['pless_temp']} | {c['baseline_method']} "
            f"| {c['baseline_temp']} | {c['dataset']} | {c['diff']:+.4f} | "
            f"{c['p_value']:.4f}{sig} | {c['outcome']} |"
        )
    report_lines.append("")

    # Win/Tie/Lose
    wtl_summary = compute_win_tie_loss(comparisons, p_less_method="p_less")
    report_lines.append("### Win/Tie/Lose Summary\n")
    report_lines.append(generate_wtl_table(wtl_summary))
    report_lines.append("")

    return "\n".join(report_lines)


def build_tost_report(tost_results: list[dict]) -> str:
    """Build TOST equivalence test report."""
    if not tost_results:
        return ""

    lines = ["## TOST Equivalence Test Results\n"]
    lines.append(f"Equivalence margin (δ): {tost_results[0].get('delta', 0.02):.1%}\n")
    lines.append("| p-less | Baseline | Dataset | Δ | CI | Equivalent? |")
    lines.append("|--------|----------|---------|---|----|-------------|")

    for t in tost_results:
        eq = "YES" if t.get("is_equivalent", False) else "NO"
        ci = f"[{t.get('diff_ci_lower', 0):+.4f}, {t.get('diff_ci_upper', 0):+.4f}]"
        lines.append(
            f"| {t.get('method_a', '')} | {t.get('method_b', '')} | "
            f"{t.get('dataset', '')} | {t.get('mean_diff', 0):+.4f} | {ci} | {eq} |"
        )
    lines.append("")

    equiv_count = sum(1 for t in tost_results if t.get("is_equivalent", False))
    lines.append(f"**Equivalence rate: {equiv_count}/{len(tost_results)}**\n")

    return "\n".join(lines)


def build_temperature_plot(
    samples: dict, output_dir: str,
) -> str:
    """Generate temperature robustness plot."""
    # Organize: {method: {temperature: accuracy}}
    method_temp_acc = defaultdict(dict)

    for (dataset, model, method, temp), sample_list in samples.items():
        if method in ("p_less", "p_less_norm", "top_p", "top_k", "min_p"):
            acc = sum(s["correct"] for s in sample_list) / len(sample_list) * 100
            method_temp_acc[method][temp] = acc

    if not method_temp_acc:
        return ""

    plot_dir = os.path.join(output_dir, "figures")
    filepath = plot_temperature_robustness(dict(method_temp_acc), plot_dir)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis and report generation")
    parser.add_argument("--results-dir", required=True,
                        help="Root directory with benchmark results")
    parser.add_argument("--search-dir", default="",
                        help="Directory with grid search results (for context)")
    parser.add_argument("--output-dir", default="verification/outputs/report",
                        help="Output directory for report and figures")
    parser.add_argument("--delta", type=float, default=0.02,
                        help="TOST equivalence margin for re-computation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Statistical Analysis ===")
    print(f"Results directory: {args.results_dir}")

    # Load all data
    aggregated = load_aggregated_results(args.results_dir)
    samples = load_all_sample_results(args.results_dir)
    tost_results = load_tost_results(args.results_dir)

    print(f"Loaded {len(aggregated)} aggregated results")
    print(f"Loaded {len(samples)} sample groups")
    print(f"Loaded {len(tost_results)} TOST results")

    # Build report sections
    report_sections = []

    # 1. Accuracy tables
    if aggregated:
        acc_report = build_accuracy_report(aggregated, args.output_dir)
        report_sections.append(acc_report)
        print("Generated accuracy tables")

    # 2. Statistical tests
    if samples:
        stat_report = build_statistical_report(samples, args.output_dir)
        report_sections.append(stat_report)
        print("Generated statistical tests")

    # 3. TOST results
    if tost_results:
        tost_report = build_tost_report(tost_results)
        report_sections.append(tost_report)
        print("Generated TOST report")

    # 4. Temperature robustness plot
    if samples:
        plot_path = build_temperature_plot(samples, args.output_dir)
        if plot_path:
            report_sections.append(f"## Figures\n\n![Temperature Robustness]({plot_path})\n")
            print(f"Generated temperature plot: {plot_path}")

    # 5. Grid search context
    if args.search_dir:
        grid_results = load_grid_search_results(args.search_dir)
        if grid_results:
            grid_section = ["## Grid Search Best Parameters\n"]
            grid_section.append("| Method | Dataset | Model | Best Params | Best Accuracy |")
            grid_section.append("|--------|---------|-------|-------------|---------------|")
            for g in grid_results:
                grid_section.append(
                    f"| {g.get('method', '')} | {g.get('dataset', '')} | "
                    f"{g.get('model', '')} | {g.get('best_params', {})} | "
                    f"{g.get('best_accuracy', 0):.4f} |"
                )
            grid_section.append("")
            report_sections.append("\n".join(grid_section))
            print("Generated grid search summary")

    # Write final report
    report = "\n".join(report_sections)
    report_path = os.path.join(args.output_dir, "verification_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
