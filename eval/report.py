"""Format evaluation results as markdown tables.

Used by run_comparison.py and run_adversarial.py to generate
human-readable output for the README and terminal.
"""

from __future__ import annotations

from typing import Any


def format_comparison_table(comparison: dict[str, Any]) -> str:
    """Format comparison results as a markdown table.

    The format matches spec §3.4. When fallback_used is True for an arm,
    the name is appended with '(heuristic fallback)' to comply with
    AGENTS.md rule 4.
    """
    arms = comparison.get("arms", [])
    if not arms:
        return "No comparison results available.\n"

    lines: list[str] = [
        "## Comparison Results",
        "",
        f"Dataset: {comparison.get('dataset', 'N/A')} "
        f"({comparison.get('n_samples', 0)} samples)",
        "",
        "| Arm | Beh. Acc | Bal. Acc | 95% CI | WC-F1 | p50 lat. | p95 lat. | $/1k | Det. |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for arm in arms:
        name = arm["arm_name"]
        if arm.get("fallback_used", False):
            if "fallback" not in name.lower():
                name += " ⚠️ fallback"

        acc = arm.get("behaviour_accuracy", 0)
        bal_acc = arm.get("balanced_accuracy", 0)
        ci_lo = arm.get("ci_95_lower", 0)
        ci_hi = arm.get("ci_95_upper", 0)
        wc = arm.get("worst_cluster_f1", 0)
        p50 = arm.get("p50_latency_ms", 0)
        p95 = arm.get("p95_latency_ms", 0)
        cost = arm.get("cost_per_1k_usd", 0)
        det = "yes" if arm.get("deterministic", False) else "no"

        lines.append(
            f"| {name} | {acc:.3f} | {bal_acc:.3f} | [{ci_lo:.3f}, {ci_hi:.3f}] | "
            f"{wc:.1f} | {p50:.0f} ms | {p95:.0f} ms | "
            f"${cost:.2f} | {det} |"
        )

    lines.append("")
    lines.append(
        f"*Generated: {comparison.get('timestamp', 'N/A')}*"
    )
    lines.append("")

    return "\n".join(lines)


def format_adversarial_table(results: dict[str, Any]) -> str:
    """Format adversarial results as a markdown pass/fail table."""
    per_row = results.get("per_row", [])
    if not per_row:
        return "No adversarial results available.\n"

    summary = results.get("summary", {})

    lines: list[str] = [
        "## Adversarial Evaluation Results",
        "",
        f"Dataset: {results.get('dataset', 'N/A')} "
        f"({results.get('n_samples', 0)} samples)",
        "",
    ]

    # Injection resistance headline
    gate_resist = summary.get("gate_injection_resistance")
    llm_resist = summary.get("llm_injection_resistance")
    if gate_resist is not None:
        lines.append("### Injection Resistance")
        lines.append("")
        lines.append("| | Gate | LLM Judge |")
        lines.append("|---|---|---|")
        lines.append(
            f"| Resistance | {gate_resist:.0%} | {llm_resist:.0%} |"
        )
        lines.append("")

    # Per-row table
    lines.extend(
        [
            "### Per-Row Results",
            "",
            "| ID | Category | Gate | LLM | Gate ✓ | LLM ✓ |",
            "|---|---|---|---|---|---|",
        ]
    )

    for row in per_row:
        gate_ok = "✅" if row["gate_correct"] else "❌"
        llm_ok = "✅" if row["llm_correct"] else "❌"
        lines.append(
            f"| {row['id']} | {row['category']} | "
            f"{row['gate_prediction']} | {row['llm_prediction']} | "
            f"{gate_ok} | {llm_ok} |"
        )

    lines.append("")
    lines.append(
        f"*Generated: {results.get('timestamp', 'N/A')}*"
    )
    lines.append("")

    return "\n".join(lines)
