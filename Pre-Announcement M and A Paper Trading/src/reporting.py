from __future__ import annotations

import pandas as pd


QUESTIONS = [
    "Can public information predict transformative transactions before announcement?",
    "What is the baseline event rate?",
    "How much lift does the model produce?",
    "Which features matter?",
    "Does the signal survive out-of-sample testing?",
    "Does it survive placebo tests?",
    "Does it survive realistic transaction costs?",
    "Does it work specifically among microcaps?",
    "Does it work specifically for biotech/medtech?",
    "Does it work across sectors?",
    "How early does the signal appear?",
    "How many false positives occur?",
    "What characteristics distinguish the biggest pre-announcement winners?",
    "Does AEMD represent a repeatable pattern or an outlier?",
    "What would be required to deploy this as a live daily scanner?",
]


def final_report_template(metrics: pd.DataFrame, quality: dict[str, object]) -> str:
    lines = [
        "# Transformative Transaction Prediction Research Report",
        "",
        "## Dataset",
        "",
        f"- Rows: {quality.get('rows')}",
        f"- Date range: {quality.get('date_min')} to {quality.get('date_max')}",
        f"- Duplicate observations: {quality.get('duplicate_observations')}",
        "",
        "## Model Results",
        "",
    ]
    if metrics.empty:
        lines.append("No metrics available yet.")
    else:
        lines.append(metrics.to_markdown(index=False))
    lines.extend(["", "## Required Answers", ""])
    for i, q in enumerate(QUESTIONS, start=1):
        lines.append(f"{i}. {q}")
        lines.append("   - Pending empirical result.")
    return "\n".join(lines)
