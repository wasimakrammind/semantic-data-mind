from typing import List, Optional

from src.models.validation_result import ValidationReport
from src.models.analysis_result import EDAResult


class Explainer:
    """Generates plain-English explanations from analysis results.

    This module provides template-based explanations. When the Anthropic client
    is available, the agent orchestrator handles LLM-powered explanations directly
    through the ReAct loop, making this a lightweight fallback.
    """

    def explain_validation(self, report: ValidationReport) -> str:
        """Turn a validation report into a narrative."""
        if not report.findings:
            return (
                f"Great news! Your data scored {report.data_quality_score:.0f}/100 "
                "on our quality assessment. No significant issues were found."
            )

        parts = [
            f"I analyzed your data quality and found **{len(report.findings)} issues** "
            f"(score: {report.data_quality_score:.0f}/100).\n"
        ]

        errors = [f for f in report.findings if f.severity == "error"]
        warnings = [f for f in report.findings if f.severity == "warning"]

        if errors:
            parts.append(f"**{len(errors)} critical issue(s) need attention:**\n")
            for i, e in enumerate(errors[:5], 1):
                parts.append(f"{i}. {e.message}")
                if e.suggestion:
                    parts.append(f"   *Suggestion:* {e.suggestion}\n")

        if warnings:
            parts.append(f"\n**{len(warnings)} warning(s) to review:**\n")
            for i, w in enumerate(warnings[:5], 1):
                parts.append(f"{i}. {w.message}")

        return "\n".join(parts)

    def explain_eda(self, result: EDAResult, question: str = "") -> str:
        """Explain EDA results in plain English."""
        parts = [f"**Analysis: {result.analysis_type}**\n"]

        if result.summary:
            parts.append(result.summary)

        if result.charts:
            parts.append(f"\n*Generated {len(result.charts)} visualization(s) to illustrate the findings.*")

        return "\n".join(parts)

    def suggest_next_steps(
        self, has_data: bool, has_validation: bool, has_eda: bool
    ) -> List[str]:
        """Suggest what the user might want to do next."""
        suggestions = []

        if not has_data:
            return ["Upload a dataset to get started (CSV, Excel, JSON, or Parquet)"]

        if not has_validation:
            suggestions.append("Run data validation to check for quality issues")

        if not has_eda:
            suggestions.append("Ask me to explore your data (e.g., 'What are the trends in revenue?')")
            suggestions.append("Ask about correlations between columns")

        if has_validation and has_eda:
            suggestions.extend([
                "Ask about specific columns or patterns",
                "Request a forecast or trend analysis",
                "Ask what drives a particular metric",
            ])

        return suggestions
