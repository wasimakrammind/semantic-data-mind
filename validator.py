from typing import List, Optional

from src.models.dataset import Dataset
from src.models.validation_result import ValidationFinding, ValidationReport
from src.validation.rules.statistical import StatisticalValidator
from src.validation.rules.temporal import TemporalValidator
from src.validation.rules.schema_rules import SchemaValidator
from src.validation.rules.cross_column import CrossColumnValidator
from src.validation.rules.semantic import SemanticValidator


class DataValidator:
    """Orchestrates all validation rules against a Dataset."""

    def __init__(
        self,
        z_threshold: float = 3.0,
        missing_warning_pct: float = 5.0,
        missing_critical_pct: float = 30.0,
    ):
        self.statistical = StatisticalValidator(z_threshold=z_threshold)
        self.temporal = TemporalValidator()
        self.schema = SchemaValidator(
            missing_warning_pct=missing_warning_pct,
            missing_critical_pct=missing_critical_pct,
        )
        self.cross_column = CrossColumnValidator()
        self.semantic = SemanticValidator()

    def validate(
        self,
        dataset: Dataset,
        checks: Optional[List[str]] = None,
        anthropic_client=None,
    ) -> ValidationReport:
        """Run validation checks and return a consolidated report."""
        if checks is None:
            checks = ["statistical", "temporal", "schema", "cross_column"]
            if anthropic_client:
                checks.append("semantic")

        all_findings: List[ValidationFinding] = []

        if "statistical" in checks or "all" in checks:
            all_findings.extend(
                self.statistical.validate(dataset.df, dataset.metadata)
            )

        if "temporal" in checks or "all" in checks:
            all_findings.extend(
                self.temporal.validate(dataset.df, dataset.metadata)
            )

        if "schema" in checks or "all" in checks:
            all_findings.extend(
                self.schema.validate(dataset.df, dataset.metadata)
            )

        if "cross_column" in checks or "all" in checks:
            all_findings.extend(
                self.cross_column.validate(dataset.df, dataset.metadata)
            )

        if ("semantic" in checks or "all" in checks) and anthropic_client:
            all_findings.extend(
                self.semantic.validate(dataset, anthropic_client)
            )

        score = self._compute_quality_score(all_findings, dataset)
        summary = self._generate_summary(all_findings, score)

        return ValidationReport(
            findings=all_findings,
            summary=summary,
            data_quality_score=score,
        )

    def _compute_quality_score(
        self, findings: List[ValidationFinding], dataset: Dataset
    ) -> float:
        """Compute a 0-100 data quality score."""
        score = 100.0
        for f in findings:
            if f.severity == "error":
                score -= 10
            elif f.severity == "warning":
                score -= 3
            elif f.severity == "info":
                score -= 1
        return max(0.0, min(100.0, score))

    def _generate_summary(
        self, findings: List[ValidationFinding], score: float
    ) -> str:
        errors = [f for f in findings if f.severity == "error"]
        warnings = [f for f in findings if f.severity == "warning"]
        infos = [f for f in findings if f.severity == "info"]

        parts = [f"Data Quality Score: {score:.0f}/100"]

        if not findings:
            parts.append("No data quality issues found. Your data looks clean!")
            return "\n".join(parts)

        parts.append(
            f"Found {len(findings)} issues: "
            f"{len(errors)} errors, {len(warnings)} warnings, {len(infos)} informational."
        )

        if errors:
            parts.append("\nCritical issues that need attention:")
            for e in errors[:5]:
                parts.append(f"  - {e.message}")

        if warnings:
            parts.append("\nWarnings to review:")
            for w in warnings[:5]:
                parts.append(f"  - {w.message}")

        return "\n".join(parts)
