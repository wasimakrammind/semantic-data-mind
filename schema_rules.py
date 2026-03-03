from typing import List

import pandas as pd
import numpy as np

from src.models.dataset import DatasetMetadata
from src.models.validation_result import ValidationFinding


class SchemaValidator:
    """Detects schema-level issues: type consistency, missing patterns, duplicates."""

    def __init__(self, missing_warning_pct: float = 5.0, missing_critical_pct: float = 30.0):
        self.missing_warning_pct = missing_warning_pct
        self.missing_critical_pct = missing_critical_pct

    def validate(self, df: pd.DataFrame, metadata: DatasetMetadata) -> List[ValidationFinding]:
        findings = []
        findings.extend(self._check_mixed_types(df, metadata))
        findings.extend(self._check_missing_patterns(df, metadata))
        findings.extend(self._check_duplicate_rows(df))
        findings.extend(self._check_constant_columns(df))
        return findings

    def _check_mixed_types(
        self, df: pd.DataFrame, metadata: DatasetMetadata
    ) -> List[ValidationFinding]:
        findings = []
        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().head(1000)
            if len(sample) == 0:
                continue

            types_found = set()
            for val in sample:
                try:
                    float(val)
                    types_found.add("numeric")
                except (ValueError, TypeError):
                    try:
                        pd.to_datetime(val)
                        types_found.add("datetime")
                    except (ValueError, TypeError):
                        types_found.add("string")

            if len(types_found) > 1:
                findings.append(
                    ValidationFinding(
                        severity="warning",
                        category="schema",
                        column=col,
                        row_indices=None,
                        message=f"Column '{col}' contains mixed types: {types_found}",
                        suggestion="This column may need type conversion or cleaning. Numeric values mixed with text often indicate data entry issues.",
                        evidence={"types_found": list(types_found)},
                    )
                )
        return findings

    def _check_missing_patterns(
        self, df: pd.DataFrame, metadata: DatasetMetadata
    ) -> List[ValidationFinding]:
        findings = []
        for col, pct in metadata.missing_summary.items():
            if pct >= self.missing_critical_pct:
                findings.append(
                    ValidationFinding(
                        severity="error",
                        category="schema",
                        column=col,
                        row_indices=None,
                        message=f"Column '{col}' has {pct:.1f}% missing values (critical threshold: {self.missing_critical_pct}%)",
                        suggestion="Consider dropping this column or investigating why so much data is missing.",
                        evidence={"missing_pct": pct, "threshold": self.missing_critical_pct},
                    )
                )
            elif pct >= self.missing_warning_pct:
                findings.append(
                    ValidationFinding(
                        severity="warning",
                        category="schema",
                        column=col,
                        row_indices=None,
                        message=f"Column '{col}' has {pct:.1f}% missing values",
                        suggestion="Consider imputation strategies (mean/median for numeric, mode for categorical).",
                        evidence={"missing_pct": pct, "threshold": self.missing_warning_pct},
                    )
                )
        return findings

    def _check_duplicate_rows(self, df: pd.DataFrame) -> List[ValidationFinding]:
        findings = []
        dup_mask = df.duplicated(keep="first")
        dup_count = dup_mask.sum()

        if dup_count > 0:
            pct = dup_count / len(df) * 100
            findings.append(
                ValidationFinding(
                    severity="warning" if pct < 5 else "error",
                    category="schema",
                    column=None,
                    row_indices=df.index[dup_mask].tolist()[:20],
                    message=f"Found {dup_count} duplicate rows ({pct:.1f}% of dataset)",
                    suggestion="Consider deduplicating the dataset. Check if duplicates are intentional.",
                    evidence={"count": int(dup_count), "percentage": round(pct, 2)},
                )
            )
        return findings

    def _check_constant_columns(self, df: pd.DataFrame) -> List[ValidationFinding]:
        findings = []
        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            if nunique <= 1 and len(df) > 1:
                findings.append(
                    ValidationFinding(
                        severity="info",
                        category="schema",
                        column=col,
                        row_indices=None,
                        message=f"Column '{col}' has only {nunique} unique value(s) — provides no analytical value",
                        suggestion="Consider dropping this column from analysis.",
                        evidence={"unique_values": nunique},
                    )
                )
        return findings
