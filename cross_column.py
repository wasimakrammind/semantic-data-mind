from typing import List

import pandas as pd
import numpy as np

from src.models.dataset import DatasetMetadata
from src.models.validation_result import ValidationFinding


class CrossColumnValidator:
    """Detects cross-column logic violations."""

    def validate(self, df: pd.DataFrame, metadata: DatasetMetadata) -> List[ValidationFinding]:
        findings = []
        findings.extend(self._check_sum_consistency(df, metadata))
        findings.extend(self._check_range_consistency(df, metadata))
        return findings

    def _check_sum_consistency(
        self, df: pd.DataFrame, metadata: DatasetMetadata
    ) -> List[ValidationFinding]:
        """Check if 'total' columns equal the sum of their parts."""
        findings = []
        numeric_cols = metadata.numeric_columns

        total_cols = [c for c in numeric_cols if "total" in c.lower() or "sum" in c.lower()]

        for total_col in total_cols:
            prefix = total_col.lower().replace("total", "").replace("sum", "").strip("_")
            # Find component columns that share the prefix
            component_cols = [
                c for c in numeric_cols
                if c != total_col and prefix and prefix in c.lower()
            ]

            if len(component_cols) >= 2:
                computed_sum = df[component_cols].sum(axis=1)
                actual_total = df[total_col]

                # Allow small floating point differences
                mismatch_mask = (
                    (computed_sum.notna() & actual_total.notna())
                    & (np.abs(computed_sum - actual_total) > 0.01)
                )
                mismatch_indices = df.index[mismatch_mask].tolist()

                if mismatch_indices:
                    findings.append(
                        ValidationFinding(
                            severity="error",
                            category="cross_column",
                            column=total_col,
                            row_indices=mismatch_indices[:20],
                            message=(
                                f"'{total_col}' doesn't equal sum of {component_cols} "
                                f"in {len(mismatch_indices)} rows"
                            ),
                            suggestion="Verify the calculation. The total column may need recalculation.",
                            evidence={
                                "total_column": total_col,
                                "component_columns": component_cols,
                                "count": len(mismatch_indices),
                            },
                        )
                    )
        return findings

    def _check_range_consistency(
        self, df: pd.DataFrame, metadata: DatasetMetadata
    ) -> List[ValidationFinding]:
        """Check min/max column pairs for consistency."""
        findings = []
        numeric_cols = metadata.numeric_columns

        # Find min/max pairs
        for col in numeric_cols:
            col_lower = col.lower()
            if "min" in col_lower:
                base = col_lower.replace("min", "")
                max_candidates = [
                    c for c in numeric_cols
                    if "max" in c.lower() and c.lower().replace("max", "") == base
                ]
                for max_col in max_candidates:
                    violation_mask = (
                        df[col].notna() & df[max_col].notna() & (df[col] > df[max_col])
                    )
                    violation_indices = df.index[violation_mask].tolist()

                    if violation_indices:
                        findings.append(
                            ValidationFinding(
                                severity="error",
                                category="cross_column",
                                column=f"{col}, {max_col}",
                                row_indices=violation_indices[:20],
                                message=f"'{col}' > '{max_col}' in {len(violation_indices)} rows",
                                suggestion="Min values should not exceed max values. Check for swapped columns.",
                                evidence={"count": len(violation_indices)},
                            )
                        )
        return findings
