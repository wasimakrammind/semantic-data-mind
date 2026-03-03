from typing import List, Tuple
from itertools import combinations

import pandas as pd
import numpy as np

from src.models.dataset import DatasetMetadata
from src.models.validation_result import ValidationFinding


class TemporalValidator:
    """Detects temporal inconsistencies in datetime columns."""

    def validate(self, df: pd.DataFrame, metadata: DatasetMetadata) -> List[ValidationFinding]:
        findings = []

        if not metadata.datetime_columns:
            return findings

        for col in metadata.datetime_columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            findings.extend(self._check_future_dates(df, col, parsed))
            findings.extend(self._check_date_gaps(df, col, parsed))

        # Check date ordering across column pairs
        findings.extend(self._check_date_ordering(df, metadata))

        return findings

    def _check_future_dates(
        self, df: pd.DataFrame, col: str, parsed: pd.Series
    ) -> List[ValidationFinding]:
        findings = []
        now = pd.Timestamp.now()
        future_mask = parsed > now
        future_indices = df.index[future_mask].tolist()

        if future_indices:
            findings.append(
                ValidationFinding(
                    severity="warning",
                    category="temporal",
                    column=col,
                    row_indices=future_indices[:20],
                    message=f"Found {len(future_indices)} future dates in '{col}'",
                    suggestion="Verify whether future dates are expected (e.g., scheduled events) or data errors.",
                    evidence={
                        "count": len(future_indices),
                        "example_dates": parsed.loc[future_indices[:5]].dt.strftime("%Y-%m-%d").tolist(),
                    },
                )
            )
        return findings

    def _check_date_gaps(
        self, df: pd.DataFrame, col: str, parsed: pd.Series
    ) -> List[ValidationFinding]:
        findings = []
        valid_dates = parsed.dropna().sort_values()
        if len(valid_dates) < 10:
            return findings

        diffs = valid_dates.diff().dropna()
        median_gap = diffs.median()

        if median_gap.total_seconds() == 0:
            return findings

        # Find gaps > 5x the median gap
        large_gap_mask = diffs > (5 * median_gap)
        if large_gap_mask.any():
            gap_indices = diffs.index[large_gap_mask].tolist()
            gap_sizes = diffs.loc[gap_indices]
            findings.append(
                ValidationFinding(
                    severity="info",
                    category="temporal",
                    column=col,
                    row_indices=gap_indices[:10],
                    message=f"Found {large_gap_mask.sum()} large gaps in '{col}' (>{5}x median interval)",
                    suggestion="Check if data is missing for these periods or if gaps are expected.",
                    evidence={
                        "median_gap": str(median_gap),
                        "count": int(large_gap_mask.sum()),
                        "largest_gap": str(gap_sizes.max()),
                    },
                )
            )
        return findings

    def _check_date_ordering(
        self, df: pd.DataFrame, metadata: DatasetMetadata
    ) -> List[ValidationFinding]:
        """Check that date pairs follow expected ordering (e.g., order_date < ship_date)."""
        findings = []

        # Common ordering patterns
        ordering_hints = [
            ("start", "end"),
            ("begin", "end"),
            ("create", "update"),
            ("create", "close"),
            ("order", "ship"),
            ("order", "deliver"),
            ("open", "close"),
            ("birth", "death"),
            ("hire", "termin"),
        ]

        date_cols = metadata.datetime_columns
        for col_a, col_b in combinations(date_cols, 2):
            a_lower = col_a.lower()
            b_lower = col_b.lower()

            should_check = False
            for before_kw, after_kw in ordering_hints:
                if (before_kw in a_lower and after_kw in b_lower):
                    should_check = True
                    expected_first, expected_second = col_a, col_b
                    break
                if (before_kw in b_lower and after_kw in a_lower):
                    should_check = True
                    expected_first, expected_second = col_b, col_a
                    break

            if not should_check:
                continue

            parsed_a = pd.to_datetime(df[expected_first], errors="coerce")
            parsed_b = pd.to_datetime(df[expected_second], errors="coerce")

            # Find rows where first date is after second date
            violation_mask = (parsed_a > parsed_b) & parsed_a.notna() & parsed_b.notna()
            violation_indices = df.index[violation_mask].tolist()

            if violation_indices:
                findings.append(
                    ValidationFinding(
                        severity="error",
                        category="temporal",
                        column=f"{expected_first}, {expected_second}",
                        row_indices=violation_indices[:20],
                        message=(
                            f"Found {len(violation_indices)} rows where '{expected_first}' "
                            f"is after '{expected_second}'"
                        ),
                        suggestion="This likely indicates a data entry error. Review and correct these rows.",
                        evidence={
                            "count": len(violation_indices),
                            "examples": [
                                {
                                    "row": int(idx),
                                    expected_first: str(parsed_a.loc[idx]),
                                    expected_second: str(parsed_b.loc[idx]),
                                }
                                for idx in violation_indices[:5]
                            ],
                        },
                    )
                )

        return findings
