from typing import List

import pandas as pd
import numpy as np
from scipy import stats

from src.models.dataset import DatasetMetadata
from src.models.validation_result import ValidationFinding


class StatisticalValidator:
    """Detects statistical outliers and impossible values."""

    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier

    def validate(self, df: pd.DataFrame, metadata: DatasetMetadata) -> List[ValidationFinding]:
        findings = []

        for col in metadata.numeric_columns:
            series = df[col].dropna()
            if len(series) < 3:
                continue

            findings.extend(self._check_zscore_outliers(df, col, series))
            findings.extend(self._check_iqr_outliers(df, col, series))
            findings.extend(self._check_impossible_values(df, col, series))

        return findings

    def _check_zscore_outliers(
        self, df: pd.DataFrame, col: str, series: pd.Series
    ) -> List[ValidationFinding]:
        findings = []
        z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
        outlier_mask = z_scores > self.z_threshold
        outlier_indices = series.index[outlier_mask].tolist()

        if outlier_indices:
            findings.append(
                ValidationFinding(
                    severity="warning",
                    category="statistical",
                    column=col,
                    row_indices=outlier_indices[:20],  # Cap at 20 for display
                    message=f"Found {len(outlier_indices)} outliers in '{col}' (Z-score > {self.z_threshold})",
                    suggestion=f"Review these rows for data entry errors. Consider capping or removing extreme values.",
                    evidence={
                        "method": "zscore",
                        "threshold": self.z_threshold,
                        "count": len(outlier_indices),
                        "example_values": series.loc[outlier_indices[:5]].tolist(),
                        "column_mean": float(series.mean()),
                        "column_std": float(series.std()),
                    },
                )
            )
        return findings

    def _check_iqr_outliers(
        self, df: pd.DataFrame, col: str, series: pd.Series
    ) -> List[ValidationFinding]:
        findings = []
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr

        outlier_mask = (series < lower) | (series > upper)
        outlier_indices = series.index[outlier_mask].tolist()

        if outlier_indices and len(outlier_indices) / len(series) < 0.1:
            findings.append(
                ValidationFinding(
                    severity="info",
                    category="statistical",
                    column=col,
                    row_indices=outlier_indices[:20],
                    message=f"Found {len(outlier_indices)} IQR outliers in '{col}' (outside [{lower:.2f}, {upper:.2f}])",
                    suggestion="These may be legitimate extreme values or data quality issues.",
                    evidence={
                        "method": "iqr",
                        "q1": float(q1),
                        "q3": float(q3),
                        "iqr": float(iqr),
                        "lower_bound": float(lower),
                        "upper_bound": float(upper),
                        "count": len(outlier_indices),
                    },
                )
            )
        return findings

    def _check_impossible_values(
        self, df: pd.DataFrame, col: str, series: pd.Series
    ) -> List[ValidationFinding]:
        findings = []
        col_lower = col.lower()

        # Negative values in typically-positive columns
        positive_keywords = ["age", "price", "cost", "revenue", "salary", "quantity", "count", "amount", "weight", "height"]
        if any(kw in col_lower for kw in positive_keywords):
            negative_mask = series < 0
            neg_indices = series.index[negative_mask].tolist()
            if neg_indices:
                findings.append(
                    ValidationFinding(
                        severity="error",
                        category="statistical",
                        column=col,
                        row_indices=neg_indices[:20],
                        message=f"Found {len(neg_indices)} negative values in '{col}' which should be positive",
                        suggestion="Check for data entry errors or sign inversions.",
                        evidence={
                            "count": len(neg_indices),
                            "example_values": series.loc[neg_indices[:5]].tolist(),
                        },
                    )
                )

        # Percentage columns > 100 or < 0
        pct_keywords = ["percent", "pct", "rate", "ratio", "proportion"]
        if any(kw in col_lower for kw in pct_keywords):
            invalid_mask = (series > 100) | (series < 0)
            invalid_indices = series.index[invalid_mask].tolist()
            if invalid_indices:
                findings.append(
                    ValidationFinding(
                        severity="error",
                        category="statistical",
                        column=col,
                        row_indices=invalid_indices[:20],
                        message=f"Found {len(invalid_indices)} values outside [0, 100] in percentage column '{col}'",
                        suggestion="Verify these values are actual percentages and not raw numbers.",
                        evidence={
                            "count": len(invalid_indices),
                            "example_values": series.loc[invalid_indices[:5]].tolist(),
                        },
                    )
                )

        return findings
