from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

from src.models.dataset import DatasetMetadata


class CorrelationAnalyzer:
    """Correlation analysis between columns."""

    def compute(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
        min_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        cols = columns or metadata.numeric_columns
        cols = [c for c in cols if c in df.columns and c in metadata.numeric_columns]

        if len(cols) < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis."}

        numeric_df = df[cols].dropna()
        if len(numeric_df) < 3:
            return {"error": "Not enough non-null rows for correlation analysis."}

        corr_matrix = numeric_df.corr(method=method)

        # Extract significant correlations
        significant = []
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                val = corr_matrix.loc[c1, c2]
                if abs(val) >= min_threshold:
                    significant.append({
                        "column_1": c1,
                        "column_2": c2,
                        "correlation": round(float(val), 4),
                        "strength": self._classify_strength(abs(val)),
                        "direction": "positive" if val > 0 else "negative",
                    })

        significant.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        result = {
            "method": method,
            "num_columns": len(cols),
            "num_rows_used": len(numeric_df),
            "significant_correlations": significant[:20],
            "strongest_positive": significant[0] if significant and significant[0]["direction"] == "positive" else None,
            "strongest_negative": next(
                (s for s in significant if s["direction"] == "negative"), None
            ),
            "matrix": {
                c1: {c2: round(float(corr_matrix.loc[c1, c2]), 4) for c2 in cols}
                for c1 in cols
            },
        }

        return result

    def _classify_strength(self, abs_corr: float) -> str:
        if abs_corr >= 0.8:
            return "very strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        else:
            return "weak"
