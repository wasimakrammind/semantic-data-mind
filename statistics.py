from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from src.models.dataset import DatasetMetadata


class StatisticsCalculator:
    """Descriptive statistics, distributions, and outlier detection."""

    def compute(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        columns: Optional[List[str]] = None,
        group_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        if columns:
            cols = [c for c in columns if c in df.columns]
        else:
            cols = metadata.numeric_columns

        if group_by and group_by in df.columns:
            return self._grouped_stats(df, cols, group_by)

        result = {"columns": {}}
        for col in cols:
            if col in metadata.numeric_columns:
                result["columns"][col] = self._numeric_stats(df[col])
            elif col in metadata.categorical_columns:
                result["columns"][col] = self._categorical_stats(df[col])

        return result

    def _numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        clean = series.dropna()
        if len(clean) == 0:
            return {"count": 0, "all_missing": True}

        desc = clean.describe()
        result = {
            "count": int(desc["count"]),
            "mean": float(desc["mean"]),
            "std": float(desc["std"]),
            "min": float(desc["min"]),
            "25%": float(desc["25%"]),
            "median": float(desc["50%"]),
            "75%": float(desc["75%"]),
            "max": float(desc["max"]),
            "skewness": float(clean.skew()),
            "kurtosis": float(clean.kurtosis()),
            "missing_count": int(series.isna().sum()),
        }

        # Normality test (only for reasonable sample sizes)
        if 8 <= len(clean) <= 5000:
            try:
                stat, p_value = scipy_stats.shapiro(clean.sample(min(len(clean), 5000), random_state=42))
                result["normality_test"] = {
                    "test": "Shapiro-Wilk",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "is_normal": p_value > 0.05,
                }
            except Exception:
                pass

        return result

    def _categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        vc = series.value_counts()
        return {
            "unique_count": int(series.nunique()),
            "mode": str(vc.index[0]) if len(vc) > 0 else None,
            "mode_frequency": int(vc.iloc[0]) if len(vc) > 0 else 0,
            "top_values": {str(k): int(v) for k, v in vc.head(10).items()},
            "missing_count": int(series.isna().sum()),
        }

    def _grouped_stats(
        self, df: pd.DataFrame, columns: List[str], group_by: str
    ) -> Dict[str, Any]:
        result = {"grouped_by": group_by, "groups": {}}
        for name, group in df.groupby(group_by):
            group_stats = {}
            for col in columns:
                if col in df.select_dtypes(include=[np.number]).columns:
                    group_stats[col] = self._numeric_stats(group[col])
            result["groups"][str(name)] = group_stats
        return result

    def detect_outliers(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        cols = columns or metadata.numeric_columns
        result = {"method": method, "columns": {}}

        for col in cols:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 3:
                continue

            if method in ("iqr", "both"):
                iqr_thresh = threshold or 1.5
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - iqr_thresh * iqr, q3 + iqr_thresh * iqr
                outlier_mask = (series < lower) | (series > upper)
                outlier_indices = series.index[outlier_mask].tolist()

                result["columns"][col] = {
                    "method": "iqr",
                    "outlier_count": len(outlier_indices),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "outlier_values": series.loc[outlier_indices[:10]].tolist(),
                    "row_indices": outlier_indices[:20],
                }

            if method in ("zscore", "both"):
                z_thresh = threshold or 3.0
                z_scores = np.abs(scipy_stats.zscore(series, nan_policy="omit"))
                outlier_mask = z_scores > z_thresh
                outlier_indices = series.index[outlier_mask].tolist()

                key = f"{col}_zscore" if method == "both" else col
                result["columns"][key] = {
                    "method": "zscore",
                    "threshold": z_thresh,
                    "outlier_count": len(outlier_indices),
                    "outlier_values": series.loc[outlier_indices[:10]].tolist(),
                    "row_indices": outlier_indices[:20],
                }

        return result

    def analyze_missing(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        cols_to_check = columns or [
            c for c, pct in metadata.missing_summary.items() if pct > 0
        ]

        if not cols_to_check:
            return {"message": "No missing data found in the dataset."}

        result = {"total_rows": len(df), "columns": {}}

        for col in cols_to_check:
            if col not in df.columns:
                continue
            missing_count = int(df[col].isna().sum())
            missing_pct = missing_count / len(df) * 100

            col_result = {
                "missing_count": missing_count,
                "missing_pct": round(missing_pct, 2),
                "present_count": len(df) - missing_count,
            }

            # Suggest imputation
            if col in metadata.numeric_columns:
                col_result["suggested_imputation"] = "median"
                col_result["imputation_value"] = float(df[col].median())
            elif col in metadata.categorical_columns:
                col_result["suggested_imputation"] = "mode"
                mode = df[col].mode()
                col_result["imputation_value"] = str(mode.iloc[0]) if len(mode) > 0 else None
            else:
                col_result["suggested_imputation"] = "review_manually"

            result["columns"][col] = col_result

        # Missing data correlation
        missing_cols = [c for c in cols_to_check if c in df.columns and df[c].isna().any()]
        if len(missing_cols) >= 2:
            missing_matrix = df[missing_cols].isna().astype(int)
            corr = missing_matrix.corr()
            high_corr = []
            for i, c1 in enumerate(missing_cols):
                for c2 in missing_cols[i + 1:]:
                    val = corr.loc[c1, c2]
                    if abs(val) > 0.5:
                        high_corr.append({
                            "column_1": c1,
                            "column_2": c2,
                            "correlation": round(float(val), 3),
                        })
            if high_corr:
                result["missing_correlations"] = high_corr
                result["note"] = "High correlation between missing patterns suggests data is Missing At Random (MAR)."

        return result
