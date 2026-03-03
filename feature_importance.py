from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.models.dataset import DatasetMetadata


class FeatureImportanceAnalyzer:
    """Compute feature importance relative to a target variable."""

    def compute(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        target_column: str,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found."}

        # Determine if target is numeric or categorical
        is_numeric_target = target_column in metadata.numeric_columns

        # Select feature columns (exclude identifiers and the target)
        feature_cols = [
            c for c in metadata.numeric_columns + metadata.categorical_columns
            if c != target_column and c not in metadata.identifier_columns
        ]

        if not feature_cols:
            return {"error": "No suitable feature columns found."}

        # Prepare data
        work_df = df[feature_cols + [target_column]].dropna()
        if len(work_df) < 10:
            return {"error": "Not enough non-null rows for feature importance analysis."}

        # Encode categoricals
        X = work_df[feature_cols].copy()
        encoders = {}
        for col in metadata.categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le

        y = work_df[target_column]
        if not is_numeric_target:
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))

        result = {"target": target_column, "methods": {}}

        # Mutual Information
        try:
            if is_numeric_target:
                mi_scores = mutual_info_regression(X, y, random_state=42)
            else:
                mi_scores = mutual_info_classif(X, y, random_state=42)

            mi_ranking = sorted(
                zip(feature_cols, mi_scores), key=lambda x: -x[1]
            )[:top_n]
            result["methods"]["mutual_information"] = [
                {"feature": f, "score": round(float(s), 4)} for f, s in mi_ranking
            ]
        except Exception as e:
            result["methods"]["mutual_information"] = {"error": str(e)}

        # Random Forest importance
        try:
            if is_numeric_target:
                rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            else:
                rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)

            rf.fit(X, y)
            importances = rf.feature_importances_
            rf_ranking = sorted(
                zip(feature_cols, importances), key=lambda x: -x[1]
            )[:top_n]
            result["methods"]["random_forest"] = [
                {"feature": f, "importance": round(float(s), 4)} for f, s in rf_ranking
            ]
        except Exception as e:
            result["methods"]["random_forest"] = {"error": str(e)}

        # Correlation-based (for numeric targets only)
        if is_numeric_target:
            numeric_features = [c for c in feature_cols if c in metadata.numeric_columns]
            if numeric_features:
                correlations = []
                for col in numeric_features:
                    corr = df[col].corr(df[target_column])
                    if not pd.isna(corr):
                        correlations.append({"feature": col, "correlation": round(float(corr), 4)})
                correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                result["methods"]["correlation"] = correlations[:top_n]

        # Combined ranking
        combined = self._combine_rankings(result["methods"], feature_cols, top_n)
        result["combined_ranking"] = combined

        return result

    def _combine_rankings(
        self, methods: Dict, feature_cols: List[str], top_n: int
    ) -> List[Dict]:
        """Average rankings across methods."""
        scores = {f: [] for f in feature_cols}

        for method_name, rankings in methods.items():
            if isinstance(rankings, dict) and "error" in rankings:
                continue
            if isinstance(rankings, list):
                for i, item in enumerate(rankings):
                    feature = item.get("feature")
                    if feature:
                        # Normalize score: rank position / total features
                        scores[feature].append(1.0 - i / len(rankings))

        combined = []
        for feature, s in scores.items():
            if s:
                combined.append({"feature": feature, "avg_score": round(sum(s) / len(s), 4)})

        combined.sort(key=lambda x: -x["avg_score"])
        return combined[:top_n]
