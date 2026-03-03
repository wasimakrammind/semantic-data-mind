from typing import List

import pandas as pd
import numpy as np

from src.models.dataset import DatasetMetadata


class SchemaDetector:
    """Infers semantic column types beyond raw pandas dtypes."""

    def detect(self, df: pd.DataFrame, filename: str, file_format: str) -> DatasetMetadata:
        datetime_cols = self._detect_datetime_columns(df)
        identifier_cols = self._detect_identifier_columns(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        boolean_cols = df.select_dtypes(include=["bool"]).columns.tolist()

        # Categoricals: object/category columns that aren't datetime, identifiers, or text
        text_cols = []
        categorical_cols = []
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if col in datetime_cols or col in identifier_cols:
                continue
            if self._is_text_column(df[col]):
                text_cols.append(col)
            else:
                categorical_cols.append(col)

        missing_summary = {}
        for col in df.columns:
            pct = df[col].isna().sum() / len(df) * 100 if len(df) > 0 else 0.0
            missing_summary[col] = round(pct, 2)

        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        return DatasetMetadata(
            filename=filename,
            file_format=file_format.lstrip("."),
            row_count=len(df),
            column_count=len(df.columns),
            column_names=df.columns.tolist(),
            dtypes={col: str(df[col].dtype) for col in df.columns},
            memory_usage_mb=round(memory_mb, 2),
            has_datetime_index=isinstance(df.index, pd.DatetimeIndex),
            categorical_columns=categorical_cols,
            numeric_columns=numeric_cols,
            datetime_columns=datetime_cols,
            text_columns=text_cols,
            identifier_columns=identifier_cols,
            boolean_columns=boolean_cols,
            missing_summary=missing_summary,
        )

    def _detect_datetime_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns that are or can be parsed as datetimes."""
        datetime_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()

        for col in df.select_dtypes(include=["object"]).columns:
            if col in datetime_cols:
                continue
            sample = df[col].dropna().head(20)
            if len(sample) == 0:
                continue
            try:
                pd.to_datetime(sample, format="mixed")
                # If >80% parse successfully, treat as datetime
                parsed = pd.to_datetime(df[col], errors="coerce")
                success_rate = parsed.notna().sum() / max(df[col].notna().sum(), 1)
                if success_rate > 0.8:
                    datetime_cols.append(col)
            except (ValueError, TypeError):
                continue

        return datetime_cols

    def _detect_identifier_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns that look like IDs (high cardinality, sequential-like)."""
        id_cols = []
        for col in df.columns:
            col_lower = col.lower()
            # Name-based heuristic
            if any(kw in col_lower for kw in ["_id", "id_", "identifier", "key", "index", "uuid"]):
                id_cols.append(col)
                continue
            if col_lower in ("id", "pk"):
                id_cols.append(col)
                continue

            # Cardinality-based: if unique values > 90% of rows, likely an ID
            if df[col].dtype in ("int64", "float64", "object"):
                nunique = df[col].nunique()
                if len(df) > 10 and nunique / len(df) > 0.9:
                    # Check if it's not a continuous numeric feature
                    if df[col].dtype == "object" or (
                        df[col].dtype in ("int64", "float64")
                        and nunique == len(df)
                    ):
                        id_cols.append(col)

        return id_cols

    def _is_text_column(self, series: pd.Series) -> bool:
        """Check if a string column contains freeform text (avg length > 50 chars)."""
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        avg_len = sample.astype(str).str.len().mean()
        return avg_len > 50
