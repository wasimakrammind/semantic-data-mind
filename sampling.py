from typing import Optional

import pandas as pd
import numpy as np

from src.models.dataset import DatasetMetadata


class DataSampler:
    """Creates representative samples optimized for LLM context."""

    def create_llm_sample(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        max_rows: int = 50,
        strategy: str = "stratified",
    ) -> pd.DataFrame:
        if len(df) <= max_rows:
            return df.copy()

        if strategy == "stratified" and metadata.categorical_columns:
            return self._stratified_sample(df, metadata.categorical_columns[0], max_rows)
        elif strategy == "head_tail":
            return self._head_tail_sample(df, max_rows)
        else:
            return self._random_sample(df, max_rows)

    def create_statistical_summary(self, df: pd.DataFrame, metadata: DatasetMetadata) -> str:
        """Create a text summary of the dataset for the LLM system prompt."""
        lines = [
            f"DATASET CONTEXT:",
            f"- File: {metadata.filename}",
            f"- Shape: {metadata.row_count:,} rows x {metadata.column_count} columns",
            f"- Memory: {metadata.memory_usage_mb:.1f} MB",
            f"- Columns:",
        ]

        for col in metadata.column_names:
            dtype = metadata.dtypes[col]
            missing_pct = metadata.missing_summary.get(col, 0)
            col_type = self._classify_column(col, metadata)

            detail = f"  * {col} ({dtype}, {col_type}"
            if col in metadata.numeric_columns:
                series = df[col].dropna()
                if len(series) > 0:
                    detail += f", mean: {series.mean():.2f}, range: {series.min():.2f}-{series.max():.2f}"
            elif col in metadata.categorical_columns:
                nunique = df[col].nunique()
                top_vals = df[col].value_counts().head(5).index.tolist()
                detail += f", {nunique} unique, top: {top_vals}"
            elif col in metadata.datetime_columns:
                parsed = pd.to_datetime(df[col], errors="coerce").dropna()
                if len(parsed) > 0:
                    detail += f", range: {parsed.min()} to {parsed.max()}"

            if missing_pct > 0:
                detail += f", {missing_pct:.1f}% missing"
            detail += ")"
            lines.append(detail)

        # Missing data summary
        cols_with_missing = {k: v for k, v in metadata.missing_summary.items() if v > 0}
        if cols_with_missing:
            lines.append(f"- Missing Data: {len(cols_with_missing)} columns have missing values")
            for col, pct in sorted(cols_with_missing.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  * {col}: {pct:.1f}%")

        return "\n".join(lines)

    def _classify_column(self, col: str, metadata: DatasetMetadata) -> str:
        if col in metadata.identifier_columns:
            return "identifier"
        if col in metadata.datetime_columns:
            return "datetime"
        if col in metadata.numeric_columns:
            return "numeric"
        if col in metadata.categorical_columns:
            return "categorical"
        if col in metadata.text_columns:
            return "text"
        if col in metadata.boolean_columns:
            return "boolean"
        return "unknown"

    def _stratified_sample(
        self, df: pd.DataFrame, key_column: str, n: int
    ) -> pd.DataFrame:
        """Sample proportionally from each category."""
        counts = df[key_column].value_counts(normalize=True)
        samples = []
        for value, proportion in counts.items():
            group = df[df[key_column] == value]
            group_n = max(1, int(n * proportion))
            samples.append(group.sample(n=min(group_n, len(group)), random_state=42))

        result = pd.concat(samples).head(n)
        return result

    def _head_tail_sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        half = n // 2
        return pd.concat([df.head(half), df.tail(n - half)])

    def _random_sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.sample(n=n, random_state=42)
