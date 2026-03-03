from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import pandas as pd


@dataclass
class DatasetMetadata:
    filename: str
    file_format: str
    row_count: int
    column_count: int
    column_names: List[str]
    dtypes: Dict[str, str]
    memory_usage_mb: float
    has_datetime_index: bool
    categorical_columns: List[str]
    numeric_columns: List[str]
    datetime_columns: List[str]
    text_columns: List[str]
    identifier_columns: List[str]
    boolean_columns: List[str]
    missing_summary: Dict[str, float]  # column -> % missing


@dataclass
class Dataset:
    df: pd.DataFrame
    metadata: DatasetMetadata
    sample_for_llm: Optional[pd.DataFrame] = None
    profile_summary: Optional[str] = None
    validation_results: Optional[List[Any]] = field(default_factory=list)
    eda_results: Optional[Dict[str, Any]] = field(default_factory=dict)
