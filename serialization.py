import json
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 30) -> str:
    """Convert a DataFrame to a compact markdown table for LLM consumption."""
    if len(df) > max_rows:
        display_df = pd.concat([df.head(max_rows // 2), df.tail(max_rows // 2)])
        truncated = True
    else:
        display_df = df
        truncated = False

    result = display_df.to_markdown(index=False)
    if truncated:
        result += f"\n\n... ({len(df) - max_rows} rows omitted) ..."
    return result


def dataframe_to_csv_string(df: pd.DataFrame, max_rows: int = 30) -> str:
    """Convert a DataFrame to a CSV string for LLM consumption."""
    if len(df) > max_rows:
        display_df = pd.concat([df.head(max_rows // 2), df.tail(max_rows // 2)])
    else:
        display_df = df
    return display_df.to_csv(index=False)


def make_json_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if pd.isna(obj):
        return None
    return obj


def result_to_json(data: Dict[str, Any]) -> str:
    """Serialize analysis results to a JSON string for LLM tool responses."""
    clean_data = make_json_serializable(data)
    return json.dumps(clean_data, indent=2, default=str)
