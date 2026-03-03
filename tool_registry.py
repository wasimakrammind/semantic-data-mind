from typing import Callable, Dict, Any, List, Optional
import json

import pandas as pd
import numpy as np

from src.models.dataset import Dataset
from src.models.analysis_result import EDAResult
from src.agent.conversation import ConversationState
from src.utils.serialization import dataframe_to_markdown, result_to_json


class ToolRegistry:
    """Central registry: maps tool names to handler functions."""

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}

    def register(self, name: str, handler: Callable) -> None:
        self._handlers[name] = handler

    def execute(
        self, tool_name: str, tool_input: Dict[str, Any], session: ConversationState
    ) -> str:
        """Execute a tool and return string result for the LLM."""
        if tool_name not in self._handlers:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            result = self._handlers[tool_name](tool_input, session)
            return result if isinstance(result, str) else json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "tool": tool_name})


def create_tool_registry(anthropic_client=None) -> ToolRegistry:
    """Create and populate the tool registry with all handlers."""
    registry = ToolRegistry()

    registry.register("describe_dataset", _handle_describe_dataset)
    registry.register("get_data_sample", _handle_get_data_sample)
    registry.register("validate_data", _handle_validate_data_factory(anthropic_client))
    registry.register("compute_statistics", _handle_compute_statistics)
    registry.register("compute_correlations", _handle_compute_correlations)
    registry.register("detect_trends", _handle_detect_trends)
    registry.register("create_visualization", _handle_create_visualization)
    registry.register("detect_outliers", _handle_detect_outliers)
    registry.register("analyze_missing_data", _handle_analyze_missing_data)
    registry.register("compute_feature_importance", _handle_compute_feature_importance)
    registry.register("run_custom_analysis", _handle_custom_analysis_factory(anthropic_client))

    return registry


# --- Tool Handlers ---


def _handle_describe_dataset(tool_input: Dict, session: ConversationState) -> str:
    if not session.dataset:
        return json.dumps({"error": "No dataset loaded. Please upload a file first."})

    ds = session.dataset
    m = ds.metadata
    df = ds.df

    result = {
        "filename": m.filename,
        "shape": {"rows": m.row_count, "columns": m.column_count},
        "memory_mb": m.memory_usage_mb,
        "columns": {},
    }

    for col in m.column_names:
        col_info = {"dtype": m.dtypes[col], "missing_pct": m.missing_summary.get(col, 0)}
        if col in m.numeric_columns:
            desc = df[col].describe()
            col_info["stats"] = {
                "mean": float(desc["mean"]),
                "std": float(desc["std"]),
                "min": float(desc["min"]),
                "25%": float(desc["25%"]),
                "50%": float(desc["50%"]),
                "75%": float(desc["75%"]),
                "max": float(desc["max"]),
            }
        elif col in m.categorical_columns:
            vc = df[col].value_counts().head(10)
            col_info["top_values"] = {str(k): int(v) for k, v in vc.items()}
            col_info["unique_count"] = int(df[col].nunique())

        result["columns"][col] = col_info

    return result_to_json(result)


def _handle_get_data_sample(tool_input: Dict, session: ConversationState) -> str:
    if not session.dataset:
        return json.dumps({"error": "No dataset loaded."})

    df = session.dataset.df
    n_rows = min(tool_input.get("n_rows", 10), 50)
    columns = tool_input.get("columns")
    strategy = tool_input.get("strategy", "head")

    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            return json.dumps({"error": f"Columns not found: {missing}"})
        df = df[columns]

    if strategy == "random":
        sample = df.sample(n=min(n_rows, len(df)), random_state=42)
    elif strategy == "tail":
        sample = df.tail(n_rows)
    else:
        sample = df.head(n_rows)

    return dataframe_to_markdown(sample, max_rows=n_rows)


def _handle_validate_data_factory(anthropic_client):
    def handler(tool_input: Dict, session: ConversationState) -> str:
        if not session.dataset:
            return json.dumps({"error": "No dataset loaded."})

        from src.validation.validator import DataValidator

        checks = tool_input.get("checks", ["all"])
        validator = DataValidator()
        report = validator.validate(session.dataset, checks=checks, anthropic_client=anthropic_client)
        session.validation_report = report

        result = {
            "data_quality_score": report.data_quality_score,
            "summary": report.summary,
            "finding_count": len(report.findings),
            "errors": report.error_count,
            "warnings": report.warning_count,
            "findings": [
                {
                    "severity": f.severity,
                    "category": f.category,
                    "column": f.column,
                    "message": f.message,
                    "suggestion": f.suggestion,
                }
                for f in report.findings[:20]
            ],
        }
        return result_to_json(result)

    return handler


def _handle_compute_statistics(tool_input: Dict, session: ConversationState) -> str:
    if not session.dataset:
        return json.dumps({"error": "No dataset loaded."})

    from src.eda.statistics import StatisticsCalculator

    calc = StatisticsCalculator()
    columns = tool_input.get("columns")
    group_by = tool_input.get("group_by")

    result = calc.compute(session.dataset.df, session.dataset.metadata, columns, group_by)
    return result_to_json(result)


def _handle_compute_correlations(tool_input: Dict, session: ConversationState) -> str:
    if not session.dataset:
        return json.dumps({"error": "No dataset loaded."})

    from src.eda.correlations import CorrelationAnalyzer

    analyzer = CorrelationAnalyzer()
    columns = tool_input.get("columns")
    method = tool_input.get("method", "pearson")
    threshold = tool_input.get("min_threshold", 0.3)

    result = analyzer.compute(session.dataset.df, session.dataset.metadata, columns, method, threshold)
    return result_to_json(result)


def _handle_detect_trends(tool_input: Dict, session: ConversationState) -> str:
    if not session.dataset:
        return json.dumps({"error": "No dataset loaded."})

    from src.eda.trends import TrendAnalyzer

    analyzer = TrendAnalyzer()
    result = analyzer.detect(
        session.dataset.df,
        date_column=tool_input["date_column"],
        value_column=tool_input["value_column"],
        decompose=tool_input.get("decompose", True),
    )
    return result_to_json(result)


def _handle_create_visualization(tool_input: Dict, session: ConversationState) -> str:
    if not session.dataset:
        return json.dumps({"error": "No dataset loaded."})

    from src.eda.visualizer import Visualizer

    viz = Visualizer()
    chart = viz.create_chart(session.dataset.df, **tool_input)
    if chart:
        session.charts.append(chart)
        return json.dumps({"status": "Chart created successfully", "chart_type": tool_input["chart_type"]})
    return json.dumps({"error": "Could not create chart with given parameters."})


def _handle_detect_outliers(tool_input: Dict, session: ConversationState) -> str:
    if not session.dataset:
        return json.dumps({"error": "No dataset loaded."})

    from src.eda.statistics import StatisticsCalculator

    calc = StatisticsCalculator()
    columns = tool_input.get("columns")
    method = tool_input.get("method", "iqr")
    threshold = tool_input.get("threshold")

    result = calc.detect_outliers(
        session.dataset.df, session.dataset.metadata, columns, method, threshold
    )
    return result_to_json(result)


def _handle_analyze_missing_data(tool_input: Dict, session: ConversationState) -> str:
    if not session.dataset:
        return json.dumps({"error": "No dataset loaded."})

    from src.eda.statistics import StatisticsCalculator

    calc = StatisticsCalculator()
    columns = tool_input.get("columns")
    result = calc.analyze_missing(session.dataset.df, session.dataset.metadata, columns)
    return result_to_json(result)


def _handle_compute_feature_importance(tool_input: Dict, session: ConversationState) -> str:
    if not session.dataset:
        return json.dumps({"error": "No dataset loaded."})

    from src.eda.feature_importance import FeatureImportanceAnalyzer

    analyzer = FeatureImportanceAnalyzer()
    result = analyzer.compute(
        session.dataset.df,
        session.dataset.metadata,
        target_column=tool_input["target_column"],
        top_n=tool_input.get("top_n", 10),
    )
    return result_to_json(result)


def _handle_custom_analysis_factory(anthropic_client):
    def handler(tool_input: Dict, session: ConversationState) -> str:
        if not session.dataset:
            return json.dumps({"error": "No dataset loaded."})

        if not anthropic_client:
            return json.dumps({"error": "Custom analysis requires an LLM connection."})

        operation = tool_input["operation"]
        df = session.dataset.df
        metadata = session.dataset.metadata

        # Ask Claude to generate pandas code
        code_prompt = f"""Given a pandas DataFrame `df` with these columns:
{json.dumps({c: metadata.dtypes[c] for c in metadata.column_names}, indent=2)}

Write a single Python expression or short code block that performs this operation:
"{operation}"

Rules:
- The variable `df` is already available as a pandas DataFrame
- You can use `pd` (pandas) and `np` (numpy)
- No imports allowed
- No file I/O or network access
- Assign the final result to a variable called `result`
- Return ONLY the Python code, no markdown, no explanation"""

        try:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": code_prompt}],
            )
            code = response.content[0].text.strip()
            # Strip markdown code fences if present
            if code.startswith("```"):
                code = "\n".join(code.split("\n")[1:])
            if code.endswith("```"):
                code = "\n".join(code.split("\n")[:-1])

            # Execute in restricted scope
            local_scope = {"df": df.copy(), "pd": pd, "np": np, "result": None}
            exec(code, {"__builtins__": {}}, local_scope)

            result = local_scope.get("result")
            if result is None:
                return json.dumps({"error": "Code did not produce a result."})

            if isinstance(result, pd.DataFrame):
                return dataframe_to_markdown(result, max_rows=30)
            elif isinstance(result, pd.Series):
                return result.to_string()
            else:
                return str(result)

        except Exception as e:
            return json.dumps({"error": f"Custom analysis failed: {e}"})

    return handler
