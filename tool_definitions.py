"""Claude API tool schemas for the Data Intelligence Agent."""


def get_all_tool_definitions():
    """Return all tool definitions in Anthropic API format."""
    return [
        {
            "name": "describe_dataset",
            "description": (
                "Get a comprehensive description of the currently loaded dataset. "
                "Returns shape, column names and types, descriptive statistics, "
                "value counts for categoricals, and missing data summary. "
                "Use when the user asks general questions about their data."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "get_data_sample",
            "description": (
                "Retrieve a sample of rows from the dataset as a markdown table. "
                "Use when you need to inspect actual values or show the user examples."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "n_rows": {
                        "type": "integer",
                        "description": "Number of rows to return (default 10, max 50).",
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Column names to include. Omit for all.",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["random", "head", "tail"],
                        "description": "Sampling strategy. Default 'head'.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "validate_data",
            "description": (
                "Run comprehensive data validation. Checks for: statistical outliers, "
                "temporal inconsistencies, missing data patterns, schema issues, and "
                "cross-column logic violations. Returns a report with severity levels, "
                "affected rows, and suggested fixes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "checks": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "statistical", "temporal", "schema",
                                "cross_column", "semantic", "all",
                            ],
                        },
                        "description": "Which checks to run. Default 'all'.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "compute_statistics",
            "description": (
                "Compute descriptive statistics and distribution analysis. "
                "Returns mean, median, std, min, max, quartiles, skewness, kurtosis "
                "for numeric columns. Frequency counts for categoricals. "
                "Use when the user asks about distributions or statistical properties."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to analyze. Omit for all numeric.",
                    },
                    "group_by": {
                        "type": "string",
                        "description": "Optional column to group statistics by.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "compute_correlations",
            "description": (
                "Compute correlation analysis between columns. "
                "Supports Pearson, Spearman, and Kendall methods. "
                "Returns correlation matrix and highlights strongest relationships. "
                "Use when the user asks about relationships between variables."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to include. Omit for all numeric.",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["pearson", "spearman", "kendall"],
                        "description": "Correlation method. Default 'pearson'.",
                    },
                    "min_threshold": {
                        "type": "number",
                        "description": "Minimum |correlation| to report. Default 0.3.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "detect_trends",
            "description": (
                "Detect trends, patterns, and seasonality in time series data. "
                "Performs linear trend detection, moving averages, and seasonal decomposition. "
                "Use when the user asks about trends over time or seasonal patterns."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "date_column": {
                        "type": "string",
                        "description": "Name of the datetime column.",
                    },
                    "value_column": {
                        "type": "string",
                        "description": "Name of the numeric column to analyze.",
                    },
                    "decompose": {
                        "type": "boolean",
                        "description": "Whether to perform seasonal decomposition. Default true.",
                    },
                },
                "required": ["date_column", "value_column"],
            },
        },
        {
            "name": "create_visualization",
            "description": (
                "Create an interactive Plotly chart. Supports: histogram, scatter, "
                "line, bar, box, heatmap, pie. Always create a visualization when "
                "discussing distributions, trends, or comparisons."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": [
                            "histogram", "scatter", "line", "bar",
                            "box", "heatmap", "pie",
                        ],
                        "description": "Type of chart to create.",
                    },
                    "x_column": {
                        "type": "string",
                        "description": "Column for x-axis.",
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Column for y-axis (not needed for histogram/pie).",
                    },
                    "color_column": {
                        "type": "string",
                        "description": "Optional column for color coding.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title.",
                    },
                    "aggregation": {
                        "type": "string",
                        "enum": ["none", "sum", "mean", "count", "median"],
                        "description": "Aggregation for y values. Default 'none'.",
                    },
                },
                "required": ["chart_type", "x_column"],
            },
        },
        {
            "name": "detect_outliers",
            "description": (
                "Detect outliers in numeric columns using Z-score, IQR, or both. "
                "Returns outlier counts, row indices, and values. "
                "Use when the user asks about unusual values or anomalies."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to check. Omit for all numeric.",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["zscore", "iqr", "both"],
                        "description": "Detection method. Default 'iqr'.",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Z-score threshold (default 3.0) or IQR multiplier (default 1.5).",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "analyze_missing_data",
            "description": (
                "Analyze missing data patterns. Returns per-column counts, "
                "missing data correlations, and suggests imputation strategies. "
                "Use when the user asks about missing values or data completeness."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to analyze. Omit for all with missing data.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "compute_feature_importance",
            "description": (
                "Compute feature importance relative to a target variable using "
                "mutual information and correlation. "
                "Use when the user asks what drives or influences a metric."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "target_column": {
                        "type": "string",
                        "description": "The target variable.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top features to return. Default 10.",
                    },
                },
                "required": ["target_column"],
            },
        },
        {
            "name": "run_custom_analysis",
            "description": (
                "Execute a custom pandas analysis described in natural language. "
                "Supports groupby, pivot, filtering, sorting, and calculated columns. "
                "Use as a fallback when other tools don't cover the specific analysis needed."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Natural language description of the pandas operation.",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["table", "scalar", "series"],
                        "description": "Expected output format. Default 'table'.",
                    },
                },
                "required": ["operation"],
            },
        },
    ]
