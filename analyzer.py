from typing import Dict, Any, List

from src.models.dataset import Dataset
from src.models.analysis_result import EDAResult
from src.eda.statistics import StatisticsCalculator
from src.eda.correlations import CorrelationAnalyzer
from src.eda.trends import TrendAnalyzer
from src.eda.feature_importance import FeatureImportanceAnalyzer
from src.eda.visualizer import Visualizer


class EDAAnalyzer:
    """Orchestrates automated EDA operations."""

    def __init__(self):
        self.stats = StatisticsCalculator()
        self.correlations = CorrelationAnalyzer()
        self.trends = TrendAnalyzer()
        self.features = FeatureImportanceAnalyzer()
        self.visualizer = Visualizer()

    def auto_eda(self, dataset: Dataset) -> EDAResult:
        """Run a comprehensive automated EDA pipeline."""
        df = dataset.df
        m = dataset.metadata
        results = {}
        charts = []

        # 1. Descriptive statistics
        results["statistics"] = self.stats.compute(df, m)

        # 2. Correlations (if enough numeric columns)
        if len(m.numeric_columns) >= 2:
            results["correlations"] = self.correlations.compute(df, m)

            # Correlation heatmap
            heatmap = self.visualizer.create_chart(
                df, "heatmap", m.numeric_columns[0],
                title="Correlation Heatmap"
            )
            if heatmap:
                charts.append(heatmap)

        # 3. Distributions for top numeric columns
        for col in m.numeric_columns[:5]:
            hist = self.visualizer.create_chart(
                df, "histogram", col,
                title=f"Distribution of {col}"
            )
            if hist:
                charts.append(hist)

        # 4. Trend analysis (if datetime column exists)
        if m.datetime_columns and m.numeric_columns:
            date_col = m.datetime_columns[0]
            value_col = m.numeric_columns[0]
            results["trend"] = self.trends.detect(df, date_col, value_col)

            line_chart = self.visualizer.create_chart(
                df, "line", date_col, value_col,
                title=f"{value_col} over time"
            )
            if line_chart:
                charts.append(line_chart)

        # 5. Categorical breakdowns
        for col in m.categorical_columns[:3]:
            bar = self.visualizer.create_chart(
                df, "bar", col,
                title=f"Distribution of {col}"
            )
            if bar:
                charts.append(bar)

        summary = self._generate_summary(results, m)

        return EDAResult(
            analysis_type="auto_eda",
            columns_analyzed=m.column_names,
            results=results,
            summary=summary,
            charts=charts,
        )

    def _generate_summary(self, results: Dict, metadata) -> str:
        parts = ["Automated EDA Summary:"]

        # Stats summary
        if "statistics" in results:
            stats = results["statistics"].get("columns", {})
            parts.append(f"- Analyzed {len(stats)} columns")

        # Correlation highlights
        if "correlations" in results:
            sig = results["correlations"].get("significant_correlations", [])
            if sig:
                top = sig[0]
                parts.append(
                    f"- Strongest correlation: {top['column_1']} ↔ {top['column_2']} "
                    f"({top['correlation']:.2f}, {top['strength']})"
                )

        # Trend summary
        if "trend" in results:
            trend = results["trend"].get("linear_trend", {})
            direction = trend.get("direction", "unknown")
            r2 = trend.get("r_squared", 0)
            parts.append(f"- Overall trend: {direction} (R²={r2:.2f})")

        return "\n".join(parts)
