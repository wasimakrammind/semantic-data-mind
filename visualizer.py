from typing import Optional, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class Visualizer:
    """Generate interactive Plotly charts."""

    def create_chart(
        self,
        df: pd.DataFrame,
        chart_type: str,
        x_column: str,
        y_column: Optional[str] = None,
        color_column: Optional[str] = None,
        title: Optional[str] = None,
        aggregation: str = "none",
        **kwargs,
    ) -> Optional[Any]:
        if x_column not in df.columns:
            return None
        if y_column and y_column not in df.columns:
            return None

        # Apply aggregation if needed
        plot_df = df.copy()
        if aggregation != "none" and y_column:
            agg_func = {"sum": "sum", "mean": "mean", "count": "count", "median": "median"}
            if aggregation in agg_func:
                plot_df = df.groupby(x_column, as_index=False)[y_column].agg(agg_func[aggregation])

        title = title or f"{chart_type.title()}: {x_column}" + (f" vs {y_column}" if y_column else "")

        chart_builders = {
            "histogram": self._histogram,
            "scatter": self._scatter,
            "line": self._line,
            "bar": self._bar,
            "box": self._box,
            "heatmap": self._heatmap,
            "pie": self._pie,
        }

        builder = chart_builders.get(chart_type)
        if not builder:
            return None

        fig = builder(plot_df, x_column, y_column, color_column, title)
        if fig:
            fig.update_layout(
                template="plotly_white",
                height=450,
                margin=dict(l=60, r=30, t=50, b=60),
            )
        return fig

    def _histogram(self, df, x, y, color, title):
        return px.histogram(df, x=x, color=color, title=title, nbins=30)

    def _scatter(self, df, x, y, color, title):
        if not y:
            return None
        return px.scatter(df, x=x, y=y, color=color, title=title, opacity=0.7)

    def _line(self, df, x, y, color, title):
        if not y:
            return None
        return px.line(df, x=x, y=y, color=color, title=title)

    def _bar(self, df, x, y, color, title):
        if y:
            return px.bar(df, x=x, y=y, color=color, title=title)
        # Bar chart of value counts
        vc = df[x].value_counts().head(20).reset_index()
        vc.columns = [x, "count"]
        return px.bar(vc, x=x, y="count", title=title)

    def _box(self, df, x, y, color, title):
        if y:
            return px.box(df, x=x, y=y, color=color, title=title)
        return px.box(df, y=x, title=title)

    def _heatmap(self, df, x, y, color, title):
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) < 2:
            return None
        corr = df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
        ))
        fig.update_layout(title=title)
        return fig

    def _pie(self, df, x, y, color, title):
        if y:
            return px.pie(df, names=x, values=y, title=title)
        vc = df[x].value_counts().head(10)
        return px.pie(values=vc.values, names=vc.index.tolist(), title=title)
