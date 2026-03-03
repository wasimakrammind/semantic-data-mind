from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


class TrendAnalyzer:
    """Detect trends, patterns, and seasonality in time series data."""

    def detect(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        decompose: bool = True,
    ) -> Dict[str, Any]:
        if date_column not in df.columns:
            return {"error": f"Column '{date_column}' not found."}
        if value_column not in df.columns:
            return {"error": f"Column '{value_column}' not found."}

        # Parse dates and sort
        ts_df = df[[date_column, value_column]].copy()
        ts_df[date_column] = pd.to_datetime(ts_df[date_column], errors="coerce")
        ts_df = ts_df.dropna().sort_values(date_column)

        if len(ts_df) < 3:
            return {"error": "Not enough data points for trend analysis."}

        result = {}

        # Basic trend (linear regression)
        x_numeric = np.arange(len(ts_df))
        y_values = ts_df[value_column].values.astype(float)
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_numeric, y_values)

        result["linear_trend"] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat",
            "is_significant": p_value < 0.05,
        }

        # Summary statistics over time
        result["time_range"] = {
            "start": str(ts_df[date_column].min()),
            "end": str(ts_df[date_column].max()),
            "data_points": len(ts_df),
        }

        result["value_summary"] = {
            "start_value": float(y_values[0]),
            "end_value": float(y_values[-1]),
            "min_value": float(y_values.min()),
            "max_value": float(y_values.max()),
            "mean": float(y_values.mean()),
            "total_change": float(y_values[-1] - y_values[0]),
            "pct_change": float((y_values[-1] - y_values[0]) / y_values[0] * 100)
            if y_values[0] != 0
            else None,
        }

        # Moving averages
        result["moving_averages"] = {}
        for window in [7, 30, 90]:
            if len(ts_df) >= window:
                ma = ts_df[value_column].rolling(window=window).mean()
                result["moving_averages"][f"{window}_period"] = {
                    "latest_value": float(ma.iloc[-1]) if not pd.isna(ma.iloc[-1]) else None,
                    "trend_vs_mean": "above" if ma.iloc[-1] > y_values.mean() else "below"
                    if not pd.isna(ma.iloc[-1])
                    else None,
                }

        # Seasonal decomposition (simple approach)
        if decompose and len(ts_df) >= 14:
            result["seasonality"] = self._detect_seasonality(ts_df, date_column, value_column)

        # Change point detection (simple variance-based)
        if len(ts_df) >= 20:
            result["change_points"] = self._detect_change_points(ts_df, date_column, value_column)

        return result

    def _detect_seasonality(
        self, ts_df: pd.DataFrame, date_col: str, value_col: str
    ) -> Dict[str, Any]:
        """Simple seasonality detection using day-of-week and monthly patterns."""
        result = {}

        ts_df = ts_df.copy()
        ts_df["_dow"] = ts_df[date_col].dt.dayofweek
        ts_df["_month"] = ts_df[date_col].dt.month

        # Day-of-week pattern
        if len(ts_df) >= 14:
            dow_means = ts_df.groupby("_dow")[value_col].mean()
            dow_std = dow_means.std()
            overall_mean = ts_df[value_col].mean()
            if overall_mean != 0 and dow_std / abs(overall_mean) > 0.05:
                result["weekly_pattern"] = {
                    "detected": True,
                    "strongest_day": int(dow_means.idxmax()),
                    "weakest_day": int(dow_means.idxmin()),
                    "variation_pct": round(float(dow_std / abs(overall_mean) * 100), 2),
                }
            else:
                result["weekly_pattern"] = {"detected": False}

        # Monthly pattern
        if len(ts_df) >= 60:
            month_means = ts_df.groupby("_month")[value_col].mean()
            month_std = month_means.std()
            if overall_mean != 0 and month_std / abs(overall_mean) > 0.1:
                result["monthly_pattern"] = {
                    "detected": True,
                    "peak_month": int(month_means.idxmax()),
                    "trough_month": int(month_means.idxmin()),
                    "variation_pct": round(float(month_std / abs(overall_mean) * 100), 2),
                }
            else:
                result["monthly_pattern"] = {"detected": False}

        return result

    def _detect_change_points(
        self, ts_df: pd.DataFrame, date_col: str, value_col: str
    ) -> list:
        """Simple change point detection using rolling mean shifts."""
        values = ts_df[value_col].values.astype(float)
        dates = ts_df[date_col].values

        window = max(5, len(values) // 10)
        change_points = []

        for i in range(window, len(values) - window):
            left_mean = values[i - window : i].mean()
            right_mean = values[i : i + window].mean()

            if left_mean != 0:
                shift_pct = abs(right_mean - left_mean) / abs(left_mean) * 100
                if shift_pct > 20:  # >20% shift
                    change_points.append({
                        "date": str(pd.Timestamp(dates[i])),
                        "shift_pct": round(float(shift_pct), 2),
                        "direction": "increase" if right_mean > left_mean else "decrease",
                    })

        # Deduplicate nearby change points (keep the strongest)
        if change_points:
            filtered = [change_points[0]]
            for cp in change_points[1:]:
                if abs(pd.Timestamp(cp["date"]) - pd.Timestamp(filtered[-1]["date"])).days > window:
                    filtered.append(cp)
                elif cp["shift_pct"] > filtered[-1]["shift_pct"]:
                    filtered[-1] = cp
            change_points = filtered[:5]

        return change_points
