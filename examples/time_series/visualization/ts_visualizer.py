from __future__ import annotations

from typing import Optional
import pandas as pd

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore


class TSVisualizer:
    """Utility to plot time series and forecasts."""

    def plot(self, series: pd.Series, forecast: Optional[pd.Series] = None) -> None:
        if plt is None:
            raise RuntimeError("matplotlib is required for plotting")
        plt.figure()
        series.plot(label="actual")
        if forecast is not None:
            forecast.plot(label="forecast")
        plt.legend()
        plt.tight_layout()
