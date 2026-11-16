"""Price-growth forecast with an honest baseline comparison.

Model: damped log-linear trend fitted on the most recent window of monthly prices.
Baseline: naive trailing 12-month growth carried forward.
Backtest: time-based split (never random). Train on data up to the cut, predict the next
`horizon` months, compare absolute error of both methods.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _fit_predict(prices: np.ndarray, horizon: int, window: int = 36, damp: float = 0.97) -> np.ndarray:
    y = np.log(prices[-window:])
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    out, level, s = [], y[-1], slope
    for _ in range(horizon):
        s *= damp
        level += s
        out.append(np.exp(level))
    return np.array(out)


def _naive(prices: np.ndarray, horizon: int) -> np.ndarray:
    g = (prices[-1] / prices[-13]) ** (1 / 12) if len(prices) > 12 else 1.0
    return np.array([prices[-1] * g ** (k + 1) for k in range(horizon)])


def forecast_growth(series: pd.Series, horizon: int = 12) -> float:
    """Expected % price change over `horizon` months."""
    p = series.dropna().to_numpy(dtype=float)
    if len(p) < 24:
        return float("nan")
    return float(_fit_predict(p, horizon)[-1] / p[-1] - 1)


def backtest(panel: pd.DataFrame, horizon: int = 12, cut_months_back: int = 12) -> dict:
    """panel: standard schema. Returns MAE% of model vs naive on the held-out final months."""
    errs_m, errs_b = [], []
    for _, g in panel.groupby(["area", "property_type"]):
        s = g.sort_values("month")["avg_price"].to_numpy(dtype=float)
        if len(s) < 24 + cut_months_back:
            continue
        train, test = s[:-cut_months_back], s[-cut_months_back:][:horizon]
        h = len(test)
        errs_m.append(np.mean(np.abs(_fit_predict(train, h) - test) / test))
        errs_b.append(np.mean(np.abs(_naive(train, h) - test) / test))
    if not errs_m:
        return {"n": 0}
    m, b = float(np.mean(errs_m)), float(np.mean(errs_b))
    return {"n": len(errs_m), "model_mape": m, "baseline_mape": b, "beats_baseline": m < b}
