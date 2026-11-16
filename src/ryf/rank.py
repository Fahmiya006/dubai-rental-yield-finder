"""Combine yield + growth + data confidence into an 'invest here' ranking."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .costs import net_yield
from .forecast import forecast_growth


def summarize(panel: pd.DataFrame, cfg: dict, horizon: int = 12) -> pd.DataFrame:
    rows = []
    for (area, ptype), g in panel.groupby(["area", "property_type"]):
        g = g.sort_values("month")
        last = g.tail(6)  # smooth over the latest 6 months
        price, rent, size = last.avg_price.mean(), last.avg_rent_annual.mean(), last.avg_size_sqft.mean()
        ny = net_yield(price, rent, size, cfg)
        growth = forecast_growth(g.set_index("month")["avg_price"], horizon)
        growth = float(np.clip(growth, -0.20, 0.20)) if growth == growth else growth  # cap extrapolation
        vol = g["avg_price"].pct_change().tail(24).std()
        rows.append({"area": area, "property_type": ptype, "price": price, "rent": rent, "size_sqft": size,
                     "gross_yield": ny["gross_yield"], "net_yield": ny["net_yield"],
                     "payback_years": ny["payback_years"], "growth_12m": growth,
                     "volatility": vol, "months": len(g)})
    df = pd.DataFrame(rows)
    # total return = net yield + HALF the forecast growth (growth is a noisy trend estimate); risk-adjust by volatility
    df["total_return"] = df.net_yield + 0.5 * df.growth_12m.fillna(0)
    z = lambda s: (s - s.mean()) / (s.std() or 1)
    df["score"] = (0.6 * z(df.total_return) - 0.2 * z(df.volatility.fillna(df.volatility.mean()))
                   + 0.2 * z(df.months.clip(upper=60)))
    return df.sort_values("score", ascending=False).reset_index(drop=True)
