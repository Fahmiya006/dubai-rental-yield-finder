"""SYNTHETIC demo data. Invented numbers shaped like Dubai's market. NOT real DLD data.
Used so the app and tests run without a Dubai Pulse key."""
from __future__ import annotations
import numpy as np
import pandas as pd

# area: (price AED/sqft in 2019, rent yield 2019, annual drift, vol, tier)
AREAS = {
    "Jumeirah Village Circle": (700, 0.075, 0.115, 0.012),
    "Dubai Marina": (1250, 0.063, 0.085, 0.010),
    "Downtown Dubai": (1650, 0.056, 0.090, 0.011),
    "Business Bay": (1100, 0.066, 0.100, 0.012),
    "Dubai Sports City": (620, 0.080, 0.095, 0.014),
    "Dubai Silicon Oasis": (640, 0.078, 0.105, 0.013),
    "International City": (430, 0.090, 0.120, 0.016),
    "Arjan": (720, 0.074, 0.110, 0.014),
    "Jumeirah Lake Towers": (1000, 0.068, 0.088, 0.010),
    "Palm Jumeirah": (2200, 0.050, 0.110, 0.013),
    "Dubai Hills Estate": (1350, 0.058, 0.115, 0.012),
    "Al Furjan": (780, 0.072, 0.100, 0.013),
    "Dubai Creek Harbour": (1500, 0.058, 0.105, 0.015),
    "Meydan": (1100, 0.062, 0.105, 0.015),
}
TYPES = {"Studio": 420, "1 Bedroom": 780, "2 Bedroom": 1280}  # avg sqft
TYPE_YIELD_ADJ = {"Studio": 1.06, "1 Bedroom": 1.0, "2 Bedroom": 0.94}


def generate(seed: int = 7, start="2019-01-01", end="2026-06-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range(start, end, freq="MS")
    rows = []
    for area, (p0, y0, drift, vol) in AREAS.items():
        area_shock = rng.normal(0, 0.003)
        for t, tadj in TYPE_YIELD_ADJ.items():
            size = TYPES[t] * rng.uniform(0.95, 1.05)
            lp, lr = np.log(p0 * rng.uniform(0.96, 1.04)), None
            price_psf = []
            for i, m in enumerate(months):
                boost = 1.0 if 2021 <= m.year <= 2024 else 0.35  # synthetic boom
                lp += (drift / 12) * boost + area_shock + rng.normal(0, vol)
                price_psf.append(np.exp(lp))
            price_psf = np.array(price_psf)
            # rents lag prices and grow slower, so yields compress in the boom
            rent_psf = p0 * y0 * tadj * np.exp(np.cumsum(np.full(len(months), 0.085 / 12) + rng.normal(0, 0.006, len(months))))
            for m, pp, rp in zip(months, price_psf, rent_psf):
                rows.append((area, t, m, pp * size, rp * size, size))
    return pd.DataFrame(rows, columns=["area", "property_type", "month", "avg_price", "avg_rent_annual", "avg_size_sqft"])
