"""Adapter for the datadubai.ae open dataset (built from Dubai Land Department open data, CC BY 4.0).

Inputs (from github.com/datadubai/dubai-real-estate-dld, folder data/):
  community_bedrooms.csv  current median price, rent and size per community x bedroom count
  community_monthly.csv   monthly median price per sqft per community (about 5 years)

Output: the standard panel (area, property_type, month, avg_price, avg_rent_annual, avg_size_sqft).
Honest limits, kept visible on purpose:
  * Price and rent are CURRENT medians per community x bedroom.
  * Price HISTORY exists per community only (not per bedroom). Each bedroom series is the current
    bedroom price moved by that community's monthly price-per-sqft trend, so growth is a community trend.
  * There is no rent history, so rent is held flat across months.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

BED = {"Studio": "Studio", "1 B/R": "1 Bedroom", "2 B/R": "2 Bedroom", "3 B/R": "3 Bedroom",
       "4 B/R": "4 Bedroom", "5 B/R": "5 Bedroom"}
FIX = {"DownTown Dubai": "Downtown Dubai", "Down Town Jabal Ali": "Downtown Jebel Ali",
       "DAMAC HILLS": "DAMAC Hills", "DAMAC HILLS 2": "DAMAC Hills 2"}
ATTRIBUTION = ("Dubai Data (datadubai.ae), Continental Club Property, based on Dubai Land Department "
               "open data. Licence CC BY 4.0. Not an official DLD product.")


def build(folder, min_sales: int = 30, min_rents: int = 30, min_months: int = 24):
    folder = Path(folder)
    b = pd.read_csv(folder / "community_bedrooms.csv")
    m = pd.read_csv(folder / "community_monthly.csv")
    b = b[b.code.isin(BED) & (b.quality != "insufficient")
          & b.price_median.notna() & b.rent_median.notna() & b.sqft_median.notna()
          & (b.n >= min_sales) & (b.n_rents >= min_rents)].copy()
    b["gy"] = b.rent_median / b.price_median
    b = b[(b.gy.between(0.03, 0.12))].copy()                      # drops data errors such as 94% yields
    m["month"] = pd.to_datetime(m["month"] + "-01")
    idx = {}
    for slug, g in m.groupby("community_slug"):
        s = g.sort_values("month").set_index("month")["median"]          # median AED per sqft
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
        if g["n"].median() < 10:
            continue                                                  # months too thin to trust a trend
        s = s.interpolate(limit_area="inside").rolling(3, min_periods=1, center=True).mean().dropna()
        if len(s) >= min_months:
            idx[slug] = s / s.tail(6).mean()                              # latest 6 months = 1.0
    rows = []
    for r in b.itertuples():
        name = FIX.get(r.community, r.community)
        if r.community_slug in idx:
            for d, k in idx[r.community_slug].items():
                rows.append((name, BED[r.code], d, r.price_median * k, r.rent_median, r.sqft_median))
        else:  # not enough history: one current point, growth shows as unavailable
            rows.append((name, BED[r.code], m.month.max(), r.price_median, r.rent_median, r.sqft_median))
    cols = ["area", "property_type", "month", "avg_price", "avg_rent_annual", "avg_size_sqft"]
    return pd.DataFrame(rows, columns=cols)


def data_through(folder) -> str:
    m = pd.read_csv(Path(folder) / "community_monthly.csv", usecols=["month"])
    return str(m.month.max())
