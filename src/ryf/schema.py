"""Standard schema every country/source adapter must produce."""
from __future__ import annotations
import pandas as pd

REQUIRED = ["area", "property_type", "month", "avg_price", "avg_rent_annual", "avg_size_sqft"]


class SchemaError(ValueError):
    pass


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    out = df.copy()
    out["month"] = pd.to_datetime(out["month"]).dt.to_period("M").dt.to_timestamp()
    for c in ["avg_price", "avg_rent_annual", "avg_size_sqft"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    bad = out[REQUIRED].isna().any(axis=1) | (out["avg_price"] <= 0) | (out["avg_rent_annual"] <= 0) | (out["avg_size_sqft"] <= 0)
    return out.loc[~bad].reset_index(drop=True)
