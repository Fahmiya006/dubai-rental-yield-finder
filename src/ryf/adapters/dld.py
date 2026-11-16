"""Adapter: Dubai Land Department open data (Dubai Pulse) -> standard schema.

Expected inputs (CSV exports you download after registering on Dubai Pulse):
  sales : columns area_name_en, property_type_en, property_sub_type_en (optional),
          instance_date, actual_worth (AED), procedure_area (sq m)
  rents : Ejari contracts with area_name_en, ejari_property_type_en,
          contract_start_date, annual_amount (AED), actual_area (sq m)
Column names change between releases. Edit COLUMN_MAP rather than the logic.
"""
from __future__ import annotations
import pandas as pd
from ..schema import validate

SQM_TO_SQFT = 10.7639
COLUMN_MAP = {
    "sales": {"area": "area_name_en", "ptype": "property_sub_type_en", "date": "instance_date",
              "price": "actual_worth", "size": "procedure_area"},
    "rents": {"area": "area_name_en", "ptype": "ejari_property_sub_type_en", "date": "contract_start_date",
              "rent": "annual_amount", "size": "actual_area"},
}


def _monthly(df, cols, value_col, size_col):
    d = pd.DataFrame({
        "area": df[cols["area"]].astype(str).str.strip().str.title(),
        "property_type": df[cols["ptype"]].astype(str).str.strip().str.title(),
        "month": pd.to_datetime(df[cols["date"]], errors="coerce").dt.to_period("M").dt.to_timestamp(),
        "value": pd.to_numeric(df[value_col], errors="coerce"),
        "size_sqft": pd.to_numeric(df[size_col], errors="coerce") * SQM_TO_SQFT,
    }).dropna()
    return d[(d.value > 0) & (d.size_sqft > 0)]


def build(sales: pd.DataFrame, rents: pd.DataFrame, min_deals: int = 3) -> pd.DataFrame:
    s = _monthly(sales, COLUMN_MAP["sales"], COLUMN_MAP["sales"]["price"], COLUMN_MAP["sales"]["size"])
    r = _monthly(rents, COLUMN_MAP["rents"], COLUMN_MAP["rents"]["rent"], COLUMN_MAP["rents"]["size"])
    key = ["area", "property_type", "month"]
    sg = s.groupby(key).agg(avg_price=("value", "mean"), deals=("value", "size"), avg_size_sqft=("size_sqft", "mean"))
    rg = r.groupby(key).agg(avg_rent_annual=("value", "mean"), rent_contracts=("value", "size"))
    m = sg.join(rg, how="inner").reset_index()
    m = m[(m.deals >= min_deals) & (m.rent_contracts >= min_deals)]
    return validate(m)
