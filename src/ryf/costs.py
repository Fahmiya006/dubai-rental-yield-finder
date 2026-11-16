"""UAE buying costs and net-yield maths. All money in AED."""
from __future__ import annotations
import yaml
from pathlib import Path

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "config" / "uae.yaml"


def load_config(path=DEFAULT_CFG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def buying_costs(price: float, cfg: dict, financed: bool = False, loan: float = 0.0) -> float:
    b = cfg["buying_costs"]
    agent = price * b["agent_pct"] * (1 + b["agent_vat_pct"])
    trustee_base = b["trustee_fee_high"] if price >= b["trustee_threshold"] else b["trustee_fee_low"]
    trustee = trustee_base * (1 + b["trustee_vat_pct"])
    total = price * b["dld_transfer_pct"] + b["dld_admin_fee"] + agent + trustee
    if financed:
        total += loan * b["mortgage_registration_pct"]
    return total


def gross_yield(price: float, annual_rent: float) -> float:
    return annual_rent / price


def net_yield(price: float, annual_rent: float, size_sqft: float, cfg: dict,
              service_charge_sqft: float | None = None) -> dict:
    r = cfg["running_costs"]
    sc = (service_charge_sqft if service_charge_sqft is not None else r["service_charge_aed_sqft"]) * size_sqft
    vacancy = annual_rent * r["vacancy_pct"]
    mgmt = annual_rent * r["management_pct"]
    maint = annual_rent * r["maintenance_pct_of_rent"]
    noi = annual_rent - sc - vacancy - mgmt - maint
    buy = buying_costs(price, cfg)
    invested = price + buy
    return {
        "gross_yield": gross_yield(price, annual_rent),
        "net_yield": noi / invested,
        "noi": noi,
        "buying_costs": buy,
        "buying_costs_pct": buy / price,
        "service_charge": sc,
        "payback_years": invested / noi if noi > 0 else float("inf"),
    }


def mortgage_payment(loan: float, rate: float, years: int) -> float:
    i = rate / 12
    n = years * 12
    return loan / n if i == 0 else loan * i / (1 - (1 + i) ** -n)
