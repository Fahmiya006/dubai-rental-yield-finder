"""CLI: python -m ryf.pipeline [--csv path] [--out data/]
Without --csv it uses SYNTHETIC demo data. Writes ranking.csv and app_data.json."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from .costs import load_config
from .forecast import backtest
from .rank import summarize
from .schema import validate
from .synthetic import generate


def run(panel: pd.DataFrame, out: Path, synthetic: bool, source: dict | None = None):
    cfg = load_config()
    panel = validate(panel)
    rank = summarize(panel, cfg)
    bt = backtest(panel)
    out.mkdir(parents=True, exist_ok=True)
    rank.to_csv(out / "ranking.csv", index=False)
    series = {}
    for (a, t), g in panel.groupby(["area", "property_type"]):
        g = g.sort_values("month").iloc[::3]  # quarterly points keep the app light
        series[f"{a}|{t}"] = {"m": [d.strftime("%Y-%m") for d in g.month],
                              "p": [round(x) for x in g.avg_price],
                              "r": [round(x) for x in g.avg_rent_annual]}
    payload = {"synthetic": synthetic, "source": source, "config": cfg, "backtest": bt,
               "rows": json.loads(rank.round(5).to_json(orient="records")), "series": series}
    (out / "app_data.json").write_text(json.dumps(payload, separators=(",", ":")))
    return rank, bt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--datadubai", help="folder with community_bedrooms.csv and community_monthly.csv")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    src = None
    if a.datadubai:
        from .adapters import datadubai
        df, syn = datadubai.build(a.datadubai), False
        src = {"name": "datadubai.ae", "text": datadubai.ATTRIBUTION,
               "through": datadubai.data_through(a.datadubai)}
    elif a.csv:
        df, syn = pd.read_csv(a.csv), False
    else:
        df, syn = generate(), True
    rank, bt = run(df, Path(a.out), syn, src)
    print(rank.head(10)[["area", "property_type", "gross_yield", "net_yield", "growth_12m", "score"]].round(3))
    print("backtest:", bt)
