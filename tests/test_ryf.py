import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, pytest
from ryf.costs import load_config, buying_costs, net_yield, gross_yield, mortgage_payment
from ryf.schema import validate, SchemaError
from ryf.forecast import forecast_growth, backtest
from ryf.rank import summarize
from ryf.synthetic import generate
from ryf.adapters import dld

CFG = load_config()


def test_gross_yield():
    assert gross_yield(1_000_000, 70_000) == pytest.approx(0.07)


def test_buying_costs_known_example():
    # 1,000,000: DLD 40,000 + 580 + agent 20,000*1.05=21,000 + trustee 4,000*1.05=4,200
    assert buying_costs(1_000_000, CFG) == pytest.approx(40_000 + 580 + 21_000 + 4_200)


def test_trustee_low_band():
    assert buying_costs(400_000, CFG) == pytest.approx(16_000 + 580 + 8_400 + 2_100)


def test_net_below_gross():
    r = net_yield(1_000_000, 70_000, 800, CFG)
    assert r["net_yield"] < r["gross_yield"]
    assert r["noi"] == pytest.approx(70_000 - 14 * 800 - 70_000 * (0.06 + 0.05 + 0.04))


def test_mortgage_payment():
    assert mortgage_payment(800_000, 0.05, 25) == pytest.approx(4676.6, abs=1)
    assert mortgage_payment(120, 0, 1) == pytest.approx(10)


def test_schema_rejects_missing_and_bad_rows():
    with pytest.raises(SchemaError):
        validate(pd.DataFrame({"area": ["x"]}))
    df = generate().head(5).copy()
    df.loc[0, "avg_price"] = -1
    assert len(validate(df)) == 4


def test_forecast_needs_history():
    assert np.isnan(forecast_growth(pd.Series(np.linspace(100, 110, 10))))
    up = pd.Series(100 * 1.01 ** np.arange(48))
    assert forecast_growth(up) > 0


def test_backtest_runs_time_split():
    bt = backtest(generate())
    assert bt["n"] > 0 and 0 < bt["model_mape"] < 1


def test_rank_sorted_and_complete():
    r = summarize(generate(), CFG)
    assert r.score.is_monotonic_decreasing
    assert len(r) == 14 * 3
    assert (r.net_yield < r.gross_yield).all()


def test_dld_adapter():
    n = 8
    sales = pd.DataFrame({"area_name_en": ["marina"] * n, "property_sub_type_en": ["Flat"] * n,
                          "instance_date": ["2025-03-05"] * n, "actual_worth": [1_000_000] * n,
                          "procedure_area": [80] * n})
    rents = pd.DataFrame({"area_name_en": ["Marina"] * n, "ejari_property_sub_type_en": ["flat"] * n,
                          "contract_start_date": ["2025-03-20"] * n, "annual_amount": [70_000] * n,
                          "actual_area": [80] * n})
    out = dld.build(sales, rents)
    assert len(out) == 1 and out.iloc[0].area == "Marina"
    assert out.iloc[0].avg_size_sqft == pytest.approx(80 * 10.7639)



def test_datadubai_adapter(tmp_path):
    from ryf.adapters import datadubai
    months = pd.date_range("2022-01-01", periods=30, freq="MS").strftime("%Y-%m")
    pd.DataFrame({"community": "Marina", "community_slug": "marina", "month": months, "n": 50,
                  "median": [1000 + 10 * i for i in range(30)]}).to_csv(tmp_path / "community_monthly.csv", index=False)
    pd.DataFrame([
        dict(community="Marina", community_slug="marina", code="1 B/R", n=100, quality="ok", price_median=1_000_000,
             sqft_median=700, rent_median=70_000, n_rents=100),
        dict(community="Marina", community_slug="marina", code="2 B/R", n=100, quality="ok", price_median=1_000_000,
             sqft_median=900, rent_median=940_000, n_rents=100),   # 94% yield = data error, must be dropped
        dict(community="Marina", community_slug="marina", code="3 B/R", n=5, quality="low", price_median=2_000_000,
             sqft_median=1500, rent_median=120_000, n_rents=5),    # too few deals, dropped
    ]).to_csv(tmp_path / "community_bedrooms.csv", index=False)
    out = datadubai.build(tmp_path)
    assert set(out.property_type) == {"1 Bedroom"} and len(out) == 30
    assert out.tail(6).avg_price.mean() == pytest.approx(1_000_000)   # latest 6 months anchor to the real median
    assert (out.avg_rent_annual == 70_000).all()
    r = summarize(out, CFG)
    assert 0 < r.growth_12m.iloc[0] <= 0.20
