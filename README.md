# UAE Rental Yield Finder

Find where a Dubai property pays the best rent for the price, after real UAE costs, and where prices are likely to grow.
Built for buyers and investors (including expats and overseas Indian buyers). Sellers and agents can use the listing check to price a flat.

> The bundled data is **real**: median sale prices, rents and sizes per Dubai community and bedroom count, from the [datadubai.ae](https://datadubai.ae) open dataset built on Dubai Land Department data (CC BY 4.0, data through 2026-09/10). The old invented demo data is still available through `python -m ryf.pipeline` with no flags.
>
> Credit: *Dubai Data (datadubai.ae), Continental Club Property, based on Dubai Land Department open data. Not an official DLD product.*

## What it does
1. **Adapters** turn raw data (DLD sales, Ejari rent contracts, any CSV) into one standard table: `area, property_type, month, avg_price, avg_rent_annual, avg_size_sqft`.
2. **Costs** (`config/uae.yaml`) apply DLD transfer fee, agent fee + VAT, trustee fee, service charge, vacancy, management and maintenance.
3. **Yield**: gross = rent / price. Net = (rent − running costs) / (price + buying costs).
4. **Forecast**: damped log-linear trend, tested against a naive baseline with a time-based split (`ryf.forecast.backtest`).
5. **Rank**: net yield + forecast growth, adjusted for volatility and history length, gives the "invest here" list.
6. **App** (`app/index.html`): phone-style app with login, ranking, explorer, calculator, listing check and an AI assistant.

## Run it end to end (accounts, saved data, AI agent)
```bash
python server.py                          # only needs Python 3.9+, no pip install
# optional, enables the AI agent:
export ANTHROPIC_API_KEY=sk-ant-...       # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
python server.py                          # open http://localhost:8000
```
- Sign up, log in and log out. Accounts are stored in `data/app.db` (SQLite) with PBKDF2-hashed passwords and HttpOnly session cookies.
- Saved per user and fetched on every login: profile, watchlist, calculator inputs, settings (INR, theme), chat history and imported data.
- Phone: set `HOST=0.0.0.0`, open `http://<your-computer-ip>:8000` on the same Wi-Fi, then "Add to Home screen".
- Deploy for real: run behind HTTPS, set `COOKIE_SECURE=1`, and keep `data/` on a persistent disk (see `Dockerfile`).
- Back up `data/app.db` to keep user data.

## Analysis pipeline and tests
```bash
pip install pandas numpy pyyaml pytest
python -m pytest                          # 29 tests: maths, forecast, adapters, server and accounts
PYTHONPATH=src python -m ryf.pipeline     # demo data -> data/ranking.csv, data/app_data.json
PYTHONPATH=src python -m ryf.pipeline --csv my_panel.csv --out data/
python app/build.py                       # embeds data/app_data.json into app/index.html
```
Without `server.py` (for example opened as a plain file or published as an artifact) the app falls back to a browser-only demo login.

## AI agent
The assistant calls tools on the app's data (search, compare, yield, costs, mortgage, listing check, watchlist, open screens). The browser runs the tools, `server.py` relays to the Anthropic API so your key stays on the server. Logged-in users only, 20 questions per minute each. Tools live in `app/template.html` under `TOOLS`.

## Mini-GPT: a language model written from scratch
`llm/tinygpt.py` is a small GPT in plain NumPy, with the backward pass written by hand (no PyTorch, no libraries):
token and position embeddings, pre-LayerNorm transformer blocks, causal multi-head self-attention, a GELU MLP, softmax cross-entropy, Adam and sampling.
`tests/test_tinygpt.py` checks every gradient against finite differences and checks that attention cannot see the future.

How it is used: the app finds the matching rows (its normal search), writes them as a `F: ...` facts line, and the model writes the sentence.
The app then checks that every number in the reply appears in the facts, and falls back to the built-in helper if not.
The model is small (about 180,000 parameters) and cannot memorise 229 rows, so the task is copying and phrasing, which small models do well.

```bash
python3 -m llm.train --steps 2400 --holdout 0.15 --out /tmp/stage1.npz            # stage 1, tested on areas it never saw
python3 -m llm.train --init /tmp/stage1.npz --steps 3500 --lr 3.5e-3 --holdout 0.15   # stage 2 -> llm/model.npz (the app uses this)
python3 server.py                                                       # Assistant tab -> "Mini-GPT (my own model)"
```
Training takes about 30 to 60 minutes per stage on a laptop CPU. Needs `numpy` and `pandas` only. Retrain after you rebuild `data/ranking.csv`.
Measured result: on 38 real rows from areas held out of training, 75% of answers match the expected sentence exactly and 90% have every number copied correctly. The app's number check catches most of the rest and falls back to the built-in helper.
Honest limits: it only answers about one area plus bedroom type, or "best yield"; it is not a general chatbot.

## Deploy (Render, from GitHub)
Fastest: **Blueprint**. `render.yaml` (free) is in the repo: Render > New > Blueprint > pick the repo > Apply. Use `render.paid.yaml` for permanent accounts.
Or by hand:
1. Push this folder to a GitHub repo (`git init`, `git add .`, `git commit -m "first"`, create the repo, `git push`).
2. On render.com: New > Web Service > pick the repo. Runtime: **Docker**. Render sets `PORT` for you.
3. Environment variables: `COOKIE_SECURE=1`. Optional: `ANTHROPIC_API_KEY` (Claude-powered assistant). The Mini-GPT needs no key.
4. Accounts live in SQLite at `RYF_DB` (default `/data/app.db`). Free web services have no persistent disk, so on free hosting set `RYF_DB=/tmp/app.db` and accept that accounts reset when the service restarts. For real users add a persistent disk mounted at `/data` (paid) or use Fly.io with a volume.
5. Open the `https://...onrender.com` link on your phone and use "Add to Home screen".

## Use real data
**Rebuild from the datadubai files** (`community_bedrooms.csv`, `community_monthly.csv` from github.com/datadubai/dubai-real-estate-dld):
```bash
PYTHONPATH=src python -m ryf.pipeline --datadubai data/source --out data && python app/build.py
```
How it is built: rows with fewer than 30 sales or 30 rent contracts, or a gross yield outside 3-12%, are dropped. Price history exists per community, not per bedroom, so each bedroom series follows its community's price-per-sqft trend. There is no rent history, so rent is held flat. Growth forecasts are capped at ±20% and count half in the ranking (net yield + 0.5 × growth).

**Or use the official source directly:**
1. Register on [Dubai Pulse](https://www.dubaipulse.gov.ae) and download the DLD transactions and Ejari rent contracts datasets. Check the "last updated" date on each.
2. Load both CSVs and call `ryf.adapters.dld.build(sales, rents)`. If column names changed, edit `COLUMN_MAP` in `adapters/dld.py`.
3. Save the result as CSV and run the pipeline with `--csv`.
4. Or open the app → Profile → Import CSV with the standard columns.

## Add another country
Write an adapter that returns the standard schema and copy `config/uae.yaml` with that country's fees. Nothing else changes.

## Limits
- Fee rates in `uae.yaml` are approximate. Verify them before using the numbers for a real purchase.
- Averages mix good and bad buildings. Always check the specific building, service charge and a recent comparable.
- Forecasts are trend extrapolations. On the real data the model beats the naive baseline only narrowly (about 10.8% vs 11.3% error), and the app says so.
- Not financial advice.
