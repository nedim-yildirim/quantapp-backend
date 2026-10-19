# QuantApp Backend

**Live app:** https://quantapp-api-r9nc.onrender.com

FastAPI service that turns the QuantProjectV2
research engine into a data API for the QuantApp iOS app.

It is a decision-support tool for knowledgeable investors: it surfaces the model's
quantitative output (weekly rankings, backtest performance, feature explanations)
as one input. It is not investment advice and does not place trades.

## Architecture

```
QuantProjectV2 engine            QuantApp backend                 clients
(main.py, weekly)                                                 (iOS app,
   results/*.csv,  ──►  build_data.py  ──►  data/*.json  ──►  app.py  ──►  web)
   predictions.parquet    (heavy: pandas)     (snapshots)   (light: FastAPI)
```

- `build_data.py` reads the engine outputs and writes three JSON snapshots.
- `app.py` serves those snapshots. No pandas or model loading at request time.

## The honest result

Backtested 2018 to 2026, out of sample, after removing two look-ahead leaks
(fundamentals in August 2026, a missing purge gap between training and test in
October 2026). The 60-stock book is always invested and equal weighted, and every
row below is traded by the same code with the same costs:

| | Per year | Volatility | Worst fall | Alpha (t) |
|---|---|---|---|---|
| S&P 500 | 15.4% | 18.2% | -32.2% | |
| Random 60 stocks (median of 200) | 15.4% | 20.4% | -37.1% | |
| **This model** | **20.6%** | **25.9%** | **-43.8%** | **2.0% (t 0.57)** |
| 52-week momentum rule | 25.6% | 23.2% | -35.6% | 8.2% (t 2.20) |

The model beats the index on return mainly because its book has a beta of 1.26.
After adjusting for that risk the alpha is indistinguishable from zero, and the
model's Sharpe (0.85) is no better than the index's (0.87). A published one-line
momentum rule does better. The app shows all of this on the Performance page.

Since the backtest years have now been looked at many times, they cannot test a
new idea fairly. From October 2026 the weekly job saves each book to
`data/live/` and commits it, so git records the picks before their returns exist.
That live record is the real test.

## Run locally

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Point at your engine checkout if it is not ~/QuantProjectV2
export ENGINE_DIR=~/QuantProjectV2

python build_data.py          # generate data/rankings.json, backtest.json, features.json
uvicorn app:app --reload      # http://127.0.0.1:8000  (docs at /docs)
```

## Endpoints

| Method | Path | Returns |
|--------|------|---------|
| GET | `/` | service info |
| GET | `/health` | liveness + which snapshots are built |
| GET | `/rankings` | latest weekly rankings (`?limit=N` for top+bottom N) |
| GET | `/backtest` | equity curve, performance metrics, IC table |
| GET | `/features` | the 25 feature explanations |

## Weekly refresh

Every Saturday `.github/workflows/weekly-update.yml` runs `weekly_update.py` on
GitHub Actions, then commits `data/rankings.json` and the dated book in
`data/live/`. The push redeploys Render. The backtest snapshot only changes when
the engine is retrained:

```
cd ~/QuantProjectV2 && python main.py
cd ~/QuantApp/backend && ./bundle_engine.sh && python build_data.py
```

## Disclaimer

Educational and informational use only. Not investment advice. Backtest results
contain survivorship bias and may overstate live performance.
