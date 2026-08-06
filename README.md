# QuantApp Backend

FastAPI service that turns the [QuantProjectV2](https://github.com/Nedim21Y/QuantProjectV2)
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
| GET | `/features` | the 30 feature explanations |

## Weekly refresh

Run the engine, then rebuild snapshots:

```
cd ~/QuantProjectV2 && python main.py && python signals.py
cd ~/QuantApp/backend && python build_data.py
```

In production this is a scheduled job (cron / Render cron) that runs over the
weekend so Monday's rankings are fresh.

## Disclaimer

Educational and informational use only. Not investment advice. Backtest results
contain survivorship bias and may overstate live performance.
