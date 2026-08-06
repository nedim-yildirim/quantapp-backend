"""
QuantApp backend API.

Lightweight FastAPI service that serves the JSON snapshots produced by
build_data.py. No pandas or model loading at request time, so it starts fast
and is cheap to host.

Run locally:
    pip install -r requirements.txt
    python build_data.py        # generate data/*.json from the engine outputs
    uvicorn app:app --reload     # serve at http://127.0.0.1:8000

Endpoints:
    GET /                 service info
    GET /health          liveness probe
    GET /rankings        latest weekly stock rankings
    GET /backtest        equity curve, performance metrics, IC table
    GET /features        the 30 feature explanations
"""
import json
import os
import sys
import threading
import subprocess
import datetime as dt
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

DATA = Path(__file__).parent / "data"
_STAMP = DATA / ".last_weekly"


def _scheduler_loop():
    """
    Minimal in-process weekly scheduler (ENABLE_SCHEDULER=1). Hosted cron
    services get a separate filesystem from the web service, so the refresh
    runs here instead. Checks hourly; runs weekly_update.py as a subprocess
    (memory released on exit) on Saturdays, at most once per 5 days.
    """
    import time
    while True:
        try:
            now = dt.datetime.now(dt.timezone.utc)
            last = 0.0
            if _STAMP.exists():
                last = float(_STAMP.read_text().strip() or 0)
            stale = (time.time() - last) > 5 * 86400
            if now.weekday() == 5 and now.hour >= 8 and stale:
                print("scheduler: running weekly_update.py ...", flush=True)
                r = subprocess.run(
                    [sys.executable, str(Path(__file__).parent / "weekly_update.py")],
                    timeout=3600,
                )
                if r.returncode == 0:
                    _STAMP.write_text(str(time.time()))
                    print("scheduler: weekly refresh OK", flush=True)
                else:
                    print("scheduler: weekly refresh FAILED, keeping old snapshot",
                          flush=True)
        except Exception as e:
            print(f"scheduler: error {e}", flush=True)
        time.sleep(3600)


if os.environ.get("ENABLE_SCHEDULER") == "1":
    threading.Thread(target=_scheduler_loop, daemon=True).start()

app = FastAPI(
    title="QuantApp API",
    version="0.1.0",
    description="Decision-support analytics for a market-neutral S&P 500 model. "
                "Educational use only, not investment advice.",
)

# The iOS app and any web client call this cross-origin. Open for v1; tighten to
# specific origins before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _load(name: str) -> dict:
    path = DATA / f"{name}.json"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"{name}.json not built yet. Run 'python build_data.py' first.",
        )
    return json.loads(path.read_text())


@app.get("/")
def root():
    return {
        "service": "QuantApp API",
        "version": "0.1.0",
        "endpoints": ["/rankings", "/backtest", "/features", "/health"],
        "disclaimer": "Educational use only. Not investment advice.",
    }


@app.get("/health")
def health():
    built = {n: (DATA / f"{n}.json").exists()
             for n in ("rankings", "backtest", "features")}
    return {"status": "ok", "data_built": built}


@app.get("/rankings")
def rankings(limit: int | None = None):
    """Latest weekly rankings. Optional ?limit=N returns the top and bottom N."""
    data = _load("rankings")
    if limit is not None and limit > 0:
        rows = data["rankings"]
        data = {**data, "rankings": rows[:limit] + rows[-limit:],
                "note": data["note"] + f" (showing top and bottom {limit})"}
    return data


@app.get("/backtest")
def backtest():
    return _load("backtest")


@app.get("/features")
def features():
    return _load("features")
