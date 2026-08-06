"""
weekly_update.py - the production heartbeat.

Runs once a week (cron, Saturday after Friday close). Downloads the latest
weekly prices, computes features, scores every stock with the saved model
ensemble, and atomically replaces data/rankings.json. The API never computes
anything at request time; it just serves what this job wrote.

Run:  python weekly_update.py
Env:  ENGINE_DIR  path to the QuantProjectV2 checkout (default ~/QuantProjectV2)

Failure behavior: any exception leaves the previous rankings.json untouched
(atomic os.replace only happens after a fully successful build) and exits
non-zero so the scheduler can alert.
"""
import os
import sys
import json
import pickle
import tempfile
import datetime as dt
from pathlib import Path

ENGINE_DIR = Path(os.environ.get("ENGINE_DIR", Path.home() / "QuantProjectV2"))
sys.path.insert(0, str(ENGINE_DIR))

import config                                    # engine config
from signals import download_recent_data          # reuse engine logic
from main import compute_features, fetch_fundamentals
# Single source of truth for the book: the same selection the backtest trades.
from build_data import _pick_top_n, LONG_TOP_N, LONG_MAX_PER_SECTOR

OUT = Path(__file__).parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

DISCLAIMER = (
    "For educational and informational use only. Not investment advice and not "
    "a recommendation to buy or sell any security. Quantitative output is one "
    "input among many. Past performance does not guarantee future results."
)


def _signal_label(pct: float) -> str:
    if pct >= 0.95:  return "Strong Buy"
    if pct >= 0.85:  return "Buy"
    if pct >= 0.75:  return "Weak Buy"
    if pct <= 0.05:  return "Strong Sell"
    if pct <= 0.15:  return "Sell"
    if pct <= 0.25:  return "Weak Sell"
    return "Neutral"


def main():
    print("weekly_update: loading model bundle...")
    with open(config.MODEL_PATH, "rb") as fh:
        bundle = pickle.load(fh)
    feature_cols = bundle["feature_cols"]

    print("weekly_update: downloading recent market data...")
    data = download_recent_data(n_weeks=config.LOOKBACK_WEEKS + 5)
    fund_df = fetch_fundamentals(config.UNIVERSE)

    print("weekly_update: computing features...")
    # require_target=False: live scoring keeps the newest weeks, which have no
    # forward-return target yet. Without this, rankings lag a month behind.
    panel = compute_features(data, fund_df, require_target=False)

    dates = panel.index.get_level_values("Date").unique().sort_values()
    if len(dates) == 0:
        raise RuntimeError("No feature rows produced; aborting without touching rankings.json")
    latest_date = dates[-1]
    latest = panel.xs(latest_date, level="Date").dropna(subset=feature_cols)
    if latest.empty:
        raise RuntimeError(f"No valid rows on {latest_date}; aborting")

    print(f"weekly_update: scoring {len(latest)} stocks as of {latest_date.date()}...")
    X = latest[feature_cols].values
    s_lgbm  = bundle["lgbm"].predict(X)
    s_xgb   = bundle["xgb"].predict(X)
    s_ridge = bundle["ridge"].predict(bundle["scaler"].transform(X))
    latest = latest.copy()
    latest["score_raw"] = (s_lgbm + s_xgb + s_ridge) / 3.0
    latest["pct"] = latest["score_raw"].rank(pct=True)
    latest = latest.sort_values("pct", ascending=False)

    # The headline strategy does not simply hold the top 25 by score: it applies
    # a per-sector cap. Mark the names it would actually hold so the app can show
    # the real book instead of a raw ranking the backtest never traded.
    book = set(_pick_top_n(latest["pct"], LONG_TOP_N, LONG_MAX_PER_SECTOR))

    rows = [{
        "ticker": str(t),
        "score": round(float(r["pct"]), 4),      # cross-sectional rank, 0-1
        "percentile": round(float(r["pct"]), 4),
        "signal": _signal_label(float(r["pct"])),
        "sector": config.SECTOR_MAP.get(str(t), "Unknown"),
        "in_book": str(t) in book,
    } for t, r in latest.iterrows()]

    payload = {
        "as_of": latest_date.date().isoformat(),
        "universe_size": len(rows),
        "note": ("Live weekly rankings: cross-sectional model scores for the most "
                 "recent completed week. Higher score = higher predicted relative "
                 "rank over the next 4 weeks."),
        "book_note": (f"in_book marks the {LONG_TOP_N} holdings of the headline "
                      f"long-only strategy, which caps any one GICS sector at "
                      f"{LONG_MAX_PER_SECTOR} names, so a very highly ranked stock "
                      f"can be skipped once its sector is full."),
        "disclaimer": DISCLAIMER,
        "rankings": rows,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    # Atomic swap: write temp in same dir, then replace. Readers never see a
    # partial file; a crash before replace leaves the old snapshot serving.
    fd, tmp = tempfile.mkstemp(dir=OUT, suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, OUT / "rankings.json")
    print(f"weekly_update: OK — {len(rows)} stocks, as_of {payload['as_of']}, "
          f"rankings.json swapped atomically")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"weekly_update: FAILED — {e}", file=sys.stderr)
        sys.exit(1)
