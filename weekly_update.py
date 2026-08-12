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

import numpy as np

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

    # Walk back from the newest date to the most recent COMPLETE week. The job
    # runs on Saturdays, when the provider often returns a partial in-progress
    # week whose features are all NaN. Taking dates[-1] blindly made the job
    # abort every time that happened, which is most of the time.
    MIN_STOCKS = 100
    latest_date, latest = None, None
    for d in reversed(dates):
        rows = panel.xs(d, level="Date").dropna(subset=feature_cols)
        if len(rows) >= MIN_STOCKS:
            latest_date, latest = d, rows
            break
        print(f"weekly_update: skipping {d.date()} ({len(rows)} valid rows, "
              f"need {MIN_STOCKS}) - incomplete week")
    if latest is None:
        raise RuntimeError(
            f"No week in the panel has {MIN_STOCKS}+ scorable stocks; aborting "
            "without touching rankings.json")

    print(f"weekly_update: scoring {len(latest)} stocks as of {latest_date.date()}...")
    X = latest[feature_cols].values
    s_lgbm  = bundle["lgbm"].predict(X)
    s_xgb   = bundle["xgb"].predict(X)
    X_scaled = bundle["scaler"].transform(X)
    s_ridge = bundle["ridge"].predict(X_scaled)
    latest = latest.copy()
    latest["score_raw"] = (s_lgbm + s_xgb + s_ridge) / 3.0
    latest["pct"] = latest["score_raw"].rank(pct=True)

    # ── Per-stock attribution ────────────────────────────────────────────────
    # Exact SHAP values, no `shap` package needed: LightGBM and XGBoost both
    # compute tree SHAP natively, and for the ridge term the contribution of a
    # feature is just coefficient times its standardised value. Each model's
    # contributions plus its base value sum exactly to that model's prediction,
    # so averaging the three reproduces the ensemble score the same way the
    # scoring code does.
    print("weekly_update: computing SHAP attributions...")
    c_lgbm = bundle["lgbm"].predict(X, pred_contrib=True)          # (n, k+1)
    import xgboost as _xgb
    c_xgb = bundle["xgb"].get_booster().predict(
        _xgb.DMatrix(X), pred_contribs=True)                        # (n, k+1)
    c_ridge = np.hstack([
        X_scaled * bundle["ridge"].coef_,
        np.full((len(X), 1), float(bundle["ridge"].intercept_)),
    ])
    contrib = (c_lgbm + c_xgb + c_ridge) / 3.0
    shap_base = float(contrib[:, -1].mean())
    contrib = contrib[:, :-1]                                       # drop base column

    # Keep the attribution aligned to the row order before sorting.
    contrib_by_ticker = {str(t): contrib[i] for i, t in enumerate(latest.index)}
    # Panel features are already cross-sectional percentiles in [0,1]; see
    # compute_features(), which rank-normalises within each week.
    pct_by_ticker = {str(t): latest.loc[t, feature_cols].values
                     for t in latest.index}

    latest = latest.sort_values("pct", ascending=False)

    # The headline strategy does not simply hold the top 25 by score: it applies
    # a per-sector cap. Mark the names it would actually hold so the app can show
    # the real book instead of a raw ranking the backtest never traded.
    book = set(_pick_top_n(latest["pct"], LONG_TOP_N, LONG_MAX_PER_SECTOR))

    # fp and fc are positional arrays aligned to `feature_order` at the top of
    # the payload. Storing them as objects with repeated key names would roughly
    # quadruple the file for 490 stocks times 25 features.
    rows = [{
        "ticker": str(t),
        "score": round(float(r["pct"]), 4),      # cross-sectional rank, 0-1
        "percentile": round(float(r["pct"]), 4),
        "signal": _signal_label(float(r["pct"])),
        "sector": config.SECTOR_MAP.get(str(t), "Unknown"),
        "in_book": str(t) in book,
        "fp": [int(round(v * 100)) for v in pct_by_ticker[str(t)]],
        "fc": [round(float(v), 5) for v in contrib_by_ticker[str(t)]],
    } for t, r in latest.iterrows()]

    payload = {
        "as_of": latest_date.date().isoformat(),
        "universe_size": len(rows),
        "feature_order": list(feature_cols),
        "shap_base": round(shap_base, 5),
        "shap_note": ("fp is each feature's cross-sectional percentile this week, 0 to "
                      "100. fc is that feature's SHAP contribution to this stock's raw "
                      "score; the 25 contributions plus shap_base sum to the raw score. "
                      "SHAP explains why the MODEL ranked the stock where it did. It is "
                      "not a claim about why the stock will move."),
        "note": ("Live weekly rankings: cross-sectional model scores for the most "
                 "recent completed week. Higher score = higher predicted relative "
                 "rank over the next 4 weeks."),
        "book_note": (f"The {LONG_TOP_N} positions the headline strategy holds. No "
                      f"more than {LONG_MAX_PER_SECTOR} may come from one sector, so a "
                      f"highly ranked stock is skipped once its sector is full."),
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
