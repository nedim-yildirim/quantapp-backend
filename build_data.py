"""
build_data.py - snapshot builder for the QuantApp backend.

Reads the QuantProjectV2 engine outputs (backtest.csv, ic_log.csv,
predictions.parquet) plus the feature metadata, computes summary metrics, and
writes three small JSON files that the API serves:

    data/rankings.json   - latest weekly stock rankings + signal labels
    data/backtest.json   - equity curve + performance metrics + IC table
    data/features.json   - the 30 feature explanations

Run:  python build_data.py
Point ENGINE_DIR at your QuantProjectV2 checkout (default: ~/QuantProjectV2).
"""
import os
import sys
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from features_meta import FEATURES, FUNDAMENTAL_NOTE

ENGINE_DIR = Path(os.environ.get("ENGINE_DIR", Path.home() / "QuantProjectV2"))
sys.path.insert(0, str(ENGINE_DIR))
import config                                    # engine config (SECTOR_MAP)

RESULTS = ENGINE_DIR / "results"
OUT = Path(__file__).parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

WEEKS_PER_YEAR = 52
CASH = 10_000          # starting capital for the equity curve
DISCLAIMER = (
    "For educational and informational use only. Not investment advice and not "
    "a recommendation to buy or sell any security. Quantitative output is one "
    "input among many. Past performance does not guarantee future results. "
    "Backtest contains survivorship bias and may overstate live results."
)


def _signal_label(pct: float) -> str:
    """Rank percentile (0-1) -> human signal bucket. Mirrors signals.py."""
    if pct >= 0.95:  return "Strong Buy"
    if pct >= 0.85:  return "Buy"
    if pct >= 0.75:  return "Weak Buy"
    if pct <= 0.05:  return "Strong Sell"
    if pct <= 0.15:  return "Sell"
    if pct <= 0.25:  return "Weak Sell"
    return "Neutral"


def build_rankings() -> dict:
    """Latest available week's cross-sectional scores from predictions.parquet."""
    preds = pd.read_parquet(RESULTS / "predictions.parquet")
    latest_date = preds.index.get_level_values("Date").max()
    latest = preds.xs(latest_date, level="Date").copy()

    # score column is already a cross-sectional rank in [0,1]; recompute a clean
    # percentile rank so ties and scale are consistent for labelling.
    latest["pct"] = latest["score"].rank(pct=True)
    latest = latest.sort_values("score", ascending=False)

    rows = []
    for ticker, r in latest.iterrows():
        rows.append({
            "ticker": str(ticker),
            "score": round(float(r["score"]), 4),
            "percentile": round(float(r["pct"]), 4),
            "signal": _signal_label(float(r["pct"])),
        })

    return {
        "as_of": pd.Timestamp(latest_date).date().isoformat(),
        "universe_size": len(rows),
        "note": ("Rankings are the most recent cross-sectional scores produced by "
                 "the walk-forward engine. A higher score means a higher predicted "
                 "relative rank over the next 4 weeks."),
        "disclaimer": DISCLAIMER,
        "rankings": rows,
    }


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


# Headline strategy parameters (long-only, regime-filtered).
LONG_TOP_N = 25
LONG_REBAL_WEEKS = 4
LONG_TC = 5 / 10_000       # 5 bps per side
LONG_SPY_MA_WEEKS = 40

# Maximum names from any one GICS sector in the 25-stock book.
# Chosen on economic grounds, not by tuning: 8/25 = 32%, close to Information
# Technology's own weight in the S&P 500, so the portfolio may still tilt toward
# a sector but cannot become a single-sector bet. Without this the book ran
# 18/25 Information Technology, which is a concentrated semiconductor position
# rather than a diversified portfolio.
LONG_MAX_PER_SECTOR = 8


def _pick_top_n(scores: pd.Series, top_n: int, max_per_sector: int | None) -> pd.Index:
    """
    Highest-scoring names subject to a per-sector cap, filled greedily by score.
    A ticker with no sector mapping is treated as its own sector so it can never
    be silently excluded by the cap.
    """
    ranked = scores.sort_values(ascending=False)
    if not max_per_sector:
        return ranked.head(top_n).index
    counts: dict = {}
    picked = []
    for ticker in ranked.index:
        sector = config.SECTOR_MAP.get(ticker) or f"_unmapped_{ticker}"
        if counts.get(sector, 0) >= max_per_sector:
            continue
        counts[sector] = counts.get(sector, 0) + 1
        picked.append(ticker)
        if len(picked) == top_n:
            break
    return pd.Index(picked)


def _run_long_only(max_per_sector: int | None = LONG_MAX_PER_SECTOR):
    """
    Backtest the long-only top-N regime-filtered strategy from the stored
    out-of-sample predictions and cached prices. Returns
    (dates, strat_equity, spy_equity, strat_weekly_returns, spy_weekly_returns).
    """
    preds = pd.read_parquet(RESULTS / "predictions.parquet")
    store = pd.read_parquet(ENGINE_DIR / "data" / "prices.parquet")
    close = store["close"]
    spy = store["spy_close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.squeeze()

    wret = close.pct_change()
    spy_ret = spy.pct_change()
    in_regime = spy > spy.rolling(LONG_SPY_MA_WEEKS).mean()

    dates = preds.index.get_level_values("Date").unique().sort_values()
    prev_w = pd.Series(dtype=float)
    val = float(CASH)
    weeks = 0
    rows = []
    for d in dates:
        pnl = 0.0
        if len(prev_w) and d in wret.index:
            rr = wret.loc[d]
            valid = prev_w.index.intersection(rr.dropna().index)
            pnl = float((prev_w[valid] * rr[valid]).sum())
        val *= (1 + pnl)
        weeks += 1
        if weeks >= LONG_REBAL_WEEKS or not len(prev_w):
            weeks = 0
            if not bool(in_regime.get(d, True)):
                new_w = pd.Series(dtype=float)          # flat in bear regime
            else:
                s = preds.xs(d, level="Date")["score"].dropna()
                top = _pick_top_n(s, LONG_TOP_N, max_per_sector)
                new_w = pd.Series(1.0 / len(top), index=top) if len(top) else pd.Series(dtype=float)
        else:
            new_w = prev_w
        old = prev_w.reindex(new_w.index.union(prev_w.index), fill_value=0.0)
        new = new_w.reindex(old.index, fill_value=0.0)
        turnover = (new - old).abs().sum() / 2
        val *= (1 - turnover * LONG_TC)
        rows.append((d, pnl - turnover * LONG_TC, val))
        prev_w = new_w

    bt = pd.DataFrame(rows, columns=["Date", "ret", "val"]).set_index("Date")
    sp = spy_ret.reindex(bt.index).fillna(0.0)
    spy_equity = (1 + sp).cumprod().to_numpy() * float(CASH)
    return (bt.index, bt["val"].to_numpy(), spy_equity,
            bt["ret"].to_numpy(), sp.to_numpy())


def build_backtest() -> dict:
    """Equity curve + performance metrics + per-fold IC table."""
    # Headline strategy is LONG-ONLY top-N, regime filtered: hold the model's
    # highest ranked names, step to cash when SPY is below its 40-week average.
    # Computed live from the out-of-sample predictions so it always matches the
    # ranking engine. (Market-neutral remains available as a diversifier mode.)
    dates_idx, strat_equity, spy_equity, r, rm = _run_long_only()
    T = len(r)

    ann_ret = float((strat_equity[-1] / strat_equity[0]) ** (WEEKS_PER_YEAR / T) - 1)
    spy_ann_ret = float((spy_equity[-1] / spy_equity[0]) ** (WEEKS_PER_YEAR / T) - 1)
    ann_vol = float(r.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))
    spy_ann_vol = float(rm.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))
    sharpe = float(np.sqrt(WEEKS_PER_YEAR) * r.mean() / r.std(ddof=1)) if r.std() else 0.0
    spy_sharpe = float(np.sqrt(WEEKS_PER_YEAR) * rm.mean() / rm.std(ddof=1)) if rm.std() else 0.0
    downside = r[r < 0]
    sortino = float(np.sqrt(WEEKS_PER_YEAR) * r.mean() /
                    np.sqrt((downside ** 2).mean())) if len(downside) else 0.0
    mdd = _max_drawdown(strat_equity)
    spy_mdd = _max_drawdown(spy_equity)

    # CAPM alpha/beta via OLS of strategy weekly returns on SPY weekly returns
    beta, alpha_w = np.polyfit(rm, r, 1)
    alpha_ann = float(alpha_w * WEEKS_PER_YEAR)
    win_rate = float((r > 0).mean())

    # Full weekly equity curve (~349 points, fine to send whole).
    curve = [
        {"date": d.date().isoformat(),
         "strategy": round(float(s), 2),
         "spy": round(float(m), 2)}
        for d, s, m in zip(dates_idx, strat_equity, spy_equity)
    ]

    # IC table
    ic = pd.read_csv(RESULTS / "ic_log.csv")
    ic_models = {}
    for col, label in [("ic_lgbm", "LightGBM"), ("ic_xgb", "XGBoost"),
                       ("ic_ridge", "Ridge"), ("mean_ic", "Ensemble")]:
        vals = ic[col].dropna()
        ic_models[label] = {
            "mean_ic": round(float(vals.mean()), 4),
            "pct_positive": round(float((vals > 0).mean()), 3),
        }
    ic_folds = [
        {"fold": int(row["fold"]),
         "pred_start": str(row["pred_start"]),
         "ensemble_ic": None if pd.isna(row["mean_ic"]) else round(float(row["mean_ic"]), 4)}
        for _, row in ic.iterrows()
    ]

    return {
        "strategy_name": "Long-only top 25, regime filtered",
        "period": {"start": pd.Timestamp(dates_idx[0]).date().isoformat(),
                   "end": pd.Timestamp(dates_idx[-1]).date().isoformat(),
                   "weeks": int(T)},
        "metrics": {
            "annual_return": round(ann_ret, 4),
            "annual_volatility": round(ann_vol, 4),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "max_drawdown": round(mdd, 4),
            "alpha_annual": round(alpha_ann, 4),
            "beta": round(float(beta), 3),
            "win_rate": round(win_rate, 4),
        },
        "benchmark": {
            "annual_return": round(spy_ann_ret, 4),
            "annual_volatility": round(spy_ann_vol, 4),
            "sharpe": round(spy_sharpe, 3),
            "max_drawdown": round(spy_mdd, 4),
        },
        "equity_curve": curve,
        "ic": {"by_model": ic_models, "folds": ic_folds},
        "disclaimer": DISCLAIMER,
    }


def build_features() -> dict:
    return {
        "count": len(FEATURES),
        "technical_count": sum(1 for f in FEATURES if f["category"] not in
                               ("Valuation", "Quality", "Growth", "Leverage")),
        "fundamental_note": FUNDAMENTAL_NOTE,
        "features": FEATURES,
    }


def main():
    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    # backtest + features are the static snapshots (change only on a quarterly
    # retrain). Rankings are the LIVE weekly heartbeat, owned exclusively by
    # weekly_update.py, so build_data must not touch rankings.json here.
    builders = [("backtest", build_backtest), ("features", build_features)]
    if "--seed-rankings" in sys.argv:
        # One-time seed only, e.g. before the first weekly job has run.
        builders.insert(0, ("rankings", build_rankings))
    for name, fn in builders:
        payload = fn()
        payload["generated_at"] = generated
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {path}  ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
