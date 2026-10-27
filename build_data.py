"""
build_data.py - snapshot builder for the QuantApp backend.

Reads the QuantProjectV2 engine outputs (backtest.csv, ic_log.csv,
predictions.parquet) plus the feature metadata, computes summary metrics, and
writes three small JSON files that the API serves:

    data/rankings.json   - latest weekly stock rankings + signal labels
    data/backtest.json   - equity curve + performance metrics + IC table
    data/features.json   - the 25 feature explanations

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

from features_meta import FEATURES, REMOVED_NOTE, RANK_NOTE

ENGINE_DIR = Path(os.environ.get("ENGINE_DIR", Path.home() / "QuantProjectV2"))
sys.path.insert(0, str(ENGINE_DIR))
import config                                    # engine config (SECTOR_MAP)

RESULTS = ENGINE_DIR / "results"
# Take the price-cache filename from the engine config rather than hardcoding it.
# The name changed once already (prices.parquet to prices_extended_2026.parquet)
# and four separate files had it spelled out, which is one drift away from the
# container silently re-downloading the entire universe at boot.
PRICES = ENGINE_DIR / "data" / config.PRICES_PATH.name
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
    # Describes where the score sits, never what to do. "Buy" and "Sell"
    # labels read as advice, which this app does not give.
    if pct >= 0.95:  return "Top 5% of scores"
    if pct >= 0.85:  return "Top 15% of scores"
    if pct >= 0.75:  return "Top 25% of scores"
    if pct <= 0.05:  return "Bottom 5% of scores"
    if pct <= 0.15:  return "Bottom 15% of scores"
    if pct <= 0.25:  return "Bottom 25% of scores"
    return "Middle of the range"


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


# Headline strategy parameters (long-only, always invested).
#
# Book size raised from 25 to 60 on 2026-08-12. The 25-name book ran 20.2%
# volatility for a beta of 0.65, which is a large amount of stock-specific risk
# that nothing was paying for. Widening the book diversifies it away. Measured
# on the same fixed signal, changing only the construction:
#
#     names     25      40      60      80     100
#     Sharpe  0.669   0.817   0.953   0.888   0.840
#     vol     20.2%   18.5%   17.2%   16.6%   16.1%
#
# A smooth peak at 60, not an isolated spike, and volatility falls monotonically
# as predicted. Controls confirm the selection is doing the work: 60 names drawn
# at random score 8.95% a year at Sharpe 0.609 against 16.42% and 0.953 here.
#
# Every optimiser from MITx 15.455x Week 8 lost to plain equal weighting on this
# data: Michaud resampling 0.783, mean-variance with Ledoit-Wolf shrinkage 0.706,
# minimum variance 0.693. With 60 names and a 156-week window, T/N is 2.6, so the
# covariance matrix cannot be estimated well enough to optimise against. Equal
# weighting wins precisely because it assumes nothing about it.
LONG_TOP_N = 60
LONG_REBAL_WEEKS = 4
LONG_TC = 5 / 10_000       # 5 bps per side
LONG_SPY_MA_WEEKS = 40
# The "step to cash below the 40-week average" switch. Off since 2026-10-17:
# once the purge fix removed the label leak, the switch cut the model from
# about 24% a year to about 10%, and cut 60 random names from about 15% to
# about 4%. It sold near every bottom and bought back after the rebound, so it
# hurt every portfolio it was put on, not just this one. Kept as an option so
# the comparison can be re-run.
LONG_REGIME_FILTER = False

# Weighting experiments (Week 8): trailing window for volatility and covariance,
# and the single-name cap, matching config.MAX_STOCK_WEIGHT.
COV_WEEKS = 52
MAX_NAME_WEIGHT = 0.05

# Maximum names from any one GICS sector, holding the same 32% ceiling used at
# N=25 (8/25), which is close to Information Technology's own S&P 500 weight.
# The cap costs a little performance rather than adding it (uncapped scored
# 0.872 against 0.840 at N=100); it is kept for concentration risk, not return.
LONG_MAX_PER_SECTOR = 20


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


def _min_variance_weights(names: pd.Index, wret: pd.DataFrame, d) -> pd.Series:
    """
    Long-only minimum-variance weights (MITx 15.455x Week 8) for the chosen
    names, from the trailing COV_WEEKS of weekly returns up to and including d,
    with Ledoit-Wolf shrinkage and the same 5% single-name cap the engine uses.
    """
    from scipy.optimize import minimize
    from sklearn.covariance import LedoitWolf
    hist = wret.loc[:d, names].tail(COV_WEEKS).dropna(axis=1, thresh=COV_WEEKS // 2)
    hist = hist.fillna(0.0)
    n = hist.shape[1]
    if n < 2:
        return pd.Series(1.0 / len(names), index=names)
    cov = LedoitWolf().fit(hist.to_numpy()).covariance_
    cap = max(MAX_NAME_WEIGHT, 1.0 / n)
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1.0 / n),
                   jac=lambda w: 2 * cov @ w, method="SLSQP",
                   bounds=[(0.0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=hist.columns)


def _inverse_vol_weights(names: pd.Index, wret: pd.DataFrame, d) -> pd.Series:
    """Weights proportional to 1 / trailing volatility. Uses no correlations."""
    vol = wret.loc[:d, names].tail(COV_WEEKS).std()
    inv = (1.0 / vol.replace(0, np.nan)).dropna()
    return inv / inv.sum()


def _run_long_only(max_per_sector: int | None = LONG_MAX_PER_SECTOR,
                   delay_weeks: int = 0,
                   regime_filter: bool = LONG_REGIME_FILTER,
                   weighting: str = "equal",
                   scores: pd.Series | None = None):
    """
    Backtest the long-only top-N strategy (optionally regime filtered) from the stored
    out-of-sample predictions and cached prices. Returns
    (dates, strat_equity, spy_equity, strat_weekly_returns, spy_weekly_returns).

    delay_weeks > 0 acts on signals that many weeks late (scores and regime
    from an earlier Friday), as a check that the result does not depend on
    trading at the exact close the signal was computed from.

    scores replaces the model's scores with any other ranking on the same
    (Date, Ticker) index, so baselines are traded by exactly the same code.
    """
    preds = pd.read_parquet(RESULTS / "predictions.parquet")
    if scores is not None:
        preds = scores.reindex(preds.index).to_frame("score")
    store = pd.read_parquet(PRICES)
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
    for i, d in enumerate(dates):
        sig_d = dates[i - delay_weeks] if i >= delay_weeks else None
        pnl = 0.0
        if len(prev_w) and d in wret.index:
            rr = wret.loc[d]
            valid = prev_w.index.intersection(rr.dropna().index)
            pnl = float((prev_w[valid] * rr[valid]).sum())
        val *= (1 + pnl)
        weeks += 1
        if weeks >= LONG_REBAL_WEEKS or not len(prev_w):
            weeks = 0
            if sig_d is None or (regime_filter and not bool(in_regime.get(sig_d, True))):
                new_w = pd.Series(dtype=float)          # flat in bear regime
            else:
                s = preds.xs(sig_d, level="Date")["score"].dropna()
                top = _pick_top_n(s, LONG_TOP_N, max_per_sector)
                if not len(top):
                    new_w = pd.Series(dtype=float)
                elif weighting == "min_variance":
                    new_w = _min_variance_weights(top, wret, sig_d)
                elif weighting == "inverse_vol":
                    new_w = _inverse_vol_weights(top, wret, sig_d)
                else:
                    new_w = pd.Series(1.0 / len(top), index=top)
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


def _capm(r: np.ndarray, rm: np.ndarray, lags: int = 4) -> tuple[float, float, float]:
    """
    Beta, annual alpha and the alpha's t statistic from an OLS of weekly
    strategy returns on weekly S&P returns. The standard error is Newey-West
    with `lags` weeks, because a 4-week holding period makes neighbouring
    weekly returns overlap in the same book.
    """
    X = np.column_stack([np.ones_like(rm), rm])
    b = np.linalg.lstsq(X, r, rcond=None)[0]
    e = r - X @ b
    T = len(r)
    S = np.zeros((2, 2))
    for k in range(lags + 1):
        G = (X[k:] * e[k:, None]).T @ (X[:T - k] * e[:T - k, None]) / T
        S += G if k == 0 else (1 - k / (lags + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / T)
    se = np.sqrt(np.diag(Q @ S @ Q / T))
    return float(b[1]), float(b[0] * WEEKS_PER_YEAR), float(b[0] / se[0])


def _summary(eq: np.ndarray, r: np.ndarray, rm: np.ndarray) -> dict:
    T = len(r)
    beta, alpha, t = _capm(r, rm)
    return {
        "annual_return": round(float((eq[-1] / eq[0]) ** (WEEKS_PER_YEAR / T) - 1), 4),
        "annual_volatility": round(float(r.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)), 4),
        "sharpe": round(float(np.sqrt(WEEKS_PER_YEAR) * r.mean() / r.std(ddof=1)), 3),
        "max_drawdown": round(_max_drawdown(eq), 4),
        "beta": round(beta, 3),
        "alpha_annual": round(alpha, 4),
        "alpha_t": round(t, 2),
    }


RANDOM_BOOKS = 200    # random 60-name books, enough for a stable 5 to 95% band


def build_comparison(model: dict) -> dict:
    """
    The model next to simple rules, every one traded by _run_long_only so the
    costs, rebalancing and sector cap are identical. Answers the question a
    reader should ask of any backtest: would something dumber have done as well?
    """
    preds = pd.read_parquet(RESULTS / "predictions.parquet")
    close = pd.read_parquet(PRICES)["close"]
    idx = preds.index

    # 52-week momentum, the best known published rule (Jegadeesh and Titman
    # 1993). Uses only prices up to the signal date, like the model.
    mom = (close / close.shift(52) - 1).stack()
    mom.index.names = ["Date", "Ticker"]
    _, eq, _, r, rm = _run_long_only(scores=mom.reindex(idx))
    momentum = _summary(eq, r, rm)

    rng = np.random.default_rng(0)
    rand = []
    for _ in range(RANDOM_BOOKS):
        s = pd.Series(rng.random(len(idx)), index=idx)
        _, eq, _, r, rm = _run_long_only(scores=s)
        rand.append(_summary(eq, r, rm))
    rand_ann = np.array([x["annual_return"] for x in rand])
    rand_sh = np.array([x["sharpe"] for x in rand])

    # The old design, kept so the reason for removing it stays on record.
    _, eq, _, r, rm = _run_long_only(regime_filter=True)
    switch_on = _summary(eq, r, rm)

    return {
        "momentum": momentum,
        "random": {
            "books": RANDOM_BOOKS,
            "annual_return_median": round(float(np.median(rand_ann)), 4),
            "annual_return_p5": round(float(np.percentile(rand_ann, 5)), 4),
            "annual_return_p95": round(float(np.percentile(rand_ann, 95)), 4),
            "sharpe_median": round(float(np.median(rand_sh)), 3),
            "max_drawdown_median": round(float(np.median([x["max_drawdown"] for x in rand])), 4),
            "annual_volatility_median": round(float(np.median([x["annual_volatility"] for x in rand])), 4),
            # Share of random books whose Sharpe beat the model's: a p-value
            # for "the selection did better than chance on a risk-adjusted basis".
            "share_beating_model_sharpe": round(float((rand_sh >= model["sharpe"]).mean()), 3),
        },
        "with_cash_switch": switch_on,
    }


def build_backtest() -> dict:
    """Equity curve + performance metrics + per-fold IC table."""
    # Headline strategy is LONG-ONLY top-N, always invested: hold the model's
    # highest ranked names at equal weight. Computed live from the
    # out-of-sample predictions so it always matches the ranking engine.
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

    # CAPM alpha/beta via OLS of strategy weekly returns on SPY weekly returns,
    # with a Newey-West t on the alpha so it is never shown without its test.
    beta, alpha_ann, alpha_t = _capm(r, rm)
    win_rate = float((r > 0).mean())

    # Robustness: the same strategy acting one week late.
    _, eq_lag, _, r_lag, _ = _run_long_only(delay_weeks=1)
    lag_ann = float((eq_lag[-1] / eq_lag[0]) ** (WEEKS_PER_YEAR / len(r_lag)) - 1)
    lag_sharpe = float(np.sqrt(WEEKS_PER_YEAR) * r_lag.mean() / r_lag.std(ddof=1)) if r_lag.std() else 0.0

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

    model_summary = {"sharpe": sharpe}
    comparison = build_comparison(model_summary)

    return {
        # Derived, not typed. It said "top 25" for the first hour after the book
        # was widened to 60, which is exactly the kind of stale label that makes
        # the whole screen untrustworthy.
        "strategy_name": f"Long-only top {LONG_TOP_N}, equal weight, always invested",
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
            "alpha_t": round(alpha_t, 2),
            "beta": round(float(beta), 3),
            "win_rate": round(win_rate, 4),
        },
        "benchmark": {
            "annual_return": round(spy_ann_ret, 4),
            "annual_volatility": round(spy_ann_vol, 4),
            "sharpe": round(spy_sharpe, 3),
            "max_drawdown": round(spy_mdd, 4),
        },
        "delayed_one_week": {
            "annual_return": round(lag_ann, 4),
            "sharpe": round(lag_sharpe, 3),
            "max_drawdown": round(_max_drawdown(eq_lag), 4),
        },
        "rf_note": ("Sharpe, Sortino and alpha use a risk-free rate of 0, for the "
                    "strategy and the S&P 500 alike. With a cash rate subtracted, "
                    "both Sharpe ratios and the alpha would be lower."),
        "comparison": comparison,
        "equity_curve": curve,
        "ic": {"by_model": ic_models, "folds": ic_folds},
        "disclaimer": DISCLAIMER,
    }


def build_features() -> dict:
    """The feature dictionary the app renders. Every entry now carries its
    formula, because showing the mathematics is the product."""
    return {
        "count": len(FEATURES),
        "technical_count": len(FEATURES),   # all 25 are price/volume derived
        "rank_note": RANK_NOTE,
        "removed_note": REMOVED_NOTE,
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
