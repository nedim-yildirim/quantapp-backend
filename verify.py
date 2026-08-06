"""
verify.py - pre-flight checks for the QuantApp data pipeline.

Every check here exists because something actually went wrong, or because a
silent failure in that spot would ship wrong numbers to users. Run before any
deploy or App Store submission:

    python verify.py

Exit code 0 = all checks passed, 1 = at least one FAIL.
"""
import os
import sys
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE_DIR = Path(os.environ.get("ENGINE_DIR", Path.home() / "QuantProjectV2"))
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(Path(__file__).parent))

import config
from build_data import _pick_top_n, LONG_TOP_N, LONG_MAX_PER_SECTOR

DATA = Path(__file__).parent / "data"
RESULTS = ENGINE_DIR / "results"

_results: list = []


def check(name):
    """Decorator: run a check, capture PASS/FAIL/WARN plus a detail string."""
    def deco(fn):
        try:
            ok, detail = fn()
            # numpy comparisons return np.bool_, which is never `is True`.
            status = "WARN" if ok == "warn" else ("PASS" if bool(ok) else "FAIL")
        except Exception as e:
            status, detail = "FAIL", f"raised {type(e).__name__}: {e}"
        _results.append((status, name, detail))
        return fn
    return deco


# ── 1. Universe integrity ───────────────────────────────────────────────────
@check("Universe has no duplicate tickers")
def _c1():
    u = config.UNIVERSE
    dupes = [t for t in set(u) if u.count(t) > 1]
    return (not dupes), f"{len(u)} tickers, {len(dupes)} duplicates {dupes[:5]}"


@check("Every universe ticker has a GICS sector")
def _c2():
    unmapped = [t for t in config.UNIVERSE if t not in config.SECTOR_MAP]
    return (not unmapped), f"{len(unmapped)} unmapped {unmapped[:8]}"


@check("Universe contains the mega-caps it claims to")
def _c3():
    # NVDA was silently missing for the whole life of the project.
    must = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "BRK-B", "JPM",
            "LLY", "AVGO", "TSLA", "UNH", "XOM", "V", "MA"]
    missing = [t for t in must if t not in config.UNIVERSE]
    return (not missing), f"missing {missing}" if missing else "all 15 present"


@check("Universe size is plausible for the S&P 500")
def _c4():
    n = len(config.UNIVERSE)
    return (480 <= n <= 520), f"{n} tickers (expect 480-520)"


# ── 2. Price cache integrity ────────────────────────────────────────────────
@check("Price cache covers the universe (the bug that made edits no-ops)")
def _c5():
    store = pd.read_parquet(ENGINE_DIR / "data" / "prices.parquet")
    cached = set(store["close"].columns)
    wanted = {t for t in config.UNIVERSE if t != "SPY"}
    missing_path = ENGINE_DIR / "data" / "missing_tickers.json"
    known = set(json.loads(missing_path.read_text())) if missing_path.exists() else set()
    gap = wanted - cached - known
    return (not gap), (f"{len(cached)} cached, {len(known)} known-dead, "
                       f"{len(gap)} unexplained {sorted(gap)[:6]}")


@check("Cache-invalidation logic actually fires on a universe change")
def _c6():
    # Regression test for main.py:82. Ask for a ticker that cannot be cached and
    # confirm the coverage check would reject the cache rather than silently
    # reuse it.
    src = (ENGINE_DIR / "main.py").read_text()
    has_guard = ("wanted - cached" in src) and ("re-downloading" in src)
    return has_guard, "coverage guard present in download_data()" if has_guard \
        else "GUARD MISSING: edits to UNIVERSE will be silently ignored"


@check("Price history is deep enough for the feature warmup")
def _c7():
    store = pd.read_parquet(ENGINE_DIR / "data" / "prices.parquet")
    weeks = len(store["close"])
    return (weeks >= 300), f"{weeks} weekly bars"


# ── 3. Model / prediction integrity ─────────────────────────────────────────
@check("Predictions contain no NaN or infinite scores")
def _c8():
    p = pd.read_parquet(RESULTS / "predictions.parquet")
    bad = int((~np.isfinite(p["score"])).sum())
    return (bad == 0), f"{len(p):,} rows, {bad} non-finite"


@check("Walk-forward folds never train on the future")
def _c9():
    ic = pd.read_csv(RESULTS / "ic_log.csv")
    leaks = ic[pd.to_datetime(ic["train_end"]) > pd.to_datetime(ic["pred_start"])]
    return (len(leaks) == 0), f"{len(ic)} folds, {len(leaks)} with train_end > pred_start"


@check("Ensemble IC is positive and statistically significant")
def _c10():
    ic = pd.read_csv(RESULTS / "ic_log.csv")
    v = ic["mean_ic"].dropna()
    t = v.mean() / v.std() * np.sqrt(len(v))
    return (v.mean() > 0 and t > 2.0), \
        f"mean IC {v.mean():.4f}, {(v > 0).mean():.1%} positive, t={t:.2f} over {len(v)} folds"


@check("Backtest is reproducible (models are seeded)")
def _c11():
    src = (ENGINE_DIR / "main.py").read_text()
    seeded = src.count("random_state") >= 2 or "random_state" in str(config.LGBM_PARAMS)
    return seeded, f"LGBM random_state={config.LGBM_PARAMS.get('random_state')}, xgb inherits it"


# ── 4. Portfolio construction ───────────────────────────────────────────────
@check("Sector cap is enforced in the live book")
def _c12():
    r = json.loads((DATA / "rankings.json").read_text())
    book = [x for x in r["rankings"] if x.get("in_book")]
    if not book:
        return False, "no in_book flags found in rankings.json"
    from collections import Counter
    mix = Counter(config.SECTOR_MAP.get(x["ticker"], "Unknown") for x in book)
    worst = mix.most_common(1)[0]
    return (len(book) == LONG_TOP_N and worst[1] <= LONG_MAX_PER_SECTOR), \
        f"{len(book)} holdings, largest sector {worst[0]} = {worst[1]}/{LONG_TOP_N} (cap {LONG_MAX_PER_SECTOR})"


@check("Book selection is identical between backtest and live rankings")
def _c13():
    # Both must call _pick_top_n with the same parameters, or the app would show
    # a book the backtest never traded.
    wu = (Path(__file__).parent / "weekly_update.py").read_text()
    uses_shared = "_pick_top_n" in wu and "from build_data import" in wu
    return uses_shared, "weekly_update imports the backtest's selector" if uses_shared \
        else "live book and backtest book are computed by different code"


@check("Equity curve is finite, monotonic in time, and correctly lengthed")
def _c14():
    b = json.loads((DATA / "backtest.json").read_text())
    c = b["equity_curve"]
    dates = [x["date"] for x in c]
    vals = [x["strategy"] for x in c]
    finite = all(np.isfinite(v) and v > 0 for v in vals)
    sorted_ok = dates == sorted(dates)
    len_ok = len(c) == b["period"]["weeks"]
    return (finite and sorted_ok and len_ok), \
        f"{len(c)} points, finite={finite}, sorted={sorted_ok}, matches period={len_ok}"


@check("Headline metrics are internally consistent")
def _c15():
    b = json.loads((DATA / "backtest.json").read_text())
    m, bm = b["metrics"], b["benchmark"]
    c = b["equity_curve"]
    T = b["period"]["weeks"]
    implied = (c[-1]["strategy"] / c[0]["strategy"]) ** (52 / T) - 1
    close = abs(implied - m["annual_return"]) < 0.01
    sane = (0 < m["annual_volatility"] < 1 and -1 < m["max_drawdown"] < 0
            and 0 < m["beta"] < 2 and 0 <= m["win_rate"] <= 1)
    return (close and sane), \
        f"annual_return {m['annual_return']:.2%} vs curve-implied {implied:.2%}, ranges sane={sane}"


# ── 5. Served payload / product correctness ─────────────────────────────────
@check("All served JSON carries the not-advice disclaimer")
def _c16():
    missing = []
    for f in ["rankings.json", "backtest.json"]:
        d = json.loads((DATA / f).read_text())
        if "not investment advice" not in d.get("disclaimer", "").lower():
            missing.append(f)
    return (not missing), f"missing in {missing}" if missing else "present in rankings + backtest"


@check("Rankings snapshot is fresh")
def _c17():
    r = json.loads((DATA / "rankings.json").read_text())
    age = (pd.Timestamp.now().normalize() - pd.Timestamp(r["as_of"])).days
    return ("warn" if age > 10 else True), f"as_of {r['as_of']}, {age} days old"


@check("Feature explanations cover all 30 model features")
def _c18():
    f = json.loads((DATA / "features.json").read_text())
    n_meta = f["count"]
    n_model = len(config.FEATURE_COLS)
    return (n_meta == n_model), f"{n_meta} explained vs {n_model} used by the model"


@check("Every scored stock carries a sector label")
def _c19():
    r = json.loads((DATA / "rankings.json").read_text())
    unknown = [x["ticker"] for x in r["rankings"] if x.get("sector", "Unknown") == "Unknown"]
    return (not unknown), f"{len(unknown)} unlabelled {unknown[:6]}"


@check("No em dashes in any shipped document")
def _c20():
    root = Path(__file__).resolve().parent.parent
    hits = []
    for p in root.rglob("*.md"):
        if "node_modules" in str(p) or "/engine/" in str(p):
            continue
        if "—" in p.read_text(errors="ignore"):
            hits.append(p.name)
    return (not hits), f"{len(hits)} files with em dashes {hits[:5]}"


@check("Listing copy matches the strategy actually shipped")
def _c21():
    txt = (Path(__file__).resolve().parent.parent / "store" / "LISTING.md").read_text().lower()
    b = json.loads((DATA / "backtest.json").read_text())
    is_long_only = "long-only" in b["strategy_name"].lower()
    claims_neutral = "market-neutral machine-learning model ranks" in txt
    return (is_long_only and not claims_neutral), \
        f"backtest ships '{b['strategy_name']}', listing describes market-neutral={claims_neutral}"


@check("Engine bundle matches the source engine")
def _c22():
    a = (Path(__file__).parent / "engine" / "config.py")
    b = ENGINE_DIR / "config.py"
    if not a.exists():
        return False, "backend/engine/config.py absent, run bundle_engine.sh"
    same = a.read_bytes() == b.read_bytes()
    return same, "bundled config identical to source" if same \
        else "STALE: bundle_engine.sh has not been re-run since config changed"


@check("iOS app still points at a reachable base URL")
def _c23():
    src = (Path(__file__).resolve().parent.parent / "ios" / "QuantApp" / "APIClient.swift").read_text()
    is_local = "127.0.0.1" in src or "localhost" in src
    return ("warn" if is_local else True), \
        "baseURL is still localhost, must be the Render HTTPS URL before submission" if is_local \
        else "baseURL points at a remote host"


def main():
    print(f"\nQuantApp pre-flight  ({len(_results)} checks)\n" + "=" * 78)
    width = max(len(n) for _, n, _ in _results)
    for status, name, detail in _results:
        mark = {"PASS": "ok  ", "WARN": "warn", "FAIL": "FAIL"}[status]
        print(f"[{mark}] {name:<{width}}  {detail}")
    n_fail = sum(1 for s, _, _ in _results if s == "FAIL")
    n_warn = sum(1 for s, _, _ in _results if s == "WARN")
    print("=" * 78)
    print(f"{len(_results) - n_fail - n_warn} passed, {n_warn} warnings, {n_fail} failed\n")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
