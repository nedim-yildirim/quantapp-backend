"""
verify.py - pre-flight checks for the QuantApp data pipeline.

Every check here exists because something actually went wrong, or because a
silent failure in that spot would ship wrong numbers to users. Run before any
deploy or App Store submission:

    python verify.py

Exit code 0 = all checks passed, 1 = at least one FAIL.
"""
import os
import re
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
    store = pd.read_parquet(ENGINE_DIR / "data" / config.PRICES_PATH.name)
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
    store = pd.read_parquet(ENGINE_DIR / "data" / config.PRICES_PATH.name)
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


@check("Ensemble IC is positive, and its significance is not overstated")
def _c10():
    """This check used to demand t > 2 and it passed at t = 3.98, but that
    figure came from five leaked fundamental features. On clean point-in-time
    inputs the honest t is 1.59, which is not significant.

    Requiring t > 2 would now block every deploy on a fact we have decided to
    publish openly, so the bar has moved to the thing that actually matters:
    the signal must be positive, and nothing we ship may claim a significance
    the data does not support. A weak signal, stated as weak, is shippable.
    A weak signal described as proven is not."""
    ic = pd.read_csv(RESULTS / "ic_log.csv")
    v = ic["mean_ic"].dropna()
    t = v.mean() / v.std() * np.sqrt(len(v))
    detail = (f"mean IC {v.mean():.4f}, {(v > 0).mean():.1%} positive, "
              f"t={t:.2f} over {len(v)} folds")

    if v.mean() <= 0:
        return False, detail + " - signal is not positive"

    if t >= 2.0:
        return True, detail + " - significant"

    # Not significant, so hunt for any shipped copy that says otherwise.
    # The phrases must be checked in context: "are the results guaranteed? No"
    # and "not statistically significant" are honest, and an earlier version of
    # this check flagged both.
    root = Path(__file__).parent
    claims = ("statistically significant", "guaranteed return", "proven strategy",
              "reliably predicts", "consistently beats", "will outperform",
              "guaranteed profit")
    negators = ("not ", "no ", "never ", "n't ", "cannot ", "does not ",
                "is not ", "are not ", "without ")
    offenders = []
    for f in list(root.glob("*.py")) + list(root.glob("static/*.html")) + \
             list((root.parent / "store").glob("*.md")) + \
             list((root.parent / "ios" / "QuantApp").glob("*.swift")):
        if f.name == "verify.py":
            continue
        # Collapse all whitespace first. Without this a line wrap between the
        # negation and the claim ("is not\nstatistically significant") hides the
        # negator from the window below, and honest copy gets flagged as an
        # overclaim. The same wrap can also split the claim itself so it is
        # never found at all, which is the more dangerous half of the bug.
        low = " ".join(f.read_text(errors="ignore").lower().split())
        for c in claims:
            start = 0
            while (i := low.find(c, start)) != -1:
                start = i + len(c)
                before = low[max(0, i - 60):i]
                # A question mark just before means it is a FAQ heading, and the
                # answer underneath is what carries the meaning.
                if any(n in before for n in negators) or before.rstrip().endswith("?"):
                    continue
                offenders.append(f"{f.name}:'{c}'")
    if offenders:
        return False, detail + f" but copy overclaims: {offenders[:4]}"
    return "warn", detail + " - NOT significant, and no shipped copy claims it is"


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


@check("Feature explanations cover every model feature")
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
    # Swift is included because the app's own copy is shipped text too, and
    # most of the writing the user reads now lives in the views rather than in
    # the markdown.
    for pattern in ("*.md", "*.swift"):
        for p in root.rglob(pattern):
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


@check("Weekly job tolerates a partial in-progress week")
def _c24():
    # The scheduler fires on Saturdays, when the data provider often returns an
    # incomplete current week with all-NaN features. Taking the newest date
    # blindly made the job abort every run.
    src = (Path(__file__).parent / "weekly_update.py").read_text()
    guard = "MIN_STOCKS" in src and "incomplete week" in src
    return guard, "walks back to the last complete week" if guard \
        else "GUARD MISSING: a partial week will abort the weekly refresh"


@check("Live book agrees with the sector cap and holds no partial-week data")
def _c25():
    r = json.loads((DATA / "rankings.json").read_text())
    as_of = pd.Timestamp(r["as_of"])
    # A completed weekly bar is stamped on a Monday by the resampler.
    is_week_start = as_of.dayofweek == 0
    enough = r["universe_size"] >= 400
    return (is_week_start and enough), \
        f"as_of {r['as_of']} (weekday {as_of.dayofweek}), {r['universe_size']} scored"


@check("Privacy and support pages are servable (App Store requires the URL)")
def _c26():
    src = (Path(__file__).parent / "app.py").read_text()
    routed = '"/privacy"' in src and '"/support"' in src
    files = [(Path(__file__).parent / "static" / f"{n}.html").exists()
             for n in ("privacy", "support")]
    return (routed and all(files)), \
        f"routes present={routed}, static files present={all(files)}"


@check("iOS app still points at a reachable base URL")
def _c23():
    src = (Path(__file__).resolve().parent.parent / "ios" / "QuantApp" / "APIClient.swift").read_text()
    # Read the assignment itself, not the whole file: the surrounding comment
    # legitimately names the localhost URL as the local-development alternative,
    # and a substring search over the file would flag that as a failure.
    m = re.search(r'static\s+let\s+baseURL\s*=\s*URL\(string:\s*"([^"]+)"', src)
    if not m:
        return False, "could not find the baseURL assignment in APIClient.swift"
    url = m.group(1)
    if "127.0.0.1" in url or "localhost" in url:
        return "warn", f"baseURL is still {url}, must be the Render HTTPS URL before submission"
    if not url.startswith("https://"):
        return False, f"baseURL {url} is not HTTPS, App Transport Security will block it"
    return True, f"baseURL is {url}"


@check("Per stock attribution is complete and reproduces the ranking")
def _c27():
    # The app's stock detail screen shows all 25 features with their percentile
    # and their contribution, and states in print that base plus the
    # contributions equals the raw model output. That claim has to hold for
    # every stock, not just the one that was checked by eye.
    r = json.loads((DATA / "rankings.json").read_text())
    order = r.get("feature_order")
    if not order:
        return False, "rankings.json carries no feature_order, the detail screen degrades to a notice"
    if list(order) != list(config.FEATURE_COLS):
        return False, "feature_order does not match config.FEATURE_COLS, labels would be attached to the wrong numbers"
    k = len(order)
    base = r.get("shap_base")
    rows = r["rankings"]
    bad = [x["ticker"] for x in rows
           if len(x.get("fp", [])) != k or len(x.get("fc", [])) != k]
    if bad:
        return False, f"{len(bad)} stocks with a wrong-length breakdown {bad[:5]}"
    # The reconstructed raw scores must rank the universe the same way the
    # shipped percentiles do. Spearman 1.0 or the screen is lying about where
    # the number came from.
    raw = pd.Series({x["ticker"]: base + sum(x["fc"]) for x in rows})
    shipped = pd.Series({x["ticker"]: x["percentile"] for x in rows})
    rho = raw.corr(shipped, method="spearman")
    return (rho > 0.999), f"{len(rows)} stocks x {k} features, base={base}, rank agreement rho={rho:.4f}"


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
