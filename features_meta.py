"""
Metadata for the 25 features the engine uses.

Each entry carries the display name, category, a plain description, the
intuition for why it might predict relative returns, the actual formula as
computed in QuantProjectV2/main.py compute_features(), and a one-line note on
what a low cross-sectional score means for that feature.

The formulas are the point of the app, so they must match the code exactly.
Notation used throughout:
    P_t   closing price this week          S_t   SPY closing price this week
    V_t   volume this week                 H, L  weekly high and low
    r_t   log return, ln(P_t / P_{t-1})    r^S_t SPY log return
    MA_n  n-week simple moving average     EMA_n n-week exponential MA
    sd_n  n-week rolling standard deviation

The five yfinance fundamentals (P/E, P/B, profit margin, revenue growth,
debt/equity) were REMOVED on 2026-08-12. They were current values broadcast
across every historical date, so the backtest was reading 2026 accounts into
2018 decisions. See config.py for the measured size of that error.
"""

# Applies to every feature, so it is stated once rather than 25 times.
RANK_NOTE = (
    "Every feature is converted to a cross-sectional percentile within the same "
    "week before the model sees it, so only a stock's rank against its peers "
    "matters, never the raw level. A score of 100 means the highest value in the "
    "S&P 500 that week, 0 the lowest."
)

FEATURES = [
    # ---- Momentum ----
    {"key": "mom_4w", "name": "4-Week Momentum", "category": "Momentum",
     "formula": "P_t / P_{t-4} - 1",
     "description": "Price change over the last 4 weeks.",
     "intuition": "Recent relative winners tend to keep outperforming over short horizons.",
     "low_note": "Scored low: the stock fell, or rose less than its peers, over the past month."},
    {"key": "mom_13w", "name": "13-Week Momentum", "category": "Momentum",
     "formula": "P_t / P_{t-13} - 1",
     "description": "Price change over the last 3 months.",
     "intuition": "The classic 3 to 12 month momentum effect (Jegadeesh and Titman).",
     "low_note": "Scored low: three-month performance trails the rest of the index."},
    {"key": "mom_26w", "name": "26-Week Momentum", "category": "Momentum",
     "formula": "P_t / P_{t-26} - 1",
     "description": "Price change over the last 6 months.",
     "intuition": "Medium-term trend persistence.",
     "low_note": "Scored low: the six-month trend is weak relative to peers."},
    {"key": "mom_52w", "name": "52-Week Momentum", "category": "Momentum",
     "formula": "P_t / P_{t-52} - 1",
     "description": "Price change over the last year.",
     "intuition": "Long-horizon momentum, the strongest of the classic momentum windows.",
     "low_note": "Scored low: a weak year against the index. This is the single most "
                 "influential feature in the model, so a low reading matters more here."},
    # ---- Oscillators ----
    {"key": "rsi_14", "name": "RSI (14)", "category": "Oscillator",
     "formula": "100 - 100 / (1 + RS),   RS = mean(gains)_14 / mean(losses)_14",
     "description": "Wilder's Relative Strength Index over 14 weekly bars.",
     "intuition": "Near 100 means overbought, near 0 oversold. Captures short-term exhaustion.",
     "low_note": "Scored low: losing weeks have outweighed winning weeks recently."},
    {"key": "bb_pos", "name": "Bollinger Band Position", "category": "Oscillator",
     "formula": "(P_t - (MA_20 - 2*sd_20)) / (4*sd_20)",
     "description": "Where price sits inside a 2 standard deviation band around its 20-week mean.",
     "intuition": "Values outside 0 to 1 flag statistically stretched prices.",
     "low_note": "Scored low: price sits near or below the lower band, statistically depressed."},
    # ---- Volume ----
    {"key": "vol_zscore_4w", "name": "Volume Z-Score", "category": "Volume",
     "formula": "(V_t - MA_4(V)) / sd_4(V)",
     "description": "How unusual this week's volume is versus its own 4-week history.",
     "intuition": "Abnormal volume often accompanies new information hitting the stock.",
     "low_note": "Scored low: quieter than usual trading, so little new information arriving."},
    {"key": "vol_trend", "name": "Volume Trend", "category": "Volume",
     "formula": "MA_4(V) / MA_13(V)",
     "description": "Short-term average volume divided by longer-term average volume.",
     "intuition": "Above 1 means participation is building relative to its own quarter.",
     "low_note": "Scored low: participation is fading against its own quarterly average."},
    # ---- Trend / anchors ----
    {"key": "price_52w_high", "name": "Price vs 52-Week High", "category": "Trend",
     "formula": "P_t / max(P)_52",
     "description": "Current price divided by its highest price in the past year.",
     "intuition": "Stocks near their yearly high tend to keep outperforming (anchoring bias).",
     "low_note": "Scored low: trading well below its 52-week high."},
    {"key": "dist_52w_low", "name": "Distance from 52-Week Low", "category": "Trend",
     "formula": "P_t / min(P)_52",
     "description": "Current price divided by its lowest price in the past year.",
     "intuition": "Position within the yearly range, the mirror of the 52-week-high anchor.",
     "low_note": "Scored low: sitting close to the bottom of its yearly range."},
    {"key": "macd_hist", "name": "MACD Histogram", "category": "Trend",
     "formula": "M - EMA_4(M),   M = EMA_4(P) - EMA_9(P)",
     "description": "Gap between the MACD line and its signal line (weekly-adapted EMAs).",
     "intuition": "Measures the acceleration of a trend, not just its direction.",
     "low_note": "Scored low: the trend is decelerating, whichever way it points."},
    {"key": "ma_cross_4_13", "name": "MA Crossover 4/13", "category": "Trend",
     "formula": "MA_4(P) / MA_13(P) - 1",
     "description": "4-week moving average divided by the 13-week moving average, minus 1.",
     "intuition": "Positive means a short-term uptrend has established over the medium term.",
     "low_note": "Scored low: the short average sits below the medium one, a downtrend."},
    {"key": "ma_cross_13_26", "name": "MA Crossover 13/26", "category": "Trend",
     "formula": "MA_13(P) / MA_26(P) - 1",
     "description": "13-week moving average divided by the 26-week moving average, minus 1.",
     "intuition": "A slower, more stable trend confirmation than the 4/13 pair.",
     "low_note": "Scored low: the slower trend confirmation is negative too."},
    {"key": "price_to_ma26", "name": "Price vs 26-Week MA", "category": "Trend",
     "formula": "P_t / MA_26(P)",
     "description": "Current price divided by its 26-week moving average.",
     "intuition": "Distance above or below the medium-term trend line.",
     "low_note": "Scored low: price is below its six-month trend line."},
    # ---- Volatility ----
    {"key": "idio_vol", "name": "Idiosyncratic Volatility", "category": "Volatility",
     "formula": "sqrt(max(Var_52(r) - b^2 * Var_52(r^S), 0)) * sqrt(52),   "
                "b = Cov_52(r, r^S) / Var_52(r^S)",
     "description": "Volatility left over after removing the stock's market exposure.",
     "intuition": "The low-volatility anomaly: high idiosyncratic vol has predicted lower returns.",
     "low_note": "Scored low: little stock-specific noise. Historically this is the "
                 "favourable direction for this feature."},
    {"key": "vol_4w", "name": "Realised Volatility (4w)", "category": "Volatility",
     "formula": "sd_4(r) * sqrt(52)",
     "description": "Annualised standard deviation of returns over 4 weeks.",
     "intuition": "Short-term risk level of the stock.",
     "low_note": "Scored low: unusually calm over the past month."},
    {"key": "vol_13w", "name": "Realised Volatility (13w)", "category": "Volatility",
     "formula": "sd_13(r) * sqrt(52)",
     "description": "Annualised standard deviation of returns over 13 weeks.",
     "intuition": "Medium-term risk level, smoother than the 4-week version.",
     "low_note": "Scored low: calm over the quarter as well, not just the month."},
    {"key": "atr_4w_pct", "name": "ATR % of Price", "category": "Volatility",
     "formula": "100 * MA_4(TR) / P_t,   TR = max(H-L, |H-P_{t-1}|, |L-P_{t-1}|)",
     "description": "Average true range over 4 weeks as a percentage of price.",
     "intuition": "A price-scale-free measure of how wide the trading range is.",
     "low_note": "Scored low: narrow weekly trading ranges."},
    {"key": "up_down_vol", "name": "Up/Down Volatility Ratio", "category": "Volatility",
     "formula": "sd_13(max(r,0)) / sd_13(min(r,0))",
     "description": "Volatility of up-weeks divided by volatility of down-weeks (13w).",
     "intuition": "Above 1 means variability comes mostly from up-moves, an asymmetry.",
     "low_note": "Scored low: the big moves have mostly been downward ones."},
    # ---- Relative strength ----
    {"key": "rel_str_spy_4w", "name": "Relative Strength vs SPY (4w)", "category": "Relative Strength",
     "formula": "(P_t/P_{t-4} - 1) - (S_t/S_{t-4} - 1)",
     "description": "The stock's 4-week return minus the market's 4-week return.",
     "intuition": "Isolates stock-specific strength from the overall market tide.",
     "low_note": "Scored low: underperformed the index over the month."},
    {"key": "rel_str_spy_13w", "name": "Relative Strength vs SPY (13w)", "category": "Relative Strength",
     "formula": "(P_t/P_{t-13} - 1) - (S_t/S_{t-13} - 1)",
     "description": "The stock's 13-week return minus the market's 13-week return.",
     "intuition": "Medium-term outperformance versus the index.",
     "low_note": "Scored low: underperformed the index over the quarter."},
    # ---- Reversal ----
    {"key": "reversal_1w", "name": "1-Week Reversal", "category": "Reversal",
     "formula": "-(P_t / P_{t-1} - 1)",
     "description": "The negative of last week's return.",
     "intuition": "Last week's extreme movers tend to partially revert (liquidity pressure).",
     "low_note": "Scored low: the stock jumped last week, and sharp jumps often give "
                 "part of the move back. The minus sign is deliberate."},
    # ---- Risk / market link ----
    {"key": "rolling_beta", "name": "Rolling Beta (52w)", "category": "Risk",
     "formula": "b_{t-1},   b = Cov_52(r, r^S) / Var_52(r^S)",
     "description": "Sensitivity of the stock to the market over 52 weeks, lagged one week.",
     "intuition": "How much the stock amplifies or dampens market moves.",
     "low_note": "Scored low: moves less than the market, a defensive profile."},
    {"key": "corr_spy_13w", "name": "Correlation with SPY (13w)", "category": "Risk",
     "formula": "corr_13(r, r^S)",
     "description": "Rolling 13-week correlation between the stock and the market.",
     "intuition": "How tightly the stock tracks the index right now.",
     "low_note": "Scored low: currently marching to its own drum rather than the index."},
    # ---- Distribution ----
    {"key": "skew_13w", "name": "Return Skewness (13w)", "category": "Distribution",
     "formula": "skew_13(r)",
     "description": "Asymmetry of the return distribution over 13 weeks.",
     "intuition": "Investors overpay for lottery-like positive skew, depressing its future returns.",
     "low_note": "Scored low: returns are skewed negative, occasional sharp falls."},
]

assert len(FEATURES) == 25, f"expected 25 features, found {len(FEATURES)}"

# Shown wherever the removed fundamentals might be expected.
REMOVED_NOTE = (
    "Five fundamental features (P/E, P/B, profit margin, revenue growth, debt to "
    "equity) were removed in August 2026. The data source supplied only current "
    "values, which the pipeline applied to every historical date, so the backtest "
    "was effectively reading today's accounts into decisions made years ago. "
    "Removing them cut the measured signal strength by roughly three quarters, "
    "which is the honest size of the error. They can only return with "
    "point-in-time data carrying filing dates."
)


# What each measure IS, for someone who has never met the term. The other
# fields assume the reader knows what RSI or a moving average is; this one
# does not. Shown first wherever a measure is explained.
PLAIN = {
    "mom_4w": "Momentum just means: has the price been going up? This one looks at the last month and compares the stock with every other stock.",
    "mom_13w": "Momentum just means: has the price been going up? This one looks at the last 3 months and compares the stock with every other stock.",
    "mom_26w": "Momentum just means: has the price been going up? This one looks at the last 6 months and compares the stock with every other stock.",
    "mom_52w": "Momentum just means: has the price been going up? This one looks at the whole last year and compares the stock with every other stock.",
    "rsi_14": "RSI (Relative Strength Index) is a 0 to 100 gauge of how one-sided recent trading has been. It compares the size of the up weeks with the size of the down weeks over the last 14 weeks. Above 70 usually means the price has risen fast and may be overheated; below 30 means it has fallen fast.",
    "bb_pos": "Bollinger Bands draw a channel around the stock's average price over the last 20 weeks, wide when the price is jumpy and narrow when it is calm. This measure says where today's price sits inside that channel: near the top, the middle or the bottom.",
    "vol_zscore_4w": "Volume is how many shares changed hands. This asks whether this week's trading was unusually busy or unusually quiet compared with the stock's own last month. A z-score just means how many normal-sized steps away from usual it is.",
    "vol_trend": "Compares recent trading activity with its longer-term level. Above 1 means more people have been trading the stock lately.",
    "price_52w_high": "How close the price is to the highest it has been in the past year. A value of 1 means it is at its yearly high right now.",
    "dist_52w_low": "How far the price has climbed from its lowest point in the past year. A value of 1 means it is at its yearly low right now.",
    "macd_hist": "MACD compares a fast-moving average of the price with a slower one to show whether a trend is speeding up or slowing down. The histogram is the gap between MACD and its own average: positive and growing means the upward push is getting stronger.",
    "ma_cross_4_13": "A moving average is the average price over the last few weeks, which smooths out the noise. This compares the 1-month average with the 3-month average: positive means the recent price is above the longer trend.",
    "ma_cross_13_26": "A moving average is the average price over the last few weeks, which smooths out the noise. This compares the 3-month average with the 6-month average: positive means the medium trend is above the longer one.",
    "price_to_ma26": "Today's price compared with its average over the last 6 months. Above 1 means it is trading above its usual level.",
    "idio_vol": "Idiosyncratic means its own. This is how much the stock jumps around for reasons that have nothing to do with the overall market, such as company news.",
    "vol_4w": "Volatility is how much the price swings from week to week. High volatility means big ups and downs. This one uses the last month.",
    "vol_13w": "Volatility is how much the price swings from week to week. High volatility means big ups and downs. This one uses the last 3 months.",
    "atr_4w_pct": "ATR (Average True Range) is the typical distance between a week's highest and lowest price. It is shown as a percentage of the price so that expensive and cheap stocks can be compared fairly.",
    "up_down_vol": "Compares how big the stock's up weeks have been with how big its down weeks have been. Above 1 means its rises have been larger than its falls.",
    "rel_str_spy_4w": "SPY is a fund that tracks the S&P 500, the 500 largest US companies, so it stands for the market. This is the stock's return over the last month minus the market's: positive means it beat the market.",
    "rel_str_spy_13w": "SPY is a fund that tracks the S&P 500, the 500 largest US companies, so it stands for the market. This is the stock's return over the last 3 months minus the market's: positive means it beat the market.",
    "reversal_1w": "Last week's return turned upside down. Stocks that drop sharply in one week often bounce back a little the next, so a big fall gives a high score here.",
    "rolling_beta": "Beta measures how strongly a stock follows the market. A beta of 1 moves with the market, 2 moves twice as much, 0.5 half as much. It is measured over the last year.",
    "corr_spy_13w": "Correlation measures how closely two things move together, from -1 (always opposite) to 1 (always in step). This is the stock against the market over the last 3 months.",
    "skew_13w": "Skewness describes whether a stock's surprises have been mostly good or mostly bad. Positive skew means a few unusually large up weeks; negative means a few unusually large down weeks.",
}

for _f in FEATURES:
    _f["plain"] = PLAIN[_f["key"]]
