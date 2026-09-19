'''Cross-sectional momentum: ranks the universe by risk-adjusted (12-1
month) momentum. See backtest.py's vol_weighted_signals / equal_weight_signals
for how the resulting top-N names are turned into long-only book weights.
'''

import pandas as pd


def momentum_scores(price_data: pd.DataFrame, lookback: int) -> pd.Series:
    """Risk-adjusted momentum score for every ticker with enough history --
    the same score CMOM.risk_adj_signals ranks by, exposed directly so it
    can be filtered by strength, not just cross-sectional rank."""
    daily_ret = price_data.pct_change(fill_method=None)
    momentum  = price_data.shift(21) / price_data.shift(lookback) - 1
    vol       = daily_ret.rolling(lookback).std()
    return (momentum / vol).iloc[-1].dropna()


def top_momentum_names(price_data: pd.DataFrame, lookback: int, top_n: int,
                        min_zscore: float | None = None) -> pd.Index:
    """Top-N tickers by risk-adjusted momentum -- ranking only, no weights.

    If `min_zscore` is given, a name must also clear that many cross-
    sectional standard deviations above that rebalance's own mean score
    (e.g. 2.33 ~ one-tailed 1% under a normal cross-section) -- a weak
    month can then return fewer than top_n names, rather than always
    filling the book from whatever ranked highest even when the whole
    cross-section's momentum is statistically indistinguishable from noise."""
    score = momentum_scores(price_data, lookback)
    if min_zscore is not None:
        z = (score - score.mean()) / score.std()
        score = score[z >= min_zscore]
    return score.nlargest(top_n).index
