"""Research idea: instead of letting a vol-spike (or stop-loss/capital-halve)
cut sit in cash, redirect the cut fraction into a "safe-haven" asset so the
book stays 100% invested at all times -- momentum + hedge asset rather than
momentum + cash.

Rationale: some assets have historically tended to hold up or rally during
equity stress, so the periods when the vol-spike overlay is cutting momentum
exposure (precisely the highest-stress periods) may be periods the hedge
asset performs well -- partially offsetting what would otherwise be idle
cash, without reintroducing the momentum book's own crash risk. Three
candidates tested:

  - GOLD (GLD, SPDR Gold Shares): classic flight-to-safety commodity.
  - TLT (iShares 20+ Year Treasury Bond ETF): long-duration bonds, the
    classic equity-crash hedge (benefits from both flight-to-safety AND
    the rate cuts that typically accompany a real equity crisis) -- but
    also the most exposed of the three to a "bad" outcome where inflation
    or rate-hike fears are THEMSELVES the equity stress trigger (2022 is
    the textbook case: stocks and long bonds fell together).
  - IEF (iShares 7-10 Year Treasury Bond ETF): the same idea with roughly
    half TLT's duration -- less rate-cut upside, but also less exposure to
    the "bonds and stocks fall together" failure mode.

See `run_long_only`'s `hedge_ret` parameter in backtest.py for the actual
mechanism: raw_gross = effective_exposure * momentum_return +
(1 - effective_exposure) * hedge_return, where effective_exposure =
capital_scalar * spike_scalar. With the champion's non-continuous vol-spike
scalar (VOL_SPIKE_SCALAR = 0.5), "cut" days put exactly 50% of capital into
the hedge asset.

Data: Data/{gold,tlt,ief}_daily.parquet, all fetched via LSEG
(loader.DataLoader().load_daily_close(ric, start, end) for "GLD.P", "TLT.O",
"IEF.O" respectively), covering the same 2010-01-04 to 2025-12-31 window as
the equity universe.
"""
import os

import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
GOLD_PATH = os.path.join(_DATA_DIR, "gold_daily.parquet")
TLT_PATH  = os.path.join(_DATA_DIR, "tlt_daily.parquet")
IEF_PATH  = os.path.join(_DATA_DIR, "ief_daily.parquet")


def _load_hedge_returns(parquet_path: str, dates: pd.DatetimeIndex) -> pd.Series:
    """Daily returns for the asset at `parquet_path`, reindexed/ffilled onto
    `dates` (the universe's own date index), with any residual gap (before
    the asset's data starts, or an isolated missing print) filled with 0.0
    -- meaning those specific days silently fall back to the "sits in cash"
    behavior rather than crashing or fabricating a return."""
    close = pd.read_parquet(parquet_path)["close"]
    ret = close.pct_change(fill_method=None)
    return ret.reindex(dates).fillna(0.0)


def load_gold_returns(dates: pd.DatetimeIndex) -> pd.Series:
    return _load_hedge_returns(GOLD_PATH, dates)


def load_tlt_returns(dates: pd.DatetimeIndex) -> pd.Series:
    return _load_hedge_returns(TLT_PATH, dates)


def load_ief_returns(dates: pd.DatetimeIndex) -> pd.Series:
    return _load_hedge_returns(IEF_PATH, dates)
