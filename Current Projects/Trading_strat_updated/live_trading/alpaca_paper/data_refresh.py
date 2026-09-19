"""Live daily price refresh for the momentum/vol-ratio lookback window.

The cached Data/universe_prices.parquet is LSEG-sourced and re-pulling it
via loader.py costs ~60-90 minutes for the full S&P 500 history (see that
module's own docstring) and needs an active LSEG Workspace/Eikon session
open on the machine -- both impractical for something meant to run
unattended once a day.

Instead, this pulls only the trailing ~1.5 years of daily bars -- comfortably
more than the strategy's longest lookback (252-day momentum window, 126-day
vol window) -- directly from Alpaca's own historical data API for the
current universe's ticker list. That's fast (a handful of seconds, no
desktop session required) and keeps the live pipeline on one data venue:
the same one orders execute against, so there's no LSEG-vs-Alpaca price
convention mismatch to worry about for the numbers that actually drive
today's decision.

The *set* of tickers is normally the live S&P 500 membership from
sp500_membership.py (Wikipedia-sourced, refreshed every run -- see
live_runner.py's load_universe), not the static cached LSEG list; this
module only fetches PRICES for whatever ticker list it's given, RIC-style
or already Alpaca-native (see `already_alpaca_native`).

Caveat worth knowing about, not hidden: because only the tail of the
lookback window comes from Alpaca while the strategy was originally
validated on LSEG's own "Adjusted" close, a live momentum score computed
here can differ very slightly from what an equivalent LSEG-only calculation
would give, due to differing dividend/split-adjustment conventions between
the two vendors. This is a live-inputs testing bridge, not a claim that the
two are pixel-identical.
"""
import os

import pandas as pd
from datetime import datetime, timedelta, timezone

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca_client import to_alpaca_symbol
from dotenv_util import load_dotenv

LOOKBACK_CALENDAR_DAYS = 550  # comfortably covers 252 + 126 trading days incl. weekends/holidays


def _data_client() -> StockHistoricalDataClient:
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set -- see alpaca_client.py")
    return StockHistoricalDataClient(key, secret)


def fetch_live_universe(ric_columns: list[str], already_alpaca_native: bool = False) -> pd.DataFrame:
    """Daily close prices for `ric_columns` over the trailing lookback
    window, fetched fresh from Alpaca and returned with the SAME column
    labels so it's a drop-in replacement for the cached parquet in the
    momentum/vol-ratio functions. Tickers Alpaca has no data for (delisted,
    mismapped RIC suffix, etc.) come back as an all-NaN column -- the
    existing `.ffill()` handling in those functions already tolerates that
    the same way missing history does in the backtest.

    `already_alpaca_native=True` skips the RIC -> Alpaca symbol conversion
    (see alpaca_client.to_alpaca_symbol) -- use this for a ticker list that
    already came back in Alpaca/NYSE format, e.g. from
    sp500_membership.get_current_sp500_tickers(). Applying the RIC
    conversion to an already-native list would wrongly mangle a real
    dot-format ticker like "BRK.B" down to "BRK"."""
    if already_alpaca_native:
        ric_to_alpaca = {t: t for t in ric_columns}
    else:
        ric_to_alpaca = {ric: to_alpaca_symbol(ric) for ric in ric_columns}
    alpaca_symbols = sorted(set(ric_to_alpaca.values()))

    client = _data_client()
    start = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    request = StockBarsRequest(symbol_or_symbols=alpaca_symbols, timeframe=TimeFrame.Day, start=start)
    bars = client.get_stock_bars(request).df  # MultiIndex (symbol, timestamp)

    close = bars["close"].unstack(level=0)
    close.index = pd.to_datetime(close.index.date)  # drop intraday timestamp component
    close = close[~close.index.duplicated(keep="last")].sort_index()

    missing = [ric for ric, sym in ric_to_alpaca.items() if sym not in close.columns]
    series_list = [close[sym] if sym in close.columns else pd.Series(float("nan"), index=close.index)
                   for sym in ric_to_alpaca.values()]
    # explicit `keys=` (rather than relying on dict-input ordering, which
    # pd.concat doesn't guarantee) pins each column's label to its data
    out = pd.concat(series_list, axis=1, keys=list(ric_to_alpaca.keys()))
    if missing:
        print(f"  [data_refresh] no Alpaca data for {len(missing)}/{len(ric_columns)} tickers "
              f"(left as NaN): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return out


if __name__ == "__main__":
    import sys
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT)
    from config import UNIVERSE_PATH

    cached = pd.read_parquet(UNIVERSE_PATH)
    live = fetch_live_universe(list(cached.columns))
    print(f"fetched {live.shape[0]} days x {live.shape[1]} tickers, "
          f"{live.index[0].date()} -> {live.index[-1].date()}")
    print(f"non-null tickers: {live.notna().any().sum()}/{live.shape[1]}")
