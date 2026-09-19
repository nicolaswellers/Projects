"""Live S&P 500 constituent list -- decouples the LIVE universe from the
static, LSEG-built Data/universe_prices.parquet cache (which only reflects
whatever the index looked like whenever `python main.py --force` was last
run). Fetched from Wikipedia's community-maintained "List of S&P 500
companies" page -- no LSEG desktop session, no paid data subscription,
takes ~1 second.

Why this is safe to rely on: it's one of the most-watched, most-edited
pages on Wikipedia specifically because of how many people/tools depend on
it for exactly this purpose; index changes typically show up within a day
of being announced. It's still a community source, not an official S&P
Dow Jones Indices feed -- for anything where correctness matters more than
convenience (e.g. rebuilding the historical backtest), keep using LSEG's
own chain (loader.DataLoader.get_sp500_constituents).

Tickers come back already in Alpaca/NYSE-native format (e.g. "BRK.B",
"BF.B" with a literal dot for share classes) -- NOT LSEG RIC format, so
these should NOT be passed through alpaca_client.to_alpaca_symbol (that
would incorrectly strip "BRK.B" down to "BRK").
"""
import pandas as pd
import requests
from io import StringIO

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def get_current_sp500_tickers() -> list[str]:
    """Current S&P 500 constituent tickers, Alpaca-ready (no RIC suffix to
    strip). Raises on network failure / page-structure changes -- callers
    should decide whether to fall back to the cached ticker list rather
    than silently trading a stale or empty universe."""
    resp = requests.get(WIKIPEDIA_URL, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    table = pd.read_html(StringIO(resp.text))[0]
    return table["Symbol"].str.strip().tolist()


if __name__ == "__main__":
    tickers = get_current_sp500_tickers()
    print(f"{len(tickers)} tickers, e.g. {tickers[:5]}")
    print(f"dot-format (share class) tickers: {[t for t in tickers if '.' in t]}")
