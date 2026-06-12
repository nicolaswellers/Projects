'''this is the data loader
I will be taking data from the LSEG refinitiv workspace
login details are called from an external .json file (lseg-data.config.json)

For backtesting without LSEG, use load_spy_cached() which pulls from yfinance
and caches to the Data/ directory as a CSV.
'''
import os
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

try:
    import lseg.data as ld
    _HAS_LSEG = True
except ImportError:
    _HAS_LSEG = False

class DataLoader:
    def __init__(self, config_path="./lseg-data.config.json", open_session=True):
        self.config_path = config_path
        if open_session and _HAS_LSEG:
            ld.open_session()

    def clean_up(self, data):
        data = data.copy()                                  # avoid mutating the original
        data.index = pd.to_datetime(data.index)             # ensure index is datetime
        data = data.sort_index()                            # chronological order
        data = data[~data.index.duplicated(keep="last")]    # de-duplicate
        data = data.apply(pd.to_numeric, errors="coerce")   # non-numerics -> NaN
        data = data.dropna(how="all")                       # drop fully-NaN rows
        return data

    def _to_series(self, data):
        """Convert single-column DataFrame from get_history into a 1-D Series."""
        clean = self.clean_up(data)
        if isinstance(clean, pd.DataFrame):
            clean = clean.iloc[:, 0]
        return clean

    def load_daily_close(self, ticker, start, end):
        data = ld.get_history(
            universe=[ticker],
            fields=["TR.PriceClose"],
            parameters={"Adjusted": 1},
            start=start,
            end=end,
            interval="1D",
        )
        return self._to_series(data)

    def _load_etf_cached(self, ticker: str, start: str, end: str, force: bool) -> pd.Series:
        cache_path = os.path.join(DATA_DIR, f"{ticker.lower()}_close.csv")

        if not force and os.path.exists(cache_path):
            print(f"  [loader] {ticker} loaded from cache")
            return pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze()

        print(f"  [loader] downloading {ticker} {start} → {end} ...")
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        s = raw["Close"].squeeze()
        s.index = pd.to_datetime(s.index)
        s.name = ticker
        os.makedirs(DATA_DIR, exist_ok=True)
        s.to_csv(cache_path)
        print(f"  [loader] {ticker} cached to {cache_path}")
        return s

    def load_spy_cached(self, start="2010-01-01", end="2026-01-01", force=False) -> pd.Series:
        return self._load_etf_cached("SPY", start, end, force)

    def load_iwm_cached(self, start="2010-01-01", end="2026-01-01", force=False) -> pd.Series:
        return self._load_etf_cached("IWM", start, end, force)