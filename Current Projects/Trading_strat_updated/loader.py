'''this is the data loader
I will be taking data from the LSEG refinitiv workspace
login details are called from an external .json file (lseg-data.config.json)
'''
import time
import pandas as pd

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

    def get_sp500_constituents(self) -> list:
        """Current S&P 500 constituent RICs via the .SPX chain."""
        data = ld.get_data(universe=["0#.SPX"], fields=["TR.RIC"])
        return data["RIC"].dropna().unique().tolist()

    def load_universe_prices(self, start, end, batch_size=20, retries=3) -> pd.DataFrame:
        """Daily close prices for every current S&P 500 constituent.

        Fetched in batches: asking the UDF gateway for ~500 tickers over a
        16-year window in one call reliably times it out, so tickers are
        chunked and each chunk retried on a gateway timeout. Note batching
        doesn't speed anything up (each ticker costs ~7-10s regardless of
        how many share a call) -- it's here purely for fault isolation and
        visible progress. Expect ~60-90 minutes total for the full S&P 500,
        one-time (cached to parquet after).
        """
        tickers = self.get_sp500_constituents()
        n_batches = -(-len(tickers) // batch_size)  # ceil division
        chunks = []

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            print(f"  [loader] universe batch {i // batch_size + 1}/{n_batches} ({len(batch)} tickers) ...")

            for attempt in range(1, retries + 1):
                try:
                    data = ld.get_history(
                        universe=batch,
                        fields=["TR.PriceClose"],
                        parameters={"Adjusted": 1},
                        start=start,
                        end=end,
                        interval="1D",
                    )
                    break
                except Exception as e:
                    if attempt == retries:
                        raise
                    print(f"  [loader] batch failed ({e}), retrying ({attempt}/{retries}) ...")
                    time.sleep(5 * attempt)

            chunks.append(data)

        combined = pd.concat(chunks, axis=1)
        return self.clean_up(combined)
