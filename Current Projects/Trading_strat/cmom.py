'''cross sectional momentum strategy
takes a cross sectio of russel 2000 stocks and ranks by risk-adjusted momentum
'''

import pandas as pd

class CMOM:
    def __init__(self, universe, lookback=252, top_n=50):
        self.universe = universe
        self.lookback = lookback
        self.top_n = top_n
        
        
    def risk_adj_signals(self, price_data: pd.DataFrame) -> pd.Series:

        daily_ret = price_data.pct_change()

        # cumulative return over lookback, skip last month
        momentum = price_data.shift(21) / price_data.shift(self.lookback) - 1

        # vol over same lookback window
        vol = daily_ret.rolling(self.lookback).std()

        # risk-adjusted signal at latest date
        score = (momentum / vol).iloc[-1]
        score = score.dropna()

        top_stocks    = score.nlargest(self.top_n).index
        bottom_stocks = score.nsmallest(self.top_n).index

        signals = pd.Series(0, index=price_data.columns)
        signals[top_stocks]    =  1
        signals[bottom_stocks] = -1

        return signals
