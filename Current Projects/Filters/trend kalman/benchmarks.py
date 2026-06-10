'''here are some simple benchmarks for trend extraction

to test:
trend detection accuracy
lag
noise robustness
parameter stability


HP filter is ground truth for trend classification and reference!!lamba = 1600 is quarterly data, 129600 is minutely data, 1296000 is secondly data.
'''
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
class SimpleTrend:
    def __init__(self, price):
        self.price = price
    
    def SMA(self,window):
        return {"slope": self.price.rolling(window=window).mean().diff(), "trend": self.price.rolling(window=window).mean()}
    
    def EMA(self,window):
        return {"slope": self.price.ewm(span=window, adjust=False).mean().diff(), "trend": self.price.ewm(span=window, adjust=False).mean()}
    
    def SG_filter(self,window,polyorder):
        return {"slope": savgol_filter(self.price, window_length=window, polyorder=polyorder, deriv=1), "trend": savgol_filter(self.price, window_length=window, polyorder=polyorder, deriv=0)} 
    
    def SG_causal(self, window, polyorder):
        # one-sided SG: at each bar fit polynomial to the trailing window only
        n = len(self.price)
        trend = np.full(n, np.nan)
        slope = np.full(n, np.nan)
        x = np.arange(window)
        for i in range(window - 1, n):
            y = self.price.iloc[i - window + 1 : i + 1].values
            coeffs = np.polyfit(x, y, polyorder)
            trend[i] = np.polyval(coeffs, window - 1)
            # derivative of polynomial at the last point
            slope[i] = np.polyval(np.polyder(coeffs), window - 1)
        return {
            "trend": pd.Series(trend, index=self.price.index),
            "slope": pd.Series(slope, index=self.price.index),
        }

    def HP_filter(self, lamb=1600):
        cycle, trend = hpfilter(self.price, lamb=lamb) # cycle is the residual, cycle = price - trend
        return {"slope": trend.diff(), "trend": trend}