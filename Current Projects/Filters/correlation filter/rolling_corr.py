'''rolling correlation:

takes two price series and a window for corr and smoothing:

recommended: 
window_corr = 30
window_smooth = 10

we can iterate this for larger sets of assets, but start simple'''

import numpy as np
class RollingCorr:
    def __init__ (self, price1, price2, window_corr, window_smooth):
        self.price1 = price1
        self.price2 = price2
        self.window_corr = window_corr
        self.window_smooth = window_smooth
        
    def log_ret(self, price):
        return np.log(price / price.shift(1))
    
    def rolling_corr(self):
        log_ret1 = self.log_ret(self.price1)
        log_ret2 = self.log_ret(self.price2)
        return log_ret1.rolling(window=self.window_corr).corr(log_ret2)
    
    def smooth_corr(self, corr_series):
        return corr_series.rolling(window=self.window_smooth).mean()
    
    def fisher_transform(self, corr_series):
        return 0.5 * np.log((1 + corr_series) / (1 - corr_series))
    
    def run_smoothed_fisher_corr(self):
        corr_series = self.rolling_corr()
        fisher_series = self.fisher_transform(corr_series)
        smoothed_corr = self.smooth_corr(fisher_series)
        return smoothed_corr
        
    
