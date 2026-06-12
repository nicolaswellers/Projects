'''I will be measuring the autocorrelation of returns 
I will be taking pearson correlation of returns with itself at different lags
I will also provide an option for a mean autocorrelation across a period

if rho > 0 then positive autocorrelation
if rho < 0 then negative autocorrelation
if rho = 0 then no autocorrelation, random walk (no memory)

the maths would look like:

rho = sum((r_t - mean(r)) * (r_(t-lag) - mean(r))) / sum((r_t - mean(r))^2)
'''

import pandas as pd
class Autocorrelation:
    def __init__(self, returns : pd.Series):
        self.returns = returns

    def calculate_autocorrelation(self, lag):
        return self.returns.autocorr(lag=lag)
    
    def mean_autocorrelation(self, max_lag=4):
        autocorrelations = []
        
        for i in range(1, max_lag + 1):
            autocorrelations.append(self.calculate_autocorrelation(i))
        
        return sum(autocorrelations) / len(autocorrelations)