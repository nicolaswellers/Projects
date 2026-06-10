'''Two-sided CUSUM for short-term structural break detection, should be alongside HMM

Idea: 
S_t = sigma(x_i - mu_0) for i = 1 to t
where mu_0 is the expected value of x_i under the null hypothesis (no change), or simply rolling mean
x_i is the observed value at time i

S_t+ = max(0, S_t + x_t - k)
S_t- = min(0, S_t + x_t + k)

x_t is the observed value at time t
S_t is the cumulative sum up to time t

because all x_t are z-scored,  0.2< k < 0.5 is a common choice for the reference value,
0.2 for weak shifts,
0.3 is balanced,
0.5 for very strong shifts,

if S_t+ > h, signal positive change
if S_t- < -h, signal negative change

inputs: shocks of credit, liquidity, vol, correlation!

output: 
1 upward break
-1 downward break
0 stable regime
'''

import numpy as np
import pandas as pd

class CUSUMDetector:
    def __init__(self, k=0.5, h=2.0):
        self.k = k
        self.h = h

    def detect(self, x):
        x = x.dropna().values

        s_pos = np.zeros(len(x))
        s_neg = np.zeros(len(x))

        signals = np.zeros(len(x))

        for t in range(1, len(x)):
            s_pos[t] = max(0, s_pos[t-1] + x[t] - self.k)
            s_neg[t] = min(0, s_neg[t-1] + x[t] + self.k)

            if s_pos[t] > self.h:
                signals[t] = 1   # upward break
                s_pos[t] = 0     # reset

            elif s_neg[t] < -self.h:
                signals[t] = -1  # downward break
                s_neg[t] = 0     # reset

        return pd.Series(signals, index=x.index)