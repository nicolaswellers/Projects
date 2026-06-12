'''Symmetric CUSUM filters for Hurst and Autocorrelation signals.

Reduces sensitivity by requiring sustained evidence before flipping regime.
State only changes when the opposing accumulator builds enough evidence to cross h.

Parameters:
    k : allowance per step — noise smaller than this is ignored
    h : trigger threshold — accumulated evidence required to flip state

State outputs:
     1 : trending
    -1 : choppy
     0 : undecided (before first trigger)
'''

import numpy as np
import pandas as pd


class HurstPersistence:
    '''Simple persistence filter on the Hurst exponent.

    Only flips state after N consecutive days on the same side of the threshold.
    A single day crossing back resets the counter but does NOT flip state —
    the current state holds until the opposite side accumulates N consecutive days.

    State outputs:
         1 : trending  (N consecutive days above threshold)
        -1 : choppy    (N consecutive days below threshold)
         0 : undecided (before first trigger)
    '''

    def __init__(self, n=10, window=15, threshold=0.5):
        self.n         = n       # required hits within the window
        self.window    = window  # rolling window size in days
        self.threshold = threshold

    def filter(self, hurst: pd.Series) -> pd.Series:

        clean = hurst.dropna()
        above = (clean > self.threshold).values
        n     = len(above)

        state         = np.zeros(n)
        current_state = 0

        for t in range(n):
            window_start = max(0, t - self.window + 1)
            window       = above[window_start : t + 1]

            count_above = window.sum()
            count_below = len(window) - count_above

            # only evaluate once we have a full window
            if len(window) == self.window:
                if count_above >= self.n:
                    current_state = 1
                elif count_below >= self.n:
                    current_state = -1

            state[t] = current_state

        return pd.Series(state, index=clean.index, name="hurst_persistence")


class HurstCUSUM:
    '''Symmetric CUSUM on the rolling Hurst exponent.

    Centres Hurst around 0.5 so:
        x_t > 0  →  evidence of trending
        x_t < 0  →  evidence of choppy

    Recommended starting params:
        k = 0.02  (ignore sub-2% Hurst deviations from threshold)
        h = 0.10  (require ~5 consistent steps to flip, since 5 * 0.02 = 0.10)
    '''

    def __init__(self, k=0.02, h=0.10, threshold=0.5):
        self.k         = k
        self.h         = h
        self.threshold = threshold

    def filter(self, hurst: pd.Series) -> pd.DataFrame:

        clean = hurst.dropna()
        x     = (clean - self.threshold).values
        n     = len(x)

        c_pos         = np.zeros(n)
        c_neg         = np.zeros(n)
        state         = np.zeros(n)
        current_state = 0

        for t in range(1, n):
            c_pos[t] = max(0.0, c_pos[t-1] + x[t] - self.k)
            c_neg[t] = max(0.0, c_neg[t-1] - x[t] - self.k)

            if c_pos[t] >= self.h:
                current_state = 1
                c_neg[t]      = 0.0

            elif c_neg[t] >= self.h:
                current_state = -1
                c_pos[t]      = 0.0

            state[t] = current_state

        return pd.DataFrame({
            "c_pos": c_pos,
            "c_neg": c_neg,
            "state": state,
        }, index=clean.index)


class AutocorrCUSUM:
    '''Symmetric CUSUM on the rolling mean autocorrelation.

    Centres autocorrelation around 0 so:
        x_t > 0  →  evidence of trending (positive serial dependence)
        x_t < 0  →  evidence of choppy (mean-reverting)

    Recommended starting params:
        k = 0.01  (autocorrelation values are small, tighter allowance)
        h = 0.05  (require ~5 consistent steps to flip, since 5 * 0.01 = 0.05)
    '''

    def __init__(self, k=0.01, h=0.05, threshold=0.0):
        self.k         = k
        self.h         = h
        self.threshold = threshold

    def filter(self, autocorr: pd.Series) -> pd.DataFrame:

        clean = autocorr.dropna()
        x     = (clean - self.threshold).values
        n     = len(x)

        c_pos         = np.zeros(n)
        c_neg         = np.zeros(n)
        state         = np.zeros(n)
        current_state = 0

        for t in range(1, n):
            c_pos[t] = max(0.0, c_pos[t-1] + x[t] - self.k)
            c_neg[t] = max(0.0, c_neg[t-1] - x[t] - self.k)

            if c_pos[t] >= self.h:
                current_state = 1
                c_neg[t]      = 0.0

            elif c_neg[t] >= self.h:
                current_state = -1
                c_pos[t]      = 0.0

            state[t] = current_state

        return pd.DataFrame({
            "c_pos": c_pos,
            "c_neg": c_neg,
            "state": state,
        }, index=clean.index)
