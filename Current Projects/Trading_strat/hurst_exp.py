'''Hurst Exponent via Rescaled Range (R/S) analysis.

Methodology:
    1. Compute R/S at multiple sub-window sizes n
    2. Fit OLS of log(R/S) on log(n)
    3. Slope = H

Interpretation:
    H < 0.5 : mean-reverting / choppy
    H = 0.5 : random walk
    H > 0.5 : trending / persistent
'''

import numpy as np
import pandas as pd


class HurstExponent:

    def __init__(self, min_window=20):
        self.min_window = min_window

    # ==================================================
    # R/S for a single window of returns
    # ==================================================

    def _rs(self, returns: np.ndarray) -> float:

        n = len(returns)
        if n < 4:
            return np.nan

        r_bar = returns.mean()
        Y = np.cumsum(returns - r_bar)

        R = Y.max() - Y.min()
        S = returns.std(ddof=1)

        if S < 1e-12:
            return np.nan

        return R / S

    # ==================================================
    # Hurst exponent via OLS across sub-window sizes
    # ==================================================

    def compute(self, returns: np.ndarray) -> float:

        n = len(returns)

        # sub-window sizes: powers of 2 from min_window up to n//2
        sizes = []
        s = self.min_window
        while s <= n // 2:
            sizes.append(s)
            s = int(s * 1.5)

        if len(sizes) < 2:
            return np.nan

        log_n  = []
        log_rs = []

        for size in sizes:
            # average R/S across non-overlapping sub-windows of this size
            rs_vals = []
            for start in range(0, n - size + 1, size):
                rs = self._rs(returns[start : start + size])
                if not np.isnan(rs):
                    rs_vals.append(rs)

            if rs_vals:
                log_n.append(np.log(size))
                log_rs.append(np.log(np.mean(rs_vals)))

        if len(log_n) < 2:
            return np.nan

        # OLS slope
        log_n  = np.array(log_n)
        log_rs = np.array(log_rs)
        H = np.polyfit(log_n, log_rs, 1)[0]

        return float(H)

    # ==================================================
    # Rolling Hurst over a price series
    # ==================================================

    def rolling(self, prices: pd.Series, window: int = 120) -> pd.Series:

        log_ret = np.log(prices / prices.shift(1)).dropna().values
        n = len(log_ret)

        hurst_vals = np.full(n, np.nan)

        for t in range(window - 1, n):
            hurst_vals[t] = self.compute(log_ret[t - window + 1 : t + 1])

        index = np.log(prices / prices.shift(1)).dropna().index
        return pd.Series(hurst_vals, index=index, name="hurst")
