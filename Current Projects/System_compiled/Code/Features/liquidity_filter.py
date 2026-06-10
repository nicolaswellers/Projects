'''this is the liquidity file:

General output: 

1. Liquidity level (z-score)
2. Liquidity shock (z-score)
3. Liquidity acceleration (z-score)

the liquidity level structure is broken down into:
A. cost (tightness of the spread)
B. volume (volume available at the best bid and ask)
C. stability (stability of the turnover)

the liquidity shock is the change in liquidity level, 
and the liquidity acceleration is the change in liquidity shock


inputs: price series and volume series of chosen security
'''
import numpy as np
import pandas as pd

class LiquidityFilter:
    def __init__(self, price, volume):
        self.price = price
        self.volume = volume

    def returns(self):
        return np.log(self.price / self.price.shift(1))

    def zscore(self, x, window=20):
        return (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-9)

    def amihud(self, window=20):
        # Amihud illiquidity: |return| / dollar volume — higher = more illiquid
        ret = self.returns().abs()
        dollar_vol = self.price * self.volume
        raw = ret / (dollar_vol + 1e-9)
        smoothed = raw.rolling(window).mean()
        return self.zscore(smoothed, 504)  # normalize vs 2yr window

    def volume_proxy(self, volume, window=504):
        vol = np.log(volume.replace(0, np.nan))
        return self.zscore(vol, window)

    def stability(self, price, volume, window=504):
        turnover = np.log(price * volume)
        stability = 1 / (turnover.rolling(20).std() + 1e-9)
        return self.zscore(stability, window)

    def liquidity_level(self, window=20):
        amihud_z = self.amihud(window)
        vol_z    = self.volume_proxy(self.volume)
        stab_z   = self.stability(self.price, self.volume)

        # amihud is illiquidity (higher = worse), invert so composite is illiquidity score
        return 0.5*amihud_z + 0.3*(-vol_z) + 0.2*stab_z

    def liquidity_shock(self, liquidity_level, window=504):
        shock = liquidity_level.diff().rolling(5).mean()
        return self.zscore(shock, window)

    def liquidity_acceleration(self, liquidity_shock, window=504):
        accel = liquidity_shock.diff().rolling(5).mean()
        return self.zscore(accel, window)