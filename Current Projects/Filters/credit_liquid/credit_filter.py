'''this is the credit filter:

General structure:

1. credit level (z-score)
2. credit shock (z-score)
3. credit acceleration (z-score)

inputs: ICE BofA HY OAS from FRED (BAMLH0A0HYM2)
- actual option-adjusted spread over Treasuries
- rate moves are already stripped out, pure credit risk signal
'''
import numpy as np

class CreditFilter:
    def __init__(self, hy_oas):
        self.oas = hy_oas  # ICE BofA HY OAS series (basis points)

    def zscore(self, x, window=20):
        return (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-9)

    def spread_level(self, window=504):
        # z-score vs 2yr window — credit regimes are slow-moving
        return self.zscore(self.oas, window)

    def spread_shock(self, window=504):
        # smoothed daily change in OAS, normalized vs history
        shock = self.oas.diff().rolling(5).mean()
        return self.zscore(shock, window)

    def spread_acceleration(self, window=504):
        # smoothed change-of-change in OAS, normalized vs history
        shock = self.oas.diff().rolling(5).mean()
        accel = shock.diff().rolling(5).mean()
        return self.zscore(accel, window)