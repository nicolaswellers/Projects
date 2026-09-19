'''GJR-GARCH(1,1,1) volatility model.

Mean equation: constant (daily returns are ~white noise in mean).
Variance equation: GJR-GARCH captures the leverage effect (negative
shocks raise conditional variance more than positive ones of the same size).
Distribution: Student-t, for fatter tails than Gaussian.

Class is instantiated with a price series; apply_gjr_garch() fits on
whatever window of prices it's given, so lookahead is controlled entirely
by the caller (see backtest.py's walk-forward refit).
'''
import numpy as np
from arch import arch_model


def _log_returns_pct(prices):
    """Daily log returns in % with NaN/inf rows removed."""
    p = prices.replace(0, np.nan).dropna()
    r = np.log(p / p.shift(1)) * 100
    return r.replace([np.inf, -np.inf], np.nan).dropna()


class GARCHModel:
    def __init__(self, prices_close):
        self.prices_close  = prices_close
        self.returns_close = _log_returns_pct(prices_close)

    def apply_gjr_garch(self):
        model = arch_model(
            self.returns_close,
            mean="constant", vol="GARCH",
            p=1, o=1, q=1, dist="t", rescale=False,
        )
        self.fit_gjr = model.fit(disp="off", show_warning=True)
        return self.fit_gjr
