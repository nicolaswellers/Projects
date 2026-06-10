'''

models:
partial corr
rolling corr
dcc with garch?

Inputs for rolling_corr look like:

SPY <----> IWM (breadth)
SPY <----> XLU (sector specific)
SPY <---->VIX (stress)
SPY <---->TLT (macro)
HYG <----> TLT (liquidity)
SPY <----> HYG (credit)

SPY = s and p "SPY.P"
IWM = russell 2000 "IWM.P"
XLU = utilities "XLU.P"
VIX = volatility index "VIX"  (routed to load_vix -> '.VIX' internally)
TLT = long term treasury "TLT.P"
HYG = high yield bonds "HYG.P"

Inputs for partial_corr look like:

IWM <-----> HYG whilst controlling for SPY
XLU <-----> VIX whilst controlling for SPY

Inputs for DCC_garch_corr look like:

SPY, IWM, XLU, VIX, TLT, HYG returns
SPY, IWM, XLU, VIX, TLT, HYG garch vol estimates

'''


# -------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from loader import DataLoader
from rolling_corr import RollingCorr
from partial_corr import PartialCorr
from DCC_garch_corr import DCC


# =========================
# CONFIG
# =========================

DATE_START, DATE_END = "2020-01-01", "2025-01-01"

# LSEG ticker -> short label
TICKERS = {
    "SPY.P": "SPY",
    "IWM.P": "IWM",
    "XLU.P": "XLU",
    "VIX":   "VIX",       # routed to load_vix -> '.VIX' inside the loader
    "TLT.P": "TLT",
    "HYG.P": "HYG",
}

ROLLING_PAIRS = [
    ("SPY", "IWM"),   # breadth
    ("SPY", "XLU"),   # sector
    ("SPY", "VIX"),   # stress
    ("SPY", "TLT"),   # macro
    ("HYG", "TLT"),   # liquidity
    ("SPY", "HYG"),   # credit
]

PARTIAL_TRIPLES = [
    ("IWM", "HYG", "SPY"),
    ("XLU", "VIX", "SPY"),
]

WINDOW_CORR    = 30
WINDOW_SMOOTH  = 10
DCC_A, DCC_B   = 0.02, 0.97


# =========================
# DATA LOADING (with cache)
# =========================

def load_data(fetch_fn, start, end, cache_path="cache/prices.csv"):
    if os.path.exists(cache_path):
        print("Loading cached data...")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True).loc[start:end]
    else:
        print("Fetching data...")
        df = fetch_fn(start, end)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path)

    # Force numeric dtype and joint NaN-drop so all columns share an index.
    # Models that call .dropna() per series (e.g. PartialCorr.log_rets) need
    # the inputs to be pre-aligned, otherwise OLS gets misaligned Series and
    # statsmodels coerces them through object dtype.
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df


def fetch_lseg(start, end):
    """fetch_fn for load_data: pulls every ticker in TICKERS from LSEG
    and returns a single aligned DataFrame indexed by date."""
    loader = DataLoader()
    cols = []
    for ticker, short in TICKERS.items():
        print(f"  fetching {ticker} ({short})...")
        if short == "VIX":
            s = loader.load_vix(start, end)
        else:
            s = loader.load_daily_close(ticker, start, end)
        s.name = short
        cols.append(s)
    return pd.concat(cols, axis=1).dropna()


# =========================
# GARCH(1,1)  -- BUILT IN
# =========================

class GARCH11:
    """Gaussian GARCH(1,1) by direct quasi-MLE.

        r_t        = mu + eps_t,        eps_t = sigma_t * z_t,  z_t ~ N(0, 1)
        sigma_t^2  = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

    Constraints enforced:  omega > 0,  alpha >= 0,  beta >= 0,  alpha + beta < 1
    Returns conditional sigmas aligned to the input return series.
    """

    def __init__(self):
        self.mu = self.omega = self.alpha = self.beta = None
        self.sigma2 = None

    @staticmethod
    def _recursion(eps, omega, alpha, beta, s2_init):
        T = len(eps)
        sigma2 = np.empty(T)
        sigma2[0] = s2_init
        for t in range(1, T):
            sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
        return sigma2

    def _neg_loglik(self, params, ret):
        mu, omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1 - 1e-6:
            return 1e12
        eps = ret - mu
        sigma2 = self._recursion(eps, omega, alpha, beta, eps.var())
        if np.any(sigma2 <= 0) or not np.all(np.isfinite(sigma2)):
            return 1e12
        return 0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)

    def fit(self, ret):
        ret = np.asarray(ret, dtype=float)
        v  = ret.var()
        x0 = [ret.mean(), v * 0.05, 0.05, 0.90]
        bounds = [(None, None), (1e-10, None), (0.0, 0.5), (0.0, 0.999)]
        cons   = [{'type': 'ineq', 'fun': lambda p: 1 - p[2] - p[3] - 1e-6}]
        res = minimize(self._neg_loglik, x0, args=(ret,),
                       method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-9, 'maxiter': 300})
        self.mu, self.omega, self.alpha, self.beta = res.x
        eps = ret - self.mu
        self.sigma2 = self._recursion(eps, self.omega, self.alpha, self.beta,
                                      eps.var())
        return self

    @property
    def sigma(self):
        return np.sqrt(self.sigma2)


def fit_garch_all(returns_df):
    """Fit GARCH(1,1) per column; return a DataFrame of conditional sigmas
    aligned to returns_df. Returns are scaled to ~% units internally for
    numerical stability, then descaled on the way out."""
    out = {}
    for col in returns_df.columns:
        r = returns_df[col].dropna().values
        m = GARCH11().fit(r * 100.0)
        sigma = m.sigma / 100.0
        out[col] = pd.Series(sigma, index=returns_df[col].dropna().index)
        print(f"  {col:>4s}:  mu={m.mu/100:+.2e}  omega={m.omega:.2e}  "
              f"alpha={m.alpha:.3f}  beta={m.beta:.3f}  "
              f"(a+b={m.alpha+m.beta:.3f})")
    return pd.DataFrame(out)


# =========================
# MODEL RUNNERS
# =========================

def run_rolling(prices):
    out = {}
    for a, b in ROLLING_PAIRS:
        rc = RollingCorr(prices[a], prices[b], WINDOW_CORR, WINDOW_SMOOTH)
        out[f"{a}_{b}"] = rc.run_smoothed_fisher_corr()
    return pd.DataFrame(out)


def run_partial(prices):
    rows = []
    for x, y, ctrl in PARTIAL_TRIPLES:
        pc  = PartialCorr(prices[x], prices[y], prices[ctrl])
        val = pc.compute_partial_corr()
        rows.append({"x": x, "y": y, "control": ctrl,
                     "partial_corr": float(val)})
    return pd.DataFrame(rows)


def run_dcc(z):
    cols  = list(z.columns)
    dcc   = DCC(a=DCC_A, b=DCC_B)
    dcc.set_inputs(z.values)
    R     = dcc.fit()                     # (T-1, N, N)
    dates = z.index[1:]
    out = {}
    for a, b in ROLLING_PAIRS:
        i, j = cols.index(a), cols.index(b)
        out[f"{a}_{b}"] = pd.Series(R[:, i, j], index=dates)
    return pd.DataFrame(out), R


# =========================
# PLOTS
# =========================

def save_plots(rolling_df, dcc_df, plots_dir="results/plots"):
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    rolling_df.plot(ax=ax, lw=1.0)
    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax.set_title(f"Rolling correlations  (Fisher-z, window={WINDOW_CORR}, "
                 f"smooth={WINDOW_SMOOTH})")
    ax.set_ylabel("Fisher-z correlation")
    fig.tight_layout(); fig.savefig(f"{plots_dir}/rolling_corr.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    dcc_df.plot(ax=ax, lw=1.0)
    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax.set_title(f"DCC-GARCH correlations  (a={DCC_A}, b={DCC_B})")
    ax.set_ylabel("Correlation")
    fig.tight_layout(); fig.savefig(f"{plots_dir}/dcc_corr.png", dpi=140)
    plt.close(fig)

    # Side-by-side: rolling output back-transformed (tanh) to live in [-1, 1]
    rolling_rho = np.tanh(rolling_df)
    n = len(ROLLING_PAIRS)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.4 * n), sharex=True)
    for ax, (a, b) in zip(axes, ROLLING_PAIRS):
        col = f"{a}_{b}"
        if col in rolling_rho.columns:
            ax.plot(rolling_rho.index, rolling_rho[col],
                    label="Rolling (tanh of Fisher-z)", lw=0.9)
        if col in dcc_df.columns:
            ax.plot(dcc_df.index, dcc_df[col],
                    label="DCC-GARCH", lw=0.9, alpha=0.85)
        ax.axhline(0, color='k', lw=0.4, alpha=0.4)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=8, loc='best')
    fig.tight_layout(); fig.savefig(f"{plots_dir}/rolling_vs_dcc.png", dpi=140)
    plt.close(fig)


# =========================
# MAIN
# =========================

def main():
    os.makedirs("results", exist_ok=True)

    # -------------------------
    # 1. LOAD DATA (cached)
    # -------------------------
    prices = load_data(fetch_lseg, DATE_START, DATE_END,
                       cache_path="cache/prices.csv")
    print(f"  -> {prices.shape[0]} rows from "
          f"{prices.index.min().date()} to {prices.index.max().date()}")

    # -------------------------
    # 2. LOG RETURNS
    # -------------------------
    log_ret = np.log(prices / prices.shift(1)).dropna()

    # -------------------------
    # 3. ROLLING CORRELATION
    # -------------------------
    print("\nRunning rolling correlations...")
    roll_corr = run_rolling(prices)
    roll_corr.to_csv("results/rolling_corr.csv")

    # -------------------------
    # 4. PARTIAL CORRELATION
    # -------------------------
    print("Running partial correlations...")
    pcorr = run_partial(prices)
    pcorr.to_csv("results/partial_corr.csv", index=False)

    # -------------------------
    # 5. GARCH(1,1) FITS
    # -------------------------
    print("\nFitting GARCH(1,1) per series...")
    sigma  = fit_garch_all(log_ret)
    common = log_ret.index.intersection(sigma.index)
    z      = (log_ret.loc[common] / sigma.loc[common]).dropna()

    # -------------------------
    # 6. DCC
    # -------------------------
    print("\nRunning DCC recursion...")
    dcc_corr, R = run_dcc(z)
    dcc_corr.to_csv("results/dcc_corr.csv")
    dcc_latest = pd.DataFrame(R[-1], index=z.columns, columns=z.columns)

    # -------------------------
    # 7. PLOTS
    # -------------------------
    save_plots(roll_corr, dcc_corr)

    # -------------------------
    # 8. OUTPUT COMPARISON
    # -------------------------
    print("\n====================")
    print("ROLLING CORRELATION  (tail, Fisher-z space)")
    print("====================")
    print(roll_corr.tail())

    print("\n====================")
    print("PARTIAL CORRELATION")
    print("====================")
    print(pcorr.to_string(index=False))

    print("\n====================")
    print("DCC CORRELATION  (LATEST)")
    print("====================")
    print(dcc_latest.round(3))

    print("\nWrote results to ./results/  and cached prices to ./cache/prices.csv")


if __name__ == "__main__":
    main()