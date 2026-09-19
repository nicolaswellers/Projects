"""Market-wide signal-construction functions from this project's regime/vol-
timing research that did NOT end up in the production overlay (see
table/00_MASTER_eval_table.png for the full comparison). Kept here, out of
backtest.py's main path, so they're still importable/runnable if any of
these directions are worth revisiting, without adding to the core file's
bulk.

  - compute_market_garch_kalman_zscore / _market_garch_vol_path: GARCH vol,
    Kalman-smoothed and z-scored, used as a hysteresis trigger -- lost to
    the plain realized-vol-ratio champion.
  - compute_market_pca_index: cross-sectional PCA market factor, used as an
    input to the HMM/SMA+Kalman variants below and in research/HSMM.py.
  - compute_pc1_variance_share: PC1 variance-share stress proxy -- test was
    inconclusive (threshold miscalibration meant it never actually
    triggered at COVID).
  - compute_panic_mode_weight: compound vol+return trigger with a fixed
    30-day cooldown (Duriez 2025's "Panic Mode") -- underperformed.
  - compute_market_sma_kalman_zscore / market_trend_weight: continuous
    trend-following exposure dial -- the closest of these to competitive
    (won a handful of CPCV folds combined with the champion), but still net
    worse Sharpe/CAGR standalone.

See research/HSMM.py and research/ms_garch.py for the HMM/HSMM and
Markov-switching-GARCH regime-detection attempts.

Run with the project root on sys.path (same pattern as backtest.py callers):
    sys.path.insert(0, ROOT); from research import research_signals
"""
import numpy as np
import pandas as pd

from GARCH import GARCHModel
from config import (
    REBALANCE_FREQ, GARCH_WINDOW, GARCH_MIN_OBS,
    KALMAN_ZSCORE_WINDOW, KALMAN_ZSCORE_MIN_PERIODS,
)


def _market_garch_vol_path(price_index: pd.Series, refit_freq: int = REBALANCE_FREQ,
                            window: int = GARCH_WINDOW, min_obs: int = GARCH_MIN_OBS) -> pd.Series:
    """Daily-updating GJR-GARCH(1,1,1) conditional volatility for a single
    price series (e.g. the equal-weight market index). Refits parameters
    every `refit_freq` days but recursively propagates the variance
    recursion forward every day in between using realized returns, giving a
    genuinely daily, still lookahead-safe path (each day's value uses only
    returns up to and including that day's own close) suitable as an input
    to something meant to track a smooth trend, like a Kalman filter."""
    dates       = price_index.index
    log_ret_pct = np.log(price_index / price_index.shift(1)) * 100

    vol    = pd.Series(np.nan, index=dates)
    params = None
    sigma2 = eps_prev = np.nan

    for i, date in enumerate(dates):
        if i < min_obs:
            continue
        if params is None or (i - min_obs) % refit_freq == 0:
            window_prices = price_index.iloc[max(0, i - window + 1): i + 1]
            try:
                fit      = GARCHModel(prices_close=window_prices).apply_gjr_garch()
                params   = fit.params
                sigma2   = float(fit.conditional_volatility.iloc[-1]) ** 2
                eps_prev = float(fit.resid.iloc[-1])
                vol.loc[date] = np.sqrt(sigma2)
                continue
            except Exception:
                params = None
                continue

        mu, omega, alpha, gamma, beta = (params["mu"], params["omega"], params["alpha[1]"],
                                          params["gamma[1]"], params["beta[1]"])
        sigma2 = omega + (alpha + gamma * (eps_prev < 0)) * eps_prev ** 2 + beta * sigma2
        vol.loc[date] = np.sqrt(sigma2)
        eps_prev = log_ret_pct.iloc[i] - mu

    return vol


def compute_market_garch_kalman_zscore(universe: pd.DataFrame) -> pd.Series:
    """Market-wide risk-on/off signal: fit a daily-updating GJR-GARCH vol
    path for the equal-weight book average (see `_market_garch_vol_path`),
    smooth it with the same adaptive UKF used for the momentum exit gate,
    and z-score its slope against its own recent regime. A rising z-score
    means market vol is accelerating upward -- the same idea as the
    realized-vol-ratio trigger, but built from GARCH's own filtered
    conditional variance and Kalman-smoothed rather than a raw rolling-std
    ratio."""
    from unscent_kalman import AdaptiveUKF

    daily_ret    = universe.ffill().pct_change(fill_method=None)
    market_ret   = daily_ret.mean(axis=1).fillna(0.0)
    market_price = (1 + market_ret).cumprod()

    garch_vol = _market_garch_vol_path(market_price)

    ukf_out = AdaptiveUKF().filter(garch_vol.bfill())
    slope   = ukf_out["slope"]
    rolling = slope.rolling(KALMAN_ZSCORE_WINDOW, min_periods=KALMAN_ZSCORE_MIN_PERIODS)
    return (slope - rolling.mean()) / rolling.std()


def compute_market_pca_index(universe: pd.DataFrame, window: int = GARCH_WINDOW,
                              refit_freq: int = REBALANCE_FREQ, min_obs: int = GARCH_MIN_OBS) -> pd.Series:
    """Cross-sectional PCA market factor, as a walk-forward, lookahead-safe
    alternative to the plain equal-weight average used elsewhere: refits
    PC1 of the standardized return panel every `refit_freq` days on the
    trailing `window` (only names with a complete history over that
    window), oriented to be positively correlated with the equal-weight
    market return over it, then projects each subsequent day's own
    realized returns onto those fixed loadings to get a daily PCA-weighted
    market return. Returned as a cumulative price index (starts at 1.0),
    matching the plain market price series elsewhere."""
    daily_ret = universe.ffill().pct_change(fill_method=None)
    dates     = universe.index
    pca_ret   = pd.Series(0.0, index=dates)

    coef = cols = None

    for i, date in enumerate(dates):
        if i < min_obs:
            continue
        if coef is None or (i - min_obs) % refit_freq == 0:
            window_ret = daily_ret.iloc[max(0, i - window + 1): i + 1]
            valid = window_ret.columns[window_ret.notna().all()]
            window_ret = window_ret[valid]
            if len(valid) < 10:
                continue
            mean = window_ret.mean()
            std  = window_ret.std().replace(0, np.nan)
            z    = ((window_ret - mean) / std).dropna(axis=1).astype(np.float64)
            valid = z.columns
            try:
                z_arr = z.to_numpy(dtype=np.float64)
                _, s, vt = np.linalg.svd(z_arr, full_matrices=False)
                loading = vt[0]
                score   = z_arr @ loading
                ew_ret  = window_ret[valid].mean(axis=1).to_numpy(dtype=np.float64)
                if np.corrcoef(score, ew_ret)[0, 1] < 0:
                    loading = -loading
                # `loading`/`std` are only the right combination to recover the
                # PCA *score* in de-meaned/standardized space (score = loading . z);
                # for an actual weighted-index return we instead just want
                # portfolio-style weights applied straight to raw returns, with
                # no re-centering -- otherwise every refit silently strips out
                # each stock's own historical drift and replaces it with a
                # persistent negative drag over time.
                coef = pd.Series(loading, index=valid) / std[valid].astype(np.float64)
                coef = coef / coef.abs().sum()  # normalize to portfolio-style weights (sum |w| = 1)
                cols = valid
            except Exception:
                coef = None
                continue

        day = daily_ret.loc[date, cols].fillna(0.0)
        pca_ret.loc[date] = float((coef * day).sum())

    return (1 + pca_ret).cumprod()


def compute_pc1_variance_share(universe: pd.DataFrame, window: int = 63, min_stocks: int = 10) -> pd.Series:
    """Rolling PC1 variance share: the fraction of total cross-sectional
    return variance explained by the first principal component of the
    (centered, not standardized) covariance structure over a trailing
    `window` -- a correlation-concentration / stress proxy, since it rises
    whenever stocks start moving together more than usual (as in a crisis).
    Computed fresh every day using only names with a complete history over
    the window."""
    daily_ret = universe.ffill().pct_change(fill_method=None)
    dates = universe.index
    share = pd.Series(np.nan, index=dates)

    for i, date in enumerate(dates):
        if i < window:
            continue
        window_ret = daily_ret.iloc[i - window + 1: i + 1]
        valid = window_ret.columns[window_ret.notna().all()]
        if len(valid) < min_stocks:
            continue
        x = window_ret[valid].to_numpy(dtype=np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        try:
            s = np.linalg.svd(x, compute_uv=False)
            share.loc[date] = float(s[0] ** 2 / (s ** 2).sum())
        except Exception:
            continue

    return share


def compute_panic_mode_weight(universe: pd.DataFrame, vol_window: int = 20, ret_window: int = 5,
                               vol_threshold: float = 0.60, ret_threshold: float = -0.01,
                               cooldown_days: int = 30) -> pd.Series:
    """'Panic Mode' (Duriez 2025): flattens exposure to zero for
    `cooldown_days` trading days whenever BOTH the trailing `vol_window`
    annualized realized vol exceeds `vol_threshold` AND the trailing
    `ret_window` mean daily return falls below `ret_threshold` -- a
    compound vol-and-loss trigger with a fixed-duration cooldown, rather
    than the threshold-based reset the ratio-hysteresis overlay uses.
    Checked and decremented daily (no lookahead)."""
    daily_ret  = universe.ffill().pct_change(fill_method=None)
    market_ret = daily_ret.mean(axis=1).fillna(0.0)
    dates      = market_ret.index

    ann_vol  = market_ret.rolling(vol_window).std() * np.sqrt(252)
    mean_ret = market_ret.rolling(ret_window).mean()

    weight   = pd.Series(1.0, index=dates)
    cooldown = 0
    for i, date in enumerate(dates):
        v, r = ann_vol.iloc[i], mean_ret.iloc[i]
        if not np.isnan(v) and not np.isnan(r) and v > vol_threshold and r < ret_threshold:
            cooldown = cooldown_days
        if cooldown > 0:
            weight.loc[date] = 0.0
            cooldown -= 1
        else:
            weight.loc[date] = 1.0

    return weight


def compute_market_sma_kalman_zscore(universe: pd.DataFrame, sma_window: int = 50,
                                      price_index: pd.Series | None = None) -> pd.Series:
    """Market-wide trend signal: take a market price index -- the
    equal-weight book average by default, or `price_index` (e.g. from
    compute_market_pca_index) -- smooth it with a simple moving average
    (`sma_window` days), then run the same adaptive UKF used for the
    per-stock momentum gate over that SMA (not the raw price) to get a
    trend slope, and z-score it against its own recent regime. A rising
    z-score means the smoothed market trend is accelerating upward; falling
    means decelerating or reversing -- meant to drive `market_trend_weight`
    below, cutting or restoring overall book exposure with the broad
    market trend rather than its volatility or regime."""
    from unscent_kalman import AdaptiveUKF

    if price_index is None:
        daily_ret    = universe.ffill().pct_change(fill_method=None)
        market_ret   = daily_ret.mean(axis=1).fillna(0.0)
        price_index  = (1 + market_ret).cumprod()
    sma = price_index.rolling(sma_window).mean()

    ukf_out = AdaptiveUKF().filter(sma.bfill())
    slope   = ukf_out["slope"]
    rolling = slope.rolling(KALMAN_ZSCORE_WINDOW, min_periods=KALMAN_ZSCORE_MIN_PERIODS)
    return (slope - rolling.mean()) / rolling.std()


def market_trend_weight(zscore: pd.Series, sensitivity: float = 0.25,
                         floor: float = 0.3, cap: float = 1.0) -> pd.Series:
    """Maps a trend z-score (e.g. from compute_market_sma_kalman_zscore) to
    a continuous exposure scalar around 1.0 -- cuts exposure toward `floor`
    as the trend deteriorates, restores it toward `cap` as the trend
    improves, rather than the discrete on/off cut the vol-spike overlay
    uses."""
    return (1.0 + sensitivity * zscore).clip(lower=floor, upper=cap)
