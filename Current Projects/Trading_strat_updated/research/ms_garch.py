"""2-regime Markov-Switching GJR-GARCH(1,1,1).

Exact regime-conditional GARCH is path-dependent (Cai 1994 / Hamilton-Susmel 1994):
h_{k,t} depends on which regime was active at every past date, so the likelihood
requires summing over all 2^t regime paths. This module uses the standard practical
fix (Gray 1996, extended with a leverage term as in Klaassen 2002): at every step,
the regimes' conditional variances are "collapsed" into a single filtered-probability
-weighted variance before being fed into next period's GARCH recursion. This keeps the
recursion Markovian (tractable) at the cost of being an approximation to the true
infinite-mixture likelihood.

Mean equation: constant (zero, since inputs are already-demeaned-ish daily log returns).
Variance equation per regime k: h_k,t = omega_k + (alpha_k + gamma_k*I[eps<0]) * eps_{t-1}^2
                                          + beta_k * h_tilde_{t-1}
  where h_tilde_{t-1} = sum_j filt_{t-1}[j] * h_{j,t-1}  (Gray's collapse).
Innovations: Gaussian (kept simple -- the collapsing step is already an approximation
to the exact likelihood, and per-regime Student-t roughly doubles the optimizer's
dimensionality for a second-order refinement).
Regimes are sorted ascending by unconditional variance, so regime 1 is always
"high-vol / stress" and regime 0 is "low-vol / calm".

The per-step recursion is written in plain scalar Python (not numpy) because it's
inherently sequential (h_t depends on filt_{t-1} depends on h_{t-1}, ...) and gets
called many thousands of times during MLE (each L-BFGS-B gradient step needs 11
function evals for 10 params) -- numpy's per-call overhead on length-2 arrays made
a single fit take minutes; scalar math brings it down to ~1s.
"""
import math
import numpy as np
from scipy.optimize import minimize

N_REGIMES = 2


def _unpack(theta: np.ndarray):
    omega = np.exp(theta[0:2])
    alpha = 1.0 / (1.0 + np.exp(-theta[2:4])) * 0.30
    gamma = 1.0 / (1.0 + np.exp(-theta[4:6])) * 0.30
    beta  = 1.0 / (1.0 + np.exp(-theta[6:8])) * 0.95
    p11   = 1.0 / (1.0 + np.exp(-theta[8]))
    p22   = 1.0 / (1.0 + np.exp(-theta[9]))
    return omega, alpha, gamma, beta, (p11, 1 - p11, 1 - p22, p22)


def _filter_step(r_t, eps_prev, f0, f1, hp0, hp1,
                  omega0, omega1, alpha0, alpha1, gamma0, gamma1, beta0, beta1,
                  p11, p12, p21, p22):
    ind = 1.0 if eps_prev < 0.0 else 0.0
    eps2 = eps_prev * eps_prev
    h_tilde = f0 * hp0 + f1 * hp1
    h0 = omega0 + (alpha0 + gamma0 * ind) * eps2 + beta0 * h_tilde
    h1 = omega1 + (alpha1 + gamma1 * ind) * eps2 + beta1 * h_tilde
    if h0 < 1e-8:
        h0 = 1e-8
    if h1 < 1e-8:
        h1 = 1e-8
    pred0 = f0 * p11 + f1 * p21
    pred1 = f0 * p12 + f1 * p22
    d0 = math.exp(-0.5 * r_t * r_t / h0) / math.sqrt(2.0 * math.pi * h0)
    d1 = math.exp(-0.5 * r_t * r_t / h1) / math.sqrt(2.0 * math.pi * h1)
    j0 = pred0 * d0
    j1 = pred1 * d1
    lik = j0 + j1
    if lik <= 0.0 or not math.isfinite(lik):
        return pred0, pred1, h0, h1, 1e-300
    return j0 / lik, j1 / lik, h0, h1, lik


def _run_filter_fast(r_list, omega, alpha, gamma, beta, P, filt0, h0):
    """Scalar-loop Hamilton filter. Returns (loglik, (f0,f1)_last, (h0,h1)_last)."""
    omega0, omega1 = omega
    alpha0, alpha1 = alpha
    gamma0, gamma1 = gamma
    beta0, beta1 = beta
    p11, p12, p21, p22 = P
    f0, f1 = filt0
    hp0, hp1 = h0
    eps_prev = 0.0
    ll = 0.0
    for r_t in r_list:
        f0, f1, hp0, hp1, lik = _filter_step(
            r_t, eps_prev, f0, f1, hp0, hp1,
            omega0, omega1, alpha0, alpha1, gamma0, gamma1, beta0, beta1,
            p11, p12, p21, p22)
        ll += math.log(lik)
        eps_prev = r_t
    return ll, (f0, f1), (hp0, hp1)


def run_filter_path(r, omega, alpha, gamma, beta, P, filt0, h0):
    """Same recursion as `_run_filter_fast` but also returns the full per-day filtered
    probability and variance paths, for diagnostics/plotting. Not used in the MLE inner
    loop (called once per fit, so numpy convenience here doesn't matter for speed)."""
    r_list = list(r)
    T = len(r_list)
    filt_path = np.zeros((T, 2))
    h_path = np.zeros((T, 2))
    omega0, omega1 = omega
    alpha0, alpha1 = alpha
    gamma0, gamma1 = gamma
    beta0, beta1 = beta
    p11, p12, p21, p22 = P
    f0, f1 = filt0
    hp0, hp1 = h0
    eps_prev = 0.0
    ll = 0.0
    for t, r_t in enumerate(r_list):
        f0, f1, hp0, hp1, lik = _filter_step(
            r_t, eps_prev, f0, f1, hp0, hp1,
            omega0, omega1, alpha0, alpha1, gamma0, gamma1, beta0, beta1,
            p11, p12, p21, p22)
        ll += math.log(lik)
        filt_path[t] = (f0, f1)
        h_path[t] = (hp0, hp1)
        eps_prev = r_t
    return ll, filt_path, h_path, (f0, f1), (hp0, hp1)


def _neg_loglik(theta, r_list, filt0):
    omega, alpha, gamma, beta, P = _unpack(theta)
    denom0 = max(1 - alpha[0] - beta[0] - gamma[0] / 2, 1e-3)
    denom1 = max(1 - alpha[1] - beta[1] - gamma[1] / 2, 1e-3)
    h0 = (omega[0] / denom0, omega[1] / denom1)
    ll, *_ = _run_filter_fast(r_list, omega, alpha, gamma, beta, P, filt0, h0)
    if not math.isfinite(ll):
        return 1e10
    return -ll


def fit_ms_garch(returns: np.ndarray, n_restarts: int = 3, seed: int = 0,
                  maxiter: int = 150) -> dict:
    """Fit the 2-regime MS-GJR-GARCH on a window of (already-demeaned) log returns in %.

    Tries a few restarts (regime-switching likelihoods can have multiple local optima /
    can degenerate to near-identical regimes) and keeps the best log-likelihood. Returns
    fitted params plus the filtered state at the *last* observation, so a caller can
    continue the recursion forward day-by-day with no refit (see
    `compute_walkforward_ms_garch`).
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=np.float64)
    r = r - r.mean()
    r_list = r.tolist()
    var = float(r.var())
    filt0 = (0.5, 0.5)

    best = None
    inits = [
        np.array([np.log(var * 0.5), np.log(var * 2.0), -1.0, 0.5, -2.0, 0.5, 1.5, 0.5, 2.0, 2.0]),
        np.array([np.log(var * 0.8), np.log(var * 3.0), 0.0, 1.0, -1.0, 1.0, 1.0, 0.0, 1.5, 1.5]),
    ]
    for i in range(n_restarts):
        x0 = inits[i % len(inits)] + rng.normal(scale=0.15, size=10)
        try:
            res = minimize(_neg_loglik, x0, args=(r_list, filt0), method="L-BFGS-B",
                            options={"maxiter": maxiter})
        except Exception:
            continue
        if res.fun is not None and np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res

    if best is None:
        return None

    omega, alpha, gamma, beta, P = _unpack(best.x)
    denom = np.clip(1 - alpha - beta - gamma / 2, 1e-3, None)
    h0 = (omega[0] / denom[0], omega[1] / denom[1])
    ll, filt_last, h_last = _run_filter_fast(r_list, omega, alpha, gamma, beta, P, filt0, h0)

    uncond_var = omega / denom
    order = np.argsort(uncond_var)  # ascending: 0=calm, 1=stress
    omega, alpha, gamma, beta = omega[order], alpha[order], gamma[order], beta[order]
    p11, p12, p21, p22 = P
    Pmat = np.array([[p11, p12], [p21, p22]])[np.ix_(order, order)]
    P_ordered = (Pmat[0, 0], Pmat[0, 1], Pmat[1, 0], Pmat[1, 1])
    filt_last_o = tuple(np.array(filt_last)[order])
    h_last_o = tuple(np.array(h_last)[order])

    return dict(omega=omega, alpha=alpha, gamma=gamma, beta=beta, P=P_ordered,
                filt_last=filt_last_o, h_last=h_last_o, loglik=-best.fun,
                uncond_var=uncond_var[order])


def compute_walkforward_ms_garch(price_index, refit_freq: int = 21, window: int = 750,
                                  min_obs: int = 300, stress_state: int = 1,
                                  n_restarts: int = 2, maxiter: int = 150):
    """Walk-forward P(stress regime) from the 2-regime MS-GJR-GARCH above.

    Refits on the trailing `window` of returns every `refit_freq` days (matching the
    cadence used elsewhere for GARCH/HMM refits). Between refits, propagates the causal
    Hamilton filter forward one real trading day at a time with the *fixed* fitted
    params -- exactly the "refit periodically, update daily" pattern used for the
    plain single-regime GARCH vol path, chosen for the same reason: naively holding a
    periodic point estimate flat between refits produced a sawtooth artifact there, and
    would do the same to a regime probability here.
    """
    import pandas as pd
    log_ret = np.log(price_index).diff().dropna() * 100
    dates = log_ret.index
    r = log_ret.to_numpy()
    n = len(r)

    out = np.full(n, np.nan)
    params = None
    filt = None
    h_prev = None
    eps_prev = 0.0

    for i in range(n):
        if i < min_obs:
            eps_prev = r[i]
            continue
        if params is None or i % refit_freq == 0:
            lo = max(0, i - window)
            fitted = fit_ms_garch(r[lo:i], n_restarts=n_restarts, seed=i, maxiter=maxiter)
            if fitted is not None:
                params = fitted
                filt = fitted["filt_last"]
                h_prev = fitted["h_last"]
        if params is None:
            eps_prev = r[i]
            continue

        f0, f1, hp0, hp1, lik = _filter_step(
            r[i], eps_prev, filt[0], filt[1], h_prev[0], h_prev[1],
            params["omega"][0], params["omega"][1], params["alpha"][0], params["alpha"][1],
            params["gamma"][0], params["gamma"][1], params["beta"][0], params["beta"][1],
            params["P"][0], params["P"][1], params["P"][2], params["P"][3])
        filt = (f0, f1)
        h_prev = (hp0, hp1)
        out[i] = filt[stress_state]
        eps_prev = r[i]

    s = pd.Series(out, index=dates).ffill()
    return s
