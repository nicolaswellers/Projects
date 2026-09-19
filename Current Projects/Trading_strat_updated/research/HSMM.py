'''Hidden (Semi-)Markov regime detection for a market return series.

Two variants, both walk-forward (refit every `refit_freq` days on a
trailing window, then daily-filtered forward between refits using that
fit's fixed parameters -- no lookahead, matching the walk-forward
discipline used for GARCH/Kalman elsewhere in this project):

  - HMM: a plain n_states Gaussian HMM (hmmlearn). Regime durations are
    implicitly geometric (memoryless) under an HMM -- the well-known
    limitation that motivates HSMM (Bulla and Bulla 2006).
  - HSMM-style: the same HMM, with a minimum-dwell-time filter applied to
    the raw state calls -- a regime switch only registers once the new
    state has been the argmax for `min_duration` consecutive days. This
    imposes the positive duration dependence a true semi-Markov model
    would estimate explicitly, without the local-optima-prone custom EM
    a generalized HSMM estimator requires (see Nystrup et al. 2017, who
    show a duration-augmented HMM can reproduce these effects with fewer
    parameters than an HSMM).

States are relabelled by fitted mean at every refit (state 0 = lowest
mean = "bear") since HMM state indices aren't identified across
independent fits.

A third function, `compute_regime_weight`, implements the paper's own
(Baitinger & Hoch, "Simplicity versus Complexity") dynamic allocation
rule directly, for an n_states > 2 model where a hard bull/bear split no
longer makes sense: w*_t = (1/risk_aversion) * (Rhat_t / sigma2hat_t),
with Rhat_t and sigma2hat_t the state means/variances blended by that
day's filtered state probabilities (their Eq. 9-11), clipped to
[0, weight_cap].
'''
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm


def _sorted_by_mean(model: GaussianHMM):
    """Reorder a fitted model's parameters so state 0 has the lowest
    fitted mean ('bear') through state n-1 ('bull')."""
    order = np.argsort(model.means_.ravel())
    means = model.means_.ravel()[order]
    varsq = model.covars_.reshape(-1)[order]
    transmat = model.transmat_[np.ix_(order, order)]
    startprob = model.startprob_[order]
    return means, varsq, transmat, startprob


def _forward_filter(x: np.ndarray, startprob, transmat, means, varsq) -> np.ndarray:
    """Pure forward (causal) filtering: alpha_t only ever uses x[0..t],
    never future observations -- unlike hmmlearn's own predict_proba,
    which is a smoother and would leak future information here."""
    n = len(x)
    k = len(startprob)
    alphas = np.zeros((n, k))
    emiss = norm.pdf(x[0], means, np.sqrt(varsq))
    a = startprob * emiss
    a = a / a.sum() if a.sum() > 1e-300 else startprob
    alphas[0] = a
    for t in range(1, n):
        emiss = norm.pdf(x[t], means, np.sqrt(varsq))
        a = (alphas[t - 1] @ transmat) * emiss
        s = a.sum()
        a = a / s if s > 1e-300 else alphas[t - 1]
        alphas[t] = a
    return alphas


def compute_walkforward_regime(returns: pd.Series, n_states: int = 2,
                                refit_freq: int = 21, window: int = 750,
                                min_obs: int = 300, random_state: int = 0,
                                stress_state: int = 0) -> pd.Series:
    """Walk-forward P(stress regime) for each day in `returns.index`,
    refitting a Gaussian HMM every `refit_freq` days on the trailing
    `window` observations, and forward-filtering daily in between with
    that fit's fixed parameters (each day's value uses only returns up
    to and including that day's own close). NaN before `min_obs`.

    `stress_state` selects which of the mean-ascending-sorted states
    counts as "stress": 0 (the default) is the lowest-mean state, e.g.
    "bear" for a return series; -1 is the highest-mean state, appropriate
    for a feature where a HIGH value signals stress, like realized
    volatility or PC1 variance share."""
    dates = returns.index
    x = returns.values
    p_stress = pd.Series(np.nan, index=dates)

    params = None
    alpha = None

    for i, date in enumerate(dates):
        if i < min_obs:
            continue
        if params is None or (i - min_obs) % refit_freq == 0:
            window_x = x[max(0, i - window + 1): i + 1]
            try:
                model = GaussianHMM(n_components=n_states, covariance_type="diag",
                                     n_iter=200, random_state=random_state)
                model.fit(window_x.reshape(-1, 1))
                means, varsq, transmat, startprob = _sorted_by_mean(model)
                alphas = _forward_filter(window_x, startprob, transmat, means, varsq)
                params = (transmat, means, varsq)
                alpha = alphas[-1]
                p_stress.loc[date] = alpha[stress_state]
                continue
            except Exception:
                params = None
                continue

        transmat, means, varsq = params
        emiss = norm.pdf(x[i], means, np.sqrt(varsq))
        a = (alpha @ transmat) * emiss
        s = a.sum()
        alpha = a / s if s > 1e-300 else alpha
        p_stress.loc[date] = alpha[stress_state]

    return p_stress


def compute_regime_weight(returns: pd.Series, n_states: int = 5, refit_freq: int = 21,
                           window: int = 750, min_obs: int = 300, risk_aversion: float = 6.0,
                           weight_cap: float = 1.5, random_state: int = 0) -> pd.Series:
    """Walk-forward dynamic exposure weight, w*_t = (1/risk_aversion) *
    (Rhat_t / sigma2hat_t), clipped to [0, weight_cap] -- no short-sales,
    a capped leverage limit (matching the paper's own 0-150% constraint).
    Refits an n_states Gaussian HMM every `refit_freq` days on the
    trailing `window`, forward-filtering state probabilities daily in
    between with fixed parameters (each day's value uses only returns up
    to and including that day's own close -- no lookahead)."""
    dates = returns.index
    x = returns.values
    weight = pd.Series(np.nan, index=dates)

    params = None
    alpha = None

    for i, date in enumerate(dates):
        if i < min_obs:
            continue
        if params is None or (i - min_obs) % refit_freq == 0:
            window_x = x[max(0, i - window + 1): i + 1]
            try:
                model = GaussianHMM(n_components=n_states, covariance_type="diag",
                                     n_iter=200, random_state=random_state)
                model.fit(window_x.reshape(-1, 1))
                means     = model.means_.ravel()
                varsq     = model.covars_.reshape(-1)
                transmat  = model.transmat_
                startprob = model.startprob_
                alphas = _forward_filter(window_x, startprob, transmat, means, varsq)
                params = (transmat, means, varsq)
                alpha  = alphas[-1]
            except Exception:
                params = None
                continue
        else:
            transmat, means, varsq = params
            emiss = norm.pdf(x[i], means, np.sqrt(varsq))
            a = (alpha @ transmat) * emiss
            s = a.sum()
            alpha = a / s if s > 1e-300 else alpha

        r_hat   = float(np.dot(alpha, means))
        var_hat = float(np.dot(alpha, varsq))
        w = (r_hat / var_hat) / risk_aversion if var_hat > 1e-12 else 0.0
        weight.loc[date] = min(max(w, 0.0), weight_cap)

    return weight


def apply_min_duration(hard_state: pd.Series, min_duration: int) -> pd.Series:
    """HSMM-style positive-duration-dependence filter: a state switch in
    `hard_state` only registers once the new state has persisted for
    `min_duration` consecutive raw calls; until then, the previously
    confirmed state is carried forward. This is the practical, numerically
    robust stand-in for a generalized semi-Markov sojourn-time model."""
    out = hard_state.copy()
    confirmed = hard_state.iloc[0]
    candidate = confirmed
    streak = 0
    for i in range(len(hard_state)):
        s = hard_state.iloc[i]
        if s == candidate:
            streak += 1
        else:
            candidate = s
            streak = 1
        if streak >= min_duration and candidate != confirmed:
            confirmed = candidate
        out.iloc[i] = confirmed
    return out
