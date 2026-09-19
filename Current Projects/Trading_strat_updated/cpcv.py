'''Combinatorial Purged Cross-Validation (Lopez de Prado) over a set of
already-computed candidate daily-return series.

Each candidate here is a full risk-overlay configuration (e.g. "vol_spike",
"stop_15pct") backtested once over the entire date range -- these daily
returns are the fixed inputs. CPCV never re-runs the simulation; it only
decides, fold by fold, which subset of days counts as "train" (for
selecting the best candidate by Sharpe) vs "test" (for scoring that
selection out-of-sample). This is valid because compounding a return
series is just sequential multiplication -- it doesn't matter which fold a
day's already-realized return came from.

Method: split the date range into `n_groups` contiguous blocks. For every
way of choosing `k_test` of those blocks as the test set (all others
train), the candidate with the highest Sharpe on the train days is
selected; what gets reported is THAT candidate's Sharpe on the held-out
test days. An embargo drops a few days immediately after each test block
from the train side, to limit serial-correlation leakage across the
train/test boundary.

Repeating this over every C(n_groups, k_test) combination gives a full
distribution of out-of-sample Sharpes under selection, plus how often each
candidate actually wins -- both far more honest than the single in-sample
"best of N" comparison a plain grid search gives you.
'''
import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm


def _sharpe(ret: pd.Series) -> float:
    ret = ret.dropna()
    if len(ret) < 2 or ret.std() < 1e-12:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(252))


def combinatorial_purged_cv(returns: pd.DataFrame, n_groups: int = 8, k_test: int = 2,
                             embargo_days: int = 21) -> pd.DataFrame:
    """returns: DataFrame of daily returns, one column per candidate
    strategy, one row per date. Returns one row per (n_groups choose
    k_test) fold: which groups were held out, which candidate won the
    in-sample (train) selection, and that candidate's out-of-sample (test)
    Sharpe."""
    dates  = returns.index
    bounds = np.linspace(0, len(dates), n_groups + 1).astype(int)
    groups = [dates[bounds[i]:bounds[i + 1]] for i in range(n_groups)]

    rows = []
    for test_idx in itertools.combinations(range(n_groups), k_test):
        test_dates = dates[np.isin(np.arange(len(dates)),
                                    np.concatenate([np.arange(bounds[i], bounds[i + 1]) for i in test_idx]))]

        embargoed = pd.DatetimeIndex([])
        for i in test_idx:
            block_end = groups[i][-1]
            after     = dates[dates > block_end][:embargo_days]
            embargoed = embargoed.union(after)

        train_dates = dates.difference(test_dates).difference(embargoed)

        train_sharpes = {c: _sharpe(returns.loc[train_dates, c]) for c in returns.columns}
        winner        = max(train_sharpes, key=train_sharpes.get)
        test_sharpe   = _sharpe(returns.loc[test_dates, winner])

        rows.append({
            "test_groups":  test_idx,
            "winner":       winner,
            "train_sharpe": train_sharpes[winner],
            "test_sharpe":  test_sharpe,
        })

    return pd.DataFrame(rows)


def summarize(cv_result: pd.DataFrame, full_sample_winner: str, full_sample_sharpe: float):
    print("\n" + "=" * 58)
    print("  COMBINATORIAL PURGED CROSS-VALIDATION")
    print("=" * 58)
    print(f"  Folds run           : {len(cv_result)}")
    print(f"  Full-sample winner  : {full_sample_winner} (Sharpe {full_sample_sharpe:.3f})")

    print("\n  Selection frequency (how often each candidate won its fold's train Sharpe):")
    for name, count in cv_result["winner"].value_counts().items():
        print(f"    {name:<32}: {count}/{len(cv_result)}")

    ts = cv_result["test_sharpe"]
    print("\n  Out-of-sample test Sharpe of the selected candidate, across folds:")
    print(f"    mean   : {ts.mean():.3f}")
    print(f"    median : {ts.median():.3f}")
    print(f"    std    : {ts.std():.3f}")
    print(f"    min    : {ts.min():.3f}")
    print(f"    max    : {ts.max():.3f}")
    print(f"\n  Deflation vs full-sample: {full_sample_sharpe:.3f} -> {ts.mean():.3f} mean OOS "
          f"({(ts.mean() - full_sample_sharpe):+.3f})")
    print("=" * 58 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# DEFLATED / PROBABILISTIC SHARPE RATIO (Bailey & Lopez de Prado)
# ════════════════════════════════════════════════════════════════════════════
_EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(trial_sharpes: np.ndarray, n_trials: int | None = None) -> float:
    """E[max Sharpe] across `n_trials` independent, skill-less strategies,
    given the empirical variance of the Sharpe ratios actually observed
    across trials as a stand-in for the null's variance. Same units
    (daily or annualized) as `trial_sharpes` in, same units out."""
    trial_sharpes = np.asarray(trial_sharpes, dtype=float)
    n = n_trials if n_trials is not None else len(trial_sharpes)
    var = trial_sharpes.var(ddof=1)
    if n <= 1 or var <= 0:
        return 0.0
    z1 = norm.ppf(1 - 1.0 / n)
    z2 = norm.ppf(1 - 1.0 / (n * np.e))
    return float(np.sqrt(var) * ((1 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def probabilistic_sharpe_ratio(sr_hat: float, sr_benchmark: float, n: int,
                                skew: float, kurtosis: float) -> float:
    """P(true Sharpe > sr_benchmark), given an estimated Sharpe `sr_hat`
    over `n` observations with return skew/kurtosis (raw, not excess --
    normal = 3). All Sharpe inputs must be in the same (non-annualized,
    per-period) units as the returns `n`/`skew`/`kurtosis` came from."""
    denom = np.sqrt(max(1e-12, 1 - skew * sr_hat + ((kurtosis - 1) / 4) * sr_hat ** 2))
    z = (sr_hat - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio(champion_ret: pd.Series, trial_sharpes_annualized: list[float]) -> dict:
    """Corrects the champion strategy's Sharpe for (a) selection bias from
    having tried len(trial_sharpes_annualized) configurations and kept the
    best, and (b) non-normality of its own return distribution.
    `trial_sharpes_annualized` must include the champion's own value (that's
    what makes it the observed max). Returns the DSR probability plus the
    intermediate figures, for transparency."""
    from scipy.stats import skew as _skew, kurtosis as _kurt

    trial_sharpes_daily = np.asarray(trial_sharpes_annualized, dtype=float) / np.sqrt(252)
    n_trials  = len(trial_sharpes_daily)
    sr0_daily = expected_max_sharpe(trial_sharpes_daily, n_trials)

    ret = champion_ret.dropna()
    n = len(ret)
    sr_hat_daily = ret.mean() / ret.std()
    g3 = float(_skew(ret))
    g4 = float(_kurt(ret, fisher=False))  # raw kurtosis (normal = 3)

    dsr = probabilistic_sharpe_ratio(sr_hat_daily, sr0_daily, n, g3, g4)

    return {
        "n_trials":          n_trials,
        "sr0_annualized":    sr0_daily * np.sqrt(252),
        "sr_hat_annualized": sr_hat_daily * np.sqrt(252),
        "n_obs":             n,
        "skew":              g3,
        "kurtosis":          g4,
        "dsr":               dsr,
    }
