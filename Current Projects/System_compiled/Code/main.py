"""
main.py — Market Regime System backtest 2010-2025
Pipeline (per system diagram):
  GJR-GARCH → Credit/Liquidity → Stress index → UKF Kalman
  → Signal bus → CUSUM / BOCPD / HMM → Regime classification

Run:
    python main.py              # use cached results where available
    python main.py --force      # recompute everything
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import pandas_datareader.data as web

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in [ROOT,
           os.path.join(ROOT, "Features"),
           os.path.join(ROOT, "Estimators"),
           os.path.join(ROOT, "Loading")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from GARCH import GARCHModel
from credit_filter import CreditFilter
from liquidity_filter import LiquidityFilter
from rolling_corr import RollingCorr
from unscent_kalman import AdaptiveUKF
from CUSUM import CUSUMDetector
from BOPCD import StudentTBOCPD
from HMM_gaussian import GaussianHMMRegimeModel

# ── output dirs ──────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(ROOT, "cache")
PLOT_DIR  = os.path.join(ROOT, "plots")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# ── config ───────────────────────────────────────────────────────────────────
START         = "2010-01-01"
END           = "2026-01-01"
ASSET         = "SPY"
ASSET2        = "TLT"   # SPY-TLT flight-to-safety correlation
N_HMM_STATES  = 3

REGIME_COLORS = {"Bull": "#2ecc71", "Bear": "#e74c3c", "Sideways": "#f39c12"}



# ════════════════════════════════════════════════════════════════════════════
# CACHE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _pkl(name):
    return os.path.join(CACHE_DIR, f"{name}.pkl")

def _load(name):
    p = _pkl(name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def _save(name, obj):
    with open(_pkl(name), "wb") as f:
        pickle.dump(obj, f)


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_data(force=False):
    if not force:
        cached = _load("raw_data")
        if cached is not None:
            print("  [data] loaded from cache")
            return cached

    print(f"  [data] downloading {ASSET}, {ASSET2}, VIX, HY OAS  ({START} → {END}) ...")

    spy = yf.download(ASSET,  start=START, end=END, auto_adjust=True, progress=False)
    tlt = yf.download(ASSET2, start=START, end=END, auto_adjust=True, progress=False)
    vix = yf.download("^VIX",  start=START, end=END, auto_adjust=True, progress=False)["Close"]

    # ICE BofA HY OAS from FRED — pure credit-risk signal, rate moves stripped
    hy_oas = web.DataReader("BAMLH0A0HYM2", "fred", START, END)["BAMLH0A0HYM2"].ffill()

    data = {
        "spy_close":  spy["Close"].squeeze(),
        "spy_volume": spy["Volume"].squeeze(),
        "tlt_close":  tlt["Close"].squeeze(),
        "vix_close":  vix.squeeze(),
        "hy_oas":     hy_oas,
    }
    _save("raw_data", data)
    print("  [data] done")
    return data


# ════════════════════════════════════════════════════════════════════════════
# 2. FEATURES  (GJR-GARCH · Credit · Liquidity · Correlation)
# ════════════════════════════════════════════════════════════════════════════

def _gjr_walk_forward(prices_close, init_window=252, refit_every=21):
    """
    Walk-forward GJR-GARCH(1,1,1) — fully causal, no lookahead in parameters.

    At each refit point t:
      - Parameters estimated on returns[0..t]  (expanding window)
      - GARCH recursion then applied manually to the NEXT window [t..t+refit_every]
        using only those parameters and the last h from the training tail.
    => Parameters at day t never see data from day t+1 onward.
    Refits every `refit_every` days (~monthly). Prints progress every 6 months.
    """
    from arch import arch_model as _arch_model

    log_ret = np.log(prices_close / prices_close.shift(1)) * 100
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan).dropna()
    r_vals  = log_ret.values
    n       = len(r_vals)

    vol_out = np.full(n, np.nan)
    mu, omega, alpha, gamma, beta = 0.0, 0.05, 0.05, 0.05, 0.85
    h_prev  = float(np.var(r_vals[:init_window]))

    refit_points = list(range(init_window, n, refit_every))
    if refit_points[-1] != n:
        refit_points.append(n)

    for idx, t_start in enumerate(refit_points[:-1]):
        t_end = refit_points[idx + 1]

        # ── fit on all data BEFORE this new window ──────────────────────────
        r_train = log_ret.iloc[:t_start]
        m = _arch_model(r_train, mean="constant", vol="GARCH",
                        p=1, o=1, q=1, dist="t", rescale=False)
        try:
            fit    = m.fit(disp="off", show_warning=False)
            mu     = float(fit.params["mu"])
            omega  = float(fit.params["omega"])
            alpha  = float(fit.params["alpha[1]"])
            gamma  = float(fit.params["gamma[1]"])
            beta   = float(fit.params["beta[1]"])
            h_prev = float(fit.conditional_volatility.iloc[-1]) ** 2
        except Exception:
            pass   # keep previous params if fit fails

        if idx % 6 == 0:
            print(f"      GJR walk-forward: {t_start}/{n} days")

        # ── apply recursion to new window using the just-estimated params ───
        for t in range(t_start, t_end):
            eps       = r_vals[t - 1] - mu
            indicator = 1.0 if eps < 0.0 else 0.0
            h         = omega + alpha * eps**2 + gamma * eps**2 * indicator + beta * h_prev
            h         = max(h, 1e-8)
            vol_out[t] = np.sqrt(h)
            h_prev     = h

    result = pd.Series(vol_out, index=log_ret.index, name="gjr_vol")
    return result.dropna()

def run_features(data, force=False):
    if not force:
        cached = _load("features")
        if cached is not None:
            print("  [features] loaded from cache")
            return cached

    print("  [features] computing ...")
    spy_close  = data["spy_close"]
    spy_volume = data["spy_volume"]
    tlt_close  = data["tlt_close"]
    hy_oas     = data["hy_oas"]

    # ── GJR-GARCH(1,1,1) walk-forward — fully causal, no parameter lookahead ─
    print("    GJR-GARCH (walk-forward — a few minutes) ...")
    gjr_vol = _gjr_walk_forward(spy_close, init_window=252, refit_every=21)
    gjr_vol.index = pd.to_datetime(gjr_vol.index)

    # ── Credit filter (HY OAS) ───────────────────────────────────────────────
    print("    Credit filter ...")
    hy_aligned = hy_oas.reindex(spy_close.index, method="ffill")
    credit  = CreditFilter(hy_aligned)
    c_level = credit.spread_level()
    c_shock = credit.spread_shock()
    c_accel = credit.spread_acceleration()

    # ── Liquidity filter ─────────────────────────────────────────────────────
    print("    Liquidity filter ...")
    liq       = LiquidityFilter(spy_close, spy_volume)
    liq_level = liq.liquidity_level()
    liq_shock = liq.liquidity_shock(liq_level)
    liq_accel = liq.liquidity_acceleration(liq_shock)

    # ── Stress index: equal-weight composite of credit + liquidity stress ────
    c_aligned  = c_level.reindex(spy_close.index, method="ffill")
    stress_idx = (c_aligned + liq_level) / 2

    # ── Rolling SPY-TLT correlation (Fisher-transformed, smoothed) ──────────
    print("    Rolling correlation ...")
    rc        = RollingCorr(spy_close, tlt_close, window_corr=30, window_smooth=10)
    roll_corr = rc.run_smoothed_fisher_corr()

    feats = {
        "gjr_vol":    gjr_vol,
        "c_level":    c_level,
        "c_shock":    c_shock,
        "c_accel":    c_accel,
        "liq_level":  liq_level,
        "liq_shock":  liq_shock,
        "liq_accel":  liq_accel,
        "stress_idx": stress_idx,
        "roll_corr":  roll_corr,
    }
    _save("features", feats)
    print("  [features] done")
    return feats


# ════════════════════════════════════════════════════════════════════════════
# 3. UKF KALMAN  (trend · slope · acceleration)
# ════════════════════════════════════════════════════════════════════════════

def run_kalman(data, feats, force=False):
    if not force:
        cached = _load("kalman")
        if cached is not None:
            print("  [kalman] loaded from cache")
            return cached

    print("  [kalman] running UKF ...")
    spy_close  = data["spy_close"]
    idx        = spy_close.index

    vol_aligned    = feats["gjr_vol"].reindex(idx).ffill().fillna(0)
    stress_aligned = feats["stress_idx"].reindex(idx).ffill().fillna(0)

    ukf = AdaptiveUKF()
    kalman_out = ukf.filter(
        prices=spy_close,
        volatility=vol_aligned,
        stress=stress_aligned,
    )
    _save("kalman", kalman_out)
    print("  [kalman] done")
    return kalman_out


# ════════════════════════════════════════════════════════════════════════════
# 4. SIGNAL BUS  (aligned DataFrame fed to all estimators)
# ════════════════════════════════════════════════════════════════════════════

WARMUP_DAYS = 504  # 2-year rolling window used by credit & liquidity z-scores

def build_signal_bus(data, feats, kalman_out):
    idx = data["spy_close"].index

    def _align(s):
        if isinstance(s, (int, float, np.floating)):
            return pd.Series(float(s), index=idx)
        return pd.Series(s).reindex(idx).ffill().fillna(0)

    spy_ret_20 = (np.log(data["spy_close"] / data["spy_close"].shift(20)) * 100)

    bus = pd.DataFrame(index=idx)
    bus["kalman_trend"]   = _align(kalman_out["trend"])
    bus["kalman_slope"]   = _align(kalman_out["slope"])
    bus["gjr_vol"]        = _align(feats["gjr_vol"])
    bus["spy_ret_20"]     = _align(spy_ret_20)
    bus["c_level"]        = _align(feats["c_level"])
    bus["c_shock"]        = _align(feats["c_shock"])
    bus["c_accel"]        = _align(feats["c_accel"])
    bus["liq_level"]      = _align(feats["liq_level"])
    bus["liq_shock"]      = _align(feats["liq_shock"])
    bus["liq_accel"]      = _align(feats["liq_accel"])
    bus["stress_idx"]     = _align(feats["stress_idx"])
    bus["roll_corr"]      = _align(feats["roll_corr"])
    bus = bus.dropna()
    # drop warm-up period — credit/liquidity 504-day windows are unreliable before this
    return bus.iloc[WARMUP_DAYS:]


# ════════════════════════════════════════════════════════════════════════════
# 5. ESTIMATORS  (CUSUM · BOCPD · HMM)
# ════════════════════════════════════════════════════════════════════════════

def _expanding_zscore(s, min_periods=120):
    """Causal z-score: at time t uses only data up to t. No lookahead."""
    mu  = s.expanding(min_periods=min_periods).mean()
    std = s.expanding(min_periods=min_periods).std()
    return ((s - mu) / (std + 1e-9)).dropna()


def _hmm_walk_forward(features_df, n_states, random_state,
                      init_window=252, refit_every=21):
    """
    Walk-forward HMM: parameters fit only on data seen so far.
    - init_window : minimum days before first fit (1 year)
    - refit_every : refit monthly (21 trading days)
    At each step we fit on [0..t], then take the last decoded state as
    the regime at t — parameters are causal, no future data leaks in.
    Slow but honest. Progress printed every 6 months.
    """
    from hmmlearn.hmm import GaussianHMM

    n       = len(features_df)
    states  = np.full(n, np.nan)
    model   = None
    next_refit = init_window

    for t in range(init_window, n):
        if t >= next_refit:
            X = features_df.iloc[:t].values
            m = GaussianHMM(n_components=n_states, covariance_type="full",
                            n_iter=100, random_state=random_state)
            try:
                m.fit(X)
                model = m
            except Exception:
                pass   # keep previous model if this fit fails
            next_refit = t + refit_every
            if t % 126 == 0:   # progress every ~6 months
                print(f"      HMM walk-forward: {t}/{n} days fitted")

        if model is not None:
            X_so_far = features_df.iloc[:t + 1].values
            decoded  = model.predict(X_so_far)
            states[t] = decoded[-1]

    regime_series = pd.Series(states, index=features_df.index)
    return regime_series.dropna().astype(int)


def run_estimators(bus, force=False):
    if not force:
        cached = _load("estimators")
        if cached is not None:
            print("  [estimators] loaded from cache")
            return cached

    print("  [estimators] running ...")
    results = {}

    # ── CUSUM ────────────────────────────────────────────────────────────────
    # Level signal: 10-day smoothed (vol + credit + liquidity).
    # Expanding z-score — causal, no full-sample mean/std leakage.
    # CUSUM detects persistent upward/downward shifts in this stress level.
    print("    CUSUM ...")
    raw_level = (
        bus["gjr_vol"].rolling(10).mean().fillna(0)
        + bus["c_level"].rolling(10).mean().fillna(0)
        + bus["liq_level"].rolling(10).mean().fillna(0)
    )
    level_signal = _expanding_zscore(raw_level)
    cusum = CUSUMDetector(k=0.3, h=6.0)
    results["cusum"]        = cusum.detect(level_signal)
    results["cusum_signal"] = level_signal

    # ── BOCPD ────────────────────────────────────────────────────────────────
    # Predictive surprise: squared deviation of each new observation from the
    # BOCPD's running regime mean, normalised by its running variance.
    # Sequential and fully causal — the model only uses past observations.
    print("    BOCPD ...")
    bocpd         = StudentTBOCPD(hazard=1 / 50, nu=5)
    surprise_list = []
    clean_level   = level_signal.dropna()
    for x in clean_level:
        mean, var = bocpd.predictive()
        surprise_list.append((float(x) - mean) ** 2 / (var + 1e-9))
        bocpd.step(x)
    surprise        = pd.Series(surprise_list, index=clean_level.index)
    surprise_smooth = surprise.rolling(20).mean()
    surprise_norm   = (surprise_smooth - surprise_smooth.min()) / \
                      (surprise_smooth.max() - surprise_smooth.min() + 1e-9)
    results["bocpd"]          = surprise_norm
    results["bocpd_pressure"] = surprise_smooth

    # ── HMM — walk-forward (no lookahead) ────────────────────────────────────
    # Features: gjr_vol (Bear detector), kalman_slope (Bull vs Sideways), c_level.
    # Expanding z-score per feature — causal standardisation.
    # Features: gjr_vol (Bear detector), spy_ret_20 (Bull/Bear direction),
    # kalman_slope (trend velocity). SPY rolling return replaces c_level —
    # more directly discriminates Bull vs Bear than slow credit spreads.
    print("    HMM (walk-forward — this will take a few minutes) ...")
    raw_hmm = pd.DataFrame({
        "gjr_vol":      bus["gjr_vol"].rolling(5).mean(),
        "spy_ret_20":   bus["spy_ret_20"].rolling(5).mean(),
        "kalman_slope": bus["kalman_slope"].rolling(5).mean(),
    }).dropna()
    # causal per-feature z-score
    hmm_features = raw_hmm.apply(lambda col: _expanding_zscore(col, min_periods=120))
    hmm_features = hmm_features.dropna()

    state_series = _hmm_walk_forward(hmm_features, N_HMM_STATES, random_state=42)

    # ── minimum duration filter: absorb regimes shorter than 5 days ──────────
    def _min_duration(states, min_days=5):
        vals = states.values.tolist()
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(vals):
                j = i
                while j < len(vals) and vals[j] == vals[i]:
                    j += 1
                if (j - i) < min_days and i > 0:
                    vals[i:j] = [vals[i - 1]] * (j - i)
                    changed = True
                i = j
        return pd.Series(vals, index=states.index)

    state_series = _min_duration(state_series, min_days=5)

    # build hmm_df from walk-forward states
    hmm_df = hmm_features.loc[state_series.index].copy()
    hmm_df["state"] = state_series

    # label by mean gjr_vol: lowest → Bull, middle → Sideways, highest → Bear
    state_vol  = hmm_df.groupby("state")["gjr_vol"].mean().sort_values()
    vol_labels = ["Bull", "Sideways", "Bear"]
    label_map  = {s: l for s, l in zip(state_vol.index, vol_labels)}
    hmm_df["regime"] = hmm_df["state"].map(label_map)

    # ── transition matrix from actual walk-forward state sequence ────────────
    n_s   = N_HMM_STATES
    trans = np.zeros((n_s, n_s))
    sv    = state_series.values
    for a, b in zip(sv[:-1], sv[1:]):
        trans[int(a), int(b)] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans = trans / row_sums
    # reorder rows/cols to match label_map so matrix labels are Bull/Sideways/Bear
    ordered_states = [k for k, _ in sorted(label_map.items(), key=lambda x: ["Bull","Sideways","Bear"].index(x[1]))]
    trans_df = pd.DataFrame(trans, index=range(n_s), columns=range(n_s))
    trans_df = trans_df.loc[ordered_states, ordered_states]
    trans_df.index   = [label_map[s] for s in ordered_states]
    trans_df.columns = [label_map[s] for s in ordered_states]

    results["hmm_df"]        = hmm_df
    results["hmm_model"]     = None
    results["transition_mx"] = trans_df

    _save("estimators", results)
    print("  [estimators] done")
    return results


# ════════════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ════════════════════════════════════════════════════════════════════════════

def _shade_regimes(ax, regime_series):
    """Background colour bands for Bear/Bull/Sideways regimes."""
    if regime_series.empty:
        return
    dates   = regime_series.index
    current = regime_series.iloc[0]
    start   = dates[0]
    for i in range(1, len(dates)):
        if regime_series.iloc[i] != current:
            ax.axvspan(start, dates[i], alpha=0.12,
                       color=REGIME_COLORS.get(current, "grey"), lw=0)
            current = regime_series.iloc[i]
            start   = dates[i]
    ax.axvspan(start, dates[-1], alpha=0.12,
               color=REGIME_COLORS.get(current, "grey"), lw=0)


def _regime_aligned(results, spy_close):
    return (results["hmm_df"]["regime"]
            .reindex(spy_close.index, method="ffill")
            .dropna())


def _shade_cusum_state(ax, cusum_sig):
    """Shade background by CUSUM state: red after upward break, green after downward."""
    state  = 0
    start  = cusum_sig.index[0]
    COLORS = {1: "#e74c3c", -1: "#2ecc71", 0: None}
    for i, (date, val) in enumerate(cusum_sig.items()):
        if val != 0:
            if COLORS[state]:
                ax.axvspan(start, date, alpha=0.12, color=COLORS[state], lw=0)
            state = int(val)
            start = date
    if COLORS[state]:
        ax.axvspan(start, cusum_sig.index[-1], alpha=0.12, color=COLORS[state], lw=0)


def _shade_bocpd_anomaly(ax, surprise_norm, threshold=0.05):
    """Shade background red where normalised surprise exceeds threshold."""
    above = surprise_norm > threshold
    in_anomaly = False
    start = None
    for date, val in above.items():
        if val and not in_anomaly:
            start = date
            in_anomaly = True
        elif not val and in_anomaly:
            ax.axvspan(start, date, alpha=0.15, color="#e74c3c", lw=0)
            in_anomaly = False
    if in_anomaly and start:
        ax.axvspan(start, above.index[-1], alpha=0.15, color="#e74c3c", lw=0)


def plot_cusum(bus, results, spy_close):
    level_sig = results["cusum_signal"]
    cusum_sig = results["cusum"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("CUSUM — Structural Break Detection", fontsize=13, fontweight="bold")

    # Panel 1: price with CUSUM state backdrop
    ax = axes[0]
    ax.plot(spy_close.index, spy_close.values, color="#2c3e50", lw=0.9, zorder=5)
    _shade_cusum_state(ax, cusum_sig)
    ax.fill_between([], [], color="#e74c3c", alpha=0.3, label="Stress elevated (post upward break)")
    ax.fill_between([], [], color="#2ecc71", alpha=0.3, label="Stress falling (post downward break)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("SPY Price — CUSUM state backdrop")
    ax.set_ylabel("Price")

    # Panel 2: level signal with CUSUM state backdrop
    ax = axes[1]
    ax.plot(level_sig.index, level_sig.values, color="#2c3e50", lw=0.9, zorder=5)
    ax.axhline(0, color="black", lw=0.4)
    _shade_cusum_state(ax, cusum_sig)
    ax.set_title("CUSUM input — level signal (10-day smoothed GJR-vol + credit + liquidity, expanding z-score)")
    ax.set_ylabel("z-score")

    # Panel 3: break markers
    ax = axes[2]
    breaks_up   = cusum_sig[cusum_sig == 1]
    breaks_down = cusum_sig[cusum_sig == -1]
    ax.vlines(breaks_up.index,   0,  1, color="#e74c3c", lw=1.3, label=f"Upward stress break ({len(breaks_up)})")
    ax.vlines(breaks_down.index, 0, -1, color="#2ecc71", lw=1.3, label=f"Downward stress break ({len(breaks_down)})")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("CUSUM break events  (k=0.3, h=6.0)")
    ax.set_ylabel("Direction")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "cusum.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_bocpd(results, spy_close):
    cp_probs = results["bocpd"]
    pressure = results["bocpd_pressure"]
    p95      = cp_probs.quantile(0.95)

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    fig.suptitle("BOCPD — Bayesian Change Point Detection", fontsize=13, fontweight="bold")

    # Panel 1: price with BOCPD anomaly backdrop
    ax = axes[0]
    ax.plot(spy_close.index, spy_close.values, color="#2c3e50", lw=0.9, zorder=5)
    _shade_bocpd_anomaly(ax, cp_probs, threshold=p95)
    ax.fill_between([], [], color="#e74c3c", alpha=0.3, label=f"Anomaly (surprise > 95th pct)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("SPY Price — BOCPD anomaly backdrop (red = model surprised)")
    ax.set_ylabel("Price")

    # Panel 2: normalised surprise with anomaly backdrop
    ax = axes[1]
    ax.fill_between(cp_probs.index, cp_probs.values, alpha=0.7, color="#9b59b6", zorder=5)
    _shade_bocpd_anomaly(ax, cp_probs, threshold=p95)
    ax.axhline(p95, color="#e74c3c", lw=0.8, ls="--", label=f"95th pct ({p95:.2f})")
    ax.set_ylim(0, 1)
    ax.set_title("BOCPD Predictive Surprise (normalised) — high = model sees anomaly")
    ax.set_ylabel("Surprise (0–1)")
    ax.legend(fontsize=8)

    # Panel 3: raw surprise
    ax = axes[2]
    ax.fill_between(pressure.index, pressure.values, alpha=0.7, color="#2980b9")
    _shade_bocpd_anomaly(ax, cp_probs, threshold=p95)
    ax.set_title("BOCPD Raw Predictive Surprise (20-day smoothed squared deviation from regime mean)")
    ax.set_ylabel("Surprise")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "bocpd.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_hmm(results, spy_close, feats):
    hmm_df = results["hmm_df"]
    regime = _regime_aligned(results, spy_close)
    trans  = results["transition_mx"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("HMM Gaussian — Latent Regime Classification", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(spy_close.index, spy_close.values, color="#2c3e50", lw=0.9, zorder=5)
    _shade_regimes(ax, regime)
    for label, color in REGIME_COLORS.items():
        ax.fill_between([], [], color=color, alpha=0.4, label=label)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("SPY Price — HMM regime shading")
    ax.set_ylabel("Price")

    ax = axes[1]
    numeric = regime.map({"Bull": 1, "Sideways": 0, "Bear": -1}).astype(float)
    ax.fill_between(numeric.index, numeric.values, step="mid", color="#3498db", alpha=0.65)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Bear", "Sideways", "Bull"])
    ax.set_title("Latent regime")
    ax.set_ylabel("Regime")

    # GJR-GARCH raw vol coloured by regime (use feats not hmm_df which holds z-scores)
    ax = axes[2]
    gjr = feats["gjr_vol"].reindex(spy_close.index, method="ffill").dropna()
    for label, color in REGIME_COLORS.items():
        mask = regime.reindex(gjr.index) == label
        ax.fill_between(gjr.index, gjr.where(mask).values,
                        alpha=0.75, color=color, label=label)
    ax.set_title("GJR-GARCH vol coloured by regime")
    ax.set_ylabel("Vol (%)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "hmm.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")

    # transition matrix heatmap — built from actual walk-forward states
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    trans = results["transition_mx"]
    labels = list(trans.index)
    im = ax2.imshow(trans.values, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(len(labels))); ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels); ax2.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax2.text(j, i, f"{trans.values[i, j]:.2f}", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax2)
    ax2.set_title("HMM Transition Matrix")
    ax2.set_xlabel("To")
    ax2.set_ylabel("From")
    fig2.tight_layout()
    path2 = os.path.join(PLOT_DIR, "hmm_transition.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [plot] {path2}")


def plot_overview(data, feats, kalman_out, bus, results):
    spy_close = data["spy_close"]
    regime    = _regime_aligned(results, spy_close)
    cp_probs  = results["bocpd"]
    cusum_sig = results["cusum"]

    fig = plt.figure(figsize=(20, 24))
    gs  = gridspec.GridSpec(7, 1, hspace=0.45)
    fig.suptitle(f"Market Regime System Overview — {START} to {END}  |  {ASSET}",
                 fontsize=14, fontweight="bold")

    # ── Panel 1: Price + HMM regime ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(spy_close.index, spy_close.values, color="#2c3e50", lw=0.9)
    _shade_regimes(ax1, regime)
    for label, color in REGIME_COLORS.items():
        ax1.fill_between([], [], color=color, alpha=0.4, label=label)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_title("SPY Price — HMM Regime Shading")
    ax1.set_ylabel("Price")

    # ── Panel 2: GJR-GARCH vol ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    gjr = feats["gjr_vol"]
    ax2.fill_between(gjr.index, gjr.values, alpha=0.75, color="#e67e22")
    ax2.set_title("GJR-GARCH(1,1,1) Conditional Volatility")
    ax2.set_ylabel("Vol (%)")

    # ── Panel 3: UKF Kalman trend + slope ────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(spy_close.index, spy_close.values, color="#bdc3c7", lw=0.6, label="Price")
    ax3.plot(kalman_out.index, kalman_out["trend"].values,
             color="#2980b9", lw=1.2, label="UKF trend")
    ax3_r = ax3.twinx()
    ax3_r.plot(kalman_out.index, kalman_out["slope"].values,
               color="#8e44ad", lw=0.8, alpha=0.7, label="Slope")
    ax3_r.axhline(0, color="#8e44ad", lw=0.4, ls="--")
    ax3_r.set_ylabel("Slope", color="#8e44ad", fontsize=8)
    ax3.legend(fontsize=7, loc="upper left")
    ax3.set_title("UKF Kalman — Trend & Velocity")
    ax3.set_ylabel("Price")

    # ── Panel 4: Credit + liquidity + stress ─────────────────────────────────
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(feats["stress_idx"].index, feats["stress_idx"].values,
             color="#c0392b", lw=1.1, label="Stress index", zorder=5)
    ax4.plot(feats["c_level"].index, feats["c_level"].values,
             color="#e74c3c", lw=0.7, alpha=0.55, label="Credit level")
    ax4.plot(feats["liq_level"].index, feats["liq_level"].values,
             color="#2980b9", lw=0.7, alpha=0.55, label="Liquidity level")
    ax4.axhline(0, color="black", lw=0.4)
    ax4.legend(fontsize=7)
    ax4.set_title("Stress Index — Credit + Liquidity (z-score)")
    ax4.set_ylabel("z-score")

    # ── Panel 5: SPY-TLT rolling correlation ─────────────────────────────────
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    rc = feats["roll_corr"]
    ax5.plot(rc.index, rc.values, color="#16a085", lw=0.9)
    ax5.axhline(0, color="black", lw=0.4)
    ax5.set_title("SPY–TLT Rolling Correlation (Fisher-transformed, 30d window)")
    ax5.set_ylabel("Fisher corr")

    # ── Panel 6: BOCPD pressure ──────────────────────────────────────────────
    pressure = results["bocpd_pressure"]
    ax6 = fig.add_subplot(gs[5], sharex=ax1)
    ax6.fill_between(pressure.index, pressure.values, alpha=0.65, color="#9b59b6")
    p95 = pressure.quantile(0.95)
    ax6.axhline(p95, color="#e74c3c", lw=0.7, ls="--", label=f"95th pct")
    ax6.legend(fontsize=7)
    ax6.set_title("BOCPD — 20-day Changepoint Pressure (rolling sum of R[0])")
    ax6.set_ylabel("Pressure")

    # ── Panel 7: CUSUM breaks ────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[6], sharex=ax1)
    cusum_sig = results["cusum"]
    bu = cusum_sig[cusum_sig == 1]
    bd = cusum_sig[cusum_sig == -1]
    ax7.vlines(bu.index, 0,  1, color="#2ecc71", lw=1.0,
               label=f"Upward ({len(bu)})")
    ax7.vlines(bd.index, 0, -1, color="#e74c3c", lw=1.0,
               label=f"Downward ({len(bd)})")
    ax7.axhline(0, color="black", lw=0.4)
    ax7.set_ylim(-1.5, 1.5)
    ax7.legend(fontsize=7)
    ax7.set_title("CUSUM Breaks")
    ax7.set_ylabel("Direction")

    path = os.path.join(PLOT_DIR, "overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")
    return path


def plot_gjr(feats, data, results):
    gjr    = feats["gjr_vol"]
    spy    = data["spy_close"]
    regime = _regime_aligned(results, spy)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("GJR-GARCH(1,1,1) — Walk-Forward Conditional Volatility", fontsize=13, fontweight="bold")

    # ── Panel 1: SPY price with regime shading ───────────────────────────────
    ax = axes[0]
    ax.plot(spy.index, spy.values, color="#2c3e50", lw=0.9)
    _shade_regimes(ax, regime)
    for label, color in REGIME_COLORS.items():
        ax.fill_between([], [], color=color, alpha=0.4, label=label)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("SPY Price (HMM regime shading)")
    ax.set_ylabel("Price")

    # ── Panel 2: Conditional vol coloured by regime ───────────────────────────
    ax = axes[1]
    for label, color in REGIME_COLORS.items():
        mask = regime.reindex(gjr.index, method="ffill") == label
        ax.fill_between(gjr.index, gjr.where(mask).values,
                        alpha=0.8, color=color, label=label)
    ax.set_title("Conditional Volatility (%) — coloured by HMM regime")
    ax.set_ylabel("Vol (%)")
    ax.legend(fontsize=8)

    # ── Panel 3: Vol with VIX overlay ────────────────────────────────────────
    ax = axes[2]
    ax.fill_between(gjr.index, gjr.values, alpha=0.6, color="#e67e22", label="GJR-GARCH vol")
    vix = data["vix_close"].reindex(gjr.index, method="ffill")
    ax2r = ax.twinx()
    ax2r.plot(vix.index, vix.values, color="#8e44ad", lw=0.8, alpha=0.7, label="VIX")
    ax2r.set_ylabel("VIX", color="#8e44ad", fontsize=8)
    ax.set_title("GJR-GARCH Conditional Vol vs VIX")
    ax.set_ylabel("Vol (%)")
    ax.legend(fontsize=8, loc="upper left")
    ax2r.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "gjr_garch.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


# ════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY PRINT
# ════════════════════════════════════════════════════════════════════════════

def print_summary(data, feats, results):
    hmm_df  = results["hmm_df"]
    regime  = hmm_df["regime"]
    total   = len(hmm_df)
    counts  = regime.value_counts()
    cu      = results["cusum"]
    cp      = results["bocpd"]
    gjr     = feats["gjr_vol"]

    print("\n" + "=" * 62)
    print("  MARKET REGIME SYSTEM — BACKTEST SUMMARY")
    print("=" * 62)
    print(f"  Period         : {START}  →  {END}")
    print(f"  Asset          : {ASSET}  /  correlation pair: {ASSET2}")
    print(f"  Trading days   : {total}")

    print(f"\n  HMM Regime distribution ({N_HMM_STATES} states):")
    for r in ["Bull", "Sideways", "Bear"]:
        c = counts.get(r, 0)
        print(f"    {r:<10}: {c:4d} days  ({100 * c / total:5.1f}%)")

    print(f"\n  HMM Transition matrix (rows = from, cols = to):")
    trans = results["transition_mx"]
    print("  " + "          ".join(["Bull", "Sideways", "Bear"][:N_HMM_STATES]))
    for i, row in trans.iterrows():
        print(f"  {i}  " + "  ".join(f"{v:.3f}" for v in row))

    print(f"\n  CUSUM breaks detected:")
    print(f"    Upward   : {(cu == 1).sum()}")
    print(f"    Downward : {(cu == -1).sum()}")
    print(f"    Total    : {(cu != 0).sum()}")

    cp_high = (cp > 0.5).sum()
    cp_max  = cp.max()
    print(f"\n  BOCPD:")
    print(f"    Days with P(cp) > 0.5 : {cp_high}")
    print(f"    Peak P(changepoint)   : {cp_max:.4f}  on {cp.idxmax().date()}")

    print(f"\n  GJR-GARCH conditional vol:")
    print(f"    Mean : {gjr.mean():.2f}%")
    print(f"    Max  : {gjr.max():.2f}%  on {gjr.idxmax().date()}")
    print(f"    Min  : {gjr.min():.2f}%  on {gjr.idxmin().date()}")

    print("=" * 62 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(force_recompute=False):
    print("\n── Market Regime System ──────────────────────────────────────")
    print(f"  {START} → {END}  |  {ASSET}")
    if force_recompute:
        print("  force recompute: all cache ignored\n")

    data       = load_data(force=force_recompute)
    feats      = run_features(data, force=force_recompute)
    kalman_out = run_kalman(data, feats, force=force_recompute)
    bus        = build_signal_bus(data, feats, kalman_out)
    results    = run_estimators(bus, force=force_recompute)

    print("\n  Generating plots ...")
    plot_gjr(feats, data, results)
    plot_cusum(bus, results, data["spy_close"])
    plot_bocpd(results, data["spy_close"])
    plot_hmm(results, data["spy_close"], feats)
    overview = plot_overview(data, feats, kalman_out, bus, results)

    print_summary(data, feats, results)
    print(f"  Overview  → {overview}")
    print(f"  All plots → {PLOT_DIR}")
    print(f"  Cache     → {CACHE_DIR}")
    print("─────────────────────────────────────────────────────────────\n")

    return data, feats, kalman_out, bus, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Regime System backtest 2010-2025")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cache and recompute everything")
    args = parser.parse_args()
    main(force_recompute=args.force)
