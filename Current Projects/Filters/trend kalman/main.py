'''this is the main file for the project

outlining theory for a Kalman Filter:

1. x_t = F* x_{t-1} + w_t, where x_t is market state, x_{t-1} is previous state, F is state transition matrix, w_t is process noise
The transition matrix F changes for each model, but typically  2D state space (price and velocity)
w_t is assumed to be Gaussian with mean 0 and covariance Q

Covariance itself is estimated: 

P_t = F* P_{t-1} * F^T + Q, where P_t is the covariance of the state estimate at time t, F is the state transition matrix, 
P_{t-1} is the covariance of the state estimate at time t-1, F^T is the transpose of F, and Q is the covariance of the process noise w_t.

2. y_t = H* x_t + v_t, where y_t is the observed price, H is the observation matrix, 
x_t is the market state, v_t is observation noise
v_t is also assumed to be Gaussian with mean 0 and covariance R

3. we measure the error (Innovation) of the state estimate: 
e_t = y_t - H* x_t, where e_t is the innovation at time t, y_t is the observed price at time t, 
H is the observation matrix, and x_t is the market state at time t
The innovation also has covariance S_t = H* P_t * H^T + R, where S_t is the covariance of the innovation at time t,
H is the observation matrix, P_t is the covariance of the state estimate at time t, H^T is the transpose of H, and R is the covariance of the observation noise v_t.

4. The Kalman Gain is calculated as K_t = P_t * H^T * S_t^{-1}, where K_t is the Kalman Gain at time t, P_t is the covariance of the state estimate at time t,
H^T is the transpose of the observation matrix H, and S_t^{-1} is the inverse of the covariance of the innovation at time t.

The idea is that the Kalman Gain probabilistically tells us how much to trust the new observation vs our current state estimate. 
If the innovation covariance S_t is large (i.e., the observation is noisy),
the Kalman Gain K_t will be small, meaning we trust our current state estimate more than the new observation.
Conversely, if S_t is small (i.e., the observation is reliable), K_t will be large, and we will give more weight to the new observation in updating our state estimate.

5. We update our state estimate with the new observation:
x_{t+1} = x_t + K_t * e_t, where x_{t+1} is the updated state estimate at time t+1, x_t is the prior state estimate at time t,
K_t is the Kalman Gain at time t, and e_t is the innovation at time t. (new state = old state + correction)

We also update the covariance of our state estimate:
P_{t+1} = (I - K_t * H) * P_t,

General comments:
If Q too large, too noisy
If R too large, too slow to react


Kalman variants to build: TAKE DAILY SPY CLOSE

2-state kalman
Adaptive kalman
Unscented Kalman (non-linear)

Particle filter does not assume Gaussian noise

Improvements: make Bayesian update with student-t distribution!


'''
# ============================================================
# S&P 500 BACKTEST — ALL TREND FILTERS
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

from kalman2D import TwoDKalman
from adapt_kalman import AdaptiveKalmanFilter
from unscent_kalman import AdaptiveUKF
from particle_filter import ParticleFilter
from benchmarks import SimpleTrend
from testing import Testing

# ────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────
TICKER    = "SPY"
START     = "2010-01-01"
END       = "2024-12-31"
HP_LAMBDA = 1600
CACHE_DIR = "./cache"
PLOT_DIR  = "./plots"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────
# 1. PRICE DATA  (cached)
# ────────────────────────────────────────────────────────────
price_cache = f"{CACHE_DIR}/spy_close.pkl"
if os.path.exists(price_cache):
    prices = pd.read_pickle(price_cache)
    print("Loaded prices from cache.")
else:
    raw = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][TICKER].dropna()
    else:
        prices = raw["Close"].dropna()
    prices.to_pickle(price_cache)
    print(f"Downloaded {len(prices)} days of {TICKER}.")

prices.index = pd.to_datetime(prices.index)
prices = prices.sort_index()

# ────────────────────────────────────────────────────────────
# 2. VOLATILITY & STRESS PROXIES
#    vol  = 21-day rolling std of returns, clipped ≥ 0
#    stress = zeros (no credit/liquidity feed available)
# ────────────────────────────────────────────────────────────
returns  = prices.pct_change().fillna(0)
vol_raw  = returns.rolling(21).std().fillna(0)
vol_norm = ((vol_raw - vol_raw.mean()) / (vol_raw.std() + 1e-9)).clip(lower=0)
stress   = pd.Series(np.zeros(len(prices)), index=prices.index)

# ────────────────────────────────────────────────────────────
# 3. RUN FILTERS  (cached)
# ────────────────────────────────────────────────────────────
results_cache = f"{CACHE_DIR}/filter_results.pkl"
if os.path.exists(results_cache):
    with open(results_cache, "rb") as f:
        results = pickle.load(f)
    print("Loaded filter results from cache.")
else:
    results = {}
    st = SimpleTrend(prices)

    # HP filter — ground truth
    hp = st.HP_filter(lamb=HP_LAMBDA)
    results["HP"] = {"trend": hp["trend"], "slope": hp["slope"]}

    # Benchmarks
    sma = st.SMA(50)
    results["SMA_50"] = {"trend": sma["trend"], "slope": sma["slope"]}

    ema = st.EMA(50)
    results["EMA_50"] = {"trend": ema["trend"], "slope": ema["slope"]}

    sg = st.SG_causal(51, 3)
    results["SG_causal"] = {"trend": sg["trend"], "slope": sg["slope"]}

    # 2D Kalman
    trend_2d, slope_2d = TwoDKalman().filter(prices)
    results["Kalman2D"] = {"trend": trend_2d, "slope": slope_2d}

    # Adaptive Kalman
    trend_ak, slope_ak = AdaptiveKalmanFilter().filter(prices, vol_norm, stress)
    results["AdaptKalman"] = {"trend": trend_ak, "slope": slope_ak}

    # UKF
    ukf_out = AdaptiveUKF().filter(prices, vol_norm, stress)
    results["UKF"] = {"trend": ukf_out["trend"], "slope": ukf_out["slope"]}

    # Particle Filter
    pf_out = ParticleFilter(n_particles=500).filter(prices, vol_norm, stress)
    results["Particle"] = {"trend": pf_out["trend"], "slope": pf_out["slope"]}

    with open(results_cache, "wb") as f:
        pickle.dump(results, f)
    print("Filter results cached.")

# ────────────────────────────────────────────────────────────
# 4. BENCHMARKS CACHE  (separate file)
# ────────────────────────────────────────────────────────────
bench_cache = f"{CACHE_DIR}/benchmarks.pkl"
if not os.path.exists(bench_cache):
    bench = {k: results[k] for k in ["HP", "SMA_50", "EMA_50", "SG"]}
    with open(bench_cache, "wb") as f:
        pickle.dump(bench, f)
    print("Benchmarks cached.")

# ────────────────────────────────────────────────────────────
# 5. PLOTS
# ────────────────────────────────────────────────────────────
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm

# Pre-compute common index so plots align with evaluation
# (duplicated here to allow plots to run before section 6)
_hp_t = results["HP"]["trend"]
_hp_s = results["HP"]["slope"]
_cidx  = _hp_t.dropna().index
for _r in results.values():
    _cidx = _cidx.intersection(_r["trend"].dropna().index)
    _cidx = _cidx.intersection(_r["slope"].dropna().index)

FILTER_NAMES = [n for n in results if n != "HP"]
dates        = _cidx
date_nums    = mdates.date2num(dates.to_pydatetime())

def _fmt_xaxis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

# ── 5a. SIGNAL REGIME HEATMAP  ──────────────────────────────
# Rows = filters, time = x-axis, colour = normalised slope
# Green = bullish trend, Red = bearish trend, white = flat
# ────────────────────────────────────────────────────────────
heatmap_path = f"{PLOT_DIR}/heatmap_signals.png"
if not os.path.exists(heatmap_path):
    slope_matrix = np.array([
        results[n]["slope"].reindex(_cidx).values for n in FILTER_NAMES
    ])
    # row-wise normalisation to [-1, 1] so all filters are comparable
    row_max = np.abs(slope_matrix).max(axis=1, keepdims=True) + 1e-9
    slope_norm_mat = slope_matrix / row_max

    hp_slope_vals = _hp_s.reindex(_cidx).values
    hp_norm       = hp_slope_vals / (np.abs(hp_slope_vals).max() + 1e-9)

    fig = plt.figure(figsize=(20, 11))
    gs  = fig.add_gridspec(
        4, 2,
        height_ratios=[2.5, 1.5, 1.5, 0.12],
        width_ratios=[1, 0.018],
        hspace=0.08, wspace=0.03
    )
    ax_price = fig.add_subplot(gs[0, 0])
    ax_heat  = fig.add_subplot(gs[1, 0], sharex=ax_price)
    ax_agree = fig.add_subplot(gs[2, 0], sharex=ax_price)
    ax_cbar  = fig.add_subplot(gs[1, 1])

    # — price panel with HP regime background —
    hp_up   = hp_slope_vals > 0
    ax_price.fill_between(dates, prices.reindex(_cidx).values,
                          where=hp_up,  alpha=0.08, color="green", label="_")
    ax_price.fill_between(dates, prices.reindex(_cidx).values,
                          where=~hp_up, alpha=0.08, color="red",   label="_")
    ax_price.plot(dates, prices.reindex(_cidx).values,
                  color="black", lw=0.7, alpha=0.6, label="SPY")
    ax_price.plot(dates, _hp_t.reindex(_cidx).values,
                  color="black", lw=1.8, ls="--", label="HP trend (truth)")
    ax_price.set_ylabel("Price")
    ax_price.set_title("SPY — All Filter Regime Signals vs HP Truth", fontsize=13)
    ax_price.legend(fontsize=8, loc="upper left")
    ax_price.set_xlim(date_nums[0], date_nums[-1])
    _fmt_xaxis(ax_price)

    # — signal heatmap —
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    X_edge = np.append(date_nums, date_nums[-1] + (date_nums[-1] - date_nums[-2]))
    Y_edge = np.arange(len(FILTER_NAMES) + 1)
    im = ax_heat.pcolormesh(
        X_edge, Y_edge, slope_norm_mat,
        cmap="RdYlGn", norm=norm, shading="flat"
    )
    ax_heat.set_yticks(np.arange(len(FILTER_NAMES)) + 0.5)
    ax_heat.set_yticklabels(FILTER_NAMES, fontsize=9)
    ax_heat.set_ylabel("Filter")
    ax_heat.set_title("Normalised slope  (green = bullish, red = bearish)", fontsize=9)
    ax_heat.xaxis_date()
    _fmt_xaxis(ax_heat)
    plt.colorbar(im, cax=ax_cbar, label="Norm. slope")

    # — agreement bar: fraction of filters agreeing with HP sign at each bar —
    hp_sign   = np.sign(hp_slope_vals)
    agree_mat = np.sign(slope_matrix) == hp_sign[None, :]
    agree_frac = agree_mat.mean(axis=0)

    ax_agree.fill_between(dates, agree_frac, 0.5,
                          where=agree_frac >= 0.5, color="green", alpha=0.5)
    ax_agree.fill_between(dates, agree_frac, 0.5,
                          where=agree_frac < 0.5,  color="red",   alpha=0.5)
    ax_agree.plot(dates, agree_frac, color="black", lw=0.6)
    ax_agree.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax_agree.set_ylim(0, 1)
    ax_agree.set_ylabel("Filter agreement\nwith HP (%)", fontsize=8)
    ax_agree.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
    )
    ax_agree.xaxis_date()
    _fmt_xaxis(ax_agree)

    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {heatmap_path}")

# ── 5b. ROLLING ERROR HEATMAP  ──────────────────────────────
# Rolling 63-day (≈ 1 quarter) RSE per filter vs HP slope
# Shows when each filter tracks HP well vs poorly over time
# ────────────────────────────────────────────────────────────
error_heatmap_path = f"{PLOT_DIR}/heatmap_rolling_error.png"
if not os.path.exists(error_heatmap_path):
    ROLL = 63
    error_matrix = np.full((len(FILTER_NAMES), len(dates)), np.nan)
    hp_s_vals = _hp_s.reindex(_cidx).values

    for i, name in enumerate(FILTER_NAMES):
        fs = results[name]["slope"].reindex(_cidx).values
        sq_err = (fs - hp_s_vals) ** 2
        for t in range(ROLL, len(dates)):
            error_matrix[i, t] = sq_err[t - ROLL:t].mean()

    # log-scale so large outliers don't dominate colour range
    log_err = np.log1p(error_matrix)
    log_err[:, :ROLL] = np.nan

    fig, (ax_p, ax_e) = plt.subplots(
        2, 1, figsize=(20, 8),
        gridspec_kw={"height_ratios": [1.5, 2]},
        sharex=True
    )
    ax_p.plot(dates, prices.reindex(_cidx).values,
              color="black", lw=0.7, alpha=0.6)
    ax_p.plot(dates, _hp_t.reindex(_cidx).values,
              color="navy", lw=1.5, ls="--", label="HP trend")
    ax_p.set_ylabel("Price")
    ax_p.set_title("Rolling 63-day Slope Error vs HP (log scale — darker = larger error)", fontsize=11)
    ax_p.legend(fontsize=8)
    _fmt_xaxis(ax_p)

    X_edge = np.append(date_nums, date_nums[-1] + (date_nums[-1] - date_nums[-2]))
    Y_edge = np.arange(len(FILTER_NAMES) + 1)
    im2 = ax_e.pcolormesh(
        X_edge, Y_edge, log_err,
        cmap="YlOrRd", shading="flat"
    )
    ax_e.set_yticks(np.arange(len(FILTER_NAMES)) + 0.5)
    ax_e.set_yticklabels(FILTER_NAMES, fontsize=9)
    ax_e.set_ylabel("Filter")
    ax_e.xaxis_date()
    _fmt_xaxis(ax_e)
    plt.colorbar(im2, ax=ax_e, label="log(1 + MSE)", pad=0.01)

    plt.tight_layout()
    plt.savefig(error_heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {error_heatmap_path}")

# ── 5c. INDIVIDUAL FILTER CHARTS  ───────────────────────────
# Price chart with regime-shaded background + slope oscillator
# Background colour = filter's own regime direction
# HP slope overlaid in black as reference
# ────────────────────────────────────────────────────────────
for name in FILTER_NAMES:
    p = f"{PLOT_DIR}/{name}.png"
    if os.path.exists(p):
        continue

    fs   = results[name]["slope"].reindex(_cidx).values
    ft   = results[name]["trend"].reindex(_cidx).values
    hp_s_vals = _hp_s.reindex(_cidx).values
    hp_t_vals = _hp_t.reindex(_cidx).values
    px   = prices.reindex(_cidx).values

    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(16, 8),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True
    )

    # regime background from this filter's slope sign
    filter_up = fs > 0
    a1.fill_between(dates, px, where=filter_up,  alpha=0.10, color="green")
    a1.fill_between(dates, px, where=~filter_up, alpha=0.10, color="red")
    a1.plot(dates, px,         color="black",  lw=0.6, alpha=0.5, label="SPY")
    a1.plot(dates, ft,         color="steelblue", lw=1.4, label=f"{name} trend")
    a1.plot(dates, hp_t_vals,  color="black",  lw=1.2, ls="--", alpha=0.7, label="HP truth")
    a1.set_ylabel("Price")
    a1.set_title(f"{name} — regime background from filter slope, dashed = HP truth")
    a1.legend(fontsize=8, loc="upper left")
    _fmt_xaxis(a1)

    # slope oscillator: filter vs HP
    a2.fill_between(dates, fs, 0, where=fs > 0, color="green", alpha=0.35)
    a2.fill_between(dates, fs, 0, where=fs < 0, color="red",   alpha=0.35)
    a2.plot(dates, fs,        color="steelblue", lw=0.9, label=f"{name} slope")
    a2.plot(dates, hp_s_vals, color="black", lw=1.0, ls="--", alpha=0.6, label="HP slope")
    a2.axhline(0, color="gray", lw=0.5, ls=":")
    a2.set_ylabel("Slope")
    a2.legend(fontsize=8, loc="upper left")
    _fmt_xaxis(a2)

    plt.tight_layout()
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()

print("All plots saved.")

# ────────────────────────────────────────────────────────────
# 6. EVALUATION vs HP FILTER
#    RSE  : residual squared error on trend and slope
#    Lag  : cross-correlation peak offset (days)
#    FP   : fraction of bars where sign(slope) ≠ sign(HP slope)
# ────────────────────────────────────────────────────────────
tester   = Testing()
hp_trend = results["HP"]["trend"]
hp_slope = results["HP"]["slope"]

# common non-NaN index across all filters
common_idx = hp_trend.dropna().index
for res in results.values():
    common_idx = common_idx.intersection(res["trend"].dropna().index)
    common_idx = common_idx.intersection(res["slope"].dropna().index)

hp_t = hp_trend.loc[common_idx].values
hp_s = hp_slope.loc[common_idx].values

metrics = {}
for name, res in results.items():
    if name == "HP":
        continue
    ft = res["trend"].loc[common_idx].values
    fs = res["slope"].loc[common_idx].values

    rse       = tester.RSE(ft, hp_t, fs, hp_s)
    lag_days  = int(tester.lag(fs, hp_s))
    fp_rate   = np.sum(np.sign(fs) != np.sign(hp_s)) / len(hp_s)

    metrics[name] = {
        "trend_RSE": rse["trend_RSE"],
        "slope_RSE": rse["slope_RSE"],
        "lag_days":  lag_days,
        "FP_rate":   fp_rate,
    }

# ────────────────────────────────────────────────────────────
# 7. RANKING  (lower = better for all metrics)
# ────────────────────────────────────────────────────────────
df_m = pd.DataFrame(metrics).T

df_m["rank_trend_RSE"] = df_m["trend_RSE"].rank()
df_m["rank_slope_RSE"] = df_m["slope_RSE"].rank()
df_m["rank_lag"]       = df_m["lag_days"].abs().rank()
df_m["rank_FP"]        = df_m["FP_rate"].rank()
df_m["overall_rank"]   = (
    df_m["rank_trend_RSE"]
    + df_m["rank_slope_RSE"]
    + df_m["rank_lag"]
    + df_m["rank_FP"]
)
df_m = df_m.sort_values("overall_rank")

pd.set_option("display.float_format", "{:.4f}".format)
print("\n" + "=" * 70)
print("  FILTER EVALUATION — S&P 500 BACKTEST (2010–2024)")
print("=" * 70)
print(df_m[["trend_RSE", "slope_RSE", "lag_days", "FP_rate",
            "overall_rank"]].to_string())
print("\nFINAL RANKING  (best → worst):")
for i, name in enumerate(df_m.index, 1):
    score = df_m.loc[name, "overall_rank"]
    print(f"  {i}. {name:<14}  composite score: {score:.1f}")


