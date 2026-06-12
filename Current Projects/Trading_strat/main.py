'''Hurst + Autocorrelation regime classification and backtest on IWM (Russell 2000) 2010-2025.

Entry gate: no entry only when CUSUM-filtered Hurst AND CUSUM-filtered autocorr
both agree the regime is choppy. Otherwise long IWM (Russell 2000).

Run:
    python main.py            # use cached IWM (Russell 2000) data
    python main.py --force    # re-download IWM (Russell 2000)
'''

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from loader import DataLoader
from hurst_exp import HurstExponent
from autocorr import Autocorrelation
from symm_CUSUM import HurstCUSUM, AutocorrCUSUM, HurstPersistence
from GARCH import GARCHModel
from unscent_kalman import AdaptiveUKF

UNIVERSE_PATH  = os.path.join(ROOT, "Data", "universe_prices.csv")
GJR_VOL_PATH   = os.path.join(ROOT, "Data", "gjr_vol.csv")
UKF_SLOPE_PATH = os.path.join(ROOT, "Data", "ukf_slopes.csv")

PLOT_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

START              = "2010-01-01"
END                = "2026-01-01"
HURST_WINDOW       = 120
AUTOCORR_WINDOW    = 60
AUTOCORR_MAX_LAG   = 5
HURST_THRESHOLD    = 0.578  # historical median — centres the CUSUM around the actual distribution
AUTOCORR_THRESHOLD = -0.04  # historical median — daily equity autocorr is structurally negative
HURST_PERSISTENCE_N      = 15   # required hits within the window
HURST_PERSISTENCE_WINDOW = 15   # rolling window in days
LOOKBACK             = 252
SKIP_DAYS            = 21
TOP_N                = 20
REBALANCE_FREQ       = 21
TRANSACTION_COST     = 0.001
STARTING_CAPITAL     = 100_000.0   # euros
FIXED_FEE_PER_STOCK  = 4.0        # euros per stock traded
HURST_CUSUM_K      = 0.02
HURST_CUSUM_H      = 0.50
AUTOCORR_CUSUM_K   = 0.01
AUTOCORR_CUSUM_H   = 0.05


# ════════════════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════════════════

def load_data(force=False) -> pd.Series:
    loader = DataLoader(open_session=False)
    return loader.load_iwm_cached(start=START, end=END, force=force)

def compute_gjr_vol(universe: pd.DataFrame, force=False) -> pd.DataFrame:
    if not force and os.path.exists(GJR_VOL_PATH):
        print("  [gjr] loading cached vol ...")
        return pd.read_csv(GJR_VOL_PATH, index_col=0, parse_dates=True)

    print(f"  [gjr] fitting GJR-GARCH on {universe.shape[1]} stocks (this will take a while) ...")
    vol_df = pd.DataFrame(index=universe.index, columns=universe.columns, dtype=float)

    for i, ticker in enumerate(universe.columns):
        prices = universe[ticker].dropna()
        if len(prices) < 100:
            continue
        try:
            model = GARCHModel(prices_close=prices, prices_intraday=prices)
            fit   = model.apply_gjr_garch()
            cv    = fit.conditional_volatility
            vol_df.loc[cv.index, ticker] = cv.values
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{universe.shape[1]} stocks fitted ...")

    vol_df.to_csv(GJR_VOL_PATH)
    print(f"  [gjr] cached to {GJR_VOL_PATH}")
    return vol_df


def compute_ukf_slopes(universe: pd.DataFrame, force=False) -> pd.DataFrame:
    if not force and os.path.exists(UKF_SLOPE_PATH):
        print("  [ukf] loading cached slopes ...")
        return pd.read_csv(UKF_SLOPE_PATH, index_col=0, parse_dates=True)

    print(f"  [ukf] fitting UKF on {universe.shape[1]} stocks (this will take a while) ...")
    slope_df = pd.DataFrame(index=universe.index, columns=universe.columns, dtype=float)

    for i, ticker in enumerate(universe.columns):
        prices = universe[ticker].dropna()
        if len(prices) < 50:
            continue
        try:
            ukf = AdaptiveUKF()
            out = ukf.filter(prices)
            slope_df.loc[out.index, ticker] = out["slope"].values
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{universe.shape[1]} stocks done ...")

    slope_df.to_csv(UKF_SLOPE_PATH)
    print(f"  [ukf] cached to {UKF_SLOPE_PATH}")
    return slope_df


def load_universe() -> pd.DataFrame:
    print("  [data] loading universe prices ...")
    prices = pd.read_csv(UNIVERSE_PATH, index_col=0, parse_dates=True).sort_index()
    print(f"  [data] {prices.shape[0]} days, {prices.shape[1]} stocks")
    return prices


# ════════════════════════════════════════════════════════════════════════════
# SIGNALS
# ════════════════════════════════════════════════════════════════════════════

def compute_signals(prices: pd.Series) -> pd.DataFrame:

    print("  [signals] computing rolling Hurst ...")
    hurst_model = HurstExponent(min_window=20)
    hurst = hurst_model.rolling(prices, window=HURST_WINDOW)

    print("  [signals] computing rolling autocorrelation ...")
    log_ret = np.log(prices / prices.shift(1)).dropna()

    autocorr_vals = []
    for t in range(len(log_ret)):
        start = max(0, t - AUTOCORR_WINDOW + 1)
        window_ret = log_ret.iloc[start : t + 1]
        ac = Autocorrelation(window_ret)
        if len(window_ret) > AUTOCORR_MAX_LAG + 2:
            autocorr_vals.append(ac.mean_autocorrelation(max_lag=AUTOCORR_MAX_LAG))
        else:
            autocorr_vals.append(np.nan)

    autocorr = pd.Series(autocorr_vals, index=log_ret.index, name="autocorr")

    print("  [signals] applying CUSUM and persistence filters ...")
    hurst_aligned    = hurst.reindex(log_ret.index).dropna()
    autocorr_aligned = autocorr.reindex(hurst_aligned.index).dropna()

    hurst_cusum    = HurstCUSUM(k=HURST_CUSUM_K, h=HURST_CUSUM_H, threshold=HURST_THRESHOLD)
    autocorr_cusum = AutocorrCUSUM(k=AUTOCORR_CUSUM_K, h=AUTOCORR_CUSUM_H, threshold=AUTOCORR_THRESHOLD)
    hurst_persist  = HurstPersistence(n=HURST_PERSISTENCE_N, window=HURST_PERSISTENCE_WINDOW, threshold=HURST_THRESHOLD)

    hurst_cusum_out    = hurst_cusum.filter(hurst_aligned)
    autocorr_cusum_out = autocorr_cusum.filter(autocorr_aligned)
    hurst_persist_out  = hurst_persist.filter(hurst_aligned)

    signals = pd.DataFrame(index=hurst_aligned.index)
    signals["log_ret"]          = log_ret.reindex(hurst_aligned.index)
    signals["hurst"]            = hurst_aligned
    signals["autocorr"]         = autocorr_aligned
    signals["hurst_state"]      = hurst_cusum_out["state"]
    signals["autocorr_state"]   = autocorr_cusum_out["state"]
    signals["hurst_persist"]    = hurst_persist_out

    return signals.dropna()


# ════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ════════════════════════════════════════════════════════════════════════════

def backtest(signals: pd.DataFrame) -> pd.DataFrame:

    bt = signals.copy()

    # no entry only when both CUSUM states agree choppy (-1)
    no_entry = (bt["hurst_state"] == -1) & (bt["autocorr_state"] == -1)

    # shift by 1: signal at close t → position from close t+1
    bt["position"]     = (~no_entry).shift(1).fillna(True).astype(int)
    bt["strat_ret"]    = bt["position"] * bt["log_ret"]
    bt["bh_ret"]       = bt["log_ret"]
    bt["strat_equity"] = bt["strat_ret"].cumsum().apply(np.exp)
    bt["bh_equity"]    = bt["bh_ret"].cumsum().apply(np.exp)

    return bt


# ════════════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════════════

def _sharpe(returns: pd.Series) -> float:
    if returns.std() < 1e-12:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))

def _max_drawdown(equity: pd.Series) -> float:
    dd = (equity - equity.cummax()) / equity.cummax()
    return float(dd.min())

def _cagr(equity: pd.Series, n_days: int) -> float:
    return float(equity.iloc[-1] ** (252 / n_days) - 1)

def print_metrics(bt: pd.DataFrame):
    n            = len(bt)
    invested_pct = bt["position"].mean() * 100

    print("\n" + "=" * 58)
    print("  HURST + AUTOCORR CUSUM GATE — BACKTEST SUMMARY")
    print("=" * 58)
    print(f"  Period          : {START}  →  {END}")
    print(f"  Trading days    : {n}")
    print(f"  Hurst CUSUM     : k={HURST_CUSUM_K}  h={HURST_CUSUM_H}")
    print(f"  Autocorr CUSUM  : k={AUTOCORR_CUSUM_K}  h={AUTOCORR_CUSUM_H}")
    print(f"  Days invested   : {invested_pct:.1f}%")

    for label, ret_col, eq_col in [
        ("Strategy (CUSUM gate)", "strat_ret", "strat_equity"),
        ("Buy & Hold IWM (Russell 2000)",        "bh_ret",    "bh_equity"),
    ]:
        r  = bt[ret_col]
        eq = bt[eq_col]
        print(f"\n  {label}:")
        print(f"    CAGR         : {_cagr(eq, n) * 100:+.2f}%")
        print(f"    Sharpe       : {_sharpe(r):.3f}")
        print(f"    Max drawdown : {_max_drawdown(eq) * 100:.2f}%")
        print(f"    Total return : {(eq.iloc[-1] - 1) * 100:+.2f}%")

    print("=" * 58 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════════════

def _shade(ax, mask, color, alpha=0.25):
    in_region = False
    start = None
    for date, val in mask.items():
        if val and not in_region:
            start = date
            in_region = True
        elif not val and in_region:
            ax.axvspan(start, date, color=color, alpha=alpha, lw=0)
            in_region = False
    if in_region and start:
        ax.axvspan(start, mask.index[-1], color=color, alpha=alpha, lw=0)


def plot_signals(signals: pd.DataFrame, prices: pd.Series):

    iwm         = prices.reindex(signals.index, method="ffill")
    h_trending  = signals["hurst_state"]    == 1
    h_choppy    = signals["hurst_state"]    == -1
    ac_trending = signals["autocorr_state"] == 1
    ac_choppy   = signals["autocorr_state"] == -1
    p_trending  = signals["hurst_persist"]  == 1
    p_choppy    = signals["hurst_persist"]  == -1

    fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=True)
    fig.suptitle("Hurst & Autocorrelation — Raw Signals and Filters  |  IWM (Russell 2000)",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: IWM shaded by raw Hurst ─────────────────────────────────────
    ax = axes[0]
    h_above = signals["hurst"] > HURST_THRESHOLD
    ax.plot(iwm.index, iwm.values, color="#2c3e50", lw=0.9, zorder=5)
    _shade(ax, h_above,  "#2ecc71", alpha=0.3)
    _shade(ax, ~h_above, "#e74c3c", alpha=0.3)
    ax.fill_between([], [], color="#2ecc71", alpha=0.5, label="Trending")
    ax.fill_between([], [], color="#e74c3c", alpha=0.5, label="Choppy")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(f"Raw Hurst  (threshold {HURST_THRESHOLD:.3f})")
    ax.set_ylabel("Price ($)")

    # ── Panel 2: IWM shaded by raw Autocorrelation ───────────────────────────
    ax = axes[1]
    ac_above = signals["autocorr"] > AUTOCORR_THRESHOLD
    ax.plot(iwm.index, iwm.values, color="#2c3e50", lw=0.9, zorder=5)
    _shade(ax, ac_above,  "#2ecc71", alpha=0.3)
    _shade(ax, ~ac_above, "#e74c3c", alpha=0.3)
    ax.fill_between([], [], color="#2ecc71", alpha=0.5, label="Trending")
    ax.fill_between([], [], color="#e74c3c", alpha=0.5, label="Choppy")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(f"Raw Autocorrelation  (threshold {AUTOCORR_THRESHOLD})")
    ax.set_ylabel("Price ($)")

    # ── Panel 3: Hurst CUSUM gate ─────────────────────────────────────────────
    ax = axes[2]
    ax.plot(iwm.index, iwm.values, color="#2c3e50", lw=0.9, zorder=5)
    _shade(ax, h_trending, "#2ecc71", alpha=0.3)
    _shade(ax, h_choppy,   "#e74c3c", alpha=0.3)
    ax.fill_between([], [], color="#2ecc71", alpha=0.5, label="Trending")
    ax.fill_between([], [], color="#e74c3c", alpha=0.5, label="Choppy")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(f"Hurst CUSUM Gate  (k={HURST_CUSUM_K}  h={HURST_CUSUM_H})")
    ax.set_ylabel("Price ($)")

    # ── Panel 4: Autocorr CUSUM gate ──────────────────────────────────────────
    ax = axes[3]
    ax.plot(iwm.index, iwm.values, color="#2c3e50", lw=0.9, zorder=5)
    _shade(ax, ac_trending, "#2ecc71", alpha=0.3)
    _shade(ax, ac_choppy,   "#e74c3c", alpha=0.3)
    ax.fill_between([], [], color="#2ecc71", alpha=0.5, label="Trending")
    ax.fill_between([], [], color="#e74c3c", alpha=0.5, label="Choppy")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(f"Autocorr CUSUM Gate  (k={AUTOCORR_CUSUM_K}  h={AUTOCORR_CUSUM_H})")
    ax.set_ylabel("Price ($)")

    # ── Panel 5: Hurst persistence gate ──────────────────────────────────────
    ax = axes[4]
    ax.plot(iwm.index, iwm.values, color="#2c3e50", lw=0.9, zorder=5)
    _shade(ax, p_trending, "#2ecc71", alpha=0.3)
    _shade(ax, p_choppy,   "#e74c3c", alpha=0.3)
    ax.fill_between([], [], color="#2ecc71", alpha=0.5, label="Trending")
    ax.fill_between([], [], color="#e74c3c", alpha=0.5, label="Choppy")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(f"Hurst Persistence Gate  ({HURST_PERSISTENCE_N} hits in {HURST_PERSISTENCE_WINDOW} days to trigger)")
    ax.set_ylabel("Price ($)")
    ax.set_xlabel("Date")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "signals.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")
    return path


def plot_backtest(bt: pd.DataFrame, prices: pd.Series):

    no_entry = (bt["hurst_state"] == -1) & (bt["autocorr_state"] == -1)
    entry    = ~no_entry

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle("Backtest — CUSUM-gated entry vs Buy & Hold IWM (Russell 2000)", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(bt.index, bt["strat_equity"], color="#2ecc71", lw=1.2, label="Strategy (CUSUM gate)")
    ax.plot(bt.index, bt["bh_equity"],    color="#2c3e50", lw=1.0, label="Buy & Hold IWM (Russell 2000)", alpha=0.7)
    _shade(ax, entry,    "#2ecc71", alpha=0.07)
    _shade(ax, no_entry, "#e74c3c", alpha=0.07)
    ax.legend(fontsize=9)
    ax.set_title("Equity Curves")
    ax.set_ylabel("Portfolio value")

    ax = axes[1]
    strat_dd = (bt["strat_equity"] - bt["strat_equity"].cummax()) / bt["strat_equity"].cummax() * 100
    bh_dd    = (bt["bh_equity"]    - bt["bh_equity"].cummax())    / bt["bh_equity"].cummax()    * 100
    ax.fill_between(bt.index, strat_dd, color="#2ecc71", alpha=0.6, label="Strategy drawdown")
    ax.fill_between(bt.index, bh_dd,    color="#2c3e50", alpha=0.4, label="Buy & Hold drawdown")
    ax.legend(fontsize=9)
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("%")

    ax = axes[2]
    ax.fill_between(bt.index, bt["position"], step="mid", alpha=0.7,
                    color="#27ae60", label="Invested")
    ax.set_ylim(-0.1, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Flat", "Long"])
    ax.legend(fontsize=9)
    ax.set_title("Position")
    ax.set_xlabel("Date")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "backtest.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════
# CMOM SIGNAL
# ════════════════════════════════════════════════════════════════════════════

def _rank_scores(price_data: pd.DataFrame) -> pd.Series:
    daily_ret = price_data.ffill().pct_change(fill_method=None)
    momentum  = price_data.shift(SKIP_DAYS) / price_data.shift(LOOKBACK) - 1
    vol       = daily_ret.rolling(LOOKBACK).std()
    return (momentum / vol).iloc[-1].dropna()


def generate_signals(price_data: pd.DataFrame) -> pd.Series:
    score   = _rank_scores(price_data)
    signals = pd.Series(0.0, index=price_data.columns)
    signals[score.nlargest(TOP_N).index]  =  1.0 / TOP_N
    signals[score.nsmallest(TOP_N).index] = -1.0 / TOP_N
    return signals


def generate_signals_gjr(price_data: pd.DataFrame, gjr_vol: pd.DataFrame) -> pd.Series:
    score         = _rank_scores(price_data)
    top_stocks    = score.nlargest(TOP_N).index
    bottom_stocks = score.nsmallest(TOP_N).index

    # use latest available GJR vol for each stock — 1/σ weighting
    latest_vol = gjr_vol.loc[:price_data.index[-1]].iloc[-1]

    # floor at 5th percentile of the vol distribution to kill degenerate estimates
    vol_floor = latest_vol.quantile(0.05)

    def _vol_weights(stocks):
        v = latest_vol.reindex(stocks).replace(0, np.nan).dropna()
        v = v.clip(lower=vol_floor)           # floor degenerate vols
        w = 1.0 / v
        w = w / w.sum()                       # normalise
        w = w.clip(upper=3.0 / TOP_N)         # cap any single position at 3x equal weight
        w = w / w.sum()                       # renormalise after cap
        return w

    signals = pd.Series(0.0, index=price_data.columns)
    lw = _vol_weights(top_stocks)
    sw = _vol_weights(bottom_stocks)
    signals[lw.index] =  lw.values
    signals[sw.index] = -sw.values
    return signals


# ════════════════════════════════════════════════════════════════════════════
# CMOM BACKTEST
# ════════════════════════════════════════════════════════════════════════════

def _apply_kalman_exits(positions: pd.Series, slopes: pd.Series,
                         slope_stds: pd.Series) -> pd.Series:
    positions = positions.copy()
    exit_threshold = -1.0  # -1 std

    for ticker in positions[positions != 0].index:
        if ticker not in slopes.index or ticker not in slope_stds.index:
            continue
        slope    = slopes[ticker]
        std      = slope_stds[ticker]
        if np.isnan(slope) or np.isnan(std) or std < 1e-10:
            continue
        z_slope  = slope / std   # slope normalised by its own history std
        pos      = positions[ticker]
        # exit long if slope < -1 std, exit short if slope > +1 std
        if pos > 0 and z_slope < exit_threshold:
            positions[ticker] = 0.0
        elif pos < 0 and z_slope > -exit_threshold:
            positions[ticker] = 0.0

    return positions


def _apply_kalman_entry_gate(new_positions: pd.Series, current_positions: pd.Series,
                              slopes: pd.Series, slope_stds: pd.Series) -> pd.Series:
    """Block new entries that would be immediately exited by the Kalman filter."""
    new_positions = new_positions.copy()
    exit_threshold = -1.0

    for ticker in new_positions[new_positions != 0].index:
        if current_positions.get(ticker, 0.0) != 0.0:
            continue  # already in position — exit logic handles it, not entry gate
        if ticker not in slopes.index or ticker not in slope_stds.index:
            continue
        slope = slopes[ticker]
        std   = slope_stds[ticker]
        if np.isnan(slope) or np.isnan(std) or std < 1e-10:
            continue
        z_slope = slope / std
        pos = new_positions[ticker]
        if pos > 0 and z_slope < exit_threshold:
            new_positions[ticker] = 0.0
        elif pos < 0 and z_slope > -exit_threshold:
            new_positions[ticker] = 0.0

    return new_positions


def _run_backtest_loop(universe: pd.DataFrame, gjr_vol: pd.DataFrame | None,
                        ukf_slopes: pd.DataFrame | None) -> pd.DataFrame:
    daily_ret         = universe.ffill().pct_change(fill_method=None)
    dates             = universe.index
    current_positions = pd.Series(0.0, index=universe.columns)
    rebalance_dates   = set(dates[LOOKBACK::REBALANCE_FREQ])

    # precompute rolling std of slopes per stock for normalisation (expanding)
    slope_stds = None
    if ukf_slopes is not None:
        slope_stds = ukf_slopes.expanding().std()

    results = []
    portfolio_value = STARTING_CAPITAL

    for i, date in enumerate(dates):
        if i < LOOKBACK:
            results.append({"date": date, "ret": 0.0, "value": portfolio_value,
                            "n_longs": 0, "n_shorts": 0})
            continue

        day_ret = daily_ret.loc[date].fillna(0)

        # apply Kalman exit before computing today's return
        if ukf_slopes is not None and date in ukf_slopes.index:
            slopes_today = ukf_slopes.loc[date]
            stds_today   = slope_stds.loc[date]
            current_positions = _apply_kalman_exits(
                current_positions, slopes_today, stds_today
            )

        portfolio_ret = (current_positions * day_ret).sum()

        if date in rebalance_dates:
            slice_ = universe.loc[:date]
            if gjr_vol is not None:
                new_pos = generate_signals_gjr(slice_, gjr_vol)
            else:
                new_pos = generate_signals(slice_)

            if ukf_slopes is not None and date in ukf_slopes.index:
                new_pos = _apply_kalman_entry_gate(
                    new_pos, current_positions,
                    ukf_slopes.loc[date], slope_stds.loc[date],
                )

            turnover    = (new_pos - current_positions).abs().sum()
            n_trades    = int(((new_pos - current_positions).abs() > 1e-10).sum())
            fixed_cost  = n_trades * FIXED_FEE_PER_STOCK / max(portfolio_value, 1.0)
            portfolio_ret -= turnover * TRANSACTION_COST + fixed_cost
            current_positions = new_pos

        portfolio_value *= (1 + portfolio_ret)
        results.append({
            "date":     date,
            "ret":      portfolio_ret,
            "value":    portfolio_value,
            "n_longs":  (current_positions > 0).sum(),
            "n_shorts": (current_positions < 0).sum(),
        })

    bt = pd.DataFrame(results).set_index("date")
    bt["equity"] = bt["value"] / STARTING_CAPITAL
    return bt


def run_cmom_backtest(universe: pd.DataFrame, gjr_vol: pd.DataFrame,
                       ukf_slopes: pd.DataFrame) -> pd.DataFrame:
    print("  [cmom] running equal-weight backtest ...")
    ew = _run_backtest_loop(universe, gjr_vol=None, ukf_slopes=None)
    print("  [cmom] running equal-weight + Kalman gate backtest ...")
    ew_kalman = _run_backtest_loop(universe, gjr_vol=None, ukf_slopes=ukf_slopes)
    print("  [cmom] running GJR-GARCH + Kalman gate backtest ...")
    gjr = _run_backtest_loop(universe, gjr_vol=gjr_vol, ukf_slopes=ukf_slopes)

    bt = pd.DataFrame(index=ew.index)
    bt["ew_ret"]        = ew["ret"]
    bt["ew_kalman_ret"] = ew_kalman["ret"]
    bt["gjr_ret"]       = gjr["ret"]
    bt["ew_equity"]        = ew["equity"]
    bt["ew_kalman_equity"] = ew_kalman["equity"]
    bt["gjr_equity"]       = gjr["equity"]
    bt["ew_value"]         = ew["value"]
    bt["ew_kalman_value"]  = ew_kalman["value"]
    bt["gjr_value"]        = gjr["value"]
    bt["n_longs"]    = gjr["n_longs"]
    bt["n_shorts"]   = gjr["n_shorts"]
    return bt


def _metrics(ret: pd.Series, eq: pd.Series) -> dict:
    n = len(eq)
    return {
        "CAGR":         f"{(eq.iloc[-1] ** (252/n) - 1)*100:+.2f}%",
        "Sharpe":       f"{ret.mean()/ret.std()*np.sqrt(252):.3f}",
        "Max drawdown": f"{((eq - eq.cummax())/eq.cummax()).min()*100:.2f}%",
        "Total return": f"{(eq.iloc[-1]-1)*100:+.2f}%",
    }

def print_cmom_metrics(bt: pd.DataFrame):
    print("\n" + "=" * 58)
    print("  CMOM BACKTEST — RUSSELL 2000")
    print("=" * 58)
    print(f"  Lookback        : {LOOKBACK}d  skip {SKIP_DAYS}d")
    print(f"  Top N           : {TOP_N} longs / {TOP_N} shorts")
    print(f"  Rebalance       : every {REBALANCE_FREQ} days")
    print(f"  Transaction cost: {TRANSACTION_COST*100:.1f}bps proportional  +  €{FIXED_FEE_PER_STOCK:.0f}/stock")
    print(f"  Starting capital: €{STARTING_CAPITAL:,.0f}")

    for label, ret_col, eq_col, val_col in [
        ("CMOM equal-weight",               "ew_ret",        "ew_equity",        "ew_value"),
        ("CMOM equal-weight + Kalman gate", "ew_kalman_ret", "ew_kalman_equity", "ew_kalman_value"),
        ("CMOM GJR-GARCH + Kalman gate",    "gjr_ret",       "gjr_equity",       "gjr_value"),
    ]:
        m = _metrics(bt[ret_col].dropna(), bt[eq_col].dropna())
        final_val = bt[val_col].dropna().iloc[-1]
        print(f"\n  {label}:")
        for k, v in m.items():
            print(f"    {k:<14}: {v}")
        print(f"    Final value  : €{final_val:,.0f}")
    print("=" * 58 + "\n")


def plot_cmom_backtest(bt: pd.DataFrame, iwm: pd.Series):

    iwm_aligned = (1 + iwm.pct_change(fill_method=None)).cumprod().reindex(bt.index, method="ffill")
    iwm_aligned = iwm_aligned / iwm_aligned.iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)
    fig.suptitle("CMOM — Equal-Weight vs EW+Kalman vs GJR-GARCH Weighted  |  Russell 2000",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(bt.index, bt["ew_equity"],        color="#2980b9", lw=1.1, label="CMOM equal-weight", alpha=0.85)
    ax.plot(bt.index, bt["ew_kalman_equity"], color="#e67e22", lw=1.1, label="CMOM EW + Kalman gate", alpha=0.85)
    ax.plot(bt.index, bt["gjr_equity"],       color="#2ecc71", lw=1.2, label="CMOM GJR-GARCH + Kalman gate")
    ax.plot(iwm_aligned.index, iwm_aligned.values, color="#2c3e50", lw=0.9,
            label="IWM (Russell 2000)", alpha=0.6)
    ax.legend(fontsize=9)
    ax.set_title(f"Equity Curves  (starting capital €{STARTING_CAPITAL:,.0f})")
    ax.set_ylabel("Portfolio value (normalised)")

    ax = axes[1]
    ew_dd         = (bt["ew_equity"]        - bt["ew_equity"].cummax())        / bt["ew_equity"].cummax()        * 100
    ew_kalman_dd  = (bt["ew_kalman_equity"] - bt["ew_kalman_equity"].cummax()) / bt["ew_kalman_equity"].cummax() * 100
    gjr_dd        = (bt["gjr_equity"]       - bt["gjr_equity"].cummax())       / bt["gjr_equity"].cummax()       * 100
    iwm_dd        = (iwm_aligned - iwm_aligned.cummax()) / iwm_aligned.cummax() * 100
    ax.fill_between(bt.index, ew_dd,        color="#2980b9", alpha=0.4, label="Equal-weight")
    ax.fill_between(bt.index, ew_kalman_dd, color="#e67e22", alpha=0.4, label="EW + Kalman gate")
    ax.fill_between(bt.index, gjr_dd,       color="#2ecc71", alpha=0.5, label="GJR-GARCH weight")
    ax.fill_between(bt.index, iwm_dd,       color="#2c3e50", alpha=0.3, label="IWM")
    ax.legend(fontsize=9)
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("%")

    ax = axes[2]
    ax.fill_between(bt.index,  bt["n_longs"],  alpha=0.7, color="#2ecc71", label="Longs")
    ax.fill_between(bt.index, -bt["n_shorts"], alpha=0.7, color="#e74c3c", label="Shorts")
    ax.axhline(0, color="black", lw=0.4)
    ax.legend(fontsize=9)
    ax.set_title("Position Count")
    ax.set_ylabel("N stocks")
    ax.set_xlabel("Date")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "cmom_backtest.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(force=False):
    print("\n── Hurst + Autocorr + CMOM ───────────────────────────────────")

    # IWM signals and classification
    iwm     = load_data(force=force)
    signals = compute_signals(iwm)
    bt      = backtest(signals)
    print_metrics(bt)
    plot_signals(signals, iwm)

    # CMOM backtest: equal-weight vs GJR-GARCH weighted + Kalman exit
    universe   = load_universe()
    gjr_vol    = compute_gjr_vol(universe)
    ukf_slopes = compute_ukf_slopes(universe)
    cmom_bt    = run_cmom_backtest(universe, gjr_vol, ukf_slopes)
    print_cmom_metrics(cmom_bt)
    plot_cmom_backtest(cmom_bt, iwm)

    print(f"  Plots → {PLOT_DIR}")
    print("─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download IWM (Russell 2000) data")
    args = parser.parse_args()
    main(force=args.force)
