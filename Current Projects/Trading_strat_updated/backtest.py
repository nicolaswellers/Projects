'''Long-only cross-sectional momentum backtest on the S&P 500 universe.

Signal: 12-1 month risk-adjusted momentum (see cmom.py), top LONG_TOP_N
names each rebalance, weighted inversely to walk-forward GJR-GARCH
volatility and scaled to a portfolio-level volatility target (see
vol_weighted_signals / the vol-target scaling below). `run_long_only`'s own
docstring lists every overlay it accepts; main.py runs the current
production config (see its own docstring) -- a market-wide realized-vol-
spike hysteresis cut (risk_layer.py). Other overlays tried during
development (Kalman trend-slope exit gate via unscent_kalman.py,
per-position stop-loss + capital halving, vol-spike tiering/continuous
scaling, GARCH/HMM/PCA-based regime detection) are still available as
`run_long_only` parameters but are documented and compared in
table/00_MASTER_eval_table.png rather than run by default; the modules
behind the ones that underperformed the champion live in research/.

Lookahead discipline
---------------------
  - Momentum ranking (cmom.py) and GJR-GARCH vol (refit at every rebalance,
    on the trailing GARCH_WINDOW window ending at t) use `universe.loc[:t]`,
    so date t's weights can only ever depend on data through t's own close.
    They set positions that take effect starting t + 1 (today's return is
    booked against yesterday's positions *before* the new ones are set).
  - The vol-target scalar for date t is computed from realized portfolio
    returns through t - 1 only (see run_long_only), so it's set before
    today's return is booked, same as the Kalman gate below.
  - The UKF slope reported for date t (see unscent_kalman.py) is the
    filter's *predicted* state, built only from data through t - 1 — it
    never sees today's own close, so it's safe to use for exits applied
    before today's return is computed: the decision was knowable before
    today even opened.
  - The stop-loss / capital-halving risk layer (risk_layer.py) is evaluated
    off each day's own close, but only takes effect starting the next day.
'''
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from GARCH import GARCHModel
from cmom import top_momentum_names
from risk_layer import RiskLayer
from config import (
    LOOKBACK, REBALANCE_FREQ, TRANSACTION_COST, STARTING_CAPITAL,
    FIXED_FEE_PER_STOCK, LONG_TOP_N,
    KALMAN_ZSCORE_WINDOW, KALMAN_ZSCORE_MIN_PERIODS, KALMAN_EXIT_THRESHOLD,
    GARCH_WINDOW, GARCH_MIN_OBS, POSITION_VOL_CAP_MULT,
    VOL_TARGET, VOL_TARGET_LOOKBACK, VOL_TARGET_SCALAR_CAP,
    VOL_SPIKE_SHORT_WINDOW, VOL_SPIKE_LONG_WINDOW,
)


# ════════════════════════════════════════════════════════════════════════════
# KALMAN EXIT / ENTRY GATING
# ════════════════════════════════════════════════════════════════════════════

def _slope_zscore(slopes: pd.Series, slope_means: pd.Series, slope_stds: pd.Series,
                   tickers) -> pd.Series:
    """(slope - rolling mean) / rolling std, each ticker normalised against
    its own recent trend-slope regime rather than its full-history level."""
    mean = slope_means.reindex(tickers)
    std  = slope_stds.reindex(tickers)
    z = (slopes.reindex(tickers) - mean) / std
    return z[std > 1e-10].dropna()


def _apply_kalman_exits(positions: pd.Series, slopes: pd.Series,
                         slope_means: pd.Series, slope_stds: pd.Series,
                         exit_threshold: float) -> pd.Series:
    positions = positions.copy()
    held = positions[positions != 0].index
    z = _slope_zscore(slopes, slope_means, slope_stds, held)
    stopped = z[z < exit_threshold].index
    positions.loc[stopped] = 0.0
    return positions


def _apply_kalman_entry_gate(new_positions: pd.Series, current_positions: pd.Series,
                              slopes: pd.Series, slope_means: pd.Series,
                              slope_stds: pd.Series, exit_threshold: float) -> pd.Series:
    """Block new entries that would be immediately exited by the Kalman filter."""
    new_positions = new_positions.copy()
    candidates = new_positions[new_positions != 0].index.difference(
        current_positions[current_positions != 0].index
    )
    z = _slope_zscore(slopes, slope_means, slope_stds, candidates)
    blocked = z[z < exit_threshold].index
    new_positions.loc[blocked] = 0.0
    return new_positions


# ════════════════════════════════════════════════════════════════════════════
# GJR-GARCH VOL WEIGHTING
# ════════════════════════════════════════════════════════════════════════════

def _walkforward_gjr_vol(prices: pd.Series) -> float:
    """GJR-GARCH(1,1,1) conditional volatility as of `prices`' last date.

    Fit on only the trailing GARCH_WINDOW observations ending at that date —
    refitting per-rebalance like this (instead of fitting once over the full
    history) is what keeps the parameter estimates themselves from being
    informed by data past the current rebalance date.
    """
    p = prices.dropna()
    if len(p) < GARCH_MIN_OBS:
        return np.nan
    p = p.iloc[-GARCH_WINDOW:]
    try:
        fit = GARCHModel(prices_close=p).apply_gjr_garch()
        return float(fit.conditional_volatility.iloc[-1])
    except Exception:
        return np.nan


def vol_weighted_signals(price_data: pd.DataFrame, lookback: int, top_n: int,
                          min_zscore: float | None = None) -> pd.Series:
    """Top-N momentum names, weighted inversely to walk-forward GJR-GARCH
    volatility, capped at POSITION_VOL_CAP_MULT x equal weight so one
    unusually low-vol name can't dominate the book. `min_zscore` (see
    cmom.top_momentum_names) can shrink the candidate set below top_n in a
    weak month -- the book size is scaled down to match, rather than
    renormalised back up to full exposure in fewer names."""
    names = top_momentum_names(price_data, lookback, top_n, min_zscore=min_zscore)
    vol = pd.Series({t: _walkforward_gjr_vol(price_data[t]) for t in names}, dtype=float)
    vol = vol.replace(0, np.nan).dropna()

    signals = pd.Series(0.0, index=price_data.columns)
    if vol.empty or len(names) == 0:
        return signals

    vol_floor = vol.quantile(0.05)
    vol = vol.clip(lower=vol_floor)
    w = 1.0 / vol
    w = w / w.sum()
    w = w.clip(upper=POSITION_VOL_CAP_MULT / top_n)
    w = w / w.sum()
    w = w * (len(names) / top_n)  # leave slots the significance filter dropped in cash

    signals.loc[w.index] = w.values
    return signals


def equal_weight_signals(price_data: pd.DataFrame, lookback: int, top_n: int,
                          min_zscore: float | None = None) -> pd.Series:
    """Top-N momentum names, each fixed at 1/top_n -- never renormalised,
    so unfilled slots (fewer than top_n ranked, or filtered out by
    `min_zscore`) are left in cash."""
    names = top_momentum_names(price_data, lookback, top_n, min_zscore=min_zscore)
    signals = pd.Series(0.0, index=price_data.columns)
    signals.loc[names] = 1.0 / top_n
    return signals


# ════════════════════════════════════════════════════════════════════════════
# BACKTEST LOOP
# ════════════════════════════════════════════════════════════════════════════

def market_vol_ratio_series(universe: pd.DataFrame, short_window: int = VOL_SPIKE_SHORT_WINDOW,
                             long_window: int = VOL_SPIKE_LONG_WINDOW
                             ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Market-wide (equal-weight universe) realized vol over `short_window`
    and `long_window`, plus the trailing `short_window` cumulative return --
    the three inputs `run_long_only` feeds into `RiskLayer.vol_spike_scalar`
    each day. Factored out so a live/paper-trading runner can read today's
    values (the last row) with the exact same definition used here, instead
    of risking a second implementation that quietly drifts from what was
    actually backtested."""
    daily_ret = universe.ffill().pct_change(fill_method=None)
    market_ret = daily_ret.mean(axis=1)
    market_vol_short = market_ret.rolling(short_window).std() * np.sqrt(252)
    market_vol_long = market_ret.rolling(long_window).std() * np.sqrt(252)
    market_ret_trailing = market_ret.rolling(short_window).sum()
    return market_vol_short, market_vol_long, market_ret_trailing


def _turnover_cost(old_pos: pd.Series, new_pos: pd.Series, portfolio_value: float) -> float:
    """Proportional + fixed-fee cost of moving from old_pos to new_pos, as a
    fraction of portfolio value."""
    diff = new_pos - old_pos
    if diff.abs().sum() < 1e-10:
        return 0.0
    turnover = diff.abs().sum()
    n_trades = int((diff.abs() > 1e-10).sum())
    return turnover * TRANSACTION_COST + n_trades * FIXED_FEE_PER_STOCK / max(portfolio_value, 1.0)


def run_long_only(universe: pd.DataFrame, stop_loss_threshold: float | None = None,
                   ukf_slopes: pd.DataFrame | None = None,
                   kalman_exit_threshold: float = KALMAN_EXIT_THRESHOLD,
                   vol_target: float | None = VOL_TARGET,
                   vol_spike_protection: bool = False,
                   vol_spike_direction: str | None = None,
                   vol_spike_multiplier: float | None = None,
                   vol_spike_continuous: bool = False,
                   vol_spike_tiers: list[tuple[float, float]] | None = None,
                   vol_spike_reset_multiplier: float | None = None,
                   vol_spike_short_window: int = VOL_SPIKE_SHORT_WINDOW,
                   vol_spike_long_window: int = VOL_SPIKE_LONG_WINDOW,
                   spike_signal: pd.Series | None = None,
                   regime_weight: pd.Series | None = None,
                   use_vol_weighting: bool = True,
                   momentum_min_zscore: float | None = None,
                   starting_capital: float = STARTING_CAPITAL,
                   top_n: int = LONG_TOP_N,
                   hedge_ret: pd.Series | None = None) -> pd.DataFrame:
    """Long-only book: top LONG_TOP_N momentum names. `use_vol_weighting`
    picks GJR-GARCH inverse-vol weighting (True) vs. plain 1/LONG_TOP_N
    equal weight (False); `vol_target` scales exposure to that annualized
    vol (None disables it, leaving raw 1x exposure). `starting_capital`
    matters beyond just rescaling the final €-value line: FIXED_FEE_PER_STOCK
    is a flat cost per trade, so it's a much bigger fraction of a smaller
    account's returns each rebalance -- CAGR/Sharpe/MaxDD are NOT scale-
    invariant once that fee is in the mix. `top_n` (defaults to LONG_TOP_N)
    trades off diversification against exactly that same fee drag: fewer
    positions means fewer trades per rebalance to pay the flat fee on --
    relevant together with `starting_capital` for a small account.
    `momentum_min_zscore`
    requires a name's momentum score to clear that many cross-sectional
    standard deviations above the rebalance's own mean before it's
    eligible (see cmom.top_momentum_names) -- a weak month can then hold
    fewer than LONG_TOP_N names, cash instead of forcing capital into
    whatever ranked highest among noise. `stop_loss_threshold`, `ukf_slopes`
    and `vol_spike_protection` are independent, optional overlays on top of
    that sizing -- pass at most one to isolate its effect. `vol_spike_direction`,
    `vol_spike_multiplier`, `vol_spike_continuous`, `vol_spike_tiers`,
    `vol_spike_reset_multiplier` and the two window sizes only matter
    together with `vol_spike_protection` (see risk_layer.py). Passing
    `spike_signal` swaps the trigger's input from the realized-vol ratio to
    that precomputed daily series (e.g. a signal from
    research/research_signals.py or research/HSMM.py) -- `vol_spike_multiplier`/
    `vol_spike_reset_multiplier`/`vol_spike_tiers` are then read as levels
    on that signal's own scale. `regime_weight` (see
    research/HSMM.py's compute_regime_weight or research_signals.py's
    market_trend_weight) is a separate, continuous exposure scalar applied
    alongside everything else, rather than a discrete on/off cut.
    `hedge_ret` (a daily return series for some "safe-haven" asset, e.g.
    gold or treasuries -- see research/safe_haven_hedge.py) changes what
    happens to the fraction of capital a stop-loss/capital-halve or
    vol-spike cut takes OUT of the momentum book: with no `hedge_ret`, that
    fraction just sits in cash (0% return) for the day; with `hedge_ret`,
    it earns that day's realized hedge-asset return instead, so the book
    stays 100% invested (momentum + hedge) rather than partly in cash
    whenever a cut is active. No effect on days with no cut, since the
    hedge fraction is exactly `1 - capital_scalar * spike_scalar`."""
    daily_ret = universe.ffill().pct_change(fill_method=None)
    close     = universe.ffill()
    dates     = universe.index
    rebalance_dates = set(dates[LOOKBACK::REBALANCE_FREQ])

    market_vol_short, market_vol_long, market_ret_trailing = market_vol_ratio_series(
        universe, vol_spike_short_window, vol_spike_long_window)

    risk = RiskLayer(stop_loss_threshold, vol_spike_protection=vol_spike_protection,
                      vol_spike_direction=vol_spike_direction, vol_spike_multiplier=vol_spike_multiplier,
                      vol_spike_continuous=vol_spike_continuous, vol_spike_tiers=vol_spike_tiers,
                      vol_spike_reset_multiplier=vol_spike_reset_multiplier)
    if ukf_slopes is not None:
        rolling     = ukf_slopes.rolling(KALMAN_ZSCORE_WINDOW, min_periods=KALMAN_ZSCORE_MIN_PERIODS)
        slope_means = rolling.mean()
        slope_stds  = rolling.std()
    else:
        slope_means = slope_stds = None
    positions       = pd.Series(0.0, index=universe.columns)
    raw_ret_history = []  # pre-vol-target-scalar returns, for the target's own realized-vol estimate

    results = []
    portfolio_value = starting_capital

    for i, date in enumerate(dates):
        if i < LOOKBACK:
            results.append({"date": date, "ret": 0.0, "value": portfolio_value, "n_positions": 0})
            continue

        cost_frac    = 0.0
        todays_close = close.loc[date]

        # Kalman exit off yesterday's predicted slope -> no lookahead (see
        # module docstring); applied before today's return is booked.
        if ukf_slopes is not None and date in ukf_slopes.index:
            exited = _apply_kalman_exits(positions, ukf_slopes.loc[date],
                                          slope_means.loc[date], slope_stds.loc[date],
                                          kalman_exit_threshold)
            cost_frac += _turnover_cost(positions, exited, portfolio_value)
            positions = exited

        # capital + vol-spike scalars for today, decided off yesterday's
        # close/vol only -> no lookahead
        capital_scalar = risk.capital_scalar(positions, close.loc[dates[i - 1]])
        if spike_signal is not None:
            spike_scalar = risk.vol_spike_scalar(np.nan, np.nan, market_ret_trailing.loc[dates[i - 1]],
                                                   signal=spike_signal.loc[dates[i - 1]])
        else:
            spike_scalar = risk.vol_spike_scalar(market_vol_short.loc[dates[i - 1]],
                                                   market_vol_long.loc[dates[i - 1]],
                                                   market_ret_trailing.loc[dates[i - 1]])
        day_ret   = daily_ret.loc[date].fillna(0)
        effective_exposure = capital_scalar * spike_scalar
        raw_gross = effective_exposure * (positions * day_ret).sum()
        if hedge_ret is not None:
            # whatever a stop-loss/capital-halve or vol-spike cut takes out
            # of the momentum book goes into the hedge asset instead of
            # sitting in cash -- see the "hedge_ret" docstring paragraph above
            raw_gross += (1.0 - effective_exposure) * hedge_ret.get(date, 0.0)

        # vol-target scalar for today, from realized vol of raw_gross through
        # yesterday only (raw_gross itself is appended *after* this) -> no lookahead
        if vol_target is None:
            vol_scalar = 1.0
        elif len(raw_ret_history) >= VOL_TARGET_LOOKBACK:
            realized_vol = float(np.std(raw_ret_history[-VOL_TARGET_LOOKBACK:])) * np.sqrt(252)
            vol_scalar = vol_target / realized_vol if realized_vol > 1e-10 else VOL_TARGET_SCALAR_CAP
            vol_scalar = min(vol_scalar, VOL_TARGET_SCALAR_CAP)
        else:
            vol_scalar = 1.0

        gross = raw_gross * vol_scalar
        if regime_weight is not None:
            rw = regime_weight.loc[dates[i - 1]]
            gross *= rw if not np.isnan(rw) else 1.0
        raw_ret_history.append(raw_gross)

        if date in rebalance_dates:
            signal_fn = vol_weighted_signals if use_vol_weighting else equal_weight_signals
            new_positions = signal_fn(universe.loc[:date], LOOKBACK, top_n,
                                       min_zscore=momentum_min_zscore)
            if ukf_slopes is not None and date in ukf_slopes.index:
                new_positions = _apply_kalman_entry_gate(
                    new_positions, positions, ukf_slopes.loc[date],
                    slope_means.loc[date], slope_stds.loc[date], kalman_exit_threshold,
                )
            cost_frac += _turnover_cost(positions, new_positions, portfolio_value)
            positions = new_positions
            risk.on_new_positions(positions, todays_close)

        # stop-loss off today's close -> takes effect starting tomorrow
        pre_stop  = positions
        positions = risk.apply_stop_loss(positions, todays_close)
        cost_frac += _turnover_cost(pre_stop, positions, portfolio_value)

        net = gross - cost_frac
        portfolio_value *= (1 + net)
        results.append({"date": date, "ret": net, "value": portfolio_value,
                         "n_positions": int((positions > 0).sum()), "vol_scalar": vol_scalar})

    bt = pd.DataFrame(results).set_index("date")
    bt["equity"] = bt["value"] / starting_capital
    return bt


def run_param_sweep(universe: pd.DataFrame, param: str, values: list,
                     baseline_bt: pd.DataFrame | None = None,
                     name_fmt=None, **kwargs) -> dict[str, pd.DataFrame]:
    """Run `run_long_only` once per value of a single keyword argument
    (e.g. param="stop_loss_threshold", values=STOP_LOSS_LEVELS), against a
    shared baseline -- the one generic sweep every param-comparison in this
    project reduces to. `name_fmt(value)` overrides the default label."""
    results = {"baseline": baseline_bt if baseline_bt is not None else run_long_only(universe, **kwargs)}
    for v in values:
        name = name_fmt(v) if name_fmt else f"{param}_{v}"
        print(f"  [long-only] running {name} ...")
        results[name] = run_long_only(universe, **{param: v}, **kwargs)
    return results


# ════════════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════════════

def raw_metrics(ret: pd.Series, eq: pd.Series, benchmark_ret: pd.Series | None = None) -> dict:
    """CAGR/Sharpe/MaxDD/etc as raw floats (NaN where undefined) -- the
    numbers themselves, not the formatted strings `_metrics`/
    `print_long_only_metrics` print. Meant for building eval-table rows
    (see plot_grouped_eval_table) where the caller wants to compare/
    highlight actual values rather than parse formatted text back apart."""
    n = len(eq)
    sharpe  = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 1e-12 else 0.0
    cagr    = eq.iloc[-1] ** (252 / n) - 1
    max_dd  = ((eq - eq.cummax()) / eq.cummax()).min()

    gross_profit  = ret[ret > 0].sum()
    gross_loss    = -ret[ret < 0].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else np.nan

    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-12 else np.nan

    info_ratio = np.nan
    if benchmark_ret is not None:
        active = ret - benchmark_ret.reindex(ret.index).fillna(0)
        if active.std() > 1e-12:
            info_ratio = active.mean() / active.std() * np.sqrt(252)

    return {
        "cagr": cagr * 100, "sharpe": sharpe, "mdd": max_dd * 100,
        "total": (eq.iloc[-1] - 1) * 100, "pf": profit_factor,
        "calmar": calmar, "ir": info_ratio, "final": None,  # caller fills "final"/"name"/"cpcv" in €
    }


def _metrics(ret: pd.Series, eq: pd.Series, benchmark_ret: pd.Series | None = None) -> dict:
    m = raw_metrics(ret, eq, benchmark_ret)
    return {
        "CAGR":               f"{m['cagr']:+.2f}%",
        "Sharpe":             f"{m['sharpe']:.3f}",
        "Max drawdown":       f"{m['mdd']:.2f}%",
        "Total return":       f"{m['total']:+.2f}%",
        "Profit factor":      f"{m['pf']:.3f}" if not np.isnan(m['pf']) else "n/a",
        "Calmar ratio":       f"{m['calmar']:.3f}" if not np.isnan(m['calmar']) else "n/a",
        "Information ratio":  f"{m['ir']:.3f}" if not np.isnan(m['ir']) else "n/a",
    }


def print_long_only_metrics(results: dict[str, pd.DataFrame], book_desc: str | None = None):
    """`book_desc` should describe the actual sizing used for this
    particular set of runs (e.g. "equal-weight, no vol target" vs.
    "GJR-GARCH inverse-vol weighted, {VOL_TARGET:.0%} vol target") --
    it's on the caller to get this right, since a dict of pre-computed
    backtests carries no record of which `run_long_only` kwargs produced
    them. Omit it to skip the line entirely rather than guess wrong."""
    print("\n" + "=" * 58)
    print("  LONG-ONLY BOOK — overlay comparison")
    print("=" * 58)
    print(f"  Max positions: {LONG_TOP_N}" + (f", {book_desc}" if book_desc else ""))

    baseline_ret = results["baseline"]["ret"].dropna() if "baseline" in results else None
    for name, bt in results.items():
        benchmark = baseline_ret if (baseline_ret is not None and name != "baseline") else None
        m = _metrics(bt["ret"].dropna(), bt["equity"].dropna(), benchmark_ret=benchmark)
        print(f"\n  {name}:")
        for k, v in m.items():
            print(f"    {k:<18}: {v}")
        print(f"    {'Final value':<18}: €{bt['value'].iloc[-1]:,.0f}")
    print("=" * 58 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════════════

def _positions_panel(ax, curves: dict[str, pd.DataFrame]):
    for name, bt in curves.items():
        ax.plot(bt.index, bt["n_positions"], lw=1.0, label=name, alpha=0.9)
    ax.axhline(LONG_TOP_N, color="black", lw=0.6, ls="--", alpha=0.5, label=f"max ({LONG_TOP_N})")
    ax.legend(fontsize=9)
    ax.set_title("Positions Held")
    ax.set_ylabel("N positions")
    ax.set_xlabel("Date")


def plot_comparison(results: dict[str, pd.DataFrame], plot_dir: str, filename: str, title: str):
    """Equity curves (top) + positions held (bottom) for an arbitrary set
    of named backtest runs -- the shared 2-panel layout behind every
    overlay-comparison plot in this project, parameterized instead of
    duplicated per comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0]
    for name, bt in results.items():
        ax.plot(bt.index, bt["equity"], lw=1.2, label=name, alpha=0.9)
    ax.legend(fontsize=9)
    ax.set_title(f"Equity Curves  (starting capital €{STARTING_CAPITAL:,.0f})")
    ax.set_ylabel("Portfolio value (normalised)")

    _positions_panel(axes[1], results)

    plt.tight_layout()
    path = os.path.join(plot_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


# fixed color per strategy identity, reused across annual-return charts so
# the same name always reads as the same color
_STRATEGY_COLORS = {
    "baseline":   "#5b6b7a",  # neutral slate
    "kalman":     "#e67e22",  # orange
    "stop":       "#3498db",  # blue
    "vol_spike":  "#2ecc71",  # green
    "vol_target": "#9b59b6",  # purple
}


def _color_for(name: str, fallback_cycle: list[str], idx: int) -> str:
    for key, color in _STRATEGY_COLORS.items():
        if key in name:
            return color
    return fallback_cycle[idx % len(fallback_cycle)]


def annual_returns(bt: pd.DataFrame) -> pd.Series:
    """Compounded return per calendar year from a backtest's daily `ret`."""
    ret = bt["ret"].dropna()
    return ret.groupby(ret.index.year).apply(lambda s: (1 + s).prod() - 1)


def plot_annual_returns(results: dict[str, pd.DataFrame], plot_dir: str):
    """Bar chart of compounded annual return per calendar year, one bar per
    strategy per year -- shows which years each overlay actually helped or
    hurt, rather than just the final cumulative number."""
    annual = {name: annual_returns(bt) for name, bt in results.items()}
    years  = sorted(set().union(*(s.index for s in annual.values())))

    n = len(results)
    width = 0.8 / n
    x = np.arange(len(years))

    fig, ax = plt.subplots(figsize=(16, 7))
    fallback_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, s) in enumerate(annual.items()):
        vals  = [s.get(y, np.nan) * 100 for y in years]
        color = _color_for(name, fallback_cycle, i)
        ax.bar(x + i * width - (n - 1) * width / 2, vals, width=width * 0.9, label=name, color=color)

    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.legend(fontsize=9)
    ax.set_title("Long-Only CMOM — Annual Returns by Strategy  |  S&P 500")
    ax.set_ylabel("Calendar-year return (%)")
    ax.set_xlabel("Year")

    plt.tight_layout()
    path = os.path.join(plot_dir, "long_only_annual_returns.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_grouped_eval_table(sections: list[tuple[str, list[dict]]], out_path: str):
    """Render a metrics-comparison table as a PNG: a list of (section_title,
    rows) groups -- each row a dict with keys
    name/cagr/sharpe/mdd/total/pf/calmar/ir/final/cpcv/cpcv_n -- rendered as
    labeled blocks within a single image, with the best value per column
    highlighted globally across every row in every section. A single
    section covers the plain (ungrouped) case."""
    cols = [
        ("name",   "Strategy",       "left"),
        ("cagr",   "CAGR",           "pct"),
        ("sharpe", "Sharpe",         "num"),
        ("mdd",    "Max DD",         "pct"),
        ("total",  "Total Return",   "pct"),
        ("pf",     "Profit Factor",  "num"),
        ("calmar", "Calmar",         "num"),
        ("ir",     "Info Ratio",     "num"),
        ("final",  "Final Value",    "eur"),
        ("cpcv",   "CPCV Win Rate",  "cpcv"),
    ]

    def fmt(key, kind, r):
        v = r.get(key)
        if kind == "left":
            return str(v)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        if kind == "pct":
            return f"{v:+.2f}%"
        if kind == "eur":
            return f"€{v:,.0f}"
        if kind == "cpcv":
            return f"{v}/{r['cpcv_n']}"
        return f"{v:.3f}"

    all_rows = [r for _, rows in sections for r in rows]
    best = {}
    for key, _, kind in cols:
        if kind in ("pct", "num", "eur", "cpcv"):
            vals = [r.get(key) for r in all_rows
                    if r.get(key) is not None and not (isinstance(r.get(key), float) and np.isnan(r.get(key)))]
            if vals:
                best[key] = max(vals)

    display = []
    for title, rows in sections:
        display.append(("section", title))
        for r in rows:
            display.append(("data", r))

    n_rows, n_cols = len(display), len(cols)
    fig_w = 3.0 + n_cols * 1.55
    fig_h = 1.0 + n_rows * 0.40
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    header_bg  = "#24344D"
    header_ink = "#FFFFFF"
    row_bg     = ["#FFFFFF", "#F7F8FA"]
    border     = "#E1E4E9"
    ink        = "#1B2432"
    ink_muted  = "#6B7280"
    good_bg    = "#E1F4EA"
    good_ink   = "#157A4F"
    section_bg = "#D3DBE8"
    section_ink = "#182338"

    cell_text = [[c[1] for c in cols]]
    for kind, item in display:
        if kind == "section":
            cell_text.append([item] + [""] * (n_cols - 1))
        else:
            cell_text.append([fmt(c[0], c[2], item) for c in cols])

    col_widths = [6.0 if c[0] == "name" else 1.3 for c in cols]
    table = ax.table(cellText=cell_text, cellLoc="right", loc="center",
                      colWidths=[w / sum(col_widths) for w in col_widths])
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1, 1.85)

    row_kind = {0: ("header", None)}
    for i, (kind, item) in enumerate(display, start=1):
        row_kind[i] = (kind, item)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(border)
        key, _, kind = cols[col]
        if row == 0:
            cell.set_facecolor(header_bg)
            cell.get_text().set_color(header_ink)
            cell.get_text().set_fontweight("bold")
            cell.set_text_props(ha="left" if key == "name" else "right")
            continue

        rowkind, item = row_kind[row]
        if rowkind == "section":
            cell.set_facecolor(section_bg)
            cell.get_text().set_color(section_ink)
            cell.get_text().set_fontweight("bold")
            cell.set_text_props(ha="left" if col == 0 else "right")
            continue

        r = item
        v = r.get(key)
        cell.set_text_props(ha="left" if key == "name" else "right")
        if key == "name":
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color(ink)
            cell.set_facecolor(row_bg[row % 2])
        elif v is not None and key in best and v == best[key]:
            cell.set_facecolor(good_bg)
            cell.get_text().set_color(good_ink)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(row_bg[row % 2])
            cell.get_text().set_color(ink if v is not None else ink_muted)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {out_path}")
