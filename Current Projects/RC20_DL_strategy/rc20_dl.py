'''RC20-DL: RC20 (5 names) + the drawdown-losers trigger. Self-contained: numpy, pandas and the cached data in DATA_DIR.

    python rc20_dl.py backtest              # all three versions, US + Europe -> output/ CSVs + summary
    python rc20_dl.py holdings us           # what to hold from the latest month end, for MODE (also: eu)

THE TRIGGER (checked at every month-end close)
  index drawdown = 1 - index TR / its highest close of the last 252 trading days (S&P 500 TR / STOXX 600 TR)
  drawdown >= 20%  -> the 5 biggest LOSERS: lowest raw total return over the last 21 trading days, from the 100 worst,
                      greedy correlation penalty 0.5 (each next pick: loss score - 0.5 x standardised avg correlation)
  drawdown <  20%  -> trigger off (HOLD_EXTRA > 0 keeps the last losers that many more month ends; tested: 0 is best)
  trades at the next close; held one month, then checked again (still >= 20% -> fresh losers)

VERSIONS (MODE)
  "switch"   100% RC20 (5 names, both gold layers) normally, 100% the 5 losers (no gold) while the trigger is on
  "mix"      70% RC20 + 30% a no-gold sleeve: the 5 losers while the trigger is on, RC20's 5 names otherwise
  "trigger"  100% the 5 losers while the trigger is on, cash otherwise (cash earns nothing in the backtest)

RC20 (the normal book), as in ../RC20_strategy/rc20.py but TOP_N = 5 and EUR 1 fee:
  universe   index members on the month-end close, priced, >= 95% of the last 254 closes, gaps <= 5 days carried forward
  signal     residual momentum on daily returns t-252..t-21: OLS on the members' equal-weight return, alpha / residual sd
  selection  from the 100 best z-scores, greedy correlation penalty 0.5
  book       trade the close after the signal; drop-outs sold, kept names untouched, freed cash split equally over new names
  gold 1     50% gold while 21d / 126d equal-weight market vol >= 1.2 (off below 1.0), previous close
  gold 2     month end: 21d market vol >= 22% -> 50% gold (off below 16%), re-split at month ends
  costs      10bp + EUR 1 per trade (100k account); European buy taxes; switch costs charged explicitly
'''
import os
import sys
import numpy as np
import pandas as pd

# ── settings ─────────────────────────────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("RC20_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data"))
MODE = "switch"                               # "switch", "mix" or "trigger" (used by `holdings`; backtest runs all three)
TOP_N, POOL, LAMBDA = 5, 100, 0.5             # names held, shortlist size, correlation penalty (RC20 picks)
DD_TRIGGER, DD_WINDOW = 0.20, 252             # trigger: index this far below its high of the last DD_WINDOW closes
LOSER_DAYS, LOSER_LAMBDA = 21, 0.5            # losers: raw return window, correlation penalty (0 = plain biggest losers)
HOLD_EXTRA = 0                                # month ends the last losers are kept after the index recovers
MIX_SLEEVE = 0.30                             # "mix": share of the account in the trigger sleeve
LOOKBACK, SKIP = 252, 21                      # 12-1 momentum window
MIN_COVERAGE, GAP_FILL = 0.95, 5              # eligibility: share of closes present, longest gap carried forward
SPIKE_SHORT, SPIKE_LONG, SPIKE_ON, SPIKE_OFF, SPIKE_GOLD = 21, 126, 1.2, 1.0, 0.5
TURB_WINDOW, TURB_ON, TURB_OFF, TURB_GOLD = 21, 0.22, 0.16, 0.5
CAPITAL, BPS, FEE = 100_000.0, 0.001, 1.0
MARKETS = {"us": "1996-01-01", "eu": "2000-01-01"}
EU_TAXES = {"L": [("1900-01-01", 0.005)], "I": [("1900-01-01", 0.01)],
            "PA": [("2012-08-01", 0.002), ("2017-01-01", 0.003), ("2025-04-01", 0.004)],
            "MI": [("2013-03-01", 0.001)], "MC": [("2021-01-16", 0.002)]}


# ── data ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def _path(name):
    return os.path.join(DATA_DIR, name)


def _daily_returns(level: pd.Series, dates) -> pd.Series:
    return level.reindex(level.index.union(dates)).ffill().reindex(dates).pct_change().fillna(0)


def load(market: str) -> dict:
    '''prices, membership, equal-weight market return, gold and index returns, and buy taxes for one market.'''
    if market == "us":
        prices = pd.read_parquet(_path("spx_pit_prices_1993_2026.parquet"))
        events = pd.read_parquet(_path("spx_joiners_leavers.parquet")).sort_values("date")
    else:
        prices = pd.read_parquet(_path("stoxx600_pit_prices_1997_2026.parquet"))
        events = pd.read_parquet(_path("stoxx600_joiners_leavers.parquet")).sort_values("date")
    dates = prices.index
    member = pd.DataFrame(False, index=dates, columns=prices.columns)
    for ric, g in events.groupby("ric"):
        if ric not in member.columns:
            continue
        start = None
        for _, r in g.iterrows():
            if r.change == "Joiner":
                start = r.date if (market == "us" or start is None) else start
            elif start is not None:
                member.loc[(dates >= start) & (dates < r.date), ric] = True
                start = None
        if start is not None:
            member.loc[dates >= start, ric] = True
    ret = prices.ffill().pct_change(fill_method=None)
    mkt = ret.where(member.shift(1, fill_value=False)).mean(axis=1)
    pre = dates <= events.date.min()
    mkt[pre] = ret[pre].mean(axis=1)
    last = prices.apply(lambda c: c.last_valid_index())
    after_end = pd.DataFrame(dates.to_numpy()[:, None] > last.to_numpy()[None, :], index=dates, columns=prices.columns)
    if market == "us":
        gold = _daily_returns(pd.read_parquet(_path("gold_spot_1993_2026.parquet"))["gold_spot"], dates)
        index = _daily_returns(pd.read_parquet(_path("spy_1993_2026.parquet"))["SPY"], dates)
        tax = None
    else:
        b = pd.read_parquet(_path("stoxx600_benchmarks_1997_2026.parquet"))
        gold, index = _daily_returns(b["gold_eur"], dates), _daily_returns(b[".STOXXR"], dates)
        suffix = pd.Series(prices.columns, index=prices.columns).str.split("^", regex=False).str[0].str.extract(r"\.([A-Z]+)$")[0]
        tax = pd.DataFrame(0.0, index=dates, columns=prices.columns)
        for sfx, steps in EU_TAXES.items():
            for start, rate in steps:
                tax.loc[tax.index >= pd.Timestamp(start), suffix.index[suffix == sfx]] = rate
    return {"prices": prices, "member": member, "ret": ret, "mkt": mkt.fillna(0), "after_end": after_end,
            "gold": gold, "index": index, "tax": tax}


# ── signal and selection ─────────────────────────────────────────────────────────────────────────────────────────
def month_ends(dates) -> np.ndarray:
    return pd.Series(np.arange(len(dates)), index=dates).groupby(dates.to_period("M")).max().to_numpy()


def clean_window(w: pd.DataFrame) -> pd.DataFrame:
    w = w.loc[:, w.notna().mean() >= MIN_COVERAGE].ffill(limit=GAP_FILL)
    return w.loc[:, w.iloc[1:].notna().all()]


def residual_momentum(px: np.ndarray, mkt: np.ndarray) -> np.ndarray:
    '''px: LOOKBACK + 1 closes (t-252 .. t) per name; mkt: market return on the same rows. alpha / residual sd on t-252..t-21.'''
    w = px[: LOOKBACK + 1 - SKIP]
    r = w[1:] / w[:-1] - 1
    m = mkt[1: LOOKBACK + 1 - SKIP]
    mean = r.mean(0)
    mc = m - m.mean()
    beta = (mc[:, None] * (r - mean)).sum(0) / (mc ** 2).sum()
    alpha = mean - beta * m.mean()
    resid = r - alpha - beta * m[:, None]
    return alpha / np.maximum(resid.std(0, ddof=1), 1e-12)


def select(score: pd.Series, daily: pd.DataFrame, lam: float = None) -> list:
    '''Greedy correlation-penalised pick of TOP_N names from the POOL best z-scores (lam: penalty, default LAMBDA).'''
    lam = LAMBDA if lam is None else lam
    s = score.replace([np.inf, -np.inf], np.nan).dropna()
    z = ((s - s.mean()) / s.std()).nlargest(POOL)
    C = np.corrcoef(daily[z.index].fillna(0).to_numpy(), rowvar=False)
    zs = z.to_numpy()
    chosen = [int(np.argmax(zs))]
    left = [i for i in range(len(zs)) if i != chosen[0]]
    while len(chosen) < TOP_N and left:
        avg = C[np.ix_(left, chosen)].mean(axis=1)
        pen = (avg - avg.mean()) / avg.std() if avg.std() > 0 else np.zeros_like(avg)
        chosen.append(left.pop(int(np.argmax(zs[left] - lam * pen))))
    return list(z.index[chosen])


def window_at(d: dict, t: int) -> pd.DataFrame:
    '''Clean closes t-252 .. t of the eligible members at month-end row t.'''
    P = d["prices"]
    ok = d["member"].iloc[t] & P.iloc[t].notna() & ~d["after_end"].iloc[t]
    return clean_window(P.loc[:, ok].iloc[t - LOOKBACK - 1: t + 1]).iloc[1:]


def holdings_at(d: dict, t: int) -> list:
    '''RC20's TOP_N names at month-end row t.'''
    win = window_at(d, t)
    score = pd.Series(residual_momentum(win.to_numpy(float), d["mkt"].to_numpy()[t - LOOKBACK: t + 1]), index=win.columns)
    return select(score, win.pct_change().iloc[1:])


def losers_at(d: dict, t: int) -> list:
    '''The TOP_N biggest raw losers over the last LOSER_DAYS closes, correlation-penalised (LOSER_LAMBDA).'''
    win = window_at(d, t)
    daily = win.pct_change().iloc[1:]
    total = (1 + daily.iloc[-LOSER_DAYS:]).prod() - 1
    return select(-total, daily, LOSER_LAMBDA)


def target_holdings(d: dict, start: str) -> dict:
    '''{month-end row: [RC20 names]} from START (with a 40-day lead) to the latest month end.'''
    dates = d["prices"].index
    first = pd.Timestamp(start) - pd.Timedelta(days=40)
    return {int(t): holdings_at(d, t) for t in month_ends(dates) if t >= LOOKBACK + 2 and dates[t] >= first}


# ── the trigger ──────────────────────────────────────────────────────────────────────────────────────────────────
def index_drawdown(d: dict) -> pd.Series:
    '''1 - index TR / its highest close over the last DD_WINDOW trading days.'''
    level = (1 + d["index"]).cumprod()
    return 1 - level / level.rolling(DD_WINDOW, min_periods=1).max()


def trigger_plan(d: dict, start: str) -> tuple[dict, dict]:
    '''{month-end row: losers or []}, {month-end row: trigger on?}. On when the index is >= DD_TRIGGER below its 1-year
    high; after it recovers the last losers are kept HOLD_EXTRA more month ends; a new >= DD_TRIGGER month end restarts.'''
    dates = d["prices"].index
    dd = index_drawdown(d).to_numpy()
    first = pd.Timestamp(start) - pd.Timedelta(days=40)
    picks, on, last, left = {}, {}, [], 0
    for t in month_ends(dates):
        t = int(t)
        if t < LOOKBACK + 2 or dates[t] < first:
            continue
        if dd[t] >= DD_TRIGGER:
            last, left = losers_at(d, t), HOLD_EXTRA
            picks[t], on[t] = last, True
        elif last and left > 0:
            left -= 1
            picks[t], on[t] = last, True
        else:
            last = []
            picks[t], on[t] = [], False
    return picks, on


# ── book and gold layers ─────────────────────────────────────────────────────────────────────────────────────────
def stock_book(d: dict, picks: dict, start: str) -> pd.Series:
    '''Monthly book: trade the close after each signal; sell drop-outs, keep the rest, split freed cash over new names.
    An empty list sells everything (cash).'''
    dates, cols = d["prices"].index, d["prices"].columns
    R = d["ret"].fillna(0).to_numpy(float)
    TAX = None if d["tax"] is None else d["tax"].to_numpy(float)
    pos = {t: cols.get_indexer(names) for t, names in picks.items()}
    t0 = dates.searchsorted(pd.Timestamp(start)) - 1
    h, cash, v_prev = np.zeros(len(cols)), CAPITAL, CAPITAL
    out, pending = np.zeros(len(dates)), None
    for t in range(t0, len(dates)):
        h *= 1 + R[t]
        if pending is not None:
            keep = np.zeros(len(h), bool)
            keep[pending] = True
            sold = (h > 0) & ~keep
            proceeds = h[sold].sum()
            cash += proceeds - (BPS * proceeds + FEE * sold.sum())
            h[sold] = 0
            new = keep & (h == 0)
            k = new.sum()
            if k:
                tax = TAX[t, new].mean() if TAX is not None else 0.0
                spend = max(cash - FEE * k, 0) / (1 + BPS + tax)
                h[new] = spend / k
                cash -= spend * (1 + BPS + tax) + FEE * k
            pending = None
        v = h.sum() + cash
        if v <= 0.05 * CAPITAL:
            out[t] = max(v, 0) / v_prev - 1
            break
        if t > t0:
            out[t] = v / v_prev - 1
        v_prev = v
        if t in pos:
            pending = pos[t]
    return pd.Series(out, index=dates)[start:]


def vol_spike_state(mkt: pd.Series, idx) -> pd.Series:
    '''Share in stocks for layer 1: 1 normally, 1 - SPIKE_GOLD while the 21d / 126d vol ratio says so (previous close).'''
    ratio = (mkt.rolling(SPIKE_SHORT).std() / mkt.rolling(SPIKE_LONG).std()).shift(1).reindex(idx).to_numpy()
    on, s = False, np.ones(len(idx))
    for t, r in enumerate(ratio):
        if not np.isnan(r):
            on = r >= SPIKE_OFF if on else r >= SPIKE_ON
        s[t] = 1 - SPIKE_GOLD if on else 1.0
    return pd.Series(s, index=idx)


def turbulent_state(mkt: pd.Series, idx) -> np.ndarray:
    vol = (mkt.rolling(TURB_WINDOW).std() * np.sqrt(252)).reindex(idx).to_numpy()
    s, on = pd.Series(np.nan, index=idx), False
    for t in month_ends(idx):
        if not np.isnan(vol[t]):
            on = vol[t] >= TURB_OFF if on else vol[t] >= TURB_ON
        s.iloc[t] = float(on)
    return s.ffill().fillna(0).astype(bool).to_numpy()


def blend(a: pd.Series, b: pd.Series, w_a: float) -> pd.Series:
    '''w_a / 1 - w_a of two sleeves, re-split at month ends (BPS on the amount moved).'''
    idx = a.index
    month_end = set(month_ends(idx).tolist())
    x, y = w_a, 1 - w_a
    out, ra, rb = np.zeros(len(idx)), a.to_numpy(), b.reindex(idx).fillna(0).to_numpy()
    for t in range(len(idx)):
        x, y = x * (1 + ra[t]), y * (1 + rb[t])
        V = x + y
        if t in month_end:
            V -= BPS * abs(x - w_a * V)
            x, y = w_a * V, (1 - w_a) * V
        out[t] = V - 1
        x, y = x / V, y / V
    return pd.Series(out, index=idx)


def with_gold(d: dict, stocks: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx = stocks.index
    gold = d["gold"].reindex(idx).fillna(0)
    s = vol_spike_state(d["mkt"], idx)
    value = CAPITAL * (1 + stocks).cumprod().shift(1).fillna(1)
    moved = s.diff().abs().fillna(0)
    layer1 = s * stocks + (1 - s) * gold - (moved * BPS * 2 + (moved > 0) * (TOP_N + 1) * FEE / value)
    w = np.where(turbulent_state(d["mkt"], idx), 1 - TURB_GOLD, 1.0)
    month_end = set(month_ends(idx).tolist())
    a, b = w[0], 1 - w[0]
    out, r1, rg = np.zeros(len(idx)), layer1.to_numpy(), gold.to_numpy()
    for t in range(len(idx)):                                   # month-end re-split between layer 1 and gold
        a, b = a * (1 + r1[t]), b * (1 + rg[t])
        V = a + b
        if t in month_end:
            V -= BPS * abs(a - w[t] * V)
            a, b = w[t] * V, (1 - w[t]) * V
        out[t] = V - 1
        a, b = a / V, b / V
    gold_weight = pd.Series(w * (1 - s.to_numpy()) + (1 - w), index=idx)
    return pd.Series(out, index=idx), gold_weight


# ── the three versions ───────────────────────────────────────────────────────────────────────────────────────────
def run(market: str = "us", d: dict | None = None) -> pd.DataFrame:
    '''Daily returns of the three versions, RC20 (5) and the index, plus the flag (1 = losers held that day).'''
    start = MARKETS[market]
    d = d or load(market)
    dates = d["prices"].index
    rc_picks = target_holdings(d, start)
    loser_picks, on = trigger_plan(d, start)
    stocks = stock_book(d, rc_picks, start)
    idx = stocks.index
    rc20, gold_weight = with_gold(d, stocks)

    # losers are bought at the close after the signal (t+1), so they earn the returns from t+2 on
    flag = pd.Series(np.nan, index=dates)
    for t, a in on.items():
        if t + 2 < len(dates):
            flag.iloc[t + 2] = float(a)
    flag = flag.ffill().fillna(0).reindex(idx).astype(bool)

    # trigger alone: losers or cash
    trigger = stock_book(d, loser_picks, start)

    # mix: 70% RC20 + 30% sleeve holding the losers when on, RC20's names otherwise (no gold)
    sleeve = stock_book(d, {t: (loser_picks[t] if on.get(t) else names) for t, names in rc_picks.items()}, start)
    mix = blend(rc20, sleeve, 1 - MIX_SLEEVE)

    # switch: RC20 returns, losers' returns while held; on entry RC20 is sold (BPS + fees), on exit re-bought
    # (BPS + fees + average member buy tax); the trigger book already pays for buying and selling the losers
    tax = (d["tax"].where(d["member"]).mean(axis=1).fillna(0) if d["tax"] is not None
           else pd.Series(0.0, index=dates)).reindex(idx)
    r_rc, r_tr, fl, tx = rc20.to_numpy(), trigger.reindex(idx).fillna(0).to_numpy(), flag.to_numpy(), tax.to_numpy()
    sw, eq = np.zeros(len(idx)), 1.0
    for i in range(len(idx)):
        r = r_tr[i] if fl[i] else r_rc[i]
        fee = TOP_N * FEE / (CAPITAL * eq)
        if i + 1 < len(idx) and fl[i + 1] and not fl[i]:
            r -= BPS + fee
        if i > 0 and fl[i - 1] and not fl[i]:
            r -= BPS + fee + tx[i]
        sw[i] = r
        eq *= 1 + r
    return pd.DataFrame({"switch": sw, "mix": mix.to_numpy(), "trigger": r_tr, "RC20 (5)": rc20.to_numpy(),
                         "index": d["index"].reindex(idx).fillna(0).to_numpy(), "losers held": fl.astype(int),
                         "RC20 gold weight": gold_weight.to_numpy()}, index=idx)


# ── command line ─────────────────────────────────────────────────────────────────────────────────────────────────
def summary(r: pd.Series) -> str:
    eq = (1 + r).cumprod()
    return (f"Sharpe {r.mean() / r.std() * np.sqrt(252):.2f}, CAGR {eq.iloc[-1] ** (252 / len(r)) - 1:+.1%}, "
            f"max drawdown {(eq / eq.cummax() - 1).min():.0%}")


def holdings(market: str):
    d = load(market)
    dates = d["prices"].index
    t = int(month_ends(dates)[-1])
    picks, on = trigger_plan(d, str(dates[max(t - 400, 0)].date()))     # last ~18 months is enough for HOLD_EXTRA
    active = on.get(t, False)
    idx = dates[: t + 1]
    rc = holdings_at(d, t)
    s = vol_spike_state(d["mkt"], idx).iloc[-1]
    w = 1 - TURB_GOLD if turbulent_state(d["mkt"], idx)[-1] else 1.0
    gw = w * (1 - s) + (1 - w)
    print(f"{market.upper()} at the {dates[t].date()} close (trade at the next close), MODE = {MODE}")
    print(f"  index is {index_drawdown(d).iloc[t]:.1%} below its 1-year high -> trigger {'ON' if active else 'off'} "
          f"(fires at {DD_TRIGGER:.0%})")
    if MODE == "switch":
        if active:
            print("  hold 100% in the 5 losers, equal weight: " + ", ".join(picks[t]))
        else:
            print(f"  hold RC20: gold {gw:.0%}, stocks {1 - gw:.0%} split over " + ", ".join(rc))
    elif MODE == "mix":
        print(f"  {1 - MIX_SLEEVE:.0%} RC20: gold {gw:.0%} of it, stocks in " + ", ".join(rc))
        print(f"  {MIX_SLEEVE:.0%} sleeve, no gold: " + ", ".join(picks[t] if active else rc)
              + (" (losers)" if active else " (RC20 names)"))
    else:
        print("  hold 100% in: " + ", ".join(picks[t]) if active else "  hold cash")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    if cmd == "holdings":
        holdings(sys.argv[2] if len(sys.argv) > 2 else "us")
        return
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    for market in MARKETS:
        R = run(market)
        R.to_csv(os.path.join(out_dir, f"daily_returns_{market}.csv"))
        print(f"{market.upper()} {R.index[0].date()} -> {R.index[-1].date()}, losers held {R['losers held'].mean():.0%} of days")
        for c in ("switch", "mix", "trigger", "RC20 (5)", "index"):
            print(f"   {c:9s} {summary(R[c])}")


if __name__ == "__main__":
    main()
