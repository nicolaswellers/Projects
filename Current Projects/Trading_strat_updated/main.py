'''Long-only cross-sectional momentum backtest on the S&P 500 universe --
production run.

Signal: 12-1 month risk-adjusted momentum (cmom.py), top LONG_TOP_N names,
EQUAL-weighted (1/LONG_TOP_N each), with a market-wide realized-vol-spike
hysteresis overlay (risk_layer.py): capital is cut to 50% while short-term
(21d) realized market vol runs at or above 1.2x its own longer-run (126d)
level, and stays cut until that ratio falls back below 1.0x (hysteresis, to
avoid whipsawing in and out at the trigger boundary). This is "champion" --
CPCV win rate 21/28 folds, Sharpe 1.057 vs. 0.999 for the no-overlay
equal-weight baseline, max drawdown -22.84% vs. -38.99%.

A further overlay, "champion_gold", redirects whatever the vol-spike cut
takes OUT of the book into gold (GLD) instead of cash (see `hedge_ret` in
backtest.run_long_only and research/safe_haven_hedge.py) -- this is now the
single best-validated configuration found in this project's research (see
table/00_MASTER_eval_table.png, Section I): CPCV win rate 26/28 folds (the
highest of anything tested), Sharpe 1.168, CAGR +23.39% (beating even the
no-overlay baseline's own return), max drawdown -23.66%.

`run_long_only` also supports GJR-GARCH inverse-vol position weighting
(`use_vol_weighting=True`, the function's own default) and a portfolio-
level vol target (`vol_target=0.20` e.g.) -- this project's entire overlay
comparison this far, including the 1.2x/1.0x hysteresis above, was run on
the plain equal-weight book (`use_vol_weighting=False, vol_target=None`)
for faster iteration, so combining the champion overlay with vol-weighting/
targeting is untested, not merely disabled -- don't assume it would still
win without re-running the comparison.

Other overlays explored during development -- a Kalman trend-slope exit
gate, per-position stop-loss, vol-spike tiering/continuous scaling,
GARCH/HMM/PCA-based regime detection, Markov-switching GARCH -- are
documented in research/ and the master eval table but are not part of this
production run; see run_long_only's docstring in backtest.py for how to
re-enable any of them for further experimentation.

Run:
    python main.py            # use cached data (Data/*.parquet)
    python main.py --force    # re-fetch the S&P 500 universe from LSEG
'''
import os
import sys
import argparse
import pandas as pd

# stdout can land on a non-UTF-8 codepage when redirected (e.g. piped to a
# file rather than a real console) -- force UTF-8 so the box-drawing/euro
# characters in the prints below never crash the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from loader import DataLoader
import backtest
from research import safe_haven_hedge
from config import DATA_DIR, PLOT_DIR, START, END, UNIVERSE_PATH

os.makedirs(PLOT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════════════════

def load_universe(force=False) -> pd.DataFrame:
    if not force and os.path.exists(UNIVERSE_PATH):
        print("  [data] universe prices loaded from cache")
        prices = pd.read_parquet(UNIVERSE_PATH)
    else:
        print(f"  [data] fetching S&P 500 universe {START} -> {END} from LSEG ...")
        loader = DataLoader(open_session=True)
        prices = loader.load_universe_prices(start=START, end=END)
        os.makedirs(DATA_DIR, exist_ok=True)
        prices.to_parquet(UNIVERSE_PATH)

    print(f"  [data] {prices.shape[0]} days, {prices.shape[1]} stocks")
    return prices


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(force=False):
    print("\n── Long-Only CMOM (S&P 500) — champion config ──────────────────")

    universe = load_universe(force=force)

    # equal-weight, no vol target -- matches the config this project's whole
    # overlay comparison was validated on (see module docstring above)
    book = dict(use_vol_weighting=False, vol_target=None)

    print("  [long-only] running baseline (equal-weight, no overlay) ...")
    baseline = backtest.run_long_only(universe, **book)

    print("  [long-only] running champion (vol-spike hysteresis 1.2x/1.0x) ...")
    champion_kwargs = dict(vol_spike_protection=True, vol_spike_multiplier=1.2,
                            vol_spike_reset_multiplier=1.0)
    champion = backtest.run_long_only(universe, **champion_kwargs, **book)

    print("  [long-only] running champion + gold hedge (cuts -> 50% gold, not cash) ...")
    gold_ret = safe_haven_hedge.load_gold_returns(universe.index)
    champion_gold = backtest.run_long_only(universe, **champion_kwargs, hedge_ret=gold_ret, **book)

    results = {"baseline": baseline, "champion": champion, "champion_gold": champion_gold}
    backtest.print_long_only_metrics(results, book_desc="equal-weight, no vol target")
    backtest.plot_comparison(results, PLOT_DIR, "long_only_champion.png",
                              "Long-Only CMOM — Champion vs. Champion+Gold-Hedge vs. Baseline  |  S&P 500")

    print(f"  Plots -> {PLOT_DIR}")
    print("─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-fetch the S&P 500 universe from LSEG")
    args = parser.parse_args()
    main(force=args.force)
