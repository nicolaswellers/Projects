''''this is main: here i combine my liquidity and credit filter'''

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf

from loader import DataLoader
from credit_filter import CreditFilter
from liquidity_filter import LiquidityFilter

# ── config ────────────────────────────────────────────────────────────────────
START       = "2016-06-01"
END         = "2026-06-01"
WINDOW      = 252   # ~1 year: level z-scores need long memory to distinguish regimes
CACHE_DIR   = "./cache"
DATA_CACHE  = os.path.join(CACHE_DIR, "raw_data.pkl")
RESULT_CACHE = os.path.join(CACHE_DIR, "filter_results.pkl")

os.makedirs(CACHE_DIR, exist_ok=True)


# ── data ──────────────────────────────────────────────────────────────────────
def load_data():
    if os.path.exists(DATA_CACHE):
        print("Loading raw data from cache...")
        with open(DATA_CACHE, "rb") as f:
            return pickle.load(f)

    print("Fetching data from LSEG + FRED + yfinance...")
    loader = DataLoader()

    hy_oas = loader.load_fred("BAMLH0A0HYM2", START, END)  # ICE BofA HY OAS (bps)
    spy    = loader.load_daily_close("SPY",  START, END)
    vol    = loader.load_daily_volume("SPY", START, END)

    # S&P 500 index via yfinance
    spx = yf.download("^GSPC", start=START, end=END, progress=False)["Close"].squeeze()
    spx.index = pd.to_datetime(spx.index)
    spx = spx.sort_index()

    data = {"hy_oas": hy_oas, "spy": spy, "spy_vol": vol, "spx": spx}

    with open(DATA_CACHE, "wb") as f:
        pickle.dump(data, f)
    print(f"Raw data cached to {DATA_CACHE}")
    return data


# ── filters ───────────────────────────────────────────────────────────────────
def run_filters(data):
    if os.path.exists(RESULT_CACHE):
        print("Loading filter results from cache...")
        with open(RESULT_CACHE, "rb") as f:
            return pickle.load(f)

    print("Running filters...")
    cf = CreditFilter(data["hy_oas"])
    lf = LiquidityFilter(data["spy"], data["spy_vol"])

    liq_level = lf.liquidity_level(WINDOW)
    liq_shock = lf.liquidity_shock(liq_level)

    results = {
        "credit_level":        cf.spread_level(),
        "credit_shock":        cf.spread_shock(),
        "credit_accel":        cf.spread_acceleration(),
        "liquidity_level":     liq_level,
        "liquidity_shock":     liq_shock,
        "liquidity_accel":     lf.liquidity_acceleration(liq_shock),
    }

    with open(RESULT_CACHE, "wb") as f:
        pickle.dump(results, f)
    print(f"Filter results cached to {RESULT_CACHE}")
    return results


# ── plotting ──────────────────────────────────────────────────────────────────
PANEL_META = [
    ("credit_level",    "Credit Level (z)",        "tab:red"),
    ("credit_shock",    "Credit Shock (z)",         "tab:orange"),
    ("credit_accel",    "Credit Acceleration (z)",  "tab:pink"),
    ("liquidity_level", "Liquidity Level (z)",      "tab:blue"),
    ("liquidity_shock", "Liquidity Shock (z)",      "tab:cyan"),
    ("liquidity_accel", "Liquidity Acceleration (z)", "tab:purple"),
]


def plot_filters(data, results):
    spx = data["spx"]
    n   = len(PANEL_META)

    fig = plt.figure(figsize=(16, 3 * n))
    fig.suptitle("Credit & Liquidity Filters vs S&P 500 (10Y)", fontsize=14, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(n, 1, hspace=0.55)

    for i, (key, label, color) in enumerate(PANEL_META):
        signal = results[key]
        common = spx.index.intersection(signal.dropna().index)

        ax1 = fig.add_subplot(gs[i])
        ax2 = ax1.twinx()

        ax1.fill_between(common, signal.loc[common], 0,
                         where=signal.loc[common] > 0,
                         alpha=0.35, color=color, label=label)
        ax1.fill_between(common, signal.loc[common], 0,
                         where=signal.loc[common] < 0,
                         alpha=0.2, color=color)
        ax1.plot(common, signal.loc[common], color=color, linewidth=0.8)
        ax1.axhline(0,   color="gray",  linewidth=0.5, linestyle="--")
        ax1.axhline(1.5, color="red",   linewidth=0.5, linestyle=":")
        ax1.axhline(-1.5,color="green", linewidth=0.5, linestyle=":")
        ax1.set_ylabel(label, color=color, fontsize=9)
        ax1.tick_params(axis="y", labelcolor=color, labelsize=8)

        ax2.plot(common, spx.loc[common], color="black", linewidth=1, alpha=0.6, label="S&P 500")
        ax2.set_ylabel("S&P 500", fontsize=9, color="black")
        ax2.tick_params(axis="y", labelsize=8)

        ax1.set_xlim(common[0], common[-1])
        ax1.tick_params(axis="x", labelsize=8)

        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8)

    plot_path = os.path.join(CACHE_DIR, "filter_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data    = load_data()
    results = run_filters(data)
    plot_filters(data, results)