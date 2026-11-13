# RC20-DL strategy (standalone)

RC20 with 5 names plus the **drawdown-losers trigger**, in one file: `rc20_dl.py` (numpy + pandas only). It reproduces the
research results exactly: every daily return matches `champions/RC20_DL.py`. It is a candidate variant, not the champion,
and it is not in the paper runner.

```
python rc20_dl.py backtest        # all three versions, US (1996-) and Europe (2000-) -> output/daily_returns_{us,eu}.csv + summary
python rc20_dl.py holdings us     # what to hold from the latest month end, for MODE (also: eu)
```

Data: reads the cached files in `../Data` (set `RC20_DATA` to point elsewhere). The US needs `spx_pit_prices_1993_2026.parquet`,
`spx_joiners_leavers.parquet`, `gold_spot_1993_2026.parquet` and `spy_1993_2026.parquet`. Europe needs
`stoxx600_pit_prices_1997_2026.parquet`, `stoxx600_joiners_leavers.parquet` and `stoxx600_benchmarks_1997_2026.parquet`.

## The trigger

| Step | Rule | In `rc20_dl.py` |
|---|---|---|
| Check | at every month-end close: index drawdown = 1 - index TR / its highest close of the last 252 trading days (S&P 500 TR, STOXX 600 TR) | `index_drawdown` |
| Fire | drawdown >= 20% | `trigger_plan` |
| Losers | the 5 stocks with the lowest raw total return over the last 21 trading days, from the 100 worst, greedy correlation penalty 0.5 (each next pick: loss score - 0.5 x standardised average correlation with the picks so far, 252-day daily returns) | `losers_at`, `select` |
| Trade | at the next close, equal weight | `stock_book` |
| Hold | one month, then check again: still >= 20% means fresh losers (names still on the list are kept); recovered means the trigger is off | `trigger_plan` |
| Exit | at the first month end the index is less than 20% below its high (`HOLD_EXTRA = 0`; holding 1-3 months longer was tested and is worse) | `trigger_plan` |

## The three versions (`MODE`)

| Mode | Normal months | Trigger months |
|---|---|---|
| `switch` (default) | 100% RC20 (5 names, both gold layers) | 100% in the 5 losers, no gold |
| `mix` | 70% RC20 + 30% RC20's 5 names without gold | 70% RC20 + 30% in the 5 losers |
| `trigger` | 100% cash | 100% in the 5 losers |

RC20 is the same as `../RC20_strategy/rc20.py`, with `TOP_N = 5` and a EUR 1 fee. Its signal is residual momentum t-252..t-21
with the correlation penalty 0.5 from the best 100. It has two gold layers: 50% while the 21d / 126d market volatility is >= 1.2
(off below 1.0), and 50% at month ends while 21-day market volatility is >= 22% (off below 16%).

Costs are 10bp + EUR 1 per trade and European buy taxes. The switch also pays for selling the RC20 book when it enters, and for
buying it back (with the average member buy tax) when it exits. Cash earns nothing.

Settings are the constants at the top of `rc20_dl.py`: `MODE`, `DD_TRIGGER`, `LOSER_DAYS`, `LOSER_LAMBDA`, `HOLD_EXTRA`,
`MIX_SLEEVE`, `TOP_N`, `FEE`, `CAPITAL` and the rest.

## Results (after costs, 100k, EUR 1 fee): Sharpe / CAGR / max drawdown

| | US 1996-2026 | Europe 2000-2026 |
|---|---|---|
| switch | 0.75 / +22.1% / -72% | 0.62 / +14.2% / -65% |
| mix (70/30) | 0.80 / +16.7% / -54% | 0.65 / +10.8% / -51% |
| trigger alone | 0.43 / +8.8% / -66% | 0.28 / +3.8% / -65% |
| RC20, 5 names | 0.69 / +12.3% / -50% | 0.60 / +9.3% / -49% |
| Index TR | 0.53 / +8.9% / -56% | 0.34 / +4.7% / -59% |

Since 2017 the switch made US 1.00 / +22.4% / -19% and Europe 0.57 / +10.9% / -49%. The losers were held 8% (US) and 11% (Europe)
of days. Charts are in `output/`: `switch.png`, `switch_since_2017.png`, `trigger_only.png` and `rc20_dl_equity.png` (mix).

## What to know before using it

- **It is a bet on a few crashes.** The trigger fired 7 times in the US (2001, 2001-02, 2002-03, 2008-09, 2020, 2022 twice) and
  6 times in Europe. 2009 alone carries 40-60% of the gain. The Sharpe gain over RC20 is not statistically significant
  (US p 0.67, Europe p 0.94 for the switch).
- **It can hurt before it pays.** The switch's -72% (US) is October-November 2008, when it was fully in losers that kept falling
  before the 2009 rebound of about 5.6x. Europe lost money in the trigger years 2002 and 2008 before 2003 and 2009 paid off.
- **The tested variations were worse.** Removing the correlation penalty, holding the losers 1-3 months longer, and triggers
  below 17.5% all did worse. Triggers of 17.5-22.5% form a plateau.
- **The data has to be updated.** The cached prices end 15 October 2026, so `holdings` currently treats that day as the latest
  month end. Run it after the last close of each month once the data is current.
