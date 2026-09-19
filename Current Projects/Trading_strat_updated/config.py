'''central configuration for the trading strategy'''
import os

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "Data")
PLOT_DIR = os.path.join(ROOT, "plots")

# ─ universe / backtest window ────────────────────────────────────────────────
UNIVERSE = "SP500"       # S&P 500 constituents, fetched via LSEG chain 0#.SPX
START    = "2010-01-01"
END      = "2026-01-01"

# ─ cached data (parquet) ──────────────────────────────────────────────────────
UNIVERSE_PATH  = os.path.join(DATA_DIR, "universe_prices.parquet")
UKF_SLOPE_PATH = os.path.join(DATA_DIR, "ukf_slopes.parquet")

# ─ cross-sectional momentum / portfolio construction ─────────────────────────
LOOKBACK        = 252
REBALANCE_FREQ  = 21

# ─ costs / capital ────────────────────────────────────────────────────────────
TRANSACTION_COST    = 0.001
STARTING_CAPITAL    = 100_000.0   # euros
FIXED_FEE_PER_STOCK = 4.0         # euros per stock traded

# ─ Kalman exit/entry gate (see backtest.py) ───────────────────────────────────
KALMAN_ZSCORE_WINDOW      = 126  # rolling window (trading days, ~6mo) for the slope z-score
KALMAN_ZSCORE_MIN_PERIODS = 20   # minimum observations before the z-score is trusted
KALMAN_EXIT_THRESHOLD     = -3.0 # exit when a held position's slope z-score falls below this

# ─ long-only book + risk overlay (see cmom.py / risk_layer.py) ───────────────
LONG_TOP_N              = 15   # max positions (unfilled slots = cash)
STOP_LOSS_LEVELS        = [0.05, 0.10, 0.15]  # per-position cut levels to compare
CAPITAL_HALVE_DRAWDOWN  = 0.10  # per-position drawdown that counts as "underwater"
CAPITAL_HALVE_COUNT     = 5     # underwater positions needed to trigger the halving
CAPITAL_HALVE_SCALAR    = 0.5   # capital multiplier while the trigger holds

# ─ market-wide vol-spike protection (see risk_layer.py) ──────────────────────
VOL_SPIKE_SHORT_WINDOW   = 21    # trading days, short-term realized market vol
VOL_SPIKE_LONG_WINDOW    = 126   # trading days, longer-run baseline market vol
VOL_SPIKE_MULTIPLIER     = 1.2   # short vol must reach this x the long-run vol to trigger (on-threshold)
VOL_SPIKE_RESET_MULTIPLIER = 1.0 # hysteresis off-threshold: stays triggered until ratio falls back below this
VOL_SPIKE_SCALAR         = 0.5   # capital multiplier while a spike is active (non-continuous mode)
VOL_SPIKE_CONTINUOUS_FLOOR = 0.3 # min exposure multiplier in continuous mode, however severe the spike
VOL_SPIKE_MULTIPLIER_LEVELS = [1.2, 2.0, 2.5]        # alternative trigger levels to compare against VOL_SPIKE_MULTIPLIER
VOL_SPIKE_WINDOW_LEVELS     = [(10, 60), (42, 252)]  # alternative (short, long) window pairs to compare
VOL_SPIKE_TIER_LEVELS = {                            # staged (trigger_multiplier, scalar) cuts to compare
    "2tier":  [(1.2, 0.5), (1.8, 0.0)],
    "3tier":  [(1.2, 0.5), (1.5, 0.25), (2.0, 0.0)],
}
VOL_SPIKE_HYSTERESIS_LEVELS = [(1.2, 1.0), (1.2, 0.8)]  # (on_multiplier, off_multiplier) pairs to compare

# ─ GJR-GARCH vol weighting + vol targeting (see backtest.py) ─────────────────
GARCH_WINDOW           = 750   # trailing trading days fed into each walk-forward fit (~3y)
GARCH_MIN_OBS          = 300   # minimum history required before a ticker gets a fit
POSITION_VOL_CAP_MULT  = 3.0   # cap any single position at this many x equal weight
VOL_TARGET             = 0.20  # annualized portfolio volatility target
VOL_TARGET_LEVELS      = [0.10, 0.15, 0.25, 0.30]  # alternative targets to compare against VOL_TARGET
VOL_TARGET_LOOKBACK    = 63    # trading days of trailing realized vol used to scale exposure
VOL_TARGET_SCALAR_CAP  = 2.0   # max leverage multiplier applied to hit the target

# ─ momentum significance filter (see cmom.py) ─────────────────────────────────
MOMENTUM_MIN_ZSCORE   = 2.33   # ~one-tailed 1% under a normal cross-section
MOMENTUM_ZSCORE_LEVELS = [1.0, 1.65, 2.33]  # ~16%, ~5%, ~1% one-tailed, to compare
