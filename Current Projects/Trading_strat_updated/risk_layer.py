'''Position-level stop-loss + portfolio-level capital scaling for a
long-only book:

  - stop_loss_threshold: a held position is cut to zero once it's down
    at or below that threshold from entry.
  - capital scalar: halved while CAPITAL_HALVE_COUNT or more held
    positions are each down at or below CAPITAL_HALVE_DRAWDOWN from entry;
    reverts to 1.0 as soon as fewer are (dynamic, not a one-way switch).
  - vol-spike scalar: cuts capital while short-term realized market
    volatility is running at or above `vol_spike_multiplier` x its own
    longer-run level -- a market-wide vol spike, independent of any single
    position's own P&L. This is aimed at momentum-crash risk: broad,
    sudden vol spikes tend to precede the market-wide reversals that hit a
    high-beta momentum book hardest, well before any one stock's own
    drawdown would flag it. Three independent knobs:
      - `vol_spike_direction`: None (any spike), "down" (only cut when the
        trailing market return is negative -- a volatile melt-up isn't
        punished) or "up" (the mirror case, mainly a diagnostic).
      - `vol_spike_continuous`: False cuts by a flat VOL_SPIKE_SCALAR once
        triggered; True scales exposure smoothly with how far short vol has
        run past long vol (floored at `vol_spike_floor`), instead of a
        single discrete step.
      - `vol_spike_multiplier`: overrides VOL_SPIKE_MULTIPLIER per instance,
        for sweeping the trigger level itself.
      - `vol_spike_tiers`: a list of (trigger_multiplier, scalar) pairs for a
        staged cut instead of a single step, e.g. [(1.2, 0.5), (1.8, 0.0)]
        halves capital once the ratio reaches 1.2x and cuts to cash at 1.8x.
        Takes priority over `vol_spike_multiplier`/`vol_spike_continuous`
        when set.
      - `vol_spike_reset_multiplier`: enables hysteresis on the single-
        multiplier trigger -- once active (ratio >= vol_spike_multiplier),
        stays active until the ratio falls back below this lower reset
        level, rather than reverting the moment it dips under the trigger.
        Ignored when `vol_spike_tiers` is set.

    `vol_spike_scalar` normally computes its own ratio from short_vol/
    long_vol, but the caller can instead pass a precomputed `signal`
    (e.g. a Kalman-filtered GARCH-vol z-score) to drive the exact same
    tiers/hysteresis/continuous/direction machinery off a different kind
    of trigger -- `vol_spike_multiplier`/`vol_spike_reset_multiplier`/
    `vol_spike_tiers` are then just read as levels on that signal's own
    scale, not literally a "multiplier x long-run vol" anymore.

All three checks are meant to be applied off a day's close, after that
day's return has already been booked against the prior day's positions, so
the decision only ever takes effect starting the next day (see backtest.py).
'''
import numpy as np
import pandas as pd

from config import (
    CAPITAL_HALVE_DRAWDOWN, CAPITAL_HALVE_COUNT, CAPITAL_HALVE_SCALAR,
    VOL_SPIKE_MULTIPLIER, VOL_SPIKE_SCALAR, VOL_SPIKE_CONTINUOUS_FLOOR,
)


class RiskLayer:
    def __init__(self, stop_loss_threshold: float | None = None, vol_spike_protection: bool = False,
                 vol_spike_direction: str | None = None, vol_spike_multiplier: float | None = None,
                 vol_spike_continuous: bool = False, vol_spike_floor: float = VOL_SPIKE_CONTINUOUS_FLOOR,
                 vol_spike_tiers: list[tuple[float, float]] | None = None,
                 vol_spike_reset_multiplier: float | None = None):
        self.enabled = stop_loss_threshold is not None
        self.stop_loss_threshold = stop_loss_threshold
        self.vol_spike_protection = vol_spike_protection
        self.vol_spike_direction   = vol_spike_direction    # None | "down" | "up"
        self.vol_spike_multiplier  = vol_spike_multiplier if vol_spike_multiplier is not None else VOL_SPIKE_MULTIPLIER
        self.vol_spike_continuous  = vol_spike_continuous
        self.vol_spike_floor       = vol_spike_floor
        self.vol_spike_tiers       = vol_spike_tiers
        self.vol_spike_reset_multiplier = vol_spike_reset_multiplier
        self._vol_spike_active     = False  # hysteresis state, only used when vol_spike_reset_multiplier is set
        self.entry_price = pd.Series(dtype=float)

    def on_new_positions(self, positions: pd.Series, prices: pd.Series):
        """Record entry price for tickers newly held today. Names that
        were already held keep their original entry, so a rebalance that
        re-picks them doesn't reset the stop clock."""
        held = positions[positions > 0].index
        new_entries = held.difference(self.entry_price.dropna().index)
        for t in new_entries:
            self.entry_price[t] = prices.get(t, float("nan"))

    def _drawdowns(self, positions: pd.Series, prices: pd.Series) -> pd.Series:
        held = positions[positions > 0].index
        return prices.reindex(held) / self.entry_price.reindex(held) - 1.0

    def apply_stop_loss(self, positions: pd.Series, prices: pd.Series) -> pd.Series:
        if not self.enabled:
            return positions
        positions = positions.copy()
        dd = self._drawdowns(positions, prices)
        stopped = dd[dd <= -self.stop_loss_threshold].index
        positions.loc[stopped] = 0.0
        self.entry_price = self.entry_price.drop(index=stopped, errors="ignore")
        return positions

    def capital_scalar(self, positions: pd.Series, prices: pd.Series) -> float:
        if not self.enabled:
            return 1.0
        dd = self._drawdowns(positions, prices)
        n_underwater = int((dd <= -CAPITAL_HALVE_DRAWDOWN).sum())
        return CAPITAL_HALVE_SCALAR if n_underwater >= CAPITAL_HALVE_COUNT else 1.0

    def vol_spike_scalar(self, short_vol: float, long_vol: float,
                          market_ret_trailing: float | None = None,
                          signal: float | None = None) -> float:
        """See class docstring. Independent of `enabled` -- controlled by
        `vol_spike_protection`. Pass `signal` to drive the trigger off a
        precomputed value instead of short_vol/long_vol."""
        if not self.vol_spike_protection:
            return 1.0
        if signal is not None:
            if np.isnan(signal):
                return 1.0
            ratio = signal
        else:
            if np.isnan(short_vol) or np.isnan(long_vol) or long_vol < 1e-10:
                return 1.0
            ratio = short_vol / long_vol

        if self.vol_spike_tiers:
            triggered = [s for m, s in self.vol_spike_tiers if ratio >= m]
            if not triggered:
                return 1.0
            if self.vol_spike_direction is not None:
                if market_ret_trailing is None or np.isnan(market_ret_trailing):
                    return 1.0
                if self.vol_spike_direction == "down" and market_ret_trailing >= 0:
                    return 1.0
                if self.vol_spike_direction == "up" and market_ret_trailing <= 0:
                    return 1.0
            return min(triggered)

        if self.vol_spike_reset_multiplier is not None:
            if self._vol_spike_active:
                if ratio < self.vol_spike_reset_multiplier:
                    self._vol_spike_active = False
            elif ratio >= self.vol_spike_multiplier:
                self._vol_spike_active = True
            if not self._vol_spike_active:
                return 1.0
        elif ratio < self.vol_spike_multiplier:
            return 1.0

        if self.vol_spike_direction is not None:
            if market_ret_trailing is None or np.isnan(market_ret_trailing):
                return 1.0
            if self.vol_spike_direction == "down" and market_ret_trailing >= 0:
                return 1.0
            if self.vol_spike_direction == "up" and market_ret_trailing <= 0:
                return 1.0

        if self.vol_spike_continuous:
            return max(self.vol_spike_floor, min(1.0, self.vol_spike_multiplier / ratio))
        return VOL_SPIKE_SCALAR
