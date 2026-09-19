"""Broker-agnostic interface for live_runner.py.

The strategy logic in live_runner.py (vol-spike hysteresis, gold-hedge
allocation, rebalance timing) should never call a broker SDK directly --
only through this interface. That's what makes swapping or adding a broker
later (e.g. IBKR) a matter of writing one new adapter class, not touching
the strategy itself.

Honest limitation, not something an interface can paper over: this makes
the CODE broker-agnostic, not the DEPLOYMENT. Alpaca is a stateless REST
API, which is why it runs fine in AWS Lambda. IBKR's TWS API requires a
persistent, already-running Gateway/TWS process to connect to -- it cannot
run inside a stateless serverless function. An IBKRBroker implementing this
same interface would need to run on an always-on host (e.g. a small VPS
alongside IB Gateway), not Lambda. The strategy code itself would be
unaffected either way.

All amounts are USD notional (see AlpacaBroker.submit_order); all methods
return plain Python types (str/float/bool/date/dict/list) -- no
broker-specific SDK objects should ever cross this boundary.
"""
from abc import ABC, abstractmethod
from datetime import date


class Broker(ABC):
    @abstractmethod
    def get_position_values(self) -> dict[str, float]:
        """{symbol: current mark-to-market USD value} for every filled
        position in the account."""

    @abstractmethod
    def get_open_order_ids_by_symbol(self) -> dict[str, list[str]]:
        """{symbol: [order_id, ...]} for every open (unfilled) order."""

    @abstractmethod
    def submit_order(self, symbol: str, notional_usd: float) -> None:
        """Market order sized in USD notional. Positive buys, negative
        sells. Implementations should no-op on ~zero orders."""

    @abstractmethod
    def close_position(self, symbol: str) -> None:
        """Close the entire filled position in `symbol`, if any."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel a single open order by id."""

    @abstractmethod
    def is_market_open(self) -> bool:
        ...

    @abstractmethod
    def is_trading_day(self, day: date | None = None) -> bool:
        """Whether `day` (default: today) has/had a session at all --
        should cover both weekends and market holidays."""
