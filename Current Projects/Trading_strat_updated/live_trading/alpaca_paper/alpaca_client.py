"""Thin wrapper around alpaca-py's TradingClient -- the Alpaca-specific
adapter implementing live_trading/broker.py's Broker interface. Everything
Alpaca-specific lives in this alpaca_paper/ subfolder; live_runner.py and
the rest of live_trading/ never import from alpaca-py directly.

Setup (one-time, outside this repo):
  1. Sign up free at https://alpaca.markets -- no funding needed for paper.
  2. In the dashboard, generate a PAPER API key + secret (separate from any
     live-trading keys, which this project should never touch).
  3. Put them in this folder's .env file (already .gitignore'd),
     one KEY=VALUE per line:
        ALPACA_API_KEY=your-paper-key-id
        ALPACA_SECRET_KEY=your-paper-secret
     This is deliberately a file, not a shell command or an OS-level env
     var -- Windows env vars set via `setx` only take effect in NEW shells
     started after the change, which makes them unreliable for an IDE-
     hosted terminal session that was already running; a plain gitignored
     file avoids that entirely and never touches shell history. Real
     process environment variables (if already set) still take priority
     over the .env file, so CI/production deployments can override this
     however they like.

Everything below only ever talks to the paper endpoint (`paper=True` is
hardcoded in `get_client`) so there's no risk of an accidental live order
while this project is still at the "does the plumbing work" stage.
"""
import os
import sys

from datetime import date

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, GetCalendarRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # live_trading/, for broker.py
from broker import Broker
from dotenv_util import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def to_alpaca_symbol(ric: str) -> str:
    """Best-effort LSEG RIC -> plain US-equity ticker: this project's
    universe columns are RICs (e.g. "WDC.OQ", "LRCX.OQ" -- exchange-suffixed),
    but Alpaca expects the bare symbol ("WDC", "LRCX"). Strips everything
    from the last "." onward, then handles the one systematic exception
    confirmed against the actual S&P 500 universe: LSEG denotes a share
    class with a trailing lowercase letter on an otherwise-uppercase root
    ("BRKb" = Berkshire Class B, "BFb" = Brown-Forman Class B), while
    Alpaca/NYSE convention uses a literal dot instead ("BRK.B", "BF.B")."""
    root = ric.rsplit(".", 1)[0]
    if len(root) > 1 and root[-1].islower() and root[:-1].isupper():
        return f"{root[:-1].upper()}.{root[-1].upper()}"
    return root.upper()


def get_client() -> TradingClient:
    """Paper-trading client, credentials from environment variables only."""
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set in the environment. "
            "See this module's docstring for how to generate and set them."
        )
    return TradingClient(key, secret, paper=True)


def account_summary(client: TradingClient) -> dict:
    a = client.get_account()
    return {
        "equity": float(a.equity),
        "cash": float(a.cash),
        "buying_power": float(a.buying_power),
        "portfolio_value": float(a.portfolio_value),
    }


def current_positions(client: TradingClient) -> dict[str, float]:
    """{ticker: qty} for every FILLED position (qty can be fractional).
    Does NOT include symbols with an order still pending -- see
    `open_order_symbols` for that, and don't use this alone to decide
    whether a symbol is already "spoken for" when submitting new orders,
    or a re-run before yesterday's orders fill will double them up."""
    return {p.symbol: float(p.qty) for p in client.get_all_positions()}


def current_position_values(client: TradingClient) -> dict[str, float]:
    """{ticker: market_value} for every FILLED position -- current
    mark-to-market dollar value (Alpaca computes this server-side), used to
    track an isolated sub-portfolio's own compounding value rather than
    reading the shared paper account's total equity."""
    return {p.symbol: float(p.market_value) for p in client.get_all_positions()}


def open_orders(client: TradingClient) -> list:
    """Every open (unfilled) order -- combine with `current_positions` to
    get the full set of symbols already "spoken for" before deciding what
    new orders to submit, and use each order's `.id` to cancel it via
    `client.cancel_order_by_id` if a rebalance drops that name before it
    fills."""
    return client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))


def is_market_open(client: TradingClient) -> bool:
    return client.get_clock().is_open


def is_trading_day(client: TradingClient, day: date | None = None) -> bool:
    """Whether `day` (default: today) had/has a NYSE session at all --
    correctly covers both weekends and market holidays via Alpaca's own
    calendar, rather than hardcoding a holiday list that would need yearly
    upkeep. An empty calendar response means no session that day."""
    day = day or date.today()
    sessions = client.get_calendar(GetCalendarRequest(start=day, end=day))
    return len(sessions) > 0


def submit_target_notional_order(client: TradingClient, symbol: str, notional_usd: float):
    """Submit a market order sized in dollars (Alpaca supports fractional
    shares natively, which maps directly onto this project's target-weight
    sizing without a separate shares-rounding step). Positive notional_usd
    buys, negative sells. No-ops on ~zero orders."""
    if abs(notional_usd) < 1.0:
        return None
    side = OrderSide.BUY if notional_usd > 0 else OrderSide.SELL
    order = MarketOrderRequest(
        symbol=symbol,
        notional=round(abs(notional_usd), 2),
        side=side,
        time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(order)


class AlpacaBroker(Broker):
    """Broker interface (see broker.py) implemented over alpaca-py. Thin
    adapter -- delegates to the free functions above rather than
    reimplementing anything, so this stays a pure wiring layer."""

    def __init__(self):
        self.client = get_client()

    def get_position_values(self) -> dict[str, float]:
        return current_position_values(self.client)

    def get_open_order_ids_by_symbol(self) -> dict[str, list[str]]:
        by_symbol: dict[str, list[str]] = {}
        for o in open_orders(self.client):
            by_symbol.setdefault(o.symbol, []).append(o.id)
        return by_symbol

    def submit_order(self, symbol: str, notional_usd: float) -> None:
        submit_target_notional_order(self.client, symbol, notional_usd)

    def close_position(self, symbol: str) -> None:
        self.client.close_position(symbol)

    def cancel_order(self, order_id: str) -> None:
        self.client.cancel_order_by_id(order_id)

    def is_market_open(self) -> bool:
        return is_market_open(self.client)

    def is_trading_day(self, day: date | None = None) -> bool:
        return is_trading_day(self.client, day)


if __name__ == "__main__":
    client = get_client()
    print("account:", account_summary(client))
    print("positions:", current_positions(client))
    print("market open:", is_market_open(client))
