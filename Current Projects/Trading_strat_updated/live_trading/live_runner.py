"""Daily paper-trading runner for the champion+gold-hedge strategy config,
run as an ISOLATED sub-portfolio starting at CAPITAL_CAP / TOP_N positions
-- deliberately sized down from this project's own backtested default (15
positions, whatever the account holds) to match a realistic real-life
starting point, not the shared paper account's total balance. Cuts are
redirected into GLD instead of cash (see ../main.py and
research/safe_haven_hedge.py). Meant to be fired once per trading day,
shortly after the close.

NOTE: CAPITAL_CAP is a plain notional starting point, not an actual EUR/USD
conversion -- this project has never done real currency conversion (the
backtest's own "€" labeling is just a label on a currency-agnostic number),
so "4,000 euros" here means "this sub-portfolio starts with $4,000 to
deploy," using Alpaca's own USD-denominated sizing as-is.

Isolation, not a cap: sizing is NOT min(account equity, CAPITAL_CAP), which
would silently throw away any gains once the sub-portfolio grew past
CAPITAL_CAP. Instead, each run marks-to-market only the specific positions
this strategy itself holds (via current_position_values) and treats THAT
value as the day's equity -- fully reinvested if it's grown, only ever
sized off the reduced amount if it's shrunk, and never topped back up or
capped. It compounds on its own, independent of whatever else might be
sitting in the same paper account.

Each run:
  1. Loads persisted state (risk-layer hysteresis flag, days since last
     rebalance, currently-held names) from live_state.json in this folder.
  2. Recomputes today's vol-spike scalar via backtest.market_vol_ratio_series
     -- the exact same function run_long_only itself uses -- so this can
     never silently diverge from what was actually backtested/validated.
  3. Marks-to-market this strategy's own current positions (see "Isolation"
     above) to get today's sub-portfolio value.
  4. Every REBALANCE_FREQ trading days, re-picks the top-TOP_N momentum
     names via cmom.top_momentum_names (again, the same function the
     backtest uses).
  5. Computes each held name's target dollar allocation from that
     sub-portfolio value, PLUS a GLD allocation sized at whatever fraction
     the vol-spike scalar is currently cutting (see backtest.py's
     `hedge_ret` mechanism this mirrors: `1 - scalar` of the book goes into
     gold rather than sitting in cash whenever a cut is active). Diffs all
     of that against current Alpaca positions, and opens/closes positions
     accordingly.
  6. Persists updated state.
  7. On Fridays, emails a weekly summary of current positions and
     risk-on/risk-off status (see send_weekly_summary) -- reuses this same
     daily trigger rather than needing a second scheduled task.

Known gaps (fine for verifying the plumbing works; not yet a finished
unattended system):
  - Only opens newly-added names/GLD and closes dropped ones; doesn't true
    up notional drift for names (or GLD) held continuously while the
    account's own equity moves. Under the champion's binary (not
    continuous/tiered) vol-spike scalar this only ever means 0% or exactly
    (1 - VOL_SPIKE_SCALAR) of equity in GLD, so the drift is just from
    equity changing day to day while a cut stays active for multiple days
    in a row -- a few percent at most, not the primary mechanism. Fine
    while position sizes are still tiny paper-trading tests; needs adding
    before this reflects the backtest's actual daily rebalancing precisely.
"""
import os
import sys
import json
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# alpaca_paper/ -- everything Alpaca-specific (AlpacaBroker, data_refresh)
# lives there, isolated from the broker-agnostic modules at this level
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpaca_paper"))

import pandas as pd

import backtest
from cmom import top_momentum_names
from risk_layer import RiskLayer
from config import UNIVERSE_PATH, LOOKBACK, REBALANCE_FREQ

from broker import Broker
from alpaca_client import AlpacaBroker, to_alpaca_symbol
from data_refresh import fetch_live_universe
from sp500_membership import get_current_sp500_tickers
from alerts import send_alert

STATE_PATH = os.path.join(os.path.dirname(__file__), "live_state.json")

# champion + gold-hedge config -- see main.py
VOL_SPIKE_MULTIPLIER = 1.2
VOL_SPIKE_RESET_MULTIPLIER = 1.0
GOLD_SYMBOL = "GLD"

# deliberately smaller than this project's backtested default (15 positions,
# whatever the account holds) -- see module docstring
CAPITAL_CAP = 4_000.0
TOP_N = 5


_DEFAULT_STATE = {"vol_spike_active": False, "days_since_rebalance": REBALANCE_FREQ, "held_names": []}
# days_since_rebalance starts at REBALANCE_FREQ so the very first run always
# rebalances (nothing held yet)

# STATE_S3_BUCKET set -> Lambda deployment (see lambda_deploy/), state lives
# in S3 since Lambda's own filesystem doesn't persist between invocations.
# Unset -> local file (Windows Task Scheduler deployment). Same live_runner.py,
# same strategy logic, either way -- only where the state JSON lives differs.
_STATE_S3_BUCKET = os.environ.get("STATE_S3_BUCKET")
_STATE_S3_KEY = "live_state.json"


def load_state() -> dict:
    if _STATE_S3_BUCKET:
        import boto3
        from botocore.exceptions import ClientError
        s3 = boto3.client("s3")
        try:
            obj = s3.get_object(Bucket=_STATE_S3_BUCKET, Key=_STATE_S3_KEY)
            return json.loads(obj["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return dict(_DEFAULT_STATE)
            raise
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return dict(_DEFAULT_STATE)


def save_state(state: dict):
    if _STATE_S3_BUCKET:
        import boto3
        s3 = boto3.client("s3")
        s3.put_object(Bucket=_STATE_S3_BUCKET, Key=_STATE_S3_KEY, Body=json.dumps(state, indent=2))
        return
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def load_universe() -> pd.DataFrame:
    """Trailing ~1.5y of daily closes, fetched fresh from Alpaca each run
    (see data_refresh.py) for the LIVE S&P 500 constituent list (Wikipedia-
    sourced, see sp500_membership.py -- refreshed every run, so index
    additions/removals are picked up automatically, no more manual
    `python main.py --force`/LSEG session needed for ongoing live
    operation). Falls back to the cached LSEG ticker list (via
    alpaca_client.to_alpaca_symbol's RIC conversion) if that fetch fails,
    e.g. a network hiccup or Wikipedia page-structure change, rather than
    letting a transient failure there take down the whole daily run."""
    try:
        ticker_list = get_current_sp500_tickers()
        return fetch_live_universe(ticker_list, already_alpaca_native=True)
    except Exception as e:
        print(f"  [load_universe] live S&P 500 list fetch failed ({e}) -- "
              f"falling back to the cached LSEG ticker list")
        ticker_list = [to_alpaca_symbol(t) for t in pd.read_parquet(UNIVERSE_PATH).columns]
        return fetch_live_universe(ticker_list, already_alpaca_native=True)


def run_once(broker: Broker):
    if not broker.is_trading_day():
        print("  not a trading day (weekend or market holiday) -- nothing to do, skipping")
        return

    if not broker.is_market_open():
        print("  market is currently closed -- expected, since this runs after the close; "
              "any market orders submitted below will queue for the next open")

    state = load_state()
    universe = load_universe()

    # -- vol-spike scalar, same definition run_long_only itself uses --
    vol_short, vol_long, _ = backtest.market_vol_ratio_series(universe)
    risk = RiskLayer(vol_spike_protection=True, vol_spike_multiplier=VOL_SPIKE_MULTIPLIER,
                      vol_spike_reset_multiplier=VOL_SPIKE_RESET_MULTIPLIER)
    risk._vol_spike_active = state["vol_spike_active"]
    scalar = risk.vol_spike_scalar(vol_short.iloc[-1], vol_long.iloc[-1])
    state["vol_spike_active"] = risk._vol_spike_active
    ratio = vol_short.iloc[-1] / vol_long.iloc[-1]
    print(f"  vol ratio: {ratio:.3f}  spike active: {state['vol_spike_active']}  scalar: {scalar}")

    # -- isolated sub-portfolio value: mark-to-market of whatever THIS
    # strategy currently holds, BEFORE today's rebalance/sizing decisions --
    # not the shared paper account's total equity. Starts at CAPITAL_CAP
    # and compounds freely from there: fully reinvested if up, only ever
    # sized off the reduced amount if down -- never topped back up to
    # CAPITAL_CAP or capped once it grows past it.
    tracked_before = set(state["held_names"]) | {GOLD_SYMBOL}
    position_values = broker.get_position_values()
    sub_portfolio_value = sum(v for sym, v in position_values.items() if sym in tracked_before)
    equity = sub_portfolio_value if sub_portfolio_value > 1.0 else CAPITAL_CAP
    print(f"  sub-portfolio value: ${equity:,.2f}  "
          f"({'first run, seeded at CAPITAL_CAP' if sub_portfolio_value <= 1.0 else 'compounded from prior runs'})")

    # -- rebalance every REBALANCE_FREQ trading days --
    state["days_since_rebalance"] += 1
    if state["days_since_rebalance"] >= REBALANCE_FREQ:
        state["held_names"] = list(top_momentum_names(universe, LOOKBACK, TOP_N))
        state["days_since_rebalance"] = 0
        print(f"  rebalanced -> {state['held_names']}")
    else:
        print(f"  no rebalance ({state['days_since_rebalance']}/{REBALANCE_FREQ} days) "
              f"-- holding {state['held_names']}")

    n_held = max(len(state["held_names"]), 1)
    target_notional = equity * scalar / n_held
    # held_names are already Alpaca-native (see load_universe) -- no RIC
    # conversion needed here
    target = {name: target_notional for name in state["held_names"]}

    # -- gold hedge: whatever the vol-spike scalar is cutting from the
    # momentum book goes into GLD instead of sitting in cash (mirrors
    # backtest.py's `hedge_ret` mechanism -- see research/safe_haven_hedge.py)
    gold_notional = equity * (1.0 - scalar)
    if gold_notional > 1.0:
        target[GOLD_SYMBOL] = gold_notional
        print(f"  gold hedge: {(1.0 - scalar):.0%} of equity -> ${gold_notional:,.2f} in {GOLD_SYMBOL}")

    # "spoken for" = filled position (from `position_values`, already
    # fetched above) OR an order already pending -- without the
    # pending-order check, re-running this before yesterday's orders fill
    # (e.g. because it fired again, or a retry after a transient failure)
    # would submit a second, duplicate batch for the same names
    pending_ids_by_symbol = broker.get_open_order_ids_by_symbol()
    spoken_for = set(position_values) | set(pending_ids_by_symbol)

    for symbol in set(target) | spoken_for:
        in_target = symbol in target
        already_spoken_for = symbol in spoken_for
        if in_target and not already_spoken_for:
            print(f"  [order] opening {symbol} @ ${target[symbol]:,.2f}")
            broker.submit_order(symbol, target[symbol])
        elif already_spoken_for and not in_target:
            if symbol in position_values:
                print(f"  [order] closing {symbol}")
                broker.close_position(symbol)
            for order_id in pending_ids_by_symbol.get(symbol, []):
                print(f"  [order] cancelling pending order for dropped name {symbol}")
                broker.cancel_order(order_id)
        # names held both before and after this rebalance are left alone --
        # see "Known gaps" in the module docstring re: notional drift.

    save_state(state)

    if date.today().weekday() == 4:  # Friday
        send_weekly_summary(broker, state, scalar)


def send_weekly_summary(broker: Broker, state: dict, scalar: float):
    """Position + risk-on/off snapshot, emailed once a week (see the Friday
    check in run_once) rather than needing a second scheduled task."""
    position_values = broker.get_position_values()
    tracked = set(state["held_names"]) | {GOLD_SYMBOL}
    sub_portfolio_value = sum(v for sym, v in position_values.items() if sym in tracked)
    risk_status = "RISK OFF" if state["vol_spike_active"] else "RISK ON"
    risk_detail = "vol-spike cut active" if state["vol_spike_active"] else "full exposure"

    lines = [
        f"Weekly summary -- {date.today().isoformat()}",
        "",
        f"Status: {risk_status} ({risk_detail})",
        f"Momentum book exposure: {scalar:.0%}   Gold hedge: {(1.0 - scalar):.0%}",
        f"Sub-portfolio value: ${sub_portfolio_value:,.2f}  (started at ${CAPITAL_CAP:,.2f})",
        f"Days since last rebalance: {state['days_since_rebalance']}/{REBALANCE_FREQ}",
        "",
        "Current positions:",
    ]
    if position_values:
        for symbol, value in sorted(position_values.items()):
            flag = "" if symbol in tracked else "  (not managed by this strategy)"
            lines.append(f"  {symbol:<8} ${value:,.2f}{flag}")
    else:
        lines.append("  (none)")

    body = "\n".join(lines)
    print(body)
    send_alert(f"Weekly strategy summary -- {risk_status}", body)


def main(broker: Broker | None = None):
    """Shared entry point for both the local (Windows Task Scheduler) CLI
    run and the Lambda handler (see lambda_deploy/lambda_handler.py) --
    same error-alerting behavior either way. `broker` defaults to
    AlpacaBroker() -- the only place a broker gets chosen; everything else
    only ever talks to the Broker interface, so adding e.g. IBKRBroker
    later is a one-line change here plus a new adapter class, not a change
    to the strategy logic itself."""
    import traceback

    broker = broker or AlpacaBroker()
    try:
        run_once(broker)
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        send_alert("live_runner.py FAILED", tb)
        raise


if __name__ == "__main__":
    main()
