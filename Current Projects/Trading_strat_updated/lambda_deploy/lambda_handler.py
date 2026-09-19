"""AWS Lambda entry point for the mom_vol_spike_gold_hedge strategy.
Deployed as a container image (see Dockerfile) -- pandas/numpy/matplotlib/
arch comfortably exceed the zip-deployment size limit, so a container image
is the standard way to run this kind of stack on Lambda.

Adds live_trading/ to sys.path so `import live_runner` resolves the exact
same way it does when run locally (`python live_trading/live_runner.py`) --
live_runner.py's own sys.path setup (based on its __file__) then finds both
the project-root modules (backtest.py etc.) and live_trading/alpaca_paper/
(AlpacaBroker) the same way in both places. Same code, same strategy logic
either way -- only the broker (AlpacaBroker for now; a future IBKRBroker
would need its own always-on host, not Lambda -- see broker.py) and where
state lives (S3 here vs. a local file when run locally -- see live_runner.py's
STATE_S3_BUCKET check) differ.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_trading"))

import live_runner
from alpaca_client import AlpacaBroker


def handler(event, context):
    live_runner.main(AlpacaBroker())
    return {"statusCode": 200}
