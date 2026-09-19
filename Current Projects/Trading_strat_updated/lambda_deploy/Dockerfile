# Build from the project ROOT as context, e.g.:
#   docker build -f lambda_deploy/Dockerfile -t mom_vol_spike_gold_hedge --provenance=false --sbom=false .
# (not from inside lambda_deploy/ -- the COPY paths below are relative to
# the project root, since this needs files from both the root and
# live_trading/). --provenance=false --sbom=false is required: Docker's
# newer buildx otherwise attaches an OCI attestation manifest that Lambda's
# container image support rejects.
FROM public.ecr.aws/lambda/python:3.12

COPY lambda_deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# core strategy modules (project root)
COPY backtest.py GARCH.py cmom.py risk_layer.py config.py ./

# fallback ticker list -- only used if the live Wikipedia fetch fails
COPY Data/universe_prices.parquet ./Data/universe_prices.parquet

# broker-agnostic live trading modules
COPY live_trading/broker.py live_trading/dotenv_util.py live_trading/sp500_membership.py \
     live_trading/alerts.py live_trading/live_runner.py \
     ./live_trading/

# Alpaca-specific adapter (see live_trading/alpaca_paper/)
COPY live_trading/alpaca_paper/alpaca_client.py live_trading/alpaca_paper/data_refresh.py \
     ./live_trading/alpaca_paper/

COPY lambda_deploy/lambda_handler.py ./

CMD ["lambda_handler.handler"]
