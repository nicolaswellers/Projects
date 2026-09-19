"""Minimal KEY=VALUE .env loader, shared by every module in live_trading/
that needs one -- each loads its OWN local .env (see alerts.py and
alpaca_paper/alpaca_client.py), matching the folder each credential set
actually belongs to."""
import os


def load_dotenv(path: str):
    """Doesn't override a real environment variable that's already set."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
