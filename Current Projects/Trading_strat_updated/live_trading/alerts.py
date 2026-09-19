"""Email alerts for the unattended live_runner.py -- so a broken scheduled
run doesn't fail silently for weeks before anyone notices.

Setup (one-time, outside this repo): both Gmail and Yahoo require an
app-specific password for SMTP (your normal account password won't work
once 2-factor auth is on, and Yahoo/Google both increasingly require it) --
  Gmail: enable 2-Step Verification (myaccount.google.com/security), then
         generate an App Password (myaccount.google.com/apppasswords).
  Yahoo: Account Info -> Account Security -> Generate app password
         (help.yahoo.com/kb/SLN15241.html).

Add to live_trading/.env (already gitignored; kept separate from
alpaca_paper/.env's broker credentials since this module is broker-agnostic):
    ALERT_EMAIL_FROM=the-account-you-send-from@{gmail.com,yahoo.com,...}
    ALERT_EMAIL_APP_PASSWORD=the-app-password-for-that-account
    ALERT_EMAIL_TO=where-alerts-should-go@example.com          # optional
    ALERT_SMTP_HOST=smtp.mail.yahoo.com                        # optional
    ALERT_SMTP_PORT=587                                        # optional
ALERT_EMAIL_TO defaults to nicolas.wellers@yahoo.com if unset. ALERT_SMTP_HOST
defaults to smtp.gmail.com -- override it if ALERT_EMAIL_FROM is a Yahoo (or
other) address instead (Yahoo: smtp.mail.yahoo.com).

If ALERT_EMAIL_FROM/ALERT_EMAIL_APP_PASSWORD aren't set, `send_alert` prints
a warning and does nothing rather than raising -- a missing alert config
shouldn't itself crash the trading script it's meant to be monitoring.
"""
import os
import smtplib
from email.mime.text import MIMEText

from dotenv_util import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

_DEFAULT_TO = "nicolas.wellers@yahoo.com"


def send_alert(subject: str, body: str):
    sender   = os.environ.get("ALERT_EMAIL_FROM")
    password = os.environ.get("ALERT_EMAIL_APP_PASSWORD")
    to       = os.environ.get("ALERT_EMAIL_TO", _DEFAULT_TO)
    host     = os.environ.get("ALERT_SMTP_HOST", "smtp.gmail.com")
    port     = int(os.environ.get("ALERT_SMTP_PORT", "587"))

    if not sender or not password:
        print(f"  [alerts] ALERT_EMAIL_FROM/ALERT_EMAIL_APP_PASSWORD not set in "
              f".env -- would have sent: {subject!r}. See alerts.py docstring for setup.")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, [to], msg.as_string())
        print(f"  [alerts] sent: {subject!r} -> {to}")
    except Exception as e:
        # deliberately swallowed: a failed *alert* must never mask or
        # replace the original error it was trying to report
        print(f"  [alerts] failed to send alert email: {e}")


if __name__ == "__main__":
    send_alert("live_runner.py test alert", "This is a test of the email alert path -- if you got this, it works.")
