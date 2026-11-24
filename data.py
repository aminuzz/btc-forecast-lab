"""Data access: Kalshi public market data + Coinbase candles."""
import time, json, datetime as dt
import urllib.request, urllib.parse, urllib.error

KALSHI = "https://external-api.kalshi.com/trade-api/v2"
COINBASE = "https://api.exchange.coinbase.com"
UA = {"User-Agent": "paper-research-bot/0.1"}


def _get(url, params=None, tries=8):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    delay = 2.0
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise
        except urllib.error.URLError:
            if i < tries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise


def kalshi(path, **params):
    return _get(KALSHI + path, params or None)


def open_events(series):
    return kalshi("/events", series_ticker=series, status="open", limit=50).get("events", [])


def event_markets(event_ticker):
    d = kalshi(f"/events/{event_ticker}")
    return d.get("event", {}), d.get("markets", [])


def candles(product="BTC-USD", hours=500):
    """Hourly OHLCV, oldest first. Coinbase returns max 300 per call."""
    end = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    out = {}
    while len(out) < hours:
        start = end - dt.timedelta(hours=300)
        rows = _get(f"{COINBASE}/products/{product}/candles",
                    {"granularity": 3600, "start": start.isoformat(), "end": end.isoformat()})
        if not rows:
            break
        for t, lo, hi, op, cl, vol in rows:
            out[t] = (t, op, hi, lo, cl, vol)
        end = start
        time.sleep(0.4)
    rows = sorted(out.values())[-hours:]
    return rows  # (unix_ts_open, open, high, low, close, volume)


def candles_1m(minutes=240, product="BTC-USD"):
    """1-minute OHLCV, oldest first, including the forming candle."""
    end = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0) + dt.timedelta(minutes=1)
    out = {}
    while len(out) < minutes:
        start = end - dt.timedelta(minutes=300)
        rows = _get(f"{COINBASE}/products/{product}/candles",
                    {"granularity": 60, "start": start.isoformat(), "end": end.isoformat()})
        if not rows:
            break
        for t, lo, hi, op, cl, vol in rows:
            out[t] = (t, op, hi, lo, cl, vol)
        end = start
    return sorted(out.values())[-minutes:]
