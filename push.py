"""Build the live-status JSON the dashboard reads (Follow-Kronos, 15-min BTC). Prints it to stdout."""
import json, os, datetime as dt
import data
from fifteen import load, summary

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "settle_values.json")
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def actual(w):
    if not w["result"]:
        return None
    if w["ticker"] not in cache:
        v = data.kalshi(f"/markets/{w['ticker']}").get("market", {}).get("expiration_value")
        if v:
            cache[w["ticker"]] = float(v)
    return cache.get(w["ticker"])


L = load()
now = dt.datetime.now(dt.timezone.utc)
spot = data.candles_1m(2)[-1][4]
S = summary(L)
W = L["windows"][-10:]  # most recent 10 windows on the page
out = {
    "updated": now.isoformat(timespec="seconds"),
    "btc": {"spot": spot},
    "summary": {k: S[k] for k in ("cash", "pnl", "trades", "wins", "settled", "windows", "dir_right",
                                  "disagree", "disagree_wins", "brier")},
    "windows": [{"open": w["open_utc"], "close": w["close_utc"], "decided": w["decided_utc"], "strike": w["strike"],
                 "spot": w["spot"], "kronos_median": w["kronos_median"], "actual": actual(w), "p": w["p"],
                 "trade": w["trade"], "agrees": w.get("agrees_with_market"), "result": w["result"], "pnl": w["pnl"]}
                for w in W],
}
json.dump(cache, open(CACHE, "w"))
print(json.dumps(out, separators=(",", ":")))
