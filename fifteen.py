"""Paper-trade Kalshi 15-minute BTC up/down markets (KXBTC15M) with Kronos-small on 1-minute candles.
No real orders. Mechanical rules (set by the desk); the agent team reviews results afterwards.

Usage: python3 fifteen.py --windows 4
Each window: wait until ~60s after it opens, run Kronos (model loaded once), re-quote, decide, then settle later.
"""
import sys, os, json, math, time, argparse, datetime as dt
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("KRONOS_DIR", os.path.join(_HERE, "Kronos-repo")))
import numpy as np, pandas as pd, torch
import data
from forecast import kalshi_fee

LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "follow_ledger.json")
RULES = {"strategy": "follow_kronos", "stake_usd": 10.0, "paths": 32, "lookback_min": 240, "decide_after_s": 60}


def now():
    return dt.datetime.now(dt.timezone.utc)


def load():
    if os.path.exists(LEDGER):
        return json.load(open(LEDGER))
    return {"cash": 1000.0, "start_cash": 1000.0, "rules": RULES, "windows": []}


def save(L):
    json.dump(L, open(LEDGER, "w"), indent=1)


def candles_1m(minutes):
    end = now().replace(second=0, microsecond=0) + dt.timedelta(minutes=1)
    out = {}
    while len(out) < minutes:
        start = end - dt.timedelta(minutes=300)
        rows = data._get(f"{data.COINBASE}/products/BTC-USD/candles",
                         {"granularity": 60, "start": start.isoformat(), "end": end.isoformat()})
        if not rows:
            break
        for t, lo, hi, op, cl, vol in rows:
            out[t] = (t, op, hi, lo, cl, vol)
        end = start
    return sorted(out.values())[-minutes:]


def current_market():
    ev = data.open_events("KXBTC15M")
    if not ev:
        return None
    _, ms = data.event_markets(ev[0]["event_ticker"])
    return ms[0] if ms else None


def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def decide(pred, L):
    m = current_market()
    close = dt.datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
    opened = dt.datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
    if any(w["ticker"] == m["ticker"] for w in L["windows"]):
        return None
    for _ in range(12):  # the start price posts a few seconds after the open
        if m.get("floor_strike"):
            break
        time.sleep(5)
        m = current_market()
    strike = float(m["floor_strike"])
    rows = candles_1m(RULES["lookback_min"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["ts"], unit="s")
    df["amount"] = df["volume"] * df["close"]
    last_open = df["timestamps"].iloc[-1]
    pred_len = max(1, int(round((close.replace(tzinfo=None) - pd.Timedelta(minutes=1) - last_open) / pd.Timedelta(minutes=1))))
    y_ts = pd.Series(pd.date_range(last_open + pd.Timedelta(minutes=1), periods=pred_len, freq="min"))
    x = df[["open", "high", "low", "close", "volume", "amount"]]
    t0 = time.time()
    outs = pred.predict_batch([x] * RULES["paths"], [df["timestamps"]] * RULES["paths"], [y_ts] * RULES["paths"],
                              pred_len=pred_len, T=1.0, top_p=1.0, sample_count=1, verbose=False)
    model_secs = time.time() - t0
    finals = np.array([o["close"].iloc[-1] for o in outs])
    lr = np.diff(np.log(df["close"].values))
    mins_left = max(1.0, (close - now()).total_seconds() / 60)
    sig = max(float(lr.std() * math.sqrt(mins_left)), float(np.std(np.log(finals))))
    spot = float(df["close"].iloc[-1])
    med = float(np.median(finals))
    p_kronos = 1 - norm_cdf((math.log(strike) - math.log(med)) / sig)
    p_zero = 1 - norm_cdf((math.log(strike) - math.log(spot)) / sig)
    # re-quote right before deciding
    m2 = current_market()
    ya, yb, na = float(m2["yes_ask_dollars"]), float(m2["yes_bid_dollars"]), float(m2["no_ask_dollars"])
    mid = (ya + yb) / 2
    e_yes = p_kronos - ya - kalshi_fee(ya)
    e_no = (1 - p_kronos) - na - kalshi_fee(na)
    z_yes = p_zero - ya - kalshi_fee(ya)
    z_no = (1 - p_zero) - na - kalshi_fee(na)
    # Follow Kronos: buy the side its predicted close points to, every window.
    side = "yes" if med >= strike else "no"
    price = ya if side == "yes" else na
    trade = (side, price) if 0 < price < 1 else None
    w = {"ticker": m["ticker"], "open_utc": opened.isoformat(), "close_utc": close.isoformat(),
         "decided_utc": now().isoformat(timespec="seconds"), "strike": strike, "spot": spot,
         "kronos_median": round(med, 2), "sigma_pct": round(sig * 100, 3), "mins_left": round(mins_left, 1),
         "model_secs": round(model_secs, 1), "yes_bid": yb, "yes_ask": ya, "no_ask": na,
         "p": {"kronos": round(p_kronos, 4), "zero_drift": round(p_zero, 4), "market": round(mid, 4)},
         "edge": {"yes": round(e_yes, 4), "no": round(e_no, 4)},
         "agrees_with_market": (side == ("yes" if mid >= 0.5 else "no")), "trade": None, "result": None, "pnl": None}
    if trade:
        side, price = trade
        n = max(1, int(RULES["stake_usd"] // price))
        fee = kalshi_fee(price, n)
        cost = round(n * price + fee, 2)
        L["cash"] = round(L["cash"] - cost, 2)
        w["trade"] = {"side": side, "price": price, "contracts": n, "fee": fee, "cost": cost}
    L["windows"].append(w)
    save(L)
    return w


def settle(L):
    for w in L["windows"]:
        if w["result"] is None and now() > dt.datetime.fromisoformat(w["close_utc"]) + dt.timedelta(seconds=90):
            r = data.kalshi(f"/markets/{w['ticker']}").get("market", {}).get("result")
            if r in ("yes", "no"):
                w["result"] = r
                if w["trade"]:
                    pay = w["trade"]["contracts"] * (1.0 if r == w["trade"]["side"] else 0.0)
                    L["cash"] = round(L["cash"] + pay, 2)
                    w["pnl"] = round(pay - w["trade"]["cost"], 2)
    save(L)


def summary(L):
    done = [w for w in L["windows"] if w["result"]]
    br = {}
    for w in done:
        o = 1 if w["result"] == "yes" else 0
        for k, p in w["p"].items():
            br.setdefault(k, []).append((p - o) ** 2)
    trades = [w for w in done if w["trade"]]
    return {"windows": len(L["windows"]), "settled": len(done), "trades": len([w for w in L["windows"] if w["trade"]]),
            "wins": sum(1 for w in trades if w["pnl"] > 0), "pnl": round(sum(w["pnl"] for w in trades), 2),
            "dir_right": sum(1 for w in done if (w["kronos_median"] >= w["strike"]) == (w["result"] == "yes")),
            "disagree": sum(1 for w in done if w.get("agrees_with_market") is False),
            "disagree_wins": sum(1 for w in trades if w.get("agrees_with_market") is False and w["pnl"] > 0),
            "cash": L["cash"], "brier": {k: round(sum(v) / len(v), 4) for k, v in br.items()}}


def main(windows):
    from model import Kronos, KronosTokenizer, KronosPredictor
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    device = os.environ.get("KRONOS_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kronos device: {device}", flush=True)
    pred = KronosPredictor(mdl, tok, device=device, max_context=512)
    L = load()
    done = 0
    while done < windows:
        t = now()
        m = current_market()
        opened = dt.datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
        target = opened + dt.timedelta(seconds=RULES["decide_after_s"])
        already = any(w["ticker"] == m["ticker"] for w in L["windows"])
        if already or t > opened + dt.timedelta(minutes=4):
            nxt = opened + dt.timedelta(minutes=15, seconds=RULES["decide_after_s"])
            settle(L)
            time.sleep(max(5, (nxt - now()).total_seconds()))
            continue
        if t < target:
            time.sleep((target - t).total_seconds())
        w = decide(pred, L)
        if w:
            done += 1
            print(json.dumps({k: w[k] for k in ("ticker", "decided_utc", "strike", "spot", "kronos_median", "model_secs", "p", "edge", "trade")}), flush=True)
    # wait for the last window to settle
    while any(w["result"] is None for w in L["windows"]):
        time.sleep(30)
        settle(L)
        if now() > dt.datetime.fromisoformat(L["windows"][-1]["close_utc"]) + dt.timedelta(minutes=10):
            break
    print("SUMMARY", json.dumps(summary(L)), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--decide-after", type=int, default=None, help="seconds after the window opens to decide")
    a = ap.parse_args()
    if a.decide_after is not None:
        RULES["decide_after_s"] = a.decide_after
    main(a.windows)
