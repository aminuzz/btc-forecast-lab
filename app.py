"""Interactive Kronos BTC desk. Runs a local web server at http://localhost:8766/.
Keeps Kronos-small loaded so each Predict click is fast. Logs trades to bot_ledger.json.
"""
import os, sys, json, math, time, threading, datetime as dt
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("KRONOS_DIR", os.path.join(_HERE, "Kronos-repo")))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np, pandas as pd, torch
from flask import Flask, jsonify, request, send_from_directory

import data
from forecast import kalshi_fee

LEDGER = os.path.join(_HERE, "bot_ledger.json")
START_CASH = 1000.0
STAKE_USD = 10.0
LOOKBACK = 240
PATHS = 32

app = Flask(__name__)
_lock = threading.Lock()
_pred_cache = {}  # ticker -> most recent prediction
_predictor = None


def _iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def now_utc():
    return dt.datetime.now(dt.timezone.utc)


def load_ledger():
    if os.path.exists(LEDGER):
        return json.load(open(LEDGER))
    return {"cash": START_CASH, "start_cash": START_CASH,
            "rules": {"stake_usd": STAKE_USD, "paths": PATHS, "lookback_min": LOOKBACK, "mode": "interactive_follow_kronos"},
            "windows": []}


def save_ledger(L):
    json.dump(L, open(LEDGER, "w"), indent=1)


def get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor
    print("Loading Kronos-small ...", flush=True)
    from model import Kronos, KronosTokenizer, KronosPredictor
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _predictor = KronosPredictor(mdl, tok, device=device, max_context=512)
    print(f"Kronos loaded on {device}.", flush=True)
    return _predictor


def current_market():
    """Return the latest open KXBTC15M market, or None."""
    try:
        evs = data.open_events("KXBTC15M")
    except Exception as e:
        return None
    if not evs:
        return None
    try:
        _, ms = data.event_markets(evs[0]["event_ticker"])
    except Exception:
        return None
    return ms[0] if ms else None


def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def run_forecast(close_time_utc):
    """Pull latest candles, run Kronos, predict up to close_time_utc. Returns a dict."""
    rows = []
    end = now_utc().replace(second=0, microsecond=0) + dt.timedelta(minutes=1)
    seen = {}
    cur = end
    while len(seen) < LOOKBACK:
        lo = cur - dt.timedelta(minutes=300)
        got = data._get(f"{data.COINBASE}/products/BTC-USD/candles",
                        {"granularity": 60, "start": lo.isoformat(), "end": cur.isoformat()})
        if not got:
            break
        for t, l, h, o, c, v in got:
            seen[t] = (t, o, h, l, c, v)
        cur = lo
    rows = sorted(seen.values())[-LOOKBACK:]
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["ts"], unit="s")
    df["amount"] = df["volume"] * df["close"]
    last_open = df["timestamps"].iloc[-1]
    close_naive = close_time_utc.replace(tzinfo=None)
    pred_len = max(1, int(round((close_naive - pd.Timedelta(minutes=1) - last_open) / pd.Timedelta(minutes=1))))
    y_ts = pd.Series(pd.date_range(last_open + pd.Timedelta(minutes=1), periods=pred_len, freq="min"))
    x = df[["open", "high", "low", "close", "volume", "amount"]]

    pred = get_predictor()
    t0 = time.time()
    outs = pred.predict_batch([x] * PATHS, [df["timestamps"]] * PATHS, [y_ts] * PATHS,
                              pred_len=pred_len, T=1.0, top_p=1.0, sample_count=1, verbose=False)
    secs = time.time() - t0
    finals = np.array([o["close"].iloc[-1] for o in outs])
    med = float(np.median(finals))
    spot = float(df["close"].iloc[-1])
    lr = np.diff(np.log(df["close"].values))
    mins_left = max(1.0, (close_time_utc - now_utc()).total_seconds() / 60)
    sig = max(float(lr.std() * math.sqrt(mins_left)), float(np.std(np.log(finals))))
    quantiles = {q: float(np.quantile(finals, q / 100)) for q in (5, 25, 50, 75, 95)}
    return {
        "spot": spot, "median": med, "sigma": sig, "pred_len": pred_len,
        "mins_left": round(mins_left, 1), "model_secs": round(secs, 1),
        "quantiles": quantiles,
        "last_candle_utc": last_open.isoformat(),
    }


def settle_open_windows(L):
    """Settle any pending windows whose close + 90s has passed."""
    changed = False
    for w in L["windows"]:
        if w.get("result") is not None:
            continue
        close = _iso(w["close_utc"])
        if now_utc() < close + dt.timedelta(seconds=90):
            continue
        try:
            r = data.kalshi(f"/markets/{w['ticker']}").get("market", {})
        except Exception:
            continue
        result = r.get("result")
        if result in ("yes", "no"):
            w["result"] = result
            try:
                w["actual"] = float(r.get("expiration_value") or 0) or None
            except (TypeError, ValueError):
                w["actual"] = None
            if w.get("trade"):
                pay = w["trade"]["contracts"] * (1.0 if result == w["trade"]["side"] else 0.0)
                L["cash"] = round(L["cash"] + pay, 2)
                w["pnl"] = round(pay - w["trade"]["cost"], 2)
            else:
                w["pnl"] = 0.0
            changed = True
    if changed:
        save_ledger(L)
    return changed


@app.route("/")
def index():
    return send_from_directory(_HERE, "bot.html")


@app.route("/api/market")
def api_market():
    m = current_market()
    if not m:
        return jsonify({"ok": False, "error": "No open KXBTC15M market"})
    try:
        strike = float(m.get("floor_strike")) if m.get("floor_strike") else None
        return jsonify({
            "ok": True,
            "ticker": m["ticker"],
            "open_utc": m["open_time"],
            "close_utc": m["close_time"],
            "strike": strike,
            "yes_bid": float(m.get("yes_bid_dollars") or 0),
            "yes_ask": float(m.get("yes_ask_dollars") or 0),
            "no_ask": float(m.get("no_ask_dollars") or 0),
            "mins_left": round((_iso(m["close_time"]) - now_utc()).total_seconds() / 60, 1),
            "has_started": _iso(m["open_time"]) <= now_utc(),
        })
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"ok": False, "error": f"Bad market data: {e}"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    with _lock:
        m = current_market()
        if not m:
            return jsonify({"ok": False, "error": "No open market"})
        close_utc = _iso(m["close_time"])
        strike = float(m.get("floor_strike")) if m.get("floor_strike") else None
        if strike is None:
            return jsonify({"ok": False, "error": "Market has no strike yet (usually posts a few seconds after open)"})
        try:
            f = run_forecast(close_utc)
        except Exception as e:
            return jsonify({"ok": False, "error": f"Forecast failed: {e}"})
        side = "yes" if f["median"] >= strike else "no"
        p_kronos = 1 - norm_cdf((math.log(strike) - math.log(f["median"])) / f["sigma"])
        p_zero = 1 - norm_cdf((math.log(strike) - math.log(f["spot"])) / f["sigma"])
        ya = float(m.get("yes_ask_dollars") or 0)
        na = float(m.get("no_ask_dollars") or 0)
        yb = float(m.get("yes_bid_dollars") or 0)
        mid = (ya + yb) / 2 if ya and yb else None
        price = ya if side == "yes" else na
        n = max(1, int(STAKE_USD // price)) if price > 0 else 0
        fee = kalshi_fee(price, n) if price > 0 else 0.0
        cost = round(n * price + fee, 2) if n else 0.0
        payload = {
            "ok": True,
            "ticker": m["ticker"],
            "open_utc": m["open_time"],
            "close_utc": m["close_time"],
            "strike": strike,
            "spot": round(f["spot"], 2),
            "kronos_median": round(f["median"], 2),
            "sigma_pct": round(100 * f["sigma"], 3),
            "mins_left": f["mins_left"],
            "model_secs": f["model_secs"],
            "yes_bid": yb, "yes_ask": ya, "no_ask": na,
            "market_mid": round(mid, 4) if mid is not None else None,
            "p": {"kronos": round(p_kronos, 4), "zero_drift": round(p_zero, 4),
                  "market": round(mid, 4) if mid is not None else None},
            "side": side,
            "price": price,
            "contracts": n,
            "fee": fee,
            "cost": cost,
            "agrees_with_market": (side == ("yes" if (mid or 0) >= 0.5 else "no")) if mid is not None else None,
            "quantiles": {k: round(v, 2) for k, v in f["quantiles"].items()},
        }
        _pred_cache[m["ticker"]] = payload
        return jsonify(payload)


@app.route("/api/trade", methods=["POST"])
def api_trade():
    with _lock:
        body = request.get_json(force=True, silent=True) or {}
        ticker = body.get("ticker")
        if not ticker or ticker not in _pred_cache:
            return jsonify({"ok": False, "error": "No prediction cached for this ticker. Click Predict first."})
        pred = _pred_cache[ticker]
        L = load_ledger()
        if any(w["ticker"] == ticker for w in L["windows"]):
            return jsonify({"ok": False, "error": "Already traded this window"})
        # re-check close time
        close_utc = _iso(pred["close_utc"])
        if now_utc() >= close_utc:
            return jsonify({"ok": False, "error": "Window already closed"})
        if pred["cost"] > L["cash"]:
            return jsonify({"ok": False, "error": f"Not enough cash ({L['cash']}) to cover trade cost ({pred['cost']})"})
        w = {
            "ticker": pred["ticker"], "open_utc": pred["open_utc"], "close_utc": pred["close_utc"],
            "decided_utc": now_utc().isoformat(timespec="seconds"),
            "strike": pred["strike"], "spot": pred["spot"],
            "kronos_median": pred["kronos_median"], "sigma_pct": pred["sigma_pct"], "mins_left": pred["mins_left"],
            "model_secs": pred["model_secs"],
            "yes_bid": pred["yes_bid"], "yes_ask": pred["yes_ask"], "no_ask": pred["no_ask"],
            "p": pred["p"], "agrees_with_market": pred["agrees_with_market"],
            "trade": {"side": pred["side"], "price": pred["price"], "contracts": pred["contracts"],
                      "fee": pred["fee"], "cost": pred["cost"]},
            "result": None, "pnl": None, "actual": None,
        }
        L["cash"] = round(L["cash"] - pred["cost"], 2)
        L["windows"].append(w)
        save_ledger(L)
        return jsonify({"ok": True, "window": w, "cash": L["cash"]})


@app.route("/api/ledger")
def api_ledger():
    L = load_ledger()
    try:
        settle_open_windows(L)
    except Exception as e:
        pass  # non-fatal
    return jsonify(L)


@app.route("/api/settle", methods=["POST"])
def api_settle():
    L = load_ledger()
    changed = settle_open_windows(L)
    return jsonify({"ok": True, "changed": changed, "cash": L["cash"]})


if __name__ == "__main__":
    # Warm up Kronos at startup so the first prediction is also fast.
    try:
        get_predictor()
    except Exception as e:
        print(f"WARN: Could not preload Kronos: {e}", flush=True)
    app.run(host="127.0.0.1", port=8766, debug=False, threaded=False)
