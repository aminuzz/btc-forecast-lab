"""Kronos probability engine for Kalshi daily BTC/ETH above/below markets.

Usage: python3 forecast.py --asset BTC --model small --paths 32 --lookback 256
Writes runs/<date>_<asset>_<model>.json
"""
import sys, os, json, math, time, argparse, datetime as dt
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("KRONOS_DIR", os.path.join(_HERE, "Kronos-repo")))
import numpy as np, pandas as pd, torch
import data

SERIES = {"BTC": ("KXBTCD", "BTC-USD"), "ETH": ("KXETHD", "ETH-USD")}
MODELS = {"small": "NeoQuasar/Kronos-small", "base": "NeoQuasar/Kronos-base",
          "mini": "NeoQuasar/Kronos-mini"}
TOKENIZERS = {"small": "NeoQuasar/Kronos-Tokenizer-base", "base": "NeoQuasar/Kronos-Tokenizer-base",
              "mini": "NeoQuasar/Kronos-Tokenizer-2k"}


def pick_event(series, min_hours=3):
    """Next daily 5pm-EST event at least `min_hours` away."""
    now = dt.datetime.now(dt.timezone.utc)
    best = None
    for e in data.open_events(series):
        t = dt.datetime.fromisoformat(e["strike_date"].replace("Z", "+00:00"))
        if e["event_ticker"].endswith("17") and t - now >= dt.timedelta(hours=min_hours):
            if best is None or t < best[1]:
                best = (e["event_ticker"], t)
    return best


def kalshi_fee(p, contracts=1):
    """Kalshi general taker fee estimate: ceil(0.07 * C * P * (1-P)) to the cent."""
    return math.ceil(0.07 * contracts * p * (1 - p) * 100) / 100


def run(asset, model_name, paths, lookback, event=None, seed=0, T=1.0, top_p=1.0):
    from model import Kronos, KronosTokenizer, KronosPredictor
    series, product = SERIES[asset]
    ev, target = event or pick_event(series)
    now = dt.datetime.now(dt.timezone.utc)

    rows = data.candles(product, hours=lookback + 2)
    # keep the still-forming candle: its close is the latest price
    rows = rows[-lookback:]
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["ts"], unit="s")
    df["amount"] = df["volume"] * df["close"]
    last_open = df["timestamps"].iloc[-1]
    target_naive = target.replace(tzinfo=None)
    # predicted candle that opens at target-1h closes at target
    pred_len = int(round((target_naive - pd.Timedelta(hours=1) - last_open) / pd.Timedelta(hours=1)))
    y_ts = pd.Series(pd.date_range(last_open + pd.Timedelta(hours=1), periods=pred_len, freq="h"))

    torch.manual_seed(seed); np.random.seed(seed)
    tok = KronosTokenizer.from_pretrained(TOKENIZERS[model_name])
    mdl = Kronos.from_pretrained(MODELS[model_name])
    device = os.environ.get("KRONOS_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    pred = KronosPredictor(mdl, tok, device=device, max_context=512)
    x = df[["open", "high", "low", "close", "volume", "amount"]]
    t0 = time.time()
    outs = pred.predict_batch([x] * paths, [df["timestamps"]] * paths, [y_ts] * paths,
                              pred_len=pred_len, T=T, top_p=top_p, sample_count=1, verbose=False)
    secs = time.time() - t0
    finals = np.array([o["close"].iloc[-1] for o in outs])

    # realized vol from 30 days of hourly closes -> calibrated spread
    hist = data.candles(product, hours=24 * 30)
    lr = np.diff(np.log([h[4] for h in hist]))
    sig_real = float(lr.std() * math.sqrt(pred_len))
    med = float(np.median(finals))
    sig_model = float(np.std(np.log(finals)))
    sig = max(sig_real, sig_model)
    from math import erf, log, sqrt
    def p_cal(k):
        z = (log(k) - log(med)) / sig
        return 1 - 0.5 * (1 + erf(z / sqrt(2)))

    _, markets = data.event_markets(ev)
    spot = float(df["close"].iloc[-1])
    out_m = []
    for m in markets:
        if m.get("strike_type") != "greater" or m.get("status") != "active":
            continue
        k = float(m["floor_strike"])
        ya, yb = float(m["yes_ask_dollars"]), float(m["yes_bid_dollars"])
        na = float(m["no_ask_dollars"])
        if not (0.02 <= ya <= 0.98 or 0.02 <= yb <= 0.98):
            continue  # skip strikes that are already near-certain
        p = (np.sum(finals > k) + 0.5) / (paths + 1)  # raw path frequency (overconfident)
        pc = p_cal(k)
        out_m.append({
            "ticker": m["ticker"], "strike": k, "yes_bid": yb, "yes_ask": ya, "no_ask": na,
            "p_yes_raw": round(float(p), 4),
            "p_yes_model": round(p_cal(k), 4),
            "p_yes_market_mid": round((ya + yb) / 2, 4),
            "edge_buy_yes": round(pc - ya - kalshi_fee(ya), 4),
            "edge_buy_no": round((1 - pc) - na - kalshi_fee(na), 4),
            "volume": float(m.get("volume_fp", 0) or 0),
        })
    result = {
        "asset": asset, "model": model_name, "event": ev, "target_utc": target.isoformat(),
        "run_utc": now.isoformat(timespec="seconds"), "spot_coinbase": spot, "spot_note": "last trade in the current hourly candle",
        "hours_ahead": pred_len, "paths": paths, "lookback_hours": lookback, "T": T, "top_p": top_p,
        "median_model": round(med, 2), "sigma_model_pct": round(sig_model * 100, 2), "sigma_realized_pct": round(sig_real * 100, 2), "secs": round(secs, 1),
        "path_quantiles": {q: round(float(np.quantile(finals, q / 100)), 2) for q in (5, 25, 50, 75, 95)},
        "markets": sorted(out_m, key=lambda r: r["strike"]),
        "method": "p_yes_model = lognormal centered on Kronos median path, width = max(Kronos spread, 30-day realized vol). p_yes_raw = raw share of Kronos paths above strike.",
        "note": "Model uses Coinbase hourly closes; Kalshi settles on CF Benchmarks 60s average, so small basis differences exist.",
    }
    os.makedirs("runs", exist_ok=True)
    fn = f"runs/{now:%Y%m%d_%H%M}_{asset}_{model_name}.json"
    json.dump(result, open(fn, "w"), indent=1)
    return fn, result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC"); ap.add_argument("--model", default="small")
    ap.add_argument("--paths", type=int, default=32); ap.add_argument("--lookback", type=int, default=256)
    a = ap.parse_args()
    fn, r = run(a.asset, a.model, a.paths, a.lookback)
    print(fn, r["secs"], "s", r["path_quantiles"], len(r["markets"]), "markets")
