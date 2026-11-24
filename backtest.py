"""Backtest Kronos on settled KXBTC15M markets.

Writes results to runs/backtest_<tag>.json. Caches markets+candles under cache/ so config sweeps don't re-fetch.
"""
import os, sys, json, math, time, argparse, pickle, datetime as dt
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("KRONOS_DIR", os.path.join(_HERE, "Kronos-repo")))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np, pandas as pd, torch
import data
from forecast import kalshi_fee

CACHE = os.path.join(_HERE, "cache")
RUNS = os.path.join(_HERE, "runs")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RUNS, exist_ok=True)

TOKENIZERS = {"small": "NeoQuasar/Kronos-Tokenizer-base", "base": "NeoQuasar/Kronos-Tokenizer-base",
              "mini": "NeoQuasar/Kronos-Tokenizer-2k"}
MODELS = {"small": "NeoQuasar/Kronos-small", "base": "NeoQuasar/Kronos-base",
          "mini": "NeoQuasar/Kronos-mini"}


def _iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def fetch_markets(n_events, cache_file):
    """Fetch the most recent n_events settled KXBTC15M markets via /markets. Caches to cache_file."""
    if os.path.exists(cache_file):
        return pickle.load(open(cache_file, "rb"))
    print(f"Fetching {n_events} settled KXBTC15M markets ...", flush=True)
    out = []
    cursor = None
    while len(out) < n_events:
        params = {"series_ticker": "KXBTC15M", "status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = data.kalshi("/markets", **params)
        mkts = r.get("markets", [])
        if not mkts:
            break
        for m in mkts:
            if m.get("result") not in ("yes", "no"):
                continue
            try:
                ev = float(m.get("expiration_value") or 0)
            except (TypeError, ValueError):
                ev = 0.0
            out.append({
                "ticker": m["ticker"],
                "open": _iso(m["open_time"]).replace(tzinfo=None),
                "close": _iso(m["close_time"]).replace(tzinfo=None),
                "strike": float(m["floor_strike"]),
                "result": m["result"],
                "expiration_value": ev,
            })
            if len(out) >= n_events:
                break
        cursor = r.get("cursor")
        if not cursor:
            break
        print(f"  markets {len(out)}/{n_events}", flush=True)
    out.sort(key=lambda m: m["open"])
    pickle.dump(out, open(cache_file, "wb"))
    return out


def fetch_candles(start, end, cache_file):
    """Fetch 1-min BTC-USD OHLCV between start and end (naive UTC). Returns DataFrame."""
    if os.path.exists(cache_file):
        return pickle.load(open(cache_file, "rb"))
    print(f"Fetching 1-min candles {start} -> {end} ...", flush=True)
    seen = {}
    cur = end
    while cur > start:
        lo = max(start, cur - dt.timedelta(minutes=300))
        rows = data._get(f"{data.COINBASE}/products/BTC-USD/candles",
                         {"granularity": 60,
                          "start": lo.replace(tzinfo=dt.timezone.utc).isoformat(),
                          "end": cur.replace(tzinfo=dt.timezone.utc).isoformat()})
        if not rows:
            break
        for t, lo_p, hi_p, op, cl, vol in rows:
            seen[t] = (t, op, hi_p, lo_p, cl, vol)
        cur = lo
        time.sleep(0.2)
    rows = sorted(seen.values())
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="s")
    df["amount"] = df["volume"] * df["close"]
    pickle.dump(df, open(cache_file, "wb"))
    return df


def build_window(mkt, candles, lookback_min, decide_offset_s):
    """Slice `lookback_min` completed 1-min candles ending strictly before (open + decide_offset_s).
    Returns dict or None.
    """
    decide_time = mkt["open"] + dt.timedelta(seconds=decide_offset_s)
    # include candles whose close <= decide_time, i.e. open + 1 min <= decide_time
    cutoff_open = decide_time - dt.timedelta(minutes=1)
    sub = candles[candles["dt"] <= cutoff_open]
    if len(sub) < lookback_min:
        return None
    hist = sub.tail(lookback_min).reset_index(drop=True)
    last_open = hist["dt"].iloc[-1]
    close = mkt["close"]
    pred_len = int(round((close - dt.timedelta(minutes=1) - last_open.to_pydatetime()) / dt.timedelta(minutes=1)))
    if pred_len < 1:
        return None
    y_ts = pd.Series(pd.date_range(last_open + pd.Timedelta(minutes=1), periods=pred_len, freq="min"))
    return {
        "mkt": mkt,
        "x": hist[["open", "high", "low", "close", "volume", "amount"]].copy(),
        "x_stamp": hist["dt"].copy(),
        "y_ts": y_ts,
        "pred_len": pred_len,
        "spot": float(hist["close"].iloc[-1]),
    }


def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def run_kronos(windows, device, model_name, paths, T, top_p, batch_series, seed=0):
    """For each window, run `paths` Kronos samples. Returns (medians, sigmas, secs)."""
    from model import Kronos, KronosTokenizer, KronosPredictor
    torch.manual_seed(seed); np.random.seed(seed)
    tok = KronosTokenizer.from_pretrained(TOKENIZERS[model_name])
    mdl = Kronos.from_pretrained(MODELS[model_name])
    pred = KronosPredictor(mdl, tok, device=device, max_context=512)
    medians, sigmas = [], []
    t_all = time.time()
    # process by pred_len (all +60s windows share one pred_len; +0s share another)
    by_pl = {}
    for i, w in enumerate(windows):
        by_pl.setdefault(w["pred_len"], []).append(i)
    for pl, idxs in by_pl.items():
        # chunk so GPU batch = batch_series * paths
        chunk = max(1, batch_series // max(1, paths // 32))  # keep rough memory parity to 32-path baseline
        for start in range(0, len(idxs), chunk):
            group = idxs[start:start + chunk]
            dfs, xstamps, ystamps = [], [], []
            for i in group:
                w = windows[i]
                dfs.extend([w["x"]] * paths)
                xstamps.extend([w["x_stamp"]] * paths)
                ystamps.extend([w["y_ts"]] * paths)
            outs = pred.predict_batch(dfs, xstamps, ystamps,
                                      pred_len=pl, T=T, top_p=top_p, sample_count=1, verbose=False)
            finals = np.array([o["close"].iloc[-1] for o in outs]).reshape(len(group), paths)
            for j, i in enumerate(group):
                medians.append((i, float(np.median(finals[j]))))
                sigmas.append((i, float(np.std(np.log(finals[j])))))
            done = min(start + chunk, len(idxs))
            print(f"  pred_len={pl}: {done}/{len(idxs)}", flush=True)
    secs = time.time() - t_all
    medians.sort(); sigmas.sort()
    return np.array([m for _, m in medians]), np.array([s for _, s in sigmas]), secs


def metrics(windows, meds, sigs, use_filter):
    n = len(windows)
    strikes = np.array([w["mkt"]["strike"] for w in windows])
    spots = np.array([w["spot"] for w in windows])
    results = np.array([1 if w["mkt"]["result"] == "yes" else 0 for w in windows])
    # realized vol sigma: use spots history across windows (per window).
    # cheap proxy: use spot-based realized vol from the window's own candles.
    realized_sig = []
    for w in windows:
        lr = np.diff(np.log(w["x"]["close"].values))
        realized_sig.append(float(lr.std() * math.sqrt(w["pred_len"])))
    realized_sig = np.array(realized_sig)
    sigs_used = np.maximum(sigs, realized_sig)
    z_k = (np.log(strikes) - np.log(meds)) / sigs_used
    p_kronos = 1 - np.array([norm_cdf(z) for z in z_k])
    z_z = (np.log(strikes) - np.log(spots)) / sigs_used
    p_zero = 1 - np.array([norm_cdf(z) for z in z_z])
    side_k = (meds >= strikes).astype(int)
    side_z = (spots >= strikes).astype(int)
    hit_k = (side_k == results).astype(int)
    hit_z = (side_z == results).astype(int)
    agree = (side_k == side_z).astype(int)
    out = {
        "n": int(n),
        "kronos_dir_right": int(hit_k.sum()),
        "zero_dir_right": int(hit_z.sum()),
        "kronos_brier": round(float(((p_kronos - results) ** 2).mean()), 4),
        "zero_brier": round(float(((p_zero - results) ** 2).mean()), 4),
        "agree_n": int(agree.sum()),
        "agree_hit_k": int(hit_k[agree == 1].sum()),
        "disagree_n": int((1 - agree).sum()),
        "disagree_hit_k": int(hit_k[agree == 0].sum()),
    }
    if use_filter:
        mask = agree == 1
        out["filtered_n"] = int(mask.sum())
        out["filtered_hit_k"] = int(hit_k[mask].sum())
    # pnl approximation: $10 stake per window at a notional $0.50 ask -> win $10, lose $10. (No historical ask prices.)
    # So just report hit rate; skip pnl.
    return out, p_kronos, p_zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=288)
    ap.add_argument("--paths", type=int, default=32)
    ap.add_argument("--lookback", type=int, default=240)
    ap.add_argument("--model", default="small", choices=list(MODELS))
    ap.add_argument("--offset", type=int, default=60, help="decide-after seconds")
    ap.add_argument("--filter", type=int, default=0, help="1 = zero-drift filter (bet only when kronos and spot agree)")
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--batch-series", type=int, default=8, help="max concurrent windows per batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mkt_cache = os.path.join(CACHE, f"markets_{a.n_events}.pkl")
    markets = fetch_markets(a.n_events, mkt_cache)
    if len(markets) < a.n_events:
        print(f"only got {len(markets)} markets", flush=True)
    span_start = markets[0]["open"] - dt.timedelta(minutes=a.lookback + 10)
    span_end = markets[-1]["close"] + dt.timedelta(minutes=5)
    cand_cache = os.path.join(CACHE, f"candles_{int(span_start.timestamp())}_{int(span_end.timestamp())}.pkl")
    candles = fetch_candles(span_start, span_end, cand_cache)
    print(f"markets={len(markets)}  candles={len(candles)}", flush=True)

    windows, skipped = [], 0
    for m in markets:
        w = build_window(m, candles, a.lookback, a.offset)
        if w is None:
            skipped += 1
        else:
            windows.append(w)
    print(f"built {len(windows)} windows  (skipped {skipped})", flush=True)

    meds, sigs, secs = run_kronos(windows, device, a.model, a.paths, a.T, a.top_p, a.batch_series, seed=a.seed)
    m, _, _ = metrics(windows, meds, sigs, bool(a.filter))

    tag = a.tag or f"{a.model}_p{a.paths}_lb{a.lookback}_off{a.offset}_f{a.filter}_n{len(windows)}"
    result = {
        "tag": tag,
        "config": {k: getattr(a, k) for k in ("n_events", "paths", "lookback", "model", "offset", "filter", "T", "top_p", "seed")},
        "device": device,
        "windows": len(windows),
        "forecast_secs": round(secs, 1),
        "metrics": m,
        "range": {"first_open": markets[0]["open"].isoformat(), "last_close": markets[-1]["close"].isoformat()},
    }
    fn = os.path.join(RUNS, f"backtest_{tag}.json")
    json.dump(result, open(fn, "w"), indent=1)
    print(json.dumps(result, indent=1))
    print(f"wrote {fn}")


if __name__ == "__main__":
    main()
