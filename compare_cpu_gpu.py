"""Replay every window in follow_ledger.json on the GPU and diff its Kronos median
against the CPU median the cloud stored.

Caveat: the original CPU run was unseeded, so every sample differs by both
(device numerics) and (sample randomness). We report the GPU vs CPU diff
plus the within-GPU sampling spread (K re-runs with different seeds), so
you can see how much of the diff is device vs noise.

Usage: python compare_cpu_gpu.py [--k 3] [--ledger follow_ledger.json]
"""
import os, sys, json, math, time, argparse, datetime as dt
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("KRONOS_DIR", os.path.join(_HERE, "Kronos-repo")))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np, pandas as pd, torch
import data


def _iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def fetch_candles_for_window(open_utc, lookback_min, decide_offset_s):
    """1-min BTC-USD candles ending at (open + decide_offset_s), excluding the forming candle."""
    open_dt = _iso(open_utc).replace(tzinfo=None)
    decide = open_dt + dt.timedelta(seconds=decide_offset_s)
    end = decide  # exclusive upper bound; last candle open <= end - 1min
    start = end - dt.timedelta(minutes=lookback_min + 20)
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
        time.sleep(0.15)
    rows = sorted(seen.values())
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="s")
    df["amount"] = df["volume"] * df["close"]
    # keep candles whose close <= decide (open + 1min <= decide)
    cutoff = decide - dt.timedelta(minutes=1)
    df = df[df["dt"] <= cutoff].tail(lookback_min).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=os.path.join(_HERE, "follow_ledger.json"))
    ap.add_argument("--k", type=int, default=3, help="GPU re-runs per window (seeds 0..k-1)")
    ap.add_argument("--paths", type=int, default=32)
    ap.add_argument("--lookback", type=int, default=240)
    ap.add_argument("--offset", type=int, default=60)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    L = json.load(open(a.ledger))
    ws = [w for w in L["windows"] if w.get("kronos_median") is not None]
    if a.max_windows:
        ws = ws[-a.max_windows:]
    print(f"ledger has {len(L['windows'])} windows, {len(ws)} with a stored kronos_median", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"replaying on device={device}, {a.k} runs per window, paths={a.paths}", flush=True)
    from model import Kronos, KronosTokenizer, KronosPredictor
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    pred = KronosPredictor(mdl, tok, device=device, max_context=512)

    rows_out = []
    t0_all = time.time()
    for i, w in enumerate(ws):
        df = fetch_candles_for_window(w["open_utc"], a.lookback, a.offset)
        if len(df) < a.lookback:
            print(f"[{i+1}/{len(ws)}] {w['ticker']}: only {len(df)} candles, skipping", flush=True)
            continue
        last_open = df["dt"].iloc[-1]
        close = _iso(w["close_utc"]).replace(tzinfo=None)
        pred_len = int(round((close - dt.timedelta(minutes=1) - last_open.to_pydatetime()) / dt.timedelta(minutes=1)))
        if pred_len < 1:
            continue
        y_ts = pd.Series(pd.date_range(last_open + pd.Timedelta(minutes=1), periods=pred_len, freq="min"))
        x = df[["open", "high", "low", "close", "volume", "amount"]]

        medians = []
        for seed in range(a.k):
            torch.manual_seed(seed); np.random.seed(seed)
            outs = pred.predict_batch([x] * a.paths, [df["dt"]] * a.paths, [y_ts] * a.paths,
                                      pred_len=pred_len, T=1.0, top_p=1.0, sample_count=1, verbose=False)
            finals = np.array([o["close"].iloc[-1] for o in outs])
            medians.append(float(np.median(finals)))
        gpu_mean = float(np.mean(medians))
        gpu_std = float(np.std(medians))
        cpu_med = float(w["kronos_median"])
        spot = float(w["spot"])
        strike = float(w["strike"])
        # direction agreement vs stored
        cpu_side = "yes" if cpu_med >= strike else "no"
        gpu_side = "yes" if gpu_mean >= strike else "no"
        diff = gpu_mean - cpu_med
        rows_out.append({
            "ticker": w["ticker"], "open_utc": w["open_utc"], "strike": strike,
            "spot": spot, "pred_len": pred_len,
            "cpu_median": cpu_med, "gpu_mean_median": round(gpu_mean, 2),
            "gpu_std_median": round(gpu_std, 2), "gpu_medians": [round(m, 2) for m in medians],
            "diff_usd": round(diff, 2), "diff_bps": round(10000 * diff / cpu_med, 2),
            "cpu_side": cpu_side, "gpu_side": gpu_side, "side_agrees": cpu_side == gpu_side,
            "result": w.get("result"),
        })
        done = len(rows_out); dt_s = time.time() - t0_all
        print(f"[{i+1}/{len(ws)}] {w['ticker']} cpu={cpu_med:.2f} "
              f"gpu={gpu_mean:.2f}±{gpu_std:.2f} ({'=' if cpu_side==gpu_side else '!'}) "
              f"diff {diff:+.2f} ({10000*diff/cpu_med:+.1f} bps)  [{dt_s:.0f}s]", flush=True)

    # summary
    diffs = np.array([r["diff_usd"] for r in rows_out])
    gpu_stds = np.array([r["gpu_std_median"] for r in rows_out])
    bps = np.array([r["diff_bps"] for r in rows_out])
    flips = sum(1 for r in rows_out if not r["side_agrees"])
    summary = {
        "n": len(rows_out),
        "k_per_window": a.k,
        "mean_diff_usd": round(float(diffs.mean()), 2),
        "median_abs_diff_usd": round(float(np.median(np.abs(diffs))), 2),
        "p95_abs_diff_usd": round(float(np.percentile(np.abs(diffs), 95)), 2),
        "median_abs_diff_bps": round(float(np.median(np.abs(bps))), 2),
        "median_gpu_sampling_std_usd": round(float(np.median(gpu_stds)), 2),
        "direction_flips_vs_cpu": int(flips),
        "direction_flip_pct": round(100 * flips / len(rows_out), 1) if rows_out else 0,
    }
    print("\n=== summary ===")
    print(json.dumps(summary, indent=1))
    out_fn = os.path.join(_HERE, "runs", f"compare_cpu_gpu_n{len(rows_out)}_k{a.k}.json")
    os.makedirs(os.path.join(_HERE, "runs"), exist_ok=True)
    json.dump({"summary": summary, "rows": rows_out}, open(out_fn, "w"), indent=1)
    print(f"wrote {out_fn}")


if __name__ == "__main__":
    main()
