"""Run a config sweep for the Kronos backtest in one process (share loaded model+cached data when configs match).
Writes each config's result to runs/backtest_<tag>.json and prints a summary table.
"""
import os, sys, json, time, argparse, datetime as dt
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("KRONOS_DIR", os.path.join(_HERE, "Kronos-repo")))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np, pandas as pd, torch
import backtest as bt

RUNS = os.path.join(_HERE, "runs")

# (model, paths, lookback, offset)
CONFIGS = [
    ("small", 32, 240, 60),   # baseline
    ("small", 256, 240, 60),  # more paths
    ("small", 32, 480, 60),   # more lookback
    ("small", 32, 240, 0),    # decide at open
    ("base",  32, 240, 60),   # bigger model
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=288)
    ap.add_argument("--batch-series", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mkt_cache = os.path.join(bt.CACHE, f"markets_{a.n_events}.pkl")
    markets = bt.fetch_markets(a.n_events, mkt_cache)
    max_lb = max(c[2] for c in CONFIGS)
    span_start = markets[0]["open"] - dt.timedelta(minutes=max_lb + 10)
    span_end = markets[-1]["close"] + dt.timedelta(minutes=5)
    cand_cache = os.path.join(bt.CACHE, f"candles_{int(span_start.timestamp())}_{int(span_end.timestamp())}.pkl")
    candles = bt.fetch_candles(span_start, span_end, cand_cache)
    print(f"markets={len(markets)}  candles={len(candles)}", flush=True)

    summary = []
    for model_name, paths, lb, off in CONFIGS:
        windows = []
        for m in markets:
            w = bt.build_window(m, candles, lb, off)
            if w is not None:
                windows.append(w)
        tag_check = f"{model_name}_p{paths}_lb{lb}_off{off}_n{len(windows)}"
        existing = os.path.join(RUNS, f"backtest_{tag_check}.json")
        if os.path.exists(existing):
            print(f"\n=== {tag_check}: already done, loading ===", flush=True)
            summary.append(json.load(open(existing)))
            continue
        print(f"\n=== {model_name} paths={paths} lb={lb} off={off}   windows={len(windows)} ===", flush=True)
        meds, sigs, secs = bt.run_kronos(windows, device, model_name, paths, 1.0, 1.0, a.batch_series, seed=a.seed)
        metrics, p_k, p_z = bt.metrics(windows, meds, sigs, use_filter=False)
        tag = f"{model_name}_p{paths}_lb{lb}_off{off}_n{len(windows)}"
        row = {
            "tag": tag,
            "config": {"model": model_name, "paths": paths, "lookback": lb, "offset": off, "n_events": a.n_events},
            "windows": len(windows),
            "forecast_secs": round(secs, 1),
            "metrics": metrics,
        }
        json.dump(row, open(os.path.join(RUNS, f"backtest_{tag}.json"), "w"), indent=1)
        summary.append(row)
        print(json.dumps(metrics, indent=1), flush=True)

    # table
    print("\n" + "=" * 95)
    print(f"{'tag':48} {'win':>5} {'k_dir':>6} {'z_dir':>6} {'k_bri':>7} {'z_bri':>7} {'dis_n':>5} {'dis_k':>5}")
    print("-" * 95)
    for r in summary:
        m = r["metrics"]
        print(f"{r['tag']:48} {m['n']:5d} {m['kronos_dir_right']:6d} {m['zero_dir_right']:6d} "
              f"{m['kronos_brier']:7.4f} {m['zero_brier']:7.4f} {m['disagree_n']:5d} {m['disagree_hit_k']:5d}")
    json.dump(summary, open(os.path.join(RUNS, f"sweep_n{a.n_events}.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(RUNS, f'sweep_n{a.n_events}.json')}")


if __name__ == "__main__":
    main()
