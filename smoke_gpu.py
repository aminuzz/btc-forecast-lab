"""Smoke test: one 32-path Kronos-small forecast on GPU, prints timing.
Usage: python smoke_gpu.py [--paths 32] [--pred-len 15] [--lookback 240]
"""
import os, sys, time, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "Kronos-repo"))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, "hf_cache"))

import numpy as np, pandas as pd, torch
import data
from model import Kronos, KronosTokenizer, KronosPredictor

ap = argparse.ArgumentParser()
ap.add_argument("--paths", type=int, default=32)
ap.add_argument("--pred-len", type=int, default=15)
ap.add_argument("--lookback", type=int, default=240)
ap.add_argument("--device", default=None)
a = ap.parse_args()

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

print(f"Fetching {a.lookback} min of BTC-USD candles ...", flush=True)
rows = data.candles_1m(a.lookback)
df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
df["timestamps"] = pd.to_datetime(df["ts"], unit="s")
df["amount"] = df["volume"] * df["close"]
last_open = df["timestamps"].iloc[-1]
y_ts = pd.Series(pd.date_range(last_open + pd.Timedelta(minutes=1), periods=a.pred_len, freq="min"))
x = df[["open", "high", "low", "close", "volume", "amount"]]

print("Loading Kronos-small ...", flush=True)
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
pred = KronosPredictor(mdl, tok, device=device, max_context=512)

print(f"Warm-up ({a.paths} paths, pred_len={a.pred_len}) ...", flush=True)
t0 = time.time()
_ = pred.predict_batch([x] * a.paths, [df["timestamps"]] * a.paths, [y_ts] * a.paths,
                       pred_len=a.pred_len, T=1.0, top_p=1.0, sample_count=1, verbose=False)
warm_s = time.time() - t0
print(f"  warm-up {warm_s:.2f}s", flush=True)

print(f"Timed run ({a.paths} paths) ...", flush=True)
if device == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
outs = pred.predict_batch([x] * a.paths, [df["timestamps"]] * a.paths, [y_ts] * a.paths,
                          pred_len=a.pred_len, T=1.0, top_p=1.0, sample_count=1, verbose=False)
if device == "cuda":
    torch.cuda.synchronize()
secs = time.time() - t0
finals = np.array([o["close"].iloc[-1] for o in outs])
spot = float(df["close"].iloc[-1])
print(f"device={device}  paths={a.paths}  pred_len={a.pred_len}  "
      f"secs={secs:.2f}  spot={spot:.2f}  median_final={float(np.median(finals)):.2f}  "
      f"spread_pct={100 * float(np.std(np.log(finals))):.3f}")
if device == "cuda":
    free, total = torch.cuda.mem_get_info()
    print(f"gpu mem: used {(total - free) / 1024 ** 2:.0f} MiB / {total / 1024 ** 2:.0f} MiB")
