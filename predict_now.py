"""One-shot BTC-USD forecast 15 minutes ahead. No Kalshi, no ledger — just Kronos on live candles.
Usage: python predict_now.py  (or double-click run_predict.bat)
"""
import os, sys, math, time, datetime as dt
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("KRONOS_DIR", os.path.join(_HERE, "Kronos-repo")))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np, pandas as pd, torch
import data

LOOKBACK = 240
PRED_MIN = 15
PATHS = 32


def main():
    print(f"\nKronos BTC forecast  ({dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("Fetching 240 min of BTC-USD candles ...", flush=True)
    rows = data.candles_1m(LOOKBACK)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["ts"], unit="s")
    df["amount"] = df["volume"] * df["close"]
    spot = float(df["close"].iloc[-1])
    last_open = df["timestamps"].iloc[-1]
    y_ts = pd.Series(pd.date_range(last_open + pd.Timedelta(minutes=1), periods=PRED_MIN, freq="min"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Kronos-small on {device} ({PATHS} paths, {PRED_MIN} min ahead) ...", flush=True)
    from model import Kronos, KronosTokenizer, KronosPredictor
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    pred = KronosPredictor(mdl, tok, device=device, max_context=512)

    t0 = time.time()
    x = df[["open", "high", "low", "close", "volume", "amount"]]
    outs = pred.predict_batch([x] * PATHS, [df["timestamps"]] * PATHS, [y_ts] * PATHS,
                              pred_len=PRED_MIN, T=1.0, top_p=1.0, sample_count=1, verbose=False)
    secs = time.time() - t0
    finals = np.array([o["close"].iloc[-1] for o in outs])
    med = float(np.median(finals))
    p5, p25, p75, p95 = (float(np.quantile(finals, q)) for q in (0.05, 0.25, 0.75, 0.95))
    spread = float(np.std(np.log(finals)))
    up_prob = float((finals > spot).mean())

    close_utc = (last_open + pd.Timedelta(minutes=PRED_MIN + 1)).to_pydatetime().replace(tzinfo=dt.timezone.utc)
    close_local = close_utc.astimezone()
    mins_from_now = (close_utc - dt.datetime.now(dt.timezone.utc)).total_seconds() / 60
    move = med - spot
    pct = 100 * move / spot
    arrow = "UP" if med > spot else "DOWN"
    arrow_sym = "▲" if med > spot else "▼"

    print()
    print(f"  Spot now         : ${spot:>10,.2f}")
    print(f"  Predicted close  : ${med:>10,.2f}  {arrow_sym} {arrow}  ({'+' if move >= 0 else ''}{move:.2f}  /  {'+' if pct >= 0 else ''}{pct:.3f}%)")
    print(f"  at approx         : {close_local.strftime('%H:%M:%S')} local  /  {close_utc.strftime('%H:%M:%S')} UTC  ({mins_from_now:.1f} min from now)")
    print()
    print(f"  Prob close > spot: {100 * up_prob:.0f}%  (fraction of 32 paths that finish above spot)")
    print(f"  Path spread (1σ) : {100 * spread:.3f}%  of price")
    print(f"  Quantiles 5/25/50/75/95 : ${p5:,.2f} / ${p25:,.2f} / ${med:,.2f} / ${p75:,.2f} / ${p95:,.2f}")
    print()
    print(f"  Forecast time    : {secs:.1f}s on {device}")
    print()


if __name__ == "__main__":
    main()
