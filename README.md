# btc-forecast-lab

A GPU workbench for forecasting short-horizon BTC prices and paper-trading Kalshi's 15-minute up/down markets (`KXBTC15M`) with [Kronos](https://github.com/shiyu-coder/Kronos), a pretrained time-series foundation model.

**Paper trading only** — no exchange accounts, no API keys, no real orders. Public market data only (Coinbase 1-min candles + Kalshi public market API, both unauthenticated).

## What's here

- **Live paper-trading loop** (`fifteen.py`) — pulls the current Kalshi KXBTC15M window, fetches 240 min of Coinbase 1-min BTC candles, runs Kronos-small on CUDA (32 Monte Carlo paths), buys the predicted side on paper, settles after close
- **Interactive web UI** (`app.py` + `bot.html`) — Flask server with a click-to-predict-and-trade dashboard; model stays warm in GPU memory so each prediction is ~2 s
- **One-shot forecast** (`predict_now.py`) — standalone "what does Kronos think BTC does in the next 15 min from right now" with no market context
- **Historical backtest pipeline** (`backtest.py` + `sweep.py`) — replays settled KXBTC15M markets through configurable Kronos configs, reports direction hit rate, Brier score, agreement/disagreement splits
- **GPU vs CPU equivalence check** (`compare_cpu_gpu.py`) — replays a recorded CPU ledger on the GPU k times with different seeds, measures whether device drift is distinguishable from sampling noise
- **Local dashboard** (`dashboard.html`) — auto-refreshing read-only view of the trade ledger: cumulative P&L, rolling direction accuracy, Brier per source

## The research result

I ran a 5-config sweep on 288 settled KXBTC15M windows (3 days of history). **Pretrained Kronos has no edge on 15-min BTC**.

| config | dir_right | zero_drift dir_right | Brier Kronos | Brier zero-drift | disagreement hit % |
|---|---|---|---|---|---|
| small, 32 paths, 240 lb, +60 s (baseline) | 156 / 288 | **181 / 288** | 0.268 | **0.232** | 38.3% |
| small, **256 paths**, 240 lb, +60 s | 158 / 288 | 181 / 288 | 0.266 | 0.232 | 39.3% |
| small, 32 paths, **480 lb**, +60 s | 142 / 288 | 181 / 288 | 0.404 | 0.232 | 36.9% |
| small, 32 paths, 240 lb, **0 s (decide at open)** | 156 / 288 | 158 / 288 | 0.282 | 0.248 | 49.2% |
| **base model**, 32 paths, 240 lb, +60 s | 158 / 288 | 181 / 288 | 0.263 | 0.232 | 38.4% |

- Zero-drift (the "spot at decision vs strike" baseline) beats Kronos on direction and calibration in every config
- On the **disagreement subset** (windows where Kronos picks a different side than zero-drift — the only place Kronos could add value), Kronos is right **36-39%** of the time, worse than a coin flip, in every serious config
- More paths (32 → 256) moves the needle ~1 pp
- More context (240 → 480 min) makes it *worse* (Brier 0.40, Kronos becomes catastrophically overconfident — this matches the HANDOFF's "bounce bias" note: after a drop, Kronos's middle path keeps landing back above the start price)
- The bigger **base** model performs the same as small
- The only near-parity config is decide-at-open — but zero-drift also degrades there, so no net edge

A separate validation run confirmed that these numbers aren't a GPU artifact: GPU-vs-CPU diffs on the same 27 cloud-recorded inputs were $27 median / 3.7 bps of price — smaller than the GPU's own seed-to-seed noise ($23 median).

**Conclusion**: Follow-Kronos as a strategy is losing money on average on the disagreement windows it depends on for edge. Next useful experiment would be fine-tuning Kronos-small on recent BTC 1-min data (the `Kronos-repo/finetune/` scripts); the pretrained model is not the signal.

## Stack

- **Python 3.13**, **PyTorch 2.11 + CUDA 12.8**, **RTX 3060 12 GB**
- **Kronos** foundation model (small ≈ 25 M params, base ≈ 100 M)
- **Flask** for the interactive bot server
- **Vanilla HTML / JS / Canvas** for the dashboard (no React, no Chart.js)
- **Kalshi** + **Coinbase** public APIs (read-only, no keys)

## Layout

| file | purpose |
|---|---|
| `fifteen.py` | Live paper-trading loop on the Kalshi 15-min grid |
| `app.py` | Flask server for the interactive bot |
| `bot.html` | Interactive click-to-predict / click-to-trade UI |
| `dashboard.html` | Read-only auto-refreshing ledger view |
| `predict_now.py` | One-shot BTC forecast at an arbitrary time |
| `backtest.py` | Historical backtest driver (one config at a time) |
| `sweep.py` | Loops `backtest.py` over a list of configs in one process (shares model + candle cache) |
| `compare_cpu_gpu.py` | Replays a CPU ledger on GPU k times to measure device drift vs sampling noise |
| `smoke_gpu.py` | One-off GPU benchmark / smoke test |
| `data.py` | Kalshi + Coinbase HTTP wrappers with retry/backoff |
| `forecast.py` | Original hourly-market forecaster + Kalshi fee estimate |
| `push.py` | JSON summary builder for an online dashboard (requires claude.ai artifact DB) |
| `run_*.bat` | Windows one-click launchers for each entry point |
| `HANDOFF.md` | Original project brief and goals that seeded this repo |

## Setup

Requires Windows + NVIDIA GPU with a CUDA 12.x-capable driver. Tested on RTX 3060 with Python 3.13.

```bat
rem 1. Clone this repo and cd into it
git clone https://github.com/<you>/btc-forecast-lab.git
cd btc-forecast-lab

rem 2. Create a venv and install PyTorch with CUDA, then the rest
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

rem 3. Clone the Kronos model code into Kronos-repo/
git clone https://github.com/shiyu-coder/Kronos.git Kronos-repo

rem 4. Weights are downloaded on first use from Hugging Face into .\hf_cache\ (~1 GB).
rem    All launchers set HF_HOME to that folder automatically.
```

## Running it

| what you want | double-click |
|---|---|
| Interactive bot UI (predict on demand, click to trade) | `run_bot.bat` → http://localhost:8766/ |
| One-shot "what does Kronos think 15 min from now" | `run_predict.bat` |
| Auto-trade the next N 15-min Kalshi windows | `run_window.bat N` |
| Read-only dashboard of the ledger | `run_dashboard.bat` → http://localhost:8765/dashboard.html |
| Backtest one config on last 3 days of history | `run_backtest.bat [n_events] [paths] [lookback] [model] [offset]` |

## Credits

- **Kronos** — Shiyu Chen et al., *Kronos: Scalable Foundation Models for Candlesticks*  
  [paper](https://arxiv.org/abs/2508.02739) · [code](https://github.com/shiyu-coder/Kronos) · MIT license
- **Kalshi** — public market data API for `KXBTC15M` series
- **Coinbase Exchange** — public 1-minute BTC-USD candles API
