# Kronos BTC desk: handoff for a Claude Code session on this PC

Paper trading only. Never create exchange accounts, place real orders, or handle API keys.

## What this is
A paper-trading desk for Kalshi's 15-minute BTC up/down markets (series `KXBTC15M`).
Each window: ~1 min after the open, Kronos-small (https://github.com/shiyu-coder/Kronos) reads the last 240 one-minute
BTC-USD candles from Coinbase (`api.exchange.coinbase.com`), samples 32 price paths to the window close, and the desk
buys (on paper, $10, at the ask) whichever side the middle path's close points to ("Follow Kronos").
Kalshi public data: `https://external-api.kalshi.com/trade-api/v2` (no key needed for market data).

Files here:
- `data.py`: Kalshi + Coinbase access (with retry/backoff for Kalshi 429s)
- `forecast.py`: `kalshi_fee()` (Kalshi taker fee estimate) and the older daily-market forecaster
- `fifteen.py`: the 15-minute desk. `python fifteen.py --windows N [--decide-after SECONDS]`. Ledger: `follow_ledger.json`
- `push.py`: builds the JSON summary the online dashboard uses

## Results so far (cloud, CPU, 24 windows, Nov 8-9 2026)
- Paper P&L +$10.01; Kronos direction right 15/24.
- Windows where Kronos disagreed with the market: 4 won of 11, net -$17.61. This is the real test of edge.
- Brier (lower is better): market 0.175, zero-drift baseline 0.232, Kronos 0.243.
- Known problem: "bounce bias". After a drop, Kronos's middle path often lands back above the start price.
- Raw Kronos path spread is ~3x too narrow vs realized vol; probabilities are widened to realized vol before use.

## Goals for the GPU (RTX 3060) setup on Windows
1. Create a venv in this folder; install CUDA PyTorch (check `nvidia-smi` for the driver's CUDA version first),
   plus `einops==0.8.1 huggingface_hub==0.33.1 safetensors pandas tqdm`. Set `HF_HOME` to `.\hf_cache` in this folder.
   Clone Kronos into `.\Kronos-repo` and set env `KRONOS_DIR` to it (fifteen.py reads `KRONOS_DIR`).
   Make `fifteen.py` use `device="cuda"` when available (it currently hardcodes `cpu` in `main()`).
2. Speed test: one 32-path forecast on GPU vs ~60 s on the cloud CPU.
3. Backtest on historical 1-min BTC candles + settled KXBTC15M results: compare 32 vs 256 paths, 240 vs 480 candles,
   small vs base, deciding at the open vs +60 s, and a "zero-drift filter" (only bet when Kronos and spot-vs-start agree).
4. Optional: fine-tune Kronos-small on recent BTC 1-min data (repo has `finetune/` scripts) if backtests justify it.
5. A local GUI: a small local web dashboard (auto-refreshing) reading `follow_ledger.json`: wins/losses, P&L over time,
   direction record, disagreement record, Brier by source, and a table of every window.
6. Easy launchers: a desktop shortcut / .bat to run the next N windows, and optionally Windows Task Scheduler.

## Ledger
The latest ledger lives in the user's claude.ai Project at `claude/kalshi-desk/follow_ledger.json`.
Copy it here as `follow_ledger.json` before running, so history continues.
