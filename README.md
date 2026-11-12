# roblox-up-and-coming-forecast-01-entry

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/FandeadAlter/roblox-up-and-coming-forecast-01-entry/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/FandeadAlter/roblox-up-and-coming-forecast-01-entry/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

A latent log-rank Monte Carlo forecaster for Roblox's Up-and-Coming top 10: it simulates each game's rank trajectory over the scored window — momentum drift, fitted diffusion, breakout jumps, 28-day aging-out and new entrants — and averages the per-place marginals into honest probabilities. Backtests about 15 log-score points above the house baseline on the kit history.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

(`input/` is the kit's input directory: `history.ndjson` plus `task.json`. The program reads the scored window from `task.json`, needs no network, and writes `out/forecast.json`.)

---

<sub>Entered on [make.fun](https://make.fun/?ref={referral_code}&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
