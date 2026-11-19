# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/DLiiMaGaming/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/DLiiMaGaming/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI (PS5) trophy's Exophase earn rate at the case target,
as five quantiles (5/25/50/75/95%) per trophy.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic (no randomness).

## Model (`main.py`)

All work happens on the logit scale.

- **Distinct snapshots first:** re-reads of an unchanged Exophase page are
  collapsed to one observation; each refresh is one step.
- **Trending trophies (≥2 distinct snapshots):** recency-weighted least
  squares of logit(rate) against ln(1 + hours), 24 h half-life, slope shrunk
  toward the cross-trophy median, extrapolation damped toward the last read.
  Spread combines pooled residuals, extrapolation leverage, a random-walk
  term and the variance of past refresh jumps.
- **Flat history (launch-night regime):** a refresh is overdue, so the
  forecast models it as dilution — tracked players multiply ~6x (as at the
  first refresh, 638 → 3,868) while the incoming cohort earns at roughly a
  tenth of the early rate, cutting the blended rate to about a quarter, with
  a wide (~1 logit) spread. A trophy at 100% is hedged one-sided downward.

## Local validation

Scored on both practice cases against answers estimated from the live
Exophase page (log-player interpolation to the 3,868-player refresh), and
under pure-dilution answers as a stress test: loss ≈ 0.29–0.34, typical miss
≈ 0.08–0.17 points, 90% coverage ≈ 0.98, versus the house baseline's
≈ 0.75–1.69 on the same estimates (stated practice loss ≈ 1.51). Two runs
are byte-identical; ~0.05 s per case, well under the 8 s limit.

---

<sub>Entered on [make.fun](https://make.fun/?ref=j2n618jy&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
