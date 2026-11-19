#!/usr/bin/env python3
"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case target.

Model, in brief
---------------
Everything is done on the logit scale, because a miss from 1% to 2% should
cost about what a miss from 33% to 50% does, and because Exophase rates are
ratios that move multiplicatively when the tracked population grows.

1. The raw history is first collapsed to *distinct snapshots*: Exophase
   refreshes its counts in jumps, so a page that has not changed is a single
   observation, and each refresh is one step.  Re-reads of the same page
   carry no new information and must not overweight the fit.
2. When a trophy has at least two distinct snapshots, its logit rate is fit
   by recency-weighted least squares against ln(1 + hours since the first
   snapshot).  Weights decay with a 24 h half-life, the slope is shrunk
   toward the cross-trophy median slope (most trophies drift together), and
   the extrapolation to the target is damped back toward the last read, the
   more so the longer the horizon.  The spread combines the pooled residual
   variance, the leverage of extrapolating, a random-walk term growing with
   the horizon, and the variance of the logit jumps seen at past refreshes.
3. When a trophy's history is flat (a single distinct snapshot, e.g. the
   launch-night practice cases), a refresh is overdue.  The forecast models
   the refresh as dilution: the tracked population multiplies by roughly the
   factor seen at the first refresh (~6x), while the incoming cohort earns
   each trophy at only a fraction (~0.18) of the early rate, so the blended
   rate falls to about a third of its pre-refresh value.  Repeated refreshes
   are applied when the stale time plus the horizon spans several refresh
   intervals, with a small upward drift for the old cohort's continued
   progress between refreshes.  The spread is wide (~1 logit) because the
   size and timing of the unseen refresh are both uncertain.
4. A trophy sitting at 100% can only fall, so its median is hedged a touch
   below 100% and its band is one-sided downward.

Deterministic: no randomness anywhere, so two runs are byte-identical.
Standard library only; reads $MAKEFUN_INPUT_DIR (else /input/) *.in files
and writes one /output/<id>.out per case ($MAKEFUN_OUTPUT_DIR else
/output/).
"""

import json
import math
import os
from datetime import datetime, timezone

# Standard-normal quantiles for the case levels [0.05, 0.25, 0.5, 0.75, 0.95].
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

# Refresh model, calibrated on the one refresh the house recorded on launch
# night: 638 -> 3,868 players tracked (~6.06x), every rate but the 100% one
# lower.  The incoming cohort earns each trophy at roughly a tenth of the
# early rate, so a refresh dilutes the blended rate to about a quarter of
# its pre-refresh value -- between pure dilution (no new earners at all)
# and a log-player interpolation to the live page, which the stated house
# baseline practice loss (~1.51) suggests would understate the drop.
REFRESH_GROWTH = 6.06        # tracked-population multiplier per refresh
NEW_COHORT_RATIO = 0.10      # new cohort's earn rate, as a fraction of the old
REFRESH_INTERVAL_H = 5.0     # hours between Exophase refreshes, roughly
BETWEEN_REFRESH_DRIFT = 1.03  # old cohort keeps earning between refreshes

HALF_LIFE_H = 24.0           # recency-weight half-life for the trend fit
SLOPE_SHRINK = 0.6           # weight on a trophy's own slope vs the median
RW_SIGMA = 0.5               # random-walk logit sigma per sqrt(day)
SPREAD_FLOOR = 0.35
SPREAD_CAP = 2.2


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(rate):
    """Clamp a percent into the range where logit is defined."""
    return min(99.99, max(0.01, float(rate)))


def logit_of(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def rate_of(logit_value):
    """Inverse logit, as a percent, guarded against overflow."""
    z = max(-30.0, min(30.0, logit_value))
    return 100.0 / (1.0 + math.exp(-z))


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def distinct_snapshots(history):
    """Collapse re-reads of an unchanged page to one snapshot each.

    A snapshot is identified by the full rates vector plus the
    players-tracked count; the first time it is seen is kept.
    """
    seen = {}
    snapshots = []
    for read in history:
        rates = read.get("rates") or []
        key = (tuple(rates), read.get("players_tracked"))
        if key in seen:
            continue
        seen[key] = True
        snapshots.append(read)
    return snapshots


def weighted_fit(points, target_x):
    """Recency-weighted least squares of y on x; returns (pred, resid_var,
    leverage_var, slope).  points are (x, y, weight)."""
    w_sum = sum(w for _, _, w in points)
    x_bar = sum(w * x for x, _, w in points) / w_sum
    y_bar = sum(w * y for _, y, w in points) / w_sum
    sxx = sum(w * (x - x_bar) ** 2 for x, _, w in points)
    sxy = sum(w * (x - x_bar) * (y - y_bar) for x, y, w in points)
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    intercept = y_bar - slope * x_bar
    pred = intercept + slope * target_x
    if len(points) >= 3 and sxx > 1e-12:
        resid_var = sum(w * (y - (intercept + slope * x)) ** 2
                        for x, y, w in points) / w_sum
        leverage = (1.0 / len(points)
                    + (target_x - x_bar) ** 2 / sxx)
    else:
        resid_var = 0.0
        leverage = 1.0
    return pred, resid_var, leverage, slope


def dilution_factor(refreshes):
    """Blended-rate multiplier after `refreshes` dilution refreshes."""
    one = (1.0 + NEW_COHORT_RATIO * (REFRESH_GROWTH - 1.0)) / REFRESH_GROWTH
    factor = 1.0
    for _ in range(refreshes):
        factor *= one * BETWEEN_REFRESH_DRIFT
    return factor


def flat_history_forecast(last_rate, stale_h, horizon_h):
    """Median logit + spread when the page has not moved yet.

    A refresh is due: count how many refresh intervals the stale time plus
    the horizon spans (at least one), dilute, and keep the band wide.
    """
    span_h = stale_h + horizon_h
    refreshes = 1 + int(max(0.0, span_h - REFRESH_INTERVAL_H)
                        // REFRESH_INTERVAL_H)
    refreshes = min(refreshes, 3)
    median_rate = last_rate * dilution_factor(refreshes)
    horizon_days = horizon_h / 24.0
    spread = min(1.3, 0.95 + 0.10 * math.sqrt(horizon_days)
                 + 0.05 * (refreshes - 1))
    return logit_of(median_rate), spread


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_h = max(0.0, (target - checkpoint).total_seconds() / 3600.0)
    horizon_days = horizon_h / 24.0

    snapshots = distinct_snapshots(history)
    n_trophies = len(trophies)

    # Per-trophy series on the logit scale, x = ln(1 + hours since the
    # first snapshot), with recency weights against the checkpoint.
    first_time = parse_time(snapshots[0]["at"]) if snapshots else checkpoint
    series = []          # per trophy: list of (x, logit, weight)
    last_rates = []
    for idx in range(n_trophies):
        points = []
        last_rate = None
        for snap in snapshots:
            rates = snap.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            last_rate = rates[idx]
            t = parse_time(snap["at"])
            hours = max(0.0, (t - first_time).total_seconds() / 3600.0)
            age_h = max(0.0, (checkpoint - t).total_seconds() / 3600.0)
            weight = 0.5 ** (age_h / HALF_LIFE_H)
            points.append((math.log1p(hours), logit_of(last_rate), weight))
        series.append(points)
        last_rates.append(last_rate if last_rate is not None else 1.0)

    # Cross-trophy median slope, for shrinkage, from trophies with a real
    # trend (at least two distinct snapshots with different logits).
    own_slopes = []
    target_x = math.log1p(max(0.0, (target - first_time).total_seconds()
                              / 3600.0)) if snapshots else 0.0
    for points in series:
        distinct = {(round(x, 9), round(y, 9)) for x, y, _ in points}
        if len(distinct) >= 2:
            _, _, _, slope = weighted_fit(points, target_x)
            own_slopes.append(slope)
    median_slope = median_of(own_slopes) if own_slopes else 0.0

    # Pooled residual variance and refresh-jump variance across trophies.
    resid_terms = []
    jump_diffs = []
    for points in series:
        if len(points) >= 3:
            _, resid_var, _, _ = weighted_fit(points, target_x)
            resid_terms.append(resid_var)
        ordered = sorted(points)
        for (_, y0, _), (_, y1, _) in zip(ordered, ordered[1:]):
            if abs(y1 - y0) > 1e-9:
                jump_diffs.append(y1 - y0)
    pooled_resid = median_of(resid_terms) if resid_terms else 0.0
    if jump_diffs:
        centre = median_of(jump_diffs)
        jump_var = min(0.8, sum((d - centre) ** 2 for d in jump_diffs)
                       / len(jump_diffs))
    else:
        jump_var = 0.5  # prior: one unseen refresh moves logits noticeably

    stale_h = 0.0
    if snapshots:
        stale_h = max(0.0, (checkpoint
                             - parse_time(snapshots[-1]["at"])
                             ).total_seconds() / 3600.0)

    rows = []
    for idx, trophy in enumerate(trophies):
        last_rate = last_rates[idx]
        points = series[idx]
        distinct = {(round(x, 9), round(y, 9)) for x, y, _ in points}

        if last_rate >= 99.99:
            # Only way is down: hedge the median just under 100%.
            median_logit, spread = logit_of(99.85), 0.9
        elif len(distinct) >= 2:
            pred, resid_var, leverage, slope = weighted_fit(points, target_x)
            slope = SLOPE_SHRINK * slope + (1.0 - SLOPE_SHRINK) * median_slope
            # Re-anchor the shrunk line through the weighted mean point.
            w_sum = sum(w for _, _, w in points)
            x_bar = sum(w * x for x, _, w in points) / w_sum
            y_bar = sum(w * y for _, y, w in points) / w_sum
            pred = y_bar + slope * (target_x - x_bar)
            last_logit = logit_of(last_rate)
            # Damp the extrapolation back toward the last read; a long
            # horizon earns a stronger pull.
            damp = 1.0 / (1.0 + horizon_days / 5.0)
            median_logit = last_logit + damp * (pred - last_logit)
            resid = max(pooled_resid, resid_var)
            var = (resid * max(leverage, 0.0)
                   + (RW_SIGMA ** 2) * horizon_days
                   + jump_var)
            spread = min(SPREAD_CAP, max(SPREAD_FLOOR, math.sqrt(var)))
        else:
            median_logit, spread = flat_history_forecast(
                last_rate, stale_h, horizon_h)

        median_logit = max(-9.2, min(9.2, median_logit))
        quantiles = []
        for z in Z_SCORES:
            q = rate_of(median_logit + z * spread)
            quantiles.append(round(min(99.99, max(0.01, q)), 4))
        # Rounding can only tie neighbours, never invert them; enforce it.
        for k in range(1, len(quantiles)):
            quantiles[k] = max(quantiles[k], quantiles[k - 1])
        rows.append({"id": trophy["id"], "q": quantiles})

    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(output_dir, exist_ok=True)
    case_files = sorted(
        name for name in os.listdir(input_dir) if name.endswith(".in"))
    for name in case_files:
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        out_name = name[:-3] + ".out" if name.endswith(".in") else name + ".out"
        with open(os.path.join(output_dir, out_name), "w",
                  encoding="utf-8") as fh:
            json.dump(forecast, fh, separators=(",", ":"))
            fh.write("\n")


if __name__ == "__main__":
    main()
