#!/usr/bin/env python3
"""
Validate the trade aggressor (buy/sell) classification.

Every delta, CVD, imbalance, and absorption feature in this project is built on
the assumption that each print can be labelled buyer- or seller-initiated. The
existing decoder infers this by comparing the trade price to [level_low,
level_high]. That is the standard quote rule and is correct *if* those bounds
are the bid and ask. The observed level widths of 1.0 and 2.0 points suggest
they may instead be footprint price buckets, since ES normally quotes a 0.25
spread. If so, the classification is close to noise and everything downstream
is meaningless.

This script settles it. Run it before building anything else.

THE PRIMARY TEST IS CONTEMPORANEOUS, NOT PREDICTIVE
---------------------------------------------------
Correctly labelled aggressor flow is mechanically related to price movement
*within the same bar*: if buyers are lifting offers, price rises as it happens.
So corr(bar delta, bar return) must be strongly positive. On 1-minute ES with
correct labels this is typically 0.6-0.85.

The predictive test -- delta at bar t against the return of bar t+1 -- is
reported too, but it is NOT a validity check. A near-zero predictive
correlation is what an efficient market looks like and says nothing about
whether the labels are right. Only use it to gauge how much forecasting signal
raw delta carries on its own.

Confidence intervals come from a bootstrap over whole sessions. Adjacent bars
are heavily autocorrelated, so a naive per-bar standard error would be far too
narrow.

Usage:
    python3 validate_direction.py /path/to/data
    python3 validate_direction.py /path/to/data --start 20240101 --end 20241231
    python3 validate_direction.py /path/to/data --window 1min --n-boot 2000
"""

import argparse

import numpy as np
import pandas as pd

from bars import build_bars
from tick_data import TICK_SIZE, classify_direction, is_rth, iter_sessions

METHODS = ('level', 'tick', 'mid')

# Interpretation bands for the contemporaneous correlation. Heuristics from
# what correctly-labelled liquid futures data looks like, not hard theory.
STRONG_CORR = 0.50
WEAK_CORR = 0.20


# ============================================================
# Statistics (numpy only -- no scipy dependency)
# ============================================================

def pearson(x, y):
    """Pearson correlation, NaN-safe."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    x = x[ok] - x[ok].mean()
    y = y[ok] - y[ok].mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0:
        return np.nan
    return float((x * y).sum() / denom)


def spearman(x, y):
    """Rank correlation, robust to the heavy tails in volume and delta."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    return pearson(rx, ry)


def session_block_bootstrap_ci(values_by_session, stat_fn, n_boot=1000,
                               alpha=0.05, seed=0):
    """
    Bootstrap a statistic by resampling whole sessions with replacement.

    values_by_session: list of (x_array, y_array), one entry per session.
    stat_fn: callable(x_concat, y_concat) -> float
    """
    if len(values_by_session) < 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    n = len(values_by_session)
    stats = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        pick = rng.integers(0, n, size=n)
        xs = np.concatenate([values_by_session[i][0] for i in pick])
        ys = np.concatenate([values_by_session[i][1] for i in pick])
        stats[b] = stat_fn(xs, ys)

    stats = stats[np.isfinite(stats)]
    if len(stats) < 10:
        return (np.nan, np.nan)
    return (float(np.quantile(stats, alpha / 2)),
            float(np.quantile(stats, 1 - alpha / 2)))


# ============================================================
# Per-session measurement
# ============================================================

def _tick_level_separation(ticks, direction):
    """
    Mean next-tick price change, in ticks, conditional on classified direction.

    The most direct possible test of the labelling: an aggressive buyer walks
    the book up, so prints labelled BUY should be followed by non-negative
    price changes on average, and SELL prints by non-positive ones. The
    separation (buy mean minus sell mean) should be clearly positive.
    """
    price = ticks['price'].to_numpy(dtype=np.float64)
    if len(price) < 3:
        return np.nan, np.nan, 0, 0

    nxt = np.diff(price, append=price[-1]) / TICK_SIZE
    nxt = nxt[:-1]
    d = direction[:-1]

    buy_mask = d == 1
    sell_mask = d == -1
    buy_mean = float(nxt[buy_mask].mean()) if buy_mask.any() else np.nan
    sell_mean = float(nxt[sell_mask].mean()) if sell_mask.any() else np.nan
    return buy_mean, sell_mean, int(buy_mask.sum()), int(sell_mask.sum())


def _level_position_profile(ticks):
    """
    Where within [level_low, level_high] do prints occur?

    If the level is a genuine bid/ask, essentially every print sits exactly on
    one bound. A large 'strictly inside' share proves the level is a bucket,
    which invalidates the quote rule outright.
    """
    price = ticks['price'].to_numpy(dtype=np.float64)
    low = ticks['level_low'].to_numpy(dtype=np.float64)
    high = ticks['level_high'].to_numpy(dtype=np.float64)

    at_low = np.isclose(price, low, atol=1e-9)
    at_high = np.isclose(price, high, atol=1e-9)
    inside = ~at_low & ~at_high & (price > low) & (price < high)
    outside = (price < low - 1e-9) | (price > high + 1e-9)

    n = len(price)
    return {
        'at_low_pct': float(at_low.sum() / n * 100),
        'at_high_pct': float(at_high.sum() / n * 100),
        'inside_pct': float(inside.sum() / n * 100),
        'outside_pct': float(outside.sum() / n * 100),
        'median_width_ticks': float(np.median(high - low) / TICK_SIZE),
    }


def collect(data_dir, window='1min', start=None, end=None, rth_only=True,
            max_sessions=None, verbose=True):
    """
    Decode every session once and, for each candidate rule, accumulate the
    per-session arrays each test needs.
    """
    per_method = {
        m: {
            'contemp': [],      # (delta_pct, return_ticks) per session
            'contemp_raw': [],  # (delta, return_ticks) per session
            'predictive': [],   # (delta_pct at t, return of t+1) per session
            'tick_sep': [],     # (buy_mean, sell_mean, n_buy, n_sell)
            'unknown_pct': [],
        }
        for m in METHODS
    }
    level_profiles = []
    n_sessions = 0

    for path, ticks in iter_sessions(data_dir, start=start, end=end,
                                     verbose=False):
        if len(ticks) < 100:
            continue

        level_profiles.append(_level_position_profile(ticks))

        for method in METHODS:
            direction = classify_direction(ticks, method=method)
            per_method[method]['unknown_pct'].append(
                float((direction == 0).sum() / len(direction) * 100))

            per_method[method]['tick_sep'].append(
                _tick_level_separation(ticks, direction))

            scoped = ticks.copy()
            scoped['direction'] = direction
            scoped['signed_volume'] = direction * scoped['volume']

            bars, _ = build_bars(scoped, window=window, with_footprint=False)
            if len(bars) < 20:
                continue

            if rth_only:
                bars = bars[is_rth(bars.index)]
                if len(bars) < 20:
                    continue

            ret_ticks = ((bars['close'] - bars['open']) / TICK_SIZE).to_numpy()
            delta = bars['delta'].to_numpy(dtype=np.float64)
            with np.errstate(invalid='ignore', divide='ignore'):
                delta_pct = np.where(bars['volume'].to_numpy() > 0,
                                     delta / bars['volume'].to_numpy(), np.nan)

            per_method[method]['contemp'].append((delta_pct, ret_ticks))
            per_method[method]['contemp_raw'].append((delta, ret_ticks))

            # Predictive: delta at t vs the *next* bar's open-to-close move.
            fwd = np.full(len(bars), np.nan)
            fwd[:-1] = ret_ticks[1:]
            per_method[method]['predictive'].append((delta_pct, fwd))

        n_sessions += 1
        if verbose and n_sessions % 25 == 0:
            print(f"  processed {n_sessions} sessions...")
        if max_sessions and n_sessions >= max_sessions:
            break

    if n_sessions == 0:
        raise ValueError("No usable sessions found")

    if verbose:
        print(f"  processed {n_sessions} sessions total")

    return per_method, level_profiles, n_sessions


# ============================================================
# Reporting
# ============================================================

def report(per_method, level_profiles, n_sessions, n_boot=1000):
    print(f"\n{'='*72}")
    print("  LEVEL GEOMETRY  (is [level_low, level_high] a quote or a bucket?)")
    print(f"{'='*72}")

    prof = pd.DataFrame(level_profiles).mean()
    print(f"  Median level width:        {prof['median_width_ticks']:.2f} ticks")
    print(f"  Prints exactly at low:     {prof['at_low_pct']:.2f}%")
    print(f"  Prints exactly at high:    {prof['at_high_pct']:.2f}%")
    print(f"  Prints strictly inside:    {prof['inside_pct']:.2f}%")
    print(f"  Prints outside the level:  {prof['outside_pct']:.2f}%")

    if prof['median_width_ticks'] > 1.5:
        print("\n  -> Width exceeds one tick, so this is very likely a footprint")
        print("     bucket rather than the bid/ask. The quote rule ('level') is")
        print("     then measuring which half of a bucket a print landed in,")
        print("     which is not the same thing as aggression.")
    else:
        print("\n  -> Width is consistent with a genuine bid/ask spread.")

    results = {}
    print(f"\n{'='*72}")
    print("  PRIMARY TEST: contemporaneous corr(bar delta, bar return)")
    print("  Correct aggressor labels are mechanically related to same-bar")
    print("  price movement. Expect 0.6-0.85 when labelling is right.")
    print(f"{'='*72}")
    print(f"  {'rule':>6} | {'pearson':>8} | {'95% CI':>18} | {'spearman':>9} | "
          f"{'r2':>6} | {'unknown':>8}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*18}-+-{'-'*9}-+-{'-'*6}-+-{'-'*8}")

    for method in METHODS:
        data = per_method[method]['contemp']
        if not data:
            continue
        xs = np.concatenate([d[0] for d in data])
        ys = np.concatenate([d[1] for d in data])
        r = pearson(xs, ys)
        rho = spearman(xs, ys)
        lo, hi = session_block_bootstrap_ci(data, pearson, n_boot=n_boot)
        unknown = float(np.mean(per_method[method]['unknown_pct']))
        results[method] = {'contemp_r': r, 'ci': (lo, hi), 'unknown_pct': unknown}
        print(f"  {method:>6} | {r:>8.3f} | [{lo:>7.3f}, {hi:>7.3f}] | "
              f"{rho:>9.3f} | {r**2 if np.isfinite(r) else np.nan:>6.3f} | "
              f"{unknown:>7.2f}%")

    print(f"\n{'='*72}")
    print("  TICK-LEVEL TEST: mean next-tick price change by label (ticks)")
    print("  Aggressive buys walk the book up, sells walk it down, so the")
    print("  separation should be clearly positive.")
    print(f"{'='*72}")
    print(f"  {'rule':>6} | {'after BUY':>10} | {'after SELL':>11} | "
          f"{'separation':>11}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*11}-+-{'-'*11}")

    for method in METHODS:
        seps = per_method[method]['tick_sep']
        if not seps:
            continue
        buy_means = np.array([s[0] for s in seps], dtype=np.float64)
        sell_means = np.array([s[1] for s in seps], dtype=np.float64)
        n_buy = np.array([s[2] for s in seps], dtype=np.float64)
        n_sell = np.array([s[3] for s in seps], dtype=np.float64)

        bm = np.nansum(buy_means * n_buy) / max(np.nansum(n_buy), 1)
        sm = np.nansum(sell_means * n_sell) / max(np.nansum(n_sell), 1)
        results.setdefault(method, {})['tick_separation'] = bm - sm
        print(f"  {method:>6} | {bm:>10.4f} | {sm:>11.4f} | {bm - sm:>11.4f}")

    print(f"\n{'='*72}")
    print("  SECONDARY: predictive corr(delta at t, return of t+1)")
    print("  NOT a validity check. Near zero is normal and expected.")
    print(f"{'='*72}")
    print(f"  {'rule':>6} | {'pearson':>8} | {'95% CI':>18}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*18}")

    for method in METHODS:
        data = per_method[method]['predictive']
        if not data:
            continue
        xs = np.concatenate([d[0] for d in data])
        ys = np.concatenate([d[1] for d in data])
        r = pearson(xs, ys)
        lo, hi = session_block_bootstrap_ci(data, pearson, n_boot=n_boot)
        results.setdefault(method, {})['predictive_r'] = r
        print(f"  {method:>6} | {r:>8.3f} | [{lo:>7.3f}, {hi:>7.3f}]")

    _verdict(results, n_sessions)
    return results


def _verdict(results, n_sessions):
    print(f"\n{'='*72}")
    print("  VERDICT")
    print(f"{'='*72}")

    scored = [(m, v.get('contemp_r', np.nan)) for m, v in results.items()]
    scored = [(m, r) for m, r in scored if np.isfinite(r)]
    if not scored:
        print("  Could not compute correlations. Check the input data.")
        return

    scored.sort(key=lambda t: -t[1])
    best_method, best_r = scored[0]
    level_r = results.get('level', {}).get('contemp_r', np.nan)

    print(f"  Sessions analysed:  {n_sessions}")
    print(f"  Best rule:          '{best_method}' (contemporaneous r = {best_r:.3f})")
    print(f"  Current rule:       'level'   (contemporaneous r = {level_r:.3f})")
    print()

    if best_r >= STRONG_CORR:
        print(f"  The '{best_method}' rule looks CORRECT. Delta tracks same-bar")
        print("  price movement as it should.")
        if best_method != 'level':
            print(f"\n  ACTION: switch to --direction-method {best_method}. The 'level'")
            print("  rule currently used by es_training and NN_model is inferior,")
            print("  so every delta/CVD/imbalance feature already computed should")
            print("  be regenerated.")
        else:
            print("\n  ACTION: none. Keep the existing rule and proceed.")
    elif best_r >= WEAK_CORR:
        print(f"  PARTIAL. r = {best_r:.3f} is well below the 0.6-0.85 expected from")
        print("  clean aggressor data, so the labels carry some signal but are")
        print("  substantially noisy.")
        print("\n  ACTION: order-flow features will be attenuated. Worth checking")
        print("  whether the feed exposes an explicit aggressor flag before")
        print("  investing further in delta-based features.")
    else:
        print(f"  BROKEN. r = {best_r:.3f} means delta is essentially unrelated to")
        print("  same-bar price movement. No rule recoverable from this file")
        print("  layout is labelling aggression correctly.")
        print("\n  ACTION: stop building delta/CVD/imbalance features. They are")
        print("  noise. Either source data with an aggressor flag, or restrict")
        print("  the model to price, volume, and volume-profile features, which")
        print("  do not depend on the buy/sell split.")

    print(f"{'='*72}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate trade aggressor classification.')
    parser.add_argument('data_dir', help='Directory containing .data files')
    parser.add_argument('--window', default='1min', help='Bar window (default: 1min)')
    parser.add_argument('--start', default=None, help='First session, YYYYMMDD')
    parser.add_argument('--end', default=None, help='Last session, YYYYMMDD')
    parser.add_argument('--max-sessions', type=int, default=None,
                        help='Cap sessions analysed (a few hundred is plenty)')
    parser.add_argument('--all-hours', action='store_true',
                        help='Include overnight bars (default: RTH only)')
    parser.add_argument('--n-boot', type=int, default=1000,
                        help='Bootstrap resamples (default: 1000)')
    parser.add_argument('--out', default=None, help='Write results to this CSV')

    args = parser.parse_args()

    print("Decoding sessions and computing diagnostics...")
    per_method, level_profiles, n_sessions = collect(
        args.data_dir,
        window=args.window,
        start=args.start,
        end=args.end,
        rth_only=not args.all_hours,
        max_sessions=args.max_sessions,
    )

    results = report(per_method, level_profiles, n_sessions, n_boot=args.n_boot)

    if args.out:
        pd.DataFrame(results).T.to_csv(args.out)
        print(f"\n  Results written to {args.out}")


if __name__ == '__main__':
    main()
