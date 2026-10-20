#!/usr/bin/env python3
"""
Per-price-level ledger and first-test vs retest features.

WHY THIS EXISTS
---------------
Reading a footprint chart, the recurring observation is: "price came back to
this level and the response was weaker than the first time." That is a *paired
comparison between two visits to the same price*. Every feature in
es_training/training_meta.json is instead a rolling-window aggregate
(pressure_ratio_5, flow_accel_3, imbalance_cluster_5, va10_price_vs_poc), and
no rolling window can express "there was 800 sell volume here 20 minutes ago
and only 90 buy volume now."

This module maintains a ledger keyed by price level, records each distinct
visit to that level, and emits features at each bar comparing the current visit
to prior visits of the same level.

NO LOOKAHEAD
------------
The ledger is built by a single forward pass over bars. At bar t the ledger is
updated with bar t's own footprint (known at bar t's close, which is when a
decision would be taken) and then features are emitted. A "prior visit" is by
construction a visit that ended before the current one began. Nothing from
t+1 onward is ever consulted.

The large-print threshold is carried across sessions as an expanding quantile
of completed sessions only, so it also never peeks forward.

Usage:
    python3 level_ledger.py /path/to/data --out retest.parquet
    python3 level_ledger.py /path/to/data --direction-method tick --near-ticks 4
"""

import argparse
import os

import numpy as np
import pandas as pd

from bars import (
    add_atr_percentile, add_basic_features, assign_bar_ids, build_bars,
    value_area_from_footprint,
)
from tick_data import TICK_SIZE, iter_sessions

# A new visit starts when the level has gone untouched for this many bars.
DEFAULT_VISIT_GAP_BARS = 10

# Only levels whose prior visit traded at least this much volume are treated as
# reference levels. Filters out incidental one-print touches.
DEFAULT_MIN_PRIOR_VOLUME = 200.0

# How close the close must be to a level for it to count as being tested.
DEFAULT_NEAR_TICKS = 4

# Percentile of print size that counts as a large participant print.
DEFAULT_BIG_PRINT_PCTL = 99.0

FEATURE_COLUMNS = [
    'has_retest_level',
    'retest_level_price',
    'retest_level_dist_ticks',
    'n_prior_visits',
    'bars_since_first_touch',
    'bars_since_prior_visit',
    'prior_visit_volume',
    'prior_visit_delta',
    'prior_visit_delta_pct',
    'prior_visit_duration_bars',
    'current_visit_volume',
    'current_visit_delta',
    'current_visit_delta_pct',
    'current_visit_duration_bars',
    'retest_vol_ratio',
    'retest_delta_ratio',
    'retest_delta_flip',
    'retest_big_prints_now',
    'retest_big_prints_prior',
    'retest_big_print_same_side',
    'absorption_at_level',
    'excursion_beyond_now',
    'excursion_beyond_prior',
    'excursion_beyond_ratio',
    'level_total_volume',
    'level_total_delta',
]


class _Visit:
    """One contiguous episode of trading at a single price level."""

    __slots__ = ('start_bar', 'last_bar', 'volume', 'buy_volume', 'sell_volume',
                 'n_big_buy', 'n_big_sell', 'n_bars', 'max_high', 'min_low')

    def __init__(self, bar_idx, high, low):
        self.start_bar = bar_idx
        self.last_bar = bar_idx
        self.volume = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.n_big_buy = 0
        self.n_big_sell = 0
        self.n_bars = 0
        self.max_high = high
        self.min_low = low

    @property
    def delta(self):
        return self.buy_volume - self.sell_volume

    @property
    def delta_pct(self):
        return self.delta / self.volume if self.volume > 0 else np.nan


class _LevelState:
    """Ledger entry for one price level (in integer ticks)."""

    __slots__ = ('first_touch_bar', 'completed', 'current',
                 'total_volume', 'total_buy', 'total_sell')

    def __init__(self):
        self.first_touch_bar = None
        self.completed = []
        self.current = None
        self.total_volume = 0.0
        self.total_buy = 0.0
        self.total_sell = 0.0

    @property
    def total_delta(self):
        return self.total_buy - self.total_sell


def build_level_footprint(ticks, bar_id, big_print_threshold):
    """
    Long-form footprint with large-print counts.

    Returns columns: bar_idx, price_ticks, volume, buy_volume, sell_volume,
    n_big_buy, n_big_sell.
    """
    volume = ticks['volume'].to_numpy(dtype=np.float64)
    direction = ticks['direction'].to_numpy(dtype=np.int8)
    price_ticks = np.rint(ticks['price'].to_numpy(dtype=np.float64) / TICK_SIZE)

    is_big = volume >= big_print_threshold

    work = pd.DataFrame({
        'bar_idx': bar_id,
        'price_ticks': price_ticks.astype(np.int64),
        'volume': volume,
        'buy_volume': np.where(direction == 1, volume, 0.0),
        'sell_volume': np.where(direction == -1, volume, 0.0),
        'n_big_buy': ((direction == 1) & is_big).astype(np.int32),
        'n_big_sell': ((direction == -1) & is_big).astype(np.int32),
    })

    return work.groupby(['bar_idx', 'price_ticks'], sort=True).agg(
        volume=('volume', 'sum'),
        buy_volume=('buy_volume', 'sum'),
        sell_volume=('sell_volume', 'sum'),
        n_big_buy=('n_big_buy', 'sum'),
        n_big_sell=('n_big_sell', 'sum'),
    ).reset_index()


def compute_retest_features(bars, footprint, visit_gap_bars=DEFAULT_VISIT_GAP_BARS,
                            min_prior_volume=DEFAULT_MIN_PRIOR_VOLUME,
                            near_ticks=DEFAULT_NEAR_TICKS):
    """
    Forward pass over one session's bars, emitting retest features per bar.

    Returns a DataFrame indexed like `bars` with FEATURE_COLUMNS.
    """
    n_bars = len(bars)
    out = {col: np.full(n_bars, np.nan) for col in FEATURE_COLUMNS}
    if n_bars == 0 or footprint is None or len(footprint) == 0:
        return pd.DataFrame(out, index=bars.index)

    # Footprint rows grouped by bar for O(1) lookup during the pass.
    by_bar = {int(b): g for b, g in footprint.groupby('bar_idx', sort=True)}

    highs = bars['high'].to_numpy(dtype=np.float64)
    lows = bars['low'].to_numpy(dtype=np.float64)
    closes = bars['close'].to_numpy(dtype=np.float64)

    ledger = {}

    for t in range(n_bars):
        grp = by_bar.get(t)

        # ---- Update the ledger with bar t's own footprint ----
        if grp is not None:
            for level, vol, buy, sell, nbb, nbs in zip(
                grp['price_ticks'].to_numpy(dtype=np.int64),
                grp['volume'].to_numpy(dtype=np.float64),
                grp['buy_volume'].to_numpy(dtype=np.float64),
                grp['sell_volume'].to_numpy(dtype=np.float64),
                grp['n_big_buy'].to_numpy(dtype=np.int64),
                grp['n_big_sell'].to_numpy(dtype=np.int64),
            ):
                state = ledger.get(level)
                if state is None:
                    state = _LevelState()
                    state.first_touch_bar = t
                    ledger[level] = state

                # Close out the running visit if the level has gone quiet.
                if state.current is not None and (t - state.current.last_bar) > visit_gap_bars:
                    state.completed.append(state.current)
                    state.current = None

                if state.current is None:
                    state.current = _Visit(t, highs[t], lows[t])

                v = state.current
                v.last_bar = t
                v.volume += vol
                v.buy_volume += buy
                v.sell_volume += sell
                v.n_big_buy += int(nbb)
                v.n_big_sell += int(nbs)
                v.n_bars += 1
                v.max_high = max(v.max_high, highs[t])
                v.min_low = min(v.min_low, lows[t])

                state.total_volume += vol
                state.total_buy += buy
                state.total_sell += sell

        # ---- Select the reference level being tested at bar t ----
        close_ticks = int(round(closes[t] / TICK_SIZE))
        best_level = None
        best_prior = None
        best_prior_volume = -1.0

        for offset in range(-near_ticks, near_ticks + 1):
            level = close_ticks + offset
            state = ledger.get(level)
            if state is None or not state.completed:
                continue
            prior = state.completed[-1]
            if prior.volume < min_prior_volume:
                continue
            if prior.volume > best_prior_volume:
                best_prior_volume = prior.volume
                best_level = level
                best_prior = prior

        if best_level is None:
            out['has_retest_level'][t] = 0.0
            continue

        state = ledger[best_level]
        current = state.current
        level_price = best_level * TICK_SIZE

        # Excursion beyond the level, measured on the far side of the approach.
        # Approaching from below means the meaningful excursion is above it.
        approaching_from_below = closes[t] <= level_price
        if approaching_from_below:
            exc_prior = max(0.0, best_prior.max_high - level_price)
            exc_now = max(0.0, (current.max_high - level_price)) if current else 0.0
        else:
            exc_prior = max(0.0, level_price - best_prior.min_low)
            exc_now = max(0.0, (level_price - current.min_low)) if current else 0.0

        cur_vol = current.volume if current else 0.0
        cur_delta = current.delta if current else np.nan
        cur_delta_pct = current.delta_pct if current else np.nan
        cur_bars = current.n_bars if current else 0
        cur_big_buy = current.n_big_buy if current else 0
        cur_big_sell = current.n_big_sell if current else 0

        out['has_retest_level'][t] = 1.0
        out['retest_level_price'][t] = level_price
        out['retest_level_dist_ticks'][t] = (closes[t] - level_price) / TICK_SIZE
        out['n_prior_visits'][t] = len(state.completed)
        out['bars_since_first_touch'][t] = t - state.first_touch_bar
        out['bars_since_prior_visit'][t] = t - best_prior.last_bar

        out['prior_visit_volume'][t] = best_prior.volume
        out['prior_visit_delta'][t] = best_prior.delta
        out['prior_visit_delta_pct'][t] = best_prior.delta_pct
        out['prior_visit_duration_bars'][t] = best_prior.n_bars

        out['current_visit_volume'][t] = cur_vol
        out['current_visit_delta'][t] = cur_delta
        out['current_visit_delta_pct'][t] = cur_delta_pct
        out['current_visit_duration_bars'][t] = cur_bars

        # The headline "lack of interest" measure: how much trade is showing up
        # on the retest relative to the original test.
        out['retest_vol_ratio'][t] = (
            cur_vol / best_prior.volume if best_prior.volume > 0 else np.nan)

        if np.isfinite(cur_delta) and abs(best_prior.delta) > 0:
            out['retest_delta_ratio'][t] = cur_delta / abs(best_prior.delta)
            out['retest_delta_flip'][t] = float(
                np.sign(cur_delta) != np.sign(best_prior.delta))

        out['retest_big_prints_now'][t] = cur_big_buy + cur_big_sell
        out['retest_big_prints_prior'][t] = best_prior.n_big_buy + best_prior.n_big_sell

        prior_big_side = np.sign(best_prior.n_big_buy - best_prior.n_big_sell)
        now_big_side = np.sign(cur_big_buy - cur_big_sell)
        if prior_big_side != 0 and now_big_side != 0:
            out['retest_big_print_same_side'][t] = float(prior_big_side == now_big_side)

        # Volume spent per tick of progress past the level. High values mean
        # size is trading without price advancing, i.e. absorption.
        out['absorption_at_level'][t] = cur_vol / (exc_now / TICK_SIZE + 1.0)

        out['excursion_beyond_now'][t] = exc_now
        out['excursion_beyond_prior'][t] = exc_prior
        out['excursion_beyond_ratio'][t] = (
            exc_now / exc_prior if exc_prior > 0 else np.nan)

        out['level_total_volume'][t] = state.total_volume
        out['level_total_delta'][t] = state.total_delta

    return pd.DataFrame(out, index=bars.index)


def build_retest_dataset(data_dir, window='1min', volume_per_bar=None,
                         direction_method='level', start=None, end=None,
                         visit_gap_bars=DEFAULT_VISIT_GAP_BARS,
                         min_prior_volume=DEFAULT_MIN_PRIOR_VOLUME,
                         near_ticks=DEFAULT_NEAR_TICKS,
                         big_print_pctl=DEFAULT_BIG_PRINT_PCTL,
                         verbose=True):
    """
    Stream sessions, building bars plus retest features for each.

    The large-print threshold is an expanding quantile over the print sizes of
    *completed* sessions. The first session is skipped for feature purposes
    because no threshold exists yet without peeking at its own data.
    """
    all_frames = []
    print_sizes_seen = []
    threshold = None

    for path, ticks in iter_sessions(data_dir, direction_method=direction_method,
                                     start=start, end=end, verbose=False):
        bar_id, _, _ = assign_bar_ids(ticks, window=window,
                                      volume_per_bar=volume_per_bar)
        bars, _ = build_bars(ticks, window=window, volume_per_bar=volume_per_bar,
                             with_footprint=False)
        if len(bars) < 20:
            continue

        if threshold is not None:
            footprint = build_level_footprint(ticks, bar_id, threshold)
            feats = compute_retest_features(
                bars, footprint,
                visit_gap_bars=visit_gap_bars,
                min_prior_volume=min_prior_volume,
                near_ticks=near_ticks,
            )

            # Per-bar value area, reused from the same footprint rather than
            # re-aggregating the ticks.
            va = value_area_from_footprint(footprint).reindex(range(len(bars)))
            va.index = bars.index

            frame = pd.concat([bars, va, feats], axis=1)
            frame['close_vs_poc'] = frame['close'] - frame['poc']
            frame['close_vs_vah'] = frame['close'] - frame['vah']
            frame['close_vs_val'] = frame['close'] - frame['val']
            frame['close_in_value_area'] = (
                (frame['close'] >= frame['val']) & (frame['close'] <= frame['vah'])
            ).astype(np.int8)
            frame['source_file'] = os.path.basename(path)
            frame['big_print_threshold'] = threshold
            all_frames.append(frame)

        # Update the threshold only after this session has been used, so it is
        # always derived from strictly prior data.
        sizes = ticks['volume'].to_numpy(dtype=np.float64)
        # Subsample to bound memory over multi-year runs.
        if len(sizes) > 20000:
            sizes = np.random.default_rng(0).choice(sizes, 20000, replace=False)
        print_sizes_seen.append(sizes)
        if len(print_sizes_seen) > 60:
            print_sizes_seen.pop(0)
        threshold = float(np.percentile(np.concatenate(print_sizes_seen),
                                        big_print_pctl))
        threshold = max(threshold, 2.0)

        if verbose and len(all_frames) % 25 == 0 and all_frames:
            print(f"  {len(all_frames)} sessions with features "
                  f"(big-print threshold {threshold:.0f})")

    if not all_frames:
        raise ValueError("No sessions produced retest features")

    combined = pd.concat(all_frames).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = add_basic_features(combined)
    combined = add_atr_percentile(combined)

    if verbose:
        n_with = int((combined['has_retest_level'] == 1).sum())
        print(f"\n  Bars: {len(combined):,}  "
              f"({n_with:,} with a reference level, "
              f"{n_with / len(combined) * 100:.1f}%)")

    return combined


def describe_features(df):
    """Coverage and distribution summary for the retest features."""
    print(f"\n{'='*72}")
    print("  RETEST FEATURE COVERAGE")
    print(f"{'='*72}")

    live = df[df['has_retest_level'] == 1]
    print(f"  Bars total:                {len(df):,}")
    print(f"  Bars with a level:         {len(live):,} "
          f"({len(live) / max(len(df), 1) * 100:.1f}%)")
    if len(live) == 0:
        return

    print(f"  Sessions:                  {df['session_date'].nunique()}")
    print()
    cols = ['retest_vol_ratio', 'retest_delta_ratio', 'absorption_at_level',
            'excursion_beyond_ratio', 'n_prior_visits', 'bars_since_prior_visit',
            'prior_visit_volume', 'current_visit_volume']
    stats = live[cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    print(stats.T[['count', 'mean', '10%', '50%', '90%']].to_string(
        float_format=lambda v: f"{v:,.3f}"))

    flip = live['retest_delta_flip'].mean()
    same = live['retest_big_print_same_side'].mean()
    print(f"\n  Delta sign flips vs prior visit:  {flip * 100:.1f}%")
    print(f"  Large prints on same side:        {same * 100:.1f}% "
          f"(of bars where both visits had large prints)")
    print(f"{'='*72}")


def main():
    parser = argparse.ArgumentParser(
        description='Build first-test vs retest level features.')
    parser.add_argument('data_dir', help='Directory containing .data files')
    parser.add_argument('--window', default='1min', help='Time bar window')
    parser.add_argument('--volume-bars', type=int, default=None,
                        help='Use volume bars of this size instead of time bars')
    parser.add_argument('--direction-method', default='level',
                        choices=['level', 'tick', 'mid'],
                        help='Aggressor rule; run validate_direction.py first')
    parser.add_argument('--start', default=None, help='First session, YYYYMMDD')
    parser.add_argument('--end', default=None, help='Last session, YYYYMMDD')
    parser.add_argument('--visit-gap-bars', type=int, default=DEFAULT_VISIT_GAP_BARS,
                        help='Bars untouched before a new visit starts')
    parser.add_argument('--min-prior-volume', type=float,
                        default=DEFAULT_MIN_PRIOR_VOLUME,
                        help='Minimum prior-visit volume for a reference level')
    parser.add_argument('--near-ticks', type=int, default=DEFAULT_NEAR_TICKS,
                        help='How close the close must be to count as a test')
    parser.add_argument('--out', default=None, help='Output .parquet or .csv')

    args = parser.parse_args()

    df = build_retest_dataset(
        args.data_dir,
        window=args.window,
        volume_per_bar=args.volume_bars,
        direction_method=args.direction_method,
        start=args.start,
        end=args.end,
        visit_gap_bars=args.visit_gap_bars,
        min_prior_volume=args.min_prior_volume,
        near_ticks=args.near_ticks,
    )

    describe_features(df)

    if args.out:
        if args.out.endswith('.parquet'):
            df.to_parquet(args.out)
        else:
            df.to_csv(args.out)
        print(f"\n  Written to {args.out}")


if __name__ == '__main__':
    main()
