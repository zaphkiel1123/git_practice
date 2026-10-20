#!/usr/bin/env python3
"""
Tick-to-bar aggregation: time bars, volume bars, and footprint profiles.

Two differences from es_training/feature_pipeline.py:

1. Volume bars are supported alongside time bars. A 1-minute bar at 09:35 and
   one at 12:30 carry very different amounts of information, which makes every
   feature computed from them heteroskedastic. Bars closed every N contracts
   have far more stable statistics.

2. Volume-at-price is returned as a long-form table (bar_idx, price_ticks,
   volume, buy_volume, sell_volume) built by a single vectorized groupby,
   rather than a Python dict per bar. It is faster, far more compact over
   multi-year runs, and is the exact input the level ledger needs.

Usage:
    python3 bars.py /path/to/data --window 1min --out bars.parquet
    python3 bars.py /path/to/data --volume-bars 2500 --out volbars.parquet
"""

import argparse
import os

import numpy as np
import pandas as pd

from tick_data import (
    TICK_SIZE, is_rth, iter_sessions, session_date, to_market_time,
)


# ============================================================
# Bar construction
# ============================================================

def _assign_time_bars(tick_df, window='1min'):
    """Bar id per tick from a time window. Returns (bar_id, bar_open_time)."""
    ts = pd.DatetimeIndex(tick_df['timestamp'])
    floored = ts.floor(window)
    codes, uniques = pd.factorize(floored, sort=True)
    return codes, pd.DatetimeIndex(uniques)


def _assign_volume_bars(tick_df, volume_per_bar):
    """
    Bar id per tick from cumulative volume.

    A bar closes once cumulative volume crosses a multiple of volume_per_bar.
    The crossing print is assigned to the bar it completes, so bars are at
    least volume_per_bar in size (never split a print across bars).
    """
    if volume_per_bar <= 0:
        raise ValueError("volume_per_bar must be positive")
    cumvol = tick_df['volume'].to_numpy(dtype=np.float64).cumsum()
    raw_id = np.floor(cumvol / float(volume_per_bar)).astype(np.int64)
    # Shift so the print that crosses the boundary closes the current bar.
    bar_id = np.maximum.accumulate(np.concatenate([[0], raw_id[:-1]])) if len(raw_id) else raw_id
    codes, uniques = pd.factorize(bar_id, sort=True)
    return codes, uniques


def assign_bar_ids(tick_df, window='1min', volume_per_bar=None):
    """
    Positional bar id per tick, plus the bar index labels.

    Public so that level_ledger.py can build its own footprint aggregation on
    exactly the same bar boundaries as build_bars().

    Returns (bar_id, labels, labels_are_time).
    """
    if volume_per_bar is not None:
        bar_id, uniques = _assign_volume_bars(tick_df, volume_per_bar)
        return bar_id, uniques, False
    bar_id, bar_times = _assign_time_bars(tick_df, window)
    return bar_id, bar_times, True


def build_bars(tick_df, window='1min', volume_per_bar=None, with_footprint=True):
    """
    Aggregate one session of ticks into bars.

    Returns (bars, footprint) where footprint is None unless with_footprint.

    bars columns:
        open/high/low/close, volume, buy_volume, sell_volume, delta,
        num_trades, vwap, first_ts, last_ts
    footprint columns:
        bar_idx, price_ticks, volume, buy_volume, sell_volume, delta
    """
    if len(tick_df) == 0:
        return pd.DataFrame(), None

    bar_id, labels, bar_index_is_time = assign_bar_ids(
        tick_df, window=window, volume_per_bar=volume_per_bar)
    bar_times = labels if bar_index_is_time else None

    price = tick_df['price'].to_numpy(dtype=np.float64)
    volume = tick_df['volume'].to_numpy(dtype=np.float64)
    direction = tick_df['direction'].to_numpy(dtype=np.int8)
    ts = pd.DatetimeIndex(tick_df['timestamp'])

    work = pd.DataFrame({
        'bar_idx': bar_id,
        'price': price,
        'volume': volume,
        'buy_volume': np.where(direction == 1, volume, 0.0),
        'sell_volume': np.where(direction == -1, volume, 0.0),
        'notional': price * volume,
        'timestamp': ts,
    })

    g = work.groupby('bar_idx', sort=True)
    bars = pd.DataFrame({
        'open': g['price'].first(),
        'high': g['price'].max(),
        'low': g['price'].min(),
        'close': g['price'].last(),
        'volume': g['volume'].sum(),
        'buy_volume': g['buy_volume'].sum(),
        'sell_volume': g['sell_volume'].sum(),
        'num_trades': g['price'].size().astype(np.float64),
        'first_ts': g['timestamp'].first(),
        'last_ts': g['timestamp'].last(),
        'notional': g['notional'].sum(),
    })
    bars['delta'] = bars['buy_volume'] - bars['sell_volume']
    bars['vwap'] = bars['notional'] / bars['volume'].replace(0, np.nan)
    bars = bars.drop(columns=['notional'])

    if bar_index_is_time:
        # Index by bar-open wall time so multi-session concatenation aligns.
        bars.index = bar_times[bars.index.to_numpy()]
        bars.index.name = 'timestamp'
    else:
        # Volume bars have no natural clock; index by the closing print time.
        bars = bars.set_index('last_ts', drop=False)
        bars.index.name = 'timestamp'

    footprint = None
    if with_footprint:
        work['price_ticks'] = np.rint(price / TICK_SIZE).astype(np.int64)
        fg = work.groupby(['bar_idx', 'price_ticks'], sort=True)
        footprint = fg.agg(
            volume=('volume', 'sum'),
            buy_volume=('buy_volume', 'sum'),
            sell_volume=('sell_volume', 'sum'),
        ).reset_index()
        footprint['delta'] = footprint['buy_volume'] - footprint['sell_volume']

    return bars, footprint


# ============================================================
# Volume profile / value area
# ============================================================

def value_area_from_footprint(footprint, value_area_pct=0.70):
    """
    POC / VAH / VAL per bar from a long-form footprint table.

    Standard 70% rule: start at the POC and repeatedly extend toward whichever
    adjacent price level holds more volume until the target share is captured.

    Returns a DataFrame indexed by bar_idx with poc, vah, val (in points),
    va_width, and poc_volume_pct.
    """
    if footprint is None or len(footprint) == 0:
        return pd.DataFrame(columns=['poc', 'vah', 'val', 'va_width', 'poc_volume_pct'])

    rows = []
    for bar_idx, grp in footprint.groupby('bar_idx', sort=True):
        levels = grp['price_ticks'].to_numpy(dtype=np.int64)
        vols = grp['volume'].to_numpy(dtype=np.float64)
        order = np.argsort(levels)
        levels = levels[order]
        vols = vols[order]

        total = vols.sum()
        if total <= 0:
            continue

        poc_i = int(np.argmax(vols))
        target = total * value_area_pct
        captured = vols[poc_i]
        lo = hi = poc_i

        while captured < target and (lo > 0 or hi < len(levels) - 1):
            vol_below = vols[lo - 1] if lo > 0 else -1.0
            vol_above = vols[hi + 1] if hi < len(levels) - 1 else -1.0
            if vol_above >= vol_below:
                hi += 1
                captured += vols[hi]
            else:
                lo -= 1
                captured += vols[lo]

        rows.append({
            'bar_idx': bar_idx,
            'poc': levels[poc_i] * TICK_SIZE,
            'val': levels[lo] * TICK_SIZE,
            'vah': levels[hi] * TICK_SIZE,
            'poc_volume_pct': vols[poc_i] / total,
        })

    out = pd.DataFrame(rows).set_index('bar_idx')
    out['va_width'] = out['vah'] - out['val']
    return out


# ============================================================
# Common bar-level derived columns
# ============================================================

def compute_atr(bars, period=14):
    """Wilder-style ATR using a simple moving average of true range."""
    high = bars['high']
    low = bars['low']
    prev_close = bars['close'].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=max(2, period // 2)).mean()


def add_basic_features(bars):
    """
    Bar-shape, volatility, and session columns used by the event study.

    Kept deliberately small: this is context for conditioning and matching,
    not a model feature set.
    """
    bars = bars.copy()

    bars['bar_range'] = bars['high'] - bars['low']
    bars['body'] = bars['close'] - bars['open']
    bars['log_return'] = np.log(bars['close'] / bars['close'].shift(1))
    bars['clv'] = (bars['close'] - bars['low']) / bars['bar_range'].replace(0, np.nan)
    bars['delta_pct'] = bars['delta'] / bars['volume'].replace(0, np.nan)

    bars['atr_14'] = compute_atr(bars, 14)
    bars['atr_5'] = compute_atr(bars, 5)
    bars['atr_ratio'] = bars['atr_5'] / bars['atr_14'].replace(0, np.nan)
    bars['range_vs_atr'] = bars['bar_range'] / bars['atr_14'].replace(0, np.nan)
    bars['realized_vol_20'] = bars['log_return'].rolling(20, min_periods=10).std()

    et = to_market_time(bars.index)
    bars['session_date'] = session_date(bars.index)
    bars['is_rth'] = is_rth(bars.index)
    bars['et_hour'] = et.hour
    bars['minute_of_day_et'] = et.hour * 60 + et.minute

    # Minutes since the 09:30 ET open, negative before it.
    rth_open = et.normalize() + pd.Timedelta(hours=9, minutes=30)
    bars['mins_from_rth_open'] = (et - rth_open).total_seconds() / 60.0

    # Session CVD, reset each session so the level is comparable across days.
    bars['cvd'] = bars.groupby('session_date')['delta'].cumsum()
    bars['cvd_slope_5'] = (bars['cvd'] - bars['cvd'].shift(5)) / 5.0

    return bars


def add_atr_percentile(bars, lookback_sessions=20):
    """
    ATR percentile rank against the trailing N sessions' median ATR.

    This is the volatility-regime axis used for baseline matching in the event
    study. Uses only completed prior sessions, so there is no lookahead.
    """
    bars = bars.copy()
    session_atr = bars.groupby('session_date')['atr_14'].median()

    ranks = {}
    sessions = session_atr.index.to_list()
    for i, sess in enumerate(sessions):
        if i < 5:
            ranks[sess] = np.nan
            continue
        window = session_atr.iloc[max(0, i - lookback_sessions):i]
        ranks[sess] = float((window < session_atr.iloc[i]).mean())

    bars['atr_percentile'] = bars['session_date'].map(ranks)
    return bars


# ============================================================
# Multi-session driver
# ============================================================

def build_dataset(data_dir, window='1min', volume_per_bar=None,
                  direction_method='level', start=None, end=None,
                  with_value_area=True, verbose=True):
    """
    Stream every session, aggregate to bars, and concatenate.

    Ticks are discarded after each session, so peak memory is one session of
    ticks plus the accumulated bar table. Multi-year runs are fine.
    """
    all_bars = []

    for path, ticks in iter_sessions(data_dir, direction_method=direction_method,
                                     start=start, end=end, verbose=verbose):
        bars, footprint = build_bars(ticks, window=window,
                                     volume_per_bar=volume_per_bar,
                                     with_footprint=with_value_area)
        if len(bars) == 0:
            continue

        if with_value_area and footprint is not None:
            va = value_area_from_footprint(footprint)
            # footprint bar_idx is positional within this session
            va = va.reindex(range(len(bars)))
            va.index = bars.index
            bars = pd.concat([bars, va], axis=1)
            bars['close_vs_poc'] = bars['close'] - bars['poc']
            bars['close_vs_vah'] = bars['close'] - bars['vah']
            bars['close_vs_val'] = bars['close'] - bars['val']
            bars['close_in_value_area'] = (
                (bars['close'] >= bars['val']) & (bars['close'] <= bars['vah'])
            ).astype(np.int8)

        bars['source_file'] = os.path.basename(path)
        all_bars.append(bars)

    if not all_bars:
        raise ValueError("No bars produced")

    combined = pd.concat(all_bars).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]

    combined = add_basic_features(combined)
    combined = add_atr_percentile(combined)

    if verbose:
        n_sessions = combined['session_date'].nunique()
        print(f"\n  Bars: {len(combined):,} across {n_sessions} sessions")
        print(f"  RTH bars: {int(combined['is_rth'].sum()):,}")

    return combined


def main():
    parser = argparse.ArgumentParser(description='Build bars from .data tick files.')
    parser.add_argument('data_dir', help='Directory containing .data files')
    parser.add_argument('--window', default='1min', help='Time bar window (default: 1min)')
    parser.add_argument('--volume-bars', type=int, default=None,
                        help='Use volume bars of this size instead of time bars')
    parser.add_argument('--direction-method', default='level',
                        choices=['level', 'tick', 'mid'],
                        help='Aggressor classification rule (default: level)')
    parser.add_argument('--start', default=None, help='First session, YYYYMMDD')
    parser.add_argument('--end', default=None, help='Last session, YYYYMMDD')
    parser.add_argument('--no-value-area', action='store_true',
                        help='Skip POC/VAH/VAL (faster)')
    parser.add_argument('--out', default=None, help='Output .parquet or .csv path')

    args = parser.parse_args()

    bars = build_dataset(
        args.data_dir,
        window=args.window,
        volume_per_bar=args.volume_bars,
        direction_method=args.direction_method,
        start=args.start,
        end=args.end,
        with_value_area=not args.no_value_area,
    )

    if args.out:
        if args.out.endswith('.parquet'):
            bars.to_parquet(args.out)
        else:
            bars.to_csv(args.out)
        print(f"  Written to {args.out}")
    else:
        print(bars.tail(10).to_string())


if __name__ == '__main__':
    main()
