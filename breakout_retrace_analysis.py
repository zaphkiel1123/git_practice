#!/usr/bin/env python3
"""
Breakout-Retrace-Continuation pattern scanner.

Pattern:
  1. CONSOLIDATION: 5-10 consecutive 1-min candles with >= 2k vol/min
     and a balanced volume profile (symmetric distribution across range).
  2. BREAKOUT: A candle that closes far beyond the range with most volume
     traded away from the range boundary (conviction close).
  3. RETRACEMENT: Within 1-5 candles, price retraces >= 50% back into
     the original consolidation range.
  4. CONTINUATION: From the retracement extreme, measure whether price
     hits +50 ticks before -20 ticks (first-touch barrier).

Outputs probability of 50-tick continuation stratified by:
  - Pre-breakout participant strength (single-event txn > 150)
  - Breakout candle properties (one-sided dominance)
  - Retracement depth (% buckets)
  - Retracement level type (least-activity vs most-one-sided)
  - Retracement volume profile characteristics

Usage:
  python3 breakout_retrace_analysis.py 20250904.data
  python3 breakout_retrace_analysis.py /path/to/data/dir
  python3 breakout_retrace_analysis.py /path/to/data/dir 20250909.data -o report.html --no-browser
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from dataclasses import dataclass

try:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:
    print(
        f"Missing dependency: {exc}\n"
        "See INSTALL.md for packages to install (numpy, pandas, plotly).",
        file=sys.stderr,
    )
    sys.exit(1)

from decode_data import decode_file

TICK_SIZE = 0.25


# ============================================================
# Data loading
# ============================================================

def load_ticks(filepath: str) -> pd.DataFrame | None:
    """Decode binary ticks into a sorted DataFrame. Returns None on failure."""
    filesize = os.path.getsize(filepath)
    if filesize == 0:
        print(f"    SKIP (empty file): {filepath}", file=sys.stderr)
        return None
    if filesize % 57 != 0:
        print(f"    SKIP (size {filesize} not divisible by 57): {filepath}",
              file=sys.stderr)
        return None

    rows = []
    for _, rec in decode_file(filepath):
        rows.append({
            'timestamp': rec['timestamp'],
            'price': rec['price'],
            'direction': rec['direction'],
            'txn_count': rec['txn_count'],
            'volume_delta': rec['volume_delta'],
            'cumulative_volume_delta': rec['cumulative_volume_delta'],
        })
    if not rows:
        print(f"    SKIP (no records decoded): {filepath}", file=sys.stderr)
        return None
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def resolve_inputs(paths: list[str]) -> list[str]:
    """Resolve a mix of files and directories into a sorted list of .data files."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, '*.data')))
            if not found:
                print(f"Warning: no .data files found in directory {p}", file=sys.stderr)
            files.extend(found)
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"Error: path not found: {p}", file=sys.stderr)
            sys.exit(1)
    if not files:
        print("Error: no .data files resolved from the given inputs.", file=sys.stderr)
        sys.exit(1)
    return sorted(set(files))


def load_multiple(filepaths: list[str]) -> pd.DataFrame:
    """Load and concatenate multiple .data files, skipping bad ones."""
    dfs = []
    skipped = 0
    for fp in filepaths:
        print(f"  Loading {fp} ...")
        result = load_ticks(fp)
        if result is not None:
            dfs.append(result)
        else:
            skipped += 1
    if not dfs:
        print("Error: no valid .data files could be decoded.", file=sys.stderr)
        sys.exit(1)
    if skipped:
        print(f"  ({skipped} file(s) skipped due to format/size issues)")
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    return df


# ============================================================
# Bar construction with per-price volume profile
# ============================================================

def build_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ticks into 1-min OHLCV bars."""
    df = ticks.set_index('timestamp').sort_index()
    df['buy_vol'] = df['txn_count'].where(df['direction'] == 'BUY', 0)
    df['sell_vol'] = df['txn_count'].where(df['direction'] == 'SELL', 0)

    resampled = df.resample('1min')
    bars = pd.DataFrame({
        'open': resampled['price'].first(),
        'high': resampled['price'].max(),
        'low': resampled['price'].min(),
        'close': resampled['price'].last(),
        'volume': resampled['txn_count'].sum(),
        'buy_volume': resampled['buy_vol'].sum(),
        'sell_volume': resampled['sell_vol'].sum(),
        'num_events': resampled['price'].count(),
        'max_single_buy': resampled['buy_vol'].max(),
        'max_single_sell': resampled['sell_vol'].max(),
    })
    bars = bars.dropna(subset=['open'])
    bars['range_ticks'] = (bars['high'] - bars['low']) / TICK_SIZE
    bars['body_ticks'] = (bars['close'] - bars['open']) / TICK_SIZE
    bars['imbalance'] = (
        (bars['buy_volume'] - bars['sell_volume']) /
        bars['volume'].replace(0, np.nan)
    )
    return bars


def build_volume_profile(ticks: pd.DataFrame, t_start, t_end) -> pd.DataFrame:
    """Per-price-level buy/sell volume in a time window."""
    mask = (ticks['timestamp'] >= t_start) & (ticks['timestamp'] < t_end)
    window = ticks.loc[mask].copy()
    if window.empty:
        return pd.DataFrame(columns=['price', 'buy_vol', 'sell_vol', 'total_vol'])

    window['buy_vol'] = window['txn_count'].where(window['direction'] == 'BUY', 0)
    window['sell_vol'] = window['txn_count'].where(window['direction'] == 'SELL', 0)

    vp = window.groupby('price').agg(
        buy_vol=('buy_vol', 'sum'),
        sell_vol=('sell_vol', 'sum'),
        total_vol=('txn_count', 'sum'),
    ).reset_index()
    return vp


# ============================================================
# Profile shape metrics
# ============================================================

def profile_balance_score(vp: pd.DataFrame, range_low: float, range_high: float) -> float:
    """Measure how balanced (symmetric) a volume profile is.

    Returns a score 0-1 where 1 = perfectly balanced.
    Uses ratio of volume in top half vs bottom half of the range.
    """
    if vp.empty or range_high <= range_low:
        return 0.0
    mid = (range_low + range_high) / 2
    top_vol = vp.loc[vp['price'] >= mid, 'total_vol'].sum()
    bot_vol = vp.loc[vp['price'] < mid, 'total_vol'].sum()
    total = top_vol + bot_vol
    if total == 0:
        return 0.0
    ratio = min(top_vol, bot_vol) / max(top_vol, bot_vol) if max(top_vol, bot_vol) > 0 else 0
    return float(ratio)


def poc_centrality(vp: pd.DataFrame, range_low: float, range_high: float) -> float:
    """How central is the POC (point of control) within the range. 0=edge, 1=center."""
    if vp.empty or range_high <= range_low:
        return 0.0
    poc_price = vp.loc[vp['total_vol'].idxmax(), 'price']
    mid = (range_low + range_high) / 2
    half_range = (range_high - range_low) / 2
    dist_from_center = abs(poc_price - mid)
    return float(max(0, 1.0 - dist_from_center / half_range))


# ============================================================
# Pattern detection
# ============================================================

@dataclass
class PatternEvent:
    """One detected breakout-retrace-continuation pattern."""
    # Consolidation
    consol_start: pd.Timestamp
    consol_end: pd.Timestamp
    consol_bars: int
    consol_high: float
    consol_low: float
    consol_range_ticks: float
    consol_avg_vol: float
    consol_balance_score: float
    consol_poc_centrality: float

    # Pre-breakout participant strength
    max_single_buy_pre: float
    max_single_sell_pre: float
    dominant_side_pre: str
    dominant_strength_pre: float

    # Breakout
    breakout_time: pd.Timestamp
    breakout_direction: str
    breakout_close: float
    breakout_body_ticks: float
    breakout_volume: float
    breakout_imbalance: float
    breakout_vwap_position: float

    # Retracement
    retrace_bars: int
    retrace_extreme: float
    retrace_pct: float
    retrace_level_type: str
    retrace_buy_vol: float
    retrace_sell_vol: float
    retrace_imbalance: float

    # Continuation (50/20 result)
    entry_price: float
    outcome: str
    ticks_to_resolve: int
    seconds_to_resolve: float


def detect_consolidations(
    bars: pd.DataFrame,
    min_bars: int = 5,
    max_bars: int = 10,
    min_vol_per_bar: float = 2000,
) -> list[tuple[int, int]]:
    """Find runs of bars that qualify as consolidation zones.

    Returns list of (start_idx, end_idx) into bars DataFrame (inclusive).
    """
    n = len(bars)
    zones = []
    i = 0
    while i <= n - min_bars:
        best_end = None
        for length in range(min_bars, min(max_bars + 1, n - i + 1)):
            window = bars.iloc[i:i + length]
            avg_vol = window['volume'].mean()
            if avg_vol < min_vol_per_bar:
                break
            best_end = i + length - 1

        if best_end is not None and (best_end - i + 1) >= min_bars:
            zones.append((i, best_end))
            i = best_end + 1
        else:
            i += 1
    return zones


def _breakout_vwap_position(
    ticks: pd.DataFrame,
    bar_start: pd.Timestamp,
    bar_end: pd.Timestamp,
    range_boundary: float,
    close: float,
) -> float:
    """Where is VWAP of breakout candle between range boundary and close.

    0 = VWAP at range boundary, 1 = VWAP at close.
    """
    mask = (ticks['timestamp'] >= bar_start) & (ticks['timestamp'] < bar_end)
    w = ticks.loc[mask]
    if w.empty or abs(close - range_boundary) < 1e-9:
        return 0.5
    vwap = (w['price'] * w['txn_count']).sum() / w['txn_count'].sum()
    position = (vwap - range_boundary) / (close - range_boundary)
    return float(np.clip(position, 0.0, 1.0))


def _classify_retrace_level(
    vp_consol: pd.DataFrame,
    retrace_price: float,
) -> str:
    """Is the retracement level at a low-activity or high-one-sided price?"""
    if vp_consol.empty:
        return 'unknown'
    diffs = (vp_consol['price'] - retrace_price).abs()
    nearest_idx = diffs.idxmin()
    row = vp_consol.loc[nearest_idx]
    total = row['total_vol']
    median_vol = vp_consol['total_vol'].median()

    if total < median_vol * 0.5:
        return 'low_activity'
    buy = row['buy_vol']
    sell = row['sell_vol']
    if (buy + sell) > 0:
        ratio = max(buy, sell) / (buy + sell)
        if ratio > 0.65:
            return 'high_one_sided'
    return 'neutral'


def scan_patterns(
    ticks: pd.DataFrame,
    bars: pd.DataFrame,
    min_consol_bars: int = 5,
    max_consol_bars: int = 10,
    min_vol_per_bar: float = 2000,
    min_balance: float = 0.4,
    min_breakout_body: float = 4.0,
    max_retrace_bars: int = 5,
    min_retrace_pct: float = 0.50,
    target_ticks: int = 50,
    stop_ticks: int = 20,
) -> list[PatternEvent]:
    """Scan bars for the full breakout-retrace-continuation pattern."""

    prices_arr = ticks['price'].to_numpy(dtype=float)
    ts_arr = ticks['timestamp'].to_numpy()
    n_ticks = len(ticks)

    zones = detect_consolidations(
        bars, min_consol_bars, max_consol_bars, min_vol_per_bar,
    )

    events: list[PatternEvent] = []

    for (z_start, z_end) in zones:
        consol = bars.iloc[z_start:z_end + 1]
        consol_high = float(consol['high'].max())
        consol_low = float(consol['low'].min())
        consol_range = (consol_high - consol_low) / TICK_SIZE
        avg_vol = float(consol['volume'].mean())

        # Volume profile for consolidation
        t_consol_start = consol.index[0]
        t_consol_end = consol.index[-1] + pd.Timedelta('1min')
        vp_consol = build_volume_profile(ticks, t_consol_start, t_consol_end)

        balance = profile_balance_score(vp_consol, consol_low, consol_high)
        if balance < min_balance:
            continue
        poc_cent = poc_centrality(vp_consol, consol_low, consol_high)

        # Pre-breakout participant strength
        max_buy_pre = float(consol['max_single_buy'].max())
        max_sell_pre = float(consol['max_single_sell'].max())
        if max_buy_pre >= max_sell_pre:
            dom_side = 'BUY'
            dom_strength = max_buy_pre
        else:
            dom_side = 'SELL'
            dom_strength = max_sell_pre

        # Look for breakout candle immediately after consolidation
        bo_idx = z_end + 1
        if bo_idx >= len(bars):
            continue
        bo_bar = bars.iloc[bo_idx]

        # Determine breakout direction
        if bo_bar['close'] > consol_high:
            bo_dir = 'UP'
            range_boundary = consol_high
        elif bo_bar['close'] < consol_low:
            bo_dir = 'DOWN'
            range_boundary = consol_low
        else:
            continue

        body = abs(bo_bar['body_ticks'])
        if body < min_breakout_body:
            continue

        bo_time = bars.index[bo_idx]
        bo_end_time = bo_time + pd.Timedelta('1min')

        vwap_pos = _breakout_vwap_position(
            ticks, bo_time, bo_end_time, range_boundary, bo_bar['close'],
        )

        # --- Retracement phase ---
        for r_len in range(1, min(max_retrace_bars + 1, len(bars) - bo_idx)):
            r_start = bo_idx + 1
            r_end = bo_idx + r_len
            if r_end >= len(bars):
                break
            retrace_window = bars.iloc[r_start:r_end + 1]

            if bo_dir == 'UP':
                retrace_extreme = float(retrace_window['low'].min())
                bo_extreme = float(bo_bar['close'])
                if bo_extreme <= consol_high:
                    continue
                retrace_dist = bo_extreme - retrace_extreme
                total_dist = bo_extreme - consol_low
                retrace_pct = retrace_dist / total_dist if total_dist > 0 else 0
            else:
                retrace_extreme = float(retrace_window['high'].max())
                bo_extreme = float(bo_bar['close'])
                if bo_extreme >= consol_low:
                    continue
                retrace_dist = retrace_extreme - bo_extreme
                total_dist = consol_high - bo_extreme
                retrace_pct = retrace_dist / total_dist if total_dist > 0 else 0

            if retrace_pct >= min_retrace_pct:
                # Retracement volume characteristics
                r_buy = float(retrace_window['buy_volume'].sum())
                r_sell = float(retrace_window['sell_volume'].sum())
                r_total = r_buy + r_sell
                r_imbalance = (r_buy - r_sell) / r_total if r_total > 0 else 0

                level_type = _classify_retrace_level(vp_consol, retrace_extreme)

                # --- Continuation: 50/20 first-touch from retrace extreme ---
                entry_time = bars.index[r_end] + pd.Timedelta('1min')
                entry_tick_idx = int(np.searchsorted(ts_arr, np.datetime64(entry_time), side='left'))
                if entry_tick_idx >= n_ticks:
                    break

                entry_price = float(prices_arr[entry_tick_idx])

                target_price = (
                    entry_price + target_ticks * TICK_SIZE if bo_dir == 'UP'
                    else entry_price - target_ticks * TICK_SIZE
                )
                stop_price = (
                    entry_price - stop_ticks * TICK_SIZE if bo_dir == 'UP'
                    else entry_price + stop_ticks * TICK_SIZE
                )

                outcome = 'timeout'
                resolve_idx = n_ticks - 1
                for j in range(entry_tick_idx + 1, n_ticks):
                    p = prices_arr[j]
                    if bo_dir == 'UP':
                        if p >= target_price:
                            outcome = 'success'
                            resolve_idx = j
                            break
                        if p <= stop_price:
                            outcome = 'fail'
                            resolve_idx = j
                            break
                    else:
                        if p <= target_price:
                            outcome = 'success'
                            resolve_idx = j
                            break
                        if p >= stop_price:
                            outcome = 'fail'
                            resolve_idx = j
                            break

                resolve_time = ticks.iloc[resolve_idx]['timestamp']
                secs = (pd.Timestamp(resolve_time) - pd.Timestamp(entry_time)).total_seconds()

                events.append(PatternEvent(
                    consol_start=t_consol_start,
                    consol_end=t_consol_end,
                    consol_bars=z_end - z_start + 1,
                    consol_high=consol_high,
                    consol_low=consol_low,
                    consol_range_ticks=consol_range,
                    consol_avg_vol=avg_vol,
                    consol_balance_score=balance,
                    consol_poc_centrality=poc_cent,
                    max_single_buy_pre=max_buy_pre,
                    max_single_sell_pre=max_sell_pre,
                    dominant_side_pre=dom_side,
                    dominant_strength_pre=dom_strength,
                    breakout_time=bo_time,
                    breakout_direction=bo_dir,
                    breakout_close=float(bo_bar['close']),
                    breakout_body_ticks=float(body),
                    breakout_volume=float(bo_bar['volume']),
                    breakout_imbalance=float(bo_bar['imbalance']),
                    breakout_vwap_position=vwap_pos,
                    retrace_bars=r_len,
                    retrace_extreme=retrace_extreme,
                    retrace_pct=retrace_pct,
                    retrace_level_type=level_type,
                    retrace_buy_vol=r_buy,
                    retrace_sell_vol=r_sell,
                    retrace_imbalance=r_imbalance,
                    entry_price=entry_price,
                    outcome=outcome,
                    ticks_to_resolve=resolve_idx - entry_tick_idx,
                    seconds_to_resolve=secs,
                ))
                break  # take first valid retracement

    return events


# ============================================================
# Analysis and statistics
# ============================================================

def events_to_dataframe(events: list[PatternEvent]) -> pd.DataFrame:
    """Convert pattern events to a DataFrame for analysis."""
    if not events:
        return pd.DataFrame()
    return pd.DataFrame([ev.__dict__ for ev in events])


def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute probability breakdowns stratified by various factors."""
    if df.empty:
        return {}

    stats = {}

    resolved = df[df['outcome'] != 'timeout']
    n_total = len(resolved)
    n_success = (resolved['outcome'] == 'success').sum()
    stats['overall'] = {
        'total_patterns': len(df),
        'resolved': n_total,
        'timeouts': (df['outcome'] == 'timeout').sum(),
        'success': int(n_success),
        'fail': int((resolved['outcome'] == 'fail').sum()),
        'hit_rate': float(n_success / n_total) if n_total > 0 else 0,
    }

    # By direction
    stats['by_direction'] = {}
    for d in ('UP', 'DOWN'):
        sub = resolved[resolved['breakout_direction'] == d]
        n = len(sub)
        s = (sub['outcome'] == 'success').sum()
        stats['by_direction'][d] = {
            'n': int(n), 'success': int(s),
            'hit_rate': float(s / n) if n > 0 else 0,
        }

    # By pre-breakout strength (>150 threshold)
    stats['by_pre_strength'] = {}
    for label, mask in [
        ('strong_dominant_>150', resolved['dominant_strength_pre'] > 150),
        ('weak_dominant_<=150', resolved['dominant_strength_pre'] <= 150),
    ]:
        sub = resolved[mask]
        n = len(sub)
        s = (sub['outcome'] == 'success').sum()
        stats['by_pre_strength'][label] = {
            'n': int(n), 'success': int(s),
            'hit_rate': float(s / n) if n > 0 else 0,
        }

    # By breakout imbalance strength
    stats['by_breakout_imbalance'] = {}
    for label, lo, hi in [
        ('strong_buy (>0.5)', 0.5, 1.01),
        ('moderate (0.2-0.5)', 0.2, 0.5),
        ('weak (<0.2)', -0.2, 0.2),
        ('moderate_sell (-0.5--0.2)', -0.5, -0.2),
        ('strong_sell (<-0.5)', -1.01, -0.5),
    ]:
        sub = resolved[
            (resolved['breakout_imbalance'] >= lo) &
            (resolved['breakout_imbalance'] < hi)
        ]
        n = len(sub)
        s = (sub['outcome'] == 'success').sum()
        if n > 0:
            stats['by_breakout_imbalance'][label] = {
                'n': int(n), 'success': int(s),
                'hit_rate': float(s / n) if n > 0 else 0,
            }

    # By retracement percentage buckets
    stats['by_retrace_pct'] = {}
    bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    for lo, hi in bins:
        label = f'{lo:.0%}-{hi:.0%}'
        sub = resolved[(resolved['retrace_pct'] >= lo) & (resolved['retrace_pct'] < hi)]
        n = len(sub)
        s = (sub['outcome'] == 'success').sum()
        if n > 0:
            stats['by_retrace_pct'][label] = {
                'n': int(n), 'success': int(s),
                'hit_rate': float(s / n) if n > 0 else 0,
            }

    # By retracement level type
    stats['by_retrace_level'] = {}
    for lvl in ('low_activity', 'high_one_sided', 'neutral', 'unknown'):
        sub = resolved[resolved['retrace_level_type'] == lvl]
        n = len(sub)
        s = (sub['outcome'] == 'success').sum()
        if n > 0:
            stats['by_retrace_level'][lvl] = {
                'n': int(n), 'success': int(s),
                'hit_rate': float(s / n) if n > 0 else 0,
            }

    # By retracement volume imbalance (against breakout direction = absorption)
    stats['by_retrace_imbalance'] = {}
    for label, condition_fn in [
        ('retrace_absorbed (counter-dir dominant)', lambda row: (
            (row['breakout_direction'] == 'UP' and row['retrace_imbalance'] < -0.3) or
            (row['breakout_direction'] == 'DOWN' and row['retrace_imbalance'] > 0.3)
        )),
        ('retrace_weak (same-dir or neutral)', lambda row: not (
            (row['breakout_direction'] == 'UP' and row['retrace_imbalance'] < -0.3) or
            (row['breakout_direction'] == 'DOWN' and row['retrace_imbalance'] > 0.3)
        )),
    ]:
        mask = resolved.apply(condition_fn, axis=1)
        sub = resolved[mask]
        n = len(sub)
        s = (sub['outcome'] == 'success').sum()
        if n > 0:
            stats['by_retrace_imbalance'][label] = {
                'n': int(n), 'success': int(s),
                'hit_rate': float(s / n) if n > 0 else 0,
            }

    # By VWAP position (conviction of breakout)
    stats['by_vwap_conviction'] = {}
    for label, lo, hi in [
        ('high_conviction (>0.7)', 0.7, 1.01),
        ('moderate (0.4-0.7)', 0.4, 0.7),
        ('low_conviction (<0.4)', 0.0, 0.4),
    ]:
        sub = resolved[
            (resolved['breakout_vwap_position'] >= lo) &
            (resolved['breakout_vwap_position'] < hi)
        ]
        n = len(sub)
        s = (sub['outcome'] == 'success').sum()
        if n > 0:
            stats['by_vwap_conviction'][label] = {
                'n': int(n), 'success': int(s),
                'hit_rate': float(s / n) if n > 0 else 0,
            }

    return stats


# ============================================================
# Reporting
# ============================================================

def print_report(stats: dict, df: pd.DataFrame) -> None:
    """Print text report of probabilities."""
    print('\n' + '=' * 70)
    print('  BREAKOUT-RETRACE-CONTINUATION PROBABILITY REPORT')
    print('=' * 70)

    ov = stats.get('overall', {})
    print(f"\n  Total patterns detected:  {ov.get('total_patterns', 0)}")
    print(f"  Resolved (excl timeout):  {ov.get('resolved', 0)}")
    print(f"  Timeouts:                 {ov.get('timeouts', 0)}")
    print(f"  Success (hit +50):        {ov.get('success', 0)}")
    print(f"  Fail (hit -20):           {ov.get('fail', 0)}")
    print(f"  Overall hit rate:         {ov.get('hit_rate', 0):.1%}")

    sections = [
        ('BY DIRECTION', 'by_direction'),
        ('BY PRE-BREAKOUT PARTICIPANT STRENGTH (single event >150)', 'by_pre_strength'),
        ('BY BREAKOUT CANDLE IMBALANCE', 'by_breakout_imbalance'),
        ('BY RETRACEMENT % INTO RANGE', 'by_retrace_pct'),
        ('BY RETRACEMENT LEVEL TYPE', 'by_retrace_level'),
        ('BY RETRACEMENT VOLUME CHARACTER', 'by_retrace_imbalance'),
        ('BY BREAKOUT VWAP CONVICTION', 'by_vwap_conviction'),
    ]

    for title, key in sections:
        data = stats.get(key, {})
        if not data:
            continue
        print(f'\n  --- {title} ---')
        print(f'  {"Category":<40} {"N":>5} {"Success":>8} {"Hit Rate":>9}')
        for cat, vals in data.items():
            print(f'  {cat:<40} {vals["n"]:>5} {vals["success"]:>8} '
                  f'{vals["hit_rate"]:>8.1%}')

    print('\n' + '=' * 70)


def build_html_report(
    stats: dict,
    df: pd.DataFrame,
    title: str,
) -> go.Figure:
    """Build interactive Plotly figure summarizing probabilities."""
    sections = [
        ('Direction', 'by_direction'),
        ('Pre-Break Strength', 'by_pre_strength'),
        ('Breakout Imbalance', 'by_breakout_imbalance'),
        ('Retrace % Bucket', 'by_retrace_pct'),
        ('Retrace Level Type', 'by_retrace_level'),
        ('Retrace Volume', 'by_retrace_imbalance'),
        ('VWAP Conviction', 'by_vwap_conviction'),
    ]

    active = [(t, k) for t, k in sections if stats.get(k)]
    n_panels = len(active) + 1
    cols = 2
    rows = (n_panels + 1) // 2

    subtitles = ['Overall Summary'] + [t for t, _ in active]
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subtitles,
        specs=[[{'type': 'bar'}] * cols for _ in range(rows)],
        vertical_spacing=0.08,
        horizontal_spacing=0.12,
    )

    # Overall summary bar
    ov = stats.get('overall', {})
    fig.add_trace(
        go.Bar(
            x=['Success', 'Fail', 'Timeout'],
            y=[ov.get('success', 0), ov.get('fail', 0), ov.get('timeouts', 0)],
            marker_color=['#2e7d32', '#c62828', '#9e9e9e'],
            text=[
                f"{ov.get('success',0)} ({ov.get('hit_rate',0):.0%})",
                str(ov.get('fail', 0)),
                str(ov.get('timeouts', 0)),
            ],
            textposition='outside',
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Each stratification as hit-rate bar chart
    for panel_i, (panel_title, key) in enumerate(active, start=1):
        data = stats[key]
        r = (panel_i // cols) + 1
        c = (panel_i % cols) + 1
        cats = list(data.keys())
        rates = [data[k]['hit_rate'] for k in cats]
        ns = [data[k]['n'] for k in cats]
        fig.add_trace(
            go.Bar(
                x=cats,
                y=rates,
                marker_color='#2e7d32',
                text=[f'{rate:.0%} (n={n})' for rate, n in zip(rates, ns)],
                textposition='outside',
                showlegend=False,
                hovertemplate='%{x}<br>hit_rate=%{y:.1%}<br>n=%{customdata}<extra></extra>',
                customdata=ns,
            ),
            row=r, col=c,
        )
        fig.update_yaxes(range=[0, 1.0], tickformat='.0%', row=r, col=c)

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=350 * rows,
        margin=dict(l=60, r=30, t=80, b=40),
    )
    return fig


# ============================================================
# Main
# ============================================================

def run(
    filepaths: list[str],
    min_consol_bars: int = 5,
    max_consol_bars: int = 10,
    min_vol_per_bar: float = 2000,
    min_balance: float = 0.4,
    min_breakout_body: float = 4.0,
    max_retrace_bars: int = 5,
    min_retrace_pct: float = 0.50,
    target_ticks: int = 50,
    stop_ticks: int = 20,
    output: str | None = None,
    no_browser: bool = False,
) -> None:
    print('Loading tick data ...')
    ticks = load_multiple(filepaths)
    print(f'  Total: {len(ticks):,} ticks')

    print('Building 1-min bars ...')
    bars = build_bars(ticks)
    print(f'  {len(bars):,} bars')

    print('Scanning for breakout-retrace-continuation patterns ...')
    events = scan_patterns(
        ticks, bars,
        min_consol_bars=min_consol_bars,
        max_consol_bars=max_consol_bars,
        min_vol_per_bar=min_vol_per_bar,
        min_balance=min_balance,
        min_breakout_body=min_breakout_body,
        max_retrace_bars=max_retrace_bars,
        min_retrace_pct=min_retrace_pct,
        target_ticks=target_ticks,
        stop_ticks=stop_ticks,
    )
    print(f'  Found {len(events)} pattern instances')

    df = events_to_dataframe(events)
    if df.empty:
        print('\nNo patterns found with current parameters. Try relaxing constraints:')
        print('  --min-vol 1000  --min-balance 0.3  --min-body 2  --min-retrace 0.4')
        return

    stats = compute_statistics(df)
    print_report(stats, df)

    # Save CSV
    stem = Path(filepaths[0]).stem
    output = output or f'{stem}_breakout_retrace.html'
    csv_path = str(Path(output).with_suffix('.csv'))
    df.to_csv(csv_path, index=False)
    print(f'\nSaved events CSV: {csv_path}')

    # Build HTML
    title = (
        f'Breakout-Retrace-Continuation | target={target_ticks} stop={stop_ticks} '
        f'| consol {min_consol_bars}-{max_consol_bars} bars >= {min_vol_per_bar} vol/min'
    )
    fig = build_html_report(stats, df, title)
    fig.write_html(output, include_plotlyjs='cdn', auto_open=not no_browser)
    print(f'Saved HTML report: {output}')


def main():
    parser = argparse.ArgumentParser(
        description='Breakout-retrace-continuation pattern probability analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 20250904.data
  %(prog)s /path/to/data/directory
  %(prog)s /path/to/dir 20250909.data --min-vol 1500
  %(prog)s 20250904.data --min-retrace 0.6 --target-ticks 40 -o report.html
        """,
    )
    parser.add_argument('input_paths', nargs='+',
                        help='Binary .data file(s) and/or directories containing .data files')
    parser.add_argument('--min-consol', type=int, default=5,
                        help='Min consolidation bars (default: 5)')
    parser.add_argument('--max-consol', type=int, default=10,
                        help='Max consolidation bars (default: 10)')
    parser.add_argument('--min-vol', type=float, default=2000,
                        help='Min volume per bar in consolidation (default: 2000)')
    parser.add_argument('--min-balance', type=float, default=0.4,
                        help='Min balance score for consolidation profile (default: 0.4)')
    parser.add_argument('--min-body', type=float, default=4.0,
                        help='Min breakout candle body in ticks (default: 4)')
    parser.add_argument('--max-retrace-bars', type=int, default=5,
                        help='Max bars for retracement (default: 5)')
    parser.add_argument('--min-retrace', type=float, default=0.50,
                        help='Min retracement %% back into range (default: 0.50)')
    parser.add_argument('--target-ticks', type=int, default=50,
                        help='Profit target in ticks (default: 50)')
    parser.add_argument('--stop-ticks', type=int, default=20,
                        help='Stop loss in ticks (default: 20)')
    parser.add_argument('-o', '--output', default=None, help='Output HTML path')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not auto-open browser')
    args = parser.parse_args()

    filepaths = resolve_inputs(args.input_paths)
    print(f'Resolved {len(filepaths)} .data file(s):')
    for fp in filepaths:
        print(f'  {fp}')

    run(
        filepaths,
        min_consol_bars=args.min_consol,
        max_consol_bars=args.max_consol,
        min_vol_per_bar=args.min_vol,
        min_balance=args.min_balance,
        min_breakout_body=args.min_body,
        max_retrace_bars=args.max_retrace_bars,
        min_retrace_pct=args.min_retrace,
        target_ticks=args.target_ticks,
        stop_ticks=args.stop_ticks,
        output=args.output,
        no_browser=args.no_browser,
    )


if __name__ == '__main__':
    main()
