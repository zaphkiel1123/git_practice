#!/usr/bin/env python3
"""
Self-contained feature pipeline for ES Mini NN model.

Reads raw .data binary tick files, aggregates to 1-min bars, and computes
all 26 sequence channels specified in NN_model_features.md.

No dependency on es_training/ or any external module beyond numpy/pandas.

Usage:
    python nn_feature_pipeline.py /path/to/data/ --output features.parquet
    python nn_feature_pipeline.py /path/to/data/ --workers 8
"""

from __future__ import annotations

import os
import struct
import glob
import argparse
import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd


# ============================================================
# Constants
# ============================================================

TICK_SIZE = 0.25
RECORD_SIZE = 57
RECORD_STRUCT = struct.Struct('<q3d6IB')
DOTNET_EPOCH = datetime(1, 1, 1)

SEQ_LEN = 60
NUM_CHANNELS = 26

CHANNEL_NAMES = [
    "log_return", "bar_range_atr", "clv",
    "volume", "volume_delta", "volume_delta_pct", "buy_volume_pct", "cvd_slope_5",
    "close_vs_poc", "close_vs_vah", "close_vs_val",
    "close_in_value_area", "close_above_vah", "close_below_val",
    "vp60_vol_at_close_pct",
    "leg_size", "retrace_pct", "bars_since_swing_high", "bars_since_swing_low",
    "dist_to_swing_high_pct", "dist_to_swing_low_pct",
    "mins_from_rth_open",
    "atr_ratio", "range_vs_atr", "realized_vol_20", "atr_percentile_20d",
]


# ============================================================
# Binary Tick Decoding
# ============================================================

def _ticks_to_datetime(ticks: int) -> datetime:
    return DOTNET_EPOCH + timedelta(microseconds=ticks // 10)


def _infer_direction(price: float, level_low: float, level_high: float) -> int:
    if abs(price - level_high) < 1e-9:
        return 1   # BUY
    elif abs(price - level_low) < 1e-9:
        return -1  # SELL
    return 0


def decode_file(filepath: str) -> pd.DataFrame:
    """Decode a binary .data file into a DataFrame of ticks."""
    filesize = os.path.getsize(filepath)
    num_records = filesize // RECORD_SIZE

    timestamps = np.empty(num_records, dtype='datetime64[us]')
    prices = np.empty(num_records, dtype=np.float64)
    level_lows = np.empty(num_records, dtype=np.float64)
    level_highs = np.empty(num_records, dtype=np.float64)
    directions = np.empty(num_records, dtype=np.int8)
    txn_counts = np.empty(num_records, dtype=np.uint32)

    with open(filepath, 'rb') as f:
        buf = f.read()

    for i in range(num_records):
        offset = i * RECORD_SIZE
        fields = RECORD_STRUCT.unpack_from(buf, offset)
        dt = DOTNET_EPOCH + timedelta(microseconds=fields[0] // 10)
        timestamps[i] = np.datetime64(dt, 'us')
        prices[i] = fields[1]
        level_lows[i] = fields[2]
        level_highs[i] = fields[3]
        directions[i] = _infer_direction(fields[1], fields[2], fields[3])
        txn_counts[i] = fields[4]

    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'price': prices,
        'level_low': level_lows,
        'level_high': level_highs,
        'direction': directions,
        'txn_count': txn_counts,
    })
    return df


# ============================================================
# Tick-to-Bar Aggregation
# ============================================================

def _aggregate_ticks_to_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Resample ticks to 1-min bars with OHLC, volume, buy/sell split, and vol_at_price."""
    df = df.set_index('timestamp').sort_index()

    df['buy_vol'] = np.where(df['direction'] == 1, df['txn_count'], 0).astype(np.float64)
    df['sell_vol'] = np.where(df['direction'] == -1, df['txn_count'], 0).astype(np.float64)

    resampled = df.resample('1min')

    bars = pd.DataFrame()
    bars['open'] = resampled['price'].first()
    bars['high'] = resampled['price'].max()
    bars['low'] = resampled['price'].min()
    bars['close'] = resampled['price'].last()
    bars['volume'] = resampled['txn_count'].sum().astype(np.float64)
    bars['buy_volume'] = resampled['buy_vol'].sum()
    bars['sell_volume'] = resampled['sell_vol'].sum()
    bars['num_trades'] = resampled['price'].count().astype(np.float64)

    bars = bars.dropna(subset=['open'])
    return bars


def _build_vol_at_price_for_bars(df: pd.DataFrame) -> list[dict]:
    """Build per-bar volume-at-price histograms from ticks. Returns list aligned to bars."""
    df = df.set_index('timestamp').sort_index()
    profiles = []
    for _, group in df.resample('1min'):
        if len(group) == 0:
            profiles.append({})
            continue
        vap = {}
        for price, vol in zip(group['price'].values, group['txn_count'].values):
            vap[price] = vap.get(price, 0.0) + float(vol)
        profiles.append(vap)
    return profiles


def decode_and_aggregate(filepath: str) -> tuple[pd.DataFrame, list[dict]]:
    """Decode a .data file and return (bars_df, vol_at_price_list)."""
    df = decode_file(filepath)
    if len(df) == 0:
        return pd.DataFrame(), []
    bars = _aggregate_ticks_to_bars(df)
    profiles = _build_vol_at_price_for_bars(df)
    # Align profiles to bars (drop empty-bar entries)
    valid_mask = []
    df_indexed = df.set_index('timestamp').sort_index()
    bar_timestamps = bars.index
    profiles_aligned = []
    for _, group in df_indexed.resample('1min'):
        if len(group) == 0:
            continue
        vap = {}
        for price, vol in zip(group['price'].values, group['txn_count'].values):
            vap[price] = vap.get(price, 0.0) + float(vol)
        profiles_aligned.append(vap)
    return bars, profiles_aligned


def load_data(data_dir: str, workers: int = 0) -> tuple[pd.DataFrame, list[dict]]:
    """
    Load all .data files, decode, and aggregate to 1-min bars.
    Uses multi-core when workers > 1.
    """
    files = sorted(glob.glob(os.path.join(data_dir, '*.data')))
    if not files:
        raise FileNotFoundError(f"No .data files in {data_dir}")

    print(f"Found {len(files)} .data file(s)")
    t0 = time.time()

    n_workers = workers if workers > 0 else min(os.cpu_count() or 1, len(files))

    if n_workers > 1 and len(files) > 1:
        print(f"  Parallel decoding: {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(decode_and_aggregate, files))
    else:
        results = [decode_and_aggregate(f) for f in files]

    all_bars = []
    all_profiles = []
    for bars, profiles in results:
        if len(bars) > 0:
            all_bars.append(bars)
            all_profiles.extend(profiles)

    if not all_bars:
        raise ValueError("No bars produced from data files")

    combined = pd.concat(all_bars).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    print(f"  Total bars: {len(combined):,} [{time.time()-t0:.1f}s]")
    return combined, all_profiles


# ============================================================
# Volume Profile Computation (per-bar POC/VAH/VAL)
# ============================================================

def _compute_value_area(vol_at_price: dict, value_area_pct: float = 0.70) -> tuple:
    """Compute POC, VAH, VAL from a volume-at-price histogram."""
    if not vol_at_price:
        return np.nan, np.nan, np.nan

    total_vol = sum(vol_at_price.values())
    if total_vol <= 0:
        return np.nan, np.nan, np.nan

    poc = max(vol_at_price, key=vol_at_price.get)
    target_vol = total_vol * value_area_pct
    captured = vol_at_price[poc]
    vah = poc
    val = poc

    while captured < target_vol:
        vol_above = vol_at_price.get(vah + TICK_SIZE, 0)
        vol_below = vol_at_price.get(val - TICK_SIZE, 0)
        if vol_above == 0 and vol_below == 0:
            break
        if vol_above >= vol_below:
            vah += TICK_SIZE
            captured += vol_above
        else:
            val -= TICK_SIZE
            captured += vol_below

    return poc, vah, val


def _compute_bar_profiles_chunk(args: tuple) -> np.ndarray:
    """Compute POC/VAH/VAL for a chunk of bars. Returns (chunk_len, 3) array."""
    profiles_chunk, = args
    n = len(profiles_chunk)
    result = np.full((n, 3), np.nan, dtype=np.float64)
    for i, vap in enumerate(profiles_chunk):
        poc, vah, val = _compute_value_area(vap)
        result[i] = [poc, vah, val]
    return result


def compute_bar_volume_profiles(profiles: list[dict], n_workers: int = 0) -> np.ndarray:
    """Compute POC/VAH/VAL for all bars. Returns (n_bars, 3) array [poc, vah, val]."""
    n = len(profiles)
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 8)

    chunk_size = max(1, n // n_workers)
    chunks = []
    for i in range(0, n, chunk_size):
        chunks.append((profiles[i:i + chunk_size],))

    if n_workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_compute_bar_profiles_chunk, chunks))
    else:
        results = [_compute_bar_profiles_chunk(c) for c in chunks]

    return np.vstack(results)


# ============================================================
# Rolling 60-bar Volume Profile (vp60_vol_at_close_pct)
# ============================================================

def _merge_profiles(profile_list: list[dict]) -> dict:
    """Merge multiple volume-at-price dicts into one."""
    merged = {}
    for prof in profile_list:
        if not prof:
            continue
        for price, vol in prof.items():
            merged[price] = merged.get(price, 0.0) + vol
    return merged


def _round_to_tick(price: float) -> float:
    return round(price / TICK_SIZE) * TICK_SIZE


def _compute_vp60_chunk(args: tuple) -> np.ndarray:
    """Compute vp60_vol_at_close_pct for a chunk of bar indices."""
    start_idx, end_idx, profiles_all, close_prices = args
    n = end_idx - start_idx
    result = np.full(n, np.nan, dtype=np.float64)

    for local_i in range(n):
        i = start_idx + local_i
        if i < 60:
            continue
        merged = _merge_profiles(profiles_all[i - 60:i])
        if not merged:
            continue
        poc_price = max(merged, key=merged.get)
        vol_at_poc = merged[poc_price]
        if vol_at_poc <= 0:
            continue
        price_key = _round_to_tick(close_prices[i])
        vol_at_close = merged.get(price_key, 0.0)
        result[local_i] = min(100.0, vol_at_close / vol_at_poc * 100.0)

    return result


def compute_vp60(profiles: list[dict], close_prices: np.ndarray,
                 n_workers: int = 0) -> np.ndarray:
    """Compute vp60_vol_at_close_pct for all bars. Returns (n_bars,) array."""
    n = len(profiles)
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 8)

    chunk_size = max(1, n // n_workers)
    chunks = []
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunks.append((i, end, profiles, close_prices))

    if n_workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_compute_vp60_chunk, chunks))
    else:
        results = [_compute_vp60_chunk(c) for c in chunks]

    return np.concatenate(results)


# ============================================================
# ATR Computation
# ============================================================

def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    """Compute Average True Range (vectorized)."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().values
    return atr


# ============================================================
# CVD Computation
# ============================================================

def compute_cvd(volume_delta: np.ndarray, timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Compute Cumulative Volume Delta with session reset at RTH open (9:30 ET)."""
    n = len(volume_delta)
    cvd = np.zeros(n, dtype=np.float64)

    try:
        ts_et = timestamps.tz_localize('America/Chicago').tz_convert('America/New_York')
    except TypeError:
        ts_et = timestamps.tz_convert('America/New_York')

    hours = ts_et.hour
    minutes = ts_et.minute
    hm = hours * 100 + minutes

    running = 0.0
    prev_hm = -1
    for i in range(n):
        cur_hm = hm[i]
        # Reset at RTH open 9:30 ET
        if cur_hm == 930 and prev_hm < 930:
            running = 0.0
        elif cur_hm < prev_hm and prev_hm >= 1600:
            running = 0.0
        running += volume_delta[i]
        cvd[i] = running
        prev_hm = cur_hm

    return cvd


# ============================================================
# Swing Detection (Market Structure)
# ============================================================

def detect_swings(high: np.ndarray, low: np.ndarray, k: int = 5) -> dict:
    """
    Detect swing highs/lows with confirmation lag k.
    Returns dict with arrays for structure features.

    Swing high at bar i: high[i] == max(high[i-k .. i+k])
    Swing low at bar i:  low[i] == min(low[i-k .. i+k])
    Only confirmed swings (known k bars after the pivot).
    Alternating filter: SH → SL → SH → SL.
    """
    n = len(high)

    # Detect raw swing points
    raw_swing_high = np.zeros(n, dtype=bool)
    raw_swing_low = np.zeros(n, dtype=bool)

    for i in range(k, n - k):
        window_high = high[max(0, i - k):i + k + 1]
        if high[i] == window_high.max() and np.sum(window_high == high[i]) == 1:
            raw_swing_high[i] = True
        window_low = low[max(0, i - k):i + k + 1]
        if low[i] == window_low.min() and np.sum(window_low == low[i]) == 1:
            raw_swing_low[i] = True

    # Alternating filter
    swing_high_indices = []
    swing_low_indices = []
    last_type = None  # 'H' or 'L'

    all_swings = []
    for i in range(n):
        if raw_swing_high[i]:
            all_swings.append((i, 'H', high[i]))
        if raw_swing_low[i]:
            all_swings.append((i, 'L', low[i]))
    all_swings.sort(key=lambda x: x[0])

    for idx, stype, price in all_swings:
        if stype == 'H':
            if last_type == 'H':
                if price > high[swing_high_indices[-1]]:
                    swing_high_indices[-1] = idx
            else:
                swing_high_indices.append(idx)
                last_type = 'H'
        else:
            if last_type == 'L':
                if price < low[swing_low_indices[-1]]:
                    swing_low_indices[-1] = idx
            else:
                swing_low_indices.append(idx)
                last_type = 'L'

    # Build per-bar structure arrays
    last_sh = np.full(n, np.nan)
    last_sl = np.full(n, np.nan)
    prev_sh = np.full(n, np.nan)
    prev_sl = np.full(n, np.nan)
    bars_since_sh = np.full(n, np.nan)
    bars_since_sl = np.full(n, np.nan)

    sh_prices = [(idx, high[idx]) for idx in swing_high_indices]
    sl_prices = [(idx, low[idx]) for idx in swing_low_indices]

    # Fill forward: at each bar t, find the most recent confirmed swing
    # Confirmed means the swing was at bar i, and we are at bar t >= i + k
    sh_ptr = 0
    sl_ptr = 0

    current_sh = np.nan
    current_sl = np.nan
    previous_sh = np.nan
    previous_sl = np.nan
    current_sh_bar = -1
    current_sl_bar = -1

    for t in range(n):
        # Check if new swing highs are confirmed by bar t
        while sh_ptr < len(sh_prices) and sh_prices[sh_ptr][0] + k <= t:
            previous_sh = current_sh
            current_sh = sh_prices[sh_ptr][1]
            current_sh_bar = sh_prices[sh_ptr][0]
            sh_ptr += 1

        while sl_ptr < len(sl_prices) and sl_prices[sl_ptr][0] + k <= t:
            previous_sl = current_sl
            current_sl = sl_prices[sl_ptr][1]
            current_sl_bar = sl_prices[sl_ptr][0]
            sl_ptr += 1

        last_sh[t] = current_sh
        last_sl[t] = current_sl
        prev_sh[t] = previous_sh
        prev_sl[t] = previous_sl
        bars_since_sh[t] = t - current_sh_bar if current_sh_bar >= 0 else np.nan
        bars_since_sl[t] = t - current_sl_bar if current_sl_bar >= 0 else np.nan

    return {
        'last_swing_high': last_sh,
        'last_swing_low': last_sl,
        'prev_swing_high': prev_sh,
        'prev_swing_low': prev_sl,
        'bars_since_swing_high': bars_since_sh,
        'bars_since_swing_low': bars_since_sl,
    }


def compute_structure_features(high: np.ndarray, low: np.ndarray,
                               close: np.ndarray, k: int = 5) -> dict:
    """Compute all market structure features from swing detection."""
    swings = detect_swings(high, low, k=k)
    n = len(high)

    last_sh = swings['last_swing_high']
    last_sl = swings['last_swing_low']
    prev_sh = swings['prev_swing_high']
    prev_sl = swings['prev_swing_low']

    # Leg size
    leg_size = last_sh - last_sl
    leg_size = np.where(leg_size <= 0, np.nan, leg_size)

    # Trend state: +1 uptrend (HH+HL), -1 downtrend (LH+LL), 0 chop
    is_hh = last_sh > prev_sh
    is_hl = last_sl > prev_sl
    is_lh = last_sh < prev_sh
    is_ll = last_sl < prev_sl
    trend_state = np.zeros(n, dtype=np.float64)
    trend_state[is_hh & is_hl] = 1.0
    trend_state[is_lh & is_ll] = -1.0

    # Retrace percent
    retrace_pct = np.full(n, np.nan)
    up_mask = trend_state == 1.0
    dn_mask = trend_state == -1.0
    valid_leg = leg_size > 0

    # Uptrend: retrace_pct = (last_swing_high - close) / leg_size
    mask = up_mask & valid_leg & ~np.isnan(last_sh)
    retrace_pct[mask] = (last_sh[mask] - close[mask]) / leg_size[mask]
    # Downtrend: retrace_pct = (close - last_swing_low) / leg_size
    mask = dn_mask & valid_leg & ~np.isnan(last_sl)
    retrace_pct[mask] = (close[mask] - last_sl[mask]) / leg_size[mask]

    retrace_pct = np.clip(retrace_pct, 0.0, 1.0)

    # Distance to swing high/low as percent of leg
    dist_to_sh_pct = np.full(n, np.nan)
    dist_to_sl_pct = np.full(n, np.nan)
    valid = valid_leg & ~np.isnan(last_sh) & ~np.isnan(last_sl)
    dist_to_sh_pct[valid] = np.clip(
        (last_sh[valid] - close[valid]) / leg_size[valid] * 100.0, 0, 100)
    dist_to_sl_pct[valid] = np.clip(
        (close[valid] - last_sl[valid]) / leg_size[valid] * 100.0, 0, 100)

    return {
        'leg_size': leg_size,
        'retrace_pct': retrace_pct,
        'bars_since_swing_high': swings['bars_since_swing_high'],
        'bars_since_swing_low': swings['bars_since_swing_low'],
        'dist_to_swing_high_pct': dist_to_sh_pct,
        'dist_to_swing_low_pct': dist_to_sl_pct,
        'trend_state': trend_state,
        'last_swing_high': last_sh,
        'last_swing_low': last_sl,
    }


# ============================================================
# Session Time Features
# ============================================================

def compute_mins_from_rth_open(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Compute minutes since RTH open (9:30 ET) for each bar."""
    try:
        ts_et = timestamps.tz_localize('America/Chicago').tz_convert('America/New_York')
    except TypeError:
        ts_et = timestamps.tz_convert('America/New_York')

    hours = ts_et.hour
    minutes_arr = ts_et.minute
    mins_since_open = (hours - 9) * 60 + (minutes_arr - 30)
    result = mins_since_open.values.astype(np.float64)
    # Pre-market bars get NaN
    result[result < 0] = np.nan
    result[result > 390] = np.nan
    return result


# ============================================================
# Main Feature Computation
# ============================================================

def compute_nn_features(bars: pd.DataFrame, profiles: list[dict],
                        n_workers: int = 0) -> pd.DataFrame:
    """
    Compute all 26 NN sequence channels from bars + volume profiles.
    Returns the bars DataFrame augmented with all feature columns.
    """
    n = len(bars)
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 8)

    high = bars['high'].values
    low = bars['low'].values
    close = bars['close'].values
    open_p = bars['open'].values
    volume = bars['volume'].values
    buy_volume = bars['buy_volume'].values
    sell_volume = bars['sell_volume'].values

    print("  Computing ATR...")
    atr_14 = compute_atr(high, low, close, period=14)
    atr_5 = compute_atr(high, low, close, period=5)
    eps = 1e-10

    # --- 1. OHLC-derived ---
    print("  Computing OHLC-derived features...")
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    log_return = np.log(close / (prev_close + eps))
    log_return[0] = 0.0

    bar_range = high - low
    bar_range_atr = bar_range / (atr_14 + eps)

    clv = (close - low) / (high - low + eps)
    clv = np.clip(clv, 0.0, 1.0)

    # --- 2. Volume ---
    # Already have volume from bars

    # --- 3. Order flow ---
    print("  Computing order flow features...")
    volume_delta = buy_volume - sell_volume
    volume_delta_pct = volume_delta / (volume + eps)
    volume_delta_pct = np.clip(volume_delta_pct, -1.0, 1.0)
    buy_volume_pct = buy_volume / (volume + eps)
    buy_volume_pct = np.clip(buy_volume_pct, 0.0, 1.0)

    # --- 4. CVD ---
    print("  Computing CVD...")
    cvd = compute_cvd(volume_delta, bars.index)
    cvd_slope_5 = np.full(n, np.nan)
    cvd_slope_5[5:] = (cvd[5:] - cvd[:-5]) / 5.0

    # --- 5. Per-bar Volume Profile ---
    print(f"  Computing per-bar volume profiles ({n_workers} workers)...")
    bar_profiles = compute_bar_volume_profiles(profiles, n_workers=n_workers)
    poc = bar_profiles[:, 0]
    vah = bar_profiles[:, 1]
    val = bar_profiles[:, 2]

    close_vs_poc = close - poc
    close_vs_vah = close - vah
    close_vs_val = close - val
    close_in_value_area = ((close >= val) & (close <= vah)).astype(np.float64)
    close_above_vah = (close > vah).astype(np.float64)
    close_below_val = (close < val).astype(np.float64)

    # Handle NaN from profiles
    nan_mask = np.isnan(poc)
    close_vs_poc[nan_mask] = 0.0
    close_vs_vah[nan_mask] = 0.0
    close_vs_val[nan_mask] = 0.0
    close_in_value_area[nan_mask] = np.nan
    close_above_vah[nan_mask] = np.nan
    close_below_val[nan_mask] = np.nan

    # --- 6. Rolling 60m Volume Profile ---
    print(f"  Computing vp60 rolling profiles ({n_workers} workers)...")
    vp60_vol_at_close_pct = compute_vp60(profiles, close, n_workers=n_workers)

    # --- 7. Market Structure ---
    print("  Computing market structure (swing detection)...")
    structure = compute_structure_features(high, low, close, k=5)

    # --- 8. Session Time ---
    print("  Computing session time...")
    mins_from_rth_open = compute_mins_from_rth_open(bars.index)

    # --- 9. Volatility ---
    print("  Computing volatility features...")
    atr_ratio = atr_5 / (atr_14 + eps)
    range_vs_atr = bar_range / (atr_14 + eps)

    log_returns_series = pd.Series(log_return)
    realized_vol_20 = log_returns_series.rolling(20, min_periods=1).std().values

    # ATR percentile over rolling 20-session window (~20 trading days * 390 bars)
    atr_series = pd.Series(atr_14)
    atr_percentile_20d = atr_series.rolling(
        390 * 20, min_periods=390
    ).apply(lambda x: (x.iloc[-1] <= x).mean(), raw=False).values

    # If not enough data for 20d percentile, use available history
    if np.all(np.isnan(atr_percentile_20d)):
        atr_percentile_20d = atr_series.expanding(min_periods=60).apply(
            lambda x: (x.iloc[-1] <= x).mean(), raw=False).values

    # --- Assemble all 26 channels ---
    print("  Assembling feature DataFrame...")
    bars = bars.copy()
    bars['log_return'] = log_return
    bars['bar_range_atr'] = bar_range_atr
    bars['clv'] = clv
    # volume already exists
    bars['volume_delta'] = volume_delta
    bars['volume_delta_pct'] = volume_delta_pct
    bars['buy_volume_pct'] = buy_volume_pct
    bars['cvd_slope_5'] = cvd_slope_5
    bars['close_vs_poc'] = close_vs_poc
    bars['close_vs_vah'] = close_vs_vah
    bars['close_vs_val'] = close_vs_val
    bars['close_in_value_area'] = close_in_value_area
    bars['close_above_vah'] = close_above_vah
    bars['close_below_val'] = close_below_val
    bars['vp60_vol_at_close_pct'] = vp60_vol_at_close_pct
    bars['leg_size'] = structure['leg_size']
    bars['retrace_pct'] = structure['retrace_pct']
    bars['bars_since_swing_high'] = structure['bars_since_swing_high']
    bars['bars_since_swing_low'] = structure['bars_since_swing_low']
    bars['dist_to_swing_high_pct'] = structure['dist_to_swing_high_pct']
    bars['dist_to_swing_low_pct'] = structure['dist_to_swing_low_pct']
    bars['mins_from_rth_open'] = mins_from_rth_open
    bars['atr_ratio'] = atr_ratio
    bars['range_vs_atr'] = range_vs_atr
    bars['realized_vol_20'] = realized_vol_20
    bars['atr_percentile_20d'] = atr_percentile_20d

    # Store intermediate columns needed by labels
    bars['atr_14'] = atr_14
    bars['trend_state'] = structure['trend_state']
    bars['last_swing_high'] = structure['last_swing_high']
    bars['last_swing_low'] = structure['last_swing_low']

    return bars


def get_sequence_channels(bars: pd.DataFrame) -> np.ndarray:
    """Extract the 26 sequence channels as a (n_bars, 26) numpy array."""
    return bars[CHANNEL_NAMES].values.astype(np.float32)


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='NN Feature Pipeline for ES Mini')
    parser.add_argument('data_dir', help='Directory containing .data files')
    parser.add_argument('--output', default=None, help='Output parquet file path')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers (0=auto)')
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: {args.data_dir} is not a directory")
        return 1

    bars, profiles = load_data(args.data_dir, workers=args.workers)
    bars = compute_nn_features(bars, profiles, n_workers=args.workers)

    output = args.output or os.path.join(args.data_dir, 'nn_features.parquet')
    cols_to_save = ['open', 'high', 'low', 'close'] + CHANNEL_NAMES + [
        'atr_14', 'trend_state', 'last_swing_high', 'last_swing_low',
        'buy_volume', 'sell_volume', 'num_trades',
    ]
    save_df = bars[[c for c in cols_to_save if c in bars.columns]]
    try:
        save_df.to_parquet(output)
    except ImportError:
        output = output.replace('.parquet', '.csv')
        save_df.to_csv(output)
    print(f"Saved features to: {output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
