#!/usr/bin/env python3
"""
Trade outcome labeling: simulates entries at each bar and checks if
TP (1.5R+) is hit before SL using subsequent OHLC data.

Provides dynamic SL sizing based on ATR and recent volatility.
"""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


TICK_SIZE = 0.25
POINT_VALUE = 50.0  # $50 per point for ES mini (4 ticks per point)
RTH_START_HOUR = 9
RTH_START_MIN = 30
RTH_END_HOUR = 15
RTH_END_MIN = 30
# NY timezone offset from UTC (Eastern): -4 (EDT) or -5 (EST)
# We'll handle this via tz-aware timestamps


def compute_atr(bars, period=14):
    """Average True Range over `period` bars."""
    high = bars['high']
    low = bars['low']
    prev_close = bars['close'].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def compute_dynamic_sl(bars, atr_period=14, atr_multiplier=1.5, min_sl=2.0, max_sl=8.0):
    """
    Dynamic stop-loss in points based on ATR.
    Clamped between min_sl and max_sl points.
    """
    atr = compute_atr(bars, period=atr_period)
    sl = (atr * atr_multiplier).clip(lower=min_sl, upper=max_sl)
    # Round to nearest tick
    sl = (sl / TICK_SIZE).round() * TICK_SIZE
    return sl


def is_rth(timestamps, tz='America/New_York'):
    """Return boolean mask for Regular Trading Hours (9:30-15:30 ET)."""
    if timestamps.tz is None:
        # Raw timestamps from CME data are in US Central (Chicago) time
        ts = timestamps.tz_localize('America/Chicago').tz_convert(tz)
    else:
        ts = timestamps.tz_convert(tz)
    hour_min = ts.hour * 100 + ts.minute
    return (hour_min >= 930) & (hour_min < 1530)


def simulate_trade_outcomes(bars, sl_points, rr_ratio=1.5, max_hold_bars=60, min_tp_ticks=20):
    """
    For each bar, simulate a LONG and SHORT entry at close price.
    Check subsequent bars to see if TP or SL is hit first.
    Uses multiprocessing to parallelize across chunks of bars.
    """
    n = len(bars)
    closes = bars['close'].values
    highs = bars['high'].values
    lows = bars['low'].values

    sl_arr = sl_points.values if hasattr(sl_points, 'values') else np.asarray(sl_points)
    min_tp_points = min_tp_ticks * TICK_SIZE
    tp_arr = np.maximum(sl_arr * rr_ratio, min_tp_points)

    n_workers = min(os.cpu_count() or 4, 8)
    chunk_size = max(1, n // n_workers)
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append((start, end, closes, highs, lows, sl_arr, tp_arr, max_hold_bars))

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_simulate_chunk, chunks))

    long_result = np.zeros(n, dtype=np.int8)
    short_result = np.zeros(n, dtype=np.int8)
    long_bars_held = np.full(n, max_hold_bars, dtype=np.int16)
    short_bars_held = np.full(n, max_hold_bars, dtype=np.int16)
    long_mae = np.zeros(n, dtype=np.float32)
    short_mae = np.zeros(n, dtype=np.float32)
    long_mfe = np.zeros(n, dtype=np.float32)
    short_mfe = np.zeros(n, dtype=np.float32)

    for (start, end, lr, sr, lbh, sbh, lmae, smae, lmfe, smfe) in results:
        long_result[start:end] = lr
        short_result[start:end] = sr
        long_bars_held[start:end] = lbh
        short_bars_held[start:end] = sbh
        long_mae[start:end] = lmae
        short_mae[start:end] = smae
        long_mfe[start:end] = lmfe
        short_mfe[start:end] = smfe

    return pd.DataFrame({
        'long_result': long_result,
        'short_result': short_result,
        'long_bars_held': long_bars_held,
        'short_bars_held': short_bars_held,
        'long_mae': long_mae,
        'short_mae': short_mae,
        'long_mfe': long_mfe,
        'short_mfe': short_mfe,
    }, index=bars.index)


def _simulate_chunk(args):
    """Process a chunk of bars for trade simulation (runs in worker process)."""
    start, end, closes, highs, lows, sl_arr, tp_arr, max_hold_bars = args
    n = len(closes)
    chunk_len = end - start

    lr = np.zeros(chunk_len, dtype=np.int8)
    sr = np.zeros(chunk_len, dtype=np.int8)
    lbh = np.full(chunk_len, max_hold_bars, dtype=np.int16)
    sbh = np.full(chunk_len, max_hold_bars, dtype=np.int16)
    lmae = np.zeros(chunk_len, dtype=np.float32)
    smae = np.zeros(chunk_len, dtype=np.float32)
    lmfe = np.zeros(chunk_len, dtype=np.float32)
    smfe = np.zeros(chunk_len, dtype=np.float32)

    for idx in range(chunk_len):
        i = start + idx
        if i >= n - 1:
            break
        entry = closes[i]
        sl = sl_arr[i]
        tp = tp_arr[i]

        # Long trade
        long_sl_price = entry - sl
        long_tp_price = entry + tp
        l_mae_val = 0.0
        l_mfe_val = 0.0
        l_result = 0
        l_bars = max_hold_bars

        for j in range(i + 1, min(i + 1 + max_hold_bars, n)):
            l_mfe_val = max(l_mfe_val, highs[j] - entry)
            l_mae_val = max(l_mae_val, entry - lows[j])
            if lows[j] <= long_sl_price:
                l_result = -1
                l_bars = j - i
                break
            if highs[j] >= long_tp_price:
                l_result = 1
                l_bars = j - i
                break

        lr[idx] = l_result
        lbh[idx] = l_bars
        lmae[idx] = l_mae_val
        lmfe[idx] = l_mfe_val

        # Short trade
        short_sl_price = entry + sl
        short_tp_price = entry - tp
        s_mae_val = 0.0
        s_mfe_val = 0.0
        s_result = 0
        s_bars = max_hold_bars

        for j in range(i + 1, min(i + 1 + max_hold_bars, n)):
            s_mfe_val = max(s_mfe_val, entry - lows[j])
            s_mae_val = max(s_mae_val, highs[j] - entry)
            if highs[j] >= short_sl_price:
                s_result = -1
                s_bars = j - i
                break
            if lows[j] <= short_tp_price:
                s_result = 1
                s_bars = j - i
                break

        sr[idx] = s_result
        sbh[idx] = s_bars
        smae[idx] = s_mae_val
        smfe[idx] = s_mfe_val

    return (start, end, lr, sr, lbh, sbh, lmae, smae, lmfe, smfe)


def create_trade_labels(bars, atr_period=14, atr_multiplier=1.5,
                        rr_ratio=1.5, max_hold_bars=60, min_tp_ticks=20):
    """
    Main entry point: compute dynamic SL and simulate trade outcomes.
    Returns the bars DataFrame augmented with trade labels.
    """
    sl_points = compute_dynamic_sl(bars, atr_period=atr_period,
                                   atr_multiplier=atr_multiplier)
    outcomes = simulate_trade_outcomes(bars, sl_points, rr_ratio=rr_ratio,
                                       max_hold_bars=max_hold_bars,
                                       min_tp_ticks=min_tp_ticks)

    bars = bars.copy()
    bars['sl_points'] = sl_points
    min_tp_points = min_tp_ticks * TICK_SIZE
    bars['tp_points'] = np.maximum(sl_points * rr_ratio, min_tp_points)

    # Primary label: best direction (which side wins?)
    # +1 if long wins, -1 if short wins, 0 if both timeout/lose
    bars['trade_label'] = 0
    bars.loc[outcomes['long_result'] == 1, 'trade_label'] = 1
    bars.loc[outcomes['short_result'] == 1, 'trade_label'] = -1
    # If both could win, prefer the one that wins faster
    both_win = (outcomes['long_result'] == 1) & (outcomes['short_result'] == 1)
    bars.loc[both_win & (outcomes['long_bars_held'] <= outcomes['short_bars_held']), 'trade_label'] = 1
    bars.loc[both_win & (outcomes['short_bars_held'] < outcomes['long_bars_held']), 'trade_label'] = -1

    bars['long_result'] = outcomes['long_result']
    bars['short_result'] = outcomes['short_result']
    bars['long_bars_held'] = outcomes['long_bars_held']
    bars['short_bars_held'] = outcomes['short_bars_held']
    bars['long_mae'] = outcomes['long_mae']
    bars['short_mae'] = outcomes['short_mae']
    bars['long_mfe'] = outcomes['long_mfe']
    bars['short_mfe'] = outcomes['short_mfe']

    # RTH filter
    bars['is_rth'] = is_rth(bars.index)

    return bars
