#!/usr/bin/env python3
"""
Trade outcome labeling: simulates entries at each bar and checks if
TP (1.5R+) is hit before SL using subsequent OHLC data.

Provides dynamic SL sizing based on ATR and recent volatility.
"""

import numpy as np
import pandas as pd


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

    Returns DataFrame with columns:
        long_result: 1=WIN, -1=LOSS, 0=TIMEOUT
        short_result: 1=WIN, -1=LOSS, 0=TIMEOUT
        long_bars_held: bars until exit
        short_bars_held: bars until exit
        long_mae: maximum adverse excursion (points)
        short_mae: maximum adverse excursion (points)
        long_mfe: maximum favorable excursion (points)
        short_mfe: maximum favorable excursion (points)
    """
    n = len(bars)
    closes = bars['close'].values
    highs = bars['high'].values
    lows = bars['low'].values

    long_result = np.zeros(n, dtype=np.int8)
    short_result = np.zeros(n, dtype=np.int8)
    long_bars_held = np.full(n, max_hold_bars, dtype=np.int16)
    short_bars_held = np.full(n, max_hold_bars, dtype=np.int16)
    long_mae = np.zeros(n, dtype=np.float32)
    short_mae = np.zeros(n, dtype=np.float32)
    long_mfe = np.zeros(n, dtype=np.float32)
    short_mfe = np.zeros(n, dtype=np.float32)

    sl_arr = sl_points.values if hasattr(sl_points, 'values') else sl_points
    min_tp_points = min_tp_ticks * TICK_SIZE
    tp_arr = np.maximum(sl_arr * rr_ratio, min_tp_points)

    for i in range(n - 1):
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

        long_result[i] = l_result
        long_bars_held[i] = l_bars
        long_mae[i] = l_mae_val
        long_mfe[i] = l_mfe_val

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

        short_result[i] = s_result
        short_bars_held[i] = s_bars
        short_mae[i] = s_mae_val
        short_mfe[i] = s_mfe_val

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
