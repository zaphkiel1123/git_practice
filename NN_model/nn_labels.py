#!/usr/bin/env python3
"""
Label computation for ES Mini NN model.

Three label groups computed from future bars [t+1 .. t+H]:
  Group A: MFE Opportunity (3 classes)
  Group B: Volatility Expansion/Contraction (3 classes)
  Group C: Directional Regime (4 classes)

All labels use H=60 bars (60 minutes) horizon.
Vectorized numpy implementation for performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os


H = 60  # Horizon: 60 bars


# ============================================================
# Group A — MFE Opportunity (Primary Head)
# ============================================================
# 0: strong_long_opp
# 1: strong_short_opp
# 2: no_edge

def compute_labels_a(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                     atr_14: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    """
    Vectorized Group A label computation.
    Returns array of shape (n,) with values 0, 1, or 2.
    Last H bars get label = -1 (invalid).
    """
    n = len(close)
    labels = np.full(n, -1, dtype=np.int64)

    for i in range(n - H):
        highs_future = high[i + 1:i + 1 + H]
        lows_future = low[i + 1:i + 1 + H]

        mfe_long = highs_future.max() - close[i]
        mfe_short = close[i] - lows_future.min()

        atr = atr_14[i]
        if atr <= 0:
            labels[i] = 2
            continue

        mfe_long_atr = mfe_long / atr
        mfe_short_atr = mfe_short / atr

        if mfe_long_atr >= threshold and mfe_long > 2 * mfe_short:
            labels[i] = 0  # strong_long_opp
        elif mfe_short_atr >= threshold and mfe_short > 2 * mfe_long:
            labels[i] = 1  # strong_short_opp
        else:
            labels[i] = 2  # no_edge

    return labels


def _compute_labels_a_chunk(args: tuple) -> tuple:
    """Process a chunk for Group A labels."""
    start, end, close, high, low, atr_14, threshold = args
    n_total = len(close)
    chunk_len = end - start
    labels = np.full(chunk_len, -1, dtype=np.int64)

    for local_i in range(chunk_len):
        i = start + local_i
        if i >= n_total - H:
            continue
        highs_future = high[i + 1:i + 1 + H]
        lows_future = low[i + 1:i + 1 + H]

        mfe_long = highs_future.max() - close[i]
        mfe_short = close[i] - lows_future.min()

        atr = atr_14[i]
        if atr <= 0:
            labels[local_i] = 2
            continue

        mfe_long_atr = mfe_long / atr
        mfe_short_atr = mfe_short / atr

        if mfe_long_atr >= threshold and mfe_long > 2 * mfe_short:
            labels[local_i] = 0
        elif mfe_short_atr >= threshold and mfe_short > 2 * mfe_long:
            labels[local_i] = 1
        else:
            labels[local_i] = 2

    return start, end, labels


# ============================================================
# Group B — Volatility Expansion / Contraction (Auxiliary Head 1)
# ============================================================
# 0: expansion
# 1: normal
# 2: contraction

def compute_labels_b(high: np.ndarray, low: np.ndarray,
                     atr_14: np.ndarray) -> np.ndarray:
    """
    Vectorized Group B label computation.
    Returns array of shape (n,) with values 0, 1, or 2.
    """
    n = len(high)
    labels = np.full(n, -1, dtype=np.int64)

    for i in range(n - H):
        highs_future = high[i + 1:i + 1 + H]
        lows_future = low[i + 1:i + 1 + H]
        future_range = highs_future.max() - lows_future.min()

        atr = atr_14[i]
        if atr <= 0:
            labels[i] = 1
            continue

        range_ratio = future_range / atr

        if range_ratio >= 1.5:
            labels[i] = 0  # expansion
        elif range_ratio >= 0.7:
            labels[i] = 1  # normal
        else:
            labels[i] = 2  # contraction

    return labels


def _compute_labels_b_chunk(args: tuple) -> tuple:
    """Process a chunk for Group B labels."""
    start, end, high, low, atr_14 = args
    n_total = len(high)
    chunk_len = end - start
    labels = np.full(chunk_len, -1, dtype=np.int64)

    for local_i in range(chunk_len):
        i = start + local_i
        if i >= n_total - H:
            continue
        highs_future = high[i + 1:i + 1 + H]
        lows_future = low[i + 1:i + 1 + H]
        future_range = highs_future.max() - lows_future.min()

        atr = atr_14[i]
        if atr <= 0:
            labels[local_i] = 1
            continue

        range_ratio = future_range / atr
        if range_ratio >= 1.5:
            labels[local_i] = 0
        elif range_ratio >= 0.7:
            labels[local_i] = 1
        else:
            labels[local_i] = 2

    return start, end, labels


# ============================================================
# Group C — Directional Regime (Auxiliary Head 2)
# ============================================================
# 0: continuation
# 1: retracement
# 2: reversal
# 3: chop

def compute_labels_c(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                     atr_14: np.ndarray, trend_state: np.ndarray,
                     leg_size: np.ndarray, last_swing_high: np.ndarray,
                     last_swing_low: np.ndarray) -> np.ndarray:
    """
    Vectorized Group C label computation.
    Returns array of shape (n,) with values 0, 1, 2, or 3.
    """
    n = len(close)
    labels = np.full(n, -1, dtype=np.int64)

    for i in range(n - H):
        ts = trend_state[i]
        if ts == 0 or np.isnan(ts):
            labels[i] = 3  # chop
            continue

        highs_future = high[i + 1:i + 1 + H]
        lows_future = low[i + 1:i + 1 + H]
        close_future_H = close[min(i + H, n - 1)]

        atr = atr_14[i]
        if atr <= 0:
            labels[i] = 3
            continue

        lsh = last_swing_high[i]
        lsl = last_swing_low[i]
        ls = leg_size[i]

        # Check reversal (BOS in opposite direction)
        if ts == 1.0 and not np.isnan(lsl) and lows_future.min() < lsl:
            labels[i] = 2  # reversal
            continue
        if ts == -1.0 and not np.isnan(lsh) and highs_future.max() > lsh:
            labels[i] = 2  # reversal
            continue

        # Check continuation
        if ts == 1.0:
            move = highs_future.max() - close[i]
            retrace = close[i] - lows_future.min()
        else:
            move = close[i] - lows_future.min()
            retrace = highs_future.max() - close[i]

        if move >= 1.0 * atr and (move == 0 or retrace / move <= 0.5):
            labels[i] = 0  # continuation
            continue

        # Check retracement
        if not np.isnan(ls) and ls > 0:
            if ts == 1.0:
                pullback_depth = (close[i] - lows_future.min()) / ls
                resumed = close_future_H > close[i]
            else:
                pullback_depth = (highs_future.max() - close[i]) / ls
                resumed = close_future_H < close[i]

            if 0.38 <= pullback_depth <= 0.62 and resumed:
                labels[i] = 1  # retracement
                continue

        # Default: chop
        future_range = highs_future.max() - lows_future.min()
        if future_range < 0.5 * atr:
            labels[i] = 3
        else:
            labels[i] = 3

    return labels


def _compute_labels_c_chunk(args: tuple) -> tuple:
    """Process a chunk for Group C labels."""
    start, end, close, high, low, atr_14, trend_state, leg_size, last_swing_high, last_swing_low = args
    n_total = len(close)
    chunk_len = end - start
    labels = np.full(chunk_len, -1, dtype=np.int64)

    for local_i in range(chunk_len):
        i = start + local_i
        if i >= n_total - H:
            continue

        ts = trend_state[i]
        if ts == 0 or np.isnan(ts):
            labels[local_i] = 3
            continue

        highs_future = high[i + 1:i + 1 + H]
        lows_future = low[i + 1:i + 1 + H]
        close_future_H = close[min(i + H, n_total - 1)]

        atr = atr_14[i]
        if atr <= 0:
            labels[local_i] = 3
            continue

        lsh = last_swing_high[i]
        lsl = last_swing_low[i]
        ls = leg_size[i]

        if ts == 1.0 and not np.isnan(lsl) and lows_future.min() < lsl:
            labels[local_i] = 2
            continue
        if ts == -1.0 and not np.isnan(lsh) and highs_future.max() > lsh:
            labels[local_i] = 2
            continue

        if ts == 1.0:
            move = highs_future.max() - close[i]
            retrace = close[i] - lows_future.min()
        else:
            move = close[i] - lows_future.min()
            retrace = highs_future.max() - close[i]

        if move >= 1.0 * atr and (move == 0 or retrace / move <= 0.5):
            labels[local_i] = 0
            continue

        if not np.isnan(ls) and ls > 0:
            if ts == 1.0:
                pullback_depth = (close[i] - lows_future.min()) / ls
                resumed = close_future_H > close[i]
            else:
                pullback_depth = (highs_future.max() - close[i]) / ls
                resumed = close_future_H < close[i]

            if 0.38 <= pullback_depth <= 0.62 and resumed:
                labels[local_i] = 1
                continue

        labels[local_i] = 3

    return start, end, labels


# ============================================================
# Parallel Label Computation
# ============================================================

def compute_all_labels(bars: pd.DataFrame, n_workers: int = 0,
                       threshold_a: float = 1.5) -> pd.DataFrame:
    """
    Compute all 3 label groups and add them to bars DataFrame.
    Uses multiprocessing for parallel computation across chunks.

    Requires bars to have columns: close, high, low, atr_14,
    trend_state, leg_size, last_swing_high, last_swing_low
    (all produced by nn_feature_pipeline.compute_nn_features).
    """
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 8)

    close = bars['close'].values
    high = bars['high'].values
    low = bars['low'].values
    atr_14 = bars['atr_14'].values
    trend_state = bars['trend_state'].values
    leg_size = bars['leg_size'].values
    last_swing_high = bars['last_swing_high'].values
    last_swing_low = bars['last_swing_low'].values

    n = len(bars)
    chunk_size = max(1, n // n_workers)

    # --- Group A ---
    print("  Computing Group A labels (MFE opportunity)...")
    chunks_a = []
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunks_a.append((i, end, close, high, low, atr_14, threshold_a))

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results_a = list(pool.map(_compute_labels_a_chunk, chunks_a))
    else:
        results_a = [_compute_labels_a_chunk(c) for c in chunks_a]

    label_a = np.full(n, -1, dtype=np.int64)
    for start, end, chunk_labels in results_a:
        label_a[start:end] = chunk_labels

    # --- Group B ---
    print("  Computing Group B labels (vol regime)...")
    chunks_b = []
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunks_b.append((i, end, high, low, atr_14))

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results_b = list(pool.map(_compute_labels_b_chunk, chunks_b))
    else:
        results_b = [_compute_labels_b_chunk(c) for c in chunks_b]

    label_b = np.full(n, -1, dtype=np.int64)
    for start, end, chunk_labels in results_b:
        label_b[start:end] = chunk_labels

    # --- Group C ---
    print("  Computing Group C labels (directional regime)...")
    chunks_c = []
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunks_c.append((i, end, close, high, low, atr_14,
                         trend_state, leg_size, last_swing_high, last_swing_low))

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results_c = list(pool.map(_compute_labels_c_chunk, chunks_c))
    else:
        results_c = [_compute_labels_c_chunk(c) for c in chunks_c]

    label_c = np.full(n, -1, dtype=np.int64)
    for start, end, chunk_labels in results_c:
        label_c[start:end] = chunk_labels

    # Add to DataFrame
    bars = bars.copy()
    bars['label_a'] = label_a
    bars['label_b'] = label_b
    bars['label_c'] = label_c

    # Print distribution
    for name, lbl, n_cls in [('A (MFE)', label_a, 3), ('B (Vol)', label_b, 3), ('C (Dir)', label_c, 4)]:
        valid = lbl[lbl >= 0]
        print(f"    Group {name}: {len(valid)} valid samples")
        for c in range(n_cls):
            cnt = (valid == c).sum()
            pct = cnt / len(valid) * 100 if len(valid) > 0 else 0
            print(f"      class {c}: {cnt:,} ({pct:.1f}%)")

    return bars


# ============================================================
# Exclusion Rules
# ============================================================

def get_valid_mask(bars: pd.DataFrame) -> np.ndarray:
    """
    Return boolean mask for valid decision points.
    Excludes:
      - First 60 bars (warmup for vp60 and rolling features)
      - Last H bars (no future data for labels)
      - Bars with ATR_14 <= 0
      - Bars with leg_size <= 0 or NaN
      - Bars with invalid labels (-1)
    """
    n = len(bars)
    mask = np.ones(n, dtype=bool)

    # Warmup
    mask[:60] = False

    # Last H bars
    mask[n - H:] = False

    # Degenerate ATR
    mask[bars['atr_14'].values <= 0] = False

    # Invalid leg_size
    leg = bars['leg_size'].values
    mask[np.isnan(leg) | (leg <= 0)] = False

    # Invalid labels
    mask[bars['label_a'].values < 0] = False
    mask[bars['label_b'].values < 0] = False
    mask[bars['label_c'].values < 0] = False

    return mask
