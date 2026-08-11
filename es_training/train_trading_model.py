#!/usr/bin/env python3
"""
Multi-model training for ES mini trading system.

Models:
1. Entry Signal Model — predicts trade direction (long/short/no-trade)
2. Volatility Model — predicts next-bar ATR for dynamic SL sizing
3. Trade Quality Model — filters signals by predicted win probability

All use walk-forward validation with no look-ahead bias.

Usage:
    python3 train_trading_model.py /path/to/data/ --window 1min --rr 1.5
    python3 train_trading_model.py /path/to/data/ --window 1min --rr 2.0 --folds 7
"""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import sys
import json
import time
import warnings
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import pandas as pd
import joblib

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, r2_score, classification_report
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from feature_pipeline import decode_file_to_dataframe, compute_window_features, add_rolling_features, add_time_features, add_value_area_features, load_files_to_bars
from labels import create_trade_labels, compute_atr, is_rth

warnings.filterwarnings('ignore', category=UserWarning)

TICK_SIZE = 0.25


# ============================================================
# Enhanced Feature Engineering
# ============================================================

def add_microstructure_features(bars, raw_df=None):
    """Add order-flow microstructure features. No MAs or close-price features."""

    # Delta (net buy-sell volume per bar)
    bars['delta'] = bars['buy_volume'] - bars['sell_volume']

    # Delta divergence: bar direction (high+low midpoint vs open) vs delta direction
    bars['mid'] = (bars['high'] + bars['low']) / 2
    bars['bar_direction'] = np.sign(bars['mid'] - bars['open'])
    bars['delta_direction'] = np.sign(bars['delta'])
    bars['delta_divergence'] = (bars['bar_direction'] != bars['delta_direction']).astype(int)

    # CVD (cumulative volume delta) — resets at session boundary (17:00 CT / 18:00 ET)
    bars['cvd'] = _compute_session_cvd(bars['delta'], bars.index)
    bars['cvd_slope_3'] = bars['cvd'].diff(3) / 3
    bars['cvd_slope_5'] = bars['cvd'].diff(5) / 5
    bars['cvd_slope_10'] = bars['cvd'].diff(10) / 10
    bars['cvd_accel'] = bars['cvd_slope_3'].diff()
    bars['cvd_accel_5'] = bars['cvd_slope_5'].diff()

    # CVD divergence from price: price making new highs but CVD not, or vice versa
    bars['high_5'] = bars['high'].rolling(5).max()
    bars['cvd_5_max'] = bars['cvd'].rolling(5).max()
    bars['cvd_bull_div'] = ((bars['high'] >= bars['high_5']) &
                            (bars['cvd'] < bars['cvd_5_max'])).astype(int)
    bars['low_5'] = bars['low'].rolling(5).min()
    bars['cvd_5_min'] = bars['cvd'].rolling(5).min()
    bars['cvd_bear_div'] = ((bars['low'] <= bars['low_5']) &
                            (bars['cvd'] > bars['cvd_5_min'])).astype(int)

    # Delta as percent of total volume (normalized)
    bars['delta_pct'] = bars['delta'] / bars['volume'].clip(lower=1)

    # Cumulative delta rate of change
    bars['cvd_roc_3'] = bars['cvd'].diff(3) / bars['cvd'].shift(3).abs().clip(lower=1)
    bars['cvd_roc_10'] = bars['cvd'].diff(10) / bars['cvd'].shift(10).abs().clip(lower=1)

    # Absorption: high volume but small price movement
    bars['bar_range'] = bars['high'] - bars['low']
    bars['vol_per_tick'] = bars['volume'] / (bars['bar_range'] / TICK_SIZE).clip(lower=1)
    bars['absorption'] = bars['vol_per_tick'] / bars['vol_per_tick'].rolling(20, min_periods=1).mean()

    # Large trade concentration
    if 'max_single_txn' in bars.columns:
        bars['large_trade_ratio'] = bars['max_single_txn'] / bars['volume'].clip(lower=1)

    # Volatility features
    bars['atr_5'] = compute_atr(bars, period=5)
    bars['atr_14'] = compute_atr(bars, period=14)
    bars['atr_ratio'] = bars['atr_5'] / bars['atr_14'].clip(lower=TICK_SIZE)

    # Range relative to ATR (expansion/contraction)
    bars['range_vs_atr'] = bars['bar_range'] / bars['atr_14'].clip(lower=TICK_SIZE)

    # Consolidation detection (narrowing range)
    bars['range_ma_5'] = bars['bar_range'].rolling(5).mean()
    bars['range_ma_20'] = bars['bar_range'].rolling(20).mean()
    bars['consolidation'] = bars['range_ma_5'] / bars['range_ma_20'].clip(lower=TICK_SIZE)

    # Order flow acceleration
    bars['flow_accel'] = bars['flow_imbalance'].diff()
    bars['flow_accel_3'] = bars['flow_imbalance'].diff(3)

    # Volume surge detection
    bars['vol_ma_20'] = bars['volume'].rolling(20).mean()
    bars['vol_surge'] = bars['volume'] / bars['vol_ma_20'].clip(lower=1)

    # Volume rate of change over 5 bars (trend direction)
    bars['vol_roc_5'] = (
        (bars['volume'] - bars['volume'].shift(5)) /
        bars['volume'].shift(5).clip(lower=1)
    )
    bars['vol_trend_5'] = np.where(
        bars['vol_roc_5'] > 0.10, 1,
        np.where(bars['vol_roc_5'] < -0.10, -1, 0)
    )

    # Transaction speed: intensity acceleration
    bars['intensity_accel'] = bars['intensity'].diff()
    bars['intensity_accel_3'] = bars['intensity'].diff(3)
    bars['intensity_surge'] = bars['intensity'] / bars['intensity'].rolling(10, min_periods=1).mean()

    # Delta momentum: consecutive bars with same delta sign
    bars['delta_sign'] = np.sign(bars['delta'])
    bars['delta_streak'] = _compute_streak(bars['delta_sign'].values)

    # Buy/sell pressure ratio over rolling windows
    bars['pressure_ratio_5'] = (bars['buy_volume'].rolling(5).sum() /
                                bars['sell_volume'].rolling(5).sum().clip(lower=1))
    bars['pressure_ratio_10'] = (bars['buy_volume'].rolling(10).sum() /
                                 bars['sell_volume'].rolling(10).sum().clip(lower=1))

    # Imbalance cluster rolling features
    if 'buy_imbalance_cluster' in bars.columns:
        bars['imbalance_cluster_net'] = bars['buy_imbalance_cluster'] - bars['sell_imbalance_cluster']
        bars['imbalance_cluster_3'] = bars['imbalance_cluster_net'].rolling(3).sum()
        bars['imbalance_cluster_5'] = bars['imbalance_cluster_net'].rolling(5).sum()
        bars['imbalance_strength'] = (bars['buy_imbalance_count'] - bars['sell_imbalance_count']).rolling(5).sum()

    # Gap from previous bar (open vs previous high/low, not close)
    bars['gap_from_prev_high'] = bars['open'] - bars['high'].shift(1)
    bars['gap_from_prev_low'] = bars['open'] - bars['low'].shift(1)

    # CVD-vs-range divergence: CVD moving but price stuck (setup detection)
    bars['cvd_range_div_5'] = bars['cvd_slope_5'].abs() / bars['bar_range'].rolling(5).mean().clip(lower=TICK_SIZE)
    bars['cvd_range_div_10'] = bars['cvd_slope_10'].abs() / bars['bar_range'].rolling(10).mean().clip(lower=TICK_SIZE)

    # Direction context: is CVD pushing with or against recent flow?
    # Positive = CVD and flow agree (strength signal), Negative = disagree (absorption signal)
    bars['cvd_flow_alignment'] = np.sign(bars['cvd_slope_5']) * bars['flow_imbalance']

    # Absorption vs strength classifier inputs:
    # When cvd_range_div is high, these help determine outcome:
    # - If opposing side's imbalance clusters are forming → absorption (reversal)
    # - If same-side pressure ratio stays dominant → strength (continuation)
    bars['cvd_div_x_pressure'] = bars['cvd_range_div_5'] * (bars['pressure_ratio_5'] - 1.0)
    # Positive = CVD diverging + buy pressure dominant → bullish strength
    # Negative = CVD diverging + sell pressure dominant → bearish strength

    bars['cvd_div_x_absorption'] = bars['cvd_range_div_5'] * bars['absorption']
    # High = CVD diverging + high absorption → trapped traders, likely reversal

    bars['cvd_div_x_imbalance'] = bars['cvd_range_div_5'] * bars.get('net_imbalance', 0)
    # CVD diverging + footprint imbalance stacking → confirms direction (strength)

    return bars


def _compute_streak(signs):
    """Compute streak length of consecutive same-sign values."""
    streak = np.zeros(len(signs), dtype=np.int32)
    for i in range(1, len(signs)):
        if signs[i] == signs[i-1] and signs[i] != 0:
            streak[i] = streak[i-1] + 1
        elif signs[i] != 0:
            streak[i] = 1
    return streak


def _compute_session_cvd(delta_series, timestamps):
    """Compute CVD that resets at each session boundary (17:00 CT = 18:00 ET)."""
    SESSION_RESET_HOUR_CT = 17
    deltas = delta_series.values
    hours = timestamps.hour
    cvd = np.zeros(len(deltas), dtype=np.float64)
    running = 0.0
    prev_hour = -1
    for i in range(len(deltas)):
        cur_hour = hours[i]
        # Reset when crossing 17:00 CT
        if prev_hour < SESSION_RESET_HOUR_CT and cur_hour >= SESSION_RESET_HOUR_CT:
            running = 0.0
        elif cur_hour < prev_hour and prev_hour >= SESSION_RESET_HOUR_CT:
            running = 0.0
        running += deltas[i]
        cvd[i] = running
        prev_hour = cur_hour
    return pd.Series(cvd, index=delta_series.index)


def add_multi_timeframe_features(bars):
    """Add features from higher timeframe aggregation (5-bar, 15-bar). Order-flow focused."""
    for tf in [5, 15]:
        prefix = f'tf{tf}'
        bars[f'{prefix}_range'] = bars['high'].rolling(tf).max() - bars['low'].rolling(tf).min()
        bars[f'{prefix}_delta_sum'] = bars['delta'].rolling(tf).sum()
        bars[f'{prefix}_vol_sum'] = bars['volume'].rolling(tf).sum()
        bars[f'{prefix}_imbalance'] = bars['flow_imbalance'].rolling(tf).mean()
        bars[f'{prefix}_absorption'] = bars['absorption'].rolling(tf).mean()
        bars[f'{prefix}_intensity'] = bars['intensity'].rolling(tf).mean()
        if 'buy_imbalance_count' in bars.columns:
            bars[f'{prefix}_imb_cluster_sum'] = bars['imbalance_cluster_net'].rolling(tf).sum()
    return bars


# ============================================================
# Walk-Forward Validation
# ============================================================

def walk_forward_split(n, n_folds=5, min_train_ratio=0.5):
    """Expanding window walk-forward splits."""
    test_size = int(n * (1 - min_train_ratio) / n_folds)
    splits = []
    for i in range(n_folds):
        test_start = int(n * min_train_ratio) + i * test_size
        test_end = min(test_start + test_size, n)
        if test_end > n:
            break
        train_idx = list(range(0, test_start))
        test_idx = list(range(test_start, test_end))
        if len(train_idx) > 100 and len(test_idx) > 20:
            splits.append((train_idx, test_idx))
    return splits


# ============================================================
# Model Training Functions
# ============================================================

def get_feature_columns(df):
    """Feature columns: everything except labels, metadata, raw OHLC, and pruned groups."""
    exclude = {
        # Labels and metadata
        'trade_label', 'long_result', 'short_result',
        'long_bars_held', 'short_bars_held',
        'long_mae', 'short_mae', 'long_mfe', 'short_mfe',
        'sl_points', 'tp_points', 'is_rth',
        'open', 'high', 'low', 'close', 'mid', 'vwap',
        'direction_label', 'magnitude_label', 'future_return', 'future_close',
        'atr_target', 'quality_label', 'has_trade_outcome',
        'high_5', 'low_5', 'cvd_5_max', 'cvd_5_min',
        # Duplicate of delta_pct
        'flow_imbalance',
        # Pruned group 5: Rolling bar features
        'return_1',
        'roll_3_volatility', 'roll_3_flow_imb', 'roll_3_intensity', 'roll_3_vol_ratio',
        'roll_5_volatility', 'roll_5_flow_imb', 'roll_5_intensity', 'roll_5_vol_ratio',
        'roll_10_volatility', 'roll_10_flow_imb', 'roll_10_intensity', 'roll_10_vol_ratio',
        # Pruned group 6: Time & session
        'hour_sin', 'hour_cos',
        'session_asia', 'session_europe', 'session_us',
        'session_overnight', 'session_premarket', 'session_rth',
        # Pruned group 8: CVD and CVD-derived interaction
        'cvd', 'cvd_slope_3', 'cvd_slope_5', 'cvd_slope_10',
        'cvd_accel', 'cvd_accel_5', 'cvd_roc_3', 'cvd_roc_10',
        'cvd_bull_div', 'cvd_bear_div',
        'cvd_range_div_5', 'cvd_range_div_10', 'cvd_flow_alignment',
        'cvd_div_x_pressure', 'cvd_div_x_absorption', 'cvd_div_x_imbalance',
        # Pruned group 14: Gap
        'gap_from_prev_high', 'gap_from_prev_low',
        # Pruned group 15: Multi-timeframe tf5
        'tf5_range', 'tf5_delta_sum', 'tf5_vol_sum', 'tf5_imbalance',
        'tf5_absorption', 'tf5_intensity', 'tf5_imb_cluster_sum',
        # Pruned group 16: Multi-timeframe tf15
        'tf15_range', 'tf15_delta_sum', 'tf15_vol_sum', 'tf15_imbalance',
        'tf15_absorption', 'tf15_intensity', 'tf15_imb_cluster_sum',
        # Raw VA levels (not normalized)
        'va10_poc', 'va10_vah', 'va10_val',
    }
    return [c for c in df.columns if c not in exclude and not c.startswith('_')]


def train_entry_signal_model(X_train, y_train, X_test, y_test, feature_names):
    """Train entry signal classifier: +1 (long), -1 (short), 0 (no trade)."""
    # Convert to binary per-side: we train a 3-class model
    if HAS_LIGHTGBM:
        # Map labels: -1→0, 0→1, +1→2
        y_tr_mapped = y_train + 1
        y_te_mapped = y_test + 1

        model = lgb.LGBMClassifier(
            objective='multiclass', num_class=3,
            n_estimators=500, learning_rate=0.03,
            num_leaves=63, max_depth=8,
            feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=5,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
            verbosity=-1, n_jobs=-1,
        )
        model.fit(X_train, y_tr_mapped,
                  eval_X=X_test, eval_y=y_te_mapped,
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        y_pred_mapped = model.predict(X_test)
        y_pred = y_pred_mapped - 1
    else:
        y_tr_mapped = y_train + 1
        y_te_mapped = y_test + 1
        model = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.7, random_state=42
        )
        model.fit(X_train, y_tr_mapped)
        y_pred_mapped = model.predict(X_test)
        y_pred = y_pred_mapped - 1

    # Only evaluate on non-zero labels (actual trade opportunities)
    trade_mask = y_test != 0
    if trade_mask.sum() > 0:
        acc = accuracy_score(y_test[trade_mask], y_pred[trade_mask])
        prec = precision_score(y_test[trade_mask], y_pred[trade_mask], average='weighted', zero_division=0)
    else:
        acc = 0.0
        prec = 0.0

    metrics = {
        'accuracy_trades': float(acc),
        'precision_trades': float(prec),
        'accuracy_all': float(accuracy_score(y_test, y_pred)),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_trades_test': int(trade_mask.sum()),
    }
    return model, metrics


def train_volatility_model(X_train, y_train, X_test, y_test, feature_names):
    """Train ATR/range predictor for SL sizing."""
    mask_tr = y_train > 0
    mask_te = y_test > 0
    X_tr, y_tr = X_train[mask_tr], y_train[mask_tr]
    X_te, y_te = X_test[mask_te], y_test[mask_te]

    if len(X_tr) < 50 or len(X_te) < 10:
        return None, None

    if HAS_LIGHTGBM:
        model = lgb.LGBMRegressor(
            objective='regression', metric='mae',
            n_estimators=300, learning_rate=0.05,
            num_leaves=31, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=5,
            verbosity=-1, n_jobs=-1,
        )
        model.fit(X_tr, y_tr,
                  eval_X=X_te, eval_y=y_te,
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    else:
        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42
        )
        model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    metrics = {
        'mae': float(mean_absolute_error(y_te, y_pred)),
        'r2': float(r2_score(y_te, y_pred)),
        'mean_actual': float(y_te.mean()),
        'mean_predicted': float(y_pred.mean()),
    }
    return model, metrics


def train_quality_model(X_train, y_train, X_test, y_test, feature_names):
    """Train trade quality filter: predicts P(win) given a signal."""
    if len(X_train) < 50 or len(X_test) < 10:
        return None, None

    if HAS_LIGHTGBM:
        model = lgb.LGBMClassifier(
            objective='binary', n_estimators=400,
            learning_rate=0.03, num_leaves=31, max_depth=6,
            feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=5,
            min_child_samples=30, reg_alpha=0.2, reg_lambda=0.2,
            verbosity=-1, n_jobs=-1,
        )
        model.fit(X_train, y_train,
                  eval_X=X_test, eval_y=y_test,
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.7, random_state=42
        )
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'win_rate_actual': float(y_test.mean()),
        'win_rate_predicted': float(y_prob.mean()),
    }
    return model, metrics


# ============================================================
# Full Training Pipeline
# ============================================================

def _load_files_to_bars(data_dir, window='1min', workers=0):
    """Load .data files one at a time and aggregate to bars (memory-efficient)."""
    return load_files_to_bars(data_dir, window=window, workers=workers)


def _engineer_bar_features(bars):
    """Add derived features on bar-level data."""
    t0 = time.time()
    print("Adding rolling features...")
    bars = add_rolling_features(bars)
    print("Adding time features...")
    bars = add_time_features(bars)
    print("Adding microstructure features...")
    bars = add_microstructure_features(bars)
    print("Adding multi-timeframe features...")
    bars = add_multi_timeframe_features(bars)
    print("Adding value area features...")
    bars = add_value_area_features(bars)
    print(f"  Feature engineering [{time.time()-t0:.1f}s]")
    return bars


def prepare_features(data_dir, window='1min', rr_ratio=1.5, max_hold_bars=60,
                     include_labels=True):
    """Run full feature (+ optional label) pipeline on all .data files in directory."""
    pipeline_start = time.time()

    bars = _load_files_to_bars(data_dir, window=window)
    bars = _engineer_bar_features(bars)

    if include_labels:
        t0 = time.time()
        print(f"Simulating trade outcomes (RR={rr_ratio}, max_hold={max_hold_bars} bars)...")
        bars = create_trade_labels(bars, rr_ratio=rr_ratio, max_hold_bars=max_hold_bars)
        print(f"  Trade simulation [{time.time()-t0:.1f}s]")

        bars['atr_target'] = compute_atr(bars, period=5).shift(-5)
        bars['has_trade_outcome'] = ((bars['long_result'] != 0) | (bars['short_result'] != 0)).astype(int)
        bars['quality_label'] = ((bars['long_result'] == 1) | (bars['short_result'] == 1)).astype(int)
    else:
        bars['is_rth'] = is_rth(bars.index)

    bars = bars.drop(columns=['_vol_profile'], errors='ignore')
    bars = bars.dropna()
    print(f"Complete bars after dropna: {len(bars)}")
    print(f"  Total pipeline [{time.time()-pipeline_start:.1f}s]")

    return bars


def run_training(data_dir, window='1min', rr_ratio=1.5, max_hold_bars=60,
                 n_folds=5, output_dir=None):
    """Full training pipeline with walk-forward validation."""

    bars = prepare_features(data_dir, window=window, rr_ratio=rr_ratio,
                            max_hold_bars=max_hold_bars)

    if output_dir is None:
        output_dir = os.path.join(_SCRIPT_DIR, 'data', 'models')
    os.makedirs(output_dir, exist_ok=True)

    feature_cols = get_feature_columns(bars)
    print(f"\nFeature columns: {len(feature_cols)}")

    # Filter to RTH only for training signal/quality models
    rth_mask = bars['is_rth'].values.astype(bool)
    print(f"RTH bars: {rth_mask.sum()} / {len(bars)} ({100*rth_mask.sum()/len(bars):.1f}%)")

    X_all = bars[feature_cols].values
    y_signal = bars['trade_label'].values
    y_atr = bars['atr_target'].values
    y_quality = bars['quality_label'].values

    # Walk-forward splits (on RTH bars only for signal/quality)
    rth_indices = np.where(rth_mask)[0]
    splits = walk_forward_split(len(rth_indices), n_folds=n_folds)

    # ---- Entry Signal Model ----
    print(f"\n{'='*60}")
    print(f"  ENTRY SIGNAL MODEL (direction prediction)")
    print(f"{'='*60}")

    t0 = time.time()
    signal_metrics = []
    best_signal_model = None
    best_signal_acc = 0

    for fold_idx, (tr_idx, te_idx) in enumerate(splits):
        actual_tr = rth_indices[tr_idx]
        actual_te = rth_indices[te_idx]

        model, metrics = train_entry_signal_model(
            X_all[actual_tr], y_signal[actual_tr],
            X_all[actual_te], y_signal[actual_te],
            feature_cols
        )
        signal_metrics.append(metrics)
        print(f"  Fold {fold_idx+1}: acc_trades={metrics['accuracy_trades']:.4f}, "
              f"acc_all={metrics['accuracy_all']:.4f} "
              f"(train={metrics['n_train']}, test={metrics['n_test']})")

        if metrics['accuracy_trades'] > best_signal_acc:
            best_signal_acc = metrics['accuracy_trades']
            best_signal_model = model
    print(f"  Signal model training [{time.time()-t0:.1f}s]")

    # ---- Volatility Model (uses all bars, not just RTH) ----
    print(f"\n{'='*60}")
    print(f"  VOLATILITY MODEL (ATR prediction for SL sizing)")
    print(f"{'='*60}")

    t0 = time.time()
    vol_splits = walk_forward_split(len(bars), n_folds=n_folds)
    vol_metrics = []
    best_vol_model = None
    best_vol_mae = float('inf')

    for fold_idx, (tr_idx, te_idx) in enumerate(vol_splits):
        model, metrics = train_volatility_model(
            X_all[tr_idx], y_atr[tr_idx],
            X_all[te_idx], y_atr[te_idx],
            feature_cols
        )
        if metrics is None:
            continue
        vol_metrics.append(metrics)
        print(f"  Fold {fold_idx+1}: MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")

        if metrics['mae'] < best_vol_mae:
            best_vol_mae = metrics['mae']
            best_vol_model = model
    print(f"  Volatility model training [{time.time()-t0:.1f}s]")

    # ---- Trade Quality Model (RTH only, only on trade signals) ----
    print(f"\n{'='*60}")
    print(f"  TRADE QUALITY MODEL (win probability filter)")
    print(f"{'='*60}")

    t0 = time.time()
    # Train on all bars with a definitive trade outcome (win or loss, not just winners)
    y_has_outcome = bars['has_trade_outcome'].values
    trade_mask_rth = (y_has_outcome[rth_indices] != 0)
    trade_rth_indices = rth_indices[trade_mask_rth]

    if len(trade_rth_indices) > 200:
        quality_splits = walk_forward_split(len(trade_rth_indices), n_folds=n_folds)
        quality_metrics = []
        best_quality_model = None
        best_quality_f1 = 0

        for fold_idx, (tr_idx, te_idx) in enumerate(quality_splits):
            actual_tr = trade_rth_indices[tr_idx]
            actual_te = trade_rth_indices[te_idx]

            model, metrics = train_quality_model(
                X_all[actual_tr], y_quality[actual_tr],
                X_all[actual_te], y_quality[actual_te],
                feature_cols
            )
            if metrics is None:
                continue
            quality_metrics.append(metrics)
            print(f"  Fold {fold_idx+1}: acc={metrics['accuracy']:.4f}, "
                  f"prec={metrics['precision']:.4f}, f1={metrics['f1']:.4f}, "
                  f"win_rate={metrics['win_rate_actual']:.4f}")

            if metrics['f1'] > best_quality_f1:
                best_quality_f1 = metrics['f1']
                best_quality_model = model
    else:
        print("  Insufficient trade samples for quality model")
        quality_metrics = []
        best_quality_model = None
    print(f"  Quality model training [{time.time()-t0:.1f}s]")

    # ---- Feature Importance ----
    print(f"\n{'='*60}")
    print(f"  TOP FEATURES")
    print(f"{'='*60}")

    if best_signal_model:
        imp = best_signal_model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:20]
        print("\n  Entry Signal Model (top 20):")
        for rank, idx in enumerate(top_idx, 1):
            print(f"    {rank:2d}. {feature_cols[idx]:<30} {imp[idx]:.0f}")

    if best_quality_model:
        imp = best_quality_model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:15]
        print("\n  Trade Quality Model (top 15):")
        for rank, idx in enumerate(top_idx, 1):
            print(f"    {rank:2d}. {feature_cols[idx]:<30} {imp[idx]:.0f}")

    # ---- Save Models ----
    print(f"\n{'='*60}")
    print(f"  SAVING MODELS")
    print(f"{'='*60}")

    models_saved = {}

    if best_signal_model:
        path = os.path.join(output_dir, 'signal_model.joblib')
        joblib.dump(best_signal_model, path)
        models_saved['signal_model'] = path
        print(f"  Signal model → {path}")

    if best_vol_model:
        path = os.path.join(output_dir, 'volatility_model.joblib')
        joblib.dump(best_vol_model, path)
        models_saved['volatility_model'] = path
        print(f"  Volatility model → {path}")

    if best_quality_model:
        path = os.path.join(output_dir, 'quality_model.joblib')
        joblib.dump(best_quality_model, path)
        models_saved['quality_model'] = path
        print(f"  Quality model → {path}")

    # Save feature column list
    meta = {
        'feature_columns': feature_cols,
        'window': window,
        'rr_ratio': rr_ratio,
        'max_hold_bars': max_hold_bars,
        'n_folds': n_folds,
        'signal_metrics': signal_metrics,
        'volatility_metrics': vol_metrics,
        'quality_metrics': quality_metrics,
        'models': models_saved,
        'trained_at': datetime.now().isoformat(),
    }
    meta_path = os.path.join(output_dir, 'training_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Metadata → {meta_path}")

    # Save prepared features for backtesting
    features_path = os.path.join(output_dir, 'training_features.parquet')
    try:
        bars.to_parquet(features_path)
    except ImportError:
        features_path = features_path.replace('.parquet', '.csv')
        bars.to_csv(features_path, date_format='%Y-%m-%d %H:%M:%S')
    print(f"  Features → {features_path}")

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")

    # ---- Trading Pattern Report ----
    if best_signal_model:
        _generate_pattern_report(best_signal_model, X_all, y_signal, feature_cols,
                                 rth_indices, bars.index, output_dir)

    return bars, best_signal_model, best_vol_model, best_quality_model


# ============================================================
# Pattern Discovery Report
# ============================================================

PATTERN_MIN_WIN_RATE = 0.58
PATTERN_MIN_TRADES_PER_WEEK = 5
PATTERN_N_CONDITIONS = 3


def _pattern_trades_per_week(mask, trade_mask, timestamps):
    """Average matching trades per ISO calendar week."""
    hit_mask = mask & trade_mask
    n_trades = int(hit_mask.sum())
    if n_trades == 0:
        return 0.0, 0
    ts = pd.to_datetime(np.asarray(timestamps)[hit_mask])
    n_weeks = pd.Series(ts).dt.to_period('W').nunique()
    if n_weeks == 0:
        return 0.0, 0
    return n_trades / n_weeks, n_weeks


def _generate_pattern_report(model, X_all, y_signal, feature_cols, rth_indices,
                             timestamps, output_dir):
    """Mine 3-condition trading rules from the trained model and actual outcomes."""
    print(f"\n{'='*60}")
    print(f"  TRADING PATTERN REPORT")
    print(f"{'='*60}")

    X_rth = X_all[rth_indices]
    y_rth = y_signal[rth_indices]
    ts_rth = np.asarray(timestamps)[rth_indices]

    # Get per-sample leaf contributions for direction classes
    has_contribs = hasattr(model, 'booster_')
    if has_contribs:
        raw = model.booster_.predict(X_rth, pred_contrib=True)
        n_feats = len(feature_cols)
        # LightGBM multiclass: shape (n_samples, (n_features+1)*n_classes)
        if raw.ndim == 2:
            n_classes = 3
            raw = raw.reshape(len(X_rth), n_feats + 1, n_classes)
        contribs_long = raw[:, :n_feats, 2]   # contributions toward long
        contribs_short = raw[:, :n_feats, 0]  # contributions toward short

    # Define condition thresholds using percentiles
    conditions = _build_conditions(X_rth, feature_cols)

    # Find multi-condition patterns with high win rates
    patterns = []
    for direction, dir_label in [(1, 'LONG'), (-1, 'SHORT')]:
        dir_mask = (y_rth == direction)
        opp_mask = (y_rth == -direction)
        trade_mask = dir_mask | opp_mask  # bars where either direction won

        if trade_mask.sum() < 50:
            continue

        # Get top contributing features for this direction
        if has_contribs:
            contribs = contribs_long if direction == 1 else contribs_short
            mean_contrib = contribs[dir_mask].mean(axis=0) if dir_mask.sum() > 0 else np.zeros(len(feature_cols))
            top_features = np.argsort(np.abs(mean_contrib))[::-1][:8]
        else:
            imp = model.feature_importances_
            top_features = np.argsort(imp)[::-1][:8]

        # Search for 3-condition patterns among top features
        for i in range(len(top_features)):
            fi = top_features[i]
            for cond_i in conditions.get(fi, []):
                mask_i = cond_i['mask']
                for j in range(i + 1, len(top_features)):
                    fj = top_features[j]
                    for cond_j in conditions.get(fj, []):
                        mask_ij = mask_i & cond_j['mask']
                        for k in range(j + 1, len(top_features)):
                            fk = top_features[k]
                            for cond_k in conditions.get(fk, []):
                                mask_ijk = mask_ij & cond_k['mask']
                                n_trades = int((mask_ijk & trade_mask).sum())
                                if n_trades == 0:
                                    continue
                                trades_per_week, n_weeks = _pattern_trades_per_week(
                                    mask_ijk, trade_mask, ts_rth)
                                if trades_per_week < PATTERN_MIN_TRADES_PER_WEEK:
                                    continue
                                n_wins = int((mask_ijk & dir_mask).sum())
                                win_rate = n_wins / n_trades
                                if win_rate >= PATTERN_MIN_WIN_RATE:
                                    patterns.append({
                                        'direction': dir_label,
                                        'conditions': [
                                            cond_i['desc'], cond_j['desc'], cond_k['desc']],
                                        'rules': [
                                            cond_i.get('rule', {}),
                                            cond_j.get('rule', {}),
                                            cond_k.get('rule', {})],
                                        'win_rate': win_rate,
                                        'n_trades': n_trades,
                                        'n_wins': n_wins,
                                        'trades_per_week': round(trades_per_week, 1),
                                        'n_weeks': n_weeks,
                                    })

    # Sort by win_rate * n_trades (balance quality and quantity)
    patterns.sort(key=lambda p: p['win_rate'] * p['n_trades'], reverse=True)
    patterns = patterns[:30]  # top 30 patterns

    # Print report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  DISCOVERED TRADING PATTERNS")
    report_lines.append(
        f"  ({PATTERN_N_CONDITIONS}-condition rules, >={PATTERN_MIN_WIN_RATE:.0%} win rate, "
        f">={PATTERN_MIN_TRADES_PER_WEEK} trades/week)")
    report_lines.append("=" * 70)

    # Collect all feature names used in patterns for glossary
    used_features = set()

    for idx, p in enumerate(patterns, 1):
        tpw = p.get('trades_per_week', 0)
        report_lines.append(
            f"\n  Pattern #{idx}: {p['direction']} -- {p['win_rate']:.0%} win rate "
            f"({p['n_wins']}/{p['n_trades']} trades, {tpw:.1f}/week)")
        report_lines.append(f"  When:")
        for cond in p['conditions']:
            report_lines.append(f"    * {cond}")
            # Extract feature name from condition description
            for fname in feature_cols:
                if fname in cond:
                    used_features.add(fname)
                    break
        report_lines.append(f"  -> {p['win_rate']:.0%} chance of {p['direction'].lower()} continuation")

    # Add glossary of terms used
    report_lines.append(f"\n{'='*70}")
    report_lines.append("  GLOSSARY: What each term means and how it is calculated")
    report_lines.append("=" * 70)
    for fname in sorted(used_features):
        desc = _get_feature_description(fname)
        if desc:
            report_lines.append(f"\n  {fname}")
            report_lines.append(f"    {desc}")

    report_lines.append(f"\n{'='*70}")

    report_text = '\n'.join(report_lines)
    print(report_text)

    # Save to file
    report_path = os.path.join(output_dir, 'trading_patterns.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n  Pattern report → {report_path}")

    # Also save as JSON for programmatic use
    json_path = os.path.join(output_dir, 'trading_patterns.json')
    with open(json_path, 'w') as f:
        json.dump({'patterns': patterns, 'generated_at': datetime.now().isoformat()}, f, indent=2)


def _get_feature_description(fname):
    """Return a human-readable description of how a feature is calculated."""
    descriptions = {
        # CVD features
        'cvd': "Cumulative Volume Delta: running sum of (buy_volume - sell_volume) across all bars. "
               "Positive = net buying over time, negative = net selling. Resets at session open (18:00 ET).",
        'cvd_slope_3': "CVD slope over 3 bars: cvd.diff(3)/3. How fast CVD is changing in the short term.",
        'cvd_slope_5': "CVD slope over 5 bars: cvd.diff(5)/5. Medium-term rate of change of cumulative delta.",
        'cvd_slope_10': "CVD slope over 10 bars: cvd.diff(10)/10. Longer-term CVD momentum.",
        'cvd_accel': "CVD acceleration: derivative of cvd_slope_3. Positive = buying accelerating, negative = selling accelerating.",
        'cvd_accel_5': "CVD acceleration (5-bar): derivative of cvd_slope_5.",
        'cvd_roc_3': "CVD rate of change over 3 bars: cvd.diff(3) / abs(cvd_3_bars_ago). Normalized CVD momentum.",
        'cvd_roc_10': "CVD rate of change over 10 bars: normalized longer-term CVD shift.",
        'cvd_bull_div': "CVD bullish divergence: price making new 5-bar highs but CVD is not. Signals hidden selling.",
        'cvd_bear_div': "CVD bearish divergence: price making new 5-bar lows but CVD is not. Signals hidden buying.",

        # Delta features
        'delta': "Bar delta: buy_volume - sell_volume for this single bar. Positive = more buying, negative = more selling.",
        'delta_pct': "Delta as % of total volume: delta / volume. Normalized measure of directional aggression.",
        'delta_divergence': "1 if bar direction (mid vs open) disagrees with delta sign. Price went one way but flow went the other.",
        'delta_sign': "Sign of delta: +1 (net buying), -1 (net selling), 0 (neutral).",
        'delta_streak': "Consecutive bars with same delta direction. Longer streak = sustained one-sided flow.",

        # Pressure ratios
        'pressure_ratio_5': "5-bar buy/sell pressure: sum(buy_volume over 5 bars) / sum(sell_volume over 5 bars). "
                           ">1.5 = strong buy pressure, <0.67 = strong sell pressure.",
        'pressure_ratio_10': "10-bar buy/sell pressure: same as pressure_ratio_5 but over 10 bars. Longer-term aggression balance.",

        # Imbalance features
        'buy_imbalance_count': "Number of price levels in this bar where buy volume >= 3x the sell volume at the level below (footprint imbalance).",
        'sell_imbalance_count': "Number of price levels where sell volume >= 3x the buy volume at the level above.",
        'buy_imbalance_cluster': "Longest consecutive stack of buy imbalances (ascending prices). 3+ = strong buying wall.",
        'sell_imbalance_cluster': "Longest consecutive stack of sell imbalances (descending prices). 3+ = strong selling wall.",
        'imbalance_cluster_net': "buy_imbalance_cluster - sell_imbalance_cluster. Positive = buy-side stacking dominates.",
        'imbalance_cluster_3': "Rolling 3-bar sum of imbalance_cluster_net. Short-term stacking trend.",
        'imbalance_cluster_5': "Rolling 5-bar sum of imbalance_cluster_net. Medium-term stacking trend.",
        'imbalance_strength': "Rolling 5-bar sum of (buy_imbalance_count - sell_imbalance_count). Overall imbalance bias.",
        'net_imbalance': "buy_imbalance_count - sell_imbalance_count for this bar. Positive = more buy imbalances.",

        # Multi-timeframe
        'tf5_range': "Price range (high-low) over last 5 bars. Measures short-term volatility expansion.",
        'tf5_delta_sum': "Sum of delta over last 5 bars. Net order flow direction over ~5 minutes.",
        'tf5_vol_sum': "Total volume over last 5 bars.",
        'tf5_imbalance': "Mean flow_imbalance over last 5 bars. Smoothed directional bias.",
        'tf5_absorption': "Mean absorption over last 5 bars.",
        'tf5_intensity': "Mean trade intensity over last 5 bars.",
        'tf5_imb_cluster_sum': "Rolling 5-bar sum of imbalance_cluster_net.",
        'tf15_range': "Price range over last 15 bars (~15 min). Higher timeframe volatility context.",
        'tf15_delta_sum': "Sum of delta over last 15 bars. Net flow on a higher timeframe.",
        'tf15_vol_sum': "Total volume over last 15 bars. Activity level on higher timeframe.",
        'tf15_imbalance': "Mean flow_imbalance over last 15 bars. Higher-timeframe directional bias.",
        'tf15_absorption': "Mean absorption over last 15 bars. Sustained trapping activity.",
        'tf15_intensity': "Mean trade intensity (events/sec) over last 15 bars.",
        'tf15_imb_cluster_sum': "Rolling 15-bar sum of imbalance_cluster_net. Are imbalance clusters consistently one-sided?",

        # Flow features
        'flow_imbalance': "Per-bar order flow imbalance: (buy_vol - sell_vol) / total_vol. Ranges -1 to +1.",
        'flow_accel': "Flow imbalance acceleration: flow_imbalance.diff(). Positive = flow turning more bullish.",
        'flow_accel_3': "Flow imbalance acceleration over 3 bars.",
        'roll_3_flow_imb': "Rolling 3-bar mean of flow_imbalance. Short-term directional flow.",
        'roll_5_flow_imb': "Rolling 5-bar mean of flow_imbalance.",
        'roll_10_flow_imb': "Rolling 10-bar mean of flow_imbalance. Longer-term flow trend.",

        # Volume/Intensity
        'volume': "Total contracts traded in this bar.",
        'vol_surge': "Current volume / 20-bar average volume. >2 = volume spike.",
        'vol_ma_20': "20-bar moving average of volume. Baseline activity level.",
        'intensity': "Trade events per second in this bar. High = fast market.",
        'intensity_surge': "Current intensity / 10-bar mean intensity. >2 = sudden speed increase.",
        'intensity_accel': "Change in intensity from previous bar.",
        'intensity_accel_3': "Change in intensity over 3 bars.",
        'roll_3_vol_ratio': "Current volume / 3-bar rolling mean volume.",
        'roll_5_vol_ratio': "Current volume / 5-bar rolling mean volume.",
        'roll_10_vol_ratio': "Current volume / 10-bar rolling mean volume.",
        'roll_3_intensity': "Rolling 3-bar mean of trade intensity.",
        'roll_5_intensity': "Rolling 5-bar mean of trade intensity.",
        'roll_10_intensity': "Rolling 10-bar mean of trade intensity.",

        # Absorption
        'absorption': "Volume per tick / 20-bar mean of volume per tick. "
                     ">1.5 = high volume absorbed in small range (trapped traders, institutional activity).",
        'vol_per_tick': "Volume divided by number of ticks in bar range. High = lots of contracts at each price level.",

        # Volatility/Range
        'bar_range': "High - low of this bar in points.",
        'atr_5': "Average True Range over 5 bars. Short-term volatility.",
        'atr_14': "Average True Range over 14 bars. Standard volatility measure.",
        'atr_ratio': "atr_5 / atr_14. >1 = volatility expanding, <1 = contracting.",
        'range_vs_atr': "bar_range / atr_14. >1.5 = unusually large bar, <0.5 = unusually small.",
        'range_ma_5': "5-bar mean of bar_range.",
        'range_ma_20': "20-bar mean of bar_range. Baseline range.",
        'consolidation': "range_ma_5 / range_ma_20. <0.7 = narrowing (consolidation), >1.3 = expanding (breakout).",
        'roll_3_volatility': "Rolling 3-bar std of 1-bar returns.",
        'roll_5_volatility': "Rolling 5-bar std of 1-bar returns.",
        'roll_10_volatility': "Rolling 10-bar std of 1-bar returns. Longer-term vol regime.",

        # Time
        'hour_sin': "Cyclical time encoding (sin). Captures time-of-day patterns without hard boundaries.",
        'hour_cos': "Cyclical time encoding (cos). Together with hour_sin, encodes the 24-hour cycle.",
        'session_rth': "1 if within Regular Trading Hours (9:30 AM - 4:00 PM ET), 0 otherwise.",
        'session_premarket': "1 if premarket (4:00 AM - 9:30 AM ET).",
        'session_overnight': "1 if overnight session (6:00 PM - 4:00 AM ET).",

        # Other
        'large_trade_ratio': "Largest single transaction / total bar volume. High = institutional block trade.",
        'wide_spread_frac': "Fraction of trades in bar with bid-ask spread >= 2 ticks. High = low liquidity.",
        'max_buy_run': "Longest consecutive buy ticks within this bar.",
        'max_sell_run': "Longest consecutive sell ticks within this bar.",
        'gap_from_prev_high': "Open price - previous bar's high. Positive = gapped above prior high.",
        'gap_from_prev_low': "Open price - previous bar's low. Negative = gapped below prior low.",
        'max_events_per_tick': "Max number of trade events at any single price level in this bar.",
        'level_concentration': "Fraction of total volume at the most-traded price level. High = heavy activity at one price.",
        'cvd_range_div_5': "abs(CVD slope 5-bar) / 5-bar avg range. High = CVD moving fast but price stuck. Setup forming.",
        'cvd_range_div_10': "Same as cvd_range_div_5 but over 10 bars. Longer-term divergence.",
        'cvd_flow_alignment': "sign(cvd_slope) * flow_imbalance. Positive = CVD and flow agree (strength). Negative = disagree (absorption).",
        'cvd_div_x_pressure': "cvd_range_div * (pressure_ratio - 1). High positive = CVD stuck + buy dominant (bullish strength). High negative = sell dominant (bearish strength).",
        'cvd_div_x_absorption': "cvd_range_div * absorption. High = CVD diverging while volume is being absorbed. Signals trapped traders, likely reversal.",
        'cvd_div_x_imbalance': "cvd_range_div * net_imbalance. CVD stuck + footprint imbalance stacking same direction = confirms strength/continuation.",

        # Volume trend
        'vol_roc_5': "Volume rate of change over 5 bars: (vol - vol_5_ago) / vol_5_ago. Positive = increasing, negative = decreasing.",
        'vol_trend_5': "Discrete volume trend over 5 bars: +1 (increasing >10%), -1 (decreasing >10%), 0 (flat).",

        # Value area (10-bar cumulative volume profile)
        'va10_price_vs_poc': "(mid - POC) / ATR_14. Price relative to 10-bar cumulative fair value. Positive = above POC.",
        'va10_price_vs_vah': "(mid - VAH) / ATR_14. Positive = above value area high (breakout territory).",
        'va10_price_vs_val': "(mid - VAL) / ATR_14. Negative = below value area low (breakdown territory).",
        'va10_in_value_area': "1 if current price is inside the 10-bar 70% value area (between VAL and VAH).",
        'va10_above_vah': "1 if current price is above value area high (potential breakout).",
        'va10_below_val': "1 if current price is below value area low (potential breakdown).",
        'va10_va_width': "(VAH - VAL) / ATR_14. Width of accepted value range. Low = tight consolidation.",
        'va10_volume_at_price': "Volume at current price / max volume in 10-bar profile. High = at accepted value.",
        'va10_price_percentile': "Percentile of current price in 10-bar volume distribution. 0.5 = middle of range.",
    }
    return descriptions.get(fname, None)


def _build_conditions(X, feature_cols):
    """Build testable conditions with machine-readable rules for each feature."""
    conditions = {}

    def _c(mask, desc, fname, op, thresh):
        return {'mask': mask, 'desc': desc, 'rule': {'feature': fname, 'op': op, 'threshold': float(thresh)}}

    for fi, fname in enumerate(feature_cols):
        vals = X[:, fi]
        conditions[fi] = []

        if 'cvd' == fname or fname.startswith('cvd_slope') or fname.startswith('cvd_accel'):
            # CVD: rising vs falling, extreme levels
            med = np.median(vals)
            p75 = np.percentile(vals, 75)
            p25 = np.percentile(vals, 25)
            conditions[fi].append(_c(vals > p75, f'{fname} elevated (>{p75:.0f}, top 25%)', fname, '>', p75))
            conditions[fi].append(_c(vals < p25, f'{fname} depressed (<{p25:.0f}, bottom 25%)', fname, '<', p25))
            conditions[fi].append(_c(vals > 0, f'{fname} positive (buyers leading)', fname, '>', 0))
            conditions[fi].append(_c(vals < 0, f'{fname} negative (sellers leading)', fname, '<', 0))

        elif 'imbalance' in fname:
            if 'cluster' in fname:
                conditions[fi].append(_c(vals >= 3, f'{fname} >= 3 (strong cluster)', fname, '>=', 3))
                conditions[fi].append(_c(vals <= -3, f'{fname} <= -3 (strong sell cluster)', fname, '<=', -3))
            else:
                p75 = np.percentile(vals, 75)
                p25 = np.percentile(vals, 25)
                conditions[fi].append(_c(vals > p75, f'{fname} high (>{p75:.1f})', fname, '>', p75))
                conditions[fi].append(_c(vals < p25, f'{fname} low (<{p25:.1f})', fname, '<', p25))

        elif 'delta' in fname:
            p80 = np.percentile(vals, 80)
            p20 = np.percentile(vals, 20)
            conditions[fi].append(_c(vals > p80, f'{fname} strongly positive (>{p80:.0f})', fname, '>', p80))
            conditions[fi].append(_c(vals < p20, f'{fname} strongly negative (<{p20:.0f})', fname, '<', p20))
            conditions[fi].append(_c(vals > 0, f'{fname} > 0 (net buying)', fname, '>', 0))
            conditions[fi].append(_c(vals < 0, f'{fname} < 0 (net selling)', fname, '<', 0))

        elif 'absorption' in fname:
            p75 = np.percentile(vals, 75)
            conditions[fi].append(_c(vals > p75, f'high {fname} (>{p75:.1f}x, trapped traders)', fname, '>', p75))
            conditions[fi].append(_c(vals > 2.0, f'{fname} > 2x (extreme absorption)', fname, '>', 2.0))

        elif 'divergence' in fname or '_div' in fname:
            conditions[fi].append(_c(vals == 1, f'{fname} active (price/flow disagree)', fname, '==', 1))

        elif 'intensity' in fname or 'surge' in fname:
            p75 = np.percentile(vals, 75)
            conditions[fi].append(_c(vals > p75, f'{fname} surging (>{p75:.1f})', fname, '>', p75))
            conditions[fi].append(_c(vals > 2.0, f'{fname} > 2x (acceleration)', fname, '>', 2.0))

        elif 'pressure' in fname:
            conditions[fi].append(_c(vals > 1.5, f'{fname} > 1.5 (strong buy pressure)', fname, '>', 1.5))
            conditions[fi].append(_c(vals < 0.67, f'{fname} < 0.67 (strong sell pressure)', fname, '<', 0.67))

        elif 'consolidation' in fname:
            conditions[fi].append(_c(vals < 0.7, 'range contracting (consolidation)', fname, '<', 0.7))
            conditions[fi].append(_c(vals > 1.3, 'range expanding (breakout)', fname, '>', 1.3))

        elif 'flow' in fname:
            p75 = np.percentile(vals, 75)
            p25 = np.percentile(vals, 25)
            conditions[fi].append(_c(vals > p75, f'{fname} bullish (>{p75:.3f})', fname, '>', p75))
            conditions[fi].append(_c(vals < p25, f'{fname} bearish (<{p25:.3f})', fname, '<', p25))

        else:
            # Generic: above/below median, extreme quartiles
            p75 = np.percentile(vals, 75)
            p25 = np.percentile(vals, 25)
            if not np.isnan(p75) and p75 != p25:
                conditions[fi].append(_c(vals > p75, f'{fname} high (>{p75:.2f})', fname, '>', p75))
                conditions[fi].append(_c(vals < p25, f'{fname} low (<{p25:.2f})', fname, '<', p25))

    return conditions


def main():
    parser = argparse.ArgumentParser(
        description='Train multi-model ES mini trading system.')
    parser.add_argument('data_dir', help='Directory containing .data files')
    parser.add_argument('--window', default='1min', help='Bar aggregation window (default: 1min)')
    parser.add_argument('--rr', type=float, default=1.5, help='Minimum risk-reward ratio (default: 1.5)')
    parser.add_argument('--max-hold', type=int, default=60, help='Max bars to hold a trade (default: 60)')
    parser.add_argument('--folds', type=int, default=5, help='Walk-forward folds (default: 5)')
    parser.add_argument('--output-dir', default=None, help='Output directory for models (default: ./data/models)')

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: Not a directory: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    run_training(
        args.data_dir, window=args.window, rr_ratio=args.rr,
        max_hold_bars=args.max_hold, n_folds=args.folds,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
