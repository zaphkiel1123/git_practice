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

import argparse
import os
import sys
import json
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

from feature_pipeline import decode_file_to_dataframe, compute_window_features, add_rolling_features, add_time_features
from labels import create_trade_labels, compute_atr, is_rth

warnings.filterwarnings('ignore', category=UserWarning)

TICK_SIZE = 0.25


# ============================================================
# Enhanced Feature Engineering
# ============================================================

def add_microstructure_features(bars, raw_df=None):
    """Add order-flow microstructure features beyond basic pipeline."""

    # Delta (net buy-sell volume per bar) — already have flow_imbalance, add raw delta
    bars['delta'] = bars['buy_volume'] - bars['sell_volume']

    # Delta divergence: price direction vs delta direction
    bars['price_direction'] = np.sign(bars['close'] - bars['open'])
    bars['delta_direction'] = np.sign(bars['delta'])
    bars['delta_divergence'] = (bars['price_direction'] != bars['delta_direction']).astype(int)

    # CVD (cumulative volume delta) and its slope
    bars['cvd'] = bars['delta'].cumsum()
    bars['cvd_slope_3'] = bars['cvd'].diff(3) / 3
    bars['cvd_slope_5'] = bars['cvd'].diff(5) / 5
    bars['cvd_slope_10'] = bars['cvd'].diff(10) / 10

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

    # Close position within bar range (0=low, 1=high)
    bars['close_position'] = (bars['close'] - bars['low']) / bars['bar_range'].clip(lower=TICK_SIZE)

    # Momentum features
    bars['roc_3'] = bars['close'].pct_change(3)
    bars['roc_5'] = bars['close'].pct_change(5)
    bars['roc_10'] = bars['close'].pct_change(10)

    # Price distance from VWAP
    if 'vwap' in bars.columns:
        bars['vwap_distance'] = (bars['close'] - bars['vwap']) / bars['atr_14'].clip(lower=TICK_SIZE)

    # Swing high/low detection (simplified)
    bars['swing_high_dist'] = bars['high'].rolling(20).max() - bars['close']
    bars['swing_low_dist'] = bars['close'] - bars['low'].rolling(20).min()
    bars['swing_range_pct'] = bars['swing_low_dist'] / (bars['swing_high_dist'] + bars['swing_low_dist']).clip(lower=TICK_SIZE)

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

    # Multi-timeframe context (5-bar = ~5min if 1min bars)
    bars['close_5bar_ma'] = bars['close'].rolling(5).mean()
    bars['close_10bar_ma'] = bars['close'].rolling(10).mean()
    bars['close_20bar_ma'] = bars['close'].rolling(20).mean()
    bars['ma_cross_5_20'] = (bars['close_5bar_ma'] - bars['close_20bar_ma']) / bars['atr_14'].clip(lower=TICK_SIZE)

    # Gap from previous bar close to current open
    bars['gap'] = bars['open'] - bars['close'].shift(1)
    bars['gap_atr'] = bars['gap'] / bars['atr_14'].clip(lower=TICK_SIZE)

    return bars


def add_multi_timeframe_features(bars):
    """Add features from higher timeframe aggregation (5-bar, 15-bar)."""
    for tf in [5, 15]:
        prefix = f'tf{tf}'
        bars[f'{prefix}_high'] = bars['high'].rolling(tf).max()
        bars[f'{prefix}_low'] = bars['low'].rolling(tf).min()
        bars[f'{prefix}_range'] = bars[f'{prefix}_high'] - bars[f'{prefix}_low']
        bars[f'{prefix}_close_pos'] = (bars['close'] - bars[f'{prefix}_low']) / bars[f'{prefix}_range'].clip(lower=TICK_SIZE)
        bars[f'{prefix}_delta_sum'] = bars['delta'].rolling(tf).sum()
        bars[f'{prefix}_vol_sum'] = bars['volume'].rolling(tf).sum()
        bars[f'{prefix}_imbalance'] = bars['flow_imbalance'].rolling(tf).mean()
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
    """Feature columns: everything except labels, metadata, raw OHLC."""
    exclude = {
        'trade_label', 'long_result', 'short_result',
        'long_bars_held', 'short_bars_held',
        'long_mae', 'short_mae', 'long_mfe', 'short_mfe',
        'sl_points', 'tp_points', 'is_rth',
        'open', 'high', 'low', 'close',
        'direction_label', 'magnitude_label', 'future_return', 'future_close',
        'atr_target', 'quality_label',
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
            device='gpu',
        )
        model.fit(X_train, y_tr_mapped,
                  eval_set=[(X_test, y_te_mapped)],
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
            device='gpu',
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
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
            device='gpu',
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
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

def _load_single_file(fp):
    """Load one .data file and return (basename, len, dataframe)."""
    df = decode_file_to_dataframe(fp)
    return os.path.basename(fp), len(df), df


def prepare_features(data_dir, window='1min', rr_ratio=1.5, max_hold_bars=60):
    """Run full feature + label pipeline on all .data files in directory."""
    import glob
    from concurrent.futures import ProcessPoolExecutor

    pattern = os.path.join(data_dir, '*.data')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: No .data files in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} data file(s)")
    all_dfs = []
    with ProcessPoolExecutor() as executor:
        for name, n_ticks, df in executor.map(_load_single_file, files):
            all_dfs.append(df)
            print(f"  {name}: {n_ticks:,} ticks")

    raw_df = pd.concat(all_dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    print(f"Total ticks: {len(raw_df):,}")

    # Compute bars
    print(f"Computing {window} bars...")
    bars = compute_window_features(raw_df, window=window)
    print(f"  {len(bars)} bars")

    # Add features
    print("Adding rolling features...")
    bars = add_rolling_features(bars)
    print("Adding time features...")
    bars = add_time_features(bars)
    print("Adding microstructure features...")
    bars = add_microstructure_features(bars)
    print("Adding multi-timeframe features...")
    bars = add_multi_timeframe_features(bars)

    # Add trade labels
    print(f"Simulating trade outcomes (RR={rr_ratio}, max_hold={max_hold_bars} bars)...")
    bars = create_trade_labels(bars, rr_ratio=rr_ratio, max_hold_bars=max_hold_bars)

    # ATR target for volatility model
    bars['atr_target'] = compute_atr(bars, period=5).shift(-5)

    # Quality label: for rows where trade_label != 0, did that direction win?
    bars['quality_label'] = 0
    long_mask = bars['trade_label'] == 1
    short_mask = bars['trade_label'] == -1
    bars.loc[long_mask, 'quality_label'] = (bars.loc[long_mask, 'long_result'] == 1).astype(int)
    bars.loc[short_mask, 'quality_label'] = (bars.loc[short_mask, 'short_result'] == 1).astype(int)

    # Drop NaN rows from rolling features
    bars = bars.dropna()
    print(f"Complete bars after dropna: {len(bars)}")

    return bars


def run_training(data_dir, window='1min', rr_ratio=1.5, max_hold_bars=60,
                 n_folds=5, output_dir=None):
    """Full training pipeline with walk-forward validation."""

    bars = prepare_features(data_dir, window=window, rr_ratio=rr_ratio,
                            max_hold_bars=max_hold_bars)

    if output_dir is None:
        output_dir = data_dir
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

    # ---- Volatility Model (uses all bars, not just RTH) ----
    print(f"\n{'='*60}")
    print(f"  VOLATILITY MODEL (ATR prediction for SL sizing)")
    print(f"{'='*60}")

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

    # ---- Trade Quality Model (RTH only, only on trade signals) ----
    print(f"\n{'='*60}")
    print(f"  TRADE QUALITY MODEL (win probability filter)")
    print(f"{'='*60}")

    # Only train on bars where a trade direction is indicated
    trade_mask_rth = (y_signal[rth_indices] != 0)
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
        bars.to_csv(features_path)
    print(f"  Features → {features_path}")

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")

    return bars, best_signal_model, best_vol_model, best_quality_model


def main():
    parser = argparse.ArgumentParser(
        description='Train multi-model ES mini trading system.')
    parser.add_argument('data_dir', help='Directory containing .data files')
    parser.add_argument('--window', default='1min', help='Bar aggregation window (default: 1min)')
    parser.add_argument('--rr', type=float, default=1.5, help='Minimum risk-reward ratio (default: 1.5)')
    parser.add_argument('--max-hold', type=int, default=60, help='Max bars to hold a trade (default: 60)')
    parser.add_argument('--folds', type=int, default=5, help='Walk-forward folds (default: 5)')
    parser.add_argument('--output-dir', default=None, help='Output directory for models')

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
