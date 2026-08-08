#!/usr/bin/env python3
"""
Baseline Model: LightGBM with Walk-Forward Validation.

Trains two models:
1. Direction classifier: predicts UP (+1) vs DOWN (-1)
2. Magnitude regressor: predicts absolute price change

Uses walk-forward (expanding window) validation to avoid look-ahead bias.

Usage:
    python3 train_model.py features.parquet
    python3 train_model.py features.parquet --folds 5 --test-ratio 0.2
"""

import argparse
import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

warnings.filterwarnings('ignore', category=UserWarning)


# Columns that are labels or metadata, not features
EXCLUDE_COLS = [
    'direction_label', 'magnitude_label', 'future_return', 'future_close',
    'open', 'close', 'high', 'low',  # raw OHLC (use derived features instead)
]


def get_feature_columns(df):
    """Identify feature columns (everything except labels and raw OHLC)."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def walk_forward_split(df, n_folds=5, min_train_ratio=0.3):
    """
    Generate walk-forward (expanding window) train/test splits.

    Each fold uses all data up to a cutoff for training,
    and the next chunk for testing. The training window expands each fold.
    """
    n = len(df)
    test_size = int(n * (1 - min_train_ratio) / n_folds)

    splits = []
    for i in range(n_folds):
        test_start = int(n * min_train_ratio) + i * test_size
        test_end = min(test_start + test_size, n)

        if test_end > n:
            break

        train_idx = list(range(0, test_start))
        test_idx = list(range(test_start, test_end))

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


def train_direction_model(X_train, y_train, X_test, y_test):
    """Train a classifier for direction prediction (LightGBM or sklearn fallback)."""
    # Filter out FLAT (0) labels for cleaner binary classification
    train_mask = y_train != 0
    test_mask = y_test != 0

    X_tr = X_train[train_mask]
    y_tr = ((y_train[train_mask] + 1) / 2).astype(int)  # Map -1→0, +1→1

    X_te = X_test[test_mask]
    y_te = ((y_test[test_mask] + 1) / 2).astype(int)

    if len(X_tr) == 0 or len(X_te) == 0:
        return None, None, None

    if HAS_LIGHTGBM:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'n_estimators': 300,
            'early_stopping_rounds': 30,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42
        )
        model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_te, y_pred),
        'precision': precision_score(y_te, y_pred, zero_division=0),
        'recall': recall_score(y_te, y_pred, zero_division=0),
        'f1': f1_score(y_te, y_pred, zero_division=0),
        'n_train': len(X_tr),
        'n_test': len(X_te),
        'train_up_pct': float(y_tr.mean()),
        'test_up_pct': float(y_te.mean()),
    }

    return model, metrics, y_prob


def train_magnitude_model(X_train, y_train, X_test, y_test):
    """Train a regressor for magnitude prediction (LightGBM or sklearn fallback)."""
    # Remove rows where magnitude is 0 or NaN
    train_mask = y_train > 0
    test_mask = y_test > 0

    X_tr = X_train[train_mask]
    y_tr = y_train[train_mask]
    X_te = X_test[test_mask]
    y_te = y_test[test_mask]

    if len(X_tr) == 0 or len(X_te) == 0:
        return None, None

    if HAS_LIGHTGBM:
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'n_estimators': 300,
            'early_stopping_rounds': 30,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])
    else:
        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42
        )
        model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)

    metrics = {
        'mae': mean_absolute_error(y_te, y_pred),
        'rmse': float(np.sqrt(mean_squared_error(y_te, y_pred))),
        'r2': r2_score(y_te, y_pred),
        'mean_actual': float(y_te.mean()),
        'mean_predicted': float(y_pred.mean()),
        'n_train': len(X_tr),
        'n_test': len(X_te),
    }

    return model, metrics


def compute_feature_importance(model, feature_names, top_n=15):
    """Extract top feature importances from a trained model."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    return [(feature_names[i], float(importance[i])) for i in indices]


def run_training(parquet_path, n_folds=5, output_dir=None):
    """Run the full walk-forward training and evaluation."""
    print(f"Loading features from: {parquet_path}")

    if parquet_path.endswith('.parquet'):
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(parquet_path, index_col=0, parse_dates=True)

    print(f"  Shape: {df.shape}")
    print(f"  Time range: {df.index.min()} → {df.index.max()}")
    if HAS_LIGHTGBM:
        print(f"  Backend: LightGBM")
    else:
        print(f"  Backend: sklearn GradientBoosting (LightGBM not available)")

    feature_cols = get_feature_columns(df)
    print(f"  Feature columns: {len(feature_cols)}")

    X = df[feature_cols].values
    y_dir = df['direction_label'].values
    y_mag = df['magnitude_label'].values

    # Walk-forward splits
    splits = walk_forward_split(df, n_folds=n_folds)
    print(f"\n  Walk-forward folds: {len(splits)}")

    # ---- Direction Model ----
    print(f"\n{'='*60}")
    print(f"  DIRECTION CLASSIFIER (UP vs DOWN)")
    print(f"{'='*60}")

    dir_metrics_all = []
    best_dir_model = None
    best_dir_acc = 0

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_dir[train_idx], y_dir[test_idx]

        model, metrics, _ = train_direction_model(X_train, y_train, X_test, y_test)
        if metrics is None:
            print(f"  Fold {fold_idx+1}: SKIPPED (insufficient data)")
            continue

        dir_metrics_all.append(metrics)
        print(f"  Fold {fold_idx+1}: acc={metrics['accuracy']:.4f}, "
              f"prec={metrics['precision']:.4f}, rec={metrics['recall']:.4f}, "
              f"f1={metrics['f1']:.4f} (train={metrics['n_train']}, test={metrics['n_test']})")

        if metrics['accuracy'] > best_dir_acc:
            best_dir_acc = metrics['accuracy']
            best_dir_model = model

    if dir_metrics_all:
        avg_dir = {k: np.mean([m[k] for m in dir_metrics_all])
                   for k in dir_metrics_all[0]}
        print(f"\n  AVERAGE: acc={avg_dir['accuracy']:.4f}, "
              f"prec={avg_dir['precision']:.4f}, rec={avg_dir['recall']:.4f}, "
              f"f1={avg_dir['f1']:.4f}")
        print(f"  Baseline (always predict majority): {max(avg_dir['test_up_pct'], 1-avg_dir['test_up_pct']):.4f}")

    # ---- Magnitude Model ----
    print(f"\n{'='*60}")
    print(f"  MAGNITUDE REGRESSOR (absolute price change)")
    print(f"{'='*60}")

    mag_metrics_all = []
    best_mag_model = None
    best_mag_mae = float('inf')

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_mag[train_idx], y_mag[test_idx]

        model, metrics = train_magnitude_model(X_train, y_train, X_test, y_test)
        if metrics is None:
            print(f"  Fold {fold_idx+1}: SKIPPED (insufficient data)")
            continue

        mag_metrics_all.append(metrics)
        print(f"  Fold {fold_idx+1}: MAE={metrics['mae']:.4f}, "
              f"RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f} "
              f"(mean_actual={metrics['mean_actual']:.4f})")

        if metrics['mae'] < best_mag_mae:
            best_mag_mae = metrics['mae']
            best_mag_model = model

    if mag_metrics_all:
        avg_mag = {k: np.mean([m[k] for m in mag_metrics_all])
                   for k in mag_metrics_all[0]}
        print(f"\n  AVERAGE: MAE={avg_mag['mae']:.4f}, "
              f"RMSE={avg_mag['rmse']:.4f}, R2={avg_mag['r2']:.4f}")
        print(f"  Naive baseline (predict mean): MAE≈{avg_mag['mean_actual']:.4f}")

    # ---- Feature Importance ----
    print(f"\n{'='*60}")
    print(f"  FEATURE IMPORTANCE (top 15)")
    print(f"{'='*60}")

    if best_dir_model:
        print(f"\n  Direction model:")
        imp = compute_feature_importance(best_dir_model, feature_cols)
        for name, score in imp:
            print(f"    {name:<30} {score}")

    if best_mag_model:
        print(f"\n  Magnitude model:")
        imp = compute_feature_importance(best_mag_model, feature_cols)
        for name, score in imp:
            print(f"    {name:<30} {score}")

    # ---- Save results ----
    if output_dir is None:
        output_dir = os.path.dirname(parquet_path) or '.'

    results = {
        'run_time': datetime.now().isoformat(),
        'input_file': parquet_path,
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'n_folds': len(splits),
        'direction_metrics': dir_metrics_all,
        'magnitude_metrics': mag_metrics_all,
    }

    results_path = os.path.join(output_dir, 'model_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    # Save models
    if best_dir_model:
        if HAS_LIGHTGBM:
            dir_model_path = os.path.join(output_dir, 'direction_model.txt')
            best_dir_model.booster_.save_model(dir_model_path)
        else:
            import pickle
            dir_model_path = os.path.join(output_dir, 'direction_model.pkl')
            with open(dir_model_path, 'wb') as f:
                pickle.dump(best_dir_model, f)
        print(f"  Direction model saved to: {dir_model_path}")

    if best_mag_model:
        if HAS_LIGHTGBM:
            mag_model_path = os.path.join(output_dir, 'magnitude_model.txt')
            best_mag_model.booster_.save_model(mag_model_path)
        else:
            import pickle
            mag_model_path = os.path.join(output_dir, 'magnitude_model.pkl')
            with open(mag_model_path, 'wb') as f:
                pickle.dump(best_mag_model, f)
        print(f"  Magnitude model saved to: {mag_model_path}")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train LightGBM models with walk-forward validation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s features.parquet                 # Train with default 5 folds
  %(prog)s features.parquet --folds 10      # Use 10 walk-forward folds
  %(prog)s features.parquet --output-dir .  # Save models to current dir
        """)
    parser.add_argument('parquet_file', help='Path to the features parquet file')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of walk-forward folds (default: 5)')
    parser.add_argument('--output-dir', default=None,
                        help='Directory to save models and results')

    args = parser.parse_args()

    if not os.path.isfile(args.parquet_file):
        print(f"Error: File not found: {args.parquet_file}", file=sys.stderr)
        sys.exit(1)

    run_training(args.parquet_file, n_folds=args.folds, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
