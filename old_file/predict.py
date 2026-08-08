#!/usr/bin/env python3
"""
Prediction script: Load trained models and generate signals.

Given new .data files, runs the feature pipeline then applies
the trained direction + magnitude models to produce predictions.

Usage:
    python3 predict.py /path/to/new/data --models-dir /path/to/models/
"""

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from feature_pipeline import run_pipeline


EXCLUDE_COLS = [
    'direction_label', 'magnitude_label', 'future_return', 'future_close',
    'open', 'close', 'high', 'low',
]


def load_models(models_dir):
    """Load saved LightGBM models."""
    dir_model_path = os.path.join(models_dir, 'direction_model.txt')
    mag_model_path = os.path.join(models_dir, 'magnitude_model.txt')

    dir_model = None
    mag_model = None

    if os.path.isfile(dir_model_path):
        dir_model = lgb.Booster(model_file=dir_model_path)
        print(f"  Loaded direction model: {dir_model_path}")
    else:
        print(f"  WARNING: Direction model not found at {dir_model_path}")

    if os.path.isfile(mag_model_path):
        mag_model = lgb.Booster(model_file=mag_model_path)
        print(f"  Loaded magnitude model: {mag_model_path}")
    else:
        print(f"  WARNING: Magnitude model not found at {mag_model_path}")

    return dir_model, mag_model


def get_feature_columns(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def predict(df, dir_model, mag_model):
    """Run predictions on feature dataframe."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values

    results = pd.DataFrame(index=df.index)
    results['close'] = df['close']

    if dir_model:
        prob_up = dir_model.predict(X)
        results['prob_up'] = prob_up
        results['prob_down'] = 1 - prob_up
        results['direction_pred'] = np.where(prob_up > 0.5, 'UP', 'DOWN')

        # Confidence: distance from 0.5 decision boundary
        results['confidence'] = np.abs(prob_up - 0.5) * 2  # Scale to 0-1

    if mag_model:
        results['expected_magnitude'] = mag_model.predict(X)

    # Composite signal
    if dir_model and mag_model:
        # Signal strength = probability * expected magnitude
        signed_magnitude = np.where(
            prob_up > 0.5,
            results['expected_magnitude'],
            -results['expected_magnitude']
        )
        results['signal_strength'] = (results['confidence'] * signed_magnitude)

        # Categorize confidence
        results['signal'] = 'HOLD'
        results.loc[results['confidence'] > 0.3, 'signal'] = results['direction_pred']

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions using trained models.')
    parser.add_argument('data_dir', help='Directory containing .data files')
    parser.add_argument('--models-dir', required=True,
                        help='Directory containing trained model files')
    parser.add_argument('--window', default='1min',
                        help='Aggregation window (must match training)')
    parser.add_argument('--horizon', default='5min',
                        help='Prediction horizon (must match training)')
    parser.add_argument('--output', default=None,
                        help='Output CSV path for predictions')
    parser.add_argument('--tail', type=int, default=20,
                        help='Number of recent predictions to display')

    args = parser.parse_args()

    print("Loading models...")
    dir_model, mag_model = load_models(args.models_dir)

    if dir_model is None and mag_model is None:
        print("ERROR: No models found.", file=sys.stderr)
        sys.exit(1)

    print("\nRunning feature pipeline...")
    # Run pipeline but capture the dataframe instead of just saving
    features_path = os.path.join(args.data_dir, '_tmp_features.parquet')
    run_pipeline(args.data_dir, window=args.window,
                 horizon=args.horizon, output=features_path)

    df = pd.read_parquet(features_path)
    os.remove(features_path)

    print(f"\nGenerating predictions on {len(df)} bars...")
    predictions = predict(df, dir_model, mag_model)

    # Display recent predictions
    print(f"\n{'='*70}")
    print(f"  PREDICTIONS (last {args.tail} bars)")
    print(f"{'='*70}")
    tail = predictions.tail(args.tail)
    cols_to_show = ['close', 'prob_up', 'expected_magnitude', 'confidence', 'signal']
    cols_available = [c for c in cols_to_show if c in tail.columns]
    print(tail[cols_available].to_string())

    # Save predictions
    if args.output:
        predictions.to_csv(args.output)
        print(f"\nPredictions saved to: {args.output}")

    # Summary stats
    if 'signal' in predictions.columns:
        print(f"\n  Signal distribution:")
        vc = predictions['signal'].value_counts()
        for signal, count in vc.items():
            print(f"    {signal:>5}: {count} ({count/len(predictions)*100:.1f}%)")


if __name__ == '__main__':
    main()
