#!/usr/bin/env python3
"""
Live/paper trading signal generator for ES mini.

Watches a directory for new/updated .data files, maintains a rolling
feature window, and emits real-time trade signals with SL/TP levels.

Usage:
    python3 live_signal.py /path/to/models/ --watch /path/to/live_data/
    python3 live_signal.py /path/to/models/ --file new_session.data
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import pandas as pd
import joblib

from feature_pipeline import decode_file_to_dataframe, compute_window_features, add_rolling_features, add_time_features, add_value_area_features
from train_trading_model import add_microstructure_features, add_multi_timeframe_features
from labels import compute_atr, is_rth, TICK_SIZE
from core_features import check_core_alignment, DEFAULT_CORE_CONFIG


def load_models(models_dir):
    """Load all three models and metadata."""
    meta_path = os.path.join(models_dir, 'training_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)

    signal_model = None
    vol_model = None
    quality_model = None

    sp = os.path.join(models_dir, 'signal_model.joblib')
    vp = os.path.join(models_dir, 'volatility_model.joblib')
    qp = os.path.join(models_dir, 'quality_model.joblib')

    if os.path.isfile(sp):
        signal_model = joblib.load(sp)
    if os.path.isfile(vp):
        vol_model = joblib.load(vp)
    if os.path.isfile(qp):
        quality_model = joblib.load(qp)

    return signal_model, vol_model, quality_model, meta


def compute_features_for_file(filepath, window='1min'):
    """Decode a .data file and compute full feature set."""
    raw_df = decode_file_to_dataframe(filepath)
    if len(raw_df) == 0:
        return None

    bars = compute_window_features(raw_df, window=window)
    bars = add_rolling_features(bars)
    bars = add_time_features(bars)
    bars = add_microstructure_features(bars)
    bars = add_multi_timeframe_features(bars)
    bars = add_value_area_features(bars)
    bars = bars.drop(columns=['_vol_profile'], errors='ignore')
    bars = bars.dropna()
    return bars


def generate_signal(bars, feature_cols, signal_model, vol_model, quality_model,
                    rr_ratio=1.5, signal_threshold=0.55, quality_threshold=0.50,
                    sl_multiplier=1.5, min_sl=2.0, max_sl=8.0, core_config=None):
    """Generate signal for the latest bar."""
    if len(bars) == 0:
        return None

    latest = bars.iloc[-1:]
    X = latest[feature_cols].values

    # Check RTH
    rth = is_rth(latest.index)
    if not rth.iloc[0]:
        return {'signal': 'NO_TRADE', 'reason': 'Outside RTH (9:30-15:30 ET)'}

    # Direction prediction
    if hasattr(signal_model, 'predict_proba'):
        probs = signal_model.predict_proba(X)[0]
        prob_long = probs[2] if len(probs) == 3 else probs[1]
        prob_short = probs[0]
    else:
        pred = signal_model.predict(X)[0]
        prob_long = 1.0 if pred == 2 else 0.0
        prob_short = 1.0 if pred == 0 else 0.0

    direction = 0
    confidence = 0.0
    if prob_long > signal_threshold and prob_long > prob_short:
        direction = 1
        confidence = prob_long
    elif prob_short > signal_threshold and prob_short > prob_long:
        direction = -1
        confidence = prob_short

    if direction == 0:
        return {
            'signal': 'NO_TRADE',
            'reason': f'Low confidence (long={prob_long:.3f}, short={prob_short:.3f})',
            'prob_long': float(prob_long),
            'prob_short': float(prob_short),
        }

    # Core feature alignment gate
    feature_dict = dict(zip(feature_cols, X[0]))
    aligned, gate_reason = check_core_alignment(feature_dict, direction, core_config)
    if not aligned:
        return {
            'signal': 'NO_TRADE',
            'reason': f'Core gate: {gate_reason}',
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'prob_long': float(prob_long),
            'prob_short': float(prob_short),
        }

    # Quality filter
    if quality_model is not None:
        if hasattr(quality_model, 'predict_proba'):
            q_prob = quality_model.predict_proba(X)[0][1]
        else:
            q_prob = float(quality_model.predict(X)[0])
        if q_prob < quality_threshold:
            return {
                'signal': 'NO_TRADE',
                'reason': f'Quality filter rejected (q={q_prob:.3f} < {quality_threshold})',
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'quality_prob': float(q_prob),
            }
    else:
        q_prob = None

    # Compute SL from volatility model
    if vol_model is not None:
        pred_atr = vol_model.predict(X)[0]
        sl_points = np.clip(pred_atr * sl_multiplier, min_sl, max_sl)
    else:
        sl_points = float(latest['atr_14'].iloc[0]) * sl_multiplier
        sl_points = np.clip(sl_points, min_sl, max_sl)

    sl_points = round(sl_points / TICK_SIZE) * TICK_SIZE
    tp_points = sl_points * rr_ratio

    entry_price = float(latest['close'].iloc[0])
    if direction == 1:
        sl_price = entry_price - sl_points
        tp_price = entry_price + tp_points
    else:
        sl_price = entry_price + sl_points
        tp_price = entry_price - tp_points

    return {
        'signal': 'LONG' if direction == 1 else 'SHORT',
        'time': str(latest.index[0]),
        'entry_price': entry_price,
        'sl_price': float(sl_price),
        'tp_price': float(tp_price),
        'sl_points': float(sl_points),
        'tp_points': float(tp_points),
        'rr_ratio': float(rr_ratio),
        'confidence': float(confidence),
        'quality_prob': float(q_prob) if q_prob is not None else None,
        'prob_long': float(prob_long),
        'prob_short': float(prob_short),
    }


def watch_mode(models_dir, watch_dir, window, rr_ratio, signal_threshold, quality_threshold):
    """Continuously watch for new data and emit signals."""
    signal_model, vol_model, quality_model, meta = load_models(models_dir)
    feature_cols = meta['feature_columns']
    core_config = meta.get('core_feature_config', DEFAULT_CORE_CONFIG)

    print(f"Watching {watch_dir} for .data file changes...")
    print(f"  Window: {window} | RR: {rr_ratio} | Signal threshold: {signal_threshold}")
    print(f"  Models loaded from: {models_dir}")
    print(f"  Core gate: delta_pct_min={core_config['delta_pct_min']}, "
          f"delta_min={core_config['delta_min_contracts']}")
    print(f"  Press Ctrl+C to stop\n")

    last_mtime = {}

    while True:
        import glob
        files = sorted(glob.glob(os.path.join(watch_dir, '*.data')))

        for fp in files:
            mtime = os.path.getmtime(fp)
            if fp in last_mtime and mtime == last_mtime[fp]:
                continue
            last_mtime[fp] = mtime

            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] File updated: {os.path.basename(fp)}")
            bars = compute_features_for_file(fp, window=window)
            if bars is None or len(bars) == 0:
                print("  No bars after feature computation")
                continue

            signal = generate_signal(
                bars, feature_cols, signal_model, vol_model, quality_model,
                rr_ratio=rr_ratio, signal_threshold=signal_threshold,
                quality_threshold=quality_threshold, core_config=core_config,
            )

            if signal['signal'] == 'NO_TRADE':
                print(f"  → NO TRADE: {signal.get('reason', '')}")
            else:
                print(f"  ★ SIGNAL: {signal['signal']}")
                print(f"    Entry: {signal['entry_price']:.2f}")
                print(f"    SL:    {signal['sl_price']:.2f} ({signal['sl_points']:.2f} pts)")
                print(f"    TP:    {signal['tp_price']:.2f} ({signal['tp_points']:.2f} pts)")
                print(f"    RR:    {signal['rr_ratio']:.1f}")
                print(f"    Conf:  {signal['confidence']:.3f}")

        time.sleep(5)


def single_file_mode(models_dir, filepath, window, rr_ratio, signal_threshold, quality_threshold, tail_n):
    """Process a single file and show recent signals."""
    signal_model, vol_model, quality_model, meta = load_models(models_dir)
    feature_cols = meta['feature_columns']
    core_config = meta.get('core_feature_config', DEFAULT_CORE_CONFIG)

    print(f"Processing: {filepath}")
    bars = compute_features_for_file(filepath, window=window)
    if bars is None or len(bars) == 0:
        print("No bars.")
        return

    print(f"Bars: {len(bars)} | Last: {bars.index[-1]}")
    print(f"\nLast {tail_n} signals:")
    print("-" * 70)

    signals = []
    for i in range(max(0, len(bars) - tail_n), len(bars)):
        bar_slice = bars.iloc[:i+1]
        if len(bar_slice) < 30:
            continue
        sig = generate_signal(
            bar_slice, feature_cols, signal_model, vol_model, quality_model,
            rr_ratio=rr_ratio, signal_threshold=signal_threshold,
            quality_threshold=quality_threshold, core_config=core_config,
        )
        sig['bar_time'] = str(bars.index[i])
        signals.append(sig)

    for sig in signals:
        if sig['signal'] == 'NO_TRADE':
            print(f"  {sig['bar_time']}  —  (no trade)")
        else:
            pref = '▲' if sig['signal'] == 'LONG' else '▼'
            print(f"  {sig['bar_time']}  {pref} {sig['signal']}  "
                  f"@ {sig['entry_price']:.2f}  "
                  f"SL={sig['sl_price']:.2f}  TP={sig['tp_price']:.2f}  "
                  f"conf={sig['confidence']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Live signal generator for ES mini.')
    parser.add_argument('models_dir', help='Directory with trained models')
    parser.add_argument('--watch', default=None, help='Directory to watch for live data updates')
    parser.add_argument('--file', default=None, help='Single .data file to process')
    parser.add_argument('--window', default='1min', help='Bar window (default: 1min)')
    parser.add_argument('--rr', type=float, default=1.5, help='Risk-reward ratio')
    parser.add_argument('--signal-threshold', type=float, default=0.55)
    parser.add_argument('--quality-threshold', type=float, default=0.50)
    parser.add_argument('--tail', type=int, default=30, help='Number of recent bars to show signals for')

    args = parser.parse_args()

    if args.watch:
        watch_mode(args.models_dir, args.watch, args.window, args.rr,
                   args.signal_threshold, args.quality_threshold)
    elif args.file:
        single_file_mode(args.models_dir, args.file, args.window, args.rr,
                         args.signal_threshold, args.quality_threshold, args.tail)
    else:
        print("Specify --watch <dir> or --file <path>", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
