#!/usr/bin/env python3
"""
Event-driven backtester for the ES mini trading system.

Processes bars sequentially (no look-ahead), applies entry/exit rules
with SL/TP management, enforces 1.5R minimum, RTH only, flat by 15:30 ET.

Usage:
    python3 backtester.py /path/to/models/ --data /path/to/data/ --workers 4
    python3 backtester.py /path/to/models/ --features training_features.parquet
	
	# More selective (fewer but higher-quality trades)
	python3 backtester.py ./data/models --signal-threshold 0.45
	# Less selective (more trades)
	python3 backtester.py ./data/models --signal-threshold 0.35
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

from feature_pipeline import (
    add_rolling_features, add_time_features, add_value_area_features,
    load_files_to_bars,
)
from train_trading_model import add_microstructure_features, add_multi_timeframe_features
from labels import TICK_SIZE, is_rth, compute_atr
from core_features import check_core_alignment, DEFAULT_CORE_CONFIG

POINT_VALUE = 50.0  # $50 per point per ES contract
SLIPPAGE_TICKS = 1  # 1 tick slippage per side
COMMISSION = 2.50   # per side per contract


class Trade:
    def __init__(self, entry_time, entry_price, direction, sl_price, tp_price, bar_idx, entry_reason='', pattern=''):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction  # +1 long, -1 short
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.bar_idx = bar_idx
        self.entry_reason = entry_reason
        self.pattern = pattern
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.bars_held = 0
        self.pnl_points = 0.0
        self.pnl_dollars = 0.0
        self.mae = 0.0
        self.mfe = 0.0


class Backtester:
    def __init__(self, signal_model, vol_model, quality_model, feature_cols,
                 rr_ratio=1.5, signal_threshold=0.40, quality_threshold=0.50,
                 max_hold_bars=60, sl_multiplier=1.5, min_sl=2.0, max_sl=8.0,
                 core_config=None):
        self.signal_model = signal_model
        self.vol_model = vol_model
        self.quality_model = quality_model
        self.feature_cols = feature_cols
        self.rr_ratio = rr_ratio
        self.signal_threshold = signal_threshold
        self.quality_threshold = quality_threshold
        self.max_hold_bars = max_hold_bars
        self.sl_multiplier = sl_multiplier
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.core_config = core_config if core_config is not None else DEFAULT_CORE_CONFIG
        self.patterns = []

    def load_patterns(self, patterns_path):
        """Load trading patterns from JSON for trade tagging."""
        if os.path.isfile(patterns_path):
            with open(patterns_path) as f:
                data = json.load(f)
            self.patterns = data.get('patterns', [])
            print(f"  Loaded {len(self.patterns)} trading patterns")

    def _match_pattern(self, X, direction):
        """Find the best matching pattern for this trade entry."""
        dir_label = 'LONG' if direction == 1 else 'SHORT'
        feature_vals = {name: X[0, i] for i, name in enumerate(self.feature_cols)}

        best = None
        best_wr = 0
        for p in self.patterns:
            if p['direction'] != dir_label:
                continue
            rules = p.get('rules', [])
            if not rules:
                continue
            match = True
            for rule in rules:
                feat = rule.get('feature', '')
                op = rule.get('op', '')
                thresh = rule.get('threshold', 0)
                val = feature_vals.get(feat)
                if val is None:
                    match = False
                    break
                if op == '>' and not (val > thresh):
                    match = False
                elif op == '<' and not (val < thresh):
                    match = False
                elif op == '>=' and not (val >= thresh):
                    match = False
                elif op == '<=' and not (val <= thresh):
                    match = False
                elif op == '==' and not (val == thresh):
                    match = False
            if match and p['win_rate'] > best_wr:
                best = p
                best_wr = p['win_rate']

        if best:
            return f"P{self.patterns.index(best)+1}_{best['win_rate']:.0%}"
        return ''

    def _get_entry_reason(self, X, direction):
        """Explain why this trade was taken using per-prediction feature contributions."""
        dir_label = 'long' if direction == 1 else 'short'

        # LightGBM: get per-sample leaf SHAP contributions
        if hasattr(self.signal_model, 'booster_'):
            # pred_contrib returns shape (n_samples, n_features+1, n_classes)
            # Last column is bias. Classes: 0=short, 1=no_trade, 2=long
            raw = self.signal_model.booster_.predict(X, pred_contrib=True)
            if raw.ndim == 2:
                # Binary or flattened: reshape to (1, n_features+1, n_classes)
                n_classes = 3
                raw = raw.reshape(1, -1, n_classes)
            class_idx = 2 if direction == 1 else 0  # long=2, short=0
            contribs = raw[0, :-1, class_idx]  # exclude bias term
            top_idx = int(np.argmax(np.abs(contribs)))
            feat_name = self.feature_cols[top_idx]
            contrib_val = contribs[top_idx]
            feat_val = X[0, top_idx]
            return self._format_reason(feat_name, feat_val, contrib_val, dir_label)

        # Fallback for sklearn: use importance * z-score direction
        imp = self.signal_model.feature_importances_
        vals = X[0]
        scores = imp * np.abs(vals)
        top_idx = int(np.argmax(scores))
        feat_name = self.feature_cols[top_idx]
        feat_val = vals[top_idx]
        return f"{feat_name}={feat_val:.1f} → {dir_label}"

    def _format_reason(self, feat_name, feat_val, contrib_val, dir_label):
        """Format a human-readable entry reason from feature contribution."""
        # Determine if the feature is pushing toward or confirming the direction
        strength = 'strongly' if abs(contrib_val) > 0.5 else ''

        if 'cvd' in feat_name:
            if feat_val > 0 and dir_label == 'short':
                ctx = f"CVD elevated ({feat_val:.0f}), buyers exhausted"
            elif feat_val < 0 and dir_label == 'long':
                ctx = f"CVD negative ({feat_val:.0f}), sellers exhausted"
            elif feat_val > 0 and dir_label == 'long':
                ctx = f"CVD rising ({feat_val:.0f}), momentum continuation"
            else:
                ctx = f"CVD falling ({feat_val:.0f}), momentum continuation"
        elif 'imbalance' in feat_name:
            if 'cluster' in feat_name:
                ctx = f"imbalance cluster ({feat_val:.0f} levels stacked)"
            else:
                ctx = f"order imbalance {feat_val:+.0f}"
        elif 'delta' in feat_name:
            ctx = f"delta {'positive' if feat_val > 0 else 'negative'} ({feat_val:.0f})"
        elif 'absorption' in feat_name:
            ctx = f"volume absorption ({feat_val:.1f}x normal)"
        elif 'vol_sum' in feat_name or 'volume' in feat_name:
            ctx = f"volume {'surge' if feat_val > 1.5 else 'elevated'} ({feat_val:.0f})"
        elif 'intensity' in feat_name:
            ctx = f"trade speed {'accelerating' if feat_val > 0 else 'decelerating'}"
        elif 'pressure' in feat_name:
            side = 'buy' if feat_val > 1 else 'sell'
            ctx = f"{side} pressure dominant ({feat_val:.2f})"
        elif 'flow' in feat_name:
            ctx = f"flow {'bullish' if feat_val > 0 else 'bearish'} ({feat_val:+.3f})"
        else:
            ctx = f"{feat_name}={feat_val:.2f}"

        return f"{ctx} → {strength} {dir_label}".strip()

    def run(self, bars):
        """Run backtest on prepared feature DataFrame."""
        trades = []
        position = None
        equity = 0.0
        equity_curve = []

        feature_data = bars[self.feature_cols].values
        timestamps = bars.index
        rth_mask = bars['is_rth'].values.astype(bool) if 'is_rth' in bars.columns else is_rth(timestamps).values

        # Detect session end (15:25 ET)
        if timestamps.tz is None:
            ts_et = timestamps.tz_localize('America/Chicago').tz_convert('America/New_York')
        else:
            ts_et = timestamps.tz_convert('America/New_York')
        session_end_mask = (ts_et.hour == 15) & (ts_et.minute >= 25)

        for i in range(len(bars)):
            bar = bars.iloc[i]
            current_time = timestamps[i]

            # Manage existing position
            if position is not None:
                position.bars_held += 1
                high = bar['high']
                low = bar['low']

                if position.direction == 1:
                    position.mfe = max(position.mfe, high - position.entry_price)
                    position.mae = max(position.mae, position.entry_price - low)
                else:
                    position.mfe = max(position.mfe, position.entry_price - low)
                    position.mae = max(position.mae, high - position.entry_price)

                exit_price = None
                exit_reason = None

                # Check SL (checked before TP — conservative)
                if position.direction == 1 and low <= position.sl_price:
                    exit_price = position.sl_price
                    exit_reason = 'SL'
                elif position.direction == -1 and high >= position.sl_price:
                    exit_price = position.sl_price
                    exit_reason = 'SL'

                # Check TP
                if exit_reason is None:
                    if position.direction == 1 and high >= position.tp_price:
                        exit_price = position.tp_price
                        exit_reason = 'TP'
                    elif position.direction == -1 and low <= position.tp_price:
                        exit_price = position.tp_price
                        exit_reason = 'TP'

                # Force close at session end or max hold
                if exit_reason is None:
                    if session_end_mask[i] or position.bars_held >= self.max_hold_bars:
                        exit_price = bar['close']
                        exit_reason = 'EOD' if session_end_mask[i] else 'TIMEOUT'

                if exit_price is not None:
                    # Apply slippage on exit
                    slippage = SLIPPAGE_TICKS * TICK_SIZE
                    if position.direction == 1:
                        exit_price -= slippage
                    else:
                        exit_price += slippage

                    position.exit_time = current_time
                    position.exit_price = exit_price
                    position.exit_reason = exit_reason
                    position.pnl_points = (exit_price - position.entry_price) * position.direction
                    position.pnl_dollars = position.pnl_points * POINT_VALUE - 2 * COMMISSION
                    trades.append(position)
                    equity += position.pnl_dollars
                    position = None

            # Entry logic (only when flat and in RTH)
            if position is None and rth_mask[i] and not session_end_mask[i]:
                X = feature_data[i:i+1]

                # Get signal model prediction
                if hasattr(self.signal_model, 'predict_proba'):
                    probs = self.signal_model.predict_proba(X)[0]
                    # probs: [P(short), P(no_trade), P(long)] for mapped classes [0,1,2]
                    prob_long = probs[2] if len(probs) == 3 else probs[1]
                    prob_short = probs[0]
                    prob_no_trade = probs[1] if len(probs) == 3 else 0.0
                    pred_direction = 0

                    # Pick the direction with highest probability, but only if it
                    # exceeds the threshold AND beats the no-trade probability
                    if prob_long > self.signal_threshold and prob_long > prob_short and prob_long > prob_no_trade:
                        pred_direction = 1
                    elif prob_short > self.signal_threshold and prob_short > prob_long and prob_short > prob_no_trade:
                        pred_direction = -1
                else:
                    pred = self.signal_model.predict(X)[0]
                    pred_direction = int(pred) - 1

                if pred_direction == 0:
                    equity_curve.append({'time': current_time, 'equity': equity})
                    continue

                # Core feature alignment gate
                feature_dict = dict(zip(self.feature_cols, X[0]))
                aligned, _ = check_core_alignment(feature_dict, pred_direction, self.core_config)
                if not aligned:
                    equity_curve.append({'time': current_time, 'equity': equity})
                    continue

                # Quality filter
                if self.quality_model is not None:
                    if hasattr(self.quality_model, 'predict_proba'):
                        q_prob = self.quality_model.predict_proba(X)[0][1]
                    else:
                        q_prob = self.quality_model.predict(X)[0]
                    if q_prob < self.quality_threshold:
                        equity_curve.append({'time': current_time, 'equity': equity})
                        continue

                # Compute SL from volatility model
                if self.vol_model is not None:
                    if hasattr(self.vol_model, 'predict'):
                        pred_atr = self.vol_model.predict(X)[0]
                    else:
                        pred_atr = float(self.vol_model.predict(X))
                    sl_points = np.clip(pred_atr * self.sl_multiplier, self.min_sl, self.max_sl)
                else:
                    sl_points = bar.get('atr_14', 3.0) * self.sl_multiplier
                    sl_points = np.clip(sl_points, self.min_sl, self.max_sl)

                # Round SL to tick
                sl_points = round(sl_points / TICK_SIZE) * TICK_SIZE
                tp_points = sl_points * self.rr_ratio

                # Entry price with slippage
                entry_price = bar['close']
                slippage = SLIPPAGE_TICKS * TICK_SIZE
                if pred_direction == 1:
                    entry_price += slippage
                    sl_price = entry_price - sl_points
                    tp_price = entry_price + tp_points
                else:
                    entry_price -= slippage
                    sl_price = entry_price + sl_points
                    tp_price = entry_price - tp_points

                position = Trade(
                    entry_time=current_time,
                    entry_price=entry_price,
                    direction=pred_direction,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    bar_idx=i,
                    entry_reason=self._get_entry_reason(X, pred_direction),
                    pattern=self._match_pattern(X, pred_direction),
                )

            equity_curve.append({'time': current_time, 'equity': equity})

        # Close any remaining position at last bar
        if position is not None:
            bar = bars.iloc[-1]
            position.exit_time = timestamps[-1]
            position.exit_price = bar['close']
            position.exit_reason = 'END'
            position.pnl_points = (position.exit_price - position.entry_price) * position.direction
            position.pnl_dollars = position.pnl_points * POINT_VALUE - 2 * COMMISSION
            trades.append(position)
            equity += position.pnl_dollars

        return trades, pd.DataFrame(equity_curve)


def compute_metrics(trades):
    """Compute performance metrics from trade list."""
    if not trades:
        return {
            'n_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
            'avg_winner_$': 0.0, 'avg_loser_$': 0.0, 'expectancy_$': 0.0,
            'total_pnl_$': 0.0, 'total_pnl_pts': 0.0, 'max_drawdown_$': 0.0,
            'sharpe_approx': 0.0, 'avg_bars_held': 0.0,
            'n_longs': 0, 'n_shorts': 0,
            'long_win_rate': 0.0, 'short_win_rate': 0.0,
            'exit_reasons': {},
        }

    pnls = np.array([t.pnl_dollars for t in trades])
    points = np.array([t.pnl_points for t in trades])
    winners = pnls > 0
    losers = pnls < 0

    n_trades = len(trades)
    win_rate = winners.sum() / n_trades if n_trades > 0 else 0

    avg_winner = pnls[winners].mean() if winners.any() else 0
    avg_loser = abs(pnls[losers].mean()) if losers.any() else 1
    profit_factor = (pnls[winners].sum() / abs(pnls[losers].sum())) if losers.any() and pnls[losers].sum() != 0 else float('inf')

    # Max drawdown
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl
    max_dd = drawdown.max()

    # Sharpe (daily approx: assume ~4 trades/day)
    if len(pnls) > 1:
        sharpe = (pnls.mean() / pnls.std()) * np.sqrt(252 * 4) if pnls.std() > 0 else 0
    else:
        sharpe = 0

    # Trade duration
    bars_held = np.array([t.bars_held for t in trades])

    # Direction breakdown
    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]
    long_wr = sum(1 for t in longs if t.pnl_dollars > 0) / len(longs) if longs else 0
    short_wr = sum(1 for t in shorts if t.pnl_dollars > 0) / len(shorts) if shorts else 0

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    metrics = {
        'n_trades': n_trades,
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'avg_winner_$': float(avg_winner),
        'avg_loser_$': float(avg_loser),
        'expectancy_$': float(pnls.mean()),
        'total_pnl_$': float(pnls.sum()),
        'total_pnl_pts': float(points.sum()),
        'max_drawdown_$': float(max_dd),
        'sharpe_approx': float(sharpe),
        'avg_bars_held': float(bars_held.mean()),
        'n_longs': len(longs),
        'n_shorts': len(shorts),
        'long_win_rate': float(long_wr),
        'short_win_rate': float(short_wr),
        'exit_reasons': reasons,
    }
    return metrics


def trades_to_dataframe(trades):
    """Convert trade list to DataFrame for analysis."""
    records = []
    for t in trades:
        records.append({
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'direction': 'LONG' if t.direction == 1 else 'SHORT',
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'sl_price': t.sl_price,
            'tp_price': t.tp_price,
            'exit_reason': t.exit_reason,
            'entry_reason': t.entry_reason,
            'pattern': t.pattern,
            'pnl_points': t.pnl_points,
            'pnl_dollars': t.pnl_dollars,
            'bars_held': t.bars_held,
            'mae': t.mae,
            'mfe': t.mfe,
        })
    return pd.DataFrame(records)


def print_report(metrics, trades_df=None):
    """Print formatted backtest report."""
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total trades:      {metrics['n_trades']}")
    print(f"  Win rate:          {metrics['win_rate']:.1%}")
    print(f"  Profit factor:     {metrics['profit_factor']:.2f}")
    print(f"  Avg winner:        ${metrics['avg_winner_$']:.2f}")
    print(f"  Avg loser:         ${metrics['avg_loser_$']:.2f}")
    print(f"  Expectancy:        ${metrics['expectancy_$']:.2f}/trade")
    print(f"  Total PnL:         ${metrics['total_pnl_$']:.2f} ({metrics['total_pnl_pts']:.2f} pts)")
    print(f"  Max drawdown:      ${metrics['max_drawdown_$']:.2f}")
    print(f"  Sharpe (approx):   {metrics['sharpe_approx']:.2f}")
    print(f"  Avg bars held:     {metrics['avg_bars_held']:.1f}")
    print(f"  Longs/Shorts:      {metrics['n_longs']}/{metrics['n_shorts']}")
    print(f"  Long WR / Short WR: {metrics['long_win_rate']:.1%} / {metrics['short_win_rate']:.1%}")
    print(f"  Exit reasons:      {metrics['exit_reasons']}")
    print(f"{'='*60}")


def prepare_backtest_features(data_dir, window='1min', workers=0):
    """Memory-efficient feature pipeline for backtesting (no label simulation)."""
    pipeline_start = time.time()

    bars = load_files_to_bars(data_dir, window=window, workers=workers)

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

    bars['is_rth'] = is_rth(bars.index)
    bars = bars.drop(columns=['_vol_profile'], errors='ignore')
    bars = bars.dropna()
    print(f"Complete bars after dropna: {len(bars)}")
    print(f"  Total pipeline [{time.time()-pipeline_start:.1f}s]")
    return bars


def main():
    parser = argparse.ArgumentParser(description='Backtest ES mini trading system.')
    parser.add_argument('models_dir', help='Directory with trained models')
    parser.add_argument('--data', default=None, help='Directory with .data files (re-compute features)')
    parser.add_argument('--features', default=None, help='Pre-computed features file (parquet/csv)')
    parser.add_argument('--rr', type=float, default=1.5, help='Risk-reward ratio')
    parser.add_argument('--signal-threshold', type=float, default=0.40, help='Signal confidence threshold (default: 0.40 for 3-class model)')
    parser.add_argument('--quality-threshold', type=float, default=0.50, help='Quality filter threshold')
    parser.add_argument('--max-hold', type=int, default=60, help='Max bars to hold')
    parser.add_argument('--workers', type=int, default=0,
                        help='File pipeline workers: 0=prefetch thread (default), 1=sequential, >1=parallel processes')
    parser.add_argument('--output', default=None, help='Output directory for results')

    args = parser.parse_args()

    # Load models
    print("Loading models...")
    signal_model = None
    vol_model = None
    quality_model = None

    signal_path = os.path.join(args.models_dir, 'signal_model.joblib')
    vol_path = os.path.join(args.models_dir, 'volatility_model.joblib')
    quality_path = os.path.join(args.models_dir, 'quality_model.joblib')

    if os.path.isfile(signal_path):
        signal_model = joblib.load(signal_path)
        print(f"  Loaded signal model")
    else:
        print(f"  ERROR: Signal model not found at {signal_path}", file=sys.stderr)
        sys.exit(1)

    if os.path.isfile(vol_path):
        vol_model = joblib.load(vol_path)
        print(f"  Loaded volatility model")

    if os.path.isfile(quality_path):
        quality_model = joblib.load(quality_path)
        print(f"  Loaded quality model")

    # Load metadata for feature columns
    meta_path = os.path.join(args.models_dir, 'training_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta['feature_columns']
    core_config = meta.get('core_feature_config', DEFAULT_CORE_CONFIG)

    # Load or compute features
    if args.features:
        print(f"Loading features from {args.features}...")
        if args.features.endswith('.parquet'):
            bars = pd.read_parquet(args.features)
        else:
            bars = pd.read_csv(args.features, index_col=0, parse_dates=True)
    elif args.data:
        bars = prepare_backtest_features(
            args.data, window=meta.get('window', '1min'), workers=args.workers,
        )
    else:
        # Default: look for training_features in models_dir
        fp = os.path.join(args.models_dir, 'training_features.parquet')
        if os.path.isfile(fp):
            bars = pd.read_parquet(fp)
        else:
            fp = os.path.join(args.models_dir, 'training_features.csv')
            bars = pd.read_csv(fp, index_col=0, parse_dates=True)

    print(f"Bars loaded: {len(bars)}")

    # Run backtest
    bt = Backtester(
        signal_model=signal_model,
        vol_model=vol_model,
        quality_model=quality_model,
        feature_cols=feature_cols,
        rr_ratio=args.rr,
        signal_threshold=args.signal_threshold,
        quality_threshold=args.quality_threshold,
        max_hold_bars=args.max_hold,
        core_config=core_config,
    )

    patterns_path = os.path.join(args.models_dir, 'trading_patterns.json')
    bt.load_patterns(patterns_path)

    print("Running backtest...")
    trades, equity_curve = bt.run(bars)

    metrics = compute_metrics(trades)
    print_report(metrics)

    # Save results
    output_dir = args.output or args.models_dir
    os.makedirs(output_dir, exist_ok=True)

    trades_df = trades_to_dataframe(trades)
    trades_path = os.path.join(output_dir, 'backtest_trades.csv')
    trades_df.to_csv(trades_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f"\nTrades saved to: {trades_path}")

    equity_path = os.path.join(output_dir, 'equity_curve.csv')
    equity_curve.to_csv(equity_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f"Equity curve saved to: {equity_path}")

    metrics_path = os.path.join(output_dir, 'backtest_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()
