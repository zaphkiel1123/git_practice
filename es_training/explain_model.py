#!/usr/bin/env python3
"""
Model interpretability via SHAP: explains what factors the model uses
to determine trade direction and strength.

Generates:
- Global feature importance ranking with direction of influence
- Per-trade explanation ("why was this trade taken?")
- Winning vs losing trade factor comparison
- Human-readable summary report

Usage:
    python3 explain_model.py /path/to/models/ --trades backtest_trades.csv
"""

import argparse
import os
import sys
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import pandas as pd
import joblib

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def _normalize_shap_values(shap_values):
    """Convert any SHAP output format to a consistent numpy array/list."""
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values
    return shap_values


def _mean_abs_importance(shap_values):
    """Compute 1D mean absolute SHAP importance per feature."""
    if isinstance(shap_values, list):
        return np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        return np.abs(shap_values).mean(axis=(0, 2))
    else:
        return np.abs(shap_values).mean(axis=0)


def _get_class_shap(shap_values, class_idx=-1):
    """Extract SHAP values for a specific class from multiclass output."""
    if isinstance(shap_values, list):
        return shap_values[class_idx]
    elif shap_values.ndim == 3:
        return shap_values[:, :, class_idx]
    else:
        return shap_values


def explain_global(model, X, feature_names, model_name='model', top_n=20):
    """Compute global SHAP feature importance."""
    if not HAS_SHAP:
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1][:top_n]
        return {
            'method': 'built-in',
            'features': [
                {
                    'rank': rank,
                    'name': feature_names[int(i)],
                    'importance': float(imp[i]),
                    'mean_abs_shap': float(imp[i]),
                    'effect': 'importance-based (SHAP unavailable)',
                }
                for rank, i in enumerate(idx, 1)
            ],
        }

    explainer = shap.TreeExplainer(model)

    sample_size = min(2000, len(X))
    if len(X) > sample_size:
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X

    shap_values = _normalize_shap_values(explainer.shap_values(X_sample))
    mean_abs = _mean_abs_importance(shap_values)
    idx = np.argsort(mean_abs)[::-1][:top_n]

    # Get SHAP values for the "long" class to determine direction of influence
    sv_long = _get_class_shap(shap_values, class_idx=-1)

    result = {
        'method': 'SHAP_TreeExplainer',
        'model_name': model_name,
        'sample_size': sample_size,
        'features': []
    }

    for rank, i in enumerate(idx, 1):
        i = int(i)
        feature_info = {
            'rank': rank,
            'name': feature_names[i],
            'mean_abs_shap': float(mean_abs[i]),
        }

        feature_vals = X_sample[:, i]
        shap_vals = sv_long[:, i]
        if len(feature_vals) > 10:
            corr = np.corrcoef(feature_vals, shap_vals)[0, 1]
            feature_info['direction_correlation'] = float(corr)
            if corr > 0.1:
                feature_info['effect'] = 'higher value → more bullish'
            elif corr < -0.1:
                feature_info['effect'] = 'higher value → more bearish'
            else:
                feature_info['effect'] = 'non-linear effect'

        result['features'].append(feature_info)

    return result


def explain_trades(model, X_trades, feature_names, trade_results, top_n=5):
    """Explain individual trades: what drove each decision."""
    if not HAS_SHAP:
        return {'method': 'unavailable', 'note': 'Install shap package for per-trade explanations'}

    explainer = shap.TreeExplainer(model)
    shap_values = _normalize_shap_values(explainer.shap_values(X_trades))

    sv_long = _get_class_shap(shap_values, class_idx=-1)
    sv_short = _get_class_shap(shap_values, class_idx=0)

    explanations = []
    for i in range(min(len(X_trades), 50)):
        direction = trade_results[i].get('direction', 'LONG') if isinstance(trade_results, list) else 'LONG'
        sv = sv_long[i] if direction == 'LONG' else sv_short[i]

        top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
        factors = []
        for j in top_idx:
            j = int(j)
            factors.append({
                'feature': feature_names[j],
                'shap_value': float(sv[j]),
                'feature_value': float(X_trades[i, j]),
                'contribution': 'bullish' if sv[j] > 0 else 'bearish',
            })

        explanations.append({
            'trade_idx': i,
            'direction': direction,
            'top_factors': factors,
        })

    return explanations


def compare_winners_losers(model, X_all, feature_names, pnl_array, top_n=15):
    """Compare SHAP profiles of winning vs losing trades."""
    if not HAS_SHAP:
        return {'method': 'unavailable'}

    winners_mask = pnl_array > 0
    losers_mask = pnl_array < 0

    if winners_mask.sum() < 10 or losers_mask.sum() < 10:
        return {'note': 'Insufficient trades for comparison'}

    explainer = shap.TreeExplainer(model)

    n_sample = min(500, int(winners_mask.sum()), int(losers_mask.sum()))
    win_idx = np.random.choice(np.where(winners_mask)[0], n_sample, replace=False)
    lose_idx = np.random.choice(np.where(losers_mask)[0], n_sample, replace=False)

    sv_win = _normalize_shap_values(explainer.shap_values(X_all[win_idx]))
    sv_lose = _normalize_shap_values(explainer.shap_values(X_all[lose_idx]))

    mean_win = _mean_abs_importance(sv_win)
    mean_lose = _mean_abs_importance(sv_lose)

    # Ensure 1D (should be after _mean_abs_importance, but guard against edge cases)
    mean_win = np.asarray(mean_win).ravel()
    mean_lose = np.asarray(mean_lose).ravel()

    diff = mean_win - mean_lose
    diff_idx = np.argsort(np.abs(diff))[::-1][:top_n]

    comparison = []
    for i in diff_idx:
        i = int(i)
        comparison.append({
            'feature': feature_names[i],
            'winner_importance': float(mean_win[i]),
            'loser_importance': float(mean_lose[i]),
            'difference': float(diff[i]),
            'more_important_for': 'winners' if diff[i] > 0 else 'losers',
        })

    return comparison


def generate_report(global_explanation, comparison=None):
    """Generate human-readable text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  MODEL FACTOR ANALYSIS REPORT")
    lines.append("  What the model looks at to determine direction and strength")
    lines.append("=" * 70)

    lines.append("\n  TOP FACTORS FOR TRADE DIRECTION:")
    lines.append("  " + "-" * 50)
    for f in global_explanation['features'][:15]:
        effect = f.get('effect', 'complex')
        lines.append(f"    {f['rank']:2d}. {f['name']:<30} (importance: {f['mean_abs_shap']:.4f})")
        lines.append(f"        → {effect}")

    if comparison and isinstance(comparison, list):
        lines.append("\n\n  FACTORS THAT DISTINGUISH WINNERS FROM LOSERS:")
        lines.append("  " + "-" * 50)
        for c in comparison[:10]:
            lines.append(f"    {c['feature']:<30} more important for {c['more_important_for']}")
            lines.append(f"        winners: {c['winner_importance']:.4f} | losers: {c['loser_importance']:.4f}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Explain trading model decisions via SHAP.')
    parser.add_argument('models_dir', help='Directory with trained models')
    parser.add_argument('--features', default=None, help='Features file (parquet/csv)')
    parser.add_argument('--trades', default=None, help='Backtest trades CSV')
    parser.add_argument('--output', default=None, help='Output directory')

    args = parser.parse_args()

    if not HAS_SHAP:
        print("WARNING: shap package not installed. Using built-in feature importance only.")
        print("  Install with: pip install shap")

    # Load model
    signal_path = os.path.join(args.models_dir, 'signal_model.joblib')
    quality_path = os.path.join(args.models_dir, 'quality_model.joblib')

    signal_model = joblib.load(signal_path) if os.path.isfile(signal_path) else None
    quality_model = joblib.load(quality_path) if os.path.isfile(quality_path) else None

    if signal_model is None:
        print("ERROR: No signal model found.", file=sys.stderr)
        sys.exit(1)

    # Load metadata
    meta_path = os.path.join(args.models_dir, 'training_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta['feature_columns']

    # Load features
    if args.features:
        fp = args.features
    else:
        fp = os.path.join(args.models_dir, 'training_features.parquet')
        if not os.path.isfile(fp):
            fp = fp.replace('.parquet', '.csv')

    if fp.endswith('.parquet'):
        bars = pd.read_parquet(fp)
    else:
        bars = pd.read_csv(fp, index_col=0, parse_dates=True)

    X = bars[feature_cols].values
    print(f"Loaded {len(X)} bars with {len(feature_cols)} features")

    # Global explanation
    print("\nComputing global feature importance (this may take a moment)...")
    global_expl = explain_global(signal_model, X, feature_cols, 'Entry Signal Model')

    # Load trades for per-trade and winner/loser analysis
    comparison = None
    trade_explanations = None

    if args.trades:
        trades_df = pd.read_csv(args.trades, parse_dates=['entry_time', 'exit_time'])
        print(f"Loaded {len(trades_df)} trades")

        # Match trades to feature bars
        if 'pnl_dollars' in trades_df.columns:
            pnl_arr = trades_df['pnl_dollars'].values
            # Get feature rows at entry times
            entry_times = pd.to_datetime(trades_df['entry_time'])
            matched_idx = []
            for et in entry_times:
                idx = bars.index.get_indexer([et], method='nearest')[0]
                matched_idx.append(idx)
            X_trades = X[matched_idx]

            print("Comparing winners vs losers...")
            comparison = compare_winners_losers(signal_model, X_trades, feature_cols, pnl_arr)

            print("Generating per-trade explanations...")
            trade_explanations = explain_trades(
                signal_model, X_trades[:50], feature_cols,
                trades_df.to_dict('records')[:50]
            )

    # Generate report
    report = generate_report(global_expl, comparison)
    print(report)

    # Save outputs
    output_dir = args.output or args.models_dir
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, 'factor_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    full_results = {
        'global_importance': global_expl,
        'winner_loser_comparison': comparison,
        'trade_explanations': trade_explanations,
    }
    json_path = os.path.join(output_dir, 'factor_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"Full analysis saved to: {json_path}")


if __name__ == '__main__':
    main()
