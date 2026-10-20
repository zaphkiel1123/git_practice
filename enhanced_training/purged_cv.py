#!/usr/bin/env python3
"""
Purged cross-validation, conditional meta-labels, and a multiple-testing check.

THREE PROBLEMS THIS FIXES
-------------------------
1. OVERLAPPING LABELS. es_training labels every bar with a 60-bar forward
   horizon, so bar t and bar t+1 share 59 of 60 future bars.
   walk_forward_split() in train_trading_model.py is a plain expanding window
   with no purge, so the tail of every training set overlaps the head of its
   test set. Scores come out biased upward, and any confidence interval derived
   from the raw sample count is off by roughly sqrt(60).

2. THE WRONG META-LABEL. train_trading_model.py builds

       quality_label = (long_result == 1) | (short_result == 1)

   which asks "did *either* side win at this bar?". That is a property of the
   bar alone and is independent of what any strategy actually signalled. It
   cannot answer "will this strategy's signal work here", which is the question
   behind strategy selection. make_meta_labels() below conditions on the
   signal's own direction, which is the whole point.

3. WIN RATES THAT CONDITION ON THE FUTURE. Both the fold metrics and the
   pattern miner compute win rate over `trade_mask = y != 0`, i.e. only bars
   where some side won. Bars where neither side won are dropped from the
   denominator -- but in live trading you cannot skip those, you get stopped
   out on them. Every function here keeps them in.

WHY META-LABELING ANSWERS "WHICH STRATEGY WHEN"
-----------------------------------------------
Do not try to pick a strategy in advance. Let every strategy signal, then train
one secondary model per strategy that predicts P(this signal wins | context).
At decision time you run them all and take whichever cleared its probability
threshold. Selection becomes a calibrated number instead of a hand-built regime
taxonomy. The secondary problem is also much easier than direction prediction:
binary rather than three-class, and the primary rule has already narrowed the
sample to plausible setups.

Usage:
    python3 purged_cv.py --features retest.parquet --setup weak_retest_short
    python3 purged_cv.py --features retest.parquet --setup all --n-splits 6
"""

import argparse

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================
# Conditional meta-labels (triple barrier)
# ============================================================

def make_meta_labels(bars, signal_idx, direction, sl_atr=1.0, tp_atr=1.5,
                     max_hold=60):
    """
    Path-dependent outcome for each signal, conditional on its own direction.

    For every signalled bar the trade is simulated forward within the same
    session: stop and target are placed at sl_atr / tp_atr multiples of ATR at
    entry, then bars are scanned in order. When a bar's range spans both
    barriers the stop is assumed hit first, which is the conservative choice.

    Returns a DataFrame indexed by the signal bar positions with:
        label        1 if the target was reached first, else 0
        exit_reason  'tp' | 'sl' | 'timeout' | 'session_end'
        label_end    positional index where the trade resolved -- REQUIRED for
                     purging, since a trade that resolves in 3 bars leaks far
                     less than one that runs the full horizon
        bars_held, entry_price, sl_price, tp_price
    """
    close = bars['close'].to_numpy(dtype=np.float64)
    high = bars['high'].to_numpy(dtype=np.float64)
    low = bars['low'].to_numpy(dtype=np.float64)
    atr = bars['atr_14'].to_numpy(dtype=np.float64)
    sessions = bars['session_date'].to_numpy()

    rows = []
    n = len(bars)

    for t in signal_idx:
        if not np.isfinite(atr[t]) or atr[t] <= 0:
            continue

        entry = close[t]
        risk = sl_atr * atr[t]
        reward = tp_atr * atr[t]

        if direction > 0:
            sl_price, tp_price = entry - risk, entry + reward
        else:
            sl_price, tp_price = entry + risk, entry - reward

        label = 0
        reason = 'timeout'
        end = min(t + max_hold, n - 1)

        for j in range(t + 1, min(t + 1 + max_hold, n)):
            if sessions[j] != sessions[t]:
                label, reason, end = 0, 'session_end', j - 1
                break

            if direction > 0:
                hit_sl = low[j] <= sl_price
                hit_tp = high[j] >= tp_price
            else:
                hit_sl = high[j] >= sl_price
                hit_tp = low[j] <= tp_price

            if hit_sl:
                label, reason, end = 0, 'sl', j
                break
            if hit_tp:
                label, reason, end = 1, 'tp', j
                break
        else:
            end = min(t + max_hold, n - 1)

        rows.append({
            'signal_idx': t,
            'label': label,
            'exit_reason': reason,
            'label_end': end,
            'bars_held': end - t,
            'entry_price': entry,
            'sl_price': sl_price,
            'tp_price': tp_price,
        })

    if not rows:
        return pd.DataFrame(columns=['signal_idx', 'label', 'exit_reason',
                                     'label_end', 'bars_held', 'entry_price',
                                     'sl_price', 'tp_price'])
    return pd.DataFrame(rows).set_index('signal_idx')


# ============================================================
# Purged / embargoed splits
# ============================================================

def purged_kfold(sample_idx, label_end, n_splits=5, embargo_pct=0.01):
    """
    Purged K-fold with embargo (Lopez de Prado, Advances in Financial ML ch7).

    sample_idx: positional index of each sample, ascending.
    label_end:  positional index at which each sample's label resolves.

    A training sample is dropped when its [start, end] label window overlaps
    the test window, and additionally when it starts inside the embargo period
    immediately after the test block.

    Yields (train_positions, test_positions) as offsets into sample_idx.
    """
    sample_idx = np.asarray(sample_idx)
    label_end = np.asarray(label_end)
    n = len(sample_idx)
    if n < n_splits * 2:
        raise ValueError(f"Too few samples ({n}) for {n_splits} splits")

    embargo = int(n * embargo_pct)
    bounds = np.linspace(0, n, n_splits + 1).astype(int)

    for k in range(n_splits):
        test_pos = np.arange(bounds[k], bounds[k + 1])
        if len(test_pos) == 0:
            continue

        test_start = sample_idx[test_pos[0]]
        test_end = label_end[test_pos].max()

        train_pos = []
        for p in range(n):
            if bounds[k] <= p < bounds[k + 1]:
                continue
            start_p = sample_idx[p]
            end_p = label_end[p]

            # Purge: any overlap with the test window.
            if end_p >= test_start and start_p <= test_end:
                continue
            # Embargo: samples starting just after the test block.
            if 0 < start_p - test_end <= embargo:
                continue
            train_pos.append(p)

        if len(train_pos) > 50:
            yield np.array(train_pos), test_pos


def purged_walk_forward(sample_idx, label_end, n_splits=5, min_train_frac=0.4,
                        embargo_pct=0.01):
    """
    Expanding-window walk-forward with purging and embargo.

    Closer to live conditions than K-fold because training data always precedes
    the test block. This is the drop-in replacement for walk_forward_split() in
    train_trading_model.py.
    """
    sample_idx = np.asarray(sample_idx)
    label_end = np.asarray(label_end)
    n = len(sample_idx)

    start = int(n * min_train_frac)
    test_size = max(1, (n - start) // n_splits)
    embargo = int(n * embargo_pct)

    for k in range(n_splits):
        t0 = start + k * test_size
        t1 = min(t0 + test_size, n)
        if t1 - t0 < 20:
            break

        test_pos = np.arange(t0, t1)
        test_start = sample_idx[t0]

        # Only prior samples, and only those whose label resolved before the
        # test window opens (plus embargo).
        train_pos = np.array([
            p for p in range(t0)
            if label_end[p] < test_start - embargo
        ])

        if len(train_pos) > 50:
            yield train_pos, test_pos


def effective_sample_size(label_end, sample_idx):
    """
    Report how much independent information the labels actually contain.

    Average label lifespan in bars is the redundancy factor: with a 60-bar
    horizon sampled every bar it is close to 60, meaning 7,700 rows carry about
    128 independent observations.
    """
    sample_idx = np.asarray(sample_idx)
    label_end = np.asarray(label_end)
    n = len(sample_idx)
    if n == 0:
        return {}

    span = label_end - sample_idx + 1
    total_bars = sample_idx.max() - sample_idx.min() + 1

    # Concurrency: how many labels are live at each bar.
    live = np.zeros(total_bars + 1, dtype=np.int64)
    for s, e in zip(sample_idx - sample_idx.min(), label_end - sample_idx.min()):
        live[max(0, s):min(total_bars, e) + 1] += 1
    mean_conc = float(live[live > 0].mean()) if (live > 0).any() else 1.0

    return {
        'n_samples': n,
        'mean_label_span_bars': float(span.mean()),
        'mean_concurrency': mean_conc,
        'effective_n': float(n / max(mean_conc, 1.0)),
    }


# ============================================================
# Multiple testing
# ============================================================

def multiple_testing_report(observed_t, n_trials):
    """
    How impressive is a t-statistic once the search effort is accounted for?

    The maximum of n_trials draws from a standard normal is approximately
    sqrt(2*ln(n_trials)). A result that does not clear that threshold is not
    distinguishable from the best of a pile of noise.
    """
    n_trials = max(int(n_trials), 1)
    expected_max = float(np.sqrt(2 * np.log(n_trials))) if n_trials > 1 else 0.0

    print(f"\n{'='*72}")
    print("  MULTIPLE-TESTING CHECK")
    print(f"{'='*72}")
    print(f"  Observed t-statistic:            {observed_t:.2f}")
    print(f"  Configurations tried:            {n_trials}")
    print(f"  Expected max t from pure noise:  {expected_max:.2f}")
    if observed_t > expected_max:
        print(f"\n  Clears the noise threshold by {observed_t - expected_max:.2f}.")
    else:
        print(f"\n  Does NOT clear the noise threshold. A search this wide would")
        print("  be expected to turn up a result this good even if no edge")
        print("  existed. Note that the 3-condition pattern miner in")
        print("  train_trading_model.py evaluates thousands of combinations.")
    print(f"{'='*72}")
    return expected_max


# ============================================================
# Meta-model evaluation
# ============================================================

def evaluate_meta_model(bars, feature_cols, setup_name, sl_atr=1.0, tp_atr=1.5,
                        max_hold=60, n_splits=5, embargo_pct=0.01,
                        method='walk_forward'):
    """
    Train and evaluate a per-setup meta-model under purged validation.

    The number that matters is precision at the operating threshold measured
    against the base rate, with every signalled bar in the denominator.
    """
    from event_study import SETUPS

    scope = bars[bars['is_rth']].reset_index(drop=False)
    mask, direction = SETUPS[setup_name](scope)
    mask = mask.fillna(False).to_numpy(dtype=bool)
    signal_idx = np.flatnonzero(mask)

    print(f"\n{'='*72}")
    print(f"  META-MODEL: {setup_name}  "
          f"direction={'LONG' if direction > 0 else 'SHORT'}")
    print(f"{'='*72}")
    print(f"  Signals: {len(signal_idx):,}")

    if len(signal_idx) < 200:
        print("  Too few signals to train a meta-model (need ~200+).")
        return None

    meta = make_meta_labels(scope, signal_idx, direction, sl_atr=sl_atr,
                            tp_atr=tp_atr, max_hold=max_hold)
    if len(meta) < 200:
        print("  Too few resolvable signals.")
        return None

    base_rate = float(meta['label'].mean())
    breakeven = 1.0 / (1.0 + tp_atr / sl_atr)

    ess = effective_sample_size(meta['label_end'].to_numpy(), meta.index.to_numpy())
    print(f"  Resolvable signals:      {len(meta):,}")
    print(f"  Base rate (win share):   {base_rate:.3f}")
    print(f"  Breakeven at {tp_atr/sl_atr:.2f}R:        {breakeven:.3f}")
    print(f"  Exit mix:                {meta['exit_reason'].value_counts().to_dict()}")
    print(f"  Mean label span:         {ess['mean_label_span_bars']:.1f} bars")
    print(f"  Mean concurrency:        {ess['mean_concurrency']:.1f}")
    print(f"  Effective sample size:   {ess['effective_n']:.0f} "
          f"(vs {ess['n_samples']} rows)")

    if base_rate > breakeven:
        print(f"\n  The raw setup already clears breakeven by "
              f"{(base_rate - breakeven)*100:.1f} points before any filtering.")
    else:
        print(f"\n  The raw setup is {(breakeven - base_rate)*100:.1f} points below")
        print("  breakeven, so the meta-model has to add real discrimination")
        print("  for this to be tradeable at all.")

    if not HAS_SKLEARN:
        print("\n  scikit-learn not available; skipping model fit.")
        return {'base_rate': base_rate, 'breakeven': breakeven, **ess}

    usable = [c for c in feature_cols if c in scope.columns]
    X_all = scope.loc[meta.index, usable].to_numpy(dtype=np.float64)
    y_all = meta['label'].to_numpy(dtype=np.int64)

    splitter = (purged_walk_forward if method == 'walk_forward' else purged_kfold)
    kwargs = ({'min_train_frac': 0.4} if method == 'walk_forward' else {})

    fold_rows = []
    for fold, (tr, te) in enumerate(
        splitter(meta.index.to_numpy(), meta['label_end'].to_numpy(),
                 n_splits=n_splits, embargo_pct=embargo_pct, **kwargs), 1
    ):
        if len(np.unique(y_all[tr])) < 2:
            continue

        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05,
            l2_regularization=1.0, random_state=0,
        )
        model.fit(X_all[tr], y_all[tr])
        prob = model.predict_proba(X_all[te])[:, 1]

        auc = (roc_auc_score(y_all[te], prob)
               if len(np.unique(y_all[te])) > 1 else np.nan)

        # Precision at a threshold, with every signal kept in the denominator.
        taken = prob >= 0.55
        prec = float(y_all[te][taken].mean()) if taken.any() else np.nan

        fold_rows.append({
            'fold': fold,
            'n_train': len(tr),
            'n_test': len(te),
            'test_base_rate': float(y_all[te].mean()),
            'auc': auc,
            'n_taken': int(taken.sum()),
            'precision_at_055': prec,
            'lift': (prec - float(y_all[te].mean())
                     if np.isfinite(prec) else np.nan),
        })

    if not fold_rows:
        print("\n  No valid folds after purging. Widen the date range.")
        return None

    folds = pd.DataFrame(fold_rows)
    print(f"\n  Purged {method.replace('_', '-')} results:")
    print(folds.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    mean_auc = folds['auc'].mean()
    mean_lift = folds['lift'].mean()
    print(f"\n  Mean AUC:  {mean_auc:.3f}  (0.5 = no discrimination)")
    print(f"  Mean lift: {mean_lift:+.3f} over each fold's own base rate")

    if np.isfinite(mean_lift) and folds['lift'].notna().sum() > 1:
        lift_std = folds['lift'].std(ddof=1)
        if lift_std > 0:
            t_stat = mean_lift / (lift_std / np.sqrt(folds['lift'].notna().sum()))
            print(f"  Lift t-stat across folds: {t_stat:.2f}")

    print(f"{'='*72}")
    return {'folds': folds, 'base_rate': base_rate, 'breakeven': breakeven, **ess}


DEFAULT_META_FEATURES = [
    # Level-relative context -- the reason this model can discriminate at all.
    'retest_vol_ratio', 'retest_delta_ratio', 'retest_delta_flip',
    'absorption_at_level', 'excursion_beyond_ratio', 'n_prior_visits',
    'bars_since_prior_visit', 'bars_since_first_touch',
    'prior_visit_volume', 'prior_visit_delta_pct',
    'current_visit_volume', 'current_visit_delta_pct',
    'retest_big_prints_now', 'retest_big_print_same_side',
    'retest_level_dist_ticks', 'level_total_delta',
    # Regime context -- lets the model learn the conditioning itself instead of
    # you hand-building a regime taxonomy.
    'atr_percentile', 'atr_ratio', 'range_vs_atr', 'realized_vol_20',
    'mins_from_rth_open', 'delta_pct', 'cvd_slope_5',
    'close_vs_poc', 'close_vs_vah', 'close_vs_val', 'close_in_value_area',
]


def main():
    parser = argparse.ArgumentParser(
        description='Purged CV and conditional meta-labeling.')
    parser.add_argument('--features', required=True,
                        help='Parquet/CSV from level_ledger.py')
    parser.add_argument('--setup', default='all',
                        help="Setup name, comma-separated list, or 'all'")
    parser.add_argument('--sl-atr', type=float, default=1.0,
                        help='Stop distance in ATR (default: 1.0)')
    parser.add_argument('--tp-atr', type=float, default=1.5,
                        help='Target distance in ATR (default: 1.5)')
    parser.add_argument('--max-hold', type=int, default=60,
                        help='Max bars held (default: 60)')
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--embargo-pct', type=float, default=0.01)
    parser.add_argument('--method', default='walk_forward',
                        choices=['walk_forward', 'kfold'])
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Configurations tried so far, for the noise check')

    args = parser.parse_args()

    from event_study import SETUPS

    if args.features.endswith('.parquet'):
        bars = pd.read_parquet(args.features)
    else:
        bars = pd.read_csv(args.features, index_col=0, parse_dates=True)
    print(f"Loaded {len(bars):,} bars from {args.features}")

    names = sorted(SETUPS) if args.setup == 'all' else args.setup.split(',')

    t_stats = []
    for name in names:
        res = evaluate_meta_model(
            bars, DEFAULT_META_FEATURES, name.strip(),
            sl_atr=args.sl_atr, tp_atr=args.tp_atr, max_hold=args.max_hold,
            n_splits=args.n_splits, embargo_pct=args.embargo_pct,
            method=args.method,
        )
        if res and 'folds' in res:
            folds = res['folds']
            if folds['lift'].notna().sum() > 1:
                sd = folds['lift'].std(ddof=1)
                if sd > 0:
                    t_stats.append(
                        folds['lift'].mean() / (sd / np.sqrt(folds['lift'].notna().sum())))

    if t_stats:
        n_trials = args.n_trials or len(names)
        multiple_testing_report(max(t_stats), n_trials)


if __name__ == '__main__':
    main()
