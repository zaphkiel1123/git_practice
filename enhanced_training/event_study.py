#!/usr/bin/env python3
"""
Conditional event study: does a setup carry information, before any backtest?

WHY NOT JUST BACKTEST
---------------------
A backtest entangles signal quality with stop placement, target placement,
sizing, and threshold choices. When it fails you cannot tell which ingredient
was wrong, and when it succeeds you cannot tell whether it was the signal or a
lucky exit rule. An event study isolates the signal: define the event, collect
every occurrence, and compare the forward price distribution against a matched
baseline.

WHAT "MATCHED" MEANS
--------------------
A setup that mostly fires at 09:45 in high volatility will show large forward
excursions for reasons that have nothing to do with the setup. So the baseline
is not the unconditional average over all bars. It is post-stratified: bars are
binned by (time-of-day, volatility percentile), the baseline outcome is
computed per bin, and those bin outcomes are then re-weighted to match the
bins the events actually landed in. The reported edge is the difference between
the event mean and that standardized baseline.

CONFIDENCE INTERVALS
--------------------
Whole sessions are resampled with replacement. Bars within a session are
heavily autocorrelated and 60-bar forward windows on adjacent bars overlap
almost completely, so a per-bar standard error would be roughly sqrt(60) times
too narrow.

Usage:
    python3 event_study.py --features retest.parquet --setup weak_retest_short
    python3 event_study.py --features retest.parquet --setup all
    python3 event_study.py --data /path/to/data --setup all --list-setups
"""

import argparse

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = (5, 15, 30, 60)

# Post-stratification bins.
TOD_BUCKET_MINUTES = 30
N_VOL_BUCKETS = 3


# ============================================================
# Forward outcome measurement
# ============================================================

def _forward_extreme(values, horizon, kind='max'):
    """
    Forward-looking rolling extreme over bars [i+1 .. i+horizon].

    NaN where the window is incomplete, so a truncated tail never biases the
    result toward smaller excursions.
    """
    n = len(values)
    out = np.full(n, np.nan)
    if n <= horizon:
        return out

    rev = pd.Series(values[::-1])
    roll = (rev.rolling(horizon).max() if kind == 'max'
            else rev.rolling(horizon).min()).to_numpy()

    # rolling value at reversed index j covers original [n-1-j .. n-j+horizon-2],
    # which is the window [i+1 .. i+horizon] for i = n-2-j.
    idx = np.arange(n - horizon - 1, -1, -1)
    j = n - 2 - idx
    out[idx] = roll[j]
    return out


def add_forward_outcomes(bars, horizons=DEFAULT_HORIZONS):
    """
    Add per-horizon forward excursions, computed within each session only.

    Columns added per horizon h:
        fwd_max_h, fwd_min_h  -- extreme high/low over [t+1 .. t+h]
        fwd_close_h           -- close at t+h
    """
    bars = bars.copy()
    if 'session_date' not in bars.columns:
        raise ValueError("bars must carry a 'session_date' column")

    for h in horizons:
        fwd_max = np.full(len(bars), np.nan)
        fwd_min = np.full(len(bars), np.nan)
        fwd_close = np.full(len(bars), np.nan)

        for _, idx in bars.groupby('session_date', sort=True).indices.items():
            idx = np.sort(idx)
            highs = bars['high'].to_numpy()[idx]
            lows = bars['low'].to_numpy()[idx]
            closes = bars['close'].to_numpy()[idx]

            fwd_max[idx] = _forward_extreme(highs, h, 'max')
            fwd_min[idx] = _forward_extreme(lows, h, 'min')

            shifted = np.full(len(idx), np.nan)
            if len(idx) > h:
                shifted[:-h] = closes[h:]
            fwd_close[idx] = shifted

        bars[f'fwd_max_{h}'] = fwd_max
        bars[f'fwd_min_{h}'] = fwd_min
        bars[f'fwd_close_{h}'] = fwd_close

    return bars


def signed_outcomes(bars, direction, horizon):
    """
    Direction-signed MFE, MAE, and terminal return in ATR units.

    direction: +1 for a long hypothesis, -1 for a short hypothesis.
    """
    close = bars['close'].to_numpy(dtype=np.float64)
    atr = bars['atr_14'].to_numpy(dtype=np.float64)
    fwd_max = bars[f'fwd_max_{horizon}'].to_numpy(dtype=np.float64)
    fwd_min = bars[f'fwd_min_{horizon}'].to_numpy(dtype=np.float64)
    fwd_close = bars[f'fwd_close_{horizon}'].to_numpy(dtype=np.float64)

    up = fwd_max - close
    down = close - fwd_min

    if direction > 0:
        mfe, mae, ret = up, down, fwd_close - close
    else:
        mfe, mae, ret = down, up, close - fwd_close

    with np.errstate(invalid='ignore', divide='ignore'):
        safe_atr = np.where(atr > 0, atr, np.nan)
        return {
            'mfe_atr': mfe / safe_atr,
            'mae_atr': mae / safe_atr,
            'ret_atr': ret / safe_atr,
            'edge_atr': (mfe - mae) / safe_atr,
        }


# ============================================================
# Post-stratification bins
# ============================================================

def assign_bins(bars, n_vol_buckets=N_VOL_BUCKETS):
    """Composite (time-of-day, volatility) bin id per bar."""
    minute = bars['minute_of_day_et'].to_numpy(dtype=np.float64)
    tod = np.floor(minute / TOD_BUCKET_MINUTES).astype(np.int64)

    pct = bars['atr_percentile'].to_numpy(dtype=np.float64)
    vol = np.full(len(bars), -1, dtype=np.int64)
    ok = np.isfinite(pct)
    if ok.any():
        edges = np.linspace(0, 1, n_vol_buckets + 1)[1:-1]
        vol[ok] = np.searchsorted(edges, pct[ok])

    return pd.Series(tod * 1000 + vol, index=bars.index, name='bin_id')


class _Strata:
    """
    Per-(session, bin) sufficient statistics for the standardized comparison.

    Post-stratified means are sums divided by counts, and both are additive
    over sessions. Reducing to a (n_sessions x n_bins) table of counts and sums
    up front means a bootstrap resample is a row-index plus a column sum,
    instead of re-running a groupby over hundreds of thousands of rows. The
    result is identical; at 1,000 sessions it is the difference between hours
    and milliseconds.
    """

    __slots__ = ('n_ev', 'sum_ev', 'n_ct', 'sum_ct', 'n_sessions', 'n_bins')

    def __init__(self, outcome, bin_codes, n_bins, event_mask, control_mask,
                 session_codes, n_sessions):
        outcome = np.asarray(outcome, dtype=np.float64)
        valid = np.isfinite(outcome)
        flat = session_codes * n_bins + bin_codes
        size = n_sessions * n_bins

        ev = valid & event_mask
        ct = valid & control_mask

        self.n_ev = np.bincount(flat[ev], minlength=size).reshape(n_sessions, n_bins).astype(np.float64)
        self.sum_ev = np.bincount(flat[ev], weights=outcome[ev], minlength=size).reshape(n_sessions, n_bins)
        self.n_ct = np.bincount(flat[ct], minlength=size).reshape(n_sessions, n_bins).astype(np.float64)
        self.sum_ct = np.bincount(flat[ct], weights=outcome[ct], minlength=size).reshape(n_sessions, n_bins)
        self.n_sessions = n_sessions
        self.n_bins = n_bins

    def standardize(self, rows=None):
        """
        Event mean, bin-matched baseline, and matched share.

        rows selects which sessions to include (all of them for the point
        estimate, a bootstrap resample otherwise). Bins holding events but no
        controls are excluded from both sides and reflected in matched_share,
        so the comparison never silently comes from unmatched strata.
        """
        if rows is None:
            NE, SE = self.n_ev.sum(0), self.sum_ev.sum(0)
            NC, SC = self.n_ct.sum(0), self.sum_ct.sum(0)
        else:
            NE, SE = self.n_ev[rows].sum(0), self.sum_ev[rows].sum(0)
            NC, SC = self.n_ct[rows].sum(0), self.sum_ct[rows].sum(0)

        total_events = NE.sum()
        shared = (NE > 0) & (NC > 0)
        if total_events <= 0 or not shared.any():
            return np.nan, np.nan, 0.0

        ne = NE[shared]
        weights = ne / ne.sum()
        baseline = float((SC[shared] / NC[shared] * weights).sum())
        event_mean = float(SE[shared].sum() / ne.sum())
        return event_mean, baseline, float(ne.sum() / total_events)

    def bootstrap_ci(self, n_boot=1000, alpha=0.05, seed=0):
        """Session-resampled CI for (event mean - standardized baseline)."""
        if self.n_sessions < 5:
            return (np.nan, np.nan)

        rng = np.random.default_rng(seed)
        diffs = np.full(n_boot, np.nan)
        for b in range(n_boot):
            rows = rng.integers(0, self.n_sessions, size=self.n_sessions)
            ev, base, _ = self.standardize(rows)
            if np.isfinite(ev) and np.isfinite(base):
                diffs[b] = ev - base

        diffs = diffs[np.isfinite(diffs)]
        if len(diffs) < 20:
            return (np.nan, np.nan)
        return (float(np.quantile(diffs, alpha / 2)),
                float(np.quantile(diffs, 1 - alpha / 2)))


# ============================================================
# Setup definitions
# ============================================================

def _setup_weak_retest_short(df):
    """
    Annotation 1 and 3 from the chart: price returns up to a level that
    previously traded heavy sell-side volume, and the retest brings materially
    less trade. Buyers are not showing up. Expect continuation down.
    """
    return (
        (df['has_retest_level'] == 1)
        & (df['prior_visit_delta_pct'] < -0.10)
        & (df['retest_vol_ratio'] < 0.40)
        & (df['retest_level_dist_ticks'].abs() <= 4)
        & (df['bars_since_prior_visit'] >= 10)
    ), -1


def _setup_weak_retest_long(df):
    """Mirror of weak_retest_short: failed retest of a buy-dominant level."""
    return (
        (df['has_retest_level'] == 1)
        & (df['prior_visit_delta_pct'] > 0.10)
        & (df['retest_vol_ratio'] < 0.40)
        & (df['retest_level_dist_ticks'].abs() <= 4)
        & (df['bars_since_prior_visit'] >= 10)
    ), 1


def _absorption_is_high(df):
    """
    Absorption above its own trailing median.

    Uses an expanding median lagged by one bar rather than the full-sample
    median: a full-sample statistic would let the late part of the history
    define the threshold for the early part, which is lookahead.
    """
    a = df['absorption_at_level']
    trailing = a.expanding(min_periods=200).median().shift(1)
    return a > trailing


def _setup_absorption_reversal_long(df):
    """
    Heavy volume at a level with almost no downward progress past it, and the
    delta has flipped from the prior visit. Sellers are being absorbed.
    """
    return (
        (df['has_retest_level'] == 1)
        & _absorption_is_high(df)
        & (df['excursion_beyond_ratio'] < 0.5)
        & (df['retest_delta_flip'] == 1)
        & (df['current_visit_delta_pct'] > 0)
    ), 1


def _setup_absorption_reversal_short(df):
    """Mirror of absorption_reversal_long."""
    return (
        (df['has_retest_level'] == 1)
        & _absorption_is_high(df)
        & (df['excursion_beyond_ratio'] < 0.5)
        & (df['retest_delta_flip'] == 1)
        & (df['current_visit_delta_pct'] < 0)
    ), -1


def _setup_big_print_confirm_short(df):
    """
    Annotation 2: large participants are still on the sell side at the level on
    the retest, i.e. the original seller is defending it.
    """
    return (
        (df['has_retest_level'] == 1)
        & (df['retest_big_print_same_side'] == 1)
        & (df['prior_visit_delta'] < 0)
        & (df['retest_big_prints_now'] >= 1)
    ), -1


def _setup_third_test_short(df):
    """Repeated tests weaken a level: third or later visit, still thin."""
    return (
        (df['has_retest_level'] == 1)
        & (df['n_prior_visits'] >= 2)
        & (df['retest_vol_ratio'] < 0.60)
        & (df['prior_visit_delta_pct'] < 0)
    ), -1


def _setup_any_level_test_short(df):
    """
    Deliberately weak control setup: any level test at all, no flow condition.
    If a real setup does not beat this, its conditions are doing nothing.
    """
    return (df['has_retest_level'] == 1), -1


SETUPS = {
    'weak_retest_short': _setup_weak_retest_short,
    'weak_retest_long': _setup_weak_retest_long,
    'absorption_reversal_long': _setup_absorption_reversal_long,
    'absorption_reversal_short': _setup_absorption_reversal_short,
    'big_print_confirm_short': _setup_big_print_confirm_short,
    'third_test_short': _setup_third_test_short,
    'any_level_test_short': _setup_any_level_test_short,
}


# ============================================================
# CUSUM event sampling
# ============================================================

def cusum_events(bars, threshold_atr=0.5):
    """
    Symmetric CUSUM filter on close prices, with the threshold scaled by ATR.

    Returns a boolean mask of bars where cumulative drift in either direction
    exceeded the threshold, resetting on each trigger. Use it to restrict
    modelling to bars where something actually happened, which removes most of
    the redundancy created by labelling every single bar.
    """
    close = bars['close'].to_numpy(dtype=np.float64)
    atr = bars['atr_14'].to_numpy(dtype=np.float64)
    sessions = bars['session_date'].to_numpy()

    mask = np.zeros(len(bars), dtype=bool)
    s_pos = s_neg = 0.0
    prev_session = None

    for i in range(1, len(close)):
        if sessions[i] != prev_session:
            s_pos = s_neg = 0.0
            prev_session = sessions[i]
            continue
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue

        diff = close[i] - close[i - 1]
        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)
        limit = threshold_atr * atr[i]

        if s_pos > limit:
            mask[i] = True
            s_pos = 0.0
        elif -s_neg > limit:
            mask[i] = True
            s_neg = 0.0

    return mask


# ============================================================
# Evaluation
# ============================================================

def evaluate_setup(bars, name, horizons=DEFAULT_HORIZONS, n_boot=1000,
                   rth_only=True, min_events=100):
    """Run the full event study for one named setup across all horizons."""
    if name not in SETUPS:
        raise KeyError(f"Unknown setup {name!r}. Known: {sorted(SETUPS)}")

    scope = bars[bars['is_rth']] if rth_only else bars
    scope = scope.copy()

    mask, direction = SETUPS[name](scope)
    mask = mask.fillna(False).to_numpy(dtype=bool)
    n_events = int(mask.sum())

    bin_ids = assign_bins(scope)
    # Controls are every other in-scope bar. Bars that are themselves events
    # are excluded so the baseline is not contaminated by the signal.
    control_mask = ~mask

    print(f"\n{'='*72}")
    print(f"  SETUP: {name}   direction={'LONG' if direction > 0 else 'SHORT'}")
    print(f"{'='*72}")
    print(f"  Events:    {n_events:,} of {len(scope):,} in-scope bars "
          f"({n_events / max(len(scope), 1) * 100:.2f}%)")
    print(f"  Sessions:  {scope['session_date'].nunique()}  "
          f"(events on {scope.loc[mask, 'session_date'].nunique()} of them)")

    if n_events < min_events:
        print(f"\n  TOO FEW EVENTS (< {min_events}). No conclusion is possible.")
        print("  Either loosen the conditions or extend the date range.")
        return None

    # Factorize bins and sessions once; every metric and horizon reuses them.
    bin_codes, bin_uniq = pd.factorize(bin_ids.to_numpy(), sort=True)
    sess_codes, sess_uniq = pd.factorize(scope['session_date'].to_numpy(),
                                         sort=True)
    n_bins, n_sessions = len(bin_uniq), len(sess_uniq)

    rows = []
    for h in horizons:
        out = signed_outcomes(scope, direction, h)
        for metric in ('mfe_atr', 'mae_atr', 'ret_atr', 'edge_atr'):
            strata = _Strata(out[metric], bin_codes, n_bins, mask,
                             control_mask, sess_codes, n_sessions)
            ev_mean, base, matched = strata.standardize()
            lo, hi = strata.bootstrap_ci(n_boot=n_boot)
            rows.append({
                'horizon': h,
                'metric': metric,
                'event': ev_mean,
                'baseline': base,
                'diff': ev_mean - base,
                'ci_lo': lo,
                'ci_hi': hi,
                'significant': bool(np.isfinite(lo) and np.isfinite(hi)
                                    and (lo > 0 or hi < 0)),
                'matched_share': matched,
            })

    res = pd.DataFrame(rows)

    print(f"\n  Outcomes in ATR units. 'diff' is event mean minus the")
    print(f"  time-of-day and volatility matched baseline.")
    print(f"\n  {'h':>4} | {'metric':>9} | {'event':>7} | {'base':>7} | "
          f"{'diff':>7} | {'95% CI':>18} | sig")
    print(f"  {'-'*4}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*18}-+----")
    for _, r in res.iterrows():
        star = ' *' if r['significant'] else ''
        print(f"  {r['horizon']:>4} | {r['metric']:>9} | {r['event']:>7.3f} | "
              f"{r['baseline']:>7.3f} | {r['diff']:>+7.3f} | "
              f"[{r['ci_lo']:>+7.3f}, {r['ci_hi']:>+7.3f}] |{star}")

    _interpret(res, name)
    return res


def _interpret(res, name):
    edge = res[res['metric'] == 'edge_atr']
    sig = edge[edge['significant']]

    print(f"\n  Reading:")
    if len(sig) == 0:
        print(f"    No horizon shows a significant edge for '{name}'. The")
        print("    conditional forward distribution is indistinguishable from")
        print("    matched control bars. No exit rule or regime filter will")
        print("    rescue this -- the signal carries no information.")
        return

    best = sig.loc[sig['diff'].abs().idxmax()]
    print(f"    Significant at horizon(s): {sorted(sig['horizon'].tolist())}")
    print(f"    Strongest at h={int(best['horizon'])}: "
          f"{best['diff']:+.3f} ATR of edge (MFE - MAE) vs baseline.")

    mfe = res[(res['metric'] == 'mfe_atr') & (res['horizon'] == best['horizon'])]
    mae = res[(res['metric'] == 'mae_atr') & (res['horizon'] == best['horizon'])]
    if len(mfe) and len(mae):
        m_ev = float(mfe['event'].iloc[0])
        a_ev = float(mae['event'].iloc[0])
        print(f"    At that horizon events average {m_ev:.2f} ATR favorable")
        print(f"    excursion against {a_ev:.2f} ATR adverse. That ratio, not a")
        print("    guessed 1.5R, is what should set the target and stop.")
    print(f"    Horizon {int(best['horizon'])} bars is the holding period the")
    print("    signal actually supports.")


def main():
    parser = argparse.ArgumentParser(
        description='Conditional event study on level-retest setups.')
    parser.add_argument('--features', default=None,
                        help='Parquet/CSV from level_ledger.py')
    parser.add_argument('--data', default=None,
                        help='Directory of .data files (builds features first)')
    parser.add_argument('--setup', default='all',
                        help="Setup name, comma-separated list, or 'all'")
    parser.add_argument('--list-setups', action='store_true',
                        help='Print available setups and exit')
    parser.add_argument('--horizons', default='5,15,30,60',
                        help='Comma-separated forward horizons in bars')
    parser.add_argument('--n-boot', type=int, default=1000,
                        help='Bootstrap resamples (default: 1000)')
    parser.add_argument('--min-events', type=int, default=100,
                        help='Refuse to report below this many events')
    parser.add_argument('--all-hours', action='store_true',
                        help='Include overnight bars (default: RTH only)')
    parser.add_argument('--direction-method', default='level',
                        choices=['level', 'tick', 'mid'])
    parser.add_argument('--out', default=None, help='Write results to this CSV')

    args = parser.parse_args()

    if args.list_setups:
        print("Available setups:")
        for name, fn in sorted(SETUPS.items()):
            doc = (fn.__doc__ or '').strip().split('\n')[0]
            print(f"  {name:<28} {doc}")
        return

    if args.features:
        if args.features.endswith('.parquet'):
            bars = pd.read_parquet(args.features)
        else:
            bars = pd.read_csv(args.features, index_col=0, parse_dates=True)
        print(f"Loaded {len(bars):,} bars from {args.features}")
    elif args.data:
        from level_ledger import build_retest_dataset
        bars = build_retest_dataset(args.data, direction_method=args.direction_method)
    else:
        parser.error("one of --features or --data is required")

    horizons = tuple(int(h) for h in args.horizons.split(','))
    print("Computing forward outcomes...")
    bars = add_forward_outcomes(bars, horizons)

    names = sorted(SETUPS) if args.setup == 'all' else args.setup.split(',')

    all_res = []
    for name in names:
        res = evaluate_setup(bars, name.strip(), horizons=horizons,
                             n_boot=args.n_boot, rth_only=not args.all_hours,
                             min_events=args.min_events)
        if res is not None:
            res['setup'] = name.strip()
            all_res.append(res)

    if all_res:
        combined = pd.concat(all_res, ignore_index=True)
        _summary_table(combined)
        if args.out:
            combined.to_csv(args.out, index=False)
            print(f"\n  Results written to {args.out}")


def _summary_table(combined):
    print(f"\n{'='*72}")
    print("  SUMMARY: edge_atr (MFE - MAE vs matched baseline)")
    print(f"{'='*72}")
    edge = combined[combined['metric'] == 'edge_atr']
    pivot = edge.pivot_table(index='setup', columns='horizon', values='diff')
    print(pivot.to_string(float_format=lambda v: f"{v:+.3f}"))

    n_sig = int(edge['significant'].sum())
    n_tests = len(edge)
    print(f"\n  Significant cells: {n_sig} of {n_tests} tested.")
    print(f"  At a 5% level roughly {0.05 * n_tests:.1f} would be expected by")
    print("  chance alone. Treat marginal results accordingly, and remember")
    print("  every setup you try adds to that multiple-testing budget.")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
