#!/usr/bin/env python3
"""
End-to-end self-test on synthetic .data files with known ground truth.

Run this once before pointing anything at real data. It writes synthetic
sessions in the real 57-byte record format, then checks that each stage
recovers properties that were deliberately planted.

Two synthetic feeds are generated:

  'quote'  -- [level_low, level_high] really is the bid/ask, width always 0.25,
              and aggressive buys genuinely push price up. The 'level' rule
              must score well here.
  'bucket' -- level bounds are 1.0-point buckets, as the real files appear to
              be. The 'level' rule must degrade badly here while the 'tick'
              rule holds up. This is the scenario validate_direction.py exists
              to detect.

Usage:
    python3 selftest.py
    python3 selftest.py --keep /tmp/synth   # keep the generated files
"""

import argparse
import os
import shutil
import struct
import tempfile
from datetime import datetime, timedelta

import numpy as np

import bars as bars_mod
import event_study
import level_ledger
import purged_cv
import tick_data

RECORD_FORMAT = '<q3d6IB'
DOTNET_EPOCH = datetime(1, 1, 1)
TICK = 0.25

PASS = "  PASS"
FAIL = "  FAIL"


class SelfTestFailure(AssertionError):
    pass


def _check(condition, label, detail=''):
    if condition:
        print(f"{PASS}  {label}")
        return True
    print(f"{FAIL}  {label}  {detail}")
    raise SelfTestFailure(f"{label}: {detail}")


# ============================================================
# Synthetic data generation
# ============================================================

def _dotnet_ticks(ts):
    """
    .NET DateTime ticks via integer arithmetic.

    total_seconds() * 1e7 would be about 6.4e17, well past the 2^53 exact range
    of a float64, so it silently quantises to ~10 microseconds. Integer math
    keeps the round-trip exact.
    """
    delta = ts - DOTNET_EPOCH
    return ((delta.days * 86400 + delta.seconds) * 10_000_000
            + delta.microseconds * 10)


def make_synthetic_session(path, session_start, n_prints=40000, mode='quote',
                           seed=0, start_price=5000.0):
    """
    Write one synthetic session in the real record format.

    Aggressor side is generated first with positive autocorrelation, then price
    is moved by it. That ordering is what makes the contemporaneous
    delta-to-return relationship real rather than incidental, which is exactly
    the property validate_direction.py measures.
    """
    rng = np.random.default_rng(seed)

    # Autocorrelated aggressor sides.
    sides = np.empty(n_prints, dtype=np.int8)
    sides[0] = 1
    flip = rng.random(n_prints) < 0.35
    for i in range(1, n_prints):
        sides[i] = -sides[i - 1] if flip[i] else sides[i - 1]

    # Aggression moves price: a buy sometimes lifts the offer one tick.
    # int64 explicitly -- an int8 cumsum over 40k prints would be at risk of
    # wrapping if numpy's upcasting rules ever changed.
    moves = sides.astype(np.int64) * (rng.random(n_prints) < 0.30)
    price_ticks = np.cumsum(moves) + int(round(start_price / TICK))
    prices = price_ticks * TICK

    sizes = np.maximum(1, rng.geometric(0.35, n_prints)).astype(np.uint32)
    # Plant occasional large prints so big-print features have something to see.
    big = rng.random(n_prints) < 0.005
    sizes[big] = rng.integers(150, 400, big.sum())

    if mode == 'quote':
        level_low = np.where(sides == 1, prices - TICK, prices)
        level_high = np.where(sides == 1, prices, prices + TICK)
    elif mode == 'bucket':
        bucket = np.floor(prices)
        level_low = bucket
        level_high = bucket + 1.0
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # Spread prints across the RTH session, in Chicago local time.
    total_seconds = 6.5 * 3600
    offsets = np.sort(rng.random(n_prints)) * total_seconds

    cum = np.cumsum(sizes.astype(np.uint64))

    with open(path, 'wb') as f:
        for i in range(n_prints):
            ts = session_start + timedelta(seconds=float(offsets[i]))
            net_ticks = _dotnet_ticks(ts)
            f.write(struct.pack(
                RECORD_FORMAT,
                net_ticks,
                float(prices[i]), float(level_low[i]), float(level_high[i]),
                int(sizes[i]), int(sizes[i]), int(cum[i] % (2**32)),
                0, 0, 0, 1,
            ))

    return {
        'n_prints': n_prints,
        'total_volume': float(sizes.sum()),
        'price_min': float(prices.min()),
        'price_max': float(prices.max()),
        'true_sides': sides,
    }


def build_synthetic_dir(root, n_sessions=40, mode='quote', n_prints=40000):
    """Write n_sessions consecutive weekday sessions of synthetic data."""
    os.makedirs(root, exist_ok=True)
    truth = []
    day = datetime(2024, 1, 2, 8, 30)  # 08:30 Chicago == 09:30 ET
    made = 0
    seed = 0
    while made < n_sessions:
        if day.weekday() < 5:
            path = os.path.join(root, f"{day.strftime('%Y%m%d')}.data")
            truth.append(make_synthetic_session(
                path, day, n_prints=n_prints, mode=mode, seed=seed))
            made += 1
            seed += 1
        day += timedelta(days=1)
    return truth


# ============================================================
# Stage tests
# ============================================================

def test_decoder(data_dir, truth):
    print("\n[1] Decoder round-trip")
    files = tick_data.find_data_files(data_dir)
    _check(len(files) == len(truth), "file discovery",
           f"found {len(files)}, expected {len(truth)}")

    df = tick_data.decode_file(files[0])
    _check(len(df) == truth[0]['n_prints'], "record count",
           f"{len(df)} vs {truth[0]['n_prints']}")
    _check(abs(df['volume'].sum() - truth[0]['total_volume']) < 1e-6,
           "volume round-trip",
           f"{df['volume'].sum()} vs {truth[0]['total_volume']}")
    _check(abs(df['price'].min() - truth[0]['price_min']) < 1e-9
           and abs(df['price'].max() - truth[0]['price_max']) < 1e-9,
           "price round-trip")
    _check(df['timestamp'].is_monotonic_increasing, "timestamps monotonic")

    off_grid = (~np.isclose(np.remainder(df['price'].to_numpy(), TICK), 0,
                            atol=1e-9)).sum()
    _check(off_grid == 0, "prices on the 0.25 grid", f"{off_grid} off-grid")
    return df


def test_direction_recovery(quote_dir, bucket_dir):
    print("\n[2] Aggressor classification recovery")

    df_q = tick_data.decode_file(tick_data.find_data_files(quote_dir)[0])
    truth_sides = tick_data.classify_direction(df_q, method='level')
    # In quote mode the level rule is exact by construction.
    _check((truth_sides != 0).all(), "quote mode leaves no unknown prints")

    def contemp_r(directory, method):
        rs = []
        for _, ticks in tick_data.iter_sessions(directory, verbose=False):
            d = tick_data.classify_direction(ticks, method=method)
            scoped = ticks.copy()
            scoped['direction'] = d
            b, _ = bars_mod.build_bars(scoped, window='1min',
                                       with_footprint=False)
            if len(b) < 30:
                continue
            ret = ((b['close'] - b['open']) / TICK).to_numpy()
            vol = b['volume'].to_numpy()
            delta = b['delta'].to_numpy()
            with np.errstate(invalid='ignore', divide='ignore'):
                dpct = np.where(vol > 0, delta / vol, np.nan)
            r = _pearson(dpct, ret)
            if np.isfinite(r):
                rs.append(r)
        return float(np.mean(rs)) if rs else np.nan

    r_quote_level = contemp_r(quote_dir, 'level')
    r_bucket_level = contemp_r(bucket_dir, 'level')
    r_bucket_tick = contemp_r(bucket_dir, 'tick')

    print(f"       quote  feed, 'level' rule: r = {r_quote_level:+.3f}")
    print(f"       bucket feed, 'level' rule: r = {r_bucket_level:+.3f}")
    print(f"       bucket feed, 'tick'  rule: r = {r_bucket_tick:+.3f}")

    _check(r_quote_level > 0.5,
           "correct labels give strong contemporaneous correlation",
           f"r={r_quote_level:.3f}, expected > 0.5")
    _check(r_bucket_level < r_quote_level - 0.15,
           "bucket geometry degrades the quote rule",
           f"{r_bucket_level:.3f} vs {r_quote_level:.3f}")
    _check(r_bucket_tick > r_bucket_level,
           "tick rule beats the quote rule on bucket geometry",
           f"{r_bucket_tick:.3f} vs {r_bucket_level:.3f}")


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    x = x[ok] - x[ok].mean()
    y = y[ok] - y[ok].mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else np.nan


def test_bars(ticks):
    print("\n[3] Bar construction and footprint")

    b, fp = bars_mod.build_bars(ticks, window='1min', with_footprint=True)
    _check(len(b) > 0, "bars produced")
    _check(abs(b['volume'].sum() - ticks['volume'].sum()) < 1e-6,
           "bar volume conserves tick volume")
    _check(abs(fp['volume'].sum() - ticks['volume'].sum()) < 1e-6,
           "footprint volume conserves tick volume")
    _check((b['high'] >= b['low']).all(), "high >= low")
    _check((b['high'] >= b['close']).all() and (b['low'] <= b['close']).all(),
           "close inside the bar range")
    _check(b.index.is_monotonic_increasing, "bar index sorted")

    va = bars_mod.value_area_from_footprint(fp)
    _check(len(va) > 0, "value area computed")
    _check((va['val'] <= va['poc']).all() and (va['poc'] <= va['vah']).all(),
           "VAL <= POC <= VAH")

    vb, _ = bars_mod.build_bars(ticks, window='1min', volume_per_bar=2000,
                                with_footprint=False)
    _check(len(vb) > 0, "volume bars produced")
    _check(abs(vb['volume'].sum() - ticks['volume'].sum()) < 1e-6,
           "volume bars conserve tick volume")
    vol_cv = vb['volume'].std() / vb['volume'].mean()
    time_cv = b['volume'].std() / b['volume'].mean()
    print(f"       volume dispersion: time bars CV={time_cv:.3f}, "
          f"volume bars CV={vol_cv:.3f}")
    _check(vol_cv < time_cv,
           "volume bars are more homogeneous than time bars",
           f"{vol_cv:.3f} vs {time_cv:.3f}")


def test_level_ledger(data_dir):
    print("\n[4] Level ledger and retest features")

    df = level_ledger.build_retest_dataset(data_dir, verbose=False)
    _check(len(df) > 0, "dataset built")

    live = df[df['has_retest_level'] == 1]
    _check(len(live) > 100, "reference levels found",
           f"only {len(live)} bars with a level")
    print(f"       {len(live):,} of {len(df):,} bars have a reference level "
          f"({len(live)/len(df)*100:.1f}%)")

    for col in ('retest_vol_ratio', 'prior_visit_volume', 'current_visit_volume',
                'bars_since_prior_visit', 'n_prior_visits'):
        _check(live[col].notna().mean() > 0.9,
               f"{col} populated when a level exists",
               f"{live[col].notna().mean()*100:.1f}% non-null")

    _check((live['prior_visit_volume'] >= level_ledger.DEFAULT_MIN_PRIOR_VOLUME).all(),
           "prior visits respect the minimum volume filter")
    _check((live['bars_since_prior_visit'] >= 0).all(),
           "prior visit precedes the current bar")
    _check((live['retest_vol_ratio'] >= 0).all(), "retest volume ratio non-negative")
    _check((live['retest_level_dist_ticks'].abs()
            <= level_ledger.DEFAULT_NEAR_TICKS + 0.51).all(),
           "selected level is within the near-ticks window")
    return df


def test_forward_outcomes(df):
    print("\n[5] Forward outcome measurement")

    out = event_study.add_forward_outcomes(df, horizons=(5, 30))
    for h in (5, 30):
        cov = out[f'fwd_max_{h}'].notna().mean()
        print(f"       horizon {h}: {cov*100:.1f}% of bars have a complete window")
        _check(cov > 0.5, f"horizon {h} coverage", f"{cov*100:.1f}%")

        ok = out[[f'fwd_max_{h}', f'fwd_min_{h}']].notna().all(axis=1)
        sub = out[ok]
        _check((sub[f'fwd_max_{h}'] >= sub[f'fwd_min_{h}']).all(),
               f"horizon {h}: fwd_max >= fwd_min")

    # A forward window must never cross a session boundary. The last bar of
    # each session therefore cannot have a complete window.
    last_of_session = out.groupby('session_date').tail(1)
    _check(last_of_session['fwd_max_5'].isna().all(),
           "forward windows do not cross session boundaries")

    long_out = event_study.signed_outcomes(out, 1, 30)
    ok = np.isfinite(long_out['mfe_atr']) & np.isfinite(long_out['mae_atr'])
    _check((long_out['mfe_atr'][ok] >= -1e-9).all(), "MFE non-negative")
    _check((long_out['mae_atr'][ok] >= -1e-9).all(), "MAE non-negative")
    return out


def test_purging():
    print("\n[6] Purged cross-validation")

    # 1000 samples, each with a 60-bar label window, sampled every bar.
    sample_idx = np.arange(1000)
    label_end = sample_idx + 60

    stats = purged_cv.effective_sample_size(label_end, sample_idx)
    print(f"       mean label span {stats['mean_label_span_bars']:.0f} bars, "
          f"concurrency {stats['mean_concurrency']:.1f}, "
          f"effective n {stats['effective_n']:.0f} of {stats['n_samples']}")
    _check(stats['effective_n'] < stats['n_samples'] / 10,
           "overlap collapses the effective sample size",
           f"effective_n={stats['effective_n']:.0f}")

    n_folds = 0
    for train, test in purged_cv.purged_kfold(sample_idx, label_end,
                                              n_splits=5, embargo_pct=0.01):
        n_folds += 1
        test_start = sample_idx[test[0]]
        test_end = label_end[test].max()
        overlap = [(sample_idx[p], label_end[p]) for p in train
                   if label_end[p] >= test_start and sample_idx[p] <= test_end]
        _check(len(overlap) == 0,
               f"k-fold {n_folds}: no training label overlaps the test window",
               f"{len(overlap)} overlapping samples")
    _check(n_folds >= 3, "k-fold produced usable folds", f"{n_folds} folds")

    n_wf = 0
    for train, test in purged_cv.purged_walk_forward(sample_idx, label_end,
                                                     n_splits=5):
        n_wf += 1
        test_start = sample_idx[test[0]]
        _check((label_end[train] < test_start).all(),
               f"walk-forward {n_wf}: all training labels resolve before test")
        _check((train < test[0]).all(),
               f"walk-forward {n_wf}: training strictly precedes test")
    _check(n_wf >= 3, "walk-forward produced usable folds", f"{n_wf} folds")


def test_meta_labels(df):
    print("\n[7] Conditional meta-labels")

    scope = df[df['is_rth']].reset_index(drop=False)
    mask, direction = event_study.SETUPS['any_level_test_short'](scope)
    signal_idx = np.flatnonzero(mask.fillna(False).to_numpy())
    _check(len(signal_idx) > 200, "enough signals to label",
           f"{len(signal_idx)}")

    meta = purged_cv.make_meta_labels(scope, signal_idx, direction,
                                      sl_atr=1.0, tp_atr=1.5, max_hold=60)
    _check(len(meta) > 100, "meta-labels produced", f"{len(meta)}")
    _check(set(meta['label'].unique()) <= {0, 1}, "labels are binary")
    _check((meta['label_end'] >= meta.index).all(),
           "label resolves at or after the signal bar")
    _check((meta['bars_held'] <= 60).all(), "holding period respects max_hold")

    # Every signal is kept, including the ones that resolve badly. This is the
    # denominator that es_training's win-rate calculation drops.
    reasons = meta['exit_reason'].value_counts().to_dict()
    print(f"       exits: {reasons}")
    print(f"       base rate: {meta['label'].mean():.3f} "
          f"(breakeven at 1.5R = {1/(1+1.5):.3f})")
    _check('timeout' in reasons or 'session_end' in reasons or 'sl' in reasons,
           "losing and unresolved outcomes are retained")

    # Stops must be on the correct side of entry for a short hypothesis.
    _check((meta['sl_price'] > meta['entry_price']).all(),
           "short stops sit above entry")
    _check((meta['tp_price'] < meta['entry_price']).all(),
           "short targets sit below entry")


def test_event_study(df):
    print("\n[8] Event study end to end")

    out = event_study.add_forward_outcomes(df, horizons=(15,))
    res = event_study.evaluate_setup(out, 'any_level_test_short',
                                     horizons=(15,), n_boot=120,
                                     min_events=50)
    _check(res is not None, "event study returned results")
    _check(set(res['metric']) == {'mfe_atr', 'mae_atr', 'ret_atr', 'edge_atr'},
           "all four outcome metrics reported")
    _check(res['matched_share'].min() > 0.5,
           "most events matched to a control bin",
           f"min matched share {res['matched_share'].min():.2f}")
    _check(res['ci_lo'].notna().all() and res['ci_hi'].notna().all(),
           "bootstrap intervals computed")
    _check((res['ci_lo'] <= res['ci_hi']).all(), "intervals ordered")

    # Synthetic prices are a driftless random walk, so a setup with no real
    # conditions must not show a meaningful edge. If it does, the baseline
    # matching is broken.
    edge = res[res['metric'] == 'edge_atr']['diff'].abs().max()
    print(f"       |edge| on random-walk data: {edge:.3f} ATR")
    _check(edge < 0.35, "no spurious edge on a driftless random walk",
           f"|edge|={edge:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Self-test the pipeline.')
    parser.add_argument('--keep', default=None,
                        help='Directory to write synthetic data into and keep')
    parser.add_argument('--sessions', type=int, default=40,
                        help='Synthetic sessions to generate (default: 40)')
    parser.add_argument('--prints', type=int, default=40000,
                        help='Prints per session (default: 40000)')
    args = parser.parse_args()

    root = args.keep or tempfile.mkdtemp(prefix='enhanced_selftest_')
    quote_dir = os.path.join(root, 'quote')
    bucket_dir = os.path.join(root, 'bucket')

    print("=" * 72)
    print("  ENHANCED_TRAINING SELF-TEST")
    print("=" * 72)
    print(f"  Synthetic data root: {root}")
    print(f"  Sessions: {args.sessions}, prints/session: {args.prints:,}")

    failed = False
    try:
        print("\n[0] Generating synthetic sessions...")
        truth_q = build_synthetic_dir(quote_dir, args.sessions, 'quote',
                                      args.prints)
        build_synthetic_dir(bucket_dir, min(args.sessions, 10), 'bucket',
                            args.prints)
        print(f"  wrote {args.sessions} quote-mode and "
              f"{min(args.sessions, 10)} bucket-mode sessions")

        ticks = test_decoder(quote_dir, truth_q)
        test_direction_recovery(quote_dir, bucket_dir)
        test_bars(ticks)
        df = test_level_ledger(quote_dir)
        test_forward_outcomes(df)
        test_purging()
        test_meta_labels(df)
        test_event_study(df)

    except SelfTestFailure as exc:
        failed = True
        print(f"\n{'='*72}")
        print(f"  SELF-TEST FAILED: {exc}")
        print(f"{'='*72}")
    finally:
        if args.keep is None:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"\n  Synthetic data kept at {root}")

    if not failed:
        print(f"\n{'='*72}")
        print("  ALL CHECKS PASSED")
        print(f"{'='*72}")
        print("  The pipeline is behaving correctly on data with known")
        print("  properties. Safe to point at real .data files.")
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
