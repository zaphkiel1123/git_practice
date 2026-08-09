#!/usr/bin/env python3
"""
Feature Extraction Pipeline for .data binary transaction files.

Reads all .data files in a directory, computes per-window features
(order flow, intensity, momentum, volatility), constructs labels
(direction and magnitude), and outputs a Parquet or CSV file ready for modeling.

Usage:
    python3 feature_pipeline.py /path/to/data/dir --window 1min --horizon 5min
    python3 feature_pipeline.py /path/to/data/dir --window 1min --horizon 5min --output features.csv
"""

import struct
import os
import sys
import argparse
import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================
# Binary decoder (same logic as decode_data.py)
# ============================================================

RECORD_SIZE = 57
RECORD_STRUCT = struct.Struct('<q3d6IB')
DOTNET_EPOCH = datetime(1, 1, 1)


def ticks_to_datetime(ticks):
    return DOTNET_EPOCH + timedelta(microseconds=ticks // 10)


def infer_direction(price, level_low, level_high):
    if abs(price - level_high) < 1e-9:
        return 1   # BUY
    elif abs(price - level_low) < 1e-9:
        return -1  # SELL
    else:
        return 0   # UNKNOWN


def decode_file_to_dataframe(filepath):
    """Decode a binary .data file into a pandas DataFrame."""
    filesize = os.path.getsize(filepath)
    num_records = filesize // RECORD_SIZE

    timestamps = []
    prices = []
    level_lows = []
    level_highs = []
    directions = []
    txn_counts = []
    cumulative_txns = []

    with open(filepath, 'rb') as f:
        for _ in range(num_records):
            raw = f.read(RECORD_SIZE)
            if len(raw) < RECORD_SIZE:
                break
            fields = RECORD_STRUCT.unpack(raw)

            timestamps.append(ticks_to_datetime(fields[0]))
            prices.append(fields[1])
            level_lows.append(fields[2])
            level_highs.append(fields[3])
            directions.append(infer_direction(fields[1], fields[2], fields[3]))
            txn_counts.append(fields[4])
            cumulative_txns.append(fields[6])

    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'level_low': level_lows,
        'level_high': level_highs,
        'direction': directions,
        'txn_count': txn_counts,
        'cumulative_txn': cumulative_txns,
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['source_file'] = os.path.basename(filepath)
    return df


# ============================================================
# Feature Engineering
# ============================================================

def compute_window_features(df, window='1min'):
    """
    Aggregate tick-level data into time-windowed bars with features.

    Features computed per window:
    - OHLC prices
    - Total volume (sum of txn_count)
    - Number of trade events
    - Order flow imbalance: (buy_vol - sell_vol) / total_vol
    - Trade intensity: events per second
    - Spread distribution: fraction of trades with spread=2 vs spread=1
    - Volume clustering: max single txn_count in window
    - Tick direction runs: longest consecutive same-direction streak
    - VWAP: volume-weighted average price
    """
    df = df.set_index('timestamp').sort_index()

    # Pre-compute trade-level fields
    df['spread'] = df['level_high'] - df['level_low']
    df['buy_vol'] = df['txn_count'].where(df['direction'] == 1, 0)
    df['sell_vol'] = df['txn_count'].where(df['direction'] == -1, 0)
    df['price_x_vol'] = df['price'] * df['txn_count']

    # Resample into windows
    resampled = df.resample(window)

    bars = pd.DataFrame()
    bars['open'] = resampled['price'].first()
    bars['high'] = resampled['price'].max()
    bars['low'] = resampled['price'].min()
    bars['close'] = resampled['price'].last()
    bars['volume'] = resampled['txn_count'].sum()
    bars['num_events'] = resampled['price'].count()

    # Order flow imbalance
    bars['buy_volume'] = resampled['buy_vol'].sum()
    bars['sell_volume'] = resampled['sell_vol'].sum()
    bars['flow_imbalance'] = (
        (bars['buy_volume'] - bars['sell_volume']) /
        bars['volume'].replace(0, np.nan)
    )

    # Trade intensity (events per second)
    window_seconds = pd.Timedelta(window).total_seconds()
    bars['intensity'] = bars['num_events'] / window_seconds

    # Spread distribution (fraction with spread >= 2)
    bars['wide_spread_frac'] = resampled['spread'].apply(
        lambda x: (x >= 2.0).sum() / max(len(x), 1)
    )

    # Volume clustering (max single-event txn_count in the window)
    bars['max_single_txn'] = resampled['txn_count'].max()

    # VWAP
    bars['vwap'] = resampled['price_x_vol'].sum() / bars['volume'].replace(0, np.nan)

    # Tick direction longest run
    bars['max_buy_run'] = resampled['direction'].apply(_max_consecutive, target=1)
    bars['max_sell_run'] = resampled['direction'].apply(_max_consecutive, target=-1)

    # Per-level imbalance features (3:1 ratio = imbalance per footprint convention)
    imb_features = _compute_all_imbalance_features(df, window)
    for col in imb_features.columns:
        bars[col] = imb_features[col]
    bars['net_imbalance'] = bars['buy_imbalance_count'] - bars['sell_imbalance_count']

    # Transaction speed features
    bars['max_events_per_tick'] = resampled['price'].apply(lambda s: s.value_counts().max() if len(s) > 0 else 0)
    total_vol_by_level = resampled.apply(lambda g: g.groupby('price')['txn_count'].sum().max() / g['txn_count'].sum() if len(g) > 0 and g['txn_count'].sum() > 0 else 0)
    bars['level_concentration'] = total_vol_by_level

    # Drop windows with no data
    bars = bars.dropna(subset=['open'])

    return bars


TICK_SIZE_FP = 0.25


def _compute_all_imbalance_features(df, window):
    """Compute per-bar imbalance features by iterating over resampled groups."""
    records = []
    for ts, group in df.resample(window):
        if len(group) == 0:
            continue
        levels = _compute_level_buysell(group)
        records.append({
            'timestamp': ts,
            'buy_imbalance_count': _count_buy_imb(levels),
            'sell_imbalance_count': _count_sell_imb(levels),
            'buy_imbalance_cluster': _max_buy_cluster(levels),
            'sell_imbalance_cluster': _max_sell_cluster(levels),
        })
    result = pd.DataFrame(records).set_index('timestamp')
    return result


def _compute_level_buysell(group):
    """Build dict of {price: [buy_vol, sell_vol]} for a bar's ticks."""
    levels = {}
    for _, row in group.iterrows():
        p = row['price']
        vol = row['txn_count']
        if p not in levels:
            levels[p] = [0, 0]
        if row['direction'] == 1:
            levels[p][0] += vol
        elif row['direction'] == -1:
            levels[p][1] += vol
    return levels


def _count_buy_imb(levels):
    """Count price levels with buy_at_P >= 3 * sell_at_(P - tick)."""
    count = 0
    for p, (buy, sell) in levels.items():
        below = levels.get(p - TICK_SIZE_FP)
        if below and below[1] > 0 and buy >= 3 * below[1]:
            count += 1
    return count


def _count_sell_imb(levels):
    """Count price levels with sell_at_P >= 3 * buy_at_(P + tick)."""
    count = 0
    for p, (buy, sell) in levels.items():
        above = levels.get(p + TICK_SIZE_FP)
        if above and above[0] > 0 and sell >= 3 * above[0]:
            count += 1
    return count


def _max_buy_cluster(levels):
    """Longest consecutive stack of buy imbalances (prices ascending)."""
    prices = sorted(levels.keys())
    max_cluster = 0
    current = 0
    for p in prices:
        buy = levels[p][0]
        below = levels.get(p - TICK_SIZE_FP)
        if below and below[1] > 0 and buy >= 3 * below[1]:
            current += 1
            max_cluster = max(max_cluster, current)
        else:
            current = 0
    return max_cluster


def _max_sell_cluster(levels):
    """Longest consecutive stack of sell imbalances (prices descending)."""
    prices = sorted(levels.keys(), reverse=True)
    max_cluster = 0
    current = 0
    for p in prices:
        sell = levels[p][1]
        above = levels.get(p + TICK_SIZE_FP)
        if above and above[0] > 0 and sell >= 3 * above[0]:
            current += 1
            max_cluster = max(max_cluster, current)
        else:
            current = 0
    return max_cluster


def _max_consecutive(series, target):
    """Find the longest consecutive run of `target` in a series."""
    if len(series) == 0:
        return 0
    vals = (series == target).values
    max_run = 0
    current = 0
    for v in vals:
        if v:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def add_rolling_features(bars, lookback_windows=[3, 5, 10]):
    """
    Add rolling/momentum/volatility features computed from prior bars.
    All features are strictly backward-looking (no data leakage).
    Uses mid-price (high+low)/2 instead of close as representative price.
    """
    bars['mid'] = (bars['high'] + bars['low']) / 2
    bars['return_1'] = bars['mid'].pct_change()

    for w in lookback_windows:
        prefix = f'roll_{w}'

        # Volatility: rolling std of 1-bar returns
        bars[f'{prefix}_volatility'] = bars['return_1'].rolling(w).std()

        # Mean flow imbalance over last w bars
        bars[f'{prefix}_flow_imb'] = bars['flow_imbalance'].rolling(w).mean()

        # Mean intensity over last w bars
        bars[f'{prefix}_intensity'] = bars['intensity'].rolling(w).mean()

        # Volume trend: current volume vs rolling mean
        rolling_vol = bars['volume'].rolling(w).mean()
        bars[f'{prefix}_vol_ratio'] = bars['volume'] / rolling_vol.replace(0, np.nan)

    return bars


def add_time_features(bars):
    """Add time-of-day cyclical encoding and session indicators.
    Timestamps are in US Central (CME/Chicago) time."""
    # Convert to Eastern for session logic
    ts_et = bars.index.tz_localize('America/Chicago').tz_convert('America/New_York')

    # Cyclical hour encoding (ET)
    hour_frac = ts_et.hour + ts_et.minute / 60.0
    bars['hour_sin'] = np.sin(2 * np.pi * hour_frac / 24)
    bars['hour_cos'] = np.cos(2 * np.pi * hour_frac / 24)

    # Session indicators based on ET hours
    bars['session_overnight'] = ((ts_et.hour >= 18) | (ts_et.hour < 4)).astype(int)
    bars['session_premarket'] = ((ts_et.hour >= 4) & (ts_et.hour < 9)).astype(int)
    bars['session_rth'] = (((ts_et.hour * 100 + ts_et.minute) >= 930) &
                           ((ts_et.hour * 100 + ts_et.minute) < 1600)).astype(int)

    return bars


# ============================================================
# Label Construction
# ============================================================

def add_labels(bars, horizon='5min'):
    """
    Add forward-looking labels for supervised learning.

    Labels:
    - direction_label: +1 if price goes up, -1 if down, 0 if flat
    - magnitude_label: absolute price change in the next `horizon` bars
    - future_return: signed return over horizon (for analysis)
    """
    horizon_bars = int(pd.Timedelta(horizon) / pd.Timedelta(bars.index.freq or
                       bars.index.to_series().diff().median()))

    # Future close price N bars ahead
    bars['future_close'] = bars['close'].shift(-horizon_bars)
    bars['future_return'] = (bars['future_close'] - bars['close']) / bars['close']
    bars['magnitude_label'] = (bars['future_close'] - bars['close']).abs()

    # Direction: +1 up, -1 down, 0 unchanged (keep as float to handle NaN at edges)
    bars['direction_label'] = np.sign(bars['future_close'] - bars['close'])

    return bars


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(data_dir, window='1min', horizon='5min', output=None):
    """Run full feature extraction + labeling pipeline."""

    # Find all .data files
    pattern = os.path.join(data_dir, '*.data')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"ERROR: No .data files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} data file(s):")
    for f in files:
        print(f"  {f} ({os.path.getsize(f):,} bytes)")

    # Decode all files
    print(f"\nDecoding...")
    all_dfs = []
    for filepath in files:
        df = decode_file_to_dataframe(filepath)
        all_dfs.append(df)
        print(f"  {os.path.basename(filepath)}: {len(df):,} records, "
              f"{df['timestamp'].min()} → {df['timestamp'].max()}")

    raw_df = pd.concat(all_dfs, ignore_index=True)
    raw_df = raw_df.sort_values('timestamp').reset_index(drop=True)
    print(f"\nTotal raw records: {len(raw_df):,}")

    # Compute windowed features
    print(f"\nComputing {window} bar features...")
    bars = compute_window_features(raw_df, window=window)
    print(f"  Generated {len(bars)} bars")

    # Add rolling/momentum features
    print("Adding rolling features...")
    bars = add_rolling_features(bars)

    # Add time features
    print("Adding time features...")
    bars = add_time_features(bars)

    # Add labels
    print(f"Adding labels (horizon={horizon})...")
    bars = add_labels(bars, horizon=horizon)

    # Drop rows with NaN from rolling windows and labels
    complete_bars = bars.dropna()
    print(f"\nComplete rows (after dropping NaN): {len(complete_bars)} "
          f"(dropped {len(bars) - len(complete_bars)} edge rows)")

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"  FEATURE MATRIX SUMMARY")
    print(f"{'='*60}")
    print(f"  Rows:     {len(complete_bars)}")
    print(f"  Columns:  {len(complete_bars.columns)}")
    print(f"  Time:     {complete_bars.index.min()} → {complete_bars.index.max()}")
    print(f"  Label distribution (direction):")
    vc = complete_bars['direction_label'].value_counts()
    for label, count in vc.items():
        pct = count / len(complete_bars) * 100
        name = {1: 'UP', -1: 'DOWN', 0: 'FLAT'}[label]
        print(f"    {name:>5}: {count:>6} ({pct:.1f}%)")
    print(f"  Magnitude (ticks): mean={complete_bars['magnitude_label'].mean():.3f}, "
          f"std={complete_bars['magnitude_label'].std():.3f}, "
          f"median={complete_bars['magnitude_label'].median():.3f}")
    print(f"{'='*60}")

    # Save output
    if output is None:
        output = os.path.join(data_dir, 'features.csv')

    if output.endswith('.parquet'):
        try:
            complete_bars.to_parquet(output)
        except ImportError:
            output = output.replace('.parquet', '.csv')
            print(f"  pyarrow not available, falling back to CSV")
            complete_bars.to_csv(output)
    else:
        complete_bars.to_csv(output)

    print(f"\nSaved feature matrix to: {output}")
    print(f"File size: {os.path.getsize(output):,} bytes")

    # Also print feature columns for reference
    print(f"\nFeature columns ({len(complete_bars.columns)}):")
    for col in complete_bars.columns:
        print(f"  - {col}")

    return complete_bars


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from binary .data files for ML modeling.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s .                              # Process .data files in current dir
  %(prog)s /path/to/data --window 5min    # Use 5-minute bars
  %(prog)s . --horizon 10min              # Predict 10 minutes ahead
  %(prog)s . --output my_features.parquet # Custom output path
        """)
    parser.add_argument('data_dir', help='Directory containing .data files')
    parser.add_argument('--window', default='1min',
                        help='Aggregation window size (default: 1min)')
    parser.add_argument('--horizon', default='5min',
                        help='Prediction horizon for labels (default: 5min)')
    parser.add_argument('--output', default=None,
                        help='Output parquet file path (default: data_dir/features.parquet)')

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: Directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    run_pipeline(args.data_dir, window=args.window,
                 horizon=args.horizon, output=args.output)


if __name__ == '__main__':
    main()
