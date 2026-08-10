#!/usr/bin/env python3
"""
Generate weekly output files for visualization in candlestick_viewer.html:
- 57-byte tick .data files split by ISO week (from original raw data)
- trades.json sidecar files split by matching week

Usage:
    python3 generate_trade_output.py /path/to/models/ --data /path/to/raw/data/
    python3 generate_trade_output.py /path/to/models/ --data /path/to/raw/data/ --output output/
"""

import argparse
import glob
import os
import sys
import struct
import json
from datetime import datetime, timedelta

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import pandas as pd

DOTNET_EPOCH = datetime(1, 1, 1)
UNIX_EPOCH = datetime(1970, 1, 1)
TICK_RECORD_SIZE = 57
TICK_STRUCT = struct.Struct('<q3d6IB')


def _naive_to_unix(dt):
    """Convert naive datetime to unix seconds matching the viewer's tick-to-time conversion."""
    if hasattr(dt, 'to_pydatetime'):
        dt = dt.to_pydatetime()
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return int((dt - UNIX_EPOCH).total_seconds())


def ticks_to_datetime(ticks):
    return DOTNET_EPOCH + timedelta(microseconds=ticks // 10)


def datetime_to_ticks(dt):
    if hasattr(dt, 'to_pydatetime'):
        dt = dt.to_pydatetime()
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    delta = dt - DOTNET_EPOCH
    return int(delta.total_seconds() * 10_000_000)


def read_raw_tick_records(filepath):
    """Read a 57-byte tick .data file and return list of (datetime, raw_bytes) tuples."""
    filesize = os.path.getsize(filepath)
    num_records = filesize // TICK_RECORD_SIZE
    records = []
    with open(filepath, 'rb') as f:
        for _ in range(num_records):
            raw = f.read(TICK_RECORD_SIZE)
            if len(raw) < TICK_RECORD_SIZE:
                break
            fields = TICK_STRUCT.unpack(raw)
            ts = ticks_to_datetime(fields[0])
            records.append((ts, raw))
    return records


def split_records_by_week(records):
    """Group tick records by ISO year-week. Returns dict of {week_label: [raw_bytes]}."""
    weeks = {}
    for ts, raw in records:
        iso = ts.isocalendar()
        label = f"{iso[0]}-W{iso[1]:02d}"
        if label not in weeks:
            weeks[label] = []
        weeks[label].append(raw)
    return weeks


def write_tick_data(raw_records, output_path):
    """Write raw 57-byte tick records to a .data file."""
    with open(output_path, 'wb') as f:
        for raw in raw_records:
            f.write(raw)
    print(f"  {len(raw_records):,} ticks → {output_path}")


def split_trades_by_week(trades_df):
    """Group trades by ISO week of entry_time."""
    trades_df = trades_df.copy()
    entry_ts = pd.to_datetime(trades_df['entry_time'])
    trades_df['_week'] = entry_ts.dt.isocalendar().year.astype(str) + '-W' + \
                         entry_ts.dt.isocalendar().week.astype(str).str.zfill(2)
    grouped = {}
    for week, group in trades_df.groupby('_week'):
        grouped[week] = group.drop(columns=['_week'])
    return grouped


def trades_to_json(trades_df, output_path):
    """Convert trades DataFrame to JSON for the viewer's marker overlay."""
    trades = []
    for _, row in trades_df.iterrows():
        entry_time = pd.Timestamp(row['entry_time'])
        exit_time = pd.Timestamp(row['exit_time'])

        entry_unix = _naive_to_unix(entry_time)
        exit_unix = _naive_to_unix(exit_time)

        pnl = float(row.get('pnl_points', 0))
        result = 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'SCRATCH')

        trades.append({
            'entry_time': entry_unix,
            'exit_time': exit_unix,
            'direction': row['direction'],
            'entry_price': float(row['entry_price']),
            'sl_price': float(row['sl_price']),
            'tp_price': float(row['tp_price']),
            'exit_price': float(row['exit_price']),
            'exit_reason': row['exit_reason'],
            'entry_reason': str(row.get('entry_reason', '')),
            'pnl_points': pnl,
            'result': result,
        })

    with open(output_path, 'w') as f:
        json.dump({'trades': trades, 'generated_at': datetime.now().isoformat()}, f, indent=2)

    n_wins = sum(1 for t in trades if t['result'] == 'WIN')
    n_losses = sum(1 for t in trades if t['result'] == 'LOSS')
    print(f"  {len(trades)} trades ({n_wins}W/{n_losses}L) → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate tick .data and trades.json for candlestick viewer.')
    parser.add_argument('models_dir', help='Directory with models and backtest results')
    parser.add_argument('--data', required=True, help='Directory with raw 57-byte tick .data files')
    parser.add_argument('--trades', default=None, help='Backtest trades CSV')
    parser.add_argument('--output', default=None, help='Output directory')
    parser.add_argument('--per-trade', action='store_true',
                        help='Generate one .data + .json per trade with context')
    parser.add_argument('--context-bars', type=int, default=30,
                        help='1-min bars of context before entry (default: 30)')

    args = parser.parse_args()

    output_dir = args.output or os.path.join(args.models_dir, 'viewer_output')
    os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(args.data, '*.data')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: No .data files in {args.data}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {len(files)} tick file(s)...")
    all_records = []
    for fp in files:
        recs = read_raw_tick_records(fp)
        print(f"  {os.path.basename(fp)}: {len(recs):,} ticks")
        all_records.extend(recs)

    all_records.sort(key=lambda r: r[0])
    print(f"Total: {len(all_records):,} ticks")

    trades_path = args.trades or os.path.join(args.models_dir, 'backtest_trades.csv')

    if args.per_trade:
        _generate_per_trade(all_records, trades_path, output_dir, args.context_bars)
    else:
        _generate_weekly(all_records, trades_path, output_dir)


def _generate_weekly(all_records, trades_path, output_dir):
    """Split ticks by ISO week and write weekly .data + .json files."""
    weeks = split_records_by_week(all_records)
    print(f"\nGenerating weekly tick .data files ({len(weeks)} weeks)...")
    for label in sorted(weeks.keys()):
        out_path = os.path.join(output_dir, f"{label}.data")
        write_tick_data(weeks[label], out_path)

    if os.path.isfile(trades_path) and os.path.getsize(trades_path) > 0:
        trades_df = pd.read_csv(trades_path)
        trade_weeks = split_trades_by_week(trades_df)
        print(f"\nGenerating weekly trades.json files...")
        for label in sorted(trade_weeks.keys()):
            json_path = os.path.join(output_dir, f"{label}.json")
            trades_to_json(trade_weeks[label], json_path)

        for label in sorted(weeks.keys()):
            json_path = os.path.join(output_dir, f"{label}.json")
            if not os.path.isfile(json_path):
                with open(json_path, 'w') as f:
                    json.dump({'trades': [], 'generated_at': datetime.now().isoformat()}, f, indent=2)

    print(f"\nWeekly output in: {output_dir}")


def _generate_per_trade(all_records, trades_path, output_dir, context_bars):
    """Generate one .data + .json per trade with context window."""
    if not os.path.isfile(trades_path) or os.path.getsize(trades_path) == 0:
        print(f"ERROR: No trades file at {trades_path}", file=sys.stderr)
        sys.exit(1)

    trades_dir = os.path.join(output_dir, 'trades')
    os.makedirs(trades_dir, exist_ok=True)

    trades_df = pd.read_csv(trades_path)
    print(f"\nGenerating per-trade files ({len(trades_df)} trades, {context_bars} bars context)...")

    timestamps = np.array([r[0].timestamp() for r in all_records])
    after_exit_bars = 5

    for idx, row in trades_df.iterrows():
        entry_time = pd.Timestamp(row['entry_time'])
        exit_time = pd.Timestamp(row['exit_time'])
        direction = row['direction']
        result = 'W' if row.get('pnl_points', 0) > 0 else 'L'

        window_start = entry_time - timedelta(minutes=context_bars)
        window_end = exit_time + timedelta(minutes=after_exit_bars)

        i_start = int(np.searchsorted(timestamps, window_start.timestamp(), side='left'))
        i_end = int(np.searchsorted(timestamps, window_end.timestamp(), side='right'))

        if i_end <= i_start:
            continue

        entry_str = entry_time.strftime('%Y%m%d_%H%M')
        prefix = f"trade_{idx+1:04d}_{entry_str}_{direction[0]}_{result}"

        data_path = os.path.join(trades_dir, f"{prefix}.data")
        with open(data_path, 'wb') as f:
            for rec_idx in range(i_start, i_end):
                f.write(all_records[rec_idx][1])

        entry_unix = _naive_to_unix(entry_time)
        exit_unix = _naive_to_unix(exit_time)
        pnl = float(row.get('pnl_points', 0))
        trade_json = {
            'trades': [{
                'entry_time': entry_unix,
                'exit_time': exit_unix,
                'direction': direction,
                'entry_price': float(row['entry_price']),
                'sl_price': float(row['sl_price']),
                'tp_price': float(row['tp_price']),
                'exit_price': float(row['exit_price']),
                'exit_reason': row['exit_reason'],
                'entry_reason': str(row.get('entry_reason', '')),
                'pnl_points': pnl,
                'result': 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'SCRATCH'),
            }],
        }
        json_path = os.path.join(trades_dir, f"{prefix}.json")
        with open(json_path, 'w') as f:
            json.dump(trade_json, f, indent=2)

        print(f"  {prefix}: {i_end - i_start:,} ticks")

    print(f"\nPer-trade output in: {trades_dir}")


if __name__ == '__main__':
    main()
