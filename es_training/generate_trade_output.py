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
TICK_RECORD_SIZE = 57
TICK_STRUCT = struct.Struct('<q3d6IB')


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

        entry_unix = int(entry_time.timestamp()) if entry_time.tzinfo else int(entry_time.tz_localize('UTC').timestamp())
        exit_unix = int(exit_time.timestamp()) if exit_time.tzinfo else int(exit_time.tz_localize('UTC').timestamp())

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
        description='Generate weekly tick .data and trades.json for candlestick viewer.')
    parser.add_argument('models_dir', help='Directory with models and backtest results')
    parser.add_argument('--data', required=True, help='Directory with raw 57-byte tick .data files')
    parser.add_argument('--trades', default=None, help='Backtest trades CSV')
    parser.add_argument('--output', default=None, help='Output directory')

    args = parser.parse_args()

    output_dir = args.output or os.path.join(args.models_dir, 'viewer_output')
    os.makedirs(output_dir, exist_ok=True)

    # Read all raw tick files
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

    # Split by ISO week and write
    weeks = split_records_by_week(all_records)
    print(f"\nGenerating weekly tick .data files ({len(weeks)} weeks)...")
    for label in sorted(weeks.keys()):
        out_path = os.path.join(output_dir, f"{label}.data")
        write_tick_data(weeks[label], out_path)

    # Split trades by week
    trades_path = args.trades or os.path.join(args.models_dir, 'backtest_trades.csv')
    if os.path.isfile(trades_path) and os.path.getsize(trades_path) > 0:
        trades_df = pd.read_csv(trades_path)
        trade_weeks = split_trades_by_week(trades_df)
        print(f"\nGenerating weekly trades.json files...")
        for label in sorted(trade_weeks.keys()):
            json_path = os.path.join(output_dir, f"{label}.json")
            trades_to_json(trade_weeks[label], json_path)

        # Also write any weeks with ticks but no trades as empty json
        for label in sorted(weeks.keys()):
            json_path = os.path.join(output_dir, f"{label}.json")
            if not os.path.isfile(json_path):
                with open(json_path, 'w') as f:
                    json.dump({'trades': [], 'generated_at': datetime.now().isoformat()}, f, indent=2)
    else:
        print(f"  No trades file found at {trades_path}, skipping trades.json")

    print(f"\nOutput in: {output_dir}")
    print(f"  Load {label}.data with 'tick .data' button in viewer")
    print(f"  Load {label}.json for matching trade markers")


if __name__ == '__main__':
    main()
