#!/usr/bin/env python3
"""
Generate output files for visualization in candlestick_viewer.html:
- 60-byte minute-bar .data file (compatible with existing viewer)
- trades.json sidecar file with entry/SL/TP/exit annotations

Usage:
    python3 generate_trade_output.py /path/to/models/ --trades backtest_trades.csv
    python3 generate_trade_output.py /path/to/models/ --data /path/to/data/ --output output/
"""

import argparse
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
MINUTE_RECORD_SIZE = 60
MINUTE_STRUCT = struct.Struct('<q4d3I')  # timestamp(8) + OHLC(32) + sellCount(4) + buyCount(4) + volume(4) = 60 fake but matches


def datetime_to_ticks(dt):
    """Convert Python datetime to .NET ticks."""
    if hasattr(dt, 'to_pydatetime'):
        dt = dt.to_pydatetime()
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    delta = dt - DOTNET_EPOCH
    return int(delta.total_seconds() * 10_000_000)


def bars_to_binary(bars, output_path):
    """
    Write bars DataFrame as 60-byte minute-bar binary file
    compatible with candlestick_viewer.html's minute decoder.

    Format per record (60 bytes):
        int64   timestamp (.NET ticks, little-endian)
        float64 high
        float64 low
        float64 open
        float64 close
        uint32  sell_count
        uint32  buy_count
        uint32  volume
    """
    with open(output_path, 'wb') as f:
        for i in range(len(bars)):
            row = bars.iloc[i]
            ts = bars.index[i]
            ticks = datetime_to_ticks(ts)

            high = float(row['high'])
            low = float(row['low'])
            open_p = float(row['open'])
            close_p = float(row['close'])

            sell_count = int(row.get('sell_volume', 0))
            buy_count = int(row.get('buy_volume', 0))
            volume = int(row.get('volume', 0))

            record = MINUTE_STRUCT.pack(ticks, high, low, open_p, close_p,
                                        sell_count, buy_count, volume)
            f.write(record)

    print(f"  Written {len(bars)} bars ({len(bars) * MINUTE_RECORD_SIZE} bytes) → {output_path}")


def trades_to_json(trades_df, output_path):
    """
    Convert trades DataFrame to JSON for the viewer's marker overlay.

    Each trade entry:
    {
        "entry_time": unix_seconds,
        "exit_time": unix_seconds,
        "direction": "LONG" | "SHORT",
        "entry_price": float,
        "sl_price": float,
        "tp_price": float,
        "exit_price": float,
        "exit_reason": "TP" | "SL" | "EOD" | "TIMEOUT",
        "pnl_points": float,
        "result": "WIN" | "LOSS" | "SCRATCH"
    }
    """
    trades = []
    for _, row in trades_df.iterrows():
        entry_time = pd.Timestamp(row['entry_time'])
        exit_time = pd.Timestamp(row['exit_time'])

        # Convert to unix seconds for the viewer
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
            'pnl_points': pnl,
            'result': result,
        })

    with open(output_path, 'w') as f:
        json.dump({'trades': trades, 'generated_at': datetime.now().isoformat()}, f, indent=2)

    n_wins = sum(1 for t in trades if t['result'] == 'WIN')
    n_losses = sum(1 for t in trades if t['result'] == 'LOSS')
    print(f"  Written {len(trades)} trades ({n_wins}W / {n_losses}L) → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate binary .data and trades.json for candlestick viewer.')
    parser.add_argument('models_dir', help='Directory with models and features')
    parser.add_argument('--trades', default=None, help='Backtest trades CSV')
    parser.add_argument('--features', default=None, help='Features file with OHLC bars')
    parser.add_argument('--output', default=None, help='Output directory')

    args = parser.parse_args()

    output_dir = args.output or os.path.join(args.models_dir, 'viewer_output')
    os.makedirs(output_dir, exist_ok=True)

    # Load features (bars with OHLC)
    if args.features:
        fp = args.features
    else:
        fp = os.path.join(args.models_dir, 'training_features.parquet')
        if not os.path.isfile(fp):
            fp = fp.replace('.parquet', '.csv')

    print("Loading bar data...")
    if fp.endswith('.parquet'):
        bars = pd.read_parquet(fp)
    else:
        bars = pd.read_csv(fp, index_col=0, parse_dates=True)

    # Write binary .data file
    data_path = os.path.join(output_dir, 'bars.data')
    print("Generating binary .data file...")
    bars_to_binary(bars, data_path)

    # Load and convert trades
    trades_path = args.trades or os.path.join(args.models_dir, 'backtest_trades.csv')
    if os.path.isfile(trades_path):
        print("Generating trades.json...")
        trades_df = pd.read_csv(trades_path)
        json_path = os.path.join(output_dir, 'trades.json')
        trades_to_json(trades_df, json_path)
    else:
        print(f"  No trades file found at {trades_path}, skipping trades.json")

    print(f"\nOutput files in: {output_dir}")
    print(f"  Load bars.data in candlestick_viewer.html (minute button)")
    print(f"  Load trades.json for trade markers overlay")


if __name__ == '__main__':
    main()
