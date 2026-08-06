#!/usr/bin/env python3
# python3 decode_minute_data.py minute/20250611.data --all
# python3 decode_minute_data.py minute/20250611.data --csv out.csv
"""
Decoder for minute-bar .data binary files (e.g., minute/20250611.data).

Thought Process
===============
1. IDENTIFYING THE FORMAT:
   - Same family as the 57-byte tick files, but a different record size.
   - File size 81,300; .NET DateTime ticks (high byte 0x08) repeat every
     60 bytes. 81300 / 60 = 1,355 records exactly.
   - Timestamps are 1-minute aligned (occasional 2–4 minute gaps when idle).

2. DETERMINING FIELD LAYOUT:
   - Bytes 0-7: little-endian int64 .NET DateTime ticks.
   - Bytes 8-39: four IEEE 754 doubles. Always High >= Low; every OHLC
     permutation with High=field0 / Low=field1 is valid. Continuity
     (prev close ≈ next open) selects Open=field2, Close=field3.
   - Bytes 40-59: five little-endian uint32. Fields 0+1 ≈ field 2 (volume);
     field 3 is the running cumulative volume; field 4 is always 0.

3. UNDERSTANDING THE SEMANTICS:
   - OHLC minute bar: High, Low, Open, Close (tick size 0.25).
   - First two integers are directional txn counts. Correlation of
     (i1 - i0) with (close - open) is positive (~0.24) when i0=SELL and
     i1=BUY — matching the tick-file convention (trade at ask=BUY,
     trade at bid=SELL).
   - volume ≈ buy_count + sell_count; cumulative_volume increments by
     volume each bar (session cumulative from the first bar).

4. TIMESTAMP FORMAT:
   - Same as tick files: .NET DateTime ticks = 100 ns since 0001-01-01.
"""

import struct
import sys
import os
import csv
from datetime import datetime, timedelta


RECORD_SIZE = 60
RECORD_FORMAT = '<q4d5I'  # int64 + 4 doubles + 5 uint32
RECORD_STRUCT = struct.Struct(RECORD_FORMAT)

DOTNET_EPOCH = datetime(1, 1, 1)


def ticks_to_datetime(ticks):
    """Convert .NET DateTime ticks to Python datetime."""
    return DOTNET_EPOCH + timedelta(microseconds=ticks // 10)


def decode_record(raw_bytes):
    """Decode a single 60-byte minute bar into a dictionary."""
    fields = RECORD_STRUCT.unpack(raw_bytes)
    high, low, open_, close = fields[1:5]
    sell_count = fields[5]
    buy_count = fields[6]
    volume = fields[7]
    return {
        'timestamp': ticks_to_datetime(fields[0]),
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'sell_count': sell_count,
        'buy_count': buy_count,
        'volume': volume,
        'volume_delta': buy_count - sell_count,
        'cumulative_volume': fields[8],
        'reserved': fields[9],
    }


def decode_file(filepath):
    """Generator that yields decoded minute bars from a binary .data file.

    Adds cumulative_volume_delta (running buy - sell from the start of the
    file / session).
    """
    filesize = os.path.getsize(filepath)
    if filesize % RECORD_SIZE != 0:
        print(f"WARNING: File size ({filesize}) is not evenly divisible by "
              f"record size ({RECORD_SIZE}). Trailing bytes will be ignored.",
              file=sys.stderr)

    num_records = filesize // RECORD_SIZE
    cum_vd = 0

    with open(filepath, 'rb') as f:
        for i in range(num_records):
            raw = f.read(RECORD_SIZE)
            if len(raw) < RECORD_SIZE:
                break
            rec = decode_record(raw)
            cum_vd += rec['volume_delta']
            rec['cumulative_volume_delta'] = cum_vd
            yield i, rec


def print_text(filepath, limit=None):
    """Print decoded records in a human-readable text table."""
    header = (f"{'#':>6} | {'Timestamp':<19} | {'Open':>9} | {'High':>9} | "
              f"{'Low':>9} | {'Close':>9} | {'Buy':>5} | {'Sell':>5} | "
              f"{'Vol':>5} | {'Vol Δ':>6} | {'Cum Vol Δ':>9} | "
              f"{'Cum Vol':>8}")
    print(header)
    print('-' * len(header))

    for i, rec in decode_file(filepath):
        if limit and i >= limit:
            break
        ts_str = rec['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i:>6} | {ts_str:<19} | {rec['open']:>9.2f} | "
              f"{rec['high']:>9.2f} | {rec['low']:>9.2f} | "
              f"{rec['close']:>9.2f} | {rec['buy_count']:>5} | "
              f"{rec['sell_count']:>5} | {rec['volume']:>5} | "
              f"{rec['volume_delta']:>6} | "
              f"{rec['cumulative_volume_delta']:>9} | "
              f"{rec['cumulative_volume']:>8}")


def export_csv(filepath, output_path):
    """Export decoded records to a CSV file."""
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'record_num', 'timestamp', 'open', 'high', 'low', 'close',
            'buy_count', 'sell_count', 'volume', 'volume_delta',
            'cumulative_volume_delta', 'cumulative_volume',
        ])
        for i, rec in decode_file(filepath):
            writer.writerow([
                i,
                rec['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                f"{rec['open']:.2f}",
                f"{rec['high']:.2f}",
                f"{rec['low']:.2f}",
                f"{rec['close']:.2f}",
                rec['buy_count'],
                rec['sell_count'],
                rec['volume'],
                rec['volume_delta'],
                rec['cumulative_volume_delta'],
                rec['cumulative_volume'],
            ])
    print(f"Exported {i + 1} records to {output_path}")


def print_summary(filepath):
    """Print a summary of the file contents."""
    filesize = os.path.getsize(filepath)
    num_records = filesize // RECORD_SIZE

    first_rec = last_rec = None
    total_vol = 0
    max_vol = 0
    price_min = float('inf')
    price_max = float('-inf')
    up_bars = down_bars = flat_bars = 0
    final_cum_vd = 0

    for i, rec in decode_file(filepath):
        if first_rec is None:
            first_rec = rec
        last_rec = rec
        total_vol += rec['volume']
        max_vol = max(max_vol, rec['volume'])
        price_min = min(price_min, rec['low'])
        price_max = max(price_max, rec['high'])
        final_cum_vd = rec['cumulative_volume_delta']
        change = rec['close'] - rec['open']
        if change > 1e-9:
            up_bars += 1
        elif change < -1e-9:
            down_bars += 1
        else:
            flat_bars += 1

    print("=" * 60)
    print(f"  File:            {filepath}")
    print(f"  File size:       {filesize:,} bytes")
    print(f"  Record size:     {RECORD_SIZE} bytes")
    print(f"  Total bars:      {num_records:,}")
    print(f"  Time range:      {first_rec['timestamp']} → {last_rec['timestamp']}")
    print(f"  Duration:        {last_rec['timestamp'] - first_rec['timestamp']}")
    print(f"  Price range:     {price_min:.2f} → {price_max:.2f}")
    print(f"  Tick size:       0.25")
    print(f"  Total volume:    {total_vol:,}")
    print(f"  Max vol/bar:     {max_vol}")
    print(f"  Up bars:         {up_bars:,} ({up_bars/num_records*100:.1f}%)")
    print(f"  Down bars:       {down_bars:,} ({down_bars/num_records*100:.1f}%)")
    print(f"  Flat bars:       {flat_bars:,} ({flat_bars/num_records*100:.1f}%)")
    print(f"  Final cum vol Δ: {final_cum_vd:,}")
    print(f"  Final cum vol:   {last_rec['cumulative_volume']:,}")
    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Decode binary minute-bar .data files to human-readable format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s minute/20250611.data                  # Summary + first 50 bars
  %(prog)s minute/20250611.data --all            # Print all bars
  %(prog)s minute/20250611.data -n 100           # Print first 100 bars
  %(prog)s minute/20250611.data --csv out.csv    # Export all bars to CSV
  %(prog)s minute/20250611.data --summary        # Print only the summary
        """)
    parser.add_argument('input_file', help='Path to the binary minute .data file')
    parser.add_argument('-n', '--limit', type=int, default=50,
                        help='Number of records to display (default: 50)')
    parser.add_argument('--all', action='store_true',
                        help='Display all records (overrides -n)')
    parser.add_argument('--csv', metavar='OUTPUT',
                        help='Export decoded data to a CSV file')
    parser.add_argument('--summary', action='store_true',
                        help='Print only the summary, no records')

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    print_summary(args.input_file)
    print()

    if args.csv:
        export_csv(args.input_file, args.csv)
    elif not args.summary:
        limit = None if args.all else args.limit
        print_text(args.input_file, limit=limit)


if __name__ == '__main__':
    main()
