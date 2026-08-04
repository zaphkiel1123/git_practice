#!/usr/bin/env python3
"""
Decoder for .data binary transaction files (e.g., 20250909.data).

Thought Process
===============
1. IDENTIFYING THE FORMAT:
   - The file is raw binary with no text header or magic bytes.
   - By searching for repeating byte patterns (specifically the .NET DateTime
     tick signature where the high byte is 0x08 for year ~2025), I discovered
     timestamps appearing at a fixed interval of 57 bytes.
   - File size (786,372) divides evenly by 57, confirming 13,796 records.

2. DETERMINING FIELD LAYOUT:
   - Bytes 0-7: Interpreted as little-endian int64. Values decode to valid
     .NET DateTime ticks in the Sep 2025 range — confirmed as timestamps.
   - Bytes 8-31: Three groups of 8 bytes. Ending in 0x40B9xx pattern which
     is characteristic of IEEE 754 doubles in the 6500-6600 range. These
     are price/level values.
   - Bytes 32-55: Six groups of 4 bytes. Little-endian uint32 integers.
     First two are always equal (transaction count). Third is monotonically
     increasing (cumulative counter). Last three are always zero (reserved).
   - Byte 56: Single byte, always 1 (flag field).

3. UNDERSTANDING THE SEMANTICS:
   - The three doubles represent: trade price, level lower bound, level upper
     bound. The price always equals one of the bounds (trade at bid or ask).
   - The spread (high - low) is always 1.0 or 2.0, defining the "level."
   - The first integer is the KEY FIELD: "number of transactions at a level
     at a time" — how many trades executed at this price level simultaneously.
   - The cumulative counter increments by exactly the transaction count each
     record, confirming it's a running total.
   - Tick size is 0.25 (prices are exact multiples of 0.25).

4. TIMESTAMP FORMAT:
   - .NET DateTime ticks = 100-nanosecond intervals since 0001-01-01 00:00:00.
   - Convert to Python datetime by dividing ticks by 10 to get microseconds,
     then adding to the epoch datetime(1, 1, 1).
"""

import struct
import sys
import os
import csv
from datetime import datetime, timedelta
from pathlib import Path


RECORD_SIZE = 57
RECORD_FORMAT = '<q3d6IB'  # int64 + 3 doubles + 6 uint32 + 1 byte
RECORD_STRUCT = struct.Struct(RECORD_FORMAT)

DOTNET_EPOCH = datetime(1, 1, 1)


def ticks_to_datetime(ticks):
    """Convert .NET DateTime ticks to Python datetime."""
    return DOTNET_EPOCH + timedelta(microseconds=ticks // 10)


def decode_record(raw_bytes):
    """Decode a single 57-byte record into a dictionary."""
    fields = RECORD_STRUCT.unpack(raw_bytes)
    return {
        'timestamp': ticks_to_datetime(fields[0]),
        'price': fields[1],
        'level_low': fields[2],
        'level_high': fields[3],
        'txn_count': fields[4],
        'txn_count_dup': fields[5],
        'cumulative_txn': fields[6],
        'reserved_1': fields[7],
        'reserved_2': fields[8],
        'reserved_3': fields[9],
        'flag': fields[10],
    }


def decode_file(filepath):
    """Generator that yields decoded records from a binary .data file."""
    filesize = os.path.getsize(filepath)
    if filesize % RECORD_SIZE != 0:
        print(f"WARNING: File size ({filesize}) is not evenly divisible by "
              f"record size ({RECORD_SIZE}). Trailing bytes will be ignored.",
              file=sys.stderr)

    num_records = filesize // RECORD_SIZE

    with open(filepath, 'rb') as f:
        for i in range(num_records):
            raw = f.read(RECORD_SIZE)
            if len(raw) < RECORD_SIZE:
                break
            yield i, decode_record(raw)


def print_text(filepath, limit=None):
    """Print decoded records in a human-readable text table."""
    header = (f"{'#':>6} | {'Timestamp':<26} | {'Price':>9} | "
              f"{'Level Low':>9} | {'Level High':>10} | "
              f"{'Txn Count':>9} | {'Cumulative':>10} | {'Flag':>4}")
    print(header)
    print('-' * len(header))

    for i, rec in decode_file(filepath):
        if limit and i >= limit:
            break
        ts_str = rec['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')
        print(f"{i:>6} | {ts_str:<26} | {rec['price']:>9.2f} | "
              f"{rec['level_low']:>9.2f} | {rec['level_high']:>10.2f} | "
              f"{rec['txn_count']:>9} | {rec['cumulative_txn']:>10} | "
              f"{rec['flag']:>4}")


def export_csv(filepath, output_path):
    """Export decoded records to a CSV file."""
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'record_num', 'timestamp', 'price', 'level_low', 'level_high',
            'txn_count', 'cumulative_txn', 'flag'
        ])
        for i, rec in decode_file(filepath):
            writer.writerow([
                i,
                rec['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f'),
                f"{rec['price']:.2f}",
                f"{rec['level_low']:.2f}",
                f"{rec['level_high']:.2f}",
                rec['txn_count'],
                rec['cumulative_txn'],
                rec['flag'],
            ])
    print(f"Exported {i + 1} records to {output_path}")


def print_summary(filepath):
    """Print a summary of the file contents."""
    filesize = os.path.getsize(filepath)
    num_records = filesize // RECORD_SIZE

    first_rec = last_rec = None
    total_txn = 0
    max_txn = 0
    price_min = float('inf')
    price_max = float('-inf')

    for i, rec in decode_file(filepath):
        if first_rec is None:
            first_rec = rec
        last_rec = rec
        total_txn += rec['txn_count']
        max_txn = max(max_txn, rec['txn_count'])
        price_min = min(price_min, rec['price'])
        price_max = max(price_max, rec['price'])

    print("=" * 60)
    print(f"  File:            {filepath}")
    print(f"  File size:       {filesize:,} bytes")
    print(f"  Record size:     {RECORD_SIZE} bytes")
    print(f"  Total records:   {num_records:,}")
    print(f"  Time range:      {first_rec['timestamp']} → {last_rec['timestamp']}")
    print(f"  Duration:        {last_rec['timestamp'] - first_rec['timestamp']}")
    print(f"  Price range:     {price_min:.2f} → {price_max:.2f}")
    print(f"  Tick size:       0.25")
    print(f"  Total txns:      {total_txn:,}")
    print(f"  Max txns/event:  {max_txn}")
    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Decode binary .data transaction files to human-readable format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 20250909.data                  # Print summary + first 50 records
  %(prog)s 20250909.data --all            # Print all records
  %(prog)s 20250909.data -n 100           # Print first 100 records
  %(prog)s 20250909.data --csv out.csv    # Export all records to CSV
  %(prog)s 20250909.data --summary        # Print only the summary
        """)
    parser.add_argument('input_file', help='Path to the binary .data file')
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
