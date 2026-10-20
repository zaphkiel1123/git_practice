#!/usr/bin/env python3
"""
Vectorized decoder for .data binary transaction files.

Derived from old_file/decode_data.py and NN_model/nn_feature_pipeline.py, with
two changes that matter at 3-5 year scale:

1. Decoding is vectorized via a packed numpy structured dtype instead of a
   per-record struct.unpack_from loop (~100x faster on large files).
2. Access is per-session streaming. 3-5 years of tick data does not fit in
   memory, so callers iterate sessions and reduce to bars/features as they go.

Record layout (57 bytes, little-endian, unaligned):
    0-7    int64    .NET DateTime ticks (100ns since 0001-01-01)
    8-15   float64  trade price
    16-23  float64  level lower bound
    24-31  float64  level upper bound
    32-35  uint32   txn_count  (volume for this print)
    36-39  uint32   txn_count duplicate
    40-43  uint32   cumulative txn counter
    44-55  uint32*3 reserved (always zero)
    56     uint8    flag (always 1)

Usage:
    python3 tick_data.py /path/to/data --summary
    python3 tick_data.py /path/to/data --check-format
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

TICK_SIZE = 0.25
RECORD_SIZE = 57

# align=False (numpy default for a list-of-tuples spec) gives an exactly packed
# 57-byte itemsize. Asserted at import so a numpy behaviour change fails loudly
# rather than silently misparsing every file.
REC_DTYPE = np.dtype([
    ('ticks', '<i8'),
    ('price', '<f8'),
    ('level_low', '<f8'),
    ('level_high', '<f8'),
    ('txn_count', '<u4'),
    ('txn_count_dup', '<u4'),
    ('cum_txn', '<u4'),
    ('reserved_1', '<u4'),
    ('reserved_2', '<u4'),
    ('reserved_3', '<u4'),
    ('flag', 'u1'),
])
assert REC_DTYPE.itemsize == RECORD_SIZE, (
    f"Expected packed 57-byte record, numpy produced {REC_DTYPE.itemsize}"
)

DOTNET_EPOCH_US = np.datetime64('0001-01-01T00:00:00', 'us')

# Raw file timestamps are US Central (CME matching engine local time).
SOURCE_TZ = 'America/Chicago'
MARKET_TZ = 'America/New_York'

# The CVD/session boundary. Globex reopens at 17:00 CT; the original decoder
# used 18:00 which is 17:00 CT expressed in ET. Kept in ET for clarity.
SESSION_RESET_HOUR_ET = 18


# ============================================================
# Decoding
# ============================================================

def decode_file(filepath, direction_method='level'):
    """
    Decode one .data file into a tick DataFrame.

    Returns columns: timestamp (tz-naive, source tz), price, level_low,
    level_high, level_width, volume, direction, signed_volume.

    direction_method selects the aggressor classification rule -- see
    classify_direction(). Defaults to 'level' to reproduce the original
    decoder; run validate_direction.py before trusting any choice.
    """
    filesize = os.path.getsize(filepath)
    if filesize == 0:
        return _empty_tick_frame()

    n_trailing = filesize % RECORD_SIZE
    if n_trailing != 0:
        print(f"  WARNING: {os.path.basename(filepath)} size {filesize} is not a "
              f"multiple of {RECORD_SIZE}; ignoring {n_trailing} trailing bytes",
              file=sys.stderr)

    with open(filepath, 'rb') as f:
        buf = f.read(filesize - n_trailing)

    rec = np.frombuffer(buf, dtype=REC_DTYPE)
    if len(rec) == 0:
        return _empty_tick_frame()

    ts = DOTNET_EPOCH_US + (rec['ticks'] // 10).astype('timedelta64[us]')

    price = rec['price'].astype(np.float64)
    level_low = rec['level_low'].astype(np.float64)
    level_high = rec['level_high'].astype(np.float64)
    volume = rec['txn_count'].astype(np.float64)

    df = pd.DataFrame({
        'timestamp': pd.to_datetime(ts),
        'price': price,
        'level_low': level_low,
        'level_high': level_high,
        'level_width': level_high - level_low,
        'volume': volume,
    })

    # Records are expected to be in time order already, but a stable sort is
    # cheap insurance -- every downstream feature assumes monotonic time.
    if not df['timestamp'].is_monotonic_increasing:
        df = df.sort_values('timestamp', kind='stable').reset_index(drop=True)

    df['direction'] = classify_direction(df, method=direction_method)
    df['signed_volume'] = df['direction'] * df['volume']
    return df


def _empty_tick_frame():
    return pd.DataFrame({
        'timestamp': pd.to_datetime([]),
        'price': np.array([], dtype=np.float64),
        'level_low': np.array([], dtype=np.float64),
        'level_high': np.array([], dtype=np.float64),
        'level_width': np.array([], dtype=np.float64),
        'volume': np.array([], dtype=np.float64),
        'direction': np.array([], dtype=np.int8),
        'signed_volume': np.array([], dtype=np.float64),
    })


# ============================================================
# Aggressor classification
# ============================================================

def classify_direction(df, method='level'):
    """
    Infer trade aggressor side. Returns int8 array: +1 buy, -1 sell, 0 unknown.

    Three rules are provided because it is not established which is correct for
    this feed. If [level_low, level_high] is the bid/ask quote then 'level' is
    the right rule (it is the standard quote rule). If instead the level is a
    footprint price bucket -- which the observed 1.0 and 2.0 point widths
    suggest, since ES normally quotes a 0.25 spread -- then 'level' is close to
    meaningless and 'tick' or 'mid' will do better.

    Every delta, CVD, and imbalance feature depends on this being right, so
    resolve it with validate_direction.py before building anything on top.

    method:
        'level' -- quote rule. price at level_high => buy, at level_low => sell.
        'tick'  -- tick rule. price rising => buy, falling => sell, unchanged
                   inherits the previous direction.
        'mid'   -- price above level midpoint => buy, below => sell, exactly at
                   the midpoint falls back to the tick rule.
    """
    if len(df) == 0:
        return np.array([], dtype=np.int8)

    price = df['price'].to_numpy(dtype=np.float64)

    if method == 'level':
        low = df['level_low'].to_numpy(dtype=np.float64)
        high = df['level_high'].to_numpy(dtype=np.float64)
        direction = np.zeros(len(price), dtype=np.int8)
        direction[np.isclose(price, high, atol=1e-9)] = 1
        direction[np.isclose(price, low, atol=1e-9)] = -1
        return direction

    if method == 'tick':
        return _tick_rule(price)

    if method == 'mid':
        low = df['level_low'].to_numpy(dtype=np.float64)
        high = df['level_high'].to_numpy(dtype=np.float64)
        mid = 0.5 * (low + high)
        direction = np.sign(price - mid).astype(np.int8)
        at_mid = direction == 0
        if at_mid.any():
            direction[at_mid] = _tick_rule(price)[at_mid]
        return direction

    raise ValueError(f"Unknown direction method: {method!r}")


def _tick_rule(price):
    """Tick rule with carry-forward of the last non-zero direction."""
    diff = np.diff(price, prepend=price[0])
    direction = np.sign(diff).astype(np.int8)

    # Carry the last non-zero sign forward over unchanged prices.
    nonzero_idx = np.flatnonzero(direction)
    if len(nonzero_idx) == 0:
        return np.zeros(len(price), dtype=np.int8)

    # For each position, index of the most recent non-zero direction.
    fill_src = np.maximum.accumulate(np.where(direction != 0, np.arange(len(direction)), -1))
    valid = fill_src >= 0
    out = np.zeros(len(price), dtype=np.int8)
    out[valid] = direction[fill_src[valid]]
    return out


# ============================================================
# Session handling
# ============================================================

def to_market_time(timestamps):
    """Convert tz-naive source timestamps to tz-aware market (ET) time."""
    idx = pd.DatetimeIndex(timestamps)
    if idx.tz is None:
        # ambiguous/nonexistent arise on DST boundaries; shift rather than raise
        idx = idx.tz_localize(SOURCE_TZ, ambiguous='NaT', nonexistent='shift_forward')
    return idx.tz_convert(MARKET_TZ)


def session_date(timestamps):
    """
    Trading session date, keyed to the 18:00 ET reset boundary.

    Bars from 18:00 ET onward belong to the *next* calendar day's session,
    matching how CME dates the Globex session.
    """
    et = to_market_time(timestamps)
    shifted = et + pd.Timedelta(hours=(24 - SESSION_RESET_HOUR_ET))
    return pd.Index(shifted.date, name='session_date')


def is_rth(timestamps, start=(9, 30), end=(15, 30)):
    """Boolean mask for Regular Trading Hours in ET. End is exclusive."""
    et = to_market_time(timestamps)
    hm = et.hour * 100 + et.minute
    return np.asarray((hm >= start[0] * 100 + start[1]) & (hm < end[0] * 100 + end[1]))


# ============================================================
# File discovery and streaming
# ============================================================

def find_data_files(data_dir, start=None, end=None):
    """
    Sorted list of .data files. Filenames are expected to be YYYYMMDD.data, so
    lexical sort is chronological. start/end are inclusive 'YYYYMMDD' strings.
    """
    files = sorted(glob.glob(os.path.join(data_dir, '*.data')))
    if not files:
        raise FileNotFoundError(f"No .data files found in {data_dir}")

    def stem(p):
        return os.path.splitext(os.path.basename(p))[0]

    if start:
        files = [f for f in files if stem(f) >= start]
    if end:
        files = [f for f in files if stem(f) <= end]
    if not files:
        raise FileNotFoundError(f"No .data files in {data_dir} within [{start}, {end}]")
    return files


def iter_sessions(data_dir, direction_method='level', start=None, end=None,
                  files=None, verbose=True):
    """
    Yield (filepath, tick_df) one file at a time.

    This is the memory-safe entry point for multi-year runs: decode a session,
    reduce it to bars or features, discard the ticks, move on.
    """
    files = files if files is not None else find_data_files(data_dir, start, end)
    for i, path in enumerate(files, 1):
        df = decode_file(path, direction_method=direction_method)
        if verbose:
            print(f"  [{i}/{len(files)}] {os.path.basename(path)}: {len(df):,} ticks")
        if len(df) == 0:
            continue
        yield path, df


# ============================================================
# Format diagnostics
# ============================================================

def check_format(data_dir, n_files=5, start=None, end=None):
    """
    Report on record layout and the level-width distribution.

    The level width is the key diagnostic for aggressor classification: a
    consistent 0.25 means [level_low, level_high] is the ES bid/ask and the
    quote rule is valid. Widths of 1.0 or 2.0 mean the level is a price bucket
    and the quote rule is not measuring aggression.
    """
    files = find_data_files(data_dir, start, end)[:n_files]
    print(f"{'='*64}")
    print("  FORMAT CHECK")
    print(f"{'='*64}")
    print(f"  Record dtype itemsize: {REC_DTYPE.itemsize} bytes")

    for path in files:
        size = os.path.getsize(path)
        df = decode_file(path)
        if len(df) == 0:
            print(f"\n  {os.path.basename(path)}: empty")
            continue

        et = to_market_time(df['timestamp'])
        widths = df['level_width'].round(4)
        width_counts = widths.value_counts().sort_index()
        total = len(df)

        print(f"\n  {os.path.basename(path)}  ({size:,} bytes, {total:,} records)")
        print(f"    remainder mod {RECORD_SIZE}: {size % RECORD_SIZE}")
        print(f"    time range (ET):   {et.min()} -> {et.max()}")
        print(f"    price range:       {df['price'].min():.2f} -> {df['price'].max():.2f}")
        print(f"    total volume:      {df['volume'].sum():,.0f}")
        print(f"    reserved non-zero: "
              f"{'yes (layout may differ)' if _reserved_nonzero(path) else 'no (as expected)'}")
        print(f"    level width distribution:")
        for w, c in width_counts.items():
            print(f"      {w:>6.2f} pts ({w / TICK_SIZE:>4.0f} ticks): "
                  f"{c:>9,} ({c / total * 100:>5.1f}%)")

        off_tick = (~np.isclose(np.remainder(df['price'].to_numpy(), TICK_SIZE), 0,
                                atol=1e-9)).sum()
        print(f"    prices off {TICK_SIZE} grid: {off_tick:,}")

        for method in ('level', 'tick', 'mid'):
            d = classify_direction(df, method=method)
            n_buy = int((d == 1).sum())
            n_sell = int((d == -1).sum())
            n_unk = int((d == 0).sum())
            print(f"    direction[{method:>5}]: buy {n_buy/total*100:5.1f}%  "
                  f"sell {n_sell/total*100:5.1f}%  unknown {n_unk/total*100:5.1f}%")

    print(f"\n{'='*64}")
    print("  If level width is not a constant 0.25, the 'level' (quote rule)")
    print("  aggressor classification is suspect. Run validate_direction.py.")
    print(f"{'='*64}")


def _reserved_nonzero(path):
    """True if any reserved field is non-zero, which would imply a different layout."""
    with open(path, 'rb') as f:
        buf = f.read((os.path.getsize(path) // RECORD_SIZE) * RECORD_SIZE)
    rec = np.frombuffer(buf, dtype=REC_DTYPE)
    if len(rec) == 0:
        return False
    return bool(rec['reserved_1'].any() or rec['reserved_2'].any() or rec['reserved_3'].any())


def summarize(data_dir, start=None, end=None):
    """Print a per-session inventory: coverage, volume, and gaps."""
    files = find_data_files(data_dir, start, end)
    print(f"{'='*64}")
    print(f"  DATA INVENTORY: {len(files)} file(s)")
    print(f"{'='*64}")

    rows = []
    for path, df in iter_sessions(data_dir, files=files, verbose=False):
        et = to_market_time(df['timestamp'])
        rth = is_rth(df['timestamp'])
        rows.append({
            'file': os.path.basename(path),
            'first_et': et.min(),
            'last_et': et.max(),
            'ticks': len(df),
            'volume': df['volume'].sum(),
            'rth_ticks': int(rth.sum()),
            'price_lo': df['price'].min(),
            'price_hi': df['price'].max(),
        })

    if not rows:
        print("  No decodable records.")
        return None

    inv = pd.DataFrame(rows)
    print(f"  Sessions:      {len(inv)}")
    print(f"  Span:          {inv['first_et'].min()} -> {inv['last_et'].max()}")
    print(f"  Total ticks:   {inv['ticks'].sum():,}")
    print(f"  Total volume:  {inv['volume'].sum():,.0f}")
    print(f"  Ticks/session: median {inv['ticks'].median():,.0f}, "
          f"min {inv['ticks'].min():,}, max {inv['ticks'].max():,}")
    print(f"  RTH share:     {inv['rth_ticks'].sum() / inv['ticks'].sum() * 100:.1f}%")

    # Flag calendar gaps larger than a long weekend, which usually means
    # missing downloads rather than holidays.
    days = pd.to_datetime(inv['first_et'].dt.date.astype(str))
    gaps = days.diff().dt.days
    big = gaps[gaps > 4]
    if len(big) > 0:
        print(f"\n  Calendar gaps > 4 days ({len(big)}):")
        for idx in big.index[:20]:
            print(f"    {days[idx-1].date()} -> {days[idx].date()} ({int(gaps[idx])} days)")
        if len(big) > 20:
            print(f"    ... and {len(big) - 20} more")

    thin = inv[inv['ticks'] < inv['ticks'].median() * 0.2]
    if len(thin) > 0:
        print(f"\n  Suspiciously thin sessions ({len(thin)}, <20% of median ticks):")
        for _, r in thin.head(20).iterrows():
            print(f"    {r['file']}: {r['ticks']:,} ticks")

    print(f"{'='*64}")
    return inv


def main():
    parser = argparse.ArgumentParser(
        description='Decode and inspect binary .data tick files.')
    parser.add_argument('data_dir', help='Directory containing .data files')
    parser.add_argument('--summary', action='store_true',
                        help='Per-session inventory with gap detection')
    parser.add_argument('--check-format', action='store_true',
                        help='Record layout and level-width diagnostics')
    parser.add_argument('--n-files', type=int, default=5,
                        help='Files to inspect for --check-format (default: 5)')
    parser.add_argument('--start', default=None, help='First session, YYYYMMDD')
    parser.add_argument('--end', default=None, help='Last session, YYYYMMDD')
    parser.add_argument('--out', default=None,
                        help='Write the inventory table to this CSV path')

    args = parser.parse_args()

    if args.check_format:
        check_format(args.data_dir, n_files=args.n_files,
                     start=args.start, end=args.end)

    if args.summary or not args.check_format:
        inv = summarize(args.data_dir, start=args.start, end=args.end)
        if args.out and inv is not None:
            inv.to_csv(args.out, index=False)
            print(f"\n  Inventory written to {args.out}")


if __name__ == '__main__':
    main()
