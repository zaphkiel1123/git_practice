#!/usr/bin/env python3
"""
Exploratory pattern analysis for 50/20 first-touch directional moves.

For each non-overlapping candidate start with reference price P0:
  UP success:   first touch of P0 + target_ticks before P0 - stop_ticks
  DOWN success: first touch of P0 - target_ticks before P0 + stop_ticks
  Fail:         adverse barrier hit first
  Timeout:      neither barrier hit by EOF (reported, excluded from primary plots)

Then compares lookback precursor features and aligned price/CVD paths
for success vs fail outcomes.

Usage:
  python3 pattern_analysis.py 20250909.data
  python3 pattern_analysis.py 20250909.data --target-ticks 50 --stop-ticks 20 \\
      --lookback 5min --grid 30s -o patterns.html --no-browser
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:
    print(
        f"Missing dependency: {exc}\n"
        "See INSTALL.md for packages to install (numpy, pandas, plotly).\n"
        "Nothing was auto-installed on this machine.",
        file=sys.stderr,
    )
    sys.exit(1)

from decode_data import decode_file

TICK_SIZE = 0.25

# Precursor feature columns used in rankings / box plots
FEATURE_COLS = [
    'flow_imbalance',
    'buy_volume',
    'sell_volume',
    'intensity',
    'price_return_ticks',
    'price_slope_ticks_per_min',
    'cvd_change',
    'range_ticks',
    'dist_from_high_ticks',
    'dist_from_low_ticks',
]


# ============================================================
# Data loading
# ============================================================

def load_ticks(filepath: str) -> pd.DataFrame:
    """Decode binary ticks into a sorted DataFrame indexed by timestamp."""
    rows = []
    for _, rec in decode_file(filepath):
        rows.append({
            'timestamp': rec['timestamp'],
            'price': rec['price'],
            'direction': rec['direction'],
            'txn_count': rec['txn_count'],
            'volume_delta': rec['volume_delta'],
            'cumulative_volume_delta': rec['cumulative_volume_delta'],
        })
    if not rows:
        raise ValueError(f"No records found in {filepath}")
    df = pd.DataFrame(rows)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


# ============================================================
# First-touch labeling (non-overlapping)
# ============================================================

def _resolve_barrier(
    prices: np.ndarray,
    start_idx: int,
    direction: str,
    target_ticks: int,
    stop_ticks: int,
) -> tuple[str, int]:
    """Scan forward from start_idx+1 until target, stop, or EOF.

    Returns (outcome, resolve_idx) where outcome is 'success', 'fail',
    or 'timeout'. resolve_idx is the index that hit a barrier, or
    len(prices)-1 on timeout.
    """
    p0 = prices[start_idx]
    target_move = target_ticks * TICK_SIZE
    stop_move = stop_ticks * TICK_SIZE

    if direction == 'UP':
        target = p0 + target_move
        stop = p0 - stop_move
    else:
        target = p0 - target_move
        stop = p0 + stop_move

    n = len(prices)
    for j in range(start_idx + 1, n):
        p = prices[j]
        if direction == 'UP':
            if p >= target:
                return 'success', j
            if p <= stop:
                return 'fail', j
        else:
            if p <= target:
                return 'success', j
            if p >= stop:
                return 'fail', j
    return 'timeout', n - 1


def label_events(
    ticks: pd.DataFrame,
    target_ticks: int = 50,
    stop_ticks: int = 20,
    grid: str = '30s',
) -> pd.DataFrame:
    """Label non-overlapping first-touch events on a time grid.

    Candidate starts are taken on `grid` (e.g. every 30s). After each
    candidate resolves, the next candidate must start at or after the
    resolution timestamp (non-overlapping).
    """
    prices = ticks['price'].to_numpy(dtype=float)
    timestamps = pd.to_datetime(ticks['timestamp'])
    n = len(ticks)
    if n < 2:
        return pd.DataFrame()

    grid_delta = pd.Timedelta(grid)
    t0 = timestamps.iloc[0]
    t_end = timestamps.iloc[-1]

    # Precompute grid candidate indices (first tick at/after each grid time)
    candidates: list[int] = []
    grid_t = t0
    ts_values = timestamps.to_numpy()
    while grid_t <= t_end:
        idx = int(np.searchsorted(ts_values, np.datetime64(grid_t), side='left'))
        if idx >= n:
            break
        if not candidates or candidates[-1] != idx:
            candidates.append(idx)
        grid_t = grid_t + grid_delta

    events = []
    next_allowed = 0

    for start_idx in candidates:
        if start_idx < next_allowed:
            continue
        if start_idx >= n - 1:
            break

        p0 = float(prices[start_idx])
        t_start = timestamps.iloc[start_idx]

        # Evaluate both directions from the same start; advance past the
        # later resolution so subsequent starts are non-overlapping.
        up_outcome, up_resolve = _resolve_barrier(
            prices, start_idx, 'UP', target_ticks, stop_ticks)
        down_outcome, down_resolve = _resolve_barrier(
            prices, start_idx, 'DOWN', target_ticks, stop_ticks)

        for direction, outcome, resolve_idx in (
            ('UP', up_outcome, up_resolve),
            ('DOWN', down_outcome, down_resolve),
        ):
            events.append({
                'start_idx': start_idx,
                'resolve_idx': resolve_idx,
                't_start': t_start,
                't_resolve': timestamps.iloc[resolve_idx],
                'p0': p0,
                'direction': direction,
                'outcome': outcome,
                'bars_to_resolve': resolve_idx - start_idx,
                'seconds_to_resolve': (
                    timestamps.iloc[resolve_idx] - t_start
                ).total_seconds(),
            })

        # Non-overlapping: next start must be after both directions resolve
        next_allowed = max(up_resolve, down_resolve) + 1

    return pd.DataFrame(events)


# ============================================================
# Lookback precursor features
# ============================================================

def compute_lookback_features(
    ticks: pd.DataFrame,
    events: pd.DataFrame,
    lookback: str = '5min',
) -> pd.DataFrame:
    """Attach precursor features from the lookback window before each start."""
    if events.empty:
        return events

    lookback_delta = pd.Timedelta(lookback)
    timestamps = pd.to_datetime(ticks['timestamp'])
    prices = ticks['price'].to_numpy(dtype=float)
    txn = ticks['txn_count'].to_numpy(dtype=float)
    directions = ticks['direction'].to_numpy()
    cvd = ticks['cumulative_volume_delta'].to_numpy(dtype=float)
    ts_values = timestamps.to_numpy()

    feature_rows = []
    for _, ev in events.iterrows():
        start_idx = int(ev['start_idx'])
        t_start = timestamps.iloc[start_idx]
        t_lb = t_start - lookback_delta
        lb_start = int(np.searchsorted(ts_values, np.datetime64(t_lb), side='left'))
        # Window is [lb_start, start_idx) — strictly before the decision tick
        if lb_start >= start_idx:
            # No lookback ticks available
            feats = {c: np.nan for c in FEATURE_COLS}
            feats['n_lookback_ticks'] = 0
            feature_rows.append(feats)
            continue

        sl = slice(lb_start, start_idx)
        w_price = prices[sl]
        w_txn = txn[sl]
        w_dir = directions[sl]
        w_cvd = cvd[sl]
        w_ts = timestamps.iloc[lb_start:start_idx]

        buy_vol = float(w_txn[w_dir == 'BUY'].sum())
        sell_vol = float(w_txn[w_dir == 'SELL'].sum())
        total_vol = buy_vol + sell_vol
        flow_imbalance = (
            (buy_vol - sell_vol) / total_vol if total_vol > 0 else np.nan
        )

        duration_s = max((t_start - w_ts.iloc[0]).total_seconds(), 1e-9)
        intensity = len(w_price) / duration_s

        p_first = float(w_price[0])
        p_last = float(w_price[-1])
        price_return_ticks = (p_last - p_first) / TICK_SIZE
        duration_min = duration_s / 60.0
        price_slope = price_return_ticks / duration_min if duration_min > 0 else np.nan

        cvd_change = float(w_cvd[-1] - w_cvd[0]) if len(w_cvd) else np.nan
        range_ticks = (float(w_price.max()) - float(w_price.min())) / TICK_SIZE
        p0 = float(ev['p0'])
        dist_from_high = (float(w_price.max()) - p0) / TICK_SIZE
        dist_from_low = (p0 - float(w_price.min())) / TICK_SIZE

        feature_rows.append({
            'n_lookback_ticks': len(w_price),
            'flow_imbalance': flow_imbalance,
            'buy_volume': buy_vol,
            'sell_volume': sell_vol,
            'intensity': intensity,
            'price_return_ticks': price_return_ticks,
            'price_slope_ticks_per_min': price_slope,
            'cvd_change': cvd_change,
            'range_ticks': range_ticks,
            'dist_from_high_ticks': dist_from_high,
            'dist_from_low_ticks': dist_from_low,
        })

    feat_df = pd.DataFrame(feature_rows, index=events.index)
    return pd.concat([events.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


# ============================================================
# Aligned path extraction
# ============================================================

def extract_aligned_paths(
    ticks: pd.DataFrame,
    events: pd.DataFrame,
    lookback: str = '5min',
    forward_pad: str = '10min',
    sample_interval: str = '5s',
) -> dict[str, dict[str, np.ndarray]]:
    """Align price (ticks from P0) and CVD change around each event.

    Returns nested dict: direction -> outcome -> array of shape (n_events, n_bins)
    for 'price_ticks' and 'cvd_delta'. Also returns shared 'time_offsets_s'.
    """
    lookback_delta = pd.Timedelta(lookback)
    forward_delta = pd.Timedelta(forward_pad)
    sample_delta = pd.Timedelta(sample_interval)

    # Relative time axis from -lookback to +forward_pad
    neg_steps = int(lookback_delta / sample_delta)
    pos_steps = int(forward_delta / sample_delta)
    offsets = np.arange(-neg_steps, pos_steps + 1) * sample_delta.total_seconds()

    timestamps = pd.to_datetime(ticks['timestamp'])
    prices = ticks['price'].to_numpy(dtype=float)
    cvd = ticks['cumulative_volume_delta'].to_numpy(dtype=float)
    ts_ns = timestamps.to_numpy()

    result: dict = {'time_offsets_s': offsets}

    for direction in ('UP', 'DOWN'):
        result[direction] = {}
        for outcome in ('success', 'fail'):
            subset = events[
                (events['direction'] == direction) & (events['outcome'] == outcome)
            ]
            price_paths = []
            cvd_paths = []
            for _, ev in subset.iterrows():
                start_idx = int(ev['start_idx'])
                p0 = float(ev['p0'])
                cvd0 = float(cvd[start_idx])
                t_start = timestamps.iloc[start_idx]
                window_start = t_start - lookback_delta
                window_end = t_start + forward_delta

                # Sample at regular offsets via searchsorted
                sample_times = [
                    np.datetime64(t_start + pd.Timedelta(seconds=float(off)))
                    for off in offsets
                ]
                idxs = np.searchsorted(ts_ns, sample_times, side='right') - 1
                idxs = np.clip(idxs, 0, len(prices) - 1)

                # Mask points outside available data / window
                path_price = (prices[idxs] - p0) / TICK_SIZE
                path_cvd = cvd[idxs] - cvd0

                # Invalidate samples before first tick or after last if out of range
                for k, off in enumerate(offsets):
                    t_k = t_start + pd.Timedelta(seconds=float(off))
                    if t_k < timestamps.iloc[0] or t_k > timestamps.iloc[-1]:
                        path_price[k] = np.nan
                        path_cvd[k] = np.nan
                    if t_k < window_start or t_k > window_end:
                        path_price[k] = np.nan
                        path_cvd[k] = np.nan

                price_paths.append(path_price)
                cvd_paths.append(path_cvd)

            if price_paths:
                result[direction][outcome] = {
                    'price_ticks': np.vstack(price_paths),
                    'cvd_delta': np.vstack(cvd_paths),
                }
            else:
                result[direction][outcome] = {
                    'price_ticks': np.empty((0, len(offsets))),
                    'cvd_delta': np.empty((0, len(offsets))),
                }

    return result


# ============================================================
# Summary / effect sizes
# ============================================================

def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    """Per direction/outcome counts, hit rates, median time-to-resolve."""
    rows = []
    for direction in ('UP', 'DOWN'):
        sub = events[events['direction'] == direction]
        n_total = len(sub)
        for outcome in ('success', 'fail', 'timeout'):
            s = sub[sub['outcome'] == outcome]
            med_s = s['seconds_to_resolve'].median() if len(s) else np.nan
            med_bars = s['bars_to_resolve'].median() if len(s) else np.nan
            rows.append({
                'direction': direction,
                'outcome': outcome,
                'count': len(s),
                'pct_of_direction': 100.0 * len(s) / n_total if n_total else 0.0,
                'median_seconds': med_s,
                'median_bars': med_bars,
            })
    return pd.DataFrame(rows)


def feature_effect_sizes(events: pd.DataFrame) -> pd.DataFrame:
    """Rank precursor features by |Cohen's d| between success and fail.

    Computed separately for UP and DOWN.
    """
    rows = []
    for direction in ('UP', 'DOWN'):
        succ = events[
            (events['direction'] == direction) & (events['outcome'] == 'success')
        ]
        fail = events[
            (events['direction'] == direction) & (events['outcome'] == 'fail')
        ]
        for col in FEATURE_COLS:
            a = succ[col].dropna().to_numpy(dtype=float)
            b = fail[col].dropna().to_numpy(dtype=float)
            if len(a) < 2 or len(b) < 2:
                d = np.nan
            else:
                # Pooled std Cohen's d
                va = a.var(ddof=1)
                vb = b.var(ddof=1)
                pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) /
                                 max(len(a) + len(b) - 2, 1))
                d = (a.mean() - b.mean()) / pooled if pooled > 1e-12 else 0.0
            rows.append({
                'direction': direction,
                'feature': col,
                'success_mean': float(a.mean()) if len(a) else np.nan,
                'fail_mean': float(b.mean()) if len(b) else np.nan,
                'mean_diff': (float(a.mean()) - float(b.mean()))
                if len(a) and len(b) else np.nan,
                'cohens_d': d,
                'abs_d': abs(d) if d == d else np.nan,
                'n_success': len(a),
                'n_fail': len(b),
            })
    out = pd.DataFrame(rows)
    return out.sort_values(['direction', 'abs_d'], ascending=[True, False])


# ============================================================
# Visualization
# ============================================================

def _band_traces(offsets, paths, name, color, row, col, fig):
    """Add median + IQR band for a set of aligned paths."""
    if paths.shape[0] == 0:
        return
    with np.errstate(all='ignore'):
        med = np.nanmedian(paths, axis=0)
        q25 = np.nanpercentile(paths, 25, axis=0)
        q75 = np.nanpercentile(paths, 75, axis=0)

    fig.add_trace(
        go.Scatter(
            x=offsets, y=q75, mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
        ),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=offsets, y=q25, mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=color.replace('1.0', '0.2').replace('rgb', 'rgba')
            if color.startswith('rgb') else f'rgba({_hex_to_rgb(color)},0.2)',
            name=f'{name} IQR',
            showlegend=False,
            hoverinfo='skip',
        ),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=offsets, y=med, mode='lines',
            line=dict(color=color, width=2),
            name=name,
            hovertemplate='%{x:.0f}s<br>%{y:.1f}<extra>' + name + '</extra>',
        ),
        row=row, col=col,
    )


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip('#')
    if len(h) == 6:
        return f'{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}'
    return '128,128,128'


def build_figure(
    events: pd.DataFrame,
    aligned: dict,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    title: str,
    target_ticks: int,
    stop_ticks: int,
) -> go.Figure:
    """Multi-panel HTML figure: summary, aligned paths, feature boxes."""
    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            'UP: aligned price (ticks from P0)',
            'DOWN: aligned price (ticks from P0)',
            'UP: aligned CVD change',
            'DOWN: aligned CVD change',
            'UP: precursor features by outcome',
            'DOWN: precursor features by outcome',
            'Feature effect size |Cohen\'s d| (success vs fail)',
            'Event outcome counts',
        ),
        vertical_spacing=0.07,
        horizontal_spacing=0.08,
        row_heights=[0.28, 0.22, 0.28, 0.22],
        specs=[
            [{'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'xy'}, {'type': 'bar'}],
        ],
    )

    offsets = aligned['time_offsets_s']
    colors = {
        ('UP', 'success'): '#2e7d32',
        ('UP', 'fail'): '#c62828',
        ('DOWN', 'success'): '#1565c0',
        ('DOWN', 'fail'): '#ef6c00',
    }

    # Aligned price paths
    for col_i, direction in enumerate(('UP', 'DOWN'), start=1):
        for outcome in ('success', 'fail'):
            paths = aligned[direction][outcome]['price_ticks']
            n = paths.shape[0]
            _band_traces(
                offsets, paths,
                name=f'{direction} {outcome} (n={n})',
                color=colors[(direction, outcome)],
                row=1, col=col_i, fig=fig,
            )
        fig.add_hline(y=0, line_dash='dot', line_color='#888', row=1, col=col_i)
        if direction == 'UP':
            fig.add_hline(y=target_ticks, line_dash='dash', line_color='#2e7d32',
                          row=1, col=col_i, annotation_text=f'+{target_ticks}')
            fig.add_hline(y=-stop_ticks, line_dash='dash', line_color='#c62828',
                          row=1, col=col_i, annotation_text=f'-{stop_ticks}')
        else:
            fig.add_hline(y=-target_ticks, line_dash='dash', line_color='#1565c0',
                          row=1, col=col_i, annotation_text=f'-{target_ticks}')
            fig.add_hline(y=stop_ticks, line_dash='dash', line_color='#ef6c00',
                          row=1, col=col_i, annotation_text=f'+{stop_ticks}')
        fig.add_vline(x=0, line_dash='dot', line_color='#666', row=1, col=col_i)

    # Aligned CVD
    for col_i, direction in enumerate(('UP', 'DOWN'), start=1):
        for outcome in ('success', 'fail'):
            paths = aligned[direction][outcome]['cvd_delta']
            n = paths.shape[0]
            _band_traces(
                offsets, paths,
                name=f'{direction} {outcome} CVD (n={n})',
                color=colors[(direction, outcome)],
                row=2, col=col_i, fig=fig,
            )
        fig.add_hline(y=0, line_dash='dot', line_color='#888', row=2, col=col_i)
        fig.add_vline(x=0, line_dash='dot', line_color='#666', row=2, col=col_i)

    # Feature box plots (top features by |d|, up to 6)
    for col_i, direction in enumerate(('UP', 'DOWN'), start=1):
        top = effects[effects['direction'] == direction].head(6)
        feat_list = top['feature'].tolist()
        if not feat_list:
            continue
        for outcome, color in (('success', colors[(direction, 'success')]),
                               ('fail', colors[(direction, 'fail')])):
            sub = events[
                (events['direction'] == direction) & (events['outcome'] == outcome)
            ]
            for fi, feat in enumerate(feat_list):
                fig.add_trace(
                    go.Box(
                        y=sub[feat],
                        x=[feat] * len(sub),
                        name=f'{direction} {outcome}',
                        legendgroup=f'{direction}-{outcome}-box',
                        showlegend=(fi == 0),
                        marker_color=color,
                        boxmean=True,
                        offsetgroup=outcome,
                    ),
                    row=3, col=col_i,
                )

    # Effect size bars (both directions)
    for direction, color in (('UP', '#2e7d32'), ('DOWN', '#1565c0')):
        sub = effects[effects['direction'] == direction].sort_values('abs_d')
        fig.add_trace(
            go.Bar(
                y=sub['feature'],
                x=sub['abs_d'],
                orientation='h',
                name=f'{direction} |d|',
                marker_color=color,
                opacity=0.75,
                hovertemplate=(
                    '%{y}<br>|d|=%{x:.3f}<br>'
                    'diff=%{customdata[0]:.3f}<extra></extra>'
                ),
                customdata=np.column_stack([sub['mean_diff'].to_numpy()]),
            ),
            row=4, col=1,
        )

    # Outcome counts
    for _, row in summary.iterrows():
        color = {
            ('UP', 'success'): '#2e7d32',
            ('UP', 'fail'): '#c62828',
            ('UP', 'timeout'): '#9e9e9e',
            ('DOWN', 'success'): '#1565c0',
            ('DOWN', 'fail'): '#ef6c00',
            ('DOWN', 'timeout'): '#bdbdbd',
        }[(row['direction'], row['outcome'])]
        fig.add_trace(
            go.Bar(
                x=[f"{row['direction']} {row['outcome']}"],
                y=[row['count']],
                name=f"{row['direction']} {row['outcome']}",
                marker_color=color,
                text=[f"{int(row['count'])} ({row['pct_of_direction']:.0f}%)"],
                textposition='outside',
                showlegend=False,
                hovertemplate=(
                    '%{x}<br>count=%{y}<br>'
                    f"median_s={row['median_seconds']:.0f}<extra></extra>"
                    if pd.notna(row['median_seconds']) else
                    '%{x}<br>count=%{y}<extra></extra>'
                ),
            ),
            row=4, col=2,
        )

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=1400,
        margin=dict(l=80, r=40, t=80, b=40),
        boxmode='group',
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
    )
    fig.update_xaxes(title_text='Seconds from start (0 = P0)', row=1, col=1)
    fig.update_xaxes(title_text='Seconds from start (0 = P0)', row=1, col=2)
    fig.update_xaxes(title_text='Seconds from start', row=2, col=1)
    fig.update_xaxes(title_text='Seconds from start', row=2, col=2)
    fig.update_xaxes(title_text="|Cohen's d|", row=4, col=1)
    fig.update_yaxes(title_text='Price ticks from P0', row=1, col=1)
    fig.update_yaxes(title_text='Price ticks from P0', row=1, col=2)
    fig.update_yaxes(title_text='CVD Δ from start', row=2, col=1)
    fig.update_yaxes(title_text='CVD Δ from start', row=2, col=2)
    fig.update_yaxes(title_text='Count', row=4, col=2)

    return fig


def print_report(summary: pd.DataFrame, effects: pd.DataFrame) -> None:
    """Print text summary and ranked features to stdout."""
    print('\n' + '=' * 60)
    print('  EVENT SUMMARY')
    print('=' * 60)
    print(summary.to_string(index=False))

    print('\n' + '=' * 60)
    print("  PRECURSOR FEATURE RANKING (|Cohen's d|, success vs fail)")
    print('=' * 60)
    for direction in ('UP', 'DOWN'):
        print(f'\n  --- {direction} ---')
        sub = effects[effects['direction'] == direction][
            ['feature', 'success_mean', 'fail_mean', 'mean_diff', 'cohens_d',
             'n_success', 'n_fail']
        ]
        print(sub.to_string(index=False, float_format=lambda x: f'{x:8.3f}'))
    print('=' * 60)


# ============================================================
# Main
# ============================================================

def run(
    filepath: str,
    target_ticks: int = 50,
    stop_ticks: int = 20,
    lookback: str = '5min',
    grid: str = '30s',
    forward_pad: str = '10min',
    sample_interval: str = '5s',
    output: str | None = None,
    no_browser: bool = False,
) -> None:
    stem = Path(filepath).stem
    output = output or f'{stem}_patterns_{target_ticks}_{stop_ticks}.html'

    print(f'Decoding {filepath} ...')
    ticks = load_ticks(filepath)
    print(f'  {len(ticks):,} ticks  '
          f'({ticks["timestamp"].iloc[0]} → {ticks["timestamp"].iloc[-1]})')

    print(f'Labeling first-touch events '
          f'(target={target_ticks}, stop={stop_ticks}, grid={grid}) ...')
    events = label_events(
        ticks,
        target_ticks=target_ticks,
        stop_ticks=stop_ticks,
        grid=grid,
    )
    print(f'  {len(events):,} labeled direction-events '
          f'({events["start_idx"].nunique()} unique starts)')

    print(f'Computing lookback features (lookback={lookback}) ...')
    events = compute_lookback_features(ticks, events, lookback=lookback)

    summary = summarize_events(events)
    effects = feature_effect_sizes(events)
    print_report(summary, effects)

    print('Extracting aligned paths ...')
    aligned = extract_aligned_paths(
        ticks, events,
        lookback=lookback,
        forward_pad=forward_pad,
        sample_interval=sample_interval,
    )

    title = (
        f'{stem}  |  first-touch ±{target_ticks}/∓{stop_ticks} ticks  '
        f'| lookback {lookback}  | grid {grid}'
    )
    fig = build_figure(
        events, aligned, summary, effects, title, target_ticks, stop_ticks,
    )
    fig.write_html(output, include_plotlyjs='cdn', auto_open=not no_browser)
    print(f'\nWrote {output}')

    # Also dump a CSV of events+features for further inspection
    csv_path = str(Path(output).with_suffix('.csv'))
    events.to_csv(csv_path, index=False)
    print(f'Wrote {csv_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Exploratory 50/20 first-touch move pattern analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 20250909.data
  %(prog)s 20250909.data --target-ticks 50 --stop-ticks 20 --lookback 5min
  %(prog)s 20250909.data --grid 1min -o patterns.html --no-browser
        """,
    )
    parser.add_argument('input_file', help='Path to the binary .data file')
    parser.add_argument('--target-ticks', type=int, default=50,
                        help='Profit barrier in ticks (default: 50)')
    parser.add_argument('--stop-ticks', type=int, default=20,
                        help='Adverse barrier in ticks (default: 20)')
    parser.add_argument('--lookback', default='5min',
                        help='Lookback window for precursor features (default: 5min)')
    parser.add_argument('--grid', default='30s',
                        help='Candidate start time grid (default: 30s)')
    parser.add_argument('--forward-pad', default='10min',
                        help='Forward window for aligned path plots (default: 10min)')
    parser.add_argument('--sample-interval', default='5s',
                        help='Alignment sample interval (default: 5s)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output HTML path')
    parser.add_argument('--no-browser', action='store_true',
                        help='Write HTML only; do not open a browser')
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f'Error: File not found: {args.input_file}', file=sys.stderr)
        sys.exit(1)

    run(
        args.input_file,
        target_ticks=args.target_ticks,
        stop_ticks=args.stop_ticks,
        lookback=args.lookback,
        grid=args.grid,
        forward_pad=args.forward_pad,
        sample_interval=args.sample_interval,
        output=args.output,
        no_browser=args.no_browser,
    )


if __name__ == '__main__':
    main()
