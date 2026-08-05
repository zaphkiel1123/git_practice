#!/usr/bin/env python3
"""
Interactive 1-minute candlestick viewer for .data transaction files.

Panels:
  1. OHLC candlesticks with footprint (sell left | buy right at each price)
  2. Volume bars with numeric labels
  3. Cumulative volume delta as candlesticks (resets at 18:00)

Usage:
  python visualize.py 20250904.data
  python visualize.py 20250904.data --interval 5min
  python visualize.py 20250904.data -o chart.html --no-browser
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from decode_data import decode_file


def load_ticks(filepath: str) -> pd.DataFrame:
    """Decode binary ticks into a DataFrame."""
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
    df = df.set_index('timestamp').sort_index()
    return df


def build_footprint(ticks: pd.DataFrame, interval: str = '1min') -> pd.DataFrame:
    """Per-bar, per-price buy/sell transaction counts."""
    df = ticks.copy()
    df['buy'] = np.where(df['direction'] == 'BUY', df['txn_count'], 0)
    df['sell'] = np.where(df['direction'] == 'SELL', df['txn_count'], 0)
    df['bar_time'] = df.index.floor(interval)

    footprint = (
        df.groupby(['bar_time', 'price'], sort=True)[['buy', 'sell']]
        .sum()
        .reset_index()
    )
    # Drop levels with no directed volume (UNKNOWN-only).
    footprint = footprint[(footprint['buy'] > 0) | (footprint['sell'] > 0)]
    return footprint


def build_bars(ticks: pd.DataFrame, interval: str = '1min') -> pd.DataFrame:
    """Aggregate ticks into OHLCV + buy/sell totals + CVD OHLC bars."""
    df = ticks.copy()
    df['buy'] = np.where(df['direction'] == 'BUY', df['txn_count'], 0)
    df['sell'] = np.where(df['direction'] == 'SELL', df['txn_count'], 0)

    resampled = df.resample(interval)
    bars = pd.DataFrame({
        'open': resampled['price'].first(),
        'high': resampled['price'].max(),
        'low': resampled['price'].min(),
        'close': resampled['price'].last(),
        'volume': resampled['txn_count'].sum(),
        'buy_volume': resampled['buy'].sum(),
        'sell_volume': resampled['sell'].sum(),
        'cvd_open': resampled['cumulative_volume_delta'].first(),
        'cvd_high': resampled['cumulative_volume_delta'].max(),
        'cvd_low': resampled['cumulative_volume_delta'].min(),
        'cvd_close': resampled['cumulative_volume_delta'].last(),
    })
    bars = bars.dropna(subset=['open'])
    bars['up'] = bars['close'] >= bars['open']
    return bars


def make_figure(
    bars: pd.DataFrame,
    footprint: pd.DataFrame,
    title: str,
) -> go.Figure:
    """Build a 3-panel Plotly figure: candles+footprint, volume, CVD candles."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.20, 0.25],
        subplot_titles=(
            'Price (sell left | buy right at each level)',
            'Volume',
            'Cumulative Volume Delta',
        ),
    )

    # --- Price candlesticks (semi-transparent so footprint text stays readable)
    fig.add_trace(
        go.Candlestick(
            x=bars.index,
            open=bars['open'],
            high=bars['high'],
            low=bars['low'],
            close=bars['close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='rgba(38, 166, 154, 0.25)',
            decreasing_fillcolor='rgba(239, 83, 80, 0.25)',
            opacity=0.85,
        ),
        row=1,
        col=1,
    )

    # --- Footprint: sell counts on the left of each price level
    if len(footprint):
        fig.add_trace(
            go.Scatter(
                x=footprint['bar_time'],
                y=footprint['price'],
                text=[str(int(v)) if v else '' for v in footprint['sell']],
                mode='text',
                name='Sell',
                textposition='middle left',
                textfont=dict(size=9, color='#c62828'),
                hovertemplate=(
                    '%{x}<br>Price %{y:.2f}<br>'
                    'Sell %{customdata[0]} | Buy %{customdata[1]}'
                    '<extra></extra>'
                ),
                customdata=np.column_stack([
                    footprint['sell'].astype(int),
                    footprint['buy'].astype(int),
                ]),
            ),
            row=1,
            col=1,
        )
        # Buy counts on the right of each price level
        fig.add_trace(
            go.Scatter(
                x=footprint['bar_time'],
                y=footprint['price'],
                text=[str(int(v)) if v else '' for v in footprint['buy']],
                mode='text',
                name='Buy',
                textposition='middle right',
                textfont=dict(size=9, color='#2e7d32'),
                hoverinfo='skip',
            ),
            row=1,
            col=1,
        )

    # --- Volume bars with value labels
    colors = ['#26a69a' if up else '#ef5350' for up in bars['up']]
    fig.add_trace(
        go.Bar(
            x=bars.index,
            y=bars['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.85,
            text=bars['volume'].astype(int).astype(str),
            textposition='outside',
            textfont=dict(size=9),
            cliponaxis=False,
            hovertemplate='%{x}<br>Volume %{y}<extra></extra>',
        ),
        row=2,
        col=1,
    )

    # --- CVD candlesticks
    fig.add_trace(
        go.Candlestick(
            x=bars.index,
            open=bars['cvd_open'],
            high=bars['cvd_high'],
            low=bars['cvd_low'],
            close=bars['cvd_close'],
            name='CVD',
            increasing_line_color='#42a5f5',
            decreasing_line_color='#ab47bc',
            increasing_fillcolor='#42a5f5',
            decreasing_fillcolor='#ab47bc',
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line_dash='dot', line_color='#888', row=3, col=1)

    fig.update_layout(
        title=title,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
        showlegend=False,
        height=1000,
        margin=dict(l=60, r=30, t=60, b=40),
        hovermode='closest',
    )
    # Disable rangeslider on both candlestick axes
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(title_text='Time', row=3, col=1)
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    fig.update_yaxes(title_text='CVD', row=3, col=1)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize .data files as interactive candlestick charts.',
    )
    parser.add_argument('input_file', help='Path to the binary .data file')
    parser.add_argument(
        '--interval', default='1min',
        help='Bar interval (default: 1min). Examples: 1min, 5min, 15min',
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help='Output HTML path (default: <input_stem>_<interval>_chart.html)',
    )
    parser.add_argument(
        '--no-browser', action='store_true',
        help='Write HTML only; do not open a browser',
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    stem = Path(args.input_file).stem
    interval_tag = args.interval.replace(' ', '')
    output = args.output or f"{stem}_{interval_tag}_chart.html"

    print(f"Decoding {args.input_file} ...")
    ticks = load_ticks(args.input_file)
    print(f"  {len(ticks):,} ticks  "
          f"({ticks.index[0]} → {ticks.index[-1]})")

    print(f"Building {args.interval} bars + footprint ...")
    bars = build_bars(ticks, args.interval)
    footprint = build_footprint(ticks, args.interval)
    print(f"  {len(bars):,} bars, {len(footprint):,} price levels")

    title = f"{stem}  |  {args.interval} candles + footprint + volume + CVD"
    fig = make_figure(bars, footprint, title)

    fig.write_html(output, include_plotlyjs='cdn', auto_open=not args.no_browser)
    print(f"Wrote {output}")


if __name__ == '__main__':
    main()
