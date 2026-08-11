#!/usr/bin/env python3
"""
Core feature alignment gate for ES mini trading system.

Enforces that delta_pct, raw volume delta, and 10-bar volume-profile zone
all agree with the predicted trade direction before a signal is taken.
Used by both pattern mining and live/backtest inference.
"""

import numpy as np

# Features that must align with trade direction
CORE_FEATURES = ['delta_pct', 'delta', 'va10_zone']

# Default thresholds — overridden by calibrated values from training_meta.json
DEFAULT_CORE_CONFIG = {
    'delta_pct_min': 0.05,
    'delta_min_contracts': 200,
    'require_zone_alignment': True,
    'features': CORE_FEATURES,
}


def compute_va10_zone(bars):
    """Derive va10_zone from existing zone flags.

    Returns:
        +1: above VAH (breakout territory)
         0: inside value area
        -1: below VAL (breakdown territory)
       NaN: profile not yet valid (first ~10 bars)
    """
    zone = np.select(
        [bars['va10_above_vah'] == 1, bars['va10_below_val'] == 1],
        [1, -1],
        default=0,
    )
    zone = zone.astype(float)
    zone[bars['va10_above_vah'].isna()] = np.nan
    return zone


def check_core_alignment(feature_dict, direction, config=None):
    """Check whether core features align with the predicted direction.

    Args:
        feature_dict: dict mapping feature name -> value for the current bar
        direction: +1 (LONG) or -1 (SHORT)
        config: dict with thresholds (uses DEFAULT_CORE_CONFIG if None)

    Returns:
        (aligned: bool, reason: str)
        reason is empty string when aligned, explanatory message when not.
    """
    if config is None:
        config = DEFAULT_CORE_CONFIG

    delta_pct_min = config.get('delta_pct_min', DEFAULT_CORE_CONFIG['delta_pct_min'])
    delta_min = config.get('delta_min_contracts', DEFAULT_CORE_CONFIG['delta_min_contracts'])
    require_zone = config.get('require_zone_alignment', True)

    delta_pct = feature_dict.get('delta_pct')
    delta = feature_dict.get('delta')
    va10_zone = feature_dict.get('va10_zone')

    if delta_pct is None or delta is None:
        return False, 'core features missing (delta_pct or delta)'

    # --- delta_pct alignment ---
    if direction == 1 and delta_pct < delta_pct_min:
        return False, f'delta_pct={delta_pct:.3f} below {delta_pct_min} for LONG'
    if direction == -1 and delta_pct > -delta_pct_min:
        return False, f'delta_pct={delta_pct:.3f} above {-delta_pct_min} for SHORT'

    # --- raw delta alignment (absolute conviction) ---
    if direction == 1 and delta < delta_min:
        return False, f'delta={delta:.0f} below {delta_min} contracts for LONG'
    if direction == -1 and delta > -delta_min:
        return False, f'delta={delta:.0f} above {-delta_min} contracts for SHORT'

    # --- va10 zone alignment ---
    if require_zone:
        if va10_zone is None or np.isnan(va10_zone):
            return False, 'va10_zone unavailable (profile warmup)'
        if direction == 1 and va10_zone < 0:
            return False, f'va10_zone={va10_zone:.0f} (below VAL) conflicts with LONG'
        if direction == -1 and va10_zone > 0:
            return False, f'va10_zone={va10_zone:.0f} (above VAH) conflicts with SHORT'

    return True, ''


def calibrate_thresholds(bars, y_signal):
    """Derive core thresholds from winning trades in training data.

    Uses 25th percentile of winning-direction values so the gate
    is permissive enough to keep most good trades while filtering noise.

    Returns:
        dict suitable for core_feature_config in training_meta.json
    """
    long_wins = (y_signal == 1)
    short_wins = (y_signal == -1)

    if long_wins.sum() < 20 or short_wins.sum() < 20:
        return DEFAULT_CORE_CONFIG.copy()

    long_delta_pct_p25 = float(np.percentile(bars.loc[long_wins, 'delta_pct'], 25))
    short_delta_pct_p75 = float(np.percentile(bars.loc[short_wins, 'delta_pct'], 75))
    delta_pct_min = max(0.01, min(abs(long_delta_pct_p25), abs(short_delta_pct_p75)))

    long_delta_p25 = float(np.percentile(bars.loc[long_wins, 'delta'], 25))
    short_delta_p75 = float(np.percentile(bars.loc[short_wins, 'delta'], 75))
    delta_min_contracts = max(10, min(abs(long_delta_p25), abs(short_delta_p75)))

    return {
        'delta_pct_min': round(delta_pct_min, 4),
        'delta_min_contracts': round(delta_min_contracts, 1),
        'require_zone_alignment': True,
        'features': CORE_FEATURES,
    }
