# Part 2.8 — Derived Structure Features: Implementation Guide

How to compute `choch_bull`, `choch_bear`, `swing_failure`, `equal_highs`, and `equal_lows`.

**Prerequisites:** Part 1.6 market structure must be implemented first:
- Confirmed swing highs (SH) and swing lows (SL) with alternating filter
- At each bar `t`: `last_swing_high`, `last_swing_low`, `prev_swing_high`, `prev_swing_low`
- `trend_state` ∈ {-1, 0, +1}
- `is_HH`, `is_LH`, `is_HL`, `is_LL`

**Parameters (defaults):**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `k` | 5 | Swing confirmation lag (bars each side) |
| `tick_size` | 0.25 | ES mini minimum price increment |
| `equal_tolerance` | 0.25 | Two swings "equal" if within 1 tick |
| `sf_penetration_atr` | 0.1 | Min break beyond swing level (× ATR) |
| `sf_reclaim_bars` | 3 | Bars allowed to reclaim after break |
| `choch_confirm_bars` | 1 | Close must hold beyond broken level |

All features use **past data only** at bar `t`.

---

## Shared data structure: swing history

Maintain an ordered list of confirmed swings up to bar `t`:

```python
# Each entry: {bar_idx, type, price, confirmed_at}
# type ∈ {"SH", "SL"}
# confirmed_at = bar_idx + k  (when swing becomes known)

swings = [
    {"bar_idx": 120, "type": "SH", "price": 5845.0, "confirmed_at": 125},
    {"bar_idx": 145, "type": "SL", "price": 5830.0, "confirmed_at": 150},
    ...
]
```

Helper functions used below:

```python
def last_n_swings(swings, swing_type, n=2):
    """Return last n swings of given type, newest last."""
    s = [x for x in swings if x["type"] == swing_type]
    return s[-n:] if len(s) >= n else s

def last_n_lh(swings):
    """Last two swing highs where the newer is a Lower High."""
    shs = last_n_swings(swings, "SH", 2)
    if len(shs) < 2:
        return None
    if shs[-1]["price"] < shs[-2]["price"]:  # LH
        return shs[-1]
    return None

def last_n_hl(swings):
    """Last two swing lows where the newer is a Higher Low."""
    sls = last_n_swings(swings, "SL", 2)
    if len(sls) < 2:
        return None
    if sls[-1]["price"] > sls[-2]["price"]:  # HL
        return sls[-1]
    return None
```

---

## 1. `equal_highs`

**Meaning:** Last two confirmed swing highs are at essentially the same price → potential double-top / liquidity pool above.

**Algorithm at bar `t`:**

```
1. sh = last_n_swings(swings, "SH", 2)
2. if len(sh) < 2:
       equal_highs = 0
3. else:
       equal_highs = 1  if  abs(sh[-1].price - sh[-2].price) <= equal_tolerance
                 else  0
```

**Optional strength feature:**

```
equal_highs_dist = abs(sh[-1].price - sh[-2].price) / ATR_14
# Use dist for model; binary equal_highs = 1 when dist <= equal_tolerance / ATR
```

**Example:**

```
SH at 5850.00 (older)
SH at 5850.25 (newer)   → |5850.25 - 5850.00| = 0.25 ≤ 0.25  → equal_highs = 1
SH at 5851.00 (newer)   → equal_highs = 0
```

---

## 2. `equal_lows`

**Meaning:** Last two confirmed swing lows at same price → potential double-bottom / liquidity pool below.

**Algorithm at bar `t`:**

```
1. sl = last_n_swings(swings, "SL", 2)
2. if len(sl) < 2:
       equal_lows = 0
3. else:
       equal_lows = 1  if  abs(sl[-1].price - sl[-2].price) <= equal_tolerance
                 else  0
```

Same logic as `equal_highs`, applied to swing lows.

---

## 3. `choch_bull` (Change of Character — bullish)

**Meaning:** Market was in downtrend (LH + LL sequence). Price breaks above the most recent **Lower High** — first sign uptrend may be starting.

**Downtrend context at `t`:**

```
trend_state == -1
OR (is_LH == 1 AND is_LL == 1)
```

**Reference level:** price of the most recent LH swing high.

```
lh = last_n_lh(swings)
if lh is None:
    choch_bull = 0
else:
    lh_price = lh["price"]
```

**Trigger at bar `t`:**

```
choch_bull = 1  if ALL of:
    - downtrend context is true
    - close[t] > lh_price + choch_confirm_bars * 0   # close above LH
    - close[t] > lh_price
    - (optional) close[t-1] <= lh_price   # first break bar only (edge trigger)
else:
    choch_bull = 0
```

**Recommended: edge-trigger vs level-trigger**

| Mode | Behavior | Feature column |
|------|----------|----------------|
| Edge | 1 only on first bar that closes above LH | `choch_bull` |
| Level | 1 while close remains above LH | `above_lh_in_downtrend` |

For training, **edge-trigger** is cleaner (one event per CHoCH). Store `bars_since_choch_bull` as a continuous companion feature.

**Pseudocode (edge-trigger):**

```python
def compute_choch_bull(t, bars, swings, trend_state):
    lh = last_n_lh(swings_up_to_t)
    if lh is None or trend_state[t] != -1:
        return 0
    lh_price = lh["price"]
    if bars.close[t] > lh_price and bars.close[t - 1] <= lh_price:
        return 1
    return 0
```

**Diagram:**

```
Downtrend: LH at 5840, LL at 5820

Price
5840 ---- LH (reference for choch_bull)
      \    /
       \  /
5820    \/  LL
         \
          \____ close breaks above 5840 → choch_bull = 1
```

---

## 4. `choch_bear` (Change of Character — bearish)

**Meaning:** Market was in uptrend (HH + HL). Price breaks below the most recent **Higher Low** — first sign downtrend may be starting.

**Uptrend context:**

```
trend_state == +1
OR (is_HH == 1 AND is_HL == 1)
```

**Reference level:** price of the most recent HL swing low.

```python
def compute_choch_bear(t, bars, swings, trend_state):
    hl = last_n_hl(swings_up_to_t)
    if hl is None or trend_state[t] != +1:
        return 0
    hl_price = hl["price"]
    if bars.close[t] < hl_price and bars.close[t - 1] >= hl_price:
        return 1
    return 0
```

---

## 5. `swing_failure` (liquidity grab / failed breakout)

**Meaning:** Price **penetrates** a key swing level (takes liquidity) then **reclaims** back inside range within `sf_reclaim_bars` — classic stop-hunt / trap.

Two variants: **failed breakout above SH** and **failed breakdown below SL**.

### 5.1 State machine per swing level

Track for the most recent SH and SL:

```python
# Per swing level state:
#   IDLE → PENETRATED → RECLAIMED (swing_failure=1) or EXPIRED
```

### 5.2 Failed breakout above swing high (bearish trap)

**At bar `t`, reference = `last_swing_high` = SH_price**

```
penetration = high[t] > SH_price + sf_penetration_atr * ATR_14

if penetration:
    mark state = PENETRATED at bar t
    deadline = t + sf_reclaim_bars

For bars after penetration until deadline:
    if close[t] < SH_price:          # reclaimed below SH
        swing_failure_bear = 1       # failed rally above SH
        reset state = IDLE
    elif t > deadline:
        state = EXPIRED              # break held → not a failure
        swing_failure_bear = 0
```

### 5.3 Failed breakdown below swing low (bullish trap)

**Reference = `last_swing_low` = SL_price**

```
penetration = low[t] < SL_price - sf_penetration_atr * ATR_14

if penetration:
    mark state = PENETRATED

For bars after penetration until deadline:
    if close[t] > SL_price:          # reclaimed above SL
        swing_failure_bull = 1       # failed breakdown
    elif t > deadline:
        state = EXPIRED
```

### 5.4 Combined feature

```
swing_failure = 1  if  swing_failure_bull == 1 OR swing_failure_bear == 1
                 else  0
```

Or keep separate columns (recommended for model):

| Column | Meaning |
|--------|---------|
| `swing_failure_bull` | Penetrated SL, reclaimed above — bullish trap |
| `swing_failure_bear` | Penetrated SH, reclaimed below — bearish trap |
| `swing_failure` | Either (OR of above) |

### 5.5 Example (bearish trap at SH)

```
SH = 5850.00

Bar 100: high=5850.50, close=5849.75  → penetration (high > 5850 + 0.1*ATR)
Bar 101: close=5848.00                → reclaimed below 5850 → swing_failure_bear = 1
```

### 5.6 Implementation notes

- Only track **one active penetration** per swing level (the most recent SH or SL).
- Use **confirmed** swing prices only (from alternating swing list).
- Do not fire `swing_failure` if `bos_bull/bos_bear` already confirmed a real break with follow-through (optional filter: failure only if break distance < 0.5× ATR).

---

## Full computation order at each bar `t`

```
1. Update swing list (if new confirmed SH/SL)
2. Update trend_state, is_HH/HL/LH/LL
3. equal_highs, equal_lows          ← need last 2 SH/SL
4. choch_bull, choch_bear         ← need LH/HL + trend_state + close cross
5. swing_failure_*                ← state machine on SH/SL penetration
```

---

## Output columns to add to bar table

| Column | Type | Values |
|--------|------|--------|
| `equal_highs` | int | 0/1 |
| `equal_lows` | int | 0/1 |
| `equal_highs_dist_atr` | float | distance / ATR (optional) |
| `equal_lows_dist_atr` | float | distance / ATR (optional) |
| `choch_bull` | int | 0/1 edge-trigger |
| `choch_bear` | int | 0/1 edge-trigger |
| `bars_since_choch_bull` | int | 0 if just fired, else bars since |
| `bars_since_choch_bear` | int | same |
| `swing_failure_bull` | int | 0/1 |
| `swing_failure_bear` | int | 0/1 |
| `swing_failure` | int | 0/1 OR of bull/bear |

---

## Pandas-style loop skeleton

```python
import numpy as np
import pandas as pd

TICK = 0.25
EQUAL_TOL = 0.25
SF_PEN_ATR = 0.1
SF_RECLAIM = 3

def compute_derived_structure(bars, swings_df):
    """
    bars: OHLCV + ATR_14 + trend_state + last_sh/sl prices
    swings_df: confirmed swings [{bar_idx, type, price, confirmed_at}, ...]
    Returns bars with Part 2.8 columns added.
    """
    n = len(bars)
    equal_highs = np.zeros(n, dtype=np.int8)
    equal_lows  = np.zeros(n, dtype=np.int8)
    choch_bull  = np.zeros(n, dtype=np.int8)
    choch_bear  = np.zeros(n, dtype=np.int8)
    sf_bull     = np.zeros(n, dtype=np.int8)
    sf_bear     = np.zeros(n, dtype=np.int8)

    # Active penetration trackers
    sh_pen = None   # {"level", "start_bar", "deadline"}
    sl_pen = None

    sh_history, sl_history = [], []

    swing_ptr = 0
    swing_rows = swings_df.sort_values("confirmed_at").to_dict("records")

    for t in range(n):
        # ingest newly confirmed swings
        while swing_ptr < len(swing_rows) and swing_rows[swing_ptr]["confirmed_at"] <= t:
            sw = swing_rows[swing_ptr]
            if sw["type"] == "SH":
                sh_history.append(sw["price"])
            else:
                sl_history.append(sw["price"])
            swing_ptr += 1

        atr = bars["atr_14"].iloc[t]
        close = bars["close"].iloc[t]
        high  = bars["high"].iloc[t]
        low   = bars["low"].iloc[t]
        prev_close = bars["close"].iloc[t - 1] if t > 0 else close
        trend = bars["trend_state"].iloc[t]

        # equal highs / lows
        if len(sh_history) >= 2:
            equal_highs[t] = int(abs(sh_history[-1] - sh_history[-2]) <= EQUAL_TOL)
        if len(sl_history) >= 2:
            equal_lows[t] = int(abs(sl_history[-1] - sl_history[-2]) <= EQUAL_TOL)

        # CHoCH — need LH/HL from history
        if len(sh_history) >= 2 and sh_history[-1] < sh_history[-2] and trend == -1:
            lh_price = sh_history[-1]
            if close > lh_price and prev_close <= lh_price:
                choch_bull[t] = 1
        if len(sl_history) >= 2 and sl_history[-1] > sl_history[-2] and trend == +1:
            hl_price = sl_history[-1]
            if close < hl_price and prev_close >= hl_price:
                choch_bear[t] = 1

        # Swing failure — SH penetration
        if len(sh_history) >= 1:
            sh_price = sh_history[-1]
            pen = high > sh_price + SF_PEN_ATR * atr
            if sh_pen is None and pen:
                sh_pen = {"level": sh_price, "start": t, "deadline": t + SF_RECLAIM}
            if sh_pen is not None:
                if close < sh_pen["level"]:
                    sf_bear[t] = 1
                    sh_pen = None
                elif t > sh_pen["deadline"]:
                    sh_pen = None

        # Swing failure — SL penetration
        if len(sl_history) >= 1:
            sl_price = sl_history[-1]
            pen = low < sl_price - SF_PEN_ATR * atr
            if sl_pen is None and pen:
                sl_pen = {"level": sl_price, "start": t, "deadline": t + SF_RECLAIM}
            if sl_pen is not None:
                if close > sl_pen["level"]:
                    sf_bull[t] = 1
                    sl_pen = None
                elif t > sl_pen["deadline"]:
                    sl_pen = None

    bars["equal_highs"] = equal_highs
    bars["equal_lows"] = equal_lows
    bars["choch_bull"] = choch_bull
    bars["choch_bear"] = choch_bear
    bars["swing_failure_bull"] = sf_bull
    bars["swing_failure_bear"] = sf_bear
    bars["swing_failure"] = ((sf_bull | sf_bear) > 0).astype(np.int8)
    return bars
```

---

## Common pitfalls

| Pitfall | Fix |
|---------|-----|
| Using unconfirmed swings | Only add swing when `confirmed_at <= t` |
| CHoCH fires in chop | Require `trend_state != 0` |
| `equal_highs` on first 2 SH ever | Return 0 until 2 SH exist |
| Swing failure fires on every wick | Require penetration ≥ `sf_penetration_atr × ATR` |
| Lookahead in labels | Derived structure features use bar `t` OHLC only; labels use `t+1..t+H` |

---

## See also

- Core structure definitions: [NN_model_features.md](NN_model_features.md) § 1.6
- Part 2.8 feature list: [NN_model_features.md](NN_model_features.md) § 2.8
