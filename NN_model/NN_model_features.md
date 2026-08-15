# ES Mini NN Model — Feature Specification

Curated feature list for continuation / retracement / reversal / unknown classification.
Derived from tick data aggregated to **1-minute bars**. No dependency on legacy training scripts.

**Model:** Neural network only (TCN / LSTM / hybrid sequence encoder). No LightGBM or other tree models.

Each **decision event** at bar `t` uses:
- **Past sequence** — bars `[t-L+1 .. t]` as a `(L, C)` matrix fed to the NN
- **Future bars** `[t+1 .. t+H]` — label assignment only, never model input

---

## Part 1 — Core Features (start here)

These are the minimum features to implement first.

### 1.1 OHLC (price bar shape)

Computed from tick `price` aggregated to 1m bars.

| Feature | Formula | Unit / Range | Notes |
|---------|---------|--------------|-------|
| `open` | first tick price in bar | points | Raw price; normalize for model |
| `high` | max tick price in bar | points | |
| `low` | min tick price in bar | points | |
| `close` | last tick price in bar | points | |
| `log_return` | `log(close / close_prev)` | dimensionless | Preferred over raw price for NN |
| `bar_range` | `high - low` | points | |
| `bar_range_atr` | `bar_range / ATR_14` | dimensionless | Vol-normalized range |
| `body` | `close - open` | points | Signed candle body |
| `body_atr` | `body / ATR_14` | dimensionless | |
| `upper_wick` | `high - max(open, close)` | points | |
| `lower_wick` | `min(open, close) - low` | points | |
| `clv` | `(close - low) / (high - low + eps)` | [0, 1] | Close location value — where close sits in bar |

**Sequence input:** `log_return`, `bar_range_atr`, `clv` — not raw OHLC prices.

**Intermediate only (not fed to NN):** `open`, `high`, `low`, `close`, `bar_range`, `body`, `upper_wick`, `lower_wick`, `body_atr`.

**ATR_14:** average true range over prior 14 bars — intermediate; used to compute `bar_range_atr`, `range_vs_atr`, etc.

---

### 1.2 Volume

From tick `size` (or `txn_count`) summed per bar.

| Feature | Formula | Notes |
|---------|---------|-------|
| `volume` | sum of tick sizes in bar | Total contracts traded |
| `num_trades` | count of ticks in bar | Trade arrival rate proxy |
| `avg_trade_size` | `volume / num_trades` | Large vs small print mix |

---

### 1.3 Volume Delta

Requires classifying each tick as buy or sell (aggressor side or tick rule: price uptick → buy, downtick → sell).

| Feature | Formula | Notes |
|---------|---------|-------|
| `buy_volume` | sum of tick size where direction = buy | |
| `sell_volume` | sum of tick size where direction = sell | |
| `volume_delta` | `buy_volume - sell_volume` | Signed net aggression |
| `volume_delta_pct` | `volume_delta / (volume + eps)` | Normalized to [-1, +1] |
| `buy_volume_pct` | `buy_volume / (volume + eps)` | Fraction of bar that was bought |

**Sequence input:** `volume_delta` **and** `volume_delta_pct` are both used (not interchangeable). Also `buy_volume_pct`.

**Intermediate only:** `buy_volume`, `sell_volume` (components used to compute delta features).

**Tick classification (when no aggressor flag):**
```
if price > prev_price  → buy
if price < prev_price  → sell
if price == prev_price → inherit previous direction (or split 50/50)
```

---

### 1.4 CVD (Cumulative Volume Delta)

Running sum of `volume_delta`, reset at each session open (RTH or full session — pick one and stay consistent).

| Feature | Formula | Notes |
|---------|---------|-------|
| `cvd` | `cumsum(volume_delta)` per session | Absolute level — non-stationary |
| `cvd_change_1` | `cvd - cvd.shift(1)` | Same as `volume_delta` — redundant but useful as channel |
| `cvd_slope_5` | `(cvd - cvd.shift(5)) / 5` | Short-term CVD momentum |
| `cvd_slope_20` | `(cvd - cvd.shift(20)) / 20` | Medium-term CVD momentum |
| `cvd_z` | `(cvd - rolling_mean(cvd, 60)) / rolling_std(cvd, 60)` | intermediate only — uses normalization; not in sequence |

**Sequence input:** `cvd_slope_5`.

**Intermediate only:** `cvd`, `cvd_change_1` (redundant with `volume_delta`), `cvd_slope_20`, `cvd_z`.

---

### 1.5 POC / VAL / VAH (per-bar volume profile)

Built from tick volume at each **price level** within the bar.

**Step 1 — volume-at-price histogram for bar:**
```
For each tick (price, size):
    vol_at_price[price] += size
```

**Step 2 — derive levels:**

| Feature | Definition | Notes |
|---------|------------|-------|
| `poc` | price level with highest volume in bar | Point of Control |
| `vah` | upper bound of value area | See below |
| `val` | lower bound of value area | See below |
| `va_width` | `vah - val` | Width of value area |
| `poc_volume_pct` | `vol_at_price[poc] / volume` | How dominant POC is |

**Value area (70% rule):**
1. Start at POC; captured_vol = vol_at_price[poc]
2. Expand one tick up or down, adding the side with more volume
3. Repeat until captured_vol ≥ 70% of bar volume
4. Upper bound = VAH, lower bound = VAL

**Distance features:**

| Feature | Formula | Sequence | Notes |
|---------|---------|----------|-------|
| `close_vs_poc` | `close - poc` | **yes** | points — raw distance |
| `close_vs_vah` | `close - vah` | **yes** | points — raw distance |
| `close_vs_val` | `close - val` | **yes** | points — raw distance |
| `close_in_value_area` | 1 if `val <= close <= vah`, else 0 | **yes** | binary |
| `close_above_vah` | 1 if `close > vah` | **yes** | breakout above value |
| `close_below_val` | 1 if `close < val` | **yes** | breakdown below value |
| `close_vs_poc_atr` | `(close - poc) / ATR_14` | no | intermediate / optional ablation only |

**Intermediate only:** `poc`, `vah`, `val` (raw price levels), `va_width`, `poc_volume_pct`, `vol_at_price` histogram.

---

### 1.6 Market Structure (from bars)

Swing points detected on **1m or 5m bars** using local extrema with confirmation lag `k` (recommended k=5).

**Swing detection:**
```
swing_high at bar i  if  high[i] == max(high[i-k .. i+k])
swing_low  at bar i  if  low[i]  == min(low[i-k .. i+k])
Only use confirmed swings (known k bars after the pivot).
Apply alternating filter: SH → SL → SH → SL (drop consecutive same-type).
```

**Structure features at decision bar `t`:**

| Feature | Formula | Values | Notes |
|---------|---------|--------|-------|
| `last_swing_high` | price of most recent confirmed SH | points | |
| `last_swing_low` | price of most recent confirmed SL | points | |
| `prev_swing_high` | SH before last_swing_high | points | For HH/LH |
| `prev_swing_low` | SL before last_swing_low | points | For HL/LL |
| `is_HH` | 1 if `last_swing_high > prev_swing_high` | 0/1 | Higher high |
| `is_LH` | 1 if `last_swing_high < prev_swing_high` | 0/1 | Lower high |
| `is_HL` | 1 if `last_swing_low > prev_swing_low` | 0/1 | Higher low |
| `is_LL` | 1 if `last_swing_low < prev_swing_low` | 0/1 | Lower low |
| `leg_size` | `last_swing_high - last_swing_low` | points | **Sequence** — last completed leg size |
| `retrace_pct` | `(last_swing_high - close) / leg_size` in uptrend | [0, 1] | **Sequence** — pullback depth |
| `bars_since_swing_high` | bars since last SH confirmed | integer | **Sequence** — scale ÷ 60 for NN |
| `bars_since_swing_low` | bars since last SL confirmed | integer | **Sequence** — scale ÷ 60 for NN |
| `dist_to_swing_high_pct` | `(last_swing_high - close) / leg_size × 100` | [0, 100] | **Sequence** — % of leg below SH |
| `dist_to_swing_low_pct` | `(close - last_swing_low) / leg_size × 100` | [0, 100] | **Sequence** — % of leg above SL |
| `is_HH` | 1 if `last_swing_high > prev_swing_high` | 0/1 | optional sequence / Part 2 |
| `is_LH` | 1 if `last_swing_high < prev_swing_high` | 0/1 | optional sequence / Part 2 |
| `is_HL` | 1 if `last_swing_low > prev_swing_low` | 0/1 | optional sequence / Part 2 |
| `is_LL` | 1 if `last_swing_low < prev_swing_low` | 0/1 | optional sequence / Part 2 |
| `trend_state` | +1 uptrend (HH+HL), -1 downtrend (LH+LL), 0 chop | {-1,0,+1} | optional sequence / Part 2 |
| `bos_bull` | 1 if close breaks above last_swing_high | 0/1 | optional sequence / Part 2 |
| `bos_bear` | 1 if close breaks below last_swing_low | 0/1 | optional sequence / Part 2 |

**Intermediate only:** `last_swing_high`, `last_swing_low`, `prev_swing_high`, `prev_swing_low` (raw swing prices), `leg_size_atr`, `dist_to_swing_high_atr`, `dist_to_swing_low_atr` (replaced by `_pct`).

**Why percentage for swing distance:** easier to interpret than ATR units — "price is 35% of the leg below the swing high" vs "0.8 ATR away".

**Retrace_pct in downtrend:**
```
retrace_pct = (close - last_swing_low) / (last_swing_high - last_swing_low)
```

---

### 1.7 Session Time

| Feature | Formula | Notes |
|---------|---------|-------|
| `mins_from_rth_open` | minutes since 9:30 AM ET on current RTH session | 0 at open, ~390 at close |
| `session_rth` | 1 if bar is within RTH (9:30–16:00 ET), else 0 | Filter only — not a sequence channel |

**Computation:**
```
1. Convert bar timestamp to America/New_York
2. Find current session's RTH open datetime (today 9:30 ET)
3. mins_from_rth_open = (bar_timestamp - rth_open).total_seconds() / 60
4. Reset to 0 at each new RTH session; NaN or negative for pre-market bars
```

**Why core:** open-drive (first 30 min) vs midday vs close behave very differently for continuation vs reversal.

---

### 1.8 Volatility / Regime

| Feature | Formula | Notes |
|---------|---------|-------|
| `atr_14` | 14-bar average true range | TR = max(H-L, \|H-Cprev\|, \|L-Cprev\|) — intermediate |
| `atr_5` | 5-bar average true range | intermediate |
| `atr_ratio` | `atr_5 / atr_14` | **Sequence** — >1 expanding, <1 contracting |
| `range_vs_atr` | `bar_range / atr_14` | **Sequence** |
| `realized_vol_20` | std(`log_return`, window=20) | **Sequence** |
| `atr_percentile_20d` | percentile rank of `atr_14` vs prior 20 sessions' median ATR | **Sequence** |

**Why core:** retracement vs reversal often separates by vol expansion; `bar_range_atr` (§1.1) depends on ATR.

---

### 1.9 Rolling 60-min volume profile — `vp60_vol_at_close_pct`

Measures **how much volume traded at the current price relative to the POC** in a rolling **60-minute** cumulative volume profile. Not profile shape (P/b/B/D) and not share of total volume.

**Data source:** merge per-bar `vol_at_price` histograms (§1.5) from the last **60 completed** 1-min bars `[t-60 .. t-1]` at bar `t` (no lookahead).

**Step 1 — merge histograms:**
```
vp60 = merge vol_at_price from bars[t-60 .. t-1]
total_vol = sum(vp60.values())
poc = price with max volume in vp60
vol_at_poc = vp60[poc]
```

**Step 2 — volume at close (tick-rounded, ES tick = 0.25):**
```
price_key = round_to_tick(close)
vol_at_close = vp60.get(price_key, 0)    # 0 if no trades at that level
```

**Step 3 — feature (0–100% scale):**
```
vp60_vol_at_close_pct = min(100, vol_at_close / vol_at_poc × 100)
```

| Value | Meaning |
|-------|---------|
| **100%** | Price is at the **POC** (strongest volume node in 60-min profile) |
| **30%–99%** | Meaningful volume at this price, but below POC strength |
| **< 30%** | Thin node — typically near **VAL / VAH** or low-acceptance area |
| **0%** | **No volume** traded at this price in the last 60 minutes (LVN) |

**Examples:**

| Close price | vol at price | vol at POC | `vp60_vol_at_close_pct` |
|-------------|--------------|------------|-------------------------|
| 5850.00 (POC) | 12,000 | 12,000 | **100%** |
| 5849.75 | 2,400 | 12,000 | **20%** |
| 5851.25 | 0 | 12,000 | **0%** |
| 5849.50 (VAL area) | 2,000 | 12,000 | **~17%** |

**Sequence input:** `vp60_vol_at_close_pct` — raw [0, 100].

**Intermediate only:** merged `vp60` dict, `vp60_poc`, `vp60_val`, `vp60_vah`, `vol_at_close`, `vol_at_poc`, `total_vol` (for debugging and optional future features).

**Warmup:** if `i < 60` (fewer than 60 prior bars), set `NaN`.

**This is NOT:**
- `vol_at_close / total_vol × 100` (share of total — different metric)
- `vp60_price_percentile` (CDF position in profile)
- Profile shape classification (P / b / B / D)

**Python reference:**
```python
TICK = 0.25

def merge_profiles(profile_list):
    merged = {}
    for prof in profile_list:
        if not prof:
            continue
        for price, vol in prof.items():
            merged[price] = merged.get(price, 0.0) + vol
    return merged

def round_to_tick(price, tick=TICK):
    return round(price / tick) * tick

def vp60_vol_at_close_pct(vp60, close, tick=TICK):
    if not vp60:
        return float("nan")
    poc = max(vp60, key=vp60.get)
    vol_at_poc = vp60[poc]
    if vol_at_poc <= 0:
        return float("nan")
    vol_at_close = vp60.get(round_to_tick(close, tick), 0.0)
    return min(100.0, vol_at_close / vol_at_poc * 100.0)
```

---

All model inputs are a **single sequence tensor** per decision event: shape `(L, C)` where `L=60` bars, `C=` number of channels below.

### Core sequence channels (C = 26)

| # | Channel | Group | Stored value |
|---|---------|-------|--------------|
| 1 | `log_return` | OHLC | raw |
| 2 | `bar_range_atr` | OHLC | raw |
| 3 | `clv` | OHLC | raw [0,1] |
| 4 | `volume` | Volume | raw (contracts) |
| 5 | `volume_delta` | Order flow | raw (contracts) |
| 6 | `volume_delta_pct` | Order flow | raw [-1,1] |
| 7 | `buy_volume_pct` | Order flow | raw [0,1] |
| 8 | `cvd_slope_5` | Order flow | raw |
| 9 | `close_vs_poc` | Volume profile (bar) | raw (points) |
| 10 | `close_vs_vah` | Volume profile (bar) | raw (points) |
| 11 | `close_vs_val` | Volume profile (bar) | raw (points) |
| 12 | `close_in_value_area` | Volume profile (bar) | 0/1 |
| 13 | `close_above_vah` | Volume profile (bar) | 0/1 |
| 14 | `close_below_val` | Volume profile (bar) | 0/1 |
| 15 | `vp60_vol_at_close_pct` | Volume profile (60m) | raw [0,100] percent vs POC |
| 16 | `leg_size` | Structure | raw (points) |
| 17 | `retrace_pct` | Structure | raw [0,1] |
| 18 | `bars_since_swing_high` | Structure | raw (integer bars) |
| 19 | `bars_since_swing_low` | Structure | raw (integer bars) |
| 20 | `dist_to_swing_high_pct` | Structure | raw [0,100] percent |
| 21 | `dist_to_swing_low_pct` | Structure | raw [0,100] percent |
| 22 | `mins_from_rth_open` | Session | raw (minutes) |
| 23 | `atr_ratio` | Volatility | raw |
| 24 | `range_vs_atr` | Volatility | raw |
| 25 | `realized_vol_20` | Volatility | raw |
| 26 | `atr_percentile_20d` | Volatility | raw [0,1] |

| Group | Channels |
|-------|----------|
| OHLC-derived | 3 |
| Volume | 1 |
| Order flow (delta + CVD) | 4 |
| Volume profile — per bar (§1.5) | 6 |
| Volume profile — 60m roll (§1.9) | 1 |
| Market structure | 6 |
| Session time | 1 |
| Volatility / Regime | 4 |
| **Total C** | **26** |

Input tensor shape: **`(60, 26)`** per decision event.

### What are `close_vs_poc`, `close_vs_vah`, `close_vs_val`? (channels #9–11)

These three features measure **how far the bar's close is from the volume-profile levels inside that same 1-minute bar**. They are in **ES points** (each point = 4 ticks = $50 per contract).

**How POC / VAH / VAL are built for one bar:**
1. From ticks, build a histogram: at each price level, sum traded volume.
2. **POC** = price with the most volume (Point of Control).
3. **VAL / VAH** = lower / upper bound of the "value area" — expand from POC until 70% of bar volume is captured.

**The three distance features:**

| Feature | Formula | Example | Meaning |
|---------|---------|---------|---------|
| `close_vs_poc` | `close - poc` | close=5850.50, poc=5850.00 → **+0.50** | Close is 0.50 pts above where most volume traded |
| `close_vs_vah` | `close - vah` | close=5851.00, vah=5850.75 → **+0.25** | Close is 0.25 pts above top of value area (breakout) |
| `close_vs_val` | `close - val` | close=5849.50, val=5850.00 → **-0.50** | Close is 0.50 pts below bottom of value area |

**Why three separate channels:** POC tells you where volume concentrated; VAH/VAL define the "fair range" for that bar. Close above VAH suggests acceptance higher; close below VAL suggests rejection lower; close near POC suggests balance.

**Positive vs negative:**
- `close_vs_poc > 0` → close above POC (bullish close relative to volume center)
- `close_vs_vah > 0` → close above value area (breakout)
- `close_vs_val < 0` → close below value area (breakdown)

Optional Part 2 channels (add later): `is_HH`, `is_HL`, `is_LH`, `is_LL`, `trend_state`, `bos_bull`, `bos_bear`, `choch_bull`, etc.

---

---

## Part 2 — Good-to-Have Features (add after core works)

These may improve reversal vs retracement separation. Implement only after baseline macro-F1 is measured with Part 1.

### 2.1 Session / VWAP

| Feature | Formula | Why useful |
|---------|---------|------------|
| `session_vwap` | cumsum(price × volume) / cumsum(volume) per session | Institutional reference price |
| `close_vs_vwap_atr` | `(close - session_vwap) / ATR_14` | Premium/discount to fair value |
| `vwap_slope_10` | slope of VWAP over 10 bars | VWAP trend |
| `hour_sin`, `hour_cos` | cyclical time encoding | Periodic session effects |

### 2.2 Multi-Timeframe Context

Resample to 5m and 15m; compute at decision time (no lookahead):

| Feature | Formula | Why useful |
|---------|---------|------------|
| `trend_5m` | sign of 5m slope over 20 bars | Higher-TF direction |
| `trend_15m` | sign of 15m slope over 20 bars | Macro bias |
| `tf_agreement` | 1 if 1m/5m/15m trends agree | Aligned vs counter-trend setups |
| `rsi_14_5m` | RSI on 5m bars | Overbought/oversold at HTF |
| `ema20_dist_atr_5m` | distance to 5m EMA20 / ATR | HTF mean reversion distance |

### 2.3 Rolling Volume Profile (multi-bar)

Optional extensions beyond §1.9 `vp60_vol_at_close_pct`. Single-bar POC is noisy; longer windows are smoother.

| Feature | Formula | Why useful |
|---------|---------|------------|
| `roll_poc_10` | POC from merged vol-at-price of last 10 bars | Stable support/resistance |
| `roll_vah_10`, `roll_val_10` | value area from last 10 bars | |
| `close_vs_roll_poc` | `close - roll_poc_10` (points) | distance to rolling POC |
| `poc_shift_5` | change in rolling POC over 5 bars | Profile migration |

### 2.4 Order Flow — Advanced

| Feature | Formula | Why useful |
|---------|---------|------------|
| `delta_divergence` | price up but volume_delta negative (or vice versa) | Hidden weakness/strength |
| `delta_streak` | consecutive bars with same delta sign | Sustained pressure |
| `absorption` | high volume + small bar_range_atr | Limit order absorption |
| `large_trade_ratio` | volume from ticks ≥ p90 size / total volume | Institutional activity |
| `trade_intensity` | num_trades / 60 seconds | Speed of trading |
| `intensity_surge` | trade_intensity / rolling_mean(intensity, 20) | Activity spike |

### 2.5 Price Level Imbalance (from tick price levels)

| Feature | Formula | Why useful |
|---------|---------|------------|
| `stacked_buy_imbalance` | count of price levels where buy_vol ≥ 3× sell_vol at level below | Bid stacking |
| `stacked_sell_imbalance` | count of price levels where sell_vol ≥ 3× buy_vol at level above | Offer stacking |
| `net_level_imbalance` | stacked_buy - stacked_sell | Order book pressure proxy |

### 2.6 Gap / Prior Session

| Feature | Formula | Why useful |
|---------|---------|------------|
| `overnight_gap` | today RTH open - yesterday RTH close | Gap context |
| `gap_atr` | overnight_gap / ATR_14 | Normalized gap |
| `dist_to_prior_session_high_atr` | distance to yesterday high | Resistance proximity |
| `dist_to_prior_session_low_atr` | distance to yesterday low | Support proximity |

### 2.7 Derived Structure (advanced)

| Feature | Formula | Why useful |
|---------|---------|------------|
| `choch_bull` | break of last LH in downtrend → potential reversal up | Change of character |
| `choch_bear` | break of last HL in uptrend → potential reversal down | |
| `swing_failure` | price breaks SH/SL then closes back inside | Liquidity grab / trap |
| `equal_highs` | last two SH within 0.25 pts | Double top zone |
| `equal_lows` | last two SL within 0.25 pts | Double bottom zone |

**Implementation:** see [NN_model_features_part2_8_impl.md](NN_model_features_part2_8_impl.md)

---

## Part 2 Summary — Good-to-have count

| Group | Features |
|-------|----------|
| Session / VWAP | 4 |
| Multi-Timeframe | 5 |
| Rolling Volume Profile | 4 |
| Order Flow Advanced | 6 |
| Price Level Imbalance | 3 |
| Gap / Prior Session | 4 |
| Derived Structure | 5 |
| **Total** | **~31** |

---

## Implementation order

```
Step 1:  ticks → 1m bars → OHLC + volume
Step 2:  tick direction → volume_delta, volume_delta_pct, CVD
Step 3:  vol-at-price histogram → POC, VAL, VAH per bar
Step 4:  rolling 60m merge → vp60_vol_at_close_pct (§1.9)
Step 5:  swing detection → structure columns (leg_size, retrace_pct, dist_*_pct, bars_since_*)
Step 6:  session time + volatility regime (§1.7, §1.8)
Step 7:  decision points + labels
Step 8:  sequence builder → (60, 26) numpy array per decision event
Step 9:  train TCN/LSTM on sequences + walk-forward validation
Step 10: add Part 2 channels one at a time, measure lift
```

---

## Feature storage rules (no z-score)

**Do not apply z-score (or any rolling mean/std normalization) when creating or storing features.** Each column is saved as its computed raw value from the formula.

| Feature type | Stored as |
|--------------|-----------|
| Raw OHLC prices | intermediate only — not sequence channels |
| `close_vs_poc`, `close_vs_vah`, `close_vs_val` | raw point distance (e.g. +0.75, -1.25) |
| `volume`, `volume_delta`, `leg_size` | raw numeric value |
| `volume_delta_pct`, `buy_volume_pct`, `retrace_pct` | raw ratio |
| `dist_to_swing_*_pct` | raw percent [0–100] |
| `bars_since_swing_*` | raw integer bar count |
| `mins_from_rth_open` | raw minutes since 9:30 ET |
| `vp60_vol_at_close_pct` | raw [0–100] percent vs 60m POC |
| Binary flags | 0/1 |
| CVD | `cvd_slope_5` only — not raw `cvd` |

**Normalization (if needed) is a separate model-training step** — not part of the feature pipeline. The NN can use a built-in `BatchNorm` layer or a one-time scaler fit on the training split only.

---

## Model input vs intermediate columns

**Pipeline:** compute all base columns → store in Parquet → select **26 sequence channels** for `.npy` export.

### Intermediate only — computed but NOT fed to NN

| Category | Columns |
|----------|---------|
| OHLC raw | `open`, `high`, `low`, `close`, `bar_range`, `body`, `upper_wick`, `lower_wick`, `body_atr` |
| Volatility base | `atr_14`, `atr_5` (used to derive `bar_range_atr`, `range_vs_atr`) |
| Volume components | `num_trades`, `avg_trade_size`, `buy_volume`, `sell_volume` |
| CVD raw | `cvd`, `cvd_change_1`, `cvd_slope_20`, `cvd_z` |
| POC raw prices | `poc`, `vah`, `val`, `va_width`, `poc_volume_pct`, `vol_at_price` |
| VP60 intermediate | merged `vp60` dict, `vp60_poc`, `vp60_val`, `vp60_vah`, `vol_at_close`, `vol_at_poc` |
| POC ATR distance | `close_vs_poc_atr` (superseded by `close_vs_poc` in sequence) |
| Swing raw prices | `last_swing_high`, `last_swing_low`, `prev_swing_high`, `prev_swing_low` |
| Structure ATR | `leg_size_atr`, `dist_to_swing_high_atr`, `dist_to_swing_low_atr` |
| Session filter | `session_rth` (used to filter data, not a channel) |

### Explicit sequence inclusion rules (user-defined)

| Rule | Columns |
|------|---------|
| **Both used** | `volume_delta` **and** `volume_delta_pct` |
| **Points, not ATR** | `close_vs_poc`, `close_vs_vah`, `close_vs_val` |
| **Leg in points** | `leg_size` (not `leg_size_atr`) |
| **Swing dist in %** | `dist_to_swing_high_pct`, `dist_to_swing_low_pct` (not `_atr`) |
| **Structure in sequence** | `retrace_pct`, `bars_since_swing_high`, `bars_since_swing_low` |

### Pairs to avoid duplicating in sequence

```
bar_range + bar_range_atr     →  keep bar_range_atr only
cvd + cvd_slope_5             →  keep cvd_slope_5 only
poc + close_vs_poc            →  keep close_vs_poc only (poc is intermediate)
leg_size + leg_size_atr       →  keep leg_size only
dist_*_pct + dist_*_atr       →  keep dist_*_pct only
```

`volume_delta` + `volume_delta_pct` are **intentionally both kept** — different scale (contracts vs ratio).

---

## Notes

- **Single-bar POC/VAL/VAH** is noisier on low-volume bars; no volume gate — compute for all bars with tick data.
- **Market structure** should use confirmed swings only (k-bar lag) to avoid lookahead.
- **CVD** must reset at a consistent session boundary (recommend RTH open 9:30 ET).
- Part 2 features are candidates only — add after Part 1 baseline is working.
