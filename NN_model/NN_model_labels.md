# ES Mini NN Model — Label Specification

Label groups for multi-task training. All labels are computed from **future bars** `[t+1 .. t+H]` only — never used as model input.

**Horizon H:** 60 bars (60 minutes). Consistent across all heads.

**Shared intermediate values (computed once, used by multiple groups):**
```
mfe_long  = max(high[t+1 .. t+H]) - close[t]
mfe_short = close[t] - min(low[t+1 .. t+H])
future_range = max(high[t+1 .. t+H]) - min(low[t+1 .. t+H])
```

---

## Group A — MFE Opportunity (Primary Head)

Classifies whether a **high-conviction directional opportunity** existed in the next H bars, based on Maximum Favorable Excursion normalized by ATR.

### 3-class label

| Label | Value | Condition |
|-------|-------|-----------|
| `strong_long_opp` | 0 | `mfe_long / ATR_14 >= 1.5` **and** `mfe_long > 2 * mfe_short` |
| `strong_short_opp` | 1 | `mfe_short / ATR_14 >= 1.5` **and** `mfe_short > 2 * mfe_long` |
| `no_edge` | 2 | Everything else |

**Why 3 classes (no "weak" tier):**
- Cleaner decision boundaries — the model focuses on separating high-conviction setups from noise
- More actionable — you would not trade a "weak" signal anyway
- Weak samples absorbed into `no_edge`, which is the correct framing if only high-conviction trades matter
- Precision on the classes that matter is higher with fewer boundaries to learn

**Threshold tuning:** 1.5 ATR is the starting point. Sweep [1.0, 1.2, 1.5, 2.0] on the validation set and pick the cutoff that maximizes precision on the strong classes while keeping at least ~5% representation per class.

### What to do when `strong_long_opp` fires — SL and TP placement

When the model outputs high probability for `strong_long_opp`, the label definition itself tells you the price moved at least 1.5 ATR upward within H bars. Use that structure to place orders:

**Stop Loss (SL):**

| Method | Formula | Rationale |
|--------|---------|-----------|
| Structure-based (preferred) | `SL = last_swing_low - 0.25` | Places stop just below the most recent confirmed swing low. If price breaks that level, the structure that generated the signal is invalidated. |
| ATR-based (fallback) | `SL = close[t] - 1.0 * ATR_14` | Fixed 1-ATR risk. Use when the nearest swing low is too far away (> 2 ATR) or too close (< 0.3 ATR). |
| Hybrid | `SL = max(last_swing_low - 0.25, close[t] - 1.5 * ATR_14)` | Structure stop floored by a maximum risk cap. Prevents catastrophic loss if swing low is abnormally far. |

**Take Profit (TP):**

| Method | Formula | Rationale |
|--------|---------|-----------|
| MFE-aligned | `TP = close[t] + 1.5 * ATR_14` | Matches the label threshold — if the model says "strong long opp," you're targeting the same magnitude it was trained to recognize. |
| Structure-based | `TP = last_swing_high` or `prev_swing_high` | Takes profit at the next overhead resistance. Conservative but high hit rate. |
| R-multiple | `TP = close[t] + 2 * (close[t] - SL)` | 2R reward-to-risk. SL determines TP dynamically. If SL is 4 pts below entry, TP is 8 pts above. |

**Recommended approach:**

```
risk     = close[t] - SL                    # distance to stop
TP_1     = close[t] + 1.0 * risk            # 1R — take partial (50%)
TP_2     = close[t] + 2.0 * risk            # 2R — take remaining (50%)
max_hold = H bars                           # time stop — exit if neither TP nor SL hit
```

Scale out: sell half at 1R, trail stop to breakeven, let the rest run to 2R or time stop.

**Mirror for `strong_short_opp`:** flip all directions. `SL = last_swing_high + 0.25`, `TP = close[t] - 1.5 * ATR_14`, etc.

### Label computation (Python reference)

```python
def label_mfe_opportunity(close_t, highs_future, lows_future, atr_14):
    """
    close_t:      close price at decision bar t
    highs_future: array of high prices for bars [t+1 .. t+H]
    lows_future:  array of low prices for bars [t+1 .. t+H]
    atr_14:       ATR(14) at bar t
    """
    mfe_long  = highs_future.max() - close_t
    mfe_short = close_t - lows_future.min()

    mfe_long_atr  = mfe_long / atr_14
    mfe_short_atr = mfe_short / atr_14

    if mfe_long_atr >= 1.5 and mfe_long > 2 * mfe_short:
        return 0  # strong_long_opp
    elif mfe_short_atr >= 1.5 and mfe_short > 2 * mfe_long:
        return 1  # strong_short_opp
    else:
        return 2  # no_edge
```

---

## Group B — Volatility Expansion / Contraction (Auxiliary Head 1)

Predicts whether the market is about to move significantly, regardless of direction.

```
range_ratio = future_range / ATR_14
```

### 3-class label

| Label | Value | Condition |
|-------|-------|-----------|
| `expansion` | 0 | `range_ratio >= 1.5` |
| `normal` | 1 | `0.7 <= range_ratio < 1.5` |
| `contraction` | 2 | `range_ratio < 0.7` |

**Why this is useful as an auxiliary head:**
- Easier to predict than direction — volatility clusters (GARCH effect)
- Directly actionable: expansion = widen targets, contraction = skip or scalp
- Features `atr_ratio`, `atr_percentile_20d`, `realized_vol_20`, `range_vs_atr` are purpose-built for this
- Forces the shared encoder to learn volatility regime representations that also help the primary MFE head

### Label computation (Python reference)

```python
def label_vol_regime(highs_future, lows_future, atr_14):
    future_range = highs_future.max() - lows_future.min()
    range_ratio  = future_range / atr_14

    if range_ratio >= 1.5:
        return 0  # expansion
    elif range_ratio >= 0.7:
        return 1  # normal
    else:
        return 2  # contraction
```

---

## Group C — Directional Regime (Auxiliary Head 2)

Classifies the type of price action that follows, based on market structure behavior over the next H bars.

### 4-class label

| Label | Value | Condition |
|-------|-------|-----------|
| `continuation` | 0 | Price moves >= 1 ATR in `trend_state` direction without retracing > 50% of the move |
| `retracement` | 1 | Price retraces 38–62% of last `leg_size`, then resumes trend direction (close[t+H] beyond the retracement low/high) |
| `reversal` | 2 | Price breaks the opposing swing level (BOS opposite to `trend_state`) within H bars |
| `chop` | 3 | None of the above — total excursion < 0.5 ATR in either direction, or conflicting signals |

**Definitions require `trend_state` at bar t:**
- `trend_state = +1` (uptrend): last swing pattern is HH + HL
- `trend_state = -1` (downtrend): last swing pattern is LH + LL
- `trend_state = 0` (chop): mixed — label defaults to `chop` regardless of future price action

### Label computation (Python reference)

```python
def label_regime(close_t, highs_future, lows_future, close_future_H,
                 trend_state, leg_size, last_swing_high, last_swing_low, atr_14):
    """
    close_future_H: close price at bar t+H
    trend_state:    +1 (uptrend), -1 (downtrend), 0 (chop) at bar t
    """
    if trend_state == 0:
        return 3  # chop

    future_range = highs_future.max() - lows_future.min()

    # Check reversal first (BOS in opposite direction)
    if trend_state == +1 and lows_future.min() < last_swing_low:
        return 2  # reversal — broke below swing low in uptrend
    if trend_state == -1 and highs_future.max() > last_swing_high:
        return 2  # reversal — broke above swing high in downtrend

    # Check continuation
    if trend_state == +1:
        move = highs_future.max() - close_t
        retrace = close_t - lows_future.min()
    else:
        move = close_t - lows_future.min()
        retrace = highs_future.max() - close_t

    if move >= 1.0 * atr_14 and (move == 0 or retrace / move <= 0.5):
        return 0  # continuation

    # Check retracement (pullback 38-62% of leg, then resume)
    if leg_size > 0:
        if trend_state == +1:
            pullback_depth = (close_t - lows_future.min()) / leg_size
            resumed = close_future_H > close_t
        else:
            pullback_depth = (highs_future.max() - close_t) / leg_size
            resumed = close_future_H < close_t

        if 0.38 <= pullback_depth <= 0.62 and resumed:
            return 1  # retracement

    # Default
    if future_range < 0.5 * atr_14:
        return 3  # chop
    return 3  # ambiguous → chop
```

---

## Training Strategy

### Phase 1 — Independent evaluation (train separately, compare)

Train three **separate models**, each with the same encoder architecture but a single classification head. This isolates each label group so you can evaluate which one the features support best.

```
Run 1:  Input (128, 26) → Encoder → Head A (Linear → 3 classes)   loss = CE_A
Run 2:  Input (128, 26) → Encoder → Head B (Linear → 3 classes)   loss = CE_B
Run 3:  Input (128, 26) → Encoder → Head C (Linear → 4 classes)   loss = CE_C
```

Each run is a standalone model trained from scratch with `loss = 1.0 * CE` for its own head. No weight sharing between runs.

**Compare across runs using:**

| Metric | What it tells you |
|--------|-------------------|
| Macro-F1 | Overall balance across all classes (most important) |
| Precision on minority class | How reliable the signal is (e.g. precision on `strong_long_opp`) |
| Recall on minority class | How many real opportunities the model catches |
| Validation loss curve | Whether the model is learning or just memorizing |

The group with the best macro-F1 and highest minority-class precision is your strongest label definition.

### Phase 2 — Multi-task combination (optional, after Phase 1)

After Phase 1 identifies which groups work, combine the best ones into a single multi-task model to see if the combination beats the individual:

```
Input (128, 26) → Shared Encoder → hidden representation
                                      ├── Head A: Linear → 3 classes (MFE)
                                      ├── Head B: Linear → 3 classes (Vol regime)
                                      └── Head C: Linear → 4 classes (Direction regime)
```

```python
loss = 1.0 * CE_A + 0.3 * CE_B + 0.3 * CE_C
```

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Primary head | 1.0 | The label group that won in Phase 1 |
| Auxiliary heads | 0.3 each | Supporting tasks — regularize the encoder |

**What the loss weights mean:**

The weights control how much each head's error influences the shared encoder's gradient update. During each training step:

```
CE_A = cross-entropy for Head A prediction    (e.g. 0.82)
CE_B = cross-entropy for Head B prediction    (e.g. 0.45)
CE_C = cross-entropy for Head C prediction    (e.g. 0.61)

total_loss = 1.0 * 0.82  +  0.3 * 0.45  +  0.3 * 0.61
           = 0.82        +  0.135       +  0.183
           = 1.138
```

The optimizer calls `total_loss.backward()`. The weights determine **relative gradient magnitude** from each head:

- **1.0** = full gradient strength — the encoder prioritizes this task
- **0.3** = 30% gradient strength — provides useful learning signal without overpowering the primary task

If the primary head's validation accuracy improves with auxiliaries, the multi-task setup is helping. If it degrades, drop to 0.1 or remove the weaker auxiliary.

**Only move to Phase 2 if Phase 1 shows at least two groups with meaningful accuracy.** If only one group works, single-task training is the right answer.

### Class imbalance handling

| Head | Expected distribution | Mitigation |
|------|----------------------|------------|
| A | no_edge ~85%, strong_long ~7%, strong_short ~8% | Class weights or focal loss (gamma=2) |
| B | normal ~55%, expansion ~25%, contraction ~20% | Mild class weights |
| C | chop ~50%, continuation ~20%, retracement ~18%, reversal ~12% | Class weights |

### Exclusion rules

- Exclude bars where `i < 60` (warmup for vp60 and rolling features)
- Exclude last H bars of each session (no future data for labels)
- Exclude bars where `ATR_14 <= 0` or `leg_size <= 0` (degenerate)

---

## Notes

- All thresholds (1.5 ATR, 0.7, etc.) are starting points — tune on validation set
- H = 60 bars is the baseline; sweep [30, 45, 60, 90] after baseline works
- Labels are computed once during data preparation, not during training
- The `mfe_long > 2 * mfe_short` condition in Group A ensures directional dominance — prevents labeling a bar as "strong long" when the downside was nearly as large (which would be a chop or reversal, not a clean opportunity)
