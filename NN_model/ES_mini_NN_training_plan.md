# ES Mini: Continuation / Retracement / Reversal — Training Plan

## 1. Is years of tick data enough?

**Yes.** Years of ES mini tick data is more than enough for sample count.

| Aggregation        | Approx. samples/year |
|--------------------|----------------------|
| 1-min bars         | ~100k–250k           |
| 5-min bars         | ~20k–50k             |
| Tick bars (500-tick)| ~50k–200k+          |
| Event samples      | ~5k–30k              |

Neural nets typically need tens of thousands of labeled examples per class. Tick history easily exceeds that once aggregated.

**What actually limits success:**
1. Label clarity (retracement vs reversal must be rule-based)
2. Regime coverage (crash, bear, low-vol grind)
3. No leakage (features = past only; labels = future path only)
4. Walk-forward validation (not random split)
5. Class balance (reversals are often rare)

---

## 2. Training pipeline

**Decision points:** after swing confirmation, after trend leg ≥ 1× ATR, at structure retest — not every tick/bar.

---

## 3. Label definition

Horizon H (e.g. 30–120 min or N bars). Example in uptrend:

| Class         | Rule |
|---------------|------|
| Continuation  | New high within H without breaking swing low; move ≥ X× ATR |
| Retracement   | Pullback 23.6%–61.8% of last leg (or 0.5–1.5× ATR), then trend resumes within H |
| Reversal      | Breaks swing low / prior HL + follow-through ≥ Y× ATR within H |

Also: minimum trend leg, fixed H, neutral/discard bucket for ambiguous cases (~20–40% discarded).

---

## 4. Model options

| Model              | Best for |
|--------------------|----------|
| LightGBM           | Engineered tabular features (baseline) |
| TCN / LSTM         | Raw sequence windows (OHLCV + microstructure) |
| Hybrid (fusion NN) | Sequence branch + tabular branch combined |

LightGBM vs NN on same tabular features: often 0–3 pp accuracy; LightGBM wins more often.
NN on sequences can gain 2–5 pp if path-dependent patterns matter.

---

## 5. Feature plan

### A. Multi-timeframe price / structure (1m, 5m, 15m, 60m)
- Log returns (1, 3, 5, 10, 20 bars)
- Trend slope (linear reg on close)
- MA distance: (price − EMA) / ATR
- HH/HL/LH/LL flags, swing leg size, retrace depth
- BOS / CHoCH flags
- Donchian position, RSI, ADX (optional)

### B. Volatility
- ATR (multi-window), ATR percentile, realized vol, range ratio, BB width

### C. Volume / microstructure (from ticks)
- Volume delta, CVD slope, VWAP distance
- Volume profile: POC distance, value area position
- Trade intensity, large-trade ratio

### D. Session / time
- Minutes from RTH open, gap/ATR, prior session H/L proximity, day of week

### E. Sequence input for NN
Window of L bars (e.g. 128 × 1m), channels per bar:
- log return, range/ATR, volume, volume delta, CLV, VWAP distance, etc.
- Normalize per window (z-score)

---

## 6. Evaluation

1. Walk-forward: train 2–3 yr → val 3–6 mo → roll
2. Purged embargo ≥ horizon H
3. Metrics: macro-F1, confusion matrix, reversal precision, simulated PnL (1 tick slippage)
4. Compare: LightGBM, TCN/LSTM, hybrid
5. Ablations: drop microstructure, multi-TF, structure features

---

## 7. Roadmap (8–12 weeks)

1. Tick cleaning, continuous contract, bars, VWAP
2. Swing detection + labels + neutral bucket
3. Feature store (tabular + sequences)
4. LightGBM baseline
5. TCN/hybrid NN + ablations
6. No-trade zone + execution sim