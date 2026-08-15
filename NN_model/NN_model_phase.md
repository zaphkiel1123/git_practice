# ES Mini NN Model — Implementation Phases

## Phase 1 — Data (no ML yet)

- Ticks → 1m bars CSV/Parquet
- Swing detection → HH/HL columns
- Label script: for each decision point, look forward H bars → class

## Phase 2 — Baseline

- Feature CSV: one row per decision point, ~50 columns
- LightGBM + walk-forward split
- Check: macro-F1 > 0.35? confusion matrix sensible?

## Phase 3 — Sequences

- For each decision point, save numpy array (128, 8)
- Train small TCN: input (128, 8), output 3 classes

## Phase 4 — Hybrid

- PyTorch/TensorFlow model with two inputs
- Same train/val splits as Phase 2
- If hybrid beats both alone → keep; else use simpler model

## Phase 5 — Trading check

- Only trade when prob > threshold (e.g. 0.55)
- Backtest with 1 tick slippage on ES
