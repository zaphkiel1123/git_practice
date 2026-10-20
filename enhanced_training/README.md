# enhanced_training

Tooling for establishing whether a footprint/order-flow setup carries an edge,
before committing to a backtest. Built alongside `es_training/` and
`NN_model/`, reusing their record format and conventions but replacing their
measurement approach.

## Why this exists

Three findings from reviewing `es_training/`:

**The reported backtest was in-sample.** `run_trading.sh` invokes
`backtester.py` with neither `--data` nor `--features`, so it falls through to
the default branch and loads `training_features.parquet` from the models
directory — the exact file the models were fit on. The log confirms it:
`Bars loaded: 32429`, matching the row count of `training_features.csv`. The
85.4% win rate, 7.93 profit factor and Sharpe of 32.8 are the models grading
their own homework. The honest out-of-sample numbers in the same log are
`acc_trades` of 0.259, 0.313, 0.449, 0.377, 0.459 against a 0.50 coin-flip
baseline on that subset.

**Labels overlap 60-to-1.** Every bar is labelled with a 60-bar forward
horizon, so bar `t` and bar `t+1` share 59 of 60 future bars.
`walk_forward_split()` is a plain expanding window with no purge or embargo, so
each training set's tail overlaps its test set's head. 7,705 RTH bars carry
roughly 128 independent observations, not 7,705.

**Win rates conditioned on the future.** Both the fold metrics and the pattern
miner compute win rate over `trade_mask = y != 0` — only bars where some side
won. Bars where neither side won are dropped from the denominator, but live
you cannot skip those; you get stopped out on them.

The consequence for the original question — "some strategies only work in
certain conditions, and filtering which one beforehand is tricky" — is that
the condition-dependence was mostly a measurement artifact. Regime selection
built on top of signals whose edge was never established has nothing to stand
on.

## The missing feature family

Reading a footprint chart, the recurring observation is *"price came back to
this level and the response was weaker than the first time."* That is a paired
comparison between two visits to the same price.

Every feature in `es_training/data/models/training_meta.json` is a rolling
window aggregate — `pressure_ratio_5`, `flow_accel_3`, `imbalance_cluster_5`,
`va10_price_vs_poc`. No rolling window can express "there was 800 sell volume
here 20 minutes ago and only 90 buy volume now." `level_ledger.py` adds that
family.

## Run order

Steps 3 onward are wasted effort until 1 and 2 pass.

### 0. Self-test

```bash
python3 selftest.py
```

Generates synthetic `.data` files in the real 57-byte format with deliberately
planted properties, then asserts every stage recovers them. Includes a check
that the pipeline reports **no** edge on a driftless random walk, which is the
guard against the baseline matching being broken.

### 1. Validate the aggressor classification — blocking

```bash
python3 tick_data.py /path/to/data --check-format
python3 validate_direction.py /path/to/data --max-sessions 200
```

Every delta, CVD, imbalance and absorption feature in this project assumes
each print can be labelled buyer- or seller-initiated. The existing decoder
infers it by comparing price to `[level_low, level_high]`, which is the
standard quote rule and correct *if* those bounds are the bid and ask. The
decoder's own notes record level widths of 1.0 and 2.0 points, but ES ticks in
0.25 and normally quotes a 1-tick spread — so the "level" is more likely a
footprint bucket, in which case the rule is close to noise.

The primary test is **contemporaneous**: `corr(bar delta, bar return)`.
Correctly labelled aggression is mechanically related to same-bar price
movement, and on 1-minute ES should land around 0.6–0.85. The predictive test
(delta at `t` versus the return of `t+1`) is reported but is *not* a validity
check — near zero there is what an efficient market looks like.

If a rule other than `level` wins, pass `--direction-method` to everything
downstream and regenerate any features already built.

### 2. Establish a real out-of-sample baseline

Re-run the existing backtester against days excluded from training:

```bash
cd ../es_training
python3 backtester.py ./data/models --data /path/to/heldout_data
```

The `--data` flag is what forces genuinely unseen bars.

### 3. Build bars and level features

```bash
python3 level_ledger.py /path/to/data \
    --direction-method <winner from step 1> \
    --out retest.parquet
```

Add `--volume-bars 2500` to use information-driven bars instead of 1-minute
bars. A 1-minute bar at 09:35 and one at 12:30 carry very different amounts of
information, which makes every derived feature heteroskedastic; bars closed
every N contracts are far more stationary. `selftest.py` verifies the volume
dispersion actually drops.

### 4. Event studies — find the edge

```bash
python3 event_study.py --features retest.parquet --list-setups
python3 event_study.py --features retest.parquet --setup all --out study.csv
```

For each setup: collect every occurrence, measure forward MFE/MAE/return in
ATR units at several horizons, and compare against a baseline post-stratified
by time-of-day and volatility percentile then re-weighted to the bins the
events actually landed in. Confidence intervals come from resampling whole
sessions.

Read the output this way:

- **No significant horizon** — the signal carries no information. No exit rule
  or regime filter will rescue it. Discard it.
- **Significant** — the horizon tells you the holding period the signal
  supports, and the MFE-to-MAE ratio tells you where the target and stop
  belong. That measured ratio replaces a guessed 1.5R.

`any_level_test_short` is a deliberately weak control setup with no flow
conditions. A real setup that does not beat it has conditions that are doing
nothing.

### 5. Strategy selection via meta-labeling

```bash
python3 purged_cv.py --features retest.parquet --setup all --n-trials 40
```

Do not try to pick a strategy in advance. Let every strategy signal, then train
one secondary model per strategy predicting `P(this signal wins | context)`. At
decision time, run them all and take whichever clears its threshold. Selection
becomes a calibrated number rather than a hand-built regime taxonomy, and the
secondary problem is much easier than direction prediction: binary instead of
three-class, on a sample the primary rule has already narrowed.

This is what `es_training`'s `quality_model` was reaching for, but its label

```python
quality_label = (long_result == 1) | (short_result == 1)
```

asks "did *either* side win at this bar?" — a property of the bar, independent
of what any strategy signalled. `make_meta_labels()` conditions on the signal's
own direction, which is the entire point.

Validation uses purged walk-forward with an embargo, and the reported
precision keeps every signalled bar in the denominator. `--n-trials` feeds the
multiple-testing check: the maximum of *n* noise draws is about
`sqrt(2*ln(n))`, and a result that does not clear that is indistinguishable
from the best of a pile of noise. For reference, the 3-condition pattern miner
in `train_trading_model.py` evaluates thousands of combinations.

## Modules

| File | Purpose |
|---|---|
| `tick_data.py` | Vectorized 57-byte record decoder, session/RTH logic, three aggressor rules, format diagnostics |
| `bars.py` | Time and volume bars, long-form footprint, value area, ATR/regime columns |
| `validate_direction.py` | Step 1. Contemporaneous, tick-level and predictive tests per aggressor rule |
| `level_ledger.py` | Per-price-level visit ledger and first-test vs retest features |
| `event_study.py` | CUSUM sampling, setup registry, matched-baseline forward distributions, session bootstrap |
| `purged_cv.py` | Purged K-fold and walk-forward, conditional meta-labels, effective sample size, multiple-testing check |
| `selftest.py` | Synthetic ground-truth verification of every stage |

## Design notes

**Memory.** 3–5 years of tick data does not fit in RAM. `iter_sessions()`
yields one session at a time and callers reduce to bars before moving on, so
peak usage is one session of ticks plus the accumulated bar table.

**Speed.** Decoding uses a packed numpy structured dtype via `np.frombuffer`
rather than a per-record `struct.unpack_from` loop, which is roughly two orders
of magnitude faster and matters at this scale.

**No lookahead.** The level ledger is a single forward pass; at bar `t` it is
updated with bar `t`'s own footprint (known at the close, when a decision would
be taken) and then emits features. A "prior visit" is by construction one that
ended before the current visit began. The large-print threshold is an expanding
quantile over completed sessions only. `atr_percentile` ranks against prior
sessions only. Forward outcome windows return NaN when incomplete rather than
truncating, and never cross a session boundary.

## Requirements

`numpy`, `pandas`, `pyarrow` (for parquet). `scikit-learn` only for the
meta-model in `purged_cv.py`, which degrades gracefully without it. No scipy —
the correlation and bootstrap statistics are implemented directly.
