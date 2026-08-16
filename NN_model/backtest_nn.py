#!/usr/bin/env python3
"""
Backtest saved ES Mini NN models on new data.

Loads saved model checkpoints, runs the feature + label pipeline on new data,
and reports classification metrics (macro-F1, per-class precision/recall,
confusion matrix) for all 3 heads.

Usage:
    python backtest_nn.py /path/to/new_data/
    python backtest_nn.py /path/to/new_data/ --model-dir ./models
    python backtest_nn.py /path/to/new_data/ --head A
    python backtest_nn.py /path/to/new_data/ --device xpu
"""

from __future__ import annotations

import os
import sys
import argparse
import time

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from nn_feature_pipeline import load_data, compute_nn_features, CHANNEL_NAMES, SEQ_LEN
from nn_labels import compute_all_labels, get_valid_mask, H
from nn_dataset import extract_all_sequences
from nn_model import ESMiniModel, HEAD_CLASSES


# ============================================================
# Device
# ============================================================

def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ============================================================
# Model Loading
# ============================================================

def load_model(model_path: str, device: torch.device) -> tuple[ESMiniModel, dict]:
    """Load a saved model checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    head = checkpoint['head']
    n_classes = checkpoint['n_classes']
    config = checkpoint['config']

    model = ESMiniModel(
        n_classes=n_classes,
        hidden_dim=config['hidden_dim'],
        n_blocks=config['n_blocks'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict(model: ESMiniModel, sequences: np.ndarray,
            device: torch.device, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on sequences.
    Returns (predictions, probabilities) arrays.
    """
    n = len(sequences)
    all_preds = []
    all_probs = []

    for i in range(0, n, batch_size):
        batch = torch.from_numpy(sequences[i:i + batch_size]).to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    if device.type == 'xpu':
        torch.xpu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

    return np.concatenate(all_preds), np.concatenate(all_probs)


# ============================================================
# Metrics Reporting
# ============================================================

HEAD_CLASS_NAMES = {
    'A': ['strong_long_opp', 'strong_short_opp', 'no_edge'],
    'B': ['expansion', 'normal', 'contraction'],
    'C': ['continuation', 'retracement', 'reversal', 'chop'],
}


def print_metrics(head: str, y_true: np.ndarray, y_pred: np.ndarray,
                  y_probs: np.ndarray) -> dict:
    """Print and return classification metrics for a head."""
    n_classes = HEAD_CLASSES[head]
    class_names = HEAD_CLASS_NAMES[head]

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None,
                                     zero_division=0, labels=list(range(n_classes)))
    per_class_rec = recall_score(y_true, y_pred, average=None,
                                 zero_division=0, labels=list(range(n_classes)))
    per_class_f1 = f1_score(y_true, y_pred, average=None,
                            zero_division=0, labels=list(range(n_classes)))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    print(f"\n  {'='*60}")
    print(f"  HEAD {head} — BACKTEST RESULTS")
    print(f"  {'='*60}")
    print(f"  Samples: {len(y_true):,}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<20} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>10}")
    print(f"  {'-'*56}")
    for c in range(n_classes):
        support = (y_true == c).sum()
        print(f"  {class_names[c]:<20} {per_class_prec[c]:>8.4f} "
              f"{per_class_rec[c]:>8.4f} {per_class_f1[c]:>8.4f} {support:>10,}")

    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    header = "  " + " " * 20 + "".join(f"{name[:8]:>10}" for name in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join(f"{val:>10,}" for val in row)
        print(f"  {class_names[i]:<20}{row_str}")

    # Confidence analysis for minority classes
    print(f"\n  Prediction confidence (mean probability for predicted class):")
    for c in range(n_classes):
        mask = y_pred == c
        if mask.sum() > 0:
            mean_conf = y_probs[mask, c].mean()
            correct = (y_true[mask] == c).mean()
            print(f"    {class_names[c]:<20}: conf={mean_conf:.3f}, "
                  f"accuracy={correct:.3f} ({mask.sum():,} predictions)")

    return {
        'head': head,
        'n_samples': len(y_true),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class_precision': per_class_prec.tolist(),
        'per_class_recall': per_class_rec.tolist(),
        'per_class_f1': per_class_f1.tolist(),
        'confusion_matrix': cm.tolist(),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Backtest ES Mini NN models on new data')
    parser.add_argument('data_dir', help='Directory containing .data files for backtesting')
    parser.add_argument('--model-dir', default=None,
                        help='Directory with saved model .pt files (default: ./models)')
    parser.add_argument('--head', default='all', choices=['A', 'B', 'C', 'all'],
                        help='Which head to backtest (default: all)')
    parser.add_argument('--device', default=None, help='Device: xpu, cuda, cpu (default: auto)')
    parser.add_argument('--workers', type=int, default=0, help='Pipeline workers (0=auto)')
    parser.add_argument('--batch-size', type=int, default=512, help='Inference batch size')
    parser.add_argument('--threshold-a', type=float, default=1.5,
                        help='ATR threshold for Group A labels')

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: {args.data_dir} is not a directory", file=sys.stderr)
        return 1

    model_dir = args.model_dir or os.path.join(_SCRIPT_DIR, 'models')
    if not os.path.isdir(model_dir):
        print(f"Error: model directory not found: {model_dir}", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    print(f"Device: {device}")

    heads_to_test = ['A', 'B', 'C'] if args.head == 'all' else [args.head.upper()]

    # Check which models exist
    available_models = {}
    for head in heads_to_test:
        model_path = os.path.join(model_dir, f'model_head_{head}.pt')
        if os.path.exists(model_path):
            available_models[head] = model_path
        else:
            print(f"  Warning: model not found for head {head}: {model_path}")

    if not available_models:
        print("Error: no model files found to backtest", file=sys.stderr)
        return 1

    # ---- Data Pipeline ----
    print(f"\n{'='*70}")
    print(f"  DATA PIPELINE (backtest)")
    print(f"{'='*70}")

    t0 = time.time()
    print("Loading and decoding data files...")
    bars, profiles = load_data(args.data_dir, workers=args.workers)

    print("\nComputing NN features (26 channels)...")
    bars = compute_nn_features(bars, profiles, n_workers=args.workers)

    print("\nComputing labels (3 groups)...")
    bars = compute_all_labels(bars, n_workers=args.workers, threshold_a=args.threshold_a)

    print("\nApplying exclusion rules...")
    valid_mask = get_valid_mask(bars)
    n_valid = valid_mask.sum()
    print(f"  Valid decision points: {n_valid:,} / {len(bars):,} ({100*n_valid/len(bars):.1f}%)")
    print(f"  Pipeline elapsed: {time.time()-t0:.1f}s")

    # Get feature matrix and valid indices
    feature_matrix = bars[CHANNEL_NAMES].values.astype(np.float32)

    # NaN handling
    for col in range(feature_matrix.shape[1]):
        col_data = feature_matrix[:, col]
        mask = np.isnan(col_data)
        if mask.any():
            indices = np.where(~mask, np.arange(len(col_data)), 0)
            np.maximum.accumulate(indices, out=indices)
            col_data[:] = col_data[indices]
            still_nan = np.isnan(col_data)
            if still_nan.any():
                first_valid = np.where(~still_nan)[0]
                if len(first_valid) > 0:
                    col_data[still_nan] = col_data[first_valid[0]]
                else:
                    col_data[still_nan] = 0.0

    valid_indices = np.where(valid_mask)[0]
    valid_indices = valid_indices[valid_indices >= SEQ_LEN - 1]

    print(f"\nExtracting {len(valid_indices):,} sequences...")
    t_seq = time.time()
    sequences = extract_all_sequences(feature_matrix, valid_indices,
                                      n_workers=args.workers)
    print(f"  Sequences extracted: {sequences.shape} [{time.time()-t_seq:.1f}s]")

    # ---- Run Backtest Per Head ----
    all_results = {}

    for head, model_path in available_models.items():
        print(f"\nLoading model for head {head}...")
        model, checkpoint = load_model(model_path, device)
        train_f1 = checkpoint.get('best_macro_f1', 'N/A')
        print(f"  Model trained macro-F1: {train_f1}")

        # Get labels
        label_col = f'label_{head.lower()}'
        labels = bars[label_col].values[valid_indices].astype(np.int64)

        # Predict
        t_pred = time.time()
        preds, probs = predict(model, sequences, device, batch_size=args.batch_size)
        print(f"  Inference: {len(sequences):,} samples in {time.time()-t_pred:.1f}s")

        # Metrics
        result = print_metrics(head, labels, preds, probs)
        all_results[head] = result

    # ---- Final Summary ----
    print(f"\n{'='*70}")
    print(f"  BACKTEST SUMMARY")
    print(f"{'='*70}")
    print(f"  Data: {args.data_dir}")
    print(f"  Bars: {len(bars):,} | Valid samples: {len(valid_indices):,}")
    print(f"  Device: {device}")
    print(f"\n  {'Head':<8} {'Macro-F1':>10} {'Weighted-F1':>12} {'Samples':>10}")
    print(f"  {'-'*42}")
    for head, result in all_results.items():
        print(f"  {head:<8} {result['macro_f1']:>10.4f} "
              f"{result['weighted_f1']:>12.4f} {result['n_samples']:>10,}")

    total_elapsed = time.time() - t0
    print(f"\n  Total elapsed: {total_elapsed:.1f}s")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
