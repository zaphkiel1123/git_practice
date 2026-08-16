#!/usr/bin/env python3
"""
Training script for ES Mini NN model (Phase 1).

Trains 3 independent TCN models (one per label group) with walk-forward
validation. Supports auto-detection of Intel XPU, CUDA, or CPU.

Usage:
    python train_nn.py /path/to/data/ --head A
    python train_nn.py /path/to/data/ --head B
    python train_nn.py /path/to/data/ --head C
    python train_nn.py /path/to/data/ --head all
    python train_nn.py /path/to/data/ --head A --device cpu --epochs 50
"""

from __future__ import annotations

import os
import sys
import argparse
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from nn_feature_pipeline import load_data, compute_nn_features, CHANNEL_NAMES, SEQ_LEN
from nn_labels import compute_all_labels, get_valid_mask, H
from nn_dataset import build_train_val_datasets
from nn_model import create_model, count_parameters, HEAD_CLASSES


# ============================================================
# Device Resolution
# ============================================================

def resolve_device(requested: str | None = None) -> torch.device:
    """Auto-detect best available device: XPU > CUDA > CPU."""
    if requested:
        device = torch.device(requested)
        _print_device_info(device)
        return device

    # Try Intel XPU (works with PyTorch +xpu builds, no IPEX required)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        _print_device_info(device)
        return device

    # Try CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
        _print_device_info(device)
        return device

    # Fallback to CPU
    device = torch.device('cpu')
    _print_device_info(device)
    return device


def _print_device_info(device: torch.device):
    """Print detailed device information."""
    print(f"\n  {'='*50}")
    print(f"  DEVICE INFO")
    print(f"  {'='*50}")
    if device.type == 'xpu':
        try:
            name = torch.xpu.get_device_name(0)
        except Exception:
            name = "Intel XPU (name unavailable)"
        try:
            props = torch.xpu.get_device_properties(0)
            mem = getattr(props, 'total_memory', None)
            if mem:
                print(f"  Using: Intel XPU (GPU)")
                print(f"  Device: {name}")
                print(f"  Memory: {mem / 1024**3:.1f} GB")
            else:
                print(f"  Using: Intel XPU (GPU)")
                print(f"  Device: {name}")
        except Exception:
            print(f"  Using: Intel XPU (GPU)")
            print(f"  Device: {name}")
        print(f"  XPU device count: {torch.xpu.device_count()}")
    elif device.type == 'cuda':
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem = props.total_mem
        print(f"  Using: NVIDIA CUDA (GPU)")
        print(f"  Device: {name}")
        print(f"  Memory: {mem / 1024**3:.1f} GB")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        import multiprocessing
        n_cores = multiprocessing.cpu_count()
        print(f"  Using: CPU")
        print(f"  Cores available: {n_cores}")
        print(f"  No GPU detected")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  {'='*50}\n")


# ============================================================
# Walk-Forward Splits
# ============================================================

def walk_forward_splits(n: int, n_folds: int = 5,
                        min_train_ratio: float = 0.5,
                        embargo: int = H) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window walk-forward splits with embargo gap.
    Returns list of (train_indices, val_indices) tuples.
    """
    test_size = int(n * (1 - min_train_ratio) / n_folds)
    splits = []

    for i in range(n_folds):
        test_start = int(n * min_train_ratio) + i * test_size
        test_end = min(test_start + test_size, n)

        if test_end > n:
            break

        # Train up to embargo gap before test
        train_end = test_start - embargo
        if train_end < 100:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        if len(train_idx) > 100 and len(test_idx) > 50:
            splits.append((train_idx, test_idx))

    return splits


# ============================================================
# Class Weight Computation
# ============================================================

def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# Training Loop
# ============================================================

def _optimize_for_xpu(model: nn.Module, optimizer: torch.optim.Optimizer,
                      device: torch.device) -> tuple[nn.Module, torch.optim.Optimizer]:
    """Apply Intel Extension for PyTorch optimization if available (optional)."""
    if device.type != 'xpu':
        return model, optimizer
    try:
        import intel_extension_for_pytorch as ipex
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
        print("  Applied ipex.optimize() for XPU acceleration")
    except ImportError:
        pass  # IPEX not required with PyTorch +xpu builds
    except Exception as e:
        print(f"  Note: ipex.optimize() skipped ({e})")
    return model, optimizer


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device) -> float:
    """Train for one epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    # Synchronize XPU to get accurate timing
    if device.type == 'xpu':
        torch.xpu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device,
             n_classes: int) -> dict:
    """Evaluate model on a dataset. Returns metrics dict."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        logits = model(sequences)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    if device.type == 'xpu':
        torch.xpu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_prec = precision_score(all_labels, all_preds, average=None,
                                     zero_division=0, labels=list(range(n_classes)))
    per_class_rec = recall_score(all_labels, all_preds, average=None,
                                 zero_division=0, labels=list(range(n_classes)))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))

    return {
        'loss': total_loss / max(n_batches, 1),
        'macro_f1': float(macro_f1),
        'per_class_precision': per_class_prec.tolist(),
        'per_class_recall': per_class_rec.tolist(),
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'labels': all_labels,
    }


# ============================================================
# Single Head Training
# ============================================================

def train_head(head: str, bars: pd.DataFrame, valid_mask: np.ndarray,
               device: torch.device, config: dict) -> dict:
    """
    Train a single model for the specified label head using walk-forward validation.

    Returns dict with per-fold metrics and best model state.
    """
    head = head.upper()
    n_classes = HEAD_CLASSES[head]
    n = len(bars)

    print(f"\n{'='*70}")
    print(f"  TRAINING HEAD {head} ({n_classes} classes)")
    print(f"{'='*70}")

    splits = walk_forward_splits(n, n_folds=config['n_folds'], embargo=H)
    print(f"  Walk-forward splits: {len(splits)} folds")

    all_fold_metrics = []
    best_f1 = 0.0
    best_model_state = None

    n_workers_data = min(os.cpu_count() or 1, 4)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n  --- Fold {fold_idx + 1}/{len(splits)} ---")
        print(f"  Train: bars [0..{train_idx[-1]}] ({len(train_idx):,} bars)")
        print(f"  Val:   bars [{val_idx[0]}..{val_idx[-1]}] ({len(val_idx):,} bars)")

        train_ds, val_ds = build_train_val_datasets(
            bars, valid_mask, head, train_idx, val_idx, n_workers=n_workers_data
        )

        if len(train_ds) < 100 or len(val_ds) < 50:
            print(f"  Skipping fold: insufficient samples (train={len(train_ds)}, val={len(val_ds)})")
            continue

        print(f"  Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

        # Class weights from training labels
        class_weights = compute_class_weights(train_ds.labels, n_classes).to(device)

        # DataLoaders
        dl_workers = min(os.cpu_count() or 1, 4)
        pin = (device.type == 'cuda')  # pin_memory only benefits CUDA, not XPU
        train_loader = DataLoader(
            train_ds, batch_size=config['batch_size'], shuffle=True,
            num_workers=dl_workers, pin_memory=pin,
            persistent_workers=(dl_workers > 0),
        )
        val_loader = DataLoader(
            val_ds, batch_size=config['batch_size'], shuffle=False,
            num_workers=dl_workers, pin_memory=pin,
            persistent_workers=(dl_workers > 0),
        )

        # Model
        model = create_model(
            head,
            hidden_dim=config['hidden_dim'],
            n_blocks=config['n_blocks'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout'],
        ).to(device)

        if fold_idx == 0:
            print(f"  Model parameters: {count_parameters(model):,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                     weight_decay=config.get('weight_decay', 1e-5))

        # Apply Intel XPU optimization if on discrete GPU
        model, optimizer = _optimize_for_xpu(model, optimizer, device)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_fold_state = None

        t0 = time.time()
        for epoch in range(config['epochs']):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

            if (epoch + 1) % config['eval_every'] == 0 or epoch == config['epochs'] - 1:
                val_metrics = evaluate(model, val_loader, criterion, device, n_classes)
                val_loss = val_metrics['loss']
                val_f1 = val_metrics['macro_f1']

                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']

                print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, macro_f1={val_f1:.4f}, "
                      f"lr={current_lr:.2e}, elapsed={time.time()-t0:.1f}s")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_fold_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= config['patience']:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        elapsed = time.time() - t0
        print(f"  Fold training time: {elapsed:.1f}s")

        # Final evaluation with best model
        if best_fold_state:
            model.load_state_dict(best_fold_state)
        final_metrics = evaluate(model, val_loader, criterion, device, n_classes)

        fold_result = {
            'fold': fold_idx + 1,
            'train_samples': len(train_ds),
            'val_samples': len(val_ds),
            'best_val_loss': float(best_val_loss),
            'macro_f1': final_metrics['macro_f1'],
            'per_class_precision': final_metrics['per_class_precision'],
            'per_class_recall': final_metrics['per_class_recall'],
            'confusion_matrix': final_metrics['confusion_matrix'],
            'elapsed_s': elapsed,
        }
        all_fold_metrics.append(fold_result)

        print(f"  Fold {fold_idx+1} result: macro_f1={final_metrics['macro_f1']:.4f}")
        print(f"    Precision: {final_metrics['per_class_precision']}")
        print(f"    Recall:    {final_metrics['per_class_recall']}")
        print(f"    Confusion matrix:")
        for row in final_metrics['confusion_matrix']:
            print(f"      {row}")

        if final_metrics['macro_f1'] > best_f1:
            best_f1 = final_metrics['macro_f1']
            best_model_state = best_fold_state

    # Summary
    if all_fold_metrics:
        avg_f1 = np.mean([m['macro_f1'] for m in all_fold_metrics])
        print(f"\n  HEAD {head} SUMMARY:")
        print(f"    Average macro-F1 across folds: {avg_f1:.4f}")
        print(f"    Best single-fold macro-F1: {best_f1:.4f}")
    else:
        avg_f1 = 0.0
        print(f"\n  HEAD {head}: No valid folds completed")

    return {
        'head': head,
        'n_classes': n_classes,
        'folds': all_fold_metrics,
        'avg_macro_f1': float(avg_f1),
        'best_macro_f1': float(best_f1),
        'best_model_state': best_model_state,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train ES Mini NN model (Phase 1)')
    parser.add_argument('data_dir', help='Directory containing .data files')
    parser.add_argument('--head', default='all', choices=['A', 'B', 'C', 'all'],
                        help='Which label head to train (default: all)')
    parser.add_argument('--device', default=None, help='Device: xpu, cuda, cpu (default: auto)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Pipeline workers (0=auto)')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs per fold')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='TCN hidden dimension')
    parser.add_argument('--n-blocks', type=int, default=6, help='Number of TCN blocks')
    parser.add_argument('--kernel-size', type=int, default=3, help='Conv kernel size')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--n-folds', type=int, default=5, help='Walk-forward folds')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--eval-every', type=int, default=2, help='Evaluate every N epochs')
    parser.add_argument('--output-dir', default=None, help='Output directory for models')
    parser.add_argument('--threshold-a', type=float, default=1.5,
                        help='ATR threshold for Group A labels')

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: {args.data_dir} is not a directory", file=sys.stderr)
        return 1

    # Config
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'hidden_dim': args.hidden_dim,
        'n_blocks': args.n_blocks,
        'kernel_size': args.kernel_size,
        'dropout': args.dropout,
        'n_folds': args.n_folds,
        'patience': args.patience,
        'eval_every': args.eval_every,
        'weight_decay': 1e-5,
    }

    # Device
    device = resolve_device(args.device)
    print(f"Device: {device}")

    # Output directory
    output_dir = args.output_dir or os.path.join(_SCRIPT_DIR, 'models')
    os.makedirs(output_dir, exist_ok=True)

    # ---- Data Pipeline ----
    print(f"\n{'='*70}")
    print(f"  DATA PIPELINE")
    print(f"{'='*70}")

    t_pipeline = time.time()
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

    print(f"\n  Pipeline complete [{time.time()-t_pipeline:.1f}s]")

    # ---- Training ----
    t_training = time.time()
    heads_to_train = ['A', 'B', 'C'] if args.head == 'all' else [args.head.upper()]
    all_results = {}

    for head in heads_to_train:
        result = train_head(head, bars, valid_mask, device, config)
        all_results[head] = result

        # Save best model
        if result['best_model_state'] is not None:
            model_path = os.path.join(output_dir, f'model_head_{head}.pt')
            torch.save({
                'model_state_dict': result['best_model_state'],
                'head': head,
                'n_classes': result['n_classes'],
                'config': config,
                'best_macro_f1': result['best_macro_f1'],
            }, model_path)
            print(f"  Saved best model → {model_path}")

    training_elapsed = time.time() - t_training
    total_elapsed = time.time() - t_pipeline

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    for head, result in all_results.items():
        print(f"  Head {head}: avg_macro_f1={result['avg_macro_f1']:.4f}, "
              f"best={result['best_macro_f1']:.4f}")
    print(f"\n  Elapsed time:")
    print(f"    Data pipeline:  {time.time()-t_pipeline - training_elapsed:.1f}s")
    print(f"    Training:       {training_elapsed:.1f}s")
    print(f"    Total:          {total_elapsed:.1f}s")

    # Save metadata
    meta = {
        'heads': list(all_results.keys()),
        'config': config,
        'device': str(device),
        'data_dir': args.data_dir,
        'n_bars': len(bars),
        'n_valid': int(n_valid),
        'trained_at': datetime.now().isoformat(),
        'results': {
            head: {
                'avg_macro_f1': r['avg_macro_f1'],
                'best_macro_f1': r['best_macro_f1'],
                'folds': r['folds'],
            }
            for head, r in all_results.items()
        },
    }
    meta_path = os.path.join(output_dir, 'training_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Metadata → {meta_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
