#!/usr/bin/env python3
"""
PyTorch Dataset for ES Mini NN model.

Extracts (60, 26) sequences from the feature DataFrame, handles exclusions,
and provides labels for a selected head (A, B, or C).

Supports parallel sequence extraction via multiprocessing.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset

from nn_feature_pipeline import CHANNEL_NAMES, SEQ_LEN


# ============================================================
# Sequence Extraction (parallelized)
# ============================================================

def _extract_sequences_chunk(args: tuple) -> tuple:
    """Extract sequences for a chunk of valid indices."""
    valid_indices_chunk, feature_matrix = args
    n_seq = len(valid_indices_chunk)
    sequences = np.empty((n_seq, SEQ_LEN, len(CHANNEL_NAMES)), dtype=np.float32)

    for local_i, t in enumerate(valid_indices_chunk):
        seq = feature_matrix[t - SEQ_LEN + 1:t + 1]
        sequences[local_i] = seq

    return sequences


def extract_all_sequences(feature_matrix: np.ndarray,
                          valid_indices: np.ndarray,
                          n_workers: int = 0) -> np.ndarray:
    """
    Extract all (60, 26) sequences for valid decision points.
    Uses multiprocessing for parallel extraction.

    Args:
        feature_matrix: (n_bars, 26) float32 array of channel values
        valid_indices: indices where valid decision points occur (must be >= SEQ_LEN-1)
        n_workers: number of parallel workers (0=auto)

    Returns:
        (n_valid, 60, 26) float32 array
    """
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 8)

    # Filter indices that have enough history
    valid_indices = valid_indices[valid_indices >= SEQ_LEN - 1]
    n = len(valid_indices)

    if n == 0:
        return np.empty((0, SEQ_LEN, len(CHANNEL_NAMES)), dtype=np.float32)

    chunk_size = max(1, n // n_workers)
    chunks = []
    for i in range(0, n, chunk_size):
        idx_chunk = valid_indices[i:i + chunk_size]
        chunks.append((idx_chunk, feature_matrix))

    if n_workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_extract_sequences_chunk, chunks))
    else:
        results = [_extract_sequences_chunk(c) for c in chunks]

    return np.concatenate(results, axis=0)


# ============================================================
# PyTorch Dataset
# ============================================================

class ESMiniDataset(Dataset):
    """
    PyTorch Dataset for ES Mini NN training.

    Each sample is a tuple:
        (sequence, label)
    where sequence is shape (60, 26) float32 and label is int64.
    """

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Args:
            sequences: (N, 60, 26) float32 array
            labels: (N,) int64 array
        """
        assert len(sequences) == len(labels), \
            f"sequences ({len(sequences)}) and labels ({len(labels)}) must have same length"
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = torch.from_numpy(self.sequences[idx])  # (60, 26) float32
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label


# ============================================================
# Dataset Builder
# ============================================================

def build_dataset(bars_df, valid_mask: np.ndarray, head: str,
                  n_workers: int = 0) -> ESMiniDataset:
    """
    Build a dataset for a specific label head.

    Args:
        bars_df: DataFrame with all 26 channel columns + label columns
        valid_mask: boolean array marking valid decision points
        head: 'A', 'B', or 'C'
        n_workers: parallel workers for sequence extraction

    Returns:
        ESMiniDataset ready for DataLoader
    """
    # Get feature matrix
    feature_matrix = bars_df[CHANNEL_NAMES].values.astype(np.float32)

    # Handle NaN: forward-fill then backward-fill column-wise
    for col in range(feature_matrix.shape[1]):
        col_data = feature_matrix[:, col]
        mask = np.isnan(col_data)
        if mask.any():
            # Forward fill
            indices = np.where(~mask, np.arange(len(col_data)), 0)
            np.maximum.accumulate(indices, out=indices)
            col_data[:] = col_data[indices]
            # Backward fill remaining leading NaNs
            still_nan = np.isnan(col_data)
            if still_nan.any():
                first_valid = np.where(~still_nan)[0]
                if len(first_valid) > 0:
                    col_data[still_nan] = col_data[first_valid[0]]
                else:
                    col_data[still_nan] = 0.0

    # Get valid indices
    valid_indices = np.where(valid_mask)[0]
    # Must have enough history for sequence
    valid_indices = valid_indices[valid_indices >= SEQ_LEN - 1]

    print(f"  Extracting {len(valid_indices):,} sequences ({n_workers} workers)...")
    sequences = extract_all_sequences(feature_matrix, valid_indices, n_workers=n_workers)

    # Get labels for the selected head
    label_col = f'label_{head.lower()}'
    all_labels = bars_df[label_col].values
    labels = all_labels[valid_indices].astype(np.int64)

    print(f"  Dataset: {len(sequences):,} samples, shape {sequences.shape}")
    return ESMiniDataset(sequences, labels)


def build_train_val_datasets(bars_df, valid_mask: np.ndarray, head: str,
                             train_indices: np.ndarray, val_indices: np.ndarray,
                             n_workers: int = 0) -> tuple[ESMiniDataset, ESMiniDataset]:
    """
    Build train and validation datasets for a walk-forward fold.

    Args:
        bars_df: full DataFrame
        valid_mask: boolean mask for valid bars
        head: 'A', 'B', or 'C'
        train_indices: bar indices for training
        val_indices: bar indices for validation
        n_workers: parallel workers

    Returns:
        (train_dataset, val_dataset)
    """
    feature_matrix = bars_df[CHANNEL_NAMES].values.astype(np.float32)

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

    label_col = f'label_{head.lower()}'
    all_labels = bars_df[label_col].values

    # Filter to valid + has enough history
    train_valid = train_indices[
        valid_mask[train_indices] & (train_indices >= SEQ_LEN - 1)
    ]
    val_valid = val_indices[
        valid_mask[val_indices] & (val_indices >= SEQ_LEN - 1)
    ]

    # Extract sequences
    train_seqs = extract_all_sequences(feature_matrix, train_valid, n_workers=n_workers)
    val_seqs = extract_all_sequences(feature_matrix, val_valid, n_workers=n_workers)

    train_labels = all_labels[train_valid].astype(np.int64)
    val_labels = all_labels[val_valid].astype(np.int64)

    return (
        ESMiniDataset(train_seqs, train_labels),
        ESMiniDataset(val_seqs, val_labels),
    )
