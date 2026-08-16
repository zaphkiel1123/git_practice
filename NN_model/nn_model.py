#!/usr/bin/env python3
"""
TCN-based neural network model for ES Mini classification.

Architecture:
  - TCN encoder: 6 residual blocks with dilated causal convolutions
  - Dilations: [1, 2, 4, 8, 16, 32] → receptive field = 63 bars (covers 60-bar input)
  - Each block: Conv1d → BatchNorm → ReLU → Dropout → Conv1d → BatchNorm → residual
  - Global average pooling → classification head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_feature_pipeline import NUM_CHANNELS, SEQ_LEN


# ============================================================
# TCN Building Blocks
# ============================================================

class CausalConv1d(nn.Module):
    """Conv1d with causal (left) padding so output doesn't see future."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """
    Residual block for the TCN.
    Two causal convolutions with BatchNorm, ReLU, and dropout.
    1x1 conv for residual if channel mismatch.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + res)
        out = self.dropout(out)

        return out


# ============================================================
# TCN Encoder
# ============================================================

class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network encoder.

    Input: (batch, seq_len, n_channels) — transposed internally to (batch, n_channels, seq_len)
    Output: (batch, hidden_dim) — global representation of the sequence
    """

    def __init__(self, in_channels: int = NUM_CHANNELS, hidden_dim: int = 64,
                 n_blocks: int = 6, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)

        layers = []
        for i in range(n_blocks):
            dilation = 2 ** i
            ch_in = in_channels if i == 0 else hidden_dim
            layers.append(TCNBlock(ch_in, hidden_dim, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels) → (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = self.network(x)
        x = self.pool(x).squeeze(-1)  # (batch, hidden_dim)
        return x


# ============================================================
# Full Model
# ============================================================

class ESMiniModel(nn.Module):
    """
    ES Mini classification model: TCN encoder + single linear head.

    Args:
        n_classes: number of output classes (3 for heads A/B, 4 for head C)
        hidden_dim: encoder hidden dimension
        n_blocks: number of TCN residual blocks
        kernel_size: convolution kernel size
        dropout: dropout rate
    """

    def __init__(self, n_classes: int = 3, hidden_dim: int = 64,
                 n_blocks: int = 6, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.encoder = TCNEncoder(
            in_channels=NUM_CHANNELS,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 60, 26) input sequence

        Returns:
            logits: (batch, n_classes)
        """
        h = self.encoder(x)
        return self.head(h)


# ============================================================
# Model Factory
# ============================================================

HEAD_CLASSES = {
    'A': 3,  # strong_long_opp, strong_short_opp, no_edge
    'B': 3,  # expansion, normal, contraction
    'C': 4,  # continuation, retracement, reversal, chop
    'D': 3,  # long_2R_win, short_2R_win, no_trigger
}


def create_model(head: str, hidden_dim: int = 64, n_blocks: int = 6,
                 kernel_size: int = 3, dropout: float = 0.2) -> ESMiniModel:
    """Create a model for the specified label head."""
    head = head.upper()
    if head not in HEAD_CLASSES:
        raise ValueError(f"Unknown head '{head}'. Must be one of {list(HEAD_CLASSES.keys())}")
    return ESMiniModel(
        n_classes=HEAD_CLASSES[head],
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        kernel_size=kernel_size,
        dropout=dropout,
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Receptive Field Info
# ============================================================

def compute_receptive_field(n_blocks: int = 6, kernel_size: int = 3) -> int:
    """Compute the receptive field of the TCN in bars."""
    rf = 1
    for i in range(n_blocks):
        dilation = 2 ** i
        rf += 2 * (kernel_size - 1) * dilation
    return rf


if __name__ == '__main__':
    rf = compute_receptive_field()
    print(f"TCN receptive field: {rf} bars (input length: {SEQ_LEN})")

    for head in ['A', 'B', 'C']:
        model = create_model(head)
        n_params = count_parameters(model)
        print(f"Head {head}: {HEAD_CLASSES[head]} classes, {n_params:,} parameters")

        # Test forward pass
        x = torch.randn(4, SEQ_LEN, NUM_CHANNELS)
        logits = model(x)
        print(f"  Input: {x.shape} → Output: {logits.shape}")
