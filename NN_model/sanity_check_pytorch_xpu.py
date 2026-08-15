#!/usr/bin/env python3
"""
Sanity check: PyTorch + Intel GPU (XPU) for ES mini NN training.

Validates the stack against NN_model_features.md and NN_model_labels.md:
  - Input sequence shape: (batch, 128, 26)
  - Head A (MFE opportunity): 3 classes
  - Head B (vol regime):        3 classes
  - Head C (direction regime):    4 classes

Install (Intel GPU, example):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
  pip install intel-extension-for-pytorch

Usage:
  python sanity_check_pytorch_xpu.py
  python sanity_check_pytorch_xpu.py --device xpu
  python sanity_check_pytorch_xpu.py --device cpu   # skip GPU, still test model math
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

# --- Spec constants (from NN_model_features.md / NN_model_labels.md) ---
SEQ_LEN = 60
NUM_CHANNELS = 26
HEAD_A_CLASSES = 3  # strong_long_opp, strong_short_opp, no_edge
HEAD_B_CLASSES = 3  # expansion, normal, contraction
HEAD_C_CLASSES = 4  # continuation, retracement, reversal, chop

CHANNEL_NAMES = [
    "log_return", "bar_range_atr", "clv",
    "volume", "volume_delta", "volume_delta_pct", "buy_volume_pct", "cvd_slope_5",
    "close_vs_poc", "close_vs_vah", "close_vs_val",
    "close_in_value_area", "close_above_vah", "close_below_val",
    "vp60_vol_at_close_pct",
    "leg_size", "retrace_pct", "bars_since_swing_high", "bars_since_swing_low",
    "dist_to_swing_high_pct", "dist_to_swing_low_pct",
    "mins_from_rth_open",
    "atr_ratio", "range_vs_atr", "realized_vol_20", "atr_percentile_20d",
]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _ok(msg: str) -> CheckResult:
    return CheckResult(msg, True, "")


def _fail(msg: str, detail: str) -> CheckResult:
    return CheckResult(msg, False, detail)


def check_imports() -> tuple[list[CheckResult], object, object | None]:
    results: list[CheckResult] = []
    ipex = None

    try:
        import torch
    except ImportError as exc:
        results.append(_fail("import torch", str(exc)))
        return results, None, None
    results.append(_ok(f"import torch ({torch.__version__})"))

    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        results.append(_ok(f"import intel_extension_for_pytorch ({ipex.__version__})"))
    except ImportError:
        results.append(
            _fail(
                "import intel_extension_for_pytorch",
                "Not installed. Install with: pip install intel-extension-for-pytorch",
            )
        )

    return results, torch, ipex


def resolve_device(torch, requested: str | None) -> tuple[object, list[CheckResult]]:
    results: list[CheckResult] = []

    if requested:
        device = torch.device(requested)
        if requested == "xpu" and not torch.xpu.is_available():
            results.append(_fail("xpu available", "torch.xpu.is_available() is False"))
        elif requested == "cuda" and not torch.cuda.is_available():
            results.append(_fail("cuda available", "torch.cuda.is_available() is False"))
        else:
            results.append(_ok(f"device={requested} (user override)"))
        return device, results

    if torch.xpu.is_available():
        device = torch.device("xpu")
        results.append(_ok("Intel XPU detected"))
        try:
            name = torch.xpu.get_device_name(0)
            results.append(_ok(f"XPU device: {name}"))
        except Exception as exc:  # pragma: no cover
            results.append(_ok(f"XPU device name unavailable ({exc})"))
        return device, results

    if torch.cuda.is_available():
        results.append(
            _fail(
                "Intel XPU",
                f"XPU not found; CUDA available ({torch.cuda.get_device_name(0)}). "
                "Use --device cuda to test CUDA, or install Intel GPU drivers + XPU PyTorch.",
            )
        )
        return torch.device("cpu"), results

    results.append(
        _fail(
            "Intel XPU",
            "No XPU or CUDA device. Falling back to CPU for model math check only.",
        )
    )
    return torch.device("cpu"), results


def build_sanity_model(torch):
    """Minimal sequence encoder + three heads matching the label spec."""
    import torch.nn as nn

    class SanityModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(NUM_CHANNELS, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head_a = nn.Linear(64, HEAD_A_CLASSES)
            self.head_b = nn.Linear(64, HEAD_B_CLASSES)
            self.head_c = nn.Linear(64, HEAD_C_CLASSES)

        def forward(self, x):
            # x: (B, L, C) -> conv expects (B, C, L)
            h = self.encoder(x.transpose(1, 2)).squeeze(-1)
            return self.head_a(h), self.head_b(h), self.head_c(h)

    return SanityModel()


def synthetic_batch(torch, device, batch_size: int):
    """Random tensor with plausible value ranges for core channels.

    Build on CPU then transfer to device. Intel XPU (Level Zero) can reject
    in-place slice writes and bool-mask ops directly on the device.
    """
    x = torch.randn(batch_size, SEQ_LEN, NUM_CHANNELS)

    # clv [0, 1]
    x[:, :, 2] = torch.rand(batch_size, SEQ_LEN)
    # buy_volume_pct [0, 1]
    x[:, :, 6] = torch.rand(batch_size, SEQ_LEN)
    # volume_delta_pct [-1, 1]
    x[:, :, 5] = torch.rand(batch_size, SEQ_LEN) * 2 - 1
    # binary flags — use randint, not (rand > 0.5).float() (XPU-unfriendly)
    for idx in (11, 12, 13):
        x[:, :, idx] = torch.randint(0, 2, (batch_size, SEQ_LEN), dtype=torch.float32)
    # vp60_vol_at_close_pct [0, 100]
    x[:, :, 14] = torch.rand(batch_size, SEQ_LEN) * 100
    # atr_percentile_20d [0, 1]
    x[:, :, 25] = torch.rand(batch_size, SEQ_LEN)

    y_a = torch.randint(0, HEAD_A_CLASSES, (batch_size,))
    y_b = torch.randint(0, HEAD_B_CLASSES, (batch_size,))
    y_c = torch.randint(0, HEAD_C_CLASSES, (batch_size,))

    return (
        x.to(device),
        y_a.to(device),
        y_b.to(device),
        y_c.to(device),
    )


def run_training_step(torch, ipex, device, batch_size: int, steps: int) -> list[CheckResult]:
    results: list[CheckResult] = []
    import torch.nn as nn

    if len(CHANNEL_NAMES) != NUM_CHANNELS:
        results.append(_fail("channel spec", f"Expected {NUM_CHANNELS} channels, got {len(CHANNEL_NAMES)}"))
        return results
    results.append(_ok(f"channel count = {NUM_CHANNELS}"))

    model = build_sanity_model(torch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    if ipex is not None and device.type == "xpu":
        try:
            model, optimizer = ipex.optimize(model, optimizer=optimizer)
            results.append(_ok("ipex.optimize(model, optimizer)"))
        except Exception as exc:
            results.append(_fail("ipex.optimize", str(exc)))
            return results

    x, y_a, y_b, y_c = synthetic_batch(torch, device, batch_size)

    t0 = time.perf_counter()
    last_loss = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits_a, logits_b, logits_c = model(x)
        loss = (
            1.0 * loss_fn(logits_a, y_a)
            + 0.3 * loss_fn(logits_b, y_b)
            + 0.3 * loss_fn(logits_c, y_c)
        )
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0

    for name, logits, n_cls in (
        ("head_a", logits_a, HEAD_A_CLASSES),
        ("head_b", logits_b, HEAD_B_CLASSES),
        ("head_c", logits_c, HEAD_C_CLASSES),
    ):
        if logits.shape != (batch_size, n_cls):
            results.append(_fail(f"{name} shape", f"got {tuple(logits.shape)}, expected ({batch_size}, {n_cls})"))
            return results
    results.append(_ok(f"output shapes: A={tuple(logits_a.shape)}, B={tuple(logits_b.shape)}, C={tuple(logits_c.shape)}"))

    if last_loss is None or not (last_loss == last_loss):  # NaN check
        results.append(_fail("training loss", f"invalid loss={last_loss}"))
        return results

    results.append(_ok(f"forward + backward + {steps} step(s), loss={last_loss:.4f}, time={elapsed:.3f}s"))
    return results


def print_report(all_results: list[CheckResult]) -> int:
    width = max(len(r.name) for r in all_results) if all_results else 20
    passed = 0
    failed = 0

    print()
    print("=" * 60)
    print("ES Mini NN — PyTorch / Intel XPU sanity check")
    print("=" * 60)
    for r in all_results:
        status = "PASS" if r.passed else "FAIL"
        line = f"[{status}] {r.name:<{width}}"
        if r.passed and r.detail:
            line += f"  {r.detail}"
        elif not r.passed:
            line += f"  {r.detail}"
        else:
            # extract version from name if embedded
            pass
        print(line)
        if r.passed:
            passed += 1
        else:
            failed += 1

    print("-" * 60)
    print(f"Summary: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("All checks passed. Stack is ready for ES mini NN training.")
        return 0

    print("Some checks failed. Fix the items above before training.")
    return 1


def parse_args():
    p = argparse.ArgumentParser(description="PyTorch + Intel XPU sanity check for ES mini NN")
    p.add_argument("--device", choices=("xpu", "cuda", "cpu"), default=None,
                   help="Force device (default: auto-detect XPU)")
    p.add_argument("--batch-size", type=int, default=32, help="Synthetic batch size (default: 32)")
    p.add_argument("--steps", type=int, default=3, help="Training steps to run (default: 3)")
    p.add_argument("--require-xpu", action="store_true",
                   help="Exit with error if Intel XPU is not available")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    all_results: list[CheckResult] = []

    import_results, torch, ipex = check_imports()
    all_results.extend(import_results)
    if torch is None:
        return print_report(all_results)

    device, device_results = resolve_device(torch, args.device)
    all_results.extend(device_results)

    if args.require_xpu and device.type != "xpu":
        all_results.append(_fail("--require-xpu", f"Expected XPU but got device={device.type}"))
        return print_report(all_results)

    all_results.append(_ok(f"input spec: (batch, {SEQ_LEN}, {NUM_CHANNELS})"))
    train_results = run_training_step(torch, ipex, device, args.batch_size, args.steps)
    all_results.extend(train_results)

    return print_report(all_results)


if __name__ == "__main__":
    sys.exit(main())
