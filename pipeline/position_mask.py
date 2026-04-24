"""
Shared position-masking helper.

Every downstream analysis step (train, evaluate, intervention, composition,
causal, promote_loop) flattens `activations` / `annotations` / `tokens`
from (N, T, ...) to ((N * T'), ...) and reasons about per-position
statistics. Position 0 in a transformer has degenerate attention (attends
only to itself), no prior context, anomalous residual-stream magnitude,
and acts as an attention sink — including it in supervised SAE analysis
corrupts target_dirs (sequence-level features collapse to the shared
"position-0 vs rest" axis), dominates reconstruction, and wastes
promote-loop rounds on BOS detectors.

This helper applies `cfg.mask_first_n_positions` uniformly at load time
so every step sees the same effective sequence length (T − n). Cached
tensors on disk don't need to be re-extracted.
"""

from __future__ import annotations

from typing import Sequence, Union
import torch

from .config import Config


def mask_leading(
    *tensors: torch.Tensor, cfg: Config,
) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Return the input tensors with the first `cfg.mask_first_n_positions`
    positions (second dim) sliced off.

    Accepts any number of tensors; each must have at least 2 dims. Returns
    a single tensor if one was passed, a tuple otherwise.

    Tensors whose second dim is already shorter than the mask (edge case:
    caller already sliced, or T ≤ mask) are passed through unchanged with
    a warning.
    """
    n = max(0, int(getattr(cfg, "mask_first_n_positions", 0)))
    out = []
    for t in tensors:
        if n == 0:
            out.append(t)
            continue
        if t.dim() < 2:
            out.append(t)
            continue
        if t.shape[1] <= n:
            print(
                f"  [mask_leading] WARNING: tensor shape {tuple(t.shape)} "
                f"has second dim <= mask ({n}); passing through."
            )
            out.append(t)
            continue
        out.append(t[:, n:].contiguous())
    return out[0] if len(out) == 1 else tuple(out)
