"""
Oracle-unsup readout (Type 2 appendix, v8.19.0).

For each Opus-designed feature `c` (from `opus_catalog.json`), find the
unsup SAE latent whose firing pattern best matches the Opus labels
`y_c` on a HELD-OUT split, then report F1 on a separate, untouched
TEST split. Two outputs per feature:

  • `oracle_f1_test`: F1 on the test split, using the unsup latent
    that maximized F1 on the val split. Honest oracle (val-select /
    test-report).
  • `realistic_f1_test`: not computed here yet — would be the F1 of
    the unsup latent whose Sonnet/Delphi description embedding-matches
    the Opus description. Plumbed in once Delphi descriptions are
    available + an embedding model is wired.

Why honest oracle matters: selecting the best latent and reporting F1
on the same slice over-fits to that slice. Maximum-of-N selection is
biased upward by O(sqrt(2 ln N) * sigma) per Bonferroni; with N=24576
unsup latents and feature-specific noise, that bias can be substantial.
The val-select / test-report split eliminates it.

Memory model:
  - Streams token batches through the SAE; per batch computes a
    (B*T, n_lat) bool firing tensor.
  - Per Opus feature `c` and per VAL batch: accumulates per-latent
    (tp, fp) counts vectorized across n_lat. Total fn = total positives
    in val_y_c.
  - Never materializes the full (T_total, n_lat) fires matrix. Peak
    GPU memory is one batch worth of activations + (B*T, n_lat) bool
    (~1 GB at B*T=4K, n_lat=24576).
  - Test pass: only forwards the SAE for the chosen oracle_latent per
    feature, so it's much cheaper.

Output: `pipeline_data/oracle_unsup.json`. Reuses Opus annotation
labels; no extra annotation cost.

Run with: python -m pipeline.run --step oracle-unsup
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import Config


def _load_sup_arm_leaves(cfg: Config) -> list[dict]:
    """Read the sup arm's feature_catalog.json (= Opus catalog under v8.19.2
    two-arm flow). All leaves are Opus features; annotations columns
    align 1:1 with leaves order."""
    cat_path = cfg.catalog_path
    if not cat_path.exists():
        raise FileNotFoundError(
            f"Need {cat_path}. Run --step opus-catalog + --step annotate "
            f"in this output_dir first."
        )
    catalog = json.loads(cat_path.read_text())
    if catalog.get("source") == "delphi":
        raise RuntimeError(
            f"{cat_path} is a Delphi-arm catalog, not Opus. The "
            f"oracle_unsup appendix only makes sense in the SUP arm "
            f"(it searches all unsup latents for the best match per "
            f"OPUS-designed feature). Re-run with the sup arm's "
            f"output_dir (typically pipeline_data/, not "
            f"pipeline_data_unsup/)."
        )
    return [f for f in catalog.get("features", []) if f.get("type") == "leaf"]


def _stream_firing_counts(
    cfg: Config,
    tokens: torch.Tensor,
    val_y: torch.Tensor,
    val_idx_to_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Streaming pass to compute per-latent (tp, fp) for every Opus feature.

    Args:
        cfg: pipeline config (for hook_point + SAE loading).
        tokens: (n_seqs, T) ALL tokens (val + test). We forward all
            positions but only score the val ones.
        val_y: (val_T_total, n_features) bool — Opus labels on the
            val flat-positions.
        val_idx_to_pos: (val_T_total,) long — flat index in (n_seqs * T)
            that this row of val_y corresponds to. Used to mask the
            streaming pass.

    Returns:
        tp:   (n_features, n_lat) long. Per-feature per-latent true positives on val.
        fp:   (n_features, n_lat) long. Per-feature per-latent false positives on val.
        fn:   (n_features,) long. Per-feature total positives - tp[:, k]
              gives per-latent FN; we return total positives only,
              caller derives FN.
    """
    from .inventory import load_sae, load_target_model
    sae, _ = load_sae(cfg)
    model = load_target_model(cfg)
    sae = sae.to(cfg.device)

    n_lat = (
        sae.W_enc.shape[1] if sae.W_enc is not None
        else sae.native_sae.cfg.d_sae
    )
    n_features = val_y.shape[1]
    n_seqs, T_per = tokens.shape

    # Build a dense bool mask: which flat positions are val positions?
    val_mask_flat = torch.zeros(n_seqs * T_per, dtype=torch.bool)
    val_mask_flat[val_idx_to_pos] = True

    # Build dense val_y_full: (n_seqs*T_per, n_features) with rows zero
    # for non-val positions. Only val rows ever multiply against fires
    # in the count accumulation. Storage cost: n_seqs*T*F bool ≈ 5K*128
    # *300 ≈ 192MB at our scale.
    val_y_full = torch.zeros((n_seqs * T_per, n_features), dtype=torch.bool)
    val_y_full[val_idx_to_pos] = val_y

    # Per-feature (n_lat,) tp and fp accumulators on CPU long.
    tp = torch.zeros((n_features, n_lat), dtype=torch.long)
    fp = torch.zeros((n_features, n_lat), dtype=torch.long)
    pos_count = val_y.long().sum(dim=0)  # (n_features,) total val positives

    bs = max(1, getattr(cfg, "causal_batch_size", 4))
    flat_offset = 0
    with torch.no_grad():
        for s in range(0, n_seqs, bs):
            tk = tokens[s : s + bs].to(cfg.device)
            B, Tp = tk.shape
            chunk_T = B * Tp
            _, cache = model.run_with_cache(
                tk, names_filter=cfg.hook_point, return_type=None
            )
            x = cache[cfg.hook_point]  # (B, T, d)
            z = sae.encode(x.reshape(-1, x.shape[-1]))  # (B*T, n_lat)
            fires = (z > 0).cpu()  # (chunk_T, n_lat) bool

            # Slice this chunk's val mask + val labels.
            chunk_val_mask = val_mask_flat[flat_offset : flat_offset + chunk_T]
            if chunk_val_mask.any():
                chunk_y = val_y_full[flat_offset : flat_offset + chunk_T]
                # Restrict to val positions only: small (n_val_in_chunk, n_lat)
                # and (n_val_in_chunk, n_features) tensors.
                fires_v = fires[chunk_val_mask]            # (n_v, n_lat)
                y_v = chunk_y[chunk_val_mask]              # (n_v, n_features)

                # tp_inc[c, k] = sum over n_v of (fires_v[:, k] & y_v[:, c])
                # = y_v.T @ fires_v   (matmul over n_v dimension)
                fires_long = fires_v.long()
                y_long = y_v.long()
                tp_inc = y_long.T @ fires_long             # (n_features, n_lat)
                # fp_inc[c, k] = sum_{n_v} fires_v[:, k] & ~y_v[:, c]
                # = (~y_v).T @ fires_v
                fp_inc = (~y_v).long().T @ fires_long      # (n_features, n_lat)
                tp += tp_inc
                fp += fp_inc

            flat_offset += chunk_T

    return tp, fp, pos_count


def _eval_oracle_on_test(
    cfg: Config,
    tokens: torch.Tensor,
    test_y: torch.Tensor,
    test_idx_to_pos: torch.Tensor,
    oracle_latent_per_feature: torch.Tensor,  # (n_features,) long, -1 = skip
) -> torch.Tensor:
    """Compute F1 on test for each feature's chosen oracle latent.

    Streams tokens; for each batch, only computes activations for the
    UNIQUE set of chosen latents (typically far fewer than 24576).
    Vectorizes per-feature counts via index lookup.
    """
    from .inventory import load_sae, load_target_model
    sae, _ = load_sae(cfg)
    model = load_target_model(cfg)
    sae = sae.to(cfg.device)

    n_features = test_y.shape[1]
    n_seqs, T_per = tokens.shape

    test_mask_flat = torch.zeros(n_seqs * T_per, dtype=torch.bool)
    test_mask_flat[test_idx_to_pos] = True
    test_y_full = torch.zeros((n_seqs * T_per, n_features), dtype=torch.bool)
    test_y_full[test_idx_to_pos] = test_y

    valid_features = oracle_latent_per_feature >= 0
    # Counts per feature.
    tp = torch.zeros(n_features, dtype=torch.long)
    fp = torch.zeros(n_features, dtype=torch.long)
    fn = torch.zeros(n_features, dtype=torch.long)

    bs = max(1, getattr(cfg, "causal_batch_size", 4))
    flat_offset = 0
    with torch.no_grad():
        for s in range(0, n_seqs, bs):
            tk = tokens[s : s + bs].to(cfg.device)
            B, Tp = tk.shape
            chunk_T = B * Tp
            _, cache = model.run_with_cache(
                tk, names_filter=cfg.hook_point, return_type=None
            )
            x = cache[cfg.hook_point]
            z = sae.encode(x.reshape(-1, x.shape[-1]))
            fires_all = (z > 0).cpu()  # (chunk_T, n_lat)

            chunk_test_mask = test_mask_flat[flat_offset : flat_offset + chunk_T]
            if chunk_test_mask.any():
                chunk_y = test_y_full[flat_offset : flat_offset + chunk_T]
                fires_t = fires_all[chunk_test_mask]   # (n_t, n_lat)
                y_t = chunk_y[chunk_test_mask]         # (n_t, n_features)

                for c in range(n_features):
                    if not valid_features[c]:
                        continue
                    k = int(oracle_latent_per_feature[c])
                    pred = fires_t[:, k]
                    yc = y_t[:, c]
                    tp[c] += int((pred & yc).sum())
                    fp[c] += int((pred & ~yc).sum())
                    fn[c] += int((~pred & yc).sum())

            flat_offset += chunk_T

    denom = (2 * tp + fp + fn).clamp(min=1)
    f1 = (2 * tp.float()) / denom.float()
    f1[~valid_features] = float("nan")
    return f1


def run(cfg: Config = None) -> dict:
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print("ORACLE-UNSUP  (Type-2 appendix: best unsup latent vs Opus labels)")
    print("=" * 70)

    annot_path = cfg.annotations_path
    tokens_path = cfg.tokens_path
    if not (annot_path.exists() and tokens_path.exists()):
        raise FileNotFoundError(
            f"Need {annot_path}, {tokens_path}. Run --step annotate first."
        )

    opus_leaves = _load_sup_arm_leaves(cfg)
    opus_cols: list[tuple[int, dict]] = list(enumerate(opus_leaves))
    if not opus_cols:
        raise RuntimeError(
            f"No leaves in {cfg.catalog_path}. Empty catalog?"
        )

    annotations = torch.load(annot_path, weights_only=True).bool()
    tokens = torch.load(tokens_path, weights_only=True)

    n_seqs, T_per = tokens.shape
    flat_total = n_seqs * T_per
    print(f"  Opus columns:    {len(opus_cols)}  (= n_features in catalog)")
    print(f"  Sequences:       {n_seqs}  Tokens/seq: {T_per}")
    print(f"  Total positions: {flat_total:,}")

    # v8.19.2 methodology fix: use the same shuffled flat-position
    # permutation as evaluate.py. Oracle val ↔ evaluate val_idx;
    # oracle test ↔ evaluate test_idx — so the Type-2 oracle compares
    # to sup F1 on identical positions.
    if not cfg.split_path.exists():
        raise FileNotFoundError(
            f"Need {cfg.split_path}. Run --step train (sup arm) first; "
            f"it writes split_indices.pt that this Type-2 appendix reuses."
        )
    perm = torch.load(cfg.split_path, weights_only=True)
    if perm.numel() != flat_total:
        raise RuntimeError(
            f"split_indices.pt has {perm.numel()} entries but tokens "
            f"flatten to {flat_total}. Mismatch."
        )
    split_idx = int(cfg.train_fraction * flat_total)
    remaining = flat_total - split_idx
    val_size = remaining // 2
    val_split = split_idx + val_size
    val_idx_to_pos = perm[split_idx:val_split]
    test_idx_to_pos = perm[val_split:]
    print(f"  val/test (flat positions, evaluate-aligned): "
          f"val={val_idx_to_pos.numel():,} test={test_idx_to_pos.numel():,}")
    if val_idx_to_pos.numel() == 0 or test_idx_to_pos.numel() == 0:
        raise RuntimeError(
            f"Held-out split has empty val or test (n_total={flat_total}, "
            f"train_fraction={cfg.train_fraction})."
        )

    # Restrict annotations to Opus columns and slice to val / test.
    opus_col_idx = torch.tensor([c for c, _ in opus_cols], dtype=torch.long)
    annot_flat = annotations.reshape(flat_total, -1).index_select(-1, opus_col_idx)
    val_y = annot_flat[val_idx_to_pos]                # (n_val, n_features)
    test_y = annot_flat[test_idx_to_pos]              # (n_test, n_features)

    # Filter sparse-positive features (oracle is uninformative below 5
    # positives). Keep their indices.
    val_pos = val_y.long().sum(dim=0)
    keep = val_pos >= 5
    n_eval = int(keep.sum())
    if n_eval == 0:
        print("  No features with ≥ 5 positives in val; skipping.")
        out = {
            "n_opus_features": len(opus_cols),
            "n_evaluated": 0,
            "per_feature": [],
        }
        out_path = cfg.output_dir / "oracle_unsup.json"
        out_path.write_text(json.dumps(out, indent=2))
        return out

    print(f"  Features with ≥5 val positives:  {n_eval}/{len(opus_cols)}")
    print(f"\n  [val pass] streaming SAE over all {n_seqs} sequences "
          f"(scoring only val positions)...")
    tp, fp, pos_count = _stream_firing_counts(cfg, tokens, val_y, val_idx_to_pos)
    fn = pos_count.unsqueeze(1) - tp
    denom = (2 * tp + fp + fn).clamp(min=1)
    val_f1_per_lat = (2 * tp.float()) / denom.float()  # (n_features, n_lat)

    # Oracle latent: argmax over n_lat per feature. -1 if skipped.
    oracle_latent = val_f1_per_lat.argmax(dim=1)
    oracle_latent[~keep] = -1
    val_f1_at_oracle = val_f1_per_lat.gather(
        1, oracle_latent.clamp(min=0).unsqueeze(1)
    ).squeeze(1)
    val_f1_at_oracle[~keep] = float("nan")

    print(f"  [test pass] computing F1 at chosen oracle latents on test "
          f"split...")
    test_f1 = _eval_oracle_on_test(
        cfg, tokens, test_y, test_idx_to_pos, oracle_latent
    )

    records = []
    for i, (col, leaf) in enumerate(opus_cols):
        records.append({
            "id": leaf["id"],
            "n_pos_val": int(pos_count[i]),
            "oracle_latent": int(oracle_latent[i].item()) if keep[i] else None,
            "val_f1": (
                round(float(val_f1_at_oracle[i].item()), 4) if keep[i] else None
            ),
            "test_f1": (
                round(float(test_f1[i].item()), 4) if keep[i] else None
            ),
            "skipped": (None if keep[i] else "n_pos_val<5"),
        })

    valid = [r for r in records if r["test_f1"] is not None]
    if valid:
        test_f1s = np.array([r["test_f1"] for r in valid])
        val_f1s = np.array([r["val_f1"] for r in valid])
        print(f"\n  Oracle-1 unsup F1 on TEST: "
              f"mean={test_f1s.mean():.3f}  median={np.median(test_f1s):.3f}")
        print(f"                       VAL:  "
              f"mean={val_f1s.mean():.3f}  median={np.median(val_f1s):.3f}")
        print(f"  Selection bias (val − test): "
              f"{(val_f1s.mean() - test_f1s.mean()):+.3f}")

    out = {
        "n_opus_features": len(opus_cols),
        "n_evaluated": len(valid),
        "split": {
            "train_fraction": cfg.train_fraction,
            "val_positions_flat": int(val_idx_to_pos.numel()),
            "test_positions_flat": int(test_idx_to_pos.numel()),
            "split_path": str(cfg.split_path),
        },
        "test_f1_mean": float(test_f1s.mean()) if valid else None,
        "test_f1_median": float(np.median(test_f1s)) if valid else None,
        "val_f1_mean": float(val_f1s.mean()) if valid else None,
        "val_f1_median": float(np.median(val_f1s)) if valid else None,
        "selection_bias_val_minus_test": (
            float(val_f1s.mean() - test_f1s.mean()) if valid else None
        ),
        "per_feature": records,
    }
    out_path = cfg.output_dir / "oracle_unsup.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")
    return out
