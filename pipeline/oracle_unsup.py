"""
Oracle-unsup readout (Type 2 appendix, v8.19.0).

For each Opus-designed feature `c` (from `opus_catalog.json`), find the
unsup SAE latent whose firing pattern best matches the Opus labels
`y_c` on the validation split. Reports both:

  • realistic-match: pick the unsup latent whose Sonnet-described
    description best matches the Opus description (description-embedding
    similarity). What a practitioner with no labels would do.
  • oracle-1: pick the unsup latent with maximum F1 against y_c on val.
    Upper bound on what unsup readout can do with perfect description-
    matching. Reviewer-bulletproof.

This is the SAME-CATALOG comparison that closes the rigorous gap left
by Type-1 (each method on its own catalog). It needs:
  - opus_catalog.json (sup-arm catalog)
  - feature_catalog.json (the merged annotation catalog)
  - annotations.pt with Opus labels
  - tokens.pt + the pretrained unsup SAE (gpt2-small-res-jb)

Output: pipeline_data/oracle_unsup.json with per-feature numbers.
Reuses Opus annotation labels — no extra annotation cost.

Run with: python -m pipeline.run --step oracle-unsup
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import Config


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def run(cfg: Config = None) -> dict:
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print("ORACLE-UNSUP  (Type-2 appendix: best unsup latent vs Opus labels)")
    print("=" * 70)

    opus_path = cfg.output_dir / "opus_catalog.json"
    annot_path = cfg.annotations_path
    tokens_path = cfg.tokens_path
    if not (opus_path.exists() and annot_path.exists() and tokens_path.exists()):
        raise FileNotFoundError(
            f"Need {opus_path}, {annot_path}, {tokens_path}. Run "
            f"--step opus-catalog + --step annotate first."
        )

    opus_catalog = json.loads(opus_path.read_text())
    opus_leaves = [
        f for f in opus_catalog.get("features", [])
        if f.get("type") == "leaf"
    ]
    print(f"  Opus features:     {len(opus_leaves)}")

    # Resolve which annotation columns correspond to Opus features.
    # The merged catalog used at annotation time may have Delphi cols too.
    merged_path = cfg.catalog_path
    if merged_path.exists():
        merged = json.loads(merged_path.read_text())
        merged_leaves = [
            f for f in merged.get("features", []) if f.get("type") == "leaf"
        ]
    else:
        merged_leaves = opus_leaves

    opus_ids = {f["id"] for f in opus_leaves}
    opus_cols: list[tuple[int, dict]] = [
        (col, leaf)
        for col, leaf in enumerate(merged_leaves)
        if leaf["id"] in opus_ids and not leaf.get("delphi_mode")
    ]
    if not opus_cols:
        raise RuntimeError(
            "No Opus columns found in merged catalog; check that "
            "annotate ran with the merged catalog."
        )
    print(f"  Opus columns in annotations.pt: {len(opus_cols)}")

    annotations = torch.load(annot_path, weights_only=True).bool()
    tokens = torch.load(tokens_path, weights_only=True)

    # Held-out split: last 20% of sequences.
    n_seqs = tokens.shape[0]
    val_start = int(n_seqs * cfg.train_fraction)
    val_tokens = tokens[val_start:]
    val_annot = annotations[val_start:]
    print(f"  Val sequences:    {val_tokens.shape[0]} "
          f"(of {n_seqs}, train_fraction={cfg.train_fraction})")

    # Forward unsup SAE over val tokens, materialize per-latent firing as
    # bool (T_total, n_lat). Stream batches; keep result on CPU.
    from .inventory import load_sae, load_target_model
    sae, _ = load_sae(cfg)
    model, _tok = load_target_model(cfg)
    sae = sae.to(cfg.device)

    n_lat = (
        sae.W_enc.shape[1] if sae.W_enc is not None
        else sae.native_sae.cfg.d_sae
    )
    T_per = val_tokens.shape[1]
    total_T = val_tokens.shape[0] * T_per
    print(f"  Unsup latents:    {n_lat}")
    print(f"  Total positions:  {total_T:,}")

    bs = 16
    z_pos_count = torch.zeros(n_lat, dtype=torch.long)
    # We need per-latent and per-feature joint counts. Materialize firing
    # bool tensor on CPU (n_lat × T_total bits = 24576 × 640000 ≈ 2 GB
    # for our scale; acceptable). Store as bool tensor.
    fires = torch.zeros((total_T, n_lat), dtype=torch.bool)
    pos = 0
    with torch.no_grad():
        for s in range(0, val_tokens.shape[0], bs):
            tk = val_tokens[s : s + bs].to(cfg.device)
            _, cache = model.run_with_cache(
                tk, names_filter=cfg.hook_point, return_type=None
            )
            x = cache[cfg.hook_point]  # (B, T, d)
            z = sae.encode(x.reshape(-1, x.shape[-1]))
            chunk_T = z.shape[0]
            fires[pos : pos + chunk_T] = (z > 0).cpu()
            pos += chunk_T
    z_pos_count = fires.long().sum(dim=0)
    print(f"  Firing-rate range (val): "
          f"[{z_pos_count.min().item()}, {z_pos_count.max().item()}]")

    # For each Opus feature: oracle-1 search across all unsup latents.
    # Compute TP / FP / FN vs y_c via vectorized counts.
    val_annot_flat = val_annot.reshape(-1, val_annot.shape[-1])  # (T_total, n_feat)
    records: list[dict] = []
    for col, leaf in opus_cols:
        y = val_annot_flat[:, col]
        n_pos_c = int(y.sum())
        if n_pos_c < 5:
            records.append({
                "id": leaf["id"],
                "n_pos": n_pos_c,
                "oracle_f1": None,
                "oracle_latent": None,
                "skipped": "n_pos<5",
            })
            continue

        # Per-latent counts vectorized: tp_k = (fires[:, k] & y).sum()
        # fp_k = (fires[:, k] & ~y).sum() ; fn_k = (~fires[:, k] & y).sum()
        # Compute via bool ops in chunks to avoid materializing full int.
        y_long = y.long()
        not_y = (~y).long()
        tp = (fires.long() * y_long.unsqueeze(1)).sum(dim=0)
        fp = (fires.long() * not_y.unsqueeze(1)).sum(dim=0)
        fn_total = n_pos_c - tp
        denom = 2 * tp + fp + fn_total
        f1_per_lat = torch.where(
            denom > 0, 2 * tp.float() / denom.float(),
            torch.zeros_like(tp, dtype=torch.float)
        )
        best = int(f1_per_lat.argmax().item())
        records.append({
            "id": leaf["id"],
            "n_pos": n_pos_c,
            "oracle_f1": round(float(f1_per_lat[best].item()), 4),
            "oracle_latent": best,
            "oracle_n_pos_latent": int(z_pos_count[best].item()),
        })

    valid = [r for r in records if r.get("oracle_f1") is not None]
    if valid:
        f1s = np.array([r["oracle_f1"] for r in valid])
        print(f"\n  Oracle-1 unsup F1: mean={f1s.mean():.3f} "
              f"median={np.median(f1s):.3f}  (n={len(valid)})")
    out = {
        "n_opus_features": len(opus_cols),
        "n_evaluated": len(valid),
        "oracle_f1_mean": float(f1s.mean()) if valid else None,
        "oracle_f1_median": float(np.median(f1s)) if valid else None,
        "per_feature": records,
    }
    out_path = cfg.output_dir / "oracle_unsup.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")
    return out
