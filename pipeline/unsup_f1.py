"""
Unsup-arm F1 (v8.19.2) — the headline number for the unsup side of the
Delphi-vs-Opus comparison.

Reads from `cfg.output_dir/`:
  - feature_catalog.json (Delphi-described latents; each leaf has
    `source_latents=[<unsup latent idx>]` from delphi_runner)
  - annotations.pt   (per-feature labels from --step annotate)
  - tokens.pt        (corpus tokens that were annotated)

Computes for each feature `c`:
  - unsup latent firing pattern over all corpus tokens (forward pass
    through the pretrained sae_lens SAE, threshold > 0)
  - F1(unsup_latent_fires vs label_c) on a held-out test split

Used as the unsup-arm number in the Type-1 native-pipeline F1
comparison: "unsup latent firing matches its own Delphi description's
labels at F1 = X."

NOTE: this assumes the cfg.output_dir IS the unsup arm's directory
(typically `pipeline_data_unsup/`). To be safe it checks the catalog
has `source: "delphi"` set; aborts with a clear error otherwise so
you don't accidentally run unsup-f1 against the sup arm's catalog.

Memory: streams batches through the SAE; per batch only materializes
(B*T, n_features) bools (small) for the labeled latents — no need
for the full unsup activation matrix since we only score the latents
named in the catalog.

Output: pipeline_data_unsup/unsup_f1.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import Config


def _load_unsup_arm(cfg: Config) -> tuple[list[dict], list[int]]:
    """Read the Delphi-mode feature_catalog.json from output_dir.

    Returns (leaves, latent_indices). The latent_indices list is
    aligned with leaves so leaves[i] describes unsup latent
    latent_indices[i].
    """
    cat_path = cfg.catalog_path
    if not cat_path.exists():
        raise FileNotFoundError(
            f"Missing {cat_path}. The unsup arm runs --step delphi-run "
            f"in this output_dir to write feature_catalog.json before "
            f"--step annotate; run those first."
        )
    catalog = json.loads(cat_path.read_text())
    if catalog.get("source") != "delphi":
        raise RuntimeError(
            f"{cat_path} does not look like a Delphi-arm catalog "
            f"(source={catalog.get('source')!r}). Are you sure "
            f"output_dir={cfg.output_dir} is the unsup arm? Re-run "
            f"with --output_dir pipeline_data_unsup."
        )

    leaves = [
        f for f in catalog.get("features", [])
        if f.get("type") == "leaf"
    ]
    latent_indices: list[int] = []
    bad: list[str] = []
    for leaf in leaves:
        srcs = leaf.get("source_latents") or []
        if not srcs:
            bad.append(leaf.get("id", "<no id>"))
            continue
        latent_indices.append(int(srcs[0]))
    if bad:
        raise RuntimeError(
            f"{len(bad)} Delphi leaves have no source_latents; first "
            f"few: {bad[:3]}. delphi_runner should always set this."
        )
    return leaves, latent_indices


def _compute_per_feature_f1(
    cfg: Config,
    tokens: torch.Tensor,
    test_y: torch.Tensor,
    test_mask_flat: torch.Tensor,
    test_y_full: torch.Tensor,
    latent_indices: list[int],
) -> torch.Tensor:
    """Streaming F1 over the test split.

    For each test position: unsup latent k fires iff sae.encode(x)[:, k] > 0.
    Computes per-feature (tp, fp, fn) → F1.
    """
    from .inventory import load_sae, load_target_model
    sae, _ = load_sae(cfg)
    model, _tok = load_target_model(cfg)
    sae = sae.to(cfg.device)

    n_seqs, T_per = tokens.shape
    n_features = test_y.shape[1]

    # Latent indices we care about, as a tensor for index_select.
    lat_idx_t = torch.tensor(latent_indices, dtype=torch.long, device=cfg.device)

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

            chunk_test_mask = test_mask_flat[flat_offset : flat_offset + chunk_T]
            if chunk_test_mask.any():
                _, cache = model.run_with_cache(
                    tk, names_filter=cfg.hook_point, return_type=None
                )
                x = cache[cfg.hook_point]
                z = sae.encode(x.reshape(-1, x.shape[-1]))
                # Project down to only the n_features latents we want.
                z_subset = z.index_select(1, lat_idx_t)  # (chunk_T, n_features)
                fires = (z_subset > 0).cpu()              # (chunk_T, n_features)

                chunk_y = test_y_full[flat_offset : flat_offset + chunk_T]
                fires_t = fires[chunk_test_mask]          # (n_t, n_features)
                y_t = chunk_y[chunk_test_mask]            # (n_t, n_features)

                # Per-feature counts: each column k matches latent k
                # against feature label k (1:1 alignment from delphi_runner).
                tp += (fires_t & y_t).sum(dim=0).long()
                fp += (fires_t & ~y_t).sum(dim=0).long()
                fn += (~fires_t & y_t).sum(dim=0).long()

            flat_offset += chunk_T

    denom = (2 * tp + fp + fn).clamp(min=1)
    f1 = (2 * tp.float()) / denom.float()
    return f1


def run(cfg: Config = None) -> dict:
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print(f"UNSUP-F1  (output_dir = {cfg.output_dir})")
    print("=" * 70)

    if not cfg.tokens_path.exists() or not cfg.annotations_path.exists():
        raise FileNotFoundError(
            f"Need {cfg.tokens_path} and {cfg.annotations_path}. "
            f"Run --step annotate in this output_dir first."
        )

    leaves, latent_indices = _load_unsup_arm(cfg)
    print(f"  Delphi leaves:      {len(leaves)}")

    annotations = torch.load(cfg.annotations_path, weights_only=True).bool()
    tokens = torch.load(cfg.tokens_path, weights_only=True)
    n_seqs, T_per = tokens.shape

    # Held-out: last (1 - train_fraction) of sequences. Match the
    # convention used by train.py's split for the sup arm so the
    # comparison is on the same proportion of data.
    test_start = int(n_seqs * cfg.train_fraction)
    test_seq = list(range(test_start, n_seqs))
    if not test_seq:
        raise RuntimeError(
            f"Held-out split has 0 sequences (train_fraction="
            f"{cfg.train_fraction}, n_seqs={n_seqs})."
        )
    print(f"  Test sequences:     {len(test_seq)} of {n_seqs} "
          f"(train_fraction={cfg.train_fraction})")

    # Annotations columns: assume aligned with leaves order (annotate.py
    # writes one column per leaf in catalog order; we order leaves by
    # iterating catalog["features"] type==leaf, same as annotate).
    if annotations.shape[-1] != len(leaves):
        raise RuntimeError(
            f"annotations.pt last-dim {annotations.shape[-1]} != "
            f"{len(leaves)} leaves in feature_catalog.json. Re-run "
            f"--step annotate after the catalog change."
        )

    test_y = annotations[test_seq].reshape(-1, len(leaves))
    test_mask_flat = torch.zeros(n_seqs * T_per, dtype=torch.bool)
    test_idx_to_pos = torch.tensor(
        [s * T_per + t for s in test_seq for t in range(T_per)],
        dtype=torch.long,
    )
    test_mask_flat[test_idx_to_pos] = True
    test_y_full = torch.zeros(
        (n_seqs * T_per, len(leaves)), dtype=torch.bool
    )
    test_y_full[test_idx_to_pos] = test_y

    print(f"  Streaming SAE over {n_seqs} sequences "
          f"(scoring only test positions)...")
    f1 = _compute_per_feature_f1(
        cfg, tokens, test_y, test_mask_flat, test_y_full, latent_indices
    )

    test_pos = test_y.long().sum(dim=0)
    valid = (test_pos >= 5).numpy()

    records: list[dict] = []
    for i, leaf in enumerate(leaves):
        records.append({
            "id": leaf["id"],
            "latent": int(latent_indices[i]),
            "n_pos_test": int(test_pos[i]),
            "f1": (
                round(float(f1[i].item()), 4) if valid[i] else None
            ),
            "skipped": (None if valid[i] else "n_pos_test<5"),
        })

    valid_records = [r for r in records if r["f1"] is not None]
    if valid_records:
        f1s = np.array([r["f1"] for r in valid_records])
        print(f"\n  Unsup-arm F1 (n={len(valid_records)}): "
              f"mean={f1s.mean():.3f}  median={np.median(f1s):.3f}")
    else:
        print("\n  No features with ≥5 test positives.")

    out = {
        "n_features": len(leaves),
        "n_evaluated": len(valid_records),
        "split": {
            "train_fraction": cfg.train_fraction,
            "test_seqs": len(test_seq),
        },
        "f1_mean": float(f1s.mean()) if valid_records else None,
        "f1_median": float(np.median(f1s)) if valid_records else None,
        "per_feature": records,
    }
    out_path = cfg.output_dir / "unsup_f1.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")
    return out
