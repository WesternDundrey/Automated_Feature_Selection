"""
Polysemy / monosemy report for the supervised SAE catalog (v8.18.38).

Two analyses, both READ-ONLY against existing artifacts:

  1. Pairwise overlap (from overlap_check.json, already computed at
     annotation time): fraction of feature pairs with IoU > 0.5,
     subset rate, redundant rate. The supervised catalog should show
     near-zero IoU>0.5 because catalog quality gates curate for
     distinguishability; unsup SAE latents are well-known to have
     5-15% high-IoU pairs from polysemy.

  2. Per-feature monosemy ratio (computed fresh from activations.pt +
     annotations.pt + supervised_sae.pt): for each feature k,
        monosemy = mean(ReLU(sup_pre_k), positive positions) /
                   mean(ReLU(sup_pre_k), negative positions)
     ReLU(pre_act) is the GATED ACTIVATION in hinge mode: equals 0 at
     properly-suppressed negative positions, equals pre_act magnitude
     at correctly-firing positive positions. Using `|pre_act|` instead
     of ReLU would invert the sign — hinge loss with margin=0 lets the
     encoder push pre_act arbitrarily negative at negatives without
     penalty, so `|pre_act|` at negatives can be LARGER than at
     positives even for a correctly trained feature. ReLU restores the
     intended semantics: how strongly does the feature ACTIVATE here.

     Supervised features by construction should have monosemy >> 1
     (gate fires at positive positions, ~0 at negative positions).
     Unsup SAE latents typically have monosemy 1-3 due to polysemy
     across multiple unrelated firing contexts.

Output: pipeline_data/polysemy_report.json + a printed summary.
Reads only: overlap_check.json, activations.pt, annotations.pt,
            supervised_sae.pt, feature_catalog.json.
Writes only: pipeline_data/polysemy_report.json (new).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import Config


def _summarize_overlap(overlap_path: Path, iou_threshold: float = 0.5) -> dict:
    """Read existing overlap_check.json and report polysemy proxies."""
    if not overlap_path.exists():
        return {"skipped": True, "reason": f"no overlap report at {overlap_path}"}

    data = json.loads(overlap_path.read_text())
    pairs = data.get("pairs") or []
    n_features = data.get("n_features_analyzed") or len(set(
        [p.get("a") for p in pairs] + [p.get("b") for p in pairs]
    ))

    n_pairs_total = len(pairs)
    redundant = [p for p in pairs if (p.get("iou") or 0) >= 0.8]
    high_iou = [p for p in pairs if (p.get("iou") or 0) >= iou_threshold]
    subset_strict = [
        p for p in pairs
        if max(p.get("p_a_given_b") or 0, p.get("p_b_given_a") or 0) >= 0.95
    ]

    return {
        "skipped": False,
        "n_features_analyzed": n_features,
        "n_pairs_total": n_pairs_total,
        "n_redundant_iou_ge_0.8": len(redundant),
        f"n_high_iou_ge_{iou_threshold}": len(high_iou),
        "n_subset_pairs_p_ge_0.95": len(subset_strict),
        "redundant_rate": (
            len(redundant) / max(n_pairs_total, 1)
        ),
        "high_iou_rate": (
            len(high_iou) / max(n_pairs_total, 1)
        ),
        "subset_rate": (
            len(subset_strict) / max(n_pairs_total, 1)
        ),
    }


def _per_feature_monosemy(cfg: Config) -> list[dict]:
    """Compute the supervised pre-activation monosemy ratio per feature.

    Loads activations.pt, runs them through the trained supervised SAE,
    and measures sup_pre[k] at positive vs negative positions. Reports
    mean(|sup_pre|, pos) / mean(|sup_pre|, neg) — a feature with high
    monosemy fires strongly only at labeled positions.

    Returns one record per feature with monosemy ratio + supporting
    statistics. None for features with insufficient positives.
    """
    from .train import load_trained_sae

    activations = torch.load(cfg.activations_path, weights_only=True)
    annotations = torch.load(cfg.annotations_path, weights_only=True)
    catalog = json.loads(cfg.catalog_path.read_text())
    leaves = [f for f in catalog.get("features", []) if f.get("type") == "leaf"]

    # Mask BOS/leading positions to match training-time conventions.
    # mask_leading expects token-included triples; we don't load tokens
    # here (this is a read-only analysis), so apply the same slice manually.
    if cfg.mask_first_n_positions > 0:
        activations = activations[:, cfg.mask_first_n_positions:, :].contiguous()
        annotations = annotations[:, cfg.mask_first_n_positions:, :].contiguous()

    n_seq, T, d_model = activations.shape
    n_features = annotations.shape[-1]

    # Load trained SAE.
    if not cfg.checkpoint_path.exists():
        raise FileNotFoundError(
            f"No supervised_sae.pt at {cfg.checkpoint_path}; can't compute "
            f"per-feature monosemy."
        )
    model_cfg = torch.load(
        cfg.checkpoint_config_path, map_location="cpu", weights_only=True,
    )
    sae = load_trained_sae(model_cfg)
    sae.load_state_dict(
        torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    )
    sae = sae.to(cfg.device).eval()

    # Forward in batches; capture sup_pre across all positions.
    flat_acts = activations.reshape(-1, d_model)
    flat_annot = annotations.reshape(-1, n_features)
    n_pos = flat_acts.shape[0]
    bs = max(getattr(cfg, "batch_size", 512), 256)

    sup_pre_chunks = []
    with torch.no_grad():
        for i in range(0, n_pos, bs):
            chunk = flat_acts[i : i + bs].to(cfg.device)
            _, sup_pre, _, _ = sae(chunk)
            # sup_pre shape: (chunk, n_supervised). Take only first n_features
            # (catalog leaves) — matches the BCE-supervised slice. Use ReLU
            # to match the gated activation semantics of hinge mode (the
            # actual feature ACTIVATION the model would see, which is 0 at
            # properly-suppressed negatives). |sup_pre| would invert the
            # sign of the comparison — hinge with margin=0 pushes pre_act
            # arbitrarily negative at negatives, making |pre_act| at
            # negatives spuriously larger than at positives.
            sup_pre_chunks.append(sup_pre[:, : n_features].clamp(min=0).cpu())
    sup_pre_all = torch.cat(sup_pre_chunks, dim=0)  # (n_pos, n_features)

    records = []
    for k, feat in enumerate(leaves[:n_features]):
        mask_pos = flat_annot[:, k].bool()
        n_p = int(mask_pos.sum())
        if n_p < 5:
            records.append({
                "id": feat["id"],
                "n_positive": n_p,
                "monosemy_ratio": None,
                "mean_abs_pre_pos": None,
                "mean_abs_pre_neg": None,
            })
            continue

        # Sample matched negatives for stable mean estimation.
        neg_idx_all = (~mask_pos).nonzero(as_tuple=True)[0]
        n_n = min(n_p, len(neg_idx_all))
        if n_n > 0:
            rng = np.random.RandomState(cfg.seed)
            chosen = rng.choice(neg_idx_all.numpy(), size=n_n, replace=False)
            mask_neg = torch.zeros_like(mask_pos)
            mask_neg[torch.from_numpy(chosen)] = True
        else:
            mask_neg = torch.zeros_like(mask_pos)

        mean_pos = float(sup_pre_all[mask_pos, k].mean().item())
        mean_neg = float(sup_pre_all[mask_neg, k].mean().item()) if n_n > 0 else 0.0
        eps = 1e-9
        ratio = mean_pos / max(mean_neg, eps) if mean_neg > eps else None

        records.append({
            "id": feat["id"],
            "n_positive": n_p,
            "n_negative_sampled": n_n,
            "mean_relu_pre_pos": round(mean_pos, 4),
            "mean_relu_pre_neg": round(mean_neg, 4),
            "monosemy_ratio": round(ratio, 3) if ratio is not None else None,
        })

    return records


def run(cfg: Config = None) -> dict:
    """Aggregate polysemy/monosemy report. Read-only on existing artifacts."""
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print("POLYSEMY / MONOSEMY REPORT")
    print("=" * 70)

    overlap_path = cfg.output_dir / "overlap_check.json"
    overlap_summary = _summarize_overlap(overlap_path)

    print(f"\n  Pairwise overlap (from {overlap_path.name}):")
    if overlap_summary.get("skipped"):
        print(f"    {overlap_summary.get('reason')}")
    else:
        print(f"    n features analyzed:  {overlap_summary['n_features_analyzed']}")
        print(f"    n pairs total:        {overlap_summary['n_pairs_total']}")
        print(f"    redundant (IoU≥0.8):  "
              f"{overlap_summary['n_redundant_iou_ge_0.8']} "
              f"({overlap_summary['redundant_rate']*100:.2f}%)")
        print(f"    high IoU (≥0.5):      "
              f"{overlap_summary.get('n_high_iou_ge_0.5', 0)} "
              f"({overlap_summary.get('high_iou_rate', 0)*100:.2f}%)")
        print(f"    subset (P≥0.95):      "
              f"{overlap_summary['n_subset_pairs_p_ge_0.95']} "
              f"({overlap_summary['subset_rate']*100:.2f}%)")
        print(f"    Comparison literature: unsupervised SAE latents typically "
              f"have 5-15% high-IoU pairs (polysemy). The supervised SAE's "
              f"catalog-quality gates curate for distinguishability, so the "
              f"redundant + subset rates are normally near-zero.")

    print(f"\n  Per-feature monosemy ratio (mean ReLU(sup_pre) at pos / neg):")
    monosemy = _per_feature_monosemy(cfg)

    valid = [r for r in monosemy if r.get("monosemy_ratio") is not None]
    if valid:
        ratios = sorted([r["monosemy_ratio"] for r in valid], reverse=True)
        median = float(np.median(ratios))
        mean = float(np.mean(ratios))
        n_strong = sum(1 for r in ratios if r >= 5.0)
        n_weak = sum(1 for r in ratios if r < 1.5)

        print(f"    n features evaluated:  {len(valid)}")
        print(f"    median ratio:          {median:.2f}")
        print(f"    mean ratio:            {mean:.2f}")
        print(f"    monosemy ≥ 5 (strong): {n_strong}/{len(valid)}")
        print(f"    monosemy < 1.5 (weak): {n_weak}/{len(valid)}")
        print(f"    Comparison: unsupervised SAE latents typically have monosemy "
              f"1-3 due to polysemy; supervised features should have ≥5.")

        # Top + bottom 10
        sorted_by_ratio = sorted(
            valid, key=lambda r: -(r["monosemy_ratio"] or 0),
        )
        print(f"\n    Top-10 most monosemantic:")
        for r in sorted_by_ratio[:10]:
            print(f"      {r['monosemy_ratio']:6.2f}× — {r['id']}  "
                  f"(n_pos={r['n_positive']})")
        print(f"\n    Bottom-10 (least monosemantic):")
        for r in sorted_by_ratio[-10:]:
            print(f"      {r['monosemy_ratio']:6.2f}× — {r['id']}  "
                  f"(n_pos={r['n_positive']})")

    out_path = cfg.output_dir / "polysemy_report.json"
    summary = {
        "overlap": overlap_summary,
        "monosemy": monosemy,
        "monosemy_aggregate": {
            "n_evaluated": len(valid),
            "median": median if valid else None,
            "mean": mean if valid else None,
            "n_strong_ge_5": n_strong if valid else None,
            "n_weak_lt_1.5": n_weak if valid else None,
        },
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")
    return summary
