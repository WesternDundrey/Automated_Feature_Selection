"""
Probe-vs-SAE causal asymmetry test (v8.18.38).

Mirror of `pipeline.causal.test_feature_necessity`, but ablating along
the linear-probe baseline's weight vectors instead of the supervised
SAE's decoder columns. Same labels, same train/test split, same KL
methodology.

Predicted outcome: probe directions show ~zero causal effect because
they are classifier weights optimizing BCE, not residual-stream
directions optimizing reconstruction. Per-feature ablation along
W_probe[k] removes a small, classification-relevant slice of variance
from the residual; the model's downstream computation depends very
little on the projection along that direction. Per-feature ablation
along the supervised SAE's decoder column k removes the data-anchored
mean-shift direction, which the model's downstream computation does
depend on.

If confirmed, this is the cleanest "supervised SAE >> linear probe"
claim: same labels, comparable F1, same train/test split, but only
the SAE's directions actually steer the model's predictions.

Methodological note: the SAE's existing test_feature_necessity uses a
slightly more invasive intervention (replace residual with the SAE's
full reconstruction, with feature k zeroed). The probe ablation here
uses pure projection-out (residual − projection along d_k), which is
LESS invasive. So if the probe ablation already shows lower KL than
the SAE ablation, the gap is real (and would likely grow further with
a fully-matched intervention scheme). The numbers should be read as
"probe directions show ≤ X KL when ablated; SAE directions show ≥ Y
KL" rather than as a head-to-head.

Reads (read-only):
  - cfg.tokens_path
  - cfg.activations_path
  - cfg.annotations_path
  - cfg.catalog_path
  - cfg.split_path  (split_indices.pt if present; falls back to seeded RNG)

Writes (new file, doesn't touch existing artifacts):
  - cfg.output_dir / "probe_causal.json"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import Config


def _train_probe_baseline(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    cfg: Config,
    n_features: int,
    epochs: int = 10,
) -> torch.nn.Linear:
    """Re-train the same linear probe baseline that evaluate.py builds.

    Identical recipe: nn.Linear(d_model, n_features), Adam lr=1e-3, BCE
    with class-balanced pos_weight, 10 epochs, cfg.batch_size mini-batches.
    Probe is deterministic given the seed + input ordering, so this
    produces the same weight matrix as evaluate.py's probe.
    """
    n_pos = y_train.sum(dim=0)
    pos_weight = ((y_train.shape[0] - n_pos) / n_pos.clamp(min=1)).clamp(max=100)

    d_model = x_train.shape[-1]
    probe = torch.nn.Linear(d_model, n_features)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    probe.train()
    for epoch in range(epochs):
        idx = torch.randperm(x_train.shape[0])
        for i in range(0, x_train.shape[0], cfg.batch_size):
            b = idx[i : i + cfg.batch_size]
            logits = probe(x_train[b])
            loss = loss_fn(logits, y_train[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    probe.eval()
    return probe


def _make_probe_projectout_hook(probe_weight: torch.Tensor, feature_idx: int):
    """Hook factory: project out the probe direction d_k from the residual.

        d_k = probe_weight[feature_idx]  (shape: d_model,)
        resid_ablated = resid - (resid @ d_k / ‖d_k‖²) · d_k

    This zeroes the residual's component along d_k at every position,
    leaving all orthogonal components intact. Standard "ablate this
    direction" intervention used in mech-interp work.
    """
    d = probe_weight[feature_idx]
    d_norm_sq = (d * d).sum().clamp(min=1e-8)

    def _hook(resid, hook=None):
        # resid: (batch, seq, d_model). proj_coef shape: (batch, seq).
        proj_coef = (resid @ d) / d_norm_sq
        return resid - proj_coef.unsqueeze(-1) * d
    return _hook


def run(cfg: Config = None) -> dict:
    """Per-feature causal ablation along linear-probe directions.

    Returns a dict {"features": [{"id", "n_active", "mean_kl", ...}, ...]}.
    Also saves to `cfg.output_dir / probe_causal.json`.
    """
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print("PROBE-CAUSAL: per-feature ablation along linear-probe directions")
    print("=" * 70)

    # ── Pre-flight: read-only artifact checks ───────────────────────────
    for path, name in [
        (cfg.tokens_path, "tokens.pt"),
        (cfg.activations_path, "activations.pt"),
        (cfg.annotations_path, "annotations.pt"),
        (cfg.catalog_path, "feature_catalog.json"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"probe-causal needs {name} at {path}. Run --step train + "
                f"--step evaluate first so the labeled-corpus artifacts exist."
            )

    # ── Load model ──────────────────────────────────────────────────────
    from .inventory import load_target_model
    print("Loading base model...")
    model = load_target_model(cfg)

    # ── Load data + mask BOS ────────────────────────────────────────────
    print("Loading tokens / activations / annotations...")
    tokens = torch.load(cfg.tokens_path, weights_only=True)
    activations = torch.load(cfg.activations_path, weights_only=True)
    annotations = torch.load(cfg.annotations_path, weights_only=True)

    from .position_mask import mask_leading
    tokens, activations, annotations = mask_leading(
        tokens, activations, annotations, cfg=cfg,
    )

    catalog = json.loads(cfg.catalog_path.read_text())
    features = [f for f in catalog.get("features", []) if f.get("type") == "leaf"]
    n_features = annotations.shape[-1]
    if len(features) != n_features:
        # The catalog might have groups in addition to leaves; the
        # annotation tensor is per-leaf. Use the leaves in the order they
        # appear (matches annotate.py's writing order).
        if len(features) < n_features:
            raise RuntimeError(
                f"catalog has {len(features)} leaves but annotations.pt has "
                f"{n_features} columns. Reset and re-annotate."
            )
        features = features[:n_features]

    # ── Recover the same train/val/test split evaluate.py used ──────────
    n_total_pos = activations.shape[0] * activations.shape[1]
    x_flat = activations.reshape(-1, activations.shape[-1])
    y_flat = annotations.reshape(-1, n_features).float()

    if cfg.split_path.exists():
        perm = torch.load(cfg.split_path, weights_only=True)
        print(f"  split: loaded {cfg.split_path.name} ({perm.shape[0]} indices)")
    else:
        torch.manual_seed(cfg.seed)
        perm = torch.randperm(n_total_pos)
        print(f"  split: regenerated via cfg.seed={cfg.seed} (no split_indices.pt)")

    split_idx = int(cfg.train_fraction * n_total_pos)
    train_idx = perm[:split_idx]

    x_train = x_flat[train_idx]
    y_train = y_flat[train_idx]
    print(f"  probe training set: {x_train.shape[0]:,} vectors × {n_features} features")

    # ── Train probe baseline (matches evaluate.py exactly) ──────────────
    print("Training linear probe baseline (10 epochs, BCE + class-balanced pos_weight)...")
    probe = _train_probe_baseline(x_train, y_train, cfg, n_features)
    probe_weight = probe.weight.detach()
    probe_weight_dev = probe_weight.to(cfg.device)
    print(f"  probe trained: weight matrix {tuple(probe_weight.shape)}")

    # ── Compute baseline log-probs (no ablation, normal forward pass) ───
    n_seqs = min(cfg.causal_n_sequences, tokens.shape[0])
    tokens_sub = tokens[:n_seqs]
    annot_sub = annotations[:n_seqs]
    print(f"\nCausal test subset: {n_seqs} sequences × {n_features} features")

    base_logprobs = []
    with torch.no_grad():
        for i in tqdm(range(n_seqs), desc="  Baseline (no ablation)"):
            toks = tokens_sub[i : i + 1].to(cfg.device)
            logits = model(toks, return_type="logits")
            # bf16 → fp32 before log_softmax to avoid underflow on
            # high-frequency tokens (matches the SAE-causal recipe).
            base_logprobs.append(F.log_softmax(logits[0].float().cpu(), dim=-1))

    # ── Per-feature probe-direction ablation + KL measurement ───────────
    rng_np = np.random.RandomState(cfg.seed)
    T = tokens_sub.shape[1]
    results: list[dict] = []

    for k in tqdm(range(n_features), desc="  Probe-direction ablation"):
        feat = features[k]
        active_mask = annot_sub[:, :, k].bool()
        n_active = int(active_mask.sum())

        if n_active < 5:
            results.append({
                "id": feat["id"],
                "n_active": n_active,
                "mean_kl": None,
                "pred_change_rate": None,
                "mean_kl_neg": None,
                "targeting_ratio": None,
            })
            continue

        # Match `n_active` random negative positions for targeting-ratio.
        neg_mask_flat = (~active_mask).reshape(-1).numpy()
        neg_flat = np.where(neg_mask_flat)[0]
        n_neg = min(n_active, len(neg_flat))
        chosen = rng_np.choice(neg_flat, size=n_neg, replace=False) if n_neg > 0 else []
        neg_mask = torch.zeros_like(active_mask)
        for i in chosen:
            neg_mask[int(i) // T, int(i) % T] = True

        ablate_hook = _make_probe_projectout_hook(probe_weight_dev, k)

        kl_pos_sum = 0.0
        kl_neg_sum = 0.0
        pred_changes = 0
        total_pos = 0
        total_neg = 0

        with torch.no_grad():
            for i in range(n_seqs):
                has_pos = active_mask[i].any()
                has_neg = neg_mask[i].any()
                if not has_pos and not has_neg:
                    continue
                toks = tokens_sub[i : i + 1].to(cfg.device)
                abl_logits = model.run_with_hooks(
                    toks, fwd_hooks=[(cfg.hook_point, ablate_hook)]
                )

                if has_pos:
                    pos = active_mask[i].nonzero(as_tuple=True)[0]
                    base_lp = base_logprobs[i][pos]
                    abl_lp = F.log_softmax(
                        abl_logits[0, pos].float().cpu(), dim=-1,
                    )
                    kl = (base_lp.exp() * (base_lp - abl_lp)).sum(dim=-1)
                    kl_pos_sum += kl.clamp(min=0).sum().item()
                    pred_changes += (
                        base_lp.argmax(-1) != abl_lp.argmax(-1)
                    ).sum().item()
                    total_pos += len(pos)

                if has_neg:
                    neg = neg_mask[i].nonzero(as_tuple=True)[0]
                    base_lp = base_logprobs[i][neg]
                    abl_lp = F.log_softmax(
                        abl_logits[0, neg].float().cpu(), dim=-1,
                    )
                    kl = (base_lp.exp() * (base_lp - abl_lp)).sum(dim=-1)
                    kl_neg_sum += kl.clamp(min=0).sum().item()
                    total_neg += len(neg)

        mean_kl = kl_pos_sum / total_pos if total_pos > 0 else 0.0
        mean_kl_neg = kl_neg_sum / total_neg if total_neg > 0 else 0.0
        change_rate = pred_changes / total_pos if total_pos > 0 else 0.0
        eps = 1e-9
        targeting_ratio = (
            mean_kl / max(mean_kl_neg, eps)
            if mean_kl_neg > eps else None
        )

        results.append({
            "id": feat["id"],
            "n_active": n_active,
            "n_neg_sampled": total_neg,
            "mean_kl": round(mean_kl, 6),
            "mean_kl_neg": round(mean_kl_neg, 6),
            "targeting_ratio": (
                round(targeting_ratio, 3)
                if targeting_ratio is not None else None
            ),
            "pred_change_rate": round(change_rate, 4),
        })

    # ── Summary print + write ───────────────────────────────────────────
    print(f"\n  {'Feature':<46} {'KL_pos':>8} {'KL_neg':>8} {'ratio':>8} {'dPred':>7} {'Active':>7}")
    print("  " + "─" * 90)

    sorted_results = sorted(
        results,
        key=lambda r: -(r.get("mean_kl") or 0),
    )
    for r in sorted_results:
        kl_str = f"{r['mean_kl']:8.4f}" if r.get("mean_kl") is not None else "      --"
        kln_str = f"{r['mean_kl_neg']:8.4f}" if r.get("mean_kl_neg") is not None else "      --"
        ratio_str = (
            f"{r['targeting_ratio']:8.2f}"
            if r.get("targeting_ratio") is not None else "     n/a"
        )
        dpred_str = (
            f"{r['pred_change_rate']:7.3f}"
            if r.get("pred_change_rate") is not None else "     --"
        )
        print(f"  {r['id'][:46]:<46} {kl_str} {kln_str} {ratio_str} {dpred_str} {r['n_active']:>7}")

    # Aggregate stats
    valid = [r for r in results if r.get("mean_kl") is not None]
    if valid:
        mean_kl = float(np.mean([r["mean_kl"] for r in valid]))
        mean_kl_neg_agg = float(np.mean([r["mean_kl_neg"] for r in valid]))
        ratios = [r["targeting_ratio"] for r in valid if r["targeting_ratio"] is not None]
        median_ratio = float(np.median(ratios)) if ratios else None
        n_causal_active = sum(
            1 for r in valid
            if (r["mean_kl"] or 0) > 0.01
            and (r["targeting_ratio"] or 0) > 3.0
        )

        print(f"\n  Mean KL_pos: {mean_kl:.4f}  |  Mean KL_neg: {mean_kl_neg_agg:.4f}")
        print(f"  Median targeting ratio: "
              + (f"{median_ratio:.2f}" if median_ratio is not None else "n/a"))
        print(f"  Causally specific (KL>0.01 AND ratio>3): "
              f"{n_causal_active}/{len(valid)}  ← compare to SAE-causal's 12/46")

    out_path = cfg.output_dir / "probe_causal.json"
    summary = {
        "features": results,
        "aggregate": {
            "mean_kl_pos": mean_kl if valid else None,
            "mean_kl_neg": mean_kl_neg_agg if valid else None,
            "median_targeting_ratio": median_ratio if valid else None,
            "n_causal_active": n_causal_active if valid else None,
            "n_features_evaluated": len(valid),
        },
        "config": {
            "n_sequences": n_seqs,
            "hook_point": cfg.hook_point,
            "ablation": "project-out",
            "probe_epochs": 10,
            "probe_loss": "bce_with_pos_weight",
        },
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")

    return summary
