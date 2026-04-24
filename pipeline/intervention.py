"""
Experiment C — Intervention Precision

For each of 5 high-causal-KL supervised features, compare single-latent
ablation precision across three latent pools:

    (S) our supervised latent for f
    (U) the single unsupervised latent in our SAE with highest coverage of f
    (P) the single pretrained latent with highest coverage of f

For each pool, measure KL divergence at positive positions (where f fires) AND
at an equal-size random sample of negative positions. The ratio

    targeting_ratio = mean_KL(P_pos) / mean_KL(P_neg)

measures how targeted the intervention is:
  - High ratio → clean, concept-specific effect
  - Low ratio → diffuse, polysemantic effect

Output:
    pipeline_data/intervention_precision.json

Usage:
    python -m pipeline.run --step intervention
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import Config
from .train import SupervisedSAE


TARGET_FEATURE_HINTS = [
    "digit_or_number",
    "markup_tag",
    "database_schema",
    "programming_keyword",
    "article_or_determiner",
]


def _find_target_indices(
    features: list[dict],
    annotations: torch.Tensor | None = None,
    min_positives: int = 30,
    target_count: int = 5,
) -> list[tuple[int, str]]:
    """Match TARGET_FEATURE_HINTS against leaf features, fall back to top-k
    by positive count if hints don't cover enough features."""
    matched: list[tuple[int, str]] = []
    used: set[int] = set()
    for hint in TARGET_FEATURE_HINTS:
        for idx, feat in enumerate(features):
            if feat.get("type") == "group":
                continue
            if idx in used:
                continue
            if hint in feat["id"]:
                matched.append((idx, feat["id"]))
                used.add(idx)
                break

    min_required = max(3, target_count // 2)
    if len(matched) < min_required and annotations is not None:
        pos_counts = annotations.reshape(-1, annotations.shape[-1]).sum(dim=0)
        leaf_flags = [
            i for i, f in enumerate(features)
            if f.get("type") != "group" and i < len(pos_counts)
        ]
        leaf_flags.sort(key=lambda i: -float(pos_counts[i].item()))
        for idx in leaf_flags:
            if idx in used:
                continue
            if float(pos_counts[idx].item()) < min_positives:
                break
            matched.append((idx, features[idx]["id"]))
            used.add(idx)
            if len(matched) >= target_count:
                break

    return matched[:target_count]


def _encode_all(
    sae_sup, sae_pre, activations_test, cfg, n_supervised, encode_dtype,
) -> tuple:
    """Encode the full test set through both SAEs, return fire masks.

    `encode_dtype` must match the dtype that sae_sup and sae_pre have been
    cast to. The input activations (fp32 on disk) are cast to this dtype
    before encoding to avoid silent PyTorch type-promotion surprises.

    Returns:
        fires_S: (n_test, n_supervised) bool
        fires_U: (n_test, n_unsup) bool
        fires_P: (n_test, d_sae_pre) bool
    """
    bs = cfg.batch_size
    fires_S, fires_U, fires_P = [], [], []
    with torch.no_grad():
        for i in range(0, activations_test.shape[0], bs):
            x_b = activations_test[i : i + bs].to(cfg.device).to(encode_dtype)
            _, _, sup_acts, all_acts = sae_sup(x_b)
            fires_S.append((sup_acts > 0).cpu())
            fires_U.append((all_acts[..., n_supervised:] > 0).cpu())
            z = sae_pre.encode(x_b)
            fires_P.append((z > 0).cpu())
    return (
        torch.cat(fires_S), torch.cat(fires_U), torch.cat(fires_P),
    )


def _best_match_latent(
    fires_pool: torch.Tensor, pos_mask: torch.Tensor,
) -> int:
    """Return latent index in the pool with highest coverage of pos_mask."""
    pos_fires = fires_pool[pos_mask]  # (n_pos, n_latents)
    cov = pos_fires.float().mean(dim=0)
    return int(cov.argmax().item())


# ── KL ablation for our supervised SAE (S and U share the baseline) ─────

def _make_sup_full_hook(sae_sup):
    def hook(resid, hook=None):
        flat = resid.reshape(-1, resid.shape[-1])
        recon, _, _, _ = sae_sup(flat)
        return recon.reshape(resid.shape)
    return hook


def _make_sup_ablate_hook(sae_sup, latent_idx):
    """Ablate ONE latent (supervised or unsupervised, by global index) from
    the full supervised SAE reconstruction."""
    def hook(resid, hook=None):
        flat = resid.reshape(-1, resid.shape[-1])
        _, _, _, acts = sae_sup(flat)
        acts = acts.clone()
        acts[:, latent_idx] = 0.0
        recon = sae_sup.decoder(acts)
        return recon.reshape(resid.shape)
    return hook


def _make_pre_full_hook(sae_pre):
    def hook(resid, hook=None):
        flat = resid.reshape(-1, resid.shape[-1])
        z = sae_pre.encode(flat)
        recon = sae_pre.decode(z)
        return recon.reshape(resid.shape)
    return hook


def _make_pre_ablate_hook(sae_pre, latent_idx):
    def hook(resid, hook=None):
        flat = resid.reshape(-1, resid.shape[-1])
        z = sae_pre.encode(flat).clone()
        z[:, latent_idx] = 0.0
        recon = sae_pre.decode(z)
        return recon.reshape(resid.shape)
    return hook


def _kl_on_positions(
    model, tokens, positions_by_seq, hook_point,
    baseline_hook, ablated_hook,
) -> tuple[float, int]:
    """Compute mean KL(baseline || ablated) at the given positions.

    Args:
        tokens: (N, T) int64
        positions_by_seq: dict {seq_idx: [pos_idx, ...]}
        hook_point: str
        baseline_hook: callable taking (resid) and returning full SAE recon
        ablated_hook: callable taking (resid) and returning ablated recon

    Returns:
        (mean_kl, total_positions_evaluated)
    """
    kl_sum = 0.0
    total = 0
    with torch.no_grad():
        for seq_idx, pos_list in positions_by_seq.items():
            toks = tokens[seq_idx : seq_idx + 1].to(model.cfg.device)

            base_logits = model.run_with_hooks(
                toks, fwd_hooks=[(hook_point, baseline_hook)]
            )
            abl_logits = model.run_with_hooks(
                toks, fwd_hooks=[(hook_point, ablated_hook)]
            )
            pos_t = torch.tensor(pos_list, device=model.cfg.device)
            base_lp = F.log_softmax(
                base_logits[0, pos_t].float().cpu(), dim=-1,
            )
            abl_lp = F.log_softmax(
                abl_logits[0, pos_t].float().cpu(), dim=-1,
            )
            kl = (base_lp.exp() * (base_lp - abl_lp)).sum(dim=-1)
            kl_sum += float(kl.clamp(min=0).sum().item())
            total += len(pos_list)
    return (kl_sum / total if total > 0 else 0.0), total


# ── Main run ────────────────────────────────────────────────────────────────

def run(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    from .inventory import load_sae, load_target_model

    print("Loading base model...")
    model = load_target_model(cfg)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype_torch = dtype_map.get(cfg.model_dtype, torch.float32)

    # ── Load supervised SAE ──────────────────────────────────────────────
    print("Loading supervised SAE...")
    model_cfg = torch.load(
        cfg.checkpoint_config_path, map_location="cpu", weights_only=True,
    )
    from .train import load_trained_sae
    sae_sup = load_trained_sae(model_cfg)
    sae_sup.load_state_dict(
        torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    )
    sae_sup = sae_sup.to(cfg.device).to(model_dtype_torch).eval()
    n_supervised = sae_sup.n_supervised

    # ── Load pretrained SAE ──────────────────────────────────────────────
    print("Loading pretrained SAE...")
    sae_pre, _ = load_sae(cfg)
    sae_pre = sae_pre.to(cfg.device)
    sae_pre.W_enc = sae_pre.W_enc.to(model_dtype_torch)
    sae_pre.W_dec = sae_pre.W_dec.to(model_dtype_torch)
    sae_pre.b_enc = sae_pre.b_enc.to(model_dtype_torch)
    sae_pre.b_dec = sae_pre.b_dec.to(model_dtype_torch)
    if sae_pre.threshold is not None:
        sae_pre.threshold = sae_pre.threshold.to(model_dtype_torch)
    d_sae_pre = sae_pre.d_sae

    # ── Load data ────────────────────────────────────────────────────────
    tokens = torch.load(cfg.tokens_path, weights_only=True)
    activations = torch.load(cfg.activations_path, weights_only=True)
    annotations = torch.load(cfg.annotations_path, weights_only=True)

    # Mask position 0 — same masking as train + eval, so positive/negative
    # position sets exclude BOS-artifact positions that would dominate
    # targeting_ratio numerator.
    from .position_mask import mask_leading
    tokens, activations, annotations = mask_leading(
        tokens, activations, annotations, cfg=cfg,
    )

    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]

    N, T, d_model = activations.shape
    n_features = annotations.shape[-1]

    # For best-match latent search we use the same test split as Exp A
    x_flat = activations.reshape(-1, d_model)
    y_flat = annotations.reshape(-1, n_features)
    n_total_vecs = x_flat.shape[0]
    if cfg.split_path.exists():
        perm = torch.load(cfg.split_path, weights_only=True)
    else:
        perm = torch.arange(n_total_vecs)
    split_idx = int(cfg.train_fraction * n_total_vecs)
    val_split = split_idx + (n_total_vecs - split_idx) // 2
    test_idx = perm[val_split:]
    x_test = x_flat[test_idx]
    y_test = y_flat[test_idx]

    print(f"Encoding test set for best-match search ({x_test.shape[0]:,} vectors)...")
    n_unsup = sae_sup.n_total - n_supervised
    fires_S, fires_U, fires_P = _encode_all(
        sae_sup, sae_pre, x_test, cfg, n_supervised, model_dtype_torch,
    )

    # ── Identify target features ─────────────────────────────────────────
    targets = _find_target_indices(features, annotations=annotations)
    if not targets:
        print("ERROR: no target features found. Check catalog/annotations.")
        return
    print(f"\nTarget features ({len(targets)}):")
    for idx, fid in targets:
        print(f"  [{idx}] {fid}")

    # ── For each feature, find (sequence, position) sets ────────────────
    # Positive set: positions where annotation[feat_idx] == 1
    # Negative set: sampled positions where annotation == 0
    # Limit to the first cfg.causal_n_sequences sequences for tractability
    n_causal = min(cfg.causal_n_sequences, tokens.shape[0])
    tokens_sub = tokens[:n_causal]
    annot_sub = annotations[:n_causal]  # (n_causal, T, n_features)

    rng = np.random.RandomState(cfg.seed)

    hook_point = cfg.hook_point
    sup_full_hook = _make_sup_full_hook(sae_sup)
    pre_full_hook = _make_pre_full_hook(sae_pre)

    results_per_feature = []

    print("\n" + "=" * 78)
    print(f"{'Feature':<34} {'Pool':<4} {'KL_pos':>8} {'KL_neg':>8} {'ratio':>8}")
    print("-" * 78)

    for feat_idx, feat_id in targets:
        if feat_idx >= n_supervised:
            # The catalog has more features than the SAE's supervised slice.
            # We can't ablate an (S) latent for this feature. Skip.
            print(f"  {feat_id:<32}  SKIP (feat_idx {feat_idx} >= n_sup {n_supervised})")
            continue
        if feat_idx >= annot_sub.shape[-1]:
            print(f"  {feat_id:<32}  SKIP (feat_idx out of annotation range)")
            continue

        # ── Build positive/negative position sets ───────────────────────
        active = annot_sub[:, :, feat_idx].bool()  # (n_causal, T)
        pos_coords = active.nonzero(as_tuple=False).tolist()  # list of [seq, pos]
        n_pos = len(pos_coords)
        if n_pos < 10:
            print(f"  {feat_id:<32}  SKIP (n_pos={n_pos})")
            continue

        # Sample equal number of negative positions across sequences
        neg_mask_flat = (~active).reshape(-1).numpy()
        neg_indices_flat = np.where(neg_mask_flat)[0]
        if len(neg_indices_flat) == 0:
            continue
        chosen = rng.choice(
            neg_indices_flat, size=min(n_pos, len(neg_indices_flat)), replace=False,
        )
        neg_coords = [
            [int(i) // T, int(i) % T] for i in chosen
        ]

        def group(coords):
            g: dict[int, list[int]] = {}
            for s, p in coords:
                g.setdefault(int(s), []).append(int(p))
            return g

        pos_by_seq = group(pos_coords)
        neg_by_seq = group(neg_coords)

        # ── Find best-match U and P latents ──────────────────────────────
        # y_test is (n_test_vecs, n_features), but fires_* are aligned to x_test.
        pos_mask_test = y_test[:, feat_idx].bool()
        u_match = _best_match_latent(fires_U, pos_mask_test)
        p_match = _best_match_latent(fires_P, pos_mask_test)

        def _ratio(kp: float, kn: float) -> float | None:
            """Return kp/kn, but treat (~0)/(~0) as None (undefined)."""
            eps = 1e-12
            if kp < eps and kn < eps:
                return None  # degenerate: ablation did nothing
            if kn < eps:
                return float("inf")
            return kp / kn

        # ── (S) ablate supervised latent feat_idx ────────────────────────
        s_hook = _make_sup_ablate_hook(sae_sup, feat_idx)
        kl_pos_S, _ = _kl_on_positions(
            model, tokens_sub, pos_by_seq, hook_point, sup_full_hook, s_hook,
        )
        kl_neg_S, _ = _kl_on_positions(
            model, tokens_sub, neg_by_seq, hook_point, sup_full_hook, s_hook,
        )
        ratio_S = _ratio(kl_pos_S, kl_neg_S)

        # ── (U) ablate unsupervised latent (global idx = n_supervised + u) ──
        u_global = n_supervised + u_match
        u_hook = _make_sup_ablate_hook(sae_sup, u_global)
        kl_pos_U, _ = _kl_on_positions(
            model, tokens_sub, pos_by_seq, hook_point, sup_full_hook, u_hook,
        )
        kl_neg_U, _ = _kl_on_positions(
            model, tokens_sub, neg_by_seq, hook_point, sup_full_hook, u_hook,
        )
        ratio_U = _ratio(kl_pos_U, kl_neg_U)

        # ── (P) ablate pretrained latent p_match ─────────────────────────
        p_hook = _make_pre_ablate_hook(sae_pre, p_match)
        kl_pos_P, _ = _kl_on_positions(
            model, tokens_sub, pos_by_seq, hook_point, pre_full_hook, p_hook,
        )
        kl_neg_P, _ = _kl_on_positions(
            model, tokens_sub, neg_by_seq, hook_point, pre_full_hook, p_hook,
        )
        ratio_P = _ratio(kl_pos_P, kl_neg_P)

        def _fmt_ratio(r: float | None) -> str:
            if r is None:
                return "   n/a"
            if np.isinf(r):
                return "   inf"
            return f"{r:8.2f}"

        def _save_ratio(r: float | None) -> float | None:
            if r is None or np.isinf(r) or np.isnan(r):
                return None
            return round(float(r), 4)

        for pool, kp, kn, r in [
            ("S", kl_pos_S, kl_neg_S, ratio_S),
            ("U", kl_pos_U, kl_neg_U, ratio_U),
            ("P", kl_pos_P, kl_neg_P, ratio_P),
        ]:
            print(f"  {feat_id:<32} {pool:<4} {kp:>8.4f} {kn:>8.4f} {_fmt_ratio(r)}")

        results_per_feature.append({
            "feature_idx": int(feat_idx),
            "feature_id": feat_id,
            "n_positives": n_pos,
            "n_negatives": len(neg_coords),
            "supervised": {
                "latent_idx": int(feat_idx),
                "kl_pos": round(kl_pos_S, 6),
                "kl_neg": round(kl_neg_S, 6),
                "targeting_ratio": _save_ratio(ratio_S),
            },
            "unsupervised": {
                "best_match_idx": int(u_match),
                "global_idx": int(u_global),
                "kl_pos": round(kl_pos_U, 6),
                "kl_neg": round(kl_neg_U, 6),
                "targeting_ratio": _save_ratio(ratio_U),
            },
            "pretrained": {
                "best_match_idx": int(p_match),
                "kl_pos": round(kl_pos_P, 6),
                "kl_neg": round(kl_neg_P, 6),
                "targeting_ratio": _save_ratio(ratio_P),
            },
        })
        print()

    # ── Aggregate ────────────────────────────────────────────────────────
    def _mean_ratio(pool: str) -> float:
        vals = [
            r[pool]["targeting_ratio"] for r in results_per_feature
            if r[pool]["targeting_ratio"] is not None
        ]
        return float(np.mean(vals)) if vals else 0.0

    agg = {
        "supervised_mean_ratio": round(_mean_ratio("supervised"), 4),
        "unsupervised_mean_ratio": round(_mean_ratio("unsupervised"), 4),
        "pretrained_mean_ratio": round(_mean_ratio("pretrained"), 4),
    }

    print("\n" + "=" * 78)
    print("AGGREGATE")
    print("-" * 78)
    print(f"  mean targeting_ratio  (S): {agg['supervised_mean_ratio']:>8.2f}")
    print(f"  mean targeting_ratio  (U): {agg['unsupervised_mean_ratio']:>8.2f}")
    print(f"  mean targeting_ratio  (P): {agg['pretrained_mean_ratio']:>8.2f}")

    output = {
        "experiment": "C_intervention_precision",
        "aggregate": agg,
        "per_feature": results_per_feature,
    }
    out_path = cfg.output_dir / "intervention_precision.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")
