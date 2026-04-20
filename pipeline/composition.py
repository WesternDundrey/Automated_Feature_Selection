"""
Composition — Multi-feature Joint Ablation Linearity

Tests whether supervised SAE features compose additively under intervention:
if we ablate feature A alone and feature B alone, does ablating A and B
together produce a combined effect ≈ KL_A + KL_B?

Scientific claim: supervised SAEs with frozen (mean-shift) decoders produce
interventions that are approximately linear in the feature set. The nonlinearity
residual — `|KL_{A∪B} - (KL_A + KL_B)| / (KL_A + KL_B)` — grows with
decoder cosine between the pair, confirming the geometric prediction that
orthogonal decoder directions give independently composable interventions.

Usage:
    python -m pipeline.run --step composition
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import Config
from .train import SupervisedSAE
from .intervention import (
    _find_target_indices,
    _encode_all,
    _best_match_latent,
    _make_sup_full_hook,
    _make_pre_full_hook,
    _kl_on_positions,
)


# ── Multi-latent ablation hooks ─────────────────────────────────────────────

def _make_sup_ablate_multi_hook(sae_sup, latent_indices: Iterable[int]):
    """Ablate a set of latents (global indices into sae_sup's n_total)."""
    idx = torch.as_tensor(list(latent_indices), dtype=torch.long)

    def hook(resid, hook=None):
        flat = resid.reshape(-1, resid.shape[-1])
        _, _, _, acts = sae_sup(flat)
        acts = acts.clone()
        acts[:, idx] = 0.0
        recon = sae_sup.decoder(acts)
        return recon.reshape(resid.shape)

    return hook


def _make_pre_ablate_multi_hook(sae_pre, latent_indices: Iterable[int]):
    idx = torch.as_tensor(list(latent_indices), dtype=torch.long)

    def hook(resid, hook=None):
        flat = resid.reshape(-1, resid.shape[-1])
        z = sae_pre.encode(flat).clone()
        z[:, idx] = 0.0
        recon = sae_pre.decode(z)
        return recon.reshape(resid.shape)

    return hook


# ── Position-set construction ───────────────────────────────────────────────

def _union_position_set(
    annotations_sub: torch.Tensor, feat_indices: list[int],
) -> dict[int, list[int]]:
    """Return {seq_idx: [pos_idx, ...]} for positions where ANY feature fires.

    The joint ablation is evaluated on the UNION of positive sets so the
    comparison of KL_A, KL_B, KL_{A,B} is over the same positions.
    """
    n_seq, T, _ = annotations_sub.shape
    mask = torch.zeros(n_seq, T, dtype=torch.bool)
    for fi in feat_indices:
        if fi < annotations_sub.shape[-1]:
            mask = mask | annotations_sub[:, :, fi].bool()
    coords = mask.nonzero(as_tuple=False).tolist()
    g: dict[int, list[int]] = {}
    for s, p in coords:
        g.setdefault(int(s), []).append(int(p))
    return g, int(mask.sum().item())


def _sample_neg_position_set(
    annotations_sub: torch.Tensor, feat_indices: list[int],
    n_samples: int, rng: np.random.RandomState,
) -> dict[int, list[int]]:
    """Sample negative positions (no listed feature fires) matching n_samples."""
    n_seq, T, _ = annotations_sub.shape
    pos_mask = torch.zeros(n_seq, T, dtype=torch.bool)
    for fi in feat_indices:
        if fi < annotations_sub.shape[-1]:
            pos_mask = pos_mask | annotations_sub[:, :, fi].bool()
    neg_flat = (~pos_mask).reshape(-1).numpy()
    neg_indices = np.where(neg_flat)[0]
    if len(neg_indices) == 0:
        return {}
    k = min(n_samples, len(neg_indices))
    chosen = rng.choice(neg_indices, size=k, replace=False)
    g: dict[int, list[int]] = {}
    for i in chosen:
        g.setdefault(int(i) // T, []).append(int(i) % T)
    return g


# ── Decoder cosine (supervised / pretrained) ────────────────────────────────

def _decoder_cosine(sae, i: int, j: int) -> float:
    """Cosine of decoder columns i and j. sae.decoder.weight is (d_model, n_total)."""
    col_i = sae.decoder.weight[:, i].detach().float()
    col_j = sae.decoder.weight[:, j].detach().float()
    denom = (col_i.norm() * col_j.norm()).clamp(min=1e-12)
    return float((col_i @ col_j / denom).item())


def _pre_decoder_cosine(sae_pre, i: int, j: int) -> float:
    """Pretrained-SAE decoder-column cosine. W_dec is (d_sae, d_model)."""
    col_i = sae_pre.W_dec[i].detach().float()
    col_j = sae_pre.W_dec[j].detach().float()
    denom = (col_i.norm() * col_j.norm()).clamp(min=1e-12)
    return float((col_i @ col_j / denom).item())


# ── Linearity metric ────────────────────────────────────────────────────────

def _linearity_score(joint: float, individual_sum: float) -> float:
    """Return a bounded linearity measure in [0, 1].

      - 1.0 if joint KL exactly equals the sum of individual KLs
      - -> 0 as the deviation grows large relative to max(joint, sum).

    Symmetric in joint / sum; avoids the sign artefact where sum >> joint
    (subadditive / interference) vs joint >> sum (superadditive) both count.
    """
    denom = max(abs(joint), abs(individual_sum), 1e-12)
    return max(0.0, 1.0 - abs(joint - individual_sum) / denom)


# ── Core evaluation per feature subset ──────────────────────────────────────

def _eval_subset(
    model, tokens_sub, annot_sub, feat_indices,
    sae_sup, sae_pre,
    sup_full_hook, pre_full_hook,
    u_match_by_feat: dict[int, int], p_match_by_feat: dict[int, int],
    n_supervised: int,
    cfg: Config, rng: np.random.RandomState,
) -> dict:
    """Measure KL at union positions for individual and joint ablations.

    Returns a dict with per-pool (S, U, P) metrics.
    """
    pos_by_seq, n_pos = _union_position_set(annot_sub, feat_indices)
    if n_pos < cfg.composition_min_positives:
        return {"skipped": True, "n_pos": n_pos}

    neg_by_seq = _sample_neg_position_set(
        annot_sub, feat_indices, n_pos, rng,
    )

    result = {
        "feat_indices": feat_indices,
        "n_pos": n_pos,
        "n_neg": sum(len(v) for v in neg_by_seq.values()),
    }

    hook_point = cfg.hook_point

    # ── Supervised pool ──
    sup_indiv_kl = []
    for fi in feat_indices:
        h = _make_sup_ablate_multi_hook(sae_sup, [fi])
        kp, _ = _kl_on_positions(
            model, tokens_sub, pos_by_seq, hook_point, sup_full_hook, h,
        )
        sup_indiv_kl.append(kp)

    sup_joint_hook = _make_sup_ablate_multi_hook(sae_sup, feat_indices)
    kl_sup_joint, _ = _kl_on_positions(
        model, tokens_sub, pos_by_seq, hook_point, sup_full_hook, sup_joint_hook,
    )
    sup_sum = float(np.sum(sup_indiv_kl))
    result["supervised"] = {
        "individual_kls": [round(float(k), 6) for k in sup_indiv_kl],
        "sum_individual": round(sup_sum, 6),
        "joint": round(float(kl_sup_joint), 6),
        "linearity": round(_linearity_score(kl_sup_joint, sup_sum), 4),
        "ratio_joint_over_sum": (
            round(kl_sup_joint / sup_sum, 4) if sup_sum > 1e-12 else None
        ),
    }

    # ── Unsupervised best-match pool ──
    u_latents = [n_supervised + u_match_by_feat[fi] for fi in feat_indices]
    u_indiv_kl = []
    for ul in u_latents:
        h = _make_sup_ablate_multi_hook(sae_sup, [ul])
        kp, _ = _kl_on_positions(
            model, tokens_sub, pos_by_seq, hook_point, sup_full_hook, h,
        )
        u_indiv_kl.append(kp)

    u_joint_hook = _make_sup_ablate_multi_hook(sae_sup, u_latents)
    kl_u_joint, _ = _kl_on_positions(
        model, tokens_sub, pos_by_seq, hook_point, sup_full_hook, u_joint_hook,
    )
    u_sum = float(np.sum(u_indiv_kl))
    result["unsupervised"] = {
        "latent_indices_global": u_latents,
        "individual_kls": [round(float(k), 6) for k in u_indiv_kl],
        "sum_individual": round(u_sum, 6),
        "joint": round(float(kl_u_joint), 6),
        "linearity": round(_linearity_score(kl_u_joint, u_sum), 4),
        "ratio_joint_over_sum": (
            round(kl_u_joint / u_sum, 4) if u_sum > 1e-12 else None
        ),
    }

    # ── Pretrained best-match pool ──
    p_latents = [p_match_by_feat[fi] for fi in feat_indices]
    p_indiv_kl = []
    for pl in p_latents:
        h = _make_pre_ablate_multi_hook(sae_pre, [pl])
        kp, _ = _kl_on_positions(
            model, tokens_sub, pos_by_seq, hook_point, pre_full_hook, h,
        )
        p_indiv_kl.append(kp)

    p_joint_hook = _make_pre_ablate_multi_hook(sae_pre, p_latents)
    kl_p_joint, _ = _kl_on_positions(
        model, tokens_sub, pos_by_seq, hook_point, pre_full_hook, p_joint_hook,
    )
    p_sum = float(np.sum(p_indiv_kl))
    result["pretrained"] = {
        "latent_indices": p_latents,
        "individual_kls": [round(float(k), 6) for k in p_indiv_kl],
        "sum_individual": round(p_sum, 6),
        "joint": round(float(kl_p_joint), 6),
        "linearity": round(_linearity_score(kl_p_joint, p_sum), 4),
        "ratio_joint_over_sum": (
            round(kl_p_joint / p_sum, 4) if p_sum > 1e-12 else None
        ),
    }

    # ── Decoder-cosine diagnostics (K=2 only) ──
    if len(feat_indices) == 2:
        i, j = feat_indices
        ui, uj = u_latents
        pi, pj = p_latents
        result["decoder_cosines"] = {
            "supervised": round(_decoder_cosine(sae_sup, i, j), 4),
            "unsupervised": round(_decoder_cosine(sae_sup, ui, uj), 4),
            "pretrained": round(_pre_decoder_cosine(sae_pre, pi, pj), 4),
        }

    return result


# ── Main run ────────────────────────────────────────────────────────────────

def run(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    from .inventory import load_sae, load_target_model

    # Attach composition defaults if caller didn't set them (keeps Config
    # dataclass untouched; these knobs live here because only this step uses them).
    if not hasattr(cfg, "composition_n_targets"):
        cfg.composition_n_targets = 5
    if not hasattr(cfg, "composition_pair_ks"):
        cfg.composition_pair_ks = (2, 3)
    if not hasattr(cfg, "composition_min_positives"):
        cfg.composition_min_positives = 10
    if not hasattr(cfg, "composition_max_subsets_per_k"):
        cfg.composition_max_subsets_per_k = 20

    print("Loading base model...")
    model = load_target_model(cfg)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype_torch = dtype_map.get(cfg.model_dtype, torch.float32)

    # Supervised SAE
    print("Loading supervised SAE...")
    model_cfg = torch.load(
        cfg.checkpoint_config_path, map_location="cpu", weights_only=True,
    )
    sae_sup = SupervisedSAE(
        model_cfg["d_model"], model_cfg["n_supervised"],
        model_cfg["n_unsupervised"], model_cfg.get("n_lista_steps", 0),
    )
    sae_sup.load_state_dict(
        torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    )
    sae_sup = sae_sup.to(cfg.device).to(model_dtype_torch).eval()
    n_supervised = sae_sup.n_supervised

    # Pretrained SAE
    print("Loading pretrained SAE...")
    sae_pre, _ = load_sae(cfg)
    sae_pre = sae_pre.to(cfg.device)
    for attr in ("W_enc", "W_dec", "b_enc", "b_dec"):
        setattr(sae_pre, attr, getattr(sae_pre, attr).to(model_dtype_torch))
    if sae_pre.threshold is not None:
        sae_pre.threshold = sae_pre.threshold.to(model_dtype_torch)

    # Data
    tokens = torch.load(cfg.tokens_path, weights_only=True)
    activations = torch.load(cfg.activations_path, weights_only=True)
    annotations = torch.load(cfg.annotations_path, weights_only=True)
    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]
    n_features = annotations.shape[-1]

    # Select target features (same heuristic as intervention.py)
    targets = _find_target_indices(
        features, annotations=annotations, target_count=cfg.composition_n_targets,
    )
    if len(targets) < 2:
        raise RuntimeError(
            f"Need ≥2 target features for composition, got {len(targets)}. "
            f"Check TARGET_FEATURE_HINTS in intervention.py against your catalog."
        )
    print(f"\nTarget features ({len(targets)}):")
    for idx, fid in targets:
        print(f"  [{idx}] {fid}")

    target_idxs = [t[0] for t in targets if t[0] < n_supervised]
    if len(target_idxs) < 2:
        raise RuntimeError(
            f"Only {len(target_idxs)} targets fall within n_supervised={n_supervised}. "
            f"Need ≥2. Consider retraining with a catalog that matches."
        )

    # Best-match U and P for each target (needs test-set encoding)
    N, T, d_model = activations.shape
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
    fires_S, fires_U, fires_P = _encode_all(
        sae_sup, sae_pre, x_test, cfg, n_supervised, model_dtype_torch,
    )

    u_match_by_feat: dict[int, int] = {}
    p_match_by_feat: dict[int, int] = {}
    for fi in target_idxs:
        pos_mask_test = y_test[:, fi].bool()
        u_match_by_feat[fi] = _best_match_latent(fires_U, pos_mask_test)
        p_match_by_feat[fi] = _best_match_latent(fires_P, pos_mask_test)

    # Position-set source: first cfg.causal_n_sequences sequences
    n_causal = min(cfg.causal_n_sequences, tokens.shape[0])
    tokens_sub = tokens[:n_causal]
    annot_sub = annotations[:n_causal]

    sup_full_hook = _make_sup_full_hook(sae_sup)
    pre_full_hook = _make_pre_full_hook(sae_pre)

    rng = np.random.RandomState(cfg.seed)
    fid_by_idx = {fi: features[fi]["id"] for fi in target_idxs}

    # ── Iterate subsets of size k ──────────────────────────────────────────
    results_by_k: dict[int, list[dict]] = {}
    for k in cfg.composition_pair_ks:
        if k > len(target_idxs):
            continue
        subsets = list(itertools.combinations(target_idxs, k))
        if len(subsets) > cfg.composition_max_subsets_per_k:
            # Deterministic truncation — keeps reproducibility across runs.
            subsets = subsets[: cfg.composition_max_subsets_per_k]
        print(f"\n── K={k}: {len(subsets)} subsets ──")
        print(f"  {'features':<40} {'lin(S)':>8} {'lin(U)':>8} {'lin(P)':>8} "
              f"{'cos(S)':>8}")
        print("  " + "-" * 78)

        records: list[dict] = []
        for subset in tqdm(subsets, desc=f"K={k}", leave=False):
            r = _eval_subset(
                model, tokens_sub, annot_sub, list(subset),
                sae_sup, sae_pre,
                sup_full_hook, pre_full_hook,
                u_match_by_feat, p_match_by_feat,
                n_supervised, cfg, rng,
            )
            if r.get("skipped"):
                continue

            fids = [fid_by_idx[i] for i in subset]
            label = ",".join(f.split(".")[-1][:10] for f in fids)
            cos_s = r.get("decoder_cosines", {}).get("supervised")
            print(
                f"  {label:<40} "
                f"{r['supervised']['linearity']:>8.3f} "
                f"{r['unsupervised']['linearity']:>8.3f} "
                f"{r['pretrained']['linearity']:>8.3f} "
                f"{('-' if cos_s is None else f'{cos_s:.3f}'):>8}"
            )
            r["feature_ids"] = fids
            records.append(r)
        results_by_k[k] = records

    # ── Aggregate ──────────────────────────────────────────────────────────
    def _agg(pool: str, k: int) -> dict:
        vals = [r[pool]["linearity"] for r in results_by_k.get(k, [])]
        ratios = [
            r[pool]["ratio_joint_over_sum"] for r in results_by_k.get(k, [])
            if r[pool]["ratio_joint_over_sum"] is not None
        ]
        return {
            "n": len(vals),
            "mean_linearity": round(float(np.mean(vals)), 4) if vals else None,
            "median_linearity": round(float(np.median(vals)), 4) if vals else None,
            "mean_ratio_joint_over_sum": (
                round(float(np.mean(ratios)), 4) if ratios else None
            ),
        }

    summary: dict = {"per_k": {}}
    for k in results_by_k:
        summary["per_k"][str(k)] = {
            "supervised": _agg("supervised", k),
            "unsupervised": _agg("unsupervised", k),
            "pretrained": _agg("pretrained", k),
        }

    # Decoder-cosine vs linearity correlation (K=2 only)
    k2 = results_by_k.get(2, [])
    if k2:
        cos_s = np.array([r["decoder_cosines"]["supervised"] for r in k2])
        lin_s = np.array([r["supervised"]["linearity"] for r in k2])
        if len(cos_s) >= 3 and float(cos_s.std()) > 1e-6:
            corr = float(np.corrcoef(cos_s, lin_s)[0, 1])
            summary["cosine_linearity_correlation_sup_k2"] = round(corr, 4)

    print("\n" + "=" * 78)
    print("AGGREGATE")
    print("-" * 78)
    for k_str, agg in summary["per_k"].items():
        print(f"  K={k_str}:")
        for pool in ("supervised", "unsupervised", "pretrained"):
            a = agg[pool]
            if a["mean_linearity"] is None:
                continue
            print(
                f"    {pool:<14} n={a['n']:<3}  "
                f"mean_lin={a['mean_linearity']:.3f}  "
                f"median={a['median_linearity']:.3f}  "
                f"joint/sum={a['mean_ratio_joint_over_sum']}"
            )
    if "cosine_linearity_correlation_sup_k2" in summary:
        print(
            f"  corr(decoder_cos, linearity)[S, K=2] = "
            f"{summary['cosine_linearity_correlation_sup_k2']}"
        )

    output = {
        "experiment": "composition_linearity",
        "target_features": [{"idx": i, "id": fid_by_idx[i]} for i in target_idxs],
        "config": {
            "n_targets": cfg.composition_n_targets,
            "pair_ks": list(cfg.composition_pair_ks),
            "min_positives": cfg.composition_min_positives,
            "n_causal_sequences": n_causal,
        },
        "results_by_k": {str(k): v for k, v in results_by_k.items()},
        "summary": summary,
    }
    out_path = cfg.output_dir / "composition.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")
