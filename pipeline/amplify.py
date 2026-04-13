"""
Experiment D — Activation Amplification Sweep

For each supervised feature, scale its activation by multipliers [1x, 2x, 5x, 10x]
during inference and measure KL divergence at positive (on-target) vs negative
(off-target) positions. The hypothesis:

    Frozen-decoder features maintain a cleaner on-target / off-target KL ratio
    as the multiplier increases, because the decoder column IS the analytical
    mean-shift direction. Learned-decoder features degrade at high multipliers
    because their direction is an approximation that drifts off-target when scaled.

The practical value: a frozen-decoder supervised latent is a "sharp scalpel" —
you can crank the multiplier and get a proportional, predictable effect on
behavior without amplifying directional noise.

Output:
    pipeline_data/amplify_sweep.json

Usage:
    python -m pipeline.run --step amplify
"""

import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import Config
from .train import SupervisedSAE


MULTIPLIERS = [1.0, 2.0, 5.0, 10.0]

TARGET_FEATURE_HINTS = [
    "digit_or_number",
    "numeric_digit",
    "article_or_determiner",
    "bracket_or_paren",
    "section_header",
    "dash_or_hyphen",
    "quotation_mark",
]


def _find_targets(features, annotations, target_count=5, min_pos=30):
    matched = []
    used = set()
    for hint in TARGET_FEATURE_HINTS:
        for idx, feat in enumerate(features):
            if feat.get("type") == "group" or idx in used:
                continue
            if hint in feat["id"]:
                matched.append((idx, feat["id"]))
                used.add(idx)
                break

    if len(matched) < max(3, target_count // 2) and annotations is not None:
        pos_counts = annotations.reshape(-1, annotations.shape[-1]).sum(dim=0)
        leaves = [
            i for i, f in enumerate(features)
            if f.get("type") != "group" and i < len(pos_counts)
        ]
        leaves.sort(key=lambda i: -float(pos_counts[i].item()))
        for idx in leaves:
            if idx in used:
                continue
            if float(pos_counts[idx].item()) < min_pos:
                break
            matched.append((idx, features[idx]["id"]))
            used.add(idx)
            if len(matched) >= target_count:
                break
    return matched[:target_count]


def _make_amplify_hook(sae, latent_idx, multiplier):
    """Hook: reconstruct with one latent's activation scaled by `multiplier`."""
    def hook(resid, hook=None):
        flat = resid.reshape(-1, resid.shape[-1])
        _, _, _, acts = sae(flat)
        acts = acts.clone()
        acts[:, latent_idx] = acts[:, latent_idx] * multiplier
        recon = sae.decoder(acts)
        return recon.reshape(resid.shape)
    return hook


def _make_full_hook(sae):
    def hook(resid, hook=None):
        flat = resid.reshape(-1, resid.shape[-1])
        recon, _, _, _ = sae(flat)
        return recon.reshape(resid.shape)
    return hook


def _kl_at_positions(model, tokens, positions_by_seq, hook_point,
                     baseline_hook, amplified_hook):
    """Mean KL(baseline || amplified) at the given positions."""
    kl_sum = 0.0
    total = 0
    with torch.no_grad():
        for seq_idx, pos_list in positions_by_seq.items():
            toks = tokens[seq_idx : seq_idx + 1].to(model.cfg.device)
            base_logits = model.run_with_hooks(
                toks, fwd_hooks=[(hook_point, baseline_hook)]
            )
            amp_logits = model.run_with_hooks(
                toks, fwd_hooks=[(hook_point, amplified_hook)]
            )
            pos_t = torch.tensor(pos_list, device=model.cfg.device)
            base_lp = F.log_softmax(base_logits[0, pos_t].float().cpu(), dim=-1)
            amp_lp = F.log_softmax(amp_logits[0, pos_t].float().cpu(), dim=-1)
            kl = (base_lp.exp() * (base_lp - amp_lp)).sum(dim=-1)
            kl_sum += float(kl.clamp(min=0).sum().item())
            total += len(pos_list)
    return (kl_sum / total if total > 0 else 0.0), total


def run(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    from transformer_lens import HookedTransformer

    print("Loading base model...")
    model = HookedTransformer.from_pretrained(
        cfg.model_name, device=cfg.device, dtype=cfg.model_dtype,
    )
    model.eval()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    model_dtype_torch = dtype_map.get(cfg.model_dtype, torch.float32)

    print("Loading supervised SAE...")
    model_cfg = torch.load(
        cfg.checkpoint_config_path, map_location="cpu", weights_only=True,
    )
    sae = SupervisedSAE(
        model_cfg["d_model"], model_cfg["n_supervised"],
        model_cfg["n_unsupervised"], model_cfg.get("n_lista_steps", 0),
    )
    sae.load_state_dict(
        torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    )
    sae = sae.to(cfg.device).to(model_dtype_torch).eval()
    n_supervised = sae.n_supervised

    tokens = torch.load(cfg.tokens_path, weights_only=True)
    annotations = torch.load(cfg.annotations_path, weights_only=True)
    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]

    targets = _find_targets(features, annotations)
    if not targets:
        print("ERROR: no target features found.")
        return
    print(f"\nTarget features ({len(targets)}):")
    for idx, fid in targets:
        print(f"  [{idx}] {fid}")

    n_causal = min(cfg.causal_n_sequences, tokens.shape[0])
    tokens_sub = tokens[:n_causal]
    annot_sub = annotations[:n_causal]
    N_sub, T = tokens_sub.shape[0], tokens_sub.shape[1]

    rng = np.random.RandomState(cfg.seed)
    hook_point = cfg.hook_point
    full_hook = _make_full_hook(sae)

    results_per_feature = []

    print(f"\nMultipliers: {MULTIPLIERS}")
    print(f"Sequences: {n_causal}")
    print("\n" + "=" * 90)
    print(f"{'Feature':<34} {'mult':>5} {'KL_pos':>8} {'KL_neg':>8} "
          f"{'ratio':>8} {'ratio/1x':>8}")
    print("-" * 90)

    for feat_idx, feat_id in targets:
        if feat_idx >= n_supervised or feat_idx >= annot_sub.shape[-1]:
            print(f"  {feat_id:<32}  SKIP (out of range)")
            continue

        active = annot_sub[:, :, feat_idx].bool()
        pos_coords = active.nonzero(as_tuple=False).tolist()
        n_pos = len(pos_coords)
        if n_pos < 10:
            print(f"  {feat_id:<32}  SKIP (n_pos={n_pos})")
            continue

        neg_mask_flat = (~active).reshape(-1).numpy()
        neg_flat = np.where(neg_mask_flat)[0]
        if len(neg_flat) == 0:
            continue
        chosen = rng.choice(neg_flat, size=min(n_pos, len(neg_flat)), replace=False)
        neg_coords = [[int(i) // T, int(i) % T] for i in chosen]

        def group(coords):
            g = {}
            for s, p in coords:
                g.setdefault(int(s), []).append(int(p))
            return g

        pos_by_seq = group(pos_coords)
        neg_by_seq = group(neg_coords)

        sweep = []
        base_ratio = None

        for mult in MULTIPLIERS:
            amp_hook = _make_amplify_hook(sae, feat_idx, mult)
            kl_pos, _ = _kl_at_positions(
                model, tokens_sub, pos_by_seq, hook_point, full_hook, amp_hook,
            )
            kl_neg, _ = _kl_at_positions(
                model, tokens_sub, neg_by_seq, hook_point, full_hook, amp_hook,
            )

            eps = 1e-12
            if kl_pos < eps and kl_neg < eps:
                ratio = None
            elif kl_neg < eps:
                ratio = float("inf")
            else:
                ratio = kl_pos / kl_neg

            if mult == 1.0 and ratio is not None and ratio != float("inf"):
                base_ratio = ratio

            ratio_vs_1x = None
            if ratio is not None and base_ratio is not None and base_ratio > 0:
                if not np.isinf(ratio):
                    ratio_vs_1x = ratio / base_ratio

            def fmt(r):
                if r is None:
                    return "   n/a"
                if np.isinf(r):
                    return "   inf"
                return f"{r:8.2f}"

            print(f"  {feat_id:<32} {mult:>5.1f} {kl_pos:>8.4f} {kl_neg:>8.4f} "
                  f"{fmt(ratio)} {fmt(ratio_vs_1x)}")

            sweep.append({
                "multiplier": mult,
                "kl_pos": round(kl_pos, 6),
                "kl_neg": round(kl_neg, 6),
                "targeting_ratio": round(ratio, 4) if ratio is not None and not np.isinf(ratio) else None,
                "ratio_vs_1x": round(ratio_vs_1x, 4) if ratio_vs_1x is not None else None,
            })

        results_per_feature.append({
            "feature_idx": int(feat_idx),
            "feature_id": feat_id,
            "n_positives": n_pos,
            "n_negatives": len(neg_coords),
            "sweep": sweep,
        })
        print()

    # ── Aggregate: how does targeting_ratio scale with multiplier? ──────
    print("=" * 90)
    print("AGGREGATE: mean targeting_ratio at each multiplier")
    print("-" * 90)
    for i, mult in enumerate(MULTIPLIERS):
        ratios = [
            r["sweep"][i]["targeting_ratio"]
            for r in results_per_feature
            if len(r["sweep"]) > i and r["sweep"][i]["targeting_ratio"] is not None
        ]
        mean_r = float(np.mean(ratios)) if ratios else 0.0
        print(f"  {mult:>5.1f}x  mean targeting_ratio = {mean_r:.2f}  "
              f"(n={len(ratios)} features)")

    print(f"\nInterpretation: if targeting_ratio HOLDS or INCREASES with multiplier,")
    print(f"the decoder direction is clean and amplification is predictable.")
    print(f"If targeting_ratio DEGRADES, directional drift means the decoder")
    print(f"column is an approximation that breaks under amplification.")

    output = {
        "experiment": "D_amplification_sweep",
        "multipliers": MULTIPLIERS,
        "per_feature": results_per_feature,
    }
    out_path = cfg.output_dir / "amplify_sweep.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")
