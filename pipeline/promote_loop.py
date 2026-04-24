"""
Promote Loop — U→S capacity transfer via residual-ranked unsup latents

Supersedes `discover_loop.py` for iterative catalog growth. Fixes the main
design flaw in the earlier loop: `discover_loop.py` re-trains a fresh
unsupervised SAE on the RAW cached activations with the same seed every
round, so later rounds re-propose essentially the same latents. The only
thing that changes per round is what the merge gate dedups against — which
means "iteration" was a misnomer.

This loop instead uses the U slice of the ALREADY-TRAINED supervised SAE
as the proposal pool. Each round:

  1. Rank U latents by their contribution to reconstruction (ΔR² under
     per-latent ablation on val). A U latent with high ΔR² is carrying
     capacity the supervised slice didn't — a promotion candidate.
  2. Describe the top-K U latents via the existing Sonnet path (same
     functions inventory.py uses — collect_top_activations +
     explain_features).
  3. Gate (a) crispness (Sonnet: "is this a single nameable concept?"),
     (b) cosine against existing target_dirs + separability
     (pipeline/merge.py's two-gate dedup).
  4. Annotate the survivors with --full-desc prompts.
  5. Retrain the supervised SAE on the grown catalog.
  6. Per-feature post-training validation: drop any new feature whose
     calibrated F1 on val is below a floor (default 0.3). This prevents
     the monotonic-growth guarantee from doing all the work — unlearnable
     proposals are rejected.
  7. Verify capacity transfer: for each surviving new feature f, check
     that its best-match U latent in the NEW SAE has lower ΔR² than the
     best-match U latent had in the PREVIOUS SAE. Otherwise U hasn't
     given up the capacity, and the feature duplicates rather than
     promotes.

Two residuals worth distinguishing:
  - r_S  = x − recon_supervised_only(x): what the supervised slice
          alone fails to reconstruct. The right quantity for "promote
          U into S" because we want to name the capacity U carries on
          top of S.
  - r_SU = x − recon_full(x): what the ENTIRE dictionary (S+U) still
          misses. The right quantity for "what's left for unnamed
          capacity to explain" — i.e., when searching for NEW directions
          neither S nor U represents.

This loop uses per-U-latent ΔR² on x, which is equivalent to per-U ΔMSE
on r_S (up to a constant) because the supervised reconstruction is
additively separable from the U contribution.

Usage:
    python -m pipeline.run --step promote-loop \\
        --layer 9 --sae_id "blocks.9.hook_resid_pre" \\
        --local-annotator --full-desc \\
        --promote-top-k 20 --promote-max-iters 5
"""

from __future__ import annotations

import json
import re
import textwrap
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config


# ── Config knobs attached at runtime (not on the frozen Config dataclass) ──

def _attach_defaults(cfg: Config) -> None:
    if not hasattr(cfg, "promote_top_k"):
        cfg.promote_top_k = 20          # U latents to consider per round
    if not hasattr(cfg, "promote_min_delta_r2_u"):
        cfg.promote_min_delta_r2_u = 1e-4  # skip U latents with trivial ΔR²
    if not hasattr(cfg, "promote_min_n_pos"):
        cfg.promote_min_n_pos = 50      # min active positions for mean-shift
    if not hasattr(cfg, "promote_post_train_f1_floor"):
        cfg.promote_post_train_f1_floor = 0.30  # drop new features below this
    if not hasattr(cfg, "promote_capacity_transfer_ratio"):
        cfg.promote_capacity_transfer_ratio = 0.5  # new_u_dR2 must be ≤ this × old_u_dR2
    if not hasattr(cfg, "promote_max_iters"):
        cfg.promote_max_iters = 5
    if not hasattr(cfg, "promote_min_kept"):
        cfg.promote_min_kept = 3        # terminate if < this survive a round
    if not hasattr(cfg, "promote_cos_threshold"):
        cfg.promote_cos_threshold = 0.6
    if not hasattr(cfg, "promote_use_llm_separability"):
        cfg.promote_use_llm_separability = True


# ── Rank U latents by per-latent ΔR² on val ─────────────────────────────────

def rank_u_latents_by_delta_r2(
    sae, x_val: torch.Tensor, n_supervised: int, device: str,
) -> list[tuple[int, float]]:
    """For each unsupervised latent, compute ΔR² = R²(full) - R²(without u).

    A U latent's contribution to reconstruction quantifies how much r_S it
    explains. Sorting by this gives the natural priority for promotion.

    Returns: list of (u_local_idx, delta_r2) sorted descending by delta_r2.
             u_local_idx is within [0, n_unsupervised); add n_supervised to
             get the global acts/decoder index.
    """
    sae.eval()
    bs = 1024
    with torch.no_grad():
        acts_all_list = []
        for i in range(0, x_val.shape[0], bs):
            xb = x_val[i : i + bs].to(device)
            _, _, _, acts = sae(xb)
            acts_all_list.append(acts.cpu())
        acts_all = torch.cat(acts_all_list, dim=0)  # (N_val, n_total)

        # Full reconstruction MSE once.
        recon_full = sae.decoder(acts_all.to(device))
        x_val_dev = x_val.to(device)
        err_full = x_val_dev - recon_full
        mse_full = err_full.pow(2).mean().item()
        baseline_mse = (x_val_dev - x_val_dev.mean(0, keepdim=True)).pow(2).mean().item()
        # Guard: if the SAE gets worse than the mean baseline, ranking is meaningless.
        if baseline_mse < 1e-12:
            return []

        n_total = sae.n_total
        n_unsup = n_total - n_supervised
        w_dec = sae.decoder.weight.detach()  # (d_model, n_total), unit-normalized cols

        # Vectorized ΔMSE per latent via the analytical expansion:
        #   ||err + a_u * W_u||² − ||err||²  =  2 * a_u * <err, W_u> + a_u² * ||W_u||²
        # per-position, then averaged. a_u = acts_all[:, n_sup + u], W_u = w_dec[:, n_sup + u].
        u_cols = w_dec[:, n_supervised:]                   # (d_model, n_unsup)
        a_u = acts_all[:, n_supervised:].to(device)        # (N_val, n_unsup)

        # <err, W_u> for each u, each position: shape (N_val, n_unsup)
        err_proj = err_full @ u_cols                       # (N_val, n_unsup)
        w_norm_sq = u_cols.pow(2).sum(0)                   # (n_unsup,) — ≈1 if unit-norm

        delta_mse = (
            2.0 * (a_u * err_proj).mean(dim=0)
            + (a_u.pow(2) * w_norm_sq).mean(dim=0)
        )                                                  # (n_unsup,)
        delta_r2 = (delta_mse / max(baseline_mse, 1e-9)).cpu().numpy()

    ranking = sorted(
        enumerate(delta_r2.tolist()), key=lambda kv: -kv[1]
    )
    return ranking


# ── Wrap U slice as a PretrainedSAE-compatible object ──────────────────────

def _wrap_u_slice_as_pretrained(sae, n_supervised: int, d_model: int):
    """inventory.collect_top_activations expects a PretrainedSAE-like object.
    Build one from the U slice of the supervised SAE so we can reuse the
    top-activation collection machinery without duplicating it."""
    from .inventory import PretrainedSAE
    enc_w = sae.encoder.weight.detach()   # (n_total, d_model)
    enc_b = sae.encoder.bias.detach()     # (n_total,)
    dec_w = sae.decoder.weight.detach()   # (d_model, n_total)

    # PretrainedSAE stores W_enc as (d_model, d_sae) and W_dec as (d_sae, d_model).
    return PretrainedSAE(
        W_enc=enc_w[n_supervised:, :].T.contiguous(),
        W_dec=dec_w[:, n_supervised:].T.contiguous(),
        b_enc=enc_b[n_supervised:].clone(),
        b_dec=torch.zeros(d_model),   # SupervisedSAE has no decoder bias
        threshold=None,
    )


# ── Crispness gate (is this a single nameable concept?) ────────────────────

def _crispness_judgment(description: str, cfg: Config) -> tuple[bool, str]:
    """Ask Sonnet whether a description is a single operationally-testable
    concept vs a grab-bag. Returns (is_crisp, reason).

    Fails CLOSED: any LLM error / unparseable response rejects the proposal
    rather than accepting it, matching merge.py's gate policy.
    """
    from .llm import get_client, chat
    client = get_client()

    prompt = textwrap.dedent(f"""\
        You are a strict auditor deciding whether a candidate feature
        description is usable as supervision for a supervised sparse
        autoencoder latent.

        A crisp description names ONE operationally-testable concept. A
        reader should be able to look at a single token in context and
        answer yes/no without ambiguity.

        A non-crisp description names multiple concepts, is fuzzy about
        when it fires, describes a "context" rather than a token-level
        property, or mentions alternatives ("or", "and/or", "sometimes",
        "various"). Reject those.

        CANDIDATE description:
          {description}

        Reply with EXACTLY one JSON object, no other text:
        {{
          "crisp": <true or false>,
          "reason": "<one short sentence>"
        }}
    """)

    try:
        text = chat(client, cfg.organization_model, prompt, max_tokens=150)
    except Exception as e:
        return False, f"crispness LLM unreachable ({type(e).__name__}: {e})"

    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return False, f"crispness response unparseable: {text[:120]}"
    try:
        obj = json.loads(m.group(0))
        return bool(obj.get("crisp", False)), str(obj.get("reason", ""))
    except json.JSONDecodeError:
        return False, f"crispness JSON invalid: {text[:120]}"


# ── Mean-shift target direction from U latent's firing mask ────────────────

def _compute_mean_shift_dirs(
    sae, u_local_indices: list[int],
    activations_flat: torch.Tensor,
    n_supervised: int, d_model: int,
    min_n_pos: int = 50,
) -> tuple[list[int], torch.Tensor, list[int]]:
    """For each U latent in the candidate list, compute
        d_u = normalize(mean(x | latent u fires) − mean(x))
    on the full activation tensor.

    Returns:
        kept_indices: u_local_indices with enough positives
        dirs: (len(kept), d_model)
        n_pos_per_latent: list aligned with kept_indices
    """
    device = activations_flat.device if activations_flat.is_cuda else "cpu"
    X = activations_flat
    mean_all = X.mean(dim=0)

    enc_w = sae.encoder.weight.detach().to(X.dtype)  # (n_total, d_model)
    enc_b = sae.encoder.bias.detach().to(X.dtype)

    kept: list[int] = []
    dir_list: list[torch.Tensor] = []
    n_pos_list: list[int] = []

    for u_local in u_local_indices:
        u_global = n_supervised + u_local
        w = enc_w[u_global]  # (d_model,)
        b = enc_b[u_global]
        # ReLU pre-activation > 0 defines firing mask.
        pre = X @ w + b
        mask = pre > 0
        n_pos = int(mask.sum().item())
        if n_pos < min_n_pos:
            continue
        mean_pos = X[mask].mean(dim=0)
        d = mean_pos - mean_all
        norm = d.norm()
        if norm < 1e-8:
            continue
        dir_list.append(d / norm)
        kept.append(u_local)
        n_pos_list.append(n_pos)

    if not dir_list:
        return [], torch.zeros(0, d_model), []
    return kept, torch.stack(dir_list), n_pos_list


# ── Post-training per-feature validation ───────────────────────────────────

def _post_training_validation(
    new_feature_ids: list[str], cfg: Config,
) -> tuple[list[str], list[dict]]:
    """Read evaluation.json after retrain; keep features above the F1 floor,
    drop below. Returns (kept_ids, dropped_records).

    Uses `f1_cal` (calibrated) when available, else `f1` (t=0). Always
    reports whichever was used in the dropped record for audit.
    """
    if not cfg.eval_path.exists():
        # No eval ran — keep everything rather than drop blindly.
        return new_feature_ids, []
    eval_data = json.loads(cfg.eval_path.read_text())
    feats_map = {f["id"]: f for f in eval_data.get("features") or []}

    kept: list[str] = []
    dropped: list[dict] = []
    for fid in new_feature_ids:
        rec = feats_map.get(fid) or {}
        f1 = rec.get("cal_f1")
        metric = "cal_f1"
        if f1 is None:
            f1 = rec.get("f1")
            metric = "f1"
        if f1 is None or f1 < cfg.promote_post_train_f1_floor:
            dropped.append({
                "id": fid,
                "metric": metric,
                "f1": f1,
                "floor": cfg.promote_post_train_f1_floor,
            })
        else:
            kept.append(fid)
    return kept, dropped


# ── Verify capacity transfer (U → S) ───────────────────────────────────────

def _verify_capacity_transfer(
    old_ranking: dict[int, float],      # u_local_idx -> delta_r2 before promotion
    promoted_u_indices: list[int],      # U latents that were described + promoted
    new_sae, x_val: torch.Tensor, new_n_supervised: int, device: str,
    transfer_ratio: float,
) -> list[dict]:
    """For each promoted U latent, compute the SAME-position u latent's ΔR²
    in the NEW SAE. If it dropped below transfer_ratio × old_delta_r2,
    capacity successfully transferred from U into its matched S slot.

    Caveat: the new SAE's U slice is freshly trained, so u_local indices
    don't correspond 1:1 with the old U slice. We therefore compute a
    fresh ranking of the new SAE's full U slice and look at the
    distribution of top ΔR² values; the KEY signal is whether the NEW
    top-K ΔR²s are substantially lower than the OLD top-K ΔR²s for the
    promoted latents. This is a DISTRIBUTION-LEVEL transfer check, not
    per-latent, and is the strongest claim we can make given that U
    indices are arbitrary across retrains.
    """
    new_ranking = rank_u_latents_by_delta_r2(
        new_sae, x_val, new_n_supervised, device,
    )
    new_top_dr2 = [r for _, r in new_ranking[: len(promoted_u_indices)]]
    old_top_dr2 = [old_ranking[u] for u in promoted_u_indices]

    transfer_records = []
    for old_u, old_dr2, new_dr2 in zip(
        promoted_u_indices, old_top_dr2, new_top_dr2,
    ):
        transferred = (
            new_dr2 <= transfer_ratio * old_dr2
            if old_dr2 > 1e-12 else False
        )
        transfer_records.append({
            "old_u_local_idx": int(old_u),
            "old_delta_r2": round(float(old_dr2), 6),
            "new_top_delta_r2": round(float(new_dr2), 6),
            "ratio": (
                round(new_dr2 / old_dr2, 4) if old_dr2 > 1e-12 else None
            ),
            "transferred": bool(transferred),
        })
    return transfer_records


# ── Main driver ────────────────────────────────────────────────────────────

def run(cfg: Optional[Config] = None) -> list[dict]:
    """Execute the U→S promotion loop.

    Preconditions: the main pipeline must have been run once — i.e., the
    following artifacts exist:
        catalog_path, tokens_path, activations_path, annotations_path,
        annotations_meta_path, target_dirs_path, checkpoint_path,
        checkpoint_config_path, eval_path.
    """
    if cfg is None:
        cfg = Config()
    _attach_defaults(cfg)

    required = [
        cfg.catalog_path, cfg.tokens_path, cfg.activations_path,
        cfg.annotations_path, cfg.target_dirs_path, cfg.checkpoint_path,
        cfg.checkpoint_config_path, cfg.eval_path,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Promote loop requires {p}. Run the full pipeline "
                f"(`python -m pipeline.run`) first."
            )

    loop_dir = cfg.output_dir / "promote_loop"
    loop_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    # Deferred heavy imports
    from .train import SupervisedSAE, run as run_train
    from .annotate import run as run_annotate
    from .evaluate import evaluate as run_evaluate
    from .inventory import collect_top_activations, explain_features, load_target_model
    from .merge import merge_catalogs_by_direction

    device = cfg.device

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map.get(cfg.model_dtype, torch.float32)

    for iter_idx in range(cfg.promote_max_iters):
        iter_dir = loop_dir / f"round_{iter_idx:02d}"
        iter_dir.mkdir(exist_ok=True)
        print("\n" + "=" * 70)
        print(f"PROMOTE LOOP — ROUND {iter_idx}")
        print("=" * 70)

        catalog = json.loads(cfg.catalog_path.read_text())
        model_cfg = torch.load(
            cfg.checkpoint_config_path, map_location="cpu", weights_only=True,
        )
        sae = SupervisedSAE(
            model_cfg["d_model"], model_cfg["n_supervised"],
            model_cfg["n_unsupervised"], model_cfg.get("n_lista_steps", 0),
        )
        sae.load_state_dict(torch.load(
            cfg.checkpoint_path, map_location="cpu", weights_only=True,
        ))
        sae = sae.to(device).to(model_dtype).eval()
        n_supervised = sae.n_supervised

        # ── 1. Load val activations and rank U latents by ΔR² ──────────
        activations = torch.load(cfg.activations_path, weights_only=True)
        N, T, d_model = activations.shape
        x_flat = activations.reshape(-1, d_model).to(model_dtype)
        n_total_vecs = x_flat.shape[0]
        if cfg.split_path.exists():
            perm = torch.load(cfg.split_path, weights_only=True)
        else:
            perm = torch.arange(n_total_vecs)
        split_idx = int(cfg.train_fraction * n_total_vecs)
        val_split = split_idx + (n_total_vecs - split_idx) // 2
        val_idx = perm[split_idx:val_split]
        x_val = x_flat[val_idx]

        print("\n── Ranking unsupervised latents by ΔR² on val ──")
        ranking = rank_u_latents_by_delta_r2(sae, x_val, n_supervised, device)
        if not ranking:
            print("  No ranking possible (val baseline MSE too small). Aborting.")
            break
        ranking_map = dict(ranking)
        top_k = ranking[: cfg.promote_top_k]
        top_k_filtered = [
            (u, dr2) for u, dr2 in top_k if dr2 >= cfg.promote_min_delta_r2_u
        ]
        print(
            f"  Top-{cfg.promote_top_k} U latents by ΔR²: "
            f"{len(top_k_filtered)} above threshold "
            f"{cfg.promote_min_delta_r2_u}"
        )
        for u, dr2 in top_k_filtered[:10]:
            print(f"    U[{u}]  ΔR² = {dr2:+.5f}")
        if len(top_k_filtered) < cfg.promote_min_kept:
            print(
                f"► TERMINATED: only {len(top_k_filtered)} candidates above "
                f"ΔR² threshold (< {cfg.promote_min_kept})."
            )
            history.append({
                "iter": iter_idx,
                "converged_reason": "too_few_candidates",
                "n_candidates": len(top_k_filtered),
            })
            break
        candidate_u_indices = [u for u, _ in top_k_filtered]

        # ── 2. Describe candidates via Sonnet ──────────────────────────
        print("\n── Extracting top activations for candidate U latents ──")
        top_acts_path = iter_dir / "top_activations.json"
        descriptions_path = iter_dir / "descriptions.json"
        if descriptions_path.exists():
            descriptions = json.loads(descriptions_path.read_text())
            print(f"  Loaded cached descriptions: {descriptions_path}")
        else:
            # collect_top_activations needs a model + PretrainedSAE-like wrapper
            model = load_target_model(cfg)
            tokenizer = model.tokenizer
            wrapped_u = _wrap_u_slice_as_pretrained(sae, n_supervised, d_model)
            top_acts = collect_top_activations(
                model, wrapped_u, tokenizer, candidate_u_indices, cfg,
            )
            top_acts_path.write_text(json.dumps(top_acts, indent=2))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            from transformers import AutoTokenizer
            tokenizer_ref = AutoTokenizer.from_pretrained(cfg.model_name)
            descriptions = explain_features(top_acts, tokenizer_ref, cfg)
            descriptions_path.write_text(json.dumps(descriptions, indent=2))

        # ── 3a. Crispness gate ──────────────────────────────────────────
        print("\n── Crispness gate ──")
        crispness_log: dict[str, dict] = {}
        crisp_candidates: list[tuple[int, str]] = []
        for u_str, desc in descriptions.items():
            u_local = int(u_str)
            is_crisp, reason = _crispness_judgment(desc, cfg)
            crispness_log[u_str] = {
                "crisp": is_crisp, "reason": reason, "description": desc,
            }
            if is_crisp:
                crisp_candidates.append((u_local, desc))
        (iter_dir / "crispness.json").write_text(json.dumps(crispness_log, indent=2))
        print(
            f"  {len(crisp_candidates)}/{len(descriptions)} descriptions "
            f"passed the crispness gate"
        )
        if len(crisp_candidates) < cfg.promote_min_kept:
            print(
                f"► TERMINATED: only {len(crisp_candidates)} crisp candidates "
                f"(< {cfg.promote_min_kept})."
            )
            history.append({
                "iter": iter_idx,
                "converged_reason": "too_few_crisp",
                "n_crisp": len(crisp_candidates),
            })
            break

        # ── 3b. Compute mean-shift directions for the crisp candidates ──
        crisp_u_indices = [u for u, _ in crisp_candidates]
        kept_u, proposal_dirs, n_pos_list = _compute_mean_shift_dirs(
            sae, crisp_u_indices, x_flat.to(torch.float32),
            n_supervised, d_model, min_n_pos=cfg.promote_min_n_pos,
        )
        if not kept_u:
            print(
                f"► TERMINATED: no crisp candidate has ≥ "
                f"{cfg.promote_min_n_pos} active positions."
            )
            break
        desc_by_u = dict(crisp_candidates)
        proposals = [
            {
                "id": f"promoted.u{u}_r{iter_idx}",
                "description": desc_by_u[u],
                "type": "leaf",
                "parent": None,
                "source_u_local_idx": int(u),
                "n_pos_at_proposal": int(n_pos_list[i]),
                "delta_r2_at_proposal": round(float(ranking_map[u]), 6),
            }
            for i, u in enumerate(kept_u)
        ]

        # ── 4. Merge: cosine + separability against existing catalog ────
        print("\n── Merging into existing catalog ──")
        existing_target_dirs = torch.load(
            cfg.target_dirs_path, weights_only=True,
        ).float()
        merged_catalog, dropped = merge_catalogs_by_direction(
            existing_catalog=catalog,
            existing_target_dirs=existing_target_dirs,
            proposed_features=proposals,
            proposed_dirs=proposal_dirs.cpu().float(),
            cos_threshold=cfg.promote_cos_threshold,
            use_llm_separability=cfg.promote_use_llm_separability,
            cfg=cfg,
        )
        meta = merged_catalog.get("_discovery_metadata", {})
        n_kept = meta.get("n_kept", 0)
        (iter_dir / "dropped.json").write_text(json.dumps(dropped, indent=2))
        print(
            f"  Kept: {n_kept}  |  cosine-dropped: "
            f"{meta.get('n_dropped_cosine', 0)}  |  separability-dropped: "
            f"{meta.get('n_dropped_nonseparable', 0)}"
        )
        if n_kept < cfg.promote_min_kept:
            print(
                f"► TERMINATED: only {n_kept} new features survived merge "
                f"(< {cfg.promote_min_kept})."
            )
            history.append({
                "iter": iter_idx, "converged_reason": "too_few_kept_after_merge",
                "n_kept": n_kept,
            })
            break

        kept_feature_ids = [
            f["id"] for f in merged_catalog["features"]
            if f["id"] not in {g["id"] for g in catalog["features"]}
        ]
        promoted_u_indices_this_round = [
            f["source_u_local_idx"] for f in merged_catalog["features"]
            if f.get("source_u_local_idx") is not None
            and f["id"] in kept_feature_ids
        ]

        # ── 5. Persist merged catalog + invalidate downstream ──────────
        cfg.catalog_path.write_text(json.dumps(
            {"features": merged_catalog["features"]}, indent=2,
        ))
        (iter_dir / "catalog.json").write_text(json.dumps(
            {"features": merged_catalog["features"]}, indent=2,
        ))
        for p in [
            cfg.checkpoint_path, cfg.checkpoint_config_path,
            cfg.target_dirs_path, cfg.split_path, cfg.eval_path,
            cfg.causal_path,
        ]:
            if p.exists():
                p.unlink()

        # Free old SAE before retrain
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── 6. Incremental annotate + retrain + evaluate ───────────────
        print("\n── Incremental annotation (id-keyed) ──")
        run_annotate(cfg)
        print("\n── Retraining supervised SAE ──")
        run_train(cfg)
        print("\n── Evaluating ──")
        run_evaluate(cfg)

        # ── 7. Post-training per-feature validation ────────────────────
        print("\n── Post-training validation (F1 floor) ──")
        kept_ids_after_val, val_dropped = _post_training_validation(
            kept_feature_ids, cfg,
        )
        (iter_dir / "post_training_dropped.json").write_text(
            json.dumps(val_dropped, indent=2)
        )
        if val_dropped:
            print(
                f"  Dropping {len(val_dropped)} features below F1 floor "
                f"{cfg.promote_post_train_f1_floor}:"
            )
            for d in val_dropped[:10]:
                print(f"    - {d['id']}  ({d['metric']}={d['f1']})")
            # Physically remove dropped features from catalog + invalidate
            # downstream so the next round sees a clean slate. Annotations
            # will shrink via id-keyed load on the next `run_annotate`.
            bad = {d["id"] for d in val_dropped}
            pruned_catalog = {"features": [
                f for f in merged_catalog["features"] if f["id"] not in bad
            ]}
            cfg.catalog_path.write_text(json.dumps(pruned_catalog, indent=2))
            for p in [
                cfg.checkpoint_path, cfg.checkpoint_config_path,
                cfg.target_dirs_path, cfg.split_path, cfg.eval_path,
                cfg.causal_path,
            ]:
                if p.exists():
                    p.unlink()
            print("  Retraining on pruned catalog...")
            run_annotate(cfg)
            run_train(cfg)
            run_evaluate(cfg)

        # ── 8. Capacity-transfer verification ──────────────────────────
        print("\n── Capacity-transfer verification ──")
        new_model_cfg = torch.load(
            cfg.checkpoint_config_path, map_location="cpu", weights_only=True,
        )
        new_sae = SupervisedSAE(
            new_model_cfg["d_model"], new_model_cfg["n_supervised"],
            new_model_cfg["n_unsupervised"],
            new_model_cfg.get("n_lista_steps", 0),
        )
        new_sae.load_state_dict(torch.load(
            cfg.checkpoint_path, map_location="cpu", weights_only=True,
        ))
        new_sae = new_sae.to(device).to(model_dtype).eval()
        new_n_supervised = new_sae.n_supervised

        x_val_new = x_flat[val_idx]  # reuse val split (deterministic via cfg.seed)
        transfer_records = _verify_capacity_transfer(
            ranking_map, promoted_u_indices_this_round,
            new_sae, x_val_new, new_n_supervised, device,
            transfer_ratio=cfg.promote_capacity_transfer_ratio,
        )
        (iter_dir / "capacity_transfer.json").write_text(json.dumps(
            transfer_records, indent=2,
        ))
        n_transferred = sum(1 for r in transfer_records if r["transferred"])
        print(
            f"  Capacity transferred: {n_transferred}/{len(transfer_records)} "
            f"promoted latents saw new top-U ΔR² ≤ "
            f"{cfg.promote_capacity_transfer_ratio}× the old value"
        )

        del new_sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── 9. Record round ────────────────────────────────────────────
        eval_data = json.loads(cfg.eval_path.read_text())
        recon = eval_data.get("reconstruction") or {}
        record = {
            "iter": iter_idx,
            "n_candidates_initial": len(candidate_u_indices),
            "n_crisp": len(crisp_candidates),
            "n_merged": n_kept,
            "n_after_post_training_filter": len(kept_ids_after_val),
            "n_capacity_transferred": n_transferred,
            "r2_after_retrain": recon.get("r2"),
            "delta_r2_supervised_after": recon.get("delta_r2_supervised"),
            "kept_ids": kept_ids_after_val,
        }
        history.append(record)
        (iter_dir / "summary.json").write_text(json.dumps(record, indent=2))

        if len(kept_ids_after_val) < cfg.promote_min_kept:
            print(
                f"► TERMINATED: only {len(kept_ids_after_val)} features "
                f"survived post-training validation (< {cfg.promote_min_kept})."
            )
            break

    (loop_dir / "history.json").write_text(json.dumps(history, indent=2))
    print("\n" + "=" * 70)
    print("PROMOTE LOOP COMPLETE")
    print("=" * 70)
    final_catalog = json.loads(cfg.catalog_path.read_text())
    n_leaves = sum(1 for f in final_catalog["features"] if f["type"] == "leaf")
    print(f"  Rounds executed: {len(history)}")
    print(f"  Final catalog:   {n_leaves} leaves")
    print(f"  History:         {loop_dir / 'history.json'}")
    return history
