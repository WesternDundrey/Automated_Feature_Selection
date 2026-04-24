"""Discovery loop — iterative supervised-SAE catalog growth.

DEPRECATED in v8.1 in favor of `pipeline/promote_loop.py`.

Design flaw: this loop re-trains a fresh unsupervised SAE on the RAW cached
activations every round with the same `cfg.seed`, so later rounds re-propose
essentially the same latents. The only thing that changes per round is what
the merge gate dedups against. `promote_loop.py` instead uses the U slice of
the ALREADY-TRAINED supervised SAE as the proposal pool, ranks U latents by
residual contribution (ΔR² under ablation), promotes high-ranked ones to S,
and verifies capacity transfer after retrain. That is the right design.

This file is retained for reproducibility of earlier runs only — new
experiments should use `--step promote-loop`.

Round N of the (deprecated) loop:
  1. Load current state (catalog, target_dirs, evaluation)
  2. Train a fresh unsupervised SAE on cached activations
  3. Describe its latents via Sonnet (reuses inventory.explain_features)
  4. Use each unsup latent's decoder column as a direction proxy
  5. Merge proposals into the current catalog:
       Gate A: cosine dedup against existing target_dirs (threshold 0.8)
       Gate B: Sonnet separability judgment vs nearest existing features
  6. Incrementally annotate only the surviving new features
  7. Retrain supervised SAE on the grown catalog
  8. Re-evaluate (R², F1, per-feature metrics)
  9. Terminate if (a) < K new features survived the merge OR (b) ΔR² < ε

The loop is monotonic — discovered features are never removed once added.
This is what "discoveries aren't taken away" means operationally.

Round-N artifacts live in pipeline_data/discover_loop/round_N/:
  - unsupervised_sae.pt     — the round's unsup SAE (resumable)
  - top_activations.json    — top examples per selected unsup latent
  - descriptions.json       — Sonnet descriptions of those latents
  - dropped.json            — merge rejections with reason per proposal
  - catalog.json            — snapshot of the catalog after this round's merge
  - summary.json            — round stats (kept/dropped/ΔR²)

Usage:
    python -m pipeline.run --step discover-loop \
        --layer 9 --sae_id "blocks.9.hook_resid_pre" \
        --discover-loop-max-iters 5 \
        --discover-loop-min-new 3 \
        --discover-loop-min-delta-r2 0.005
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import Config


# ── Helpers ───────────────────────────────────────────────────────────────

def _read_r2(cfg: Config) -> float | None:
    """Read `reconstruction.r2` from evaluation.json. evaluate.py stores
    it under key "r2" (not "r_squared" — a naming inconsistency worth
    noting; kept as-is to avoid churn across the codebase)."""
    if not cfg.eval_path.exists():
        return None
    try:
        data = json.loads(cfg.eval_path.read_text())
        return data.get("reconstruction", {}).get("r2")
    except (json.JSONDecodeError, KeyError):
        return None


def _count_leaves(catalog: dict) -> int:
    return sum(1 for f in catalog["features"] if f.get("type") == "leaf")


def _invalidate_downstream(cfg: Config) -> None:
    """Remove artifacts that must be recomputed after a catalog change.

    Keeps: tokens.pt, activations.pt (corpus-level, catalog-independent),
           feature_catalog.json (just written with merged contents),
           annotations.pt (handled incrementally by annotate.py).
    """
    for p in [
        cfg.checkpoint_path,
        cfg.checkpoint_config_path,
        cfg.target_dirs_path,
        cfg.split_path,
        cfg.eval_path,
        cfg.causal_path,
    ]:
        if p.exists():
            p.unlink()


def _propose_via_unsup_sae(cfg: Config, iter_dir: Path):
    """Train a fresh unsupervised SAE on cached activations, describe its
    latents via Sonnet, return a list of (feature_dict, direction_tensor).

    Direction proxy: unit-normalized decoder column of the unsup SAE latent.
    This is what the merge function uses for cosine-based dedup against the
    existing supervised target_dirs.
    """
    from .discover import UnsupervisedSAE, train_unsupervised_sae, compute_firing_rates
    from .inventory import (
        collect_top_activations, PretrainedSAE, load_target_model, explain_features,
    )

    activations = torch.load(cfg.activations_path, weights_only=True)
    _, _, d_model = activations.shape
    d_sae = 8 * d_model

    unsup_path = iter_dir / "unsupervised_sae.pt"
    if unsup_path.exists():
        print(f"  Loading cached unsup SAE: {unsup_path}")
        sae = UnsupervisedSAE(d_model, d_sae)
        sae.load_state_dict(
            torch.load(unsup_path, map_location="cpu", weights_only=True)
        )
    else:
        print(f"  Training unsupervised SAE (d_sae={d_sae})...")
        sae = train_unsupervised_sae(activations, cfg, d_sae)
        torch.save(sae.state_dict(), unsup_path)

    # Select latents by firing rate (same window as inventory)
    firing_rates = compute_firing_rates(sae, activations, cfg)
    mask = (firing_rates >= cfg.min_firing_rate) & (firing_rates <= cfg.max_firing_rate)
    candidates = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    candidates.sort(key=lambda i: -firing_rates[i].item())
    selected = candidates[: cfg.n_latents_to_explain]

    if not selected:
        print("  No latents passed the firing-rate window — returning empty proposal set.")
        return [], torch.zeros(0, d_model)

    print(f"  Selected {len(selected)} unsup latents by firing rate")

    # Top activations (streaming through target model + SAE encoder)
    top_acts_path = iter_dir / "top_activations.json"
    if top_acts_path.exists():
        top_acts = json.loads(top_acts_path.read_text())
    else:
        model = load_target_model(cfg)
        tokenizer = model.tokenizer
        wrapped = PretrainedSAE(
            W_enc=sae.encoder.weight.T.contiguous(),
            W_dec=sae.decoder.weight.T.contiguous(),
            b_enc=sae.encoder.bias.data.clone(),
            b_dec=torch.zeros(d_model),
            threshold=None,
        )
        top_acts = collect_top_activations(
            model, wrapped, tokenizer, selected, cfg,
        )
        top_acts_path.write_text(json.dumps(top_acts, indent=2))
        del model

    # Sonnet descriptions
    desc_path = iter_dir / "descriptions.json"
    if desc_path.exists():
        descriptions = json.loads(desc_path.read_text())
    else:
        from transformers import AutoTokenizer
        tokenizer_ref = AutoTokenizer.from_pretrained(cfg.model_name)
        descriptions = explain_features(top_acts, tokenizer_ref, cfg)
        desc_path.write_text(json.dumps(descriptions, indent=2))

    # Direction proxy for merge-time dedup:
    #
    # The supervised target_dir is a MEAN-SHIFT direction:
    #     target_dir_i = normalize(mean(x | label_i = 1) - mean(x))
    #
    # To get apples-to-apples cosines against it, we use the SAME formula
    # for each unsup latent, substituting its firing mask for labels:
    #     d_k = normalize(mean(x | latent_k fires) - mean(x))
    #
    # This is the third proxy attempt. The first two failed:
    #
    # - Decoder column (writing direction): decoder columns can be
    #   arbitrarily aligned, so cosines were near 0 even for clearly
    #   redundant latents.
    #
    # - Encoder row (reading direction): cosines came out ~1/sqrt(d_model)
    #   = 0.036 on layer 9 (d=768), indistinguishable from random
    #   directions. Encoder rows are separator-hyperplane normals learned
    #   by gradient descent; two methods can perfectly capture the same
    #   concept while having encoder rows that are nearly orthogonal.
    #   The linear direction that maximally SEPARATES a class is not the
    #   same geometric object as the mean-shift BETWEEN the class means.
    #
    # Mean-shift lives entirely in activation space, is a function of the
    # firing mask only, and is architecture-agnostic — so it's the only
    # proxy that can be meaningfully compared across supervision regimes.
    N_tot = activations.shape[0] * activations.shape[1]
    X = activations.reshape(N_tot, d_model).float()
    mean_all = X.mean(dim=0)

    selected_to_pos = {l: i for i, l in enumerate(selected)}
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    W_enc_sel = sae.encoder.weight[selected_tensor, :].detach().float()
    b_enc_sel = sae.encoder.bias[selected_tensor].detach().float()

    # Firing masks in chunks to cap peak memory.
    chunk = 16384
    active_mask = torch.zeros(N_tot, len(selected), dtype=torch.bool)
    for start in range(0, N_tot, chunk):
        end = min(start + chunk, N_tot)
        pre = X[start:end] @ W_enc_sel.T + b_enc_sel
        active_mask[start:end] = (pre > 0)

    proposal_features = []
    proposal_dirs_list = []
    n_too_sparse = 0
    for latent_idx_str, desc in sorted(descriptions.items(), key=lambda kv: int(kv[0])):
        latent_idx = int(latent_idx_str)
        sel_pos = selected_to_pos.get(latent_idx)
        if sel_pos is None:
            continue
        mask = active_mask[:, sel_pos]
        n_pos = int(mask.sum())
        if n_pos < 10:
            n_too_sparse += 1
            continue
        mean_pos = X[mask].mean(dim=0)
        d = mean_pos - mean_all
        d = d / (d.norm() + 1e-8)
        proposal_dirs_list.append(d)
        proposal_features.append({
            "id": f"discovered.unsup_{latent_idx}",
            "description": desc,
            "type": "leaf",
            "parent": None,
        })
    if n_too_sparse:
        print(f"  Skipped {n_too_sparse} latents with <10 active positions")
    print(
        f"  Computed mean-shift directions for {len(proposal_features)} "
        f"proposals (matches supervised target_dir formula)"
    )

    proposal_dirs = (
        torch.stack(proposal_dirs_list) if proposal_dirs_list
        else torch.zeros(0, d_model)
    )
    return proposal_features, proposal_dirs


# ── Main driver ───────────────────────────────────────────────────────────

def run(
    cfg: Config | None = None,
    max_iters: int = 5,
    min_new_features: int = 3,
    min_delta_r2: float = 0.005,
    use_llm_separability: bool = True,
    cos_threshold: float = 0.8,
) -> list[dict]:
    """Run the iterative discovery loop.

    Requires an existing catalog + trained SAE + evaluation (i.e., the
    normal pipeline must have been run at least once). Each round grows
    the catalog and retrains; the loop terminates when no novel features
    survive the merge or R² stops improving.

    Returns the per-round history (list of dicts) and writes it to
    `pipeline_data/discover_loop/history.json`.
    """
    if cfg is None:
        cfg = Config()

    # Sanity-check round-0 state.
    required = [
        (cfg.catalog_path, "feature_catalog.json"),
        (cfg.target_dirs_path, "target_directions.pt"),
        (cfg.tokens_path, "tokens.pt"),
        (cfg.activations_path, "activations.pt"),
        (cfg.annotations_path, "annotations.pt"),
    ]
    for path, name in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Discovery loop requires {name} at {path}. Run the full "
                f"pipeline (`python -m pipeline.run`) at least once before "
                f"invoking --step discover-loop."
            )

    loop_dir = cfg.output_dir / "discover_loop"
    loop_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []

    from .merge import merge_catalogs_by_direction
    # Deferred imports for the round loop — avoid cold-load when not needed.
    # Note: evaluate.py exposes its entry point as `evaluate`, not `run`.
    from .annotate import run as run_annotate
    from .train import run as run_train
    from .evaluate import evaluate as run_evaluate

    for iter_idx in range(max_iters):
        iter_dir = loop_dir / f"round_{iter_idx:02d}"
        iter_dir.mkdir(exist_ok=True)

        print("\n" + "=" * 70)
        print(f"DISCOVERY LOOP — ROUND {iter_idx}")
        print("=" * 70)

        current_catalog = json.loads(cfg.catalog_path.read_text())
        current_target_dirs = torch.load(
            cfg.target_dirs_path, weights_only=True,
        ).float()
        current_leaves = _count_leaves(current_catalog)
        current_r2 = _read_r2(cfg)

        print(
            f"Current catalog: {current_leaves} leaves, "
            f"target_dirs shape {tuple(current_target_dirs.shape)}, "
            f"R²={current_r2}"
        )

        # 1. Propose new features from a fresh unsup SAE + Sonnet descriptions
        print("\n── Proposing new features from unsupervised SAE ──")
        proposals, proposal_dirs = _propose_via_unsup_sae(cfg, iter_dir)
        n_proposed = len(proposals)
        print(f"  Sonnet produced {n_proposed} candidate descriptions")

        # 2. Merge (cosine + optional LLM separability)
        print("\n── Merging into existing catalog ──")
        merged_catalog, dropped = merge_catalogs_by_direction(
            existing_catalog=current_catalog,
            existing_target_dirs=current_target_dirs,
            proposed_features=proposals,
            proposed_dirs=proposal_dirs,
            cos_threshold=cos_threshold,
            use_llm_separability=use_llm_separability,
            cfg=cfg,
        )
        meta = merged_catalog.get("_discovery_metadata", {})
        n_kept = meta.get("n_kept", 0)
        n_dropped_cosine = meta.get("n_dropped_cosine", 0)
        n_dropped_ns = meta.get("n_dropped_nonseparable", 0)
        print(
            f"  Kept: {n_kept}  |  "
            f"cosine-dropped: {n_dropped_cosine}  |  "
            f"separability-dropped: {n_dropped_ns}"
        )
        (iter_dir / "dropped.json").write_text(json.dumps(dropped, indent=2))

        # 3. Termination check BEFORE retraining (saves a wasted round)
        if n_kept < min_new_features:
            record = {
                "iter": iter_idx,
                "n_existing": current_leaves,
                "n_proposed": n_proposed,
                "n_kept": n_kept,
                "n_dropped_cosine": n_dropped_cosine,
                "n_dropped_nonseparable": n_dropped_ns,
                "r2_before": current_r2,
                "r2_after": None,
                "delta_r2": None,
                "converged_reason": "insufficient_novel_features",
                "dropped_sample": dropped[:10],
            }
            history.append(record)
            (iter_dir / "summary.json").write_text(json.dumps(record, indent=2))
            print(
                f"\n► TERMINATED: only {n_kept} new features survived merge "
                f"(< {min_new_features} threshold)."
            )
            break

        # 4. Persist merged catalog, invalidate downstream state
        cfg.catalog_path.write_text(json.dumps(
            {"features": merged_catalog["features"]}, indent=2,
        ))
        (iter_dir / "catalog.json").write_text(json.dumps(
            {"features": merged_catalog["features"]}, indent=2,
        ))
        _invalidate_downstream(cfg)

        # 5. Incrementally annotate the new features (annotate.py detects
        #    that the cached tensor has fewer features than the catalog
        #    and only runs vLLM on the new leaves).
        print("\n── Incremental annotation ──")
        run_annotate(cfg)

        # 6. Retrain supervised SAE on the grown catalog
        print("\n── Retraining supervised SAE on grown catalog ──")
        run_train(cfg)

        # 7. Re-evaluate
        print("\n── Re-evaluating ──")
        run_evaluate(cfg)

        # 8. Record round outcomes
        new_r2 = _read_r2(cfg)
        delta_r2 = (
            new_r2 - current_r2
            if (current_r2 is not None and new_r2 is not None) else None
        )
        print(
            f"\n  R² {current_r2} → {new_r2}  "
            f"(Δ = {delta_r2:+.4f})" if delta_r2 is not None else ""
        )

        record = {
            "iter": iter_idx,
            "n_existing": current_leaves,
            "n_proposed": n_proposed,
            "n_kept": n_kept,
            "n_dropped_cosine": n_dropped_cosine,
            "n_dropped_nonseparable": n_dropped_ns,
            "r2_before": current_r2,
            "r2_after": new_r2,
            "delta_r2": delta_r2,
            "dropped_sample": dropped[:10],
        }
        history.append(record)
        (iter_dir / "summary.json").write_text(json.dumps(record, indent=2))

        # 9. Termination: ΔR² below threshold
        if delta_r2 is not None and delta_r2 < min_delta_r2:
            record["converged_reason"] = "insufficient_delta_r2"
            (iter_dir / "summary.json").write_text(json.dumps(record, indent=2))
            print(
                f"\n► TERMINATED: ΔR² = {delta_r2:+.4f} < {min_delta_r2} threshold."
            )
            break

    # Loop-level summary
    history_path = loop_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))

    print("\n" + "=" * 70)
    print("DISCOVERY LOOP COMPLETE")
    print("=" * 70)
    final_catalog = json.loads(cfg.catalog_path.read_text())
    final_leaves = _count_leaves(final_catalog)
    print(f"  Rounds executed: {len(history)}")
    print(f"  Final catalog: {final_leaves} leaves "
          f"(started with {history[0]['n_existing'] if history else '?'})")
    print(f"  History: {history_path}")
    return history


if __name__ == "__main__":
    run()
