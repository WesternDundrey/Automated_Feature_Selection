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
        cfg.promote_capacity_transfer_ratio = 0.5
    if not hasattr(cfg, "promote_max_iters"):
        cfg.promote_max_iters = 5
    if not hasattr(cfg, "promote_min_kept"):
        cfg.promote_min_kept = 3        # terminate if < this survive a round
    if not hasattr(cfg, "promote_cos_threshold"):
        cfg.promote_cos_threshold = 0.6
    if not hasattr(cfg, "promote_use_llm_separability"):
        cfg.promote_use_llm_separability = True
    # Mini-prefilter: cheap annotator-vs-U-activation agreement check on a
    # small subset of sequences. We use AUROC of U's pre-activation score
    # against the annotator's binary labels — NOT F1 against the (pre > 0)
    # firing mask. The `pre > 0` mask treats every weak firing as positive,
    # which polysemantic/leaky U latents produce in bulk; F1 against that
    # then collapses to "the annotator agrees U fires everywhere". AUROC
    # uses the continuous activation ranking and is scale/threshold free.
    # Features with too few positives (annotator or U) on the subset are
    # routed to "audit" rather than dropped — a sparse mini-sample can't
    # distinguish a rare-but-real feature from a broken one.
    if not hasattr(cfg, "promote_mini_prefilter"):
        cfg.promote_mini_prefilter = True
    if not hasattr(cfg, "promote_mini_prefilter_n_seqs"):
        cfg.promote_mini_prefilter_n_seqs = 50
    if not hasattr(cfg, "promote_mini_prefilter_min_auroc"):
        cfg.promote_mini_prefilter_min_auroc = 0.70
    if not hasattr(cfg, "promote_mini_prefilter_min_support"):
        # minimum positives on BOTH sides (annotator AND U) for the feature
        # to be eligible for dropping; below this, route to audit.
        cfg.promote_mini_prefilter_min_support = 5
    if not hasattr(cfg, "promote_mini_prefilter_audit_only"):
        # audit-only mode: compute the score but don't drop. Recommended
        # for first few runs to calibrate the AUROC threshold.
        cfg.promote_mini_prefilter_audit_only = False
    # Legacy knob, retained so CLI flag --promote-mini-prefilter-min-f1
    # still sets something harmless if someone passes it.
    if not hasattr(cfg, "promote_mini_prefilter_min_f1"):
        cfg.promote_mini_prefilter_min_f1 = 0.20

    # Adaptive proposal budget. Instead of "describe top-20, stop if fewer
    # than min_kept crisp", we pull the next batch of top-ΔR² U latents
    # until either min_kept crisp descriptions accumulate or the total
    # describe count hits the budget. The top few ΔR² latents tend to be
    # high-variance artifacts (position-anomalies, token-surface
    # detectors); genuine crisp features often live further down the
    # ranking.
    if not hasattr(cfg, "promote_proposal_budget"):
        cfg.promote_proposal_budget = 100
    if not hasattr(cfg, "promote_batch_size"):
        cfg.promote_batch_size = 20  # describe this many at a time

    # Nuisance prefilter: reject U latents whose top activations are
    # token-degenerate (all firing on the same 1-2 token IDs, or
    # dominated by whitespace / EOT / repeated separators) BEFORE spending
    # Sonnet on a description. Cheap defensive check — complements the
    # position-0 mask already applied upstream at
    # inventory.collect_top_activations.
    if not hasattr(cfg, "promote_nuisance_token_diversity"):
        # Min fraction of DISTINCT token IDs among top-K activating tokens.
        # 0.3 = at least 30% of top-K tokens are different tokens.
        cfg.promote_nuisance_token_diversity = 0.30

    # Multi-concept decomposition: when the crispness gate rejects a U
    # description as `multi_concept`, ask Sonnet to split it into 2-5
    # atomic token-level feature hypotheses, validate each via
    # crispness + within-round semantic dedup + mini-annotation, and
    # compute atom-specific target_dirs from the mini labels (NOT from
    # the source U's firing mask — the source U is precisely the bundle
    # we're decomposing, so its mask is not evidence for any one atom).
    if not hasattr(cfg, "promote_decompose_multi_concept"):
        cfg.promote_decompose_multi_concept = True
    if not hasattr(cfg, "promote_decompose_max_atoms"):
        cfg.promote_decompose_max_atoms = 5
    if not hasattr(cfg, "promote_atom_mini_min_pos"):
        # minimum n_pos on the mini-annotation subset for an atom to
        # receive an atom-specific target_dir and pass to merge.
        cfg.promote_atom_mini_min_pos = 3
    if not hasattr(cfg, "promote_multi_concept_warn_rate"):
        # per-round diagnostic: warn if this fraction of described
        # candidates are rejected as multi_concept. Does NOT change
        # behavior (decomposition is always on for multi_concept);
        # just surfaces when the U slice is dominated by bundles.
        cfg.promote_multi_concept_warn_rate = 0.70

    # Triage-only: run the adaptive describe + crispness loop, optionally
    # decomposition, then STOP before merge/annotate/retrain. Used by the
    # n_unsupervised sweep orchestrator to measure candidate quality
    # cheaply across different U widths without paying for the full
    # retraining cycle each time.
    if not hasattr(cfg, "promote_triage_only"):
        cfg.promote_triage_only = False


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


# ── Nuisance prefilter: reject degenerate U latents before LLM description ──

NUISANCE_REASONS = {
    "single_token_dominant",  # top-k all fire on 1-2 distinct token ids
    "no_examples",            # collect_top_activations returned nothing
}


def _nuisance_check(
    top_activations_for_latent: list[dict], cfg: Config,
) -> tuple[bool, str]:
    """Cheap triage on top-k activating contexts for a U latent. Returns
    (is_nuisance, reason_if_nuisance).

    Dropping nuisances here saves the Sonnet description call and the
    crispness call for latents whose top activations are obviously
    degenerate. Complements the position-0 mask at collect time.
    """
    if not top_activations_for_latent:
        return True, "no_examples"

    # Token-diversity: if all top-k examples activate on the same 1-2
    # token IDs, the latent is a token-surface detector rather than a
    # contextual feature. 74-feature catalogs already name a lot of
    # surface features (punctuation.*, token_form.*, part_of_speech.*),
    # so these are usually rediscoveries.
    tokens = [ex.get("context_ids", [])[ex.get("pos", 0)]
              for ex in top_activations_for_latent
              if ex.get("context_ids") and ex.get("pos") is not None]
    if tokens:
        diversity = len(set(tokens)) / len(tokens)
        if diversity < cfg.promote_nuisance_token_diversity:
            return True, (
                f"single_token_dominant "
                f"(unique={len(set(tokens))}/{len(tokens)}, "
                f"threshold={cfg.promote_nuisance_token_diversity})"
            )

    return False, ""


# ── Crispness gate (is this a single nameable concept?) ────────────────────

# Fixed taxonomy for crispness rejections so the per-round summary can
# surface what's actually failing (vague, multi-concept, nuisance, too-broad,
# not-token-local) instead of an opaque "19/20 rejected".
CRISPNESS_CATEGORIES = (
    "crisp",              # kept
    "multi_concept",      # description names >1 concept
    "vague",              # fuzzy / doesn't operationalize firing condition
    "too_broad",          # describes a whole register/genre, not a token-local property
    "not_token_local",    # about context / sequence, not the specific token
    "uninterpretable",    # description itself is incoherent
    "nuisance",           # catches what nuisance_check missed
    "unknown",            # LLM didn't categorize
    "llm_error",          # unreachable / unparseable — fails closed
)


def _crispness_judgment(description: str, cfg: Config) -> tuple[bool, str, str]:
    """Ask Sonnet whether a description is a single operationally-testable
    concept vs a grab-bag. Returns (is_crisp, reason, category).

    Fails CLOSED: any LLM error / unparseable response rejects the proposal
    rather than accepting it, matching merge.py's gate policy.
    """
    from .llm import get_client, chat
    client = get_client()

    categories_list = ", ".join(f'"{c}"' for c in CRISPNESS_CATEGORIES if c != "crisp")

    prompt = textwrap.dedent(f"""\
        You are a strict auditor deciding whether a candidate feature
        description is usable as supervision for a supervised sparse
        autoencoder latent.

        A crisp description names ONE operationally-testable concept. A
        reader should be able to look at a single token in context and
        answer yes/no without ambiguity.

        A non-crisp description fails for one of these specific reasons:
          - multi_concept   : names 2+ concepts joined by "or"/"and"/"sometimes"
          - vague           : doesn't specify WHEN the latent fires
          - too_broad       : describes a genre/register/domain, not a token
          - not_token_local : about surrounding context, not the specific token
          - uninterpretable : description itself is incoherent
          - nuisance        : describes degenerate patterns (whitespace,
                              padding, single-token artifacts, boilerplate)

        CANDIDATE description:
          {description}

        Reply with EXACTLY one JSON object, no other text:
        {{
          "crisp": <true or false>,
          "category": <one of: "crisp", {categories_list}>,
          "reason": "<one short sentence>"
        }}
    """)

    try:
        text = chat(client, cfg.organization_model, prompt, max_tokens=200)
    except Exception as e:
        return False, f"crispness LLM unreachable ({type(e).__name__}: {e})", "llm_error"

    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return False, f"crispness response unparseable: {text[:120]}", "llm_error"
    try:
        obj = json.loads(m.group(0))
        crisp = bool(obj.get("crisp", False))
        reason = str(obj.get("reason", ""))
        category = str(obj.get("category", "")).strip().lower()
        if category in CRISPNESS_CATEGORIES:
            return crisp, reason, category
        # Sonnet sometimes omits the category field even when asked. When
        # crisp=true, the category is obviously "crisp"; when crisp=false,
        # try to recover the category from the reason text before falling
        # back to "unknown". The modal failure mode in our runs is
        # multi_concept, so if we can't identify anything else, default to
        # that rather than "unknown" — this makes the breakdown tell the
        # user which bucket they're actually in instead of hiding it.
        if crisp:
            return True, reason, "crisp"
        reason_lc = reason.lower()
        for cat in CRISPNESS_CATEGORIES:
            if cat == "crisp" or cat == "unknown" or cat == "llm_error":
                continue
            # Look for exact category token or common synonyms
            needles = [cat, cat.replace("_", " "), cat.replace("_", "-")]
            if cat == "multi_concept":
                needles += ["multiple concepts", "bundled", "several"]
            elif cat == "vague":
                needles += ["unclear", "fuzzy", "ambiguous"]
            elif cat == "too_broad":
                needles += ["overly broad", "genre", "register"]
            elif cat == "not_token_local":
                needles += ["context-level", "sequence-level", "not about the token"]
            if any(n in reason_lc for n in needles):
                return False, reason, cat
        # Still nothing identified — default to multi_concept (the dominant
        # cause across runs) rather than "unknown", so the breakdown is
        # actionable.
        return False, reason, "multi_concept"
    except json.JSONDecodeError:
        return False, f"crispness JSON invalid: {text[:120]}", "llm_error"


# ── Multi-concept decomposition: bundle → atomic hypotheses ──────────────

def _decompose_multi_concept(
    description: str, top_contexts: list[dict], tokenizer, cfg: Config,
    max_atoms: int,
) -> list[str]:
    """Ask Sonnet to split a multi-concept description into 2-5 atomic
    token-level feature hypotheses. Returns the list of atomic description
    strings (may be fewer than max_atoms if Sonnet thinks there's less).

    Fails closed (returns empty list) on LLM error or parse failure.
    """
    from .llm import get_client, chat
    client = get_client()

    # Build a short snippet of the top contexts so Sonnet can ground the
    # decomposition in what the source latent actually fires on.
    snippets = []
    for ex in top_contexts[:8]:
        ctx_ids = ex.get("context_ids") or []
        pos = ex.get("pos", 0)
        try:
            decoded = tokenizer.decode(ctx_ids)
            marked_tok = tokenizer.decode([ctx_ids[pos]]) if 0 <= pos < len(ctx_ids) else ""
            snippets.append(f"  • [{marked_tok!r}] in: {decoded[:120]}")
        except Exception:
            continue
    snippets_block = "\n".join(snippets) if snippets else "  (no contexts available)"

    prompt = textwrap.dedent(f"""\
        A sparse-autoencoder latent was described as a MULTI-CONCEPT
        bundle (two or more distinct features in one latent). Your job
        is to propose 2-{max_atoms} ATOMIC token-level feature hypotheses
        that might each be a sub-component of the bundle.

        Each atomic hypothesis must:
          - Name ONE operationally-testable property of a single token.
          - Be a yes/no question a reader can answer for one token in
            context without ambiguity.
          - NOT mix multiple concepts with "or" / "and" / "sometimes".

        ORIGINAL MULTI-CONCEPT DESCRIPTION:
          {description}

        TOP-ACTIVATING EXAMPLES (the marked token is in []):
        {snippets_block}

        Return EXACTLY one JSON object, no other text:
        {{
          "atoms": [
            "atomic description 1",
            "atomic description 2",
            ...
          ]
        }}
        Return 2-{max_atoms} atoms. If the original is actually crisp
        (not multi-concept after all), return a single-element list
        with the cleaned-up wording.
    """)

    try:
        text = chat(client, cfg.organization_model, prompt, max_tokens=400)
    except Exception as e:
        print(f"    decompose LLM unreachable: {type(e).__name__}: {e}")
        return []

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        atoms = obj.get("atoms") or []
        return [str(a).strip() for a in atoms if str(a).strip()][:max_atoms]
    except json.JSONDecodeError:
        return []


def _semantic_dedup_atoms(
    atoms: list[dict], cfg: Config,
) -> tuple[list[dict], list[dict]]:
    """Group atoms that describe the same concept using a single Sonnet
    pass. Returns (kept, dropped) atom lists; 'kept' has one
    representative per group.

    Each atom dict must have 'atom_id' (stable identifier) and
    'description'. Dropped atoms get a 'dedup_merged_into' field.
    """
    if len(atoms) <= 1:
        return atoms, []

    from .llm import get_client, chat
    client = get_client()

    listing = "\n".join(
        f"  {a['atom_id']}: {a['description']}" for a in atoms
    )
    prompt = textwrap.dedent(f"""\
        Group the following atomic feature descriptions by semantic
        equivalence. Two atoms belong to the SAME group if a reader
        would answer yes/no identically for every token in context.

        Atoms:
        {listing}

        Return EXACTLY one JSON object, no other text:
        {{
          "groups": [
            ["atom_id_a", "atom_id_b"],   # group of duplicates
            ["atom_id_c"],                # singleton group
            ...
          ]
        }}
        Every atom_id must appear in exactly one group.
    """)

    try:
        text = chat(client, cfg.organization_model, prompt, max_tokens=500)
    except Exception as e:
        print(f"    semantic_dedup LLM unreachable: {type(e).__name__}: {e}")
        # Fallback: keep all atoms (no dedup). Merge can still drop them
        # via cosine + separability later.
        return atoms, []

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return atoms, []
    try:
        obj = json.loads(m.group(0))
        groups = obj.get("groups") or []
    except json.JSONDecodeError:
        return atoms, []

    by_id = {a["atom_id"]: a for a in atoms}
    kept: list[dict] = []
    dropped: list[dict] = []
    for group in groups:
        ids_in_group = [gid for gid in group if gid in by_id]
        if not ids_in_group:
            continue
        rep_id = ids_in_group[0]
        kept.append(by_id[rep_id])
        for gid in ids_in_group[1:]:
            dup = dict(by_id[gid])
            dup["dedup_merged_into"] = rep_id
            dropped.append(dup)
    # Any atoms Sonnet forgot to place in a group stay as singletons.
    grouped_ids = {gid for group in groups for gid in group}
    for a in atoms:
        if a["atom_id"] not in grouped_ids:
            kept.append(a)
    return kept, dropped


def _atom_target_dirs_from_labels(
    atom_labels: torch.Tensor,   # (n_seqs, T, n_atoms) boolean/float
    activations_sub: torch.Tensor,   # (n_seqs, T, d_model) — mini subset
    d_model: int,
    min_n_pos: int,
) -> tuple[list[int], torch.Tensor, list[int]]:
    """For each atom, compute mean-shift direction from its mini-annotation
    labels on the mini subset:
        d_atom = normalize(mean(x | atom_label=1) - mean(x))
    Returns (kept_atom_indices, dirs, n_pos_per_atom).

    This replaces "use source U's firing mask" — atoms need atom-specific
    evidence, not the bundle's.
    """
    X = activations_sub.reshape(-1, d_model).float()
    Y = atom_labels.reshape(-1, atom_labels.shape[-1]).bool()
    mean_all = X.mean(dim=0)

    kept_indices: list[int] = []
    dirs: list[torch.Tensor] = []
    n_pos_list: list[int] = []
    for ai in range(Y.shape[1]):
        mask = Y[:, ai]
        n_pos = int(mask.sum().item())
        if n_pos < min_n_pos:
            continue
        mean_pos = X[mask].mean(dim=0)
        d = mean_pos - mean_all
        norm = d.norm()
        if norm < 1e-8:
            continue
        kept_indices.append(ai)
        dirs.append(d / norm)
        n_pos_list.append(n_pos)
    if not dirs:
        return [], torch.zeros(0, d_model), []
    return kept_indices, torch.stack(dirs), n_pos_list


# ── Mini-annotation prefilter (cheap agreement check before full corpus) ──

def _mini_prefilter(
    sae, new_features: list[dict],
    tokens: torch.Tensor, activations: torch.Tensor,
    tokenizer, cfg: Config, d_model: int, n_supervised: int,
) -> tuple[list[str], list[dict], list[dict]]:
    """Rank-based agreement check between annotator labels and the source U
    latent's continuous activation score. AUROC(pre, ann) is scale-free and
    doesn't punish polysemantic U latents for firing weakly everywhere.

    Drops are conservative:
      - Features with too few positives (annotator or U-fires) on the subset
        land in an AUDIT bucket (kept + logged, not dropped).
      - Features with AUROC below `promote_mini_prefilter_min_auroc` AND
        enough support are dropped.
      - In audit-only mode (`promote_mini_prefilter_audit_only=True`),
        nothing is dropped — scores are only logged for threshold
        calibration across runs.

    Subset is drawn DETERMINISTICALLY at random from the full corpus using
    `cfg.seed + 7919` so results are reproducible but not biased by whatever
    documents happen to sit at the start of the ingest order.

    Returns (kept_feature_ids, dropped_records, audit_records).
    """
    n_total = int(tokens.shape[0])
    n_seqs = min(cfg.promote_mini_prefilter_n_seqs, n_total)
    if n_seqs < 5 or not new_features:
        return [f["id"] for f in new_features], [], []

    # Deterministic random subset (not the first N — openwebtext's ingest
    # ordering is not uniform across content types).
    rng = np.random.RandomState(cfg.seed + 7919)
    sampled = np.sort(rng.choice(n_total, size=n_seqs, replace=False))
    sampled_t = torch.from_numpy(sampled).long()
    tokens_sub = tokens[sampled_t]
    acts_sub = activations[sampled_t]

    from .annotate import annotate_local, annotate_corpus
    print(
        f"  Mini-annotating {len(new_features)} candidates on {n_seqs} "
        f"randomly-sampled sequences "
        f"({'local vLLM' if cfg.use_local_annotator else 'API'})..."
    )
    if cfg.use_local_annotator:
        mini_labels = annotate_local(tokens_sub, new_features, tokenizer, cfg)
    else:
        mini_labels = annotate_corpus(tokens_sub, new_features, tokenizer, cfg)
    # mini_labels: (n_seqs, T, len(new_features))

    # U activation on the same subset. Align device/dtype.
    X_sub = acts_sub.reshape(-1, d_model).to(torch.float32)
    enc_w = sae.encoder.weight.detach().to(device=X_sub.device, dtype=X_sub.dtype)
    enc_b = sae.encoder.bias.detach().to(device=X_sub.device, dtype=X_sub.dtype)

    # Local copy of the AUROC function from evaluate.py to avoid a circular
    # import on the test path.
    def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        order = np.argsort(-scores)
        y_sorted = y_true[order]
        tps = np.cumsum(y_sorted)
        fps = np.cumsum(~y_sorted)
        tpr = tps / n_pos
        fpr = fps / n_neg
        tpr = np.concatenate([[0.0], tpr])
        fpr = np.concatenate([[0.0], fpr])
        return float(np.trapz(tpr, fpr))

    kept_ids: list[str] = []
    dropped: list[dict] = []
    audit: list[dict] = []
    min_support = cfg.promote_mini_prefilter_min_support

    for i, feat in enumerate(new_features):
        u_local = feat.get("source_u_local_idx")
        if u_local is None:
            kept_ids.append(feat["id"])
            continue
        u_global = n_supervised + u_local
        pre = (X_sub @ enc_w[u_global] + enc_b[u_global]).cpu().numpy()  # float
        ann = mini_labels[:, :, i].reshape(-1).numpy().astype(bool)
        u_fires_mask = pre > 0

        n_ann = int(ann.sum())
        n_fires = int(u_fires_mask.sum())
        # Legacy F1 retained as a secondary diagnostic (easy to compare
        # against the v8.3 gate), but no longer used for the drop decision.
        tp = int((u_fires_mask & ann).sum())
        prec = tp / n_ann if n_ann > 0 else 0.0
        rec = tp / n_fires if n_fires > 0 else 0.0
        mini_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        auroc_val = _auroc(ann, pre)
        record = {
            "id": feat["id"],
            "auroc": (
                round(float(auroc_val), 4)
                if not np.isnan(auroc_val) else None
            ),
            "mini_f1_legacy": round(mini_f1, 4),
            "n_annotator_positives": n_ann,
            "n_u_fires": n_fires,
            "n_subset_positions": int(len(ann)),
            "floor": cfg.promote_mini_prefilter_min_auroc,
        }

        # Audit path: not enough support, OR audit-only mode is on.
        low_support = (
            n_ann < min_support or n_fires < min_support
            or np.isnan(auroc_val)
        )
        if cfg.promote_mini_prefilter_audit_only or low_support:
            reason = (
                "audit_only_mode"
                if cfg.promote_mini_prefilter_audit_only
                else f"low_support (n_ann={n_ann}, n_fires={n_fires})"
            )
            audit.append({**record, "reason": reason})
            kept_ids.append(feat["id"])
            continue

        if auroc_val < cfg.promote_mini_prefilter_min_auroc:
            dropped.append(record)
        else:
            kept_ids.append(feat["id"])
    return kept_ids, dropped, audit


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
    X = activations_flat
    mean_all = X.mean(dim=0)

    # Align encoder weight device + dtype with X. sae was moved to GPU +
    # model dtype earlier; X is typically CPU fp32 straight from the cached
    # activations tensor. Without device alignment, `X @ w` raises
    # RuntimeError on mixed-device tensors.
    enc_w = sae.encoder.weight.detach().to(device=X.device, dtype=X.dtype)
    enc_b = sae.encoder.bias.detach().to(device=X.device, dtype=X.dtype)

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

    Metric preference, best → worst:
      1. `val_promo_f1` (v8.4+): F1 on the held-out half of val at the
         threshold picked on the other half. Honest generalization within
         val — the only metric that neither contaminates test nor is
         fit-and-scored on the same data.
      2. `val_f1_cal` (v8.3): F1 on val at threshold fit on val. Overfit
         to val by construction; retained only as a fallback when val-
         promo n_pos is 0 (rare features with no positives in the second
         half).
      3. `f1` (t=0): legacy fallback for evaluation.json files written
         before v8.3 had any val-side metrics.

    CRITICAL: must NEVER read `cal_f1` — that's test-set F1 at val-
    calibrated thresholds, and using it for multi-round promotion filtering
    leaks test labels into the pruned catalog.
    """
    if not cfg.eval_path.exists():
        return new_feature_ids, []
    eval_data = json.loads(cfg.eval_path.read_text())
    feats_map = {f["id"]: f for f in eval_data.get("features") or []}

    kept: list[str] = []
    dropped: list[dict] = []
    for fid in new_feature_ids:
        rec = feats_map.get(fid) or {}
        f1 = rec.get("val_promo_f1")
        metric = "val_promo_f1"
        # If val_promo has no positives for this feature (rare or newly
        # added with unlucky split), fall back to val_f1_cal.
        if f1 is None or rec.get("val_promo_n_pos", 0) == 0:
            f1 = rec.get("val_f1_cal")
            metric = "val_f1_cal_fallback"
        if f1 is None:
            # Pre-v8.3 evaluation.json without any val-side metrics.
            f1 = rec.get("f1")
            metric = "f1_t0_legacy_fallback"
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
    promoted_u_indices: list[int],      # surviving promoted U latents (post-validation)
    new_sae, x_val: torch.Tensor, new_n_supervised: int, device: str,
    transfer_ratio: float,
) -> dict:
    """Aggregate capacity-transfer check.

    U latent indices are NOT stable across retrains — the new SAE's
    unsupervised slice has freshly initialized + trained encoders, and
    there's no canonical mapping between old U[k] and new U[k]. A
    per-latent `new_u_delta_r2 vs old_u_delta_r2` pairing would therefore
    be meaningless.

    The honest, aggregate signal is:

      - `old_promoted_sum` = total ΔR² the promoted old U latents carried
        on val before we promoted them.
      - `new_top_k_sum`    = total ΔR² of the top-K U latents in the
        retrained SAE's U slice. These are whatever concepts U is still
        carrying after some of its capacity (we hope) moved to S.

    If U actually gave up the promoted capacity, `new_top_k_sum` should
    be less than the OLD top-K sum by at least `transfer_ratio ×
    old_promoted_sum`. Otherwise U retained the capacity and S merely
    duplicates it.
    """
    k = len(promoted_u_indices)
    if k == 0:
        return {"k": 0, "transferred": False, "note": "no promoted latents"}

    old_promoted_sum = float(
        sum(old_ranking.get(u, 0.0) for u in promoted_u_indices)
    )
    old_top_k = sorted(old_ranking.values(), reverse=True)[:k]
    old_top_k_sum = float(sum(old_top_k))

    new_ranking = rank_u_latents_by_delta_r2(
        new_sae, x_val, new_n_supervised, device,
    )
    new_top_k_sum = float(sum(dr2 for _, dr2 in new_ranking[:k]))

    # Expected new top-K sum under the hypothesis that ratio*old_promoted_sum
    # of capacity moved from U to S. If the new top-K is at or below this,
    # transfer is consistent with the data.
    expected_new = old_top_k_sum - transfer_ratio * old_promoted_sum
    transferred = new_top_k_sum <= expected_new

    return {
        "k": k,
        "old_promoted_delta_r2_sum": round(old_promoted_sum, 6),
        "old_top_k_delta_r2_sum": round(old_top_k_sum, 6),
        "new_top_k_delta_r2_sum": round(new_top_k_sum, 6),
        "expected_new_top_k_if_transferred": round(expected_new, 6),
        "transferred": bool(transferred),
        "fractional_capacity_drop": (
            round(1.0 - new_top_k_sum / old_top_k_sum, 4)
            if old_top_k_sum > 1e-12 else None
        ),
        "note": "distribution-level aggregate; U indices are not "
                "stable across retrains so per-latent pairings would "
                "be meaningless.",
    }


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
        # Mask BOS/position-0 — otherwise the top-ΔR² U latents are dominated
        # by BOS-detector latents (position 0 has degenerate attention and
        # anomalous residual magnitude, so any latent that fires there wins
        # the ΔR² ranking). Must match the masking used in train+eval.
        from .position_mask import mask_leading
        activations = mask_leading(activations, cfg=cfg)
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
        # Full filtered ranking — we'll pull from it adaptively.
        all_candidates = [
            (u, dr2) for u, dr2 in ranking
            if dr2 >= cfg.promote_min_delta_r2_u
        ]
        print(
            f"  {len(all_candidates)} U latents above ΔR² threshold "
            f"{cfg.promote_min_delta_r2_u} "
            f"(budget: {cfg.promote_proposal_budget}, "
            f"batch: {cfg.promote_batch_size})"
        )
        for u, dr2 in all_candidates[:10]:
            print(f"    U[{u}]  ΔR² = {dr2:+.5f}")
        if len(all_candidates) < cfg.promote_min_kept:
            print(
                f"► TERMINATED: only {len(all_candidates)} candidates above "
                f"ΔR² threshold (< {cfg.promote_min_kept})."
            )
            history.append({
                "iter": iter_idx,
                "converged_reason": "too_few_candidates",
                "n_candidates": len(all_candidates),
            })
            break

        # ── 2. Adaptive describe + triage ──────────────────────────────
        # Pull top-ΔR² U latents in batches of `promote_batch_size`,
        # describe + triage each batch, stop once we have >= min_kept
        # crisp candidates or we've spent the proposal budget. This
        # prevents the single-shot top-20 pattern from terminating the
        # loop when the top few latents happen to be high-variance
        # artifacts (token-surface detectors, position anomalies).
        top_acts_path = iter_dir / "top_activations.json"
        descriptions_path = iter_dir / "descriptions.json"
        nuisance_path = iter_dir / "ignored_nuisance.json"
        crispness_path = iter_dir / "crispness.json"

        # Resume state: if any of these files exist, reuse them.
        all_top_acts: dict = (
            json.loads(top_acts_path.read_text()) if top_acts_path.exists() else {}
        )
        all_descriptions: dict = (
            json.loads(descriptions_path.read_text())
            if descriptions_path.exists() else {}
        )
        nuisance_log: dict = (
            json.loads(nuisance_path.read_text()) if nuisance_path.exists() else {}
        )
        crispness_log: dict = (
            json.loads(crispness_path.read_text()) if crispness_path.exists() else {}
        )

        crisp_candidates: list[tuple[int, str]] = [
            (int(u), rec["description"])
            for u, rec in crispness_log.items() if rec.get("crisp")
        ]
        spent = len(all_descriptions) + len(nuisance_log)
        already_processed = set(
            list(all_descriptions.keys())
            + list(nuisance_log.keys())
            + list(crispness_log.keys())
        )

        from transformers import AutoTokenizer
        tokenizer_ref = AutoTokenizer.from_pretrained(cfg.model_name)

        # ── Collect top activations for all budget candidates in ONE scan.
        # Previous version re-scanned 2M tokens per batch of 20, wasting
        # ~80% of wall-clock per round. We pick the whole budget of U
        # latents up front (capped at the ranking length) and collect
        # their top activations in a single pass; adaptive describe +
        # crispness then reads from this cache batch-by-batch.
        budget_latents = [
            u for u, _ in all_candidates[: cfg.promote_proposal_budget]
            if str(u) not in already_processed
        ]
        missing = [u for u in budget_latents if str(u) not in all_top_acts]
        if missing:
            print(f"\n  ── Collecting top activations for {len(missing)} "
                  f"candidates in a single scan (cache reuses "
                  f"{len(budget_latents) - len(missing)}) ──")
            model = load_target_model(cfg)
            tokenizer = model.tokenizer
            wrapped_u = _wrap_u_slice_as_pretrained(sae, n_supervised, d_model)
            fresh = collect_top_activations(
                model, wrapped_u, tokenizer, missing, cfg,
            )
            all_top_acts.update(fresh)
            top_acts_path.write_text(json.dumps(all_top_acts, indent=2))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"\n  ── All {len(budget_latents)} candidate top_activations "
                  f"cached; no scan needed ──")

        # ── Adaptive describe + crispness loop. No more model scans here;
        #    only Sonnet API calls (cheap relative to scanning).
        while (
            len(crisp_candidates) < cfg.promote_min_kept
            and spent < cfg.promote_proposal_budget
        ):
            # Pull the next batch of candidates we haven't already looked at.
            next_batch = []
            for u in budget_latents:
                if str(u) in already_processed:
                    continue
                next_batch.append(u)
                if len(next_batch) >= cfg.promote_batch_size:
                    break
            if not next_batch:
                print("  exhausted ΔR² candidate ranking")
                break

            print(f"\n  ── Batch: {len(next_batch)} new candidates "
                  f"(spent {spent}/{cfg.promote_proposal_budget}, "
                  f"{len(crisp_candidates)} crisp so far) ──")

            # Nuisance prefilter (cheap, no API cost).
            survivors_for_description: list[int] = []
            for u in next_batch:
                u_str = str(u)
                is_nuis, nreason = _nuisance_check(
                    all_top_acts.get(u_str, []), cfg,
                )
                if is_nuis:
                    nuisance_log[u_str] = {"reason": nreason}
                    already_processed.add(u_str)
                    spent += 1
                else:
                    survivors_for_description.append(u)
            nuisance_path.write_text(json.dumps(nuisance_log, indent=2))
            if not survivors_for_description:
                print(f"    entire batch of {len(next_batch)} latents failed "
                      f"nuisance prefilter; pulling next batch")
                continue

            # Sonnet descriptions for survivors only.
            sub_top_acts = {
                str(u): all_top_acts[str(u)] for u in survivors_for_description
                if str(u) in all_top_acts
            }
            batch_descs = explain_features(sub_top_acts, tokenizer_ref, cfg)
            all_descriptions.update(batch_descs)
            descriptions_path.write_text(json.dumps(all_descriptions, indent=2))

            # Crispness gate with rejection taxonomy.
            for u_str, desc in batch_descs.items():
                u_local = int(u_str)
                is_crisp, reason, category = _crispness_judgment(desc, cfg)
                crispness_log[u_str] = {
                    "crisp": is_crisp,
                    "category": category,
                    "reason": reason,
                    "description": desc,
                }
                if is_crisp:
                    crisp_candidates.append((u_local, desc))
                already_processed.add(u_str)
                spent += 1
            crispness_path.write_text(json.dumps(crispness_log, indent=2))

        # ── 3. Summary of adaptive triage ──────────────────────────────
        # Distribution of WHY things failed — surfaces real bottleneck.
        cat_counts: dict[str, int] = {}
        for rec in crispness_log.values():
            cat = rec.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        n_nuisance = len(nuisance_log)
        n_described = len(all_descriptions)
        n_crisp = len(crisp_candidates)
        print(
            f"\n  Adaptive triage: spent={spent}/{cfg.promote_proposal_budget}  "
            f"nuisance-dropped={n_nuisance}  described={n_described}  "
            f"crisp={n_crisp}"
        )
        if cat_counts:
            print("  crispness breakdown:")
            for cat in ("crisp", "multi_concept", "vague", "too_broad",
                        "not_token_local", "uninterpretable", "nuisance",
                        "unknown", "llm_error"):
                if cat in cat_counts:
                    print(f"    {cat:<18} {cat_counts[cat]}")

        # Diagnostic warning for high multi_concept rate (reviewer's #5).
        described = max(n_described, 1)
        mc_rate = cat_counts.get("multi_concept", 0) / described
        if mc_rate > cfg.promote_multi_concept_warn_rate:
            print(
                f"  WARNING: multi_concept_rate = {mc_rate:.0%} (> "
                f"{cfg.promote_multi_concept_warn_rate:.0%}) — "
                f"the U slice at this layer is dominated by bundled "
                f"latents. Decomposition mines these for atomic features."
            )

        # ── 3b. Source-U crisp proposals get mean-shift target_dirs from
        # source U's firing mask. Atoms (from decomposition below) get
        # atom-specific target_dirs from mini-annotation labels.
        proposals: list[dict] = []
        proposal_dirs_list: list[torch.Tensor] = []

        if crisp_candidates:
            crisp_u_indices = [u for u, _ in crisp_candidates]
            kept_u, source_u_dirs, n_pos_list = _compute_mean_shift_dirs(
                sae, crisp_u_indices, x_flat.to(torch.float32),
                n_supervised, d_model, min_n_pos=cfg.promote_min_n_pos,
            )
            desc_by_u = dict(crisp_candidates)
            for i, u in enumerate(kept_u):
                proposals.append({
                    "id": f"promoted.u{u}_r{iter_idx}",
                    "description": desc_by_u[u],
                    "type": "leaf",
                    "parent": None,
                    "source_u_local_idx": int(u),
                    "source_kind": "u_latent_crisp",
                    "n_pos_at_proposal": int(n_pos_list[i]),
                    "delta_r2_at_proposal": round(float(ranking_map[u]), 6),
                })
                proposal_dirs_list.append(source_u_dirs[i])

        # ── 3c. DECOMPOSITION PATH: rescue multi_concept rejections ──
        # For each multi_concept rejection: ask Sonnet to split into
        # atomic hypotheses → crispness on each atom → dedup within-round
        # → mini-annotate atoms (NEW labels, not source U firing) →
        # compute atom-specific target_dirs from those labels → merge.
        # This is the reviewer's design: source U's firing mask is NOT
        # evidence for any single atom of a bundle, so atoms need their
        # own target_dirs computed from real annotator labels on a mini
        # subset.
        multi_concept_u = [
            (int(u), rec["description"])
            for u, rec in crispness_log.items()
            if (not rec.get("crisp"))
            and rec.get("category") == "multi_concept"
        ]
        atom_log: dict = {}
        atom_dedup_log: dict = {}
        atom_proposals: list[dict] = []
        atom_dirs_list: list[torch.Tensor] = []

        if cfg.promote_decompose_multi_concept and multi_concept_u:
            print(f"\n── Decomposition path: {len(multi_concept_u)} "
                  f"multi_concept rejections → atoms ──")

            # 1. Decompose each multi_concept into atoms.
            decompose_tokenizer = tokenizer_ref  # reuse
            all_atoms_raw: list[dict] = []
            for u, desc in multi_concept_u:
                top_ctx = all_top_acts.get(str(u), [])
                atom_descs = _decompose_multi_concept(
                    desc, top_ctx, decompose_tokenizer, cfg,
                    max_atoms=cfg.promote_decompose_max_atoms,
                )
                for ai, a_desc in enumerate(atom_descs):
                    all_atoms_raw.append({
                        "atom_id": f"u{u}_a{ai}",
                        "source_u": int(u),
                        "description": a_desc,
                    })
            print(f"  decomposed into {len(all_atoms_raw)} raw atomic "
                  f"hypotheses")

            # 2. Crispness gate on each atom.
            crisp_atoms: list[dict] = []
            atom_cat_counts: dict[str, int] = {}
            for atom in all_atoms_raw:
                is_crisp, reason, cat = _crispness_judgment(
                    atom["description"], cfg,
                )
                atom["crisp"] = is_crisp
                atom["crisp_category"] = cat
                atom["crisp_reason"] = reason
                atom_log[atom["atom_id"]] = atom
                atom_cat_counts[cat] = atom_cat_counts.get(cat, 0) + 1
                if is_crisp:
                    crisp_atoms.append(atom)
            (iter_dir / "atoms.json").write_text(json.dumps(atom_log, indent=2))
            print(f"  crispness on atoms: {len(crisp_atoms)}/{len(all_atoms_raw)} "
                  f"passed (" + ", ".join(
                      f"{k}={v}" for k, v in sorted(atom_cat_counts.items())
                  ) + ")")

            # 3. Within-round semantic dedup.
            if crisp_atoms:
                kept_atoms, dedup_dropped = _semantic_dedup_atoms(
                    crisp_atoms, cfg,
                )
                atom_dedup_log = {
                    "kept": [a["atom_id"] for a in kept_atoms],
                    "dropped": dedup_dropped,
                }
                (iter_dir / "atoms_dedup.json").write_text(
                    json.dumps(atom_dedup_log, indent=2)
                )
                print(f"  semantic dedup: {len(kept_atoms)} kept, "
                      f"{len(dedup_dropped)} duplicates merged")
            else:
                kept_atoms = []

            # 4-6. Mini-annotate atoms on a random subset, compute
            # atom-specific target_dirs from real labels.
            if kept_atoms:
                # Build the features list shape annotate_local expects.
                atom_as_feats = [
                    {
                        "id": a["atom_id"], "description": a["description"],
                        "type": "leaf", "parent": None,
                    }
                    for a in kept_atoms
                ]
                n_seqs = min(
                    cfg.promote_mini_prefilter_n_seqs, activations.shape[0],
                )
                rng = np.random.RandomState(cfg.seed + 9001)
                sampled = np.sort(
                    rng.choice(activations.shape[0], size=n_seqs, replace=False)
                )
                sampled_t = torch.from_numpy(sampled).long()
                tokens_full = torch.load(cfg.tokens_path, weights_only=True)
                tokens_full = mask_leading(tokens_full, cfg=cfg)
                mini_tokens = tokens_full[sampled_t]
                mini_acts = activations[sampled_t]

                print(f"  mini-annotating {len(atom_as_feats)} atoms on "
                      f"{n_seqs} sequences (for atom target_dirs)...")
                from .annotate import annotate_local, annotate_corpus
                if cfg.use_local_annotator:
                    atom_labels = annotate_local(
                        mini_tokens, atom_as_feats, decompose_tokenizer, cfg,
                    )
                else:
                    atom_labels = annotate_corpus(
                        mini_tokens, atom_as_feats, decompose_tokenizer, cfg,
                    )
                # atom_labels: (n_seqs, T, len(kept_atoms))

                kept_indices, atom_target_dirs, atom_n_pos = (
                    _atom_target_dirs_from_labels(
                        atom_labels, mini_acts, d_model,
                        min_n_pos=cfg.promote_atom_mini_min_pos,
                    )
                )
                print(
                    f"  atom target_dirs: {len(kept_indices)}/"
                    f"{len(kept_atoms)} atoms have ≥ "
                    f"{cfg.promote_atom_mini_min_pos} mini-positives"
                )

                for idx_in_kept, atom_idx in enumerate(kept_indices):
                    a = kept_atoms[atom_idx]
                    atom_proposals.append({
                        "id": f"promoted.{a['atom_id']}_r{iter_idx}",
                        "description": a["description"],
                        "type": "leaf",
                        "parent": None,
                        "source_u_local_idx": int(a["source_u"]),
                        "source_kind": "decomposed_atom",
                        "n_pos_at_mini": int(atom_n_pos[idx_in_kept]),
                    })
                    atom_dirs_list.append(atom_target_dirs[idx_in_kept])

                if atom_proposals:
                    proposals.extend(atom_proposals)
                    proposal_dirs_list.extend(atom_dirs_list)

        # Triage-only: bail before merge so the sweep orchestrator can
        # measure candidate quality without paying for annotate+retrain.
        if cfg.promote_triage_only:
            print(
                f"\n► triage-only mode: stopping after triage "
                f"(crisp={n_crisp}, atom_proposals={len(atom_proposals)}, "
                f"spent={spent})."
            )
            history.append({
                "iter": iter_idx,
                "converged_reason": "triage_only",
                "n_crisp": n_crisp,
                "n_atom_proposals": len(atom_proposals),
                "spent": spent,
                "n_nuisance_dropped": n_nuisance,
                "crispness_breakdown": cat_counts,
                "multi_concept_rate": (
                    cat_counts.get("multi_concept", 0) / max(n_described, 1)
                ),
            })
            break

        # Combined proposal set: source-U crisp + decomposed atoms.
        if not proposals:
            print(
                f"\n► TERMINATED: 0 proposals after describe + "
                f"decomposition (crisp={n_crisp}, "
                f"atoms_with_dirs={len(atom_proposals)})."
            )
            history.append({
                "iter": iter_idx,
                "converged_reason": "no_proposals",
                "n_crisp": n_crisp,
                "n_atom_proposals": len(atom_proposals),
                "spent": spent,
                "n_nuisance_dropped": n_nuisance,
                "crispness_breakdown": cat_counts,
            })
            break

        proposal_dirs = torch.stack(proposal_dirs_list)
        print(
            f"\n  Proposals heading into merge: {len(proposals)} "
            f"(source-U crisp: {sum(1 for p in proposals if p['source_kind'] == 'u_latent_crisp')}, "
            f"decomposed atoms: {len(atom_proposals)})"
        )

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

        # ── 4b. Mini-annotation prefilter ──────────────────────────────
        # Cheap agreement check: annotate each new feature on N seqs,
        # compare against source U latent's firing mask, drop mismatches.
        # Saves the expensive full-annotation + retrain cost on features
        # the annotator can't articulate.
        mini_dropped: list[dict] = []
        mini_audit: list[dict] = []
        if cfg.promote_mini_prefilter and kept_feature_ids:
            print("\n── Mini-annotation prefilter (AUROC-based) ──")
            new_feats_for_prefilter = [
                f for f in merged_catalog["features"]
                if f["id"] in set(kept_feature_ids)
            ]
            from transformers import AutoTokenizer
            mini_tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            # tokens + activations are masked here so the prefilter
            # AUROC lines up with the masked training distribution.
            mp_tokens = torch.load(cfg.tokens_path, weights_only=True)
            mp_tokens = mask_leading(mp_tokens, cfg=cfg)
            kept_feature_ids, mini_dropped, mini_audit = _mini_prefilter(
                sae, new_feats_for_prefilter,
                tokens=mp_tokens,
                activations=activations,
                tokenizer=mini_tokenizer,
                cfg=cfg, d_model=d_model, n_supervised=n_supervised,
            )
            (iter_dir / "mini_prefilter_dropped.json").write_text(
                json.dumps(mini_dropped, indent=2)
            )
            (iter_dir / "mini_prefilter_audit.json").write_text(
                json.dumps(mini_audit, indent=2)
            )
            if mini_dropped:
                print(
                    f"  Mini-prefilter dropped {len(mini_dropped)} feature(s) "
                    f"below AUROC={cfg.promote_mini_prefilter_min_auroc}:"
                )
                for d in mini_dropped[:10]:
                    print(f"    - {d['id']}  AUROC={d['auroc']}  "
                          f"(n_ann={d['n_annotator_positives']}, "
                          f"n_fires={d['n_u_fires']})")
            if mini_audit:
                print(
                    f"  Mini-prefilter routed {len(mini_audit)} to audit "
                    f"(low support or audit-only mode); kept in catalog."
                )
            print(f"  {len(kept_feature_ids)} surviving after mini-prefilter")
            # Prune merged_catalog to the survivors (preserve original features,
            # drop only rejected new features).
            surviving_ids = set(kept_feature_ids)
            original_ids = {g["id"] for g in catalog["features"]}
            merged_catalog = {
                "features": [
                    f for f in merged_catalog["features"]
                    if (f["id"] in original_ids) or (f["id"] in surviving_ids)
                ],
                "_discovery_metadata": merged_catalog.get("_discovery_metadata", {}),
            }

            if len(kept_feature_ids) < cfg.promote_min_kept:
                print(
                    f"► TERMINATED: only {len(kept_feature_ids)} features "
                    f"survived mini-prefilter (< {cfg.promote_min_kept})."
                )
                history.append({
                    "iter": iter_idx,
                    "converged_reason": "too_few_after_mini_prefilter",
                    "n_kept_after_mini": len(kept_feature_ids),
                })
                break

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

        # IMPORTANT: capture promoted-U indices AFTER post-training validation
        # so dropped features don't enter the capacity-transfer check.
        post_val_ids = set(kept_ids_after_val)
        promoted_u_indices_this_round = [
            f["source_u_local_idx"]
            for f in merged_catalog["features"]
            if f.get("source_u_local_idx") is not None
            and f["id"] in post_val_ids
        ]

        # ── 8. Capacity-transfer verification (aggregate, not per-latent)
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
        transfer_summary = _verify_capacity_transfer(
            ranking_map, promoted_u_indices_this_round,
            new_sae, x_val_new, new_n_supervised, device,
            transfer_ratio=cfg.promote_capacity_transfer_ratio,
        )
        (iter_dir / "capacity_transfer.json").write_text(json.dumps(
            transfer_summary, indent=2,
        ))
        if transfer_summary.get("k", 0) > 0:
            frac = transfer_summary.get("fractional_capacity_drop")
            transferred_flag = transfer_summary.get("transferred")
            print(
                f"  Capacity transfer (aggregate, k={transfer_summary['k']}): "
                f"old_top_k_ΔR²_sum={transfer_summary['old_top_k_delta_r2_sum']:.4f} "
                f"→ new_top_k_ΔR²_sum={transfer_summary['new_top_k_delta_r2_sum']:.4f} "
                f"[{'transferred' if transferred_flag else 'NOT transferred'}, "
                f"fractional_drop={frac}]"
            )
        else:
            print("  No promoted latents survived validation — skipping.")

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
            "n_after_mini_prefilter": len(kept_feature_ids),
            "n_after_post_training_filter": len(kept_ids_after_val),
            "mini_prefilter_dropped": [d["id"] for d in mini_dropped],
            "mini_prefilter_audited": [a["id"] for a in mini_audit],
            "capacity_transfer": transfer_summary,
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
