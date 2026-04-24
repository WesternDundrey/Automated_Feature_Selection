"""Catalog merge with direction-based dedup and LLM separability judgment.

The discovery loop needs: given an existing supervised catalog and a set of
freshly-proposed feature descriptions + their direction proxies, decide which
proposals are genuinely novel and which are rediscoveries of existing features.

Two dedup gates:

  1. **Cosine-direction dedup (cheap).** For each proposed feature, compare its
     direction proxy (unit-normalized) against every existing target_dir. If
     max cosine > threshold (default 0.8), it's geometrically close to an
     existing feature — drop as rediscovery.

  2. **LLM separability judgment (optional).** For proposals that pass cosine,
     ask Sonnet: "given the N closest existing features by description, is
     this proposed description a DIFFERENT concept or the same one reworded?"
     This catches semantic duplicates that happen to have cosine < 0.8 (e.g.,
     two features that describe the same thing from different angles).

Output:
  - merged_catalog: existing_catalog + surviving proposals (monotonic growth)
  - dropped: list of proposals rejected with reason (redundant / nonseparable)

Usage (standalone):
    from pipeline.merge import merge_catalogs_by_direction
    merged, dropped = merge_catalogs_by_direction(
        existing_catalog=..., existing_target_dirs=...,
        proposed_features=..., proposed_dirs=...,
    )
"""

from __future__ import annotations

import json
import textwrap
from typing import Optional

import torch
import torch.nn.functional as F

from .config import Config


def _normalize_dirs(dirs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-normalize a (n, d_model) direction matrix; zero-norm rows stay zero."""
    if dirs.dim() == 1:
        dirs = dirs.unsqueeze(0)
    norms = dirs.norm(dim=-1, keepdim=True).clamp(min=eps)
    out = dirs / norms
    # Zero-out originally-zero rows (don't introduce unit vectors from nothing)
    zero_mask = (dirs.norm(dim=-1) < eps).unsqueeze(-1)
    return torch.where(zero_mask, torch.zeros_like(out), out)


def _closest_existing_by_cosine(
    proposed_dirs: torch.Tensor,        # (n_prop, d_model)
    existing_dirs: torch.Tensor,        # (n_exist, d_model)
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each proposal, return (max_cosine, argmax_index) over existing features."""
    p = _normalize_dirs(proposed_dirs)
    e = _normalize_dirs(existing_dirs)
    sim = p @ e.T                       # (n_prop, n_exist)
    max_cos, argmax = sim.max(dim=1)
    return max_cos, argmax


def _sonnet_separability_judgment(
    proposed_description: str,
    neighbor_descriptions: list[str],
    cfg: Config,
) -> tuple[bool, str]:
    """Ask Sonnet: is this proposal semantically different from its cosine neighbors?

    Returns (is_separable, reason_string). Separability is a YES/NO judgment
    with a short rationale for audit logs.
    """
    from .llm import get_client, chat
    client = get_client()

    neighbors_block = "\n".join(
        f"  {i+1}. {d}" for i, d in enumerate(neighbor_descriptions)
    )
    # This prompt is DELIBERATELY CONSERVATIVE. An overly-permissive prompt
    # (earlier version accepted "partial overlap is OK") resulted in
    # 495/500 proposals being judged separable — which means the gate did
    # no filtering work. The catalog must grow selectively: the default
    # answer is SAME; only clearly-novel concepts should flip to SEPARABLE.
    prompt = textwrap.dedent(f"""\
        You are a strict auditor for a supervised sparse-autoencoder
        feature catalog. A candidate feature has been proposed. Your job
        is to determine whether it is MEANINGFULLY NOVEL relative to the
        existing features, or whether any existing feature already
        substantially covers it.

        DEFAULT ANSWER IS "SAME" (separable: false). Flip to SEPARABLE
        (separable: true) only if you are confident the candidate fires
        on a substantially different set of tokens than every existing
        feature — not merely edge cases, not a narrower or broader
        wording of the same concept.

        Decision rules (apply in order):

        1. Paraphrase test: if the candidate's description can be mapped
           to an existing description by routine rephrasing — e.g.
           "comma tokens" vs "the comma punctuation feature" — answer
           SAME, no exceptions.

        2. Subset test: if the candidate is a strict subset of an
           existing feature (e.g. "serial comma" vs "comma";
           "British place name" vs "place name"), answer SAME. Subsets
           do not warrant their own catalog entry unless you can argue
           the subset has distinct mechanism.

        3. Superset test: if the candidate is a strict superset of an
           existing feature (e.g. "punctuation" vs "comma" when "comma"
           is in the catalog), answer SAME.

        4. Overlap ≥ 70% test: estimate what fraction of positive tokens
           for the candidate would ALSO be positive for at least one
           existing feature. If the estimated overlap is ≥ 70%, answer
           SAME.

        5. Domain test: if the candidate is about text genre, semantic
           domain, or register, it almost certainly overlaps with an
           existing semantic_domain.* or text_genre.* feature — answer
           SAME unless it covers a genre none of the existing ones do.

        6. Surface-feature test: if the candidate is about a common
           punctuation mark, stop-word, or part-of-speech and any
           existing feature already covers that surface, answer SAME.

        Only answer SEPARABLE when the candidate clears ALL six tests
        AND you can name one concrete token context where the candidate
        fires but no existing feature does.

        CANDIDATE feature:
          {proposed_description}

        CLOSEST EXISTING features (ranked by direction cosine; #1 is
        most similar):
        {neighbors_block}

        Reply with EXACTLY one JSON object, no other text:
        {{
          "separable": <true or false>,
          "reason": "<one short sentence. If SAME, cite which existing feature by number and why. If SEPARABLE, describe one concrete token context where candidate fires but no existing feature does.>"
        }}
    """)

    try:
        text = chat(client, cfg.organization_model, prompt, max_tokens=200)
    except Exception as e:
        # FAIL CLOSED: if the LLM call fails, reject the proposal rather than
        # promote it. The gate's job is to catch rediscoveries; defaulting to
        # "separable" on error makes outages or rate-limits auto-accept
        # everything. Previous fail-open behavior let a single Sonnet outage
        # promote hundreds of bogus proposals.
        return False, f"separability LLM unreachable ({type(e).__name__}: {e})"

    # Parse the JSON response
    import re
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return False, f"separability response unparseable: {text[:120]}"
    try:
        obj = json.loads(m.group(0))
        # The prompt explicitly defaults to SAME (separable=False); we mirror
        # that default when the key is missing.
        return bool(obj.get("separable", False)), str(obj.get("reason", ""))
    except json.JSONDecodeError:
        return False, f"separability JSON invalid: {text[:120]}"


def merge_catalogs_by_direction(
    existing_catalog: dict,
    existing_target_dirs: torch.Tensor,      # (n_existing_sup, d_model)
    proposed_features: list[dict],           # each: {id, description, type, parent, ...}
    proposed_dirs: torch.Tensor,             # (n_proposed, d_model) — decoder-column proxies
    cos_threshold: float = 0.8,
    use_llm_separability: bool = True,
    n_neighbors_for_llm: int = 5,
    cfg: Optional[Config] = None,
) -> tuple[dict, list[dict]]:
    """Merge proposals into the existing catalog with two-gate dedup.

    Args:
        existing_catalog: JSON catalog dict ({'features': [...]}); all fields
            preserved as-is in the output.
        existing_target_dirs: (n_sup, d_model). Must correspond 1:1 with the
            LEAF features of existing_catalog in the same order; groups have
            no target_dir and are skipped for the cosine comparison.
        proposed_features: list of proposal dicts; each needs 'id',
            'description', 'type' (leaf), 'parent' (optional).
        proposed_dirs: (n_proposed, d_model) direction proxies (e.g., decoder
            columns of an unsupervised SAE, or mean-shift directions computed
            from preliminary labels). Will be unit-normalized inside.
        cos_threshold: max cosine above which a proposal is rejected as a
            rediscovery. 0.8 is the default from the north-star plan.
        use_llm_separability: if True, for proposals that pass cosine, also
            ask Sonnet whether they are semantically distinct from their
            cosine neighbors. If False, rely on cosine alone.
        n_neighbors_for_llm: how many nearest existing features to show the
            LLM per separability check.
        cfg: pipeline Config (needed only if use_llm_separability=True).

    Returns:
        merged_catalog: existing + surviving proposals (groups from proposals
            are carried through if referenced by a surviving leaf).
        dropped: list of {id, description, reason, max_cos,
                         closest_existing_id, ...} for audit.
    """
    # target_dirs has one row per supervised feature in catalog order —
    # BOTH leaves AND groups (train.py computes a target_dir per feature
    # column, regardless of type). Shape must match total features count.
    existing_features = existing_catalog["features"]
    n_existing_features = len(existing_features)
    n_existing_leaves = sum(
        1 for f in existing_features if f.get("type") == "leaf"
    )
    n_existing_groups = sum(
        1 for f in existing_features if f.get("type") == "group"
    )

    if n_existing_features != existing_target_dirs.shape[0]:
        raise ValueError(
            f"existing_target_dirs has {existing_target_dirs.shape[0]} rows but "
            f"existing_catalog has {n_existing_features} features "
            f"({n_existing_leaves} leaves + {n_existing_groups} groups). "
            f"target_dirs is expected to have one row per feature in catalog "
            f"order (leaves AND groups), matching train.compute_target_directions."
        )

    if len(proposed_features) != proposed_dirs.shape[0]:
        raise ValueError(
            f"proposed_features ({len(proposed_features)}) does not match "
            f"proposed_dirs ({proposed_dirs.shape[0]}) row count."
        )

    # Existing feature IDs for downstream conflict checks.
    existing_ids = {f["id"] for f in existing_features}

    # Cosine-based dedup: for each proposal, find nearest existing feature
    # (may be a leaf OR a group). Comparing against groups is still
    # meaningful — if a proposal's direction matches a group (an OR of
    # children), the concept is likely already covered.
    max_cos, argmax = _closest_existing_by_cosine(
        proposed_dirs, existing_target_dirs,
    )

    # Diagnostic: print distribution of max cosines so we can tell
    # whether the cosine gate is doing any work. An early bug used
    # decoder columns (writing directions) against target_dirs (reading
    # directions), which produced a distribution centered near 0 and
    # rejected nothing. Encoder rows are the right proxy. The spread
    # should show some mass at 0.3-0.7 and some at > cos_threshold.
    _mc = max_cos.cpu().float()
    print(f"  Max-cosine distribution over {len(_mc)} proposals:")
    print(
        f"    min={_mc.min():.3f}  median={_mc.median():.3f}  "
        f"mean={_mc.mean():.3f}  max={_mc.max():.3f}"
    )
    if len(_mc) >= 10:
        q = torch.quantile(
            _mc, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),
        )
        print(
            f"    quantiles: 10%={q[0]:.3f}  25%={q[1]:.3f}  "
            f"50%={q[2]:.3f}  75%={q[3]:.3f}  90%={q[4]:.3f}  "
            f"95%={q[5]:.3f}  99%={q[6]:.3f}"
        )
    print(
        f"    at cos_threshold={cos_threshold}: "
        f"{int((_mc > cos_threshold).sum())}/{len(_mc)} "
        f"proposals would be cosine-dropped"
    )

    # For LLM separability, pre-compute the top-k nearest neighbors per proposal.
    if use_llm_separability:
        p = _normalize_dirs(proposed_dirs)
        e = _normalize_dirs(existing_target_dirs)
        sim = p @ e.T                               # (n_prop, n_existing_features)
        topk = min(n_neighbors_for_llm, n_existing_features)
        _, neighbors_idx = sim.topk(topk, dim=1)    # (n_prop, topk)
    else:
        neighbors_idx = None

    kept: list[dict] = []
    dropped: list[dict] = []

    for i, feat in enumerate(proposed_features):
        cos_i = float(max_cos[i].item())
        close_idx = int(argmax[i].item())
        close_feat = existing_features[close_idx]

        # Gate 1: cosine dedup
        if cos_i > cos_threshold:
            dropped.append({
                "id": feat.get("id", f"proposal_{i}"),
                "description": feat.get("description", ""),
                "reason": f"redundant_direction (cos={cos_i:.3f} > {cos_threshold})",
                "closest_existing_id": close_feat["id"],
                "closest_existing_type": close_feat.get("type", ""),
                "max_cos": round(cos_i, 4),
            })
            continue

        # Gate 2: LLM separability on survivors
        if use_llm_separability and cfg is not None:
            neighbor_descs = [
                existing_features[int(j)]["description"]
                for j in neighbors_idx[i].tolist()
            ]
            separable, reason = _sonnet_separability_judgment(
                feat.get("description", ""), neighbor_descs, cfg,
            )
            if not separable:
                dropped.append({
                    "id": feat.get("id", f"proposal_{i}"),
                    "description": feat.get("description", ""),
                    "reason": f"nonseparable_semantic ({reason})",
                    "closest_existing_id": close_feat["id"],
                    "closest_existing_type": close_feat.get("type", ""),
                    "max_cos": round(cos_i, 4),
                })
                continue

        # Gate 3: id conflict with existing catalog — auto-rename if collision.
        fid = feat.get("id", f"discovered.auto_{i}")
        if fid in existing_ids:
            base = fid
            k = 2
            while f"{base}_r{k}" in existing_ids:
                k += 1
            feat = {**feat, "id": f"{base}_r{k}"}
            fid = feat["id"]
        existing_ids.add(fid)

        kept.append(feat)

    # Build merged catalog. Groups from proposals are pulled in if any surviving
    # leaf references them as parent.
    kept_parents = {f.get("parent") for f in kept if f.get("parent")}
    merged_features = list(existing_features)
    # Add proposal groups first (if not already in existing)
    for feat in proposed_features:
        if feat.get("type") == "group" and feat["id"] in kept_parents:
            if feat["id"] not in existing_ids:
                merged_features.append(feat)
                existing_ids.add(feat["id"])
    # Add surviving proposal leaves
    merged_features.extend(kept)

    merged_catalog = {
        **existing_catalog,
        "features": merged_features,
        "_discovery_metadata": {
            "n_existing_before_merge": n_existing_features,
            "n_existing_leaves_before_merge": n_existing_leaves,
            "n_existing_groups_before_merge": n_existing_groups,
            "n_proposed": len(proposed_features),
            "n_kept": len(kept),
            "n_dropped_cosine": sum(
                1 for d in dropped if "redundant_direction" in d["reason"]
            ),
            "n_dropped_nonseparable": sum(
                1 for d in dropped if "nonseparable_semantic" in d["reason"]
            ),
            "cos_threshold": cos_threshold,
            "use_llm_separability": use_llm_separability,
        },
    }

    return merged_catalog, dropped


def load_target_dirs(cfg: Config) -> torch.Tensor:
    """Load existing target_dirs tensor from pipeline_data/target_directions.pt."""
    if not cfg.target_dirs_path.exists():
        raise FileNotFoundError(
            f"target_directions.pt not found at {cfg.target_dirs_path}. "
            f"Run `python -m pipeline.run --step train` first."
        )
    return torch.load(cfg.target_dirs_path, weights_only=True).float()
