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
    prompt = textwrap.dedent(f"""\
        You are auditing a feature catalog for a supervised sparse autoencoder.
        A candidate feature has been proposed. Determine whether it represents
        a GENUINELY DIFFERENT concept from the existing features below, or
        whether it is a rewording of one of them.

        Guidance:
          • DIFFERENT means the candidate would fire on distinct tokens/contexts
            than any existing feature — not merely a stylistic rephrasing.
          • Partial overlap is OK as long as there exist tokens/contexts where
            the candidate and each existing feature diverge.
          • If the candidate is a strict subset of an existing feature (e.g.,
            "British comma" vs "comma"), judge as SAME.
          • If the candidate is a strict superset, judge as SAME unless the
            broader form adds meaningful coverage the existing one misses.

        CANDIDATE feature:
          {proposed_description}

        CLOSEST EXISTING features (by direction cosine):
        {neighbors_block}

        Reply with EXACTLY one JSON object, no other text:
        {{
          "separable": <true or false>,
          "reason": "<one-sentence justification; cite an existing feature by number if 'same'>"
        }}
    """)

    try:
        text = chat(client, cfg.organization_model, prompt, max_tokens=200)
    except Exception as e:
        # Fail open: if the LLM call fails, don't block the merge on this check.
        return True, f"separability LLM unreachable ({type(e).__name__}: {e})"

    # Parse the JSON response
    import re
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return True, f"separability response unparseable: {text[:120]}"
    try:
        obj = json.loads(m.group(0))
        return bool(obj.get("separable", True)), str(obj.get("reason", ""))
    except json.JSONDecodeError:
        return True, f"separability JSON invalid: {text[:120]}"


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
    # Identify existing leaf features in catalog order (target_dirs align to these).
    existing_features = existing_catalog["features"]
    existing_leaves = [f for f in existing_features if f.get("type") == "leaf"]
    n_existing_leaves = len(existing_leaves)

    if n_existing_leaves != existing_target_dirs.shape[0]:
        raise ValueError(
            f"existing_target_dirs has {existing_target_dirs.shape[0]} rows but "
            f"existing_catalog has {n_existing_leaves} leaves — mismatch. Ensure "
            f"target_dirs was computed on this exact catalog."
        )

    if len(proposed_features) != proposed_dirs.shape[0]:
        raise ValueError(
            f"proposed_features ({len(proposed_features)}) does not match "
            f"proposed_dirs ({proposed_dirs.shape[0]}) row count."
        )

    # Existing feature IDs for downstream conflict checks.
    existing_ids = {f["id"] for f in existing_features}

    # Cosine-based dedup: for each proposal, find nearest existing leaf.
    max_cos, argmax = _closest_existing_by_cosine(proposed_dirs, existing_target_dirs)

    # For LLM separability, pre-compute the top-k nearest neighbors per proposal.
    if use_llm_separability:
        p = _normalize_dirs(proposed_dirs)
        e = _normalize_dirs(existing_target_dirs)
        sim = p @ e.T                               # (n_prop, n_existing_leaves)
        topk = min(n_neighbors_for_llm, n_existing_leaves)
        _, neighbors_idx = sim.topk(topk, dim=1)    # (n_prop, topk)
    else:
        neighbors_idx = None

    kept: list[dict] = []
    dropped: list[dict] = []

    for i, feat in enumerate(proposed_features):
        cos_i = float(max_cos[i].item())
        close_idx = int(argmax[i].item())
        close_leaf = existing_leaves[close_idx]

        # Gate 1: cosine dedup
        if cos_i > cos_threshold:
            dropped.append({
                "id": feat.get("id", f"proposal_{i}"),
                "description": feat.get("description", ""),
                "reason": f"redundant_direction (cos={cos_i:.3f} > {cos_threshold})",
                "closest_existing_id": close_leaf["id"],
                "max_cos": round(cos_i, 4),
            })
            continue

        # Gate 2: LLM separability on survivors
        if use_llm_separability and cfg is not None:
            neighbor_descs = [
                existing_leaves[int(j)]["description"]
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
                    "closest_existing_id": close_leaf["id"],
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
            "n_existing_before_merge": n_existing_leaves,
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
