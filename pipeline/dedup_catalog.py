"""
Dedup feature catalog by target-direction cosine similarity (v8.19.x).

The chunked Opus / Sonnet catalog generator (opus_catalog.py) splits
the shortlist across N independent calls, and each chunk independently
designs digit / surface / punctuation families. Result: near-duplicates
like `digits.0` / `digits.zero` / `digits.d0` / `digit.d0` all firing
on the same tokens, sharing decoder direction (cos ≈ 1.0), inflating
the leaf count, and dragging mean F1 down because the shared variance
is split across multiple columns.

This step:
  1. Loads `target_directions.pt` (written by train.py)
  2. Computes pairwise cosine across feature decoder columns
  3. Groups features with cosine ≥ `dedup_cos_threshold` (default 0.95)
  4. For each group keeps ONE canonical feature — preferring the one
     with highest cal F1 from `evaluation.json` if available, otherwise
     the shortest id (tie-broken alphabetically)
  5. Writes a cleaned catalog to `feature_catalog.deduped.json`
     (does NOT overwrite the original; user opts in by replacing)
  6. Writes a human-readable `dedup_report.json` showing clusters
     and drop reasons

Run with: python -m pipeline.run --step dedup-catalog
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import Config


def _load_eval_f1(cfg: Config) -> dict[str, float]:
    """Map feature id → cal F1 from evaluation.json (if present)."""
    eval_path = cfg.output_dir / "evaluation.json"
    if not eval_path.exists():
        return {}
    try:
        ev = json.loads(eval_path.read_text())
        out = {}
        for r in ev.get("features", []):
            fid = r.get("id")
            f1 = r.get("f1_cal") or r.get("f1") or 0.0
            if fid:
                out[fid] = float(f1)
        return out
    except Exception:
        return {}


def _build_clusters(
    cos: np.ndarray, leaf_ids: list[str], threshold: float
) -> list[list[int]]:
    """Connected-components over the cosine-threshold graph."""
    n = len(leaf_ids)
    visited = [False] * n
    clusters: list[list[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        # BFS
        stack = [start]
        component = []
        while stack:
            i = stack.pop()
            if visited[i]:
                continue
            visited[i] = True
            component.append(i)
            for j in range(n):
                if not visited[j] and cos[i, j] >= threshold:
                    stack.append(j)
        clusters.append(sorted(component))
    return clusters


def _pick_canonical(
    cluster: list[int], leaf_ids: list[str], f1_by_id: dict[str, float]
) -> int:
    """Pick canonical feature index from a cluster.

    Priority:
      1. Highest cal F1 if available
      2. Shortest id length (more canonical: 'digits.0' over 'digits.d0')
      3. Alphabetical tie-break
    """
    def key(i):
        fid = leaf_ids[i]
        f1 = f1_by_id.get(fid, 0.0)
        return (-f1, len(fid), fid)
    return min(cluster, key=key)


def run(cfg: Config = None) -> dict:
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print("DEDUP CATALOG  (target-direction cosine clustering)")
    print("=" * 70)

    catalog_path = cfg.catalog_path
    target_dirs_path = cfg.target_dirs_path
    if not catalog_path.exists():
        raise FileNotFoundError(f"Need {catalog_path}")
    if not target_dirs_path.exists():
        raise FileNotFoundError(
            f"Need {target_dirs_path}. Run --step train first; target "
            f"directions are written there. (If you only need dedup, "
            f"train can be killed shortly after target_dirs are saved — "
            f"that happens early in the run.)"
        )

    catalog = json.loads(catalog_path.read_text())
    leaves = [f for f in catalog.get("features", []) if f.get("type") == "leaf"]
    leaf_ids = [f["id"] for f in leaves]
    n = len(leaves)
    print(f"  Loaded catalog: {n} leaves")

    target_dirs = torch.load(target_dirs_path, weights_only=True).cpu().numpy()
    if target_dirs.shape[0] != n:
        raise RuntimeError(
            f"target_dirs has {target_dirs.shape[0]} rows but catalog "
            f"has {n} leaves. Re-run --step train after the last "
            f"catalog change."
        )

    # Cosine similarity. target_dirs are already unit-norm post-train,
    # but normalize defensively.
    norms = np.linalg.norm(target_dirs, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    td_n = target_dirs / norms
    cos = td_n @ td_n.T
    np.fill_diagonal(cos, 0.0)  # don't self-link

    threshold = float(getattr(cfg, "dedup_cos_threshold", 0.95))
    f1_by_id = _load_eval_f1(cfg)
    if f1_by_id:
        print(f"  Loaded cal F1 for {len(f1_by_id)} features from evaluation.json")
    else:
        print(f"  No evaluation.json found; tie-break by id length only")

    clusters = _build_clusters(cos, leaf_ids, threshold)
    n_singletons = sum(1 for c in clusters if len(c) == 1)
    n_groups = sum(1 for c in clusters if len(c) > 1)
    print(f"  Cos ≥ {threshold} clusters: {len(clusters)}  "
          f"({n_singletons} singletons, {n_groups} multi-feature groups)")

    keep_ids: set[str] = set()
    drop_records: list[dict] = []
    for cluster in clusters:
        if len(cluster) == 1:
            keep_ids.add(leaf_ids[cluster[0]])
            continue
        canonical = _pick_canonical(cluster, leaf_ids, f1_by_id)
        keep_ids.add(leaf_ids[canonical])
        for i in cluster:
            if i == canonical:
                continue
            drop_records.append({
                "dropped_id": leaf_ids[i],
                "kept_canonical": leaf_ids[canonical],
                "cosine_to_canonical": round(float(cos[i, canonical]), 4),
                "cluster_size": len(cluster),
                "f1_dropped": f1_by_id.get(leaf_ids[i]),
                "f1_kept": f1_by_id.get(leaf_ids[canonical]),
            })

    print(f"\n  Kept:    {len(keep_ids)} / {n} leaves")
    print(f"  Dropped: {len(drop_records)} duplicates")

    # Show top dropped clusters
    if drop_records:
        print(f"\n  Top dropped (by cluster size, then cosine):")
        sorted_drops = sorted(
            drop_records, key=lambda r: (-r["cluster_size"], -r["cosine_to_canonical"])
        )
        for r in sorted_drops[:20]:
            print(f"    {r['dropped_id']:40s} → {r['kept_canonical']:30s}  "
                  f"cos={r['cosine_to_canonical']:.3f}  "
                  f"(f1: {r['f1_dropped']} → {r['f1_kept']})")

    # Build deduped catalog. Preserve groups whose surviving children > 0.
    kept_leaves = [f for f in leaves if f["id"] in keep_ids]
    kept_parents = {f.get("parent") for f in kept_leaves if f.get("parent")}
    kept_groups = [
        f for f in catalog["features"]
        if f.get("type") == "group" and f.get("id") in kept_parents
    ]
    deduped = dict(catalog)
    deduped["features"] = kept_groups + kept_leaves
    deduped["dedup_threshold"] = threshold
    deduped["n_dropped"] = len(drop_records)

    out_path = cfg.output_dir / "feature_catalog.deduped.json"
    out_path.write_text(json.dumps(deduped, indent=2))
    print(f"\n  Saved: {out_path}")

    report_path = cfg.output_dir / "dedup_report.json"
    report_path.write_text(json.dumps({
        "n_leaves_before": n,
        "n_leaves_after": len(keep_ids),
        "n_dropped": len(drop_records),
        "threshold": threshold,
        "clusters": [
            {
                "size": len(c),
                "ids": [leaf_ids[i] for i in c],
                "canonical": leaf_ids[_pick_canonical(c, leaf_ids, f1_by_id)],
            }
            for c in clusters if len(c) > 1
        ],
        "drops": drop_records,
    }, indent=2))
    print(f"  Saved: {report_path}")
    print(f"\n  To use: review {out_path.name}, then:")
    print(f"    cp {out_path} {catalog_path}     # opt in")
    return deduped
