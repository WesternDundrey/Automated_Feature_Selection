"""
One-shot diagnostic for a finished promote-loop round (default round_00).

Prints:
  1. Pairwise cosine distribution among existing `target_directions.pt`.
     Tells us whether the catalog is geometrically well-distributed or
     whether some existing directions are near-parallel (duplicated).
  2. The cosine-dropped proposals — each with its closest existing feature
     and (if available) the description. Lets us eyeball whether proposals
     were semantically redundant with their nearest match or novel concepts
     that the gate over-rejected.
  3. The crispness-passing descriptions themselves (so we can see what
     concepts round 0 tried to add).

Usage:
    python -m pipeline.diagnose_promote_round               # defaults to round_00
    python -m pipeline.diagnose_promote_round --round 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument(
        "--output_dir", default="pipeline_data",
        help="root of the pipeline output (default: pipeline_data)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    round_dir = out / "promote_loop" / f"round_{args.round:02d}"
    if not round_dir.exists():
        raise FileNotFoundError(f"no promote-loop round at {round_dir}")

    # ── (1) pairwise cosine among existing target_dirs ────────────────
    tdirs_path = out / "target_directions.pt"
    if tdirs_path.exists():
        t = torch.load(tdirs_path, weights_only=True).float()
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
        n = t.shape[0]
        sim = t @ t.T - torch.eye(n)
        off = sim.flatten()
        off = off[off != 0]
        print("=" * 72)
        print("(1) PAIRWISE COSINE AMONG EXISTING TARGET_DIRS")
        print("=" * 72)
        print(f"  n_features:      {n}")
        print(f"  mean:            {float(off.mean()):.4f}")
        print(f"  median:          {float(off.median()):.4f}")
        print(f"  max (off-diag):  {float(sim.max()):.4f}")
        print(f"  min (off-diag):  {float(sim.min()):.4f}")
        print(f"  pairs > 0.8:     {int((sim > 0.8).sum() // 2)}")
        print(f"  pairs > 0.9:     {int((sim > 0.9).sum() // 2)}")
        print(f"  pairs > 0.95:    {int((sim > 0.95).sum() // 2)}")

        # If there are high-cosine pairs, print the top 5 so we can see
        # which features are duplicates of which.
        cat_path = out / "feature_catalog.json"
        if cat_path.exists():
            cat = json.loads(cat_path.read_text())
            feats = cat["features"]
            if sim.max() > 0.8:
                tri = torch.triu(sim, diagonal=1)
                flat = tri.flatten()
                k = min(10, int((flat > 0.8).sum().item()))
                if k > 0:
                    top_vals, top_idx = flat.topk(k)
                    print(f"\n  TOP {k} OFF-DIAGONAL PAIRS (cos > 0.8):")
                    for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
                        i, j = idx // n, idx % n
                        ida = feats[i]["id"] if i < len(feats) else f"#{i}"
                        idb = feats[j]["id"] if j < len(feats) else f"#{j}"
                        print(f"    cos={v:.4f}  {ida}  ↔  {idb}")
    else:
        print(f"  (target_directions.pt missing at {tdirs_path})")

    # ── (2) cosine-dropped proposals this round ────────────────────────
    drop_path = round_dir / "dropped.json"
    desc_path = round_dir / "descriptions.json"
    crisp_path = round_dir / "crispness.json"
    descs = json.loads(desc_path.read_text()) if desc_path.exists() else {}
    crispness = json.loads(crisp_path.read_text()) if crisp_path.exists() else {}

    print("\n" + "=" * 72)
    print(f"(2) DROPPED PROPOSALS — round {args.round}")
    print("=" * 72)
    if not drop_path.exists():
        print(f"  (no dropped.json at {drop_path})")
    else:
        drops = json.loads(drop_path.read_text())
        print(f"  {len(drops)} proposals dropped\n")
        for d in drops:
            # Proposal id looks like `promoted.u173_r0`; extract the u_local
            pid = d["id"]
            u_local = pid.split("u")[-1].split("_")[0] if "u" in pid else ""
            desc = descs.get(u_local, "")
            print(f"  cos={d.get('max_cos')}  {pid}")
            print(f"    → nearest existing: {d.get('closest_existing_id')}  "
                  f"({d.get('closest_existing_type')})")
            print(f"    reason: {d.get('reason')}")
            if desc:
                print(f"    desc: {desc[:160]}")
            print()

    # ── (3) crispness-passing descriptions ────────────────────────────
    print("=" * 72)
    print(f"(3) CRISPNESS-PASSING DESCRIPTIONS — round {args.round}")
    print("=" * 72)
    if not crispness:
        print(f"  (no crispness.json at {crisp_path})")
    else:
        passing = {u: r for u, r in crispness.items() if r.get("crisp")}
        rejected = {u: r for u, r in crispness.items() if not r.get("crisp")}
        print(f"  {len(passing)} passing  |  {len(rejected)} rejected\n")
        print("  PASSING:")
        for u, r in passing.items():
            print(f"    U[{u}]: {r.get('description', '')[:160]}")
        if rejected:
            print("\n  REJECTED (sample of 5):")
            for u, r in list(rejected.items())[:5]:
                print(f"    U[{u}]: {r.get('description', '')[:120]}")
                print(f"      reason: {r.get('reason', '')[:100]}")


if __name__ == "__main__":
    main()
