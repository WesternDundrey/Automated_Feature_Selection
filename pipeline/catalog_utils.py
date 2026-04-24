"""
Shared catalog manipulation helpers.

Scaffold merge: preseed a feature_catalog.json with control/scaffold
features before the promote-loop runs. Without scaffold, the unsupervised
slice spends its top-ΔR² capacity on surface-artifact features
(document-boundary, whitespace, code-identifier, bracket variants) that
the discovery loop then "rediscovers" as multi_concept bundles. Seeding
them pre-discovery lets the supervised slice absorb surface capacity so
the discovery loop can focus on semantic features.

Scaffold features carry `role = "control"`; evaluation stats split
"headline / discovery" from "all features (incl. control)" so the paper
numbers don't credit the method for finding things we hand-seeded.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_catalog(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    # Normalize: ensure every feature has a role (default "discovery" for
    # backward compat with pre-v8.10 catalogs).
    for f in data.get("features", []):
        f.setdefault("role", "discovery")
    return data


def save_catalog(path: Path, catalog: dict) -> None:
    Path(path).write_text(json.dumps(catalog, indent=2))


def merge_scaffold(
    main_catalog_path: Path,
    scaffold_path: Path,
    overwrite_existing_ids: bool = False,
) -> dict:
    """Merge scaffold features into `main_catalog_path` in-place.

    The scaffold JSON has two top-level lists: `groups` (added first) and
    `features` (added after). Entries from the scaffold are appended to
    the main catalog's `features` list. If an id collides with an
    existing feature in the main catalog:
        - `overwrite_existing_ids=False` (default): the scaffold entry is
          SKIPPED (existing catalog wins).
        - `overwrite_existing_ids=True`: the existing entry is replaced
          by the scaffold entry.

    Every feature added by scaffold gets `role="control"` automatically
    if the scaffold entry didn't set one.

    Returns the merged catalog dict.
    """
    main = load_catalog(main_catalog_path)
    scaffold = json.loads(Path(scaffold_path).read_text())

    existing_ids = {f["id"] for f in main.get("features", [])}

    added, skipped, overwritten = 0, 0, 0

    # Scaffold groups first (so children's parent references resolve).
    scaffold_groups = scaffold.get("groups", [])
    for g in scaffold_groups:
        g.setdefault("role", "control")
        if g["id"] in existing_ids:
            if overwrite_existing_ids:
                main["features"] = [
                    x for x in main["features"] if x["id"] != g["id"]
                ]
                main["features"].append(g)
                overwritten += 1
            else:
                skipped += 1
            continue
        main["features"].append(g)
        existing_ids.add(g["id"])
        added += 1

    # Scaffold leaves.
    for f in scaffold.get("features", []):
        f.setdefault("role", "control")
        if f["id"] in existing_ids:
            if overwrite_existing_ids:
                main["features"] = [
                    x for x in main["features"] if x["id"] != f["id"]
                ]
                main["features"].append(f)
                overwritten += 1
            else:
                skipped += 1
            continue
        main["features"].append(f)
        existing_ids.add(f["id"])
        added += 1

    save_catalog(main_catalog_path, main)
    print(
        f"  scaffold merge: +{added} added, {skipped} skipped "
        f"(id collision, keep existing), {overwritten} overwritten"
    )
    return main


def split_by_role(features: list[dict]) -> dict[str, list[dict]]:
    """Group features by their `role` field. Features without a role
    default to 'discovery'. Returns {role_name: [features]}."""
    buckets: dict[str, list[dict]] = {}
    for f in features:
        role = f.get("role") or "discovery"
        buckets.setdefault(role, []).append(f)
    return buckets


def discovery_only_ids(features: list[dict]) -> set[str]:
    """Feature ids to include in 'headline / discovery-only' metric
    aggregates — excludes role='control' by default."""
    return {
        f["id"] for f in features
        if (f.get("role") or "discovery") == "discovery"
    }
