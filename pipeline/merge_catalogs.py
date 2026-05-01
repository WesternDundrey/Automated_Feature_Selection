"""
Merge Opus + Delphi catalogs into feature_catalog.json (v8.19.0).

The full benchmark path is:
   shortlist → delphi-run + opus-catalog → MERGE → annotate → train ...

Without the explicit merge step, `--step annotate` reads whatever stale
`feature_catalog.json` happens to exist. This module concatenates leaves
from both catalogs (preserving group structure for each side), tags
Delphi leaves so downstream stages can route them correctly, and writes
the canonical `feature_catalog.json` that the rest of the pipeline
expects.

Routing of merged leaves:
  • Opus leaves:  full boundary-discipline contract (positive_examples,
                  negative_examples, exclusions). Trained against by
                  supSAE.
  • Delphi leaves: tagged `delphi_mode: True` and `source_kind: "delphi"`.
                  Annotated for the unsup arm's F1 evaluation but
                  EXCLUDED from supSAE training (see train.py filter).
                  Exempt from the boundary-discipline validator.

ID collisions: Opus uses dotted ids (group.value). Delphi uses
`delphi.latent_<idx>`. No collision in normal operation, but if Opus
ever produces a `delphi.*` id, the Opus version wins (sup features
are the headline).
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config


def _is_leaf(f: dict) -> bool:
    return f.get("type") == "leaf"


def _is_group(f: dict) -> bool:
    return f.get("type") == "group"


def _tag_delphi(f: dict) -> dict:
    """Ensure Delphi leaves carry the right tags for downstream filters."""
    out = dict(f)
    if _is_leaf(out):
        out.setdefault("delphi_mode", True)
        out.setdefault("source_kind", "delphi")
    return out


def _normalize_opus(f: dict) -> dict:
    """Ensure Opus leaves carry source_kind for catalog_quality exemption."""
    out = dict(f)
    if _is_leaf(out):
        # Symmetry-completing leaves with no grounding latents — Opus is
        # instructed to set source: "symmetry"; mirror to source_kind so
        # catalog_quality._is_boundary_discipline_exempt recognizes them.
        if not out.get("source_latents") and out.get("source") == "symmetry":
            out.setdefault("source_kind", "symmetry")
    return out


def run(cfg: Config = None) -> dict:
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print("MERGE CATALOGS  (Opus + Delphi → feature_catalog.json)")
    print("=" * 70)

    opus_path = cfg.output_dir / "opus_catalog.json"
    delphi_path = cfg.output_dir / "delphi_catalog.json"

    have_opus = opus_path.exists()
    have_delphi = delphi_path.exists()
    if not (have_opus or have_delphi):
        raise FileNotFoundError(
            f"Need at least one of {opus_path}, {delphi_path}. "
            f"Run --step opus-catalog and/or --step delphi-run first."
        )

    features: list[dict] = []
    seen_ids: set[str] = set()
    n_opus_leaves = n_opus_groups = 0
    n_delphi_leaves = n_delphi_groups = 0

    if have_opus:
        opus = json.loads(opus_path.read_text())
        for f in opus.get("features", []):
            fid = f.get("id")
            if not fid or fid in seen_ids:
                continue
            features.append(_normalize_opus(f))
            seen_ids.add(fid)
            if _is_leaf(f):
                n_opus_leaves += 1
            elif _is_group(f):
                n_opus_groups += 1

    if have_delphi:
        delphi = json.loads(delphi_path.read_text())
        for f in delphi.get("features", []):
            fid = f.get("id")
            if not fid or fid in seen_ids:
                continue
            features.append(_tag_delphi(f))
            seen_ids.add(fid)
            if _is_leaf(f):
                n_delphi_leaves += 1
            elif _is_group(f):
                n_delphi_groups += 1

    merged = {
        "features": features,
        "merged_from": {
            "opus": str(opus_path) if have_opus else None,
            "delphi": str(delphi_path) if have_delphi else None,
        },
        "n_opus_leaves": n_opus_leaves,
        "n_delphi_leaves": n_delphi_leaves,
    }

    out_path = cfg.catalog_path
    out_path.write_text(json.dumps(merged, indent=2))

    print(f"  Opus leaves:    {n_opus_leaves} (in {n_opus_groups} groups)")
    print(f"  Delphi leaves:  {n_delphi_leaves} (in {n_delphi_groups} groups)")
    print(f"  Total leaves:   {n_opus_leaves + n_delphi_leaves}")
    print(f"  Saved: {out_path}")
    return merged
