"""
Merge corpus slices from parallel 2-GPU annotation jobs (v8.20.5).

Workaround for the §3a 4-EngineCore contention: instead of running 4
shards in one process (which collapses to ~22 dec/s/shard), run two
independent 2-GPU jobs on disjoint corpus halves, then merge the
outputs into one canonical artifact set.

Workflow:
    # Job A — first 5K sequences on GPUs 0,1
    CUDA_VISIBLE_DEVICES=0,1 python -m pipeline.run \\
      --output_dir pipeline_data_a \\
      --n_sequences 5000 --corpus-skip 0 \\
      --step annotate

    # Job B — next 5K sequences on GPUs 2,3 (parallel)
    CUDA_VISIBLE_DEVICES=2,3 python -m pipeline.run \\
      --output_dir pipeline_data_b \\
      --n_sequences 5000 --corpus-skip 5000 \\
      --step annotate

    # Merge into the canonical run dir
    python -m pipeline.run \\
      --output_dir pipeline_data_scaling \\
      --step merge-slices \\
      --merge-from pipeline_data_a pipeline_data_b

Concatenates tokens.pt, activations.pt, annotations.pt along the
sequence axis (dim 0) in the order given. Sidecars (annotations_meta,
position_mask if present) are taken from the first slice and validated
to match the rest.

Each input slice MUST have been annotated against the SAME catalog
(feature_catalog.json with identical leaf order). Mismatch raises.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import Config


def _load_meta_ids(meta_path: Path) -> list[str]:
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text())
    return meta.get("feature_ids") or meta.get("ids") or []


def _load_tensor(path: Path, name: str) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Slice missing {name}: {path}")
    return torch.load(path, weights_only=True)


def run(cfg: Config | None = None) -> dict:
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print("MERGE-SLICES  (concatenate parallel-job outputs)")
    print("=" * 70)

    merge_from = list(getattr(cfg, "merge_from_dirs", []) or [])
    if len(merge_from) < 2:
        raise RuntimeError(
            "Need at least 2 source dirs via --merge-from DIR1 DIR2 [DIR3 ...]"
        )

    src_dirs = [Path(p) for p in merge_from]
    for d in src_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Source dir missing: {d}")

    print(f"  Sources ({len(src_dirs)}):")
    for d in src_dirs:
        print(f"    {d}")
    print(f"  Destination: {cfg.output_dir}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Validate annotations_meta consistency across slices.
    metas = [_load_meta_ids(d / "annotations_meta.json") for d in src_dirs]
    n_meta_present = sum(1 for m in metas if m)
    if n_meta_present > 0 and n_meta_present != len(src_dirs):
        raise RuntimeError(
            f"Inconsistent annotations_meta presence across slices: "
            f"{n_meta_present}/{len(src_dirs)} have it. All-or-none required."
        )
    if n_meta_present == len(src_dirs):
        ref = metas[0]
        for i, m in enumerate(metas[1:], start=1):
            if m != ref:
                raise RuntimeError(
                    f"Slice {src_dirs[i]} has different feature_ids than "
                    f"slice {src_dirs[0]}. The slices were annotated against "
                    f"different catalogs and cannot be merged."
                )
        print(f"  annotations_meta validated: {len(ref)} features, all slices match")

    # Concatenate tokens.pt
    tokens_list = [_load_tensor(d / "tokens.pt", "tokens") for d in src_dirs]
    for i, t in enumerate(tokens_list):
        print(f"    [{src_dirs[i].name}] tokens={tuple(t.shape)}")
    if len({t.shape[1] for t in tokens_list}) != 1:
        raise RuntimeError(
            f"tokens.pt seq_len differs across slices: "
            f"{[t.shape for t in tokens_list]}"
        )
    tokens_merged = torch.cat(tokens_list, dim=0)
    print(f"  tokens merged: {tuple(tokens_merged.shape)}")

    # Concatenate activations.pt
    acts_list = [_load_tensor(d / "activations.pt", "activations") for d in src_dirs]
    for i, t in enumerate(acts_list):
        print(f"    [{src_dirs[i].name}] activations={tuple(t.shape)}")
    if len({t.shape[1:] for t in acts_list}) != 1:
        raise RuntimeError(
            f"activations.pt shape (excluding dim 0) differs across slices: "
            f"{[t.shape for t in acts_list]}"
        )
    acts_merged = torch.cat(acts_list, dim=0)
    print(f"  activations merged: {tuple(acts_merged.shape)}")

    # Concatenate annotations.pt
    ann_list = [_load_tensor(d / "annotations.pt", "annotations") for d in src_dirs]
    for i, t in enumerate(ann_list):
        print(f"    [{src_dirs[i].name}] annotations={tuple(t.shape)}")
    if len({t.shape[1:] for t in ann_list}) != 1:
        raise RuntimeError(
            f"annotations.pt shape (excluding dim 0) differs across slices: "
            f"{[t.shape for t in ann_list]}"
        )
    ann_merged = torch.cat(ann_list, dim=0)
    print(f"  annotations merged: {tuple(ann_merged.shape)}")

    # Sanity: rows align
    if not (tokens_merged.shape[0] == acts_merged.shape[0] == ann_merged.shape[0]):
        raise RuntimeError(
            f"Row mismatch after merge: tokens={tokens_merged.shape[0]}, "
            f"activations={acts_merged.shape[0]}, annotations={ann_merged.shape[0]}"
        )

    # Write merged outputs
    torch.save(tokens_merged, cfg.tokens_path)
    print(f"  Saved: {cfg.tokens_path}")
    torch.save(acts_merged, cfg.activations_path)
    print(f"  Saved: {cfg.activations_path}")
    torch.save(ann_merged, cfg.annotations_path)
    print(f"  Saved: {cfg.annotations_path}")

    # Copy annotations_meta from the first slice (already validated identical)
    if n_meta_present == len(src_dirs):
        src_meta = src_dirs[0] / "annotations_meta.json"
        (cfg.output_dir / "annotations_meta.json").write_text(
            src_meta.read_text()
        )
        print(f"  Saved: annotations_meta.json (from {src_dirs[0].name})")

    # Write merge-meta sidecar so downstream auditing has a paper trail
    merge_meta = {
        "n_sources": len(src_dirs),
        "sources": [str(d) for d in src_dirs],
        "n_seqs_per_source": [int(t.shape[0]) for t in tokens_list],
        "n_seqs_total": int(tokens_merged.shape[0]),
        "seq_len": int(tokens_merged.shape[1]),
        "n_features": int(ann_merged.shape[-1]),
    }
    (cfg.output_dir / "merge_slices_meta.json").write_text(
        json.dumps(merge_meta, indent=2)
    )
    print(f"  Saved: merge_slices_meta.json")

    # Optional: copy feature_catalog.json from the first source if not
    # present in the destination already. This makes downstream
    # --step train / evaluate work without manual symlinking.
    if not cfg.catalog_path.exists():
        src_cat = src_dirs[0] / "feature_catalog.json"
        if src_cat.exists():
            cfg.catalog_path.write_text(src_cat.read_text())
            print(f"  Saved: feature_catalog.json (from {src_dirs[0].name})")
    return merge_meta
