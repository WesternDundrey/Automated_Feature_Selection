"""
Corpus extension (v8.18.35).

Bumps `tokens.pt` / `activations.pt` / `annotations.pt` from N_old
sequences to N_new sequences while preserving the existing labels for
seqs [0, N_old) and only computing new tensors for [N_old, N_new).
Saves ~3× cost vs re-running the whole pipeline when N grows from
3000 → 5000.

Safety contract:
  1. Tokenization is deterministic given (corpus_dataset, corpus_split,
     min-text-length filter, seq_len). The first N_old sequences of the
     newly-tokenized corpus MUST exactly match the existing tokens.pt
     bit-for-bit. If they don't, the run aborts and restores backups —
     this catches corpus drift (different dataset version, different
     filter, etc.) before stale annotations get concatenated to fresh
     ones.
  2. Originals (tokens.pt / activations.pt / annotations.pt) are copied
     to `.bak.extend` siblings before any write. On any exception, the
     backups are restored. On success, backups are deleted.
  3. Cache-meta sidecars are updated to reflect the new n_sequences so
     downstream cache checks accept the extended artifacts.
  4. Catalog identity: annotations_meta.json's feature_ids must equal
     the current catalog's leaf IDs in order, OR the tail annotation
     would land in the wrong column. Verified before any work begins.

Triggered via `--step extend-corpus`. The current `cfg.n_sequences`
defines the target; if it is <= existing n_old, the call is a no-op.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from .config import Config


def _backup(src: Path, suffix: str = ".bak.extend") -> Optional[Path]:
    """Copy src → src.bak.extend if not already present. Returns the
    backup path (or None if src doesn't exist)."""
    if not src.exists():
        return None
    bak = src.with_name(src.name + suffix)
    if not bak.exists():
        shutil.copy2(src, bak)
    return bak


def _restore_from_backups(backups: dict[Path, Path]) -> None:
    """Restore each src from its backup (best-effort)."""
    for src, bak in backups.items():
        if bak is not None and bak.exists():
            try:
                shutil.copy2(bak, src)
                print(f"  [extend-corpus] restored {src.name} from backup")
            except Exception as e:
                print(f"  [extend-corpus] WARNING: could not restore {src}: {e}")


def _cleanup_backups(backups: dict[Path, Path]) -> None:
    """Drop backups after a successful extension."""
    for bak in backups.values():
        if bak is not None and bak.exists():
            try:
                bak.unlink()
            except Exception:
                pass


def _run_extend_subprocess(
    cfg: Config,
    n_old: int,
    target_n: int,
    tokens_full_tmp: Path,
    acts_tail_tmp: Optional[Path],
) -> None:
    """Subprocess-isolated step that loads the model, tokenizes the
    full target_n sequences, and extracts activations for ONLY the tail
    [n_old:target_n].

    Outputs:
      - tokens_full_tmp  (target_n, seq_len) int64
      - acts_tail_tmp    (target_n - n_old, seq_len, d_model) float32  (if requested)

    The model is loaded once and used for both tokenization and
    extraction. CUDA context is contained to the subprocess so the
    parent process can subsequently launch vLLM without contamination.
    """
    if acts_tail_tmp is not None:
        acts_save_block = (
            f"tokens_tail = tokens_full[{n_old}:].contiguous()\n"
            f"acts_tail = extract_activations(model, tokens_tail, cfg)\n"
            f"torch.save(acts_tail, {str(acts_tail_tmp)!r})\n"
            f'print(f"[extend-corpus subproc] activations tail saved: '
            f'shape={{tuple(acts_tail.shape)}}")\n'
        )
    else:
        acts_save_block = (
            'print("[extend-corpus subproc] '
            'activations.pt not present in parent — skipping tail extraction")\n'
        )

    script = f"""
import sys
import torch
from pipeline.annotate import prepare_corpus, extract_activations
from pipeline.inventory import load_target_model
from pipeline.config import Config

cfg = Config(
    model_name={cfg.model_name!r},
    device={cfg.device!r},
    model_dtype={cfg.model_dtype!r},
    n_sequences={target_n},
    seq_len={cfg.seq_len},
    corpus_batch_size={cfg.corpus_batch_size},
    target_layer={cfg.target_layer},
    output_dir={str(cfg.output_dir)!r},
    hook_point={cfg.hook_point!r},
    sae_release={cfg.sae_release!r},
    sae_id={cfg.sae_id!r},
    corpus_dataset={cfg.corpus_dataset!r},
    corpus_split={cfg.corpus_split!r},
)
cfg.output_dir.mkdir(parents=True, exist_ok=True)
model = load_target_model(cfg)

tokens_full = prepare_corpus(model, cfg)
if tokens_full.shape[0] < {target_n}:
    print(f"ERROR: corpus yielded {{tokens_full.shape[0]}} sequences, expected {target_n}")
    sys.exit(2)

torch.save(tokens_full, {str(tokens_full_tmp)!r})
print(f"[extend-corpus subproc] tokens saved: shape={{tuple(tokens_full.shape)}}")

{acts_save_block}"""
    print("  [extend-corpus] launching tokenization + tail-activation subprocess...")
    subprocess.run([sys.executable, "-c", script], check=True)


def run(cfg: Config = None) -> dict:
    """Extend pipeline_data artifacts to cfg.n_sequences, preserving
    existing seqs [0, N_old). Returns a summary dict."""
    if cfg is None:
        cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    target_n = cfg.n_sequences
    tokens_path = cfg.tokens_path
    acts_path = cfg.activations_path
    ann_path = cfg.annotations_path
    ann_meta_path = cfg.annotations_meta_path

    # ── Pre-flight checks ────────────────────────────────────────────────
    if not tokens_path.exists():
        raise FileNotFoundError(
            f"extend-corpus requires existing {tokens_path}. Run the regular "
            f"pipeline first (so we have something to extend), or run without "
            f"--step extend-corpus."
        )

    tokens_old = torch.load(tokens_path, weights_only=True)
    n_old = int(tokens_old.shape[0])
    t_old = int(tokens_old.shape[1])

    if t_old != cfg.seq_len:
        raise ValueError(
            f"existing tokens.pt has seq_len={t_old} but cfg.seq_len={cfg.seq_len}. "
            f"Cannot extend across different seq_len; delete pipeline_data/tokens.pt "
            f"(and downstream artifacts) to regenerate from scratch."
        )

    if n_old >= target_n:
        print(f"  [extend-corpus] STATUS=noop  tokens.pt has {n_old} >= target {target_n}")
        return {"skipped": True, "n_old": n_old, "n_new": n_old, "added": 0}

    n_added = target_n - n_old
    print(f"\n  [extend-corpus] extending {n_old} → {target_n} (+{n_added} new sequences)")

    # If annotations.pt exists, verify its feature_ids match current catalog
    # leaves IN ORDER before we do any expensive work.
    catalog: Optional[dict] = None
    leaf_features: list[dict] = []
    if cfg.catalog_path.exists():
        catalog = json.loads(cfg.catalog_path.read_text())
        leaf_features = [f for f in catalog.get("features", []) if f.get("type") == "leaf"]

    if ann_path.exists():
        if not leaf_features:
            raise FileNotFoundError(
                f"annotations.pt exists but {cfg.catalog_path} is missing or has "
                f"no leaves. Cannot extend annotations without a catalog."
            )
        ann_old_shape = torch.load(ann_path, weights_only=True, map_location="cpu").shape
        if ann_old_shape[0] != n_old:
            raise RuntimeError(
                f"annotations.pt has {ann_old_shape[0]} sequences but tokens.pt has "
                f"{n_old}. Inconsistent state — refusing to extend. Delete one and "
                f"regenerate."
            )
        if ann_meta_path.exists():
            try:
                meta = json.loads(ann_meta_path.read_text())
                cached_ids = meta.get("feature_ids") or []
                current_ids = [f["id"] for f in leaf_features]
                if cached_ids != current_ids:
                    raise RuntimeError(
                        f"annotations_meta.json feature_ids ({len(cached_ids)} ids) "
                        f"do not match current catalog leaves ({len(current_ids)} ids) "
                        f"in order. The catalog has changed since last annotation; "
                        f"sequence-extension would corrupt column alignment. Reset "
                        f"and re-annotate from scratch instead."
                    )
            except json.JSONDecodeError as e:
                raise RuntimeError(f"annotations_meta.json is malformed: {e}")
        elif ann_old_shape[-1] != len(leaf_features):
            raise RuntimeError(
                f"annotations.pt has {ann_old_shape[-1]} feature columns but catalog "
                f"has {len(leaf_features)} leaves and there's no meta sidecar to "
                f"verify alignment. Refusing to extend; reset and re-annotate."
            )

    # ── Backups (taken before any write) ────────────────────────────────
    backups: dict[Path, Optional[Path]] = {
        tokens_path: _backup(tokens_path),
        acts_path:   _backup(acts_path),
        ann_path:    _backup(ann_path),
    }
    print(f"  [extend-corpus] backups created: "
          f"{[str(b) for b in backups.values() if b]}")

    tokens_full_tmp = tokens_path.with_name(tokens_path.name + ".extending.tmp")
    acts_tail_tmp = (
        acts_path.with_name(acts_path.name + ".extending_tail.tmp")
        if acts_path.exists() else None
    )

    try:
        # ── Step 1/3: tokenize full target_n + extract tail activations ──
        # Both happen in one subprocess (model loaded once).
        print(f"  [extend-corpus] step 1/3: tokenize {target_n} sequences"
              + (" + extract activations for tail" if acts_tail_tmp else ""))
        t0 = time.time()
        _run_extend_subprocess(cfg, n_old, target_n, tokens_full_tmp, acts_tail_tmp)
        print(f"    subprocess done in {time.time() - t0:.1f}s")

        # Verify head match: tokens_full[:n_old] MUST equal existing tokens_old.
        tokens_full = torch.load(tokens_full_tmp, weights_only=True)
        if tokens_full.shape != (target_n, cfg.seq_len):
            raise RuntimeError(
                f"newly tokenized corpus has shape {tuple(tokens_full.shape)}, "
                f"expected ({target_n}, {cfg.seq_len})"
            )
        if not torch.equal(tokens_full[:n_old], tokens_old):
            n_diff = int((tokens_full[:n_old] != tokens_old).sum().item())
            raise RuntimeError(
                f"first {n_old} sequences of newly-tokenized corpus DO NOT match "
                f"existing tokens.pt ({n_diff} token mismatches across the head). "
                f"The corpus has drifted (different dataset version, filter, or "
                f"seed). Refusing to concatenate stale annotations to fresh "
                f"sequences. Either delete pipeline_data/ and rerun from scratch, "
                f"or pin the dataset version that produced the existing tokens.pt."
            )
        print(f"    ✓ head match verified: first {n_old} sequences identical")

        # Commit tokens.pt (atomic rename).
        from .cache_meta import write_cache_meta
        torch.save(tokens_full, tokens_path)
        write_cache_meta(tokens_path, "tokens", cfg)
        print(f"    ✓ tokens.pt extended {n_old} → {target_n}")

        # Commit activations.pt if applicable.
        if acts_tail_tmp is not None and acts_tail_tmp.exists():
            acts_old = torch.load(acts_path, weights_only=True)
            acts_tail = torch.load(acts_tail_tmp, weights_only=True)
            if acts_tail.shape[0] != n_added:
                raise RuntimeError(
                    f"activations tail has {acts_tail.shape[0]} sequences, expected "
                    f"{n_added}"
                )
            acts_full = torch.cat([acts_old, acts_tail], dim=0)
            torch.save(acts_full, acts_path)
            write_cache_meta(acts_path, "activations", cfg)
            print(f"    ✓ activations.pt extended {n_old} → {target_n}  "
                  f"(shape {tuple(acts_full.shape)})")
            # Free memory
            del acts_old, acts_tail, acts_full

        # ── Step 2/3: annotate tail ──
        if ann_path.exists():
            print(f"  [extend-corpus] step 2/3: annotate tail [{n_old}:{target_n}] "
                  f"for {len(leaf_features)} features (subprocess)")
            t0 = time.time()
            from .annotate import _annotate_local_subprocess

            tokens_tail = tokens_full[n_old:].contiguous()
            ann_tail = _annotate_local_subprocess(tokens_tail, leaf_features, cfg)
            print(f"    annotation done in {time.time() - t0:.1f}s")

            if ann_tail.shape[0] != n_added or ann_tail.shape[1] != cfg.seq_len \
                    or ann_tail.shape[2] != len(leaf_features):
                raise RuntimeError(
                    f"annotation tail shape {tuple(ann_tail.shape)} doesn't match "
                    f"({n_added}, {cfg.seq_len}, {len(leaf_features)})"
                )

            ann_old = torch.load(ann_path, weights_only=True)
            ann_full = torch.cat([ann_old, ann_tail], dim=0)
            torch.save(ann_full, ann_path)
            ann_meta_path.write_text(json.dumps({
                "feature_ids": [f["id"] for f in leaf_features],
            }, indent=2))
            print(f"    ✓ annotations.pt extended {n_old} → {target_n}  "
                  f"(shape {tuple(ann_full.shape)})")
            del ann_old, ann_tail, ann_full
        else:
            print(f"  [extend-corpus] step 2/3: annotations.pt not present, skipping")

        # ── Step 3/3: cleanup backups + tmp files ──
        for tmp in [tokens_full_tmp, acts_tail_tmp]:
            if tmp is not None and tmp.exists():
                tmp.unlink()
        _cleanup_backups(backups)
        print(f"  [extend-corpus] STATUS=ok  cleanup done")

        return {
            "skipped": False,
            "n_old": n_old,
            "n_new": target_n,
            "added": n_added,
        }

    except Exception as e:
        print(f"\n  [extend-corpus] FAILED: {type(e).__name__}: {e}")
        print(f"  [extend-corpus] restoring from backups...")
        _restore_from_backups(backups)
        # Drop tmp files
        for tmp in [tokens_full_tmp, acts_tail_tmp]:
            if tmp is not None and tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass
        raise
