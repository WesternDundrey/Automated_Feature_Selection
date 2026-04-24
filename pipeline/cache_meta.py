"""
Shared cache-identity sidecar protocol.

Each expensive artifact (tokens.pt, activations.pt, supervised_sae.pt,
evaluation.json) is keyed by file presence today, with no check that its
content matches the current `Config`. Switching `target_layer` from 9 to 6
without deleting `pipeline_data/` would silently reuse layer-9 activations
in a layer-6 training run — bad labels quietly multiplied across the rest
of the pipeline.

This module adds a small `.meta.json` sidecar alongside each artifact and
a `verify_cache_meta` helper for load-time validation. Missing sidecars
are treated as "legacy" — accepted with a printed warning so existing
runs keep working — but any NEW save always writes one.

Invariant fields per cached artifact:

    tokens.pt           → model_name, corpus_dataset, corpus_split,
                          n_sequences, seq_len
    activations.pt      → above + target_layer, hook_point, model_dtype,
                          sae_release, sae_id
    supervised_sae.pt   → above + n_features (from catalog at train time)
    evaluation.json     → above + n_features

The verifier only compares the fields that are present on BOTH sides. A
forward-compat policy: when a new field is added, old sidecars just don't
report it, and we accept them with a note.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

from .config import Config


_VERSION = 1

# Field groups keyed by which artifact needs which fields. Use a tuple per
# artifact so the verifier reads exactly the same keys that the writer
# wrote.
CACHE_FIELDS = {
    "tokens": (
        "model_name", "corpus_dataset", "corpus_split",
        "n_sequences", "seq_len",
    ),
    "activations": (
        "model_name", "model_dtype", "corpus_dataset", "corpus_split",
        "n_sequences", "seq_len", "target_layer", "hook_point",
        "sae_release", "sae_id",
    ),
    "supervised_sae": (
        "model_name", "model_dtype", "corpus_dataset", "corpus_split",
        "n_sequences", "seq_len", "target_layer", "hook_point",
        "sae_release", "sae_id", "n_features",
    ),
    "evaluation": (
        "model_name", "model_dtype", "corpus_dataset", "corpus_split",
        "n_sequences", "seq_len", "target_layer", "hook_point",
        "sae_release", "sae_id", "n_features",
    ),
}


def _hash_of(meta: dict) -> str:
    # Deterministic short hash for easy comparison in logs / CI.
    payload = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def sidecar_path(artifact_path: Path) -> Path:
    """Companion meta file next to the artifact: `foo.pt` → `foo.pt.meta.json`."""
    return artifact_path.with_name(artifact_path.name + ".meta.json")


def build_meta(
    artifact_kind: str, cfg: Config, **extra,
) -> dict:
    """Gather the relevant cfg fields for this artifact kind + user extras."""
    fields = CACHE_FIELDS.get(artifact_kind, ())
    meta: dict = {"_artifact": artifact_kind, "_version": _VERSION}
    for name in fields:
        if hasattr(cfg, name):
            val = getattr(cfg, name)
            # Path → str for JSON serialization.
            meta[name] = str(val) if isinstance(val, Path) else val
    meta.update(extra)
    meta["_hash"] = _hash_of({k: v for k, v in meta.items() if not k.startswith("_")})
    return meta


def write_cache_meta(
    artifact_path: Path, artifact_kind: str, cfg: Config, **extra,
) -> dict:
    """Write `<artifact>.meta.json` next to artifact_path. Returns the meta dict."""
    meta = build_meta(artifact_kind, cfg, **extra)
    sidecar_path(artifact_path).write_text(json.dumps(meta, indent=2))
    return meta


def verify_cache_meta(
    artifact_path: Path, artifact_kind: str, cfg: Config,
    extra_required: Optional[dict] = None,
    strict: bool = False,
) -> tuple[bool, str]:
    """Verify artifact_path matches current cfg.

    Args:
        artifact_path: path to the data file (sidecar is derived).
        artifact_kind: key into CACHE_FIELDS.
        cfg: current Config to compare against.
        extra_required: additional key/value pairs that must match exactly
            (e.g., {"n_features": len(current_catalog)}).
        strict: if True, absence of a sidecar is treated as a mismatch.
            If False (default), a missing sidecar is a warning — the cache
            is accepted but the user is told to re-save for safety.

    Returns:
        (ok, reason). If ok is False, `reason` names the first mismatched
        field; if ok is True, `reason` is "ok" or "legacy (no sidecar)".
    """
    meta_path = sidecar_path(artifact_path)
    if not meta_path.exists():
        if strict:
            return False, f"no sidecar at {meta_path}"
        return True, "legacy (no sidecar)"

    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as e:
        return False, f"sidecar JSON invalid: {e}"

    mismatches: list[str] = []
    for field in CACHE_FIELDS.get(artifact_kind, ()):
        if field not in meta:
            continue
        cached = meta[field]
        expected = getattr(cfg, field, None)
        if isinstance(expected, Path):
            expected = str(expected)
        if cached != expected:
            mismatches.append(f"{field}: cached={cached!r} vs cfg={expected!r}")

    if extra_required:
        for k, v in extra_required.items():
            if k in meta and meta[k] != v:
                mismatches.append(f"{k}: cached={meta[k]!r} vs expected={v!r}")

    if mismatches:
        return False, "; ".join(mismatches[:3])
    return True, "ok"


def load_or_die(
    artifact_path: Path, artifact_kind: str, cfg: Config,
    extra_required: Optional[dict] = None,
    action_if_stale: str = "warn",
) -> bool:
    """Convenience: check the cache and print a message.

    action_if_stale ∈ {"warn", "raise"}:
      - "warn": print WARNING and return False so caller can regenerate.
      - "raise": raise FileExistsError with the mismatch reason.

    Returns True if the cache is fresh OR the mismatch is the "legacy"
    no-sidecar case. Returns False if cache should be regenerated.
    """
    ok, reason = verify_cache_meta(
        artifact_path, artifact_kind, cfg, extra_required=extra_required,
    )
    if ok:
        if reason != "ok":
            print(f"  [cache] {artifact_path.name}: {reason}")
        return True
    msg = (
        f"stale cache at {artifact_path} (kind={artifact_kind}): {reason}. "
        f"Delete {artifact_path} (and its sidecar) OR pass matching cfg."
    )
    if action_if_stale == "raise":
        raise FileExistsError(msg)
    print(f"  [cache] WARNING: {msg}")
    return False
