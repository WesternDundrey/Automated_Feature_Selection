"""
Strict-rewrite of `feature_catalog.json` into atomic descriptions
with explicit positive / negative / exclusion examples.

Why: reviewer feedback called out that the dominant cause of mediocre
cal_F1 is FUZZY DESCRIPTIONS — not annotator capacity, not SAE capacity.
A description like "token belongs to a syntactic_construction expressing
a relative clause" is technically defensible but operationally
ambiguous: a competent annotator gives different answers across passes
on a sentence like "the man who came" because the boundary is unclear.

The rewrite step asks a strong model (Sonnet by default) to:

  1. Rewrite each leaf's description into a SINGLE-TOKEN-PROPERTY clause
     that a reader can answer yes/no for without ambiguity.
  2. Generate 3-5 POSITIVE example phrases — short context with the
     target token wrapped in **bold** — to ground the description.
  3. Generate 3-5 NEGATIVE examples (similar but should NOT fire), so
     the boundary is explicit.
  4. List 1-3 EXCLUSIONS naming similar features the description must
     NOT match.

The new fields go into the catalog without breaking existing pipelines:
the original `description` is preserved verbatim (renamed to
`description_legacy`) and the new atomic version becomes the active
description if `--apply-rewrite` is passed. Without `--apply-rewrite`
we write `feature_catalog.rewritten.json` and leave the active catalog
alone so the user can audit before committing.

If `annotations.pt` + `tokens.pt` exist from a previous run, the rewrite
samples up to 5 annotator-positive contexts per feature as GROUNDING
in the Sonnet prompt. This dramatically improves rewrite quality: the
model sees what was actually labeled positive and can sharpen the
boundary against those concrete examples. Without grounding, the
rewrite uses the description text alone (works, but riskier on vague
inputs).

Usage:
    # Inspect first (no in-place change)
    python -m pipeline.run --step rewrite-catalog

    # Apply (back up + replace active catalog; re-run annotate + train
    # for downstream impact)
    python -m pipeline.run --step rewrite-catalog --apply-rewrite

    # Skip features already rewritten
    python -m pipeline.run --step rewrite-catalog --rewrite-skip-existing
"""

from __future__ import annotations

import json
import re
import textwrap
import time
from pathlib import Path

import numpy as np

from .config import Config


def _has_been_rewritten(feature: dict) -> bool:
    """Heuristic: a feature is "already rewritten" if it carries the new
    fields. Used by --rewrite-skip-existing for incremental runs."""
    return all(
        k in feature
        for k in ("description_atomic", "positive_examples", "exclusions")
    )


def _sibling_summary(feature: dict, all_features: list[dict]) -> str:
    """Build a short listing of sibling features in the same parent group.
    Sonnet uses this to pick a description that's distinguishable from its
    siblings (otherwise it tends to write descriptions that overlap)."""
    parent_id = feature.get("parent")
    if not parent_id:
        return "  (no siblings — this is a top-level feature)"
    sibs = [
        f for f in all_features
        if f.get("parent") == parent_id and f.get("id") != feature["id"]
        and f.get("type") == "leaf"
    ]
    if not sibs:
        return "  (no other leaves in this group)"
    lines = [f"  - {s['id']}: {s.get('description', '')}" for s in sibs[:12]]
    if len(sibs) > 12:
        lines.append(f"  - ... and {len(sibs) - 12} more")
    return "\n".join(lines)


def _parent_description(feature: dict, all_features: list[dict]) -> str:
    parent_id = feature.get("parent")
    if not parent_id:
        return "(top-level)"
    for f in all_features:
        if f.get("id") == parent_id:
            return f.get("description", "(undocumented group)")
    return "(parent not in catalog)"


def _sample_positive_contexts(
    feature_idx: int,
    annotations: "np.ndarray | None",
    tokens: "np.ndarray | None",
    tokenizer,
    n: int,
    seed: int,
) -> list[str]:
    """Sample up to n positive (annotator=1) contexts for a feature, render
    each as a markdown-quoted phrase with the target token in **bold**."""
    if annotations is None or tokens is None:
        return []
    pos_mask = annotations[..., feature_idx] > 0
    if not pos_mask.any():
        return []
    n_pos = int(pos_mask.sum())
    if n_pos == 0:
        return []
    # Sample deterministically (so reruns of the rewrite step are stable).
    rng = np.random.RandomState(seed + feature_idx)
    pos_idx = np.argwhere(pos_mask)
    if len(pos_idx) > n:
        choice = rng.choice(len(pos_idx), size=n, replace=False)
        pos_idx = pos_idx[choice]
    out: list[str] = []
    for n_seq, pos in pos_idx.tolist():
        seq = tokens[n_seq].tolist() if hasattr(tokens, "shape") else list(tokens[n_seq])
        start = max(0, pos - 8)
        end = min(len(seq), pos + 9)
        parts: list[str] = []
        for i in range(start, end):
            tok = tokenizer.decode([int(seq[i])])
            tok = tok.replace("\n", "\\n")
            if i == pos:
                parts.append(f"**{tok}**")
            else:
                parts.append(tok)
        out.append("".join(parts))
    return out


_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are auditing and rewriting a feature description for a supervised
    sparse autoencoder. The goal is a CRISP ATOMIC description: a reader
    should be able to look at one token in context and answer yes/no
    without ambiguity.

    GROUP CONTEXT:
      parent group:        {parent_id}
      parent description:  {parent_desc}

    SIBLING LEAVES IN THE SAME GROUP (your description must DISTINGUISH from these):
    {siblings}

    CURRENT FEATURE:
      id:           {feature_id}
      type:         {feature_type}
      description:  {description}

    ANNOTATOR-POSITIVE EXAMPLES (real corpus contexts where the annotator
    said this feature fires; the target token is **bolded**):
    {positives}

    STRICT TOKEN-LEVEL CONSTRAINT:

    The description must be a YES/NO question about ONE specific
    token. The SAE annotates one token at a time. Predicate is the
    TOKEN, never the text/sentence/paragraph/context/document/
    topic/register/genre/domain. Reformulate as a token-local
    property or skip.

    REJECT (predicate is wrong unit):
      - "Text is about politics"
      - "Sentence is in past tense"
      - "Context belongs to news"
      - "Register is informal"
    ACCEPT (predicate is the token):
      - "Token is the comma immediately after a person's surname"
      - "Token is the verb in a quote-attribution clause"

    YOUR JOB — produce all four:

    1. ATOMIC DESCRIPTION. One sentence. Names a single
       operationally-testable property of ONE token. No "or" / "and" /
       "sometimes". No "the text/sentence/context [verb]" structures.
       Specifies WHEN the feature fires (what token, in what context).

    2. POSITIVE EXAMPLES. 3-5 short phrases (5-10 words each) with
       the target token in **bold**. Each must clearly match the atomic
       description.

    3. NEGATIVE EXAMPLES. 3-5 phrases that look similar but should NOT
       fire (different concept, ambiguous boundary, sibling-feature
       territory). Target token in **bold**.

    4. EXCLUSIONS. 1-3 short clauses of the form "NOT X (because Y)"
       naming similar features that this description must distinguish
       from. Use sibling ids when they collide.

    Reply with EXACTLY one JSON object, no other text:

    {{
      "id": "{feature_id}",
      "description_atomic": "...",
      "positive_examples": ["...", "...", "..."],
      "negative_examples": ["...", "...", "..."],
      "exclusions": ["NOT ... (because ...)", "..."]
    }}
""")


def _extract_json_object(text: str) -> dict | None:
    """Extract the first balanced JSON object from text. Handles nested
    braces and string escapes — same approach as inventory.py."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _rewrite_one_feature(
    feature: dict, all_features: list[dict],
    positives: list[str], cfg: Config,
) -> dict | None:
    """One Sonnet round-trip per feature. Fails closed (returns None) on
    any LLM error or unparseable response — the caller skips and keeps
    the original description so a flaky rewrite never silently drops a
    feature."""
    from .llm import get_client, chat
    client = get_client()

    pos_block = (
        "\n".join(f"  • {p}" for p in positives)
        if positives else "  (none available — rewrite from description alone)"
    )

    prompt = _PROMPT_TEMPLATE.format(
        parent_id=feature.get("parent") or "(none)",
        parent_desc=_parent_description(feature, all_features),
        siblings=_sibling_summary(feature, all_features),
        feature_id=feature["id"],
        feature_type=feature.get("type", "leaf"),
        description=feature.get("description", "(no description)"),
        positives=pos_block,
    )

    last_err = None
    for attempt in range(3):
        try:
            text = chat(client, cfg.organization_model, prompt, max_tokens=900)
            obj = _extract_json_object(text)
            if obj is not None:
                return obj
            last_err = ValueError(
                f"Could not parse JSON. Response: {text[:200]!r}"
            )
        except Exception as e:
            last_err = e
        if attempt < 2:
            time.sleep(2 ** attempt)

    print(f"    rewrite failed for {feature['id']}: {last_err}")
    return None


def _validate_rewrite(rewrite: dict, feature_id: str) -> tuple[bool, str]:
    """Sanity-check Sonnet's output before merging into the catalog. We
    don't want a 1-character `description_atomic` or an empty examples
    list silently replacing a working feature."""
    if rewrite.get("id") != feature_id:
        return False, f"id mismatch ({rewrite.get('id')!r} vs {feature_id!r})"
    desc = rewrite.get("description_atomic")
    if not isinstance(desc, str) or len(desc.strip()) < 12:
        return False, f"description_atomic too short: {desc!r}"
    pos = rewrite.get("positive_examples")
    if not isinstance(pos, list) or len(pos) < 1:
        return False, "positive_examples must be a non-empty list"
    neg = rewrite.get("negative_examples")
    if not isinstance(neg, list):
        return False, "negative_examples must be a list"
    excl = rewrite.get("exclusions")
    if not isinstance(excl, list):
        return False, "exclusions must be a list"
    return True, "ok"


def run(
    cfg: Config = None,
    apply_to_disk: bool = False,
    skip_existing: bool = False,
) -> dict:
    if cfg is None:
        cfg = Config()

    if not cfg.catalog_path.exists():
        raise FileNotFoundError(
            f"Feature catalog not found at {cfg.catalog_path}."
        )
    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]

    leaves = [f for f in features if f.get("type") == "leaf"]
    print(f"  Catalog has {len(leaves)} leaves to rewrite "
          f"(of {len(features)} total entries; groups are skipped)")

    # ── Optional grounding: positives from annotations.pt ──
    annotations = None
    tokens = None
    tokenizer = None
    if cfg.annotations_path.exists() and cfg.tokens_path.exists():
        try:
            import torch
            from transformers import AutoTokenizer
            ann_t = torch.load(cfg.annotations_path, weights_only=True)
            tok_t = torch.load(cfg.tokens_path, weights_only=True)
            annotations = ann_t.numpy()
            tokens = tok_t.numpy()
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            print(f"  Grounding from annotations.pt ({annotations.shape}) "
                  f"+ tokens.pt ({tokens.shape}) — Sonnet will see annotator-"
                  f"positive examples for each feature")
        except Exception as e:
            print(f"  Could not load grounding ({type(e).__name__}: {e}); "
                  f"falling back to description-only rewrites")
            annotations = tokens = tokenizer = None
    else:
        print(f"  No annotations.pt + tokens.pt available — "
              f"rewriting from description text alone")

    # Build feature_id → annotations index map. If annotations exists, the
    # feature ordering in the catalog matches the annotation column ordering
    # (enforced by the annotations_meta.json sidecar). If we have a meta
    # sidecar use it; otherwise fall back to positional alignment.
    feature_to_ann_col: dict[str, int] = {}
    if annotations is not None:
        if cfg.annotations_meta_path.exists():
            try:
                meta = json.loads(cfg.annotations_meta_path.read_text())
                cached_ids = meta.get("feature_ids") or []
                feature_to_ann_col = {fid: i for i, fid in enumerate(cached_ids)}
            except json.JSONDecodeError:
                print("  WARNING: annotations_meta.json is malformed; "
                      "falling back to positional alignment")
        if not feature_to_ann_col:
            feature_to_ann_col = {f["id"]: i for i, f in enumerate(features)}

    # ── Per-feature rewrite ──
    rewrites: dict[str, dict] = {}
    skipped_existing = 0
    failed: list[str] = []
    n_done = 0

    for f in leaves:
        if skip_existing and _has_been_rewritten(f):
            skipped_existing += 1
            continue

        positives: list[str] = []
        if annotations is not None and tokenizer is not None:
            ann_col = feature_to_ann_col.get(f["id"])
            if ann_col is not None:
                positives = _sample_positive_contexts(
                    feature_idx=ann_col,
                    annotations=annotations,
                    tokens=tokens,
                    tokenizer=tokenizer,
                    n=5,
                    seed=cfg.seed,
                )

        n_done += 1
        print(f"  [{n_done}/{len(leaves)}] {f['id']}  "
              f"({len(positives)} grounding examples)")

        rewrite = _rewrite_one_feature(f, features, positives, cfg)
        if rewrite is None:
            failed.append(f["id"])
            continue

        ok, why = _validate_rewrite(rewrite, f["id"])
        if not ok:
            print(f"    invalid rewrite ({why}); skipping")
            failed.append(f["id"])
            continue

        rewrites[f["id"]] = rewrite

        # Polite pacing — same as inventory.explain_features
        time.sleep(0.2)

    print(f"\n  Rewrote {len(rewrites)} of {len(leaves)} leaves "
          f"({len(failed)} failed, {skipped_existing} skipped as already-rewritten)")
    if failed:
        print(f"  Failed: {failed[:10]}{' ...' if len(failed) > 10 else ''}")

    # ── Merge rewrites into the catalog (mutating a copy) ──
    out_features: list[dict] = []
    for f in features:
        if f.get("type") != "leaf" or f["id"] not in rewrites:
            out_features.append(dict(f))
            continue
        r = rewrites[f["id"]]
        merged = dict(f)
        merged["description_legacy"] = f.get("description", "")
        # Use the atomic description as the active one when the rewrite is
        # accepted. Downstream pipelines (annotate/train/eval) read
        # `feature["description"]` — switching its content here is what makes
        # the rewrite actually take effect. The original is preserved under
        # `description_legacy` so we can roll back without re-running Sonnet.
        merged["description"] = r["description_atomic"]
        merged["description_atomic"] = r["description_atomic"]
        merged["positive_examples"] = r["positive_examples"]
        merged["negative_examples"] = r.get("negative_examples", [])
        merged["exclusions"] = r["exclusions"]
        out_features.append(merged)

    out_catalog = dict(catalog)
    out_catalog["features"] = out_features

    rewritten_path = cfg.output_dir / "feature_catalog.rewritten.json"
    rewritten_path.write_text(json.dumps(out_catalog, indent=2))
    print(f"\n  Rewritten catalog written to {rewritten_path}")

    # v8.18.21: re-run the quality validator on the rewritten catalog
    # (rewrite can introduce broadness, vague exclusions, or "or" in
    # unexpected ways). Per user's note: "Run the same validator post-
    # rewrite too, because rewrite can introduce or, broadness, or
    # vague exclusions." Report-only by default — doesn't overwrite the
    # rewritten file. User decides whether to drop based on the report.
    gate_mode = getattr(cfg, "catalog_gate_mode", "quarantine")
    if gate_mode != "off":
        try:
            from .catalog_quality import (
                apply_catalog_gates, write_quality_report,
            )
            print(f"\n  [catalog-quality post-rewrite] applying gates "
                  f"(mode=report — rewrite output is auditable, not auto-dropped)")
            _, _, post_records = apply_catalog_gates(
                out_catalog, cfg=cfg, mode="report",
                use_llm_crispness=getattr(cfg, "catalog_gate_use_llm", True),
            )
            write_quality_report(
                post_records,
                cfg.output_dir / "catalog_quality_report_post_rewrite.json",
                mode="report",
            )
        except Exception as e:
            print(f"  [catalog-quality post-rewrite] error "
                  f"({type(e).__name__}: {e}); skipping.")

    if apply_to_disk:
        backup_path = cfg.output_dir / "feature_catalog.before_rewrite.json"
        if not backup_path.exists():
            backup_path.write_text(json.dumps(catalog, indent=2))
            print(f"  Backup of pre-rewrite catalog: {backup_path}")
        cfg.catalog_path.write_text(json.dumps(out_catalog, indent=2))
        print(f"  Catalog REPLACED at {cfg.catalog_path}")
        print(f"  IMPORTANT: existing annotations.pt encodes the OLD "
              f"descriptions. To benefit from the rewrite, delete "
              f"annotations.pt + the associated meta sidecar and re-run "
              f"--step annotate so the annotator scores against the "
              f"atomic descriptions instead.")

    return {
        "n_rewritten": len(rewrites),
        "n_failed": len(failed),
        "n_skipped_existing": skipped_existing,
        "rewritten_catalog_path": str(rewritten_path),
        "applied": apply_to_disk,
    }
