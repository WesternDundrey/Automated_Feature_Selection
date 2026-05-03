"""
Cascade stage 2: deterministic + cheap filters on Haiku candidates.

Reduces ~3000 Haiku candidates → ~1500 clean candidates that Opus-judge
(stage 3) can read efficiently. Free deterministic gates:

  1. catalog_quality lexical scan (existing hard-fail patterns:
     vague phrases, context-level predicates, prefix-undecidable
     constructs)
  2. User-requested word blacklist (context, political, financial,
     official, function word, common word, noun phrase, relative
     clause, grammatical head, etc.)
  3. Description length cap (>25 words → almost always over-broad)
  4. Multi-concept bundle detection (POS bundles like "noun or verb";
     long item lists like "Like/Perhaps/Maybe/Why/...")
  5. Boundary-discipline minimum (must have ≥2 positive_examples +
     ≥2 negative_examples)
  6. Local self-consistency check for "Token is X" / "Token begins
     with X" patterns (catches Haiku hallucinations like
     "first-person pronoun" with example "<<Like>>")
  7. Lexical dedup (normalize description → group identicals → keep
     shortest canonical)

Output:
  feature_candidates_filtered.json — survivors, ready for Opus-judge
  filter_candidates_report.json — per-reason drop counts + IDs

Run with: python -m pipeline.run --step filter-candidates
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .catalog_quality import _lexical_scan, _word_count
from .config import Config


# User-requested additional blacklist (beyond catalog_quality's set).
# These are descriptions that signal context-level / register / domain
# orientation rather than token-local properties — they're flagged here
# because Haiku produces them readily even after the prompt asks for
# token-level features.
_USER_HARD_FAIL_PATTERNS = [
    re.compile(r"\bcontext(?:ual|s)?\b", re.IGNORECASE),
    re.compile(r"\bpolitic(?:al|s)?\b", re.IGNORECASE),
    re.compile(r"\bfinancial\b", re.IGNORECASE),
    re.compile(r"\bofficial\b", re.IGNORECASE),
    re.compile(r"\bhigh[- ]?information\b", re.IGNORECASE),
    re.compile(r"\blow[- ]?information\b", re.IGNORECASE),
    re.compile(r"\bfunction word\b", re.IGNORECASE),
    re.compile(r"\bcommon word\b", re.IGNORECASE),
    re.compile(r"\bnoun phrase\b", re.IGNORECASE),
    re.compile(r"\brelative clause\b", re.IGNORECASE),
    re.compile(r"\bgrammatical (?:head|role|category)\b", re.IGNORECASE),
    re.compile(r"\bsemantic (?:role|domain|category|field)\b", re.IGNORECASE),
    re.compile(r"\b(?:formal|informal) (?:writing|prose|register)\b", re.IGNORECASE),
]


# POS terms — descriptions with ≥2 of these joined by "or" are bundles.
_POS_TERMS = [
    "noun", "verb", "adjective", "adverb", "pronoun", "preposition",
    "determiner", "conjunction", "interjection", "auxiliary",
    "modal", "participle", "gerund",
]


def _is_pos_bundle(desc: str) -> bool:
    """Detect MULTI-POS bundles: ≥2 distinct part-of-speech terms.

    Drops things like "Token is a noun or verb in a clause" — these
    have no shared variable; the latent is genuinely polysemantic
    across POS categories and Opus can't rewrite it into a coherent
    feature.

    Does NOT drop list-bundles like "Like/Perhaps/Maybe/Why/Now/So":
    those frequently share a residual-stream direction (e.g.
    "discourse-start after punctuation") and Opus-judge can REWRITE
    them as the shared-variable description for an FVE-rich feature.
    Hence the asymmetry — POS bundles drop here; list bundles pass to
    Opus.
    """
    desc_l = desc.lower()
    hits = sum(1 for term in _POS_TERMS if re.search(rf"\b{term}\b", desc_l))
    return hits >= 2


def _is_list_bundle(desc: str) -> bool:
    """Detect long list-bundles for tagging (NOT for dropping).

    Returns True for descriptions like "Like/Perhaps/Maybe/Why/Now/So"
    or "Token is one of: a, b, c, d, e". These are FLAGGED for Opus
    judge to examine carefully — the latent likely fires on a shared
    variable the items have in common (discourse-start, sentence-
    initial uppercase, etc.) and Opus should rewrite the description
    as that shared variable.

    Threshold > 4 items because surface families ("comma or period",
    "open or close paren") have only 2 items and shouldn't be flagged.
    """
    parts = re.split(r"\s*,\s*|\s+or\s+|/|;", desc)
    nontrivial = [p.strip() for p in parts if len(p.strip()) > 1]
    return len(nontrivial) > 4


def _normalize_description(desc: str) -> str:
    """Lexical normalization for dedup: lowercase, strip punctuation,
    sort tokens. Two descriptions hash identically iff they're token
    permutations of the same word set."""
    s = re.sub(r"[^\w\s]", " ", desc.lower())
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    # Drop very common stop tokens that don't change meaning
    stop = {"the", "a", "an", "is", "are", "of", "in", "to"}
    tokens = [t for t in tokens if t not in stop]
    return " ".join(sorted(tokens))


def _extract_highlighted(example: str) -> str | None:
    """Pull the <<token>> highlight out of an example. Returns None
    if no highlight marker is present."""
    if not isinstance(example, str):
        return None
    m = re.search(r"<<([^>]+)>>", example)
    return m.group(1).strip() if m else None


def _check_self_consistency(cand: dict) -> tuple[bool, str | None]:
    """Local self-consistency rules for the most common Haiku patterns.

    Catches descriptions like "first-person pronoun" with example
    "<<Like>>" — where the highlighted token visibly violates the
    description. We can't catch all violations locally (some require
    semantic judgment) but the surface-pattern checks catch the
    cheapest hallucinations for free.

    Returns (is_consistent, reason). True if no rule applies or all
    rules pass.
    """
    desc = cand.get("description", "")
    pos_examples = cand.get("positive_examples", []) or []

    # Pattern A: "Token is 'X'" / "Token is the word 'X'" / 'Token is X.'
    m = re.match(
        r"^Token is (?:the word )?['\"`]?([A-Za-z][A-Za-z0-9\-']*)['\"`]?\s*[\.,!]?\s*$",
        desc.strip(),
    )
    if m:
        target = m.group(1).lower()
        for ex in pos_examples[:3]:
            highlighted = _extract_highlighted(str(ex))
            if highlighted is None:
                continue  # no marker; can't check
            if highlighted.lower().strip("'\".,!?") != target:
                return False, (
                    f"'Token is {target}' but highlighted "
                    f"'{highlighted[:30]}'"
                )

    # Pattern B: "Token begins with X" / "Token starts with X"
    m = re.match(
        r"^Token (?:begins?|starts?) with ['\"`]?([A-Za-z0-9\-']+)['\"`]?",
        desc.strip(),
        re.IGNORECASE,
    )
    if m:
        prefix = m.group(1).lower()
        for ex in pos_examples[:3]:
            highlighted = _extract_highlighted(str(ex))
            if highlighted is None:
                continue
            if not highlighted.lower().lstrip("'\"` ").startswith(prefix):
                return False, (
                    f"'Token begins with {prefix}' but highlighted "
                    f"'{highlighted[:30]}'"
                )

    # Pattern C: "Token contains X" — highlighted token must contain X
    m = re.match(
        r"^Token contains ['\"`]?([A-Za-z0-9\-']+)['\"`]?",
        desc.strip(),
        re.IGNORECASE,
    )
    if m:
        substr = m.group(1).lower()
        for ex in pos_examples[:3]:
            highlighted = _extract_highlighted(str(ex))
            if highlighted is None:
                continue
            if substr not in highlighted.lower():
                return False, (
                    f"'Token contains {substr}' but highlighted "
                    f"'{highlighted[:30]}' doesn't"
                )

    return True, None


def run(cfg: Config | None = None) -> dict:
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print("FILTER-CANDIDATES  (v8.21 cascade stage 2: hard gates + dedup)")
    print("=" * 70)

    in_path = cfg.output_dir / "feature_candidates_raw.json"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Need {in_path}. Run --step propose-haiku first."
        )
    raw = json.loads(in_path.read_text())
    candidates = raw.get("candidates", [])
    n_in = len(candidates)
    n_latents = raw.get("n_latents_seen", "?")
    print(f"  Loaded: {n_in} raw candidates from {n_latents} latents")
    print(f"  Source: {raw.get('model', 'unknown')}")

    survivors: list[dict] = []
    drops_by_reason: dict[str, list[str]] = defaultdict(list)

    for cand in candidates:
        cid = cand.get("id", "?")
        desc = cand.get("description", "").strip()
        if not desc:
            drops_by_reason["empty_description"].append(cid)
            continue

        # 1. catalog_quality lexical scan
        hard, _soft = _lexical_scan(desc, legacy_prompts=False)
        if hard:
            drops_by_reason[f"catalog_quality:{hard[0][:30]}"].append(cid)
            continue

        # 2. User-requested additional blacklist
        user_hit = None
        for pat in _USER_HARD_FAIL_PATTERNS:
            m = pat.search(desc)
            if m:
                user_hit = m.group(0)
                break
        if user_hit:
            drops_by_reason[f"user_blacklist:{user_hit}"].append(cid)
            continue

        # 3. Word-count cap
        if _word_count(desc) > 25:
            drops_by_reason["too_long_>25_words"].append(cid)
            continue

        # 4. POS-bundle detection (drops "noun or verb" but NOT
        #    "Like/Perhaps/Maybe/...", which may be FVE-rich shared-
        #    variable features for Opus to rewrite).
        if _is_pos_bundle(desc):
            drops_by_reason["pos_bundle"].append(cid)
            continue
        # List-bundle: don't drop, but flag for Opus judge attention.
        if _is_list_bundle(desc):
            cand = dict(cand)  # don't mutate input
            cand["_list_bundle_flag"] = True

        # 5. Boundary-discipline minimum
        n_pos_ex = len(cand.get("positive_examples", []) or [])
        n_neg_ex = len(cand.get("negative_examples", []) or [])
        if n_pos_ex < 2:
            drops_by_reason["lt_2_positive_examples"].append(cid)
            continue
        if n_neg_ex < 2:
            drops_by_reason["lt_2_negative_examples"].append(cid)
            continue

        # 6. Local self-consistency
        ok, reason = _check_self_consistency(cand)
        if not ok:
            drops_by_reason[f"self_consistency:{(reason or '')[:40]}"].append(cid)
            continue

        survivors.append(cand)

    n_after_gates = len(survivors)
    print(f"  After hard gates: {n_after_gates} / {n_in} "
          f"({100 * n_after_gates / max(1, n_in):.1f}%)")

    # 7. Lexical dedup
    norm_groups: dict[str, list[dict]] = defaultdict(list)
    for cand in survivors:
        norm = _normalize_description(cand["description"])
        norm_groups[norm].append(cand)

    deduped: list[dict] = []
    n_dedup_drops = 0
    for norm, group in norm_groups.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue
        # Canonical = shortest description, tie-break alphabetical
        canonical = min(group, key=lambda c: (len(c["description"]), c["description"]))
        deduped.append(canonical)
        for c in group:
            if c is canonical:
                continue
            n_dedup_drops += 1
            drops_by_reason["lexical_dedup"].append(
                f"{c['id']} (kept {canonical['id']})"
            )

    n_after_dedup = len(deduped)
    print(f"  After lexical dedup: {n_after_dedup} / {n_after_gates} "
          f"(dropped {n_dedup_drops} dups)")

    # Top drop reasons summary
    sorted_drops = sorted(drops_by_reason.items(), key=lambda kv: -len(kv[1]))
    print(f"\n  Top drop reasons:")
    for reason, ids in sorted_drops[:15]:
        print(f"    {len(ids):>5}  {reason}")

    # Save outputs
    out = {
        "source": "filter_candidates",
        "n_input": n_in,
        "n_after_hard_gates": n_after_gates,
        "n_after_dedup": n_after_dedup,
        "n_dropped": n_in - n_after_dedup,
        "drops_by_reason_count": {k: len(v) for k, v in drops_by_reason.items()},
        "candidates": deduped,
    }
    out_path = cfg.output_dir / "feature_candidates_filtered.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {out_path}")

    report_path = cfg.output_dir / "filter_candidates_report.json"
    report_path.write_text(json.dumps({
        "n_input": n_in,
        "n_output": n_after_dedup,
        "drops_by_reason": dict(drops_by_reason),
    }, indent=2))
    print(f"  Saved: {report_path}")

    print(f"\n  Next stage (not yet shipped):")
    print(f"    --step opus-judge → final ~500-feature catalog")
    print(f"  Until then: feature_candidates_filtered.json is ready to")
    print(f"  feed manually into --step opus-catalog or for hand-review.")

    return out
