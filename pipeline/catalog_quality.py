"""
Catalog quality gates: strict validator + quarantine + reporting.

Per the user's design: lexical flags trigger a strict crispness check
(via Sonnet, reusing promote_loop._crispness_judgment), they don't
auto-delete. Genuinely broken patterns (`sometimes`, `various`,
`general`, `associated with`, `one of`, `and/or`, hedging modals) are
hard-fail. Surface variants like `opening or closing quotation mark`
must reach Sonnet for a real call — they may be a single atomic
family, not a multi-concept bundle.

Output per feature:
  - status:   "pass" | "quarantine" | "fail"
  - score:    rough quality score in [0,1]
  - findings: list[str] of the specific checks the feature passed/failed

The pipeline can apply gates as:
  - mode="hard":       drop fail, drop quarantine, keep pass
  - mode="quarantine": drop fail, keep quarantine + pass (default)
  - mode="report":     keep everything, just write report

A `catalog_quality_report.json` always gets written under
`cfg.output_dir/`. The report has the per-feature record + summary
counts + the worst-rejected list so the user can audit.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# Hard-fail phrases. These are operationally-undefined. No surface
# variant of "sometimes" should be accepted — there's no token-level
# yes/no answer to "this feature SOMETIMES fires."
_HARD_FAIL_PATTERNS = [
    re.compile(r"\bsometimes\b", re.IGNORECASE),
    re.compile(r"\bvarious\b", re.IGNORECASE),
    re.compile(r"\bin general\b", re.IGNORECASE),
    re.compile(r"\bgenerally\b", re.IGNORECASE),
    re.compile(r"\brelated to\b", re.IGNORECASE),
    re.compile(r"\bassociated with\b", re.IGNORECASE),
    re.compile(r"\bone of\b", re.IGNORECASE),
    re.compile(r"\band/or\b", re.IGNORECASE),
    re.compile(r"\bmay (?:be|indicate|refer)\b", re.IGNORECASE),
    re.compile(r"\bmight (?:be|indicate|refer)\b", re.IGNORECASE),
    re.compile(r"\bcan be (?:any|either|various)\b", re.IGNORECASE),
    re.compile(r"\bsuch as\b", re.IGNORECASE),  # introduces non-exhaustive enumeration
    re.compile(r"\bfor example\b", re.IGNORECASE),
    re.compile(r"\bincluding\b", re.IGNORECASE),
    re.compile(r"\betc\.", re.IGNORECASE),
]

# Soft-flag patterns. Trigger a Sonnet crispness call (which may pass
# or fail the feature) rather than auto-rejecting. Surface variants
# like "opening or closing bracket" are legitimate; "noun or verb" is
# not. The Sonnet judgment makes that call.
_SOFT_FLAG_PATTERNS = [
    re.compile(r"\bor\b", re.IGNORECASE),
    re.compile(r"\bany\b", re.IGNORECASE),
    re.compile(r"\beither\b", re.IGNORECASE),
    re.compile(r"\bmultiple\b", re.IGNORECASE),
]

_MAX_DESCRIPTION_WORDS = 30  # over this length is almost always over-broad


def _lexical_scan(description: str) -> tuple[list[str], list[str]]:
    """Return (hard_fail_phrases, soft_flag_phrases) found in the
    description. The caller decides what to do with each list."""
    hard: list[str] = []
    soft: list[str] = []
    for pat in _HARD_FAIL_PATTERNS:
        m = pat.search(description)
        if m:
            hard.append(m.group(0))
    for pat in _SOFT_FLAG_PATTERNS:
        m = pat.search(description)
        if m:
            soft.append(m.group(0))
    return hard, soft


def _word_count(description: str) -> int:
    return len(description.split())


def _has_source_latents(feature: dict) -> bool:
    src = feature.get("source_latents")
    if src is None:
        return True   # not all features need source_latents (e.g. scaffold)
    return isinstance(src, list) and len(src) > 0


def _has_examples_metadata(feature: dict) -> tuple[bool, bool, bool]:
    """Check for v8.14 rewrite-catalog metadata: positive_examples,
    negative_examples, exclusions. Returns (has_pos, has_neg, has_excl)."""
    return (
        bool(feature.get("positive_examples")),
        bool(feature.get("negative_examples")),
        bool(feature.get("exclusions")),
    )


def assess_feature_quality(
    feature: dict,
    use_llm_crispness: bool = True,
    cfg=None,
) -> dict:
    """Assess one feature's quality. Returns a record:
        {
            "id": str,
            "status": "pass" | "quarantine" | "fail",
            "score": float in [0, 1],
            "findings": [str],
            "lexical_hard_fail": [str],
            "lexical_soft_flag": [str],
            "word_count": int,
            "has_source_latents": bool,
            "has_positive_examples": bool,
            "has_negative_examples": bool,
            "has_exclusions": bool,
            "crispness_category": str | None,  # only set when LLM check fired
        }
    """
    description = (feature.get("description") or "").strip()
    fid = feature.get("id", "?")
    findings: list[str] = []

    # Empty description → automatic fail.
    if not description:
        return {
            "id": fid,
            "status": "fail",
            "score": 0.0,
            "findings": ["empty_description"],
            "lexical_hard_fail": [],
            "lexical_soft_flag": [],
            "word_count": 0,
            "has_source_latents": _has_source_latents(feature),
            "has_positive_examples": False,
            "has_negative_examples": False,
            "has_exclusions": False,
            "crispness_category": None,
        }

    hard_fail_phrases, soft_flag_phrases = _lexical_scan(description)
    wc = _word_count(description)
    src_ok = _has_source_latents(feature)
    has_pos, has_neg, has_excl = _has_examples_metadata(feature)

    # Hard-fail conditions — operationally-undefined or vague phrases
    # the description shouldn't contain at all.
    if hard_fail_phrases:
        findings.append(
            f"hard_fail_phrases: {hard_fail_phrases}"
        )
    if not src_ok:
        findings.append("missing_source_latents")
    if wc > _MAX_DESCRIPTION_WORDS:
        findings.append(f"description_too_long ({wc} words)")

    # Soft flags trigger LLM crispness judgment if available.
    crispness_category: str | None = None
    crispness_reason: str | None = None
    if soft_flag_phrases or wc > 20:
        findings.append(
            f"soft_flags: {soft_flag_phrases}"
            + (f", word_count={wc}" if wc > 20 else "")
        )
        if use_llm_crispness and cfg is not None:
            try:
                from .promote_loop import _crispness_judgment
                is_crisp, reason, category = _crispness_judgment(description, cfg)
                crispness_category = category
                crispness_reason = reason
                if not is_crisp and category in (
                    "multi_concept", "vague", "too_broad", "not_token_local",
                    "uninterpretable", "nuisance"
                ):
                    findings.append(
                        f"crispness_judgment_fail: {category} ({reason[:80]})"
                    )
            except Exception as e:
                findings.append(
                    f"crispness_judgment_unreachable: {type(e).__name__}"
                )

    # Decide status from findings.
    has_hard_fail_finding = any(
        f.startswith("hard_fail_phrases")
        or f.startswith("crispness_judgment_fail")
        or f == "missing_source_latents"
        or f == "empty_description"
        for f in findings
    )
    has_quarantine_finding = any(
        f.startswith("description_too_long")
        or f.startswith("soft_flags")
        or f.startswith("crispness_judgment_unreachable")
        for f in findings
    )

    if has_hard_fail_finding:
        status = "fail"
    elif has_quarantine_finding:
        status = "quarantine"
    else:
        status = "pass"

    # Coarse score in [0, 1] for ordering. Pass = 1.0, quarantine ~0.5,
    # fail = 0.0; bonuses for richer metadata.
    score = (
        1.0 if status == "pass"
        else 0.5 if status == "quarantine"
        else 0.0
    )
    if has_pos and has_neg and has_excl:
        score += 0.05  # reward fully-rewritten features
    score = min(score, 1.0)

    return {
        "id": fid,
        "status": status,
        "score": round(score, 3),
        "findings": findings,
        "lexical_hard_fail": hard_fail_phrases,
        "lexical_soft_flag": soft_flag_phrases,
        "word_count": wc,
        "has_source_latents": src_ok,
        "has_positive_examples": has_pos,
        "has_negative_examples": has_neg,
        "has_exclusions": has_excl,
        "crispness_category": crispness_category,
        "crispness_reason": crispness_reason,
    }


def apply_catalog_gates(
    catalog: dict,
    cfg=None,
    mode: str = "quarantine",
    use_llm_crispness: bool = True,
) -> tuple[dict, list[dict]]:
    """Apply quality gates to a catalog. Returns (filtered_catalog,
    per_feature_records).

    `mode`:
        - "hard":       drop both fail and quarantine
        - "quarantine": drop fail only (default — never silently
                        deletes things on a soft flag)
        - "report":     drop nothing; emit findings only
    """
    if mode not in ("hard", "quarantine", "report"):
        raise ValueError(f"unknown mode {mode!r}")

    features = catalog.get("features") or []
    leaves = [f for f in features if f.get("type") == "leaf"]
    groups = [f for f in features if f.get("type") != "leaf"]

    records: list[dict] = []
    for f in leaves:
        rec = assess_feature_quality(
            f, use_llm_crispness=use_llm_crispness, cfg=cfg,
        )
        records.append(rec)

    if mode == "report":
        kept_leaves = leaves
    elif mode == "hard":
        keep_ids = {r["id"] for r in records if r["status"] == "pass"}
        kept_leaves = [f for f in leaves if f["id"] in keep_ids]
    else:  # quarantine — drop only fails
        keep_ids = {r["id"] for r in records if r["status"] != "fail"}
        kept_leaves = [f for f in leaves if f["id"] in keep_ids]

    out = dict(catalog)
    out["features"] = groups + kept_leaves
    return out, records


def write_quality_report(
    records: list[dict], out_path: Path, mode: str,
) -> dict:
    """Write the per-feature records + summary stats to disk and print
    a short summary to stdout. Returns the summary dict."""
    n_total = len(records)
    n_pass       = sum(1 for r in records if r["status"] == "pass")
    n_quarantine = sum(1 for r in records if r["status"] == "quarantine")
    n_fail       = sum(1 for r in records if r["status"] == "fail")

    # Sort fails first (worst score), then quarantine, then pass.
    sorted_records = sorted(records, key=lambda r: (r["score"], r["id"]))

    summary = {
        "mode": mode,
        "n_total":      n_total,
        "n_pass":       n_pass,
        "n_quarantine": n_quarantine,
        "n_fail":       n_fail,
        "records":      sorted_records,
    }
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n  [catalog-quality] STATUS=scored mode={mode}")
    print(f"    pass:       {n_pass}/{n_total}")
    print(f"    quarantine: {n_quarantine}/{n_total}")
    print(f"    fail:       {n_fail}/{n_total}")
    print(f"  Report: {out_path}")

    # Top-10 worst features so user can eyeball
    worst = [r for r in sorted_records if r["status"] != "pass"][:10]
    if worst:
        print(f"\n  Top-10 worst features (id  status  findings):")
        for r in worst:
            findings_str = "; ".join(r["findings"][:2])
            print(f"    {r['id']:<48}  {r['status']:<10}  {findings_str[:80]}")

    return summary
