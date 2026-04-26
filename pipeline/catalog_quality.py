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

# Hard-fail phrases — only the genuinely vague / non-operational
# patterns AND context-level predicates that have no token-level answer.
#
# v8.18.24 audit fix: dropped `such as`, `for example`, `including`,
# `etc.` from hard-fail. Per user: "Token is a unit suffix including
# km, cm, or mm" is a legitimate token-local description with examples.
# Those phrases go to soft-flag → LLM crispness now.
#
# v8.18.22: added context-level guards — the SAE annotates one token
# at a time, so descriptions whose predicate is the text / sentence /
# paragraph / context / document / topic / register / genre / domain
# can't be answered token-locally.
_HARD_FAIL_PATTERNS = [
    # Operationally-undefined / vague (these are unambiguously bad).
    re.compile(r"\bsometimes\b", re.IGNORECASE),
    re.compile(r"\bvarious\b", re.IGNORECASE),
    re.compile(r"\bin general\b", re.IGNORECASE),
    re.compile(r"\bgenerally\b", re.IGNORECASE),
    re.compile(r"\brelated to\b", re.IGNORECASE),
    re.compile(r"\bassociated with\b", re.IGNORECASE),
    re.compile(r"\bone of\b", re.IGNORECASE),
    re.compile(r"\band/or\b", re.IGNORECASE),
    # Context-level predicate (subject is text/sentence/etc., not token).
    # Article is optional because descriptions sometimes start sentence-
    # initial with the subject ("Text presents an argument..." with no
    # "the").
    re.compile(
        r"(?:^|\b)(?:(?:the|this|a)\s+)?"
        r"(?:text|sentence|paragraph|passage|context|document|article|discourse|register)"
        r"\s+(?:is|are|presents|discusses|describes|contains|gives|introduces|provides|involves|expresses|refers to)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bcontext (?:is|belongs to|consists of|involves|contains|describes)\b", re.IGNORECASE),
    re.compile(r"\btext (?:about|discussing|presenting|describing|involving|introducing|covering)\b", re.IGNORECASE),
    re.compile(r"\b(?:topic|subject|theme|register|genre) of\b", re.IGNORECASE),
    re.compile(r"\b(?:semantic|text)[_ -]?(?:domain|register|genre)\b", re.IGNORECASE),
    re.compile(r"\bdiscourse[_ -]?function\b", re.IGNORECASE),
    re.compile(r"\b(?:formal|informal|conversational|argumentative|expository|scientific|technical)\s+(?:text|prose|register|writing|tone)\b", re.IGNORECASE),
]

# Soft-flag patterns. Trigger a Sonnet crispness call (which may pass
# or fail the feature) rather than auto-rejecting. Surface variants
# like "opening or closing bracket" are legitimate; "noun or verb" is
# not. The Sonnet judgment makes that call.
#
# v8.18.24: moved `such as`, `for example`, `including`, `etc.` here
# from hard-fail. They sometimes indicate non-exhaustive enumeration
# (bad) and sometimes just illustrate (fine — "unit suffix including
# km, cm, or mm" is token-local). Let the LLM decide.
_SOFT_FLAG_PATTERNS = [
    re.compile(r"\bor\b", re.IGNORECASE),
    re.compile(r"\bany\b", re.IGNORECASE),
    re.compile(r"\beither\b", re.IGNORECASE),
    re.compile(r"\bmultiple\b", re.IGNORECASE),
    re.compile(r"\bsuch as\b", re.IGNORECASE),
    re.compile(r"\bfor example\b", re.IGNORECASE),
    re.compile(r"\bincluding\b", re.IGNORECASE),
    re.compile(r"\betc\.", re.IGNORECASE),
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


_SOURCE_LATENTS_EXEMPT_ROLES = {"control"}
_SOURCE_LATENTS_EXEMPT_KINDS = {"scaffold", "manual", "decomposed_atom"}


def _is_source_latents_exempt(feature: dict) -> bool:
    """Per user's note: scaffold / manual / control features are
    hand-written and don't have source_latents. Don't hard-fail them
    on this check. Inventory-derived leaves (no role/kind set) MUST
    have source_latents."""
    if feature.get("role") in _SOURCE_LATENTS_EXEMPT_ROLES:
        return True
    if feature.get("source_kind") in _SOURCE_LATENTS_EXEMPT_KINDS:
        return True
    return False


def _has_source_latents(feature: dict) -> bool:
    """Strictly: feature has at least one source latent ID. Returns
    True for exempt features (caller never reaches the check)."""
    src = feature.get("source_latents")
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
    is_src_exempt = _is_source_latents_exempt(feature)
    src_ok = is_src_exempt or _has_source_latents(feature)
    has_pos, has_neg, has_excl = _has_examples_metadata(feature)

    # Hard-fail conditions — operationally-undefined or vague phrases
    # the description shouldn't contain at all.
    if hard_fail_phrases:
        findings.append(
            f"hard_fail_phrases: {hard_fail_phrases}"
        )
    # Inventory-derived leaves (no role exemption, no scaffold/manual
    # provenance) MUST have source_latents per the v8.18 organize_hierarchy
    # prompt change. Scaffold/manual/control are exempt.
    if not src_ok:
        findings.append("missing_source_latents")
    if wc > _MAX_DESCRIPTION_WORDS:
        findings.append(f"description_too_long ({wc} words)")

    # Soft flags + LLM crispness (v8.18.24 corrected logic per audit).
    # Pre-fix: soft-flag finding was added BEFORE the LLM verdict, so
    # even if Sonnet said "crisp", the soft-flag finding still pushed
    # the status to quarantine. Under --catalog-gate-mode hard, valid
    # surface variants like "opening or closing quotation mark" got
    # dropped just because "or" was lexically present.
    # Now: ask the LLM first; only emit soft_flag finding if the LLM
    # confirms a real issue (or is unreachable). LLM=crisp → no
    # quarantine finding from soft flags.
    crispness_category: str | None = None
    crispness_reason: str | None = None
    has_soft_signal = bool(soft_flag_phrases) or wc > 20

    if has_soft_signal:
        if use_llm_crispness and cfg is not None:
            try:
                from .promote_loop import _crispness_judgment
                is_crisp, reason, category = _crispness_judgment(description, cfg)
                crispness_category = category
                crispness_reason = reason
                if is_crisp:
                    # LLM cleared it. Don't emit a soft-flag finding —
                    # the surface tokens were a false alarm.
                    pass
                elif category in (
                    "multi_concept", "vague", "too_broad", "not_token_local",
                    "uninterpretable", "nuisance"
                ):
                    findings.append(
                        f"crispness_judgment_fail: {category} ({reason[:80]})"
                    )
                else:
                    # LLM didn't categorize cleanly — keep soft flags
                    # as quarantine signal, surface what we saw.
                    findings.append(
                        f"soft_flags_uncategorized: {soft_flag_phrases}"
                        + (f", word_count={wc}" if wc > 20 else "")
                    )
            except Exception as e:
                # LLM unreachable → can't clear soft flags. Quarantine
                # is the safe default; user can re-run when LLM is back.
                findings.append(
                    f"crispness_judgment_unreachable: {type(e).__name__}"
                )
                findings.append(
                    f"soft_flags: {soft_flag_phrases}"
                    + (f", word_count={wc}" if wc > 20 else "")
                )
        else:
            # No LLM available → can't clear, so soft flags stand.
            findings.append(
                f"soft_flags: {soft_flag_phrases}"
                + (f", word_count={wc}" if wc > 20 else "")
            )

    # Decide status from findings.
    # `crispness_judgment_fail` is hard-fail because the LLM explicitly
    # judged the description bad. Soft flags only quarantine if the LLM
    # didn't clear them (unreachable OR uncategorized) — see the soft-
    # flag block above for the cleared/uncleared logic.
    has_hard_fail_finding = any(
        f.startswith("hard_fail_phrases")
        or f.startswith("crispness_judgment_fail")
        or f == "missing_source_latents"
        or f == "empty_description"
        for f in findings
    )
    has_quarantine_finding = any(
        f.startswith("description_too_long")
        or f.startswith("soft_flags")  # only present when LLM didn't clear
        or f.startswith("soft_flags_uncategorized")
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
) -> tuple[dict, dict, list[dict]]:
    """Apply quality gates to a catalog. Returns
    (filtered_catalog, quarantined_catalog, per_feature_records).

    `mode`:
        - "hard":       drop both fail and quarantine from filtered;
                        write all dropped to quarantined_catalog.
        - "quarantine": drop fail only from filtered; quarantined_catalog
                        contains the dropped fails.
        - "report":     drop nothing; quarantined_catalog still lists
                        anything that would have been quarantined under
                        "quarantine" mode for visibility.

    The quarantined_catalog is written alongside the main catalog so
    dropped features are auditable. Per the user's note: "Hard fail
    lexical patterns should mean quarantine, not irreversible deletion."
    """
    if mode not in ("hard", "quarantine", "report"):
        raise ValueError(f"unknown mode {mode!r}")

    features = catalog.get("features") or []
    leaves = [f for f in features if f.get("type") == "leaf"]
    groups = [f for f in features if f.get("type") != "leaf"]

    records: list[dict] = []
    record_by_id: dict[str, dict] = {}
    for f in leaves:
        rec = assess_feature_quality(
            f, use_llm_crispness=use_llm_crispness, cfg=cfg,
        )
        records.append(rec)
        record_by_id[rec["id"]] = rec

    if mode == "report":
        kept_ids = {f["id"] for f in leaves}
    elif mode == "hard":
        kept_ids = {r["id"] for r in records if r["status"] == "pass"}
    else:  # quarantine — drop only fails
        kept_ids = {r["id"] for r in records if r["status"] != "fail"}

    kept_leaves: list[dict] = []
    quarantined_leaves: list[dict] = []
    for f in leaves:
        rec = record_by_id.get(f["id"], {})
        # Annotate every feature with its quality status so downstream
        # tooling can filter on it without re-running the validator.
        f_with_status = dict(f)
        f_with_status["quality_status"] = rec.get("status", "unknown")
        f_with_status["quality_findings"] = rec.get("findings", [])
        if f["id"] in kept_ids:
            kept_leaves.append(f_with_status)
        else:
            quarantined_leaves.append(f_with_status)

    filtered = dict(catalog)
    filtered["features"] = groups + kept_leaves

    quarantined = dict(catalog)
    quarantined["features"] = quarantined_leaves
    quarantined["_note"] = (
        "Features dropped by catalog_quality gates. Auditable: "
        "review quality_findings per feature, manually re-add to the "
        "main feature_catalog.json if you disagree with the gate."
    )

    return filtered, quarantined, records


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
