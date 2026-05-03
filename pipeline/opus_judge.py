"""
Cascade stage 3: Opus 4.7 as strict judge over filtered Haiku candidates.

Reads feature_candidates_filtered.json (post-stage-2). Sends the
candidate pool (descriptions + boundary metadata) to Opus 4.7 in a
SINGLE call (no chunking — Opus's 1M context handles ~2000 candidates
easily, and chunking is what caused the v8.19 dups).

Opus outputs ID-level decisions only:
  - which candidates to keep (haiku_id list)
  - per-kept candidate: final id, parent, rewritten description
  - new symmetry-completion features (full metadata)
  - group structure

Python then merges: for each selected, copy the candidate's existing
positive_examples / negative_examples / exclusions / source_latents
with Opus's chosen id, parent, and rewritten description. Symmetry-
completion features pass through with `source_kind: "symmetry"`.

This shape keeps Opus output under its 64K max_tokens cap (500
selections × ~50 tokens + ~50 symmetry × ~400 tokens ≈ 45K), and
avoids re-asking Opus to invent metadata Haiku already provided.

Output:
  feature_catalog.json — final ~500-feature catalog, standard schema
  opus_judge_record.json — Opus's raw decisions + diff vs candidates

Run with: python -m pipeline.run --step opus-judge
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

from .config import Config


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_judge_prompt(
    candidates: list[dict], n_target: int, cfg: Config,
) -> str:
    """Single-shot Opus judge prompt. Compact candidate rendering keeps
    input tokens down so 2000+ candidates fit in 1M context."""

    # Compact candidate rendering: id + description + 3 pos + 3 neg +
    # ≤2 exclusions. Skip per-candidate top-activations (Haiku already
    # absorbed those at proposal time).
    lines = []
    for c in candidates:
        cid = c.get("id", "?")
        desc = c.get("description", "")
        pos = c.get("positive_examples", []) or []
        neg = c.get("negative_examples", []) or []
        exc = c.get("exclusions", []) or []
        block = [f"=== {cid} ==="]
        block.append(f"description: {desc}")
        if pos:
            block.append("positive: " + " | ".join(str(p)[:80] for p in pos[:3]))
        if neg:
            block.append("negative: " + " | ".join(str(p)[:80] for p in neg[:3]))
        if exc:
            block.append("exclusions: " + " | ".join(str(p)[:60] for p in exc[:2]))
        lines.append("\n".join(block))
    candidates_text = "\n\n".join(lines)

    return textwrap.dedent(f"""\
        You are the JUDGE in a two-stage catalog generation pipeline for a
        supervised sparse autoencoder on {cfg.model_name}, layer
        {cfg.target_layer}. Stage 1 (Haiku) proposed candidate
        descriptions with high recall but uneven precision — including
        candidates whose positive_examples visibly violate their own
        descriptions. Your job is to pick the {n_target} best, rewrite
        their descriptions for crispness, organize them into groups, add
        symmetry-completion features that the pool missed, AND repair
        candidates whose Haiku metadata is broken.

        CRISPNESS / LENGTH BUDGET:
          - Each rewritten description: single sentence ≤ 10 words.
          - Final id ≤ 24 chars (e.g. "punctuation.comma", not
            "punctuation_type.contrastive_dialogue_comma").
          - Form preference: "Token is X", "Token immediately follows X",
            "Token begins with X", "Token contains X", "Token is the
            word X". Drop candidates whose meaning doesn't fit one of
            these forms cleanly.

        METADATA VALIDATION (CRITICAL):
        For every candidate you put in `selected`, you are inheriting its
        positive_examples / negative_examples / exclusions UNCHANGED. Only
        `selected` a candidate if you have verified all 3-5 positive
        examples actually match your rewritten description, AND the
        negative examples are still genuine boundary cases. If even ONE
        positive example contradicts the description (e.g., description
        says "Token is a first-person pronoun" but a positive example
        is "<<Like>>"), the candidate's metadata is broken. In that case:
          OPTION A: drop the candidate entirely.
          OPTION B: put a CORRECTED version in `new_features` with full
                    fresh positive_examples + negative_examples +
                    exclusions you author. Use this only if the
                    underlying concept is worth keeping.
        Do NOT inherit metadata you haven't verified.

        FVE-RICH BUNDLES (NEW — important):
        Long item lists in Haiku descriptions like "Like/Perhaps/Maybe/
        Why/Now/So" or "Token is one of: a, b, c, d, e, f" often share a
        common residual-stream direction (e.g., "token starts a new
        discourse unit after sentence punctuation", "token is sentence-
        initial uppercase"). Do NOT drop these on sight. Either:
          (a) Rewrite the description as the SHARED VARIABLE: e.g.,
              "Token starts a new clause after punctuation" — then put
              in `selected` IF Haiku's examples actually demonstrate
              this shared property, OR `new_features` with cleaner
              examples otherwise.
          (b) Split into 2-3 semantically clean leaves under a parent
              group: e.g., discourse_start.maybe_perhaps,
              discourse_start.now_so, discourse_start.and_or — each
              with its own metadata. Use this if option (a) loses too
              much information.
        Bundles with NO coherent shared variable (e.g., a literal random
        list of unrelated tokens) should be dropped.

        STRICT REJECTS (drop the candidate even if Haiku proposed it):
          - Context-level predicates: "text is", "passage about",
            "article discusses", "paragraph contains".
          - Genre/register/domain: "formal writing", "political", "
            financial", "official document".
          - True POS bundles (e.g., "noun or verb") with no shared
            position/surface variable.
          - Right-context-dependent: "sentence-final", "subject of",
            "introduces a clause", "before a noun".
          - Vague predicates: "various", "associated with", "may",
            "sometimes", "high-information", "common word".

        REQUIRED SYMMETRY FAMILIES — propose missing values as
        new_features (full metadata, source_kind="symmetry"):
          colors:        green, blue, red, yellow, orange, purple, black,
                         white, brown, pink, gray
          days_of_week:  monday … sunday (7 leaves)
          months:        january … december (12 leaves)
          digits:        0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (10 leaves)
          directions:    north, south, east, west (4 leaves)
          pronoun_subj:  i, you, he, she, we, they, it
          pronoun_obj:   me, you, him, her, us, them
          punctuation:   comma, period, question_mark, exclamation,
                         semicolon, colon, dash, ellipsis,
                         open_paren, close_paren, open_bracket,
                         close_bracket, open_quote, close_quote,
                         apostrophe
          modal_verbs:   can, could, will, would, shall, should, may,
                         might, must

        For each symmetry family, EVERY value must be a leaf in the final
        catalog. If a candidate covers a value, select it; otherwise add
        the value as a new_features entry.

        OUTPUT: reply with ONLY this JSON, no other text. Total of
        kept-from-haiku + new-symmetry must equal exactly {n_target}.

        {{
          "groups": [
            {{
              "id": "punctuation",
              "description": "Punctuation tokens"
            }},
            {{ "id": "discovery", "description": "Discovery features" }}
          ],
          "selected": [
            {{
              "from_haiku_id": "haiku.cand_00000",
              "id": "discovery.state_of_the_capital",
              "parent": "discovery",
              "description": "Token follows 'State of the' and is capitalized"
            }}
          ],
          "new_features": [
            {{
              "id": "digits.7",
              "parent": "digits",
              "description": "Token is the digit 7",
              "positive_examples": ["...>>7<<...", "...>>7<<...", "...>>7<<..."],
              "negative_examples": ["...>>seven<<...", "...>>17<<...", "...>>0<<..."],
              "exclusions": ["NOT the spelled word 'seven'", "NOT multi-digit numbers"]
            }}
          ]
        }}

        TARGET: exactly {n_target} total leaves across selected +
        new_features. If fewer good candidates exist, fall back to
        symmetry-completion features. Quality over quantity if forced
        — drop selected entries before lowering quality.

        CANDIDATE POOL (filtered Haiku candidates, n={len(candidates)}):

        {candidates_text}
        """)


def _parse_judge_response(text: str) -> dict:
    """Parse Opus's JSON response. Tolerates code-fence wrapping."""
    if not text:
        raise RuntimeError("empty Opus judge response")
    m = _JSON_OBJECT_RE.search(text.strip())
    if m is None:
        raise RuntimeError(
            f"no JSON object found in Opus response. First 200 chars:\n"
            f"{text[:200]}"
        )
    return json.loads(m.group(0))


def _post_validate_inherited(
    description: str, examples: list[str], kind: str,
) -> tuple[list[str], list[str]]:
    """Filter inherited examples by local self-consistency rules.

    Returns (kept_examples, dropped_examples). For "Token is X" /
    "Token begins with X" / "Token contains X" descriptions, examples
    whose highlighted token violates the rule are dropped. Examples
    without a `<<token>>` marker pass through (we can't check them
    locally).

    `kind` is "positive" or "negative" — for negative examples, the
    rule is INVERTED (highlighted token should NOT match). E.g., a
    negative example for "Token is a comma" should highlight a
    non-comma token.
    """
    from .filter_candidates import _extract_highlighted

    desc = description.strip()
    kept: list[str] = []
    dropped: list[str] = []

    # Pattern A: "Token is X"
    m_a = re.match(
        r"^Token is (?:the word )?['\"`]?([A-Za-z][A-Za-z0-9\-']*)['\"`]?\s*[\.,!]?\s*$",
        desc,
    )
    # Pattern B: "Token begins with X" / "Token starts with X"
    m_b = re.match(
        r"^Token (?:begins?|starts?) with ['\"`]?([A-Za-z0-9\-']+)['\"`]?",
        desc, re.IGNORECASE,
    )
    # Pattern C: "Token contains X"
    m_c = re.match(
        r"^Token contains ['\"`]?([A-Za-z0-9\-']+)['\"`]?",
        desc, re.IGNORECASE,
    )

    for ex in examples:
        ex_str = str(ex)
        highlighted = _extract_highlighted(ex_str)
        if highlighted is None:
            kept.append(ex_str)  # no marker; can't check, give benefit of doubt
            continue
        h_clean = highlighted.lower().strip("'\"`., ")

        violated = False
        if m_a:
            target = m_a.group(1).lower()
            matches_target = (h_clean == target)
            violated = (matches_target != (kind == "positive"))
        elif m_b:
            prefix = m_b.group(1).lower()
            matches_prefix = h_clean.startswith(prefix)
            violated = (matches_prefix != (kind == "positive"))
        elif m_c:
            substr = m_c.group(1).lower()
            matches_substr = (substr in h_clean)
            violated = (matches_substr != (kind == "positive"))

        if violated:
            dropped.append(ex_str)
        else:
            kept.append(ex_str)

    return kept, dropped


def _merge_into_catalog(
    parsed: dict, candidates_by_id: dict[str, dict],
) -> dict:
    """Build final feature_catalog.json from Opus's decisions + the
    original candidates' boundary metadata.

    Applies POST-VALIDATION on inherited examples: for `selected`
    entries (which inherit Haiku metadata under Opus's rewritten
    description), the local self-consistency rules re-check each
    example. Examples that visibly violate the rewrite are dropped.
    Features that fall below the boundary-discipline minimum (≥2
    positive + ≥2 negative examples surviving) are dropped from the
    final catalog with the reason logged.
    """
    features: list[dict] = []
    drop_log: list[dict] = []
    n_dropped_unknown_id = 0
    n_dropped_post_validate = 0
    n_examples_pruned = 0

    # Groups first (top-level)
    seen_group_ids: set[str] = set()
    for g in parsed.get("groups", []) or []:
        gid = str(g.get("id", "")).strip()
        if not gid or gid in seen_group_ids:
            continue
        seen_group_ids.add(gid)
        features.append({
            "id": gid,
            "description": str(g.get("description", "")).strip(),
            "type": "group",
            "parent": None,
        })

    # Selected candidates: copy Haiku's metadata, post-validate
    # against Opus's rewritten description.
    for sel in parsed.get("selected", []) or []:
        haiku_id = sel.get("from_haiku_id")
        cand = candidates_by_id.get(haiku_id)
        if cand is None:
            n_dropped_unknown_id += 1
            drop_log.append({
                "from_haiku_id": haiku_id,
                "reason": "unknown_haiku_id",
            })
            continue

        new_desc = str(sel.get("description", "")).strip()
        raw_pos = list(cand.get("positive_examples", []) or [])
        raw_neg = list(cand.get("negative_examples", []) or [])

        kept_pos, dropped_pos = _post_validate_inherited(
            new_desc, raw_pos, "positive"
        )
        kept_neg, dropped_neg = _post_validate_inherited(
            new_desc, raw_neg, "negative"
        )
        n_examples_pruned += len(dropped_pos) + len(dropped_neg)

        # Boundary-discipline survival floor: ≥2 pos AND ≥2 neg.
        if len(kept_pos) < 2 or len(kept_neg) < 2:
            n_dropped_post_validate += 1
            drop_log.append({
                "from_haiku_id": haiku_id,
                "opus_id": str(sel.get("id", "")).strip(),
                "rewrite": new_desc,
                "n_pos_kept": len(kept_pos),
                "n_neg_kept": len(kept_neg),
                "n_pos_dropped": len(dropped_pos),
                "n_neg_dropped": len(dropped_neg),
                "reason": "post_validation_below_boundary_minimum",
            })
            continue

        features.append({
            "id": str(sel.get("id", "")).strip(),
            "description": new_desc,
            "type": "leaf",
            "parent": str(sel.get("parent", "")).strip() or None,
            "source_latents": list(cand.get("source_latents", [])
                                   or [cand.get("source_latent", -1)]),
            "source_kind": "haiku_filtered",
            "from_haiku_id": haiku_id,
            "positive_examples": kept_pos,
            "negative_examples": kept_neg,
            "exclusions": list(cand.get("exclusions", []) or []),
        })

    # New features: trust Opus's metadata (Opus authored them fresh).
    # Optional post-validate for safety.
    for nf in parsed.get("new_features", []) or []:
        new_desc = str(nf.get("description", "")).strip()
        raw_pos = list(nf.get("positive_examples", []) or [])
        raw_neg = list(nf.get("negative_examples", []) or [])

        kept_pos, dropped_pos = _post_validate_inherited(
            new_desc, raw_pos, "positive"
        )
        kept_neg, dropped_neg = _post_validate_inherited(
            new_desc, raw_neg, "negative"
        )

        if len(kept_pos) < 2 or len(kept_neg) < 2:
            n_dropped_post_validate += 1
            drop_log.append({
                "opus_id": str(nf.get("id", "")).strip(),
                "rewrite": new_desc,
                "n_pos_kept": len(kept_pos),
                "n_neg_kept": len(kept_neg),
                "reason": "new_feature_post_validation_below_minimum",
            })
            continue

        features.append({
            "id": str(nf.get("id", "")).strip(),
            "description": new_desc,
            "type": "leaf",
            "parent": str(nf.get("parent", "")).strip() or None,
            "source_latents": [],
            "source_kind": str(nf.get("source_kind", "symmetry")),
            "positive_examples": kept_pos,
            "negative_examples": kept_neg,
            "exclusions": list(nf.get("exclusions", []) or []),
        })

    return {
        "source": "opus_judge",
        "n_features": len(features),
        "n_groups": sum(1 for f in features if f["type"] == "group"),
        "n_leaves": sum(1 for f in features if f["type"] == "leaf"),
        "n_dropped_unknown_haiku_id": n_dropped_unknown_id,
        "n_dropped_post_validate": n_dropped_post_validate,
        "n_examples_pruned": n_examples_pruned,
        "features": features,
        "drop_log": drop_log,
    }


def run(cfg: Config | None = None) -> dict:
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print("OPUS-JUDGE  (v8.21 cascade stage 3: select + rewrite + complete)")
    print("=" * 70)

    in_path = cfg.output_dir / "feature_candidates_filtered.json"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Need {in_path}. Run --step filter-candidates first."
        )
    filtered = json.loads(in_path.read_text())
    candidates = filtered.get("candidates", []) or []
    n_in = len(candidates)
    if n_in == 0:
        raise RuntimeError(
            f"{in_path} has 0 candidates after filtering. Re-run "
            f"--step propose-haiku and --step filter-candidates first."
        )
    print(f"  Loaded: {n_in} filtered candidates")

    # Resume guard: skip if record exists.
    record_path = cfg.output_dir / "opus_judge_record.json"
    if record_path.exists() and not getattr(cfg, "force", False):
        existing = json.loads(record_path.read_text())
        if existing.get("n_leaves", 0) > 0:
            print(f"  [resume] {record_path} exists with "
                  f"{existing.get('n_leaves')} leaves; skipping. "
                  f"Pass --force to regenerate.")
            cfg.catalog_path.write_text(json.dumps(existing, indent=2))
            return existing

    # Locked plan target — 500 features by default; opus_n_features
    # honors user override from CLI.
    n_target = int(getattr(cfg, "opus_n_features", 500) or 500)

    print(f"  Target features (selected + symmetry): {n_target}")

    prompt = _build_judge_prompt(candidates, n_target, cfg)
    print(f"  Prompt size: {len(prompt):,} chars")

    if len(prompt) > 1_900_000:
        raise RuntimeError(
            f"Prompt too large for Opus 4.7's 1M context "
            f"({len(prompt):,} chars > 1.9M budget). Reduce candidate "
            f"count via stricter filtering, or shorten per-candidate "
            f"render in _build_judge_prompt."
        )

    # Single-shot Opus call. Output budget 64K (max_tokens cap); the
    # ID-only output shape keeps total well under that.
    from .llm import get_client, chat
    client = get_client()
    model = getattr(
        cfg, "opus_explanation_model", "anthropic/claude-opus-4-7",
    )
    print(f"  Sending to {model}...")
    text = chat(client, model=model, prompt=prompt, max_tokens=64000)
    print(f"  Response: {len(text):,} chars")

    parsed = _parse_judge_response(text)
    n_groups = len(parsed.get("groups", []) or [])
    n_selected = len(parsed.get("selected", []) or [])
    n_new = len(parsed.get("new_features", []) or [])
    print(f"  Opus picked: {n_selected} selected + {n_new} symmetry = "
          f"{n_selected + n_new} leaves across {n_groups} groups")

    # Build final catalog by merging Opus's decisions with the
    # candidates' existing boundary metadata.
    candidates_by_id = {c["id"]: c for c in candidates if "id" in c}
    catalog = _merge_into_catalog(parsed, candidates_by_id)

    if catalog["n_dropped_unknown_haiku_id"] > 0:
        print(f"  WARNING: {catalog['n_dropped_unknown_haiku_id']} "
              f"selected entries referenced unknown haiku IDs (dropped). "
              f"Opus may have hallucinated IDs.")
    if catalog.get("n_dropped_post_validate", 0) > 0:
        print(f"  Post-validate dropped: "
              f"{catalog['n_dropped_post_validate']} features had "
              f"<2 valid positive or <2 valid negative examples after "
              f"local self-consistency check against Opus's rewrite.")
    if catalog.get("n_examples_pruned", 0) > 0:
        print(f"  Examples pruned (visible violations): "
              f"{catalog['n_examples_pruned']}")

    print(f"  Final catalog: {catalog['n_leaves']} leaves, "
          f"{catalog['n_groups']} groups")

    # Save
    cfg.catalog_path.write_text(json.dumps(catalog, indent=2))
    print(f"  Saved: {cfg.catalog_path}")

    record = dict(catalog)
    record["raw_opus_response"] = text
    record["raw_opus_parsed"] = parsed
    record_path.write_text(json.dumps(record, indent=2))
    print(f"  Saved: {record_path}")

    print(f"\n  Next: review feature_catalog.json, then proceed with "
          f"--step annotate.")
    return catalog
