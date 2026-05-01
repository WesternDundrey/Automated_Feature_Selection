"""
Opus 4.7 catalog designer (v8.19.0) — sup arm of the Delphi-vs-Opus
F1 head-to-head.

Inspects all `cfg.shortlist_size` candidate latents (top-activating
contexts) and designs `cfg.opus_n_features` features with the full
boundary-discipline contract. Uses Opus 4.7's 1M-context window so the
design pass can ingest top contexts of 1000 latents in a single call
batch (vs 8-15 groups / 30-80 leaves in legacy Sonnet-organize).

Crucially: Opus has FULL DESIGN FREEDOM. It can:
  - merge multiple shortlist latents into one feature
  - drop noisy latents
  - brainstorm symmetry-completing features (color families, days,
    months, digits, directions, comparatives, pronoun grids,
    punctuation classes) that aren't 1:1 with any single latent
  - rewrite descriptions for prefix-decidability and crispness

This selection-freedom asymmetry vs Delphi (which describes 300 latents
1:1) IS the methodology being tested — sup arm's whole pitch is "feature
designer picks what to model." Do not normalize away.

Output: pipeline_data/opus_catalog.json (separate from the v8.18
production feature_catalog.json so the existing artifact is preserved).

Reuses inventory.collect_top_activations + load_target_model + load_sae.
"""

from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

from .config import Config


def _render_latents_text(
    top_activations: dict, tokenizer, top_k: int
) -> str:
    """Render the latents block at a given per-latent context budget."""
    from .inventory import format_examples_for_prompt
    blocks = []
    for lat_idx, examples in sorted(top_activations.items(), key=lambda x: int(x[0])):
        if not examples:
            continue
        examples_str = format_examples_for_prompt(
            examples[:top_k], tokenizer
        )
        blocks.append(f"--- latent_{lat_idx} ---\n{examples_str}")
    return "\n\n".join(blocks)


def _build_design_prompt(top_activations: dict, tokenizer, cfg: Config) -> str:
    """Single-shot design prompt for Opus 4.7 (1M context).

    Includes top contexts of all shortlist latents + symmetry constraints
    + boundary-discipline contract. Output target: cfg.opus_n_features
    leaves with full positive_examples / negative_examples / exclusions.

    AUTO-DOWNSHIFT: at production shortlist sizes (1000 latents × 50
    contexts × ~30 tok/context ≈ 1.5M tokens), the prompt exceeds Opus
    4.7's 1M context window. Builds the full prompt at REQUESTED_TOP_K,
    measures actual size, and iteratively rebuilds at lower top_k until
    it fits the budget. Sample-based estimates were unreliable because
    the first latent's block size is not representative of the average.
    """

    # Budget: Opus 4.7 max context = 1,000,000 tokens. Empirically
    # (measured 2026-05-01 on a real run), Anthropic's tokenizer
    # produces ~2.4 chars/token on our prompts (the latent context
    # blocks are dense with markup like `[N] (act=X.XX) <ctx>` and
    # short subword tokens). Budget = (1M − 64K output − 50K scaffold
    # safety) × 2.4 chars/tok ≈ 2.1M chars. Use 1.9M to be safe.
    BUDGET_CHARS = 1_900_000
    REQUESTED_TOP_K = cfg.top_k_examples

    n_in = sum(1 for examples in top_activations.values() if examples)
    if n_in == 0:
        raise RuntimeError("No latents have activating examples; "
                           "shortlist might be all dead latents.")

    n_out = cfg.opus_n_features
    effective_top_k = REQUESTED_TOP_K
    full_prompt = ""

    # Iteratively rebuild until it fits. At most 6 iterations (each
    # halves the worst-case overshoot); converges in 1-2 passes typically.
    for attempt in range(6):
        latents_text = _render_latents_text(
            top_activations, tokenizer, effective_top_k
        )
        full_prompt = _assemble_prompt(latents_text, n_in, n_out, cfg)
        n_chars = len(full_prompt)
        if n_chars <= BUDGET_CHARS:
            if effective_top_k != REQUESTED_TOP_K:
                print(f"  Prompt-size auto-downshift CONVERGED at "
                      f"top_k={effective_top_k} (was {REQUESTED_TOP_K}); "
                      f"prompt = {n_chars:,} chars (budget {BUDGET_CHARS:,})")
            break
        # Compute new top_k. Use 0.92 safety factor because the
        # static scaffold doesn't scale with top_k, so the linear
        # ratio undershoots slightly.
        ratio = (BUDGET_CHARS / n_chars) * 0.92
        new_top_k = max(3, int(effective_top_k * ratio))
        if new_top_k >= effective_top_k:
            new_top_k = effective_top_k - 1
        if new_top_k < 3:
            raise RuntimeError(
                f"Cannot fit prompt under {BUDGET_CHARS:,}-char budget "
                f"even at top_k=3. Reduce shortlist_size or use a "
                f"smaller corpus / larger model."
            )
        print(f"  Prompt-size auto-downshift attempt {attempt + 1}: "
              f"top_k {effective_top_k} → {new_top_k}  "
              f"({n_chars:,} chars > {BUDGET_CHARS:,} budget)")
        effective_top_k = new_top_k
    else:
        # Loop completed without break — still over budget after 6 tries.
        raise RuntimeError(
            f"Auto-downshift failed to converge under {BUDGET_CHARS:,} "
            f"chars after 6 attempts (final top_k={effective_top_k}, "
            f"prompt={len(full_prompt):,} chars)."
        )

    return full_prompt


def _assemble_prompt(latents_text: str, n_in: int, n_out: int, cfg: Config) -> str:

    return textwrap.dedent(f"""\
        You are designing a supervised feature catalog for a sparse autoencoder
        on {cfg.model_name}, layer {cfg.target_layer}.

        Below are top-activating contexts for {n_in} unsupervised SAE latents.
        Token with highest activation is marked >>token<<. Your job is to
        design exactly {n_out} feature DESCRIPTIONS for a supervised SAE
        catalog.

        STRICT PREFIX-DECIDABLE TOKEN-LEVEL CONSTRAINT:

        Every leaf description must be a YES/NO QUESTION ABOUT ONE
        SPECIFIC TOKEN, decidable from ONLY the target token and the
        tokens before it (left context). The downstream annotator
        sees only the prefix up to and including the target token.

        REJECT (predicate is wrong unit, OR requires right-context /
        full parse / document-level):
          - "Text is about politics"                    (document-level)
          - "Sentence is in past tense"                 (document-level)
          - "Token is a sentence-final period"          (needs future tokens)
          - "Token is a determiner before a noun"       (needs future tokens)
          - "Token introduces a prepositional phrase"   (needs future tokens)
          - "Token is the subject of a clause"          (needs full parse)
          - "Token is part of a named entity"           (needs entity span)
          - "Token is a comma in a list separator"      (needs future tokens)

        ACCEPT (decidable from target token + left context only):
          - "Token is a comma"                                 (surface)
          - "Token is all lowercase"                           (surface)
          - "Token starts with uppercase"                      (surface)
          - "Token is the digit 7"                             (surface)
          - "Token is the color word 'green'"                  (surface)
          - "Token is the word 'said' after a quoted phrase"   (left context)
          - "Token immediately follows a comma"                (left context)
          - "Token appears after 'Mr.' or 'Dr.'"               (left context)

        CRISPNESS / LENGTH BUDGET (v8.19.6, scaling-run requirement):
          - Each leaf `description` MUST be a single sentence ≤ 10 words.
          - Leaf `id` MUST be ≤ 24 chars (e.g. "punctuation.comma", not
            "punctuation_type.contrastive_dialogue_comma").
          - The annotator suffix renders ONLY the description (exclusions
            are kept in catalog metadata for audit but are NOT sent to
            the annotator). Short descriptions = 4-5× annotation
            throughput at 500-feature scale.
          - If you can't say it in ≤ 10 words, the feature is too
            entangled — split it or drop it.

        SELECTION FREEDOM — you may:
          - merge multiple shortlist latents into one feature
          - drop noisy / polysemantic latents that don't yield a clean
            description
          - brainstorm SYMMETRY-COMPLETING features that aren't 1:1
            with any single latent. Required symmetric families:
              colors:        green, blue, red, yellow, orange, purple,
                             black, white, brown, pink, gray
              days_of_week:  monday … sunday (7 leaves)
              months:        january … december (12 leaves)
              digits:        0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (10 leaves)
              directions:    north, south, east, west (4 leaves)
              comparatives:  more, less, equal, greater, fewer
              pronoun_subj:  i, you, he, she, we, they, it
              pronoun_obj:   me, you, him, her, us, them
              punctuation:   comma, period, question_mark, exclamation,
                             semicolon, colon, dash, ellipsis,
                             open_paren, close_paren, open_bracket,
                             close_bracket, open_brace, close_brace,
                             open_quote, close_quote, apostrophe
              quantifiers:   all, some, none, many, few, several
              modal_verbs:   can, could, will, would, shall, should,
                             may, might, must
          - rewrite descriptions for clarity and prefix-decidability

        For each symmetric family above, EVERY value gets its own leaf
        even if the shortlist doesn't have a 1:1 latent for each value.
        Symmetry is what supSAE's design freedom buys vs unsup auto-interp.

        BOUNDARY DISCIPLINE — required for every leaf:

        - `positive_examples`: 3-5 short token-level examples (>>token<<
          markup). Show the feature's true scope.
        - `negative_examples`: 3-5 BOUNDARY-CASE tokens that share the
          surface form / context cue with positives but DO NOT fire.
          If you can't think of three boundary cases, drop the leaf.
        - `exclusions`: 1-3 short noun phrases naming what's excluded,
          used directly in the annotator's prompt.

        SOURCE TRACE — every leaf needs `source_latents`: list of integer
        latent IDs from the shortlist whose contexts grounded this leaf.
        For symmetry-completing features that have no direct latent
        grounding, use `source_latents: []` and add
        `source_kind: "symmetry"` (the catalog_quality validator
        recognizes this kind and exempts the leaf from the
        source_latents check, but boundary-discipline still applies —
        all leaves need positive_examples / negative_examples /
        exclusions).

        OUTPUT — reply with ONLY this JSON, no other text:
        {{
          "features": [
            {{
              "id": "group_name",
              "description": "Categorical dimension",
              "type": "group",
              "parent": null
            }},
            {{
              "id": "group_name.value_name",
              "description": "Precise operational token-level description",
              "type": "leaf",
              "parent": "group_name",
              "source_latents": [123, 456],
              "positive_examples": ["...>>tok<<...", "...>>tok<<...", "...>>tok<<..."],
              "negative_examples": ["...>>tok<<...", "...>>tok<<...", "...>>tok<<..."],
              "exclusions": ["short boundary phrase"]
            }}
          ]
        }}

        Target: exactly {n_out} leaves total across all groups. Symmetry-
        completing families count toward the target. Do not exceed {n_out}.
        Every leaf must satisfy the contract above; leaves missing any
        required field will be rejected by the catalog_quality validator.

        SHORTLIST LATENTS (top-activating contexts):

        {latents_text}
        """)


def run(cfg: Config = None) -> dict:
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print(f"OPUS 4.7 CATALOG DESIGN  (target {cfg.opus_n_features} features "
          f"from {cfg.shortlist_size}-latent shortlist)")
    print("=" * 70)

    # v8.19.5 resume: skip if a previous Opus catalog exists. Opus 4.7
    # tokens are the single most expensive thing in the pipeline ($10-15
    # per call); a failed downstream stage shouldn't force re-generation.
    record_path = cfg.output_dir / "opus_catalog.json"
    if record_path.exists() and not cfg.force:
        try:
            existing = json.loads(record_path.read_text())
            n_leaves = sum(
                1 for f in existing.get("features", [])
                if f.get("type") == "leaf"
            )
            if n_leaves > 0:
                print(f"  [resume] {record_path} exists with "
                      f"{n_leaves} leaves; skipping Opus call. "
                      f"Pass --force to regenerate.")
                # Refresh the canonical feature_catalog.json so
                # downstream --step annotate consumes it even if it
                # was overwritten between runs.
                cfg.catalog_path.write_text(json.dumps(existing, indent=2))
                return existing
        except Exception as e:
            print(f"  [resume] couldn't validate {record_path} "
                  f"({e}); regenerating.")

    from .inventory import (
        load_sae, load_target_model, collect_top_activations
    )
    from .shortlist_latents import load_shortlist
    from .llm import get_client, chat

    shortlist = load_shortlist(cfg)
    print(f"Loaded shortlist: {len(shortlist)} latents")

    sae, _ = load_sae(cfg)
    model = load_target_model(cfg)
    tokenizer = model.tokenizer
    print(f"Collecting top-{cfg.top_k_examples} contexts for "
          f"{len(shortlist)} latents over "
          f"{cfg.n_tokens_for_activation_collection:,} tokens...")
    top_activations = collect_top_activations(
        model, sae, tokenizer, shortlist, cfg
    )

    prompt = _build_design_prompt(top_activations, tokenizer, cfg)
    n_in_chars = len(prompt)
    print(f"Design prompt: {n_in_chars:,} chars (~{n_in_chars // 4:,} tokens)")
    # Hard guard: 1M context window; reserve 64K for max_tokens.
    # Use a slightly conservative limit to absorb tokenizer variance.
    MAX_INPUT_TOKENS = 920_000
    est_in_tokens = n_in_chars // 4
    if est_in_tokens > MAX_INPUT_TOKENS:
        raise RuntimeError(
            f"Design prompt estimated at {est_in_tokens:,} tokens "
            f"exceeds {MAX_INPUT_TOKENS:,}-token budget for "
            f"{cfg.opus_explanation_model} (1M context − 64K output). "
            f"Auto-downshift didn't recover; reduce shortlist_size or "
            f"top_k_examples. Prompt was {n_in_chars:,} chars at "
            f"effective top_k={cfg.top_k_examples}."
        )

    client = get_client()
    print(f"Calling {cfg.opus_explanation_model} (max_tokens=64000)...")

    catalog = None
    last_err = None
    text = None
    for attempt in range(3):
        try:
            text = chat(
                client, cfg.opus_explanation_model, prompt, max_tokens=64000,
            )
            from .inventory import _extract_json_object
            catalog = _extract_json_object(text)
            if catalog is not None and "features" in catalog:
                break
            preview = (text[:300] if isinstance(text, str)
                       else f"<non-string response: {type(text).__name__}>")
            last_err = ValueError(
                f"Could not parse catalog JSON. Begins: {preview}"
            )
        except Exception as e:
            last_err = e
            text = None
        if attempt < 2:
            print(f"  Attempt {attempt + 1} failed: {last_err}, retrying...")
            time.sleep(2 ** attempt)

    if catalog is None:
        raise RuntimeError(
            f"Opus catalog design failed after 3 attempts: {last_err}"
        )

    n_groups = sum(1 for f in catalog["features"] if f.get("type") == "group")
    n_leaves = sum(1 for f in catalog["features"] if f.get("type") == "leaf")
    print(f"\n  Designed: {n_groups} groups, {n_leaves} leaves "
          f"(target {cfg.opus_n_features})")

    # v8.19.0: validate Opus catalog through catalog_quality (lexical
    # hard-fail + boundary-discipline). Symmetry-completing leaves with
    # source_kind="symmetry" are exempted from the source_latents check
    # (no 1:1 unsup latent grounding) but still need positive_examples /
    # negative_examples / exclusions. Quarantined features are written
    # to a sidecar for audit and dropped from the saved catalog.
    try:
        from .catalog_quality import apply_catalog_gates, write_quality_report
        filtered, quarantined, records = apply_catalog_gates(
            catalog, cfg=cfg,
            mode=cfg.catalog_gate_mode,
            use_llm_crispness=cfg.catalog_gate_use_llm,
        )
        n_kept = sum(1 for f in filtered["features"] if f.get("type") == "leaf")
        n_quar = sum(
            1 for f in quarantined["features"] if f.get("type") == "leaf"
        )
        print(f"  catalog_quality: {n_kept} kept, {n_quar} quarantined "
              f"(mode={cfg.catalog_gate_mode!r})")
        catalog = filtered
        # Save the quarantined sidecar for audit.
        quar_path = cfg.output_dir / "opus_catalog.quarantined.json"
        quar_path.write_text(json.dumps(quarantined, indent=2))
        try:
            write_quality_report(
                records,
                cfg.output_dir / "opus_catalog_quality_report.json",
                mode=cfg.catalog_gate_mode,
            )
        except Exception:
            pass
    except Exception as e:
        # v8.19.2: respect --catalog-gate-strict. The validator's whole
        # job is to fail loudly; with strict mode on, propagate the
        # error so a long benchmark run aborts rather than silently
        # using an unfiltered catalog. Non-strict (default research
        # mode) prints a warning and continues.
        if cfg.catalog_gate_strict:
            raise RuntimeError(
                f"catalog_quality validator failed in strict mode "
                f"(--catalog-gate-strict): {e}"
            ) from e
        print(f"  WARNING: catalog_quality validation skipped: {e}")

    # v8.19.2 two-arm flow: write the named record (audit trail) AND
    # the canonical feature_catalog.json that downstream --step annotate
    # consumes. The user runs the sup arm in pipeline_data/ and the
    # unsup arm in pipeline_data_unsup/, so the canonical name is
    # arm-local — no merge step needed.
    record_path = cfg.output_dir / "opus_catalog.json"
    record_path.write_text(json.dumps(catalog, indent=2))
    cfg.catalog_path.write_text(json.dumps(catalog, indent=2))
    print(f"Saved: {record_path}")
    print(f"Saved: {cfg.catalog_path} (canonical, picked up by --step annotate)")
    return catalog
