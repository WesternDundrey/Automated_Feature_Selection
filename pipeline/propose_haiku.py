"""
Haiku candidate-feature proposer (v8.21 cascade architecture, stage 1/3).

Goal: high-recall coverage of pretrained SAE latents at low cost. Haiku
is the SEARCH ENGINE; Opus (in the planned `--step opus-judge`) is the
JUDGE that selects, rewrites, and completes families. This module ONLY
runs the Haiku-as-proposer step. The downstream filter + judge stages
are deferred to v8.21+.

Why this exists:
    Current `--step opus-catalog` uses Opus 4.7 directly on a 1000-latent
    shortlist via chunked calls with no cross-chunk coordination →
    catalog dups (postmortem §1: ~50 dup features in 239 leaves = 21%
    waste). Cascade architecture fixes this by separating proposal
    (high recall, cheap) from selection (high precision, expensive).

Design:
    For each latent:
      - Haiku sees top-K activating contexts (same render as Opus path).
      - Haiku proposes 1-3 candidate descriptions, each with the
        boundary-discipline schema (positive_examples, negative_examples,
        exclusions). Token-level + prefix-decidable per inventory's
        existing prompt rules.
      - Haiku may also output `SKIP` if the latent's contexts don't
        support a token-level description.

    Async fan-out via `pipeline.llm.get_async_client` so 10K latents
    don't run serially. Each request is independent; failures are
    logged + skipped without aborting the run.

Output:
    {output_dir}/feature_candidates_raw.json
        {
          "source": "haiku",
          "model": "anthropic/claude-haiku-4.5",
          "n_latents_seen": int,
          "n_candidates": int,
          "candidates": [
            {
              "id": "haiku.cand_0001",
              "source_latent": int,
              "description": str,        # token-level, prefix-decidable
              "positive_examples": [str],
              "negative_examples": [str],
              "exclusions": [str],
            },
            ...
          ],
          "skipped_latents": [int]
        }

This file is a JUNK-RICH proposal pool. Do NOT feed it directly into
annotation. The downstream cascade applies hard filters (regex hard-fails
from catalog_quality, embedding dedup, min-support feasibility) and then
Opus selection before annotation.

Run with: python -m pipeline.run --step propose-haiku
        [--propose-n-latents N] [--propose-candidates-per-latent K]
        [--haiku-proposer-model MODEL]
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from .config import Config


_PROMPT_HEADER = """\
You are analyzing a sparse autoencoder latent from a language model. \
You see the top-activating token contexts for ONE latent. The token \
with highest activation is marked with <<token>>. The number in \
parentheses is the activation strength.

Propose UP TO {k} candidate token-level descriptions of what this \
latent detects. Each candidate must be prefix-decidable: a yes/no \
question a reader can answer for ONE specific token using ONLY the \
target token's surface string and tokens BEFORE it. Do NOT propose \
descriptions that require future tokens, full-sentence parse, document \
topic, genre, register, or semantic domain.

ACCEPT examples (decidable from target + left context):
  - "Token is a semicolon"
  - "Token is all lowercase"
  - "Token starts with uppercase"
  - "Token immediately follows a comma"
  - "Token is a leading-space word starting with 'th'"

REJECT examples (require right context, full parse, or document-level):
  - "Token is a sentence-final period"      (needs future tokens)
  - "Token is a determiner before a noun"   (needs future tokens)
  - "Token is the subject of a clause"      (needs full parse)
  - "Text is about politics"                (predicate = text)
  - "Token appears in formal writing"       (register/genre)

For EACH candidate, also output:
  - positive_examples: 2-3 short snippets where the description fires
  - negative_examples: 2-3 short snippets where it should NOT fire
  - exclusions: bullet-list of look-alikes the annotator must NOT label

If NO prefix-decidable token-level description fits this latent, output \
exactly:

  {{"skip": true, "reason": "<short reason>"}}

Otherwise output STRICT JSON of this shape (no preface, no commentary):

{{
  "candidates": [
    {{
      "description": "Token is ...",
      "positive_examples": ["...", "..."],
      "negative_examples": ["...", "..."],
      "exclusions": ["NOT abbreviation periods", "NOT decimal points"]
    }}
  ]
}}

Top-activating contexts for latent_{lat_idx}:

"""


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_haiku_response(text: str) -> tuple[list[dict] | None, str | None]:
    """Return (candidates_list_or_None, skip_reason_or_None).

    Tolerates leading/trailing whitespace and minor preamble. Returns
    (None, reason) for explicit skips, ([], None) for empty candidate
    lists, ([{...}, ...], None) for proposed candidates. Raises on
    malformed JSON.
    """
    if not text:
        return [], None
    m = _JSON_OBJECT_RE.search(text.strip())
    if m is None:
        return [], None
    obj = json.loads(m.group(0))
    if obj.get("skip"):
        return None, str(obj.get("reason", "skip"))
    cands = obj.get("candidates", [])
    if not isinstance(cands, list):
        return [], None
    return cands, None


async def _propose_one(
    client, model: str, lat_idx: int, examples_str: str,
    candidates_per_latent: int, sem: asyncio.Semaphore,
) -> tuple[int, list[dict] | None, str | None]:
    """Single-latent Haiku call. Returns (lat_idx, candidates, skip_reason).

    On API/JSON failure returns ([], "error: {msg}") so caller can log
    without aborting the whole run.
    """
    prompt = _PROMPT_HEADER.format(
        k=candidates_per_latent, lat_idx=lat_idx,
    ) + examples_str
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            cands, skip = _parse_haiku_response(text)
            return lat_idx, cands, skip
        except Exception as e:
            return lat_idx, [], f"error: {e}"


async def _run_proposals(
    top_activations: dict, tokenizer, cfg: Config,
) -> dict:
    """Fan out Haiku per-latent calls and collect candidates."""
    from .llm import get_async_client
    from .inventory import format_examples_for_prompt

    model = getattr(
        cfg, "haiku_proposer_model", "anthropic/claude-haiku-4.5",
    )
    candidates_per_latent = int(
        getattr(cfg, "propose_candidates_per_latent", 3) or 3
    )
    n_latents_cap = int(getattr(cfg, "propose_n_latents", 0) or 0)
    concurrency = int(getattr(cfg, "propose_concurrency", 16) or 16)
    top_k = int(getattr(cfg, "top_k_examples", 30) or 30)

    items = sorted(
        ((int(k), v) for k, v in top_activations.items() if v),
        key=lambda kv: kv[0],
    )
    if n_latents_cap > 0:
        items = items[:n_latents_cap]

    print(f"  Haiku proposer: model={model}")
    print(f"  Latents to query:        {len(items)}")
    print(f"  Candidates per latent:   {candidates_per_latent}")
    print(f"  Concurrency:             {concurrency}")
    print(f"  Top-K contexts/latent:   {top_k}")

    client = get_async_client()
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for lat_idx, examples in items:
        examples_str = format_examples_for_prompt(
            examples[:top_k], tokenizer,
        )
        tasks.append(_propose_one(
            client, model, lat_idx, examples_str,
            candidates_per_latent, sem,
        ))

    candidates: list[dict] = []
    skipped: list[dict] = []
    errors: list[dict] = []
    completed = 0
    n_total = len(tasks)

    # gather() in batches so we can show progress and avoid one slow
    # request stalling the whole pipeline silently.
    batch_size = max(8, concurrency * 4)
    for batch_start in range(0, n_total, batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        results = await asyncio.gather(*batch, return_exceptions=False)
        for lat_idx, cands, skip_reason in results:
            completed += 1
            if skip_reason and skip_reason.startswith("error:"):
                errors.append({"latent": lat_idx, "reason": skip_reason})
                continue
            if skip_reason:
                skipped.append({"latent": lat_idx, "reason": skip_reason})
                continue
            if not cands:
                # Empty list from a non-skip response = effectively skip
                skipped.append({"latent": lat_idx, "reason": "empty_candidates"})
                continue
            for cand in cands[:candidates_per_latent]:
                cid = f"haiku.cand_{len(candidates):05d}"
                candidates.append({
                    "id": cid,
                    "source_latent": lat_idx,
                    "description": str(cand.get("description", "")).strip(),
                    "positive_examples": list(cand.get("positive_examples", []))[:5],
                    "negative_examples": list(cand.get("negative_examples", []))[:5],
                    "exclusions": list(cand.get("exclusions", []))[:5],
                })
        # Progress line per batch
        print(f"    [propose] {completed}/{n_total} latents  "
              f"candidates={len(candidates)}  "
              f"skipped={len(skipped)}  errors={len(errors)}")

    return {
        "source": "haiku",
        "model": model,
        "n_latents_seen": len(items),
        "n_candidates": len(candidates),
        "n_skipped": len(skipped),
        "n_errors": len(errors),
        "candidates_per_latent": candidates_per_latent,
        "candidates": candidates,
        "skipped_latents": skipped,
        "errors": errors,
    }


def _load_or_collect_top_activations(cfg: Config) -> tuple[dict, object]:
    """Load top_activations.json if it exists; otherwise compute it from
    the shortlist (which is what `--step shortlist` produced) using the
    same `inventory.collect_top_activations` path that `opus-catalog`
    uses internally. Persist the result so re-runs / other cascade
    stages can reuse it.

    Returns (top_activations_dict, tokenizer).
    """
    if cfg.top_activations_path.exists():
        print(f"  Loading cached top activations: {cfg.top_activations_path}")
        top_acts = json.loads(cfg.top_activations_path.read_text())
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        return top_acts, tokenizer

    # Need to compute. Requires a shortlist first.
    shortlist_path = cfg.output_dir / "latent_shortlist.json"
    if not shortlist_path.exists():
        raise FileNotFoundError(
            f"Need either {cfg.top_activations_path} (cached top "
            f"activations) or {shortlist_path} (latent shortlist). "
            f"Run `--step shortlist` first."
        )

    print(f"  No cached top_activations.json — computing from "
          f"{shortlist_path}...")
    from .inventory import (
        load_sae, load_target_model, collect_top_activations,
    )
    from .shortlist_latents import load_shortlist

    shortlist = load_shortlist(cfg)
    print(f"  Loaded shortlist: {len(shortlist)} latents")

    sae, _ = load_sae(cfg)
    model = load_target_model(cfg)
    tokenizer = model.tokenizer
    print(f"  Collecting top-{cfg.top_k_examples} contexts for "
          f"{len(shortlist)} latents over "
          f"{cfg.n_tokens_for_activation_collection:,} tokens...")
    top_activations = collect_top_activations(
        model, sae, tokenizer, shortlist, cfg,
    )

    # Persist for re-runs and downstream cascade stages.
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.top_activations_path.write_text(json.dumps(top_activations, indent=2))
    print(f"  Saved: {cfg.top_activations_path}")
    return top_activations, tokenizer


def run(cfg: Config | None = None) -> dict:
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print("PROPOSE-HAIKU  (v8.21 cascade stage 1: high-recall proposals)")
    print("=" * 70)

    top_activations, tokenizer = _load_or_collect_top_activations(cfg)

    out = asyncio.run(_run_proposals(top_activations, tokenizer, cfg))

    out_path = cfg.output_dir / "feature_candidates_raw.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {out_path}")
    print(f"  Candidates: {out['n_candidates']} from "
          f"{out['n_latents_seen']} latents "
          f"(skipped {out['n_skipped']}, errored {out['n_errors']})")
    print(f"\n  Next stages (v8.21, not yet shipped):")
    print(f"    --step filter-candidates  # regex + dedup + min-support")
    print(f"    --step opus-judge         # canonical selection + rewrite")
    print(f"\n  For now, manually inspect feature_candidates_raw.json and "
          f"feed surviving candidates into your existing flow.")
    return out
