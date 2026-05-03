"""
Cascade stage 3.5: LLM metadata validator (Sonnet 4.6).

User identified post-opus-judge quality issues that Python's local
self-consistency check misses: features whose positive examples don't
demonstrate the description (`after_closer` had opener tokens like
`<<(>>` as positives), inconsistent boundary cases (`for<<PlayStation>>`
listed as negative under "capitalization after preposition"), examples
mixing concepts (`period_abbr` had sentence-end periods alongside
abbreviation periods).

Local regex post-validation only catches "Token is X" / "Token begins
with X" / "Token contains X" surface patterns. Left-context-dependent
descriptions ("Token immediately follows a closer", "Token is
capitalized after a preposition") need semantic judgment.

This step runs Sonnet per-feature with a tight prompt: given the
description + examples, which positive examples actually demonstrate
the description, and which negatives are real boundary cases? Sonnet
outputs index lists. Python prunes accordingly. Features below the
≥2-pos-≥2-neg floor after pruning get dropped.

Output:
  feature_catalog.validated.json — pruned catalog (does NOT overwrite
                                   feature_catalog.json; user opts in)
  validation_report.json         — per-feature Sonnet verdicts +
                                   counts + drop log

Run with: python -m pipeline.run --step validate-catalog
        [--validator-model MODEL] [--validate-concurrency N]
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from .config import Config


_PROMPT_TEMPLATE = """\
You are validating metadata for a supervised SAE feature. The feature \
will label individual tokens at annotation time, and the annotator sees \
ONLY the prefix up to the target token (no future tokens). Examples \
mark the target token with `<<token>>`.

Feature id: {fid}
Description: {desc}

Positive examples (each SHOULD match the description for the \
highlighted token):
{pos_block}

Negative examples (each SHOULD NOT match the description; should be \
genuine boundary cases that look similar but fail one criterion):
{neg_block}

For each positive: does the highlighted token, given its left-context \
prefix, satisfy the description? (Don't peek at right context.)

For each negative: is it a genuine boundary case where a careful \
annotator might be tempted to label positive but shouldn't? (Or is \
it actually a positive example mislabeled, or unrelated noise?)

Output STRICT JSON, no preface, no commentary:

{{
  "valid": <true if at least 2 positive examples and 2 negative examples \
are correct, else false>,
  "bad_positive_indices": [<list of 0-based indices of positive examples \
that DON'T demonstrate the description>],
  "bad_negative_indices": [<list of 0-based indices of negative examples \
that ARE actually positives, or are off-topic>],
  "left_context_dependent": <true if description requires checking \
tokens before the target, which is fine; false if surface-only>,
  "reason": "<one sentence on what's wrong, or 'metadata clean' if \
nothing>"
}}
"""


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _format_examples(examples: list) -> str:
    if not examples:
        return "  (none)"
    lines = []
    for i, ex in enumerate(examples):
        s = str(ex).strip().replace("\n", " ")
        if len(s) > 200:
            s = s[:200] + "..."
        lines.append(f"  {i}. {s}")
    return "\n".join(lines)


def _parse_validator_response(text: str) -> dict | None:
    """Returns the parsed dict or None on parse failure."""
    if not text:
        return None
    m = _JSON_OBJECT_RE.search(text.strip())
    if m is None:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


async def _validate_one(
    client, model: str, feature: dict, sem: asyncio.Semaphore,
) -> tuple[str, dict | None, str | None]:
    """Returns (feature_id, parsed_json_or_None, error_or_None)."""
    fid = feature.get("id", "?")
    desc = feature.get("description", "")
    pos = feature.get("positive_examples", []) or []
    neg = feature.get("negative_examples", []) or []
    prompt = _PROMPT_TEMPLATE.format(
        fid=fid,
        desc=desc,
        pos_block=_format_examples(pos),
        neg_block=_format_examples(neg),
    )
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            parsed = _parse_validator_response(text)
            return fid, parsed, None
        except Exception as e:
            return fid, None, str(e)


async def _run_validation(
    leaves: list[dict], cfg: Config,
) -> dict[str, dict | None]:
    from .llm import get_async_client

    model = getattr(
        cfg, "catalog_validator_model", "anthropic/claude-sonnet-4.6",
    )
    concurrency = int(getattr(cfg, "validate_concurrency", 16) or 16)

    print(f"  Validator model: {model}")
    print(f"  Concurrency:     {concurrency}")
    print(f"  Features to validate: {len(leaves)}")

    client = get_async_client()
    sem = asyncio.Semaphore(concurrency)
    tasks = [_validate_one(client, model, f, sem) for f in leaves]

    verdicts: dict[str, dict | None] = {}
    errors: list[dict] = []
    completed = 0
    n_total = len(tasks)
    batch_size = max(8, concurrency * 4)
    for batch_start in range(0, n_total, batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        results = await asyncio.gather(*batch, return_exceptions=False)
        for fid, parsed, err in results:
            completed += 1
            if err is not None:
                errors.append({"id": fid, "error": err})
                verdicts[fid] = None
            else:
                verdicts[fid] = parsed
        print(f"    [validate] {completed}/{n_total} features  "
              f"errors={len(errors)}")
    return verdicts


def _apply_verdicts(
    catalog: dict, verdicts: dict[str, dict | None],
) -> tuple[dict, list[dict]]:
    """Return (pruned_catalog, drop_log)."""
    drop_log: list[dict] = []
    new_features: list[dict] = []
    n_examples_pruned = 0

    for f in catalog.get("features", []):
        if f.get("type") == "group":
            new_features.append(f)
            continue
        fid = f.get("id", "?")
        verdict = verdicts.get(fid)

        # Validator failure (API error / parse fail) → keep feature
        # untouched but log a warning. Don't drop on infrastructure.
        if verdict is None:
            drop_log.append({
                "id": fid,
                "kind": "validator_failed",
                "kept": True,
                "reason": "API/parse error; feature passed through unchecked",
            })
            new_features.append(f)
            continue

        bad_pos_idx = set(verdict.get("bad_positive_indices", []) or [])
        bad_neg_idx = set(verdict.get("bad_negative_indices", []) or [])
        valid_flag = bool(verdict.get("valid", False))

        old_pos = list(f.get("positive_examples", []) or [])
        old_neg = list(f.get("negative_examples", []) or [])

        new_pos = [ex for i, ex in enumerate(old_pos) if i not in bad_pos_idx]
        new_neg = [ex for i, ex in enumerate(old_neg) if i not in bad_neg_idx]

        n_pruned = (len(old_pos) - len(new_pos)) + (len(old_neg) - len(new_neg))
        n_examples_pruned += n_pruned

        # Boundary-discipline floor: ≥ 2 valid positives AND ≥ 2 valid negatives.
        if len(new_pos) < 2 or len(new_neg) < 2:
            drop_log.append({
                "id": fid,
                "kind": "below_boundary_floor",
                "kept": False,
                "n_pos_kept": len(new_pos),
                "n_neg_kept": len(new_neg),
                "n_pruned_total": n_pruned,
                "valid": valid_flag,
                "reason": verdict.get("reason", "")[:200],
            })
            continue

        # If validator marked invalid but enough examples survived, log
        # but keep — Sonnet's `valid` flag and the example survival count
        # disagree only when many examples were pruned but the floor still
        # holds. Trust the example-level judgment over the global flag.
        if not valid_flag and n_pruned > 0:
            drop_log.append({
                "id": fid,
                "kind": "invalid_but_floor_held",
                "kept": True,
                "n_pos_kept": len(new_pos),
                "n_neg_kept": len(new_neg),
                "n_pruned_total": n_pruned,
                "reason": verdict.get("reason", "")[:200],
            })

        f_new = dict(f)
        f_new["positive_examples"] = new_pos
        f_new["negative_examples"] = new_neg
        f_new["validator_left_context_dependent"] = bool(
            verdict.get("left_context_dependent", False)
        )
        new_features.append(f_new)

    pruned = dict(catalog)
    pruned["features"] = new_features
    pruned["validator"] = {
        "n_examples_pruned": n_examples_pruned,
        "n_dropped": sum(1 for d in drop_log if not d.get("kept", False)),
    }
    return pruned, drop_log


def run(cfg: Config | None = None) -> dict:
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print("VALIDATE-CATALOG  (v8.21 cascade stage 3.5: Sonnet metadata judge)")
    print("=" * 70)

    if not cfg.catalog_path.exists():
        raise FileNotFoundError(
            f"Need {cfg.catalog_path}. Run --step opus-judge or "
            f"--step opus-catalog first."
        )
    catalog = json.loads(cfg.catalog_path.read_text())
    leaves = [f for f in catalog.get("features", []) if f.get("type") == "leaf"]
    if not leaves:
        raise RuntimeError(f"{cfg.catalog_path} has 0 leaf features.")

    n_in = len(leaves)
    print(f"  Loaded: {n_in} leaves from {cfg.catalog_path}")

    verdicts = asyncio.run(_run_validation(leaves, cfg))

    pruned, drop_log = _apply_verdicts(catalog, verdicts)
    n_out = sum(1 for f in pruned["features"] if f.get("type") == "leaf")
    n_dropped = pruned["validator"]["n_dropped"]
    n_pruned_examples = pruned["validator"]["n_examples_pruned"]

    print(f"\n  Result:")
    print(f"    Leaves before: {n_in}")
    print(f"    Leaves after:  {n_out}")
    print(f"    Dropped:       {n_dropped}")
    print(f"    Examples pruned (kept feature, dropped bad ex): "
          f"{n_pruned_examples}")

    # Show the worst drops for quick audit
    real_drops = [d for d in drop_log if not d.get("kept", False)]
    if real_drops:
        print(f"\n  Top drops:")
        for d in real_drops[:10]:
            print(f"    {d['id']:<40}  "
                  f"pos_kept={d.get('n_pos_kept', '?')} "
                  f"neg_kept={d.get('n_neg_kept', '?')}  "
                  f"reason: {d.get('reason', '')[:70]}")

    out_path = cfg.output_dir / "feature_catalog.validated.json"
    out_path.write_text(json.dumps(pruned, indent=2))
    print(f"\n  Saved: {out_path}")

    report_path = cfg.output_dir / "validation_report.json"
    report_path.write_text(json.dumps({
        "n_input": n_in,
        "n_output": n_out,
        "n_dropped": n_dropped,
        "n_examples_pruned": n_pruned_examples,
        "verdicts": {k: v for k, v in verdicts.items() if v is not None},
        "drop_log": drop_log,
    }, indent=2))
    print(f"  Saved: {report_path}")
    print(f"\n  To use:")
    print(f"    cp {out_path} {cfg.catalog_path}     # opt in to validated subset")
    return pruned
