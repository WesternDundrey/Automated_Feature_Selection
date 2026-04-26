"""
Delphi scoring gate — predict held-out latent firing from a description.

Per the user's audit: "Delphi is not really integrated into the active
pipeline. The folders are cloned local references, but the running code
uses your own pipeline.inventory + Sonnet prompts." This module fixes that
narrowly. After Sonnet generates a description for a candidate latent, we
ask a held-out judge model "for these N examples, which ones are actually
activations of the description?" The judge's accuracy on a balanced
{activating / non-activating} split is the score we gate on. Bad
descriptions can't predict what the latent fires on, so they fail the
gate before they reach annotation.

We use EleutherAI Delphi's `DetectionScorer` directly (via the local
`agentic-delphi/` checkout, sys.path-injected at import time). No
homegrown prompt — the prompt is Delphi's published one with the
neuronpedia-style few-shot examples baked in. That keeps the score
comparable to what Delphi reports in its papers.

The gate runs at two slots in the pipeline:

  inventory.run    : after explain_features, before organize_hierarchy.
                     Drops fuzzy descriptions before they enter the catalog.
  promote_loop.run : between the crispness gate and the mini-prefilter.
                     Latents whose Sonnet description fails to predict
                     their own held-out firing are dropped before any
                     corpus annotation cost is paid.

Standalone CLI: `--step delphi-score` runs the gate over the existing
top_activations.json + raw_descriptions.json and writes a scores file
without modifying the catalog. Useful for sanity-checking a catalog
that was built before v8.15.

Tradeoffs / scope:
  - Detection accuracy is a precision-on-positives + recall-on-negatives
    metric. It catches "the description names the wrong concept" but
    NOT "the annotator can't reproduce it" — for the latter you still
    need the AUROC mini-prefilter against annotator labels.
  - We use accuracy (mean of `correct`) rather than a calibrated AUC;
    Delphi's OpenRouter client doesn't expose log-probs, so AUC mode
    is unavailable. Accuracy is fine for a 0.7-style gate.
  - Each scored latent costs one judge LLM call per `n_examples_shown`
    examples shown; with 5 activating + 5 non-activating per latent and
    `n_examples_shown=5`, that's 2 calls per candidate. Cheap.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from .config import Config


# ── Bootstrap: make `import delphi` work without pip install ──────────────

_DELPHI_AVAILABLE: bool | None = None
_DELPHI_IMPORT_ERROR: str | None = None


def _bootstrap_delphi() -> bool:
    """Locate a Delphi checkout and prepend it to sys.path so `import delphi`
    resolves the cloned package rather than a system install. Idempotent."""
    global _DELPHI_AVAILABLE, _DELPHI_IMPORT_ERROR
    if _DELPHI_AVAILABLE is not None:
        return _DELPHI_AVAILABLE

    repo_root = Path(__file__).parent.parent
    candidates = [
        repo_root / "agentic-delphi",
        repo_root / "delphi-eleutherai",
        repo_root / "agentic-delphi-correct" / "agentic-delphi",
    ]
    delphi_root = None
    for c in candidates:
        if (c / "delphi" / "__init__.py").exists():
            delphi_root = c
            break

    if delphi_root is None:
        _DELPHI_AVAILABLE = False
        _DELPHI_IMPORT_ERROR = (
            f"No Delphi checkout found under {repo_root}. Expected one of: "
            f"{[str(c) for c in candidates]}. Clone "
            f"github.com/EleutherAI/delphi or "
            f"github.com/agents-for-interp/agentic-delphi into the repo root."
        )
        return False

    if str(delphi_root) not in sys.path:
        sys.path.insert(0, str(delphi_root))

    try:
        import delphi  # noqa: F401
        from delphi.scorers.classifier.detection import DetectionScorer  # noqa
        from delphi.clients.openrouter import OpenRouter  # noqa
        from delphi.latents.latents import (  # noqa
            Latent, LatentRecord, ActivatingExample, NonActivatingExample,
        )
        _DELPHI_AVAILABLE = True
        return True
    except ImportError as e:
        _DELPHI_AVAILABLE = False
        _DELPHI_IMPORT_ERROR = (
            f"Delphi checkout at {delphi_root} but `import delphi` failed: "
            f"{type(e).__name__}: {e}. Likely missing a package — check "
            f"orjson, blobfile, httpx in the venv (added to "
            f"pipeline/requirements.txt in v8.15)."
        )
        return False


def is_available() -> tuple[bool, str | None]:
    """Public probe: returns (available, error_message). Other modules call
    this to decide whether to use the Delphi gate or skip silently."""
    ok = _bootstrap_delphi()
    return ok, (_DELPHI_IMPORT_ERROR if not ok else None)


# ── LatentRecord builder from our SAE state ─────────────────────────────


def _build_record_from_top_activations(
    latent_idx: int,
    description: str,
    top_activations: list[dict],
    full_tokens: torch.Tensor,
    activations_for_latent: torch.Tensor,
    tokenizer,
    n_test: int,
    n_not_active: int,
    rng_seed: int,
):
    """Build a Delphi LatentRecord from our top_activations.json shape and
    cached activations. Returns LatentRecord or None if there's not enough
    data (fewer than n_test activating positions).

    full_tokens:        (N, T) original token IDs (CPU).
    activations_for_latent: (N, T) per-position activation values for THIS
                          latent (CPU). Used to find non-activating
                          contexts and to fill the per-position activations
                          field of each Example.
    """
    if not _bootstrap_delphi():
        return None
    from delphi.latents.latents import (
        Latent, LatentRecord, ActivatingExample, NonActivatingExample,
    )

    if len(top_activations) < n_test:
        return None

    rng = np.random.RandomState(rng_seed + latent_idx)
    N, T = full_tokens.shape

    # ── Activating examples — taken from the supplied top_activations list.
    # Each entry has {context_ids, pos, activation}. We render a short
    # window centered on `pos` so the example is a contiguous tensor
    # (Delphi expects fixed-length tokens / activations per Example).
    activating_examples = []
    used_test = top_activations[:n_test]
    for ex in used_test:
        ctx_ids = ex.get("context_ids") or []
        pos = ex.get("pos", 0)
        if not ctx_ids:
            continue
        tok = torch.tensor(list(ctx_ids), dtype=torch.long)
        # Re-derive activations for this context. We don't have the raw
        # activation profile across the window, only the peak. Approximate
        # by setting the peak position to `activation` and the rest to 0;
        # DetectionScorer renders text via _prepare_text which only uses
        # max_activation, so this is a faithful representation for the
        # plain (highlighted=False) text path.
        act_window = torch.zeros(len(ctx_ids), dtype=torch.float32)
        if 0 <= pos < len(ctx_ids):
            act_window[pos] = float(ex.get("activation", 1.0))

        try:
            str_tokens = [tokenizer.decode([int(t)]) for t in ctx_ids]
        except Exception:
            str_tokens = [str(int(t)) for t in ctx_ids]

        activating_examples.append(ActivatingExample(
            tokens=tok,
            activations=act_window,
            str_tokens=str_tokens,
            quantile=0,
        ))

    if len(activating_examples) < n_test:
        return None

    # ── Non-activating examples — sample N positions where the latent's
    # activation is exactly zero, render a small window of context.
    nonact_mask = activations_for_latent <= 0
    n_zero = int(nonact_mask.sum().item())
    if n_zero < n_not_active:
        return None
    flat_zero_idx = nonact_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1).numpy()
    chosen = rng.choice(flat_zero_idx, size=n_not_active, replace=False)

    # Match the activating-example window length so prompts are uniform.
    window_len = len(used_test[0]["context_ids"]) if used_test[0].get("context_ids") else 21
    half = window_len // 2

    nonact_examples = []
    for flat_pos in chosen:
        n = int(flat_pos // T)
        t = int(flat_pos % T)
        start = max(0, t - half)
        end = min(T, start + window_len)
        # Re-clamp start so we always have window_len tokens (when possible)
        start = max(0, end - window_len)
        ctx = full_tokens[n, start:end]
        if ctx.shape[0] < 2:
            continue
        try:
            str_tokens = [tokenizer.decode([int(x)]) for x in ctx.tolist()]
        except Exception:
            str_tokens = [str(int(x)) for x in ctx.tolist()]
        nonact_examples.append(NonActivatingExample(
            tokens=ctx.long(),
            activations=torch.zeros(ctx.shape[0], dtype=torch.float32),
            str_tokens=str_tokens,
            distance=0.0,
        ))

    if len(nonact_examples) < max(2, n_not_active // 2):
        return None

    record = LatentRecord(
        latent=Latent(
            module_name="supsae_u",
            latent_index=int(latent_idx),
        ),
        explanation=description,
    )
    record.test = activating_examples
    record.not_active = nonact_examples
    return record


# ── Async scorer runner ────────────────────────────────────────────────


async def _score_one(scorer, record):
    """Run one record through the scorer; trap errors so a single flaky
    LLM call doesn't kill a batch."""
    try:
        result = await scorer(record)
        return result
    except Exception as e:
        return ("error", f"{type(e).__name__}: {e}")


async def _score_batch(scorer, records):
    return await asyncio.gather(*[_score_one(scorer, r) for r in records])


def _accuracy_from_result(result) -> tuple[float, int, int]:
    """Pull (accuracy, n_correct, n_total) out of a Delphi ScorerResult.
    ScorerResult.score is a list[ClassifierOutput], each with `.correct`."""
    score_list = getattr(result, "score", None) or []
    n_total = len(score_list)
    if n_total == 0:
        return 0.0, 0, 0
    n_correct = sum(1 for o in score_list if getattr(o, "correct", False))
    return n_correct / n_total, n_correct, n_total


# ── Public entry point ────────────────────────────────────────────────


def score_descriptions(
    descriptions: dict[str, str],
    top_activations: dict[str, list[dict]],
    full_tokens: torch.Tensor,
    activations: torch.Tensor,   # (N, T, n_latents) — for picking non-acts
    latent_offset: int,           # global latent_idx = local idx + offset
    tokenizer,
    cfg: Config,
    threshold: float = 0.7,
    n_test: int = 5,
    n_not_active: int = 5,
    judge_model: str | None = None,
    api_key: str | None = None,
) -> dict[str, dict]:
    """Score each (latent_id → description) pair via Delphi DetectionScorer.

    Returns:
        {latent_id_str: {
            "score": float | None,        # detection accuracy in [0, 1]
            "n_correct": int, "n_total": int,
            "kept": bool,                  # score >= threshold
            "reason": str,                 # why kept/dropped (or "no_data")
        }, ...}

    `descriptions` keys must be string forms of indices into `activations`'s
    last axis after `latent_offset` is added (mirrors top_activations.json's
    key convention from inventory.collect_top_activations).

    `latent_offset` lets the caller score either the WHOLE pretrained SAE
    (offset=0) or only the U slice of a supervised SAE (offset=n_supervised).
    """
    out: dict[str, dict] = {}

    if not _bootstrap_delphi():
        msg = _DELPHI_IMPORT_ERROR or "delphi unavailable"
        print(f"  [delphi-score] SKIP: {msg}")
        for lid in descriptions:
            out[lid] = {
                "score": None, "n_correct": 0, "n_total": 0,
                "kept": True,  # default-keep so the gate never silently drops everything when delphi is missing
                "reason": "delphi_unavailable",
            }
        return out

    from delphi.scorers.classifier.detection import DetectionScorer
    from delphi.clients.openrouter import OpenRouter

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("  [delphi-score] SKIP: OPENROUTER_API_KEY not set")
        for lid in descriptions:
            out[lid] = {
                "score": None, "n_correct": 0, "n_total": 0,
                "kept": True, "reason": "no_api_key",
            }
        return out

    # Resolve judge model: explicit kwarg > cfg.delphi_judge_model (if set)
    # > cfg.organization_model. Empty-string sentinel means "fall through".
    cfg_judge = getattr(cfg, "delphi_judge_model", "")
    if not cfg_judge:
        cfg_judge = cfg.organization_model
    judge_model = judge_model or cfg_judge
    print(f"  [delphi-score] judge={judge_model} threshold={threshold:.2f} "
          f"n_test={n_test} n_not_active={n_not_active}")

    client = OpenRouter(
        model=judge_model,
        api_key=api_key,
        max_tokens=400,
        temperature=0.0,
    )
    scorer = DetectionScorer(
        client=client,
        verbose=False,
        n_examples_shown=n_test + n_not_active,
        log_prob=False,
        temperature=0.0,
    )

    # Pre-extract per-latent activation profiles (cheap; activations is in
    # RAM already). For each candidate latent, we need (N, T) for that
    # latent's activations to find non-firing positions.
    records: list = []
    record_keys: list[str] = []
    for lid_str, desc in descriptions.items():
        try:
            local_idx = int(lid_str)
        except ValueError:
            out[lid_str] = {
                "score": None, "n_correct": 0, "n_total": 0,
                "kept": True, "reason": f"bad_id_{lid_str!r}",
            }
            continue
        global_idx = local_idx + latent_offset
        if global_idx < 0 or global_idx >= activations.shape[-1]:
            out[lid_str] = {
                "score": None, "n_correct": 0, "n_total": 0,
                "kept": False, "reason": f"latent_index_oob_{global_idx}",
            }
            continue
        acts_for_lat = activations[..., global_idx]
        top = top_activations.get(lid_str, [])

        record = _build_record_from_top_activations(
            latent_idx=local_idx,
            description=desc,
            top_activations=top,
            full_tokens=full_tokens,
            activations_for_latent=acts_for_lat,
            tokenizer=tokenizer,
            n_test=n_test,
            n_not_active=n_not_active,
            rng_seed=cfg.seed,
        )
        if record is None:
            out[lid_str] = {
                "score": None, "n_correct": 0, "n_total": 0,
                "kept": False, "reason": "insufficient_examples",
            }
            continue
        records.append(record)
        record_keys.append(lid_str)

    if not records:
        return out

    # Batched async scoring. Each record is one scorer call internally.
    # Larger batches make the overall wall-clock dominated by the slowest
    # request; fine for ~50-100 latents.
    print(f"  [delphi-score] running on {len(records)} latents...")
    results = asyncio.run(_score_batch(scorer, records))

    for lid_str, result in zip(record_keys, results):
        if isinstance(result, tuple) and result and result[0] == "error":
            out[lid_str] = {
                "score": None, "n_correct": 0, "n_total": 0,
                "kept": True,  # fail open on transport errors
                "reason": result[1],
            }
            continue
        accuracy, n_correct, n_total = _accuracy_from_result(result)
        kept = accuracy >= threshold
        out[lid_str] = {
            "score": round(float(accuracy), 4),
            "n_correct": n_correct,
            "n_total": n_total,
            "kept": bool(kept),
            "reason": (
                f"score={accuracy:.3f} {'≥' if kept else '<'} {threshold:.2f}"
            ),
        }
    return out


# ── promote_loop integration ───────────────────────────────────────────


def gate_promote_loop_candidates(
    crisp_candidates: list[tuple[int, str]],
    all_top_acts: dict,
    sae,
    activations: torch.Tensor,   # (N, T, d_model) — masked, model-residual
    tokens_masked: torch.Tensor,  # (N, T) — same masking as activations
    n_supervised: int,
    d_model: int,
    tokenizer,
    cfg: Config,
    threshold: float | None = None,
    judge_model: str | None = None,
) -> tuple[list[tuple[int, str]], dict]:
    """Run Delphi DetectionScorer over the promote_loop's CRISP candidates.

    Each crisp candidate is `(u_local, description)` where u_local is the
    index into the supervised SAE's U slice. The gate computes the
    per-latent activation profile using `sae.unsup_encoder_*()` (uniform
    API across HingeSAE / JumpReLUHingeSAE / GatedBCESAE / legacy
    SupervisedSAE) so it works for every supervision_mode without special
    cases.

    Returns:
        (kept_candidates, score_log)
        - kept_candidates: subset of crisp_candidates whose Delphi
          accuracy ≥ threshold. Order preserved.
        - score_log: {f"u{u}": {score, n_correct, n_total, kept, reason}, ...}

    Fail-open semantics: if Delphi is unavailable / API key missing /
    transport error, every candidate is kept and the reason is logged.
    The gate must never silently drop everything when its own
    infrastructure is broken.
    """
    if not crisp_candidates:
        return crisp_candidates, {}

    threshold = float(
        threshold
        if threshold is not None
        else getattr(cfg, "delphi_score_threshold", 0.7)
    )

    if not _bootstrap_delphi():
        msg = _DELPHI_IMPORT_ERROR or "delphi unavailable"
        print(f"  [delphi-gate promote_loop] skipped: {msg}")
        return crisp_candidates, {
            f"u{u}": {"score": None, "kept": True, "reason": "delphi_unavailable"}
            for u, _ in crisp_candidates
        }

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("  [delphi-gate promote_loop] skipped: OPENROUTER_API_KEY not set")
        return crisp_candidates, {
            f"u{u}": {"score": None, "kept": True, "reason": "no_api_key"}
            for u, _ in crisp_candidates
        }

    # Compute the per-candidate U-latent activation map.
    # enc_w[u_local]: (d_model,)  enc_b[u_local]: scalar
    # acts_u(n, t) = ReLU(x_nt @ enc_w[u_local] + enc_b[u_local])
    candidate_indices = [int(u) for u, _ in crisp_candidates]
    enc_w = sae.unsup_encoder_weight().detach()  # (n_unsup, d_model)
    enc_b = sae.unsup_encoder_bias().detach()    # (n_unsup,)

    N, T, _ = activations.shape
    x_flat = activations.reshape(-1, d_model).to(torch.float32)
    sub_w = enc_w[candidate_indices].to(
        device=x_flat.device, dtype=x_flat.dtype,
    )  # (K, d_model)
    sub_b = enc_b[candidate_indices].to(
        device=x_flat.device, dtype=x_flat.dtype,
    )  # (K,)
    with torch.no_grad():
        pre = x_flat @ sub_w.T + sub_b               # (N*T, K)
        acts = torch.relu(pre).reshape(N, T, -1).cpu()

    # score_descriptions expects keys to be string-form indices into
    # `activations`'s last axis (after `latent_offset`). We pass
    # latent_offset=0 and key by position-in-stack.
    descriptions = {str(i): desc for i, (_, desc) in enumerate(crisp_candidates)}
    top_acts_for_gate = {
        str(i): all_top_acts.get(str(u), [])
        for i, (u, _) in enumerate(crisp_candidates)
    }

    print(f"  [delphi-gate promote_loop] scoring {len(crisp_candidates)} "
          f"crisp candidates against held-out activations "
          f"(threshold={threshold:.2f})")

    scores = score_descriptions(
        descriptions=descriptions,
        top_activations=top_acts_for_gate,
        full_tokens=tokens_masked,
        activations=acts,
        latent_offset=0,
        tokenizer=tokenizer,
        cfg=cfg,
        threshold=threshold,
        judge_model=judge_model,
        api_key=api_key,
    )

    kept: list[tuple[int, str]] = []
    log: dict = {}
    for i, (u, desc) in enumerate(crisp_candidates):
        s = scores.get(str(i), {})
        if s.get("kept", True):
            kept.append((u, desc))
        log[f"u{u}"] = {**s, "description": desc}

    n_dropped = len(crisp_candidates) - len(kept)
    if n_dropped > 0:
        print(f"  [delphi-gate promote_loop] kept {len(kept)} of "
              f"{len(crisp_candidates)} (dropped {n_dropped} below "
              f"threshold {threshold:.2f}):")
        for u, desc in crisp_candidates:
            s = log[f"u{u}"]
            if not s.get("kept", True):
                ssc = s.get("score")
                ssc_s = f"{ssc:.3f}" if ssc is not None else "?"
                print(f"    drop u{u} (score={ssc_s}): {desc[:80]}")
    return kept, log


# ── CLI: `--step delphi-score` ──────────────────────────────────────────


def run(cfg: Config = None, threshold: float = 0.7) -> dict:
    """Standalone runner: score the existing top_activations.json + raw
    descriptions, write per-latent scores to disk. Doesn't modify the
    catalog — sanity check only."""
    if cfg is None:
        cfg = Config()

    if not cfg.top_activations_path.exists():
        raise FileNotFoundError(
            f"{cfg.top_activations_path} not found. Run --step inventory first."
        )
    if not cfg.raw_descriptions_path.exists():
        raise FileNotFoundError(
            f"{cfg.raw_descriptions_path} not found. Run --step inventory first."
        )
    if not cfg.activations_path.exists():
        raise FileNotFoundError(
            f"{cfg.activations_path} not found. Run --step annotate first."
        )

    top_activations = json.loads(cfg.top_activations_path.read_text())
    descriptions = json.loads(cfg.raw_descriptions_path.read_text())
    activations = torch.load(cfg.activations_path, weights_only=True)
    tokens = torch.load(cfg.tokens_path, weights_only=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    print(f"  Scoring {len(descriptions)} descriptions over "
          f"{activations.shape} activations...")

    # For pretrained-SAE latents the offset is 0 (descriptions are for the
    # raw SAE, not the supervised slice).
    scores = score_descriptions(
        descriptions=descriptions,
        top_activations=top_activations,
        full_tokens=tokens,
        activations=activations,
        latent_offset=0,
        tokenizer=tokenizer,
        cfg=cfg,
        threshold=threshold,
    )

    out_path = cfg.output_dir / "delphi_scores.json"
    out_path.write_text(json.dumps({
        "threshold": threshold,
        "judge_model": (
            getattr(cfg, "delphi_judge_model", "") or cfg.organization_model
        ),
        "n_total": len(scores),
        "n_kept": sum(1 for v in scores.values() if v.get("kept")),
        "scores": scores,
    }, indent=2))
    print(f"  Wrote {out_path}")

    n_kept = sum(1 for v in scores.values() if v.get("kept"))
    n_dropped = sum(1 for v in scores.values()
                    if not v.get("kept") and v.get("reason") != "delphi_unavailable")
    n_no_data = sum(1 for v in scores.values()
                    if v.get("reason") == "insufficient_examples")
    print(f"  Result: kept={n_kept} dropped={n_dropped} no_data={n_no_data}")

    # Print the worst 10 + best 5 for quick eyeball
    scored = [(lid, v["score"]) for lid, v in scores.items()
              if v.get("score") is not None]
    if scored:
        scored.sort(key=lambda kv: kv[1])
        print("\n  Worst 10 by detection accuracy:")
        for lid, s in scored[:10]:
            print(f"    {s:.3f}  L{lid}: {descriptions[lid][:80]}")
        print("\n  Best 5 by detection accuracy:")
        for lid, s in scored[-5:][::-1]:
            print(f"    {s:.3f}  L{lid}: {descriptions[lid][:80]}")

    return {
        "n_total": len(scores),
        "n_kept": n_kept,
        "n_dropped": n_dropped,
        "scores_path": str(out_path),
    }
