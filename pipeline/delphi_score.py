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

The gate runs at one place in the active pipeline:

  promote_loop.run : between the crispness gate and the proposal-building
                     step. Latents whose Sonnet description fails to
                     predict their own held-out firing are dropped before
                     any corpus annotation cost is paid.

Inventory-time gating is intentionally NOT inlined in `--step inventory`:
it would need a fresh pretrained-SAE forward pass on activations.pt
(which holds residual-stream vectors, not SAE codes) and `--step
inventory` is meant to be quick. The standalone CLI below covers this
case explicitly.

Standalone CLI: `--step delphi-score` loads the pretrained SAE,
encodes activations.pt → SAE codes for the latent indices that have
descriptions, builds LatentRecords, and runs DetectionScorer. Writes
delphi_scores.json without modifying the catalog. Useful for
sanity-checking a catalog built before v8.15 or for ad-hoc audits.

Held-out positives (v8.16): both the describer (Sonnet, in
`inventory.explain_features`) and the gate take their examples from
the same `top_activations` list, but the describer uses
`top_activations[:top_k_examples - delphi_held_out_n]` (default first
20) and the gate uses `top_activations[top_k_examples - delphi_held_out_n :]`
(default last 10). This keeps the gate's positives out-of-sample so
detection accuracy isn't artificially inflated by the description
literally referencing the test examples.

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
    held_out_skip: int = 0,
    n_mid_tier: int = 3,
):
    """Build a Delphi LatentRecord from our top_activations.json shape and
    cached activations. Returns LatentRecord or None if there's not enough
    data (fewer than n_test activating positions).

    full_tokens:        (N, T) original token IDs (CPU).
    activations_for_latent: (N, T) per-position activation values for THIS
                          latent (CPU). Used to find non-activating
                          contexts and to fill the per-position activations
                          field of each Example.
    held_out_skip: skip the first `held_out_skip` entries of
                   `top_activations` before drawing positive test
                   examples — those were the entries the description LLM
                   saw, so scoring on them would be optimistic. v8.16
                   audit fix.
    """
    if not _bootstrap_delphi():
        return None
    from delphi.latents.latents import (
        Latent, LatentRecord, ActivatingExample, NonActivatingExample,
    )

    rng = np.random.RandomState(rng_seed + latent_idx)
    N, T = full_tokens.shape

    # ── Activating examples — taken from the held-out slice of
    # top_activations: examples the description LLM did NOT see. Falls
    # back to a regular slice if the held-out window is too small (e.g.
    # an old top_activations.json from pre-v8.16 inventory only has 20
    # entries, all of which the LLM saw). In the fallback case we still
    # score, but we print a warning so the user knows the score is
    # optimistic on this latent.
    held_out_pool = top_activations[held_out_skip:]
    if len(held_out_pool) < n_test:
        # Fallback: use the head of the list (overlap with describer).
        # Score remains directionally informative even if optimistic.
        held_out_pool = top_activations[:max(n_test, len(top_activations))]
    activating_examples = []
    used_test = held_out_pool[:n_test]
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

        # v8.18 audit fix #3: pre-mark the peak-activation token with
        # <<...>> so the judge focuses on the target token rather than
        # scoring "does the window contain this concept." Delphi's
        # default DetectionScorer call path uses highlighted=False, so
        # we inject the marker into str_tokens directly — _prepare_text
        # will join them and the judge sees the marker. Same convention
        # Delphi uses internally when highlighted=True.
        if 0 <= pos < len(str_tokens):
            str_tokens[pos] = f"<<{str_tokens[pos]}>>"

        activating_examples.append(ActivatingExample(
            tokens=tok,
            activations=act_window,
            str_tokens=str_tokens,
            quantile=0,
        ))

    if len(activating_examples) < n_test:
        return None

    # ── Mid-tier activating examples (v8.18 audit fix #4). Top-K-only
    # tests peak-case precision: a description that nails the strongest
    # firings but fails on mid-tier ones still passes. Sample positions
    # where the latent fires in the 25-75th percentile of nonzero
    # activations to test full-support recall. The judge labels these
    # as activating; descriptions that are too narrow ("only fires on X
    # during Y") will fail recall on these mid-tier examples.
    n_mid = int(n_mid_tier)
    flat_acts = activations_for_latent.reshape(-1).cpu().numpy()
    nonzero_vals = flat_acts[flat_acts > 0]
    if len(nonzero_vals) >= 8:
        p25 = float(np.percentile(nonzero_vals, 25))
        p75 = float(np.percentile(nonzero_vals, 75))
        mid_mask = (flat_acts > p25) & (flat_acts < p75)
        mid_positions = np.flatnonzero(mid_mask)
        if len(mid_positions) >= n_mid:
            mid_chosen = rng.choice(mid_positions, size=n_mid, replace=False)
            mid_window_len = (
                len(used_test[0]["context_ids"]) if used_test[0].get("context_ids") else 21
            )
            mid_half = mid_window_len // 2
            for flat_pos in mid_chosen:
                n = int(flat_pos // T)
                t = int(flat_pos % T)
                start = max(0, t - mid_half)
                end = min(T, start + mid_window_len)
                start = max(0, end - mid_window_len)
                ctx = full_tokens[n, start:end]
                local_pos = t - start
                if ctx.shape[0] < 2:
                    continue
                try:
                    str_tokens_mid = [tokenizer.decode([int(x)]) for x in ctx.tolist()]
                except Exception:
                    str_tokens_mid = [str(int(x)) for x in ctx.tolist()]
                if 0 <= local_pos < len(str_tokens_mid):
                    str_tokens_mid[local_pos] = f"<<{str_tokens_mid[local_pos]}>>"
                act_window_mid = torch.zeros(ctx.shape[0], dtype=torch.float32)
                if 0 <= local_pos < ctx.shape[0]:
                    act_window_mid[local_pos] = float(flat_acts[flat_pos])
                activating_examples.append(ActivatingExample(
                    tokens=ctx.long(),
                    activations=act_window_mid,
                    str_tokens=str_tokens_mid,
                    quantile=1,   # 1 = mid-tier marker
                ))

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

    # v8.18.2 hotfix: every LatentRecord must produce EXACTLY
    # n_test + n_mid_tier + n_not_active samples so the scorer's
    # batch size matches predictions count. If mid-tier sampling fell
    # short (rare latents whose nonzero-activation distribution is
    # narrow, so the 25-75 percentile band is empty), pad with extra
    # non-active examples to keep the total constant.
    target_total = n_test + n_mid + n_not_active
    actual_total = len(activating_examples) + len(nonact_examples)
    if actual_total < target_total:
        # Pad with extra random non-active positions so the batch
        # arithmetic in DetectionScorer.classifier is always exact.
        needed = target_total - actual_total
        # Reuse zero-activation pool, sample additional indices.
        leftover_zero = np.setdiff1d(flat_zero_idx, chosen, assume_unique=False)
        if len(leftover_zero) >= needed:
            extra = rng.choice(leftover_zero, size=needed, replace=False)
        elif len(leftover_zero) > 0:
            extra = rng.choice(leftover_zero, size=needed, replace=True)
        else:
            # No extra zeros — drop the record. Caller treats as no-data.
            return None
        for flat_pos in extra:
            n = int(flat_pos // T)
            t = int(flat_pos % T)
            start = max(0, t - half)
            end = min(T, start + window_len)
            start = max(0, end - window_len)
            ctx = full_tokens[n, start:end]
            if ctx.shape[0] < 2:
                continue
            try:
                str_tokens_pad = [tokenizer.decode([int(x)]) for x in ctx.tolist()]
            except Exception:
                str_tokens_pad = [str(int(x)) for x in ctx.tolist()]
            nonact_examples.append(NonActivatingExample(
                tokens=ctx.long(),
                activations=torch.zeros(ctx.shape[0], dtype=torch.float32),
                str_tokens=str_tokens_pad,
                distance=0.0,
            ))

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
    held_out_skip: int | None = None,
    n_mid_tier: int | None = None,
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

    # v8.16: held-out positives. Default skip = top_k_examples - held_out_n
    # so the gate uses examples the description LLM didn't see. Caller
    # can override (e.g., promote_loop with its own held-out logic).
    if held_out_skip is None:
        held_out_skip = max(
            0,
            int(getattr(cfg, "top_k_examples", 30))
            - int(getattr(cfg, "delphi_held_out_n", 10)),
        )

    if n_mid_tier is None:
        n_mid_tier = int(getattr(cfg, "delphi_n_mid_tier", 3))

    print(f"  [delphi-score] judge={judge_model} threshold={threshold:.2f} "
          f"n_test={n_test} n_not_active={n_not_active} "
          f"held_out_skip={held_out_skip} n_mid_tier={n_mid_tier}")

    client = OpenRouter(
        model=judge_model,
        api_key=api_key,
        max_tokens=400,
        temperature=0.0,
    )
    scorer = DetectionScorer(
        client=client,
        verbose=False,
        # v8.18.1 hotfix: include n_mid_tier in the batch size. Otherwise
        # records have (n_test + n_mid + n_not_active) samples but Delphi
        # batches at (n_test + n_not_active), leaving a remainder batch
        # whose len(predictions) doesn't match self.n_examples_shown,
        # tripping the assertion at delphi/scorers/classifier/classifier.py:147
        # ("Parsing selections failed: AssertionError()") for every record.
        n_examples_shown=n_test + n_mid_tier + n_not_active,
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
            held_out_skip=held_out_skip,
            n_mid_tier=n_mid_tier,
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
        print(f"  [delphi-gate promote_loop] STATUS=skipped reason=delphi_unavailable")
        print(f"    detail: {msg}")
        return crisp_candidates, {
            "_gate_mode": "skipped",
            "_skip_reason": "delphi_unavailable",
            **{f"u{u}": {"score": None, "kept": True, "reason": "delphi_unavailable"}
               for u, _ in crisp_candidates},
        }

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("  [delphi-gate promote_loop] STATUS=skipped reason=no_api_key")
        return crisp_candidates, {
            "_gate_mode": "skipped",
            "_skip_reason": "no_api_key",
            **{f"u{u}": {"score": None, "kept": True, "reason": "no_api_key"}
               for u, _ in crisp_candidates},
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
        for u, desc in crisp_candidates:
            s = log[f"u{u}"]
            if not s.get("kept", True):
                ssc = s.get("score")
                ssc_s = f"{ssc:.3f}" if ssc is not None else "?"
                print(f"    drop u{u} (score={ssc_s}): {desc[:80]}")

    # Final summary line.
    print(f"  [delphi-gate promote_loop] STATUS=scored "
          f"kept={len(kept)}/{len(crisp_candidates)} "
          f"dropped={n_dropped} threshold={threshold:.2f}")

    log["_gate_mode"] = "scored"
    log["_threshold"] = threshold
    log["_n_input"]   = len(crisp_candidates)
    log["_n_kept"]    = len(kept)
    return kept, log


# ── inventory-time gate: filter descriptions before the catalog ───────


def gate_inventory_descriptions(
    sae,
    activations: torch.Tensor,    # (N, T, d_model) residual stream
    tokens: torch.Tensor,          # (N, T) token IDs
    descriptions: dict[str, str],
    top_activations: dict[str, list[dict]],
    tokenizer,
    cfg: Config,
    threshold: float | None = None,
    judge_model: str | None = None,
) -> tuple[dict[str, str], dict]:
    """Gate the inventory's pretrained-SAE descriptions through Delphi
    DetectionScorer BEFORE they enter the feature catalog. Returns
    (kept_descriptions, score_log).

    This is what the user actually wanted from "Delphi integration":
    drop descriptions whose own held-out latent firing the description
    can't predict, so the annotator never wastes budget on fuzzy
    descriptions. Pre-v8.17 this was deferred to a standalone audit
    (`--step delphi-score`); v8.17 makes it the default catalog filter.

    `descriptions` keys must be string-form integer indices into
    `sae.W_enc`'s last axis (i.e., latent indices). Same convention as
    `inventory.collect_top_activations` produces.

    Fail-open: if Delphi is unavailable / no API key / transport errors,
    every description is kept and the reason is logged. The gate must
    never silently nuke the whole catalog for an environment problem.
    """
    threshold = float(
        threshold
        if threshold is not None
        else getattr(cfg, "delphi_score_threshold", 0.7)
    )

    if not _bootstrap_delphi():
        msg = _DELPHI_IMPORT_ERROR or "delphi unavailable"
        print(f"  [delphi-gate inventory] STATUS=skipped reason=delphi_unavailable")
        print(f"    detail: {msg}")
        return descriptions, {
            "_gate_mode": "skipped",
            "_skip_reason": "delphi_unavailable",
            **{lid: {"score": None, "kept": True, "reason": "delphi_unavailable"}
               for lid in descriptions},
        }

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("  [delphi-gate inventory] STATUS=skipped reason=no_api_key")
        return descriptions, {
            "_gate_mode": "skipped",
            "_skip_reason": "no_api_key",
            **{lid: {"score": None, "kept": True, "reason": "no_api_key"}
               for lid in descriptions},
        }

    # Subset to descriptions whose keys parse as valid latent indices.
    latent_ids: list[int] = []
    valid_keys: list[str] = []
    for k in descriptions.keys():
        try:
            lid = int(k)
        except ValueError:
            continue
        if 0 <= lid < sae.d_sae:
            latent_ids.append(lid)
            valid_keys.append(k)
    if not latent_ids:
        print("  [delphi-gate inventory] no valid latent ids; skipping")
        return descriptions, {}

    if activations.dim() != 3:
        print(f"  [delphi-gate inventory] unexpected activations shape "
              f"{tuple(activations.shape)}; skipping")
        return descriptions, {}
    N, T, d_model = activations.shape
    print(f"  [delphi-gate inventory] encoding {len(latent_ids)} latents "
          f"through pretrained SAE (chunked, no full d_sae materialization)...")
    x_flat = activations.reshape(-1, d_model).to(torch.float32)
    sae_codes = _encode_pretrained_sae_for_latents(
        sae, x_flat, N, T, latent_ids, device=cfg.device,
    )

    # Re-key by stack position so score_descriptions's last-axis indexing
    # lines up with sae_codes' last axis.
    pos_keyed_descs = {
        str(i): descriptions[k] for i, k in enumerate(valid_keys)
    }
    pos_keyed_top_acts = {
        str(i): top_activations.get(k, []) for i, k in enumerate(valid_keys)
    }

    print(f"  [delphi-gate inventory] scoring {len(pos_keyed_descs)} "
          f"descriptions against held-out latent firing "
          f"(threshold={threshold:.2f})")

    pos_scores = score_descriptions(
        descriptions=pos_keyed_descs,
        top_activations=pos_keyed_top_acts,
        full_tokens=tokens,
        activations=sae_codes,
        latent_offset=0,
        tokenizer=tokenizer,
        cfg=cfg,
        threshold=threshold,
        judge_model=judge_model,
        api_key=api_key,
    )

    # Reverse-map back to original latent-id keys.
    score_log: dict = {}
    kept: dict[str, str] = {}
    for i, k in enumerate(valid_keys):
        s = pos_scores.get(str(i), {
            "score": None, "kept": True, "reason": "missing_score",
        })
        score_log[k] = {**s, "description": descriptions[k]}
        if s.get("kept", True):
            kept[k] = descriptions[k]
    # Carry through any descriptions whose keys couldn't be parsed —
    # these never went through the gate but shouldn't be dropped silently.
    for k, desc in descriptions.items():
        if k not in score_log:
            kept[k] = desc
            score_log[k] = {
                "score": None, "kept": True,
                "reason": "non_integer_key",
                "description": desc,
            }

    n_dropped = len(descriptions) - len(kept)
    if n_dropped > 0:
        # Show the worst 10 drops so the user can eyeball
        dropped_with_scores = sorted(
            [(k, score_log[k]["score"]) for k in score_log
             if not score_log[k].get("kept", True)
             and score_log[k].get("score") is not None],
            key=lambda kv: kv[1] if kv[1] is not None else 0.0,
        )
        for k, s in dropped_with_scores[:10]:
            print(f"    drop L{k} (score={s:.3f}): {descriptions[k][:80]}")

    # Final summary line — grep-friendly status so the user can verify
    # Delphi actually ran without inspecting JSON.
    print(f"  [delphi-gate inventory] STATUS=scored "
          f"kept={len(kept)}/{len(descriptions)} "
          f"dropped={n_dropped} threshold={threshold:.2f}")

    score_log["_gate_mode"] = "scored"
    score_log["_threshold"] = threshold
    score_log["_n_input"]   = len(descriptions)
    score_log["_n_kept"]    = len(kept)
    return kept, score_log


# ── post-organize_hierarchy gate: score final catalog leaves ─────────


def gate_organized_leaves(
    catalog: dict,
    sae,
    activations: torch.Tensor,    # (N, T, d_model) residual stream
    tokens: torch.Tensor,
    top_activations: dict[str, list[dict]],
    tokenizer,
    cfg: Config,
    threshold: float | None = None,
) -> tuple[dict, dict]:
    """Run Delphi DetectionScorer over the FINAL catalog leaves (after
    organize_hierarchy). Each leaf must carry `source_latents: [int]` —
    the latent IDs that contributed to its construction. The gate uses
    the first source latent's held-out top activations as positive
    examples for the leaf's (possibly-rewritten) description.

    Why this is needed (v8.18 audit fix #1): organize_hierarchy is allowed
    to rewrite descriptions, drop them, and (pre-v8.18) invent
    gap-filled leaves with no source. Pre-v8.18 the gate ran only on
    raw descriptions — the rewrites and gap-fillers slipped past
    unscrutinized. v8.18 makes Sonnet declare source_latents per leaf
    (organize_hierarchy prompt updated) and re-scores the rewritten
    description against the source latent's actual firing pattern. If
    the rewrite drifted too far from what predicts the source latent,
    the leaf is dropped.

    Returns (filtered_catalog, score_log). Drops leaves with empty
    `source_latents`, leaves whose source latent isn't in
    `top_activations`, and leaves whose Delphi accuracy is below
    threshold. Group entries pass through unchanged.

    Fail-open: any infrastructure problem keeps every leaf.
    """
    threshold = float(
        threshold
        if threshold is not None
        else getattr(cfg, "delphi_score_threshold", 0.7)
    )

    if not _bootstrap_delphi():
        msg = _DELPHI_IMPORT_ERROR or "delphi unavailable"
        print(f"  [delphi-gate organized] STATUS=skipped reason=delphi_unavailable")
        print(f"    detail: {msg}")
        return catalog, {"_gate_mode": "skipped", "_skip_reason": "delphi_unavailable"}

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("  [delphi-gate organized] STATUS=skipped reason=no_api_key")
        return catalog, {"_gate_mode": "skipped", "_skip_reason": "no_api_key"}

    features = catalog.get("features") or []
    leaves = [f for f in features if f.get("type") == "leaf"]
    groups = [f for f in features if f.get("type") != "leaf"]

    # Partition: leaves with valid source_latents that exist in top_activations.
    scoreable: list[dict] = []
    untraceable: list[dict] = []   # missing or empty source_latents
    orphaned: list[dict] = []      # source latent not in top_activations
    for lf in leaves:
        srcs_raw = lf.get("source_latents") or []
        srcs: list[int] = []
        for s in srcs_raw:
            try:
                srcs.append(int(s))
            except (TypeError, ValueError):
                continue
        if not srcs:
            untraceable.append(lf)
            continue
        # Keep first source whose top_activations are available.
        chosen = None
        for s in srcs:
            if str(s) in top_activations:
                chosen = s
                break
        if chosen is None:
            orphaned.append({**lf, "_attempted_sources": srcs})
            continue
        scoreable.append({**lf, "_chosen_source": chosen, "_source_latents": srcs})

    print(f"  [delphi-gate organized] {len(scoreable)} leaves traceable, "
          f"{len(untraceable)} untraceable (no source_latents), "
          f"{len(orphaned)} orphaned (source not in top_activations)")

    if not scoreable:
        # All leaves either untraceable or orphaned — keep groups + scoreable
        # (which is empty here) and drop the unscoreable ones with a log.
        log = {
            "threshold": threshold,
            "n_leaves_in":  len(leaves),
            "n_leaves_kept": 0,
            "n_untraceable": len(untraceable),
            "n_orphaned":    len(orphaned),
            "untraceable_ids": [lf["id"] for lf in untraceable],
            "orphaned_ids":    [lf["id"] for lf in orphaned],
            "scores": {},
        }
        # Edge case: keep all leaves anyway (fail-open if nothing scoreable).
        return catalog, log

    # Encode SAE codes for the chosen source latents.
    chosen_latents = [int(lf["_chosen_source"]) for lf in scoreable]
    valid_chosen: list[int] = []
    valid_lf_idx: list[int] = []
    for i, lid in enumerate(chosen_latents):
        if 0 <= lid < sae.d_sae:
            valid_chosen.append(lid)
            valid_lf_idx.append(i)
    if not valid_chosen:
        print("  [delphi-gate organized] no valid source latents within sae.d_sae range")
        return catalog, {"reason": "no_valid_sources"}

    if activations.dim() != 3:
        print(f"  [delphi-gate organized] unexpected activations shape "
              f"{tuple(activations.shape)}; skipping")
        return catalog, {"reason": "bad_activations_shape"}
    N, T, d_model = activations.shape
    print(f"  [delphi-gate organized] encoding {len(valid_chosen)} source "
          f"latents through pretrained SAE...")
    x_flat = activations.reshape(-1, d_model).to(torch.float32)
    sae_codes = _encode_pretrained_sae_for_latents(
        sae, x_flat, N, T, valid_chosen, device=cfg.device,
    )

    # Build pos-keyed descriptions where the description is the FINAL
    # (possibly-rewritten) leaf description, and top_activations come
    # from the chosen source latent.
    pos_keyed_descs = {}
    pos_keyed_top_acts = {}
    for stack_pos, lf_i in enumerate(valid_lf_idx):
        lf = scoreable[lf_i]
        pos_keyed_descs[str(stack_pos)] = lf["description"]
        pos_keyed_top_acts[str(stack_pos)] = top_activations.get(
            str(int(lf["_chosen_source"])), [],
        )

    print(f"  [delphi-gate organized] scoring {len(pos_keyed_descs)} final "
          f"leaves against their source-latent firing "
          f"(threshold={threshold:.2f})")

    pos_scores = score_descriptions(
        descriptions=pos_keyed_descs,
        top_activations=pos_keyed_top_acts,
        full_tokens=tokens,
        activations=sae_codes,
        latent_offset=0,
        tokenizer=tokenizer,
        cfg=cfg,
        threshold=threshold,
    )

    # Reverse-map. Build the filtered catalog.
    kept_leaves: list[dict] = []
    score_log: dict = {}
    for stack_pos, lf_i in enumerate(valid_lf_idx):
        lf = scoreable[lf_i]
        s = pos_scores.get(str(stack_pos), {
            "score": None, "kept": True, "reason": "missing_score",
        })
        score_log[lf["id"]] = {
            **s,
            "description": lf["description"],
            "chosen_source": int(lf["_chosen_source"]),
            "source_latents": list(lf["_source_latents"]),
        }
        if s.get("kept", True):
            # Strip the helper fields before re-emitting.
            cleaned = {
                k: v for k, v in lf.items()
                if not k.startswith("_")
            }
            kept_leaves.append(cleaned)

    # Untraceable / orphaned leaves are dropped. Log them.
    for lf in untraceable:
        score_log[lf["id"]] = {
            "score": None, "kept": False, "reason": "no_source_latents",
            "description": lf.get("description", ""),
            "chosen_source": None,
            "source_latents": [],
        }
    for lf in orphaned:
        score_log[lf["id"]] = {
            "score": None, "kept": False, "reason": "source_not_in_top_activations",
            "description": lf.get("description", ""),
            "chosen_source": None,
            "source_latents": lf.get("_attempted_sources") or [],
        }

    n_kept = len(kept_leaves)
    n_drop = len(leaves) - n_kept

    if n_drop > 0:
        # Print worst-10 (with scores) for eyeball
        scored_drops = sorted(
            [(lid, s["score"]) for lid, s in score_log.items()
             if not s.get("kept", True) and s.get("score") is not None],
            key=lambda kv: kv[1] if kv[1] is not None else 0.0,
        )
        for lid, sc in scored_drops[:10]:
            print(f"    drop {lid} (score={sc:.3f}): {score_log[lid]['description'][:80]}")

    # Final summary line.
    print(f"  [delphi-gate organized] STATUS=scored "
          f"kept={n_kept}/{len(leaves)} "
          f"dropped={n_drop} (untraceable={len(untraceable)}, "
          f"orphaned={len(orphaned)}, low_score={n_drop - len(untraceable) - len(orphaned)}) "
          f"threshold={threshold:.2f}")

    filtered_catalog = dict(catalog)
    filtered_catalog["features"] = groups + kept_leaves
    return filtered_catalog, {
        "_gate_mode":   "scored",
        "_threshold":   threshold,
        "n_leaves_in":  len(leaves),
        "n_leaves_kept": n_kept,
        "n_untraceable": len(untraceable),
        "n_orphaned":    len(orphaned),
        "scores": score_log,
    }


# ── CLI: `--step delphi-score` ──────────────────────────────────────────


def _encode_pretrained_sae_for_latents(
    sae, x_flat: torch.Tensor, N: int, T: int,
    latent_ids: list[int], device: str,
) -> torch.Tensor:
    """Compute (N, T, K) of pretrained-SAE activations for K specific
    latents. Memory-bounded: only the K columns we're scoring, never the
    full (N, T, d_sae) tensor (which is ~25 GB for d_sae=24576).

    v8.16 audit fix: the pre-fix `score_descriptions` was passed
    `activations.pt` directly and treated its last axis as latent
    activations — but `activations.pt` holds residual-stream vectors
    (last axis = d_model), not SAE codes. For latent_ids ≥ d_model the
    indexing went out of bounds; for latent_ids < d_model it scored
    residual dimensions, not latent firing. Both modes silently
    produced garbage scores. This function fixes that by encoding only
    the requested latent columns through the actual pretrained SAE.
    """
    sae = sae.to(device)
    bs = 8192   # rows per matmul chunk

    out = torch.zeros((N * T, len(latent_ids)), dtype=torch.float32)
    latent_t = torch.tensor(latent_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        for i in range(0, x_flat.shape[0], bs):
            xb = x_flat[i : i + bs].to(device)
            try:
                # Native sae_lens path: encode → slice by latent_ids.
                z = sae.encode(xb.to(sae.W_enc.dtype))  # (bs, d_sae)
            except Exception:
                # Fallback: bare-weights computation if `sae.encode` is
                # unavailable for some reason (very rare on the supported
                # sae_lens / GemmaScope paths).
                z = xb.to(sae.W_enc.dtype) @ sae.W_enc + sae.b_enc
                if sae.threshold is not None:
                    z = (z > sae.threshold) * torch.relu(z)
                else:
                    z = torch.relu(z)
            out[i : i + bs] = z.index_select(dim=-1, index=latent_t).float().cpu()

    return out.reshape(N, T, len(latent_ids))


def run(cfg: Config = None, threshold: float = 0.7) -> dict:
    """Standalone runner: score every description in raw_descriptions.json
    against the pretrained SAE's actual latent firing on activations.pt,
    write per-latent scores to disk. Does NOT modify the catalog.

    v8.16 (audit fix): previously this function passed activations.pt
    (residual stream) directly into the scorer's last-axis indexer,
    which scored residual dimensions or went OOB instead of measuring
    latent firing. Now it loads the pretrained SAE, encodes only the
    columns corresponding to descriptions we have, and passes those
    codes to the scorer.
    """
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
    if activations.dim() != 3:
        raise RuntimeError(
            f"activations.pt has unexpected shape {tuple(activations.shape)}; "
            f"expected (N, T, d_model)."
        )
    N, T, d_model = activations.shape
    print(f"  Loaded activations: {tuple(activations.shape)}, "
          f"tokens: {tuple(tokens.shape)}")

    # Load the pretrained SAE so we can encode actual latent activations.
    print(f"  Loading pretrained SAE ({cfg.sae_release} / {cfg.sae_id})...")
    from .inventory import load_sae as _load_pretrained_sae
    sae, _sparsity = _load_pretrained_sae(cfg)

    # Subset to the latents that have descriptions (scoring only the
    # columns we actually use). The keys in `descriptions` are string-form
    # latent indices set by inventory.collect_top_activations.
    latent_ids: list[int] = []
    valid_keys: list[str] = []
    for k in descriptions.keys():
        try:
            lid = int(k)
        except ValueError:
            continue
        if 0 <= lid < sae.d_sae:
            latent_ids.append(lid)
            valid_keys.append(k)

    if not latent_ids:
        raise RuntimeError(
            "No valid latent ids in raw_descriptions.json — keys must be "
            "string-form integer indices into the pretrained SAE."
        )
    print(f"  Encoding pretrained-SAE activations for {len(latent_ids)} "
          f"latents (avoids materializing the full (N,T,d_sae) tensor)...")

    x_flat = activations.reshape(-1, d_model).to(torch.float32)
    sae_codes = _encode_pretrained_sae_for_latents(
        sae, x_flat, N, T, latent_ids, device=cfg.device,
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # score_descriptions expects the last axis of its `activations` arg to
    # be latent-indexed. We've built sae_codes to match `latent_ids`. Map
    # each description's key onto its position-in-stack so the scorer can
    # find the right column.
    pos_keyed_descs = {
        str(i): descriptions[k] for i, k in enumerate(valid_keys)
    }
    pos_keyed_top_acts = {
        str(i): top_activations.get(k, []) for i, k in enumerate(valid_keys)
    }

    print(f"  Scoring {len(pos_keyed_descs)} descriptions...")
    pos_scores = score_descriptions(
        descriptions=pos_keyed_descs,
        top_activations=pos_keyed_top_acts,
        full_tokens=tokens,
        activations=sae_codes,
        latent_offset=0,
        tokenizer=tokenizer,
        cfg=cfg,
        threshold=threshold,
    )

    # Reverse-map back to original latent-id keys for the on-disk record.
    scores: dict = {}
    for i, k in enumerate(valid_keys):
        scores[k] = pos_scores.get(str(i), {
            "score": None, "n_correct": 0, "n_total": 0,
            "kept": True, "reason": "missing_score",
        })

    out_path = cfg.output_dir / "delphi_scores.json"
    out_path.write_text(json.dumps({
        "threshold": threshold,
        "judge_model": (
            getattr(cfg, "delphi_judge_model", "") or cfg.organization_model
        ),
        "sae_release": cfg.sae_release,
        "sae_id": cfg.sae_id,
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
