"""
Experiment B — Downstream Circuit Analysis (closing-bracket prediction)

Given a local behavior (prediction of `)` after `(`), count how many latents
carry the signal at layer 20 under three different latent pools:

    (S) supervised latents in our SAE
    (U) unsupervised latents in our SAE
    (P) pretrained GemmaScope SAE latents

Per-latent contribution is computed via attribution patching (first-order
direct effect):

    contribution[i, p] = acts[i, p] * (grad_logit_diff_wrt_resid[p] @ dec_col[i])

where `dec_col[i]` is the decoder column of latent i, and the gradient is
computed by differentiating `logit_diff = logit(')') - mean(logit(non-bracket))`
at the target position with respect to the residual stream at layer 20.

Attribution patching gives a fast linear approximation to the effect of
zeroing each latent — accurate enough to rank latents and count N@80% of
cumulative |attribution| without doing 16384 individual ablation runs.

Output:
    pipeline_data/circuit_comparison.json

Usage:
    python -m pipeline.run --step circuit
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import Config
from .train import SupervisedSAE


# ── Position collection ────────────────────────────────────────────────────

def _collect_bracket_positions(
    tokens: torch.Tensor, tokenizer, max_positions: int = 400,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Find (seq_idx, pos_idx) pairs where:
        - the token at position pos is `(` (open bracket), OR
        - the preceding 1-20 tokens include an unmatched `(`
      AND the NEXT token (pos+1) is `)`.

    We predict the `)` at position pos+1 from the residual at position pos.

    Returns:
        positions: list of (seq_idx, pos_idx) in the corpus
        target_token_ids: list of all token IDs matching `)` variants — the
                          `logit_diff` is computed by summing logits over these
                          IDs and subtracting the mean of non-bracket logits.
    """
    open_ids = _find_token_ids(tokenizer, ["(", " ("])
    close_ids = _find_token_ids(tokenizer, [")", " )"])
    if not open_ids or not close_ids:
        raise RuntimeError("Could not find bracket token IDs in tokenizer")

    positions: list[tuple[int, int]] = []
    open_set = set(open_ids)
    close_set = set(close_ids)

    N, T = tokens.shape
    for seq_idx in range(N):
        depth = 0
        seq = tokens[seq_idx]
        for pos in range(T - 1):
            tid = int(seq[pos].item())
            if tid in open_set:
                depth += 1
            elif tid in close_set:
                depth = max(depth - 1, 0)
            # Check if next token is a close bracket AND depth > 0
            if depth > 0 and int(seq[pos + 1].item()) in close_set:
                positions.append((seq_idx, pos))
                if len(positions) >= max_positions:
                    return positions, close_ids
    return positions, close_ids


def _find_token_ids(tokenizer, variants: list[str]) -> list[int]:
    """Return all token IDs that decode to one of the given variants.

    Handles single-token variants cleanly. For multi-token variants, we only
    use the first token (good enough for bracket characters, which are
    single-token in every BPE tokenizer we care about).
    """
    ids: list[int] = []
    for v in variants:
        try:
            encoded = tokenizer.encode(v, add_special_tokens=False)
        except Exception:
            continue
        if not encoded:
            continue
        # Only accept if the variant is a single token — otherwise the
        # "first" token of a multi-token sequence isn't really a bracket.
        if len(encoded) == 1:
            ids.append(int(encoded[0]))
    return list(dict.fromkeys(ids))  # dedupe, preserve order


def _non_bracket_token_mask(tokenizer, vocab_size: int) -> torch.Tensor:
    """Return a bool mask over the vocabulary marking non-bracket tokens."""
    bracket_ids = set(
        _find_token_ids(tokenizer, ["(", " (", ")", " )", "[", "]", "{", "}"])
    )
    mask = torch.ones(vocab_size, dtype=torch.bool)
    for tid in bracket_ids:
        if 0 <= tid < vocab_size:
            mask[tid] = False
    return mask


# ── Attribution patching for one pool ─────────────────────────────────────

def _compute_attribution_for_pool(
    model, tokens, positions, target_token_ids, non_bracket_mask,
    layer, hook_point, decoder_cols, encode_fn, n_latents,
    model_dtype_torch,
) -> tuple[torch.Tensor, float]:
    """Compute per-latent aggregate |attribution| for one latent pool.

    Args:
        model: HookedTransformer
        tokens: (N, T) token tensor
        positions: list of (seq_idx, pos_idx)
        target_token_ids: list[int] — IDs of `)` variants. logit_diff at a
            position is `sum(logits[ids]) - mean(logits[non_bracket_mask])`.
        non_bracket_mask: (vocab,) bool — tokens used in mean for logit_diff
        layer: int
        hook_point: str
        decoder_cols: (n_latents, d_model) tensor — decoder column i
        encode_fn: callable taking (seq, d_model) activations and returning
                   (seq, n_latents) latent acts
        n_latents: int
        model_dtype_torch: torch.bfloat16 / float32 / etc.

    Returns:
        per_latent_contribution: (n_latents,) tensor of aggregate |attribution|
        baseline_logit_diff: float — logit_diff averaged over positions, without
                             intervention

    Numerical notes:
        All downstream (post-backward) computation is done in fp32 to avoid
        bf16 precision loss in the per-latent multiply. We can't force the
        backward pass itself to run in fp32 because the model is bf16, but the
        ranking is stable as long as we don't compound precision loss in the
        attribution aggregation step.
    """
    # Hoist dtype casts outside the loop
    decoder_cols_dev = decoder_cols.to(model.cfg.device)
    decoder_cols_fp32 = decoder_cols_dev.float()  # (n_latents, d_model) fp32
    non_bracket_mask_gpu = non_bracket_mask.to(model.cfg.device)
    target_ids_tensor = torch.tensor(
        list(target_token_ids), dtype=torch.long, device=model.cfg.device,
    )

    # Group positions by sequence so we can batch per-sequence forward passes
    by_seq: dict[int, list[int]] = {}
    for seq_idx, pos_idx in positions:
        by_seq.setdefault(seq_idx, []).append(pos_idx)

    per_latent = torch.zeros(n_latents, dtype=torch.float32)
    total_diff = 0.0
    count = 0

    for seq_idx in tqdm(list(by_seq.keys()), desc="    attribution"):
        pos_list = by_seq[seq_idx]
        toks = tokens[seq_idx : seq_idx + 1].to(model.cfg.device)

        # Forward pass with cache; let the hook point tensor require grad
        cached: dict = {}

        def cache_hook(resid, hook=None):
            # Keep a grad-tracking copy of resid
            r = resid.detach().clone().requires_grad_(True)
            cached["resid"] = r
            return r

        logits = model.run_with_hooks(
            toks, fwd_hooks=[(hook_point, cache_hook)]
        )

        # logit_diff at each target position. Sum logits over ALL closing-
        # bracket token variants (then subtract mean of non-bracket tokens)
        # so we don't bias attribution toward a specific tokenization variant.
        # Compute in fp32 to stabilize the backward signal at large vocab.
        position_diffs = []
        for pos in pos_list:
            lg = logits[0, pos].float()  # (vocab,) fp32
            target_logit = lg[target_ids_tensor].sum()
            mean_non_bracket = lg[non_bracket_mask_gpu].mean()
            position_diffs.append(target_logit - mean_non_bracket)
        if not position_diffs:
            del cached, logits
            continue
        logit_diff = torch.stack(position_diffs).sum()

        # Backprop: grad of (summed) logit_diff w.r.t. layer-20 residual
        logit_diff.backward()
        grad_resid = cached["resid"].grad  # (1, T, d_model), same dtype as r
        if grad_resid is None:
            del cached, logits, logit_diff
            continue
        grad_resid_fp32 = grad_resid[0].float()  # (T, d_model) fp32
        resid_detached = cached["resid"].detach()  # (1, T, d_model)

        # Encode residual to latent space and project contributions
        with torch.no_grad():
            acts = encode_fn(resid_detached[0])  # (T, n_latents)
            acts_fp32 = acts.float()  # (T, n_latents) fp32

            pos_tensor = torch.tensor(
                pos_list, dtype=torch.long, device=model.cfg.device,
            )
            grad_slice = grad_resid_fp32[pos_tensor]   # (|pos|, d_model)
            acts_slice = acts_fp32[pos_tensor]          # (|pos|, n_latents)

            # grad_slice @ decoder_cols.T -> (|pos|, n_latents)
            dec_grad = grad_slice @ decoder_cols_fp32.T
            # per-position per-latent contribution
            contrib = acts_slice * dec_grad             # (|pos|, n_latents)
            per_latent += contrib.abs().sum(dim=0).cpu()

            for pd in position_diffs:
                total_diff += float(pd.item())
                count += 1

        # Clean up the autograd graph for this sequence
        del cached, logits, logit_diff, grad_resid

    baseline = total_diff / count if count > 0 else 0.0
    return per_latent, baseline


def _n_at_80pct(per_latent: torch.Tensor) -> tuple[int, list[int]]:
    """Min number of latents whose cumulative |attribution| ≥ 80% of the total."""
    total = per_latent.sum().item()
    if total <= 0:
        return 0, []
    sorted_vals, sorted_idxs = torch.sort(per_latent, descending=True)
    cum = 0.0
    used: list[int] = []
    for val, idx in zip(sorted_vals.tolist(), sorted_idxs.tolist()):
        cum += val
        used.append(int(idx))
        if cum >= 0.80 * total:
            break
    return len(used), used


# ── Main run ────────────────────────────────────────────────────────────────

def run(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    from .inventory import load_sae, load_target_model

    # ── Load model ───────────────────────────────────────────────────────
    print("Loading base model...")
    model = load_target_model(cfg)
    tokenizer = model.tokenizer
    d_model = model.cfg.d_model
    vocab_size = model.cfg.d_vocab

    model_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype_torch = model_dtype_map.get(cfg.model_dtype, torch.float32)

    # ── Load supervised SAE ──────────────────────────────────────────────
    print("Loading supervised SAE...")
    model_cfg = torch.load(
        cfg.checkpoint_config_path, map_location="cpu", weights_only=True,
    )
    from .train import load_trained_sae
    sae_sup = load_trained_sae(model_cfg)
    sae_sup.load_state_dict(
        torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    )
    sae_sup = sae_sup.to(cfg.device).to(model_dtype_torch).eval()
    n_supervised = sae_sup.n_supervised
    n_unsup = sae_sup.n_total - n_supervised

    # ── Load pretrained SAE ──────────────────────────────────────────────
    print("Loading pretrained SAE...")
    sae_pre, _ = load_sae(cfg)
    sae_pre = sae_pre.to(cfg.device)
    # Cast pretrained SAE weights to match model dtype
    sae_pre.W_enc = sae_pre.W_enc.to(model_dtype_torch)
    sae_pre.W_dec = sae_pre.W_dec.to(model_dtype_torch)
    sae_pre.b_enc = sae_pre.b_enc.to(model_dtype_torch)
    sae_pre.b_dec = sae_pre.b_dec.to(model_dtype_torch)
    if sae_pre.threshold is not None:
        sae_pre.threshold = sae_pre.threshold.to(model_dtype_torch)
    d_sae_pre = sae_pre.d_sae
    print(f"  Pretrained SAE: {d_sae_pre} latents")

    # ── Collect bracket positions ────────────────────────────────────────
    print("Collecting bracket positions...")
    tokens = torch.load(cfg.tokens_path, weights_only=True)
    positions, target_token_ids = _collect_bracket_positions(
        tokens, tokenizer, max_positions=400,
    )
    print(f"  Found {len(positions)} bracket positions")
    print(f"  Target token IDs: {target_token_ids} "
          f"({[tokenizer.decode([t]) for t in target_token_ids]!r})")
    if len(positions) == 0:
        print("  ERROR: no closing-bracket positions found in the corpus. "
              "Either the corpus has no code/math content or the tokenizer "
              "doesn't match the bracket variants we search for.")
        return
    if len(positions) < 20:
        print("  WARNING: fewer than 20 positions — result will be noisy")

    non_bracket_mask = _non_bracket_token_mask(tokenizer, vocab_size)

    # ── Prepare encode functions + decoder columns per pool ──────────────
    # (S) and (U) share the supervised SAE; we slice afterwards.
    dec_weight_sup = sae_sup.decoder.weight.data  # (d_model, n_total)
    dec_cols_S = dec_weight_sup[:, :n_supervised].T  # (n_supervised, d_model)
    dec_cols_U = dec_weight_sup[:, n_supervised:].T  # (n_unsup, d_model)

    # For (P), use W_dec stored as (d_sae, d_model) — already the right shape
    dec_cols_P = sae_pre.W_dec  # (d_sae_pre, d_model)

    def encode_sup_full(resid: torch.Tensor) -> torch.Tensor:
        """(T, d_model) → (T, n_total) — full supervised SAE acts."""
        with torch.no_grad():
            _, _, _, all_acts = sae_sup(resid)
        return all_acts

    def encode_sup_supervised_slice(resid: torch.Tensor) -> torch.Tensor:
        return encode_sup_full(resid)[:, :n_supervised]

    def encode_sup_unsupervised_slice(resid: torch.Tensor) -> torch.Tensor:
        return encode_sup_full(resid)[:, n_supervised:]

    def encode_pre(resid: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return sae_pre.encode(resid)

    # ── Run the three pools ──────────────────────────────────────────────
    layer = cfg.target_layer
    hook = cfg.hook_point

    print("\n" + "=" * 70)
    print("EXPERIMENT B: CIRCUIT ANALYSIS — closing bracket prediction")
    print("=" * 70)

    print("\n(S) supervised latents")
    per_latent_S, baseline_S = _compute_attribution_for_pool(
        model, tokens, positions, target_token_ids, non_bracket_mask,
        layer, hook, dec_cols_S, encode_sup_supervised_slice,
        n_supervised, model_dtype_torch,
    )
    n80_S, top_S = _n_at_80pct(per_latent_S)

    print("\n(U) unsupervised latents in our SAE")
    per_latent_U, baseline_U = _compute_attribution_for_pool(
        model, tokens, positions, target_token_ids, non_bracket_mask,
        layer, hook, dec_cols_U, encode_sup_unsupervised_slice,
        n_unsup, model_dtype_torch,
    )
    n80_U, top_U = _n_at_80pct(per_latent_U)

    print("\n(P) pretrained SAE latents")
    per_latent_P, baseline_P = _compute_attribution_for_pool(
        model, tokens, positions, target_token_ids, non_bracket_mask,
        layer, hook, dec_cols_P, encode_pre,
        d_sae_pre, model_dtype_torch,
    )
    n80_P, top_P = _n_at_80pct(per_latent_P)

    # ── Report ───────────────────────────────────────────────────────────
    # Look up feature IDs for the top-5 supervised latents
    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]

    def _top_feature_ids(idxs: list[int]) -> list[str]:
        out = []
        for i in idxs[:5]:
            if 0 <= i < len(features):
                out.append(features[i]["id"])
            else:
                out.append(f"idx_{i}")
        return out

    print("\n" + "=" * 70)
    print("CIRCUIT NODE COUNTS (N@80% cumulative attribution)")
    print("-" * 70)
    print(f"  (S) supervised:    {n80_S:>5d} / {n_supervised}")
    print(f"  (U) unsupervised:  {n80_U:>5d} / {n_unsup}")
    print(f"  (P) pretrained:    {n80_P:>5d} / {d_sae_pre}")
    print(f"\nBaseline logit_diff: S={baseline_S:.3f}  U={baseline_U:.3f}  "
          f"P={baseline_P:.3f}")

    print(f"\nTop-5 supervised latents (by |attribution|):")
    for fid in _top_feature_ids(top_S[:5]):
        print(f"    {fid}")

    print(f"\nTop-5 unsupervised latents (by |attribution|):")
    for lat_idx in top_U[:5]:
        print(f"    u{lat_idx}  (attribution={per_latent_U[lat_idx]:.4f})")

    print(f"\nTop-5 pretrained latents (by |attribution|):")
    for lat_idx in top_P[:5]:
        print(f"    p{lat_idx}  (attribution={per_latent_P[lat_idx]:.4f})")

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "experiment": "B_circuit_analysis",
        "behavior": "closing_bracket_prediction",
        "n_positions": len(positions),
        "target_token_ids": [int(t) for t in target_token_ids],
        "baseline_logit_diff": {
            "S": round(baseline_S, 6),
            "U": round(baseline_U, 6),
            "P": round(baseline_P, 6),
        },
        "node_counts": {
            "S_n_at_80": int(n80_S),
            "U_n_at_80": int(n80_U),
            "P_n_at_80": int(n80_P),
        },
        "pool_sizes": {
            "S": int(n_supervised),
            "U": int(n_unsup),
            "P": int(d_sae_pre),
        },
        "top_latents": {
            "S": [
                {"latent_idx": int(i), "feature_id": fid,
                 "attribution": round(per_latent_S[i].item(), 6)}
                for i, fid in zip(top_S[:20], _top_feature_ids(top_S[:20]))
            ],
            "U": [
                {"latent_idx": int(i),
                 "attribution": round(per_latent_U[i].item(), 6)}
                for i in top_U[:20]
            ],
            "P": [
                {"latent_idx": int(i),
                 "attribution": round(per_latent_P[i].item(), 6)}
                for i in top_P[:20]
            ],
        },
    }
    out_path = cfg.output_dir / "circuit_comparison.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")
