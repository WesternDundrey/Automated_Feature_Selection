"""
Step 7 — Explain the Residual

After training a supervised SAE, analyze what the reconstruction misses.
Find token positions with highest reconstruction error, extract their context,
and ask Claude to hypothesize what concepts the residual represents.

This produces candidate features for the next iteration of the pipeline.

Outputs:
    pipeline_data/residual_features.json  — proposed new features

Usage:
    python -m pipeline.run --step residual
"""

import json
import textwrap
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import Config
from .train import SupervisedSAE, set_seed


def run(cfg: Config = None):
    """Analyze reconstruction residuals and propose new features."""
    if cfg is None:
        cfg = Config()

    if cfg.residual_path.exists():
        print(f"Residual analysis already exists: {cfg.residual_path}")
        return json.loads(cfg.residual_path.read_text())

    # Load model
    for path, name in [
        (cfg.checkpoint_path, "trained SAE"),
        (cfg.checkpoint_config_path, "SAE config"),
        (cfg.activations_path, "activations"),
        (cfg.tokens_path, "tokens"),
        (cfg.catalog_path, "feature catalog"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    model_cfg = torch.load(cfg.checkpoint_config_path, map_location="cpu", weights_only=True)
    from .train import load_trained_sae
    sae = load_trained_sae(model_cfg)
    sae.load_state_dict(
        torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    )
    sae.eval().to(cfg.device)

    activations = torch.load(cfg.activations_path, weights_only=True)
    tokens = torch.load(cfg.tokens_path, weights_only=True)
    catalog = json.loads(cfg.catalog_path.read_text())
    existing_features = catalog["features"]

    N, T, d_model = activations.shape

    # Compute per-position reconstruction error
    print("Computing per-position reconstruction error...")
    x_flat = activations.reshape(-1, d_model)

    # Sample a subset for analysis
    n_samples = min(cfg.residual_n_samples * T, x_flat.shape[0])
    set_seed(cfg.seed)
    sample_idx = torch.randperm(x_flat.shape[0])[:n_samples]
    x_sample = x_flat[sample_idx]

    errors = []
    with torch.no_grad():
        for i in range(0, x_sample.shape[0], cfg.batch_size):
            x_b = x_sample[i : i + cfg.batch_size].to(cfg.device)
            recon, _, _, _ = sae(x_b)
            err = (recon - x_b).pow(2).sum(dim=-1)  # per-position MSE
            errors.append(err.cpu())

    errors = torch.cat(errors)

    # Find top-k highest error positions
    top_k = min(cfg.residual_top_k_positions, len(errors))
    top_errors, top_local_idx = errors.topk(top_k)
    top_global_idx = sample_idx[top_local_idx]

    # Map flat indices back to (sequence, position)
    seq_indices = top_global_idx // T
    pos_indices = top_global_idx % T

    print(f"  Top {top_k} error positions: MSE range [{top_errors[-1].item():.2f}, "
          f"{top_errors[0].item():.2f}]")
    print(f"  Mean MSE (sample): {errors.mean().item():.4f}")

    # Load tokenizer (lightweight — no need for full model)
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Build context strings for high-error positions
    contexts = []
    for i in range(top_k):
        seq_i = seq_indices[i].item()
        pos_i = pos_indices[i].item()
        err_i = top_errors[i].item()

        seq_tokens = tokens[seq_i].tolist()
        start = max(0, pos_i - 10)
        end = min(len(seq_tokens), pos_i + 11)

        tok_strs = []
        for t in seq_tokens[start:end]:
            try:
                tok_strs.append(tokenizer.decode([t]))
            except Exception:
                tok_strs.append(f"<{t}>")

        target_pos = pos_i - start
        if 0 <= target_pos < len(tok_strs):
            tok_strs[target_pos] = f"<<{tok_strs[target_pos]}>>"

        context_str = "".join(tok_strs)
        contexts.append({
            "context": context_str,
            "mse": round(err_i, 4),
            "seq": seq_i,
            "pos": pos_i,
        })

    # Ask Claude to hypothesize what the residual represents
    existing_desc = "\n".join(
        f"  - {f['id']}: {f['description']}" for f in existing_features
    )

    context_block = "\n".join(
        f"  [{i+1}] (MSE={c['mse']:.2f}) {c['context']}"
        for i, c in enumerate(contexts[:50])  # send top 50 to prompt
    )

    prompt = textwrap.dedent(f"""\
        You are analyzing a supervised sparse autoencoder trained on
        {cfg.model_name}, layer {cfg.target_layer}.

        The SAE already captures these features:
        {existing_desc}

        Below are the {min(50, len(contexts))} token positions with the HIGHEST
        reconstruction error (marked with <<token>>). These are positions where
        the current feature dictionary fails to capture the model's computation.

        HIGH-ERROR POSITIONS:
        {context_block}

        YOUR TASK:
        1. Look for patterns in what the SAE is missing. What concepts, token
           types, or contextual roles appear repeatedly in the high-error positions?
        2. Propose 5-15 NEW features that would help capture these patterns.
           Each feature should be:
           - Short and operationally testable (yes/no for a token in context)
           - NOT redundant with the existing features listed above
           - Organized into groups where natural

        OUTPUT FORMAT — reply with ONLY this JSON:
        {{
          "analysis": "Brief description of what the residual seems to encode",
          "proposed_features": [
            {{
              "id": "group.feature_name",
              "description": "Precise operational description",
              "type": "leaf",
              "parent": "group",
              "rationale": "Why this feature would reduce reconstruction error"
            }}
          ]
        }}
    """)

    print(f"\nAsking {cfg.residual_model} to analyze residual patterns...")
    from .llm import get_client, chat
    client = get_client()

    from .inventory import _extract_json_object

    result = None
    for attempt in range(3):
        try:
            text = chat(client, cfg.residual_model, prompt, max_tokens=4000)
            result = _extract_json_object(text)
            if result is not None:
                break
        except Exception as e:
            if attempt < 2:
                print(f"  Attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(2 ** attempt)

    if result is None:
        raise ValueError("Failed to get residual analysis from Claude after 3 attempts")

    # Save results
    output = {
        "analysis": result.get("analysis", ""),
        "proposed_features": result.get("proposed_features", []),
        "error_stats": {
            "mean_mse": round(errors.mean().item(), 6),
            "max_mse": round(top_errors[0].item(), 4),
            "top_k_mean_mse": round(top_errors.mean().item(), 4),
        },
        "high_error_contexts": contexts[:20],  # save top 20 for reference
        "n_existing_features": len(existing_features),
    }

    n_proposed = len(output["proposed_features"])
    print(f"\n  Analysis: {output['analysis'][:200]}...")
    print(f"  Proposed {n_proposed} new features")

    print("\n  Proposed features:")
    for f in output["proposed_features"]:
        print(f"    {f['id']}: {f['description']}")

    cfg.residual_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved: {cfg.residual_path}")
    print("\nTo add these features to the catalog and retrain, manually merge them")
    print("into feature_catalog.json and re-run annotation + training steps.")

    sae.cpu()
    return output


if __name__ == "__main__":
    run()
