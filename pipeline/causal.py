"""
Step 8 — Causal Validation

For each supervised latent, zero-ablate it during model forward passes
and measure KL divergence on the output distribution. This tests whether
latents have causal influence on the model's computation — the key property
that separates a supervised SAE from a linear probe.

Protocol:
    1. Run model with SAE reconstruction at target layer (baseline)
    2. For each feature k, run model with SAE reconstruction minus
       feature k's contribution (ablation)
    3. Measure KL(baseline || ablated) at positions where feature k fires

A large KL at firing positions means the latent causally affects model
output. A near-zero KL means the latent is decorative.

Outputs:
    pipeline_data/causal.json

Usage:
    python -m pipeline.run --step causal
"""

import json

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import Config
from .train import SupervisedSAE, set_seed


def run(cfg: Config = None):
    """Run causal validation via zero-ablation of supervised latents."""
    if cfg is None:
        cfg = Config()

    if cfg.causal_path.exists():
        print(f"Causal validation already exists: {cfg.causal_path}")
        return json.loads(cfg.causal_path.read_text())

    # Check prerequisites
    for path, name in [
        (cfg.checkpoint_path, "trained SAE"),
        (cfg.checkpoint_config_path, "SAE config"),
        (cfg.tokens_path, "tokens"),
        (cfg.catalog_path, "feature catalog"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    # Load base model
    from transformer_lens import HookedTransformer

    print("Loading model...")
    model = HookedTransformer.from_pretrained(
        cfg.model_name, device=cfg.device, dtype=cfg.model_dtype
    )
    model.eval()

    # Load supervised SAE
    model_cfg = torch.load(
        cfg.checkpoint_config_path, map_location="cpu", weights_only=True
    )
    sae = SupervisedSAE(
        model_cfg["d_model"],
        model_cfg["n_supervised"],
        model_cfg["n_unsupervised"],
        n_lista_steps=model_cfg.get("n_lista_steps", 0),
    )
    sae.load_state_dict(
        torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    )
    sae.eval().to(cfg.device)

    # Load tokens and feature catalog
    tokens = torch.load(cfg.tokens_path, weights_only=True)
    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]
    n_supervised = model_cfg["n_supervised"]

    # Sample sequences for causal testing
    n_causal = min(cfg.causal_n_sequences, tokens.shape[0])
    set_seed(cfg.seed)
    causal_idx = torch.randperm(tokens.shape[0])[:n_causal]
    causal_tokens = tokens[causal_idx]

    print(f"Causal validation: {n_causal} sequences, {n_supervised} supervised latents")
    batch_size = cfg.causal_batch_size

    # Accumulate per-feature statistics
    stats = [{
        "kl_firing_sum": 0.0, "kl_all_sum": 0.0,
        "n_firing": 0, "n_total": 0, "max_kl": 0.0,
    } for _ in range(n_supervised)]

    # Shared state for hooks
    _hook_state = {}

    def baseline_hook(resid, hook):
        """Replace residual with SAE reconstruction, cache sup_acts."""
        B, T, D = resid.shape
        flat = resid.reshape(-1, D)
        recon, _, sup_acts, _ = sae(flat)
        _hook_state["sup_acts"] = sup_acts.reshape(B, T, -1)
        return recon.reshape(B, T, D)

    def make_ablation_hook(latent_idx):
        """Create a hook that ablates a specific supervised latent."""
        def hook_fn(resid, hook):
            B, T, D = resid.shape
            flat = resid.reshape(-1, D)
            recon, _, sup_acts, _ = sae(flat)
            # Subtract this latent's contribution from reconstruction
            k = latent_idx
            recon = recon - sup_acts[:, k : k + 1] * sae.decoder.weight[:, k : k + 1].T
            return recon.reshape(B, T, D)
        return hook_fn

    # Process batch by batch
    for batch_start in tqdm(
        range(0, n_causal, batch_size), desc="Causal batches"
    ):
        batch_tokens = causal_tokens[
            batch_start : batch_start + batch_size
        ].to(cfg.device)

        # Baseline forward pass
        with torch.no_grad():
            baseline_logits = model.run_with_hooks(
                batch_tokens,
                fwd_hooks=[(cfg.hook_point, baseline_hook)],
            )
            baseline_lp = F.log_softmax(baseline_logits.float(), dim=-1)
            batch_sup_acts = _hook_state["sup_acts"]  # (B, T, n_sup)

        # For each supervised feature, ablate and measure KL
        for k in range(n_supervised):
            fires_k = batch_sup_acts[:, :, k] > 0
            n_fire = int(fires_k.sum().item())

            # Skip if this feature never fires in this batch
            if n_fire == 0:
                stats[k]["n_total"] += fires_k.numel()
                continue

            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    batch_tokens,
                    fwd_hooks=[(cfg.hook_point, make_ablation_hook(k))],
                )
                ablated_lp = F.log_softmax(ablated_logits.float(), dim=-1)

            # KL(baseline || ablated) per position
            kl = (baseline_lp.exp() * (baseline_lp - ablated_lp)).sum(dim=-1)

            stats[k]["kl_firing_sum"] += kl[fires_k].sum().item()
            stats[k]["kl_all_sum"] += kl.sum().item()
            stats[k]["n_firing"] += n_fire
            stats[k]["n_total"] += fires_k.numel()
            stats[k]["max_kl"] = max(
                stats[k]["max_kl"], kl[fires_k].max().item()
            )

            del ablated_logits, ablated_lp, kl

        del baseline_logits, baseline_lp, batch_sup_acts

    # Compile results
    results = []
    for k in range(n_supervised):
        feat = features[k]
        s = stats[k]
        if s["n_firing"] == 0:
            results.append({
                "id": feat["id"], "type": feat["type"],
                "n_firing": 0,
                "mean_kl_firing": None, "mean_kl_all": None,
                "max_kl": None, "specificity": None,
            })
            continue

        mean_kl_firing = s["kl_firing_sum"] / s["n_firing"]
        mean_kl_all = s["kl_all_sum"] / s["n_total"] if s["n_total"] > 0 else 0
        specificity = (
            mean_kl_firing / mean_kl_all if mean_kl_all > 1e-10 else float("inf")
        )

        results.append({
            "id": feat["id"], "type": feat["type"],
            "n_firing": s["n_firing"],
            "mean_kl_firing": round(mean_kl_firing, 6),
            "mean_kl_all": round(mean_kl_all, 6),
            "max_kl": round(s["max_kl"], 4),
            "specificity": round(specificity, 2),
        })

    # Print summary
    print(f"\n{'Feature':<40} {'Fires':>6} {'KL(fire)':>10} "
          f"{'KL(all)':>10} {'Specif':>8} {'MaxKL':>8}")
    print("-" * 84)

    causal_features = [r for r in results if r["mean_kl_firing"] is not None]
    causal_features.sort(key=lambda r: r["mean_kl_firing"], reverse=True)

    for r in causal_features:
        tag = " [G]" if r["type"] == "group" else ""
        print(f"  {r['id']:<38} {r['n_firing']:>6} "
              f"{r['mean_kl_firing']:>10.4f} {r['mean_kl_all']:>10.6f} "
              f"{r['specificity']:>8.1f} {r['max_kl']:>8.4f}{tag}")

    # Summary stats
    kl_values = [r["mean_kl_firing"] for r in results if r["mean_kl_firing"] is not None]
    if kl_values:
        import numpy as np
        kl_arr = np.array(kl_values)
        print(f"\n  Mean KL (firing): {kl_arr.mean():.4f}")
        print(f"  Median KL (firing): {np.median(kl_arr):.4f}")
        print(f"  Features with KL > 0.01: "
              f"{sum(1 for x in kl_values if x > 0.01)}/{len(kl_values)}")
        print(f"  Features with KL > 0.1:  "
              f"{sum(1 for x in kl_values if x > 0.1)}/{len(kl_values)}")

    # Save
    output = {"features": results, "n_sequences": n_causal}
    cfg.causal_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved: {cfg.causal_path}")

    del model, sae
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return output


if __name__ == "__main__":
    run()
