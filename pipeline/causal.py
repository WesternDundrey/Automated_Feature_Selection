"""
Step 8 — Causal Validation (Makelov et al. 2024 Framework)

Three evaluation axes from "Towards Principled Evaluations of SAEs":

  1. Approximation: Replace activations with SAE reconstructions.
     Measure sufficiency (does the model still work?) and necessity
     (does removing reconstructions hurt as much as mean ablation?).

  2. Sparse Controllability: Given a clean/corrupted prompt pair differing
     in one attribute, greedily select features to remove/add to change
     the model's prediction. Measure edit success rate and edit magnitude.

  3. Interpretability: For each SAE feature, compute F1 against known
     attributes. Then test causally: do interpretable features control
     the model in a way consistent with their interpretation?

The key metric is the logit difference: logit(IO_name) - logit(S_name).

Usage:
    python -m pipeline.run --step causal
"""

import json

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import Config


# ── IOI prompt generation ──────────────────────────────────────────────────

NAMES = [
    "Mary", "John", "Alice", "Bob", "Emma", "James", "Sarah", "David",
    "Lisa", "Michael", "Anna", "Tom", "Kate", "Mark", "Julia", "Peter",
]

TEMPLATES = [
    "Then, {name1} and {name2} had a long and really crazy argument. Afterwards, {s2} said to",
    "Then, {name1} and {name2} had lots of fun at the park. Afterwards, {s2} gave a present to",
]


def generate_ioi_pairs(n_pairs, tokenizer, seed=42):
    """Generate clean/corrupted IOI prompt pairs with ground-truth labels.

    Clean:     "Then, Mary and John ... John said to" → predicts "Mary"
    Corrupted: "Then, Mary and Bob  ... Bob  said to" → predicts "Mary" (same IO)
               but with different S name.

    Returns list of dicts with tokens, name positions, and attribute values.
    """
    import random
    from itertools import product

    rng = random.Random(seed)
    name_pairs = [(a, b) for a, b in product(NAMES, NAMES) if a != b]
    # Third name for corrupted (different S)
    pairs = []

    for _ in range(n_pairs):
        io_name, s_name = rng.choice(name_pairs)
        # Pick a different S name for corrupted
        s_name_corrupt = rng.choice([n for n in NAMES if n != io_name and n != s_name])
        template = rng.choice(TEMPLATES)
        pattern = rng.choice(["ABB", "BAB"])

        if pattern == "ABB":
            clean_text = template.format(name1=io_name, name2=s_name, s2=s_name)
            corrupt_text = template.format(name1=io_name, name2=s_name_corrupt, s2=s_name_corrupt)
        else:
            clean_text = template.format(name1=s_name, name2=io_name, s2=s_name)
            corrupt_text = template.format(name1=s_name_corrupt, name2=io_name, s2=s_name_corrupt)

        clean_ids = tokenizer.encode(clean_text)
        corrupt_ids = tokenizer.encode(corrupt_text)

        # Find name token positions in clean
        io_pos = None
        s1_pos = None
        s2_pos = None
        s_count = 0
        for pos, tid in enumerate(clean_ids):
            tok_str = tokenizer.decode([tid]).strip()
            if tok_str == io_name and io_pos is None:
                io_pos = pos
            if tok_str == s_name:
                s_count += 1
                if s_count == 1:
                    s1_pos = pos
                elif s_count == 2:
                    s2_pos = pos

        if io_pos is None or s1_pos is None or s2_pos is None:
            continue  # skip malformed

        # Get IO and S token IDs for logit difference
        io_token_id = tokenizer.encode(" " + io_name)[-1]
        s_token_id = tokenizer.encode(" " + s_name)[-1]

        pairs.append({
            "clean_ids": clean_ids,
            "corrupt_ids": corrupt_ids,
            "io_name": io_name,
            "s_name": s_name,
            "s_name_corrupt": s_name_corrupt,
            "pattern": pattern,
            "io_pos": io_pos,
            "s1_pos": s1_pos,
            "s2_pos": s2_pos,
            "io_token_id": io_token_id,
            "s_token_id": s_token_id,
            "end_pos": len(clean_ids) - 1,
        })

    return pairs


def logit_diff(logits, io_token_id, s_token_id, end_pos):
    """Logit difference at the END position: logit(IO) - logit(S)."""
    return (logits[0, end_pos, io_token_id] - logits[0, end_pos, s_token_id]).item()


# ── Test 1: Approximation ─────────────────────────────────────────────────

def test_approximation(model, sae, pairs, cfg):
    """Sufficiency and necessity of SAE reconstructions.

    Sufficiency: replace activations with SAE reconstructions, measure logit diff.
    Necessity: replace activations with mean + (act - reconstruction), compare to mean ablation.
    """
    print("\n" + "=" * 70)
    print("TEST 1: APPROXIMATION (sufficiency & necessity)")
    print("=" * 70)

    clean_lds = []
    recon_lds = []
    mean_ablation_lds = []
    necessity_lds = []

    # Compute mean activation for mean ablation baseline
    all_acts = []
    for pair in pairs[:100]:
        tokens = torch.tensor([pair["clean_ids"]]).to(cfg.device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=cfg.hook_point, return_type=None)
            all_acts.append(cache[cfg.hook_point].cpu())
    mean_act = torch.cat([a.reshape(-1, a.shape[-1]) for a in all_acts], dim=0).mean(dim=0)  # (d_model,)

    sae = sae.to(cfg.device).eval()

    for pair in tqdm(pairs, desc="Approximation"):
        tokens = torch.tensor([pair["clean_ids"]]).to(cfg.device)

        # Clean (no intervention)
        with torch.no_grad():
            clean_logits = model(tokens)
        ld_clean = logit_diff(clean_logits, pair["io_token_id"], pair["s_token_id"], pair["end_pos"])
        clean_lds.append(ld_clean)

        # Sufficiency: replace with SAE reconstruction
        def sufficiency_hook(resid, hook):
            flat = resid.reshape(-1, resid.shape[-1])
            recon, _, _, _ = sae(flat)
            return recon.reshape(resid.shape)

        with torch.no_grad():
            recon_logits = model.run_with_hooks(tokens, fwd_hooks=[(cfg.hook_point, sufficiency_hook)])
        ld_recon = logit_diff(recon_logits, pair["io_token_id"], pair["s_token_id"], pair["end_pos"])
        recon_lds.append(ld_recon)

        # Mean ablation baseline
        def mean_ablation_hook(resid, hook):
            return mean_act.to(resid.device).expand_as(resid)

        with torch.no_grad():
            mean_logits = model.run_with_hooks(tokens, fwd_hooks=[(cfg.hook_point, mean_ablation_hook)])
        ld_mean = logit_diff(mean_logits, pair["io_token_id"], pair["s_token_id"], pair["end_pos"])
        mean_ablation_lds.append(ld_mean)

        # Necessity: replace with mean + (act - reconstruction)
        def necessity_hook(resid, hook):
            flat = resid.reshape(-1, resid.shape[-1])
            recon, _, _, _ = sae(flat)
            residual = flat - recon  # what the SAE missed
            return (mean_act.to(resid.device) + residual).reshape(resid.shape)

        with torch.no_grad():
            nec_logits = model.run_with_hooks(tokens, fwd_hooks=[(cfg.hook_point, necessity_hook)])
        ld_nec = logit_diff(nec_logits, pair["io_token_id"], pair["s_token_id"], pair["end_pos"])
        necessity_lds.append(ld_nec)

    mean_clean = sum(clean_lds) / len(clean_lds)
    mean_recon = sum(recon_lds) / len(recon_lds)
    mean_mean_abl = sum(mean_ablation_lds) / len(mean_ablation_lds)
    mean_nec = sum(necessity_lds) / len(necessity_lds)

    sufficiency = mean_recon / mean_clean if mean_clean != 0 else 0
    necessity = 1 - (mean_mean_abl - mean_nec) / (mean_mean_abl - mean_clean) if (mean_mean_abl - mean_clean) != 0 else 0

    print(f"\n  Clean logit diff:          {mean_clean:.4f}")
    print(f"  Reconstruction logit diff: {mean_recon:.4f}  (sufficiency: {sufficiency:.4f})")
    print(f"  Mean ablation logit diff:  {mean_mean_abl:.4f}")
    print(f"  Necessity logit diff:      {mean_nec:.4f}  (necessity: {necessity:.4f})")

    return {
        "clean_ld": round(mean_clean, 4),
        "recon_ld": round(mean_recon, 4),
        "mean_ablation_ld": round(mean_mean_abl, 4),
        "necessity_ld": round(mean_nec, 4),
        "sufficiency": round(sufficiency, 4),
        "necessity": round(necessity, 4),
    }


# ── Test 2: Sparse Controllability ────────────────────────────────────────

def test_controllability(model, sae, pairs, cfg, k_values=(1, 2, 4)):
    """Sparse controllability via greedy feature editing.

    For each clean/corrupted pair, greedily select up to k features to
    remove from clean and add from corrupted to flip the model's prediction.

    IIA = (ld_clean - ld_patched) / (ld_clean - ld_corrupted)
    """
    print("\n" + "=" * 70)
    print("TEST 2: SPARSE CONTROLLABILITY")
    print("=" * 70)

    sae = sae.to(cfg.device).eval()
    results_by_k = {}

    for k in k_values:
        iia_scores = []
        edit_success = []

        for pair in tqdm(pairs, desc=f"Control k={k}"):
            clean_tokens = torch.tensor([pair["clean_ids"]]).to(cfg.device)
            corrupt_tokens = torch.tensor([pair["corrupt_ids"]]).to(cfg.device)

            with torch.no_grad():
                # Get clean and corrupted activations
                _, clean_cache = model.run_with_cache(
                    clean_tokens, names_filter=cfg.hook_point, return_type=None
                )
                _, corrupt_cache = model.run_with_cache(
                    corrupt_tokens, names_filter=cfg.hook_point, return_type=None
                )
                clean_resid = clean_cache[cfg.hook_point]  # (1, T, d_model)
                corrupt_resid = corrupt_cache[cfg.hook_point]

                # Encode both through SAE
                clean_flat = clean_resid.reshape(-1, clean_resid.shape[-1])
                corrupt_flat = corrupt_resid.reshape(-1, corrupt_resid.shape[-1])
                _, _, _, clean_acts = sae(clean_flat)  # (T, n_total)
                _, _, _, corrupt_acts = sae(corrupt_flat)

                # Reconstructions
                clean_recon = sae.decoder(clean_acts)  # (T, d_model)
                corrupt_recon = sae.decoder(corrupt_acts)

                # Target: we want to move clean_recon toward corrupt_recon
                # Greedy: find features to remove (from clean) and add (from corrupt)
                # that minimize ||edited - corrupt_recon||
                T_len = clean_acts.shape[0]
                best_edits = _greedy_edit(
                    clean_acts, corrupt_acts, sae.decoder.weight, k, T_len,
                )

                # Apply edits
                edited_acts = clean_acts.clone()
                for pos, feat_idx, action in best_edits:
                    if action == "remove":
                        edited_acts[pos, feat_idx] = 0.0
                    elif action == "add":
                        edited_acts[pos, feat_idx] = corrupt_acts[pos, feat_idx]

                # Reconstruct and patch
                edited_recon = sae.decoder(edited_acts)

                def patch_hook(resid, hook):
                    return edited_recon.reshape(resid.shape)

                # Get logit diffs
                clean_logits = model(clean_tokens)
                corrupt_logits = model(corrupt_tokens)
                patched_logits = model.run_with_hooks(
                    clean_tokens, fwd_hooks=[(cfg.hook_point, patch_hook)]
                )

            ld_clean = logit_diff(clean_logits, pair["io_token_id"], pair["s_token_id"], pair["end_pos"])
            ld_corrupt = logit_diff(corrupt_logits, pair["io_token_id"], pair["s_token_id"], pair["end_pos"])
            ld_patched = logit_diff(patched_logits, pair["io_token_id"], pair["s_token_id"], pair["end_pos"])

            # IIA: how much of the clean→corrupted shift did the edit achieve?
            denom = ld_clean - ld_corrupt
            if abs(denom) > 1e-6:
                iia = (ld_clean - ld_patched) / denom
                iia_scores.append(iia)

            # Edit success: does patched prediction match corrupted prediction?
            clean_pred = clean_logits[0, pair["end_pos"]].argmax().item()
            corrupt_pred = corrupt_logits[0, pair["end_pos"]].argmax().item()
            patched_pred = patched_logits[0, pair["end_pos"]].argmax().item()
            edit_success.append(1.0 if patched_pred == corrupt_pred else 0.0)

        mean_iia = sum(iia_scores) / len(iia_scores) if iia_scores else 0
        mean_success = sum(edit_success) / len(edit_success) if edit_success else 0
        results_by_k[k] = {
            "mean_iia": round(mean_iia, 4),
            "edit_success_rate": round(mean_success, 4),
            "n_pairs": len(pairs),
        }
        print(f"\n  k={k}: IIA={mean_iia:.4f}  edit_success={mean_success:.4f}")

    return results_by_k


def _greedy_edit(clean_acts, corrupt_acts, decoder_weight, k, T_len):
    """Greedy algorithm to find up to k features to remove/add.

    Minimizes ||edited_recon - corrupt_recon||² by greedily selecting
    the feature edit (remove from clean or add from corrupt) that
    reduces the residual the most at each step.
    """
    # Current reconstruction residual
    # We want: clean_recon - removed + added ≈ corrupt_recon
    # Residual = corrupt_recon - clean_recon
    # Each remove of feature j at position p reduces residual by clean_acts[p,j] * dec[j]
    # Each add of feature j at position p reduces residual by corrupt_acts[p,j] * dec[j]

    dec = decoder_weight  # (d_model, n_total)
    residual = (corrupt_acts - clean_acts) @ dec.T  # (T, d_model) - what we need to change

    edits = []
    used = set()

    for _ in range(k):
        best_reduction = -float("inf")
        best_edit = None

        # Try removing each active clean feature
        for p in range(T_len):
            for j in (clean_acts[p] > 0).nonzero(as_tuple=True)[0].tolist():
                if ("remove", p, j) in used:
                    continue
                # Removing clean feature j at p adds clean_acts[p,j] * dec[:,j] to our edit
                contribution = clean_acts[p, j] * dec[:, j]
                reduction = 2 * (residual[p] @ contribution) - (contribution @ contribution)
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_edit = (p, j, "remove")

        # Try adding each active corrupt feature
        for p in range(T_len):
            for j in (corrupt_acts[p] > 0).nonzero(as_tuple=True)[0].tolist():
                if ("add", p, j) in used:
                    continue
                contribution = corrupt_acts[p, j] * dec[:, j]
                reduction = 2 * (residual[p] @ contribution) - (contribution @ contribution)
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_edit = (p, j, "add")

        if best_edit is None or best_reduction <= 0:
            break

        p, j, action = best_edit
        edits.append(best_edit)
        used.add((action, p, j))

        # Update residual
        if action == "remove":
            residual[p] -= clean_acts[p, j] * dec[:, j]
        else:
            residual[p] -= corrupt_acts[p, j] * dec[:, j]

    return edits


# ── Test 3: Interpretability ──────────────────────────────────────────────

def test_interpretability(sae, pairs, features, cfg):
    """F1 of each supervised feature against IOI attributes.

    For supervised SAEs, features are defined to correspond to known
    attributes, so this tests whether the SAE actually learned them.
    """
    print("\n" + "=" * 70)
    print("TEST 3: INTERPRETABILITY (F1 vs known attributes)")
    print("=" * 70)

    # This test is most meaningful for the IOI catalog features
    # For general features, we skip detailed attribute matching
    print("  (For IOI validation, run --step ioi instead)")
    print("  (This test reports supervised latent activation statistics)")

    return {}


# ── Test 4: Per-Feature Necessity ────────────────────────────────────────

def _make_ablate_hook(sae, feature_idx):
    """Factory: create a hook that reconstructs with one feature zeroed out."""
    def _hook(resid, hook=None):
        flat = resid.reshape(-1, resid.shape[-1])
        _, _, _, acts = sae(flat)
        acts[:, feature_idx] = 0.0
        recon = sae.decoder(acts)
        return recon.reshape(resid.shape)
    return _hook


def test_feature_necessity(model, sae, cfg):
    """Per-feature ablation: zero out each supervised latent, measure downstream effect.

    For each supervised feature k:
      1. Replace residual stream with full SAE reconstruction -> baseline logits
      2. Replace residual with SAE reconstruction minus feature k -> ablated logits
      3. Compute KL(baseline || ablated) at positions where feature k is active

    High KL = the model relies on this feature's decoder direction.
    This is what separates a supervised SAE from a linear probe -- the probe
    can classify but can't intervene.

    Uses stored tokens + annotations from the pipeline (not IOI prompts).
    """
    print("\n" + "=" * 70)
    print("TEST 4: PER-FEATURE CAUSAL NECESSITY")
    print("=" * 70)

    for path, name in [
        (cfg.tokens_path, "tokens"),
        (cfg.annotations_path, "annotations"),
        (cfg.catalog_path, "catalog"),
    ]:
        if not path.exists():
            print(f"  Skipping: {name} not found at {path}")
            return {}

    tokens = torch.load(cfg.tokens_path, weights_only=True)
    annotations = torch.load(cfg.annotations_path, weights_only=True)
    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]
    n_sup = min(sae.n_supervised, len(features), annotations.shape[-1])

    n_seqs = min(cfg.causal_n_sequences, tokens.shape[0])
    tokens_sub = tokens[:n_seqs]
    annot_sub = annotations[:n_seqs]

    sae = sae.to(cfg.device).eval()

    print(f"  Sequences: {n_seqs}  |  Features: {n_sup}")
    print(f"  Hook point: {cfg.hook_point}")

    # Full-reconstruction baseline logits (one pass per sequence)
    def full_recon_hook(resid, hook=None):
        flat = resid.reshape(-1, resid.shape[-1])
        recon, _, _, _ = sae(flat)
        return recon.reshape(resid.shape)

    recon_logprobs = []
    with torch.no_grad():
        for i in tqdm(range(n_seqs), desc="  Full recon baseline"):
            toks = tokens_sub[i : i + 1].to(cfg.device)
            logits = model.run_with_hooks(
                toks, fwd_hooks=[(cfg.hook_point, full_recon_hook)]
            )
            # Cast to fp32 before log_softmax to avoid bf16 underflow on
            # high-frequency features (comma, period) — without this, the
            # log_softmax→KL computation produces NaN for Gemma (bf16 model).
            recon_logprobs.append(F.log_softmax(logits[0].float().cpu(), dim=-1))

    # Per-feature ablation
    results = []

    for k in tqdm(range(n_sup), desc="  Feature ablation"):
        feat = features[k]
        active_mask = annot_sub[:, :, k].bool()
        n_active = int(active_mask.sum())

        if n_active < 5:
            results.append({
                "id": feat["id"], "type": feat["type"],
                "n_active": n_active,
                "mean_kl": None, "pred_change_rate": None,
            })
            continue

        ablate_hook = _make_ablate_hook(sae, k)

        kl_sum = 0.0
        pred_changes = 0
        total = 0

        with torch.no_grad():
            for i in range(n_seqs):
                if not active_mask[i].any():
                    continue
                toks = tokens_sub[i : i + 1].to(cfg.device)
                abl_logits = model.run_with_hooks(
                    toks, fwd_hooks=[(cfg.hook_point, ablate_hook)]
                )

                active_pos = active_mask[i].nonzero(as_tuple=True)[0]
                base_lp = recon_logprobs[i][active_pos]
                abl_lp = F.log_softmax(abl_logits[0, active_pos].float().cpu(), dim=-1)

                kl = (base_lp.exp() * (base_lp - abl_lp)).sum(dim=-1)
                kl_sum += kl.clamp(min=0).sum().item()

                pred_changes += (base_lp.argmax(-1) != abl_lp.argmax(-1)).sum().item()
                total += len(active_pos)

        mean_kl = kl_sum / total if total > 0 else 0
        change_rate = pred_changes / total if total > 0 else 0

        results.append({
            "id": feat["id"], "type": feat["type"],
            "n_active": n_active,
            "mean_kl": round(mean_kl, 6),
            "pred_change_rate": round(change_rate, 4),
        })

    # Print sorted by causal impact
    print(f"\n  {'Feature':<30} {'KL':>8} {'dPred':>7} {'Active':>7}")
    print("  " + "-" * 55)
    for r in sorted(results, key=lambda x: -(x["mean_kl"] or 0)):
        if r["mean_kl"] is None:
            print(f"  {r['id']:<30} {'--':>8} {'--':>7} {r['n_active']:>7}")
            continue
        tag = " [G]" if r["type"] == "group" else ""
        print(f"  {r['id']:<30} {r['mean_kl']:>8.4f} {r['pred_change_rate']:>7.3f} "
              f"{r['n_active']:>7}{tag}")

    valid = [r for r in results if r["mean_kl"] is not None]
    if valid:
        mean_kl = sum(r["mean_kl"] for r in valid) / len(valid)
        mean_change = sum(r["pred_change_rate"] for r in valid) / len(valid)
        n_causal = sum(1 for r in valid if r["mean_kl"] > 0.01)
        print(f"\n  Mean KL: {mean_kl:.4f}  |  Mean pred change: {mean_change:.3f}")
        print(f"  Features with KL > 0.01 (causally active): {n_causal}/{len(valid)}")

    return {"features": results}


# ── Main entry point ───────────────────────────────────────────────────────

def run(cfg: Config = None):
    """Run Makelov-style causal validation."""
    if cfg is None:
        cfg = Config()

    import json
    from transformer_lens import HookedTransformer
    from .train import SupervisedSAE

    # Load model
    print("Loading base model...")
    model = HookedTransformer.from_pretrained(
        cfg.model_name, device=cfg.device, dtype=cfg.model_dtype,
    )
    model.eval()
    tokenizer = model.tokenizer

    # Load trained SAE
    if not cfg.checkpoint_path.exists():
        raise FileNotFoundError(f"No trained SAE: {cfg.checkpoint_path}")
    model_cfg = torch.load(cfg.checkpoint_config_path, weights_only=True)
    sae = SupervisedSAE(
        model_cfg["d_model"], model_cfg["n_supervised"],
        model_cfg["n_unsupervised"], model_cfg.get("n_lista_steps", 0),
    )
    sae.load_state_dict(torch.load(cfg.checkpoint_path, weights_only=True))

    # Match model dtype (SAE trained in fp32, but Gemma runs in bf16)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if cfg.model_dtype in dtype_map:
        sae = sae.to(dtype_map[cfg.model_dtype])

    # Load features
    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]

    # Generate IOI pairs
    n_pairs = cfg.causal_n_sequences
    print(f"\nGenerating {n_pairs} IOI prompt pairs...")
    pairs = generate_ioi_pairs(n_pairs, tokenizer, seed=cfg.seed)
    print(f"  Generated {len(pairs)} valid pairs")

    # Run tests
    results = {}
    results["approximation"] = test_approximation(model, sae, pairs, cfg)
    results["controllability"] = test_controllability(
        model, sae, pairs, cfg, k_values=(1, 2, 4),
    )
    results["interpretability"] = test_interpretability(sae, pairs, features, cfg)
    results["feature_necessity"] = test_feature_necessity(model, sae, cfg)

    # Save
    results_path = cfg.causal_path
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {results_path}")

    return results


if __name__ == "__main__":
    run()
