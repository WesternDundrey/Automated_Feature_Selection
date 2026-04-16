"""
Plan 2 — Discovery Pipeline: Unsupervised SAE → Delphi/Sonnet → Catalog

Mentor-proposed methodology. Inverts the usual top-down flow (hand-written
catalog → supervised SAE) by using unsupervised SAE training for discovery
and LLM annotation for semantic grounding.

    Flow:
      1. Train a minimal unsupervised SAE on cached residual-stream activations.
      2. Select latents with firing rate in [min_firing_rate, max_firing_rate].
      3. Collect top-k activating contexts per selected latent (streaming).
      4. Generate descriptions via the existing inventory explainer (Sonnet).
      5. Organize descriptions into a hierarchical catalog.
      6. Save to `discovered_catalog.json` (does NOT overwrite the main catalog).

    What this validates:
      • If supervised F1 on the discovered catalog is high → LLM descriptions
        captured real directions (bottom-up discovery works).
      • If F1 is low / features collapse → descriptions hallucinated coherence
        onto polysemantic latents (discovery has a ceiling).

    Complements the top-down pipeline: use `--step discover` to generate a
    catalog, then re-run `python -m pipeline.run --catalog <path>` to train a
    supervised SAE against those discovered features.

Outputs:
    pipeline_data/unsupervised_sae.pt          Trained unsupervised SAE
    pipeline_data/discovered_top_activations.json  Top-k contexts per latent
    pipeline_data/discovered_descriptions.json     Raw LLM descriptions
    pipeline_data/discovered_catalog.json          Final hierarchical catalog

Usage:
    python -m pipeline.run --step discover

    # Then train supervised SAE against the discovered catalog:
    python -m pipeline.run --catalog pipeline_data/discovered_catalog.json
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .config import Config
from .train import set_seed


# ── Minimal unsupervised SAE ────────────────────────────────────────────────

class UnsupervisedSAE(nn.Module):
    """Vanilla ReLU SAE trained on reconstruction + L1 only. No labels."""

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x):
        acts = F.relu(self.encoder(x))
        recon = self.decoder(acts)
        return recon, acts

    def encode(self, x):
        """Return post-ReLU activations (compatible with PretrainedSAE API)."""
        return F.relu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


def train_unsupervised_sae(activations: torch.Tensor, cfg: Config,
                            d_sae: int) -> UnsupervisedSAE:
    """Train an unsupervised SAE on cached activations. No labels used."""
    set_seed(cfg.seed)
    N, T, d_model = activations.shape
    x_flat = activations.reshape(-1, d_model)

    sae = UnsupervisedSAE(d_model, d_sae).to(cfg.device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=cfg.lr, weight_decay=0.0)

    loader = DataLoader(
        TensorDataset(x_flat),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    total_steps = cfg.epochs * len(loader)
    decay_start = int(total_steps * 2 / 3)
    decay_length = total_steps - decay_start

    def lr_lambda(step):
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(decay_length, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    baseline_mse = F.mse_loss(
        x_flat.mean(0, keepdim=True).expand_as(x_flat), x_flat,
    ).item()
    print(f"  Training unsupervised SAE: d_sae={d_sae}, "
          f"{cfg.epochs} epochs, {len(loader)} steps/epoch")
    print(f"  baseline_mse={baseline_mse:.6f}")

    step = 0
    for epoch in range(1, cfg.epochs + 1):
        epoch_recon = epoch_sparse = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for (x_b,) in pbar:
            x_b = x_b.to(cfg.device)
            recon, acts = sae(x_b)
            loss_recon = F.mse_loss(recon, x_b)
            loss_sparse = acts.abs().mean()
            loss = loss_recon + cfg.lambda_sparse * loss_sparse

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            sae.normalize_decoder()

            epoch_recon += loss_recon.item()
            epoch_sparse += loss_sparse.item()
            step += 1

        n = len(loader)
        r2 = 1.0 - (epoch_recon / n) / baseline_mse
        print(f"  Epoch {epoch:2d}  recon={epoch_recon/n:.5f}  "
              f"sparse={epoch_sparse/n:.5f}  R²={r2:.3f}")

    return sae.cpu()


def compute_firing_rates(sae: UnsupervisedSAE, activations: torch.Tensor,
                          cfg: Config) -> torch.Tensor:
    """Fraction of positions where each latent fires (acts > 0)."""
    sae.eval().to(cfg.device)
    N, T, d_model = activations.shape
    x_flat = activations.reshape(-1, d_model)
    d_sae = sae.d_sae

    fire_counts = torch.zeros(d_sae)
    total = 0
    with torch.no_grad():
        for i in range(0, x_flat.shape[0], cfg.batch_size):
            x_b = x_flat[i : i + cfg.batch_size].to(cfg.device)
            acts = sae.encode(x_b).cpu()
            fire_counts += (acts > 0).float().sum(dim=0)
            total += x_b.shape[0]
    sae.cpu()
    return fire_counts / max(total, 1)


# ── Orchestration ──────────────────────────────────────────────────────────

def run(cfg: Config = None):
    """Discovery pipeline entry point."""
    if cfg is None:
        cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    discovered_catalog_path = cfg.output_dir / "discovered_catalog.json"
    unsup_sae_path = cfg.output_dir / "unsupervised_sae.pt"
    top_acts_path = cfg.output_dir / "discovered_top_activations.json"
    desc_path = cfg.output_dir / "discovered_descriptions.json"

    if discovered_catalog_path.exists():
        print(f"Discovered catalog already exists: {discovered_catalog_path}")
        return json.loads(discovered_catalog_path.read_text())

    if not cfg.activations_path.exists():
        raise FileNotFoundError(
            f"Activations not found at {cfg.activations_path}. "
            f"Run `python -m pipeline.run --step annotate` first to cache "
            f"residual-stream activations."
        )

    activations = torch.load(cfg.activations_path, weights_only=True)
    N, T, d_model = activations.shape
    # Default d_sae = 8× d_model (matches common sparse-autoencoder convention)
    d_sae = 8 * d_model

    # 1. Train unsupervised SAE (or load cached)
    if unsup_sae_path.exists():
        print(f"Loading cached unsupervised SAE: {unsup_sae_path}")
        sae = UnsupervisedSAE(d_model, d_sae)
        sae.load_state_dict(
            torch.load(unsup_sae_path, map_location="cpu", weights_only=True)
        )
    else:
        t0 = time.time()
        print(f"\n── Training unsupervised SAE ({d_sae} latents) ──")
        sae = train_unsupervised_sae(activations, cfg, d_sae)
        torch.save(sae.state_dict(), unsup_sae_path)
        print(f"  Saved: {unsup_sae_path}  ({time.time() - t0:.1f}s)")

    # 2. Compute firing rates and select latents
    print("\n── Selecting latents by firing rate ──")
    firing_rates = compute_firing_rates(sae, activations, cfg)
    mask = (firing_rates >= cfg.min_firing_rate) & (firing_rates <= cfg.max_firing_rate)
    candidates = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    candidates.sort(key=lambda i: -firing_rates[i].item())
    selected = candidates[: cfg.n_latents_to_explain]
    print(f"  {len(candidates)} latents in window "
          f"[{cfg.min_firing_rate}, {cfg.max_firing_rate}], "
          f"selected top {len(selected)} by firing rate")

    if not selected:
        raise RuntimeError(
            "No latents passed the firing-rate window. Widen "
            "min_firing_rate / max_firing_rate in config."
        )

    # 3. Collect top-k activating contexts by streaming corpus through the
    #    model, encoding through our unsupervised SAE, and heaping.
    if top_acts_path.exists():
        print(f"Loading cached top activations: {top_acts_path}")
        top_acts = json.loads(top_acts_path.read_text())
    else:
        print("\n── Collecting top activations ──")
        from transformer_lens import HookedTransformer
        from .inventory import collect_top_activations, PretrainedSAE

        model = HookedTransformer.from_pretrained(
            cfg.model_name, device=cfg.device, dtype=cfg.model_dtype
        )
        model.eval()
        tokenizer = model.tokenizer

        # Wrap the trained SAE to match PretrainedSAE interface
        wrapped = PretrainedSAE(
            W_enc=sae.encoder.weight.T.contiguous(),  # (d_model, d_sae)
            W_dec=sae.decoder.weight.T.contiguous(),  # (d_sae, d_model)
            b_enc=sae.encoder.bias.data.clone(),
            b_dec=torch.zeros(d_model),               # UnsupervisedSAE has no decoder bias
            threshold=None,                            # vanilla ReLU
        )

        top_acts = collect_top_activations(
            model, wrapped, tokenizer, selected, cfg,
        )
        top_acts_path.write_text(json.dumps(top_acts, indent=2))
        print(f"  Saved top activations: {top_acts_path}")
        del model

    # 4. Describe each latent via Sonnet (reuses inventory's explainer)
    if desc_path.exists():
        print(f"Loading cached descriptions: {desc_path}")
        descriptions = json.loads(desc_path.read_text())
    else:
        print("\n── Generating descriptions (via Sonnet) ──")
        from transformers import AutoTokenizer
        from .inventory import explain_features

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        descriptions = explain_features(top_acts, tokenizer, cfg)
        desc_path.write_text(json.dumps(descriptions, indent=2))
        print(f"  Saved: {desc_path}")

    # 5. Organize into hierarchy
    print("\n── Organizing into hierarchy ──")
    from .inventory import organize_hierarchy
    catalog = organize_hierarchy(descriptions, cfg)
    discovered_catalog_path.write_text(json.dumps(catalog, indent=2))

    n_groups = sum(1 for f in catalog["features"] if f["type"] == "group")
    n_leaves = sum(1 for f in catalog["features"] if f["type"] == "leaf")
    print(f"\n  Discovered catalog: {n_groups} groups, {n_leaves} leaves")
    print(f"  Saved: {discovered_catalog_path}")
    print(
        f"\n  Next step: train a supervised SAE on this catalog with:\n"
        f"    python -m pipeline.run --catalog {discovered_catalog_path}"
    )
    return catalog


if __name__ == "__main__":
    run()
