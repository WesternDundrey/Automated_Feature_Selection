"""
Step 2b — Train Supervised Sparse Autoencoder

Train a supervised SAE using the LLM-annotated labels as supervision.

Loss = MSE(recon, x)
     + lambda_sup   * class_balanced_BCE(sup_pre, labels)
     + lambda_sparse * L1(all_acts)
     + lambda_hier   * hierarchy_loss(sup_acts)

The supervised loss is ramped in linearly over warmup_steps.
Decoder columns are normalized to unit norm after each optimizer step.

Outputs:
    pipeline_data/supervised_sae.pt          Model state dict
    pipeline_data/supervised_sae_config.pt   Model config (for loading)

Usage:
    python -m pipeline.train
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .config import Config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Model ───────────────────────────────────────────────────────────────────

class SupervisedSAE(nn.Module):
    """Sparse autoencoder with a split latent space.

    First n_supervised latents are trained with a supervised loss (BCE against
    LLM labels). Remaining n_unsupervised latents absorb whatever the
    supervised features don't cover.

    Optional LISTA refinement: after the initial encode→decode, compute the
    reconstruction residual, re-encode it, and update pre-activations with a
    learnable step size eta. This iterative refinement (from Learned ISTA,
    Gregor & LeCun 2010) improves sparse recovery.
    """

    def __init__(self, d_model: int, n_supervised: int, n_unsupervised: int,
                 n_lista_steps: int = 0):
        super().__init__()
        self.d_model = d_model
        self.n_supervised = n_supervised
        self.n_total = n_supervised + n_unsupervised
        self.n_lista_steps = n_lista_steps

        self.encoder = nn.Linear(d_model, self.n_total, bias=True)
        self.decoder = nn.Linear(self.n_total, d_model, bias=False)

        # LISTA refinement step sizes (one per iteration)
        if n_lista_steps > 0:
            self.lista_eta = nn.ParameterList([
                nn.Parameter(torch.ones(1) * 0.1) for _ in range(n_lista_steps)
            ])

        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x):
        pre = self.encoder(x)
        acts = F.relu(pre)
        recon = self.decoder(acts)

        # LISTA refinement iterations
        for i in range(self.n_lista_steps):
            residual = x - recon
            delta = self.encoder(residual)
            pre = pre + self.lista_eta[i] * delta
            acts = F.relu(pre)
            recon = self.decoder(acts)

        sup_pre = pre[..., : self.n_supervised]
        sup_acts = acts[..., : self.n_supervised]
        return recon, sup_pre, sup_acts, acts

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


# ── Hierarchy loss ──────────────────────────────────────────────────────────

def build_hierarchy_map(features: list[dict]) -> dict[int, list[int]]:
    """Build {parent_idx: [child_idx, ...]} from the feature catalog."""
    feat_id_to_idx = {f["id"]: i for i, f in enumerate(features)}
    hierarchy = {}
    for feat in features:
        if feat.get("parent") and feat["parent"] in feat_id_to_idx:
            parent_idx = feat_id_to_idx[feat["parent"]]
            child_idx = feat_id_to_idx[feat["id"]]
            hierarchy.setdefault(parent_idx, []).append(child_idx)
    return hierarchy


def hierarchy_loss(sup_acts: torch.Tensor, hierarchy: dict[int, list[int]]) -> torch.Tensor:
    """Penalize child activating more than parent: loss = mean(relu(max_child - parent))."""
    loss = sup_acts.new_zeros(())
    for parent_idx, child_idxs in hierarchy.items():
        parent = sup_acts[..., parent_idx]
        children = sup_acts[..., child_idxs]
        max_child = children.max(dim=-1).values
        loss = loss + F.relu(max_child - parent).mean()
    if hierarchy:
        loss = loss / len(hierarchy)
    return loss


# ── Training loop ───────────────────────────────────────────────────────────

def train_supervised_sae(
    activations: torch.Tensor,
    labels: torch.Tensor,
    features: list[dict],
    cfg: Config,
    save_checkpoint: bool = True,
) -> SupervisedSAE:
    """Train a supervised SAE.

    Args:
        activations: (N, seq_len, d_model) residual stream vectors
        labels: (N, seq_len, n_features) binary labels from LLM annotation
        features: list of feature dicts from the catalog
        cfg: pipeline configuration

    Returns:
        Trained SupervisedSAE on CPU.
    """
    set_seed(cfg.seed)

    N, T, d_model = activations.shape
    n_supervised = labels.shape[-1]

    # Flatten to individual residual-stream vectors
    x_flat = activations.reshape(-1, d_model)
    y_flat = labels.reshape(-1, n_supervised)

    # Train/test split (shuffled to avoid distribution shift from document order)
    n_total = x_flat.shape[0]
    perm = torch.randperm(n_total)
    split_idx = int(cfg.train_fraction * n_total)
    train_idx, test_idx = perm[:split_idx], perm[split_idx:]

    # Save split indices for reproducible evaluation (avoids RNG coupling)
    if save_checkpoint:
        torch.save(perm, cfg.split_path)

    x_train, x_test = x_flat[train_idx], x_flat[test_idx]
    y_train, y_test = y_flat[train_idx], y_flat[test_idx]

    print(f"Training data: {x_train.shape[0]:,} vectors, "
          f"Test data: {x_test.shape[0]:,} vectors")

    # Baseline MSE for R^2
    baseline_mse = F.mse_loss(
        x_train.mean(0, keepdim=True).expand_as(x_train), x_train
    ).item()
    test_baseline_mse = F.mse_loss(
        x_test.mean(0, keepdim=True).expand_as(x_test), x_test
    ).item()

    # Class-balanced pos_weight for BCE
    pos_counts = y_train.sum(dim=0).clamp(min=1.0)
    neg_counts = y_train.shape[0] - pos_counts
    pos_weight = (neg_counts / pos_counts).clamp(max=100.0).to(cfg.device)

    # Hierarchy
    hier_map = build_hierarchy_map(features)

    # Model
    sae = SupervisedSAE(
        d_model, n_supervised, cfg.n_unsupervised,
        n_lista_steps=cfg.n_lista_steps,
    ).to(cfg.device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=cfg.lr, weight_decay=0.0)

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # LR schedule: constant for first 2/3, cosine decay over final 1/3
    total_steps = cfg.epochs * len(loader)
    decay_start = int(total_steps * 2 / 3)
    decay_length = total_steps - decay_start

    def lr_lambda(step):
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(decay_length, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\nTraining: {cfg.epochs} epochs, {len(loader)} steps/epoch")
    print(f"  n_supervised={n_supervised}  n_unsupervised={cfg.n_unsupervised}"
          f"  lista_steps={cfg.n_lista_steps}")
    print(f"  lambda_sup={cfg.lambda_sup}  lambda_sparse={cfg.lambda_sparse}  "
          f"lambda_hier={cfg.lambda_hier}")
    print(f"  baseline_mse={baseline_mse:.6f}")

    step = 0
    for epoch in range(1, cfg.epochs + 1):
        epoch_recon = epoch_sup = epoch_sparse = epoch_hier = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for x_b, y_b in pbar:
            x_b, y_b = x_b.to(cfg.device), y_b.to(cfg.device)

            recon, sup_pre, sup_acts, all_acts = sae(x_b)

            loss_recon = F.mse_loss(recon, x_b)
            loss_sup = F.binary_cross_entropy_with_logits(
                sup_pre, y_b, pos_weight=pos_weight
            )
            loss_sparse = all_acts.abs().mean()
            loss_hier = hierarchy_loss(sup_acts, hier_map)

            sup_scale = min(1.0, step / max(cfg.warmup_steps, 1))
            loss = (
                loss_recon
                + cfg.lambda_sup * sup_scale * loss_sup
                + cfg.lambda_sparse * loss_sparse
                + cfg.lambda_hier * sup_scale * loss_hier
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            sae.normalize_decoder()

            epoch_recon += loss_recon.item()
            epoch_sup += loss_sup.item()
            epoch_sparse += loss_sparse.item()
            epoch_hier += loss_hier.item()
            step += 1

            pbar.set_postfix({
                "recon": f"{loss_recon.item():.5f}",
                "sup": f"{loss_sup.item():.5f}",
            })

        n = len(loader)
        r2 = 1.0 - (epoch_recon / n) / baseline_mse
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:2d}  recon={epoch_recon/n:.5f}  sup={epoch_sup/n:.5f}  "
              f"sparse={epoch_sparse/n:.5f}  hier={epoch_hier/n:.5f}  R2={r2:.3f}  "
              f"lr={current_lr:.2e}")

        # Validation on held-out data
        sae.eval()
        val_recon = 0.0
        val_sup = 0.0
        val_batches = 0
        with torch.no_grad():
            for i in range(0, x_test.shape[0], cfg.batch_size):
                xv = x_test[i : i + cfg.batch_size].to(cfg.device)
                yv = y_test[i : i + cfg.batch_size].to(cfg.device)
                recon_v, sup_pre_v, _, _ = sae(xv)
                val_recon += F.mse_loss(recon_v, xv).item()
                val_sup += F.binary_cross_entropy_with_logits(
                    sup_pre_v, yv, pos_weight=pos_weight
                ).item()
                val_batches += 1
        sae.train()
        val_r2 = 1.0 - (val_recon / val_batches) / test_baseline_mse
        print(f"           val_recon={val_recon/val_batches:.5f}  "
              f"val_sup={val_sup/val_batches:.5f}  val_R2={val_r2:.3f}")

        if epoch == 1 and r2 < 0.5:
            print("  WARNING: R^2 < 0.5 after epoch 1. "
                  "Consider reducing lambda_sup or increasing warmup_steps.")

        # Mid-training checkpoint every 5 epochs
        if save_checkpoint and epoch % 5 == 0 and epoch < cfg.epochs:
            ckpt_path = cfg.output_dir / f"supervised_sae_epoch{epoch}.pt"
            torch.save({
                "model": sae.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "step": step,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # Save
    sae_cpu = sae.cpu()
    if save_checkpoint:
        torch.save(sae_cpu.state_dict(), cfg.checkpoint_path)
        torch.save(
            {
                "d_model": d_model,
                "n_supervised": n_supervised,
                "n_unsupervised": cfg.n_unsupervised,
                "n_lista_steps": cfg.n_lista_steps,
            },
            cfg.checkpoint_config_path,
        )
        print(f"\nModel saved: {cfg.checkpoint_path}")

    return sae_cpu


# ── Main entry point ────────────────────────────────────────────────────────

def run(cfg: Config = None):
    """Train the supervised SAE from cached data."""
    if cfg is None:
        cfg = Config()

    # Load data
    for path, name in [
        (cfg.activations_path, "activations"),
        (cfg.annotations_path, "annotations"),
        (cfg.catalog_path, "feature catalog"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]

    activations = torch.load(cfg.activations_path, weights_only=True)
    annotations = torch.load(cfg.annotations_path, weights_only=True)

    print(f"Activations: {activations.shape}")
    print(f"Annotations: {annotations.shape}")
    print(f"Features: {len(features)}")

    sae = train_supervised_sae(activations, annotations, features, cfg)
    return sae


if __name__ == "__main__":
    run()
