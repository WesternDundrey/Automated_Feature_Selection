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


# ── v2: MSE feature dictionary supervision ─────────────────────────────────

def compute_target_directions(
    x_flat: torch.Tensor, y_flat: torch.Tensor, n_supervised: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Makelov-style conditional mean directions for each feature.

    For feature i: target_dir_i = normalize(mean(x | label_i=1) - mean(x))

    Args:
        x_flat: (N, d_model) residual stream activations
        y_flat: (N, n_supervised) binary labels
        n_supervised: number of supervised features

    Returns:
        target_dirs: (n_supervised, d_model) unit-norm target directions
        raw_norms:   (n_supervised,) norms before normalization (signal strength)
        counts:      (n_supervised,) number of positive examples per feature
    """
    y = y_flat[:, :n_supervised].float()
    mean_all = x_flat.mean(dim=0)  # (d_model,)

    # Vectorized weighted mean: (y.T @ x) / counts.
    # IMPORTANT: clamp counts only for the division, NOT for validity. A
    # feature with raw_counts == 0 gets mean_pos == 0 via clamped division
    # and directions == -mean_all, whose norm is the norm of the global
    # mean. That passes a norm-only validity check but is a meaningless
    # direction — the supervised slot then gets a frozen decoder column
    # pointing anti-parallel to the activation centroid, which hurts both
    # reconstruction and interventions. Use raw_counts for the validity
    # gate so zero-positive features stay zeroed.
    raw_counts = y.sum(dim=0)                                 # (n_sup,)
    counts = raw_counts.clamp(min=1)
    mean_pos = (y.T @ x_flat) / counts.unsqueeze(1)           # (n_sup, d_model)

    directions = mean_pos - mean_all.unsqueeze(0)             # (n_sup, d_model)
    raw_norms = directions.norm(dim=1)                        # (n_sup,)

    valid = (raw_counts > 0) & (raw_norms > 1e-6)
    target_dirs = torch.zeros_like(directions)
    target_dirs[valid] = directions[valid] / raw_norms[valid].unsqueeze(1)

    # Return the honest raw counts (not the clamped version) so downstream
    # reporting reflects true positive-sample sizes.
    return target_dirs, raw_norms, raw_counts


def compute_target_directions_logistic(
    x_flat: torch.Tensor, y_flat: torch.Tensor, n_supervised: int,
    l2_lambda: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-feature ridge logistic regression weight as the target direction.

    Per the user's framing: "LR will happily use a confound if it helps
    classification." This is the cost. Versus mean-shift: LR finds the
    optimal classification hyperplane normal under logistic loss, which
    is what cleanly separates classes — but it's NOT a "feature
    direction" in the interpretation sense. Use only as an ablation
    against mean-shift.

    Implementation: solves per-feature with simple gradient descent
    (sklearn would be cleaner but we avoid the dep). Returns the
    unit-normalized weight vector.
    """
    import torch.nn.functional as F
    n_features = n_supervised
    d_model = x_flat.shape[1]
    target_dirs = torch.zeros(n_features, d_model, dtype=x_flat.dtype)
    raw_norms = torch.zeros(n_features, dtype=x_flat.dtype)
    counts = torch.zeros(n_features, dtype=x_flat.dtype)

    x = x_flat.float()
    for k in range(n_features):
        y = y_flat[:, k].float()
        n_pos = float(y.sum().item())
        counts[k] = n_pos
        if n_pos < 5 or n_pos > x.shape[0] - 5:
            # too rare or too universal for stable LR
            continue
        # Train a single-feature ridge logistic regression via Newton's
        # method (equivalent to sklearn LogisticRegression(C=1/l2_lambda)).
        # Closed-form-ish: fixed-point iteration on the normal equations.
        # 50 iterations is enough at this dim.
        w = torch.zeros(d_model, dtype=x.dtype, device=x.device,
                        requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=1.0, max_iter=50, line_search_fn="strong_wolfe")
        target_y = y.to(x.device)

        def closure():
            opt.zero_grad()
            logit = x @ w
            # BCE with logits + L2
            loss = F.binary_cross_entropy_with_logits(logit, target_y) \
                 + 0.5 * l2_lambda * w.pow(2).sum() / x.shape[0]
            loss.backward()
            return loss
        try:
            opt.step(closure)
        except Exception:
            # LBFGS occasionally fails to converge for very rare classes.
            continue
        w = w.detach().cpu()
        norm = w.norm()
        if norm < 1e-8:
            continue
        raw_norms[k] = norm
        target_dirs[k] = w / norm

    return target_dirs, raw_norms, counts


def compute_target_directions_lda(
    x_flat: torch.Tensor, y_flat: torch.Tensor, n_supervised: int,
    shrinkage: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear discriminant analysis target direction (whitened mean-shift).

    d_k = normalize((Σ + λI)^-1 (μ₁ - μ₀))

    where Σ is the within-class covariance pooled over both classes
    and λI is shrinkage regularization (Ledoit-Wolf style — at our
    sample size, full Σ⁻¹ is unstable).

    Per the user's framing: "LDA / whitened mean-shift is more
    principled for suppressing high-variance junk." Compared to
    mean-shift, LDA explicitly down-weights directions that have
    high variance regardless of class membership.
    """
    n_features = n_supervised
    d_model = x_flat.shape[1]
    target_dirs = torch.zeros(n_features, d_model, dtype=x_flat.dtype)
    raw_norms = torch.zeros(n_features, dtype=x_flat.dtype)
    counts = torch.zeros(n_features, dtype=x_flat.dtype)

    x = x_flat.float()
    mean_all = x.mean(dim=0)
    # Total covariance computed once (we approximate within-class as
    # this — at our sample size pooling is roughly equivalent and
    # avoids a per-class loop).
    centered = x - mean_all
    # Σ = (1/N) X^T X  (centered)
    sigma = (centered.T @ centered) / x.shape[0]
    # Shrinkage regularizer: λ scaled by trace(Σ)/d so it's dimension-
    # invariant (Ledoit-Wolf style with fixed coefficient).
    diag_mean = sigma.diagonal().mean()
    sigma_reg = sigma + shrinkage * diag_mean * torch.eye(
        d_model, dtype=sigma.dtype, device=sigma.device,
    )
    # Cholesky-solve once: target_dir = Σ⁻¹ (μ₁ - μ₀) per feature
    # → solve(Σ, M) where M is the matrix of mean-shifts (d_model, n_features).
    print(f"  [LDA target_dirs] inverting (d_model={d_model}, "
          f"shrinkage={shrinkage}, trace/d={float(diag_mean):.4f})")
    try:
        chol = torch.linalg.cholesky(sigma_reg)
    except Exception as e:
        print(f"    Cholesky failed ({e}); falling back to mean-shift.")
        return compute_target_directions(x_flat, y_flat, n_supervised)

    for k in range(n_features):
        y = y_flat[:, k].float()
        n_pos = float(y.sum().item())
        counts[k] = n_pos
        if n_pos < 5:
            continue
        mean_pos = (y.unsqueeze(1) * x).sum(dim=0) / max(n_pos, 1)
        diff = mean_pos - mean_all   # (d_model,)
        try:
            d_k = torch.cholesky_solve(diff.unsqueeze(1), chol).squeeze(1)
        except Exception:
            continue
        norm = d_k.norm()
        if norm < 1e-8:
            continue
        raw_norms[k] = norm
        target_dirs[k] = d_k / norm

    return target_dirs, raw_norms, counts


def compute_target_directions_dispatch(
    x_flat: torch.Tensor, y_flat: torch.Tensor, n_supervised: int, cfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch to the right target_dir computation based on
    cfg.target_dir_method ∈ {'mean_shift', 'logistic', 'lda'}."""
    method = getattr(cfg, "target_dir_method", "mean_shift")
    print(f"  Computing target_dirs via method={method!r}")
    if method == "mean_shift":
        return compute_target_directions(x_flat, y_flat, n_supervised)
    if method == "logistic":
        return compute_target_directions_logistic(
            x_flat, y_flat, n_supervised,
            l2_lambda=float(getattr(cfg, "target_dir_logistic_lambda", 1.0)),
        )
    if method == "lda":
        return compute_target_directions_lda(
            x_flat, y_flat, n_supervised,
            shrinkage=float(getattr(cfg, "target_dir_lda_shrinkage", 0.1)),
        )
    raise ValueError(f"unknown target_dir_method {method!r}")


def mse_supervision_loss(
    decoder_weight: torch.Tensor,
    sup_acts: torch.Tensor,
    target_dirs: torch.Tensor,
    activations: torch.Tensor,
    labels: torch.Tensor,
    cfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized MSE feature dictionary loss (Makelov et al. 2024).

    Two components:
      A) Direction loss: cosine alignment of decoder columns to target directions
      B) Magnitude loss: at positive positions, activation ≈ projection onto target;
         at negative positions, activation → 0 (selectivity)

    Args:
        decoder_weight: (d_model, n_total) — sae.decoder.weight
        sup_acts: (batch, n_sup) — supervised latent activations (post-ReLU)
        target_dirs: (n_sup, d_model) — precomputed target directions
        activations: (batch, d_model) — input activations
        labels: (batch, n_sup) — binary labels
        cfg: Config with direction_loss_weight and magnitude_loss_weight

    Returns:
        total_loss, direction_loss, magnitude_loss
    """
    n_sup = target_dirs.shape[0]

    # A) Direction: push decoder columns toward target directions
    # decoder_weight[:, :n_sup] is (d_model, n_sup), target_dirs.T is (d_model, n_sup)
    dec_cols = decoder_weight[:, :n_sup]  # (d_model, n_sup)
    cosines = (dec_cols * target_dirs.T).sum(dim=0)  # (n_sup,)
    direction_loss = (1 - cosines).mean()

    # B) Magnitude: at positive positions, sup_act ≈ activation projected onto target;
    #              at negative positions, sup_act ≈ 0 (selectivity)
    target_mag = activations @ target_dirs.T  # (batch, n_sup)

    # Positive: activation should match projection onto target direction
    pos_err = (sup_acts - target_mag) ** 2
    n_positive = labels.sum().clamp(min=1.0)
    pos_loss = (pos_err * labels).sum() / n_positive

    # Negative: activation should be zero (class-balanced so rare features aren't drowned)
    neg_labels = 1.0 - labels
    n_negative = neg_labels.sum().clamp(min=1.0)
    neg_loss = (sup_acts ** 2 * neg_labels).sum() / n_negative

    magnitude_loss = pos_loss + neg_loss

    total = cfg.direction_loss_weight * direction_loss + cfg.magnitude_loss_weight * magnitude_loss
    return total, direction_loss, magnitude_loss


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
    def normalize_decoder(self, skip_first_n: int = 0):
        """Normalize decoder columns to unit norm.

        If skip_first_n > 0, leaves the first `skip_first_n` columns
        untouched (e.g., when supervised decoder columns are frozen to
        target_dirs and should not be modified).
        """
        if skip_first_n > 0:
            unsup_cols = self.decoder.weight.data[:, skip_first_n:]
            self.decoder.weight.data[:, skip_first_n:] = F.normalize(unsup_cols, dim=0)
        else:
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def unsup_encoder_weight(self) -> torch.Tensor:
        """(n_unsupervised, d_model). Rows are reading directions of the
        unsupervised latents. Uniform API with HingeSAE / GatedBCESAE so
        promote_loop can read U-encoder rows without branching on SAE class."""
        return self.encoder.weight[self.n_supervised:]

    def unsup_encoder_bias(self) -> torch.Tensor:
        return self.encoder.bias[self.n_supervised:]


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

def load_trained_sae(model_cfg: dict) -> "torch.nn.Module":
    """Instantiate the right SAE class for a saved checkpoint.

    Older checkpoints written by the legacy supervision_mode ∈
    {"hybrid", "mse", "bce"} paths use the `SupervisedSAE` class in this
    module. New hinge-family checkpoints (v8.11+) use
    `HingeSAE` / `JumpReLUHingeSAE` / `GatedBCESAE` from
    `supervised_hinge.py`. Dispatches based on the `supervision_mode`
    field in the saved model_cfg dict, falling back to the legacy class
    for checkpoints that predate the field.
    """
    supervision_mode = model_cfg.get("supervision_mode", "hybrid")
    from .supervised_hinge import is_hinge_mode, build_hinge_sae
    if is_hinge_mode(supervision_mode):
        return build_hinge_sae(
            supervision_mode=supervision_mode,
            d_model=model_cfg["d_model"],
            n_supervised=model_cfg["n_supervised"],
            n_unsupervised=model_cfg["n_unsupervised"],
            gated_tie_weights=model_cfg.get("gated_tie_weights", False),
            theta_init=model_cfg.get("jumprelu_theta_init", 0.1),
        )
    return SupervisedSAE(
        model_cfg["d_model"],
        model_cfg["n_supervised"],
        model_cfg["n_unsupervised"],
        model_cfg.get("n_lista_steps", 0),
    )


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
    # Dispatch to the hinge-family trainer for the new supervision modes.
    # These modes train end-to-end (no frozen decoder, no target_dir-as-loss
    # constraint) per supervised_saes_hinge_loss.md.
    from .supervised_hinge import is_hinge_mode, train_hinge_sae
    if is_hinge_mode(cfg.supervision_mode):
        return train_hinge_sae(
            activations, labels, features, cfg, save_checkpoint=save_checkpoint,
        )

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

    # Compute target directions for direction alignment (hybrid and mse modes)
    # Also needed when freeze_supervised_decoder=True (to set the columns).
    target_dirs = None
    need_dirs = cfg.supervision_mode in ("hybrid", "mse") or cfg.freeze_supervised_decoder
    if need_dirs:
        target_dirs, raw_norms, dir_counts = compute_target_directions_dispatch(
            x_train, y_train, n_supervised, cfg,
        )
        # Report target direction quality
        valid = raw_norms > 1e-6
        print(f"\n  Target directions: {valid.sum()}/{n_supervised} features have signal")
        if valid.any():
            print(f"  Direction norms — min: {raw_norms[valid].min():.4f}, "
                  f"median: {raw_norms[valid].median():.4f}, "
                  f"max: {raw_norms[valid].max():.4f}")
        # Warn about correlated directions
        if valid.sum() > 1:
            valid_dirs = target_dirs[valid]
            pairwise = valid_dirs @ valid_dirs.T
            pairwise.fill_diagonal_(0)
            max_sim = pairwise.max().item()
            if max_sim > 0.8:
                print(f"  WARNING: max pairwise cosine similarity = {max_sim:.3f} "
                      f"(>0.8 means correlated features)")
        target_dirs = target_dirs.to(cfg.device)
        # Save for evaluation
        if save_checkpoint:
            torch.save(target_dirs.cpu(), cfg.target_dirs_path)

    # Class-balanced pos_weight for BCE (used when use_mse_supervision=False)
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

    # Frozen decoder: set supervised columns to target_dirs, zero their gradients
    if cfg.freeze_supervised_decoder and target_dirs is not None:
        with torch.no_grad():
            sae.decoder.weight.data[:, :n_supervised] = target_dirs.T
        def _zero_sup_grad(grad):
            g = grad.clone()
            g[:, :n_supervised] = 0
            return g
        sae.decoder.weight.register_hook(_zero_sup_grad)
        print(f"  Decoder: supervised columns FROZEN to target_dirs")

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
    print(f"  supervision_mode={cfg.supervision_mode}")
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

            # ── Selectivity loss (encoder: which positions fire?) ─────
            if cfg.selectivity_loss == "hinge":
                targets_hinge = 2 * y_b - 1  # {0,1} → {-1,+1}
                raw_hinge = F.relu(cfg.hinge_margin - sup_pre * targets_hinge)
                # Class-balanced: weight positive positions by pos_weight so rare
                # features aren't drowned by the 99.9% negative class
                weights = torch.where(
                    y_b > 0.5, pos_weight.unsqueeze(0), y_b.new_ones(())
                )
                loss_select = (raw_hinge * weights).mean()
            elif cfg.selectivity_loss == "none":
                loss_select = sup_pre.new_zeros(())
            else:  # "bce" (default)
                loss_select = F.binary_cross_entropy_with_logits(
                    sup_pre, y_b, pos_weight=pos_weight
                )

            # ── Direction / magnitude loss (decoder: where does it point?) ──
            if cfg.freeze_supervised_decoder and cfg.supervision_mode == "mse":
                # Frozen decoder + MSE mode: use magnitude loss only, skip direction
                _, _, loss_mag = mse_supervision_loss(
                    sae.decoder.weight, sup_acts, target_dirs, x_b, y_b, cfg,
                )
                loss_sup = loss_select + cfg.magnitude_loss_weight * loss_mag
            elif cfg.freeze_supervised_decoder:
                # Frozen decoder, non-MSE: no direction loss needed
                loss_sup = loss_select
            elif cfg.supervision_mode == "hybrid":
                dec_cols = sae.decoder.weight[:, :n_supervised]
                cosines = (dec_cols * target_dirs.T).sum(dim=0)
                loss_dir = (1 - cosines).mean()
                loss_sup = loss_select + cfg.direction_loss_weight * loss_dir
            elif cfg.supervision_mode == "mse":
                loss_sup, loss_dir, loss_mag = mse_supervision_loss(
                    sae.decoder.weight, sup_acts, target_dirs, x_b, y_b, cfg,
                )
            else:  # "bce"
                loss_sup = loss_select
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
            sae.normalize_decoder(
                skip_first_n=n_supervised if cfg.freeze_supervised_decoder else 0,
            )

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
                recon_v, sup_pre_v, sup_acts_v, _ = sae(xv)
                val_recon += F.mse_loss(recon_v, xv).item()
                if cfg.supervision_mode == "hybrid":
                    vbce = F.binary_cross_entropy_with_logits(
                        sup_pre_v, yv, pos_weight=pos_weight
                    ).item()
                    vdc = sae.decoder.weight[:, :n_supervised]
                    vcos = (vdc * target_dirs.T).sum(dim=0)
                    vdir = (1 - vcos).mean().item()
                    val_sup += vbce + cfg.direction_loss_weight * vdir
                elif cfg.supervision_mode == "mse":
                    vs, _, _ = mse_supervision_loss(
                        sae.decoder.weight, sup_acts_v, target_dirs, xv, yv, cfg,
                    )
                    val_sup += vs.item()
                else:
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
                "supervision_mode": cfg.supervision_mode,
                "use_mse_supervision": cfg.supervision_mode in ("mse", "hybrid"),
            },
            cfg.checkpoint_config_path,
        )
        # Identity sidecar so a later --step evaluate / --step intervention
        # can detect a layer/model/catalog mismatch instead of silently
        # loading an SAE trained against a different activation distribution.
        from .cache_meta import write_cache_meta
        write_cache_meta(
            cfg.checkpoint_path, "supervised_sae", cfg,
            n_features=n_supervised,
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

    # Cache identity check before loading. Train is the biggest downstream
    # consumer — a mismatched activations.pt here means the whole trained
    # model is wrong.
    from .cache_meta import load_or_die as _cache_load_or_die
    _cache_load_or_die(cfg.activations_path, "activations", cfg)

    activations = torch.load(cfg.activations_path, weights_only=True)
    annotations = torch.load(cfg.annotations_path, weights_only=True)

    # Mask leading positions (e.g., BOS/position-0) before analysis.
    # Position 0 has degenerate attention and dominant residual-stream
    # directions that collapse target_dirs for sequence-level features.
    from .position_mask import mask_leading
    activations, annotations = mask_leading(activations, annotations, cfg=cfg)

    print(f"Activations: {activations.shape} (masked first "
          f"{cfg.mask_first_n_positions} positions)")
    print(f"Annotations: {annotations.shape}")
    print(f"Features: {len(features)}")

    sae = train_supervised_sae(activations, annotations, features, cfg)
    return sae


if __name__ == "__main__":
    run()
