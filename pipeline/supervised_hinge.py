"""
Hinge / Gated-BCE supervised SAE formulations from
`supervised_saes_hinge_loss.md`.

Three new SAE classes are provided, each matching one formulation in the
methodology note:

    HingeSAE          — ReLU encoder + hinge on pre-activations.
    JumpReLUHingeSAE  — JumpReLU with per-feature θ + hinge on margin (z-θ).
    GatedBCESAE       — Two encoder paths (gate, magnitude) + BCE on gate.

All three share the methodology's design invariants:

  - Reconstruction MSE shapes magnitude alone; no sparsity-induced shrinkage
    on supervised features (L1 applies only to unsupervised latents).
  - Hinge / BCE on the supervised pre-activation produces zero gradient
    when a feature is confidently correctly gated, so MSE alone drives
    magnitude in the common case.
  - Honest gradients throughout — no straight-through estimators. For
    JumpReLU the non-differentiable Heaviside gate is fed θ's gradient
    via the hinge term directly; MSE's contribution through the Heaviside
    is the Dirac term autograd silently drops, which is the desired
    behavior here.
  - No frozen decoder by construction. These modes train encoder + decoder
    jointly. The cos = 1.0 guarantee of the frozen-decoder mode (summary6,
    summary7) is traded away for cleaner single-pass training — the
    learned decoder column may cosine-align with the analytical mean-shift
    `target_dir` to varying degrees, which we report post-hoc as a
    diagnostic rather than enforce as a constraint.

Class-balanced weighting on the supervision term is added here even though
the doc is silent on it: our per-feature imbalance spans ~2 orders of
magnitude (n_pos from ~15 to ~10,000 on a 128k-position corpus) and
uniform hinge would under-gradient rare features the same way uniform BCE
would. The weighting mirrors the `pos_weight = (n_neg / n_pos).clamp(max=100)`
policy used in the legacy BCE supervised path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
# Formulation 1: HingeSAE (default, primary)
# ═════════════════════════════════════════════════════════════════════════════


class HingeSAE(nn.Module):
    """ReLU SAE supervised by hinge on pre-activations.

    Forward:
        z = encoder(x)                             # (batch, n_total)
        acts = ReLU(z)
        recon = decoder(acts)

    Training loss (constructed outside):
        MSE(recon, x)
        + λ_sup · hinge_sup(z[:, :n_sup], labels, pos_weight)
        + λ_sparse · L1(acts[:, n_sup:])           # unsup latents only
        [+ optional hierarchy loss from existing train.py machinery]

    Returns the same 4-tuple shape as legacy SupervisedSAE so downstream
    steps (causal, intervention, composition, promote_loop) can treat
    this interchangeably:
        recon, sup_pre, sup_acts, all_acts
    """

    def __init__(self, d_model: int, n_supervised: int, n_unsupervised: int):
        super().__init__()
        self.d_model = d_model
        self.n_supervised = n_supervised
        self.n_unsupervised = n_unsupervised
        self.n_total = n_supervised + n_unsupervised

        self.encoder = nn.Linear(d_model, self.n_total, bias=True)
        self.decoder = nn.Linear(self.n_total, d_model, bias=True)

        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        with torch.no_grad():
            # Unit-normalize decoder columns at init.
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        z = self.encoder(x)                                # (batch, n_total)
        acts = F.relu(z)
        recon = self.decoder(acts)
        sup_pre = z[..., : self.n_supervised]
        sup_acts = acts[..., : self.n_supervised]
        return recon, sup_pre, sup_acts, acts

    @torch.no_grad()
    def normalize_decoder(self, skip_first_n: int = 0):
        """Unit-normalize decoder columns. `skip_first_n` preserved for
        parity with legacy SupervisedSAE.normalize_decoder() signature;
        hinge mode never needs to skip (no frozen columns)."""
        if skip_first_n > 0:
            self.decoder.weight.data[:, skip_first_n:] = F.normalize(
                self.decoder.weight.data[:, skip_first_n:], dim=0,
            )
        else:
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


# ═════════════════════════════════════════════════════════════════════════════
# Formulation 2: JumpReLU + hinge
# ═════════════════════════════════════════════════════════════════════════════


class JumpReLUHingeSAE(nn.Module):
    """JumpReLU + hinge on margin (z - θ).

    For supervised features: f_i = z_i · H(z_i - θ_i), θ_i learnable.
    For unsupervised features: plain ReLU (no θ — the gap property is
    relevant only for named features we want a decisive firing boundary on).

    θ gets gradient ONLY from the hinge term — MSE contributes zero to θ
    (autograd drops the Dirac from differentiating H). This is structurally
    simpler than the original unsupervised JumpReLU training which
    required a rectangular-kernel STE to train θ from the L0 sparsity term.

    The supervised slice returns its RAW pre-activation `z_sup` (not the
    post-JumpReLU sup_acts) as `sup_pre` so the hinge loss sees the value
    the margin is computed against.
    """

    def __init__(
        self, d_model: int, n_supervised: int, n_unsupervised: int,
        theta_init: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_supervised = n_supervised
        self.n_unsupervised = n_unsupervised
        self.n_total = n_supervised + n_unsupervised

        self.encoder = nn.Linear(d_model, self.n_total, bias=True)
        self.decoder = nn.Linear(self.n_total, d_model, bias=True)
        # Per-feature learnable threshold on the supervised slice.
        self.theta = nn.Parameter(torch.full((n_supervised,), float(theta_init)))

        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        z = self.encoder(x)                                # (batch, n_total)
        sup_pre = z[..., : self.n_supervised]
        # JumpReLU for supervised slice: z * H(z - θ). H's derivative is a
        # Dirac autograd treats as zero, so gradient to θ through MSE is 0.
        # Hinge term (outside) gives θ its gradient.
        gate = (sup_pre > self.theta).to(sup_pre.dtype)
        sup_acts = sup_pre * gate

        if self.n_unsupervised > 0:
            unsup_acts = F.relu(z[..., self.n_supervised:])
            acts = torch.cat([sup_acts, unsup_acts], dim=-1)
        else:
            acts = sup_acts

        recon = self.decoder(acts)
        return recon, sup_pre, sup_acts, acts

    @torch.no_grad()
    def normalize_decoder(self, skip_first_n: int = 0):
        if skip_first_n > 0:
            self.decoder.weight.data[:, skip_first_n:] = F.normalize(
                self.decoder.weight.data[:, skip_first_n:], dim=0,
            )
        else:
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


# ═════════════════════════════════════════════════════════════════════════════
# Formulation 3: Gated + BCE (two encoder paths)
# ═════════════════════════════════════════════════════════════════════════════


class GatedBCESAE(nn.Module):
    """Gated SAE with BCE supervision on the gate logit.

    For supervised features:
        gate_logit (π) = W_gate · x + b_gate           # continuous
        magnitude (m)  = W_mag  · x + b_mag            # continuous
        f_sup          = H(π) · ReLU(m)                # gated magnitude
    BCE(σ(π), y) supervises only the gate path; MSE supervises magnitude.

    For unsupervised features: standard ReLU path on a third encoder
    (plain linear), shared decoder.

    Optional weight tying (`tie_weights=True`): W_mag = exp(r) · W_gate
    per-feature, halving supervised-slice encoder parameters at the cost
    of some expressiveness. Following Rajamanoharan et al. (Gated SAEs).

    `sup_pre` returned by forward is the GATE LOGIT π — the quantity BCE
    supervises. Downstream code that expected `sup_pre` to be a raw ReLU
    pre-activation still works for threshold calibration (the logit is
    the classification score), and intervention/ablation code that zeros
    `sup_acts[:, k]` still zeros the gated product as expected.
    """

    def __init__(
        self, d_model: int, n_supervised: int, n_unsupervised: int,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_supervised = n_supervised
        self.n_unsupervised = n_unsupervised
        self.n_total = n_supervised + n_unsupervised
        self.tie_weights = tie_weights

        # Supervised gate path.
        self.gate_encoder = nn.Linear(d_model, n_supervised, bias=True)
        # Supervised magnitude path.
        if tie_weights:
            self.mag_log_scale = nn.Parameter(torch.zeros(n_supervised))
            self.mag_bias = nn.Parameter(torch.zeros(n_supervised))
        else:
            self.mag_encoder = nn.Linear(d_model, n_supervised, bias=True)
        # Unsupervised encoder (single ReLU path).
        if n_unsupervised > 0:
            self.unsup_encoder = nn.Linear(d_model, n_unsupervised, bias=True)
        # Shared decoder for the full n_total latents.
        self.decoder = nn.Linear(self.n_total, d_model, bias=True)

        nn.init.kaiming_uniform_(self.gate_encoder.weight)
        nn.init.zeros_(self.gate_encoder.bias)
        if not tie_weights:
            nn.init.kaiming_uniform_(self.mag_encoder.weight)
            nn.init.zeros_(self.mag_encoder.bias)
        if n_unsupervised > 0:
            nn.init.kaiming_uniform_(self.unsup_encoder.weight)
            nn.init.zeros_(self.unsup_encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
        nn.init.zeros_(self.decoder.bias)

    def _magnitude(self, x):
        if self.tie_weights:
            # W_mag[i] = exp(r_i) · W_gate[i], applied column-wise.
            scale = torch.exp(self.mag_log_scale).unsqueeze(1)   # (n_sup, 1)
            scaled_w = self.gate_encoder.weight * scale
            return F.linear(x, scaled_w, self.mag_bias)
        return self.mag_encoder(x)

    def forward(self, x):
        pi = self.gate_encoder(x)                              # (batch, n_sup) logits
        m = self._magnitude(x)                                 # (batch, n_sup)
        # Heaviside gate; zero gradient through the comparison (autograd).
        gate = (pi > 0).to(pi.dtype)
        sup_acts = gate * F.relu(m)

        if self.n_unsupervised > 0:
            unsup_acts = F.relu(self.unsup_encoder(x))
            acts = torch.cat([sup_acts, unsup_acts], dim=-1)
        else:
            acts = sup_acts

        recon = self.decoder(acts)
        # Return gate logit as sup_pre so BCE can see the pre-threshold
        # quantity and downstream calibration routines (threshold search)
        # operate on a well-defined score.
        return recon, pi, sup_acts, acts

    @torch.no_grad()
    def normalize_decoder(self, skip_first_n: int = 0):
        if skip_first_n > 0:
            self.decoder.weight.data[:, skip_first_n:] = F.normalize(
                self.decoder.weight.data[:, skip_first_n:], dim=0,
            )
        else:
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


# ═════════════════════════════════════════════════════════════════════════════
# Loss functions
# ═════════════════════════════════════════════════════════════════════════════


def hinge_supervision_loss(
    sup_pre: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-feature hinge on pre-activations.

    For each feature i, violation = max(0, -(2·y_i - 1)·z_i):
      - y_i=1, z_i<0 → violation = -z_i (pushes z_i up)
      - y_i=0, z_i>0 → violation = z_i (pushes z_i down)
      - correct-side → zero gradient (the whole point)

    When `pos_weight` is supplied, positive-label violations are scaled by
    pos_weight[i]; negatives are at weight 1. Matches the class-balanced
    BCE policy (`BCEWithLogitsLoss(pos_weight=...)`) so the comparison
    between BCE and hinge isn't confounded by imbalance handling.

    Args:
        sup_pre:    (batch, n_sup) pre-activations for the supervised slice
        labels:     (batch, n_sup) binary labels
        pos_weight: (n_sup,) per-feature positive-class weight, or None
    """
    targets = 2.0 * labels - 1.0                  # {-1, +1}
    violations = F.relu(-targets * sup_pre)       # (batch, n_sup)
    if pos_weight is not None:
        weights = torch.where(
            labels > 0,
            pos_weight.to(labels.dtype).unsqueeze(0).expand_as(labels),
            torch.ones_like(labels),
        )
        violations = violations * weights
    return violations.mean()


def jumprelu_hinge_supervision_loss(
    sup_pre: torch.Tensor,
    theta: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hinge on the JumpReLU margin (z - θ).

    violation_i = max(0, -(2·y_i - 1)·(z_i - θ_i))
    θ gets its gradient from this term (hinge is piecewise linear in θ).
    """
    margin = sup_pre - theta.unsqueeze(0)          # (batch, n_sup)
    targets = 2.0 * labels - 1.0
    violations = F.relu(-targets * margin)
    if pos_weight is not None:
        weights = torch.where(
            labels > 0,
            pos_weight.to(labels.dtype).unsqueeze(0).expand_as(labels),
            torch.ones_like(labels),
        )
        violations = violations * weights
    return violations.mean()


def gated_bce_supervision_loss(
    gate_logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE on the gate logit path of the Gated SAE.

    Identical interface to torch.nn.functional.binary_cross_entropy_with_logits;
    kept here only for API parity with the hinge loss helpers.
    """
    return F.binary_cross_entropy_with_logits(
        gate_logits, labels, pos_weight=pos_weight,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Convenience: which class to instantiate for a given supervision_mode.
# ═════════════════════════════════════════════════════════════════════════════


HINGE_MODES = {"hinge", "hinge_jumprelu", "gated_bce"}


def is_hinge_mode(supervision_mode: str) -> bool:
    """True if the mode uses one of the new end-to-end hinge formulations
    (trained without frozen decoder, without pre-computed target_dirs)."""
    return supervision_mode in HINGE_MODES


def build_hinge_sae(
    supervision_mode: str,
    d_model: int,
    n_supervised: int,
    n_unsupervised: int,
    gated_tie_weights: bool = False,
    theta_init: float = 0.1,
) -> nn.Module:
    """Instantiate the right SAE class for the given hinge-family mode."""
    if supervision_mode == "hinge":
        return HingeSAE(d_model, n_supervised, n_unsupervised)
    if supervision_mode == "hinge_jumprelu":
        return JumpReLUHingeSAE(
            d_model, n_supervised, n_unsupervised, theta_init=theta_init,
        )
    if supervision_mode == "gated_bce":
        return GatedBCESAE(
            d_model, n_supervised, n_unsupervised, tie_weights=gated_tie_weights,
        )
    raise ValueError(
        f"build_hinge_sae: unknown supervision_mode={supervision_mode!r}. "
        f"Expected one of {sorted(HINGE_MODES)}."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Training loop for hinge-family modes
# ═════════════════════════════════════════════════════════════════════════════


def train_hinge_sae(
    activations: torch.Tensor,
    labels: torch.Tensor,
    features: list[dict],
    cfg,
    save_checkpoint: bool = True,
) -> nn.Module:
    """End-to-end training for the three new supervision modes.

    Departs from `train.train_supervised_sae` in these principled ways:

      - NO frozen decoder (mentor's design: encoder + decoder trained
        jointly). Decoder columns are unit-normalized after each optimizer
        step to keep magnitudes interpretable but otherwise trained freely.
      - NO pre-computed target_dirs for the supervision loss. target_dirs
        are still computed post-hoc and saved for evaluation diagnostics
        (so intervention / composition / promote_loop can still reference
        them), but the loss does NOT constrain decoder columns to match.
      - Sparsity penalty applies ONLY to unsupervised latents. Supervised
        sparsity is inherited from the labels, per mentor's doc.
      - Hierarchy loss still applies if cfg.lambda_hier > 0 — it's an
        orthogonal regularizer, not coupled to the supervision mode.

    Uses the same split/warmup/learning-rate conventions as the legacy
    trainer so checkpoints end up in the same paths and downstream eval
    can load them the same way.
    """
    # Deferred imports — these helpers live in train.py and we don't want
    # a circular import at module-load time.
    from .train import (
        set_seed, compute_target_directions, build_hierarchy_map,
        hierarchy_loss,
    )

    set_seed(cfg.seed)

    if labels.dim() == 2:
        labels = labels.unsqueeze(0)
    if activations.dim() == 2:
        activations = activations.unsqueeze(0)

    N, T, d_model = activations.shape
    n_features = labels.shape[-1]
    n_supervised = n_features

    x_flat = activations.reshape(-1, d_model)
    y_flat = labels.reshape(-1, n_supervised)

    n_total_vecs = x_flat.shape[0]
    perm = torch.randperm(n_total_vecs)
    split_idx = int(cfg.train_fraction * n_total_vecs)
    train_idx, test_idx = perm[:split_idx], perm[split_idx:]
    if save_checkpoint:
        torch.save(perm, cfg.split_path)
    x_train, x_test = x_flat[train_idx], x_flat[test_idx]
    y_train, y_test = y_flat[train_idx], y_flat[test_idx]
    print(f"Training data: {x_train.shape[0]:,} vectors, "
          f"Test data: {x_test.shape[0]:,} vectors")

    baseline_mse = F.mse_loss(
        x_train.mean(0, keepdim=True).expand_as(x_train), x_train,
    ).item()

    # Compute target_dirs for post-hoc diagnostic + downstream merge/intervention.
    # NOT used in loss. NOT constrained on the decoder.
    post_hoc_dirs, raw_norms, raw_counts = compute_target_directions(
        x_train, y_train, n_supervised,
    )
    valid = (raw_counts > 0) & (raw_norms > 1e-6)
    print(f"\n  Post-hoc target_dirs: {valid.sum().item()}/{n_supervised} "
          f"features have signal (diagnostic only, NOT used in loss)")
    if save_checkpoint:
        torch.save(post_hoc_dirs.cpu(), cfg.target_dirs_path)

    # Per-feature pos_weight: same policy as the legacy BCE path.
    pos_counts = y_train.sum(dim=0).clamp(min=1.0)
    neg_counts = y_train.shape[0] - pos_counts
    pos_weight = (neg_counts / pos_counts).clamp(max=100.0).to(cfg.device)

    hier_map = build_hierarchy_map(features)

    # Build SAE per supervision mode.
    sae = build_hinge_sae(
        supervision_mode=cfg.supervision_mode,
        d_model=d_model,
        n_supervised=n_supervised,
        n_unsupervised=cfg.n_unsupervised,
        gated_tie_weights=getattr(cfg, "gated_tie_weights", False),
        theta_init=getattr(cfg, "jumprelu_theta_init", 0.1),
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(sae.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * (x_train.shape[0] // cfg.batch_size),
    )

    x_train_dev = x_train.to(cfg.device)
    y_train_dev = y_train.to(cfg.device)
    x_test_dev = x_test.to(cfg.device)
    y_test_dev = y_test.to(cfg.device)

    print(f"\nTraining: {cfg.epochs} epochs, "
          f"{x_train.shape[0] // cfg.batch_size} steps/epoch")
    print(f"  supervision_mode={cfg.supervision_mode}")
    print(f"  n_supervised={n_supervised}  n_unsupervised={cfg.n_unsupervised}")
    print(f"  lambda_sup={cfg.lambda_sup}  lambda_sparse={cfg.lambda_sparse}  "
          f"lambda_hier={cfg.lambda_hier}")
    print(f"  decoder: NOT FROZEN (end-to-end training)")
    print(f"  baseline_mse={baseline_mse:.6f}")

    step = 0
    warmup = cfg.warmup_steps

    for epoch in range(cfg.epochs):
        sae.train()
        shuffle_idx = torch.randperm(x_train.shape[0])
        recon_sum = sup_sum = sparse_sum = hier_sum = 0.0
        n_batches = 0

        for i in range(0, x_train.shape[0], cfg.batch_size):
            idx = shuffle_idx[i : i + cfg.batch_size]
            x_b = x_train_dev[idx]
            y_b = y_train_dev[idx]

            recon, sup_pre, sup_acts, all_acts = sae(x_b)

            recon_loss = F.mse_loss(recon, x_b)

            # Supervision loss
            if cfg.supervision_mode == "hinge":
                sup_loss = hinge_supervision_loss(sup_pre, y_b, pos_weight)
            elif cfg.supervision_mode == "hinge_jumprelu":
                sup_loss = jumprelu_hinge_supervision_loss(
                    sup_pre, sae.theta, y_b, pos_weight,
                )
            elif cfg.supervision_mode == "gated_bce":
                sup_loss = gated_bce_supervision_loss(
                    sup_pre, y_b, pos_weight,
                )
            else:
                raise RuntimeError(
                    f"train_hinge_sae called with non-hinge mode "
                    f"{cfg.supervision_mode!r}"
                )

            # Sparsity ONLY on unsupervised latents (mentor's "sparsity
            # inherited from labels" for the supervised slice).
            if cfg.n_unsupervised > 0 and cfg.lambda_sparse > 0:
                unsup_acts = all_acts[..., n_supervised:]
                sparse_loss = unsup_acts.abs().mean()
            else:
                sparse_loss = torch.tensor(0.0, device=cfg.device)

            if cfg.lambda_hier > 0 and hier_map:
                hier_l = hierarchy_loss(sup_acts, hier_map)
            else:
                hier_l = torch.tensor(0.0, device=cfg.device)

            sup_w = cfg.lambda_sup * min(1.0, step / max(1, warmup))
            loss = (
                recon_loss
                + sup_w * sup_loss
                + cfg.lambda_sparse * sparse_loss
                + cfg.lambda_hier * hier_l
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Normalize decoder columns AFTER each step (all columns — no
            # frozen slice under these modes).
            sae.normalize_decoder(skip_first_n=0)

            recon_sum += float(recon_loss.item())
            sup_sum += float(sup_loss.item())
            sparse_sum += float(sparse_loss.item() if torch.is_tensor(sparse_loss) else sparse_loss)
            hier_sum += float(hier_l.item())
            n_batches += 1
            step += 1

        # Val pass
        sae.eval()
        with torch.no_grad():
            vrec, vsp, vsa, vaa = sae(x_test_dev)
            val_recon = F.mse_loss(vrec, x_test_dev).item()
            val_r2 = 1.0 - val_recon / baseline_mse
            if cfg.supervision_mode == "hinge":
                val_sup = hinge_supervision_loss(vsp, y_test_dev, pos_weight).item()
            elif cfg.supervision_mode == "hinge_jumprelu":
                val_sup = jumprelu_hinge_supervision_loss(
                    vsp, sae.theta, y_test_dev, pos_weight,
                ).item()
            else:
                val_sup = gated_bce_supervision_loss(
                    vsp, y_test_dev, pos_weight,
                ).item()

        train_r2 = 1.0 - (recon_sum / max(1, n_batches)) / baseline_mse
        print(
            f"  Epoch {epoch + 1:>2}  "
            f"recon={recon_sum / max(1, n_batches):.5f}  "
            f"sup={sup_sum / max(1, n_batches):.5f}  "
            f"sparse={sparse_sum / max(1, n_batches):.5f}  "
            f"hier={hier_sum / max(1, n_batches):.5f}  "
            f"R2={train_r2:.3f}  lr={optimizer.param_groups[0]['lr']:.2e}\n"
            f"           val_recon={val_recon:.5f}  val_sup={val_sup:.5f}  "
            f"val_R2={val_r2:.3f}"
        )
        if (epoch + 1) % 5 == 0 and save_checkpoint:
            ckpt_path = cfg.output_dir / f"supervised_sae_epoch{epoch + 1}.pt"
            torch.save(sae.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    sae_cpu = sae.cpu()
    if save_checkpoint:
        torch.save(sae_cpu.state_dict(), cfg.checkpoint_path)
        torch.save(
            {
                "d_model": d_model,
                "n_supervised": n_supervised,
                "n_unsupervised": cfg.n_unsupervised,
                "n_lista_steps": 0,
                "supervision_mode": cfg.supervision_mode,
                "gated_tie_weights": getattr(cfg, "gated_tie_weights", False),
                "jumprelu_theta_init": getattr(cfg, "jumprelu_theta_init", 0.1),
            },
            cfg.checkpoint_config_path,
        )
        from .cache_meta import write_cache_meta
        write_cache_meta(
            cfg.checkpoint_path, "supervised_sae", cfg,
            n_features=n_supervised,
        )
        print(f"\nModel saved: {cfg.checkpoint_path}")

    return sae_cpu

