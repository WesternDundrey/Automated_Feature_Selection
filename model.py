"""
model.py — Supervised Sparse Autoencoder.

Latent space is split into:
  - n_supervised: one latent per feature in the catalog, trained with a supervised loss
  - n_unsupervised: free latents that absorb whatever supervised features don't cover

The decoder columns are kept unit-norm after each optimizer step so that the
sparsity penalty (L1 on activations) is not gamed by shrinking decoder norms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedSAE(nn.Module):
    def __init__(self, d_model: int, n_supervised: int, n_unsupervised: int):
        super().__init__()
        self.d_model = d_model
        self.n_supervised = n_supervised
        self.n_total = n_supervised + n_unsupervised

        self.encoder = nn.Linear(d_model, self.n_total, bias=True)
        self.decoder = nn.Linear(self.n_total, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        # Normalize decoder columns to unit norm at init
        with torch.no_grad():
            self._normalize_decoder_inplace()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (..., d_model)

        Returns:
            recon:      (..., d_model)   reconstructed activations
            sup_pre:    (..., n_supervised)  pre-ReLU logits for supervised latents
            sup_acts:   (..., n_supervised)  post-ReLU activations for supervised latents
            all_acts:   (..., n_total)    all post-ReLU activations (for sparsity loss)
        """
        pre = self.encoder(x)                          # (..., n_total)
        acts = F.relu(pre)                             # (..., n_total)
        recon = self.decoder(acts)                     # (..., d_model)

        sup_pre = pre[..., : self.n_supervised]
        sup_acts = acts[..., : self.n_supervised]

        return recon, sup_pre, sup_acts, acts

    @torch.no_grad()
    def normalize_decoder(self):
        """Normalize each decoder column to unit norm. Call after every optimizer step."""
        self._normalize_decoder_inplace()

    def _normalize_decoder_inplace(self):
        # decoder.weight has shape (d_model, n_total); columns are feature directions
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
