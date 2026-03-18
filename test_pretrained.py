"""Diagnostic: verify pretrained SAE reconstruction works correctly.

Run from the repo root after Step 2 (activations.pt must exist):
    python test_pretrained.py
"""
import torch
from pipeline.config import Config
from pipeline.inventory import PretrainedSAE

cfg = Config()
acts = torch.load("pipeline_data/activations.pt", weights_only=True)
x = acts[0, :5, :].float()
baseline_mse = (x - x.mean(0, keepdim=True)).pow(2).mean().item()
print(f"Input norm: {x.norm(dim=-1).mean():.2f}")
print(f"Baseline MSE (mean predictor): {baseline_mse:.2f}")

# Load raw sae_lens object
from sae_lens import SAE
raw, _, _ = SAE.from_pretrained(
    release=cfg.sae_release, sae_id=cfg.sae_id, device="cpu"
)
print(f"\nsae_lens W_enc shape: {raw.W_enc.shape}")
print(f"sae_lens W_dec shape: {raw.W_dec.shape}")
print(f"Has threshold: {hasattr(raw, 'threshold') and raw.threshold is not None}")

# Test A: sae_lens native forward (ground truth)
with torch.no_grad():
    recon_native = raw(x)
mse_a = (recon_native - x).pow(2).mean().item()
r2_a = 1 - mse_a / baseline_mse
print(f"\nA) sae_lens native forward:  MSE={mse_a:.4f}  R²={r2_a:.4f}")

# Test B: our PretrainedSAE wrapper (no transpose)
threshold = None
if hasattr(raw, "threshold") and raw.threshold is not None:
    threshold = raw.threshold.data.clone()
elif hasattr(raw, "log_threshold"):
    threshold = raw.log_threshold.data.exp()

sae_ours = PretrainedSAE(
    W_enc=raw.W_enc.data.clone(),
    W_dec=raw.W_dec.data.clone(),
    b_enc=raw.b_enc.data.clone(),
    b_dec=raw.b_dec.data.clone(),
    threshold=threshold,
)
recon_ours = sae_ours.decode(sae_ours.encode(x))
mse_b = (recon_ours - x).pow(2).mean().item()
r2_b = 1 - mse_b / baseline_mse
print(f"B) Our PretrainedSAE:        MSE={mse_b:.4f}  R²={r2_b:.4f}")

# Test C: with b_dec subtraction in encoder
z_bdec = (x - sae_ours.b_dec) @ sae_ours.W_enc + sae_ours.b_enc
if sae_ours.threshold is not None:
    z_bdec = (z_bdec > sae_ours.threshold) * torch.relu(z_bdec)
else:
    z_bdec = torch.relu(z_bdec)
recon_bdec = z_bdec @ sae_ours.W_dec + sae_ours.b_dec
mse_c = (recon_bdec - x).pow(2).mean().item()
r2_c = 1 - mse_c / baseline_mse
print(f"C) With b_dec subtraction:   MSE={mse_c:.4f}  R²={r2_c:.4f}")

print("\n--- Interpretation ---")
if r2_a > 0.5:
    print("sae_lens native works. If B or C don't match A, our wrapper has a bug.")
else:
    print("sae_lens native also broken — may be dtype or activation mismatch.")
