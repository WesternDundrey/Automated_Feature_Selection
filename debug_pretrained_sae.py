"""Diagnostic — find the activation-extraction path that gpt2-small-res-jb wants.

The pretrained SAE baseline reconstructs at R² = -20000 no matter what we do
inside the SAE wrapper. The supervised SAE trained on the SAME cached
activations reconstructs at R² = 0.988. So the activations aren't garbage —
they're just in a different space than the pretrained SAE expects.

This script runs the pretrained SAE against activations extracted four
different ways and prints the R² for each. Whichever one gives R² > 0.5 is
the right path; we hardcode that in the pipeline.

Usage (on vast.ai):
    cd /workspace/Automated_Feature_Selection
    python debug_pretrained_sae.py
"""

import torch
import torch.nn.functional as F

print("=" * 64)
print("DIAGNOSTIC: pretrained SAE reconstruction on gpt2-small-res-jb")
print("=" * 64)

# ── Load SAE ─────────────────────────────────────────────────────────
from sae_lens import SAE
sae, cfg_dict, _ = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.6.hook_resid_pre",
    device="cuda",
)
sae.eval()

print(f"\nSAE cfg:")
print(f"  hook_name: {sae.cfg.hook_name}")
print(f"  apply_b_dec_to_input: {sae.cfg.apply_b_dec_to_input}")
print(f"  normalize_activations: {getattr(sae.cfg, 'normalize_activations', '(unset)')}")
print(f"  activation_fn_str: {getattr(sae.cfg, 'activation_fn_str', '(unset)')}")
print(f"  finetuning_scaling_factor: {getattr(sae.cfg, 'finetuning_scaling_factor', '(unset)')}")
print(f"  dtype: {sae.cfg.dtype}")

mfpk = cfg_dict.get("model_from_pretrained_kwargs") or {}
print(f"\nmodel_from_pretrained_kwargs: {mfpk}")

# Check for scaling_factor buffer (stored as part of normalize_activations)
for name, buf in sae.named_buffers():
    print(f"  buffer: {name}  shape={tuple(buf.shape)}  "
          f"mean={buf.float().mean().item():.4g}  norm={buf.float().norm().item():.4g}")


# ── Helper: measure R² of sae(x) ─────────────────────────────────────
def measure(x, label):
    x = x.to("cuda").float()
    with torch.no_grad():
        try:
            recon_full = sae(x)
        except Exception as e:
            print(f"  {label}: sae(x) failed ({e})")
            recon_full = None
        try:
            z = sae.encode(x)
            recon_split = sae.decode(z)
        except Exception as e:
            print(f"  {label}: encode+decode failed ({e})")
            recon_split = None

    base = F.mse_loss(x.mean(0, keepdim=True).expand_as(x), x).item()
    x_norm = x.norm(dim=-1).mean().item()
    print(f"  {label}")
    print(f"    x_norm={x_norm:.2f}  baseline_mse={base:.2f}")

    if recon_full is not None:
        mse_f = F.mse_loss(recon_full, x).item()
        r2_f = 1 - mse_f / max(base, 1e-9)
        r_norm_f = recon_full.norm(dim=-1).mean().item()
        print(f"    sae(x):          R²={r2_f:+.4f}  MSE={mse_f:.2f}  recon_norm={r_norm_f:.2f}")
    if recon_split is not None:
        mse_s = F.mse_loss(recon_split, x).item()
        r2_s = 1 - mse_s / max(base, 1e-9)
        r_norm_s = recon_split.norm(dim=-1).mean().item()
        print(f"    decode(encode(x)): R²={r2_s:+.4f}  MSE={mse_s:.2f}  recon_norm={r_norm_s:.2f}")


# ── Reference text for all tests ─────────────────────────────────────
sample_text = "The quick brown fox jumps over the lazy dog. " * 10
hook = sae.cfg.hook_name

# ── Test 1: HookedTransformer.from_pretrained_no_processing with SAE kwargs
print("\n" + "=" * 64)
print("Test 1: HookedTransformer.from_pretrained_no_processing + SAE kwargs")
print("        (what the pipeline currently does)")
print("=" * 64)
from transformer_lens import HookedTransformer
m = HookedTransformer.from_pretrained_no_processing(
    "gpt2", device="cuda", **mfpk,
).eval()
tokens = m.to_tokens(sample_text)
_, cache = m.run_with_cache(tokens, names_filter=[hook])
x = cache[hook].reshape(-1, cache[hook].shape[-1])
measure(x, "activations from Test 1")
del m, cache

# ── Test 2: HookedTransformer.from_pretrained_no_processing WITHOUT kwargs
print("\n" + "=" * 64)
print("Test 2: HookedTransformer.from_pretrained_no_processing (no kwargs)")
print("=" * 64)
m = HookedTransformer.from_pretrained_no_processing("gpt2", device="cuda").eval()
_, cache = m.run_with_cache(tokens, names_filter=[hook])
x = cache[hook].reshape(-1, cache[hook].shape[-1])
measure(x, "activations from Test 2")
del m, cache

# ── Test 3: HookedTransformer.from_pretrained (STANDARD path) ────────
print("\n" + "=" * 64)
print("Test 3: HookedTransformer.from_pretrained (standard, with LN folding)")
print("=" * 64)
m = HookedTransformer.from_pretrained("gpt2", device="cuda").eval()
_, cache = m.run_with_cache(tokens, names_filter=[hook])
x = cache[hook].reshape(-1, cache[hook].shape[-1])
measure(x, "activations from Test 3")
del m, cache

# ── Test 4: Cached activations from pipeline_data/ ───────────────────
print("\n" + "=" * 64)
print("Test 4: cached pipeline_data/activations.pt (first 4 sequences)")
print("=" * 64)
try:
    acts = torch.load(
        "pipeline_data/activations.pt", weights_only=True, map_location="cpu",
    )
    x = acts[:4].reshape(-1, acts.shape[-1])
    print(f"  cached shape: {tuple(acts.shape)}, sampled: {tuple(x.shape)}")
    measure(x, "cached activations")
except FileNotFoundError:
    print("  pipeline_data/activations.pt not found (run step annotate first)")

print("\n" + "=" * 64)
print("INTERPRETATION")
print("=" * 64)
print(
    "Whichever test above gives R² > 0.5 is the extraction path that matches\n"
    "this SAE's training distribution. If Test 1 (our current path) is bad but\n"
    "another is good, we change load_target_model to match. If ALL paths are\n"
    "bad, the SAE expects dataset-wide normalization we're not applying and\n"
    "we need to look for a buffer/scaling_factor on the SAE object.\n"
)
