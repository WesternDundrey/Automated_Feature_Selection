"""Diagnostic — find the activation-extraction path that gpt2-small-res-jb wants.

Writes no assumptions about sae_lens API — inspects what's actually on the
SAE object and uses string fallbacks where attributes aren't there.

Usage (on vast.ai):
    cd /workspace/Automated_Feature_Selection
    python debug_pretrained_sae.py
"""

import torch
import torch.nn.functional as F

HOOK_NAME = "blocks.6.hook_resid_pre"  # hardcoded — we know what it is for this SAE
SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.6.hook_resid_pre"

print("=" * 64)
print("DIAGNOSTIC: pretrained SAE on gpt2-small-res-jb")
print("=" * 64)

# ── Load SAE — try new API first, fall back to old ───────────────────
from sae_lens import SAE

cfg_dict = None
sparsity = None
try:
    # New API (sae_lens >= 4.0 or so)
    sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
        release=SAE_RELEASE, sae_id=SAE_ID, device="cuda",
    )
    print("\nLoaded via: from_pretrained_with_cfg_and_sparsity (new API)")
except (AttributeError, TypeError):
    # Old API — still works, just deprecated
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SAE.from_pretrained(
            release=SAE_RELEASE, sae_id=SAE_ID, device="cuda",
        )
    if isinstance(result, tuple):
        sae = result[0]
        cfg_dict = result[1] if len(result) > 1 else {}
        sparsity = result[2] if len(result) > 2 else None
        print("\nLoaded via: from_pretrained (old tuple-unpacking API)")
    else:
        sae = result
        cfg_dict = {}
        print("\nLoaded via: from_pretrained (new-API-but-single-return)")

sae.eval()

# ── Dump cfg attributes so we can see what's actually there ──────────
print("\nsae type:", type(sae).__name__)
cfg = getattr(sae, "cfg", None)
print("sae.cfg type:", type(cfg).__name__ if cfg is not None else "(none)")
if cfg is not None:
    print("sae.cfg public attributes:")
    for attr in sorted(dir(cfg)):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(cfg, attr)
            if callable(val):
                continue
            sval = repr(val)
            if len(sval) > 100:
                sval = sval[:97] + "..."
            print(f"  {attr} = {sval}")
        except Exception as e:
            print(f"  {attr} = <error: {e}>")

# ── Dump any buffers (scaling_factor, activation_norm, etc.) ─────────
print("\nsae.named_buffers():")
had_buffers = False
for name, buf in sae.named_buffers():
    had_buffers = True
    print(f"  {name:<40}  shape={tuple(buf.shape)}  "
          f"mean={buf.float().mean().item():.4g}  "
          f"norm={buf.float().norm().item():.4g}")
if not had_buffers:
    print("  (no buffers — nothing to scale)")

print(f"\nhook used for extraction: {HOOK_NAME}")

mfpk = (cfg_dict or {}).get("model_from_pretrained_kwargs", {}) if isinstance(cfg_dict, dict) else {}
print(f"model_from_pretrained_kwargs (from cfg_dict): {mfpk}")


# ── Helper: measure R² of sae(x) and sae.encode/decode ───────────────
def measure(x, label):
    x = x.to("cuda").float()
    base = F.mse_loss(x.mean(0, keepdim=True).expand_as(x), x).item()
    print(f"  {label}")
    print(f"    x_norm={x.norm(dim=-1).mean().item():.2f}  "
          f"baseline_mse={base:.2f}")
    with torch.no_grad():
        try:
            recon = sae(x)
            mse = F.mse_loss(recon, x).item()
            r2 = 1 - mse / max(base, 1e-9)
            print(f"    sae(x):            R²={r2:+.4f}  MSE={mse:.2f}  "
                  f"recon_norm={recon.norm(dim=-1).mean().item():.2f}")
        except Exception as e:
            print(f"    sae(x) FAILED: {type(e).__name__}: {e}")
        try:
            z = sae.encode(x)
            recon = sae.decode(z)
            mse = F.mse_loss(recon, x).item()
            r2 = 1 - mse / max(base, 1e-9)
            print(f"    decode(encode(x)): R²={r2:+.4f}  MSE={mse:.2f}  "
                  f"recon_norm={recon.norm(dim=-1).mean().item():.2f}")
        except Exception as e:
            print(f"    encode+decode FAILED: {type(e).__name__}: {e}")


# ── Test 1: from_pretrained_no_processing + SAE kwargs (current pipeline)
print("\n" + "=" * 64)
print("Test 1: HookedTransformer.from_pretrained_no_processing + SAE kwargs")
print("        (what the pipeline currently does)")
print("=" * 64)
from transformer_lens import HookedTransformer
sample_text = "The quick brown fox jumps over the lazy dog. " * 10
try:
    m = HookedTransformer.from_pretrained_no_processing(
        "gpt2", device="cuda", **mfpk,
    ).eval()
    tokens = m.to_tokens(sample_text)
    _, cache = m.run_with_cache(tokens, names_filter=[HOOK_NAME])
    x = cache[HOOK_NAME].reshape(-1, cache[HOOK_NAME].shape[-1])
    measure(x, "Test 1 activations")
    del m, cache
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  Test 1 failed to load: {type(e).__name__}: {e}")

# ── Test 2: from_pretrained_no_processing WITHOUT kwargs ─────────────
print("\n" + "=" * 64)
print("Test 2: HookedTransformer.from_pretrained_no_processing (no kwargs)")
print("=" * 64)
try:
    m = HookedTransformer.from_pretrained_no_processing("gpt2", device="cuda").eval()
    tokens = m.to_tokens(sample_text)
    _, cache = m.run_with_cache(tokens, names_filter=[HOOK_NAME])
    x = cache[HOOK_NAME].reshape(-1, cache[HOOK_NAME].shape[-1])
    measure(x, "Test 2 activations")
    del m, cache
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  Test 2 failed: {type(e).__name__}: {e}")

# ── Test 3: standard from_pretrained (with LN folding etc.) ──────────
print("\n" + "=" * 64)
print("Test 3: HookedTransformer.from_pretrained (standard)")
print("=" * 64)
try:
    m = HookedTransformer.from_pretrained("gpt2", device="cuda").eval()
    tokens = m.to_tokens(sample_text)
    _, cache = m.run_with_cache(tokens, names_filter=[HOOK_NAME])
    x = cache[HOOK_NAME].reshape(-1, cache[HOOK_NAME].shape[-1])
    measure(x, "Test 3 activations")
    del m, cache
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  Test 3 failed: {type(e).__name__}: {e}")

# ── Test 4: cached activations from pipeline ─────────────────────────
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
    print("  pipeline_data/activations.pt not found (run annotate first)")
except Exception as e:
    print(f"  Test 4 failed: {type(e).__name__}: {e}")

print("\n" + "=" * 64)
print("READ THE RESULTS")
print("=" * 64)
print("Whichever test gives R² > 0.5 is the correct extraction path.")
print("If all tests show recon_norm >> x_norm (5x+), the SAE expects a")
print("dataset-wide scaling we're missing — look at named_buffers above.")
