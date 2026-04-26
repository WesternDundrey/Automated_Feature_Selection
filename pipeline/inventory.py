"""
Step 1 — Feature Inventory

Load a pretrained SAE, collect top-activating examples for selected latents,
generate descriptions with Claude, and organize into a hierarchical catalog.

Outputs:
    pipeline_data/top_activations.json   Top examples per latent
    pipeline_data/raw_descriptions.json  Initial Claude descriptions
    pipeline_data/feature_catalog.json   Hierarchical feature catalog

Usage:
    python -m pipeline.inventory
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .config import Config


def _extract_activations_subprocess(cfg) -> None:
    """Extract tokens.pt + activations.pt in a fresh Python subprocess so
    no CUDA context bleeds into this parent process. Mirrors the pattern
    annotate.py uses (see its `extract_script` block) — keeps the parent
    GPU-pristine for vLLM cold-start.

    v8.18.11: introduced because v8.17's in-parent extraction during the
    Delphi gate caused vLLM to hang at parallel_state init in the
    downstream annotate step (parent CUDA context was dirty from the
    HookedTransformer forward passes that produced activations.pt).
    """
    extract_script = f"""
import torch
from pipeline.annotate import prepare_corpus, extract_activations
from pipeline.inventory import load_target_model
from pipeline.config import Config

cfg = Config(
    model_name={cfg.model_name!r}, device={cfg.device!r},
    model_dtype={cfg.model_dtype!r}, n_sequences={cfg.n_sequences},
    seq_len={cfg.seq_len}, corpus_batch_size={cfg.corpus_batch_size},
    target_layer={cfg.target_layer}, output_dir={str(cfg.output_dir)!r},
    hook_point={cfg.hook_point!r},
    sae_release={cfg.sae_release!r}, sae_id={cfg.sae_id!r},
)
cfg.output_dir.mkdir(parents=True, exist_ok=True)
model = load_target_model(cfg)
from pipeline.cache_meta import write_cache_meta

if not cfg.tokens_path.exists():
    toks = prepare_corpus(model, cfg)
    torch.save(toks, cfg.tokens_path)
    write_cache_meta(cfg.tokens_path, "tokens", cfg)
else:
    toks = torch.load(cfg.tokens_path, weights_only=True)

if not cfg.activations_path.exists():
    acts = extract_activations(model, toks, cfg)
    torch.save(acts, cfg.activations_path)
    write_cache_meta(cfg.activations_path, "activations", cfg)
"""
    subprocess.run([sys.executable, "-c", extract_script], check=True)


def _run_delphi_gate_subprocess(mode: str, payload: dict, cfg) -> tuple:
    """Run a Delphi gate in a fresh Python subprocess so its imports
    (faiss, sentence_transformers, bitsandbytes, vllm via Delphi's
    pyproject) NEVER enter this process. v8.18.10 fix for the
    "delphi processes features then vllm hangs" bug — by isolating
    Delphi to a subprocess, the main pipeline process's CUDA / sys.modules
    state stays clean so the downstream vLLM annotator starts cleanly.

    `mode` must be 'inventory' or 'organized'.
    `payload` is what the subprocess reads from --input.
    `cfg` is the Config instance we serialize a relevant subset of.

    Returns (kept_descriptions_or_filtered_catalog, score_log).
    """
    cfg_dict = {
        "model_name": cfg.model_name,
        "device": cfg.device,
        "model_dtype": cfg.model_dtype,
        "target_layer": cfg.target_layer,
        "hook_point": cfg.hook_point,
        "sae_release": cfg.sae_release,
        "sae_id": cfg.sae_id,
        "seed": cfg.seed,
        "top_k_examples": getattr(cfg, "top_k_examples", 30),
        "delphi_held_out_n": getattr(cfg, "delphi_held_out_n", 10),
        "delphi_n_mid_tier": getattr(cfg, "delphi_n_mid_tier", 3),
        "delphi_score_threshold": getattr(cfg, "delphi_score_threshold", 0.7),
        "delphi_judge_model": getattr(cfg, "delphi_judge_model", ""),
        "organization_model": cfg.organization_model,
    }

    with tempfile.TemporaryDirectory(prefix="delphi_gate_") as tmpdir:
        in_path  = Path(tmpdir) / "input.json"
        out_path = Path(tmpdir) / "output.json"
        cfg_path = Path(tmpdir) / "cfg.json"
        in_path.write_text(json.dumps(payload))
        cfg_path.write_text(json.dumps(cfg_dict))

        cmd = [
            sys.executable, "-m", "pipeline.delphi_subprocess",
            mode,
            "--input",  str(in_path),
            "--output", str(out_path),
            "--cfg",    str(cfg_path),
        ]
        # Inherit env so OPENROUTER_API_KEY etc. propagate.
        env = os.environ.copy()
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            # Subprocess wrote a sentinel error to output.json; read it.
            if out_path.exists():
                err = json.loads(out_path.read_text()).get("_subprocess_error")
                raise RuntimeError(f"Delphi gate subprocess failed: {err}") from e
            raise

        result = json.loads(out_path.read_text())
        if "_subprocess_error" in result:
            raise RuntimeError(
                f"Delphi gate subprocess error: {result['_subprocess_error']}"
            )
        if mode == "inventory":
            return result["kept_descriptions"], result["score_log"]
        else:
            return result["filtered_catalog"], result["score_log"]


def _extract_json_object(text: str) -> dict | None:
    """Extract the first JSON object from text, handling nested braces and strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ── Pretrained SAE wrapper ──────────────────────────────────────────────────

class PretrainedSAE:
    """Thin wrapper for a pretrained SAE with JumpReLU or ReLU activation.

    Two construction paths depending on source:

      GemmaScope npz (bare weights):
        encode: z = JumpReLU_theta(x @ W_enc + b_enc)
        decode: x_hat = z @ W_dec + b_dec
        Used when we load weights directly from the `params.npz` file.

      sae_lens native (delegate encode/decode to the sae_lens SAE object):
        We hold a reference to the live `sae_lens.SAE` and call its own
        `encode()` / `decode()`, which handle apply_b_dec_to_input,
        activation normalization, scaling factors, and any other
        preprocessing the SAE was trained with. The bare-weights path
        couldn't replicate all of this (normalize_activations and
        run_time_activation_norm_fn alone produce R² ≈ -10^4 when missed
        on `gpt2-small-res-jb`). Delegating is both simpler and correct.

    External code (circuit.py, intervention.py) still accesses
    `sae.W_enc`, `sae.W_dec`, etc., so those attributes are always
    populated — either from the live sae_lens weights (native path) or
    from the npz tensors (GemmaScope path).
    """

    def __init__(self, W_enc=None, W_dec=None, b_enc=None, b_dec=None,
                 threshold=None, native_sae=None):
        self._native = native_sae
        if native_sae is not None:
            # Views of the native SAE's parameters. These are the actual
            # Parameter tensors, not copies — moving the native SAE to a
            # new device updates them in place.
            self.W_enc = native_sae.W_enc.data
            self.W_dec = native_sae.W_dec.data
            self.b_enc = native_sae.b_enc.data
            self.b_dec = native_sae.b_dec.data
            self.threshold = (
                native_sae.threshold.data
                if hasattr(native_sae, "threshold")
                and native_sae.threshold is not None
                else None
            )
        else:
            self.W_enc = W_enc        # (d_model, d_sae)
            self.W_dec = W_dec        # (d_sae, d_model)
            self.b_enc = b_enc        # (d_sae,)
            self.b_dec = b_dec        # (d_model,)
            self.threshold = threshold
        self.d_model = self.W_enc.shape[0]
        self.d_sae = self.W_enc.shape[1]

    def encode(self, x):
        """x: (..., d_model) -> (..., d_sae)"""
        if self._native is not None:
            return self._native.encode(x)
        z = x @ self.W_enc + self.b_enc
        if self.threshold is not None:
            mask = z > self.threshold
            z = mask * torch.relu(z)
        else:
            z = torch.relu(z)
        return z

    def decode(self, z):
        """z: (..., d_sae) -> (..., d_model)"""
        if self._native is not None:
            return self._native.decode(z)
        return z @ self.W_dec + self.b_dec

    def to(self, device):
        if self._native is not None:
            self._native = self._native.to(device)
            # Re-bind views after the move.
            self.W_enc = self._native.W_enc.data
            self.W_dec = self._native.W_dec.data
            self.b_enc = self._native.b_enc.data
            self.b_dec = self._native.b_dec.data
            if hasattr(self._native, "threshold") and self._native.threshold is not None:
                self.threshold = self._native.threshold.data
            return self
        self.W_enc = self.W_enc.to(device)
        self.W_dec = self.W_dec.to(device)
        self.b_enc = self.b_enc.to(device)
        self.b_dec = self.b_dec.to(device)
        if self.threshold is not None:
            self.threshold = self.threshold.to(device)
        return self


def load_sae(cfg: Config) -> tuple[PretrainedSAE, torch.Tensor | None]:
    """Load a pretrained SAE.

    For GemmaScope releases we prefer direct npz loading (matches the reference
    implementation in agentic-delphi/delphi/sparse_coders/custom/gemmascope.py).
    sae_lens is known to apply extra preprocessing to GemmaScope weights (e.g.,
    folding in normalization factors or adjusting for apply_b_dec_to_input),
    which breaks reconstruction when we use our bare PretrainedSAE wrapper.
    The npz path bypasses sae_lens entirely and loads the exact weights that
    the original JumpReluSae expects.

    Non-GemmaScope releases (e.g., `gpt2-small-res-jb`) still go through
    sae_lens because we don't have a direct npz loader for them.
    """
    release = cfg.sae_release or ""
    is_gemmascope = "gemma-scope" in release or "gemmascope" in release

    if is_gemmascope:
        try:
            return _load_sae_gemmascope_npz(cfg)
        except Exception as e:
            print(f"Direct npz loading failed ({e}); falling back to sae_lens")

    try:
        return _load_sae_sae_lens(cfg)
    except ImportError:
        if not is_gemmascope:
            print("sae_lens not installed, trying direct GemmaScope npz loading...")
            return _load_sae_gemmascope_npz(cfg)
        raise


def _load_sae_sae_lens(cfg: Config) -> tuple[PretrainedSAE, torch.Tensor | None]:
    from sae_lens import SAE

    print(f"Loading SAE via sae_lens: {cfg.sae_release} / {cfg.sae_id}")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=cfg.sae_release,
        sae_id=cfg.sae_id,
        device="cpu",
    )

    # Delegate encode/decode to the live sae_lens SAE. The bare-weights
    # path (computing x @ W_enc + b_enc ourselves) misses apply_b_dec_to_input,
    # normalize_activations, and any scaling the SAE carries — which are
    # what produced R² ≈ -20000 on gpt2-small-res-jb. Letting sae_lens run
    # its own forward eliminates the whole class of convention mismatches.
    sae.eval()
    wrapped = PretrainedSAE(native_sae=sae)
    cfg_ap = False
    cfg_obj = getattr(sae, "cfg", None)
    if cfg_obj is not None:
        cfg_ap = bool(getattr(cfg_obj, "apply_b_dec_to_input", False))
    print(f"  SAE: d_model={wrapped.d_model}, d_sae={wrapped.d_sae}"
          f"  (native sae_lens forward, apply_b_dec_to_input={cfg_ap})")
    if sparsity is not None:
        print(f"  Sparsity info available ({sparsity.shape})")
    return wrapped, sparsity


def _load_sae_gemmascope_npz(cfg: Config) -> tuple[PretrainedSAE, None]:
    """Load a GemmaScope SAE directly from HuggingFace npz.

    Expected repo: google/gemma-scope-2b-pt-res (or -mlp)
    Expected file: layer_N/width_Wk/average_l0_L/params.npz
    npz keys: W_enc, W_dec, b_enc, b_dec, threshold
    """
    import numpy as np
    from huggingface_hub import hf_hub_download

    # sae_id like "layer_20/width_16k/canonical" -> need to find actual file
    # For GemmaScope repos, the path structure is:
    #   layer_N/width_Wk/average_l0_L/params.npz
    # sae_release is the sae_lens name (e.g., "gemma-scope-2b-pt-res"),
    # but the HF repo needs the "google/" prefix.
    repo_id = cfg.sae_release.replace("-canonical", "")
    if "/" not in repo_id:
        repo_id = f"google/{repo_id}"
    filename = f"{cfg.sae_id}/params.npz"
    # If sae_id ends with /canonical, strip it and try to find a real L0 variant
    if filename.endswith("/canonical/params.npz"):
        filename = filename.replace("/canonical/params.npz", "")
        # Default to a reasonable L0 value
        filename = f"{filename}/average_l0_71/params.npz"

    print(f"Loading GemmaScope npz: {repo_id} / {filename}")
    path = hf_hub_download(repo_id=repo_id, filename=filename)

    params = np.load(path)
    # GemmaScope convention (matches JumpReluSae in agentic-delphi):
    #   W_enc: (d_model, d_sae)
    #   W_dec: (d_sae, d_model)
    W_enc = torch.from_numpy(params["W_enc"].copy())   # (d_model, d_sae)
    W_dec = torch.from_numpy(params["W_dec"].copy())   # (d_sae, d_model)
    b_enc = torch.from_numpy(params["b_enc"].copy())   # (d_sae,)
    b_dec = torch.from_numpy(params["b_dec"].copy())   # (d_model,)

    threshold = None
    if "threshold" in params:
        threshold = torch.from_numpy(params["threshold"].copy())

    wrapped = PretrainedSAE(
        W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec, threshold=threshold,
    )
    print(f"  SAE: d_model={wrapped.d_model}, d_sae={wrapped.d_sae}")
    return wrapped, None


# ── Select latents by firing rate ───────────────────────────────────────────

def load_target_model(cfg: Config):
    """Load the target model in the activation space the pretrained SAE expects.

    Different SAE releases were trained on activations from different
    HookedTransformer loading paths. Picking the wrong path produces
    activations in a different space than the SAE saw at training time,
    which manifests as catastrophic R² (e.g., R² ≈ -20000 on
    gpt2-small-res-jb, confirmed empirically via debug_pretrained_sae.py).

    Rule — verified by measurement:

      GemmaScope releases (contain "gemma-scope" in the name):
        Trained on activations from `from_pretrained_no_processing`
        optionally with `model_from_pretrained_kwargs` from the SAE cfg
        (things like `fold_ln=False`, `center_writing_weights=False`).
        This is what summary4/5/6's R² > 0.9 on Gemma validated.

      sae_lens-native releases (e.g., gpt2-small-res-jb):
        Trained on activations from the STANDARD `from_pretrained`
        (with default LayerNorm folding, weight centering, etc.).
        Their `model_from_pretrained_kwargs` is empty, and the sae_lens
        warning to use `from_pretrained_no_processing` is misleading
        in this case.

    The sae_lens UserWarning ("use from_pretrained_no_processing") fires
    for every SAE with non-empty kwargs — but non-empty kwargs only
    exist for GemmaScope. For sae_lens-native SAEs, kwargs are empty and
    the warning doesn't apply to the intended extraction path.
    """
    from transformer_lens import HookedTransformer

    release = cfg.sae_release or ""
    is_gemmascope = "gemma-scope" in release.lower() or "gemmascope" in release.lower()

    if is_gemmascope:
        kwargs = {}
        try:
            from sae_lens import SAE
            # Try new API first, fall back to deprecated-but-compatible unpacking
            try:
                result = SAE.from_pretrained_with_cfg_and_sparsity(
                    release=cfg.sae_release, sae_id=cfg.sae_id, device="cpu",
                )
                cfg_dict = result[1] if isinstance(result, tuple) and len(result) > 1 else {}
            except (AttributeError, TypeError):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = SAE.from_pretrained(
                        release=cfg.sae_release, sae_id=cfg.sae_id, device="cpu",
                    )
                if isinstance(result, tuple) and len(result) > 1:
                    cfg_dict = result[1]
                else:
                    cfg_dict = {}
            if isinstance(cfg_dict, dict):
                kwargs = cfg_dict.get("model_from_pretrained_kwargs") or {}
            if kwargs:
                print(f"  Target model kwargs from SAE config: {kwargs}")
        except Exception as e:
            print(f"  No sae_lens kwargs ({type(e).__name__}); "
                  f"using from_pretrained_no_processing defaults")
        print(f"  Loader: from_pretrained_no_processing (GemmaScope path)")
        model = HookedTransformer.from_pretrained_no_processing(
            cfg.model_name, device=cfg.device, dtype=cfg.model_dtype, **kwargs,
        )
    else:
        # sae_lens-native SAEs: standard from_pretrained. Verified empirically
        # for gpt2-small-res-jb: R²=+0.99 on standard, R²=-20599 on no_processing.
        print(f"  Loader: from_pretrained (standard sae_lens-native path)")
        model = HookedTransformer.from_pretrained(
            cfg.model_name, device=cfg.device, dtype=cfg.model_dtype,
        )

    model.eval()
    return model


def select_latents(sparsity, cfg: Config) -> list[int]:
    """Select latents whose firing rate falls in [min, max].

    sae_lens sparsity is log(firing_rate). If sparsity is unavailable,
    returns the first n_latents_to_explain indices (user must override).
    """
    if sparsity is None:
        print("  No sparsity info — selecting first N latents.")
        return list(range(cfg.n_latents_to_explain))

    firing_rate = sparsity.exp()
    mask = (firing_rate >= cfg.min_firing_rate) & (firing_rate <= cfg.max_firing_rate)
    candidates = mask.nonzero(as_tuple=False).squeeze(-1).tolist()

    # Sort by firing rate descending (more active = easier to explain)
    candidates.sort(key=lambda i: -firing_rate[i].item())
    selected = candidates[: cfg.n_latents_to_explain]

    print(f"  {len(candidates)} latents in firing-rate window "
          f"[{cfg.min_firing_rate}, {cfg.max_firing_rate}], selected {len(selected)}")
    return selected


# ── Collect top-activating examples ─────────────────────────────────────────

def collect_top_activations(
    model, sae: PretrainedSAE, tokenizer, selected_latents: list[int], cfg: Config
) -> dict:
    """Run model on corpus, collect top-k activating contexts per selected latent.

    Uses transformer_lens API: model.to_tokens() for tokenization,
    model.run_with_cache() for forward pass.

    Returns:
        {str(latent_idx): [{"context_ids": [...], "pos": int, "activation": float}, ...]}
    """
    from datasets import load_dataset
    import heapq

    print(f"Collecting top-{cfg.top_k_examples} activations for "
          f"{len(selected_latents)} latents over "
          f"{cfg.n_tokens_for_activation_collection:,} tokens...")

    dataset = load_dataset(
        cfg.corpus_dataset, split=cfg.corpus_split, streaming=True
    )

    seq_len = cfg.activation_collection_seq_len

    # Per-latent min-heaps: (activation_value, context_ids, pos_in_context)
    heaps: dict[int, list] = {idx: [] for idx in selected_latents}

    # For vectorized top-example extraction: keep a tensor of selected indices
    sel_tensor = torch.tensor(selected_latents, dtype=torch.long)

    n_tokens = 0
    batch_texts = []
    sae_on_device = sae.to(cfg.device)

    for example in tqdm(dataset, desc="Collecting activations", total=None):
        text = example.get("text", "")
        if len(text.strip()) < 50:
            continue
        batch_texts.append(text)

        if len(batch_texts) < cfg.activation_collection_batch_size:
            continue

        # Tokenize with transformer_lens
        try:
            input_ids = model.to_tokens(batch_texts)[:, :seq_len].to(cfg.device)
        except Exception as e:
            if "out of memory" in str(e).lower():
                raise
            batch_texts = []
            continue

        with torch.no_grad():
            _, cache = model.run_with_cache(
                input_ids, names_filter=cfg.hook_point, return_type=None
            )
            resid = cache[cfg.hook_point]  # (batch, seq, d_model)

            # Encode through pretrained SAE
            flat = resid.reshape(-1, sae.d_model)
            acts = sae_on_device.encode(flat.to(sae_on_device.W_enc.dtype))
            acts = acts.reshape(resid.shape[0], resid.shape[1], -1)

        # Only extract columns for selected latents (saves memory)
        sel_acts = acts[:, :, sel_tensor].float().cpu()  # (batch, seq, n_selected)
        ids_cpu = input_ids.cpu()

        # Mask first N positions (BOS / degenerate attention) so top
        # activating examples come from real context positions, not from
        # position-0 artifacts that produce "BOS detector" descriptions.
        _mask_n = int(getattr(cfg, "mask_first_n_positions", 0))
        if _mask_n > 0 and sel_acts.shape[1] > _mask_n:
            sel_acts[:, :_mask_n, :] = float("-inf")

        # Vectorized top-k extraction: for each latent, find the top-k
        # activations in this batch in one pass instead of triple-nested loop
        ids_list = ids_cpu.tolist()
        for si, lat_idx in enumerate(selected_latents):
            lat_acts = sel_acts[:, :, si]  # (batch, seq)
            # Find all positive activations
            positive_mask = lat_acts > 0
            if not positive_mask.any():
                continue

            heap = heaps[lat_idx]
            threshold = heap[0][0] if len(heap) >= cfg.top_k_examples else -1.0

            # Only process positions above current heap threshold
            candidates = (lat_acts > threshold).nonzero(as_tuple=False)
            for pos in candidates:
                b, t = pos[0].item(), pos[1].item()
                val = lat_acts[b, t].item()
                seq_ids = ids_list[b]
                start = max(0, t - 10)
                end = min(len(seq_ids), t + 11)
                context = seq_ids[start:end]
                pos_in_context = t - start

                entry = (val, context, pos_in_context)
                if len(heap) < cfg.top_k_examples:
                    heapq.heappush(heap, entry)
                    threshold = heap[0][0] if len(heap) >= cfg.top_k_examples else -1.0
                elif val > heap[0][0]:
                    heapq.heapreplace(heap, entry)
                    threshold = heap[0][0]

        n_tokens += input_ids.numel()
        batch_texts = []

        if n_tokens >= cfg.n_tokens_for_activation_collection:
            break

    print(f"  Processed {n_tokens:,} tokens")

    # Convert heaps to sorted lists (highest activation first)
    result = {}
    for lat_idx in selected_latents:
        examples = sorted(heaps[lat_idx], key=lambda x: -x[0])
        result[str(lat_idx)] = [
            {
                "context_ids": ex[1],
                "pos": ex[2],
                "activation": round(ex[0], 4),
            }
            for ex in examples
        ]

    n_with_examples = sum(1 for v in result.values() if len(v) > 0)
    print(f"  {n_with_examples}/{len(selected_latents)} latents have activating examples")
    return result


# ── Generate feature descriptions with Claude ───────────────────────────────

def format_examples_for_prompt(examples: list[dict], tokenizer) -> str:
    """Format top-activating examples with Delphi-style >>target<< highlighting."""
    lines = []
    for i, ex in enumerate(examples):
        context_ids = ex["context_ids"]
        pos = ex["pos"]
        # Decode each token individually
        tok_strs = []
        for t in context_ids:
            try:
                tok_strs.append(tokenizer.decode([t]))
            except Exception:
                tok_strs.append(f"<{t}>")
        # Highlight the target token
        if 0 <= pos < len(tok_strs):
            tok_strs[pos] = f"<<{tok_strs[pos]}>>"
        context_str = "".join(tok_strs)
        lines.append(f"  [{i+1}] (act={ex['activation']:.2f}) {context_str}")
    return "\n".join(lines)


def explain_features(top_activations: dict, tokenizer, cfg: Config) -> dict:
    """Generate initial descriptions for each latent using Claude."""
    from .llm import get_client, chat
    client = get_client()
    descriptions = {}

    # Filter to latents that have examples
    latent_indices = [k for k, v in top_activations.items() if len(v) > 0]
    print(f"Explaining {len(latent_indices)} latents with {cfg.explanation_model}...")

    batch_size = cfg.features_per_explanation_batch

    for i in tqdm(range(0, len(latent_indices), batch_size), desc="Explaining"):
        batch = latent_indices[i : i + batch_size]

        prompt_parts = [
            "You are analyzing sparse autoencoder latents from a language model. "
            "For each latent below, you see its top-activating token contexts. "
            "The token with highest activation is marked with <<token>>. "
            "The number in parentheses is the activation strength.\n\n"
            "For EACH latent, write a precise 1-2 sentence description of what "
            "the latent detects. The description must be operationally testable: "
            "someone should be able to read a token in context and decide yes/no "
            "whether this feature should fire on that token.\n\n"
            "Do NOT say 'this latent detects...'. Just state what the feature is, "
            "e.g., 'Token is a color adjective modifying a noun.'\n\n"
            "Reply in this exact format (one per line, no blank lines):\n"
            "LATENT <idx>: <description>\n\n"
            "Here are the latents:\n"
        ]

        # v8.16 (audit fix #3): the describer sees only the first
        # `top_k_examples - delphi_held_out_n` examples; the remaining
        # `delphi_held_out_n` are reserved for the Delphi detection gate
        # so its scoring is genuinely held-out. With defaults (top_k=30,
        # held_out=10), Sonnet sees 20 — same as it always did pre-v8.16
        # — and 10 fresh examples are reserved for scoring.
        n_held_out = int(getattr(cfg, "delphi_held_out_n", 0))
        n_for_describer = max(1, cfg.top_k_examples - n_held_out)
        for lat_idx in batch:
            examples = top_activations[lat_idx][:n_for_describer]
            examples_str = format_examples_for_prompt(examples, tokenizer)
            prompt_parts.append(f"\n--- Latent {lat_idx} ---\n{examples_str}\n")

        prompt = "".join(prompt_parts)

        text = ""
        for attempt in range(3):
            try:
                text = chat(client, cfg.explanation_model, prompt, max_tokens=200 * len(batch))
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Explanation attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(2 ** attempt)
                else:
                    print(f"  Explanation failed after 3 attempts: {e}")

        # Parse "LATENT <idx>: <description>" lines
        for line in text.strip().split("\n"):
            m = re.match(r"LATENT\s+(\d+)\s*:\s*(.+)", line.strip())
            if m:
                idx = m.group(1)
                desc = m.group(2).strip()
                if idx in [str(x) for x in batch]:
                    descriptions[idx] = desc

        time.sleep(0.3)

    print(f"  Generated {len(descriptions)} descriptions")
    return descriptions


# ── Organize into hierarchical catalog ──────────────────────────────────────

def organize_hierarchy(descriptions: dict, cfg: Config) -> dict:
    """Ask Claude to organize feature descriptions into a hierarchical catalog.

    v8.18 (audit fix): every leaf MUST carry a `source_latents: [int]`
    field naming the raw-description latent IDs the leaf derives from.
    Sonnet is no longer allowed to invent gap-filled leaves with no
    source — the post-organize Delphi pass uses `source_latents` to
    re-score each final leaf against an actual SAE latent's activation
    pattern. Without this trace, a final leaf has no testable ground
    truth and silently bypasses Delphi's quality check.

    Sonnet is still allowed to: cluster into groups, drop redundant
    descriptions, choose broad leaf IDs, and rewrite descriptions for
    precision — as long as every leaf names its source(s).
    """
    from .llm import get_client, chat
    client = get_client()

    desc_lines = "\n".join(
        f"- latent_{k}: {v}" for k, v in sorted(descriptions.items(), key=lambda x: int(x[0]))
    )

    prompt = textwrap.dedent(f"""\
        You are building a supervised feature dictionary for a sparse autoencoder
        trained on {cfg.model_name}, layer {cfg.target_layer}.

        Below are {len(descriptions)} initial feature descriptions generated from
        the SAE's top-activating examples. Your job is to turn these into a clean,
        hierarchical feature catalog.

        INITIAL DESCRIPTIONS:
        {desc_lines}

        STRUCTURE RULES:

        Use CATEGORICAL groups where features are natural alternatives:
          punctuation_type: comma, period, question_mark (a token is one or none)
          semantic_domain: politics, sports, science (context belongs to one)
        These are mutually exclusive by nature — don't force it.

        Use NON-EXCLUSIVE groups where features genuinely co-occur:
          token_properties: capitalized, numeric, multi_word (a token can be all)
          text_features: contains_quote, contains_name, past_tense (co-occur freely)

        The key rule: each feature must have a DISTINGUISHABLE activation pattern.
        Two features that always co-occur on the same tokens are redundant — merge
        them. But partial overlap is fine. "is a noun" and "is capitalized" overlap
        on proper nouns but differ on common nouns and acronyms — both are useful.

        BAD: 4 subtypes of comma (comma_contrastive, comma_evaluative, etc.)
          These fire on the same tokens (commas) and can't be distinguished.
          Just have ONE "comma" feature.

        GOOD: "comma" and "list_separator" — a comma in "red, blue, green" is
          both a comma AND a list separator, but commas in other contexts aren't.
          Distinguishable patterns, OK to co-occur.

        Target: 8-15 groups, 30-80 total leaves (depends on what survives).
        Each leaf should fire on at least 1 in 200 tokens in diverse web text.

        YOUR TASKS:
        1. IDENTIFY broad feature dimensions from the descriptions. Use categorical
           groups where natural (punctuation type, part of speech, semantic domain).
           Use non-exclusive groups where features genuinely co-occur.
        2. For each group, pick 3-8 BROAD leaves drawn FROM THE INPUT
           DESCRIPTIONS — do NOT invent leaves with no source. Every leaf
           must trace back to one or more `latent_<id>` from the input
           list above.
           "comma" not "comma_contrastive". "politics" not "official_affiliation_adjective".
        3. REMOVE features that are too narrow (would match fewer than 1 in 200
           tokens), too vague ("general language pattern"), or redundant (always
           co-occur with another feature on the same tokens).
        4. Each leaf description must be short and operationally testable: a reader
           looks at a token in context and decides yes/no.
        5. ABSOLUTELY DO NOT add gap-filling leaves that aren't in the
           input. If only "comma" is in the input among punctuation, do
           NOT add "period" / "question_mark" — those weren't grounded
           in any SAE latent's activation pattern, so we have no way to
           validate them. Stick to what was observed.

        SOURCE TRACE — required field per leaf:
          - `source_latents`: list of integer latent IDs (matching the
            `latent_<N>` numbers above) whose descriptions contributed
            to this leaf. Multiple latents allowed when descriptions
            cluster. NEVER an empty list — if you can't trace a leaf
            to any source latent, drop it instead.

        OUTPUT FORMAT — reply with ONLY this JSON, no other text:
        {{
          "features": [
            {{
              "id": "group_name",
              "description": "What categorical dimension this group represents",
              "type": "group",
              "parent": null
            }},
            {{
              "id": "group_name.value_name",
              "description": "Precise operational description of this value",
              "type": "leaf",
              "parent": "group_name",
              "source_latents": [123, 456]
            }}
          ]
        }}

        Every leaf must have a parent group AND a non-empty source_latents list.
        Groups have type "group" and parent null.
        Leaf IDs use dot notation: group.value.
    """)

    print(f"Organizing {len(descriptions)} descriptions into hierarchy "
          f"with {cfg.organization_model}...")

    catalog = None
    last_err = None
    for attempt in range(3):
        try:
            text = chat(client, cfg.organization_model, prompt, max_tokens=16000)

            catalog = _extract_json_object(text)
            if catalog is not None:
                break
            last_err = ValueError(
                "Could not parse JSON from organization response. "
                f"Response begins: {text[:300]}"
            )
        except Exception as e:
            last_err = e
        if attempt < 2:
            print(f"  Attempt {attempt + 1} failed, retrying...")
            time.sleep(2 ** attempt)

    if catalog is None:
        raise ValueError(
            f"Organization failed after 3 attempts. Last error: {last_err}"
        )

    n_groups = sum(1 for f in catalog["features"] if f["type"] == "group")
    n_leaves = sum(1 for f in catalog["features"] if f["type"] == "leaf")
    print(f"  Catalog: {n_groups} groups, {n_leaves} leaves, {n_groups + n_leaves} total")

    return catalog


# ── Main entry point ────────────────────────────────────────────────────────

def run(cfg: Config = None):
    """Run the full feature inventory pipeline.

    Skips steps whose outputs already exist on disk (resumable).
    """
    if cfg is None:
        cfg = Config()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Check if final output exists
    if cfg.catalog_path.exists():
        print(f"Feature catalog already exists: {cfg.catalog_path}")
        return json.loads(cfg.catalog_path.read_text())

    # 1. Load model and pretrained SAE
    print("Loading base model...")
    model = load_target_model(cfg)
    tokenizer = model.tokenizer

    sae, sparsity = load_sae(cfg)

    # 2. Select latents
    selected = select_latents(sparsity, cfg)

    # 3. Collect top activations (or load cached)
    if cfg.top_activations_path.exists():
        print(f"Loading cached top activations: {cfg.top_activations_path}")
        top_acts = json.loads(cfg.top_activations_path.read_text())
    else:
        top_acts = collect_top_activations(model, sae, tokenizer, selected, cfg)
        cfg.top_activations_path.write_text(json.dumps(top_acts, indent=2))
        print(f"Saved top activations: {cfg.top_activations_path}")

    # 4. Generate descriptions (or load cached)
    if cfg.raw_descriptions_path.exists():
        print(f"Loading cached descriptions: {cfg.raw_descriptions_path}")
        descriptions = json.loads(cfg.raw_descriptions_path.read_text())
    else:
        descriptions = explain_features(top_acts, tokenizer, cfg)
        cfg.raw_descriptions_path.write_text(json.dumps(descriptions, indent=2))
        print(f"Saved raw descriptions: {cfg.raw_descriptions_path}")

    # 4b. (v8.17, refactored to subprocess in v8.18.10)
    # Delphi detection gate — DROP descriptions before catalog.
    #
    # CRITICAL (v8.18.10): the gate runs in a SUBPROCESS, never inside this
    # process. User report: "now that delphi processes those features, it's
    # stuck. it was instant pre-delphi features." Root cause: importing the
    # Delphi graph (faiss, sentence_transformers, bitsandbytes, vllm via
    # Delphi's pyproject) leaves residual sys.modules / CUDA-context state
    # in the main pipeline process. The downstream vLLM annotator (when run
    # as part of a full pipeline invocation) inherits that state and hangs
    # at parallel_state init. Running the gate in a subprocess isolates
    # Delphi's imports completely — the parent process never imports
    # delphi at all, only reads the subprocess's output JSON.
    if getattr(cfg, "delphi_gate_in_inventory", True):
        try:
            # We need raw activations.pt; if it isn't there yet (first
            # inventory run before annotation), extract it via SUBPROCESS
            # so the parent-process CUDA context stays clean for the
            # downstream vLLM annotator.
            #
            # v8.18.11 root-cause fix: pre-fix this block did the extract
            # IN THE PARENT process (HookedTransformer forward passes,
            # heavy CUDA work). After inventory finished, activations.pt
            # existed, so --step annotate's own activation-extraction
            # subprocess (added specifically to "isolate CUDA context"
            # before vLLM startup) was SKIPPED — and vLLM inherited the
            # parent's polluted CUDA context. Cold-start hang at
            # parallel_state init followed. By running extraction in a
            # subprocess here too, the parent never holds a CUDA context
            # at all. Activation extraction takes ~2s + model load on a
            # 5090 — minor cost vs. the cold-start hang it prevents.
            if not cfg.activations_path.exists():
                print(f"  [delphi-gate inventory] activations.pt not "
                      f"found at {cfg.activations_path}; extracting "
                      f"in subprocess so vLLM cold-start stays clean.")
                _extract_activations_subprocess(cfg)

            # Run the Delphi gate in a subprocess.
            kept_descs, score_log = _run_delphi_gate_subprocess(
                mode="inventory",
                payload={
                    "descriptions": descriptions,
                    "top_activations": top_acts,
                    "activations_path": str(cfg.activations_path),
                    "tokens_path": str(cfg.tokens_path),
                    "sae_release": cfg.sae_release,
                    "sae_id": cfg.sae_id,
                },
                cfg=cfg,
            )

            # v8.18.15 visibility: surface subprocess skip mode in the parent
            # log. Pre-fix the subprocess could fail-open (Delphi unavailable,
            # missing API key) and the parent silently passed all
            # descriptions through with no log message — user couldn't tell
            # the gate had been bypassed.
            if isinstance(score_log, dict):
                gate_mode = score_log.get("_gate_mode")
                if gate_mode == "skipped":
                    skip_reason = score_log.get("_skip_reason", "unknown")
                    print(f"  [delphi-gate inventory] SUBPROCESS RAN BUT "
                          f"GATE SKIPPED — reason={skip_reason}. All "
                          f"{len(descriptions)} descriptions passed "
                          f"through unfiltered. To fix, ensure Delphi is "
                          f"importable: pip install -e ./delphi-eleutherai/")

            cfg.output_dir.joinpath("delphi_scores_inventory.json").write_text(
                json.dumps({
                    "threshold": getattr(cfg, "delphi_score_threshold", 0.7),
                    "n_in":   len(descriptions),
                    "n_kept": len(kept_descs),
                    "scores": score_log,
                }, indent=2)
            )
            # Replace the description set with the gated subset.
            descriptions = kept_descs
        except Exception as e:
            print(f"  [delphi-gate inventory] error "
                  f"({type(e).__name__}: {e}); skipping gate.")

    # Free model only — keep SAE for the post-organize_hierarchy gate.
    # Model isn't needed for organize_hierarchy or the second Delphi pass;
    # SAE is. We free SAE after the second gate.
    try:
        del model
    except NameError:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Organize into hierarchy
    catalog = organize_hierarchy(descriptions, cfg)

    # 5b. (v8.18 audit fix #1, refactored to subprocess in v8.18.10)
    # Delphi gate on FINAL leaves. Same isolation rationale as gate 4b —
    # runs in a subprocess so Delphi's imports never enter this process.
    if getattr(cfg, "delphi_gate_in_inventory", True):
        try:
            if cfg.activations_path.exists() and cfg.tokens_path.exists():
                catalog, organized_score_log = _run_delphi_gate_subprocess(
                    mode="organized",
                    payload={
                        "catalog": catalog,
                        "top_activations": top_acts,
                        "activations_path": str(cfg.activations_path),
                        "tokens_path": str(cfg.tokens_path),
                        "sae_release": cfg.sae_release,
                        "sae_id": cfg.sae_id,
                    },
                    cfg=cfg,
                )
                if isinstance(organized_score_log, dict):
                    gate_mode = organized_score_log.get("_gate_mode")
                    if gate_mode == "skipped":
                        skip_reason = organized_score_log.get("_skip_reason", "unknown")
                        print(f"  [delphi-gate organized] SUBPROCESS RAN BUT "
                              f"GATE SKIPPED — reason={skip_reason}. Catalog "
                              f"unfiltered. Fix: pip install -e ./delphi-eleutherai/")
                cfg.output_dir.joinpath("delphi_scores_organized.json").write_text(
                    json.dumps(organized_score_log, indent=2)
                )
            else:
                print(f"  [delphi-gate organized] activations.pt/tokens.pt "
                      f"not yet cached; second gate skipped.")
        except Exception as e:
            print(f"  [delphi-gate organized] error "
                  f"({type(e).__name__}: {e}); skipping gate.")

    # Now safe to free SAE — both gates are done.
    try:
        del sae
    except NameError:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # v8.18.20: catalog quality gates. Strict validator + quarantine
    # (per user's design: lexical flags trigger LLM crispness, not
    # auto-reject). Default mode "quarantine" drops only hard-fail
    # leaves (operationally-undefined phrases, missing source_latents,
    # LLM rejects); soft-flagged ones are kept with a written report.
    gate_mode = getattr(cfg, "catalog_gate_mode", "quarantine")
    if gate_mode != "off":
        try:
            from .catalog_quality import (
                apply_catalog_gates, write_quality_report,
            )
            print(f"\n  [catalog-quality] applying gates (mode={gate_mode})...")
            catalog, quality_records = apply_catalog_gates(
                catalog, cfg=cfg, mode=gate_mode,
                use_llm_crispness=getattr(cfg, "catalog_gate_use_llm", True),
            )
            write_quality_report(
                quality_records,
                cfg.output_dir / "catalog_quality_report.json",
                mode=gate_mode,
            )
        except Exception as e:
            print(f"  [catalog-quality] error "
                  f"({type(e).__name__}: {e}); skipping gate.")

    cfg.catalog_path.write_text(json.dumps(catalog, indent=2))
    print(f"Saved feature catalog: {cfg.catalog_path}")

    return catalog


if __name__ == "__main__":
    run()
