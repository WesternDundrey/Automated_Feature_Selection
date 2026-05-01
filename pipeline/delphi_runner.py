"""
Delphi runner (v8.19.0) — invokes EleutherAI Delphi on a SUBSET of unsup
SAE latents (top `cfg.delphi_n_features` of the shortlist).

Native `python -m delphi` only supports `--max_latents N` which selects
the first N indices. We need SPECIFIC indices from latent_shortlist.json.
Solution: subprocess invocation of an inline Python script that imports
delphi internals (process_cache, populate_cache, load_artifacts) and
calls them with a custom `latent_range = torch.tensor(shortlist[:N])`.

The subprocess gets a clean Python state (avoids torch + transformer-
lens + delphi import order conflicts) and clean CUDA context.

Outputs Delphi's per-latent explanations as `{hookpoint}_latent{idx}.txt`
files at `pipeline_data/delphi_run/explanations/`. After Delphi
finishes, parses the .txt files and converts to a delphi-mode supSAE
catalog at `pipeline_data/delphi_catalog.json`. Delphi-mode features
are exempt from the boundary-discipline contract (descriptions are
free-form auto-interp, no positive/negative_examples available).

WARNING: Imports delphi's Python API. If you upgrade delphi/
beyond the API surface tested at v8.19.0 implementation time
(2026-05-01), this wrapper may need adjustment. Re-run after:
    cd delphi && git pull && uv pip install -e .
The pilot gate validates the wrapper end-to-end before the full run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .config import Config


# ── Inline subprocess script ─────────────────────────────────────────
# Run as `python <this_script>` in a clean process with delphi importable.
# Inputs come via env vars to avoid quoting hell with json on the CLI.

DELPHI_INVOKER_SCRIPT = '''\
"""Inline Delphi runner — supplied by pipeline/delphi_runner.py.

Reads its config from env vars set by the parent process. Bypasses
Delphi\'s `load_artifacts` (which routes non-Gemma SAEs through
`sparsify.SparseCoder.load_many` and would fail on `gpt2-small-res-jb`,
a sae_lens-format release). Instead loads gpt2 via transformers\'
AutoModel and gpt2-small-res-jb via our own pipeline.inventory.load_sae,
then constructs hookpoint_to_sparse_encode = {"h.8": sae.encode_dense}
where "h.8" (output of GPT2Block 8) equals TransformerLens\'
blocks.9.hook_resid_pre bit-perfectly (LN folding is a parameter-
equivalence transformation that preserves residual-stream output).

Calls Delphi\'s `process_cache` directly with our custom
`latent_range = torch.tensor(shortlist[:N])` so only the shortlisted
indices get explained (Delphi CLI\'s `--max_latents N` only supports
the FIRST N).
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from delphi import logger
from delphi.config import (
    RunConfig, ConstructorConfig, SamplerConfig, CacheConfig,
)
from delphi.__main__ import (
    populate_cache, process_cache, non_redundant_hookpoints,
)
from delphi.utils import assert_type

logger.setLevel(logging.INFO)

# Make supsae root importable so we can use pipeline.inventory.load_sae
SUPSAE_ROOT = os.environ["SUPSAE_ROOT"]
if SUPSAE_ROOT not in sys.path:
    sys.path.insert(0, SUPSAE_ROOT)

LATENT_INDICES = json.loads(os.environ["DELPHI_LATENT_INDICES"])
TRANSFORMERS_HOOKPOINT = os.environ["DELPHI_TRANSFORMERS_HOOKPOINT"]
MODEL = os.environ["DELPHI_MODEL"]
SPARSE_MODEL = os.environ["DELPHI_SPARSE_MODEL"]
EXPLAINER_MODEL = os.environ["DELPHI_EXPLAINER_MODEL"]
EXPLAINER_PROVIDER = os.environ["DELPHI_EXPLAINER_PROVIDER"]
SCORERS = json.loads(os.environ["DELPHI_SCORERS"])
N_TOKENS = int(os.environ["DELPHI_N_TOKENS"])
RUN_DIR = os.environ["DELPHI_RUN_DIR"]
SAE_ID = os.environ["DELPHI_SAE_ID"]


def _build_artifacts():
    """Load gpt2 + gpt2-small-res-jb without Delphi\'s sparsify loader.

    Returns (hookpoints, hookpoint_to_sparse_encode, model, transcode)
    matching Delphi\'s internal contract from `load_artifacts`.
    """
    from pipeline.config import Config as SupsaeConfig
    from pipeline.inventory import load_sae as supsae_load_sae

    # Build a minimal supsae Config; only needs the SAE-loading fields.
    sup_cfg = SupsaeConfig()
    sup_cfg.sae_release = SPARSE_MODEL
    sup_cfg.sae_id = SAE_ID

    sae_wrap, _sparsity = supsae_load_sae(sup_cfg)

    # Load transformers model (Delphi requires PreTrainedModel).
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        device_map = {"": "cuda"}
    else:
        dtype = torch.float32
        device_map = {"": "cpu"}
    print(f"[delphi-invoker] loading {MODEL} via transformers AutoModel", flush=True)
    model = AutoModel.from_pretrained(
        MODEL, torch_dtype=dtype, device_map=device_map,
    )

    # Verify the chosen hookpoint exists in the loaded model.
    module_names = {n for n, _ in model.named_modules()}
    if TRANSFORMERS_HOOKPOINT not in module_names:
        # Print a few candidates to help the user diagnose.
        sample = sorted(n for n in module_names if "h." in n and len(n) < 24)[:8]
        raise RuntimeError(
            f"hookpoint {TRANSFORMERS_HOOKPOINT!r} not found in "
            f"{MODEL}. Candidates (first 8): {sample}"
        )

    # Move SAE to the same device as the model.
    sae_wrap = sae_wrap.to(model.device)

    def encode_fn(x):
        """Closure called by Delphi\'s LatentCache.run on the residual
        stream activation at the hookpoint.

        Input  x: tensor of shape (B, T, d_model), the output of the
                  transformers GPT2Block at layer 8 = resid_pre of layer 9.
        Output:   (B, T, n_lat) dense ReLU/JumpReLU activations.

        sae_lens SAE.encode handles (B, T, d) → (B, T, n_lat) directly.
        For our PretrainedSAE wrapper, both the native sae_lens path and
        the bare-weights GemmaScope path return (B, T, n_lat).
        """
        return sae_wrap.encode(x)

    hookpoint_to_sparse_encode = {TRANSFORMERS_HOOKPOINT: encode_fn}
    return [TRANSFORMERS_HOOKPOINT], hookpoint_to_sparse_encode, model, False


async def main():
    run_cfg = RunConfig(
        cache_cfg=CacheConfig(n_tokens=N_TOKENS, batch_size=8, n_splits=5),
        constructor_cfg=ConstructorConfig(),
        sampler_cfg=SamplerConfig(),
        model=MODEL,
        sparse_model=SPARSE_MODEL,
        hookpoints=[TRANSFORMERS_HOOKPOINT],
        explainer_model=EXPLAINER_MODEL,
        explainer_provider=EXPLAINER_PROVIDER,
        scorers=SCORERS,
        # Cache must be wide enough to contain all our indices. We do not
        # use this to filter explanations — the latent_range below does that.
        max_latents=max(LATENT_INDICES) + 1,
        name="delphi_run",
        verbose=True,
        seed=42,
    )

    base_path = Path(RUN_DIR)
    base_path.mkdir(parents=True, exist_ok=True)
    latents_path = base_path / "latents"
    explanations_path = base_path / "explanations"
    scores_path = base_path / "scores"

    print(f"[delphi-invoker] caching activations to {latents_path}", flush=True)
    hookpoints, hookpoint_to_sparse_encode, model, transcode = _build_artifacts()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=run_cfg.hf_token)

    nrh = assert_type(dict, non_redundant_hookpoints(
        hookpoint_to_sparse_encode, latents_path, False
    ))
    if nrh:
        populate_cache(run_cfg, model, nrh, latents_path, tokenizer, transcode)

    del model, hookpoint_to_sparse_encode

    print(f"[delphi-invoker] explaining {len(LATENT_INDICES)} latents", flush=True)
    latent_range = torch.tensor(LATENT_INDICES)
    nrh = assert_type(list, non_redundant_hookpoints(
        hookpoints, scores_path, False
    ))
    if nrh:
        await process_cache(
            run_cfg, latents_path, explanations_path, scores_path,
            nrh, tokenizer, latent_range,
        )

    print(f"DELPHI_DONE: {explanations_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
'''


def run(cfg: Config = None) -> dict:
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print(f"DELPHI RUN  ({cfg.delphi_n_features} latents from shortlist)")
    print("=" * 70)

    # v8.19.5 resume: skip if a previous Delphi run already produced
    # descriptions. Delphi calls Sonnet/Opus per latent (~$10-25 + ~1
    # hour of activation caching); don't redo when downstream stages
    # failed.
    record_path = cfg.output_dir / "delphi_catalog.json"
    if record_path.exists() and not cfg.force:
        try:
            existing = json.loads(record_path.read_text())
            n_described = existing.get("n_latents_described", 0)
            if n_described >= max(1, cfg.delphi_n_features // 2):
                print(f"  [resume] {record_path} exists with "
                      f"{n_described}/{cfg.delphi_n_features} described; "
                      f"skipping Delphi subprocess. Pass --force to "
                      f"regenerate.")
                # Refresh canonical feature_catalog.json so the unsup
                # arm's --step annotate reads it.
                cfg.catalog_path.write_text(json.dumps(existing, indent=2))
                return existing
        except Exception as e:
            print(f"  [resume] couldn't validate {record_path} "
                  f"({e}); regenerating.")

    if "OPENROUTER_API_KEY" not in os.environ:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Delphi explainer needs it for "
            "OpenRouter calls. export OPENROUTER_API_KEY=... and retry."
        )

    from .shortlist_latents import load_shortlist
    shortlist = load_shortlist(cfg)
    delphi_indices = shortlist[: cfg.delphi_n_features]

    run_dir = Path(cfg.delphi_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    script_path = run_dir / "_delphi_invoker.py"
    script_path.write_text(DELPHI_INVOKER_SCRIPT)

    # Map our TransformerLens hookpoint to a transformers AutoModel module
    # path. For gpt2 (AutoModel), GPT2Block instances live at "h.0".."h.11".
    # blocks.{L}.hook_resid_pre = output of block (L-1) = "h.{L-1}".
    # blocks.{L}.hook_resid_post = output of block L = "h.{L}".
    # This identity holds bit-perfectly: TL's LN folding is parameter-
    # equivalent and preserves residual stream output.
    tl_hp = cfg.hook_point
    if tl_hp.startswith("blocks.") and tl_hp.endswith(".hook_resid_pre"):
        layer = int(tl_hp.split(".")[1])
        transformers_hookpoint = f"h.{layer - 1}"
    elif tl_hp.startswith("blocks.") and tl_hp.endswith(".hook_resid_post"):
        layer = int(tl_hp.split(".")[1])
        transformers_hookpoint = f"h.{layer}"
    else:
        raise RuntimeError(
            f"Don't know how to map TL hookpoint {tl_hp!r} to a "
            f"transformers AutoModel module path. Supported: "
            f"blocks.<L>.hook_resid_pre / blocks.<L>.hook_resid_post."
        )

    supsae_root = str(Path(__file__).resolve().parent.parent)

    env = os.environ.copy()
    env.update({
        "SUPSAE_ROOT": supsae_root,
        "DELPHI_LATENT_INDICES": json.dumps(delphi_indices),
        "DELPHI_TRANSFORMERS_HOOKPOINT": transformers_hookpoint,
        "DELPHI_MODEL": cfg.model_name,
        "DELPHI_SPARSE_MODEL": cfg.sae_release,
        "DELPHI_SAE_ID": cfg.sae_id,
        "DELPHI_EXPLAINER_MODEL": cfg.delphi_explainer_model,
        "DELPHI_EXPLAINER_PROVIDER": cfg.delphi_explainer_provider,
        "DELPHI_SCORERS": json.dumps([cfg.delphi_scorer]),
        "DELPHI_N_TOKENS": str(cfg.shortlist_calibration_tokens),
        "DELPHI_RUN_DIR": str(run_dir),
    })

    print(f"  invoker:        {script_path}")
    print(f"  model:          {cfg.model_name}")
    print(f"  sparse_model:   {cfg.sae_release}/{cfg.sae_id}")
    print(f"  TL hookpoint:   {cfg.hook_point}")
    print(f"  → transformers: {transformers_hookpoint}  "
          f"(bit-perfect via LN-folding equivalence)")
    print(f"  explainer:      {cfg.delphi_explainer_model} "
          f"via {cfg.delphi_explainer_provider}")
    print(f"  scorer:         {cfg.delphi_scorer}")
    print(f"  n latents:      {len(delphi_indices)}")
    print(f"  run_dir:        {run_dir}")
    print(f"\n  --- delphi subprocess output ---")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Delphi subprocess exited with code {result.returncode}. "
            f"Common fixes: (1) `cd delphi && git pull && "
            f"uv pip install -e .` to sync to upstream, "
            f"(2) verify OPENROUTER_API_KEY, "
            f"(3) inspect delphi.log for explainer-side failures."
        )

    return _extract_descriptions(cfg, run_dir, delphi_indices)


def _extract_descriptions(
    cfg: Config, run_dir: Path, latent_indices: list[int]
) -> dict:
    """Parse delphi explanation .txt files into a delphi-mode catalog.

    Filename pattern (from delphi/latents/latents.py:31):
        {module_name}_latent{latent_index}.txt

    Our invoker uses transformers-style hookpoint (e.g. `h.8`); files
    look like `h.8_latent42.txt`. We compute the same TL→transformers
    mapping here that the invoker uses, so callers see consistent
    addressing.
    """
    tl_hp = cfg.hook_point
    if tl_hp.endswith(".hook_resid_pre"):
        layer = int(tl_hp.split(".")[1])
        transformers_hp = f"h.{layer - 1}"
    elif tl_hp.endswith(".hook_resid_post"):
        layer = int(tl_hp.split(".")[1])
        transformers_hp = f"h.{layer}"
    else:
        transformers_hp = tl_hp  # fall through; let path-existence check fail

    explanations_path = run_dir / "explanations"
    if not explanations_path.exists():
        raise FileNotFoundError(
            f"No explanations at {explanations_path}. Delphi may have "
            f"failed to write any descriptions; check the subprocess "
            f"output above and {run_dir}/delphi.log."
        )

    descriptions: dict[int, str] = {}
    missing: list[int] = []
    for lat_idx in latent_indices:
        path = explanations_path / f"{transformers_hp}_latent{lat_idx}.txt"
        if path.exists():
            try:
                raw = path.read_bytes()
                desc = json.loads(raw)
                descriptions[lat_idx] = (
                    desc if isinstance(desc, str) else str(desc)
                )
            except Exception as e:
                print(f"  Skipping {path.name}: parse error: {e}")
                missing.append(lat_idx)
        else:
            missing.append(lat_idx)

    print(f"\n  Parsed {len(descriptions)}/{len(latent_indices)} "
          f"delphi descriptions ({len(missing)} missing)")
    if missing[:5]:
        print(f"  First missing: {missing[:5]}")

    features = []
    features.append({
        "id": "delphi",
        "description": "Delphi-described unsup SAE latents (auto-interp baseline)",
        "type": "group",
        "parent": None,
    })
    for lat_idx, desc in descriptions.items():
        features.append({
            "id": f"delphi.latent_{lat_idx}",
            "description": desc,
            "type": "leaf",
            "parent": "delphi",
            "source_latents": [int(lat_idx)],
            "source": "delphi",
            "delphi_mode": True,
        })

    catalog = {
        "features": features,
        "source": "delphi",
        "n_latents_described": len(descriptions),
        "n_latents_requested": len(latent_indices),
        "explainer_model": cfg.delphi_explainer_model,
        "scorer": cfg.delphi_scorer,
    }
    # v8.19.2 two-arm flow: write the named record (audit trail) AND
    # the canonical feature_catalog.json. The user runs the unsup arm
    # in its own pipeline_data_unsup/ output dir, so this canonical
    # name is arm-local; --step annotate picks it up directly.
    record_path = cfg.output_dir / "delphi_catalog.json"
    record_path.write_text(json.dumps(catalog, indent=2))
    cfg.catalog_path.write_text(json.dumps(catalog, indent=2))

    raw_path = cfg.output_dir / "delphi_descriptions.json"
    raw_path.write_text(
        json.dumps({str(k): v for k, v in descriptions.items()}, indent=2)
    )

    print(f"  Saved: {record_path}")
    print(f"  Saved: {cfg.catalog_path} (canonical, picked up by --step annotate)")
    print(f"  Saved: {raw_path}")
    return catalog
