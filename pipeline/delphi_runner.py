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

Reads its config from env vars set by the parent process. Calls
delphi's process_cache directly with our custom latent_range, so we
explain only the shortlisted indices instead of [0, max_latents).
"""
import asyncio
import json
import logging
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from delphi import logger
from delphi.config import (
    RunConfig, ConstructorConfig, SamplerConfig, CacheConfig,
)
from delphi.__main__ import (
    load_artifacts, populate_cache, process_cache, non_redundant_hookpoints,
)
from delphi.utils import assert_type

logger.setLevel(logging.INFO)

LATENT_INDICES = json.loads(os.environ["DELPHI_LATENT_INDICES"])
HOOKPOINT = os.environ["DELPHI_HOOKPOINT"]
MODEL = os.environ["DELPHI_MODEL"]
SPARSE_MODEL = os.environ["DELPHI_SPARSE_MODEL"]
EXPLAINER_MODEL = os.environ["DELPHI_EXPLAINER_MODEL"]
EXPLAINER_PROVIDER = os.environ["DELPHI_EXPLAINER_PROVIDER"]
SCORERS = json.loads(os.environ["DELPHI_SCORERS"])
N_TOKENS = int(os.environ["DELPHI_N_TOKENS"])
RUN_DIR = os.environ["DELPHI_RUN_DIR"]


async def main():
    run_cfg = RunConfig(
        cache_cfg=CacheConfig(n_tokens=N_TOKENS, batch_size=8, n_splits=5),
        constructor_cfg=ConstructorConfig(),
        sampler_cfg=SamplerConfig(),
        model=MODEL,
        sparse_model=SPARSE_MODEL,
        hookpoints=[HOOKPOINT],
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
    hookpoints, hookpoint_to_sparse_encode, model, transcode = load_artifacts(run_cfg)
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

    env = os.environ.copy()
    env.update({
        "DELPHI_LATENT_INDICES": json.dumps(delphi_indices),
        "DELPHI_HOOKPOINT": cfg.hook_point,
        "DELPHI_MODEL": cfg.model_name,
        "DELPHI_SPARSE_MODEL": cfg.sae_release,
        "DELPHI_EXPLAINER_MODEL": cfg.delphi_explainer_model,
        "DELPHI_EXPLAINER_PROVIDER": cfg.delphi_explainer_provider,
        "DELPHI_SCORERS": json.dumps([cfg.delphi_scorer]),
        "DELPHI_N_TOKENS": str(cfg.shortlist_calibration_tokens),
        "DELPHI_RUN_DIR": str(run_dir),
    })

    print(f"  invoker:        {script_path}")
    print(f"  model:          {cfg.model_name}")
    print(f"  sparse_model:   {cfg.sae_release}")
    print(f"  hookpoint:      {cfg.hook_point}")
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

    For our hookpoint blocks.9.hook_resid_pre, files look like:
        blocks.9.hook_resid_pre_latent42.txt
    """
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
        path = explanations_path / f"{cfg.hook_point}_latent{lat_idx}.txt"
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
    out_path = cfg.output_dir / "delphi_catalog.json"
    out_path.write_text(json.dumps(catalog, indent=2))

    raw_path = cfg.output_dir / "delphi_descriptions.json"
    raw_path.write_text(
        json.dumps({str(k): v for k, v in descriptions.items()}, indent=2)
    )

    print(f"  Saved: {out_path}")
    print(f"  Saved: {raw_path}")
    return catalog
