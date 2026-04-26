"""
Subprocess entry point for Delphi gates.

Why this exists (v8.18.10): when the inventory step imports the Delphi
graph (faiss, sentence_transformers, bitsandbytes, possibly vllm
itself) into the main pipeline process, those imports leave residual
global state — sys.modules entries, CUDA context, env-var side effects
from bitsandbytes, etc. The next time the same Python process tries
to start vLLM (during --step annotate when the user runs the full
pipeline in one invocation), the EngineCore subprocess inherits that
polluted state and hangs at parallel_state init.

Running the Delphi gate in a fresh subprocess fixes it: this module
gets `python -m pipeline.delphi_subprocess <mode>` invocations, does
all the heavy Delphi work, writes results to disk, exits. The parent
pipeline process never imports delphi at all — only reads the result
JSON the subprocess wrote.

Inputs and outputs are passed via JSON files (paths in argv) rather
than stdin/stdout to keep the subprocess invocation simple and to
allow large catalogs without buffering issues.

Usage (called by pipeline/inventory.py, not by the user):
    python -m pipeline.delphi_subprocess inventory \\
        --input  /tmp/delphi_inventory_in.json \\
        --output /tmp/delphi_inventory_out.json \\
        --cfg    /tmp/cfg.json

    python -m pipeline.delphi_subprocess organized \\
        --input  /tmp/delphi_organized_in.json \\
        --output /tmp/delphi_organized_out.json \\
        --cfg    /tmp/cfg.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _make_cfg_from_dict(cfg_dict: dict):
    """Build a minimal Config-shaped object from a JSON dict. We can't
    pickle/unpickle the full Config dataclass cleanly across Python
    interpreters, but the gate only needs a handful of attributes."""
    class _CfgShim:
        pass
    cfg = _CfgShim()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def _run_inventory_gate(input_path: Path, output_path: Path, cfg_dict: dict) -> int:
    """Inventory-time gate: filter raw descriptions before catalog."""
    payload = json.loads(input_path.read_text())
    descriptions = payload["descriptions"]
    top_activations = payload["top_activations"]
    activations_path = payload["activations_path"]
    tokens_path = payload["tokens_path"]
    sae_release = payload["sae_release"]
    sae_id = payload["sae_id"]

    cfg = _make_cfg_from_dict(cfg_dict)

    # Inside this subprocess only — these imports stay isolated.
    from .delphi_score import gate_inventory_descriptions, _bootstrap_delphi
    from .inventory import load_sae as _load_pretrained_sae

    if not _bootstrap_delphi():
        # Fail-open passthrough.
        Path(output_path).write_text(json.dumps({
            "kept_descriptions": descriptions,
            "score_log": {"_gate_mode": "skipped",
                          "_skip_reason": "delphi_unavailable_in_subprocess"},
        }))
        return 0

    # Load resources the gate needs.
    activations = torch.load(activations_path, weights_only=True)
    tokens = torch.load(tokens_path, weights_only=True)
    cfg.sae_release = sae_release
    cfg.sae_id = sae_id

    # Lazy SAE load — match inventory's load_sae signature.
    sae, _sparsity = _load_pretrained_sae(cfg)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(getattr(cfg, "model_name", "gpt2"))

    kept, score_log = gate_inventory_descriptions(
        sae=sae,
        activations=activations,
        tokens=tokens,
        descriptions=descriptions,
        top_activations=top_activations,
        tokenizer=tokenizer,
        cfg=cfg,
        threshold=getattr(cfg, "delphi_score_threshold", 0.7),
    )
    Path(output_path).write_text(json.dumps({
        "kept_descriptions": kept,
        "score_log": score_log,
    }))
    return 0


def _run_organized_gate(input_path: Path, output_path: Path, cfg_dict: dict) -> int:
    """Post-organize_hierarchy gate: filter final catalog leaves."""
    payload = json.loads(input_path.read_text())
    catalog = payload["catalog"]
    top_activations = payload["top_activations"]
    activations_path = payload["activations_path"]
    tokens_path = payload["tokens_path"]
    sae_release = payload["sae_release"]
    sae_id = payload["sae_id"]

    cfg = _make_cfg_from_dict(cfg_dict)

    from .delphi_score import gate_organized_leaves, _bootstrap_delphi
    from .inventory import load_sae as _load_pretrained_sae

    if not _bootstrap_delphi():
        Path(output_path).write_text(json.dumps({
            "filtered_catalog": catalog,
            "score_log": {"_gate_mode": "skipped",
                          "_skip_reason": "delphi_unavailable_in_subprocess"},
        }))
        return 0

    activations = torch.load(activations_path, weights_only=True)
    tokens = torch.load(tokens_path, weights_only=True)
    cfg.sae_release = sae_release
    cfg.sae_id = sae_id

    sae, _sparsity = _load_pretrained_sae(cfg)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(getattr(cfg, "model_name", "gpt2"))

    filtered, score_log = gate_organized_leaves(
        catalog=catalog,
        sae=sae,
        activations=activations,
        tokens=tokens,
        top_activations=top_activations,
        tokenizer=tokenizer,
        cfg=cfg,
        threshold=getattr(cfg, "delphi_score_threshold", 0.7),
    )
    Path(output_path).write_text(json.dumps({
        "filtered_catalog": filtered,
        "score_log": score_log,
    }))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delphi gate subprocess runner. Called by pipeline/inventory.py"
    )
    parser.add_argument("mode", choices=["inventory", "organized"])
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cfg",    required=True)
    args = parser.parse_args()

    cfg_dict = json.loads(Path(args.cfg).read_text())

    try:
        if args.mode == "inventory":
            return _run_inventory_gate(
                Path(args.input), Path(args.output), cfg_dict,
            )
        else:
            return _run_organized_gate(
                Path(args.input), Path(args.output), cfg_dict,
            )
    except Exception as e:
        # Write a sentinel so the parent doesn't hang waiting for a file.
        Path(args.output).write_text(json.dumps({
            "_subprocess_error": f"{type(e).__name__}: {e}",
        }))
        # Re-raise so the subprocess exits non-zero and the parent can
        # surface the failure cleanly.
        raise


if __name__ == "__main__":
    sys.exit(main())
