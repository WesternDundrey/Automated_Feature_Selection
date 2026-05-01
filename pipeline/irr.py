"""
Inter-rater reliability per catalog (v8.19.0).

Per-catalog F1 ceiling estimator. For each of the two catalogs (Opus
and Delphi), randomly samples `cfg.irr_sample_size` features and
re-annotates the same corpus tokens with two independent annotator
seeds. Reports Cohen's kappa + agreement-F1 per catalog.

Why this matters: the boundary-discipline contract makes Opus catalog
descriptions much crisper than free-form Delphi descriptions. If the
annotator IRR is 0.583 on Opus but 0.40 on Delphi, the supSAE-vs-unsup
F1 comparison's headline gap could be entirely a label-noise effect.
The IRR ceiling lets the paper report F1 numbers in the form
"Δ = +X above respective IRR ceilings" instead of raw absolute F1.

This step does NOT need to run after the full annotation — it's a
small-sample validity check. Cost: ~$5 in API tokens or 30 min on the
local annotator subprocess.

Run with: python -m pipeline.run --step irr
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import Config


def _load_catalog(path: Path) -> list[dict]:
    if not path.exists():
        return []
    cat = json.loads(path.read_text())
    return [f for f in cat.get("features", []) if f.get("type") == "leaf"]


def _bootstrap_kappa(a: np.ndarray, b: np.ndarray, n_boot: int = 200) -> tuple[float, float]:
    """Bootstrap-resampled Cohen's kappa point + std error."""
    rng = np.random.RandomState(42)
    ks = []
    n = len(a)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        ks.append(_cohens_kappa(a[idx], b[idx]))
    return float(np.mean(ks)), float(np.std(ks))


def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary annotators on the same items."""
    p_obs = float((a == b).mean())
    p_a = float(a.mean())
    p_b = float(b.mean())
    p_exp = p_a * p_b + (1 - p_a) * (1 - p_b)
    if p_exp >= 1.0 - 1e-9:
        return 0.0
    return (p_obs - p_exp) / (1 - p_exp)


def _agreement_f1(a: np.ndarray, b: np.ndarray) -> float:
    """F1 of pass-A predictions against pass-B labels (treats B as 'truth').

    Symmetric in this binary setting up to definition of positive class;
    this is the most natural single-number F1 estimate of the IRR ceiling.
    """
    tp = int(((a == 1) & (b == 1)).sum())
    fp = int(((a == 1) & (b == 0)).sum())
    fn = int(((a == 0) & (b == 1)).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def _run_annotator_pass(
    cfg: Config, tokens: torch.Tensor, features: list[dict], seed: int
) -> torch.Tensor:
    """Run one annotation pass at a given seed. Returns (N, T, F) labels."""
    pass_cfg = copy.copy(cfg)
    pass_cfg.seed = seed
    pass_cfg.output_dir = cfg.output_dir / f"_irr_pass_seed{seed}"
    pass_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, pass_cfg.tokens_path)
    catalog = {"features": features}
    pass_cfg.catalog_path.write_text(json.dumps(catalog))

    from .annotate import annotate_local
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    annotations = annotate_local(tokens, features, tok, pass_cfg)
    return annotations


def run(cfg: Config = None) -> dict:
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print(f"IRR  (sample_size={cfg.irr_sample_size} per catalog, "
          f"two annotation seeds)")
    print("=" * 70)

    opus_path = cfg.output_dir / "opus_catalog.json"
    delphi_path = cfg.output_dir / "delphi_catalog.json"
    opus_leaves = _load_catalog(opus_path)
    delphi_leaves = _load_catalog(delphi_path)
    if not opus_leaves and not delphi_leaves:
        raise FileNotFoundError(
            f"Need at least one of {opus_path}, {delphi_path}. Run "
            f"--step opus-catalog and/or --step delphi-run first."
        )

    if not cfg.tokens_path.exists():
        raise FileNotFoundError(
            f"Need {cfg.tokens_path}. Run --step annotate first to "
            f"materialize tokens (or run --step pilot)."
        )
    full_tokens = torch.load(cfg.tokens_path, weights_only=True)
    irr_n_seqs = min(cfg.agreement_n_sequences, full_tokens.shape[0])
    sub_tokens = full_tokens[:irr_n_seqs]

    rng = random.Random(cfg.seed)
    results: dict[str, dict] = {}

    for arm_name, leaves in (("opus", opus_leaves), ("delphi", delphi_leaves)):
        if not leaves:
            results[arm_name] = {"skipped": "catalog empty"}
            continue

        sample_n = min(cfg.irr_sample_size, len(leaves))
        sample = rng.sample(leaves, sample_n)
        print(f"\n  [{arm_name}] sampling {sample_n} of {len(leaves)} features")

        annot_a = _run_annotator_pass(cfg, sub_tokens, sample, seed=cfg.seed)
        annot_b = _run_annotator_pass(cfg, sub_tokens, sample, seed=cfg.seed + 1)
        a = annot_a.bool().numpy().reshape(-1, sample_n)
        b = annot_b.bool().numpy().reshape(-1, sample_n)

        per_feature = []
        for k in range(sample_n):
            ka = a[:, k].astype(np.int8)
            kb = b[:, k].astype(np.int8)
            kappa = _cohens_kappa(ka, kb)
            f1 = _agreement_f1(ka, kb)
            per_feature.append({
                "id": sample[k]["id"],
                "kappa": round(kappa, 4),
                "agreement_f1": round(f1, 4),
                "n_pos_a": int(ka.sum()),
                "n_pos_b": int(kb.sum()),
                "n_total": int(len(ka)),
            })

        kappas = np.array([r["kappa"] for r in per_feature])
        f1s = np.array([r["agreement_f1"] for r in per_feature])
        kappa_mean, kappa_se = _bootstrap_kappa(
            a.flatten().astype(np.int8), b.flatten().astype(np.int8)
        )
        results[arm_name] = {
            "n_features_sampled": sample_n,
            "n_total_in_catalog": len(leaves),
            "n_sequences": int(irr_n_seqs),
            "kappa_mean": float(kappas.mean()),
            "kappa_median": float(np.median(kappas)),
            "kappa_pooled_bootstrap": kappa_mean,
            "kappa_pooled_bootstrap_se": kappa_se,
            "agreement_f1_mean": float(f1s.mean()),
            "agreement_f1_median": float(np.median(f1s)),
            "per_feature": per_feature,
        }
        print(f"    kappa mean: {kappas.mean():.3f}  "
              f"agreement F1 mean: {f1s.mean():.3f}")

    out = {"sample_size": cfg.irr_sample_size, "arms": results}
    out_path = cfg.output_dir / "irr_report.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")

    print("\n  IRR-relative F1 ceiling reminder:")
    for arm_name, r in results.items():
        if "agreement_f1_mean" in r:
            print(f"    {arm_name:7s}  ceiling ≈ {r['agreement_f1_mean']:.3f}  "
                  f"(report Δ above this in paper)")

    return out
