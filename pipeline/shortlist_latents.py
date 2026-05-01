"""
Pre-Delphi/Opus latent shortlist (v8.19.0).

Picks `cfg.shortlist_size` candidate latent indices from the pretrained
unsup SAE (gpt2-small-res-jb, 24576 latents). The shortlist is the
SHARED input to both arms of the Delphi-vs-Opus comparison:

  • Delphi arm: describes top `cfg.delphi_n_features` of the shortlist 1:1.
  • Opus arm: inspects all `cfg.shortlist_size` latents and designs
    `cfg.opus_n_features` features with full design freedom.

CURRENT IMPLEMENTATION: frequency-window filter only. Drops dead
latents (freq < shortlist_freq_min) and ultra-dense latents (freq >
shortlist_freq_max, Engels project-out territory). Within the window,
ranks by DESCENDING firing rate — most-active first, on the assumption
that more-firing latents have more stable top-context statistics for
both arms.

LIMITATION: a true "activation concentration" score (kurtosis, top-K
mean / median, or entropy of the top-1000 contexts) would better
filter polysemantic-noise latents whose top contexts look random.
That requires a forward pass we don't currently perform here. Could
be added by piggy-backing on inventory.collect_top_activations after
shortlist runs (compute concentration over the candidate set, re-rank,
re-save). Treated as follow-up; the frequency-rank shortlist is
adequate for the Delphi-vs-Opus headline experiment.

Output: pipeline_data/latent_shortlist.json — read by both arms.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import Config
from .inventory import load_sae


def run(cfg: Config = None) -> list[int]:
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print(f"LATENT SHORTLIST  ({cfg.shortlist_size} candidates from "
          f"{cfg.sae_release}/{cfg.sae_id})")
    print("=" * 70)

    sae, sparsity = load_sae(cfg)
    if sparsity is None:
        raise RuntimeError(
            f"No sparsity info for {cfg.sae_release}; shortlist needs "
            f"firing-rate stats. gpt2-small-res-jb provides this; if you "
            f"swapped to a different SAE, run `shortlist_latents` after "
            f"computing per-latent firing rate from a corpus pass."
        )

    firing_rate = sparsity.exp()
    n_latents = firing_rate.shape[0]

    in_window = (
        (firing_rate >= cfg.shortlist_freq_min)
        & (firing_rate <= cfg.shortlist_freq_max)
    )
    candidates = in_window.nonzero(as_tuple=False).squeeze(-1)
    n_in_window = candidates.numel()

    if n_in_window < cfg.shortlist_size:
        print(f"  WARNING: only {n_in_window} latents in window "
              f"[{cfg.shortlist_freq_min}, {cfg.shortlist_freq_max}] "
              f"but shortlist_size={cfg.shortlist_size}. Using all "
              f"{n_in_window} candidates and capping shortlist.")
        cfg_shortlist_size = n_in_window
    else:
        cfg_shortlist_size = cfg.shortlist_size

    rates_in_window = firing_rate[candidates]
    order = rates_in_window.argsort(descending=True)
    sorted_idx = candidates[order]
    shortlist = sorted_idx[:cfg_shortlist_size].tolist()
    rates = firing_rate[shortlist].cpu().tolist()

    n_dead = int((firing_rate < cfg.shortlist_freq_min).sum())
    n_dense = int((firing_rate > cfg.shortlist_freq_max).sum())

    print(f"  Total latents in SAE:         {n_latents}")
    print(f"  Dead (freq < {cfg.shortlist_freq_min}):     {n_dead}")
    print(f"  Dense (freq > {cfg.shortlist_freq_max}):       {n_dense}  "
          f"← project-out territory (Engels)")
    print(f"  In window:                    {n_in_window}")
    print(f"  Shortlist size:               {len(shortlist)}")
    print(f"  Firing rate range in shortlist: "
          f"[{min(rates):.5f}, {max(rates):.5f}]")

    cfg.output_dir.mkdir(exist_ok=True)
    out = {
        "shortlist_size": len(shortlist),
        "sae_release": cfg.sae_release,
        "sae_id": cfg.sae_id,
        "freq_min": cfg.shortlist_freq_min,
        "freq_max": cfg.shortlist_freq_max,
        "n_total_latents": int(n_latents),
        "n_dead": n_dead,
        "n_dense_engels": n_dense,
        "n_in_window": int(n_in_window),
        "latent_indices": shortlist,
        "firing_rates": rates,
    }
    out_path = cfg.output_dir / "latent_shortlist.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")
    return shortlist


def load_shortlist(cfg: Config) -> list[int]:
    """Read the previously-computed shortlist; raises if missing."""
    path = cfg.output_dir / "latent_shortlist.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No shortlist at {path}. Run `--step shortlist` first."
        )
    return json.loads(path.read_text())["latent_indices"]
