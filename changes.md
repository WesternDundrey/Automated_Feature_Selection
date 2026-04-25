# Supervised SAE — Change Log

> Rigorous, objective audit trail. Updated on every material change to `supsae/`.

---

## [v8.11.2] — Revert default supervision_mode back to "hybrid"

**Date:** 2026-04-25

### What happened

v8.11 flipped the default `supervision_mode` from `"hybrid"` (frozen-decoder, summary6/7 numbers) to `"hinge"` (free-decoder, mentor's methodology note). User ran the full end-to-end pipeline under the new default on a fresh vast.ai instance. Result: R²=0.696 (vs 0.97 under hybrid), cal_F1=0.583 (vs 0.612), mean cosine to target = 0.27 (vs 1.0 by construction), supervised-only R² = −0.76 (vs +0.83). Pretrained-SAE gap grew from −0.014 to −0.148 R². The linear probe now beats the SAE on calibrated F1 (0.646 vs 0.583).

This is a meaningful regression on every downstream metric the paper cares about. Flipping the default without A/B validation was wrong. The legacy hybrid pipeline is what's been validated; it stays as the default.

### Fix

`cfg.supervision_mode` default: `"hinge"` → `"hybrid"`. The three new hinge-family modes (`hinge`, `hinge_jumprelu`, `gated_bce`) remain available as opt-in via `--supervision <mode>`. Their implementation (v8.11) and subsequent fixes (v8.11.1) stay in tree so a disciplined A/B comparison can be run later.

### Why hinge underperformed — current hypotheses

(Speculative — needs investigation, not paper-worthy claims.)

1. **Under-trained.** 15 epochs from random-init decoder is harder than 15 epochs with decoder pre-locked to target_dirs. The hybrid path gets "where the decoder should point" for free; hinge has to learn it.
2. **lambda_sup miscalibrated.** `lambda_sup=2.0` was tuned for BCE loss magnitudes. Hinge and BCE have different scales; a sweep is needed before claiming hinge works.
3. **Data regime the doc didn't assume.** Mentor's doc says ReLU+hinge works "as long as ground-truth features have reasonable magnitudes relative to noise." Our catalog has features with n_pos ranging 45-11,819 — signal varies by >2 orders of magnitude. Frozen decoder absorbs this heterogeneity by not needing per-feature signal to learn direction; free decoder has to.

### If you want to continue the hinge experiment

Start from the validated frozen-decoder checkpoint and fine-tune with hinge, rather than training from scratch:

```
# 1. Train frozen-hybrid baseline.
python -m pipeline.run \
--step train \
--supervision hybrid \
--layer 9 \
--sae_id blocks.9.hook_resid_pre \
--local-annotator \
--n_sequences 1000 \
--epochs 15

# 2. (Future work — not implemented) warm-start hinge from that checkpoint
#    with decoder unfrozen, short fine-tune. Preserves direction alignment
#    from the frozen starting point while testing hinge's selectivity
#    behavior. This is probably the correct pragmatic A/B.
```

### Files changed

| File | Change |
|------|--------|
| `pipeline/config.py` | `supervision_mode` default flipped `"hinge"` → `"hybrid"`; docstring explains the trade |
| `changes.md` | This entry |

---

## [v8.11.1] — Reviewer fixes on the hinge/free-decoder batch

**Date:** 2026-04-25

Six issues from reviewer on the v8.11 drop, all addressed in this patch.

### HIGH — decoder-vs-target diagnostics now run regardless of supervision mode

`evaluate.py` gated its cosine / FVE / magnitude-correlation block on `model_cfg["use_mse_supervision"]`, which hinge checkpoints don't save. The central v8.11 trade ("free decoder learns cos ≈ X instead of frozen cos = 1.0") was therefore invisible in the logs. Fix: gate on the existence of `target_directions.pt` instead — which the hinge trainer writes post-hoc — and report `supervision_mode` alongside so readers know whether cos = 1.0 is by construction (frozen modes) or a learned value (hinge-family modes). The magnitude-correlation block is unblocked the same way.

### HIGH — `gated_bce` now compatible with promote-loop

`GatedBCESAE` has `gate_encoder`, `mag_encoder`, `unsup_encoder` — no single `.encoder` attribute. Promote-loop's four sites that read `sae.encoder.weight[n_supervised + u_local]` would have crashed. Fix: added a uniform API on all four SAE classes (HingeSAE, JumpReLUHingeSAE, GatedBCESAE, legacy SupervisedSAE):

```
sae.unsup_encoder_weight() -> (n_unsupervised, d_model)
sae.unsup_encoder_bias()   -> (n_unsupervised,)
```

Rows are indexed by `u_local` (not `u_global`). Each SAE class returns the right tensor for its architecture. All four promote-loop sites (proposal scan, mini-prefilter, mean-shift direction computation, PretrainedSAE wrapping for `collect_top_activations`) now use this API. Gated/JumpReLU/Hinge all run through the same code path.

### MEDIUM-HIGH — `weight_decay=0` on the hinge trainer's AdamW

Default AdamW weight_decay is 0.01. That imposes L2 shrinkage on encoder and decoder weights — a different shrinkage signal than what the hinge-vs-BCE argument rules out, but contaminates the comparison. Fixed to `weight_decay=0.0` explicitly so hinge's "no shrinkage bias" motivation actually holds.

### MEDIUM — `hinge_jumprelu` θ parameterization (no unconstrained negatives)

Previous θ was a plain `nn.Parameter`. Hinge's loss can drive θ arbitrarily, including negative — and a negative θ would let negative `sup_pre` pass the `sup_pre > θ` gate, yielding negative `sup_acts` (SAE latents are supposed to be nonnegative). Fixed two ways:

1. `θ` is now parameterized as `softplus(theta_raw)` where `theta_raw` is the nn.Parameter. θ is always ≥ 0 by construction. Gradient from the hinge term flows through softplus to `theta_raw` without any clamping hacks or detach tricks.
2. Magnitude is now `F.relu(sup_pre) * gate` (not `sup_pre * gate`), belt-and-suspenders: even if anything slips past the softplus, the output is never negative.

θ is exposed as a `@property` on `JumpReLUHingeSAE` so callers (training loop, hinge loss helper) uniformly see the softplussed value.

### MEDIUM — grad clipping + batched validation in hinge trainer

Legacy trainer had `torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)`; hinge trainer didn't. With `pos_weight` capped at 100 for rare features, encoder gradients can blow up. Added clipping at `max_norm=1.0` to match the legacy path.

Validation was moving the full test tensor to GPU in one shot (`sae(x_test_dev)`). Fine for 1k sequences; OOM risk for larger corpora. Batched val over `cfg.batch_size`-sized chunks with a running sum; same R² + val_sup numbers, no VRAM spike.

### LOW-MEDIUM — scheduler math + val baseline_mse

Scheduler `T_max` was `epochs * (N // batch_size)` (floor), but the training loop iterates with ceil semantics (the final partial batch is included). Fixed to `math.ceil(N / batch_size)` so the cosine schedule actually covers all optimizer steps.

Val R² was computed against the train-split baseline_mse. Fixed to use a test-split `test_baseline_mse = F.mse_loss(x_test.mean(0)..., x_test)` so val R² compares apples-to-apples with `evaluate.py`'s test-side R² line.

### Files changed

| File | Change |
|------|--------|
| `pipeline/evaluate.py` | Decoder-vs-target + magnitude-correlation diagnostics gated on target_directions.pt, not use_mse_supervision |
| `pipeline/supervised_hinge.py` | softplus-parameterized θ in JumpReLU; `weight_decay=0`; grad clipping; batched val; ceil scheduler math; `unsup_encoder_*` helpers on all three classes |
| `pipeline/train.py` | `unsup_encoder_*` helpers on legacy `SupervisedSAE` for API parity |
| `pipeline/promote_loop.py` | Four call sites use `sae.unsup_encoder_weight()` / `sae.unsup_encoder_bias()` — Gated/JumpReLU/Hinge now all work end-to-end |
| `changes.md` | This entry |

---

## [v8.11] — Hinge / JumpReLU / Gated-BCE supervision modes (per mentor's methodology note)

**Date:** 2026-04-25

### Motivation

Mentor's `supervised_saes_hinge_loss.md` lays out three principled supervised-SAE formulations that train end-to-end (encoder + decoder jointly) without the frozen-decoder workaround. The core insight: hinge loss on pre-activations gives zero gradient when a feature is correctly gated, which is exactly what you want — MSE reconstruction then shapes magnitude alone, no shrinkage bias, no straight-through estimators, no target_dir pre-computation, no validation-threshold hacks. The legacy frozen-decoder path (summary6, summary7) gave the cos = 1.000 interpretability guarantee but at the cost of a more complex training pipeline; the mentor's trade is elegance for that guarantee.

User chose to adopt this as the default. The frozen-decoder modes stay available for reproducing prior numbers; new runs default to hinge.

### Additions

**`pipeline/supervised_hinge.py` (NEW).** Three SAE classes + three loss functions + training loop for the end-to-end modes:

- `HingeSAE` — ReLU encoder, no frozen decoder, per-step unit-norm column normalization on the full decoder. Hinge loss on the pre-activation of each supervised latent; sparsity (L1) applies only to the unsupervised slice.
- `JumpReLUHingeSAE` — identical to HingeSAE plus a per-feature learnable threshold `θ_i`. Forward: `f_i = z_i · H(z_i - θ_i)`. The Heaviside gate is non-differentiable but `θ` gets its gradient from the hinge term (which is piecewise-linear in `θ` with gradient ±1 on the active side, 0 on satisfied). MSE contributes zero gradient to `θ` (autograd drops the Dirac) — this is what the doc means by "no STE needed for JumpReLU under supervised training."
- `GatedBCESAE` — two encoder paths (`gate_encoder`, `mag_encoder` or tied-via-scale variant). Forward: `f_i = H(π_i) · ReLU(m_i)`. BCE on `σ(π)` supervises the gate; MSE supervises magnitude. Optional weight tying `W_mag = exp(r) · W_gate` (per-feature scale, halves encoder params).
- `hinge_supervision_loss`, `jumprelu_hinge_supervision_loss`, `gated_bce_supervision_loss` — all accept per-feature `pos_weight` (`(n_neg / n_pos).clamp(max=100)`) so the supervised signal isn't swamped by rare features' large positive-class imbalance. The doc is silent on class balance; we add it explicitly so cross-mode comparison isn't confounded by imbalance handling.
- `build_hinge_sae(supervision_mode, ...)` dispatcher and `train_hinge_sae(activations, labels, features, cfg)` trainer — the latter replaces the legacy `train_supervised_sae` path for hinge-family modes.

**`cfg.supervision_mode` default flipped `"hybrid"` → `"hinge"`.** New fresh runs use the end-to-end design. Reproducing summary6/7 requires `--supervision hybrid` explicitly. Two new sub-knobs: `gated_tie_weights` (for `gated_bce`) and `jumprelu_theta_init` (for `hinge_jumprelu`).

**`pipeline/train.py`:**
- Early dispatch in `train_supervised_sae(...)`: if `supervision_mode` is one of the hinge-family modes, delegate to `train_hinge_sae`. Legacy path unchanged.
- New `load_trained_sae(model_cfg)` helper — instantiates the right SAE class based on `model_cfg["supervision_mode"]` (checkpoint metadata). Falls back to `SupervisedSAE` for pre-v8.11 checkpoints (which didn't record a supervision_mode).

**9 downstream sites updated** (evaluate, promote_loop ×2, composition, intervention, causal, circuit, residual, amplify, feature_splitting): each `sae = SupervisedSAE(d_model, n_sup, n_unsup, n_lista)` replaced with `sae = load_trained_sae(model_cfg)`. Checkpoints written by hinge-family training will load correctly into the right class for all downstream steps.

**CLI:** `--supervision` choices extended to include `hinge`, `hinge_jumprelu`, `gated_bce`. New flags `--gated-tie-weights` and `--jumprelu-theta-init`.

### What's different about the hinge path (vs legacy frozen-decoder)

- **No frozen decoder.** All decoder columns (supervised + unsupervised) train freely, unit-normalized after each step.
- **No pre-computed target_dirs constraint.** `compute_target_directions` still runs post-hoc so `target_directions.pt` is written for downstream diagnostics (intervention, promote_loop cosine gate, composition), but target_dirs aren't used in the loss.
- **Per-feature sparsity disabled on the supervised slice.** L1 penalty `cfg.lambda_sparse * acts.abs().mean()` applies only to `acts[..., n_supervised:]`. Rationale from the doc: supervised sparsity is inherited from the labels; adding L1 on top re-introduces the shrinkage pressure hinge was designed to avoid.
- **Hierarchy loss still applies** (orthogonal regularizer, not coupled to supervision mode).
- **Warmup of supervision weight** (linear from 0 → `cfg.lambda_sup` over `cfg.warmup_steps`) preserved.
- **Decoder init** kaiming + unit-norm columns; same conventions as legacy.
- **Checkpoint format** includes `supervision_mode` + mode-specific params (`gated_tie_weights`, `jumprelu_theta_init`). Legacy checkpoints (missing these fields) load via the legacy `SupervisedSAE` class in `load_trained_sae`.

### What this does NOT change

- Downstream pipeline semantics: `causal.py`, `intervention.py`, `composition.py`, `promote_loop.py`, `layer_sweep.py`, `usweep.py` all expect the same `(recon, sup_pre, sup_acts, all_acts)` forward signature, which all three new classes honor.
- `evaluate.py`'s reconstruction / F1 / AUROC / cosine-to-target computation. The cos-to-target number that was "always 1.000" under frozen decoder will now report a learned number (0 to 1) as a post-hoc diagnostic.
- Role tags / scaffold / denylist (v8.10) — independent of supervision mode.
- BOS masking (v8.6) — independent of supervision mode.

### What the user should run first

To validate the new default works end-to-end on the existing cached artifacts:

```
rm -f pipeline_data/supervised_sae.pt \\
pipeline_data/supervised_sae.pt.meta.json \\
pipeline_data/supervised_sae_config.pt \\
pipeline_data/target_directions.pt \\
pipeline_data/split_indices.pt \\
pipeline_data/evaluation.json \\
pipeline_data/evaluation.json.meta.json
F="--layer 9 --sae_id blocks.9.hook_resid_pre --local-annotator --n_sequences 1000 --epochs 15"
python -m pipeline.run --step train $F
python -m pipeline.run --step evaluate $F
```

The run log should print `supervision_mode=hinge` and `decoder: NOT FROZEN (end-to-end training)`. The cos-to-target-dir metric will be a learned number, not 1.000 — that's the expected change.

A/B comparison against legacy:

```
python -m pipeline.run --step train --supervision hybrid $F
python -m pipeline.run --step evaluate --supervision hybrid $F
# (output_dir becomes pipeline_data again; both runs overlap on checkpoint_path
# unless --output_dir is routed to a separate subdir — see RUNNING.md)
```

### Files changed

| File | Change |
|------|--------|
| `pipeline/supervised_hinge.py` | **NEW** — three SAE classes + losses + trainer |
| `pipeline/train.py` | `load_trained_sae` dispatcher; `train_supervised_sae` routes hinge modes to `train_hinge_sae` |
| `pipeline/config.py` | `supervision_mode` default `"hybrid"` → `"hinge"`; new `gated_tie_weights`, `jumprelu_theta_init` |
| `pipeline/run.py` | `--supervision` choices extended; `--gated-tie-weights`, `--jumprelu-theta-init` |
| `pipeline/evaluate.py` | load via `load_trained_sae` |
| `pipeline/promote_loop.py` | load via `load_trained_sae` (both main + capacity-transfer sites) |
| `pipeline/composition.py` | load via `load_trained_sae` |
| `pipeline/intervention.py` | load via `load_trained_sae` |
| `pipeline/causal.py` | load via `load_trained_sae` |
| `pipeline/circuit.py` | load via `load_trained_sae` |
| `pipeline/residual.py` | load via `load_trained_sae` |
| `pipeline/amplify.py` | load via `load_trained_sae` |
| `pipeline/feature_splitting.py` | load via `load_trained_sae` |
| `changes.md` | This entry |

---

## [v8.10] — Pre-discovery scaffold + role tags + denylist

**Date:** 2026-04-21

### Motivation

Reviewer point: the promote-loop's top-ΔR² U latents consistently rediscover surface/artifact directions (BOS, whitespace, bracket variants, code-identifier shapes) because those dominate residual-stream variance. Every round burns a portion of its budget on descriptions of features the catalog *already* knows it doesn't want to promote. Pre-seeding scaffold ("these directions are ours by construction, not by discovery") lets the supervised slice absorb surface capacity before the loop runs, freeing U's top-variance mass for actual semantic discovery.

Separately: the user wants BOS/artifact directions *identified and measured* but *not credited as discoveries* in headline stats — i.e., train/eval them normally so they're not hidden, but filter them out when reporting paper-facing numbers. Needs a principled role tag on features.

### Fixes / additions

**1. `role` field on catalog feature schema.** Optional, default `"discovery"` for backward-compat. `"control"` marks scaffold/artifact features that participate in training+eval but don't contribute to headline means.

**2. `pipeline/scaffold_catalog.json` (new).** ~20 hand-written surface features tagged `role="control"`: document_boundary, whitespace_run, tab_or_indent, list_bullet, currency_symbol, math_operator, bracket_opening/closing, semicolon, ampersand, ellipsis, repeated_character, html_tag_fragment, code_identifier, hex_or_uuid_fragment, at_mention, hashtag_fragment, byline_or_attribution, emoji_or_symbol, url_scheme, separator_rule. Descriptions are operationally testable at the token level.

**3. `pipeline/catalog_utils.py` (new).** `merge_scaffold`, `split_by_role`, `discovery_only_ids` helpers. `merge_scaffold` appends scaffold entries to an existing catalog, skipping id collisions by default so custom hand-written entries win over scaffold defaults.

**4. CLI:** `--scaffold-catalog PATH` merges scaffold into the main catalog after inventory and before annotation. Config default is empty string (opt-in).

**5. `evaluate.py` reports discovery-only means alongside full-catalog means.** `mean_f1_discovery`, `mean_auroc_discovery`, `cal_mean_f1_discovery`, `val_promo_f1_discovery` are added to `evaluation.json`. Console output prints both the full-catalog and discovery-only F1 with the discovery-only number flagged as `← HEADLINE`.

**6. Promote-loop denylist (`cfg.promote_denylist`).** Tuple of case-insensitive substrings. Any Sonnet-generated description whose text contains any denylist pattern is auto-rejected pre-crispness with `category="denylist"`. Default patterns cover BOS / endoftext / padding / start-of-sequence mentions. Complementary to the v8.7 nuisance prefilter (which operates on top-activation token diversity, not description text).

**7. Promoted features tagged `role="discovery"` explicitly** in `promote_loop` so they don't silently adopt whatever default. Scaffold features that roll into the catalog stay tagged `role="control"`.

### Usage

```
python -m pipeline.run \
--step inventory \
--layer 9 \
--sae_id blocks.9.hook_resid_pre \
--scaffold-catalog pipeline/scaffold_catalog.json \
--local-annotator \
--n_sequences 1000
```

After inventory, the main catalog has Sonnet's ~60-70 discovery features + 20 scaffold controls. Annotate/train/evaluate treat them identically; the headline eval numbers only include the ~60-70 discovery features.

### BOS tension (left deliberately unresolved)

`mask_first_n_positions=1` (v8.6) is still on by default — masking position 0 from all analysis was the critical fix for target_dir collapse and we don't want to regress that. The scaffold's `control.document_boundary` feature therefore sees *no* training signal from actual BOS positions today; it'd only be useful if you flip the mask off (`cfg.mask_first_n_positions = 0`).

Preserving the v8.6 stability AND properly training BOS as a control feature requires decoupling "mask BOS for target_dir computation" from "mask BOS everywhere in analysis". That's a separate design question; parking it until there's evidence the current role-tag infrastructure catches what the paper needs.

### Files changed

| File | Change |
|------|--------|
| `pipeline/scaffold_catalog.json` | **NEW** — 20 control/scaffold features |
| `pipeline/catalog_utils.py` | **NEW** — scaffold merge + role helpers |
| `pipeline/config.py` | `scaffold_catalog`, `promote_denylist` fields |
| `pipeline/run.py` | `--scaffold-catalog` flag + merge step post-inventory |
| `pipeline/evaluate.py` | discovery-only aggregates alongside full-catalog |
| `pipeline/promote_loop.py` | denylist gate; promoted features tagged `role="discovery"` |
| `changes.md` | This entry |

---

## [v8.9.1] — Hotfix: stale `candidate_u_indices` in promote-loop round summary

Post-round summary block in `promote_loop.run()` referenced `candidate_u_indices`, a variable removed in v8.7's adaptive-batching rewrite. When round 0 succeeded for the first time (2 promoted features survived all gates), the summary-building crashed with NameError after all the expensive work had already completed. Fix: reference `all_candidates` + `spent` (which do exist in the adaptive path), and include `crispness_breakdown` + `n_atom_proposals` in the round record for downstream diagnostic reading.

---

## [v8.9] — U-width sweep (`--step usweep`) + triage-only mode for promote-loop

**Date:** 2026-04-21

### Motivation

Reviewer question: is the 85% multi_concept rate at `n_unsupervised=256` a capacity bottleneck (U too narrow to split bundles) or a methodology bottleneck (proposal method wrong)? Only empirical sweep answers this. Feature-splitting in unsup SAEs is primarily a width problem per Bricken 2023; wider U slices produce more specialized, less bundled latents. Running the sweep is cheaper than iterating on decomposition/clustering machinery.

### New

**`pipeline/usweep.py` — `--step usweep --widths 256,512,1024`**

For each width:
1. Create `pipeline_data/usweep/u{N}/` and symlink shared artifacts (`tokens.pt`, `activations.pt`, `annotations.pt` + sidecars, `feature_catalog.json`) — only the SAE + targets + split + eval + promote-loop artifacts are regenerated per width.
2. Train supervised SAE with `n_unsupervised = N`.
3. Evaluate (so R², FVE, val_promo_f1 are recorded per width).
4. Run promote-loop in **triage-only mode** (no decomposition, no merge, no annotate/retrain). Just the adaptive describe + crispness loop, producing `crispness_breakdown` and `multi_concept_rate` per width.

Output: `pipeline_data/usweep/summary.json` + formatted stdout table with columns `n_unsup`, `R²`, `R²(P)`, `FVE`, `val_F1`, `spent`, `crisp`, `crisp%`, `mcRate`, `nuisRate`.

**`cfg.promote_triage_only`** — new config flag. When True, promote-loop runs adaptive triage + optional decomposition then breaks BEFORE the merge/annotate/retrain cycle. Used by the sweep to avoid paying the annotation cost on every width.

CLI additions:
- `--step usweep` + `--widths 256,512,1024`

### How to read the output

Three signal-to-interpretation maps:

1. **`mcRate` falls monotonically with width (85% → 60% → 40%)**: the 256 slice was capacity-starved; wider U is the correct fix. Train the main pipeline at whichever width flattens the curve.
2. **`mcRate` stays flat or only drops slightly**: bundles aren't a width problem; U is producing polysemantic latents regardless. Decomposition / cluster-U remain the correct path.
3. **`crisp%` rises sharply but `mcRate` only drops a bit**: wider U is splitting some bundles cleanly and others not; you want *both* wider U AND decomposition.

### Cost

~20-30 min per width (train ~2 min + evaluate ~3 min + triage-only round 0 ~15 min for 2M-token scan + ~100 Sonnet crispness calls). 3 widths = ~75 min. No annotation regen (cached). No decomposition cost (off in sweep mode).

### Files changed

| File | Change |
|------|--------|
| `pipeline/usweep.py` | **NEW** — U-width sweep orchestrator |
| `pipeline/promote_loop.py` | `promote_triage_only` flag — break after triage before merge |
| `pipeline/run.py` | `--step usweep`, `--widths` CLI flag |
| `changes.md` | This entry |

---

## [v8.8] — Multi-concept decomposition with atom-specific target_dirs

**Date:** 2026-04-21

### Motivation

v8.7.1 round-0 produced a clean diagnostic: 100 proposals, 22 nuisance, 78 described, 64 of those rejected as `multi_concept`, 1 crisp. The top-ΔR² U latents at layer 9 are almost all **reconstructive bundles** — real activation structure, but bundling 2–5 distinct concepts under one latent. Describing them as singletons fails the crispness gate correctly; the fix is to mine the bundle for atomic features instead of discarding it.

### Design (reviewer-specified)

For every `multi_concept` rejection:

1. **Decompose via Sonnet.** Pass the original description + top-activating contexts back to Sonnet with a prompt asking for 2-5 atomic token-level feature hypotheses ("each must be yes/no answerable for a single token in context, no 'or'/'and'/'sometimes'").
2. **Crispness gate on atoms.** Each atomic hypothesis runs through the existing crispness gate independently. Non-crisp atoms are dropped here.
3. **Within-round semantic dedup.** A single Sonnet call groups all surviving atoms by semantic equivalence; one representative is kept per group, the rest are logged as merged duplicates. Cheap (1 API call for N atoms).
4. **Mini-annotation on atoms.** Annotate each deduped atom on a random 50-sequence subset (drawn deterministically by `cfg.seed + 9001`). Produces real label tensors per atom.
5. **Atom-specific target_dirs from mini labels.** `d_atom = normalize(mean(x | atom_label=1) − mean(x))` computed on the mini subset. **NOT** the source U's firing-mask direction — the source U is exactly the bundle we're decomposing, so its mask is not evidence for any one atom. Atoms with fewer than `promote_atom_mini_min_pos` (default 3) positives in the mini subset are dropped as under-supported.
6. **Merge using atom-specific directions.** Atoms go into the same merge gate as source-U crisp proposals (cosine against existing `target_dirs` + Sonnet separability), but their cosine is computed against their atom-specific mini-label direction, not the bundle's direction.
7. **Full annotation + retrain on survivors only.** Everything past merge uses the existing v8.1+ id-keyed cache path.

### New proposal set

Each round's proposals list is now the union of:
- **source-U crisp proposals**: latents that passed crispness as singletons. `source_kind = "u_latent_crisp"`. Target_dir = source U mean-shift (existing behavior).
- **decomposed atom proposals**: atoms extracted from `multi_concept` rejections. `source_kind = "decomposed_atom"`. Target_dir = mini-annotation mean-shift.

Both are passed to `merge_catalogs_by_direction` with their respective dirs stacked; no change to merge itself.

### Diagnostic warning (reviewer's #5)

When `multi_concept_rate > promote_multi_concept_warn_rate` (default 0.70) in a round, the loop prints a warning that the U slice is dominated by bundles. This is **per-round diagnostic only** — decomposition runs unconditionally whenever multi_concept rejections exist, regardless of rate. A later round with cleaner U latents still gets normal describe→crispness handling.

### New config + CLI

Config defaults:
- `promote_decompose_multi_concept = True`
- `promote_decompose_max_atoms = 5`
- `promote_atom_mini_min_pos = 3`
- `promote_multi_concept_warn_rate = 0.70`

CLI:
- `--promote-no-decompose` (opt out, e.g., for A/B comparison)
- `--promote-decompose-max-atoms`
- `--promote-atom-mini-min-pos`

### Cost accounting (500 seqs, n_features=74, ~50 multi_concepts per round)

Rough estimates:
- Decomposition: ~50 Sonnet calls (≈$0.50)
- Atom crispness: ~200 atoms → 200 calls (≈$0.50)
- Semantic dedup: 1 batched call (≈$0.05)
- Mini-annotation: ~100 deduped atoms × 50 seqs × 128 tokens = 640K vLLM decisions. At ~800 dec/s steady-state with prefix caching, ≈13 min. Most expensive step.
- Atom target_dir computation: negligible (vectorized mean-shift on a 6400-vector mini tensor)

Total round overhead vs. v8.7.1: roughly +$1 API + ~15 min vLLM. In exchange, `multi_concept` rejections become potential crisp-atom contributions rather than dead ends.

### What's deferred (reviewer's #3)

Clustering U latents by firing-mask overlap before description ("multiple U latents are the same bundle, describe the cluster as a unit"). Bigger rewrite; leaving until we see whether decomposition alone produces enough crisp atoms per round. If the next run still terminates at `too_few_crisp`, clustering is the next lever.

### Files changed

| File | Change |
|------|--------|
| `pipeline/promote_loop.py` | `_decompose_multi_concept`, `_semantic_dedup_atoms`, `_atom_target_dirs_from_labels` helpers; decomposition path injected after crispness triage; proposals = crisp-U ∪ atoms with their respective dirs |
| `pipeline/config.py` | four new `promote_decompose_*` + `promote_atom_*` knobs |
| `pipeline/run.py` | `--promote-no-decompose`, `--promote-decompose-max-atoms`, `--promote-atom-mini-min-pos` |
| `changes.md` | This entry |

---

## [v8.7] — Adaptive promote-loop triage: nuisance prefilter, rejection taxonomy, proposal budget

**Date:** 2026-04-21

### Motivation

v8.6's BOS masking worked, but the promote-loop round-0 under v8.6 terminated at "1/20 crisp" anyway: the top-20 U latents by ΔR² were still mostly high-variance artifacts (token-surface detectors, position anomalies), not genuine novel features. The review observation: the loop's real bottleneck is not post-training validation, it's candidate *triage* — we spend an API budget on 20 arbitrary latents regardless of whether they're even plausible.

### Fixes

**1. Adaptive proposal pulling.** Instead of "describe top-20, stop if fewer than min_kept crisp", pull batches from the ΔR² ranking until either `promote_min_kept` crisp candidates accumulate OR `promote_proposal_budget` (default 100) proposals have been processed. Good crisp features often sit in the middle of the ΔR² distribution — the top few latents capture variance-dominant artifacts. Two new config knobs: `promote_proposal_budget` (default 100), `promote_batch_size` (default 20). CLI: `--promote-proposal-budget`, `--promote-batch-size`.

**2. Nuisance prefilter.** Before spending Sonnet descriptions on a candidate, check whether its top-K activating tokens are degenerate (fire on 1–2 unique token IDs, i.e. a token-surface detector). Such latents are typically rediscoveries of existing `punctuation.*` / `token_form.*` / `part_of_speech.*` features, not novel concepts worth a full description. Rejected candidates go to `ignored_nuisance.json` rather than the promotion path. Threshold: `promote_nuisance_token_diversity = 0.30` (unique / total).

**3. Rejection taxonomy on crispness.** The crispness gate now returns a category alongside the boolean: `crisp`, `multi_concept`, `vague`, `too_broad`, `not_token_local`, `uninterpretable`, `nuisance`, `llm_error`. Each round's summary prints a breakdown so the bottleneck is explicit — e.g., "16 too_broad, 3 multi_concept, 1 crisp" tells us the U slice is dominated by register-level features that Sonnet can't articulate atomically, rather than "19/20 rejected for opaque reasons".

### Design note

This does NOT weaken the crispness gate. 1/20 passing under v8.6 was a real signal: those latents mostly aren't discovery targets. Lowering the gate would just admit junk. The adaptive path lets us keep the gate strict while giving genuine candidates (which may live deeper in the ΔR² ranking) a chance.

### Files changed

| File | Change |
|------|--------|
| `pipeline/promote_loop.py` | adaptive batching loop with nuisance prefilter + crispness taxonomy + resumability per artifact |
| `pipeline/run.py` | `--promote-proposal-budget`, `--promote-batch-size` CLI flags |
| `changes.md` | This entry |

---

## [v8.6] — Mask BOS / position-0 everywhere + IOI scoping + cache-meta false-positive fix

**Date:** 2026-04-21

### Motivation

The v8.5 promote-loop found a real missing feature — the `<|endoftext|>` / document-boundary token — but promoted it 9 times (unsup feature-splitting) and rejected every copy via the cosine gate. Diagnosis revealed two underlying issues:

1. **Position 0 / BOS is a known transformer artifact** (degenerate attention — attends only to self — no prior context, anomalous residual magnitude, acts as an attention sink). Standard mech-interp practice is to mask it out of all analysis; the pipeline never did. Every target_dir computation, R² metric, causal KL, and targeting_ratio was contaminated by position-0 data. Worse, features with heavy position-0 positive sets (`syntactic_position.sentence_initial`, `punctuation.quotation_mark`, sequence-level `text_register.*`, `semantic_domain.*`) had target_dirs dominated by the shared "position 0 vs rest" direction, collapsing 329 of 2701 feature pairs to cos > 0.8 and making the promote-loop cosine gate geometrically meaningless.

2. **IOI scoping wasn't applied yet** from v8.5's plan. Running `--step ioi` clobbered `pipeline_data/supervised_sae.pt` (7-feature IOI SAE written over the 74-feature main SAE), `split_indices.pt` (500-seq vs 1000-seq), and `target_directions.pt`. Downstream steps crashed with empty-test-slice errors.

### Fixes

**1. `pipeline/position_mask.py` (NEW) — `mask_leading` helper**

Small utility that slices `tensor[:, cfg.mask_first_n_positions:]` for any number of (N, T, ...) tensors. Applied uniformly at analysis-load time so cached `tokens.pt` / `activations.pt` / `annotations.pt` don't need re-extraction — they're just sliced shorter on the fly.

Wired at: `train.py`, `evaluate.py`, `intervention.py`, `composition.py`, `causal.py`, `promote_loop.py` (both the main data load and the mini-prefilter tokens load), and `inventory.collect_top_activations` (sets `sel_acts[:, :mask, :] = -inf` before the top-k heap scan so the Sonnet-described top-activating contexts never come from position 0).

Default `cfg.mask_first_n_positions = 1`. Set to 0 to restore pre-v8.6 behavior.

**2. `pipeline/ioi.py` — scope artifacts under `pipeline_data/ioi/`**

`run()` now redirects `cfg.output_dir` to a subdir for the duration of the IOI diagnostic and restores it on exit. Every `cfg.*_path` property (catalog, tokens, activations, annotations, checkpoint, split_indices, target_dirs, eval) routes into that subdir, so IOI's 7-feature synthetic-IOI SAE never overwrites the main catalog-trained SAE.

**3. `pipeline/cache_meta.py` — skip verification of non-Config fields**

`verify_cache_meta` iterated `CACHE_FIELDS` and compared each against `getattr(cfg, field, None)`. Non-Config fields (e.g., `n_features`, which is caller-derived at write time) always returned None, producing a spurious "stale cache" warning on every fresh load. Fix: skip fields that aren't Config attributes; callers still force-check via `extra_required={...}`.

### Recovery after running pre-v8.6 steps

Existing `supervised_sae.pt`, `target_directions.pt`, `split_indices.pt`, and `evaluation.json` were trained / computed against position-0-contaminated activations. They need to be regenerated against the masked distribution:

```
rm pipeline_data/supervised_sae.pt pipeline_data/supervised_sae.pt.meta.json
rm pipeline_data/supervised_sae_config.pt
rm pipeline_data/target_directions.pt pipeline_data/split_indices.pt
rm -f pipeline_data/evaluation.json pipeline_data/evaluation.json.meta.json

python -m pipeline.run --step train    <flags>
python -m pipeline.run --step evaluate <flags>
```

`tokens.pt` / `activations.pt` / `annotations.pt` are reused (the mask is applied at load, not at save).

### Expected effect on the pairwise-cosine collapse

Before v8.6 (diagnostic output):

- mean off-diag cos: 0.065 (fine)
- median off-diag cos: 0.308 (moderate)
- pairs > 0.8: 329 out of 2701
- pairs > 0.95: 170 out of 2701
- max: 0.9997 between `quotation_mark` ↔ `sentence_initial`

Hypothesis: the 170+ high-cos pairs are driven by features whose positive sets over-represent position 0. Masking position 0 should decorrelate them. If it doesn't, there's a second bias worth residualizing against (top PC of the raw `directions` matrix). `diagnose_promote_round.py` reports this distribution for every new round.

### Files changed

| File | Change |
|------|--------|
| `pipeline/position_mask.py` | **NEW** — `mask_leading` helper |
| `pipeline/config.py` | `mask_first_n_positions: int = 1` |
| `pipeline/train.py` | mask activations + annotations at load |
| `pipeline/evaluate.py` | mask activations + annotations at load |
| `pipeline/intervention.py` | mask tokens/activations/annotations at load |
| `pipeline/composition.py` | mask tokens/activations/annotations at load |
| `pipeline/causal.py` | mask tokens + annotations at load |
| `pipeline/promote_loop.py` | mask activations for ΔR² ranking + mean-shift; mask tokens for mini-prefilter |
| `pipeline/inventory.py` | `sel_acts[:, :mask, :] = -inf` in collect_top_activations so Sonnet-described examples never come from position 0 |
| `pipeline/ioi.py` | scope artifacts under `pipeline_data/ioi/` so IOI can't clobber the main SAE |
| `pipeline/cache_meta.py` | skip verification of fields not on Config |
| `changes.md` | This entry |

---

## [v8.5] — Fourth-round reviewer fixes: target_dir validity, case-preservation, cache sidecars, silent-zero fix

**Date:** 2026-04-21

### Motivation

Fourth review on v8.4 flagged three HIGH and four MEDIUM issues: (a) `compute_target_directions` treats zero-positive features as valid; (b) the annotator prompt strips/lowercases tokens and descriptions, mangling case-sensitive features; (c) shared caches are path-presence-keyed with no identity check, so layer/model/corpus changes silently reuse stale artifacts; (d) annotation JSON parse failures silently produce zero labels; (e) probe/readout thresholds still fit on full val; (f) stale mini-prefilter CLI flag; (g) `seq_chunk` hardcoded. All seven addressed.

### Fixes

**HIGH — `compute_target_directions` zero-positive bug** (`pipeline/train.py:65-80`)

Previous code clamped `counts.min` to 1 before division, so a feature with zero positives computed `mean_pos = 0 / 1 = 0`, `direction = -mean_all`, and `raw_norm = ||mean_all||`. That passed the `raw_norms > 1e-6` validity check with a "direction" that is anti-parallel to the global activation centroid — meaningless but load-bearing under frozen decoder, where the decoder column is locked to `target_dir` by construction.

Fix: validity is now `(raw_counts > 0) & (raw_norms > 1e-6)`. The clamp stays for the division (safe), but the validity gate uses raw counts. Returned counts are now the pre-clamp raw values for honest downstream reporting.

**HIGH — annotator prompt case/whitespace preservation** (`pipeline/annotate.py:546-565, 673`)

Per-token prompts stripped whitespace from the token string (`tok_str = all_token_strs[seq_j][tok_k].strip()`) and lowercased the feature description (`.lower()`). Both break case-sensitive features:
- Stripping destroys GPT-2 BPE word-boundary markers — `" The"` (start of word) and `"The"` (subword continuation) become the same token name in the prompt.
- Lowercasing turns "token is `US`?" into "token is `us`?" — different questions for any sensible annotator.

Fix: token strings are preserved verbatim and embedded via `json.dumps(tok_str)` so newlines, quotes, and whitespace survive as escape sequences. Feature descriptions are no longer lowercased. The `.rstrip(".")` trim is kept (purely cosmetic, not semantic).

Caveat for existing runs: existing `annotations.pt` files were produced with the old (stripped/lowered) prompts. v8.1's id-keyed cache accepts them, so the incremental path will mix old-prompt labels for existing features with new-prompt labels for newly-added features. To get uniform labeling, delete `annotations.pt` (+ sidecar) and re-run annotate.

**HIGH — artifact cache identity sidecars** (`pipeline/cache_meta.py` new; `annotate.py`, `train.py`, `evaluate.py` call sites)

New `pipeline/cache_meta.py` provides `write_cache_meta` / `verify_cache_meta` / `load_or_die` helpers that write a `.meta.json` sidecar next to each expensive artifact with the minimal set of Config fields needed to detect staleness:
- `tokens.pt` → model_name, corpus_dataset/split, n_sequences, seq_len
- `activations.pt` → above + target_layer, hook_point, model_dtype, sae_release, sae_id
- `supervised_sae.pt` → above + n_features
- `evaluation.json` → above + n_features

Wired at: tokens+activations save (`annotate.py` subprocess), checkpoint save (`train.py:522`), evaluation save (`evaluate.py:915`). Verification at: activation load in annotate's need-tokens/need-acts gate, activations + checkpoint load at start of evaluate. Missing sidecars (pre-v8.5 runs) are accepted with a warning — "legacy (no sidecar)" — so existing pipelines keep working.

Deliberately NOT wired: inventory-step artifacts (`top_activations.json`, `descriptions.json`, `feature_catalog.json`). Those are cheaper to regenerate and the silent-reuse footgun there is smaller. If it becomes a problem, same pattern extends trivially.

**MEDIUM — annotation JSON parse failures no longer silent-zero** (`pipeline/annotate.py:187-210, 793-810`)

The API path's retry loop previously treated a `_extract_json_object(...) → None` as "success, no positives" because the `if result:` guard silently fell through and `break` exited the retry loop. Fix: parse failure now raises `ValueError` inside the try block, which:
1. Fires the `except Exception` handler and counts as a retry-eligible failure.
2. After `cfg.annotation_max_retries`, appends a record to a module-global `_ANNOTATION_FAILURE_COUNT` list.
3. At end of `annotate_corpus_async`, the list is flushed to `annotations_failures.json` and the failure rate is compared against `cfg.annotation_max_failure_rate` (default 0.10). A rate above that aborts the run rather than saving a catalog with silently-zeroed labels.

Same pattern wired in the batch-positions vLLM path.

**MEDIUM — probe/readout now use val_calib for thresholds** (`pipeline/evaluate.py:648, 816`)

v8.4 split val 50/50 for the supervised SAE (threshold on val_calib, honest F1 on val_promo) but left the linear probe and post-training readout still fitting thresholds on the full val set. That gave the baselines a larger calibration budget than the SAE, quietly inflating their fair-comparison F1. Both baselines now use `val_calib_slice` for threshold search, matching the SAE's protocol.

**MEDIUM — mini-prefilter CLI flag aligned with current gate** (`pipeline/run.py:94`)

v8.4 switched the drop metric to AUROC but the CLI still exposed `--promote-mini-prefilter-min-f1`. Added `--promote-mini-prefilter-min-auroc` as the new flag; the old one is now `argparse.SUPPRESS`'d from help and prints a deprecation note at runtime if passed.

**MEDIUM/LOW — `local_annotation_seq_chunk` is now a config knob**

Previously hardcoded `min(2, N)`. Added `cfg.local_annotation_seq_chunk: int = 2` (default unchanged). Users hitting the ~800 decisions/s ceiling the reviewer flagged can now sweep {2, 4, 8, 16, 32} by overriding the config; throughput vs KV-cache memory trade-off is architecture-specific so no CLI flag yet.

### Files changed

| File | Change |
|------|--------|
| `pipeline/train.py` | raw_counts validity gate in `compute_target_directions`; write cache-meta sidecar on checkpoint save; verify activations sidecar on load |
| `pipeline/annotate.py` | json.dumps-based token prompt; no .lower() on descriptions; parse-failure-as-exception in both API and vLLM paths; failure rate ceiling; `seq_chunk` from config; write/verify cache-meta sidecars on tokens+activations |
| `pipeline/evaluate.py` | verify activation/checkpoint sidecars on load; write evaluation sidecar on save; probe/readout use val_calib_slice for thresholds |
| `pipeline/cache_meta.py` | **NEW** — shared sidecar protocol |
| `pipeline/config.py` | `local_annotation_seq_chunk`, `annotation_max_failure_rate` |
| `pipeline/run.py` | `--promote-mini-prefilter-min-auroc`; deprecation of `--promote-mini-prefilter-min-f1` |
| `changes.md` | This entry |

---

## [v8.4] — Third-round reviewer fixes: honest gating metrics + AUROC prefilter

**Date:** 2026-04-21

### Motivation

Third review of the promote-loop called out four more issues: (a) the v8.3 mini-prefilter used F1 against `pre > 0`, which is a brittle proxy; (b) `val_f1_cal` is still threshold-optimized and scored on the same val set; (c) mini-prefilter used the first N sequences (non-random, biased by ingest order); (d) RUNNING.md + CLI help drifted from the code after v8.0-v8.3's default changes. All four fixed.

### Fixes

**HIGH — Mini-prefilter now uses AUROC, not F1 against `pre > 0`** (`pipeline/promote_loop.py:_mini_prefilter`)

The previous F1-against-firing-mask metric treats every weak U activation as a positive prediction. Polysemantic or leaky U latents fire at low magnitude on a huge fraction of positions, so F1 collapses — the annotator's clean positive set becomes a strict subset of U's wide "active" set, and feature rejection depends on noise rather than signal. The fix is to use the continuous `pre` score directly: `AUROC(pre, annotator_labels)` is scale-free, threshold-free, and measures the ranking agreement between U's activation and the annotator's positive set.

Default threshold: `promote_mini_prefilter_min_auroc = 0.70`. A feature with AUROC ≥ 0.7 on the subset has U activations meaningfully correlated with the annotator's positives; anything near 0.5 is random noise. The legacy F1 metric is retained as `mini_f1_legacy` in the per-feature record for cross-run comparison but is no longer used to decide drops.

Features with fewer than `promote_mini_prefilter_min_support` (default 5) annotator positives OR U-fires on the subset are routed to an AUDIT bucket rather than dropped. A sparse mini-sample can't distinguish a rare-but-real feature from a broken one. Audit records are saved to `mini_prefilter_audit.json` per round.

Audit-only mode via `--promote-no-mini-prefilter` is preserved; a new `promote_mini_prefilter_audit_only` config knob (no CLI flag yet — set it programmatically) enables "compute scores but don't drop" for runs where the team wants to calibrate the AUROC threshold before committing to drops.

**HIGH — Val split for honest post-training gating** (`pipeline/evaluate.py`)

`val_f1_cal` (v8.3) optimized per-feature thresholds on val AND scored F1 on val. That's selection bias on each feature: by construction the reported F1 is above what generalization-to-fresh-data would give. Across 5 promote-loop rounds of dropping on this metric, the pruned catalog is biased toward features that happened to overfit their val-calib thresholds.

Fix: split val 50/50 into `val_calib` (first half, threshold search) and `val_promo` (second half, scored at the val_calib threshold). The new `val_promo_f1` per feature is an honest generalization metric within val; test remains untouched. The overfit `val_f1_cal` is kept for backward compat and used only as a fallback when the feature has 0 positives in val_promo. `_post_training_validation` now prefers `val_promo_f1`.

Trade-off: halving val halves the effective sample per split. For rare features (n_pos < 10 in full val), val_promo may have 0-2 positives and the F1 estimate is noisy. The automatic fallback to `val_f1_cal` handles zero-positive cases; low-but-nonzero cases are logged via `val_promo_n_pos` so promote-loop can be made more conservative in a future iteration.

**MEDIUM — Mini-prefilter samples randomly from the full corpus**

Previous version took `tokens[:n_seqs]`, which under openwebtext's ingest order isn't uniform across content types. Fix: deterministic random subset using `RandomState(cfg.seed + 7919)` over all `tokens.shape[0]` sequences. Same seed offset across runs → reproducible, but not biased by corpus position. Also eliminates a subtle bug where the same early sequences were inspected every promote-loop round.

**LOW — Documentation drift**

- `RUNNING.md`: `--layer` default now reads 9 (was 8). `--full-desc` row now notes it's the default since v8.3 and the flag is a no-op; documents `--use-findex` as the opt-in.
- `run.py`: `--annotator-model` help text now reads `Qwen/Qwen3-4B-Base` (was `Qwen/Qwen3.5-9B`, never existed).

### Files changed

| File | Change |
|------|--------|
| `pipeline/promote_loop.py` | AUROC-based mini-prefilter; random subset; min-support audit routing; reads val_promo_f1 |
| `pipeline/evaluate.py` | Val split into calib/promo halves; saves `val_promo_f1`, `val_promo_n_pos` per feature |
| `pipeline/run.py` | Corrected --annotator-model help text |
| `RUNNING.md` | `--layer` 8→9, `--full-desc` documented as default |
| `changes.md` | This entry |

---

## [v8.3] — Second-round reviewer fixes + mini-annotation prefilter

**Date:** 2026-04-21

### Motivation

Second round of outside review on v8.2. Six issues (two HIGH, three MEDIUM, one LOW) flagged; all six are now addressed. Adds a mini-annotation prefilter for loop speed per the reviewer's "big win" recommendation.

### Fixes

**HIGH — `_compute_mean_shift_dirs` device mismatch** (`pipeline/promote_loop.py:248`)

The encoder weights were cast to X's dtype but not X's device. `sae` lives on GPU; `x_flat` is typically CPU (loaded straight from `activations.pt`). The subsequent `X @ w` would raise `RuntimeError` on mixed-device tensors. Fix: `.to(device=X.device, dtype=X.dtype)` for both `enc_w` and `enc_b`. Same fix applied to the new `_mini_prefilter` helper.

**HIGH — promote-loop post-training gate leaked test labels** (`pipeline/promote_loop.py:_post_training_validation`)

Previous version read `cal_f1` from `evaluation.json`, which is the calibrated-threshold F1 computed on the TEST set with val-selected thresholds. Using it to drop features across multiple promote-loop rounds contaminates test metrics — across 5 rounds of pruning on test, the final test F1 reflects selection bias, not honest generalization.

Fix: `evaluate.py` now captures `val_f1_cal` per feature (val-only F1 at the calibrated threshold) and saves it alongside `cal_f1`. `promote_loop._post_training_validation` now prefers `val_f1_cal` and falls back to `f1` (t=0) only if the field is absent (legacy evaluation.json). Test is never touched for promotion decisions.

**MEDIUM — capacity-transfer output was misleadingly per-latent**

The previous implementation zipped old promoted-U indices against new top-K new-U indices, making records look per-latent even though the docstring admitted it was distribution-level. New-SAE U indices are not stable across retrains, so pairing has no semantic meaning. Fix: `_verify_capacity_transfer` now returns a single aggregate dict with `old_top_k_delta_r2_sum`, `new_top_k_delta_r2_sum`, `old_promoted_delta_r2_sum`, `expected_new_top_k_if_transferred`, `fractional_capacity_drop`, and `transferred` boolean. No more per-latent claims in the output.

**MEDIUM — dropped features leaked into capacity-transfer check**

`promoted_u_indices_this_round` was captured right after the merge gate, so features that failed the post-training F1 floor still entered the capacity-transfer computation. Fix: moved the capture to AFTER `_post_training_validation`, filtering to `kept_ids_after_val` only.

**MEDIUM — `use_findex_suffix` default is now `False`** (`pipeline/config.py:93`)

The F-index path in `annotate.py:515-518` hardcodes few-shot exemplars `F0? 1` (comma) and `F5? 1` ("The"), which break for any catalog that doesn't match those positions — including every incremental promote-loop round. Flipped the default so users get the safer full-description path unless they explicitly opt in via new `--use-findex` flag.

**LOW — `summary7.md` annotated as superseded**

Added a superseded note at the top of summary7.md flagging the two specific numbers that don't survive v8.1's methodology: (a) the +0.104 / +0.038 F1 advantages over probe/readout (mixed-threshold comparison), and (b) the κ=0.583-as-F1-ceiling framing (κ is not directly an F1 upper bound). Everything else in summary7 (R², cos=1.000, loss-ablation BCE-load-bearing finding, siphoning sweep, Pattern B retirement) remains valid.

### New feature: mini-annotation prefilter

Per the reviewer's recommendation ("the biggest loop-speed win is rejecting bad promoted features on 50-100 sequences before paying for full annotation + retrain"), promote_loop now includes a mini-prefilter step between merge and full annotation:

1. Annotate the newly-merged features on a small subset of sequences (default 50).
2. For each feature, compute F1 between the annotator's labels and the source U latent's firing mask on that subset.
3. Drop any feature with `mini_f1 < promote_mini_prefilter_min_f1` (default 0.20). A mismatch at 50 sequences is a strong signal that the annotator can't articulate what the U latent fires on, so full-corpus annotation + retrain would waste compute.

Controls: `--promote-no-mini-prefilter`, `--promote-mini-prefilter-n`, `--promote-mini-prefilter-min-f1`. Defaults: on, 50 sequences, 0.20 F1 floor.

Output: `pipeline_data/promote_loop/round_{N}/mini_prefilter_dropped.json`.

### Files changed

| File | Change |
|------|--------|
| `pipeline/promote_loop.py` | Device-aligned mean-shift; val-only post-training gate; aggregate capacity transfer; survivor-filtered transfer; mini-prefilter |
| `pipeline/evaluate.py` | Saves `val_f1_cal` per feature (val-only, no test contamination) |
| `pipeline/config.py` | `use_findex_suffix` default flipped to False |
| `pipeline/run.py` | New flags: `--use-findex`, `--promote-no-mini-prefilter`, `--promote-mini-prefilter-n`, `--promote-mini-prefilter-min-f1` |
| `summary7.md` | Superseded-by-v8.1 note at top |
| `changes.md` | This entry |

---

## [v8.2] — Promote Loop (U→S capacity transfer)

**Date:** 2026-04-21

### Motivation

Resolves the remaining deferred item from v8.1's reviewer audit: `discover_loop.py` re-trained a fresh unsupervised SAE on the raw cached activations with the same `cfg.seed` every round, so successive rounds re-proposed essentially the same latents. The loop was "propose-once, re-dedup-many" rather than iterative discovery.

The right design uses the U slice of the ALREADY-TRAINED supervised SAE as the proposal pool — because its latents already captured whatever the supervised slice didn't — and ranks those U latents by their contribution to reconstructing the activation (ΔR² under per-latent ablation). High-ΔR² U latents are the natural promotion candidates: they represent capacity the supervised catalog doesn't yet name. Describe them, gate them, promote survivors into S, retrain, and verify capacity moved from U into its matched S slot.

### New file

**`pipeline/promote_loop.py`** — `--step promote-loop`

Round N:

1. **Rank U latents by ΔR² on val.** For each U latent `u`, compute `ΔR²_u = (||err + a_u·W_u||² − ||err||²) / baseline_mse` averaged over val, where `a_u` is the latent's activation and `W_u` its decoder column. Vectorized via the analytical expansion — one pass over val, no per-u forward passes.
2. **Describe top-K via Sonnet.** Reuses `inventory.collect_top_activations` + `inventory.explain_features`. The U slice is wrapped as a `PretrainedSAE` object so the existing machinery works unchanged.
3. **Crispness gate.** Sonnet asks whether each description names a single operationally-testable concept or a grab bag ("fires on X and Y and Z"). Fails closed on LLM error / unparseable response / JSON-decode error — same policy as `merge.py`'s separability gate after v8.1.
4. **Mean-shift target directions.** For each surviving candidate, compute `d_u = normalize(mean(x | u fires) − mean(x))` — same formula as `train.compute_target_directions` uses for supervised features. Provides apples-to-apples cosine with the existing supervised `target_dirs`.
5. **Merge.** Reuses `merge.py` — cosine dedup at `promote_cos_threshold` (default 0.6) + Sonnet separability. Same two-gate structure as `discover_loop.py`, but now with a coherent direction proxy.
6. **Annotate + retrain + evaluate.** Uses v8.1's id-keyed annotation cache so only the newly added feature IDs are labeled.
7. **Post-training per-feature validation.** Reads `evaluation.json`; drops any new feature whose calibrated F1 is below `promote_post_train_f1_floor` (default 0.30). This rejects unlearnable proposals instead of letting the monotonic-growth guarantee do all the work. Dropped features are physically removed from the catalog, downstream artifacts are invalidated, and the pipeline retrains on the pruned catalog before continuing.
8. **Capacity-transfer verification.** After retrain, re-rank the NEW SAE's U slice by ΔR². If the new top-K ΔR² values drop below `promote_capacity_transfer_ratio × old_delta_r2` (default 0.5), the promoted capacity successfully migrated from U to S. Distribution-level check because U indices aren't stable across retrains.

Termination: any round that ends with fewer than `promote_min_kept` survivors (post crispness, post merge, post post-training-filter) stops the loop. Maximum rounds: `promote_max_iters` (default 5).

### CLI additions

- `--promote-top-k` — U latents considered per round (default 20)
- `--promote-max-iters` — maximum rounds (default 5)
- `--promote-min-kept` — terminate below this many survivors (default 3)
- `--promote-post-train-f1-floor` — F1 floor for kept features (default 0.30)
- `--promote-cos-threshold` — merge cosine threshold (default 0.6)
- `--promote-no-llm-separability` — skip Sonnet separability gate

### Deprecation

`pipeline/discover_loop.py` is flagged `DEPRECATED` in its docstring. The `--step discover-loop` CLI entry is retained for reproducibility of earlier runs but `--step promote-loop` is the recommended path for new experiments.

### Files changed

| File | Change |
|------|--------|
| `pipeline/promote_loop.py` | **NEW** — U→S promotion loop |
| `pipeline/run.py` | Added `--step promote-loop` + six `--promote-*` flags |
| `pipeline/discover_loop.py` | Docstring marks as deprecated |
| `RUNNING.md` | Promote-loop usage block |
| `changes.md` | This entry |

---

## [v8.1] — Reviewer-directed correctness fixes

**Date:** 2026-04-21

### Motivation

Outside review raised eight concrete issues with the v7/v8 code. Five are addressed in this commit (all methodology-affecting); two (F-index hardcoded exemplars; annotation throughput tuning) are deferred because `--full-desc` is the default path and the listed tuning items are runtime sweeps, not code changes; one (residual-based discovery-loop redesign) is substantial enough to live in a separate commit.

### Fixes

**1. Separability LLM gate fails CLOSED** (`pipeline/merge.py:150-166`)

Previous behavior on LLM exception / unparseable response / invalid JSON: return `separable=True` (= keep). A single Sonnet outage could silently promote every candidate in a round. Flipped to `separable=False` on all three error paths. The parsed default when `separable` is missing from the response was also `True`; matched the prompt's stated default ("DEFAULT ANSWER IS SAME") by flipping that to `False`.

**2. Feature-ID-keyed annotation cache** (`pipeline/annotate.py`, `pipeline/config.py`)

Previous behavior: `annotations.pt` columns were assumed to match the catalog by position. Any reorder or mid-catalog insert silently re-bound labels to the wrong features. Fix: every save writes a sidecar `annotations_meta.json` (new `Config.annotations_meta_path` property) recording the `feature_ids` sequence. On load:

- Sidecar present + ids available → remap each current feature's column from the cache by ID. Features absent from the cache are scheduled for (re-)annotation; extra cached features are dropped. Safe under any catalog reorder / insert / delete.
- Sidecar absent (legacy cache) → fall back to positional reuse with an explicit warning recommending deletion for safety.
- Sidecar malformed or shape mismatch on (N, T) → full re-annotation.

Group labels are always re-derived from their leaves (cheap) so a freshly remapped tensor doesn't carry stale OR values.

**3. Agreement dispatches on annotator backend + F1 ceiling reported alongside κ** (`pipeline/agreement.py`)

Previous behavior: `annotate_corpus_async` (the API path) was called unconditionally, so `--local-annotator` runs reported inter-rater agreement for a backend that never labeled the training data. Fix: dispatch on `cfg.use_local_annotator` — local runs now route through `annotate_local`. The `annotator_backend` field is written to `agreement.json` for audit.

Separately: κ is not directly an F1 ceiling (it measures agreement beyond chance, not retrieval). Added an annotator-vs-annotator F1 computation per feature — treats one annotation run as the reference, the other as predictions; the resulting F1 is a direct upper bound on the F1 any classifier can achieve against either run. Reported as `f1_ceiling` per feature and `mean_f1_ceiling` in aggregate. κ is retained for band classification (good/moderate/poor) but the F1-ceiling is the defensible "how high can our SAE go on this feature" number.

**4. Calibrated F1 for linear probe and post-training readout** (`pipeline/evaluate.py`)

Previous behavior: supervised SAE reported per-feature `cal_mean_f1` (thresholds optimized on val, evaluated on test). Linear probe and post-training readout reported F1 at raw `logits > 0`. Summary7's headline comparison table labeled all three as "Calibrated F1", which was apples-to-oranges — the SAE's +0.104 F1 advantage over the probe was partially attributable to calibration, not to the representation.

Fix: both baselines now compute logits on val and test, fit per-feature thresholds on val via `optimal_threshold_f1`, apply on test. Reported as `mean_f1_cal` alongside the existing `mean_f1` (t=0) for each baseline. The `evaluation.json` schema gains `probe_baseline.mean_f1_cal`, `posttrain_baseline.mean_f1_cal`, and `per_feature[*].f1_cal` / `cal_threshold` entries. Future cross-baseline comparisons should use the calibrated numbers. Existing `mean_f1` (t=0) columns retained for backward compatibility.

**5. Layer sweep: docstring + mode tagging** (`pipeline/layer_sweep.py`)

The sweep runs inventory per-layer unless `--catalog` is supplied, so each layer gets a different Sonnet-organized catalog. Cross-layer F1/R²/FVE deltas conflate layer with catalog. That's a legitimate question ("what catalog does each layer naturally yield?") but not the same as "where are the same concepts best represented?" — which requires a fixed catalog across layers.

Fix: docstring now explicitly states both questions and which flag selects each mode. At runtime, the sweep prints `Mode: per_layer_catalog` or `Mode: fixed_catalog` and emits a warning when running without `--catalog`. The output `layer_sweep_summary.json` records the `mode` and `catalog` fields so downstream plots don't conflate the two.

### Deferred

- **F-index annotator exemplars hardcode F0=comma, F5=capitalized** (`annotate.py:515-518`). Real bug, but `--full-desc` is the default path in RUNNING.md and in every current run command. Fix tracked but not a blocker.
- **Annotation throughput tuning** (`seq_chunk` sweep, prefix cache to disk, max_model_len, multi-GPU sharding). These are runtime/infra optimizations, not correctness bugs. Not addressed in this commit.
- **Residual-based discovery loop (`promote_loop.py`)**. The current `discover_loop.py` re-trains an unsup SAE on raw activations with the same seed every round, so later rounds re-propose the same latents. The right design is `r_S = x − recon_supervised_only(x)` → rank U latents by their contribution to residual / causal effect → describe → validate → promote to S → retrain → verify capacity shifted from U to S. Substantial redesign; deferred to a follow-up commit so this fix batch can land independently.

### Methodology framing (not code)

**BCE is the differentiable gate/selectivity loss; reconstruction learns magnitude.** The ablation matrix in summary7 is consistent with this framing: drop BCE and F1 collapses to 0.09 (no selectivity signal), drop reconstruction and magnitude supervision fails (F1=0.093 for `frozen_recon_only`). The pos-weight concern for imbalanced features is addressable with feature-balanced BCE + gradient-norm logging, not by abandoning BCE. Future BCE variants should be evaluated in ablation.py against this baseline.

### Files changed

| File | Change |
|------|--------|
| `pipeline/config.py` | `annotations_meta_path` property |
| `pipeline/annotate.py` | ID-keyed cache load/save with legacy fallback |
| `pipeline/merge.py` | Separability gate fails closed |
| `pipeline/agreement.py` | Dispatch on `use_local_annotator`; add F1-ceiling metric |
| `pipeline/evaluate.py` | Calibrated F1 for probe + post-training readout |
| `pipeline/layer_sweep.py` | Docstring clarification; `mode` tagging |
| `changes.md` | This entry |

---

## [v8.0] — Publication-track Experiments: Composition + Layer Sweep

**Date:** 2026-04-20

### Motivation

Summary5 established that supervised SAEs beat unsupervised-within-same-architecture by 15.4× on `targeting_ratio` at Gemma-2-2B layer 20. Summary7 established that layer 9 of GPT-2 Small gives cal_F1=0.625 with frozen-decoder cosine=1.000 and R²=0.971 at 75× fewer latents than the pretrained SAE. Both are single-layer, single-model point results. To turn these into a publishable paper, we need two additional pieces of evidence:

1. **Compositionality**: if ablating feature A alone and feature B alone gives KLs `k_A`, `k_B`, does ablating both jointly give `≈ k_A + k_B`? A "yes" would show the supervised SAE is a *usable editor* of model computation, not just a collection of classifiers.

2. **Cross-layer robustness**: does the 15× supervised-over-unsupervised targeting_ratio advantage survive at every layer, or only at layer 9 / layer 20?

Both are additions of new infrastructure — no breaking changes to existing steps. Neither requires re-training existing artifacts.

### New files

**`pipeline/composition.py`** — `--step composition`

For each subset of size `K ∈ {2, 3}` drawn from the top-5 causally-relevant features, computes:
- `KL_i` for each `i ∈ subset` (individual ablation)
- `KL_{∪ subset}` (joint ablation)
- `linearity = 1 − |KL_{∪} − Σ KL_i| / max(|KL_{∪}|, |Σ KL_i|) ∈ [0, 1]`

Reported per pool — supervised (S), best-match unsupervised (U), best-match pretrained (P) — on the same position set (union of per-feature positive positions). For `K=2` we additionally log the pairwise decoder cosine, enabling the sanity-check correlation `corr(decoder_cos, linearity)`.

Output: `pipeline_data/composition.json` with per-subset records and an aggregate `{supervised, unsupervised, pretrained} × {K=2, K=3}` table plus the `corr(decoder_cos, linearity)` scalar for `S, K=2`.

**`pipeline/layer_sweep.py`** — `--step layer-sweep --layers 4,6,8,9,10,11`

Cross-layer orchestrator that runs the pipeline under `pipeline_data/layer_sweep/layer_{N}/` for each `N` in the list. Each step is idempotent via artifact presence (`evaluation.json`, `causal.json`, `intervention_precision.json`), so re-invocation picks up where a previous run left off. Per-layer failures don't abort the sweep — errors are logged and the next layer continues.

Substitutes the `blocks.{N}.` fragment of `cfg.sae_id` to point at the layer-specific pretrained SAE (applies to `gpt2-small-res-jb`, layers 0-11 all covered).

Output: `pipeline_data/layer_sweep/layer_sweep_summary.json` with one row per layer containing `calibrated_f1`, `r2`, `pretrained_sae_r2`, `mean_cosine_to_target`, `mean_fve`, `causal_mean_kl`, `causal_n_live_features`, `intervention_supervised_mean_ratio`, `intervention_pretrained_mean_ratio`. A formatted ASCII table is also printed to stdout.

CLI additions:
- `--layers` (comma-separated list for `--step layer-sweep`)
- `--sweep-skip-intervention` (faster layer sweep without the 3-way S/U/P comparison)
- `--sweep-skip-causal` (faster layer sweep without per-feature KL necessity)

### Mathematical detail

**Composition linearity metric.** For a subset S with per-feature ablation KLs `k_i` and joint-ablation KL `k_S`:
```
lin(S) = 1 - |k_S - Σ_{i∈S} k_i| / max(|k_S|, |Σ_{i∈S} k_i|, ε)
```
This is symmetric under `joint ↔ sum`, bounded to `[0, 1]`, and robust to both *subadditive* failure (`k_S ≪ Σ k_i`, features interfering — e.g. parallel decoder columns) and *superadditive* failure (`k_S ≫ Σ k_i`, joint effect amplified nonlinearly). `lin = 1` is exact linearity, `lin = 0` is 50% relative error or worse.

**Decoder-cosine vs linearity correlation.** For frozen-decoder supervised features, `W_dec[:, i] = target_dir_i` exactly, so `cos(W_dec[:, i], W_dec[:, j]) = cos(target_dir_i, target_dir_j)`. The geometric prediction is that orthogonal `target_dir`s give independent interventions, hence `linearity ≈ 1`. Supervised features should show higher `linearity` at matched `|cos|` than unsupervised features, because the supervised decoder direction is locked to the mean-shift direction and cannot drift into a correlated subspace.

### What did NOT change

- `pipeline/intervention.py`, `pipeline/causal.py`, `pipeline/ioi.py`, `pipeline/evaluate.py`: unchanged. The new code imports their helpers.
- Existing `--step` commands: unchanged. Both new steps are additive.
- Config defaults: unchanged. Composition's internal knobs (`composition_n_targets`, `composition_pair_ks`, `composition_min_positives`, `composition_max_subsets_per_k`) are attached to `cfg` at runtime rather than added to the dataclass, to keep the existing saved `supervised_sae_config.pt` loadable.

### Expected runtimes (5090, GPT-2 Small, 50 causal sequences)

| Step | Time |
|---|---|
| `--step composition` (5 features, K ∈ {2, 3}, 20 subsets total) | 10-20 min |
| `--step layer-sweep` (6 layers, no annotator reuse) | 3-4 hours |
| `--step layer-sweep --sweep-skip-intervention` | 1.5-2 hours |

### Files changed

| File | Change |
|------|--------|
| `pipeline/composition.py` | **NEW** — K-way joint ablation linearity |
| `pipeline/layer_sweep.py` | **NEW** — Cross-layer orchestrator + aggregator |
| `pipeline/run.py` | Added `--step composition`, `--step layer-sweep`, `--layers`, `--sweep-skip-intervention`, `--sweep-skip-causal` |
| `RUNNING.md` | New step documentation |
| `changes.md` | This entry |

---

## [v7.0] — Layer 9 validation, discovery loop, mean-shift proxy fix

**Date:** 2026-04-13 to 2026-04-20

### Highlights (see `summary7.md` for the full writeup)

- Switched default layer 6 → 9 on GPT-2 Small after a direct comparison: 64 leaves vs 31, cal_F1 0.625 vs 0.484. Layer 9 is the densest semantic band in GPT-2 Small.
- Fixed the silent `from_pretrained_no_processing` bug that made the pretrained-SAE baseline read activations in the wrong distribution, yielding R²=-20,599. `load_target_model` now dispatches on release: `gemma-scope-*` → `no_processing`, everything else → standard `from_pretrained` (with LayerNorm folding). Diagnostic script at `debug_pretrained_sae.py`.
- Bumped `lambda_sparse` default 0.01 → 0.05 after the no-sparsity ablation showed zero F1 change at 0.01 (the penalty was too weak to do any work).
- Added `pipeline/discover_loop.py` and `pipeline/merge.py` for iterative catalog growth via: unsupervised SAE latents → Sonnet descriptions → two-gate dedup (cosine then LLM separability) → incremental annotation → retrain.
- Direction proxy for the merge gate went through three iterations before settling: decoder column (writing direction — wrong) → encoder row (reading direction — produced ~0.04 cosine mean, indistinguishable from random) → mean-shift direction from firing mask (`normalize(mean(x|latent fires) − mean(x))`, matches supervised `compute_target_directions` formula exactly).

### New files

| File | Purpose |
|------|---------|
| `pipeline/discover_loop.py` | Iterative catalog growth orchestrator |
| `pipeline/merge.py` | Two-gate dedup (cosine + Sonnet separability) |
| `pipeline/weaknesses.py` | Per-feature triage against 5 weakness categories |
| `pipeline/agreement.py` | Two independent annotation passes + Cohen's κ |
| `pipeline/residual.py` | Sonnet proposes new features from high-MSE positions |
| `pipeline/amplify.py` | Amplification sweep (0×, 2×, 5×, 10×) for Pattern B testing |
| `pipeline/siphoning.py` | FVE siphoning sweep across `n_unsupervised ∈ {0, 64, 128, 256, 512}` |
| `pipeline/ablation.py` | 12-variant loss-term ablation matrix |
| `debug_pretrained_sae.py` | Standalone diagnostic for `from_pretrained` vs `no_processing` |
| `summary7.md` | Run writeup |

### Breaking changes

None. All new steps are additive; legacy steps unchanged.

---

## [v6.0] — Phase 3: Supervised vs Pretrained SAE Comparison

**Date:** 2026-04-09

### Motivation

Phase 2 showed our supervised SAE achieves calibrated F1 0.601 on Gemma-2-2B with 35/58 features causally active. But the head-to-head comparison against pretrained (GemmaScope) SAE was undermined by a broken baseline (R²=-6.85) and NaN KL on the two highest-frequency features. Phase 3 fixes these and quantifies the three classic pretrained-SAE problems — feature splitting, entangled circuits, and noisy interventions — via direct 3-way comparisons against our supervised latents AND our own unsupervised latents (as an architecture control).

### Fixes

**1. GemmaScope pretrained SAE loader (evaluate.py:465-570)**
Previously used `sae_lens.SAE.from_pretrained().__call__` directly, which applied `apply_b_dec_to_input` and/or activation normalization inconsistent with the GemmaScope JumpReLU convention documented in `agentic-delphi/delphi/sparse_coders/custom/gemmascope.py` and in MEMORY.md:
```
encode: z = JumpReLU_theta(x @ W_enc + b_enc)    (NO b_dec subtraction)
decode: x_hat = z @ W_dec + b_dec
```
This produced R²=-6.85 on Gemma-2-2B residual-stream activations — i.e., the baseline was worse than predicting the mean. Fix: route through `inventory.load_sae(cfg)`, which returns a `PretrainedSAE` wrapper that follows the documented convention and is already used successfully by the inventory step. Expected R² post-fix: > 0.9.

**2. fp32 cast for KL computation (causal.py:471, 506)**
bfloat16 `log_softmax` underflows on large-magnitude logits, producing NaN KL for the two highest-frequency features in summary4 (`comma`: 764 active positions, `period`: 936 active positions). Added `.float()` cast before `log_softmax`. No numerical difference on fp32 runs; fixes Gemma (bf16).

### New experiments

Phase 3 adds three new `--step` commands to `pipeline.run`, all of which reuse the existing Gemma-2-2B artifacts (no retraining). Every experiment is a **3-way comparison**:
- **(S)** supervised latents in our SAE (n=60)
- **(U)** unsupervised latents in our SAE (n=256) — *same architecture, same training data, no supervision signal*
- **(P)** pretrained GemmaScope SAE latents (n=16384)

The (S) vs (U) comparison is the architecture control: if supervision is the causal factor for cleaner representations, (S) should dominate (U) even though they share everything except the supervision term in the loss.

**Experiment A — Feature splitting quantification** (`pipeline/feature_splitting.py`, `--step splitting`)

For each of 10 target features, compute per-pool:
- `top1_coverage`: fraction of positive positions covered by the best single latent
- `top1_specificity`: fraction of that latent's fires landing on positive positions
- `n_at_80`: greedy set-cover — minimum number of latents to cover 80% of positives

Output: `pipeline_data/feature_splitting.json`.

**Experiment B — Downstream circuit analysis** (`pipeline/circuit.py`, `--step circuit`)

Target behavior: closing-bracket prediction. Collects positions where the next token is `)` and the context has an open paren. Uses **attribution patching** (first-order direct effect via decoder columns and layer-20 residual gradient) to compute per-latent contribution to `logit_diff = logit(')') - mean(logit(non-bracket))`. Then counts how many latents per pool are needed to cover 80% of cumulative `|attribution|`.

Attribution patching is necessary because 16384 individual ablations on the pretrained SAE would take hours. The first-order approximation preserves the ranking — good enough to count circuit complexity per pool.

Output: `pipeline_data/circuit_comparison.json`.

**Experiment C — Intervention precision** (`pipeline/intervention.py`, `--step intervention`)

For each of 5 high-causal-KL features from summary4, compute single-latent ablation KL at two position sets:
- `P_pos`: positions where the feature is labeled active
- `P_neg`: equal-size random sample of non-active positions

The targeting ratio `KL(P_pos) / KL(P_neg)` measures how concept-specific the ablation is — high ratio = clean intervention, low ratio = diffuse (polysemantic) intervention.

(S) ablates the supervised latent for the feature directly. (U) and (P) ablate the single unsupervised/pretrained latent with highest coverage (best-match) computed via the same method as Experiment A. Baselines: full supervised SAE reconstruction for (S)/(U), full pretrained SAE reconstruction for (P).

Output: `pipeline_data/intervention_precision.json`.

### Files changed

| File | Change |
|------|--------|
| `pipeline/evaluate.py` | Fix 1 — replace sae_lens native forward with `inventory.load_sae()` wrapper |
| `pipeline/causal.py` | Fix 2 — `.float()` cast before log_softmax on lines 471, 506 |
| `pipeline/feature_splitting.py` | **NEW** — Experiment A |
| `pipeline/circuit.py` | **NEW** — Experiment B |
| `pipeline/intervention.py` | **NEW** — Experiment C |
| `pipeline/run.py` | Added `--step splitting`, `--step circuit`, `--step intervention` |

### What did NOT change

- `pipeline/test_catalog.json`, `pipeline/gpt2_catalog.json` — frozen
- `summary.md` through `summary4.md` — frozen
- All config defaults — backwards compatible
- The trained Gemma-2-2B `supervised_sae.pt` is reused by all three experiments (no retraining)

### Mathematical detail

**Feature splitting coverage (Experiment A):**
For latent i in pool L, on positive set P_f:
  `coverage(i, f) = |{p ∈ P_f : acts_L[p, i] > 0}| / |P_f|`
  `specificity(i, f) = |{p ∈ P_f : acts_L[p, i] > 0}| / |{∀p : acts_L[p, i] > 0}|`

`n_at_80` via greedy set cover on the boolean fire matrix at positive positions.

**Attribution patching (Experiment B):**
For a fixed set of target positions P and target logit direction d = (e_{`)`} - mean_{v∉brackets} e_v), the first-order contribution of latent i to `logit_diff` at position p is:
  `contribution(i, p) = acts_L[p, i] · (∇_{resid[p]} logit_diff · W_dec_L[:, i])`

Aggregated across positions: `total(i) = Σ_{p ∈ P} |contribution(i, p)|`. `n_at_80` = smallest k such that the top-k latents account for ≥80% of the total `Σ_i total(i)`.

**Intervention precision (Experiment C):**
`targeting_ratio = mean_{p ∈ P_pos}(KL(p_base || p_abl)) / mean_{p ∈ P_neg}(KL(p_base || p_abl))`, where p_base is the softmax output under the full SAE reconstruction at layer 20, and p_abl is the softmax under the same reconstruction minus one latent's contribution (zeroed activation → decoder column contribution dropped).

---

## [v5.1] — Ablation Learnings + Gemma-2-2B Support

**Date:** 2026-04-06

### Changes

**1. lambda_sparse: 0.001 → 0.01 (config.py)**
Ablation showed L0=254/256 unsupervised latents firing — the sparsity penalty was too weak to do anything. 10x increase should yield meaningful sparsity (~60-80% active).

**2. hook_point derivation from sae_id (config.py)**
Previously always derived `hook_resid_post`. Now inspects `sae_id`: if it contains `hook_resid_pre` (e.g., GPT-2 JB release `blocks.8.hook_resid_pre`), uses that. Otherwise defaults to `hook_resid_post` (correct for GemmaScope). Fixes the pre-existing mismatch where GPT-2 pretrained SAE baseline was evaluated on wrong activations.

**3. --model-dtype CLI flag (run.py)**
Gemma-2-2B runs in bfloat16. Added `--model-dtype` argument to override `model_dtype` from CLI.

**4. Subprocess hook_point passthrough (annotate.py)**
The subprocess that extracts activations now receives `hook_point` explicitly instead of relying on `__post_init__` derivation (which lacked the `sae_id` needed for the new derivation logic).

**5. Gemma-2-2B documentation (RUNNING.md)**
Added section with exact commands, architecture comparison table, and Phase 1 validation steps for Gemma.

---

## [v5.0] — Phase 1 Validation (Causal + Calibrated Eval + Ablation)

**Date:** 2026-04-06

### Changes

**1. Calibrated threshold evaluation (evaluate.py)**

Previous evaluation tuned per-feature thresholds on the test set ("oracle") — this overfits. Now the 20% held-out data is split into 10% val + 10% test:
- Thresholds are optimized per-feature on val via grid search over `np.linspace(min, max, 200)`
- All reported metrics use these fixed thresholds on the held-out test set
- Oracle thresholds (per-feature optimum on test) are kept but explicitly labeled "NOT honest eval"
- Three F1 variants reported: t=0 (naive), calibrated (from val), oracle (from test)

The train set (80%) is unchanged — no retraining required.

**2. Per-feature causal necessity test (causal.py)**

New Test 4: `test_feature_necessity()`. For each of the n_supervised features:
1. Replace layer-8 residual with full SAE reconstruction → baseline logits
2. Replace residual with SAE reconstruction minus feature k (zero one latent) → ablated logits
3. Compute KL(baseline || ablated) at positions where feature k's annotation label = 1

Metrics: `mean_kl` (causal importance), `pred_change_rate` (fraction of active positions where argmax prediction changes). This is the test that distinguishes a supervised SAE from a linear probe: the probe classifies but can't intervene; the SAE's decoder columns steer the model.

**3. Ablation study fix (ablation.py)**

Replaced deprecated `use_mse_supervision` flag with `supervision_mode` in ablation variants. Now correctly tests all three supervision modes: hybrid (BCE + direction), mse (Makelov-style), bce (legacy).

### Mathematical detail

**Threshold calibration:**
For feature k, the calibrated threshold t_k^* = argmax_{t} F1(y_val_k, scores_val_k > t), searched over 200 linearly-spaced values in [min(scores_k), max(scores_k)]. This is applied as a fixed decision boundary on the test set: pred_k = (scores_test_k > t_k^*).

**Causal necessity KL:**
KL(p_base || p_ablated) = Σ_v p_base(v) · log(p_base(v) / p_ablated(v)), where p_base is the softmax output under full SAE reconstruction and p_ablated is the softmax under reconstruction with feature k zeroed. Computed only at positions where the annotation label for feature k is 1.

---

## [v4.1] — Qwen3-8B Annotator + ChatML + Calibration Fix

**Date:** 2026-03-21

### Problem

Trial run with gpt-oss-20b produced near-empty annotations: most features had 0-3
positives in 12,800 test positions. L0=47.3 (supervised latents firing everywhere
without learning features). Root cause: the model was too conservative — answering "0"
for almost everything. Two contributing factors:

1. **No system instruction in vLLM path.** Prompts were raw text with no framing.
   The model didn't understand the annotation task.
2. **Base rate prior toward "0".** With greedy decoding, the model's prior for "0"
   dominates since most tokens genuinely don't match any given feature.

### Changes

**Annotator model: `Qwen/Qwen3-8B`** (was `openai/gpt-oss-20b`). 8B dense model,
~3-4x faster decode than 20B MoE. More VRAM free for KV cache = more prefix slots.

**ChatML template with no-think.** Manual ChatML construction (`<|im_start|>system`,
`<|im_start|>user`, `<|im_start|>assistant`) gives the model proper instruction framing.
No `<think>` block — bypasses Qwen3's thinking mode entirely.

**Calibration nudge.** System prompt includes "When it plausibly matches, output 1"
to counteract the conservatism. The full system instruction:
```
Feature annotator. For the last token shown, output 1 if it matches
the feature, 0 if not. When it plausibly matches, output 1.
```

**max_model_len: 512 → 1024.** Safety margin for longer sequences re-tokenized
by Qwen3's vocabulary.

Prompt structure preserves prefix caching:
```
Cached:   <|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n[tok0...tok_k]
Variable: \nFeature: {desc}<|im_end|>\n<|im_start|>assistant\n
Output:   1 token ("0" or "1")
```

### Files Changed

| File | Changes |
|------|---------|
| `pipeline/config.py` | `local_annotator_model` → `Qwen/Qwen3-8B` |
| `pipeline/annotate.py` | ChatML template, system prompt, calibration nudge |
| `pipeline/run.py` | Updated help text |
| `setup.sh` | Pre-downloads Qwen3-8B instead of gpt-oss-20b |
| `RUNNING.md` | Updated all model references |

---

## [v4.0] — MSE Feature Dictionary Loss + Local Annotation

**Date:** 2026-03-19

### Motivation

Two fundamental changes to improve both the training objective and annotation scalability:

1. **BCE supervision creates a classification/reconstruction tension.** The trial run showed
   AUROC 0.938 but F1 0.263 — the latent classified well but the threshold was miscalibrated.
   BCE says nothing about which direction the decoder should point or how much the latent
   should activate. MSE feature dictionary supervision (Makelov et al. 2024) resolves this
   by directly optimizing decoder direction and activation magnitude.

2. **API annotation is expensive and hard to scale.** 5,000 sequences cost ~$13-17 via Claude
   Haiku. Decomposing annotation into single-feature single-token binary decisions enables
   use of a local open-source model (~20B) with KV cache reuse, giving 10× more data at
   zero marginal API cost.

### New: MSE Feature Dictionary Supervision (`train.py`)

Replaces `BCE_balanced(sup_pre, labels)` with `MSE_supervised(decoder_dirs, target_dirs, sup_acts, labels)`.

**Step 1 — Precompute target directions (once, before training):**

For each supervised feature `i`, the target direction is the conditional mean shift:
```
d_i = normalize(mean(x | label_i = 1) − mean(x))
```

This is the Makelov et al. mean feature dictionary formula. `d_i` is the direction in
residual stream space uniquely associated with concept `i` being present.

Implementation is fully vectorized:
```python
counts = labels.sum(dim=0)                           # (n_sup,)
mean_pos = (labels.T @ activations) / counts          # (n_sup, d_model)
directions = mean_pos - activations.mean(dim=0)       # (n_sup, d_model)
target_dirs = directions / directions.norm(dim=1)     # unit-normalized
```

Single matmul, no Python loop over features.

**Step 2 — Two-component supervision loss during training:**

A) **Direction alignment** — push each decoder column toward its target direction:
```
L_dir = mean_k(1 − cos(W_dec[:, k], d_k))
```

B) **Magnitude alignment** — at positive positions, activation ≈ projection onto target:
```
L_mag = Σ (sup_acts[k] − x · d_k)² · labels[k] / Σ labels[k]
```

Combined: `L_sup = α · L_dir + β · L_mag` (α=1.0, β=0.5 by default)

**Why this is better than BCE:**

| Property | BCE | MSE Feature Dict |
|----------|-----|-------------------|
| Tells decoder WHERE to point | No (indirect via reconstruction) | Yes (cosine target) |
| Tells encoder HOW MUCH to fire | No (only sign of pre-activation) | Yes (projection magnitude) |
| Threshold tuning | Required (sigmoid > 0.5) | Not needed (continuous) |
| pos_weight tuning | Required (class imbalance) | Not needed (positive-only) |
| Gradient signal | Bounded [-1, 1] | Proportional to error |
| Aligned with reconstruction | Competing objective | Same objective |

The last row is the key insight: with MSE supervision, the decoder column must point toward
`d_i` because the reconstruction loss ALSO demands this — `d_i` is the direction the concept
actually occupies in the residual stream. BCE supervision and reconstruction MSE pull in
different directions; MSE supervision and reconstruction MSE pull in the same direction.

**Backward compatibility:** `use_mse_supervision: bool = True` in config. Set to `False` to
get legacy BCE behavior. The flag is saved in `supervised_sae_config.pt` so evaluation knows
which mode was used.

### New: Evaluation Metrics (`evaluate.py`)

**Cosine similarity to target direction** — per-feature `cos(W_dec[:, k], d_k)`. Direct
measure of whether training converged to the right direction. Values > 0.8 = strong; < 0.5 =
drifted.

**Fraction of Variance Explained (FVE)** — per-feature measure of how much positive-class
activation variance lies along the decoder column:
```
FVE_k = Σ (proj(x_pos − x̄_pos, dec_k))² / Σ ‖x_pos − x̄_pos‖²
```

**Fraction of Variance Unexplained (FVU)** — `1 − FVE`. The headline number for v2: lower
is better. FVU 0.05 means the decoder column captures 95% of the concept's variance.

With MSE supervision, FVE/cosine replace AUROC/F1 as primary metrics. AUROC/F1 are still
computed for comparison.

### New: Local Model Annotation (`annotate.py`)

Decomposed single-feature single-token annotation:

```
For each feature:
  System: "You are a feature annotator. Feature: {desc}. Output only 0 or 1."
  For each (sequence, position):
    Context: tok1 tok2 >>target_tok<< tok4 ...
    → model outputs "0" or "1"
```

Three backends (selected via `--annotator-backend`):

**Ollama (default):** HTTP calls to local Ollama server (`localhost:11434`). Ollama handles
quantization (MXFP4 for gpt-oss:20b — 21B total params, 3.6B active per token, fits in 16GB),
KV caching, and continuous batching out of the box. Async concurrent requests via `aiohttp`.
No model loading logic in Python, no CUDA memory management, no tokenizer fiddling.

**vLLM:** Offline batch inference with automatic prefix caching. Best raw throughput
(10K–50K decisions/sec on H100). System prompt + feature description cached across all
sequences/positions for a given feature.

**HuggingFace transformers:** Logit comparison (`logit["1"] > logit["0"]`) fallback.
No prefix caching. Slowest (~500–2K decisions/sec).

**Default annotator model:** `gpt-oss:20b` (OpenAI, Apache 2.0). MoE architecture,
only 3.6B active params per token. For the trivially simple binary annotation task
(single feature, single token, full context), reasoning is unnecessary — system prompt
explicitly suppresses chain-of-thought: "Output only 0 or 1. No other text."

**Config:**
- `use_local_annotator: bool = False` (API by default)
- `local_annotator_model: str = "gpt-oss:20b"` (Ollama model name)
- `local_annotator_backend: str = "ollama"` (ollama, vllm, or hf)
- `ollama_url: str = "http://localhost:11434"`
- `local_annotation_concurrency: int = 32`

### New: Ablation Variant (`ablation.py`)

Added `mse_vs_bce` / `bce_supervision` variant — trains the same architecture with the
opposite supervision method. Direct comparison of MSE vs BCE on identical data.

### Files Changed

| File | Changes |
|------|---------|
| `pipeline/config.py` | 6 new fields (MSE supervision, local annotation), `target_dirs_path` |
| `pipeline/train.py` | `compute_target_directions()`, `mse_supervision_loss()`, training/validation loop branching |
| `pipeline/evaluate.py` | Section 4b: cosine similarity, FVE, FVU per feature |
| `pipeline/annotate.py` | `annotate_local()`, `_format_annotator_context()`, routing in `run()` |
| `pipeline/ablation.py` | MSE vs BCE ablation variant |

### What Did NOT Change

- SupervisedSAE architecture (split latent space, encoder/decoder, LISTA)
- Reconstruction loss: `MSE(x̂, x)` — unchanged
- Sparsity loss: `L1(acts)` — unchanged
- Hierarchy loss — unchanged
- Decoder normalization — unchanged
- API annotation path — fully preserved behind `use_local_annotator=False`
- Causal validation (step 8) — unchanged
- All v1 evaluation metrics (R², F1, AUROC, L0, hierarchy) — still computed

---

## [v3.5] — Causal Validation (Zero-Ablation)

**Date:** 2026-03-18

### Motivation

The pipeline measures detection (F1/AUROC) and reconstruction (R²), but never asks:
"if I zero-ablate the `german_text` latent, does the model stop producing German?"
A linear probe can detect; only the SAE can intervene. Causal validation is the
experiment that separates a supervised SAE from a supervised linear probe.

### New Step: Causal Validation (`causal.py`)

For each supervised latent k, measures whether zeroing that latent's contribution
to the SAE reconstruction changes the model's output distribution:

**Protocol:**
1. Run the base model with SAE reconstruction at the target layer (baseline).
   A hook replaces the residual stream with `x̂ = W_dec · acts + b_dec`.
2. For each supervised latent k, run the same forward pass but subtract
   latent k's contribution: `x̂_ablated = x̂ − acts_k · W_dec[k, :]`.
3. Compute KL divergence between baseline and ablated output distributions
   at every token position: `KL(p_baseline ‖ p_ablated)`.
4. Report mean KL at **firing positions** (where `acts_k > 0`) vs all positions.

**Metrics per feature:**
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `mean_kl_firing` | `Σ KL[fires] / n_fires` | Causal effect when the feature is active |
| `mean_kl_all` | `Σ KL / n_total` | Average causal effect across all positions |
| `max_kl` | `max(KL[fires])` | Strongest single-position effect |
| `specificity` | `mean_kl_firing / mean_kl_all` | How targeted the effect is (higher = more specific) |

A large `mean_kl_firing` means the latent causally affects model output when it
fires. A near-zero value means the latent is "decorative" — it detects a pattern
but doesn't influence computation. High specificity means the effect is concentrated
at firing positions rather than spread everywhere (which would indicate the latent
is entangled with a broad reconstruction component).

**Computational cost:** Each of n_supervised features requires a separate forward
pass per batch. With 86 features × 50 sequences × 4 batch size = ~1,075 forward
passes. This is why `causal_n_sequences` defaults to 50 (vs 5000 for training).

**Files:**
- `pipeline/causal.py` — New. Full implementation of zero-ablation protocol.
- `pipeline/config.py` — `causal_n_sequences: int = 50`, `causal_batch_size: int = 4`,
  `causal_path` property.
- `pipeline/run.py` — Registered as `--step causal` (Step 8, optional).

**Output:** `pipeline_data/causal.json` with per-feature results and summary.

CLI: `python -m pipeline.run --step causal`

---

## [v3.4] — Sparse Feature Filtering + Baseline Comparisons

**Date:** 2026-03-16

### Sparse feature post-filtering

After annotation, leaf features with positive rate below `min_feature_positive_rate`
(default 0.1%) are removed along with orphaned parent groups. The catalog on disk
is updated to match. This addresses the observation that very narrow features
(e.g., `eigenface_method`, `feel_free_to`, `react_framework`) have near-zero
positives even at 5000 sequences and waste supervised latent capacity.

- `pipeline/config.py`: Added `min_feature_positive_rate: float = 0.001`
- `pipeline/annotate.py`: Added `filter_sparse_features()`, called after
  group label propagation. Prints removed features and updates catalog.
- `pipeline/inventory.py`: Strengthened organize prompt §4 — now explicitly
  warns against individual phrases, single named entities, and highly specialized
  terms, with a concrete heuristic (≥1 in 1000 tokens on diverse web text).

### Evaluation baselines

**Linear probe** (Section 5 in evaluate output): Trains a single `nn.Linear(d_model,
n_features)` on the same train split with class-balanced BCE (matching the SAE's
pos_weight strategy), evaluates on the same test split. Reports Mean F1 and AUROC
alongside the supervised SAE's scores. This is the theoretical upper bound for what
a linear classifier can extract from the residual stream — if the SAE matches the
probe, the shared decoder constraint is "free."

**Pretrained SAE reconstruction** (Section 6): Loads the original GemmaScope SAE
(16k latents via sae_lens), runs encode→decode on the test set, reports MSE and R².
Shows the reconstruction cost of replacing a large unsupervised dictionary with a
small supervised one (e.g., 107+256 = 363 latents vs 16,384). The supervised SAE's
advantage is that every latent has a known, testable meaning.

Both baselines are saved to `evaluation.json` (`probe_baseline` and
`pretrained_reconstruction` keys). Pretrained SAE comparison is wrapped in
try/except — gracefully skipped if sae_lens unavailable.

---

## [v3.3] — OpenRouter Migration

**Date:** 2026-03-16

### API Provider Change: Anthropic SDK → OpenRouter (OpenAI-compatible)

Switched all LLM calls from the Anthropic Python SDK (`anthropic`) to OpenRouter
via the OpenAI SDK (`openai`). OpenRouter is an OpenAI-compatible API that routes
to Anthropic models, providing broader model access and a unified interface.

**New file: `pipeline/llm.py`** — Central LLM client abstraction. All pipeline
API calls route through `get_client()` / `get_async_client()` (sync/async
`openai.OpenAI` instances) and `chat()` / `achat()` helpers. To switch providers,
only this file needs to change.

**Files modified:**
- `pipeline/inventory.py` — `explain_features()` and `organize_hierarchy()` now
  use `from .llm import get_client, chat` instead of `anthropic.Anthropic()`
- `pipeline/annotate.py` — `annotate_corpus_async()` uses `get_async_client()`,
  `annotate_sequence_async()` calls `client.chat.completions.create()` and reads
  `response.choices[0].message.content` (was `client.messages.create()` /
  `response.content[0].text`)
- `pipeline/residual.py` — uses `from .llm import get_client, chat`
- `pipeline/config.py` — model names prefixed with `anthropic/` for OpenRouter
  routing (e.g., `anthropic/claude-sonnet-4-6`)
- `pipeline/requirements.txt` — replaced `anthropic>=0.40.0,<1.0` with `openai>=1.0`
- `RUNNING.md` — env var changed from `ANTHROPIC_API_KEY` to `OPENROUTER_API_KEY`

**Environment variable:** `OPENROUTER_API_KEY` (was `ANTHROPIC_API_KEY`)

---

## [v3.2] — Final Bug Fixes and Running Guide

**Date:** 2026-03-15

### Bug Fixes

**agreement.py annotated group features wastefully**
- `annotate_corpus_async` was called with all features (groups + leaves),
  but `propagate_group_labels` immediately overwrites group labels. Now filters
  to `leaf_features` before annotating and expands back to the full tensor,
  matching the pattern established in `annotate.py` `run()` in v3.1. Saves
  ~20-40% of API calls per agreement run (x2 runs).

**ablation.py overwrote baseline checkpoint**
- `train_supervised_sae()` unconditionally saved to `cfg.checkpoint_path`.
  Since all ablation variants share the same config via `copy.copy()`, each
  variant overwrote the previous checkpoint. After ablation, `supervised_sae.pt`
  contained the last variant, not the baseline. Fixed by adding a
  `save_checkpoint: bool = True` parameter to `train_supervised_sae()`.
  Ablation passes `save_checkpoint=False`. The `split_path` save and
  mid-training checkpoints are also guarded. Default `True` preserves all
  existing behavior for normal training.

**toy/ scripts had broken imports after v3.1 move**
- `toy/train.py` and `toy/evaluate.py` used CWD-relative paths (`from model
  import SupervisedSAE`, `open("features.json")`) that broke when files moved
  from root to `toy/`. Added `_DIR = Path(__file__).resolve().parent` anchor
  and `sys.path.insert` for the model import. All file paths now use `_DIR /`
  prefix, so scripts work regardless of CWD.

### Improvements

**Lightweight tokenizer loading (agreement.py, residual.py)**
- Both files loaded the full 5 GB HookedTransformer just to extract the
  tokenizer, then immediately deleted the model. Replaced with
  `AutoTokenizer.from_pretrained(cfg.model_name)` from `transformers`
  (already a transitive dependency). Same tokenizer, ~1 MB vs 5 GB.

**residual.py: moved `import time` to module level**
- Was inside the `except` block of a retry loop (line 209). Moved to
  module-level imports alongside `json` and `textwrap`.

### New Files

**RUNNING.md** — Running guide covering:
- Quick start, CLI flags, configuration
- Cost estimates per step (with optimization tips)
- Trial run instructions ($2-3 for end-to-end validation)
- vast.ai setup, resumability, output file reference
- Toy validation pipeline instructions

### Files Changed

| File | Changes |
|------|---------|
| `pipeline/train.py` | `save_checkpoint` param guards all saves |
| `pipeline/ablation.py` | Passes `save_checkpoint=False` |
| `pipeline/agreement.py` | Leaf-only annotation, AutoTokenizer |
| `pipeline/residual.py` | AutoTokenizer, `import time` at module level |
| `toy/train.py` | `_DIR` anchor, fixed imports and paths |
| `toy/evaluate.py` | `_DIR` anchor, fixed imports and paths |
| `RUNNING.md` | New: comprehensive running guide |

---

## [v3.1] — Bug Fixes, Robustness, and Project Cleanup

**Date:** 2026-03-13

### Summary

Systematic fix of 13 issues identified in code review: bugs, fragile patterns,
missing resilience, and project organization. No architectural changes.

### Bug Fixes

**Partial annotation resume (annotate.py)**
- Crash recovery checkpoints (`annotations_partial.pt`) were saved after each
  wave but never loaded on resume. Added progress tracking via companion file
  (`annotations_progress.txt`). On restart, annotation resumes from the last
  completed wave instead of starting over.

**Group features wasted API calls (annotate.py)**
- `annotate_corpus()` was called with all features (groups + leaves), but
  `propagate_group_labels()` immediately overwrites group labels with OR of
  children. Now only leaf features are sent for annotation. Group labels are
  populated solely via propagation. Saves ~20-40% of API calls depending on
  group-to-leaf ratio.

**Validation R² used training-set variance (train.py)**
- Per-epoch validation R² was computed as `1 - val_mse / baseline_mse` where
  `baseline_mse` came from the training set. This leaks training-set statistics
  into the validation metric. Now computes `test_baseline_mse` from the test set
  independently. The training R² is unaffected (still uses training baseline).

### Fragile Pattern Fixes

**OOM silently swallowed (inventory.py)**
- `collect_top_activations()` had a bare `except Exception` around tokenization
  that would silently catch CUDA OOM errors, causing the pipeline to continue
  with missing data. Now re-raises any error containing "out of memory" and only
  swallows tokenization-specific failures (malformed text, etc.).

**RNG-based split coupling (train.py, evaluate.py, ablation.py)**
- Train/test split was reproduced via `set_seed()` + `torch.randperm()`, which
  couples the split to PyTorch version and RNG implementation. `train.py` now
  saves the permutation tensor to `split_indices.pt`. `evaluate.py` and
  `ablation.py` load from disk with a fallback to RNG regeneration for backward
  compatibility.

### Missing Resilience

**No retry on explain_features (inventory.py)**
- `explain_features()` made bare API calls with no retry, while every other LLM
  call in the pipeline had retry logic. Added 3-attempt retry with exponential
  backoff matching `organize_hierarchy()`.

**Mid-training checkpoints incomplete (train.py)**
- Mid-training checkpoints (every 5 epochs) saved only `sae.state_dict()`,
  making training resume impossible (optimizer/scheduler state lost). Now saves
  full checkpoint: model, optimizer, scheduler, epoch, and step count.

### Improvements

**Token decoding performance (annotate.py)**
- Replaced per-element `t.item()` calls with bulk `.tolist()` conversion in
  token decoding loop. Reduces Python↔C++ bridge overhead for large corpora.

**Config torch import guard (config.py)**
- `Config.__post_init__` CUDA check now catches `ImportError` so the config
  dataclass can be instantiated without torch installed (useful for inspecting
  configs, generating documentation, etc.).

**.gitignore scoping**
- Replaced blanket `*.pt` rule with directory-specific ignores (`pipeline_data/`,
  `data/`, `checkpoints/`). The blanket rule prevented tracking any `.pt` file
  in the repo, even if intentionally versioned.

**Root-level file organization**
- Moved toy validation scripts (`model.py`, `train.py`, `evaluate.py`,
  `annotate.py`, `extract.py`, `features.json`) from repo root to `toy/`
  subdirectory. These GPT-2 validation scripts are not part of the primary
  pipeline and were cluttering the root namespace.

**Documentation update (pipeline_steps.md)**
- Added documentation for Steps 5-7 (agreement, ablation, residual), LISTA
  refinement architecture, AUROC metric, cosine LR schedule, split persistence,
  and updated resource profile table. Removed "explain the residual" from the
  "not yet implemented" section (it's now Step 7).

### Files Changed

| File | Changes |
|------|---------|
| `pipeline/annotate.py` | Partial resume (#2), leaf-only annotation (#3), batch decode (#13) |
| `pipeline/train.py` | Test-set R² (#4), save split (#7), full checkpoints (#9) |
| `pipeline/evaluate.py` | Load split from disk (#7) |
| `pipeline/ablation.py` | Load split from disk (#7) |
| `pipeline/inventory.py` | Narrow OOM except (#6), retry explain_features (#8) |
| `pipeline/config.py` | split_path property, torch import guard (#14) |
| `.gitignore` | Scoped *.pt → directory ignores (#15) |
| `pipeline_steps.md` | v3.0 documentation (#10) |
| `toy/` | Moved 6 root-level files (#12) |

---

## [v3.0] — LISTA Refinement, Ablation, Agreement, Residual Analysis, AUROC

**Date:** 2026-03-13

### Motivation

The pipeline was functional but lacked: (a) the architectural improvement most
supported by recent results (LISTA), (b) principled annotation quality measurement,
(c) ablation evidence for which components matter, (d) the "explain the residual"
iterative loop from the proposal, and (e) AUROC as a threshold-free metric.

### New Architecture: LISTA Refinement (`train.py`)

SupervisedSAE now supports optional LISTA (Learned ISTA) refinement iterations.
After the initial encode→decode pass, the model computes the reconstruction
residual, re-encodes it, and updates pre-activations with a learnable step size η:

```
pre₀ = W_enc · x + b_enc
acts₀ = ReLU(pre₀)
recon₀ = W_dec · acts₀

For i = 1..n_lista_steps:
    residual = x - recon_{i-1}
    delta = W_enc · residual + b_enc
    pre_i = pre_{i-1} + η_i · delta
    acts_i = ReLU(pre_i)
    recon_i = W_dec · acts_i
```

Each η_i is a learnable scalar (initialized to 0.1). This iterative refinement
(from Gregor & LeCun, ICML 2010) was the single largest improvement in the
synthsaebench experiments, improving F1 from 0.88 to 0.95.

Config: `n_lista_steps: int = 0` (backward compatible, 0 = disabled).
CLI: `python -m pipeline --lista 1`.

The model config checkpoint now saves `n_lista_steps` so evaluate.py can
reconstruct the correct architecture.

### New Step: Inter-Annotator Agreement (`agreement.py`)

Measures annotation reliability by re-running LLM annotation independently
on a subset of sequences and computing Cohen's kappa per feature:

```
κ = (p_observed - p_expected) / (1 - p_expected)
```

Features are classified as:
- **Good** (κ ≥ 0.6): clean labels, will train well
- **Moderate** (0.3 ≤ κ < 0.6): some noise, may need description refinement
- **Poor** (κ < 0.3): noisy labels, description is ambiguous or subjective

This is the principled answer to "but aren't the LLM labels noisy?" — it tells
you exactly which features have clean labels and which don't.

Config: `agreement_n_sequences: int = 100`, `agreement_n_reruns: int = 2`.
CLI: `python -m pipeline.run --step agreement`.

### New Step: Ablation Study (`ablation.py`)

Trains 5-6 model variants with individual components disabled:

| Variant | Override |
|---------|----------|
| baseline | (full model) |
| no_hierarchy | λ_hier = 0 |
| no_warmup | warmup_steps = 0 |
| no_unsupervised | n_unsupervised = 0 |
| no_sparsity | λ_sparse = 0 |
| no_lista | n_lista_steps = 0 (only if baseline uses LISTA) |

Outputs a comparison table with R², mean F1, L0, and hierarchy consistency,
plus deltas from baseline. This is table 1 of any paper.

CLI: `python -m pipeline.run --step ablation`.

### New Step: Explain the Residual (`residual.py`)

Implements the iterative loop from the proposal:

1. Load trained SAE and compute per-position reconstruction error
2. Find the top-k highest-error token positions
3. Extract context windows around those positions
4. Ask Claude to identify patterns in what the SAE is missing
5. Propose 5-15 new features to reduce reconstruction error

Output: `residual_features.json` with proposed features, rationales,
and error statistics. Features can be manually merged into
`feature_catalog.json` for the next iteration.

Config: `residual_n_samples: int = 500`, `residual_top_k_positions: int = 100`,
`residual_model: str = "claude-sonnet-4-6"`.
CLI: `python -m pipeline.run --step residual`.

### AUROC Added to Evaluation (`evaluate.py`)

Per-feature AUROC (Area Under the ROC Curve) is now computed alongside F1.
Uses raw pre-activation logits as scores — no threshold needed.
Implemented without sklearn dependency (trapezoidal rule on sorted scores).

Mean AUROC is reported and saved to `evaluation.json`.

### Evaluation Split Consistency Fix (`evaluate.py`)

Evaluation now uses the same seeded `torch.randperm` shuffle as training,
ensuring the test set is identical. Previously, evaluate.py used an
unshuffled sequential split while train.py used a shuffled split.

### What Did NOT Change

- Core loss function (MSE + class-balanced BCE + L1 + hierarchy)
- Non-circular evaluation principle
- Pipeline steps 1-4 (inventory, annotate, train, evaluate)
- All existing hyperparameter defaults
- Annotation prompts and format

### New CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--lista` | int | LISTA refinement steps (default 0) |
| `--step agreement` | | Run inter-annotator agreement |
| `--step ablation` | | Run ablation study |
| `--step residual` | | Run explain-the-residual |

---

## [v2.2] — Vulnerability Fixes & Operational Hardening

**Date:** 2026-03-13

### Motivation

Systematic review identified 18 issues spanning security, data loss, correctness, and
performance. This release addresses all actionable items without changing the core
architecture or evaluation philosophy.

### Security

**1. Unsafe `torch.load` in root-level files** (`evaluate.py`)
Added `weights_only=True` to `torch.load` calls in the toy GPT-2 evaluation script.
Without this, PyTorch's pickle-based deserialization can execute arbitrary code from
a malicious `.pt` file. The `pipeline/` files already had this correctly.

**2. `.gitignore` created**
Excludes `pipeline_data/` (multi-GB tensors), `__pycache__/`, `.env`, `.DS_Store`,
Jupyter checkpoints, and cloned dependencies (`circuit-tracer/`, `agentic-delphi/`).

### Data Loss Prevention

**3. Annotation failure logging** (`annotate.py`)
When all retries are exhausted, `logger.warning()` now reports the sequence index,
chunk range, retry count, and error message. Previously, failures were entirely silent.

**4. Annotation wave checkpointing** (`annotate.py`)
After each wave of 100 sequences, partial results are saved to
`annotations_partial.pt`. If the process crashes at call 19,999 of 20,000, only the
last wave needs to be re-run. The partial file is deleted on successful completion.

**5. Dynamic `max_tokens` for annotation** (`annotate.py`)
Scaled to `max(500, n_features_in_chunk * 30)`. With 50 features per chunk, the
previous hard-coded 500 tokens could truncate the JSON response, silently zeroing
all labels for that chunk.

**6. Mid-training checkpoints** (`train.py`)
Model state is saved every 5 epochs to `supervised_sae_epoch{N}.pt`. If training
crashes at epoch 14/15, the epoch-10 checkpoint is available.

### Correctness

**7. Shuffled train/test split** (`train.py`)
The split now uses `torch.randperm(n_total)` (seeded for reproducibility) instead of
taking the first 80% in document order. OpenWebText is not randomly ordered, so the
previous split introduced distribution shift between train and test sets.

**8. Per-epoch validation monitoring** (`train.py`)
After each training epoch, reconstruction MSE, supervised BCE, and R² are computed on
the held-out test set. This detects overfitting to noisy LLM annotations before the
final evaluation step.

**9. Topologically ordered group propagation** (`annotate.py`)
Group labels (`group = OR(children)`) are now propagated bottom-up: leaf-adjacent
groups first, then their parents. Previously, nested hierarchies (group → sub-group →
leaf) could produce incorrect labels if the outer group was processed before its
sub-group children.

**10. String-aware JSON extraction** (`annotate.py`, `inventory.py`)
`_extract_json_object()` now tracks `in_string` and `escape` state so that braces
inside JSON string values (e.g., `"description": "uses {curly} braces"`) don't
corrupt the depth counter.

**11. Retry for `organize_hierarchy`** (`inventory.py`)
The single most critical Sonnet call (produces the entire feature catalog) now retries
up to 3 times with exponential backoff. Previously, a transient API failure required
re-running the full inventory step.

### Performance & Usability

**12. Vectorized activation collection** (`inventory.py`)
Replaced the O(n_latents × batch × seq) triple-nested Python loop with vectorized
candidate extraction: `(lat_acts > threshold).nonzero()` finds only positions above
the current heap minimum, then iterates only those. Reduces inner loop iterations
from ~512k per batch to typically <1k.

**13. CUDA fallback** (`config.py`)
`__post_init__` checks `torch.cuda.is_available()` and falls back to CPU with a
warning. Previously, running without a GPU produced a confusing CUDA error.

**14. Accurate tokenization progress bar** (`annotate.py`)
Progress bar now tracks collected sequences (not dataset iteration), so it correctly
shows progress toward `n_sequences`.

**15. Pinned dependency versions** (`requirements.txt`)
Added upper-bound pins to prevent breaking changes from major version bumps:
`torch>=2.0,<3.0`, `transformer_lens>=2.0,<3.0`, `sae-lens>=4.0,<6.0`,
`anthropic>=0.40.0,<1.0`.

### What Did NOT Change

- SupervisedSAE architecture (split latent space, ReLU, no decoder bias)
- Loss function (MSE + class-balanced BCE + L1 + hierarchy)
- Evaluation metrics and non-circular evaluation principle
- Pipeline step structure (inventory → annotate → train → evaluate)
- All existing hyperparameter defaults
- Root-level toy pipeline files remain as-is (validation only, not primary)

---

## [v2.1] — Training Robustness & Reproducibility

**Date:** 2026-03-13

### Motivation

v2.0 pipeline was functionally correct but lacked reproducibility guarantees, had no
learning rate scheduling, and silently lost annotation data on transient API failures.
These are conservative, additive changes — no architecture or design philosophy changes.

### Changes

**1. Reproducibility: Random seed setting** (`config.py`, `train.py`)

New `seed` field in Config (default 42). `set_seed(cfg.seed)` is called at the start
of training, setting `random`, `numpy`, `torch`, and `torch.cuda` seeds.
CLI: `python -m pipeline --seed 123`.

**2. Cosine LR decay over final 1/3** (`train.py`)

Learning rate schedule:
```
lr(step) = lr_0                                          if step < 2T/3
lr(step) = lr_0 · ½(1 + cos(π · (step - 2T/3) / (T/3)))  if step ≥ 2T/3
```
where T = total training steps. The first 2/3 of training runs at constant lr (letting
the supervised loss warmup ramp work unimpeded). The final 1/3 cosine-decays to ~0,
stabilizing final weights. Implemented via `torch.optim.lr_scheduler.LambdaLR`.
Current lr is logged per epoch.

**3. Retry with exponential backoff for annotation** (`annotate.py`)

API calls now retry up to `annotation_max_retries` (default 3) with exponential backoff
(1s, 2s, 4s). Previously, any transient API error (rate limit, network timeout) silently
dropped that chunk's labels to zero. With 20,000 Haiku calls at ~$20, losing data to
transient failures is wasteful.

New Config fields: `annotation_max_retries: int = 3`, `annotation_retry_base_delay: float = 1.0`.

**4. Robust JSON extraction in annotation** (`annotate.py`)

Replaced fragile regex `r"\{[^{}]+\}"` (which fails on any nested braces) with a proper
brace-matching parser `_extract_json_object()`. Finds the first `{`, counts brace depth,
extracts the complete object, then calls `json.loads`. Handles nested structures correctly.

### What Did NOT Change

- SupervisedSAE architecture (split latent space, ReLU, no decoder bias)
- Loss function (MSE + class-balanced BCE + L1 + hierarchy)
- Evaluation metrics and non-circular evaluation principle
- Pipeline step structure (inventory → annotate → train → evaluate)
- All existing hyperparameter defaults

---

## [v2.0] — Automated Feature Selection Pipeline

**Date:** 2026-03-12

### Motivation

v1.0/v1.1 demonstrated the supervised SAE concept on a single cherry-picked circuit
(rabbit→habit, ~5-10 features, 250 sequences, <$1 API cost). This was a proof of concept.

v2.0 scales this to the full proposal: start from a pretrained SAE with thousands of
latents, automatically propose a clean hierarchical feature dictionary, annotate a large
corpus, and train a supervised SAE whose features correspond to the specified descriptions
by construction.

### New: `pipeline/` Package

Seven files implementing the end-to-end pipeline from the proposal:

| File | Role |
|---|---|
| `pipeline/config.py` | Configuration dataclass — all hyperparameters |
| `pipeline/inventory.py` | **Step 1.** Load pretrained SAE, explain latents (Claude Sonnet), organize into hierarchical catalog |
| `pipeline/annotate.py` | **Step 2.** Tokenize corpus, extract activations, LLM-annotate tokens (Claude Haiku, async) |
| `pipeline/train.py` | **Step 3.** Train supervised SAE with class-balanced BCE + hierarchy loss |
| `pipeline/evaluate.py` | **Step 4.** Held-out evaluation: per-feature F1, R², hierarchy consistency |
| `pipeline/run.py` | CLI orchestrator: `python -m pipeline --layer 20 --n_sequences 5000` |
| `pipeline/__main__.py` | Module entry point |

### Step 1: Feature Inventory — The Novel Part

This is what makes the pipeline more than just "supervised SAE training":

1. **Load a pretrained SAE** (GemmaScope JumpReLU via sae_lens, or direct npz).
   The pretrained SAE has d_sae latents (e.g., 16,384). Most are uninterpretable
   or redundant.

2. **Select by firing rate.** Use sae_lens sparsity metadata to pick latents in
   [min_firing_rate, max_firing_rate]. Default: [0.0005, 0.1]. Too rare = no data
   for explanation. Too frequent = noise.

3. **Collect top-activating examples.** Run the base model on the corpus, extract
   residual stream at the target layer, encode through the pretrained SAE:
   ```
   z = JumpReLU_θ(W_enc · (x − b_dec) + b_enc)
   ```
   Track top-k activating contexts per selected latent using min-heaps.

4. **Explain with Claude Sonnet.** Delphi-style prompt with `<<target>>` highlighting
   and activation strengths. Batched (10 latents per call). Produces initial
   natural-language descriptions.

5. **Organize with Claude Sonnet.** Single large prompt. Claude rewrites for
   precision, groups into hierarchy, fills coverage gaps (symmetry: if "red"
   exists → add "blue", "green", etc.), removes vague features. Target: 50-200
   final features. Outputs `feature_catalog.json`.

### Step 2: LLM Annotation at Scale

- Async annotation with `asyncio` + `anthropic.AsyncAnthropic`
- Default 20 concurrent requests, features chunked (50/call)
- 5000 sequences × ~4 chunks = 20,000 Haiku calls
- ~$20 with standard pricing; negligible with org tokens
- Group labels propagated: `group = OR(children)`
- All intermediate outputs cached to disk (resumable)

### Step 3: Training

Same architecture as v1.0 but with more features (50-200 supervised vs 5-10):

```
L = MSE(x̂, x)
  + λ_sup · ramp(step) · BCE_balanced(sup_pre, labels)
  + λ_sparse · ‖acts‖₁
  + λ_hier · ramp(step) · mean_pairs(ReLU(max_child − parent))
```

Class-balanced BCE: `pos_weight_f = clamp(n_neg_f / n_pos_f, 100)` per feature.
Hierarchy loss enforces `act(parent) ≥ act(child)`.
Decoder columns unit-normalized after each optimizer step.

Default hyperparameters: 15 epochs, lr = 3 × 10⁻⁴, warmup = 500 steps.

### Step 4: Evaluation

All metrics on held-out 20% (never seen during training):

| Metric | Formula | Measures |
|---|---|---|
| R² | 1 − MSE_SAE / MSE_mean | Reconstruction quality |
| Per-feature F1 | F1(z_f > 0, A(x, D(f))) | Does latent match its description? |
| L0 | mean active latents per position | Sparsity |
| Hierarchy consistency | P(act_parent ≥ act_child \| child active) | Structural coherence |

Ground truth remains the LLM annotation — not any pretrained SAE activation.
This preserves the non-circular evaluation principle from v1.0.

### Pretrained SAE Loading

Two backends, tried in order:

1. **sae_lens** (preferred): `SAE.from_pretrained(release, sae_id)`.
   Handles GemmaScope, SAEBench, and other standard formats.

2. **GemmaScope npz** (fallback): Direct download from
   `google/gemma-scope-2b-pt-res` via `huggingface_hub`. Loads
   W_enc (d_sae × d_model), W_dec, b_enc, b_dec, threshold.
   Format matches `agentic-delphi/delphi/sparse_coders/custom/gemmascope.py`.

### Data Flow

```
Pretrained SAE (GemmaScope 16k)
    ↓ select by firing rate
500 latents × 20 top examples
    ↓ Claude Sonnet explains (batched)
500 initial descriptions
    ↓ Claude Sonnet organizes + fills gaps
100-200 hierarchical features (feature_catalog.json)
    ↓
Corpus (5000 seqs)
    ↓ model forward pass
Activations (5000 × 128 × d_model)
    ↓ Claude Haiku annotates (async)
Labels (5000 × 128 × n_features)
    ↓
SupervisedSAE training
    ↓
Evaluation (held-out F1, R², hierarchy consistency)
```

### New: `pipeline_steps.md`

Detailed implementation documentation (design decisions, math, vast.ai setup).

### File Inventory After v2.0

| File | Role |
|---|---|
| `supervised_sae_demo.ipynb` | v1.0 demo. Cherry-picked rabbit→habit. |
| `pipeline/` | **v2.0. Automated feature selection pipeline.** |
| `pipeline_steps.md` | Implementation documentation. |
| `model.py`, `train.py`, etc. | Toy GPT-2 pipeline (validation only). |
| `circuit-tracer/` | Cloned dependency. |
| `agentic-delphi/` | Cloned dependency (Eleuther Delphi). |

### Budget Estimate (v2.0)

| Call type | Est. calls | Est. cost |
|---|---|---|
| Sonnet (explanations, 50 batches) | ~50 | ~$2 |
| Sonnet (organization, 1 call) | ~1 | ~$0.50 |
| Haiku (annotation, 20k calls) | ~20,000 | ~$20 |
| **Total** | | **~$22** |
| GPU (vast.ai A100, ~4 hrs) | | ~$8 |

---

## [v1.1] — Colab Compatibility + Progress Visualization

**Date:** 2026-03-06

### Changes to `supervised_sae_demo.ipynb`

1. **Install cell rewritten for Colab.** Previous `pip install circuit-tracer` may not
   resolve correctly. New cell: pins `numpy<2.0` (fixes `dtype size changed` binary
   incompatibility error on Colab where C extensions were compiled against numpy 1.x),
   `git clone`s circuit-tracer from source, `pip install -e ./circuit-tracer`.

2. **tqdm progress bars added to every heavy operation:**
   - Model loading: wall-clock timer
   - Circuit attribution: wall-clock timer
   - Description generation (Sonnet): tqdm over features + timer
   - Activation extraction: tqdm over 300 texts + timer
   - LLM annotation (Haiku): tqdm with live postfix stats (total positives, seq/s, ETA)
   - Training (both models): tqdm per epoch with live R², recon loss, sup loss
   - Unsupervised latent matching: tqdm over features

3. **GPU info printed at startup** (`torch.cuda.get_device_name()`, VRAM).

4. **`tqdm.auto`** imported for Colab-compatible HTML progress bars.

---

## [v1.0] — Complete Rebuild from Proposal

**Date:** 2026-03-06

### Why v0.x Was Scrapped

The v0.x experiment (rabbit_habit_supervised_sae.ipynb) had a fatal design flaw: it used
Per-Layer Transcoder (PLT) activations as ground truth while the motivating proposal claims
transcoders are mechanistically unfaithful. Using the thing you're trying to improve as your
gold standard is circular reasoning. Additionally:

- CLERP labels were empty at runtime (feature IDs from `prune_graph` did not match the
  URL-embedded CLERP map)
- Neuronpedia API returned no data for top activation examples
- The "controlled comparison" measured F1 against transcoder activations, not against the
  LLM specification that defines feature correctness

### What Changed

**New file:** `supervised_sae_demo.ipynb` — complete rewrite from the proposal.

**Deleted logic (conceptual):** No transcoder ground truth. No CLERP dependency. No
Neuronpedia API dependency.

### Design: Cherry-Picked Demonstration

The notebook is a **hyper cherry-picked** demonstration of the supervised SAE concept,
not a full evaluation framework. It shows one concrete case of the approach working:

1. **Circuit tracing** (CLT) identifies which features matter in the rabbit→habit circuit
2. **LLM description** (Claude Sonnet 4.6) writes precise, circuit-aware descriptions
   using upstream/downstream edges and the logit target — not just activation examples
3. **LLM annotation** (Claude Haiku) labels corpus tokens per description
4. **Supervised SAE training** produces latents that correspond to descriptions by construction
5. **Held-out evaluation** confirms the latent fires where the description predicts
6. **Causal intervention** ablates the latent and measures P("habit") reduction

### Non-Circular Evaluation

The ground truth is the **LLM specification itself** — not any SAE or transcoder.

Let D(f) be the natural-language description of feature f, generated by Claude Sonnet.
Let A(x, D(f)) ∈ {0, 1} be the annotation: does token x match description D(f)?
This annotation is produced by Claude Haiku.

Training: minimize `MSE(x̂, x) + λ · BCE(z_f, A(x, D(f))) + λ_s · ‖z‖₁`
where z_f is the pre-activation of supervised latent f.

Evaluation (on held-out sequences never seen during training):
- **Feature recovery:** F1(z_f > 0, A(x, D(f))) — does the SAE fire where the
  description says it should?
- **vs. Unsupervised:** max over all unsupervised latents of F1(z_k > 0, A(x, D(f)))
  — what's the best the unsupervised SAE can do for each concept?
- **Reconstruction:** R² = 1 − MSE_SAE / MSE_mean
- **Causal faithfulness:** zero-ablate supervised latent, measure ΔP(target logit)

At no point does any transcoder activation appear as a label or evaluation target.

### Architecture (same as v0.x, unchanged)

```
SupervisedSAE(d_model, n_supervised, n_unsupervised):
  pre  = W_enc · x + b_enc    ∈ ℝ^{n_total}
  acts = ReLU(pre)             ∈ ℝ^{n_total}
  x̂   = W_dec · acts          ∈ ℝ^d

  Supervised latents: acts[:n_supervised]
  Unsupervised latents: acts[n_supervised:]
  Decoder columns normalized to unit norm after each step.
```

### Training Hyperparameters

| Symbol | Value | Purpose |
|---|---|---|
| n_unsupervised | 256 | Free capacity for reconstruction residual |
| Epochs | 10 | — |
| Batch size | 512 | — |
| lr | 3 × 10⁻⁴ | AdamW |
| λ_sup | 2.0 | Class-balanced BCE weight |
| λ_sparse | 10⁻³ | L1 on all activations |
| Warmup | 300 steps | Linear ramp of supervised loss |
| pos_weight | clamp(n_neg/n_pos, 100) | Per-feature class balance |

### File Inventory After v1.0

| File | Role |
|---|---|
| `supervised_sae_demo.ipynb` | **Primary.** Cherry-picked demonstration. |
| `rabbit_habit_supervised_sae.ipynb` | Deprecated. v0.x with circular ground truth. |
| `model.py`, `train.py`, `evaluate.py`, etc. | Toy GPT-2 pipeline (validation only). |
| `circuit-tracer/` | Cloned dependency (not modified). |
| `arena_12_sections.txt` | ARENA 1.4.2 reference material. |

### Budget

| Call type | Est. calls | Est. cost |
|---|---|---|
| Sonnet (descriptions + hierarchy) | ~15 | ~$0.10 |
| Haiku (annotation, 250 seqs) | ~250 | ~$0.35 |
| **Total** | | **~$0.45** |
