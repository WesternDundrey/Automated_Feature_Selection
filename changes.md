# Supervised SAE — Change Log

> Rigorous, objective audit trail. Updated on every material change to `supsae/`.

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
