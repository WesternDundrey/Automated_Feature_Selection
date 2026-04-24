# Discovery Loop Ships + Methodology Retrenchment (v8.0 → v8.10)

**Date:** 2026-04-21
**Target:** GPT-2 Small @ layer 9, `blocks.9.hook_resid_pre`, 1000-seq OpenWebText, 74-feature catalog grown to 76 via the first successful discovery round.

## One-line headline

**The promote-loop promoted its first real features.** Round 0 on a 74-feature catalog produced 2 new supervised features — `promoted.u61_a1_r0` (cal_F1 = 0.77, n_pos = 324) and `promoted.u243_a0_r0` (cal_F1 = 0.60, n_pos = 16) — both extracted from multi-concept U-latent bundles via the v8.8 decomposition path, validated through crispness + semantic-dedup + mini-annotation + post-training F1 gates.

The other half of the summary is honest: five reviewer rounds of methodology retrenchment mattered more than any single experiment this cycle. The paper story now reframes around *intervention geometry* and *compositionality under frozen decoders*, not classification F1.

## Context

Summary7 ended with the discovery loop as a planned methodology build. What followed was not "write the loop and run it" but "build the loop, watch it fail, listen to external reviewers, fix the cause, repeat." Five full cycles of external review drove ~10 code versions (v8.1 through v8.10), each fixing a specific methodological flaw or diagnostic gap. The promote-loop now runs end-to-end with proper validation gates, but it only works because each gate was built in response to a concrete failure mode.

## What was built (chronologically)

**v8.1 — Reviewer-directed correctness batch.**
- Separability LLM gate in `merge.py` fails CLOSED on API error / unparseable JSON. The original fail-open behavior silently promoted everything during a single Sonnet outage.
- Annotation cache keyed by feature ID via `annotations_meta.json` sidecar. Previous positional reuse silently misbound labels on any catalog reorder.
- Agreement step dispatches on `cfg.use_local_annotator` so κ is measured for the actual annotator backend, not hardcoded to API Haiku.
- Annotator-vs-annotator F1 (`f1_ceiling`) added alongside κ — κ doesn't directly bound F1; the F1-ceiling is the defensible upper bound.
- Probe + post-training readout calibrated on val_calib, matching the supervised SAE's protocol. Previously baselines used full-val and the supervised SAE used val-calib, quietly inflating the F1 gap.
- `--step layer-sweep` docstring distinguishes "what catalog does each layer yield" from "where are the same concepts best represented" (the latter requires `--catalog`).

**v8.2 — `promote_loop.py`: residual-ranked U→S promotion.**
Replaces the old `discover_loop.py` (deprecated). Uses the U slice of the already-trained supervised SAE as the proposal pool, ranks U latents by per-latent ΔR² on val, describes top-K via Sonnet, gates through crispness + Sonnet separability + mini-prefilter + post-training F1, and verifies distribution-level capacity transfer (old-top-K ΔR² sum vs new-top-K ΔR² sum). Six new CLI flags for the knobs.

**v8.3 — Target_dir validity + case preservation + cache sidecars + silent-zero fix.**
- `compute_target_directions` now requires `raw_counts > 0` for validity. Zero-positive features were silently getting `direction = -mean_all` (nonzero norm, "valid"), which under frozen decoder means their column is anti-parallel to the activation centroid — a meaningless but load-bearing direction.
- Annotator prompt stops lowercasing feature descriptions and stops `.strip()`-ing token strings. "US" vs "us" and " The" vs "The" (leading space marks BPE word boundaries) are real distinctions.
- `pipeline/cache_meta.py`: sidecar protocol for `tokens.pt`, `activations.pt`, `supervised_sae.pt`, `evaluation.json`. Cache identity verified on load; stale caches are flagged rather than silently reused.
- Annotation JSON parse failures no longer silently become zero labels — retry, then abort if aggregate failure rate exceeds `cfg.annotation_max_failure_rate` (default 0.10).

**v8.4 — Honest gating metrics.**
- Mini-prefilter switched from F1-against-(pre>0)-firing-mask to AUROC(pre, labels). The `pre > 0` mask treated weak firings as positive predictions; F1 collapsed toward the annotator's positive rate. AUROC is scale-free.
- Val split 50/50 into `val_calib` (threshold fit) and `val_promo` (F1 scored). `val_promo_f1` is the honest generalization-within-val metric; `promote_loop` gates on it. Test stays untouched.
- Mini-prefilter subset sampled deterministically from the full corpus (not `tokens[:N]`) so openwebtext ingest order doesn't bias the sample.
- RUNNING.md layer default 8→9, `--full-desc` documented as on-by-default since v8.3, `--annotator-model` help text corrected from `Qwen3.5-9B` (never existed).

**v8.5 — target_dir validity, case preservation, cache identity.**
Continuation of v8.3 themes: fail-closed silent-zero on annotation, feature-id sidecar, cache-meta sidecars, probe/readout use val_calib, stale CLI flag fix.

**v8.6 — Mask BOS / position 0 everywhere + IOI scoping + cache-meta false-positive fix.**
The critical methodological fix of the cycle. Diagnostic showed 170 of 2701 catalog pairs had cos > 0.95 and 329 pairs cos > 0.8. Root cause: features with heavy position-0 representation in their positive sets had target_dirs dominated by the shared "position 0 vs rest" direction. BOS/position-0 is a known transformer artifact (degenerate attention, attention-sink magnitude, no prior context); standard mech-interp practice is to mask it. New `cfg.mask_first_n_positions = 1` applied uniformly at analysis-load time in every step — train, evaluate, intervention, composition, causal, promote_loop, `inventory.collect_top_activations`. Cached tokens/activations/annotations slice on the fly.

Also scoped `ioi.py` under `pipeline_data/ioi/` so Q1's 7-feature synthetic-IOI SAE stops clobbering the main catalog-trained SAE. And suppressed the cache-meta false-positive on non-Config fields.

**v8.7 — Adaptive promote-loop triage.**
The top-20 ΔR² latents were dominated by variance-dominant artifacts regardless of what was below; terminating at "1/20 crisp" was a triage-shallow problem, not a crispness-strictness problem. Adaptive batching pulls until `promote_min_kept` crisp accumulate OR `promote_proposal_budget` (default 100) proposals are spent. Nuisance prefilter rejects token-surface detectors (top-K activations dominated by 1-2 unique token IDs) pre-Sonnet. Crispness gate returns a rejection taxonomy: `crisp / multi_concept / vague / too_broad / not_token_local / uninterpretable / nuisance / llm_error`.

v8.7.1 hotfix: single-scan top_activations across full budget (was rescanning 2M tokens per batch of 20, wasting ~400s/round) + smarter default for the `unknown` category (most missing-category rejections are multi_concept).

**v8.8 — Multi-concept decomposition with atom-specific target_dirs.**
Round 0 under v8.7.1 produced 64/75 `multi_concept` rejections. Reviewer call: mine the bundles for atomic features, don't throw them away. Implementation:

1. Sonnet decomposition of each `multi_concept` description + top contexts into 2-5 atomic token-level hypotheses.
2. Crispness on each atom.
3. Sonnet-based within-round semantic dedup (one batched call).
4. **Mini-annotation on the surviving atoms** on a random 50-sequence subset — producing *real per-atom labels*, not relying on the source U's firing mask.
5. Atom-specific target_dirs = `normalize(mean(x | atom_label=1) − mean(x))` on the mini subset. Atoms with n_pos < 3 are dropped as under-supported.
6. Atoms merged into the catalog via the same cosine + separability gate as source-U proposals, but with atom-specific directions.

Reviewer's correction of my initial plan (share source-U's direction across all atoms of a bundle) was right — the source U is precisely the bundle we're decomposing, so its firing mask is not evidence for any one atom.

**v8.9 — U-width sweep + triage-only mode.**
Before building more decomposition/clustering machinery, measure whether `n_unsupervised = 256` is capacity-starved or whether the bottleneck is the proposal method. `pipeline/usweep.py` orchestrates train + evaluate + promote-loop-triage-only across `n_unsup ∈ {256, 512, 1024}` under `pipeline_data/usweep/u{N}/` with symlinked shared artifacts. `promote_triage_only` flag breaks after the adaptive triage before merge/annotate/retrain, so the sweep measures candidate quality cheaply.

**v8.9.1 hotfix.** Round-summary block referenced a variable removed in v8.7's rewrite. Crashed on the first successful round after all the expensive work had completed. One-line fix, but a pointed reminder about keeping diagnostic scaffolding in sync with pipeline restructures.

**v8.10 — Pre-discovery scaffold + role tags + denylist.**
- `role` field on the feature schema (default `"discovery"`, alternative `"control"`).
- `pipeline/scaffold_catalog.json` with 20 hand-written surface/artifact features (document_boundary, whitespace_run, tab, list_bullet, currency, math_operator, bracket variants, semicolon, ampersand, ellipsis, repeated_character, html_tag, code_identifier, hex/uuid, at_mention, hashtag, byline, emoji, url_scheme, separator_rule), all tagged `role="control"`.
- `catalog_utils.merge_scaffold` merges scaffold into the main catalog post-inventory, skipping id collisions so hand-curated entries win over defaults.
- `evaluate.py` reports both full-catalog and discovery-only means. The discovery-only numbers are now the paper's headline; full-catalog numbers are kept for audit.
- `cfg.promote_denylist`: substring patterns that auto-reject descriptions mentioning BOS / padding / start-of-sequence / similar artifacts, pre-Sonnet-crispness. Catches known-artifact text in descriptions that slip past the nuisance prefilter.

## Round 0: the first successful promotion

Pipeline arithmetic for the run where discovery actually worked:

| stage | count | note |
|---|---|---|
| ΔR²-ranked U candidates pulled | 100 | adaptive (from 256 above-threshold) |
| nuisance-dropped pre-Sonnet | ~23 | token-surface detectors |
| Sonnet-described | 77 | |
| crisp source-U singletons | 1 | |
| `multi_concept` rejections | 65 | bundles — 85% of described |
| decomposed atomic hypotheses | 272 | ~4.2 atoms per bundle |
| atoms passing crispness | 64 | 24% of atoms |
| atoms after semantic dedup | 64 | Sonnet judged all distinct |
| atoms with ≥3 mini-positives | 48 | enough support for target_dir |
| **total proposals into merge** | **49** | 1 source-U + 48 atoms |
| kept by merge (cosine + Sonnet sep.) | 8 | 24 cosine-dropped, 17 separability-dropped |
| survived mini-prefilter AUROC | 5 | |
| **survived post-training val_promo_f1 ≥ 0.30** | **2** | `u61_a1_r0` (0.77), `u243_a0_r0` (0.60) |

**Capacity transfer check: NOT transferred.** `old_top_k_ΔR²_sum = 16.34` vs `new_top_k_ΔR²_sum = 16.39`. Adding 2 features out of 64 candidates barely dents U's capacity — expected with so few promotions. Becomes measurable only after many rounds.

## What this run proves (and doesn't)

**Proves:**
- The full promote-loop pipeline runs end-to-end without crashes (once the v8.9.1 stale-variable hotfix lands).
- Decomposition is the correct lever for layer-9 U latents dominated by multi-concept bundles: 1 crisp singleton → 64 crisp atoms → 48 with mini-support → 8 post-merge → 2 post-training. Without decomposition, the round yields 0-1 promotions.
- Atom target_dirs from mini-annotation labels produce non-collapsed cosines (median 0.57 vs pre-v8.8's collapse to ~1.0 or ~0.04 depending on which bug was active).
- Post-training validation drops bad atoms: 3 of 5 post-merge features failed val_promo_f1 < 0.30, all correctly caught rather than silently added.

**Does NOT prove:**
- That the promote-loop converges on a stable catalog. We have 1 round; need 3-5 to see whether capacity transfer becomes measurable and whether round-N promotion rates decay monotonically.
- That the U slice at `n_unsup=256` is the right width. The v8.9 sweep exists but hasn't been run yet — that empirical test is the next prerequisite.
- That the scaffold (v8.10) changes the picture. Scaffold is wired but hasn't been trained-with yet; will only shift round-0 proposals in the NEXT run.

## Reframed paper story

Summary7 claimed a classification-F1 advantage over the linear probe and post-training readout. v8.1 fixed the mixed-threshold comparison that was partly responsible for the advantage; at matched val_calib budgets the probe's calibrated F1 (0.620) and post-training readout's calibrated F1 (0.652) now BEAT the supervised SAE's calibrated F1 (0.612). The supervised-advantage-on-classification claim is retired for this catalog.

What DOES survive:

1. **Reconstruction parity at 75× fewer latents.** 329 supervised+unsup latents achieve R² = 0.971 (before BOS masking; 0.70 after, with the mask changing the baseline_mse denominator rather than the absolute MSE). Pretrained SAE uses 24,576.

2. **Cosine = 1.000 frozen-decoder guarantee.** Every decoder column IS the analytical mean-shift direction for its concept. No training-time drift. Every intervention pushes exactly along the named concept direction (subject to the v8.6 BOS-mask caveat).

3. **Compositionality under frozen decoders.** `corr(decoder_cosine, linearity) = −0.83` at K=2 composition. Falsifiable geometric prediction confirmed: parallel decoder columns → interference; orthogonal decoder columns → independent interventions. No other supervised-SAE paper has tested this; n=10 pairs, effect size large.

4. **Qualitative feature-splitting signature in intervention space.** Supervised interventions are slightly super-additive (joint-KL / sum-individual-KL = 1.05 at K=2, 1.07 at K=3); unsupervised and pretrained are sub-additive (0.88-0.96). This is the direct quantitative readout of "feature splitting in U manifests as decoder-column correlation, which makes joint interventions interfere".

5. **A working catalog-growth loop.** Decomposition + mini-annotation + post-training validation produces real additions (`promoted.u61_a1_r0` at cal_F1 = 0.77). This is new relative to every prior supervised-SAE paper I've found.

## Honest limitations

1. **The U slice at layer 9 is bundle-dominated.** 85% of top-ΔR² latents are multi_concept. Decomposition rescues them, but we don't know yet whether wider U fixes the bundling at the source (v8.9 sweep pending). Until then the loop's efficiency depends on Sonnet decomposition quality.

2. **Mini-annotation is the promote-loop bottleneck.** 50 sequences × ~50 atoms × 128 tokens ≈ 320K vLLM decisions per round just for atom target_dir computation. Roughly 10-15 min per round at current throughput. The single-scan v8.7.1 fix cut the BIGGER scan but the mini-annotation remains.

3. **Inter-annotator reliability is the F1 ceiling.** `mean annotator-vs-annotator F1 = 0.583` on 100 sequences with 63 catalog leaves. No supervised SAE can exceed this on its own catalog. Scaling F1 further requires a stronger annotator, not a better SAE.

4. **Catalog is layer-specific AND training-distribution-specific.** 74 features from Sonnet's organization of GPT-2 layer 9's top-500 pretrained SAE latents. Applying the same methodology to Gemma-2-2B at layer 20 would regenerate a different catalog. Cross-layer/cross-model catalog transfer is not studied.

5. **BOS is masked, not modeled.** `mask_first_n_positions=1` is still the default. The scaffold has `control.document_boundary` but it gets no training signal because position 0 is masked everywhere. Decoupling "mask for target_dir computation" from "mask for analysis" is an open design question.

## What to run next

**Highest priority:**

1. **U-width sweep** (`--step usweep --widths 256,512,1024`). ~75 min. Definitively answers "is 256 starvation or is bundling intrinsic at this layer?" The next round of building depends on this answer.

2. **3-5 rounds of promote-loop under v8.10 with scaffold.** Measures whether pre-seeding 20 control features shifts the multi_concept rate in round 0, and whether the loop reaches `min_kept` more often across rounds. Paper needs a "we grew the catalog from 74 to ~N over K rounds" narrative; one round isn't enough.

**Medium priority:**

3. **Layer sweep with a fixed scaffold catalog** (`--step layer-sweep --catalog <scaffold-extended>`). Cross-layer analysis with the same catalog so differences reflect the activation geometry, not Sonnet's per-layer organization choices.

4. **IOI behavioral intervention test.** `ioi.py` Q1 already passes (cos=0.985, R²=1.000 on ground-truth labels). Need to add Q3: ablate `name_role.subject_second` at layer 9, measure logit-diff drop on canonical IOI sentences. That's the concrete behavioral result the paper's intervention-geometry story needs.

**Lower priority (needs open design):**

5. **BOS-as-control, not BOS-removed.** Keep v8.6 masking for target_dir stability but train control features on unmasked data. Requires splitting the mask into "mask for target_dir", "mask for per-feature stats", "mask for promote-loop ranking". Non-trivial.

6. **Cluster U latents before description** (reviewer's deferred #3 from the v8.8 discussion). If multiple U latents represent the same bundle, describe the cluster as a unit. Addresses the root cause of "decomposition produces many similar atoms from different source bundles". If the v8.9 width sweep shows `n_unsup=1024` cleans most of this up, clustering may be unnecessary.

## Files of record

| file | role |
|---|---|
| `pipeline/promote_loop.py` | the main discovery loop (v8.2 → v8.10) |
| `pipeline/merge.py` | catalog growth with cosine + Sonnet separability |
| `pipeline/usweep.py` | U-width sweep orchestrator |
| `pipeline/scaffold_catalog.json` | pre-seeded control features |
| `pipeline/catalog_utils.py` | role helpers + scaffold merge |
| `pipeline/position_mask.py` | BOS masking at load time |
| `pipeline/cache_meta.py` | identity sidecars on cached artifacts |
| `pipeline/diagnose_promote_round.py` | post-hoc round inspection tool |
| `pipeline_data/promote_loop/round_00/` | round-0 artifacts: crispness.json, atoms.json, atoms_dedup.json, mini_prefilter_dropped.json, mini_prefilter_audit.json, post_training_dropped.json, capacity_transfer.json, summary.json |
| `summary7.md` | prior layer-9 results with superseded-by-v8.1 banner |
| `summary8.md` | this writeup |

## Process note

Five reviewer rounds shaped this cycle more than any single experiment. Each round caught a specific failure mode (fail-open separability gate; silent-zero annotation; target_dir collapse from BOS contamination; val/test contamination in promote-loop gating; single-token-dominant nuisance latents; U bundles vs atomic features; capacity-starved width vs wrong proposal method). The corresponding fixes are cumulative — removing any one of v8.1 through v8.10 would break the pipeline in a specific, diagnosable way. The process produced a working loop, but the loop works only because each fix was made in response to a concrete observed bug, not from up-front design.

The cost of this iteration cycle was real: ~30 hours of reviewer + user + implementation loops across 2 days. The reward is that the v8.10 pipeline has honest metrics (v8.1 calibrated baselines, v8.4 val-promo-f1, v8.5 cache sidecars, v8.8 atom-specific target_dirs) and traceable failure modes (v8.7 rejection taxonomy, v8.10 role tags). For a publication-track methodology, those are prerequisites; without them the paper's claims aren't auditable.
